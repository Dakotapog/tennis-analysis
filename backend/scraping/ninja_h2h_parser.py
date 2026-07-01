"""
🎾 Ninja H2H Parser — Alternativa rápida a Playwright para extracción H2H

Usa la API FlashScore Ninja (endpoint df_hh_1_{match_id}) para obtener
el historial completo de ambos jugadores + enfrentamientos directos
en ~0.5s por partido vs 2-3 minutos con Playwright.

Formato propietario:
  Secciones separadas por ~
  Campos separados por ¬
  Key-value separados por ÷
  Ganador indicado por * prefix en KJ/KK

Produce el mismo formato de salida que H2HExtractor._consolidate_result()
para ser consumido por edge_calculator.py sin cambios.
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS

logger = logging.getLogger(__name__)

DELAY_ENTRE_REQUESTS = 0.5  # No martillar la API


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE MATCHING DE NOMBRE (Nodo-36)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_accents(s: str) -> str:
    """Fix B (Nodo-36): strip Unicode accents for accent-insensitive matching.
    'fernández' → 'fernandez', 'é' → 'e' via NFD decomposition.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def _name_tokens(name: str) -> List[str]:
    """Fix C (Nodo-36): len > 1 (was > 2) to include 2-char surnames like 'Lu', 'Mi'.
    Fix B (Nodo-36): strip accents so 'fernandez' matches 'fernández'.
    """
    return [_strip_accents(t.lower()) for t in name.split() if len(t) > 1] if name != 'N/A' else []


def _token_in_kb(tok: str, kb: str) -> bool:
    """Fix B+C (Nodo-36): accent-insensitive comparison with word-boundary guard for short tokens.
    Short tokens (len ≤ 2) use .split() to avoid 'mi' matching 'michelsen'.
    """
    kb_norm = _strip_accents(kb.lower())
    if len(tok) <= 2:
        return tok in kb_norm.split()
    return tok in kb_norm


# ══════════════════════════════════════════════════════════════════════════════
# PARSER DEL FORMATO PROPIETARIO
# ══════════════════════════════════════════════════════════════════════════════

def _parse_sections(raw: str) -> List[Dict[str, str]]:
    """
    Divide la respuesta cruda en secciones (separadas por ~)
    y cada sección en registros de campos.

    Returns:
        Lista de dicts donde cada dict es un registro con sus campos.
    """
    sections = raw.split('~')
    records = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        record = {}
        for pair in section.split('¬'):
            if '÷' in pair:
                k, v = pair.split('÷', 1)
                record[k] = v
        if record:
            records.append(record)
    return records


def _timestamp_to_date(ts: str) -> str:
    """Convierte unix timestamp a dd.MM.yyyy."""
    try:
        dt = datetime.fromtimestamp(int(ts))
        return dt.strftime('%d.%m.%Y')
    except (ValueError, TypeError, OSError):
        return 'N/A'


def _normalize_surface(kd: str, ke: str = '') -> str:
    """
    Normaliza superficie desde campos KD (inglés) y KE (español).

    KD values: clay, hard, grass, i.hard (indoor hard)
    KE values: arcilla, dura, hierba, dura (i)
    """
    surface = (kd or ke or '').lower().strip()
    if 'clay' in surface or 'arcilla' in surface:
        return 'Arcilla'
    elif 'grass' in surface or 'hierba' in surface:
        return 'Hierba'
    elif 'hard' in surface or 'dura' in surface:
        return 'Dura'
    elif 'carpet' in surface or 'alfombra' in surface:
        return 'Alfombra'
    return 'N/A'


def _clean_player_name(name: str) -> str:
    """Limpia nombre de jugador: quita * prefix y espacios extra."""
    if not name:
        return 'N/A'
    name = name.lstrip('*').strip()
    return name


def _determine_winner(kj: str, kk: str) -> str:
    """
    Determina ganador por el prefijo * en KJ o KK.

    KJ = player1 del registro (home en la sección)
    KK = player2 del registro (away en la sección)
    * prefix = este jugador ganó
    """
    if kj.startswith('*'):
        return _clean_player_name(kj)
    elif kk.startswith('*'):
        return _clean_player_name(kk)
    return 'N/A'


def _extract_score_sets(kl: str) -> str:
    """Convierte KL (e.g. '3:1') a formato legible '3-1'."""
    if not kl:
        return 'N/A'
    return kl.replace(':', '-')


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN POR SECCIONES (player1 history, player2 history, H2H directo)
# ══════════════════════════════════════════════════════════════════════════════

def _is_main_section_kb(rec: Dict) -> bool:
    """
    Determina si un record KB es un encabezado de sección PRINCIPAL.

    FlashScore usa KB para dos propósitos distintos:
    1. Encabezados principales:  "Últimos partidos: Player" | "Enfrentamientos"
    2. Sub-secciones internas:   nombre de torneo, año ("2025/2026"), superficie

    Solo los encabezados principales deben usarse para delimitar bloques P1/P2/H2H.
    Los sub-KB dentro de un bloque no tienen KC → los parsers los saltan solos.
    """
    kb_val = rec.get('KB', '')
    return (
        'ltimos partidos' in kb_val or   # "Últimos partidos: Player" (es)
        'Last matches' in kb_val or       # English variant
        'ast matches' in kb_val or        # partial match variant
        'nfrentamientos' in kb_val or     # "Enfrentamientos directos" (es)
        'Head' in kb_val or               # "Head to head" (en)
        'H2H' in kb_val
    )


def _split_into_h2h_blocks(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Divide los records en 3 bloques según los marcadores KB de sección principal.

    La API de FlashScore puede devolver MÚLTIPLES vistas del mismo historial
    (overall, por superficie, por temporada), cada una con su propio trío de
    KB headers. Ejemplo real (Carnicella vs Ekstrand, match 2yfXph3M):

      [  2] Últimos partidos: Carnicella K.   ← vista 1 (overall)
      [ 24] Últimos partidos: Ekstrand M.
      [ 75] Enfrentamientos
      [ 78] Últimos partidos: Carnicella K.   ← vista 2 (superficie)
      [ 86] Últimos partidos: Ekstrand M.
      [117] Enfrentamientos
      [120] Últimos partidos: Carnicella K.   ← vista 3 (temporada)
      [135] Últimos partidos: Ekstrand M.
      [170] Enfrentamientos

    Cada vista tiene partidos ÚNICOS que no están en las otras. Fusionarlas
    da el historial completo (ej: Ekstrand 49+30+33 → 63 únicas).

    Returns:
        (player1_records, player2_records, h2h_records)
    """
    main_kb_indices = [i for i, rec in enumerate(records)
                       if 'KB' in rec and _is_main_section_kb(rec)]

    # Fallback: si no identificamos encabezados principales, usar todos los KB
    if len(main_kb_indices) < 2:
        main_kb_indices = [i for i, rec in enumerate(records) if 'KB' in rec]

    kb_indices = main_kb_indices

    if len(kb_indices) < 3:
        if len(kb_indices) == 2:
            p1_records = records[kb_indices[0] + 1:kb_indices[1]]
            p2_records = records[kb_indices[1] + 1:]
            return p1_records, p2_records, []
        elif len(kb_indices) == 1:
            p1_records = records[kb_indices[0] + 1:]
            return p1_records, [], []
        return [], [], []

    # Clasificar cada sección por tipo de KB header
    # "Últimos partidos: X" → historial de jugador X
    # "Enfrentamientos" / "Head" / "H2H" → H2H directo
    p1_name_lower = ''
    p2_name_lower = ''
    p1_all = []
    p2_all = []
    h2h_all = []

    for s in range(len(kb_indices)):
        start = kb_indices[s] + 1
        end = kb_indices[s + 1] if s + 1 < len(kb_indices) else len(records)
        section_records = records[start:end]
        kb_text = records[kb_indices[s]].get('KB', '').lower()

        is_h2h = ('nfrentamientos' in kb_text or 'head' in kb_text or 'h2h' in kb_text)

        if is_h2h:
            h2h_all.extend(section_records)
        elif not p1_name_lower:
            # Primer "Últimos partidos" → P1
            p1_name_lower = kb_text
            p1_all.extend(section_records)
        elif kb_text == p1_name_lower:
            # Misma persona que P1 → agregar a P1
            p1_all.extend(section_records)
        elif not p2_name_lower:
            # Primer nombre diferente → P2
            p2_name_lower = kb_text
            p2_all.extend(section_records)
        elif kb_text == p2_name_lower:
            # Misma persona que P2 → agregar a P2
            p2_all.extend(section_records)
        else:
            # Nombre desconocido (no debería pasar) → H2H como fallback
            h2h_all.extend(section_records)

    return p1_all, p2_all, h2h_all


def _parse_player_history(records: List[Dict], subject_player: str) -> List[Dict]:
    """
    Parsea registros de historial de un jugador al formato esperado.

    Output format (mismo que H2HExtractor._parse_player_history):
        {fecha, oponente, resultado, outcome, torneo, superficie}
    """
    history = []

    for rec in records:
        # Solo procesar registros con timestamp (KC) — son partidos reales
        if 'KC' not in rec:
            continue

        fecha = _timestamp_to_date(rec.get('KC', ''))
        surface = _normalize_surface(rec.get('KD', ''), rec.get('KE', ''))
        tournament = rec.get('KF', 'N/A')

        kj = rec.get('KJ', '')
        kk = rec.get('KK', '')

        # Determinar quién es el oponente
        p1_name = _clean_player_name(kj)
        p2_name = _clean_player_name(kk)

        # WIS = 'w' o 'l' — outcome para el jugador sujeto de la sección
        wis = rec.get('WIS', '').lower()
        outcome = 'Ganó' if wis == 'w' else 'Perdió'

        # El oponente es el que NO es el sujeto
        # En la sección de player history, KJ suele ser el jugador sujeto
        # cuando KS='home', y KK cuando KS='away'
        # Usamos WIS + winner para determinar
        if kj.startswith('*'):
            # p1 ganó → si WIS='w', el sujeto es p1 y el oponente es p2
            if wis == 'w':
                opponent = p2_name
            else:
                opponent = p1_name
        elif kk.startswith('*'):
            # p2 ganó → si WIS='l', el sujeto es p1 y el oponente es p2
            if wis == 'l':
                opponent = p2_name
            else:
                opponent = p1_name
        else:
            # Sin * prefix (empate/no terminado) — usar posición
            opponent = p2_name if rec.get('KS') == 'home' else p1_name

        # Ranking del oponente
        # CA = ranking del KJ player, CB = ranking del KK player
        ca = rec.get('CA', '')
        cb = rec.get('CB', '')
        opponent_ranking = None
        if opponent == p2_name and cb:
            try:
                opponent_ranking = int(cb)
            except ValueError:
                pass
        elif opponent == p1_name and ca:
            try:
                opponent_ranking = int(ca)
            except ValueError:
                pass

        # BUG-34-1 Fix A: corregir perspectiva del score cuando sujeto es KK
        # KL siempre es sets_KJ:sets_KK — si sujeto=KK, hay que invertir.
        if kj.startswith('*'):
            subject_is_kj = (wis == 'w')   # KJ ganó; sujeto ganó → sujeto=KJ
        elif kk.startswith('*'):
            subject_is_kj = (wis == 'l')   # KK ganó; sujeto perdió → sujeto=KJ
        else:
            subject_is_kj = (rec.get('KS') == 'home')  # sin * prefix: usar posición

        raw_kl = rec.get('KL', '')
        if raw_kl and ':' in raw_kl and not subject_is_kj:
            parts_kl = raw_kl.split(':')
            if len(parts_kl) == 2:
                raw_kl = f'{parts_kl[1]}:{parts_kl[0]}'  # invertir: perspectiva KK

        score = _extract_score_sets(raw_kl)

        # Anti-leakage: excluir partidos de las últimas 36h (Nodo-31)
        # FlashScore puede insertar partidos PROGRAMADOS con fecha de ayer.
        # Filtrar por timestamp KC >= (ahora - 36h) cubre:
        #   - Partidos de hoy (programados o en curso)
        #   - Partidos de ayer que podrían ser scheduled con fecha retrasada
        #   - Margen para diferencias de zona horaria
        _cutoff_ts = int((datetime.now() - timedelta(hours=36)).timestamp())
        try:
            _match_ts = int(rec.get('KC', '0'))
        except (ValueError, TypeError):
            _match_ts = 0
        if _match_ts >= _cutoff_ts:
            continue

        entry = {
            'fecha': fecha,
            'oponente': opponent,
            'resultado': score,
            'outcome': outcome,
            'torneo': tournament,
            'superficie': surface,
        }
        if opponent_ranking:
            entry['opponent_ranking'] = opponent_ranking

        history.append(entry)

    # Deduplicar: la API puede repetir partidos en múltiples vistas
    # (overall, por superficie, por temporada). Usar (fecha, oponente, outcome)
    # como clave única.
    seen = set()
    unique = []
    for h in history:
        key = (h['fecha'], h['oponente'], h['outcome'])
        if key not in seen:
            seen.add(key)
            unique.append(h)

    return unique


def _parse_direct_h2h(records: List[Dict], player1: str, player2: str) -> List[Dict]:
    """
    Parsea registros de enfrentamientos directos al formato esperado.

    Output format (mismo que H2HExtractor._parse_direct_h2h):
        {fecha, jugador1, jugador2, resultado, ganador, torneo, superficie, ganador_sets}
    """
    matches = []

    for rec in records:
        if 'KC' not in rec:
            continue

        # Anti-leakage: excluir H2H de las últimas 36h (Nodo-31)
        # Misma lógica que _parse_player_history — FlashScore puede insertar
        # enfrentamientos PROGRAMADOS como si fueran históricos.
        _cutoff_ts = int((datetime.now() - timedelta(hours=36)).timestamp())
        try:
            _match_ts = int(rec.get('KC', '0'))
        except (ValueError, TypeError):
            _match_ts = 0
        if _match_ts >= _cutoff_ts:
            continue

        fecha = _timestamp_to_date(rec.get('KC', ''))
        surface = _normalize_surface(rec.get('KD', ''), rec.get('KE', ''))
        tournament = rec.get('KF', 'N/A')

        kj = rec.get('KJ', '')
        kk = rec.get('KK', '')

        p1_name = _clean_player_name(kj)
        p2_name = _clean_player_name(kk)
        winner = _determine_winner(kj, kk)

        score = _extract_score_sets(rec.get('KL', ''))

        # ganador_sets: KU = sets won by KJ player, KT = sets won by KK player
        ku = rec.get('KU', '0')
        kt = rec.get('KT', '0')
        try:
            ganador_sets = int(ku) if kj.startswith('*') else int(kt)
        except ValueError:
            ganador_sets = 0

        matches.append({
            'fecha': fecha,
            'jugador1': p1_name,
            'jugador2': p2_name,
            'resultado': score,
            'ganador': winner,
            'torneo': tournament,
            'superficie': surface,
            'ganador_sets': ganador_sets,
        })

    # Deduplicar: la API repite H2H en múltiples vistas
    seen = set()
    unique = []
    for m in matches:
        key = (m['fecha'], m['ganador'], m['resultado'])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique


# ══════════════════════════════════════════════════════════════════════════════
# API CALL
# ══════════════════════════════════════════════════════════════════════════════

def fetch_h2h_from_api(match_id: str) -> Optional[str]:
    """
    Consulta df_hh_1_{match_id} y retorna la respuesta cruda.

    Args:
        match_id: ID del partido en FlashScore (e.g. 'vVC89C3C')

    Returns:
        str: Respuesta cruda de la API, o None si falló
    """
    url = f"{FLASHSCORE_BASE}/df_hh_1_{match_id}"

    try:
        resp = requests.get(url, headers=FLASHSCORE_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        else:
            logger.warning(f"   ⚠️ API retornó {resp.status_code} para match {match_id}")
            return None
    except requests.RequestException as e:
        logger.error(f"   ❌ Error HTTP para match {match_id}: {e}")
        return None


def extract_match_id_from_url(url: str) -> Optional[str]:
    """
    Extrae el match_id de una URL de FlashScore.

    URL format: https://www.flashscore.co/match/tennis/{slug}/{match_id}/#/...
    o: https://www.flashscore.co/match/{match_id}/#/...

    El match_id es el código alfanumérico de 8 chars (e.g. 'vVC89C3C').
    """
    if not url:
        return None

    # Prioridad 1: ?mid= query param (formato API Ninja / Kambi)
    m = re.search(r'[?&]mid=([A-Za-z0-9]+)', url)
    if m:
        return m.group(1)

    # Prioridad 2: segmento de 8 chars alfanuméricos en el path
    # FlashScore URLs: /match/tennis/slug/MATCH_ID/#/h2h
    parts = url.rstrip('/').split('/')
    for part in reversed(parts):
        clean = part.split('#')[0]
        if len(clean) == 8 and clean.isalnum():
            return clean

    # Prioridad 3: penúltimo segmento antes de /#/
    if '/#/' in url:
        pre_hash = url.split('/#/')[0]
        segments = pre_hash.rstrip('/').split('/')
        if segments:
            candidate = segments[-1]
            if len(candidate) >= 6 and candidate.isalnum():
                return candidate

    return None


# ══════════════════════════════════════════════════════════════════════════════
# NODO-49: PLAYWRIGHT H2H FALLBACK — cuando Ninja API + THF fallan
# ══════════════════════════════════════════════════════════════════════════════

async def _playwright_h2h_async(match_id: str, player_name: str,
                                 section_idx: int) -> List[Dict]:
    """
    Nodo-49: Extrae historial de un jugador desde el DOM de FlashScore via Playwright.
    URL validada por usuario en flashs_revisa_h2h_inspector.py (git 23d2d91):
        https://www.flashscore.co/partido/tenis/{match_id}/#/h2h/general
    Selectores validados: .h2h__section, .h2h__row, wcl-stageTime, wcl-tableScore.
    section_idx: 0=jugador1, 1=jugador2 (sección 2 = H2H directo, no usada aquí).
    """
    from playwright.async_api import async_playwright

    h2h_url = f"https://www.flashscore.co/partido/tenis/{match_id}/#/h2h/general"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
            '--disable-software-rasterizer',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        ])
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            await page.goto(h2h_url, wait_until="domcontentloaded", timeout=30000)

            # Cookie consent (mismo handler que ZitaScraper en extraer_cuotas_partidos.py)
            try:
                btn = await page.wait_for_selector("#onetrust-accept-btn-handler", timeout=5000)
                if btn:
                    await btn.click()
                    await asyncio.sleep(2)
            except Exception:
                pass

            await asyncio.sleep(5)

            sections = await page.locator('.h2h__section').all()
            if len(sections) <= section_idx:
                logger.warning(f"   ⚠️ Playwright H2H: solo {len(sections)} secciones para {player_name}")
                return []

            section = sections[section_idx]
            rows = await section.locator('.h2h__row').all()
            logger.info(f"   🌐 Playwright H2H [{player_name}]: {len(rows)} filas (sección {section_idx})")

            matches = []
            for row in rows:
                try:
                    # JS evaluate idéntico al de h2h_extractor.py _parse_player_history()
                    row_data = await row.evaluate('''el => {
                        const date_el = el.querySelector('[data-testid="wcl-stageTime"]');
                        const score_spans = el.querySelectorAll('[data-testid="wcl-tableScore"]');
                        const result = score_spans.length > 0
                            ? Array.from(score_spans).map(s => s.textContent.trim()).join('-')
                            : (el.querySelector('.h2h__result') ? el.querySelector('.h2h__result').textContent.trim() : null);
                        const participants = el.querySelectorAll('[class*="h2h__participant"]:not([class*="participantInner"])');
                        let opponent = null;
                        for (const p of participants) {
                            const nameSpan = p.querySelector('[data-testid="wcl-scores-simple-text-01"]');
                            if (nameSpan && !nameSpan.className.includes('wcl-hasBackground')) {
                                opponent = nameSpan.textContent.trim();
                                break;
                            }
                        }
                        const icon_div = el.querySelector('.h2h__icon > div');
                        const outcome = icon_div && icon_div.className.toLowerCase().includes('win') ? 'Gano' : 'Perdio';
                        const event_el = el.querySelector('.h2h__event');
                        return {
                            date: date_el ? date_el.textContent.trim() : null,
                            result: result,
                            opponent: opponent,
                            outcome: outcome,
                            tournament: event_el ? event_el.textContent.trim() : 'N/A',
                            event_class: event_el ? (event_el.getAttribute('class') || '') : '',
                        };
                    }''')

                    if not row_data.get('date') or not row_data.get('result'):
                        continue

                    ec = row_data.get('event_class', '').lower()
                    surface = 'N/A'
                    if 'hard' in ec:
                        surface = 'Dura'
                    elif 'clay' in ec:
                        surface = 'Arcilla'
                    elif 'grass' in ec:
                        surface = 'Hierba'
                    elif 'indoor' in ec:
                        surface = 'Indoor'

                    matches.append({
                        'fecha': row_data['date'],
                        'oponente': (row_data.get('opponent') or 'N/A').strip(),
                        'resultado': row_data['result'].replace('\n', '-'),
                        'outcome': row_data['outcome'],
                        'torneo': row_data['tournament'].replace('\n', ' '),
                        'ciudad': 'N/A',
                        'pais': 'N/A',
                        'superficie': surface,
                    })
                except Exception:
                    continue

            return matches

        finally:
            await browser.close()


def _fetch_player_history_playwright(match_id: str, player_name: str,
                                      section_idx: int) -> List[Dict]:
    """
    Nodo-49: Sync wrapper para el fallback Playwright.
    Usa ThreadPoolExecutor para correr async desde contexto sync de _process_match().
    timeout=90s por partido (Playwright es ~15-30s en WSL).
    """
    def _run():
        return asyncio.run(_playwright_h2h_async(match_id, player_name, section_idx))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=90)
    except concurrent.futures.TimeoutError:
        logger.warning(f"   ⚠️ Playwright H2H timeout (90s) para {player_name}")
        return []
    except Exception as e:
        logger.warning(f"   ⚠️ Playwright H2H falló para {player_name}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# NODO-45: TEMPORAL HISTORY FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_player_history_temporal(
    player_name: str,
    days_back: int = 7,
) -> List[Dict]:
    """
    Nodo-45 THF: busca el historial de un jugador en h2h_results_enhanced
    de los últimos N días cuando match_id = None o la API retorna vacío.

    Principio: el jugador apareció en una sesión anterior → su historial
    fue extraído correctamente entonces. Usarlo como baseline evita que el
    modelo opere ciego por un fallo de cruce Kambi↔FlashScore hoy.

    Estrategia de matching: reutiliza _name_tokens + _token_in_kb (Nodo-36)
    para matching fuzzy tolerante a acentos y apellidos compuestos.
    Desambiguación por overlap de tokens cuando un token corto aparece en
    ambos jugadores del partido.

    Args:
        player_name : nombre completo del jugador (ej. "Martin Maldonado")
        days_back   : ventana de búsqueda hacia atrás en días (default 7)

    Returns:
        Lista de partidos del historial (mismo formato que _parse_player_history),
        o [] si ningún archivo reciente contiene datos para este jugador.
    """
    reports = Path("reports")
    if not reports.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days_back)

    h2h_files = sorted(
        reports.glob("h2h_results_enhanced_*.json"),
        reverse=True,  # más reciente primero
    )
    recent_files = [
        f for f in h2h_files
        if f.stat().st_mtime >= cutoff.timestamp()
    ]

    if not recent_files:
        return []

    player_tokens = _name_tokens(player_name)
    if not player_tokens:
        return []

    for h2h_file in recent_files:
        try:
            data = json.loads(h2h_file.read_text(encoding="utf-8"))
            matches = data if isinstance(data, list) else data.get("partidos", [])

            for match in matches:
                j1 = match.get("jugador1", "")
                j2 = match.get("jugador2", "")
                j1_lower = j1.lower()
                j2_lower = j2.lower()

                p1_match = any(_token_in_kb(tok, j1_lower) for tok in player_tokens)
                p2_match = any(_token_in_kb(tok, j2_lower) for tok in player_tokens)

                # Desambiguar cuando un token corto aparece en ambos nombres
                if p1_match and p2_match:
                    j1_tokens = _name_tokens(j1)
                    j2_tokens = _name_tokens(j2)
                    overlap1 = sum(1 for t in player_tokens if t in j1_tokens)
                    overlap2 = sum(1 for t in player_tokens if t in j2_tokens)
                    if overlap1 >= overlap2:
                        p2_match = False
                    else:
                        p1_match = False

                if p1_match:
                    key = j1.replace(" ", "_").replace(".", "")
                    hist = match.get(f"historial_{key}", [])
                    if hist:
                        logger.info(
                            f"   📚 THF {h2h_file.name}: "
                            f"{len(hist)} partidos para {player_name}"
                        )
                        return hist
                elif p2_match:
                    key = j2.replace(" ", "_").replace(".", "")
                    hist = match.get(f"historial_{key}", [])
                    if hist:
                        logger.info(
                            f"   📚 THF {h2h_file.name}: "
                            f"{len(hist)} partidos para {player_name}"
                        )
                        return hist
        except Exception:
            continue

    return []


# ══════════════════════════════════════════════════════════════════════════════
# PROCESADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class NinjaH2HExtractor:
    """
    Extractor H2H usando la API Ninja de FlashScore.

    Alternativa rápida a Playwright — produce el mismo formato de salida
    que H2HExtractor para ser consumido por edge_calculator.py.

    Speedup: ~0.5s/partido vs 2-3 min/partido con Playwright.
    Limitación: No extrae cuotas actualizadas (usa las del PASO 1).
    """

    def __init__(self):
        from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer

        self.ranking_manager = RankingManager()
        self.elo_system = EloRatingSystem()
        self.rivalry_analyzer = RivalryAnalyzer(self.ranking_manager, self.elo_system)

        self.all_results = []
        self.all_tournaments = False

    def load_matches(self, json_file: Optional[str] = None) -> bool:
        """Cargar partidos desde JSON (misma lógica que H2HExtractor)."""
        from .file_utils import select_best_json_file
        from .data_parser import DataParser

        self.data_parser = DataParser()

        if not json_file:
            json_file = select_best_json_file(
                directory="data",
                pattern="zita_tennis_matches_*.json"
            )

        if not json_file:
            logger.error("❌ No se pudo seleccionar archivo JSON")
            return False

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.input_file = str(json_file)
            logger.info(f"📂 Cargando partidos desde: {json_file}")

            all_matches = []
            if isinstance(data, dict):
                for tournament_name, matches_in_tournament in data.items():
                    if isinstance(matches_in_tournament, list):
                        for match in matches_in_tournament:
                            if isinstance(match, dict):
                                info = self.data_parser.extract_tournament_info(tournament_name)
                                match.update({
                                    'torneo_nombre': info['nombre'],
                                    'tipo_cancha': match.get('superficie') or info['superficie'],
                                    'pais': info['pais'],
                                    'torneo_completo': info['completo']
                                })
                                all_matches.append(match)
            elif isinstance(data, list):
                all_matches = data

            valid_matches = [m for m in all_matches if m.get('match_url')]

            if not valid_matches:
                logger.error("❌ No se encontraron partidos con match_url válidas")
                return False

            # Filtrar singles con cuotas
            def es_singles_cuadro_principal(match: dict) -> bool:
                url = match.get('match_url', '')
                part = url.split('/match/tennis/')[-1].rstrip('/') if '/match/tennis/' in url else ''
                es_singles = '-' in part and len(part) > 20
                tiene_cuotas = match.get('cuota1') is not None
                return es_singles and tiene_cuotas

            if self.all_tournaments:
                target_matches = [m for m in valid_matches if es_singles_cuadro_principal(m)]
                logger.info(f"   🌍 Modo multi-torneo: {len(valid_matches)} válidos → {len(target_matches)} individuales con cuotas")
            else:
                roland_garros = [m for m in valid_matches
                                 if 'French Open' in m.get('torneo_completo', '')
                                 and 'Qualification' not in m.get('torneo_completo', '')]
                if roland_garros:
                    singles = [m for m in roland_garros if es_singles_cuadro_principal(m)]
                    target_matches = singles if singles else roland_garros
                else:
                    target_matches = valid_matches

            self.matches_queue = target_matches
            logger.info(f"✅ Cola: {len(self.matches_queue)} partidos para API Ninja")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando JSON: {e}")
            return False

    def run(self) -> None:
        """Procesar todos los partidos via API Ninja."""
        logger.info(f"🚀 INICIANDO EXTRACCIÓN VIA API NINJA — {len(self.matches_queue)} partidos")
        logger.info("=" * 80)

        successful = 0
        failed = 0
        start_time = time.time()

        for idx, match_data in enumerate(self.matches_queue, 1):
            p1 = match_data.get('jugador1', 'N/A')
            p2 = match_data.get('jugador2', 'N/A')
            logger.info(f"⚙️ [{idx}/{len(self.matches_queue)}] {p1} vs {p2}")

            if self._process_match(match_data):
                successful += 1
            else:
                failed += 1

            # Rate limiting
            if idx < len(self.matches_queue):
                time.sleep(DELAY_ENTRE_REQUESTS)

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"🏁 COMPLETADO en {elapsed:.1f}s")
        logger.info(f"   ✅ Exitosos: {successful}")
        logger.info(f"   ❌ Fallidos: {failed}")
        if successful > 0:
            logger.info(f"   ⚡ Promedio: {elapsed/successful:.2f}s/partido")

    def _process_match(self, match_data: Dict) -> bool:
        """Procesar un partido individual via API."""
        # Nodo-31: Tier4 ronda_futura — match_id es proxy de otro partido.
        # Usar ese match_id contamina historial con partidos programados.
        # Skip API call y construir resultado solo con datos ya extraídos.
        if match_data.get('ronda_futura'):
            logger.info(f"   ⏭️ Ronda futura — skip API Ninja (match_id proxy, riesgo leakage)")
            return self._process_ronda_futura(match_data)

        match_url = match_data.get('match_url', '')
        match_id = extract_match_id_from_url(match_url)
        p1 = match_data.get('jugador1', 'N/A')
        p2 = match_data.get('jugador2', 'N/A')

        if not match_id:
            logger.warning(f"   ⚠️ No se pudo extraer match_id de: {match_url}")
            # D45-04 (Nodo-45): Temporal History Fallback — buscar en archivos h2h recientes
            p1_history = _lookup_player_history_temporal(p1)
            p2_history = _lookup_player_history_temporal(p2)
            if not p1_history and not p2_history:
                logger.warning(f"   ⚠️ Sin match_id y sin historial temporal — omitido")
                return False
            logger.info(f"   📚 THF activo: {p1}={len(p1_history)} | {p2}={len(p2_history)}")
            return self._analyze_and_consolidate(match_data, p1, p2, p1_history, p2_history, [])

        # Fetch H2H data from API
        raw = fetch_h2h_from_api(match_id)
        if not raw:
            return False

        # Parse response
        records = _parse_sections(raw)
        if not records:
            logger.warning(f"   ⚠️ Respuesta vacía para match {match_id}")
            return False

        # Split into 3 blocks
        block1_records, block2_records, h2h_records = _split_into_h2h_blocks(records)

        # ── Asignación inteligente de bloques (Nodo-31) ──
        # Cuando el match_id es proxy (Tier4: URL dice Miroshnichenko-Carnicella
        # pero el partido real es Carnicella-Ekstrand), Block1 ≠ jugador1.
        # Usar KB headers para identificar qué bloque pertenece a quién.
        # Para nombres compuestos (ej: "Davidovich Fokina") FlashScore usa solo el
        # primer apellido en el KB header → probar todos los tokens del nombre.
        p1_tokens = _name_tokens(p1)
        p2_tokens = _name_tokens(p2)

        main_kbs = [rec.get('KB', '') for rec in records
                    if 'KB' in rec and _is_main_section_kb(rec)]

        p1_in_block1 = bool(p1_tokens) and any(_token_in_kb(tok, kb) for tok in p1_tokens for kb in main_kbs[:1])
        p1_in_block2 = bool(p1_tokens) and any(_token_in_kb(tok, kb) for tok in p1_tokens for kb in main_kbs[1:2])
        p2_in_block1 = bool(p2_tokens) and any(_token_in_kb(tok, kb) for tok in p2_tokens for kb in main_kbs[:1])
        p2_in_block2 = bool(p2_tokens) and any(_token_in_kb(tok, kb) for tok in p2_tokens for kb in main_kbs[1:2])

        p1_found = p1_in_block1 or p1_in_block2
        p2_found = p2_in_block1 or p2_in_block2

        if p1_found and p2_found:
            # Ambos jugadores en la API — asignar por KB header
            if p1_in_block2 and not p1_in_block1:
                logger.info(f"   🔀 Bloques invertidos: Block1→{p2}, Block2→{p1}")
                p1_records, p2_records = block2_records, block1_records
            else:
                p1_records, p2_records = block1_records, block2_records
        elif p1_found and not p2_found:
            # Solo P1 en la API (proxy) — P2 necesita su propio match_id
            if p1_in_block1:
                p1_records = block1_records
            else:
                p1_records = block2_records
            # Buscar historial de P2 con match_id_j2 si existe
            match_id_j2 = match_data.get('match_id_j2') or extract_match_id_from_url(
                match_data.get('match_url_j2', ''))
            if match_id_j2:
                logger.info(f"   📡 {p2} no en API proxy — buscando en match_id_j2={match_id_j2}")
                p2_records = None  # señal de que usamos proxy separado
            else:
                logger.info(f"   ⚠️ {p2} no en API proxy y sin match_id_j2 — historial vacío")
                p2_records = []
        elif p2_found and not p1_found:
            # Solo P2 en la API — P1 necesita su propio match_id
            if p2_in_block1:
                p2_records = block1_records
            else:
                p2_records = block2_records
            match_id_j1_alt = match_data.get('match_id_j2') or extract_match_id_from_url(
                match_data.get('match_url_j2', ''))
            if match_id_j1_alt:
                logger.info(f"   📡 {p1} no en API proxy — buscando en match_id_j2={match_id_j1_alt}")
                p1_records = None
            else:
                logger.info(f"   ⚠️ {p1} no en API proxy y sin match_id_j2 — historial vacío")
                p1_records = []
        else:
            # Ninguno encontrado por KB — usar asignación por defecto (Block1→P1)
            p1_records, p2_records = block1_records, block2_records

        # Parse each block — si es None, buscar via proxy separado
        if p1_records is None:
            alt_id = match_data.get('match_id_j2') or extract_match_id_from_url(
                match_data.get('match_url_j2', ''))
            p1_history = self._fetch_player_history_from_proxy(
                alt_id, p1, match_data.get('match_url_j2', '')) if alt_id else []
        else:
            p1_history = _parse_player_history(p1_records, p1)

        if p2_records is None:
            alt_id = match_data.get('match_id_j2') or extract_match_id_from_url(
                match_data.get('match_url_j2', ''))
            p2_history = self._fetch_player_history_from_proxy(
                alt_id, p2, match_data.get('match_url_j2', '')) if alt_id else []
        else:
            p2_history = _parse_player_history(p2_records, p2)

        # ── Nodo-45 THF Punto B: suplementar historiales vacíos desde sesiones anteriores ──
        if not p1_history:
            _t = _lookup_player_history_temporal(p1)
            if _t:
                logger.info(f"   📚 THF suplementa {p1}: {len(_t)} partidos")
                p1_history = _t
        if not p2_history:
            _t = _lookup_player_history_temporal(p2)
            if _t:
                logger.info(f"   📚 THF suplementa {p2}: {len(_t)} partidos")
                p2_history = _t

        # ── Nodo-49: Playwright H2H Fallback — si API + THF fallaron y hay match_id ──
        if not p1_history and match_id:
            logger.info(f"   🌐 Playwright fallback P1: {p1} (n_h2h=0, THF vacío)")
            _pw = _fetch_player_history_playwright(match_id, p1, section_idx=0)
            if _pw:
                logger.info(f"   ✅ Playwright recuperó {len(_pw)} partidos para {p1}")
                p1_history = _pw
        if not p2_history and match_id:
            logger.info(f"   🌐 Playwright fallback P2: {p2} (n_h2h=0, THF vacío)")
            _pw = _fetch_player_history_playwright(match_id, p2, section_idx=1)
            if _pw:
                logger.info(f"   ✅ Playwright recuperó {len(_pw)} partidos para {p2}")
                p2_history = _pw

        h2h_matches = _parse_direct_h2h(h2h_records, p1, p2)

        logger.info(f"   📊 {p1}: {len(p1_history)} partidos | {p2}: {len(p2_history)} | H2H: {len(h2h_matches)}")

        return self._analyze_and_consolidate(match_data, p1, p2, p1_history, p2_history, h2h_matches)

    def _analyze_and_consolidate(self, match_data: Dict, p1: str, p2: str,
                                  p1_history: List[Dict], p2_history: List[Dict],
                                  h2h_matches: List[Dict]) -> bool:
        """
        D45-03 (Nodo-45): Enriquece historiales, calcula ELO/form/rivalry y
        consolida el resultado.  Extraído de _process_match() para poder ser
        llamado también desde el Temporal History Fallback (Point A) y desde
        el bloque de suplemento de historial vacío (Point B).
        """
        # Enrich with rankings
        p1_hist = self._enrich_history(p1_history)
        p2_hist = self._enrich_history(p2_history)

        # Kambi ranking fallback — rellena jugadores ITF/Challenger ausentes del archivo ATP
        self._inject_kambi_ranking(p1, match_data.get('ranking1'))
        self._inject_kambi_ranking(p2, match_data.get('ranking2'))

        # Form analysis
        p1_form = self._analyze_form(p1_hist, p1)
        p2_form = self._analyze_form(p2_hist, p2)

        # ELO
        p1_elo = self.rivalry_analyzer.calculate_elo_from_history(p1, p1_hist)
        p2_elo = self.rivalry_analyzer.calculate_elo_from_history(p2, p2_hist)

        # Rivalry analysis
        current_context = {
            'country': match_data.get('pais', 'N/A'),
            'surface': match_data.get('tipo_cancha', 'N/A')
        }

        rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
            p1_hist, p2_hist, p1, p2, p1_form, p2_form,
            h2h_matches, current_context,
            p1_elo, p2_elo,
            match_data.get('torneo_completo', ''),
            None  # no optimized weights
        )

        # Consolidate
        result = self._consolidate_result(
            match_data, p1_hist, p2_hist, h2h_matches,
            rivalry_analysis, p1_form, p2_form, p1_elo, p2_elo
        )
        self.all_results.append(result)

        pred = rivalry_analysis.get('prediction', {})
        logger.info(f"   🎯 Predicción: {pred.get('favored_player', '?')} ({pred.get('confidence', 0)}%)")

        return True

    def _fetch_player_history_from_proxy(self, match_id: str, player_name: str,
                                         match_url: str = '') -> List[Dict]:
        """
        Obtiene el historial de UN jugador desde un match_id proxy.

        Determina qué bloque (P1 o P2 de la API) corresponde al jugador
        leyendo el encabezado KB de cada sección. El header dice:
          "Últimos partidos: Eala A." → este bloque es de Eala.

        Fallback: si el header no contiene el nombre, usa el slug de la URL.
        """
        raw = fetch_h2h_from_api(match_id)
        if not raw:
            return []
        records = _parse_sections(raw)
        if not records:
            return []

        surname = player_name.split()[-1].lower() if player_name else ''

        # Paso 1: encontrar headers KB principales y ver cuál contiene el nombre
        main_kbs = [(i, rec.get('KB', '')) for i, rec in enumerate(records)
                     if 'KB' in rec and _is_main_section_kb(rec)]

        player_in_block1 = False
        player_in_block2 = False
        if len(main_kbs) >= 2:
            player_in_block1 = surname in main_kbs[0][1].lower()
            player_in_block2 = surname in main_kbs[1][1].lower()

        p1_records, p2_records, _ = _split_into_h2h_blocks(records)

        # Decisión por header KB (más confiable)
        if player_in_block2 and not player_in_block1:
            logger.info(f"   🔀 Bloque 2 (header KB) para {player_name}")
            return _parse_player_history(p2_records, player_name)
        if player_in_block1 and not player_in_block2:
            logger.info(f"   🔀 Bloque 1 (header KB) para {player_name}")
            return _parse_player_history(p1_records, player_name)

        # Paso 2 fallback: slug de la URL del proxy
        if match_url and surname:
            # URL format: .../slug1-slug2/match_id/#/h2h
            # slug1 = P1 en la API, slug2 = P2 en la API
            url_path = match_url.split('/match/tennis/')[-1] if '/match/tennis/' in match_url else ''
            if url_path:
                slug_part = url_path.split('/')[0]  # "monfils-elina-eala-alexandra"
                parts = slug_part.split('-')
                mid = len(parts) // 2
                slug1 = '-'.join(parts[:mid]).lower()
                slug2 = '-'.join(parts[mid:]).lower()
                if surname in slug2:
                    logger.info(f"   🔀 Bloque 2 (URL slug) para {player_name}")
                    return _parse_player_history(p2_records, player_name)
                elif surname in slug1:
                    logger.info(f"   🔀 Bloque 1 (URL slug) para {player_name}")
                    return _parse_player_history(p1_records, player_name)

        # Paso 3 último recurso: el que tenga más partidos
        hist1 = _parse_player_history(p1_records, player_name)
        hist2 = _parse_player_history(p2_records, player_name)
        if len(hist2) > len(hist1):
            logger.info(f"   🔀 Bloque 2 (más partidos: {len(hist2)} vs {len(hist1)}) para {player_name}")
            return hist2
        logger.info(f"   🔀 Bloque 1 (default: {len(hist1)} partidos) para {player_name}")
        return hist1

    def _process_ronda_futura(self, match_data: Dict) -> bool:
        """
        Procesar partido de ronda futura con historiales reales (Nodo-31 v2).

        Usa DOS match_ids proxy separados (match_id para P1, match_id_j2 para P2)
        para obtener el historial correcto de CADA jugador sin contaminación cruzada.
        Si solo hay un match_id (match file legacy), obtiene solo P1 y deja P2 vacío.
        """
        p1 = match_data.get('jugador1', 'N/A')
        p2 = match_data.get('jugador2', 'N/A')

        # Inyectar rankings de Kambi como fallback
        self._inject_kambi_ranking(p1, match_data.get('ranking1'))
        self._inject_kambi_ranking(p2, match_data.get('ranking2'))

        # P1: historial desde su match_id proxy
        match_id_j1 = match_data.get('match_id') or extract_match_id_from_url(match_data.get('match_url', ''))
        if match_id_j1:
            logger.info(f"   📡 Obteniendo historial de {p1} via proxy {match_id_j1}")
            p1_hist_raw = self._fetch_player_history_from_proxy(match_id_j1, p1, match_data.get('match_url', ''))
        else:
            p1_hist_raw = []

        # P2: historial desde su propio match_id proxy (match_id_j2) — evita contaminación
        match_id_j2 = match_data.get('match_id_j2') or extract_match_id_from_url(match_data.get('match_url_j2', ''))
        if match_id_j2:
            logger.info(f"   📡 Obteniendo historial de {p2} via proxy {match_id_j2}")
            p2_hist_raw = self._fetch_player_history_from_proxy(match_id_j2, p2, match_data.get('match_url_j2', ''))
        else:
            logger.info(f"   ⚠️ Sin match_id_j2 para {p2} — historial vacío (solo ranking)")
            p2_hist_raw = []

        # H2H directo vacío — no hay datos confiables sin match_id real del partido
        h2h_matches = []

        p1_hist = self._enrich_history(p1_hist_raw)
        p2_hist = self._enrich_history(p2_hist_raw)

        logger.info(f"   📊 Ronda futura: {p1}: {len(p1_hist)} partidos | {p2}: {len(p2_hist)} | H2H: 0")

        p1_form = self._analyze_form(p1_hist, p1)
        p2_form = self._analyze_form(p2_hist, p2)

        p1_elo = self.rivalry_analyzer.calculate_elo_from_history(p1, p1_hist)
        p2_elo = self.rivalry_analyzer.calculate_elo_from_history(p2, p2_hist)

        current_context = {
            'country': match_data.get('pais', 'N/A'),
            'surface': match_data.get('tipo_cancha', 'N/A')
        }

        rivalry_analysis = self.rivalry_analyzer.analyze_rivalry(
            p1_hist, p2_hist, p1, p2, p1_form, p2_form,
            h2h_matches, current_context,
            p1_elo, p2_elo,
            match_data.get('torneo_completo', ''),
            None
        )

        result = self._consolidate_result(
            match_data, p1_hist, p2_hist, h2h_matches,
            rivalry_analysis, p1_form, p2_form, p1_elo, p2_elo
        )
        result['ronda_futura'] = True
        self.all_results.append(result)

        pred = rivalry_analysis.get('prediction', {})
        logger.info(f"   🎯 Ronda futura (solo ranking): {pred.get('favored_player', '?')} ({pred.get('confidence', 0)}%)")

        return True

    def _inject_kambi_ranking(self, player_name: str, ranking) -> None:
        """
        Inyecta el ranking de PASO 1 (Kambi/FlashScore) en RankingManager
        como fallback para jugadores ITF/Challenger no presentes en el archivo
        atp_rankings_complete_*.json.

        Solo actúa cuando el jugador es desconocido (get_player_info retorna None).
        pts_estimate usa log inverso: rank=1→700 | rank=37→210 | rank=100→130 | rank=300→90.
        """
        import math
        if not ranking:
            return
        # Guard optimizada (Nodo-46 fix): el ATP file indexa "Apellido Nombre" pero
        # normalize_name('Daniil Glinka') produce 'daniil glinka' — mismatch que
        # causaba que Kambi estimate sobreescribiera el ranking real ATP.
        #
        # Fast path O(1): chequea key directo + key invertido (cubre 95% ATP/WTA).
        # Slow path: get_player_info() con intelligent matching para nombres compuestos
        # (Davidovich Fokina, Moro Canas, etc.) — solo si fast path falla.
        normalized = self.ranking_manager.normalize_name(player_name)
        parts = normalized.split()
        reversed_key = ' '.join(reversed(parts)) if len(parts) == 2 else None
        rd = self.ranking_manager.rankings_data
        if rd.get(normalized) or (reversed_key and rd.get(reversed_key)):
            return  # encontrado en O(1) — no sobreescribir
        if self.ranking_manager.get_player_info(player_name):
            return  # encontrado via intelligent matching — no sobreescribir
        pts_estimate = max(1, round(700 / math.log1p(ranking)))
        player_entry = {
            'name':               player_name,
            'ranking_position':   int(ranking),
            'ranking_points':     pts_estimate,
            'prox_points':        pts_estimate,
            'max_points':         pts_estimate,
            'defense_points':     0,
            'nationality':        'N/A',
            '_source':            'kambi_fallback',
        }
        # get_player_info() busca en atp_players/wta_players — inyectar en ambos
        self.ranking_manager.rankings_data[normalized] = player_entry
        self.ranking_manager.atp_players[normalized]   = player_entry

    def _enrich_history(self, history: List[Dict]) -> List[Dict]:
        """Enriquecer historial con rankings."""
        enriched = []
        for match in history:
            enriched_match = match.copy()
            opponent = match.get('oponente', '')
            if opponent and opponent != 'N/A':
                rank = self.ranking_manager.get_player_ranking(opponent)
                enriched_match['opponent_ranking'] = rank
                enriched_match['opponent_weight'] = self.rivalry_analyzer.calculate_base_opponent_weight(rank)
            enriched.append(enriched_match)
        return enriched

    def _analyze_form(self, history: List[Dict], player: str, recent: int = 20) -> Optional[Dict]:
        """Análisis de forma reciente."""
        if not history:
            return None

        recent_matches = history[:recent]
        wins = sum(1 for m in recent_matches if m.get('outcome', '').lower() in ['ganó', 'win'])
        total = len(recent_matches)

        # Racha actual
        streak_count = 0
        streak_type = None
        for m in recent_matches:
            won = m.get('outcome', '').lower() in ['ganó', 'win']
            if streak_type is None:
                streak_type = 'win' if won else 'loss'
                streak_count = 1
            elif (streak_type == 'win' and won) or (streak_type == 'loss' and not won):
                streak_count += 1
            else:
                break

        win_pct = round((wins / total * 100), 1) if total > 0 else 0

        if win_pct >= 75:
            form_status = 'Excelente'
        elif win_pct >= 60:
            form_status = 'Buena'
        elif win_pct >= 40:
            form_status = 'Regular'
        else:
            form_status = 'Mala'

        return {
            'player_name': player,
            'recent_matches_count': total,
            'wins': wins,
            'losses': total - wins,
            'win_percentage': win_pct,
            'current_streak_count': streak_count,
            'current_streak_type': 'victorias' if streak_type == 'win' else 'derrotas',
            'form_status': form_status,
            'last_match_date': recent_matches[0].get('fecha', 'N/A') if recent_matches else 'N/A'
        }

    def _consolidate_result(self, match_data: Dict, p1_hist: List, p2_hist: List,
                           h2h_matches: List, rivalry: Dict,
                           p1_form: Optional[Dict], p2_form: Optional[Dict],
                           p1_elo: float, p2_elo: float) -> Dict:
        """Consolidar resultado — mismo formato que H2HExtractor."""
        p1 = match_data['jugador1']
        p2 = match_data['jugador2']
        p1_key = p1.replace(' ', '_').replace('.', '')
        p2_key = p2.replace(' ', '_').replace('.', '')

        p1_metrics = self.rivalry_analyzer.get_ranking_metrics(p1)
        p2_metrics = self.rivalry_analyzer.get_ranking_metrics(p2)

        return {
            'match_number': len(self.all_results) + 1,
            'match_id': match_data.get('match_id') or extract_match_id_from_url(match_data.get('match_url', '')),
            'jugador1': p1,
            'jugador1_nacionalidad': rivalry.get('player1_nationality', 'N/A'),
            'jugador2': p2,
            'jugador2_nacionalidad': rivalry.get('player2_nationality', 'N/A'),
            'torneo_nombre': match_data.get('torneo_nombre', 'N/A'),
            'tipo_cancha': match_data.get('tipo_cancha', 'N/A'),
            'torneo_completo': match_data.get('torneo_completo', 'N/A'),
            'cuota1': match_data.get('cuota1', 'N/A'),
            'cuota2': match_data.get('cuota2', 'N/A'),
            'cuota_es_real': match_data.get('cuota_es_real', True),
            'match_url': match_data.get('match_url', ''),
            f'historial_{p1_key}': p1_hist,
            f'historial_{p2_key}': p2_hist,
            'enfrentamientos_directos': h2h_matches,
            'estadisticas': {
                f'partidos_{p1_key}': len(p1_hist),
                f'partidos_{p2_key}': len(p2_hist),
                'enfrentamientos_totales': len(h2h_matches)
            },
            'data_quality': {
                'historial_extraido_p1': len(p1_hist) > 0,
                'historial_extraido_p2': len(p2_hist) > 0,
                'n_partidos_p1': len(p1_hist),
                'n_partidos_p2': len(p2_hist),
            },
            'ranking_analysis': {
                f'{p1_key}_ranking': rivalry['player1_rank'],
                f'{p2_key}_ranking': rivalry['player2_rank'],
                'common_opponents_count': rivalry['common_opponents_count'],
                'p1_rivalry_score': rivalry['p1_rivalry_score'],
                'p2_rivalry_score': rivalry['p2_rivalry_score'],
                'prediction': rivalry['prediction'],
                f'{p1_key}_metrics': p1_metrics,
                f'{p2_key}_metrics': p2_metrics,
                f'{p1_key}_elo': p1_elo,
                f'{p2_key}_elo': p2_elo
            },
            'form_analysis': {
                f'{p1_key}_form': p1_form,
                f'{p2_key}_form': p2_form,
            },
            'surface_analysis': {
                f'{p1_key}_surface_stats': rivalry.get('p1_surface_stats'),
                f'{p2_key}_surface_stats': rivalry.get('p2_surface_stats'),
            },
            'location_analysis': {
                f'{p1_key}_location_stats': rivalry.get('p1_location_stats'),
                f'{p2_key}_location_stats': rivalry.get('p2_location_stats'),
            },
            'common_opponents_detailed': self._build_common_opponents_detailed(
                rivalry.get('player1_advantages', []),
                rivalry.get('player2_advantages', []),
                p1, p2
            ),
            'markov_analysis': rivalry.get('markov_analysis'),
        }

    def _build_common_opponents_detailed(self, p1_advantages, p2_advantages, p1_name, p2_name):
        """Transforma advantages al formato esperado."""
        def normalize_score_for_player(score_str, player_won):
            """Devuelve el marcador desde la perspectiva del jugador (mis_sets-oponente_sets).
            Solo invierte si el score almacenado no coincide con el resultado:
            ganador debe tener más sets, perdedor menos."""
            if not score_str or '-' not in str(score_str):
                return score_str
            parts = str(score_str).split('-')
            try:
                left, right = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                return score_str
            player_has_more = left > right
            if player_won == player_has_more:
                return score_str  # ya está en perspectiva del jugador
            return f"{parts[1]}-{parts[0]}"  # invertir para corregir perspectiva

        def build_player_result(score_str, won, date_str, surface_str=''):
            outcome = 'Ganó' if won else 'Perdió'
            score = normalize_score_for_player(score_str, won)
            return {'outcome': outcome, 'score': score, 'date': date_str or 'N/A', 'surface': surface_str or ''}

        result = []
        for adv in (p1_advantages or []):
            p1_won = adv.get('p1_won', True)
            p2_won = adv.get('p2_won', False)
            result.append({
                'opponent_name': adv.get('opponent', 'N/A'),
                'opponent_ranking': adv.get('opponent_rank'),
                'advantage_for': p1_name,
                'reason': adv.get('reason', ''),
                'weight': adv.get('weight', 0),
                'player1_result': build_player_result(adv.get('player1_result', ''), p1_won, adv.get('player1_date', ''), adv.get('player1_surface', '')),
                'player2_result': build_player_result(adv.get('player2_result', ''), p2_won, adv.get('player2_date', ''), adv.get('player2_surface', '')),
            })
        for adv in (p2_advantages or []):
            p1_won = adv.get('p1_won', False)
            p2_won = adv.get('p2_won', True)
            result.append({
                'opponent_name': adv.get('opponent', 'N/A'),
                'opponent_ranking': adv.get('opponent_rank'),
                'advantage_for': p2_name,
                'reason': adv.get('reason', ''),
                'weight': adv.get('weight', 0),
                'player1_result': build_player_result(adv.get('player1_result', ''), p1_won, adv.get('player1_date', ''), adv.get('player1_surface', '')),
                'player2_result': build_player_result(adv.get('player2_result', ''), p2_won, adv.get('player2_date', ''), adv.get('player2_surface', '')),
            })
        return result

    def save_results(self) -> str:
        """Guardar resultados en el mismo formato que H2HExtractor."""
        # Usar fecha del archivo de entrada para que --tomorrow genere 20260626_*
        _m = re.search(r'(\d{8})_\d{6}', getattr(self, 'input_file', ''))
        match_date = _m.group(1) if _m else datetime.now().strftime('%Y%m%d')
        run_time = datetime.now().strftime('%H%M%S')
        timestamp = f"{match_date}_{run_time}"
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        filename = reports_dir / f"h2h_results_enhanced_{timestamp}.json"

        from analysis.rivalry_analyzer import RIVALRY_VERSION  # Nodo-32 Fase 3
        output_data = {
            'metadata': {
                'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_partidos_procesados': len(self.all_results),
                'version': '4.0_ninja_api',
                'modo': 'api_ninja',
                'rivalry_version': RIVALRY_VERSION,  # Nodo-32 Fase 3: validado por edge_calculator
                'funcionalidades': [
                    'Extracción H2H via API Ninja',
                    'Análisis de rankings ATP',
                    'Sistema ELO calculado',
                    'Rivalidad transitiva',
                    'Predicciones avanzadas',
                    'Análisis de superficie',
                    'Ventaja de localización'
                ]
            },
            'partidos': self.all_results,
            'estadisticas_globales': self._generate_stats()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Resultados guardados en: {filename}")
        return str(filename)

    def _generate_stats(self) -> Dict:
        """Generar estadísticas globales."""
        if not self.all_results:
            return {}

        players = set()
        total_h2h = 0
        for r in self.all_results:
            players.add(r.get('jugador1', ''))
            players.add(r.get('jugador2', ''))
            total_h2h += r.get('estadisticas', {}).get('enfrentamientos_totales', 0)

        return {
            'total_partidos': len(self.all_results),
            'jugadores_unicos': len(players),
            'total_enfrentamientos_directos': total_h2h,
        }

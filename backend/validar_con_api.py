"""
Nodo-05 — Validación Post-Partido con FlashScore Ninja API

Cierra el loop P&L: compara predicciones del modelo con resultados reales.

Flujo:
  1. Lee h2h_results_enhanced_FECHA.json (predicciones)
  2. Consulta dc_1_{event_id} por cada partido con match_id real
  3. Compara prediccion vs resultado real → accuracy
  4. Segmenta accuracy por superficie
  5. Exporta resultados_finales_FECHA.json + calibracion actualizada

API confirmada: dc_1_{event_id} → HTTP 200 para tenis
Auth: X-Fsign: SW9D1eZo | Referer: https://www.flashscore.co/
H2H endpoints → 404 para tenis (Playwright sigue siendo necesario para H2H)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS as HEADERS  # D-17

DELAY_ENTRE_REQUESTS = 0.5   # segundos — no martillar la API
CALIBRACION_FILE = "data/calibracion_edge.json"


# ──────────────────────────────────────────────────────────────────────────────
# Parser del formato propietario FlashScore
# ──────────────────────────────────────────────────────────────────────────────

def parsear_respuesta_flashscore(raw: str) -> dict:
    """
    Convierte el formato propietario KEY÷VALUE¬KEY÷VALUE en un dict.

    Ejemplo (formato real dc_1, verificado 2026-05-29):
        "DA÷3¬DC÷1780057200¬DE÷2¬DF÷0¬DJ÷H¬DV÷2¬~"
        → {'DA': '3', 'DC': '1780057200', 'DE': '2', 'DF': '0', 'DJ': 'H', 'DV': '2'}

    Claves relevantes del endpoint dc_1_{event_id} para tenis:
        DJ = ganador: 'H'=local ganó, 'A'=visitante ganó, ''=no terminado
        DE = sets ganados por local (jugador1)
        DF = sets ganados por visitante (jugador2)
        DC = Unix timestamp del inicio programado
        DV = constante de tipo partido (2=tenis, no es indicador de estado)
    """
    datos: dict = {}
    for par in raw.split('¬'):
        if '÷' in par:
            k, v = par.split('÷', 1)
            datos[k] = v
    return datos


# ──────────────────────────────────────────────────────────────────────────────
# Consulta a la API
# ──────────────────────────────────────────────────────────────────────────────

def obtener_resultado_partido(event_id: str) -> dict:
    """
    Consulta dc_1_{event_id} y extrae ganador + estado del partido.

    Retorna:
        status:        'FT' | 'LIVE' | 'NS' | 'ERROR' | 'UNKNOWN'
        ganador_lado:  'jugador1' | 'jugador2' | None
        sets_local:    str  (número de sets ganados por el local)
        sets_visitante: str
        raw_data:      dict completo para debugging
    """
    if not event_id or event_id in ('tennis', ''):
        return {'status': 'INVALID_ID', 'error': f'match_id inválido: {event_id!r}'}

    url = f"{FLASHSCORE_BASE}/dc_1_{event_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        raw = r.text
    except requests.exceptions.HTTPError as e:
        return {'status': 'ERROR', 'error': f'HTTP {e.response.status_code}'}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

    datos = parsear_respuesta_flashscore(raw)

    # Claves reales del endpoint dc_1 (verificadas 2026-05-29 — Nodo-09):
    # DJ = ganador: 'H'=local ganó, 'A'=visitante ganó, ''=no terminado
    # DE = sets ganados por local, DF = sets ganados por visitante
    # DC = Unix timestamp del inicio programado (para distinguir NS vs LIVE)
    dj = datos.get('DJ', '')

    if dj in ('H', 'A'):
        home_sets = datos.get('DE', '0')
        away_sets = datos.get('DF', '0')
        ganador_lado = 'jugador1' if dj == 'H' else 'jugador2'
        return {
            'status': 'FT',
            'ganador_lado': ganador_lado,
            'sets_local': home_sets,
            'sets_visitante': away_sets,
            'raw_data': datos,
        }

    # DJ vacío: no iniciado o en curso — usar DC para distinguir
    try:
        dc_ts = int(datos.get('DC', '0'))
        if dc_ts and datetime.fromtimestamp(dc_ts) > datetime.now():
            return {'status': 'NS', 'raw_data': datos}
    except (ValueError, TypeError):
        pass

    return {'status': 'LIVE', 'score_parcial': datos}


# ──────────────────────────────────────────────────────────────────────────────
# Detección de orden home/away FlashScore vs Kambi (Nodo-05 fix)
# ──────────────────────────────────────────────────────────────────────────────

def _detectar_jugador_home_fs(match_url: str, jugador1: str, jugador2: str) -> str:
    """
    Detecta qué jugador del partido es el 'home' en el endpoint dc_1 de FlashScore.

    El problema: Kambi y FlashScore pueden ordenar a los jugadores de forma distinta.
    En dc_1, DJ='H' significa que el jugador HOME de FlashScore ganó.
    Si el orden Kambi != orden FS, 'jugador1' en nuestros datos es el 'away' en FS,
    por lo que DJ='H' se mapearía al nombre incorrecto.

    Solución: la match_url tiene el formato
        /tennis/{slug1}-{slug2}/{match_id}/
    donde slug1 corresponde al jugador HOME de FlashScore. Buscamos qué jugador
    tiene sus tokens de nombre apareciendo más temprano en el slug combinado.

    Retorna: 'jugador1' si jugador1 es FS-home, 'jugador2' si jugador2 es FS-home.
    Ante empate o sin URL, retorna 'jugador1' (comportamiento previo conservado).
    """
    if not match_url:
        return 'jugador1'
    m = re.search(r'/tennis/([^/]+)/[^/]+/', match_url)
    if not m:
        return 'jugador1'
    slug_combined = m.group(1).lower()

    def earliest_token_pos(name: str) -> int:
        tokens = name.lower().split()
        positions = [slug_combined.find(t) for t in tokens if slug_combined.find(t) >= 0]
        return min(positions) if positions else 9999

    pos1 = earliest_token_pos(jugador1)
    pos2 = earliest_token_pos(jugador2)
    return 'jugador2' if pos2 < pos1 else 'jugador1'


# ──────────────────────────────────────────────────────────────────────────────
# Slug validation — detect wrong match_id from Kambi Tier 3 matcher
# ──────────────────────────────────────────────────────────────────────────────

def _validar_slug_ambos_jugadores(match_url: str, jugador1: str, jugador2: str) -> bool:
    """
    Check if the URL slug contains name tokens from BOTH players.
    Returns True if both players are represented, False if slug is wrong.

    The Kambi Tier 3 substring matcher can match 2+ tokens from the SAME player's
    compound name (e.g., "andrade" + "silva" from "Lucas Andrade Da Silva"),
    accepting a match against a DIFFERENT opponent.  This validator catches that.
    """
    if not match_url:
        return True  # no URL to validate — assume OK

    m = re.search(r'/tennis/([^/]+)/[^/]+/', match_url)
    if not m:
        return True  # can't parse slug — assume OK

    slug = m.group(1).lower()

    def _has_token(name: str) -> bool:
        tokens = name.lower().split()
        # Filter short tokens (<=2 chars) — "da", "de", "van" are too common
        tokens = [t for t in tokens if len(t) >= 3]
        if not tokens:
            return False
        matched = sum(1 for t in tokens if t in slug)
        # For compound names (3+ tokens), require >=2 matches to avoid
        # false positives from shared first names (e.g. "juan" in slug
        # matching "Juan Bautista Torres" when slug is actually "estevez-juan")
        if len(tokens) >= 3:
            return matched >= 2
        return matched >= 1

    return _has_token(jugador1) and _has_token(jugador2)


# ──────────────────────────────────────────────────────────────────────────────
# FlashScore feed lookup — fallback when match_id is wrong
# Adopted from resultados_finales.py
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normalize name: lowercase, no accents, no punctuation."""
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _parse_nombre(nombre: str) -> Tuple[str, str]:
    """Extract (surname, initial) from any name format."""
    normalized = _normalize_name(nombre)
    raw = normalized.split()
    if not raw:
        return ("", "")
    while raw and raw[-1] in _SUFFIXES:
        raw.pop()
    if not raw:
        return ("", "")
    iniciales = []
    apellido_parts = []
    for token in raw:
        if len(token) <= 1:
            iniciales.append(token)
        else:
            apellido_parts.append(token)
    if not apellido_parts:
        return (raw[-1], raw[0][0])
    apellido = apellido_parts[-1]
    inicial = iniciales[0][0] if iniciales else apellido_parts[0][0]
    return (apellido, inicial)


def _build_match_key(name1: str, name2: str) -> Tuple[str, str, str, str]:
    """Match key: (surname1, ini1, surname2, ini2) sorted."""
    a1, i1 = _parse_nombre(name1)
    a2, i2 = _parse_nombre(name2)
    if a1 <= a2:
        return (a1, i1, a2, i2)
    return (a2, i2, a1, i1)


def _fetch_finished_matches_feed() -> Dict[Tuple, Dict]:
    """
    Fetch FlashScore feed for today/yesterday/day-before to get
    finished matches with their correct match_id and slug.

    Returns:
        Dict mapping match_key -> {match_id, jugador1_fs, jugador2_fs, slug, ...}
    """
    lookup: Dict[Tuple, Dict] = {}

    for day_offset in [0, -1, 1, -2]:
        ep = f"f_2_{day_offset}_2_es_1"
        url = f"{FLASHSCORE_BASE}/{ep}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        sections = resp.text.split("~")
        current_tournament = ""

        for sec in sections:
            fields: dict = {}
            for pair in sec.split("¬"):
                if "÷" in pair:
                    k, v = pair.split("÷", 1)
                    fields[k] = v

            if "ZA" in fields:
                current_tournament = fields["ZA"]
                continue

            if "AA" not in fields:
                continue

            if "DOBLES" in current_tournament or "DOUBLES" in current_tournament:
                continue

            j1 = fields.get("AE", "")
            j2 = fields.get("AF", "")
            match_id = fields.get("AA", "")

            if not j1 or not j2 or not match_id:
                continue

            entry = {
                "match_id": match_id,
                "jugador1_fs": j1,
                "jugador2_fs": j2,
                "torneo_fs": current_tournament,
            }

            key = _build_match_key(j1, j2)
            lookup[key] = entry

            # Also index by surnames only (fallback tier 2)
            a1, _ = _parse_nombre(j1)
            a2, _ = _parse_nombre(j2)
            key_apellido = (min(a1, a2), "", max(a1, a2), "")
            if key_apellido not in lookup:
                lookup[key_apellido] = entry

    return lookup


def _buscar_en_feed(jugador1: str, jugador2: str,
                    feed_lookup: Dict[Tuple, Dict]) -> Optional[Dict]:
    """
    Search for correct match in FlashScore feed by player name matching.
    Returns the feed entry dict (with match_id, jugador1_fs, jugador2_fs) or None.

    3-tier matching: exact key, surnames only, substring overlap.
    """
    # Tier 1: exact key
    key = _build_match_key(jugador1, jugador2)
    if key in feed_lookup:
        return feed_lookup[key]

    # Tier 2: surnames only
    a1, _ = _parse_nombre(jugador1)
    a2, _ = _parse_nombre(jugador2)
    key_apellido = (min(a1, a2), "", max(a1, a2), "")
    if key_apellido in feed_lookup:
        return feed_lookup[key_apellido]

    # Tier 3: substring overlap (>=4 chars)
    for fkey, fval in feed_lookup.items():
        fa1, _, fa2, _ = fkey
        if not fa1 or not fa2:
            continue
        a1_match = (a1 and fa1 and (a1 in fa1 or fa1 in a1) and len(min(a1, fa1, key=len)) >= 4)
        a2_match = (a2 and fa2 and (a2 in fa2 or fa2 in a2) and len(min(a2, fa2, key=len)) >= 4)
        if a1_match and a2_match:
            return fval
        # Cross (home/away flipped)
        a1x = (a1 and fa2 and (a1 in fa2 or fa2 in a1) and len(min(a1, fa2, key=len)) >= 4)
        a2x = (a2 and fa1 and (a2 in fa1 or fa1 in a2) and len(min(a2, fa1, key=len)) >= 4)
        if a1x and a2x:
            return fval

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Validación individual (testeable sin I/O)
# ──────────────────────────────────────────────────────────────────────────────

def validar_partido_individual(partido: dict, resultado_api: Optional[dict] = None,
                               feed_lookup: Optional[Dict[Tuple, Dict]] = None) -> Optional[dict]:
    """
    Valida UN partido comparando predicción con resultado real.

    Args:
        partido:       dict del h2h_results_enhanced (con ranking_analysis, match_id, etc.)
        resultado_api: si se pasa, evita la llamada HTTP (útil en tests)
        feed_lookup:   pre-fetched FlashScore feed for fallback when match_id is wrong

    Retorna None si no se puede validar (match_id inválido, partido no terminado,
    sin predicción, etc.).
    """
    match_id = partido.get('match_id')
    match_url = partido.get('match_url', '')

    pred = partido.get('ranking_analysis', {}).get('prediction', {})
    favorito_pred = pred.get('favored_player')
    if not favorito_pred:
        return None

    jugador1 = partido.get('jugador1', '')
    jugador2 = partido.get('jugador2', '')

    # ── Step 1: Validate slug — detect wrong match_id from Kambi Tier 3 matcher ──
    slug_valid = _validar_slug_ambos_jugadores(match_url, jugador1, jugador2)
    resolved_from_feed = False
    feed_entry = None

    if not slug_valid and feed_lookup is not None:
        # Slug is wrong — re-resolve match_id from FlashScore feed
        feed_entry = _buscar_en_feed(jugador1, jugador2, feed_lookup)
        if feed_entry:
            match_id = feed_entry['match_id']
            resolved_from_feed = True
        else:
            # Can't find in feed — don't report wrong result
            return None
    elif not match_id or match_id in ('tennis', '', None):
        # No match_id at all — try feed fallback
        if feed_lookup is not None:
            feed_entry = _buscar_en_feed(jugador1, jugador2, feed_lookup)
            if feed_entry:
                match_id = feed_entry['match_id']
                resolved_from_feed = True
            else:
                return None
        else:
            return None

    if resultado_api is None:
        resultado_api = obtener_resultado_partido(match_id)

    if resultado_api.get('status') != 'FT':
        return None

    lado_ganador = resultado_api.get('ganador_lado')
    if lado_ganador is None:
        return None

    # ── Step 2: Determine home/away mapping ──
    # When resolved from feed, use feed player names for slug detection
    # since the original match_url slug was wrong.
    if resolved_from_feed and feed_entry:
        # Feed entry has correct FS player names — FS home = jugador1_fs
        fs_j1 = feed_entry.get('jugador1_fs', '')
        fs_j2 = feed_entry.get('jugador2_fs', '')
        # Determine which of our jugador1/jugador2 is the FS home
        # by matching feed player names to our player names
        a1_ours, _ = _parse_nombre(jugador1)
        a1_fs, _ = _parse_nombre(fs_j1)
        a2_fs, _ = _parse_nombre(fs_j2)
        if a1_ours and a1_fs and (a1_ours in a1_fs or a1_fs in a1_ours):
            fs_home = 'jugador1'  # our jugador1 = FS home (jugador1_fs)
        elif a1_ours and a2_fs and (a1_ours in a2_fs or a2_fs in a1_ours):
            fs_home = 'jugador2'  # our jugador1 = FS away, so our jugador2 = FS home
        else:
            fs_home = 'jugador1'  # fallback
    else:
        # DJ='H' en dc_1 significa HOME de FlashScore ganó, y DJ='A' significa AWAY ganó.
        # Kambi y FS pueden ordenar a los jugadores de forma distinta: cuando el orden
        # difiere, 'jugador1' de Kambi es el 'away' de FS, así que DJ='H' → jugador2 real.
        # _detectar_jugador_home_fs() usa la match_url para resolver el orden correcto.
        fs_home = _detectar_jugador_home_fs(match_url, jugador1, jugador2)

    if lado_ganador == 'jugador1':
        # FS-home ganó → nombre real del jugador FS-home
        ganador_real = jugador1 if fs_home == 'jugador1' else jugador2
    else:
        # FS-away ganó → nombre real del jugador FS-away
        ganador_real = jugador2 if fs_home == 'jugador1' else jugador1

    correcto = (favorito_pred.strip().lower() == ganador_real.strip().lower())

    return {
        'partido': f"{jugador1} vs {jugador2}",
        'prediccion': favorito_pred,
        'confianza': pred.get('confidence'),
        'resultado_real': ganador_real,
        'correcto': correcto,
        'match_id': match_id,
        'torneo': partido.get('torneo', 'Desconocido'),
        'superficie': partido.get('tipo_cancha') or partido.get('superficie') or 'unknown',
        'resolved_from_feed': resolved_from_feed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Métricas agregadas (testeables sin I/O)
# ──────────────────────────────────────────────────────────────────────────────

def calcular_accuracy(resultados: list[dict]) -> float:
    """Accuracy global: fracción de predicciones correctas."""
    if not resultados:
        return 0.0
    return sum(1 for r in resultados if r.get('correcto')) / len(resultados)


def accuracy_por_superficie(resultados: list[dict]) -> dict:
    """
    Segmenta accuracy por superficie.

    Retorna:
        {'clay': {'accuracy': 0.62, 'n': 18, 'correctas': 11}, ...}
    """
    por_sup: dict = defaultdict(lambda: {'correctas': 0, 'total': 0})
    for r in resultados:
        sup = r.get('superficie', 'unknown')
        por_sup[sup]['total'] += 1
        if r.get('correcto'):
            por_sup[sup]['correctas'] += 1

    return {
        sup: {
            'accuracy': round(v['correctas'] / v['total'], 4) if v['total'] > 0 else 0.0,
            'n': v['total'],
            'correctas': v['correctas'],
        }
        for sup, v in por_sup.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Actualización de calibración (feed para edge_calculator.py)
# ──────────────────────────────────────────────────────────────────────────────

def actualizar_calibracion_desde_resultados(resultados: list[dict]) -> None:
    """
    Actualiza data/calibracion_edge.json con los resultados validados.
    Conecta CX-06: accuracy real → p_historica en Kelly-KL.
    """
    if not resultados:
        return

    try:
        with open(CALIBRACION_FILE) as f:
            cal = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cal = {"global": {"wins": 0, "losses": 0}, "por_superficie": {}}

    for r in resultados:
        if r.get('correcto') is None:
            continue
        sup = r.get('superficie', 'unknown')
        key = 'wins' if r['correcto'] else 'losses'

        cal.setdefault('global', {'wins': 0, 'losses': 0})
        cal['global'][key] += 1

        cal.setdefault('por_superficie', {})
        cal['por_superficie'].setdefault(sup, {'wins': 0, 'losses': 0})
        cal['por_superficie'][sup][key] += 1

    cal['ultima_actualizacion'] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(CALIBRACION_FILE), exist_ok=True)
    with open(CALIBRACION_FILE, 'w') as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)

    print(f"  Calibración actualizada → {CALIBRACION_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────────────────────

def validar_predicciones(h2h_file: str, output_file: str, actualizar_cal: bool = True) -> dict:
    """
    Lee h2h_results_enhanced, consulta API, calcula accuracy y exporta JSON.

    Args:
        h2h_file:       ruta a h2h_results_enhanced_FECHA.json
        output_file:    ruta de salida para resultados_finales_FECHA.json
        actualizar_cal: si True, actualiza calibracion_edge.json (CX-06)
    """
    with open(h2h_file, encoding='utf-8') as f:
        raw = json.load(f)
    partidos = raw.get('partidos', raw) if isinstance(raw, dict) else raw

    print(f"Validando {len(partidos)} partidos desde {h2h_file} ...")

    # Pre-fetch FlashScore feed for fallback when match_id slug is wrong
    print("  Fetching FlashScore feed for slug validation fallback...")
    feed_lookup = _fetch_finished_matches_feed()
    print(f"  Feed: {len(feed_lookup)} partidos indexados")

    resultados: list[dict] = []
    saltados = 0
    resueltos_feed = 0

    for i, partido in enumerate(partidos, 1):
        match_id = partido.get('match_id')
        match_url = partido.get('match_url', '')
        jugador1 = partido.get('jugador1', '')
        jugador2 = partido.get('jugador2', '')

        # Check slug validity BEFORE using match_id
        slug_valid = _validar_slug_ambos_jugadores(match_url, jugador1, jugador2)

        if not slug_valid:
            # Slug is wrong — let validar_partido_individual handle feed fallback
            r = validar_partido_individual(partido, None, feed_lookup=feed_lookup)
            if r is None:
                saltados += 1
                print(f"  [{i:3d}] ⚠️  {jugador1} vs {jugador2} — slug incorrecto, no encontrado en feed")
                continue
            if r.get('resolved_from_feed'):
                resueltos_feed += 1
        elif not match_id or match_id in ('tennis', ''):
            # No match_id — try feed fallback
            r = validar_partido_individual(partido, None, feed_lookup=feed_lookup)
            if r is None:
                saltados += 1
                continue
            if r.get('resolved_from_feed'):
                resueltos_feed += 1
        else:
            resultado_api = obtener_resultado_partido(match_id)
            r = validar_partido_individual(partido, resultado_api, feed_lookup=feed_lookup)
            if r is None:
                # No terminado, sin predicción, o match_id inválido
                if resultado_api.get('status') not in ('NS', 'LIVE'):
                    saltados += 1
                continue

        resultados.append(r)
        feed_tag = ' [feed]' if r.get('resolved_from_feed') else ''
        estado = '✅' if r['correcto'] else '❌'
        print(f"  [{i:3d}] {estado} {r['partido']} → {r['resultado_real']} (pred: {r['prediccion']}){feed_tag}")

        time.sleep(DELAY_ENTRE_REQUESTS)

    accuracy = calcular_accuracy(resultados)
    por_sup = accuracy_por_superficie(resultados)

    output = {
        'fecha_validacion': datetime.now().isoformat(),
        'fuente_h2h': h2h_file,
        'total_partidos': len(partidos),
        'total_validados': len(resultados),
        'saltados': saltados,
        'correctas': sum(1 for r in resultados if r['correcto']),
        'accuracy': round(accuracy, 4),
        'accuracy_por_superficie': por_sup,
        'partidos': resultados,
    }

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Accuracy: {accuracy*100:.1f}% ({output['correctas']}/{output['total_validados']})")
    for sup, datos in por_sup.items():
        print(f"   {sup:8s}: {datos['accuracy']*100:.1f}% (n={datos['n']})")
    if resueltos_feed:
        print(f"   Resueltos por feed (slug incorrecto): {resueltos_feed}")
    print(f"   Exportado → {output_file}")

    if actualizar_cal and resultados:
        actualizar_calibracion_desde_resultados(resultados)

    return output


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _find_latest_h2h() -> Optional[str]:
    """Encuentra el h2h_results_enhanced más reciente en reports/."""
    import glob
    files = glob.glob('reports/h2h_results_enhanced_*.json')
    return max(files, key=os.path.getmtime) if files else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nodo-05 — Validación post-partido FlashScore')
    parser.add_argument('--h2h', help='Ruta a h2h_results_enhanced_FECHA.json (auto si omite)')
    parser.add_argument('--output', help='Ruta de salida (auto si omite)')
    parser.add_argument('--no-cal', action='store_true', help='No actualizar calibracion_edge.json')
    args = parser.parse_args()

    h2h_file = args.h2h or _find_latest_h2h()
    if not h2h_file:
        print("❌ No se encontró ningún h2h_results_enhanced_*.json en reports/")
        raise SystemExit(1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or f'reports/resultados_finales_{ts}.json'

    validar_predicciones(h2h_file, output_file, actualizar_cal=not args.no_cal)

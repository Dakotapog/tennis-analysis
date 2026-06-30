"""
🔬 SCRIPT DE VERIFICACIÓN DE RESULTADOS FINALES
Carga los resultados de un análisis H2H previo, consulta la API FlashScore Ninja
para obtener el resultado real y lo compara con la predicción generada.

Migrado de Playwright (~40 min) a API Ninja (<2 seg para 80 partidos).

Fix 2026-06-09: cuando event_id = None (bug conocido), busca el partido
en el feed de FlashScore por nombre de jugadores (name matching 3-tier).
"""

import json
import argparse
import logging
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS as HEADERS

try:
    from config import detectar_tier
    TIER_AVAILABLE = True
except ImportError:
    TIER_AVAILABLE = False

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DELAY_ENTRE_REQUESTS = 0.3  # segundos — no martillar la API


# ──────────────────────────────────────────────────────────────────────────────
# Parser del formato propietario FlashScore
# ──────────────────────────────────────────────────────────────────────────────

def parsear_respuesta_flashscore(raw: str) -> dict:
    """Convierte KEY÷VALUE¬KEY÷VALUE en dict."""
    datos = {}
    for par in raw.split('¬'):
        if '÷' in par:
            k, v = par.split('÷', 1)
            datos[k] = v
    return datos


def extraer_event_id(match_url: str) -> Optional[str]:
    """Extrae el event_id del parámetro mid= en la URL del partido."""
    m = re.search(r'mid=([^&]+)', match_url or '')
    return m.group(1) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Name matching — reutiliza patrón de kambi_tennis.py
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normaliza nombre: lowercase, sin acentos, sin puntuación."""
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _parse_nombre(nombre: str) -> Tuple[str, str]:
    """Extrae (apellido, inicial) de cualquier formato de nombre."""
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
    """Clave de matching: (apellido1, ini1, apellido2, ini2) ordenados."""
    a1, i1 = _parse_nombre(name1)
    a2, i2 = _parse_nombre(name2)
    if a1 <= a2:
        return (a1, i1, a2, i2)
    return (a2, i2, a1, i1)


# ──────────────────────────────────────────────────────────────────────────────
# FlashScore feed — obtener partidos terminados con match_id
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_finished_matches_feed() -> Dict[Tuple, Dict]:
    """
    Consulta el feed de FlashScore de hoy y ayer para obtener
    partidos terminados con su match_id.

    Returns:
        Dict mapeando match_key → {match_id, jugador1_fs, jugador2_fs, ...}
    """
    lookup = {}

    for day_offset in [0, -1, -2]:
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
            fields = {}
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

            key = _build_match_key(j1, j2)
            lookup[key] = {
                "match_id": match_id,
                "jugador1_fs": j1,
                "jugador2_fs": j2,
                "torneo_fs": current_tournament,
            }

            # También indexar por apellidos solos (fallback tier 2)
            a1, _ = _parse_nombre(j1)
            a2, _ = _parse_nombre(j2)
            key_apellido = (min(a1, a2), "", max(a1, a2), "")
            if key_apellido not in lookup:
                lookup[key_apellido] = lookup[key]

    logger.info(f"📡 FlashScore feed: {len(lookup)} partidos indexados (hoy + ayer + anteayer)")
    return lookup


def _buscar_match_id(p1: str, p2: str, feed_lookup: Dict) -> Optional[str]:
    """
    Busca el match_id en el feed de FlashScore por nombre.
    Tier 1: match_key exacto (apellido + inicial).
    Tier 2: solo apellidos (sin inicial).
    Tier 3: substring overlap en apellidos ≥4 chars.
    """
    # Tier 1: clave exacta
    key = _build_match_key(p1, p2)
    if key in feed_lookup:
        return feed_lookup[key]["match_id"]

    # Tier 2: solo apellidos
    a1, _ = _parse_nombre(p1)
    a2, _ = _parse_nombre(p2)
    key_apellido = (min(a1, a2), "", max(a1, a2), "")
    if key_apellido in feed_lookup:
        return feed_lookup[key_apellido]["match_id"]

    # Tier 3: substring overlap
    for fkey, fval in feed_lookup.items():
        fa1, _, fa2, _ = fkey
        if not fa1 or not fa2:
            continue
        a1_match = (a1 and fa1 and (a1 in fa1 or fa1 in a1) and len(min(a1, fa1, key=len)) >= 4)
        a2_match = (a2 and fa2 and (a2 in fa2 or fa2 in a2) and len(min(a2, fa2, key=len)) >= 4)
        if a1_match and a2_match:
            return fval["match_id"]
        # Intentar cruzado (home/away invertido)
        a1_match_cross = (a1 and fa2 and (a1 in fa2 or fa2 in a1) and len(min(a1, fa2, key=len)) >= 4)
        a2_match_cross = (a2 and fa1 and (a2 in fa1 or fa1 in a2) and len(min(a2, fa1, key=len)) >= 4)
        if a1_match_cross and a2_match_cross:
            return fval["match_id"]

    return None


def obtener_resultado_api(event_id: str) -> dict:
    """
    Consulta dc_1_{event_id} y retorna status + ganador.

    DJ = 'H' (local ganó) | 'A' (visitante ganó) | '' (no terminado)
    DE = sets local | DF = sets visitante
    """
    if not event_id:
        return {'status': 'INVALID_ID'}

    url = f"{FLASHSCORE_BASE}/dc_1_{event_id}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

    datos = parsear_respuesta_flashscore(r.text)
    dj = datos.get('DJ', '')

    if dj in ('H', 'A'):
        return {
            'status': 'FT',
            'ganador_lado': 'jugador1' if dj == 'H' else 'jugador2',
            'sets_local': datos.get('DE', '0'),
            'sets_visitante': datos.get('DF', '0'),
        }

    # No terminado — distinguir NS vs LIVE
    try:
        dc_ts = int(datos.get('DC', '0'))
        if dc_ts and datetime.fromtimestamp(dc_ts) > datetime.now():
            return {'status': 'NS'}
    except (ValueError, TypeError):
        pass

    return {'status': 'LIVE'}


# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda de archivo H2H
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_h2h_results_file():
    """Encontrar el archivo h2h_results_enhanced más reciente en reports/."""
    reports_dir = Path('reports')
    if not reports_dir.exists():
        logger.error("❌ El directorio 'reports' no existe.")
        return None

    results_files = list(reports_dir.glob('h2h_results_enhanced_*.json'))
    if not results_files:
        logger.error("❌ No se encontraron archivos h2h_results_enhanced_*.json.")
        return None

    latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


# ──────────────────────────────────────────────────────────────────────────────
# Verificación principal
# ──────────────────────────────────────────────────────────────────────────────

def verificar_partidos(h2h_file: str, tier_filter: str = None):
    """Verifica resultados de todos los partidos vía API Ninja."""

    with open(h2h_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matches = data.get('partidos', [])
    logger.info(f"📂 Cargados {len(matches)} partidos desde {h2h_file}")

    # Filtrar por tier si se especifica
    if tier_filter and TIER_AVAILABLE:
        matches = [m for m in matches if detectar_tier(m.get('torneo_nombre', '')) == tier_filter]
        logger.info(f"Filtro tier '{tier_filter}': {len(matches)} partidos")

    if not matches:
        logger.warning("⚠️ Sin partidos para verificar.")
        return

    # Pre-fetch feed de FlashScore para resolver match_ids faltantes
    feed_lookup = _fetch_finished_matches_feed()

    verification_results = []
    verified = 0
    hits = 0
    not_finished = 0
    errors = 0
    resolved_by_name = 0

    for i, match in enumerate(matches):
        p1 = match.get('jugador1', '?')
        p2 = match.get('jugador2', '?')
        match_url = match.get('match_url', '')

        # Obtener predicción
        prediction_info = match.get('ranking_analysis', {}).get('prediction', {})
        predicted_winner = prediction_info.get('favored_player')
        prediction_confidence = prediction_info.get('confidence')

        if not predicted_winner:
            logger.warning(f"  ⚠️ [{i+1}/{len(matches)}] {p1} vs {p2} — Sin predicción. Saltando.")
            continue

        # Extraer event_id de la URL
        event_id = extraer_event_id(match_url)

        # Fallback: buscar por nombre en el feed de FlashScore
        if not event_id:
            event_id = _buscar_match_id(p1, p2, feed_lookup)
            if event_id:
                resolved_by_name += 1
            else:
                logger.warning(f"  ⚠️ [{i+1}/{len(matches)}] {p1} vs {p2} — No encontrado en feed.")
                errors += 1
                continue

        # Consultar API
        resultado = obtener_resultado_api(event_id)
        status = resultado.get('status')

        if status == 'FT':
            ganador_lado = resultado['ganador_lado']
            actual_winner = match.get(ganador_lado, '?')
            sets_score = f"{resultado['sets_local']}-{resultado['sets_visitante']}"

            # Comparar predicción con resultado
            predicted_part = predicted_winner.split(' ')[0].lower()
            hit = predicted_part in actual_winner.lower()

            verified += 1
            if hit:
                hits += 1

            logger.info(f"  {'✅' if hit else '❌'} [{i+1}/{len(matches)}] {p1} vs {p2} → {actual_winner} ({sets_score}) | Pred: {predicted_winner} {'ACIERTO' if hit else 'FALLO'}")

            verification_results.append({
                'match_info': {
                    'jugador1': p1,
                    'jugador2': p2,
                    'match_url': match_url,
                    'torneo': match.get('torneo_nombre', ''),
                },
                'prediction': {
                    'predicted_winner': predicted_winner,
                    'confidence': prediction_confidence,
                },
                'actual_result': {
                    'final_score': sets_score,
                    'actual_winner': actual_winner,
                },
                'verification': {
                    'hit': hit,
                    'status': 'Verified',
                },
            })

        elif status in ('NS', 'LIVE'):
            not_finished += 1
            logger.info(f"  ⏳ [{i+1}/{len(matches)}] {p1} vs {p2} — {status}")

        else:
            errors += 1
            logger.warning(f"  ⚠️ [{i+1}/{len(matches)}] {p1} vs {p2} — Error: {resultado.get('error', status)}")

        time.sleep(DELAY_ENTRE_REQUESTS)

    # Guardar resultados
    accuracy = (hits / verified * 100) if verified > 0 else 0

    logger.info("=" * 60)
    logger.info("🏁 VERIFICACIÓN COMPLETADA 🏁")
    logger.info(f"📊 Verificados: {verified} | Aciertos: {hits} | Fallos: {verified - hits}")
    logger.info(f"📊 Accuracy: {accuracy:.1f}%")
    logger.info(f"⏳ No finalizados: {not_finished} | Errores: {errors}")
    if resolved_by_name:
        logger.info(f"🔗 Resueltos por name matching: {resolved_by_name}")
    logger.info("=" * 60)

    if not verification_results:
        logger.warning("⚠️ Ningún partido verificado — nada que guardar.")
        return

    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_dir / f"resultados_finales_{timestamp}.json"

    output_data = {
        'metadata': {
            'verification_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': h2h_file,
            'tier_filter': tier_filter,
            'total_matches_processed': len(matches),
            'total_matches_verified': verified,
            'total_not_finished': not_finished,
            'total_errors': errors,
        },
        'summary': {
            'total_hits': hits,
            'total_misses': verified - hits,
            'accuracy_percentage': round(accuracy, 2),
        },
        'detailed_results': verification_results,
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Resultados guardados en: {filename}")
    return str(filename)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verificación de resultados finales vía API Ninja')
    parser.add_argument('file', nargs='?', default=None,
                        help='Archivo H2H JSON (default: más reciente en reports/)')
    parser.add_argument('--torneo-tipo', type=str, default=None,
                        choices=['grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'],
                        help='Filtrar por tier de torneo (default: todos)')
    args = parser.parse_args()

    if args.file:
        h2h_file = args.file
    else:
        h2h_file = find_latest_h2h_results_file()

    if h2h_file:
        logger.info(f"📂 Archivo seleccionado: {h2h_file}")
        verificar_partidos(h2h_file, tier_filter=args.torneo_tipo)
    else:
        logger.error("❌ No se encontró archivo H2H para verificar.")

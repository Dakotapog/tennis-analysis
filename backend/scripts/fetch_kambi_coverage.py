#!/usr/bin/env python3
"""
fetch_kambi_coverage.py — D90-01 (Nodo-90)

Fetcha el catálogo de eventos de tenis disponibles en Kambi/Betplay y
guarda un side-car reports/kambi_coverage_FECHA.json.

edge_calculator.py lee este side-car (sin HTTP) para anotar kambi_disponible: bool
en cada pick del edge_report. El filtro real (gate por kambi) se aplica solo en
trader_ev_tenis.py y betplay_combo_builder.py, NUNCA en edge_calculator ni shadow_book.

Uso:
    python3 scripts/fetch_kambi_coverage.py          # guarda hoy
    python3 scripts/fetch_kambi_coverage.py --stdout # imprime JSON, no guarda
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    import requests
    from scraping.kambi_tennis import KAMBI_BASE, KAMBI_PARAMS, KAMBI_HEADERS
    _KAMBI_AVAILABLE = True
except ImportError:
    _KAMBI_AVAILABLE = False

REPORTS_DIR = BACKEND_DIR / 'reports'
COVERAGE_PREFIX = 'kambi_coverage_'

# Criterion labels que corresponden a "ganador del partido"
_MATCH_WINNER_LABELS = {'Match', 'Cuotas del partido', 'Match Winner', '1X2'}


def _normalize_name(name: str) -> str:
    """Normaliza nombre de jugador: minúsculas, sin tildes, sin puntuación."""
    name = unicodedata.normalize('NFD', name.lower())
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[^a-z\s]', '', name)
    return name.strip()


def fetch_coverage() -> dict | None:
    """
    Fetcha Kambi y retorna dict de cobertura, o None si falla.

    Estructura retornada:
    {
        "fecha": "2026-07-13",
        "fetched_at": "2026-07-13T09:28:00",
        "n_eventos": 42,
        "players_normalized": ["carlos alcaraz", "novak djokovic", ...],
        "event_pairs": [["carlos alcaraz", "novak djokovic"], ...],
    }
    """
    if not _KAMBI_AVAILABLE:
        return None

    try:
        url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=15)
        resp.raise_for_status()
        events = resp.json().get('events', [])
    except Exception as e:
        print(f'[kambi_coverage] WARN: fetch falló — {e}', file=sys.stderr)
        return None

    players_set = set()
    event_pairs = []
    odds_map: dict = {}  # D154-10: player_normalized → cuota decimal live

    for ev in events:
        pair = []
        for offer in ev.get('betOffers', []):
            label_crit = offer.get('criterion', {}).get('label', '')
            if label_crit not in _MATCH_WINNER_LABELS:
                continue
            for oc in offer.get('outcomes', []):
                raw = oc.get('label') or oc.get('participant') or ''
                if not raw:
                    continue
                norm = _normalize_name(raw)
                if norm:
                    players_set.add(norm)
                    pair.append(norm)
                    # D154-10: capturar cuota live (milliodds → decimal)
                    milli = oc.get('odds', 0)
                    if milli > 1000:  # sanity: >1.0x
                        odds_map[norm] = round(milli / 1000, 3)
        if len(pair) >= 2:
            event_pairs.append(pair[:2])

    return {
        'fecha': datetime.now().strftime('%Y-%m-%d'),
        'fetched_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'n_eventos': len(events),
        'n_jugadores': len(players_set),
        'players_normalized': sorted(players_set),
        'event_pairs': event_pairs,
        'odds_map': odds_map,  # D154-10: cuotas live por jugador normalizado
    }


def find_latest_coverage(reports_dir: Path = REPORTS_DIR) -> Path | None:
    """Encuentra el archivo kambi_coverage más reciente en reports/."""
    files = sorted(reports_dir.glob(f'{COVERAGE_PREFIX}*.json'), reverse=True)
    return files[0] if files else None


def load_coverage(reports_dir: Path = REPORTS_DIR) -> dict | None:
    """Carga el coverage más reciente desde disco. Retorna None si no existe."""
    path = find_latest_coverage(reports_dir)
    if path is None:
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_player_available(nombre: str, coverage: dict) -> bool:
    """
    Retorna True si el jugador está disponible en Kambi según el coverage.
    Usa coincidencia por apellido (último token) como fallback.
    """
    if not coverage:
        return False
    players = set(coverage.get('players_normalized', []))
    norm = _normalize_name(nombre)
    if norm in players:
        return True
    # Fallback: apellido — si último token ≤2 chars (inicial "N."), usar primer token
    # Nuestro formato: "Arakawa N." → "arakawa n" → apellido = "arakawa" (no "n")
    parts = norm.split()
    last = parts[-1] if parts else ''
    apellido = parts[0] if (len(parts) > 1 and len(last) <= 2) else last
    if len(apellido) > 3:
        return any(apellido in p for p in players)
    return False


def patch_edge_report_cuotas(edge_report_path: str, coverage: dict,
                              umbral_pct: float = 0.02) -> int:
    """D154-10 (B10/O4): actualiza cuota_favorito en edge_report con precios live.

    Evita que los combo builders calculen EV sobre cuotas de horas atrás.
    Solo actualiza picks donde la diferencia supere umbral_pct (default 2%)
    para evitar micro-ruido. Modifica el archivo en disco in-place.

    Args:
        edge_report_path: Ruta al edge_report_*.json a parchear.
        coverage: Dict retornado por fetch_coverage() (debe incluir 'odds_map').
        umbral_pct: Diferencia mínima para actualizar (default 2%).

    Returns:
        Número de picks actualizados.
    """
    odds_map = coverage.get('odds_map', {})
    if not odds_map:
        return 0

    try:
        with open(edge_report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    except Exception as e:
        print(f'[D154-10] No se pudo leer edge_report: {e}', file=sys.stderr)
        return 0

    updated = 0
    for seccion in ('apostar', 'watchlist'):
        for pick in report.get(seccion, []):
            jugador = pick.get('favorito_predicho', '')
            cuota_vieja = pick.get('cuota_favorito', 0)
            if not jugador or not cuota_vieja:
                continue
            norm = _normalize_name(jugador)
            cuota_live = odds_map.get(norm)
            if cuota_live is None:
                continue
            diff = abs(cuota_live - cuota_vieja) / max(cuota_vieja, 0.001)
            if diff >= umbral_pct:
                pick['cuota_favorito'] = cuota_live
                pick['cuota_favorito_stale'] = cuota_vieja  # trazabilidad
                pick['cuota_favorito_source'] = 'kambi_live_D154-10'
                updated += 1
                print(f'[D154-10] {jugador}: cuota {cuota_vieja}→{cuota_live} '
                      f'(Δ{diff*100:.1f}%)')

    if updated:
        with open(edge_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'[D154-10] {updated} picks actualizados en {edge_report_path}')

    return updated


def main():
    parser = argparse.ArgumentParser(description='Fetch cobertura Kambi tennis')
    parser.add_argument('--stdout', action='store_true',
                        help='Imprime JSON en stdout en lugar de guardar en disco')
    parser.add_argument('--reports-dir', default=str(REPORTS_DIR),
                        help='Directorio donde guardar el archivo de cobertura')
    args = parser.parse_args()

    coverage = fetch_coverage()
    if coverage is None:
        print('[kambi_coverage] ERROR: no se pudo obtener cobertura de Kambi', file=sys.stderr)
        sys.exit(1)

    if args.stdout:
        print(json.dumps(coverage, ensure_ascii=False, indent=2))
        return

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = reports_dir / f'{COVERAGE_PREFIX}{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    print(f'[D90-01] kambi_coverage guardado: {out_path}')
    print(f'  Eventos: {coverage["n_eventos"]} | Jugadores: {coverage["n_jugadores"]}')


if __name__ == '__main__':
    main()

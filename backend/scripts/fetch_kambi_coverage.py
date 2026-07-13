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
        if len(pair) >= 2:
            event_pairs.append(pair[:2])

    return {
        'fecha': datetime.now().strftime('%Y-%m-%d'),
        'fetched_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'n_eventos': len(events),
        'n_jugadores': len(players_set),
        'players_normalized': sorted(players_set),
        'event_pairs': event_pairs,
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
    # Fallback: apellido (último token)
    apellido = norm.split()[-1] if norm.split() else ''
    if len(apellido) > 3:
        return any(apellido in p for p in players)
    return False


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

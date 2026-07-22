#!/usr/bin/env python3
"""
filter_kambi_picks.py — D141-01 (Nodo-141)

Lee el edge_report_FECHA.json más reciente y produce edge_report_kambi_FECHA.json
con SOLO los picks donde kambi_disponible=True.

Los combo builders detectan automáticamente este reporte (mtime más reciente /
sort alfabético) y generan combos con picks 100% apostables en Betplay.

Uso:
    python3 scripts/filter_kambi_picks.py           # escribe edge_report_kambi_*.json
    python3 scripts/filter_kambi_picks.py --dry-run # muestra stats sin escribir
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
REPORTS_DIR = BACKEND_DIR / 'reports'


def _find_latest_full_report(reports_dir: Path = REPORTS_DIR) -> Path | None:
    """Encuentra el edge_report más reciente (excluye archivos kambi)."""
    candidates = sorted(
        [f for f in reports_dir.glob('edge_report_*.json') if 'kambi' not in f.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def filter_kambi_picks(source_path: Path) -> dict:
    """
    Lee source_path (edge_report completo) y retorna dict con solo picks kambi_disponible=True.
    Preserva toda la estructura del reporte original.

    El edge_report usa campos 'apostar' y 'watchlist' (no 'picks').
    """
    with open(source_path, encoding='utf-8') as f:
        full = json.load(f)

    # edge_report estructura: {'apostar': [...], 'watchlist': [...], 'metadata': {...}, ...}
    apostar_all = full.get('apostar', [])
    watchlist_all = full.get('watchlist', [])
    picks_all = apostar_all + watchlist_all

    apostar_kambi = [p for p in apostar_all if p.get('kambi_disponible') is True]
    watchlist_kambi = [p for p in watchlist_all if p.get('kambi_disponible') is True]

    kambi_report = dict(full)
    kambi_report['apostar'] = apostar_kambi
    kambi_report['watchlist'] = watchlist_kambi
    # Compatibilidad con código que lea 'picks'
    kambi_report['picks'] = apostar_kambi + watchlist_kambi
    kambi_report['_kambi_only'] = True
    kambi_report['_n_kambi'] = len(apostar_kambi) + len(watchlist_kambi)
    kambi_report['_n_total'] = len(picks_all)
    kambi_report['_source_report'] = source_path.name
    kambi_report['_generated_at'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    return kambi_report


def main():
    parser = argparse.ArgumentParser(description='Filtra edge_report a picks kambi disponibles')
    parser.add_argument('--source', default=None,
                        help='Ruta al edge_report fuente (default: más reciente en reports/)')
    parser.add_argument('--output', default=None,
                        help='Ruta de salida (default: reports/edge_report_kambi_TS.json)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo muestra stats, no escribe archivo')
    parser.add_argument('--reports-dir', default=str(REPORTS_DIR),
                        help='Directorio de reports')
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    if args.source:
        source_path = Path(args.source)
    else:
        source_path = _find_latest_full_report(reports_dir)

    if source_path is None or not source_path.exists():
        print('[filter_kambi_picks] ERROR: no se encontró edge_report_*.json en reports/',
              file=sys.stderr)
        sys.exit(1)

    kambi_report = filter_kambi_picks(source_path)
    n_kambi = kambi_report['_n_kambi']
    n_total = kambi_report['_n_total']

    print(f'[D141-01] Fuente: {source_path.name}')
    print(f'  Picks totales: {n_total} | kambi_disponible=True: {n_kambi}')

    apostar = kambi_report.get('apostar', [])
    watchlist = kambi_report.get('watchlist', [])
    print(f'  APOSTAR: {len(apostar)} | WATCHLIST: {len(watchlist)}')
    for p in apostar:
        print(f'    APOSTAR: {p.get("favorito_predicho")} @{p.get("cuota_favorito")} edge={p.get("edge_pct")}')

    if args.dry_run:
        print('[D141-01] --dry-run: no se escribio archivo')
        return

    if n_kambi == 0:
        print('[D141-01] Sin picks kambi_disponible=True — no se escribe edge_report_kambi',
              file=sys.stderr)
        sys.exit(0)  # exit 0 — no es error, es dia sin cobertura

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = reports_dir / f'edge_report_kambi_{ts}.json'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(kambi_report, f, indent=2, ensure_ascii=False)

    print(f'[D141-01] Kambi report escrito: {out_path}')
    print(f'  Combo builders lo detectan automaticamente (mtime reciente)')


if __name__ == '__main__':
    main()

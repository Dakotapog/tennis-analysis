#!/usr/bin/env python3
"""
backfill_evaluar_shadow.py — D124-05 (Nodo-124)

Recupera picks EVALUAR históricos de edge_reports y los inyecta en shadow_book
con resultados crosswalk de los registros ya settled del mismo día.

Proxy: sin_edge picks con p_modelo >= 0.54 (equivalente EVALUAR de tabla_favoritos).
Nota: apostar/watchlist ya están en shadow_book — sólo se procesan sin_edge.

Uso:
    python3 scripts/backfill_evaluar_shadow.py [--dry-run] [-v]
    python3 scripts/backfill_evaluar_shadow.py --fecha 2026-07-14 [--dry-run]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import shadow_book as sb  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.54          # same gate as generar_tabla_favoritos2 EVALUAR
REPORTS_DIR   = BACKEND_DIR / 'reports'
SB_DIR        = REPORTS_DIR / 'shadow_book'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_sb_records(fecha: str) -> dict:
    """Load shadow_book records for a date using shadow_book's own loader."""
    path = SB_DIR / f'sb_{fecha}.jsonl'
    if not path.exists():
        return {}
    return sb._load_jsonl(path)


def _build_resultados_map(records: dict) -> dict:
    """
    Build resultados_map from settled shadow_book records.
    Format expected by settle(): {partido_str: {p1, p2, ganador, cuota_cierre, provenance, void}}
    Settle falls back to fuzzy name matching so the key just needs to be recognizable.
    """
    rmap = {}
    for sb_id, rec in records.items():
        if rec.get('_type') == 'session_meta':
            continue
        resolucion = rec.get('resolucion')
        if not resolucion:
            continue
        resultado = resolucion.get('resultado')
        if resultado not in ('WON', 'LOST'):
            continue

        snap    = rec.get('pick_snapshot', {})
        partido = snap.get('partido', '')
        if not partido or ' vs ' not in partido:
            continue

        p1, p2    = [x.strip() for x in partido.split(' vs ', 1)]
        favorito  = snap.get('favorito_predicho', '') or ''
        rival     = p2 if favorito == p1 else p1
        ganador   = favorito if resultado == 'WON' else rival
        cuota_c   = resolucion.get('cuota_cierre') or snap.get('cuota_favorito') or 0

        rmap[partido] = {
            'p1':          p1,
            'p2':          p2,
            'ganador':     ganador,
            'cuota_cierre': float(cuota_c),
            'provenance':  'backfill_sb_crosswalk',
            'void':        False,
        }
    return rmap


def _latest_edge_report(fecha_iso: str) -> Optional[dict]:
    """Load the most recent edge_report for a given date YYYY-MM-DD."""
    compact = fecha_iso.replace('-', '')
    files   = sorted(REPORTS_DIR.glob(f'edge_report_{compact}_*.json'))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except Exception:
        return None


def _evaluar_candidates(edge_report: dict) -> List[dict]:
    """
    Extract sin_edge picks with p_modelo >= CONF_THRESHOLD.
    apostar/watchlist are already in shadow_book — excluded.
    """
    return [
        p for p in edge_report.get('sin_edge', [])
        if (p.get('p_modelo') or 0) >= CONF_THRESHOLD
    ]


def _sb_has_partido(partido: str, records: dict) -> bool:
    """Return True if any shadow_book record (any type) references this partido."""
    for rec in records.values():
        if rec.get('_type') == 'session_meta':
            continue
        snap = rec.get('pick_snapshot', {})
        if snap.get('partido', '') == partido:
            return True
    return False


def _build_evaluar_pick(edge_pick: dict) -> dict:
    """Build pick dict for shadow_book.log_evaluar_pick() from edge_report pick."""
    partido  = edge_pick.get('partido', '')
    favorito = edge_pick.get('favorito_predicho', '')
    rival    = ''
    if ' vs ' in partido:
        p1, p2 = [x.strip() for x in partido.split(' vs ', 1)]
        rival  = p2 if favorito == p1 else p1

    cuota = edge_pick.get('cuota_favorito')
    return {
        'favorito_predicho': favorito,
        'rival':             rival,
        'partido':           partido,
        'torneo':            edge_pick.get('torneo') or 'Desconocido',
        'superficie':        edge_pick.get('superficie') or 'unknown',
        'tier':              edge_pick.get('tier') or 'unknown',
        'pick_status':       'EVALUAR',
        # p_modelo used as confidence proxy (tabla_favoritos usa su propio scoring,
        # pero p_modelo >= 0.54 es la mejor señal disponible en edge_report histórico)
        'confidence':        float(edge_pick.get('p_modelo') or 0),
        'cuota_favorito':    float(cuota) if cuota else None,
        'p_modelo':          edge_pick.get('p_modelo'),
        'p_implicita':       edge_pick.get('p_implicita'),
        'edge':              edge_pick.get('edge'),
        'kelly_kl':          edge_pick.get('kelly_kl'),
        'rfi_tier':          edge_pick.get('rfi_tier'),
        'score_directo':     edge_pick.get('score_directo') or 0,
        # Provenance marker — distingue backfill de picks en vivo
        '_backfill':         True,
        '_backfill_source':  'edge_report_sin_edge_proxy',
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backfill(
    only_fecha: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    today = datetime.now().strftime('%Y-%m-%d')

    # Collect edge report dates
    all_edge = sorted(REPORTS_DIR.glob('edge_report_*.json'))
    by_date: Dict[str, Path] = {}
    for f in all_edge:
        compact = f.name.split('edge_report_')[1][:8]
        iso     = f'{compact[:4]}-{compact[4:6]}-{compact[6:8]}'
        by_date[iso] = f  # last write wins (sorted)

    target_dates = [only_fecha] if only_fecha else sorted(by_date.keys())

    totals = {'dates': 0, 'logged': 0, 'skipped_dup': 0, 'settled': 0}

    for fecha in target_dates:
        if fecha not in by_date:
            logger.warning(f'[{fecha}] No edge_report found — skipping')
            continue

        sb_path = SB_DIR / f'sb_{fecha}.jsonl'
        if not sb_path.exists():
            if verbose:
                logger.info(f'[{fecha}] No shadow_book file — skipping')
            continue

        edge_report = _latest_edge_report(fecha)
        if not edge_report:
            continue

        candidates = _evaluar_candidates(edge_report)
        if not candidates:
            if verbose:
                logger.info(f'[{fecha}] 0 sin_edge candidates with p>=0.54')
            continue

        records      = _load_sb_records(fecha)
        resultados   = _build_resultados_map(records)

        date_logged  = 0
        date_skipped = 0

        for edge_pick in candidates:
            partido = edge_pick.get('partido', '')
            if not partido:
                continue

            if _sb_has_partido(partido, records):
                date_skipped += 1
                if verbose:
                    logger.info(f'[{fecha}]   SKIP (dup) {partido}')
                continue

            pick = _build_evaluar_pick(edge_pick)

            if verbose:
                cuota_str = f"@{pick['cuota_favorito']:.2f}" if pick['cuota_favorito'] else '@?'
                logger.info(
                    f'[{fecha}]   LOG {partido} '
                    f'conf={pick["confidence"]:.2f} {cuota_str}'
                )

            if not dry_run:
                sb_id = sb.log_evaluar_pick(pick, fecha=fecha)
                if sb_id:
                    date_logged += 1
                    # Reload so next iteration dedup check is current
                    records = _load_sb_records(fecha)
            else:
                date_logged += 1

        # Settle newly logged EVAL_ picks using crosswalk from already-settled records
        settled_n = 0
        if not dry_run and date_logged > 0 and fecha < today:
            if resultados:
                settled_n = sb.settle(fecha, resultados)
            else:
                logger.info(f'[{fecha}]   No resultados_map available for settle (no settled peers)')

        if date_logged or date_skipped:
            totals['dates']      += 1
            totals['logged']     += date_logged
            totals['skipped_dup']+= date_skipped
            totals['settled']    += settled_n
            logger.info(
                f'[{fecha}] logged={date_logged}  skipped={date_skipped}  '
                f'settled={settled_n}  (resultados_map n={len(resultados)})'
            )

    print()
    print('=' * 42)
    print('  BACKFILL EVALUAR — D124-05 (Nodo-124)')
    print('=' * 42)
    print(f'  Fechas procesadas : {totals["dates"]}')
    print(f'  Picks logged      : {totals["logged"]}')
    print(f'  Skipped (dup)     : {totals["skipped_dup"]}')
    print(f'  Picks settled     : {totals["settled"]}')
    if dry_run:
        print()
        print('  (DRY RUN — cero cambios escritos)')
    print('=' * 42)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='D124-05: Backfill retroactivo picks EVALUAR al shadow_book'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Mostrar qué se haría sin escribir nada',
    )
    parser.add_argument(
        '--fecha',
        help='Procesar sólo esta fecha YYYY-MM-DD (default: todas)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Log detallado por pick',
    )
    args = parser.parse_args()
    run_backfill(
        only_fecha=args.fecha,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

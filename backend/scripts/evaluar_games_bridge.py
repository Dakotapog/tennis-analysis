#!/usr/bin/env python3
"""
evaluar_games_bridge.py — D125-02 (Nodo-125)

Lee picks EVALUAR_GAMES (pick_type=evaluar_games, cuota<1.30) del shadow_book de hoy,
busca el mercado UNDER total juegos en Kambi para cada favorito absoluto, y genera
reports/evaluar_games_signal_YYYYMMDD_HHMMSS.json en el mismo formato que
games_signal_report (consumible por betplay_combo_builder --evaluar y live_desk X4).

Proxy de dominancia:
    diff_abs = (1/cuota_favorito - 0.5) * 2
    cuota 1.06 → diff=0.886  cuota 1.20 → diff=0.666  cuota 1.28 → diff=0.562
    Todos caen en zona DOMINANTE (>0.35) → predicted_sets=2, games_range=16-19.

Uso:
    python3 scripts/evaluar_games_bridge.py [--fecha YYYY-MM-DD] [--dry-run] [-v]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import shadow_book as sb  # noqa: E402

# Importar funciones reutilizables de games_signal_calculator
from games_signal_calculator import (   # noqa: E402
    _buscar_event_id_kambi,
    _fetch_betoffer_event,
    _analizar_mercados_juegos,
    _seleccionar_señal_optima,
    _predecir_sets_y_games,
    _zona_diff,
)

# ── Constants ─────────────────────────────────────────────────────────────────
REPORTS_DIR = BACKEND_DIR / 'reports'
SB_DIR      = REPORTS_DIR / 'shadow_book'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Core helpers ──────────────────────────────────────────────────────────────

def _diff_abs_from_cuota(cuota: float) -> float:
    """
    Proxy de dominancia desde cuota de favorito absoluto.
    diff_abs = (p_implicita - 0.5) * 2  donde  p_implicita = 1/cuota
    cuota 1.06 → 0.886 | cuota 1.20 → 0.666 | cuota 1.28 → 0.562
    Todos >= 0.35 = DOMINANTE.
    """
    if not cuota or cuota <= 0:
        return 0.5
    p = 1.0 / cuota
    return round((p - 0.5) * 2, 3)


def _build_hora_map_from_zita() -> Dict[str, str]:
    """
    Fallback: lee el zita_tennis_matches más reciente y construye
    apellido_lower → hora_partido. Usado cuando pick_snapshot.hora=None
    porque el pick fue logueado antes de D125-01 o antes de tener hora.
    Soporta estructura dict {torneo: [match,...]} y lista plana.
    """
    import glob as _glob
    files = sorted(_glob.glob(str(BACKEND_DIR / 'data' / 'zita_tennis_matches_*.json')), reverse=True)
    hora_map: Dict[str, str] = {}
    for f in files[:3]:
        try:
            data = json.load(open(f))
            matches: list = data if isinstance(data, list) else [
                m for ms in data.values() for m in (ms if isinstance(ms, list) else [])
            ]
            for m in matches:
                hora = m.get('hora_partido') or m.get('hora') or m.get('hora_inicio')
                if not hora:
                    continue
                for field in ('jugador1', 'jugador2', 'player1', 'player2'):
                    nombre = (m.get(field) or '').strip()
                    if nombre:
                        apellido = nombre.lower().split()[-1]
                        hora_map[apellido] = hora
            if hora_map:
                break
        except Exception:
            continue
    return hora_map


def _enrich_hora(pick: Dict, hora_map: Dict[str, str]) -> str:
    """Devuelve la hora del pick o la busca en hora_map por apellido."""
    if pick.get('hora'):
        return pick['hora']
    for nombre in (pick.get('jugador1', ''), pick.get('jugador2', '')):
        apellido = nombre.lower().split()[-1] if nombre else ''
        if apellido and apellido in hora_map:
            return hora_map[apellido]
    return None


def _load_evaluar_games_picks(fecha: str) -> List[Dict]:
    """Carga picks EVAL_ con pick_type=evaluar_games del shadow_book de hoy."""
    path = SB_DIR / f'sb_{fecha}.jsonl'
    if not path.exists():
        return []
    records = sb._load_jsonl(path)
    hora_map = _build_hora_map_from_zita()
    picks = []
    for sid, rec in records.items():
        if not sid.startswith('EVAL_'):
            continue
        snap = rec.get('pick_snapshot', {})
        if snap.get('pick_type') != 'evaluar_games':
            continue
        pick = {
            'sb_id':             sid,
            'partido':           snap.get('partido', ''),
            'jugador1':          snap.get('partido', '').split(' vs ')[0].strip() if ' vs ' in snap.get('partido', '') else '',
            'jugador2':          snap.get('partido', '').split(' vs ')[1].strip() if ' vs ' in snap.get('partido', '') else '',
            'favorito_predicho': snap.get('favorito_predicho', ''),
            'cuota_favorito':    snap.get('cuota_favorito') or 0,
            'confidence':        (lambda c: c / 100 if c and c >= 1 else (c or 0))(snap.get('confidence')),  # D126-06
            'match_id':          snap.get('match_id'),
            'hora':              snap.get('hora'),
            'torneo':            snap.get('torneo', ''),
            'superficie':        snap.get('superficie', ''),
            'tier':              snap.get('tier', ''),
        }
        # D125-01 fix: enriquecer hora desde zita si snapshot no la tiene
        if not pick['hora']:
            pick['hora'] = _enrich_hora(pick, hora_map)
        picks.append(pick)
    return picks


def _process_pick(pick: Dict, verbose: bool = False) -> Optional[Dict]:
    """
    Para un pick EVALUAR_GAMES, busca el evento Kambi y analiza mercado UNDER juegos.
    Retorna resultado en formato games_signal_report o None si no hay mercado.
    """
    partido  = pick['partido']
    cuota    = pick['cuota_favorito']
    match_id = pick.get('match_id')

    diff_abs   = _diff_abs_from_cuota(cuota)
    zona       = _zona_diff(diff_abs)
    pred       = _predecir_sets_y_games(diff_abs, total_score=0.5)  # total_score neutro

    # ── D128-02: solo excluir ultra-menores confirmados sin Kambi ───────────────
    # D126-04 original era demasiado amplio: incluía 'itf', 'm25', 'w25' que SÍ
    # están en Kambi para torneos Brisbane/Bali/Nogent/Saskatoon/SantaFe.
    # detectar_tier() devuelve 'itf' para M25/W25/M35 indistintamente — la
    # clasificación no es granular. El lookup de Kambi es el oráculo real.
    # Solo pre-filtrar M10/M15/W10/W15 donde Kambi nunca tiene cobertura.
    _TIERS_SIN_KAMBI = {'itf_minor', 'm10', 'w10', 'm15', 'w15'}
    # REMOVIDO: 'itf', 'm25', 'w25' — pueden estar en Kambi (Nodo-128 D128-02)
    tier_norm = (pick.get('tier') or '').lower().replace(' ', '_').replace('-', '_')
    if any(t in tier_norm for t in _TIERS_SIN_KAMBI):
        # Intentar wplay SSR (lazy import para no ralentizar si no hay picks ITF)
        _wplay_seln_id = None
        try:
            import sys as _sys
            _sys.path.insert(0, str(BACKEND_DIR))
            from scripts.odds_aggregator import fetch_all_odds as _fetch_all
            _wplay_feed = _fetch_all(['wplay'])
            # Buscar por apellido de jugador1 o jugador2
            for _nombre in (pick.get('jugador1', ''), pick.get('jugador2', '')):
                _apellido = _nombre.lower().split()[-1] if _nombre else ''
                _entry = _wplay_feed.get(_apellido, {}).get('wplay')
                if _entry and _entry.get('seln_id'):
                    _wplay_seln_id = _entry['seln_id']
                    break
        except Exception as _exc:
            if verbose:
                logger.info(f'  wplay SSR lookup error: {_exc}')

        if _wplay_seln_id:
            if verbose:
                logger.info(f'  {partido}: tier ITF → wplay encontrado seln_id={_wplay_seln_id}')
            return {
                'partido':          partido,
                'zona_diff':        zona,
                'diff_abs':         diff_abs,
                'predicted_sets':   pred['predicted_sets'],
                'games_range':      pred['games_range'],
                'hora':             pick.get('hora'),
                'cuota_ml':         cuota,
                'confidence':       pick['confidence'],
                'señales_optimas':  [],
                'tiene_mercados':   True,
                '_source':          'evaluar_games',
                '_sb_id':           pick['sb_id'],
                '_source_casa':     'wplay',
                '_wplay_seln_id':   _wplay_seln_id,
            }
        else:
            if verbose:
                logger.info(f'  {partido}: tier ITF → no encontrado en wplay SSR ni betplay')
            return {
                'partido':          partido,
                'zona_diff':        zona,
                'diff_abs':         diff_abs,
                'predicted_sets':   pred['predicted_sets'],
                'games_range':      pred['games_range'],
                'hora':             pick.get('hora'),
                'cuota_ml':         cuota,
                'confidence':       pick['confidence'],
                'señales_optimas':  [],
                'tiene_mercados':   False,
                '_source':          'evaluar_games',
                '_sb_id':           pick['sb_id'],
                '_skip_reason':     'itf_sin_mercado_co',  # no está en betplay ni wplay
            }

    # Construir dict partido para Kambi lookup (mismo formato que games_signal_calculator)
    partido_dict = {
        'jugador1':    pick['jugador1'],
        'jugador2':    pick['jugador2'],
        'match_id':    match_id,
        'torneo_nombre': pick['torneo'],
        'tipo_cancha': pick['superficie'],
        'tier':        pick['tier'],
        'ranking_analysis': {
            'prediction': {
                'scores': {
                    'score_difference':   diff_abs,
                    'p1_final_weight':    0.5 + diff_abs / 2,
                    'p2_final_weight':    0.5 - diff_abs / 2,
                }
            }
        },
    }

    # Lookup event_id en Kambi
    event_id = _buscar_event_id_kambi(partido_dict)
    if not event_id:
        if verbose:
            logger.info(f'  {partido}: sin event_id Kambi')
        _res = {
            'partido':          partido,
            'zona_diff':        zona,
            'diff_abs':         diff_abs,
            'predicted_sets':   pred['predicted_sets'],
            'games_range':      pred['games_range'],
            'hora':             pick.get('hora'),
            'cuota_ml':         cuota,
            'confidence':       pick['confidence'],
            'señales_optimas':  [],
            'tiene_mercados':   False,
            '_source':          'evaluar_games',
            '_sb_id':           pick['sb_id'],
        }
        # D128-01: marcar dominantes extremos — Kambi puede publicar market 2h antes
        if diff_abs > 0.85:
            _res['_watchlist_dominante'] = True
        return _res

    betoffer = _fetch_betoffer_event(event_id)
    señales  = _analizar_mercados_juegos(betoffer, pred)
    optimas  = _seleccionar_señal_optima(señales)

    if verbose:
        logger.info(f'  {partido}: event_id={event_id}  señales={len(señales)}  optimas={len(optimas)}')

    _res = {
        'partido':          partido,
        'zona_diff':        zona,
        'diff_abs':         diff_abs,
        'predicted_sets':   pred['predicted_sets'],
        'games_range':      pred['games_range'],
        'hora':             pick.get('hora'),
        'cuota_ml':         cuota,
        'confidence':       pick['confidence'],
        'señales_optimas':  optimas,
        'tiene_mercados':   bool(optimas),
        '_source':          'evaluar_games',
        '_sb_id':           pick['sb_id'],
    }
    # D128-01: partido encontrado en Kambi pero sin mercado UNDER — watchlist si muy dominante
    if not optimas and diff_abs > 0.85:
        _res['_watchlist_dominante'] = True
    return _res


def _save_report(resultados: List[Dict], fecha: str) -> Path:
    """Guarda evaluar_games_signal_YYYYMMDD_HHMMSS.json."""
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    out    = REPORTS_DIR / f'evaluar_games_signal_{ts}.json'
    n_pick = len(resultados)
    n_under = sum(
        1 for r in resultados
        if any(s.get('apostar') and s.get('direccion') == 'UNDER'
               for s in r.get('señales_optimas', []))
    )
    payload = {
        'metadata': {
            'fecha':      fecha,
            'generado':   datetime.now().isoformat(),
            'fuente':     'evaluar_games_bridge (Nodo-125 D125-02)',
            'n_picks':    n_pick,
            'n_con_under': n_under,
            'nodo':       'Nodo-125-EvalGames-Bridge-Dashboard-X4',
        },
        # Mismo campo "apostar" que games_signal_report para que build_evaluar_games_combos lo lea
        'apostar': [r for r in resultados if r.get('tiene_mercados')],
        # D128-01: dominantes extremos sin mercado hoy — revisar 2h antes del partido
        'watchlist_dominante': [r for r in resultados if r.get('_watchlist_dominante')],
        'detalle_completo': resultados,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_bridge(
    fecha:    Optional[str] = None,
    dry_run:  bool = False,
    verbose:  bool = False,
) -> None:
    fecha = fecha or date.today().isoformat()

    picks = _load_evaluar_games_picks(fecha)
    if not picks:
        logger.info(f'[EvalGamesBridge] {fecha}: 0 picks evaluar_games en shadow_book — nada que procesar')
        return

    logger.info(f'[EvalGamesBridge] {fecha}: {len(picks)} picks evaluar_games → buscando UNDER juegos en Kambi')

    resultados = []
    for pick in picks:
        if verbose:
            logger.info(f'Procesando: {pick["partido"]} @{pick["cuota_favorito"]} conf={pick["confidence"]:.0%}')
        resultado = _process_pick(pick, verbose=verbose)
        if resultado:
            resultados.append(resultado)

    n_under = sum(1 for r in resultados if r.get('tiene_mercados'))
    logger.info(f'[EvalGamesBridge] Resultados: {len(resultados)} procesados | {n_under} con señal UNDER')

    if not dry_run and resultados:
        out_path = _save_report(resultados, fecha)
        logger.info(f'[EvalGamesBridge] Guardado → {out_path.name}')
    elif dry_run:
        logger.info('[EvalGamesBridge] DRY RUN — no se escribe archivo')

    # Resumen consola
    print()
    print('=' * 46)
    print(f'  EVALUAR_GAMES BRIDGE — Nodo-125 D125-02')
    print('=' * 46)
    print(f'  Fecha          : {fecha}')
    print(f'  Picks analizados: {len(resultados)}')
    print(f'  Con UNDER signal: {n_under}')
    for r in resultados:
        optimas = r.get('señales_optimas', [])
        under   = next((s for s in optimas if s.get('direccion') == 'UNDER' and s.get('apostar')), None)
        tag     = f"UNDER {under['linea']} @{under['cuota']:.2f} ({under['confianza_señal']})" if under else 'sin señal UNDER'
        print(f'  {r["partido"][:40]:<40} {tag}')
    print('=' * 46)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D125-02: EvalGames Bridge — EVALUAR_GAMES → UNDER games signal')
    parser.add_argument('--fecha',   help='Fecha YYYY-MM-DD (default: hoy)')
    parser.add_argument('--dry-run', action='store_true', help='No guardar archivo')
    parser.add_argument('-v', '--verbose', action='store_true', help='Log detallado')
    args = parser.parse_args()
    run_bridge(fecha=args.fecha, dry_run=args.dry_run, verbose=args.verbose)

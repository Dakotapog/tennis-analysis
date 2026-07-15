"""
Nodo-98 — Meta-Señal Convergencia: scorer de picks del día.
D98-07 / D99-08: corre como PASO 3b en run_daily.py, después de edge_calculator.

Lee edge_report_FECHA.json (secciones: apostar, watchlist, sin_edge).
Calcula por pick:
  score_directo     — señales pro-favorito (max=5)
  score_rival_value — señal contraria: rival tiene valor (max=1)
  direccion         — FAVORITO | RIVAL | SPLIT
Emite reports/meta_signal_YYYYMMDD_HHMMSS.json (REPORTE_SOLO).

INVARIANTE (D96-01 / D98-03): NO modifica edge, kelly_kl, apostar ni ningún gate.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime


# ─── Umbrales congelados (H98-01, D99-03) ───────────────────────────────────
_IRP_DELTA_UMBRAL   = -0.10   # irp_rival.delta_return < umbral → +1 score
_SCORE_DIRECTO_GATE = 3       # picks destacados en output
_RIVAL_VALUE_EDGE   = -0.10   # edge_fav < umbral → score_rival_value = 1


# ─── Cálculo score_directo ───────────────────────────────────────────────────

def calc_score_directo(pick: dict) -> int:
    """Suma de señales pro-favorito (max=5). Nodo-98 §4.1 + D99-03."""
    score = 0
    if pick.get('markov_favorito') == 'HOT':
        score += 1
    if pick.get('confidence_flag') == 'STRONG':
        score += 1
    if pick.get('elo_dominance_axis') is True:
        score += 1
    if (pick.get('rfi_tier') or 0) >= 1:
        score += 1
    irp_rv = pick.get('irp_rival') or {}
    if (irp_rv.get('delta_return') or 0.0) < _IRP_DELTA_UMBRAL:
        score += 1
    return score


def calc_score_rival_value(pick: dict) -> int:
    """Señal contraria: 1 si rival tiene valor (D99-03 / D99-10)."""
    # rival_value_flag ya calculado en edge_calculator (L1332)
    if pick.get('rival_value_flag') is True:
        return 1
    # fallback: calcular desde edge si rival_value_flag no está disponible
    edge = pick.get('edge') or pick.get('edge_vs_mercado_rival')
    if edge is not None and edge < _RIVAL_VALUE_EDGE:
        return 1
    return 0


def calc_direccion(score_d: int, score_rv: int) -> str:
    """D99-03: SPLIT cuando ambas direcciones activas (conflicto)."""
    if score_rv >= 1 and score_d >= 2:
        return 'SPLIT'
    if score_rv >= 1:
        return 'RIVAL'
    return 'FAVORITO'


def score_pick(pick: dict) -> dict:
    """Calcula los 4 campos meta-señal para un pick. REPORTE_SOLO."""
    s_d  = calc_score_directo(pick)
    s_rv = calc_score_rival_value(pick)
    dir_ = calc_direccion(s_d, s_rv)

    senales_fav = []
    if pick.get('markov_favorito') == 'HOT':
        senales_fav.append('HOT')
    if pick.get('confidence_flag') == 'STRONG':
        senales_fav.append('STRONG')
    if pick.get('elo_dominance_axis') is True:
        senales_fav.append('ELO_DOM')
    if (pick.get('rfi_tier') or 0) >= 1:
        senales_fav.append(f"RFI_tier{pick.get('rfi_tier')}")
    irp_rv = pick.get('irp_rival') or {}
    if (irp_rv.get('delta_return') or 0.0) < _IRP_DELTA_UMBRAL:
        senales_fav.append('IRP_delta_negativo')

    senales_rival = ['RIVAL_VALUE'] if s_rv else []

    return {
        'partido':                   pick.get('partido', pick.get('favorito_predicho', '?')),
        'favorito':                  pick.get('favorito_predicho', ''),
        'score_directo':             s_d,
        'score_rival_value':         s_rv,
        'direccion':                 dir_,
        'senales_activas_fav':       senales_fav,
        'senales_activas_rival':     senales_rival,
        'rival_value_delegado_h8801': bool(s_rv >= 1),
        'edge':                      pick.get('edge'),
        'cuota':                     pick.get('cuota_fav'),
        'tier':                      pick.get('tier'),
        'seccion':                   pick.get('_seccion', ''),
    }


# ─── Resolución del edge_report del día ─────────────────────────────────────

def _find_latest_edge_report(reports_dir: str) -> str | None:
    pattern = os.path.join(reports_dir, 'edge_report_*.json')
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def load_all_picks(edge_report_path: str) -> list[dict]:
    """Lee todas las secciones del edge_report y etiqueta la sección de origen."""
    with open(edge_report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    picks = []
    for section in ('apostar', 'watchlist', 'sin_edge'):
        for p in report.get(section, []):
            p['_seccion'] = section
            picks.append(p)
    return picks


# ─── Runner principal ────────────────────────────────────────────────────────

def run(reports_dir: str = 'reports') -> str:
    """
    Lee el edge_report más reciente, calcula scores y escribe meta_signal_*.json.
    Retorna la ruta del archivo generado.
    """
    edge_path = _find_latest_edge_report(reports_dir)
    if not edge_path:
        print('[meta_signal_scorer] No se encontró edge_report en reports/. Saliendo.')
        sys.exit(0)

    print(f'[meta_signal_scorer] Leyendo: {edge_path}')
    picks = load_all_picks(edge_path)
    print(f'[meta_signal_scorer] {len(picks)} picks analizados (apostar+watchlist+sin_edge)')

    scored = [score_pick(p) for p in picks]

    destacados = [s for s in scored if s['score_directo'] >= _SCORE_DIRECTO_GATE]
    splits      = [s for s in scored if s['direccion'] == 'SPLIT']
    h98_n       = len([s for s in scored if s['score_directo'] >= _SCORE_DIRECTO_GATE
                        and s['direccion'] == 'FAVORITO'])

    fecha     = datetime.now().strftime('%Y%m%d')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(reports_dir, f'meta_signal_{timestamp}.json')

    output = {
        'generated_at':            datetime.now().isoformat(),
        'edge_report_fuente':      os.path.basename(edge_path),
        'fecha':                   fecha,
        'n_picks_analizados':      len(picks),
        'n_score_directo_3plus':   len(destacados),
        'n_split':                 len(splits),
        'h98_01_n_actual':         h98_n,
        'picks_score_directo_3plus': destacados,
        'picks_split':             splits,
        'todos_los_picks':         scored,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'[meta_signal_scorer] score_directo>=3: {len(destacados)} picks | SPLIT: {len(splits)}')
    print(f'[meta_signal_scorer] Output: {out_path}')
    return out_path


if __name__ == '__main__':
    run()

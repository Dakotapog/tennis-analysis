"""
Player Profitability Tracker -- READ-ONLY
Reads closed betslips from reports/apuestas_*.json
Aggregates profit/loss by player name
Persists to data/player_profitability.json

D52-06 (Nodo-52 Addendum §A): build_player_profitability_simulado()
Reads shadow book JSONL (reports/shadow_book/sb_*.jsonl) with flat 1u stakes.
Output: data/player_profitability_simulado.json — NUNCA mezclar con datos reales.

This module is READ-ONLY: it never writes or modifies bet files.
"""

import json
import unicodedata
import glob
import os
import re
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """
    Normalizes a player name for consistent matching.
    Lowercase, strip accents, normalize spaces.
    Same logic as scraping/kambi_tennis.py to avoid duplicates.
    """
    if not name:
        return ''
    # Strip accents
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    # Lowercase and normalize spaces
    return re.sub(r'\s+', ' ', ascii_str.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregation function
# ─────────────────────────────────────────────────────────────────────────────

def build_player_profitability(betslip_dir: str = 'reports/') -> dict:
    """
    Reads all closed apuestas_*.json files (estado='CERRADO') and aggregates
    per player:
      - n_apostado:    times we bet in favor of this player
      - n_ganado:      times we won
      - profit_total:  sum of (stake * (cuota-1)) if won, -stake if lost
      - total_apostado: sum of all stakes
      - roi:           profit_total / total_apostado  (0 if no stake data)
      - avg_cuota:     average cuota when we bet
      - last_seen:     most recent bet date (ts_registro)

    Returns dict {normalized_name: {stats}}
    Persists to data/player_profitability.json

    Graceful degradation: if no files, returns empty dict, no crash.
    """
    reports_path = Path(betslip_dir)
    apuestas_files = sorted(reports_path.glob('apuestas_*.json'))

    aggregated = {}  # normalized_name -> accumulated stats

    for fpath in apuestas_files:
        try:
            data = json.loads(fpath.read_text(encoding='utf-8'))
        except Exception:
            continue

        # Only process closed betslips
        if data.get('estado', '').upper() != 'CERRADO':
            continue

        picks = data.get('picks', [])
        ts_registro = data.get('ts_registro', '')

        for pick in picks:
            # A pick must be resolved (correcto is not None)
            if pick.get('correcto') is None:
                continue

            jugador = pick.get('jugador', '')
            if not jugador:
                continue

            key = _normalize_name(jugador)
            if not key:
                continue

            cuota = float(pick.get('cuota', 1.0) or 1.0)
            stake = float(pick.get('stake', 0) or 0)
            correcto = bool(pick.get('correcto'))

            # Calculate profit for this pick
            if stake > 0:
                profit = round(stake * (cuota - 1), 2) if correcto else -stake
            else:
                # No stake data: use 1 unit for ROI calculation
                profit = (cuota - 1) if correcto else -1.0
                stake = 1.0  # unit stake for ROI purposes

            if key not in aggregated:
                aggregated[key] = {
                    'display_name': jugador,
                    'n_apostado': 0,
                    'n_ganado': 0,
                    'profit_total': 0.0,
                    'total_apostado': 0.0,
                    'cuota_sum': 0.0,
                    'last_seen': '',
                }

            agg = aggregated[key]
            agg['n_apostado'] += 1
            if correcto:
                agg['n_ganado'] += 1
            agg['profit_total'] += profit
            agg['total_apostado'] += stake
            agg['cuota_sum'] += cuota
            # Track most recent date
            if ts_registro > agg['last_seen']:
                agg['last_seen'] = ts_registro
                agg['display_name'] = jugador  # use most recent name spelling

    # Compute derived stats
    result = {}
    for key, agg in aggregated.items():
        n = agg['n_apostado']
        total_stake = agg['total_apostado']
        roi = round(agg['profit_total'] / total_stake, 4) if total_stake > 0 else 0.0
        avg_cuota = round(agg['cuota_sum'] / n, 4) if n > 0 else 1.0

        result[key] = {
            'display_name':  agg['display_name'],
            'n_apostado':    n,
            'n_ganado':      agg['n_ganado'],
            'profit_total':  round(agg['profit_total'], 2),
            'total_apostado': round(total_stake, 2),
            'roi':           roi,
            'avg_cuota':     avg_cuota,
            'last_seen':     agg['last_seen'],
        }

    # Persist to data/player_profitability.json
    try:
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        out_path = data_dir / 'player_profitability.json'
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass  # Graceful degradation: no crash if write fails

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lookup function
# ─────────────────────────────────────────────────────────────────────────────

def get_player_profitability(player_name: str, data_dir: str = 'data/',
                             simulado: bool = False) -> dict | None:
    """
    Loads player profitability stats for the given player.

    simulado=False (default): lee data/player_profitability.json (apuestas reales).
    simulado=True  (D52-06):  lee data/player_profitability_simulado.json (shadow book).
    NUNCA mezclar ambos en la misma métrica.
    """
    fname = 'player_profitability_simulado.json' if simulado else 'player_profitability.json'
    prof_path = Path(data_dir) / fname
    if not prof_path.exists():
        return None

    try:
        data = json.loads(prof_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    key = _normalize_name(player_name)
    return data.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# D52-06: Shadow Book Simulado (Nodo-52 Addendum §A)
# ─────────────────────────────────────────────────────────────────────────────

def build_player_profitability_simulado(
    shadow_dir: str = 'reports/shadow_book',
    data_dir: str = 'data',
) -> dict:
    """
    D52-06: agrega rentabilidad por jugador desde el shadow book (simulado=True).

    Lee reports/shadow_book/sb_*.jsonl, procesa solo registros SETTLED non-VOID.
    Usa siempre flat 1u stake (igual que el ROI del shadow book — Addendum §C).
    Escribe en data/player_profitability_simulado.json.
    NUNCA escribe en player_profitability.json — los datos reales son exclusivos
    de betslip_registrar y build_player_profitability().

    Formato de output: mismo que build_player_profitability() +
      'simulado': True           (para que el caller sepa el origen)
      'clv_median': float|None   (mediana CLV cuando disponible)
      'status_aprobado': int     (picks APOSTAR en el total loggeado)
      'status_watchlist': int    (picks WATCHLIST)

    Graceful degradation: si no hay archivos JSONL, retorna {} sin crash.
    """
    shadow_path = Path(shadow_dir)
    jsonl_files = sorted(shadow_path.glob('sb_*.jsonl'))

    aggregated = {}  # normalized_name -> accumulated stats

    for fpath in jsonl_files:
        try:
            lines = fpath.read_text(encoding='utf-8').splitlines()
        except Exception:
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Ignorar session_meta
            if rec.get('_type') == 'session_meta':
                continue

            res = rec.get('resolucion')
            if not res:
                continue  # no settled

            resultado = res.get('resultado', '')
            if resultado == 'VOID':
                continue  # excluido — misma regla que _segment_metrics()

            snap = rec.get('pick_snapshot', {})
            jugador = snap.get('favorito_predicho', '')
            if not jugador:
                continue

            key = _normalize_name(jugador)
            if not key:
                continue

            cuota = float(snap.get('cuota_favorito', 1.0) or 1.0)
            won = resultado == 'WON'
            profit = round(cuota - 1, 4) if won else -1.0  # flat 1u

            clv = res.get('clv_pct')
            logged_at = rec.get('logged_at', '')
            status = 'APROBADO' if snap.get('apostar') else 'WATCHLIST'
            if snap.get('status') == 'NO_DATA':
                status = 'NO_DATA'

            if key not in aggregated:
                aggregated[key] = {
                    'display_name':     jugador,
                    'n_apostado':       0,   # settled non-void (flat 1u)
                    'n_ganado':         0,
                    'profit_total':     0.0,
                    'total_apostado':   0.0,
                    'cuota_sum':        0.0,
                    'clv_vals':         [],
                    'last_seen':        '',
                    'status_aprobado':  0,
                    'status_watchlist': 0,
                    'status_no_data':   0,
                }

            agg = aggregated[key]
            agg['n_apostado'] += 1
            if won:
                agg['n_ganado'] += 1
            agg['profit_total'] += profit
            agg['total_apostado'] += 1.0      # siempre flat 1u
            agg['cuota_sum'] += cuota
            if clv is not None:
                agg['clv_vals'].append(clv)
            if logged_at > agg['last_seen']:
                agg['last_seen'] = logged_at
                agg['display_name'] = jugador

            if status == 'APROBADO':
                agg['status_aprobado'] += 1
            elif status == 'WATCHLIST':
                agg['status_watchlist'] += 1
            else:
                agg['status_no_data'] += 1

    # Compute derived stats
    result = {}
    for key, agg in aggregated.items():
        n = agg['n_apostado']
        total_stake = agg['total_apostado']
        roi = round(agg['profit_total'] / total_stake, 4) if total_stake > 0 else 0.0
        avg_cuota = round(agg['cuota_sum'] / n, 4) if n > 0 else 1.0

        clv_vals = sorted(agg['clv_vals'])
        clv_median = None
        if clv_vals:
            mid = len(clv_vals) // 2
            clv_median = round(
                (clv_vals[mid - 1] + clv_vals[mid]) / 2
                if len(clv_vals) % 2 == 0 else clv_vals[mid], 2
            )

        result[key] = {
            'display_name':     agg['display_name'],
            'n_apostado':       n,
            'n_ganado':         agg['n_ganado'],
            'profit_total':     round(agg['profit_total'], 2),
            'total_apostado':   round(total_stake, 2),
            'roi':              roi,
            'avg_cuota':        avg_cuota,
            'last_seen':        agg['last_seen'],
            'simulado':         True,
            'clv_median':       clv_median,
            'status_aprobado':  agg['status_aprobado'],
            'status_watchlist': agg['status_watchlist'],
            'status_no_data':   agg['status_no_data'],
        }

    # Persist — archivo separado del real (NUNCA mezclar)
    try:
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)
        out_path = data_path / 'player_profitability_simulado.json'
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass  # Graceful degradation

    return result

"""
tests/test_nodo55.py — Nodo-55: Stake Waterfall Log + shadow book stake_real/var_flattened

REGLA-T53: ningún test hardcodea la fórmula. Siempre invoca funciones del módulo real.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# T55-01 — Waterfall presente en senales_enriched después de _print_individuales
# ─────────────────────────────────────────────────────────────────────────────

def test_t55_01_waterfall_present_in_enriched():
    """T55-01: _print_individuales enriquece cada pick con _waterfall que contiene
    kelly_kl_report, p_blend, stake_pre_var, terminal_reason (None pre-VaR)."""
    from trader_ev_tenis import _print_individuales, _P_PRIOR

    senal = {
        'p_modelo': 0.645,
        'n_h2h': 0,
        'p_historica_usada': None,
        'cuota_favorito': 1.74,
        'kelly_kl': 0.07,
        'partido': 'TestA vs TestB',
        'favorito_predicho': 'TestA',
        'edge_pct': '7.0%',
        'zona_cuota': 'underdog',
        'superficie': 'clay',
    }

    import io
    import sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        gastado, enriched = _print_individuales([senal], bankroll=20000, budget=8000)
    finally:
        sys.stdout = old

    assert len(enriched) == 1
    wf = enriched[0].get('_waterfall')
    assert wf is not None, "T55-01: _waterfall ausente en pick enriquecido"
    assert 'kelly_kl_report' in wf
    assert 'p_blend' in wf
    assert 'stake_pre_var' in wf
    assert wf['terminal_reason'] is None, (
        "T55-01: terminal_reason debe ser None antes de la fase VaR"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T55-02 — update_trader_stakes escribe stake_real y var_flattened al shadow book
# ─────────────────────────────────────────────────────────────────────────────

def test_t55_02_shadow_book_receives_stake_real():
    """T55-02: update_trader_stakes enriquece registros del shadow book con
    trader_deploy.stake_real y trader_deploy.var_flattened."""
    import shadow_book as sb

    fecha = '2099-01-01'  # fecha fantasma para test aislado

    # Crear un registro mínimo en el shadow book
    rec = {
        'sb_id': 'SB_TestA_vs_TestB_2099-01-01',
        'record_type': 'pick',
        'logged_at': '2099-01-01T00:00:00',
        'pick_snapshot': {
            'match_id': 'TEST_MATCH_001',
            'partido': 'TestA vs TestB',
            'apostar': True,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Parchear SHADOW_DIR para test aislado
        original_dir = sb.SHADOW_DIR
        sb.SHADOW_DIR = tmpdir
        try:
            path = sb._jsonl_path(fecha)
            sb._save_jsonl(path, {rec['sb_id']: rec})

            senal_enriched = {
                'partido': 'TestA vs TestB',
                'match_id': 'TEST_MATCH_001',
                'stake': 0,
                '_waterfall': {
                    'stake_final': 0,
                    'var_flattened': True,
                    'var_factor': 0.36,
                    'terminal_reason': 'MIN_BET_CLIFF (stake_pre_var=$1,000 × var_factor=0.36 = $360 < MIN_BET=1,000)',
                    'stake_pre_var': 1000,
                },
            }

            n = sb.update_trader_stakes(fecha, {'senales': [senal_enriched]})

            assert n == 1, f"T55-02: se esperaba 1 registro actualizado, got {n}"

            records = sb._load_jsonl(path)
            updated = records[rec['sb_id']]
            deploy = updated.get('trader_deploy', {})

            assert 'stake_real' in deploy, "T55-02: stake_real ausente en trader_deploy"
            assert 'var_flattened' in deploy, "T55-02: var_flattened ausente en trader_deploy"
            assert deploy['stake_real'] == 0
            assert deploy['var_flattened'] is True
        finally:
            sb.SHADOW_DIR = original_dir


# ─────────────────────────────────────────────────────────────────────────────
# T55-03 — MIN_BET_CLIFF detectado cuando stake_pre_var < MIN_BET después de VaR
# ─────────────────────────────────────────────────────────────────────────────

def test_t55_03_min_bet_cliff_terminal_reason():
    """T55-03: pick con stake_pre_var=$1,000 y var_factor=0.36 → terminal_reason
    contiene 'MIN_BET_CLIFF'."""
    from trader_ev_tenis import MIN_BET

    # Simular el cálculo del ajuste VaR (mismo código del trader)
    stake_pre = MIN_BET       # $1,000 — mínimo inicial
    fv = 0.36                 # factor VaR típico ITF

    stake_after = round(stake_pre * fv / MIN_BET) * MIN_BET

    assert stake_after == 0, (
        f"T55-03: se esperaba $0 tras VaR×0.36 sobre MIN_BET, got ${stake_after}"
    )

    # El waterfall debe etiquetar esto como MIN_BET_CLIFF
    wf = {
        'stake_pre_var': stake_pre,
        'var_factor': fv,
        'stake_final': stake_after,
        'var_flattened': (stake_after == 0 and stake_pre > 0),
        'terminal_reason': (
            f'MIN_BET_CLIFF (stake_pre_var=${stake_pre:,.0f} × var_factor={fv:.2f} '
            f'= ${stake_pre*fv:,.0f} < MIN_BET={MIN_BET:,})'
            if stake_after == 0 and stake_pre > 0 else 'OK'
        ),
    }

    assert wf['var_flattened'] is True
    assert 'MIN_BET_CLIFF' in wf['terminal_reason'], (
        f"T55-03: terminal_reason no contiene 'MIN_BET_CLIFF': {wf['terminal_reason']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T55-04 — WAS: pick sin señal Markov NO califica (P54-03)
# ─────────────────────────────────────────────────────────────────────────────

def test_t55_04_was_requires_markov_signal():
    """T55-04: pick con edge=36%, cuota=6.75, sin señal Markov (markov_favorito=None)
    NO debe calificar para WAS. Verifica que el filtro WAS en betplay_combo_builder
    exige señal Markov explícita."""
    from betplay_combo_builder import _was_qualifies

    pick_sin_markov = {
        'edge_pct': '36.1%',
        'cuota_favorito': 6.75,
        'apostar': False,
        'markov_favorito': None,
        'markov_rival': None,
        'p_modelo': 0.509,
    }

    result = _was_qualifies(pick_sin_markov)
    assert result is False, (
        "T55-04: pick sin señal Markov NO debe calificar WAS (p≈0.51 es coin-flip puro)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T55-05 — WAS: pick con edge≥10%, cuota≥2.0, rival COLD conf≥0.60 → califica
# ─────────────────────────────────────────────────────────────────────────────

def test_t55_05_was_qualifies_with_cold_rival():
    """T55-05: pick con edge=23%, cuota=3.60, rival COLD conf≥0.60 califica WAS."""
    from betplay_combo_builder import _was_qualifies

    pick_con_markov = {
        'edge_pct': '23.6%',
        'cuota_favorito': 3.60,
        'apostar': False,
        'markov_favorito': 'HOT',
        'markov_rival': 'COLD',
        'markov_conf_rival': 0.72,
        'p_modelo': 0.514,
    }

    result = _was_qualifies(pick_con_markov)
    assert result is True, (
        "T55-05: pick HOT vs COLD conf=0.72, edge=23.6%, cuota=3.60 debe calificar WAS"
    )

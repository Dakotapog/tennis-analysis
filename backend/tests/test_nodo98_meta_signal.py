"""
Tests Nodo-98 — Meta-Señal Convergencia (REGLA-T53).
Invocan funciones reales del módulo — nunca hardcodean la fórmula.
"""

import json
import os
import sys
import tempfile

import pytest

# Importar funciones reales del scorer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from meta_signal_scorer import (
    calc_score_directo,
    calc_score_rival_value,
    calc_direccion,
    score_pick,
    load_all_picks,
    run,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _pick(**kwargs) -> dict:
    """Construye un pick mínimo con defaults seguros."""
    base = {
        'favorito_predicho': 'Jugador A',
        'partido': 'Jugador A vs Jugador B',
        'confidence_flag': 'LOW',
        'markov_favorito': 'NEUTRAL',
        'elo_dominance_axis': False,
        'rfi_tier': 0,
        'irp_rival': {},
        'edge': 0.05,
        'rival_value_flag': False,
        'cuota_fav': 2.0,
        'tier': 'challenger',
        '_seccion': 'apostar',
    }
    base.update(kwargs)
    return base


def _edge_report(picks_apostar=None, picks_watchlist=None) -> dict:
    return {
        'metadata': {},
        'apostar': picks_apostar or [],
        'watchlist': picks_watchlist or [],
        'sin_edge': [],
        'sin_datos': [],
    }


# ─── Test 1: score_directo=0 cuando ninguna señal activa ─────────────────────

def test_score_directo_0_cuando_ninguna_senal():
    pick = _pick()
    assert calc_score_directo(pick) == 0


# ─── Test 2: score_directo=3 con HOT + STRONG + RFI ─────────────────────────

def test_score_directo_3_cuando_hot_strong_rfi():
    pick = _pick(
        markov_favorito='HOT',
        confidence_flag='STRONG',
        rfi_tier=1,
    )
    result = calc_score_directo(pick)
    assert result == 3
    scored = score_pick(pick)
    assert scored['score_directo'] == 3
    assert scored['direccion'] == 'FAVORITO'
    assert 'HOT' in scored['senales_activas_fav']
    assert 'STRONG' in scored['senales_activas_fav']
    assert 'RFI_tier1' in scored['senales_activas_fav']


# ─── Test 3: IRP delta_return < -0.10 activa señal ──────────────────────────

def test_irp_delta_activa_cuando_menor_umbral():
    pick_activo = _pick(irp_rival={'delta_return': -0.15, 'n_retornos': 5})
    pick_inactivo = _pick(irp_rival={'delta_return': -0.05, 'n_retornos': 5})
    pick_sin_irp = _pick(irp_rival={})

    assert calc_score_directo(pick_activo) == 1     # IRP activa
    assert calc_score_directo(pick_inactivo) == 0   # IRP no llega al umbral
    assert calc_score_directo(pick_sin_irp) == 0    # sin perfil IRP


# ─── Test 4: rival_value va a score_rival_value, NO a score_directo ──────────

def test_rival_value_va_a_score_rival_value_no_a_directo():
    pick = _pick(rival_value_flag=True, edge=-0.15)
    s_d  = calc_score_directo(pick)
    s_rv = calc_score_rival_value(pick)

    assert s_d == 0             # rival_value NO suma al score directo
    assert s_rv == 1            # rival_value SÍ suma al score_rival_value

    scored = score_pick(pick)
    assert scored['score_rival_value'] == 1
    assert scored['rival_value_delegado_h8801'] is True
    assert 'RIVAL_VALUE' in scored['senales_activas_rival']
    assert 'RIVAL_VALUE' not in scored['senales_activas_fav']


# ─── Test 5: direccion=SPLIT cuando ambos scores activos ─────────────────────

def test_direccion_split_cuando_ambos_scores_activos():
    # score_directo >= 2 Y score_rival_value >= 1 → SPLIT
    assert calc_direccion(score_d=2, score_rv=1) == 'SPLIT'
    assert calc_direccion(score_d=3, score_rv=1) == 'SPLIT'
    # solo rival → RIVAL
    assert calc_direccion(score_d=1, score_rv=1) == 'RIVAL'
    assert calc_direccion(score_d=0, score_rv=1) == 'RIVAL'
    # solo fav → FAVORITO
    assert calc_direccion(score_d=3, score_rv=0) == 'FAVORITO'
    assert calc_direccion(score_d=0, score_rv=0) == 'FAVORITO'


# ─── Test 6: filtro score_directo_3plus en output ────────────────────────────

def test_solo_score_directo_3plus_en_output_destacados():
    picks = [
        _pick(markov_favorito='HOT', confidence_flag='STRONG', rfi_tier=1),  # score=3
        _pick(markov_favorito='HOT', confidence_flag='STRONG'),               # score=2
        _pick(markov_favorito='HOT'),                                         # score=1
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'edge_report_20260714_000000.json')
        with open(report_path, 'w') as f:
            json.dump(_edge_report(picks_apostar=picks), f)

        out_path = run(reports_dir=tmpdir)
        with open(out_path) as f:
            output = json.load(f)

    destacados = output['picks_score_directo_3plus']
    assert len(destacados) == 1
    assert destacados[0]['score_directo'] == 3
    assert output['n_picks_analizados'] == 3


# ─── Test 7: output JSON escrito en reports/ ─────────────────────────────────

def test_output_json_escrito():
    pick = _pick(markov_favorito='HOT', confidence_flag='STRONG', rfi_tier=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'edge_report_20260714_000000.json')
        with open(report_path, 'w') as f:
            json.dump(_edge_report(picks_apostar=[pick]), f)

        out_path = run(reports_dir=tmpdir)

        assert os.path.exists(out_path)
        assert 'meta_signal_' in os.path.basename(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert 'picks_score_directo_3plus' in data
        assert 'h98_01_n_actual' in data
        assert data['n_picks_analizados'] == 1


# ─── Test 8: scorer no modifica edge ni kelly (REPORTE_SOLO invariante) ──────

def test_scorer_no_modifica_edge_ni_kelly():
    original_edge = 0.22
    original_kelly = 0.08
    pick = _pick(
        edge=original_edge,
        kelly_kl=original_kelly,
        apostar=True,
        markov_favorito='HOT',
        confidence_flag='STRONG',
        rfi_tier=1,
    )
    # Guardamos copia antes
    pick_before = json.loads(json.dumps(pick))

    scored = score_pick(pick)

    # El pick original NO fue mutado
    assert pick['edge'] == pick_before['edge']
    assert pick.get('kelly_kl') == pick_before.get('kelly_kl')
    assert pick.get('apostar') == pick_before.get('apostar')

    # El scored es un dict NUEVO con los campos meta añadidos
    assert 'score_directo' in scored
    assert scored['edge'] == original_edge        # edge copiado tal cual

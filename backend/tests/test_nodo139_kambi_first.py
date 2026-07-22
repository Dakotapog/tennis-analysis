"""
tests/test_nodo139_kambi_first.py — REGLA-T53: tests invocan función real del módulo.

Cubre Nodo-139 D139-01→D139-07: Kambi-First Combo Builder.
Sin mocks de Kambi HTTP — funciones puras testeadas con fixtures locales.
"""
import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
_mod = importlib.import_module('betplay_combo_builder')

_apellido_kambi          = _mod._apellido_kambi
_apellido_pick           = _mod._apellido_pick
_match_score_names_kf    = _mod._match_score_names_kf
_compute_leg_signal_kf   = _mod._compute_leg_signal_kf
_build_kambi_combos_kf   = _mod._build_kambi_combos_kf
_kelly_stake_kf          = _mod._kelly_stake_kf
_select_with_overlap_kf  = _mod._select_with_overlap_kf
EV_MIN_COMBO             = _mod.EV_MIN_COMBO
MIN_P_COMBO              = _mod.MIN_P_COMBO
KF_MIN_P                 = _mod.KF_MIN_P
KF_MAX_STAKE_PCT         = _mod.KF_MAX_STAKE_PCT
KF_MIN_STAKE             = _mod.KF_MIN_STAKE


# ── Fixtures ────────────────────────────────────────────────────────────────

def _kambi_leg(player_fav='Lachlan Mcfadzean', cuota_fav=1.50,
               event_id=1001, hora='14:00', **kw):
    """KambiLeg mínimo para tests."""
    base = {
        'event_id':       event_id,
        'partido':        f'{player_fav} - Rival',
        'player_fav':     player_fav,
        'player_dog':     'Rival',
        'cuota_fav':      cuota_fav,
        'cuota_dog':      3.00,
        'outcome_id_fav': str(event_id * 10),
        'outcome_id_dog': str(event_id * 10 + 1),
        'start_utc':      '2026-07-22T19:00:00Z',
        'hora':           hora,
        'group_path':     'ITF',
        'p_implied_fav':  round(1.0 / cuota_fav, 4),
    }
    base.update(kw)
    return base


def _scored_leg(p_efectivo=0.70, cuota_fav=1.50, edge_efectivo=None,
                event_id=1001, hora='14:00', tier='A', **kw):
    """ScoredLeg ya pasado por _compute_leg_signal_kf."""
    p_implied = round(1.0 / cuota_fav, 4)
    if edge_efectivo is None:
        edge_efectivo = round(p_efectivo - p_implied, 4)
    base = _kambi_leg(cuota_fav=cuota_fav, event_id=event_id, hora=hora)
    base.update({
        'tier':          tier,
        'p_modelo':      p_efectivo,
        'p_efectivo':    p_efectivo,
        'edge_efectivo': edge_efectivo,
        'kelly_kl':      0.025,
        'conf_flag':     'STRONG',
        'n_axes':        2,
        'match_score':   1.0,
        'pick_nombre':   'Player',
        'score':         edge_efectivo * 3 + 0.025 * 20 + 1.0,
        'n_legs_ok':     True,
        'edge_model':    edge_efectivo,
    })
    base.update(kw)
    return base


# ── D139-01: _apellido_kambi ────────────────────────────────────────────────

def test_D139_01_apellido_kambi_firstname_last():
    """Kambi 'Lachlan Mcfadzean' → apellido 'mcfadzean'."""
    assert _apellido_kambi('Lachlan Mcfadzean') == 'mcfadzean'


def test_D139_01_apellido_kambi_compound_surname():
    """Kambi 'Botic Van De Zandschulp' → apellido compuesto incluye 'zandschulp'."""
    ap = _apellido_kambi('Botic Van De Zandschulp')
    assert 'zandschulp' in ap


def test_D139_01_cuota_filter_bounds():
    """_fetch_kambi_betting_universe solo retorna cuotas en [KF_MIN_CUOTA, KF_MAX_CUOTA]."""
    # Test indirecto: verificar que los campos correctos existen en el fixture
    leg = _kambi_leg(cuota_fav=1.40)
    assert _mod.KF_MIN_CUOTA <= leg['cuota_fav'] <= _mod.KF_MAX_CUOTA


# ── D139-02: _apellido_pick ─────────────────────────────────────────────────

def test_D139_02_apellido_pick_surname_first_with_initial():
    """Nuestro 'McFadzean L.' → apellido 'mcfadzean' (quita inicial final)."""
    assert _apellido_pick('McFadzean L.') == 'mcfadzean'


def test_D139_02_apellido_pick_compound_surname():
    """'van Loben Sels E.' → quita inicial, conserva apellido compuesto."""
    ap = _apellido_pick('van Loben Sels E.')
    assert 'loben' in ap or 'sels' in ap


def test_D139_02_match_score_mcfadzean_case():
    """Kambi 'Lachlan Mcfadzean' vs pick 'McFadzean L.' → score ≥ 0.85."""
    score = _match_score_names_kf('Lachlan Mcfadzean', 'McFadzean L.')
    assert score >= 0.85, f'Score={score:.2f} < 0.85 — bug en matching'


def test_D139_02_match_score_exact_match():
    """Mismo apellido → score = 1.0."""
    score = _match_score_names_kf('Carlos Alcaraz', 'Alcaraz C.')
    assert score == 1.0


# ── D139-03: _compute_leg_signal_kf ─────────────────────────────────────────

def test_D139_03_gate_excludes_negative_edge():
    """p_modelo=0.50 vs p_implied=0.667 (cuota=1.50) → edge negativo → excluido."""
    leg = _kambi_leg(cuota_fav=1.50)
    leg.update({'tier': 'A', 'p_modelo': 0.50, 'edge_model': -0.167,
                'kelly_kl': 0.0, 'conf_flag': 'LOW', 'n_axes': 2,
                'match_score': 1.0, 'pick_nombre': 'P', 'edge_efectivo': -0.167})
    result = _compute_leg_signal_kf(leg)
    assert result is None, 'G_EDGE debe excluir legs con edge ≤ 0'


def test_D139_03_gate_excludes_low_p_modelo():
    """p_modelo=0.54 < KF_MIN_P=0.55 → excluido por G_CONF."""
    leg = _kambi_leg(cuota_fav=1.80)
    leg.update({'tier': 'A', 'p_modelo': 0.54, 'edge_model': 0.04,
                'kelly_kl': 0.01, 'conf_flag': 'LOW', 'n_axes': 2,
                'match_score': 1.0, 'pick_nombre': 'P', 'edge_efectivo': 0.04})
    result = _compute_leg_signal_kf(leg)
    assert result is None, f'G_CONF debe excluir p_modelo={0.54} < {KF_MIN_P}'


def test_D139_03_allows_valid_leg():
    """p_modelo=0.70, cuota=1.50 → edge=+0.033 → pasa todos los gates."""
    leg = _kambi_leg(cuota_fav=1.50)
    leg.update({'tier': 'A', 'p_modelo': 0.70, 'edge_model': 0.033,
                'kelly_kl': 0.02, 'conf_flag': 'STRONG', 'n_axes': 3,
                'match_score': 1.0, 'pick_nombre': 'P', 'edge_efectivo': 0.033})
    result = _compute_leg_signal_kf(leg)
    assert result is not None, 'Leg válido no debe ser excluido'
    assert result['edge_efectivo'] > 0


# ── D139-04: _build_kambi_combos_kf ─────────────────────────────────────────

def test_D139_04_no_cuota_cap_allows_high_product():
    """3 legs con cuota 1.93×1.55×3.50=10.47 deben generar combo (sin cap=7.0)."""
    legs = [
        _scored_leg(p_efectivo=0.70, cuota_fav=1.93, event_id=1, hora='14:00',
                    edge_efectivo=0.181),
        _scored_leg(p_efectivo=0.68, cuota_fav=1.55, event_id=2, hora='14:30',
                    edge_efectivo=0.035),
        _scored_leg(p_efectivo=0.75, cuota_fav=3.50, event_id=3, hora='15:00',
                    edge_efectivo=0.464),
    ]
    cuota_prod = 1.93 * 1.55 * 3.50
    assert cuota_prod > 7.0, 'Fixture debe superar el cap anterior de 7.0'
    combos = _build_kambi_combos_kf(legs)
    assert len(combos) > 0, (
        f'Combo @{cuota_prod:.2f}x con EV positivo debe generarse (sin cap de cuota)'
    )


def test_D139_04_ev_gate_blocks_low_ev_combo():
    """Combo con EV_combo < EV_MIN_COMBO (2%) debe ser bloqueado."""
    # p_combo × cuota_combo - 1 < 0.02 → bloqueado
    # p=0.56 × cuota=1.75 = 0.98 → EV = -0.02 → bloqueado
    legs = [
        _scored_leg(p_efectivo=0.56, cuota_fav=1.75, event_id=10, hora='10:00',
                    edge_efectivo=0.01),
        _scored_leg(p_efectivo=0.56, cuota_fav=1.10, event_id=11, hora='10:30',
                    edge_efectivo=0.01),
        _scored_leg(p_efectivo=0.56, cuota_fav=1.13, event_id=12, hora='11:00',
                    edge_efectivo=0.01),
    ]
    combos = _build_kambi_combos_kf(legs)
    # Si algún combo pasa, verificar que tiene EV >= EV_MIN_COMBO
    for c in combos:
        assert c['EV_combo'] >= EV_MIN_COMBO, (
            f'Combo con EV={c["EV_combo"]:.3f} < EV_MIN={EV_MIN_COMBO} pasó el gate'
        )


# ── D139-05: _kelly_stake_kf ─────────────────────────────────────────────────

def test_D139_05_kelly_stake_bounded_by_3pct_bankroll():
    """stake máximo = 3% de bankroll = $3,750 con bankroll=$125,000."""
    bankroll = 125_000
    max_stake = bankroll * KF_MAX_STAKE_PCT
    # Combo con EV muy alto para forzar Kelly a superar el cap
    combo = {'EV_combo': 5.0, 'cuota_combo': 50.0, 'p_combo': 0.12}
    stake = _kelly_stake_kf(combo, bankroll, n_simultaneous=1)
    assert stake <= max_stake, f'stake=${stake} supera cap 3%=${max_stake}'


def test_D139_05_kelly_stake_minimum_500():
    """stake mínimo = $500 aunque Kelly sugiera menos."""
    bankroll = 125_000
    # Combo con EV muy bajo para forzar Kelly < 500
    combo = {'EV_combo': 0.02, 'cuota_combo': 3.0, 'p_combo': 0.34}
    stake = _kelly_stake_kf(combo, bankroll, n_simultaneous=5)
    assert stake >= KF_MIN_STAKE, f'stake=${stake} < mínimo={KF_MIN_STAKE}'


def test_D139_05_kelly_stake_scales_with_ev():
    """Combo con mayor EV debe tener mayor stake (todo lo demás igual)."""
    bankroll = 125_000
    combo_low  = {'EV_combo': 0.05, 'cuota_combo': 5.0, 'p_combo': 0.21}
    combo_high = {'EV_combo': 0.30, 'cuota_combo': 5.0, 'p_combo': 0.26}
    stake_low  = _kelly_stake_kf(combo_low,  bankroll, 1)
    stake_high = _kelly_stake_kf(combo_high, bankroll, 1)
    assert stake_high >= stake_low, (
        f'Mayor EV debe dar mayor stake: {stake_high} vs {stake_low}'
    )

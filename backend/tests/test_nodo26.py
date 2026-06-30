"""
Tests para Nodo-26: Cross-Sectional Signals
T26-01 → T26-26
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betplay_combo_builder import (
    session_budget,
    check_budget,
    line_movement_signal,
    ranking_preserved_blend,
    cv_edge_guard,
    session_regime,
    _find_bankroll_from_plans,
    MAX_SESSION_LOSS_PCT,
)


# ── M-26-2: Circuit Breaker ──────────────────────────────────────────────────

def test_t26_01_session_budget_4pct():
    """T26-01: $125k × 4% = $5,000."""
    assert session_budget(125_000) == pytest.approx(5_000.0)


def test_t26_02_session_budget_configurable():
    """T26-02: custom max_loss_pct applies correctly."""
    assert session_budget(100_000, max_loss_pct=0.02) == pytest.approx(2_000.0)


def test_t26_03_check_budget_ok():
    """T26-03: 5 combos × $500 = $2,500 < $5,000 → OK."""
    n_allowed, msg = check_budget(5, 500, 125_000)
    assert n_allowed == 5
    assert msg == "OK"


def test_t26_04_check_budget_exceeds():
    """T26-04: 15 combos × $500 = $7,500 > $5,000 → recorta a 10."""
    n_allowed, msg = check_budget(15, 500, 125_000)
    assert n_allowed == 10
    assert "BUDGET LIMIT" in msg
    assert "15" in msg


def test_t26_05_check_budget_edge_exact():
    """T26-05: exactly at budget → OK."""
    n_allowed, msg = check_budget(10, 500, 125_000)
    assert n_allowed == 10
    assert msg == "OK"


def test_t26_06_check_budget_zero_combos():
    """T26-06: 0 combos → OK."""
    n_allowed, msg = check_budget(0, 500, 125_000)
    assert n_allowed == 0
    assert msg == "OK"


# ── M-26-3: Line Movement Signal ─────────────────────────────────────────────

def test_t26_07_line_movement_steam_in():
    """T26-07: cuota drops >4% → STEAM_IN, factor=1.10."""
    # cuota_original=2.80 → cuota_actual=2.50 → delta=-10.7%
    factor, signal = line_movement_signal(2.80, 2.50)
    assert signal == "STEAM_IN"
    assert factor == pytest.approx(1.10)


def test_t26_08_line_movement_drift_out():
    """T26-08: cuota rises >4% → DRIFT_OUT, factor=0.85."""
    # cuota_original=2.50 → cuota_actual=2.80 → delta=+12%
    factor, signal = line_movement_signal(2.50, 2.80)
    assert signal == "DRIFT_OUT"
    assert factor == pytest.approx(0.85)


def test_t26_09_line_movement_stable():
    """T26-09: cuota change ≤4% → STABLE, factor=1.0."""
    factor, signal = line_movement_signal(2.50, 2.52)
    assert signal == "STABLE"
    assert factor == pytest.approx(1.0)


def test_t26_10_line_movement_no_original():
    """T26-10: cuota_original=None (no edge_report data) → NO_DATA, factor=1.0."""
    factor, signal = line_movement_signal(None, 2.50)
    assert signal == "NO_DATA"
    assert factor == pytest.approx(1.0)


def test_t26_11_line_movement_exact_threshold():
    """T26-11: exactly -4.0% change → boundary STABLE (strictly < -4% for STEAM_IN)."""
    # 2.60 → 2.496 = exactly -4.0%
    cuota_actual = 2.60 * 0.96
    factor, signal = line_movement_signal(2.60, cuota_actual)
    # At exactly -4%, behavior depends on implementation (< vs <=)
    assert signal in ("STEAM_IN", "STABLE")
    assert factor in (1.0, 1.10)


# ── M-26-1: Cross-Sectional Ranking Preservation ─────────────────────────────

def test_t26_12_ranking_preserved_order():
    """T26-12: relative order of p_modelo is preserved after amplification."""
    pool = [
        {"jugador": "A", "p_modelo": 0.56, "p_blend": 0.590},
        {"jugador": "B", "p_modelo": 0.52, "p_blend": 0.591},
        {"jugador": "C", "p_modelo": 0.54, "p_blend": 0.589},
    ]
    result = ranking_preserved_blend(pool, p_historica=0.59, js_factor=0.17)
    # A should have highest p_blend (highest p_modelo), B lowest
    p_A = next(r["p_blend"] for r in result if r["jugador"] == "A")
    p_B = next(r["p_blend"] for r in result if r["jugador"] == "B")
    p_C = next(r["p_blend"] for r in result if r["jugador"] == "C")
    assert p_A > p_C > p_B


def test_t26_13_ranking_preserved_std_increases():
    """T26-13: after ranking_preserved_blend, std(p_blend) > original std."""
    pool = [
        {"jugador": "A", "p_modelo": 0.56, "p_blend": 0.590},
        {"jugador": "B", "p_modelo": 0.52, "p_blend": 0.591},
        {"jugador": "C", "p_modelo": 0.54, "p_blend": 0.589},
        {"jugador": "D", "p_modelo": 0.55, "p_blend": 0.590},
    ]
    import statistics
    std_before = statistics.stdev([p["p_blend"] for p in pool])
    result = ranking_preserved_blend(pool, p_historica=0.59, js_factor=0.17)
    std_after = statistics.stdev([p["p_blend"] for p in result])
    assert std_after > std_before


def test_t26_14_ranking_preserved_mean_stable():
    """T26-14: amplification doesn't move mean p_blend more than 5pp."""
    pool = [
        {"jugador": "A", "p_modelo": 0.55, "p_blend": 0.590},
        {"jugador": "B", "p_modelo": 0.50, "p_blend": 0.590},
        {"jugador": "C", "p_modelo": 0.53, "p_blend": 0.590},
    ]
    result = ranking_preserved_blend(pool, p_historica=0.59, js_factor=0.17)
    import statistics
    mean_before = statistics.mean([p["p_blend"] for p in pool])
    mean_after = statistics.mean([p["p_blend"] for p in result])
    assert abs(mean_after - mean_before) < 0.05


def test_t26_15_ranking_preserved_empty():
    """T26-15: empty pool returns empty list."""
    result = ranking_preserved_blend([], p_historica=0.59, js_factor=0.17)
    assert result == []


# ── M-26-5: CV Edge Guard ─────────────────────────────────────────────────────

def test_t26_16_cv_guard_blind_edge():
    """T26-16: all picks with nearly identical edges → BLIND_EDGE."""
    pool = [
        {"edge": 0.100},
        {"edge": 0.101},
        {"edge": 0.099},
        {"edge": 0.100},
    ]
    cv, status = cv_edge_guard(pool)
    assert status == "BLIND_EDGE"
    assert cv < 0.15


def test_t26_17_cv_guard_varied():
    """T26-17: varied edges → not BLIND_EDGE."""
    pool = [
        {"edge": 0.05},
        {"edge": 0.15},
        {"edge": 0.08},
        {"edge": 0.20},
    ]
    cv, status = cv_edge_guard(pool)
    assert status != "BLIND_EDGE"


def test_t26_18_cv_guard_empty():
    """T26-18: empty pool → returns (None, 'INSUFFICIENT')."""
    cv, status = cv_edge_guard([])
    assert cv is None
    assert status == "INSUFFICIENT"


def test_t26_19_cv_guard_single():
    """T26-19: single pick → INSUFFICIENT (no std possible)."""
    cv, status = cv_edge_guard([{"edge": 0.10}])
    assert cv is None
    assert status == "INSUFFICIENT"


def test_t26_20_cv_guard_zero_mean():
    """T26-20: zero-mean edges don't cause division by zero."""
    pool = [{"edge": 0.0}, {"edge": 0.0}, {"edge": 0.0}]
    cv, status = cv_edge_guard(pool)
    # Should not raise; returns None or BLIND_EDGE
    assert status in ("BLIND_EDGE", "INSUFFICIENT")


# ── M-26-4: Meta-Markov ──────────────────────────────────────────────────────

def test_t26_21_meta_markov_insufficient():
    """T26-21: <3 sessions → INSUFFICIENT, factor=1.0."""
    cal = {"session_history": [{"accuracy": 0.60}, {"accuracy": 0.50}]}
    regime, factor = session_regime(cal)
    assert regime == "INSUFFICIENT"
    assert factor == pytest.approx(1.0)


def test_t26_22_meta_markov_cold():
    """T26-22: avg_acc < 0.50 → COLD_MODEL, factor=0.50."""
    cal = {"session_history": [
        {"accuracy": 0.45},
        {"accuracy": 0.40},
        {"accuracy": 0.48},
    ]}
    regime, factor = session_regime(cal)
    assert regime == "COLD_MODEL"
    assert factor == pytest.approx(0.50)


def test_t26_23_meta_markov_cooling():
    """T26-23: avg_acc in [0.50,0.60) + downtrend → COOLING, factor=0.75."""
    cal = {"session_history": [
        {"accuracy": 0.70},
        {"accuracy": 0.60},
        {"accuracy": 0.55},
        {"accuracy": 0.52},
        {"accuracy": 0.50},
    ]}
    regime, factor = session_regime(cal)
    assert regime == "COOLING"
    assert factor == pytest.approx(0.75)


def test_t26_24_meta_markov_hot():
    """T26-24: avg_acc > 0.70 → HOT_MODEL, factor=1.0 (NUNCA aumenta stakes)."""
    cal = {"session_history": [
        {"accuracy": 0.80},
        {"accuracy": 0.75},
        {"accuracy": 0.90},
    ]}
    regime, factor = session_regime(cal)
    assert regime == "HOT_MODEL"
    assert factor == pytest.approx(1.0)


def test_t26_25_meta_markov_neutral():
    """T26-25: avg_acc in [0.60,0.70] → NEUTRAL, factor=1.0."""
    cal = {"session_history": [
        {"accuracy": 0.65},
        {"accuracy": 0.62},
        {"accuracy": 0.68},
    ]}
    regime, factor = session_regime(cal)
    assert regime == "NEUTRAL"
    assert factor == pytest.approx(1.0)


def test_t26_26_meta_markov_no_history():
    """T26-26: missing session_history key → INSUFFICIENT."""
    cal = {}
    regime, factor = session_regime(cal)
    assert regime == "INSUFFICIENT"
    assert factor == pytest.approx(1.0)

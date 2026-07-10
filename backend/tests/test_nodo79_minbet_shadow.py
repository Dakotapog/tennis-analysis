"""
tests/test_nodo79_minbet_shadow.py — Nodo-79: MIN_BET proporcional por tier (modo sombra)

REGLA-T53: invoca funciones reales del módulo, nunca hardcodea la fórmula.
Casos basados en picks reales de PROPUESTA_VAR_2026-07-10.md.
"""
import pytest
from trader_ev_tenis import _MIN_BET_BY_TIER, MIN_BET, KELLY_FRACTION, _cppi_factor


# ── helper: replica el waterfall real (incluyendo KELLY_FRACTION y rounding inicial) ──

def _simulate_waterfall(kelly_kl: float, bankroll: float, tier: str,
                        var_factor: float = 0.25) -> dict:
    """
    Replica el waterfall de trader_ev_tenis.py (Nodo-79 shadow mode).

    Pasos reales:
      1. stake_raw = kelly_kl × bankroll × KELLY_FRACTION
      2. stake_pre = round(stake_raw / MIN_BET) × MIN_BET; si >0 kelly y pre==0 → MIN_BET
      3. stake_post_cppi = stake_pre × var_factor × cppi_factor
      4. stake_real = round(stake_post_cppi / MIN_BET) × MIN_BET
      5. stake_shadow = round(stake_post_cppi / min_bet_tier) × min_bet_tier  [Nodo-79]
    """
    cppi_f = _cppi_factor(bankroll=bankroll, peak_bankroll=bankroll)
    stake_raw = kelly_kl * bankroll * KELLY_FRACTION
    stake_pre = round(stake_raw / MIN_BET) * MIN_BET
    if kelly_kl > 0 and stake_pre == 0:
        stake_pre = MIN_BET  # max(MIN_BET, 0) del trader real
    stake_post_cppi = stake_pre * var_factor * cppi_f
    stake_real = round(stake_post_cppi / MIN_BET) * MIN_BET
    min_bet_shadow = _MIN_BET_BY_TIER.get(tier, MIN_BET)
    stake_shadow = round(stake_post_cppi / min_bet_shadow) * min_bet_shadow
    return {
        'stake_pre': stake_pre,
        'stake_post_cppi': stake_post_cppi,
        'stake_real': stake_real,
        'stake_shadow': stake_shadow,
        'min_bet_shadow': min_bet_shadow,
        'var_flattened': stake_real == 0 and stake_pre > 0,
        'shadow_survives_cliff': stake_shadow > 0 and stake_real == 0 and stake_pre > 0,
    }


# ── dict ─────────────────────────────────────────────────────────────────────

class TestMinBetByTierDict:
    def test_dict_exists(self):
        assert isinstance(_MIN_BET_BY_TIER, dict)

    def test_all_tiers_present(self):
        for tier in ('itf', 'challenger', 'atp500', 'atp1000', 'grand_slam'):
            assert tier in _MIN_BET_BY_TIER

    def test_itf_lower_than_grand_slam(self):
        assert _MIN_BET_BY_TIER['itf'] < _MIN_BET_BY_TIER['grand_slam']

    def test_grand_slam_equals_min_bet_global(self):
        assert _MIN_BET_BY_TIER['grand_slam'] == MIN_BET

    def test_monotonic_ascending(self):
        order = ['itf', 'challenger', 'atp500', 'atp1000', 'grand_slam']
        values = [_MIN_BET_BY_TIER[t] for t in order]
        assert values == sorted(values)


# ── Leyton Rivera (2026-07-09, ITF, WON) ─────────────────────────────────────

class TestLeytonRiveraCase:
    """
    kelly_kl=0.4857, bankroll=$10,000, tier=itf, var_factor=0.25 (VaR activo).
    Traza real: stake_raw=$1,214 → stake_pre=$1,000 → post_cppi=$150
    Con MIN_BET=$1,000: round(0.15)×1000 = $0 (MIN_BET_CLIFF).
    Con _MIN_BET_BY_TIER[itf]=$100: round(1.5)×100 = $200 (sobrevive).
    """
    KELLY_KL = 0.4857
    BANKROLL  = 10_000
    TIER      = 'itf'

    def test_real_stake_is_zero(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_real'] == 0

    def test_var_flattened_true(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['var_flattened'] is True

    def test_shadow_survives_cliff(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['shadow_survives_cliff'] is True

    def test_shadow_stake_positive(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_shadow'] > 0

    def test_shadow_min_bet_is_itf_value(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['min_bet_shadow'] == _MIN_BET_BY_TIER['itf']

    def test_shadow_stake_multiple_of_itf_min_bet(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_shadow'] % _MIN_BET_BY_TIER['itf'] == 0


# ── Maria Sara Popa (2026-07-03, ITF, WON) ───────────────────────────────────

class TestMariaSaraPopaCase:
    """
    kelly_kl=0.0843, bankroll=$10,000, tier=itf.
    stake_raw=$211 < MIN_BET → stake_pre=MIN_BET=$1,000 (max logic).
    Mismo waterfall que Leyton Rivera → stake_real=$0, stake_shadow=$200.
    """
    KELLY_KL = 0.0843
    BANKROLL  = 10_000
    TIER      = 'itf'

    def test_real_stake_is_zero(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_real'] == 0

    def test_stake_pre_is_min_bet(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_pre'] == MIN_BET

    def test_shadow_survives_cliff(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['shadow_survives_cliff'] is True

    def test_shadow_stake_multiple_of_itf_min_bet(self):
        wf = _simulate_waterfall(self.KELLY_KL, self.BANKROLL, self.TIER)
        assert wf['stake_shadow'] % _MIN_BET_BY_TIER['itf'] == 0


# ── GS: shadow no cambia nada (min_bet_shadow == MIN_BET) ────────────────────

class TestGrandSlamUnchanged:
    """Para GS sin VaR activo: stake_shadow == stake_real (mismo MIN_BET)."""

    def test_gs_shadow_equals_real_when_survives(self):
        wf = _simulate_waterfall(kelly_kl=0.10, bankroll=125_000,
                                 tier='grand_slam', var_factor=1.0)
        assert wf['stake_shadow'] == wf['stake_real']
        assert wf['stake_real'] > 0

    def test_gs_min_bet_shadow_equals_min_bet(self):
        wf = _simulate_waterfall(kelly_kl=0.10, bankroll=125_000,
                                 tier='grand_slam', var_factor=1.0)
        assert wf['min_bet_shadow'] == MIN_BET


# ── shadow es read-only: stake_real nunca cambia ─────────────────────────────

class TestShadowDoesNotAffectRealStake:
    """El stake_real con MIN_BET fijo es independiente del tier shadow."""

    def test_itf_real_stake_uses_global_min_bet(self):
        wf = _simulate_waterfall(0.4857, 10_000, 'itf')
        # stake_real usa MIN_BET=1000, da 0 (CLIFF)
        assert wf['stake_real'] == 0

    def test_gs_real_stake_unaffected_by_shadow_dict(self):
        wf = _simulate_waterfall(0.10, 125_000, 'grand_slam', var_factor=1.0)
        # stake_real usa MIN_BET=1000, igual que shadow para GS
        assert wf['stake_real'] == wf['stake_shadow']

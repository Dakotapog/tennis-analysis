"""
Tests para Nodo-38: Portfolio con Aislamiento de Riesgo (CORE / Satellite / Moonshot)
T38-01 → T38-25
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_confianza_builder import (
    _categorizar_pick,
    _select_core,
    _validate_core_pwin,
    _build_portfolio_v2,
    _calc_combo,
    _total_stakes,
    _scale_stakes,
    _build_cobertura,
    _all_combos,
    _resumen_portfolio,
    _extract_and_categorize,
    CUOTA_MIN, CUOTA_CAT_B, CUOTA_CAT_C, CUOTA_C1_MAX, CUOTA_PIPELINE_MAX,
    CONF_MIN, CONF_C1, CONF_C1_PIPELINE, CONF_MOONSHOT,
    PAREJO_CONF_MAX, PAREJO_CUOTA_MIN, PAREJO_CUOTA_MAX,
    CORE_MAX_SIZE, CORE_MIN_SIZE, CORE_MIN_PWIN,
    MAX_SAME_TOURNAMENT, SAT_BASE_SIZE, MAX_SATELLITES,
    PHASE_CONFIG,
    BUDGET_CORE_PCT, BUDGET_SAT_PCT, BUDGET_MOONSHOT_PCT,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pick(nombre, confianza, cuota, torneo='Torneo X',
               pipeline_picks=None):
    """Helper para crear un pick categorizado."""
    cat = _categorizar_pick(cuota, confianza, pipeline_picks, nombre)
    if cat is None:
        return None
    return {
        'nombre': nombre,
        'confianza': confianza,
        'cuota': cuota,
        'p_modelo': confianza / 100.0,
        'torneo': torneo,
        'rival': 'Rival',
        'cat': cat,
    }


def _make_picks_pool(n_cat_a=4, n_cat_b=3, n_cat_c1=2, n_cat_c2=1):
    """Genera un pool de picks para tests de portfolio."""
    picks = []
    # Cat-A: cuota 1.20-1.50, confianza 70-90
    for i in range(n_cat_a):
        p = _make_pick(f'SafePlayer_{i}', 90 - i * 5, 1.20 + i * 0.10,
                       torneo=f'ITF Torneo {i}')
        if p:
            picks.append(p)
    # Cat-B: cuota 1.65-2.00, confianza 60-70
    for i in range(n_cat_b):
        p = _make_pick(f'ValuePlayer_{i}', 68 - i * 3, 1.65 + i * 0.15,
                       torneo=f'Challenger Torneo {i}')
        if p:
            picks.append(p)
    # Cat-C1: cuota 2.50-3.40, confianza 62-70
    for i in range(n_cat_c1):
        p = _make_pick(f'HighValue_{i}', 65 - i * 3, 2.50 + i * 0.40,
                       torneo=f'ITF HV Torneo {i}')
        if p:
            picks.append(p)
    # Cat-C2: cuota 3.60-4.00, confianza 55-58
    for i in range(n_cat_c2):
        p = _make_pick(f'Moonshot_{i}', 57, 3.60 + i * 0.40,
                       torneo=f'Challenger MS Torneo {i}')
        if p:
            picks.append(p)

    picks.sort(key=lambda x: x['confianza'], reverse=True)
    return picks


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORIZACIÓN (_categorizar_pick)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCategorizarPick:
    """T38-01 → T38-10: categorización correcta de picks."""

    def test_t38_01_cat_a_cuota_baja(self):
        """T38-01: cuota 1.30, confianza 65% → CAT_A, entra en CORE."""
        result = _categorizar_pick(1.30, 65.0)
        assert result is not None
        assert result['categoria'] == 'CAT_A'
        assert 'CORE' in result['combos_permitidos']

    def test_t38_02_excluir_cuota_muy_baja(self):
        """T38-02: cuota < 1.15 → excluir (no aporta odds)."""
        assert _categorizar_pick(1.10, 80.0) is None
        assert _categorizar_pick(1.14, 90.0) is None

    def test_t38_03_excluir_confianza_baja(self):
        """T38-03: confianza < 53% → excluir completamente."""
        assert _categorizar_pick(1.50, 52.0) is None
        assert _categorizar_pick(2.50, 50.0) is None

    def test_t38_04_excluir_parejo(self):
        """T38-04: confianza <55%, cuota 1.55-1.70 → excluir (parejo)."""
        # Draper @1.61 conf 54% → parejo
        assert _categorizar_pick(1.61, 54.0) is None
        assert _categorizar_pick(1.55, 54.9) is None
        assert _categorizar_pick(1.70, 53.5) is None
        # Edge cases: just outside parejo
        assert _categorizar_pick(1.54, 54.0) is not None  # cuota below parejo range
        assert _categorizar_pick(1.61, 55.0) is not None  # confianza above parejo

    def test_t38_05_cat_b_cuota_media(self):
        """T38-05: cuota 1.85, confianza 60% → CAT_B."""
        result = _categorizar_pick(1.85, 60.0)
        assert result['categoria'] == 'CAT_B'
        assert 'CORE' in result['combos_permitidos']

    def test_t38_06_cat_c1_satellite_eligible(self):
        """T38-06: cuota 2.75, confianza 65% → CAT_C1, satellite + moonshot."""
        result = _categorizar_pick(2.75, 65.0)
        assert result['categoria'] == 'CAT_C1'
        assert 'SATELLITE' in result['combos_permitidos']
        assert 'MOONSHOT' in result['combos_permitidos']
        assert 'CORE' not in result['combos_permitidos']

    def test_t38_07_cat_c2_moonshot_only(self):
        """T38-07: cuota 3.60, confianza 58% → CAT_C2, moonshot only."""
        result = _categorizar_pick(3.60, 58.0)
        assert result['categoria'] == 'CAT_C2'
        assert result['combos_permitidos'] == ['MOONSHOT']

    def test_t38_08_cat_c2_high_cuota_high_conf(self):
        """T38-08: cuota 3.60, confianza 65% → CAT_C2 (cuota > 3.50)."""
        result = _categorizar_pick(3.60, 65.0)
        assert result['categoria'] == 'CAT_C2'

    def test_t38_09_pipeline_promotes_c2_to_c1(self):
        """T38-09: señal doble promueve Cat-C2 si cuota ≤4.50 y conf ≥57%."""
        # Sin pipeline: CAT_C2
        result_no_pipe = _categorizar_pick(3.60, 58.0)
        assert result_no_pipe['categoria'] == 'CAT_C2'
        # Con pipeline: promovido a CAT_C1
        result_pipe = _categorizar_pick(3.60, 58.0,
                                         pipeline_picks={'da silva'},
                                         nombre='Da Silva')
        assert result_pipe['categoria'] == 'CAT_C1'
        assert result_pipe['pipeline_flag'] is True

    def test_t38_10_pipeline_no_promote_extreme_cuota(self):
        """T38-10: pipeline NO promueve si cuota > 4.50."""
        result = _categorizar_pick(5.00, 60.0,
                                    pipeline_picks={'longshot'},
                                    nombre='Longshot')
        assert result['categoria'] == 'CAT_C2'


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoreConstruction:
    """T38-11 → T38-14: construcción del CORE."""

    def test_t38_11_core_never_has_cat_c(self):
        """T38-11: CORE nunca incluye picks Cat-C (REGLA-ISO-1)."""
        picks = _make_picks_pool(n_cat_a=4, n_cat_b=2, n_cat_c1=2)
        cat_ab = [p for p in picks if p['cat']['categoria'] in ('CAT_A', 'CAT_B')]
        core = _select_core(cat_ab)
        for p in core:
            assert p['cat']['categoria'] in ('CAT_A', 'CAT_B'), \
                f"Cat-C pick {p['nombre']} found in CORE"

    def test_t38_12_core_max_7_piernas(self):
        """T38-12: CORE no excede 7 piernas."""
        picks = _make_picks_pool(n_cat_a=6, n_cat_b=5)
        cat_ab = [p for p in picks if p['cat']['categoria'] in ('CAT_A', 'CAT_B')]
        core = _select_core(cat_ab, max_size=CORE_MAX_SIZE)
        assert len(core) <= CORE_MAX_SIZE

    def test_t38_13_tournament_concentration_guard(self):
        """T38-13: max 2 picks del mismo torneo en CORE."""
        picks = [
            _make_pick('A', 90, 1.20, torneo='ITF Brussels'),
            _make_pick('B', 85, 1.25, torneo='ITF Brussels'),
            _make_pick('C', 80, 1.30, torneo='ITF Brussels'),
            _make_pick('D', 75, 1.35, torneo='Challenger Plovdiv'),
            _make_pick('E', 70, 1.40, torneo='Challenger Plovdiv'),
        ]
        core = _select_core(picks, max_same_tournament=MAX_SAME_TOURNAMENT)
        brussels = [p for p in core if p['torneo'] == 'ITF Brussels']
        assert len(brussels) <= MAX_SAME_TOURNAMENT
        # C (confianza 80) debería ser excluido, D o E entran
        assert len(core) >= 4  # 2 Brussels + 2 Plovdiv

    def test_t38_14_core_pwin_validation(self):
        """T38-14: CORE reduce tamaño si P(win) < 25%."""
        # 7 picks con cuota alta (1.55 each) → P(C7) = (1/1.55)^7 ≈ 4.5% < 25%
        picks = [_make_pick(f'P{i}', 60 - i, 1.55, torneo=f'T{i}') for i in range(7)]
        picks = [p for p in picks if p is not None]
        validated = _validate_core_pwin(list(picks))
        # Should reduce size until P > 25%
        p_win = 1.0
        for p in validated:
            p_win *= min(1.0 / p['cuota'], 0.95)
        assert p_win >= CORE_MIN_PWIN or len(validated) == CORE_MIN_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioConstruction:
    """T38-15 → T38-20: construcción del portfolio completo."""

    def test_t38_15_fase_1_core_only(self):
        """T38-15: Fase 1 solo produce CORE, sin satellites ni moonshot."""
        picks = _make_picks_pool()
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=1)
        assert plan['core'] is not None
        assert len(plan['satellites']) == 0
        assert plan['moonshot'] is None
        assert len(plan['cobertura']) == 0

    def test_t38_16_fase_2_core_plus_1_satellite(self):
        """T38-16: Fase 2 produce CORE + max 1 satellite."""
        picks = _make_picks_pool(n_cat_c1=3)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=2)
        assert plan['core'] is not None
        assert len(plan['satellites']) <= 1
        assert plan['moonshot'] is None

    def test_t38_17_fase_3_full_architecture(self):
        """T38-17: Fase 3 produce CORE + 3 SAT + moonshot, sin cobertura."""
        picks = _make_picks_pool(n_cat_a=5, n_cat_b=3, n_cat_c1=3, n_cat_c2=1)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)
        assert plan['core'] is not None
        assert len(plan['satellites']) <= MAX_SATELLITES
        # moonshot only if ≥2 Cat-C SILVER+
        assert len(plan['cobertura']) == 0

    def test_t38_18_fase_4_full_with_cobertura(self):
        """T38-18: Fase 4 produce todo incluida cobertura."""
        picks = _make_picks_pool(n_cat_a=6, n_cat_b=4, n_cat_c1=2, n_cat_c2=1)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=4)
        assert plan['core'] is not None
        # Cobertura should exist if there are enough Cat-AB reserve picks
        # (6+4=10 Cat-AB, CORE takes ≤7, so ≥3 reserves)
        if len([p for p in picks if p['cat']['categoria'] in ('CAT_A', 'CAT_B')]) > CORE_MAX_SIZE:
            assert len(plan['cobertura']) > 0

    def test_t38_19_satellite_has_exactly_one_cat_c(self):
        """T38-19: cada satellite tiene exactamente 1 pick Cat-C."""
        picks = _make_picks_pool(n_cat_c1=2)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)
        for sat in plan['satellites']:
            cat_c_count = sum(
                1 for c in sat['categorias'] if c in ('CAT_C1', 'CAT_C2')
            )
            assert cat_c_count == 1, \
                f"Satellite {sat['nombre']} has {cat_c_count} Cat-C picks, expected 1"

    def test_t38_20_moonshot_needs_2_cat_c_silver(self):
        """T38-20: moonshot requiere ≥2 Cat-C con conf ≥57% (SILVER+)."""
        # Only 1 Cat-C SILVER → no moonshot
        picks = _make_picks_pool(n_cat_a=4, n_cat_b=2, n_cat_c1=1, n_cat_c2=0)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)
        assert plan['moonshot'] is None


# ═══════════════════════════════════════════════════════════════════════════════
# VaR GUARD AND SIZING
# ═══════════════════════════════════════════════════════════════════════════════

class TestVaRGuard:
    """T38-21 → T38-25: VaR guard y sizing."""

    def test_t38_21_var_guard_scales_stakes(self):
        """T38-21: total stakes nunca excede budget diario."""
        picks = _make_picks_pool(n_cat_a=6, n_cat_b=4, n_cat_c1=3, n_cat_c2=1)
        bankroll = 125000
        for fase in [1, 2, 3, 4]:
            plan = _build_portfolio_v2(picks, bankroll=bankroll, fase=fase)
            total = _total_stakes(plan)
            max_budget = bankroll * PHASE_CONFIG[fase]['max_daily_pct']
            assert total <= max_budget * 1.01, \
                f"Fase {fase}: total {total} > budget {max_budget}"

    def test_t38_22_budget_allocation_proportions(self):
        """T38-22: CORE gets ~45% of budget, satellites ~15% each."""
        picks = _make_picks_pool(n_cat_a=5, n_cat_b=3, n_cat_c1=2)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)
        budget = plan['budget']
        if plan['core']:
            # CORE should be roughly 45% of budget (with rounding)
            core_pct = plan['core']['stake'] / budget
            assert 0.30 <= core_pct <= 0.55, f"CORE is {core_pct:.1%} of budget"

    def test_t38_23_scale_stakes_preserves_ratios(self):
        """T38-23: _scale_stakes reduce todos proporcionalmente."""
        plan = {
            'core': _calc_combo(
                [_make_pick('A', 80, 1.30), _make_pick('B', 75, 1.25)],
                10000, 'CORE'),
            'satellites': [],
            'moonshot': None,
            'cobertura': [],
        }
        original_stake = plan['core']['stake']
        _scale_stakes(plan, 0.5)
        assert plan['core']['stake'] < original_stake

    def test_t38_24_min_stake_500(self):
        """T38-24: stake mínimo es $500 después de scaling."""
        picks = _make_picks_pool(n_cat_a=4)
        # Very small bankroll → all stakes should be at least 500
        plan = _build_portfolio_v2(picks, bankroll=5000, fase=1)
        for combo in _all_combos(plan):
            assert combo['stake'] >= 500

    def test_t38_25_phase_budget_limits(self):
        """T38-25: cada fase tiene su budget máximo correcto."""
        assert PHASE_CONFIG[1]['max_daily_pct'] == 0.02
        assert PHASE_CONFIG[2]['max_daily_pct'] == 0.04
        assert PHASE_CONFIG[3]['max_daily_pct'] == 0.07
        assert PHASE_CONFIG[4]['max_daily_pct'] == 0.12


# ═══════════════════════════════════════════════════════════════════════════════
# AISLAMIENTO CAT-C — Validación del patrón real del 26-jun (Nodo-38 §1.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAislamientoCatC:
    """
    T38-26/27/28: verifica que la arquitectura impide reconstruir el
    'C11 mezclado' del 26-jun que tenía 2 Cat-C (Da Silva + Cardozo) en
    el mismo combo no-moonshot.

    Caso real:
      - C11 incluía Da Silva @3.60 (Cat-C2) + Cardozo @2.55 (Cat-C1) + 9 Cat-A/B
      - AMBOS Cat-C fallaron; los 9 Cat-A/B ganaron
      - El patrón viola REGLA-ISO-2 (max 1 Cat-C por satellite)
    """

    def _make_multi_cat_c_pool(self, n_c1=4, n_c2=2):
        """Pool con múltiples Cat-C para probar aislamiento."""
        picks = []
        for i in range(6):
            p = _make_pick(f'Base_{i}', 85 - i * 3, 1.20 + i * 0.06,
                           torneo=f'ITF Base {i}')
            if p:
                picks.append(p)
        for i in range(n_c1):
            p = _make_pick(f'C1_{i}', 63 - i * 2, 2.50 + i * 0.20,
                           torneo=f'Challenger C1 {i}')
            if p:
                picks.append(p)
        for i in range(n_c2):
            p = _make_pick(f'C2_{i}', 58, 3.60 + i * 0.30,
                           torneo=f'Challenger C2 {i}')
            if p:
                picks.append(p)
        picks.sort(key=lambda x: x['confianza'], reverse=True)
        return picks

    def test_t38_26_satellite_max_1_cat_c_con_pool_amplio(self):
        """
        T38-26: mutación — pool con 4 Cat-C1 + 2 Cat-C2 → ningún satellite
        acumula 2+ Cat-C. El guard es estructural (loop 1-by-1), no condicional.
        """
        picks = self._make_multi_cat_c_pool(n_c1=4, n_c2=2)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)

        # Verificar: cada satellite tiene exactamente 1 Cat-C
        for sat in plan['satellites']:
            cat_c_count = sum(
                1 for c in sat['categorias'] if c in ('CAT_C1', 'CAT_C2')
            )
            assert cat_c_count == 1, (
                f"AISLAMIENTO ROTO: satellite '{sat['nombre']}' tiene "
                f"{cat_c_count} picks Cat-C (esperado: 1). "
                f"Pickers: {sat['nombres']}"
            )

        # Verificar: con 4 Cat-C1 disponibles, hay exactamente 3 satellites (max)
        assert len(plan['satellites']) <= MAX_SATELLITES

    def test_t38_27_core_cero_cat_c_con_pool_amplio(self):
        """
        T38-27: CORE no contiene ningún Cat-C incluso cuando el pool tiene
        6 Cat-C disponibles. REGLA-ISO-1 es absoluta.
        """
        picks = self._make_multi_cat_c_pool(n_c1=4, n_c2=2)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)

        if plan['core']:
            for cat in plan['core']['categorias']:
                assert cat not in ('CAT_C1', 'CAT_C2'), (
                    f"REGLA-ISO-1 VIOLADA: CORE contiene Cat-C "
                    f"picks={plan['core']['nombres']}"
                )

    def test_t38_28_moonshot_stake_menor_que_core(self):
        """
        T38-28: moonshot puede tener 2-3 Cat-C (diseño intencional) pero su
        stake es notablemente menor que el CORE — refleja el mayor riesgo.
        Ratio esperado: moonshot ≈ 5% budget vs CORE ≈ 45% budget.
        """
        picks = self._make_multi_cat_c_pool(n_c1=4, n_c2=2)
        plan = _build_portfolio_v2(picks, bankroll=125000, fase=3)

        assert plan['core'] is not None, "Se necesita CORE para comparar stakes"
        assert plan['moonshot'] is not None, "Se necesita moonshot para este test"

        # Moonshot stake debe ser significativamente menor que el CORE
        core_stake = plan['core']['stake']
        moon_stake = plan['moonshot']['stake']
        ratio = moon_stake / core_stake

        # Moonshot ≈ 5%/45% = ~11% del CORE stake (con margen por redondeo)
        assert ratio < 0.35, (
            f"Moonshot stake ({moon_stake}) debería ser mucho menor que "
            f"CORE stake ({core_stake}), ratio={ratio:.2%} > 35%"
        )

        # Moonshot sí puede tener 2+ Cat-C (diseño intencional)
        cat_c_in_moon = sum(
            1 for c in plan['moonshot']['categorias'] if c in ('CAT_C1', 'CAT_C2')
        )
        assert cat_c_in_moon >= 2, (
            f"Moonshot con pool amplio debería tener ≥2 Cat-C, tiene {cat_c_in_moon}"
        )

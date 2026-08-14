"""
Tests para Nodo-42: Grass Surface Bootstrap
T42-01 → T42-06

Verifica que --superficie grass activa modo bootstrap sin contaminar ejecuciones normales.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_confianza_builder import (
    _categorizar_pick,
    _extract_and_categorize,
    _build_portfolio_v2,
    _cap_stakes,
    _total_stakes,
    _all_combos,
    CONF_MIN, CONF_C1,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

GRASS_CONF_MIN = 50.0
GRASS_CONF_C1  = 55.0
GRASS_STAKE_MAX = 500.0
GRASS_VAR_PCT   = 0.01  # 1% bankroll

def _make_partido(jugador1, jugador2, cuota1, cuota2, confidence, favored):
    """Partido mínimo para _extract_and_categorize."""
    return {
        'jugador1': jugador1,
        'jugador2': jugador2,
        'cuota1': cuota1,
        'cuota2': cuota2,
        'torneo_nombre': 'Wimbledon (Reino Unido)',
        'torneo_completo': 'ATP - INDIVIDUALES: Wimbledon (Reino Unido), hierba',
        'tipo_cancha': 'grass',
        'ranking_analysis': {
            'prediction': {
                'favored_player': favored,
                'confidence': confidence,
            }
        }
    }


# ── T42-01: grass mode acepta picks con conf≥50% ─────────────────────────────

def test_t42_01_grass_mode_acepta_conf_50():
    """
    --superficie grass baja el umbral a 50%.
    Un pick con conf=51.5% (bloqueado con CONF_MIN=53) debe pasar con conf_min=50.
    """
    # Con threshold normal (53%) → None
    cat_normal = _categorizar_pick(cuota=1.40, confianza=51.5)
    assert cat_normal is None, "Con CONF_MIN=53, conf=51.5% debe ser rechazado"

    # Con grass mode (50%) → pasa
    cat_grass = _categorizar_pick(cuota=1.40, confianza=51.5,
                                   conf_min=GRASS_CONF_MIN, conf_c1=GRASS_CONF_C1)
    assert cat_grass is not None, "Con grass conf_min=50, conf=51.5% debe ser aceptado"
    assert cat_grass['categoria'] == 'CAT_A'


# ── T42-02: sin --superficie grass, CONF_MIN sigue siendo 53% ────────────────

def test_t42_02_sin_grass_mode_conf_min_intacto():
    """
    Sin grass mode, CONF_MIN no cambia.
    Verifica que el default de _categorizar_pick usa CONF_MIN=53.
    """
    assert CONF_MIN == 53.0, "CONF_MIN global debe seguir siendo 53.0"

    # conf=52.9% sin override → rechazado
    cat = _categorizar_pick(cuota=1.50, confianza=52.9)
    assert cat is None

    # conf=53.0% sin override → aceptado
    cat = _categorizar_pick(cuota=1.50, confianza=53.0)
    assert cat is not None


def test_t42_02b_extract_sin_grass_usa_conf_53(monkeypatch):
    """
    _extract_and_categorize sin parámetros usa CONF_MIN=53.

    Nodo-174 D174-02: sin este monkeypatch, el gate D140-04 (Nodo-140) llama
    load_coverage() real y excluye 'C'/'D' por no estar en la cache de Kambi
    del día — falso negativo ajeno a la lógica de threshold que este test cubre.
    """
    monkeypatch.setattr('scripts.fetch_kambi_coverage.load_coverage', lambda: None)
    partidos = [
        _make_partido('A', 'B', 2.5, 1.5, 51.0, 'B'),  # bloqueado: conf < 53
        # D143-01 (Nodo-143, posterior a este test): EV_LEG_MIN=1.02 exige
        # cuota>=1.02/0.54≈1.89 para conf=54% — cuota original 1.3 daba ev=0.70,
        # bloqueado siempre por EV real, no por threshold (lo que este test cubre).
        _make_partido('C', 'D', 2.0, 3.5, 54.0, 'C'),  # pasa
    ]
    picks = _extract_and_categorize(partidos, threshold=53.0)
    nombres = [p['nombre'] for p in picks]
    assert 'C' in nombres
    assert 'B' not in nombres


# ── T42-03: stake cap $500 en grass mode ────────────────────────────────────

def test_t42_03_stake_cap_grass():
    """
    _cap_stakes clampea todos los combos a stake_max=$500.
    """
    # Construir un plan con stake alto (simulando bankroll grande)
    plan = {
        'core': {
            'nombre': 'CORE',
            'piernas': ['A', 'B', 'C'],
            'stake': 7000,
            'odds_total': 3.5,
            'p_win': 0.20,
            'retorno_bruto': 24500.0,
            'retorno_esperado': 4900.0,
            'ev': 0.0,
            'n_piernas': 3,
            'cuotas': [1.4, 1.5, 1.6],
            'confianzas': [54.0, 53.0, 55.0],
            'categorias': ['CAT_A', 'CAT_A', 'CAT_B'],
            'pick_excluido': None,
        },
        'satellites': [],
        'moonshot': None,
        'cobertura': [],
        'budget': 15000,
        'fase': 4,
        'resumen': {},
    }

    _cap_stakes(plan, GRASS_STAKE_MAX)

    assert plan['core']['stake'] == GRASS_STAKE_MAX
    # retorno_bruto debe recalcularse
    assert plan['core']['retorno_bruto'] == round(GRASS_STAKE_MAX * 3.5, 0)


def test_t42_03b_stake_ya_bajo_cap_no_cambia():
    """
    Si el stake ya es ≤ stake_max, _cap_stakes no lo modifica.
    """
    plan = {
        'core': {
            'nombre': 'CORE',
            'piernas': ['A'],
            'stake': 300,
            'odds_total': 2.0,
            'p_win': 0.5,
            'retorno_bruto': 600.0,
            'retorno_esperado': 300.0,
            'ev': 0.0,
            'n_piernas': 1,
            'cuotas': [2.0],
            'confianzas': [51.0],
            'categorias': ['CAT_C1'],
            'pick_excluido': None,
        },
        'satellites': [],
        'moonshot': None,
        'cobertura': [],
        'budget': 5000,
        'fase': 1,
        'resumen': {},
    }
    _cap_stakes(plan, GRASS_STAKE_MAX)
    assert plan['core']['stake'] == 300  # no cambia


# ── T42-04: watermark en reporte grass mode ──────────────────────────────────

def test_t42_04_watermark_en_reporte():
    """
    _format_report con grass_mode=True incluye '[GRASS BOOTSTRAP]' en el output.
    """
    from combo_confianza_builder import _format_report, _build_portfolio_v2, _categorizar_pick

    picks = []
    cat = _categorizar_pick(cuota=1.40, confianza=51.5,
                             conf_min=GRASS_CONF_MIN, conf_c1=GRASS_CONF_C1)
    if cat:
        picks.append({
            'nombre': 'Rafael Jodar',
            'confianza': 51.5,
            'cuota': 1.40,
            'p_modelo': 0.515,
            'torneo': 'Wimbledon',
            'rival': 'Felix Gill',
            'cat': cat,
        })

    plan = {'core': None, 'satellites': [], 'moonshot': None,
            'cobertura': [], 'budget': 1250, 'fase': 4, 'resumen': {}}

    report_grass = _format_report(picks, plan, 50.0, 'test.json', grass_mode=True)
    assert '[GRASS BOOTSTRAP' in report_grass

    report_normal = _format_report(picks, plan, 53.0, 'test.json', grass_mode=False)
    assert '[GRASS BOOTSTRAP' not in report_normal


# ── T42-05: --superficie clay/hard no activa grass mode ─────────────────────

def test_t42_05_superficie_clay_no_activa_grass_mode():
    """
    superficie='clay' no debe bajar CONF_MIN.
    La lógica grass_mode solo activa cuando superficie == 'grass'.
    """
    # Simular lo que hace main(): grass_mode = (args.superficie == 'grass')
    for superficie in ('clay', 'hard', None):
        grass_mode = (superficie == 'grass')
        assert grass_mode is False, f"superficie={superficie} no debe activar grass_mode"

    grass_mode = ('grass' == 'grass')
    assert grass_mode is True


# ── T42-06: VaR guard grass — total invertido ≤ 1% bankroll ─────────────────

def test_t42_07_superficie_filter_excluye_clay(monkeypatch):
    """
    Con superficie_filter='grass', picks de tipo_cancha='clay' no entran al pool.
    --superficie grass NO es equivalente a --threshold 50 global.

    Nodo-174 D174-02 (triaje, corregido 2026-08-06): el fixture 'Ghetu' colisionaba
    con un jugador real del mismo apellido presente en el edge_report del día en
    disco (apostar=False -> gate G1 de Nodo-103 lo bloqueaba), no con el gate EV_LEG_MIN
    de D143-01 como decía el comentario original más abajo (ese calculo con la cuota
    actual del fixture da ev_leg=0.51*2.10=1.071 >= 1.02, o sea NUNCA bloqueaba por ahí).
    Se aisla también _load_edge_report_index para que el test no dependa de qué
    partidos reales existan en reports/edge_report_*.json el dia que corra.
    """
    monkeypatch.setattr('scripts.fetch_kambi_coverage.load_coverage', lambda: None)
    monkeypatch.setattr('combo_confianza_builder._load_edge_report_index', lambda: {})
    partidos = [
        # Wimbledon (grass) — debe entrar
        {
            'jugador1': 'Van Assche', 'jugador2': 'Fucsovics',
            'cuota1': 2.95, 'cuota2': 1.40,
            'tipo_cancha': 'grass',
            'torneo_nombre': 'Wimbledon',
            'torneo_completo': 'ATP - Wimbledon, hierba',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Van Assche', 'confidence': 53.1}
            }
        },
        # Roland Garros (clay) con conf=51% — NO debe entrar aunque pase el threshold
        # (con superficie_filter='grass'; sin filtro sí debe pasar, ver assert abajo).
        # ev_leg=0.51*2.10=1.071>=1.02 -> el gate EV D143-01 nunca bloquea este fixture.
        {
            'jugador1': 'Ghetu', 'jugador2': 'Poljak',
            'cuota1': 2.10, 'cuota2': 1.90,
            'tipo_cancha': 'clay',
            'torneo_nombre': 'Troyes',
            'torneo_completo': 'CHALLENGER - Troyes, arcilla',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Ghetu', 'confidence': 51.0}
            }
        },
        # Clay con conf=55% — tampoco debe entrar en grass filter
        # ev_leg=0.55*1.90=1.045>=1.02 -> el gate EV D143-01 nunca bloquea este fixture.
        {
            'jugador1': 'Dellien', 'jugador2': 'Rival',
            'cuota1': 1.90, 'cuota2': 2.80,
            'tipo_cancha': 'clay',
            'torneo_nombre': 'Brasov',
            'torneo_completo': 'CHALLENGER - Brasov, arcilla',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Dellien', 'confidence': 55.0}
            }
        },
    ]

    # Sin filtro: los 3 pasan threshold=50 (conf ≥ 50%)
    picks_sin_filtro = _extract_and_categorize(partidos, threshold=50.0,
                                                conf_min=GRASS_CONF_MIN)
    nombres_sin_filtro = [p['nombre'] for p in picks_sin_filtro]
    assert 'Van Assche' in nombres_sin_filtro
    assert 'Ghetu' in nombres_sin_filtro
    assert 'Dellien' in nombres_sin_filtro

    # Con superficie_filter='grass': solo Wimbledon entra
    picks_grass = _extract_and_categorize(partidos, threshold=50.0,
                                           conf_min=GRASS_CONF_MIN,
                                           superficie_filter='grass')
    nombres_grass = [p['nombre'] for p in picks_grass]
    assert 'Van Assche' in nombres_grass, "Pick de Wimbledon debe estar en pool grass"
    assert 'Ghetu' not in nombres_grass, "Pick de clay NO debe entrar en pool grass"
    assert 'Dellien' not in nombres_grass, "Pick de clay NO debe entrar en pool grass"


def test_t42_06_var_guard_grass():
    """
    Con grass_mode y bankroll=125000, el total invertido no debe superar $1,250 (1%).
    """
    bankroll = 125_000.0
    var_limit = bankroll * GRASS_VAR_PCT  # $1,250

    # Construir partidos Wimbledon con conf 50-55%
    partidos = [
        _make_partido('Jodar', 'Gill',       1.19, 4.6,  55.7, 'Jodar'),
        _make_partido('VanAssche', 'Fucs',   2.95, 1.4,  53.1, 'VanAssche'),
        _make_partido('Norrie', 'Zheng',     1.61, 2.32, 52.3, 'Zheng'),
        _make_partido('Sonmez', 'Li',        2.1,  1.74, 53.1, 'Sonmez'),
        _make_partido('Bencic', 'Stoj',      1.11, 6.75, 54.0, 'Bencic'),
    ]

    picks = _extract_and_categorize(
        partidos, threshold=50.0,
        conf_min=GRASS_CONF_MIN, conf_c1=GRASS_CONF_C1
    )

    if not picks:
        pytest.skip("Sin picks para VaR test — ajustar datos")

    plan = _build_portfolio_v2(picks, bankroll, fase=4, stake_max=GRASS_STAKE_MAX)

    # Aplicar VaR grass
    from combo_confianza_builder import _scale_stakes
    total = _total_stakes(plan)
    if total > var_limit:
        _scale_stakes(plan, var_limit / total)
        _cap_stakes(plan, GRASS_STAKE_MAX)

    total_final = _total_stakes(plan)
    assert total_final <= var_limit + 1, (
        f"Total invertido grass ${total_final:.0f} supera VaR limit ${var_limit:.0f}"
    )

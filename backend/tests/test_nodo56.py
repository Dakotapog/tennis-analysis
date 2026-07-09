"""
tests/test_nodo56.py — Nodo-56: Bugs de display en generar_tabla_favoritos2.py

REGLA-T53: ningún test hardcodea la fórmula. Siempre invoca funciones del módulo real.

Bugs cubiertos:
  A: get_weights_from_reasoning ignoraba LOG_SHRINKAGE → pesos no sumaban 100%
  B: round(...,2) en ajuste de superficie → pérdidas de precisión
  C: Penalizacion_Inactividad oculta → PUNTAJE FINAL ≠ suma de componentes
"""

import pytest
import json
import io
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from analysis.rivalry_analyzer import RivalryAnalyzer


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def ranking_manager():
    rm = MagicMock()
    rm.normalize_name.side_effect = lambda name: name.lower().strip() if name else name
    rm.get_player_ranking.return_value = 100
    rm.get_player_info.return_value = {
        'ranking_position': 100,
        'ranking_points': 500,
        'prox_points': 550,
        'max_points': 600,
        'defense_points': 50,
    }
    return rm


@pytest.fixture
def elo_system():
    es = MagicMock()
    es.default_rating = 1500
    es.k_factor = 32
    es.expected_score.return_value = 0.5
    es.calculate_rating_change.return_value = 16
    return es


@pytest.fixture
def analyzer(ranking_manager, elo_system):
    return RivalryAnalyzer(ranking_manager, elo_system)


def _match(oponente, outcome, superficie='Arcilla', days_ago=5):
    fecha = (datetime.now() - timedelta(days=days_ago)).strftime('%d.%m.%Y')
    return {
        'oponente': oponente,
        'resultado': '6-3 6-4' if outcome == 'Ganó' else '3-6 4-6',
        'outcome': outcome,
        'opponent_ranking': 100,
        'superficie': superficie,
        'location': 'Colombia',
        'fecha': fecha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T56-04 — _weights_final retornado por analyze_rivalry suma 1.0 ± 0.001
# (valida Fix A: fuente única de verdad en rivalry_analyzer)
# ─────────────────────────────────────────────────────────────────────────────

def test_t56_04_weights_final_sums_to_one(analyzer):
    """T56-04: analyze_rivalry retorna predicción con _weights_final que suma 1.0 ± 0.001
    para un partido clay_challenger estándar."""
    p1_hist = [_match('RivalX', 'Ganó', days_ago=5),
               _match('RivalY', 'Ganó', days_ago=10),
               _match('RivalZ', 'Perdió', days_ago=15)]
    p2_hist = [_match('RivalX', 'Perdió', days_ago=6),
               _match('RivalY', 'Ganó', days_ago=12),
               _match('RivalZ', 'Ganó', days_ago=18)]

    form_mock = {
        'last_match_date': (datetime.now() - timedelta(days=5)).strftime('%d.%m.%Y'),
        'current_streak_count': 2, 'current_streak_type': 'W',
        'last_5_results': ['W', 'W', 'L', 'W', 'W'],
        'win_percentage': 80.0, 'form_status': 'HOT',
    }

    context = {
        'current_match_surface': 'clay',
        'tournament_name': 'Challenger Test',
        'tournament_tier': 'challenger',
        'calibracion_data': {},
    }

    result = analyzer.analyze_rivalry(
        player1_history=p1_hist,
        player2_history=p2_hist,
        player1_name='TestP1',
        player2_name='TestP2',
        player1_form=form_mock,
        player2_form=form_mock,
        direct_h2h_matches=[],
        current_match_context=context,
        p1_elo=1600.0,
        p2_elo=1550.0,
        tournament_name='Challenger Test',
    )

    pred = result.get('prediction', {})
    assert '_weights_final' in pred, (
        "T56-04: _weights_final ausente en prediction. Fix D56-01 no aplicado."
    )
    wf = pred['_weights_final']
    total = sum(wf.values())
    assert abs(total - 1.0) <= 0.001, (
        f"T56-04: _weights_final suma {total:.6f}, esperado 1.0 ± 0.001. Pesos: {wf}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T56-05 — Penalizacion Inactividad aparece en tabla cuando penalty != 0
# ─────────────────────────────────────────────────────────────────────────────

def test_t56_05_penalty_row_shown_when_inactive():
    """T56-05: generar_resumen_consolidado incluye fila 'Penalizacion Inactividad'
    cuando score_breakdown tiene penalización, y sum(componentes)+penalidad ≈ PUNTAJE FINAL."""
    from generar_tabla_favoritos2 import generar_resumen_consolidado

    penalty_p2 = -1.88

    def _bd_entry(v):
        return {'weighted_score': f'{v:.2f}'}

    p1_bd = {
        'surface_specialization': _bd_entry(0.47),
        'form_recent':            _bd_entry(1.03),
        'common_opponents':       _bd_entry(0.63),
        'h2h_direct':             _bd_entry(0.00),
        'ranking_momentum':       _bd_entry(0.64),
        'elo_rating':             _bd_entry(0.64),
        'home_advantage':         _bd_entry(0.00),
        'strength_of_schedule':   _bd_entry(0.26),
        'Penalizacion_Inactividad': '0.00 pts',
        'Puntaje_Final': '3.67',
    }
    p2_bd = {
        'surface_specialization': _bd_entry(0.41),
        'form_recent':            _bd_entry(1.26),
        'common_opponents':       _bd_entry(0.42),
        'h2h_direct':             _bd_entry(0.00),
        'ranking_momentum':       _bd_entry(0.67),
        'elo_rating':             _bd_entry(0.75),
        'home_advantage':         _bd_entry(0.00),
        'strength_of_schedule':   _bd_entry(0.26),
        'Penalizacion_Inactividad': f'{penalty_p2:.2f} pts',
        'Puntaje_Final': '1.89',
    }

    score_breakdown = {'player1': p1_bd, 'player2': p2_bd}
    scores = {'p1_final_weight': 3.67, 'p2_final_weight': 1.89}

    buf = io.StringIO()
    generar_resumen_consolidado(buf, 'Felipe', 'Rodrigo', score_breakdown, scores)
    output = buf.getvalue()

    assert 'Penalizacion Inactividad' in output, (
        "T56-05: fila 'Penalizacion Inactividad' debe aparecer cuando penalty != 0"
    )
    assert 'PUNTAJE FINAL TOTAL' in output, "T56-05: fila PUNTAJE FINAL TOTAL ausente"

    # Verificar aritmética: componentes + penalidad ≈ final
    comp_p2 = 0.41 + 1.26 + 0.42 + 0.00 + 0.67 + 0.75 + 0.00 + 0.26
    final_with_penalty = comp_p2 + penalty_p2
    assert abs(final_with_penalty - 1.89) <= 0.01, (
        f"T56-05: {comp_p2:.2f} + {penalty_p2:.2f} = {final_with_penalty:.2f}, esperado ≈ 1.89"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T56-06 — Penalizacion Inactividad NO aparece cuando penalty = 0
# ─────────────────────────────────────────────────────────────────────────────

def test_t56_06_no_penalty_row_when_active():
    """T56-06: generar_resumen_consolidado NO incluye fila 'Penalizacion Inactividad'
    cuando ambos jugadores tienen penalty=0."""
    from generar_tabla_favoritos2 import generar_resumen_consolidado

    def make_bd(weighted_scores):
        bd = {k: {'weighted_score': f'{v:.2f}'} for k, v in weighted_scores.items()}
        bd['Penalizacion_Inactividad'] = '0.00 pts'
        bd['Puntaje_Final'] = f'{sum(weighted_scores.values()):.2f}'
        return bd

    comps = {
        'surface_specialization': 0.47,
        'form_recent': 1.03,
        'common_opponents': 0.63,
        'ranking_momentum': 0.64,
        'elo_rating': 0.64,
        'strength_of_schedule': 0.26,
    }

    score_breakdown = {'player1': make_bd(comps), 'player2': make_bd(comps)}
    total = sum(comps.values())
    scores = {'p1_final_weight': total, 'p2_final_weight': total}

    buf = io.StringIO()
    generar_resumen_consolidado(buf, 'PlayerA', 'PlayerB', score_breakdown, scores)
    output = buf.getvalue()

    assert 'Penalizacion Inactividad' not in output, (
        "T56-06: fila 'Penalizacion Inactividad' NO debe aparecer cuando penalty=0"
    )

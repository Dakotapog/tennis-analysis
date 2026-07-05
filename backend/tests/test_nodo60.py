"""
Tests Nodo-60: GCS — Grass/Surface Champion Signal

T60-01: GCS_RECENCY_BOOST aplica cuando tier>=ATP500 y days<=14
T60-02: GCS_RECENCY_BOOST NO aplica cuando tier=ITF
T60-03: GCS_RECENCY_BOOST NO aplica cuando days>21
T60-04: _extract_and_categorize marca gcs_active=True cuando torneo_completo=True + tier>=atp500
T60-05: H60-01 existe en preregistered_hypotheses.json con n_stop=30
"""
import json
import math
import os
import sys
import datetime

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_surface_matches(n_wins=4, n_total=6, surface='Hierba'):
    """Genera historial mínimo de partidos en superficie para test."""
    matches = []
    for i in range(n_total):
        ganador = 'TestPlayer' if i < n_wins else 'Opponent'
        matches.append({
            'jugador1': 'TestPlayer',
            'jugador2': 'Opponent',
            'resultado': '2-0',
            'ganador': ganador,
            'superficie': surface,
            'fecha': (datetime.datetime.today() - datetime.timedelta(days=30 + i * 7)).strftime('%d.%m.%Y'),
            'torneo': 'TestTournament',
            'oponente': 'Opponent',
        })
    return matches


def _make_complete_tournament_matches(torneo_name, days_ago, n_wins=5, surface='Hierba',
                                       best_opp_rank=8):
    """Genera partidos de un torneo completo para activar TORNEO_COMPLETO_BONUS.
    days_ago = días desde hoy hasta la FINAL (partido más reciente).
    Los partidos anteriores se espacian hacia atrás desde ahí.
    """
    matches = []
    final_date = datetime.datetime.today() - datetime.timedelta(days=days_ago)
    for i in range(n_wins):
        # i=n_wins-1 es la final (más reciente = days_ago días atrás)
        fecha = (final_date - datetime.timedelta(days=(n_wins - 1 - i))).strftime('%d.%m.%Y')
        matches.append({
            'jugador1': 'TestPlayer',
            'jugador2': f'Opp{i}',
            'resultado': '2-0',
            'ganador': 'TestPlayer',
            'superficie': surface,
            'fecha': fecha,
            'torneo': torneo_name,
            'oponente': f'Opp{i}',
            # best opponent in first match
            'opponent_ranking': best_opp_rank if i == 0 else 50,
        })
    return matches


def _get_analyzer():
    from analysis.rivalry_analyzer import RivalryAnalyzer
    rm = MagicMock()
    rm.get_player_ranking.return_value = None
    rm.get_player_info.return_value = {'ranking_position': 50, 'ranking_points': 500,
                                       'prox_points': 500, 'max_points': 500, 'defense_points': 0}
    rm.normalize_name.side_effect = lambda n: n.lower() if n else n
    es = MagicMock()
    es.default_rating = 1500
    es.k_factor = 32
    es.expected_score.return_value = 0.5
    es.calculate_rating_change.return_value = 16
    return RivalryAnalyzer(rm, es)


# ── T60-01: GCS_RECENCY_BOOST aplica con tier>=ATP500 y days<=14 ─────────────

def test_T60_01_gcs_boost_fires_atp500_recent():
    """GCS_RECENCY_BOOST activa (×1.8) cuando tier=atp500 y days=13."""
    analyzer = _get_analyzer()

    # Historial con torneo completo ATP500 hace 13 días en hierba
    history_with_bonus = _make_complete_tournament_matches(
        torneo_name='Nottingham 2026', days_ago=13, n_wins=5, surface='Hierba'
    )
    # Añadir partidos dispersos para tener volumen
    history_base = _make_surface_matches(n_wins=20, n_total=30, surface='Hierba')
    full_history = history_with_bonus + history_base

    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Hierba', 'TestPlayer'
    )

    # El score con GCS debe ser mayor que sin (ya que x1.8 aplica)
    assert result['gcs_active'] is True, "gcs_active debe ser True con torneo ATP500 en 13 días"
    assert result['gcs_days'] == 13, f"gcs_days debe ser 13, got {result['gcs_days']}"
    # Log debe contener GCS_RECENCY_BOOST
    gcs_log = [l for l in log if 'GCS_RECENCY_BOOST' in l]
    assert len(gcs_log) == 1, f"Debe haber exactamente 1 línea GCS_RECENCY_BOOST, got {gcs_log}"
    assert '×1.8' in gcs_log[0], f"Multiplicador debe ser ×1.8 para days=13, got: {gcs_log[0]}"


# ── T60-02: GCS_RECENCY_BOOST NO aplica con tier=ITF ─────────────────────────

def test_T60_02_gcs_boost_no_fires_itf():
    """GCS_RECENCY_BOOST NO activa cuando tier=ITF aunque el torneo sea reciente."""
    analyzer = _get_analyzer()

    # Torneo completo ITF hace 7 días
    history_with_bonus = _make_complete_tournament_matches(
        torneo_name='M25 Figueira da Foz 2026', days_ago=7, n_wins=5, surface='Arcilla'
    )
    history_base = _make_surface_matches(n_wins=15, n_total=25, surface='Arcilla')
    full_history = history_with_bonus + history_base

    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Arcilla', 'TestPlayer'
    )

    assert result['gcs_active'] is False, "gcs_active debe ser False para tier=ITF"
    gcs_log = [l for l in log if 'GCS_RECENCY_BOOST' in l]
    assert len(gcs_log) == 0, f"No debe haber GCS_RECENCY_BOOST para ITF, got: {gcs_log}"


# ── T60-03: GCS_RECENCY_BOOST NO aplica cuando days>21 ───────────────────────

def test_T60_03_gcs_boost_no_fires_old_tournament():
    """GCS_RECENCY_BOOST NO activa cuando el torneo fue hace >21 días."""
    analyzer = _get_analyzer()

    # Torneo ATP500 hace 25 días (fuera de ventana de 21d)
    history_with_bonus = _make_complete_tournament_matches(
        torneo_name='Birmingham 2026', days_ago=25, n_wins=5, surface='Hierba'
    )
    history_base = _make_surface_matches(n_wins=20, n_total=35, surface='Hierba')
    full_history = history_with_bonus + history_base

    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Hierba', 'TestPlayer'
    )

    assert result['gcs_active'] is False, "gcs_active debe ser False cuando torneo tiene >21 días"
    gcs_log = [l for l in log if 'GCS_RECENCY_BOOST' in l]
    assert len(gcs_log) == 0, f"No debe haber GCS_RECENCY_BOOST para torneo con 25 días"


# ── T60-04: _extract_and_categorize marca gcs_active=True ────────────────────

def test_T60_04_extract_marks_gcs_active():
    """_extract_and_categorize devuelve gcs_active=True cuando surface_specialization_meta tiene torneo_completo=True + tier>=atp500."""
    from combo_confianza_builder import _extract_and_categorize

    # Partido simulado con gcs_active=True en surface_specialization_meta
    partido_gcs = {
        'jugador1': 'Alexandra Eala',
        'jugador2': 'Iga Swiatek',
        'torneo_nombre': 'WTA - INDIVIDUALES: Wimbledon (Reino Unido), hierba',
        'tipo_cancha': 'hierba',
        'cuota1': 3.80,
        'cuota2': 1.27,
        'cuota_es_real': True,
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Alexandra Eala',
                'confidence': 50.2,
                'reasoning': ['P1_LOG_SURF: TORNEO_COMPLETO_BONUS: Birmingham 2026 (5W-0L, tier=atp500)'],
                'surface_specialization_meta': {
                    'player1': {'score': 85.0, 'torneo_completo': True, 'gcs_active': True, 'gcs_days': 13},
                    'player2': {'score': 55.9, 'torneo_completo': False, 'gcs_active': False},
                },
            }
        }
    }

    # Partido sin GCS
    partido_normal = {
        'jugador1': 'Kalin Ivanovski',
        'jugador2': 'Yanaki Milev',
        'torneo_nombre': 'ITF MASCULINO: M25 Skopje',
        'tipo_cancha': 'arcilla',
        'cuota1': 1.30,
        'cuota2': 3.50,
        'cuota_es_real': True,
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Kalin Ivanovski',
                'confidence': 57.7,
                'reasoning': [],
                'surface_specialization_meta': {
                    'player1': {'score': 40.0, 'torneo_completo': False, 'gcs_active': False},
                    'player2': {'score': 25.0, 'torneo_completo': False, 'gcs_active': False},
                },
            }
        }
    }

    picks = _extract_and_categorize(
        [partido_gcs, partido_normal],
        threshold=50.0,
        pipeline_picks=None,
        conf_min=50.0,
    )

    assert len(picks) == 2, f"Deben extraerse 2 picks, got {len(picks)}"

    eala_pick = next((p for p in picks if p['nombre'] == 'Alexandra Eala'), None)
    ivan_pick = next((p for p in picks if p['nombre'] == 'Kalin Ivanovski'), None)

    assert eala_pick is not None, "Pick de Eala no encontrado"
    assert eala_pick['gcs_active'] is True, f"Eala debe tener gcs_active=True, got {eala_pick.get('gcs_active')}"
    assert eala_pick['universo'] == 'GCS', f"Eala debe tener universo=GCS, got {eala_pick.get('universo')}"

    assert ivan_pick is not None, "Pick de Ivanovski no encontrado"
    assert ivan_pick['gcs_active'] is False, f"Ivanovski NO debe tener gcs_active, got {ivan_pick.get('gcs_active')}"
    assert ivan_pick['universo'] == 'ITF', f"Ivanovski debe tener universo=ITF, got {ivan_pick.get('universo')}"


# ── T60-05: H60-01 existe en preregistered_hypotheses.json ───────────────────

def test_T60_05_h60_01_exists_in_hypotheses():
    """H60-01 existe en preregistered_hypotheses.json con n_stop=30 y estado=ACUMULANDO."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hyp_path = os.path.join(base_dir, 'validation', 'preregistered_hypotheses.json')

    assert os.path.exists(hyp_path), "preregistered_hypotheses.json no encontrado"

    with open(hyp_path) as f:
        data = json.load(f)

    hypotheses = data.get('hypotheses', {})
    assert 'H60-01' in hypotheses, "H60-01 no encontrado en hypotheses"

    h60 = hypotheses['H60-01']
    assert h60.get('n_stop') == 30, f"n_stop debe ser 30, got {h60.get('n_stop')}"
    assert h60.get('estado') == 'ACUMULANDO', f"estado debe ser ACUMULANDO, got {h60.get('estado')}"

    umbrales = h60.get('umbrales_congelados', {})
    assert umbrales.get('tier_min') == 'atp500', "tier_min debe ser atp500"
    assert umbrales.get('dias_max') == 21, "dias_max debe ser 21"
    assert umbrales.get('gcs_mult_days_14') == 1.8, "gcs_mult_days_14 debe ser 1.8"

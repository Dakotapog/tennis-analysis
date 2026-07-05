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


# ── T60-06: GCS gate en edge_calculator aplica cuando edge>=15% + gcs_active ─────

def test_T60_06_gcs_gate_fires_edge_calculator():
    """D60-06: GCS gate en edge_calculator aplica apostar=True cuando edge>=15% + gcs_active + kelly>2%."""
    from edge_calculator import calcular_edge_completo

    # Simular Eala vs Swiatek: edge=23.9%, confidence=50.2%, gcs_active=True
    # Estructura de score_breakdown compatible con phi_idiosincratico
    _sb = {
        'player1': {
            'surface_specialization': {'contribution': '15%'},
            'form_recent': {'contribution': '20%'},
            'common_opponents': {'contribution': '10%'},
            'h2h_direct': {'contribution': '0%'},
            'ranking_momentum': {'contribution': '18%'},
            'elo_rating': {'contribution': '12%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '25%'},
        },
        'player2': {
            'surface_specialization': {'contribution': '10%'},
            'form_recent': {'contribution': '25%'},
            'common_opponents': {'contribution': '8%'},
            'h2h_direct': {'contribution': '20%'},
            'ranking_momentum': {'contribution': '15%'},
            'elo_rating': {'contribution': '12%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '10%'},
        },
    }

    partido = {
        'jugador1': 'Alexandra Eala',
        'jugador2': 'Iga Swiatek',
        'cuota1': 3.80,
        'cuota2': 1.27,
        'cuota_es_real': True,
        'torneo_nombre': 'WTA - INDIVIDUALES: Wimbledon (Reino Unido), hierba',
        'tipo_cancha': 'hierba',
        'superficie': 'Hierba',
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Alexandra Eala',
                'confidence': 50.2,  # p_modelo=0.502 (LOW, pero GCS rescata)
                'reasoning': ['GCS_RECENCY_BOOST: Birmingham 2026 hace 13 días'],
                'score_breakdown': _sb,
                'markov_analysis': {
                    'jugador1': {'estado_actual': 'NEUTRAL', 'win_rate_reciente': 0.5, 'win_rate_anterior': 0.5, 'confianza': 0.50},
                    'jugador2': {'estado_actual': 'NEUTRAL', 'win_rate_reciente': 0.5, 'win_rate_anterior': 0.5, 'confianza': 0.50},
                },
                'surface_specialization_meta': {
                    'player1': {
                        'score': 85.0, 'raw_score': 47.2, 'win_rate': 0.72, 'matches': 18,
                        'skill_factor': 1.8, 'alpha_bonus': 1.3, 'volume_confidence': 0.9,
                        'surface_alpha': 0.22, 'torneo_completo': True,
                        'gcs_active': True,  # ← KEY: GCS activated
                        'gcs_days': 13,
                    },
                    'player2': {
                        'score': 55.9, 'raw_score': 40.1, 'win_rate': 0.65, 'matches': 20,
                        'skill_factor': 1.5, 'alpha_bonus': 1.0, 'volume_confidence': 0.95,
                        'surface_alpha': 0.18, 'torneo_completo': False,
                        'gcs_active': False,
                        'gcs_days': None,
                    },
                },
            },
            'ranking_fav': {'ranking_position': 75, 'elo': 1787},
            'ranking_rival': {'ranking_position': 22, 'elo': 1915},
        },
        'enfrentamientos_directos': [],  # sin H2H directo
    }

    calibracion = {
        'global': {'wins': 467, 'losses': 239},
        'por_superficie': {'grass': {'wins': 150, 'losses': 60}},
        'por_superficie_y_tier': {'grass_grand_slam': {'wins': 31, 'losses': 10}},
    }

    resultado = calcular_edge_completo(partido, calibracion)

    assert resultado is not None, "Debe retornar un resultado para Eala"
    assert resultado['gcs_bonus'] is True, f"gcs_bonus debe ser True, got {resultado.get('gcs_bonus')}"
    assert resultado['gcs_gate_applied'] is True, f"gcs_gate_applied debe ser True con edge=23.9%, got {resultado.get('gcs_gate_applied')}"
    assert resultado['apostar'] is True, f"apostar debe ser True con GCS gate, got {resultado.get('apostar')}"
    assert resultado['edge'] >= 0.15, f"edge debe ser >=15%, got {resultado['edge']}"


# ── T60-07: GCS gate NO aplica cuando edge<15% ────────────────────────────────

def test_T60_07_gcs_gate_no_fires_edge_low():
    """D60-06: GCS gate NO aplica cuando edge<15% aunque gcs_active=True."""
    from edge_calculator import calcular_edge_completo

    _sb = {
        'player1': {
            'surface_specialization': {'contribution': '12%'},
            'form_recent': {'contribution': '18%'},
            'common_opponents': {'contribution': '8%'},
            'h2h_direct': {'contribution': '3%'},
            'ranking_momentum': {'contribution': '16%'},
            'elo_rating': {'contribution': '12%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '31%'},
        },
        'player2': {
            'surface_specialization': {'contribution': '10%'},
            'form_recent': {'contribution': '20%'},
            'common_opponents': {'contribution': '8%'},
            'h2h_direct': {'contribution': '10%'},
            'ranking_momentum': {'contribution': '15%'},
            'elo_rating': {'contribution': '12%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '25%'},
        },
    }

    # Simular partido con GCS activo pero edge bajo
    partido = {
        'jugador1': 'Test Player 1',
        'jugador2': 'Test Player 2',
        'cuota1': 1.95,  # edge ~10% (p_modelo=0.56, p_implicita=0.513)
        'cuota2': 1.88,
        'cuota_es_real': True,
        'torneo_nombre': 'ATP - INDIVIDUALES: London (Inglaterra), hierba',
        'tipo_cancha': 'hierba',
        'superficie': 'Hierba',
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Test Player 1',
                'confidence': 56.0,  # p_modelo=0.56 (MODERATE)
                'reasoning': [],
                'score_breakdown': _sb,
                'markov_analysis': {
                    'player1': {'estado_actual': 'NEUTRAL', 'win_rate_reciente': 0.5, 'win_rate_anterior': 0.5, 'confianza': 0.50},
                    'player2': {'estado_actual': 'NEUTRAL', 'win_rate_reciente': 0.5, 'win_rate_anterior': 0.5, 'confianza': 0.50},
                },
                'surface_specialization_meta': {
                    'player1': {
                        'score': 70.0, 'raw_score': 40.0, 'win_rate': 0.68, 'matches': 15,
                        'skill_factor': 1.5, 'alpha_bonus': 1.2, 'volume_confidence': 0.85,
                        'surface_alpha': 0.12, 'torneo_completo': True,
                        'gcs_active': True,  # GCS está activo
                        'gcs_days': 10,
                    },
                    'player2': {
                        'score': 50.0, 'raw_score': 35.0, 'win_rate': 0.60, 'matches': 18,
                        'skill_factor': 1.3, 'alpha_bonus': 1.0, 'volume_confidence': 0.80,
                        'surface_alpha': 0.10, 'torneo_completo': False,
                        'gcs_active': False,
                        'gcs_days': None,
                    },
                },
            },
            'ranking_fav': {'ranking_position': 50, 'elo': 1800},
            'ranking_rival': {'ranking_position': 80, 'elo': 1700},
        },
        'enfrentamientos_directos': [],
    }

    calibracion = {
        'global': {'wins': 467, 'losses': 239},
        'por_superficie': {'grass': {'wins': 150, 'losses': 60}},
        'por_superficie_y_tier': {'grass_atp500': {'wins': 25, 'losses': 8}},
    }

    resultado = calcular_edge_completo(partido, calibracion)

    assert resultado is not None, "Debe retornar un resultado"
    assert resultado['gcs_bonus'] is True, f"gcs_bonus debe ser True, got {resultado.get('gcs_bonus')}"
    assert resultado['edge'] < 0.15, f"edge debe ser <15%, got {resultado['edge']}"
    # GCS gate NO aplica porque edge<15%, aunque gcs_active=True
    assert resultado['gcs_gate_applied'] is False, f"gcs_gate_applied debe ser False con edge<15%, got {resultado.get('gcs_gate_applied')}"


# ── T60-08: GCS score_boost clamped a ×1.15 máximo ──────────────────────────────

def test_T60_08_gcs_score_boost_clamped():
    """D60-06: gcs_score_boost se clampea a ×1.15 máximo."""
    from edge_calculator import calcular_edge_completo

    _sb = {
        'player1': {
            'surface_specialization': {'contribution': '18%'},
            'form_recent': {'contribution': '25%'},
            'common_opponents': {'contribution': '10%'},
            'h2h_direct': {'contribution': '3%'},
            'ranking_momentum': {'contribution': '16%'},
            'elo_rating': {'contribution': '10%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '18%'},
        },
        'player2': {
            'surface_specialization': {'contribution': '5%'},
            'form_recent': {'contribution': '15%'},
            'common_opponents': {'contribution': '5%'},
            'h2h_direct': {'contribution': '8%'},
            'ranking_momentum': {'contribution': '12%'},
            'elo_rating': {'contribution': '8%'},
            'home_advantage': {'contribution': '0%'},
            'strength_of_schedule': {'contribution': '47%'},
        },
    }

    # Partido con edge muy alto (~40%) → gcs_score_boost sería ~1.30 sin clamp
    partido = {
        'jugador1': 'Test Player 1',
        'jugador2': 'Test Player 2',
        'cuota1': 5.50,  # edge~35% (p_modelo=0.56, p_implicita=0.182)
        'cuota2': 1.10,
        'cuota_es_real': True,
        'torneo_nombre': 'ATP - INDIVIDUALES: London (Inglaterra), hierba',
        'tipo_cancha': 'hierba',
        'superficie': 'Hierba',
        'ranking_analysis': {
            'prediction': {
                'favored_player': 'Test Player 1',
                'confidence': 56.0,
                'reasoning': [],
                'score_breakdown': _sb,
                'markov_analysis': {
                    'player1': {'estado_actual': 'HOT', 'win_rate_reciente': 0.75, 'win_rate_anterior': 0.50, 'confianza': 0.70},
                    'player2': {'estado_actual': 'COLD', 'win_rate_reciente': 0.25, 'win_rate_anterior': 0.50, 'confianza': 0.70},
                },
                'surface_specialization_meta': {
                    'player1': {
                        'score': 80.0, 'raw_score': 45.0, 'win_rate': 0.75, 'matches': 20,
                        'skill_factor': 1.8, 'alpha_bonus': 1.3, 'volume_confidence': 0.90,
                        'surface_alpha': 0.25, 'torneo_completo': True,
                        'gcs_active': True,
                        'gcs_days': 7,
                    },
                    'player2': {
                        'score': 30.0, 'raw_score': 20.0, 'win_rate': 0.40, 'matches': 10,
                        'skill_factor': 1.0, 'alpha_bonus': 0.9, 'volume_confidence': 0.60,
                        'surface_alpha': 0.05, 'torneo_completo': False,
                        'gcs_active': False,
                        'gcs_days': None,
                    },
                },
            },
            'ranking_fav': {'ranking_position': 5, 'elo': 2000},
            'ranking_rival': {'ranking_position': 200, 'elo': 1400},
        },
        'enfrentamientos_directos': [],
    }

    calibracion = {
        'global': {'wins': 467, 'losses': 239},
        'por_superficie': {'grass': {'wins': 150, 'losses': 60}},
        'por_superficie_y_tier': {'grass_atp500': {'wins': 25, 'losses': 8}},
    }

    resultado = calcular_edge_completo(partido, calibracion)

    assert resultado is not None, "Debe retornar un resultado"
    assert resultado['gcs_bonus'] is True, f"gcs_bonus debe ser True, got {resultado.get('gcs_bonus')}"
    # gcs_score_boost no puede exceder 1.15 aunque edge sea alto
    assert resultado['gcs_score_boost'] <= 1.15, f"gcs_score_boost debe estar clamped <=1.15, got {resultado['gcs_score_boost']}"
    assert resultado['gcs_gate_applied'] is True, f"gcs_gate_applied debe ser True con edge alto + gcs_active, got {resultado.get('gcs_gate_applied')}"

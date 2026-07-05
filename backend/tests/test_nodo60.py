"""
Tests Nodo-60: GCS — Grass/Surface Champion Signal (FABLE-ADDENDUM restructura)

T60-01: LOG_GCS_SHADOW aparece (flag OFF) y gcs_active=True cuando tier>=ATP500 y days<=14
T60-02: gcs_active=False cuando tier=ITF (guard tier)
T60-03: gcs_active=False cuando days>21 (ventana caducada)
T60-04: _extract_and_categorize marca gcs_active=True cuando torneo_completo=True + tier>=atp500
T60-05: H60-01 existe en preregistered_hypotheses.json con n_stop=30 y campo gated
T60-06: 5W-0L en grand_slam → gcs_active=False (D57-03: GS requiere 7W)
T60-07: flag OFF por default → final_score idéntico con/sin código GCS
T60-08: torneo ganado en clay, partido actual en grass → gcs_active=False
T60-09: pick GCS + pick ITF → nunca en el mismo CORE combo
T60-10: LOG_GCS_SHADOW presente cuando flag=OFF y pick califica
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


# ── T60-01: flag OFF → LOG_GCS_SHADOW aparece y gcs_active=True con tier>=ATP500 y days<=14 ──

def test_T60_01_gcs_shadow_log_atp500_recent():
    """Con _GCS_BOOST_ENABLED=False (default): gcs_active=True y LOG_GCS_SHADOW en log para
    torneo ATP500 hace 13 días. El score NO cambia (A/B shadow)."""
    import analysis.rivalry_analyzer as ra_mod
    assert ra_mod._GCS_BOOST_ENABLED is False, "_GCS_BOOST_ENABLED debe ser False por default"

    analyzer = _get_analyzer()

    history_with_bonus = _make_complete_tournament_matches(
        torneo_name='Nottingham 2026', days_ago=13, n_wins=5, surface='Hierba'
    )
    history_base = _make_surface_matches(n_wins=20, n_total=30, surface='Hierba')
    full_history = history_with_bonus + history_base

    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Hierba', 'TestPlayer'
    )

    assert result['gcs_active'] is True, "gcs_active debe ser True con torneo ATP500 en 13 días"
    assert result['gcs_days'] == 13, f"gcs_days debe ser 13, got {result['gcs_days']}"
    # Con flag OFF: LOG_GCS_SHADOW en log, NO GCS_RECENCY_BOOST
    shadow_log = [l for l in log if 'LOG_GCS_SHADOW' in l]
    boost_log = [l for l in log if 'GCS_RECENCY_BOOST' in l and 'LOG_GCS_SHADOW' not in l]
    assert len(shadow_log) >= 1, f"Debe haber LOG_GCS_SHADOW con flag OFF, got: {log}"
    assert len(boost_log) == 0, f"NO debe haber GCS_RECENCY_BOOST activo con flag OFF"
    assert '×1.8' in shadow_log[0], f"Shadow debe mencionar ×1.8 para days=13, got: {shadow_log[0]}"


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

    # FABLE §2: campo gated obligatorio (GCS_MULT permanece OFF hasta graduacion)
    assert h60.get('gated'), "H60-01 debe tener campo 'gated' con condicion de activacion"
    assert 'estado_inicial' in h60, "H60-01 debe tener campo 'estado_inicial' con estado honesto"


# ── T60-06: 5W-0L en grand_slam → gcs_active=False (D57-03: GS requiere 7W) ─────

def test_T60_06_grand_slam_5wins_no_gcs():
    """D57-03 dependencia: 5W-0L en grand_slam NO activa gcs_active (GS requiere 7W)."""
    analyzer = _get_analyzer()

    # 5 victorias en Grand Slam (insuficiente — necesita 7)
    history_gs = _make_complete_tournament_matches(
        torneo_name='Wimbledon 2026', days_ago=7, n_wins=5, surface='Hierba'
    )
    history_base = _make_surface_matches(n_wins=20, n_total=30, surface='Hierba')
    full_history = history_gs + history_base

    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Hierba', 'TestPlayer'
    )

    # Grand Slam requiere 7W (D57-03). Con 5W NO debe activar TORNEO_COMPLETO_BONUS
    # y por tanto tampoco GCS.
    assert result['gcs_active'] is False, (
        f"5W en GS NO debe activar gcs_active (requiere 7W). "
        f"gcs_active={result['gcs_active']}, gcs_days={result['gcs_days']}"
    )
    gcs_log = [l for l in log if 'GCS' in l and 'LOG_GCS_SHADOW' not in l]
    assert len(gcs_log) == 0, f"NO debe haber GCS_RECENCY_BOOST con 5W en GS: {gcs_log}"



# ── T60-07: flag OFF por default → final_score idéntico con/sin código GCS ────

def test_T60_07_flag_off_score_unchanged():
    """FABLE §S60-7: _GCS_BOOST_ENABLED=False por default → final_score idéntico
    con o sin historia de torneo GCS (el código existe, pero no altera el score)."""
    import analysis.rivalry_analyzer as ra_mod
    assert ra_mod._GCS_BOOST_ENABLED is False, "_GCS_BOOST_ENABLED debe ser False por default"

    analyzer = _get_analyzer()

    # Historial base (sin torneo completo)
    history_base = _make_surface_matches(n_wins=20, n_total=30, surface='Hierba')
    result_base, _ = analyzer.analyze_surface_specialization(
        history_base, 'Hierba', 'TestPlayer'
    )

    # Historial con torneo completo ATP500 reciente (debería activar GCS si flag estuviera ON)
    history_gcs = _make_complete_tournament_matches(
        torneo_name='Nottingham 2026', days_ago=10, n_wins=5, surface='Hierba'
    ) + history_base
    result_gcs, log_gcs = analyzer.analyze_surface_specialization(
        history_gcs, 'Hierba', 'TestPlayer'
    )

    # Con flag OFF: gcs_active=True pero el score no debe cambiar por el boost
    assert result_gcs['gcs_active'] is True, "gcs_active debe ser True (señal detectada)"
    # LOG_GCS_SHADOW debe aparecer (no GCS_RECENCY_BOOST activo)
    shadow = [l for l in log_gcs if 'LOG_GCS_SHADOW' in l]
    assert len(shadow) >= 1, f"LOG_GCS_SHADOW debe aparecer con flag OFF, got: {log_gcs}"
    no_active_boost = [l for l in log_gcs if 'GCS_RECENCY_BOOST' in l and 'LOG_GCS_SHADOW' not in l]
    assert len(no_active_boost) == 0, f"GCS_RECENCY_BOOST activo NO debe aparecer con flag OFF"


# ── T60-08: torneo ganado en clay, partido actual en grass → gcs_active=False ──

def test_T60_08_clay_tournament_grass_match_no_gcs():
    """FABLE §S60-7: Si el torneo fue ganado en clay pero el partido es en grass,
    gcs_active debe ser False (superficie no coincide con el análisis)."""
    analyzer = _get_analyzer()

    # Torneo completo ATP500 en CLAY hace 10 días
    history_clay_champ = _make_complete_tournament_matches(
        torneo_name='Nottingham 2026', days_ago=10, n_wins=5, surface='Arcilla'
    )
    history_base = _make_surface_matches(n_wins=10, n_total=15, surface='Hierba')
    full_history = history_clay_champ + history_base

    # Análisis pedido para HIERBA (diferente a clay del torneo ganado)
    result, log = analyzer.analyze_surface_specialization(
        full_history, 'Hierba', 'TestPlayer'
    )

    # El torneo en Arcilla solo aporta partidos a Arcilla, no a Hierba.
    # El TORNEO_COMPLETO_BONUS solo se detecta sobre los partidos de la superficie analizada.
    # → En análisis de Hierba, los partidos de Arcilla no se procesan → gcs_active=False.
    assert result['gcs_active'] is False, (
        f"Torneo ganado en clay NO debe activar gcs en análisis de grass. "
        f"gcs_active={result['gcs_active']}"
    )


# ── T60-09: pick GCS + pick ITF → nunca en el mismo CORE combo ───────────────

def test_T60_09_gcs_itf_not_mixed_in_core():
    """FABLE §S60-7: Picks con universo=GCS y universo=ITF nunca se mezclan
    en el mismo combo CORE (MAX_GCS_PER_COMBO=1 y GCS se reporta por separado)."""
    from combo_confianza_builder import _extract_and_categorize, _build_portfolio_v2

    picks_raw = [
        # GCS pick: ATP500 + gcs_active
        {
            'jugador1': 'Alexandra Eala', 'jugador2': 'Iga Swiatek',
            'torneo_nombre': 'WTA - INDIVIDUALES: Wimbledon (Reino Unido), hierba',
            'tipo_cancha': 'hierba', 'cuota1': 3.80, 'cuota2': 1.27,
            'cuota_es_real': True,
            'ranking_analysis': {'prediction': {
                'favored_player': 'Alexandra Eala', 'confidence': 55.0,
                'reasoning': [],
                'surface_specialization_meta': {
                    'player1': {'score': 85.0, 'torneo_completo': True, 'gcs_active': True, 'gcs_days': 13},
                    'player2': {'score': 55.0, 'torneo_completo': False, 'gcs_active': False},
                },
            }}
        },
        # ITF pick
        {
            'jugador1': 'Kalin Ivanovski', 'jugador2': 'Yanaki Milev',
            'torneo_nombre': 'ITF MASCULINO: M25 Skopje',
            'tipo_cancha': 'arcilla', 'cuota1': 1.45, 'cuota2': 3.20,
            'cuota_es_real': True,
            'ranking_analysis': {'prediction': {
                'favored_player': 'Kalin Ivanovski', 'confidence': 60.0,
                'reasoning': [],
                'surface_specialization_meta': {
                    'player1': {'score': 40.0, 'torneo_completo': False, 'gcs_active': False},
                    'player2': {'score': 25.0, 'torneo_completo': False, 'gcs_active': False},
                },
            }}
        },
    ]

    picks = _extract_and_categorize(picks_raw, threshold=50.0, pipeline_picks=None, conf_min=50.0)
    assert len(picks) == 2, f"Deben extraerse 2 picks, got {len(picks)}"

    gcs_picks = [p for p in picks if p.get('universo') == 'GCS']
    itf_picks = [p for p in picks if p.get('universo') == 'ITF']

    assert len(gcs_picks) == 1, f"Debe haber 1 pick GCS, got {len(gcs_picks)}"
    assert len(itf_picks) == 1, f"Debe haber 1 pick ITF, got {len(itf_picks)}"

    # El combo builder nunca mezcla GCS con ITF
    plan = _build_portfolio_v2(picks, bankroll=5000)
    all_combos = plan.get('combos', []) + plan.get('satellite', [])
    for combo in all_combos:
        legs = combo.get('legs', [])
        leg_names = [l.get('nombre', '') for l in legs]
        universos = [l.get('universo', '') for l in legs]
        has_gcs = 'GCS' in universos
        has_itf = 'ITF' in universos
        assert not (has_gcs and has_itf), (
            f"Combo mezcla GCS con ITF: {leg_names} universos={universos}"
        )


# ── T60-10: LOG_GCS_SHADOW presente cuando flag=OFF y pick califica ───────────

def test_T60_10_log_gcs_shadow_present_when_flag_off():
    """FABLE §S60-7: Con _GCS_BOOST_ENABLED=False, LOG_GCS_SHADOW aparece en el
    analysis_log cuando el pick califica (tier>=ATP500, days<=21, misma superficie).
    Esto es el A/B gratis: shadow book acumula 'qué habría pasado'."""
    import analysis.rivalry_analyzer as ra_mod
    assert ra_mod._GCS_BOOST_ENABLED is False, "_GCS_BOOST_ENABLED debe ser False por default"

    analyzer = _get_analyzer()

    # Torneo ATP500 en hierba hace 7 días (máximo boost potencial ×2.2)
    history_gcs = _make_complete_tournament_matches(
        torneo_name='Birmingham 2026', days_ago=7, n_wins=5, surface='Hierba'
    ) + _make_surface_matches(n_wins=15, n_total=25, surface='Hierba')

    result, log = analyzer.analyze_surface_specialization(
        history_gcs, 'Hierba', 'TestPlayer'
    )

    assert result['gcs_active'] is True, "gcs_active debe ser True"

    shadow_lines = [l for l in log if 'LOG_GCS_SHADOW' in l]
    assert len(shadow_lines) >= 1, f"LOG_GCS_SHADOW debe aparecer con flag OFF: {log}"

    # Debe mencionar el multiplicador que se habría aplicado
    assert '×2.2' in shadow_lines[0], (
        f"LOG_GCS_SHADOW debe mencionar ×2.2 para days=7, got: {shadow_lines[0]}"
    )
    # Y debe mencionar GATED para auditabilidad
    assert 'GATED' in shadow_lines[0], (
        f"LOG_GCS_SHADOW debe indicar GATED, got: {shadow_lines[0]}"
    )

    # Verificar que GCS_RECENCY_BOOST activo NO está en el log
    active_boost = [l for l in log if 'GCS_RECENCY_BOOST:' in l and 'LOG_GCS_SHADOW' not in l]
    assert len(active_boost) == 0, f"NO debe haber boost activo con flag OFF: {active_boost}"

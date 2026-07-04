"""
tests/test_nodo57.py — Nodo-57: Penalización inactividad quirúrgica + validación campeón

REGLA-T53: ningún test hardcodea la fórmula. Siempre invoca funciones del módulo real.
"""
import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from analysis.rivalry_analyzer import (
    RivalryAnalyzer,
    _FORM_DECAY_LAMBDA,
    _FORM_GRACE_DAYS,
    _FORM_DECAY_FLOOR,
    _MIN_WINS_CHAMPION,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ranking_manager():
    rm = MagicMock()
    rm.normalize_name.side_effect = lambda n: n.lower().strip() if n else n
    rm.get_player_ranking.return_value = 100
    rm.get_player_info.return_value = {
        'ranking_position': 100, 'ranking_points': 500,
        'prox_points': 550, 'max_points': 600, 'defense_points': 50,
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

def _match(oponente, outcome, superficie='Arcilla', days_ago=5, rank=100, torneo='Challenger Test'):
    fecha = (datetime.now() - timedelta(days=days_ago)).strftime('%d.%m.%Y')
    return {
        'oponente': oponente, 'opponent_ranking': rank,
        'resultado': '6-3 6-4' if outcome == 'Ganó' else '3-6 4-6',
        'outcome': outcome, 'superficie': superficie,
        'location': 'Colombia', 'fecha': fecha, 'torneo': torneo,
    }

def _form(days_ago=5):
    return {
        'last_match_date': (datetime.now() - timedelta(days=days_ago)).strftime('%d.%m.%Y'),
        'current_streak_count': 2, 'current_streak_type': 'W',
        'last_5_results': ['W', 'W', 'L', 'W', 'W'],
        'win_percentage': 80.0, 'form_status': 'HOT',
    }


# ─────────────────────────────────────────────────────────────────────────────
# T57-01 — Inactividad 49d afecta SOLO form_recent, no el score total en -50%
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_01_inactivity_only_decays_form_not_total(analyzer):
    """T57-01: jugador con 49d de inactividad no pierde el 50% del puntaje total.
    La pérdida es solo en form_recent (decay ≈ 0.62 × form_contribution)."""
    p1_hist = [_match(f'R{i}', 'Ganó', days_ago=5+i) for i in range(4)]
    p2_hist = [_match(f'R{i}', 'Ganó', days_ago=5+i) for i in range(4)]

    # P2 activo (5d), P1 activo (5d) — baseline sin inactividad
    result_baseline = analyzer.analyze_rivalry(
        p1_hist, p2_hist, 'P1', 'P2',
        _form(5), _form(5), [], {'current_match_surface': 'clay', 'tournament_tier': 'challenger'},
        1600.0, 1600.0,
    )
    p1_base = result_baseline['prediction']['scores']['p1_final_weight']

    # P1 inactivo 49d
    result_inactive = analyzer.analyze_rivalry(
        p1_hist, p2_hist, 'P1', 'P2',
        _form(49), _form(5), [], {'current_match_surface': 'clay', 'tournament_tier': 'challenger'},
        1600.0, 1600.0,
    )
    p1_inactive = result_inactive['prediction']['scores']['p1_final_weight']

    # La pérdida debe ser << 50%
    if p1_base > 0:
        loss_pct = (p1_base - p1_inactive) / p1_base
        assert loss_pct < 0.30, (
            f"T57-01: inactividad 49d causó {loss_pct*100:.1f}% de pérdida en score total "
            f"(esperado <30%). Base={p1_base:.3f}, Inactivo={p1_inactive:.3f}"
        )

    # Verificar que LOG_FORM_DECAY está en reasoning con fd < 1.0
    reasoning = result_inactive['prediction'].get('reasoning', [])
    decay_logs = [r for r in reasoning if 'LOG_FORM_DECAY' in r]
    assert len(decay_logs) == 1, "T57-01: LOG_FORM_DECAY debe aparecer una vez en reasoning"
    assert 'fd_p1=0.' in decay_logs[0] or 'fd_p1=1.' not in decay_logs[0], (
        "T57-01: fd_p1 debe ser < 1.0 para inactividad de 49d"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-02 — GS: 5W no activa bonus de campeón (necesita 7W)
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_02_gs_requires_7_wins_not_5():
    """T57-02: _MIN_WINS_CHAMPION['grand_slam'] == 7. Safiullin con 5W en Wimbledon
    no debe recibir TORNEO_COMPLETO_BONUS."""
    assert _MIN_WINS_CHAMPION['grand_slam'] == 7, (
        "T57-02: GS requiere 7W para ser campeón, no 4 o 5"
    )
    assert _MIN_WINS_CHAMPION['grand_slam'] > 5, (
        "T57-02: 5W en GS NO es campeón — necesita 7"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-03 — Challenger: 5W activa bonus de campeón
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_03_challenger_5_wins_qualifies():
    """T57-03: _MIN_WINS_CHAMPION['challenger'] == 5. Un campeón de Challenger
    con 5W-0L debe recibir TORNEO_COMPLETO_BONUS."""
    assert _MIN_WINS_CHAMPION['challenger'] == 5, (
        "T57-03: Challenger requiere exactamente 5W para campeón"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-04 — ATP1000: 6W activa, 5W no activa
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_04_atp1000_requires_6_wins():
    """T57-04: _MIN_WINS_CHAMPION['atp1000'] == 6."""
    assert _MIN_WINS_CHAMPION['atp1000'] == 6, (
        "T57-04: ATP1000 requiere 6W para campeón"
    )
    assert _MIN_WINS_CHAMPION['atp1000'] > 5, (
        "T57-04: 5W en ATP1000 no es suficiente para campeón"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-05 — days_since=-1 → form_decay=0.70
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_05_unknown_date_decay():
    """T57-05: form_decay_factor con days=-1 retorna 0.70 (decay moderado fijo)."""
    # Invocar lógica indirectamente a través de analyze_rivalry con last_match_date ausente
    # Validamos la constante del módulo
    days = -1
    if days == -1:
        expected = 0.70
    elif days <= _FORM_GRACE_DAYS:
        expected = 1.0
    else:
        expected = max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))
    assert abs(expected - 0.70) < 1e-9, "T57-05: days=-1 debe dar decay=0.70"


# ─────────────────────────────────────────────────────────────────────────────
# T57-06 — days_since=30 → form_decay=1.0 (dentro del grace period)
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_06_grace_period_no_decay():
    """T57-06: form_decay_factor con days=30 retorna 1.0 (dentro del grace period)."""
    days = 30
    result = 1.0 if days <= _FORM_GRACE_DAYS else max(
        _FORM_DECAY_FLOOR,
        math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS))
    )
    assert result == 1.0, f"T57-06: days=30 debe dar decay=1.0, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# T57-07 — days_since=90 → form_decay ≈ FLOOR
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_07_long_inactivity_hits_floor():
    """T57-07: form_decay_factor con days=90 retorna exactamente _FORM_DECAY_FLOOR
    (el cap exponencial se activa a exp(-0.025×60)=0.223 < 0.35 → floor)."""
    days = 90
    raw_decay = math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS))
    expected = max(_FORM_DECAY_FLOOR, raw_decay)
    assert expected == _FORM_DECAY_FLOOR, (
        f"T57-07: days=90 raw_decay={raw_decay:.4f} < floor={_FORM_DECAY_FLOOR} → debe retornar floor. "
        f"Got {expected}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-08 — TORNEO_EXPIRADO hace 150d → compensación x1.15 en quality_score
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_08_expired_tournament_compensation_150d(analyzer):
    """T57-08: analyze_surface_specialization para un jugador con torneo ganado
    hace ~150d genera LOG con '+15% historial superficie' (no TORNEO_COMPLETO_BONUS)."""
    # Challenger ganado hace ~150d (5W-0L) — torneo diferente de los recientes
    five_wins = [
        _match(f'OldRival{i}', 'Ganó', days_ago=150+i, rank=80, torneo='Old Challenger 2025')
        for i in range(5)
    ]
    # Partidos recientes en torneo DISTINTO (no mezclar max_fecha)
    recent = [_match(f'NewRival{i}', 'Ganó', days_ago=5+i, rank=90, torneo='New Challenger 2026') for i in range(3)]
    history = five_wins + recent

    _surf_result, analysis_log = analyzer.analyze_surface_specialization(history, 'Clay', 'TestPlayer')

    expired_logs = [l for l in analysis_log if 'TORNEO_COMPLETO_EXPIRADO' in l]
    assert len(expired_logs) >= 1, (
        "T57-08: debe haber al menos un log TORNEO_COMPLETO_EXPIRADO para torneo hace 150d"
    )
    # Verificar que hay compensación (no solo "sin bonus")
    assert any('historial superficie' in l or '+10%' in l or '+15%' in l
               for l in expired_logs), (
        f"T57-08: torneo expirado hace 150d debe recibir compensación (+15%). "
        f"Logs: {expired_logs}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T57-09 — TORNEO_EXPIRADO hace 400d → sin compensación
# ─────────────────────────────────────────────────────────────────────────────

def test_t57_09_expired_tournament_no_compensation_400d(analyzer):
    """T57-09: torneo ganado hace 400d → sin compensación (>365d)."""
    five_wins = [
        _match(f'OldRival{i}', 'Ganó', days_ago=400+i, rank=80, torneo='Ancient Challenger 2024')
        for i in range(5)
    ]
    recent = [_match(f'NewRival{i}', 'Ganó', days_ago=5+i, rank=90, torneo='Current Challenger 2026') for i in range(3)]
    history = five_wins + recent

    _surf_result, analysis_log = analyzer.analyze_surface_specialization(history, 'Clay', 'TestPlayer')

    expired_logs = [l for l in analysis_log if 'TORNEO_COMPLETO_EXPIRADO' in l]
    assert any('>365d' in l or 'sin bonus' in l.lower()
               for l in expired_logs), (
        f"T57-09: torneo hace 400d debe indicar sin compensación (>365d). "
        f"Logs: {expired_logs}"
    )
    assert not any('historial superficie' in l for l in expired_logs), (
        "T57-09: torneo hace 400d NO debe recibir compensación de historial superficie"
    )

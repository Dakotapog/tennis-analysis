"""
tests/test_nodo72.py — Nodo-72: Phantom Identity Guard
12 tests: CIRCUIT_MISMATCH (Morris case) + HOMONYM_GAP (Pereyra case) + edge gate
REGLA-T53: todos invocan la función real del módulo, sin hardcodear lógica.
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta


def _make_analyzer():
    """Helper: RivalryAnalyzer con ranking_manager mockeado."""
    from analysis.rivalry_analyzer import RivalryAnalyzer
    analyzer = RivalryAnalyzer.__new__(RivalryAnalyzer)
    analyzer.ranking_manager = MagicMock()
    return analyzer


def _history(n, days_old, torneo='Some Tournament'):
    """Helper: n entradas de historial de hace days_old días."""
    fecha = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
    return [
        {'oponente': f'Player{i}', 'fecha': fecha, 'torneo': torneo, 'superficie': 'hard'}
        for i in range(n)
    ]


# ── T72-01: WTA player con >50% oponentes ATP → CIRCUIT_MISMATCH ─────────────
def test_t72_01_wta_player_atp_opponents():
    analyzer = _make_analyzer()
    player_info = {'tour': 'wta', 'ranking_position': None}
    analyzer.ranking_manager.get_player_info = lambda name: {'tour': 'atp', 'ranking_position': 50}

    result = analyzer._detect_phantom_identity('Ariana Morris', player_info, _history(8, 30))

    assert result['phantom'] is True
    assert result['type'] == 'CIRCUIT_MISMATCH'
    assert result['confidence'] > 0.6


# ── T72-02: ATP player con >50% oponentes WTA → CIRCUIT_MISMATCH ─────────────
def test_t72_02_atp_player_wta_opponents():
    analyzer = _make_analyzer()
    player_info = {'tour': 'atp', 'ranking_position': 200}
    analyzer.ranking_manager.get_player_info = lambda name: {'tour': 'wta', 'ranking_position': 30}

    result = analyzer._detect_phantom_identity('Some Player', player_info, _history(6, 30))

    assert result['phantom'] is True
    assert result['type'] == 'CIRCUIT_MISMATCH'


# ── T72-03: WTA player con oponentes WTA → NO phantom ────────────────────────
def test_t72_03_wta_player_wta_opponents_clean():
    analyzer = _make_analyzer()
    player_info = {'tour': 'wta', 'ranking_position': 150}
    analyzer.ranking_manager.get_player_info = lambda name: {'tour': 'wta', 'ranking_position': 80}

    result = analyzer._detect_phantom_identity('Clean Player', player_info, _history(6, 30))

    assert result['phantom'] is False
    assert result['type'] is None


# ── T72-04: Sin ranking, n=25, oldest=400d → HOMONYM_GAP ─────────────────────
def test_t72_04_no_ranking_old_history_homonym_gap():
    analyzer = _make_analyzer()
    player_info = {'tour': 'atp', 'ranking_position': None}

    result = analyzer._detect_phantom_identity('Facundo Pereyra', player_info, _history(25, 400))

    assert result['phantom'] is True
    assert result['type'] == 'HOMONYM_GAP'
    assert result['confidence'] >= 0.85


# ── T72-05: Sin ranking, n=15 (≤20) → NOT phantom ────────────────────────────
def test_t72_05_no_ranking_few_matches_not_phantom():
    analyzer = _make_analyzer()
    player_info = {'tour': 'atp', 'ranking_position': None}
    # n=15 ≤ 20 no activa HOMONYM_GAP; oponentes sin tour en DB → no CIRCUIT_MISMATCH
    analyzer.ranking_manager.get_player_info = MagicMock(return_value=None)

    result = analyzer._detect_phantom_identity('Young Player', player_info, _history(15, 400))

    assert result['phantom'] is False


# ── T72-06: Sin ranking, n=25 pero oldest=300d → NOT phantom ─────────────────
def test_t72_06_no_ranking_recent_history_not_phantom():
    analyzer = _make_analyzer()
    player_info = {'tour': 'atp', 'ranking_position': None}
    analyzer.ranking_manager.get_player_info = MagicMock(return_value=None)

    # 300d < 365 threshold → no HOMONYM_GAP
    result = analyzer._detect_phantom_identity('New Player', player_info, _history(25, 300))

    assert result['phantom'] is False


# ── T72-07: WTA player con torneos "M15 Lodz" → CIRCUIT_MISMATCH (prefijos) ──
def test_t72_07_wta_player_m15_tournaments():
    analyzer = _make_analyzer()
    player_info = {'tour': 'wta', 'ranking_position': 500}
    # Oponentes no en DB → Señal A no activa; depende de Señal B (prefijos)
    analyzer.ranking_manager.get_player_info = MagicMock(return_value=None)

    history = _history(10, 30, torneo='M15 Lodz Poland')
    result = analyzer._detect_phantom_identity('Ariana Morris', player_info, history)

    assert result['phantom'] is True
    assert result['type'] == 'CIRCUIT_MISMATCH'


# ── T72-08: Historial vacío → NOT phantom ────────────────────────────────────
def test_t72_08_empty_history_not_phantom():
    analyzer = _make_analyzer()
    player_info = {'tour': 'wta', 'ranking_position': 100}

    result = analyzer._detect_phantom_identity('Some Player', player_info, [])

    assert result['phantom'] is False
    assert result['confidence'] == 0.0


# ── T72-09: confidence > 0.6 en CIRCUIT_MISMATCH con 100% ratio ──────────────
def test_t72_09_circuit_mismatch_confidence_above_threshold():
    analyzer = _make_analyzer()
    player_info = {'tour': 'wta', 'ranking_position': None}
    analyzer.ranking_manager.get_player_info = lambda name: {'tour': 'atp', 'ranking_position': 50}

    result = analyzer._detect_phantom_identity('Test Player', player_info, _history(8, 30))

    assert result['phantom'] is True
    assert result['confidence'] > 0.6


# ── T72-10: player_info=None → no crash, retorna dict válido ─────────────────
def test_t72_10_none_player_info_no_crash():
    analyzer = _make_analyzer()
    analyzer.ranking_manager.get_player_info = MagicMock(return_value=None)

    # n=10 ≤ 20 → no HOMONYM_GAP; tour='' → no CIRCUIT_MISMATCH check
    result = analyzer._detect_phantom_identity('Unknown', None, _history(10, 30))

    assert isinstance(result, dict)
    assert 'phantom' in result
    assert 'type' in result
    assert 'confidence' in result


# ── T72-11: _detect_phantom_identity importable standalone (REGLA-T53) ────────
def test_t72_11_method_exists_on_class():
    from analysis.rivalry_analyzer import RivalryAnalyzer

    assert hasattr(RivalryAnalyzer, '_detect_phantom_identity')
    assert callable(RivalryAnalyzer._detect_phantom_identity)


# ── T72-12: HOMONYM_GAP boundary n=20 vs n=21 ────────────────────────────────
def test_t72_12_homonym_gap_boundary_n20_vs_n21():
    analyzer = _make_analyzer()
    player_info = {'tour': 'atp', 'ranking_position': None}
    analyzer.ranking_manager.get_player_info = MagicMock(return_value=None)

    # n=20 → NO activa (umbral es > 20)
    result_20 = analyzer._detect_phantom_identity('Boundary', player_info, _history(20, 400))
    assert result_20['phantom'] is False

    # n=21 → SÍ activa HOMONYM_GAP
    result_21 = analyzer._detect_phantom_identity('Boundary', player_info, _history(21, 400))
    assert result_21['phantom'] is True
    assert result_21['type'] == 'HOMONYM_GAP'

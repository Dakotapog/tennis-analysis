"""
Tests migrados desde SequentialH2HExtractor → H2HExtractor + DataParser.
T07-09 — Nodo-07 Fase 2: eliminación de SequentialH2HExtractor (Strangler Fig).

Cubre métodos puros (sin Playwright) de H2HExtractor y DataParser:
  - DataParser.determine_winner_from_result  (estático)
  - DataParser.extract_winner_sets           (estático)
  - H2HExtractor._classify_form             (win_pct: float)
  - H2HExtractor._analyze_recent_form
  - H2HExtractor.load_matches
  - H2HExtractor._generate_global_statistics
  - H2HExtractor.__init__ (estado inicial)
  - Surface propagation logic (Nodo-10)

Migración completada 2026-05-30 — 52 tests → 48 tests
  Eliminados: TestAnalyzeCommonOpponents (3) — cubierto en test_rivalry_analyzer.py
              test_zero_total_is_sin_datos (1) — _classify_form recibe float, no (int,int)
"""
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from scraping.data_parser import DataParser


# ── Fixture principal ─────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    """H2HExtractor con dependencias externas mockeadas."""
    with patch('scraping.h2h_extractor.RankingManager') as mock_rm, \
         patch('scraping.h2h_extractor.EloRatingSystem'), \
         patch('scraping.h2h_extractor.RivalryAnalyzer'), \
         patch('signal.signal'), \
         patch('atexit.register'):
        from scraping.h2h_extractor import H2HExtractor
        inst = H2HExtractor()
        inst.ranking_manager.normalize_name.side_effect = lambda x: x.lower().strip()
        inst.ranking_manager.get_player_ranking.return_value = 50
        yield inst


# ── __init__ — estado inicial ─────────────────────────────────────────────────

class TestInit:
    def test_default_total_matches(self, extractor):
        assert extractor.total_matches_to_process == 80

    def test_empty_matches_queue(self, extractor):
        assert extractor.matches_queue == []

    def test_empty_all_results(self, extractor):
        assert extractor.all_results == []

    def test_current_match_index_zero(self, extractor):
        assert extractor.current_match_index == 0


# ── DataParser.determine_winner_from_result ───────────────────────────────────

class TestDetermineWinnerFromResult:
    def test_player1_wins_2_0(self):
        assert DataParser.determine_winner_from_result('2-0', 'Alcaraz', 'Djokovic') == 'Alcaraz'

    def test_player2_wins_0_2(self):
        assert DataParser.determine_winner_from_result('0-2', 'Alcaraz', 'Djokovic') == 'Djokovic'

    def test_player1_wins_three_sets(self):
        assert DataParser.determine_winner_from_result('3-1', 'Nadal', 'Federer') == 'Nadal'

    def test_player2_wins_three_sets(self):
        assert DataParser.determine_winner_from_result('1-3', 'Nadal', 'Federer') == 'Federer'

    def test_none_result_returns_na(self):
        assert DataParser.determine_winner_from_result(None, 'A', 'B') == 'N/A'

    def test_na_string_returns_na(self):
        assert DataParser.determine_winner_from_result('N/A', 'A', 'B') == 'N/A'

    def test_empty_string_returns_na(self):
        assert DataParser.determine_winner_from_result('', 'A', 'B') == 'N/A'

    def test_malformed_result_returns_na(self):
        assert DataParser.determine_winner_from_result('abc-xyz', 'A', 'B') == 'N/A'

    def test_draw_returns_na(self):
        # Tenis no tiene empates — si scraping da 1-1 es un error
        assert DataParser.determine_winner_from_result('1-1', 'A', 'B') == 'N/A'


# ── DataParser.extract_winner_sets ────────────────────────────────────────────

class TestExtractWinnerSets:
    def test_2_0_returns_2(self):
        assert DataParser.extract_winner_sets('2-0') == 2

    def test_1_2_returns_2(self):
        assert DataParser.extract_winner_sets('1-2') == 2

    def test_3_1_returns_3(self):
        assert DataParser.extract_winner_sets('3-1') == 3

    def test_none_returns_na(self):
        assert DataParser.extract_winner_sets(None) == 'N/A'

    def test_na_string_returns_na(self):
        assert DataParser.extract_winner_sets('N/A') == 'N/A'

    def test_malformed_returns_na(self):
        assert DataParser.extract_winner_sets('bad') == 'N/A'


# ── H2HExtractor._classify_form ──────────────────────────────────────────────

class TestClassifyForm:
    """_classify_form recibe win_pct: float (no (wins, total))."""

    def test_100_percent_is_excelente(self, extractor):
        assert extractor._classify_form(100.0) == 'Excelente'

    def test_75_percent_boundary_is_excelente(self, extractor):
        assert extractor._classify_form(75.0) == 'Excelente'

    def test_60_percent_is_buena(self, extractor):
        assert extractor._classify_form(60.0) == 'Buena'

    def test_40_percent_is_regular(self, extractor):
        assert extractor._classify_form(40.0) == 'Regular'

    def test_39_percent_is_mala(self, extractor):
        assert extractor._classify_form(39.0) == 'Mala'

    def test_0_percent_is_mala(self, extractor):
        assert extractor._classify_form(0.0) == 'Mala'


# ── H2HExtractor._analyze_recent_form ────────────────────────────────────────

class TestAnalyzeRecentForm:
    def test_empty_history_returns_none(self, extractor):
        assert extractor._analyze_recent_form([], 'Alcaraz') is None

    def test_all_wins_returns_100_percent(self, extractor):
        history = [{'outcome': 'Ganó'}] * 5
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['win_percentage'] == 100.0
        assert form['form_status'] == 'Excelente'

    def test_all_losses_returns_0_percent(self, extractor):
        history = [{'outcome': 'Perdió'}] * 5
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['win_percentage'] == 0.0
        assert form['form_status'] == 'Mala'

    def test_counts_wins_and_losses(self, extractor):
        history = [
            {'outcome': 'Ganó'}, {'outcome': 'Ganó'}, {'outcome': 'Perdió'},
            {'outcome': 'Ganó'}, {'outcome': 'Perdió'}
        ]
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['wins'] == 3
        assert form['losses'] == 2

    def test_win_streak_detection(self, extractor):
        history = [
            {'outcome': 'Ganó'}, {'outcome': 'Ganó'}, {'outcome': 'Ganó'},
            {'outcome': 'Perdió'}
        ]
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['current_streak_count'] == 3
        assert 'victorias' in form['current_streak_type']

    def test_loss_streak_detection(self, extractor):
        history = [
            {'outcome': 'Perdió'}, {'outcome': 'Perdió'}, {'outcome': 'Ganó'}
        ]
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['current_streak_count'] == 2
        assert 'derrotas' in form['current_streak_type']

    def test_respects_recent_count(self, extractor):
        history = [{'outcome': 'Ganó'}] * 10 + [{'outcome': 'Perdió'}] * 10
        form = extractor._analyze_recent_form(history, 'Alcaraz', recent_count=5)
        assert form['recent_matches_count'] == 5
        assert form['wins'] == 5

    def test_returns_player_name(self, extractor):
        history = [{'outcome': 'Ganó'}]
        form = extractor._analyze_recent_form(history, 'Sinner')
        assert form['player_name'] == 'Sinner'

    def test_win_outcome_alias(self, extractor):
        """'win' es alias aceptado de 'Ganó'."""
        history = [{'outcome': 'win'}, {'outcome': 'win'}]
        form = extractor._analyze_recent_form(history, 'Alcaraz')
        assert form['wins'] == 2


# ── H2HExtractor.load_matches ────────────────────────────────────────────────

class TestLoadMatchesFromJson:
    def test_no_file_returns_false(self, extractor):
        with patch('scraping.h2h_extractor.select_best_json_file', return_value=None):
            assert extractor.load_matches() is False

    def test_list_structure_loads_correctly(self, extractor):
        matches = [{'jugador1': 'A', 'jugador2': 'B', 'match_url': 'http://x'}]
        with patch('scraping.h2h_extractor.select_best_json_file', return_value='data/test.json'), \
             patch('builtins.open', mock_open(read_data=json.dumps(matches))):
            result = extractor.load_matches()
        assert result is True
        assert len(extractor.matches_queue) == 1

    def test_matches_without_url_are_excluded(self, extractor):
        matches = [
            {'jugador1': 'A', 'jugador2': 'B', 'match_url': 'http://valid'},
            {'jugador1': 'C', 'jugador2': 'D'},  # sin match_url
        ]
        with patch('scraping.h2h_extractor.select_best_json_file', return_value='data/test.json'), \
             patch('builtins.open', mock_open(read_data=json.dumps(matches))):
            result = extractor.load_matches()
        assert result is True
        assert len(extractor.matches_queue) == 1

    def test_empty_list_returns_false(self, extractor):
        with patch('scraping.h2h_extractor.select_best_json_file', return_value='data/test.json'), \
             patch('builtins.open', mock_open(read_data=json.dumps([]))):
            assert extractor.load_matches() is False

    def test_roland_garros_filter_applied(self, extractor):
        matches = [
            {'jugador1': 'A', 'jugador2': 'B', 'match_url': 'http://rg',
             'torneo_completo': 'French Open (France), clay'},
            {'jugador1': 'C', 'jugador2': 'D', 'match_url': 'http://other',
             'torneo_completo': 'Wimbledon (UK), grass'},
        ]
        with patch('scraping.h2h_extractor.select_best_json_file', return_value='data/test.json'), \
             patch('builtins.open', mock_open(read_data=json.dumps(matches))):
            result = extractor.load_matches()
        assert result is True
        assert len(extractor.matches_queue) == 1
        assert extractor.matches_queue[0]['jugador1'] == 'A'


# ── H2HExtractor._generate_global_statistics ─────────────────────────────────

class TestGenerateGlobalStatistics:
    def test_empty_results_returns_empty_dict(self, extractor):
        assert extractor._generate_global_statistics() == {}

    def test_counts_total_partidos(self, extractor):
        extractor.all_results = [
            {'jugador1': 'Alcaraz', 'jugador2': 'Sinner',
             'historial_Alcaraz': [], 'historial_Sinner': [],
             'enfrentamientos_directos': [], 'ranking_analysis': {}},
            {'jugador1': 'Nadal', 'jugador2': 'Federer',
             'historial_Nadal': [], 'historial_Federer': [],
             'enfrentamientos_directos': [], 'ranking_analysis': {}},
        ]
        stats = extractor._generate_global_statistics()
        assert stats['total_partidos'] == 2

    def test_counts_unique_players(self, extractor):
        extractor.all_results = [
            {'jugador1': 'Alcaraz', 'jugador2': 'Sinner',
             'historial_Alcaraz': [], 'historial_Sinner': [],
             'enfrentamientos_directos': [], 'ranking_analysis': {}},
        ]
        stats = extractor._generate_global_statistics()
        assert stats['jugadores_unicos'] == 2


# ── surface propagation (Nodo-10) ─────────────────────────────────────────────

class TestSurfacePropagation:
    """T10-03 — superficie de S1 debe llegar a current_match_context."""

    def test_superficie_field_preserved_in_queue(self, extractor):
        """Campo 'superficie' del JSON S1 se conserva en matches_queue."""
        matches = [{'jugador1': 'A', 'jugador2': 'B',
                    'match_url': 'http://x', 'superficie': 'clay',
                    'torneo_completo': 'French Open (France), clay'}]
        with patch('scraping.h2h_extractor.select_best_json_file', return_value='data/test.json'), \
             patch('builtins.open', mock_open(read_data=json.dumps(matches))):
            extractor.load_matches()
        assert extractor.matches_queue[0].get('superficie') == 'clay'

    def test_superficie_preferred_over_tipo_cancha(self, extractor):
        """Cuando 'superficie' existe, se usa sobre 'tipo_cancha'."""
        match_data = {'superficie': 'clay', 'tipo_cancha': 'Desconocida'}
        surface = match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
        assert surface == 'clay'

    def test_tipo_cancha_fallback_when_no_superficie(self, extractor):
        """Sin 'superficie', cae al campo 'tipo_cancha'."""
        match_data = {'superficie': None, 'tipo_cancha': 'Arcilla'}
        surface = match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
        assert surface == 'Arcilla'

    def test_na_when_both_missing(self, extractor):
        """Sin ningún campo de superficie, resulta 'N/A'."""
        match_data = {}
        surface = match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
        assert surface == 'N/A'

    def test_hard_surface_propagates(self, extractor):
        match_data = {'superficie': 'hard', 'tipo_cancha': None}
        surface = match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
        assert surface == 'hard'

    def test_grass_surface_propagates(self, extractor):
        match_data = {'superficie': 'grass', 'tipo_cancha': None}
        surface = match_data.get('superficie') or match_data.get('tipo_cancha') or 'N/A'
        assert surface == 'grass'

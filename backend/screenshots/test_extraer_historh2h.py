import pytest
from unittest.mock import Mock
from analysis.rivalry_analyzer import RivalryAnalyzer

class TestRivalryAnalyzer:
    @pytest.fixture
    def mock_ranking_manager(self):
        """Fixture to create a mock RankingManager."""
        return Mock()

    @pytest.fixture
    def mock_elo_system(self):
        """Fixture to create a mock EloRatingSystem."""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_ranking_manager, mock_elo_system):
        """Fixture to create an instance of RivalryAnalyzer with mocks."""
        return RivalryAnalyzer(ranking_manager=mock_ranking_manager, elo_system=mock_elo_system)
    
    def test_initialization(self, analyzer):
        """Test that the RivalryAnalyzer class can be initialized successfully."""
        assert analyzer is not None
        assert analyzer.ranking_manager is not None
        assert analyzer.elo_system is not None

    def test_estimate_elo_from_rank_top_player(self, analyzer):
        """Test ELO estimation for a top-ranked player."""
        assert analyzer.estimate_elo_from_rank(1) == 2400

    def test_estimate_elo_from_rank_mid_rank_player(self, analyzer):
        """Test ELO estimation for a mid-ranked player."""
        assert analyzer.estimate_elo_from_rank(50) == 2005
    
    def test_estimate_elo_from_rank_low_rank_player(self, analyzer):
        """Test ELO estimation for a lower-ranked player."""
        assert analyzer.estimate_elo_from_rank(150) == 1751

    def test_estimate_elo_from_rank_unranked_player(self, analyzer):
        """Test ELO estimation for a player with no rank."""
        assert analyzer.estimate_elo_from_rank(None) == 1500

    # Tests for calculate_base_opponent_weight
    def test_calculate_base_opponent_weight_top_10(self, analyzer):
        """Test base opponent weight for a Top 10 player."""
        assert analyzer.calculate_base_opponent_weight(5) == 10

    def test_calculate_base_opponent_weight_rank_31_50(self, analyzer):
        """Test base opponent weight for a player ranked between 31 and 50."""
        assert analyzer.calculate_base_opponent_weight(45) == 4

    def test_calculate_base_opponent_weight_rank_over_100(self, analyzer):
        """Test base opponent weight for a player ranked over 100."""
        assert analyzer.calculate_base_opponent_weight(150) == 1

    def test_calculate_base_opponent_weight_unranked(self, analyzer):
        """Test base opponent weight for an unranked player."""
        assert analyzer.calculate_base_opponent_weight(None) == 1

    # Tests for analizar_contundencia
    @pytest.mark.parametrize("resultado, expected_factor", [
        ("2-0", 1.2),  # Victoria en 2 sets (default sin desglose de juegos)
        ("2-1", 1.0),  # Victoria en 3 sets
        ("Resultado inválido", 1.0), # Error case
        ("", 1.0) # Empty string
    ])
    def test_analizar_contundencia(self, analyzer, resultado, expected_factor):
        """Test the analysis of victory decisiveness based on available data."""
        assert analyzer.analizar_contundencia(resultado) == expected_factor

    # Tests for analizar_resistencia
    @pytest.mark.parametrize("resultado, expected_factor", [
        ("1-2", 0.5), # Perdió pero ganó 1 set
        ("0-2", 0.0), # Perdió en 2 sets (no se puede determinar si fue ajustado)
        ("Resultado inválido", 0.0), # Error case
        ("", 0.0) # Empty string
    ])
    def test_analizar_resistencia(self, analyzer, resultado, expected_factor):
        """Test the analysis of resistance in a loss based on available data."""
        assert analyzer.analizar_resistencia(resultado) == expected_factor

    # Tests for determine_match_winner
    @pytest.mark.parametrize("match_data, expected_win", [
        ({'outcome': 'Ganó el partido'}, True),
        ({'outcome': 'Player win'}, True),
        ({'outcome': 'Perdió el partido'}, False),
        ({'outcome': 'Player loss'}, False),
        ({'resultado': '6-2, 6-3 RET'}, True),
        ({'resultado': 'WO'}, True),
        ({'outcome': 'Resultado raro', 'resultado': '6-4, 6-4'}, True), # Fallback
        ({}, True), # Empty dict fallback
        ({'outcome': 'perdió'}, False)
    ])
    def test_determine_match_winner(self, analyzer, match_data, expected_win):
        """Test the logic for determining the match winner from various data."""
        assert analyzer.determine_match_winner(match_data, 'player_name') == expected_win

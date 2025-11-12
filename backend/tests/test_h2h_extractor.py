"""Tests para H2HExtractor."""
import pytest
from unittest.mock import patch, MagicMock
from scraping.h2h_extractor import H2HExtractor

class TestH2HExtractor:
    """Suite de tests para H2HExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Instancia limpia para tests."""
        with patch('scraping.h2h_extractor.RankingManager'), \
             patch('scraping.h2h_extractor.EloRatingSystem'), \
             patch('scraping.h2h_extractor.RivalryAnalyzer'):
            extractor = H2HExtractor()
            yield extractor
    
    def test_initialization(self, extractor):
        """Test: La clase se inicializa correctamente."""
        assert extractor is not None
        assert extractor.total_matches_to_process == 80
        assert not extractor.all_results

    @patch('scraping.h2h_extractor.select_best_json_file')
    def test_load_matches_success(self, mock_select_file, extractor):
        """Test: Carga de partidos desde un JSON válido."""
        mock_select_file.return_value = 'tests/mock_data.json'
        
        result = extractor.load_matches()
        
        assert result is True
        assert len(extractor.matches_queue) == 2
        assert extractor.matches_queue[0]['jugador1'] == 'Player A'
        assert extractor.matches_queue[0]['torneo_nombre'] == 'Test Tournament (Testland)'

    @patch('scraping.h2h_extractor.select_best_json_file')
    def test_load_matches_no_file(self, mock_select_file, extractor):
        """Test: No se encuentra un archivo JSON válido."""
        mock_select_file.return_value = None
        
        result = extractor.load_matches()
        
        assert result is False
        assert len(extractor.matches_queue) == 0

    def test_enrich_history(self, extractor):
        """Test: Enriquecimiento del historial con datos de ranking."""
        history = [{'oponente': 'Test Player'}]
        extractor.ranking_manager.get_player_ranking = MagicMock(return_value=10)
        extractor.rivalry_analyzer.calculate_base_opponent_weight = MagicMock(return_value=10)
        
        enriched_history = extractor._enrich_history(history, 'Player X')
        
        assert enriched_history[0]['opponent_ranking'] == 10
        assert enriched_history[0]['opponent_weight'] == 10

    def test_analyze_recent_form(self, extractor):
        """Test: Análisis de la forma reciente de un jugador."""
        history = [
            {'outcome': 'Ganó'}, {'outcome': 'Ganó'}, {'outcome': 'Perdió'},
            {'outcome': 'Ganó'}, {'outcome': 'Ganó'}
     ]
    # Cambiar esto:
    # form = extractor.analyze_recent_form_in_extractor(history, 'Player X', recent_count=5)
    
    # Por esto:
        form = extractor._analyze_recent_form(history, 'Player X', recent_count=5)
    
        assert form['wins'] == 4
        assert form['losses'] == 1
        assert form['win_percentage'] == 80.0
        assert form['form_status'] == 'Excelente'
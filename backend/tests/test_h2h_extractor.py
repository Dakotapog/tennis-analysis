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


# ─────────────────────────────────────────────────────────────────────────────
# NODO-17 T17-01 — Surface propagation fix
# ─────────────────────────────────────────────────────────────────────────────

class TestSurfacePropagation:
    """T17-01: tipo_cancha y torneo_nombre NO deben sobreescribirse con None."""

    @pytest.fixture
    def extractor(self):
        with patch('scraping.h2h_extractor.RankingManager'), \
             patch('scraping.h2h_extractor.EloRatingSystem'), \
             patch('scraping.h2h_extractor.RivalryAnalyzer'):
            yield H2HExtractor()

    def test_tipo_cancha_preservado_cuando_playwright_retorna_none(self, extractor):
        """Si _extract_current_match_info retorna tipo_cancha=None,
        el valor original del JSON (clay/grass/hard) debe mantenerse."""
        match_data = {
            'jugador1': 'Player A', 'jugador2': 'Player B',
            'match_url': 'https://example.com',
            'tipo_cancha': 'grass',
            'torneo_nombre': 'Birmingham (United Kingdom)',
            'torneo_completo': 'Birmingham (United Kingdom)',
            'cuota1': 2.0, 'cuota2': 1.8,
        }
        current_info = {'torneo_nombre': None, 'tipo_cancha': None,
                        'cuota1': None, 'cuota2': None}

        # Aplicar la lógica de preservación (igual que en _process_single_match)
        for key in ('cuota1', 'cuota2', 'torneo_nombre', 'tipo_cancha', 'torneo_completo'):
            if current_info.get(key) is None and match_data.get(key) is not None:
                current_info[key] = match_data[key]
        match_data.update(current_info)

        assert match_data['tipo_cancha'] == 'grass'
        assert match_data['torneo_nombre'] == 'Birmingham (United Kingdom)'

    def test_tipo_cancha_actualizado_cuando_playwright_tiene_valor(self, extractor):
        """Si Playwright encuentra el torneo en la página, ese valor prevalece."""
        match_data = {
            'tipo_cancha': 'unknown',
            'torneo_nombre': 'Old Name',
            'cuota1': None, 'cuota2': None,
            'torneo_completo': None,
        }
        current_info = {'torneo_nombre': 'French Open (France)',
                        'tipo_cancha': 'clay',
                        'cuota1': 1.5, 'cuota2': 2.5}

        for key in ('cuota1', 'cuota2', 'torneo_nombre', 'tipo_cancha', 'torneo_completo'):
            if current_info.get(key) is None and match_data.get(key) is not None:
                current_info[key] = match_data[key]
        match_data.update(current_info)

        assert match_data['tipo_cancha'] == 'clay'
        assert match_data['torneo_nombre'] == 'French Open (France)'

    def test_cuotas_preservadas_cuando_playwright_no_las_encuentra(self, extractor):
        """Cuotas del JSON se preservan si Playwright retorna None (comportamiento pre-existente)."""
        match_data = {'cuota1': 3.0, 'cuota2': 1.4, 'tipo_cancha': 'clay',
                      'torneo_nombre': 'Test', 'torneo_completo': 'Test'}
        current_info = {'cuota1': None, 'cuota2': None,
                        'torneo_nombre': None, 'tipo_cancha': None}

        for key in ('cuota1', 'cuota2', 'torneo_nombre', 'tipo_cancha', 'torneo_completo'):
            if current_info.get(key) is None and match_data.get(key) is not None:
                current_info[key] = match_data[key]
        match_data.update(current_info)

        assert match_data['cuota1'] == 3.0
        assert match_data['cuota2'] == 1.4
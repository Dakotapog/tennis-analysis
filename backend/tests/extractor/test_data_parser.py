"""
🧪 Tests para DataParser
"""

import pytest
from datetime import datetime
from scraping.data_parser import DataParser


class TestDataParser:
    """Suite de tests para DataParser."""
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: extract_tournament_info
    # ═══════════════════════════════════════════════════════════════
    
    def test_extract_tournament_info_full_string(self):
        """Test: Extraer información completa del torneo."""
        tournament_string = "ATP: Roland Garros (Francia), clay"
        
        result = DataParser.extract_tournament_info(tournament_string)
        
        assert result['nombre'] == 'Roland Garros (Francia)'
        assert result['pais'] == 'Francia'
        assert result['superficie'] == 'clay'
        assert result['completo'] == tournament_string
    
    def test_extract_tournament_info_no_surface(self):
        """Test: Torneo sin superficie explícita."""
        tournament_string = "ATP: Indian Wells (USA)"
        
        result = DataParser.extract_tournament_info(tournament_string)
        
        assert result['nombre'] == 'Indian Wells (USA)'
        assert result['pais'] == 'USA'
        assert result['superficie'] == 'Desconocida'
    
    def test_extract_tournament_info_no_country(self):
        """Test: Torneo sin país."""
        tournament_string = "ATP: Masters Cup, hard"
        
        result = DataParser.extract_tournament_info(tournament_string)
        
        assert 'Masters Cup' in result['nombre']
        assert result['pais'] == 'N/A'
        assert result['superficie'] == 'hard'
    
    def test_extract_tournament_info_empty_string(self):
        """Test: String vacío."""
        result = DataParser.extract_tournament_info("")
        
        assert result['nombre'] == ''
        assert result['pais'] == 'N/A'
        assert result['superficie'] == 'Desconocida'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: normalize_surface
    # ═══════════════════════════════════════════════════════════════
    
    def test_normalize_surface_hard(self):
        """Test: Normalizar superficie 'hard'."""
        assert DataParser.normalize_surface('hard') == 'Dura'
        assert DataParser.normalize_surface('HARD') == 'Dura'
        assert DataParser.normalize_surface('dura') == 'Dura'
    
    def test_normalize_surface_clay(self):
        """Test: Normalizar superficie 'clay'."""
        assert DataParser.normalize_surface('clay') == 'Arcilla'
        assert DataParser.normalize_surface('arcilla') == 'Arcilla'
    
    def test_normalize_surface_grass(self):
        """Test: Normalizar superficie 'grass'."""
        assert DataParser.normalize_surface('grass') == 'Hierba'
        assert DataParser.normalize_surface('hierba') == 'Hierba'
    
    def test_normalize_surface_indoor(self):
        """Test: Normalizar superficie 'indoor'."""
        assert DataParser.normalize_surface('indoor') == 'Indoor'
    
    def test_normalize_surface_unknown(self):
        """Test: Superficie desconocida se capitaliza."""
        assert DataParser.normalize_surface('carpet') == 'Carpet'
    
    def test_normalize_surface_none(self):
        """Test: None retorna 'Desconocida'."""
        assert DataParser.normalize_surface(None) == 'Desconocida'
    
    def test_normalize_surface_empty(self):
        """Test: String vacío retorna 'Desconocida'."""
        assert DataParser.normalize_surface('') == 'Desconocida'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: determine_winner_from_result
    # ═══════════════════════════════════════════════════════════════
    
    def test_determine_winner_player1_wins(self):
        """Test: Jugador 1 gana por sets."""
        winner = DataParser.determine_winner_from_result('2-0', 'Nadal', 'Federer')
        assert winner == 'Nadal'
    
    def test_determine_winner_player2_wins(self):
        """Test: Jugador 2 gana por sets."""
        winner = DataParser.determine_winner_from_result('1-2', 'Nadal', 'Federer')
        assert winner == 'Federer'
    
    def test_determine_winner_tie(self):
        """Test: Empate en sets retorna N/A."""
        winner = DataParser.determine_winner_from_result('1-1', 'Nadal', 'Federer')
        assert winner == 'N/A'
    
    def test_determine_winner_invalid_format(self):
        """Test: Formato inválido retorna N/A."""
        winner = DataParser.determine_winner_from_result('ABC', 'Nadal', 'Federer')
        assert winner == 'N/A'
    
    def test_determine_winner_none_result(self):
        """Test: Resultado None retorna N/A."""
        winner = DataParser.determine_winner_from_result(None, 'Nadal', 'Federer')
        assert winner == 'N/A'
    
    def test_determine_winner_empty_result(self):
        """Test: Resultado vacío retorna N/A."""
        winner = DataParser.determine_winner_from_result('', 'Nadal', 'Federer')
        assert winner == 'N/A'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: extract_winner_sets
    # ═══════════════════════════════════════════════════════════════
    
    def test_extract_winner_sets_valid(self):
        """Test: Extraer sets del ganador."""
        assert DataParser.extract_winner_sets('2-0') == 2
        assert DataParser.extract_winner_sets('1-2') == 2
        assert DataParser.extract_winner_sets('3-1') == 3
    
    def test_extract_winner_sets_invalid(self):
        """Test: Formato inválido retorna N/A."""
        assert DataParser.extract_winner_sets('ABC') == 'N/A'
        assert DataParser.extract_winner_sets(None) == 'N/A'
        assert DataParser.extract_winner_sets('') == 'N/A'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: clean_player_name
    # ═══════════════════════════════════════════════════════════════
    
    def test_clean_player_name_normal(self):
        """Test: Limpiar nombre normal."""
        assert DataParser.clean_player_name('Rafael Nadal') == 'Rafael Nadal'
    
    def test_clean_player_name_extra_spaces(self):
        """Test: Remover espacios extras."""
        assert DataParser.clean_player_name('  Rafael   Nadal  ') == 'Rafael Nadal'
    
    def test_clean_player_name_empty(self):
        """Test: Nombre vacío retorna N/A."""
        assert DataParser.clean_player_name('') == 'N/A'
    
    def test_clean_player_name_none(self):
        """Test: None retorna N/A."""
        assert DataParser.clean_player_name(None) == 'N/A'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: extract_location_from_title
    # ═══════════════════════════════════════════════════════════════
    
    def test_extract_location_full_info(self):
        """Test: Extraer ciudad, país y superficie."""
        title = "Toronto (Canada), hard"
        
        result = DataParser.extract_location_from_title(title)
        
        assert result['ciudad'] == 'Toronto'
        assert result['pais'] == 'Canada'
        assert result['superficie'] == 'Dura'
    
    def test_extract_location_no_country(self):
        """Test: Ubicación sin país en paréntesis."""
        title = "Paris, clay"
        
        result = DataParser.extract_location_from_title(title)
        
        assert result['ciudad'] == 'Paris'
        assert result['pais'] == 'N/A'
        assert result['superficie'] == 'Arcilla'
    
    def test_extract_location_none(self):
        """Test: title None retorna valores por defecto."""
        result = DataParser.extract_location_from_title(None)
        
        assert result['ciudad'] == 'N/A'
        assert result['pais'] == 'N/A'
        assert result['superficie'] == 'N/A'
    
    def test_extract_location_empty(self):
        """Test: title vacío retorna valores por defecto."""
        result = DataParser.extract_location_from_title('')
        
        assert result['ciudad'] == 'N/A'
        assert result['pais'] == 'N/A'
        assert result['superficie'] == 'N/A'
    
    # ═══════════════════════════════════════════════════════════════
    # TESTS: parse_match_date
    # ═══════════════════════════════════════════════════════════════
    
    def test_parse_match_date_valid(self):
        """Test: Parsear fecha válida."""
        date_str = "24.10.24"
        result = DataParser.parse_match_date(date_str, '%d.%m.%y')
        
        assert isinstance(result, datetime)
        assert result.day == 24
        assert result.month == 10
        assert result.year == 2024
    
    def test_parse_match_date_invalid_format(self):
        """Test: Formato inválido retorna None."""
        result = DataParser.parse_match_date("2024-10-24", '%d.%m.%y')
        assert result is None
    
    def test_parse_match_date_none(self):
        """Test: None retorna None."""
        result = DataParser.parse_match_date(None)
        assert result is None
    
    def test_parse_match_date_empty(self):
        """Test: String vacío retorna None."""
        result = DataParser.parse_match_date('')
        assert result is None

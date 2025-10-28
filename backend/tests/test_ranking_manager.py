import pytest
import json
from unittest.mock import patch
from analysis.ranking_manager import RankingManager

# --- Datos de ejemplo mejorados ---
ATP_DATA_COMPLETE = {
    "rankings": [
        {"name": "Carlos Alcaraz", "ranking_position": 1, "PTS": 9000, "nationality": "Spain", "prox_points": 9200},
        {"name": "Novak Djokovic", "ranking_position": 2, "PTS": 8500, "nationality": "Serbia"},
        {"name": "Felix Auger-Aliassime", "ranking_position": 9, "PTS": 3000, "nationality": "Canada"},
        {"name": "Cerundolo F.", "ranking_position": 20, "PTS": 1800, "nationality": "Argentina"}
    ]
}
WTA_DATA_COMPLETE = {"rankings": [{"name": "Iga Swiatek", "ranking_position": 1, "PTS": 10000, "nationality": "Poland"}]}
ATP_DATA_BASIC = [{"name": "Casper Ruud", "rank": 4, "points": 5000, "nationality": "Norway"}]

# --- Fixtures ---
@pytest.fixture
def mock_files(tmp_path):
    """Crea un ecosistema de archivos de ranking falsos en un directorio temporal."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    files = {
        "atp_complete": data_dir / "atp_rankings_complete_mock.json",
        "wta_complete": data_dir / "wta_rankings_complete_mock.json",
        "atp_basic": data_dir / "atp_rankings_basic_mock.json"
    }
    
    with open(files["atp_complete"], 'w') as f: json.dump(ATP_DATA_COMPLETE, f)
    with open(files["wta_complete"], 'w') as f: json.dump(WTA_DATA_COMPLETE, f)
    with open(files["atp_basic"], 'w') as f: json.dump(ATP_DATA_BASIC, f)
        
    return files

# --- Tests ---
@patch('analysis.ranking_manager.CompleteRankingScraper')
def test_load_rankings_complete(MockScraper, mock_files):
    """Verifica la carga exitosa cuando los archivos completos están presentes."""
    def glob_side_effect(pattern):
        if 'atp_rankings_complete' in pattern: return [mock_files["atp_complete"]]
        if 'wta_rankings_complete' in pattern: return [mock_files["wta_complete"]]
        return []
    
    with patch('pathlib.Path.glob', side_effect=glob_side_effect):
        manager = RankingManager(force_read_only=True)
        assert manager.metadata['atp_count'] == 4
        assert manager.metadata['wta_count'] == 1
        assert "carlos alcaraz" in manager.atp_players

@patch('analysis.ranking_manager.CompleteRankingScraper')
def test_load_rankings_fallback(MockScraper, mock_files):
    """Verifica que se cargan los archivos básicos si los completos no existen."""
    def glob_side_effect(pattern):
        if 'atp_rankings_complete' in pattern: return [] # No se encuentran completos
        if 'wta_rankings_complete' in pattern: return []
        if 'atp_rankings' in pattern: return [mock_files["atp_basic"]] # Se encuentra el básico
        return []

    with patch('pathlib.Path.glob', side_effect=glob_side_effect):
        manager = RankingManager(force_read_only=True)
        assert manager.metadata['atp_count'] == 1
        assert "casper ruud" in manager.atp_players
        assert "carlos alcaraz" not in manager.atp_players # Asegurarse de que no cargó el completo

@patch('pathlib.Path.glob', return_value=[])
def test_normalize_name(mock_glob):
    """Verifica la lógica de normalización de nombres."""
    manager = RankingManager(force_read_only=True)
    assert manager.normalize_name("Félix Auger-Aliassime") == "felix auger aliassime"
    assert manager.normalize_name("Cerundolo F.") == "cerundolo f"

@patch('analysis.ranking_manager.CompleteRankingScraper')
def test_get_player_info_and_ranking(MockScraper, mock_files):
    """Verifica la búsqueda de jugadores y la obtención de su ranking."""
    def glob_side_effect(pattern):
        if 'atp_rankings_complete' in pattern: return [mock_files["atp_complete"]]
        return []
    
    with patch('pathlib.Path.glob', side_effect=glob_side_effect):
        manager = RankingManager(force_read_only=True)
        # Búsqueda por nombre completo
        assert manager.get_player_ranking("Carlos Alcaraz") == 1
        # Búsqueda por formato Apellido Inicial
        info = manager.get_player_info("Cerundolo F.")
        assert info is not None
        assert info['ranking_position'] == 20
        # Jugador no existente
        assert manager.get_player_ranking("Unknown Player") is None

@patch('analysis.ranking_manager.CompleteRankingScraper')
def test_get_ranking_metrics(MockScraper, mock_files):
    """Verifica que se extraen correctamente las métricas avanzadas."""
    def glob_side_effect(pattern):
        if 'atp_rankings_complete' in pattern: return [mock_files["atp_complete"]]
        return []

    with patch('pathlib.Path.glob', side_effect=glob_side_effect):
        manager = RankingManager(force_read_only=True)
        metrics = manager.get_ranking_metrics("Carlos Alcaraz")
        assert metrics is not None
        assert metrics['ranking_position'] == 1
        assert metrics['PTS'] == 9000
        assert metrics['PROX'] == 9200
        # Jugador no existente
        assert manager.get_ranking_metrics("Unknown Player") is None

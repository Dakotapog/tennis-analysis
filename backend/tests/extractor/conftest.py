"""
🧪 Fixtures compartidas para tests del extractor package
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json


@pytest.fixture
def sample_match_data():
    """Datos de ejemplo de un partido."""
    return {
        'jugador1': 'Rafael Nadal',
        'jugador2': 'Roger Federer',
        'cuota1': '1.85',
        'cuota2': '2.10',
        'match_url': 'https://www.flashscore.com/match/test123',
        'torneo_nombre': 'Roland Garros',
        'tipo_cancha': 'Arcilla',
        'pais': 'Francia',
        'torneo_completo': 'ATP: Roland Garros (Francia), clay'
    }


@pytest.fixture
def sample_player_history():
    """Historial de ejemplo de un jugador."""
    return [
        {
            'fecha': '24.10.24',
            'oponente': 'Novak Djokovic',
            'resultado': '2-0',
            'outcome': 'Ganó',
            'torneo': 'Shanghai Masters',
            'ciudad': 'Shanghai',
            'pais': 'China',
            'superficie': 'Dura'
        },
        {
            'fecha': '20.10.24',
            'oponente': 'Carlos Alcaraz',
            'resultado': '1-2',
            'outcome': 'Perdió',
            'torneo': 'Paris Masters',
            'ciudad': 'Paris',
            'pais': 'Francia',
            'superficie': 'Indoor'
        }
    ]


@pytest.fixture
def sample_h2h_matches():
    """Enfrentamientos directos de ejemplo."""
    return [
        {
            'fecha': '15.09.24',
            'jugador1': 'Rafael Nadal',
            'jugador2': 'Roger Federer',
            'resultado': '2-1',
            'ganador': 'Rafael Nadal',
            'torneo': 'US Open',
            'ciudad': 'New York',
            'pais': 'USA',
            'superficie': 'Dura',
            'ganador_sets': 2
        }
    ]


@pytest.fixture
def sample_json_file(tmp_path):
    """Crear un archivo JSON temporal para tests."""
    json_file = tmp_path / "test_matches.json"
    data = {
        'ATP: Test Tournament (Spain), clay': [
            {
                'jugador1': 'Player A',
                'jugador2': 'Player B',
                'cuota1': '1.50',
                'cuota2': '2.80',
                'match_url': 'https://flashscore.com/test'
            }
        ]
    }
    json_file.write_text(json.dumps(data), encoding='utf-8')
    return json_file


@pytest.fixture
def mock_playwright_page():
    """Mock de página de Playwright."""
    page = AsyncMock()
    
    # Mock de locator
    mock_locator = AsyncMock()
    mock_locator.count = AsyncMock(return_value=1)
    mock_locator.is_visible = AsyncMock(return_value=True)
    mock_locator.inner_text = AsyncMock(return_value="Test Text")
    mock_locator.get_attribute = AsyncMock(return_value="test-class")
    mock_locator.first = mock_locator
    mock_locator.all = AsyncMock(return_value=[mock_locator])
    
    page.locator = MagicMock(return_value=mock_locator)
    page.goto = AsyncMock()
    page.screenshot = AsyncMock()
    page.title = AsyncMock(return_value="Test Page")
    page.url = "https://test.com"
    page.content = AsyncMock(return_value="<html></html>")
    
    return page


@pytest.fixture
def mock_browser_manager():
    """Mock de BrowserManager."""
    manager = MagicMock()
    manager.setup = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.page = AsyncMock()
    return manager


@pytest.fixture
def mock_ranking_manager():
    """Mock de RankingManager."""
    manager = MagicMock()
    manager.get_player_ranking = MagicMock(return_value=5)
    manager.get_player_info = MagicMock(return_value={
        'name': 'Test Player',
        'points': 5000,
        'nationality': 'ESP'
    })
    manager.normalize_name = MagicMock(side_effect=lambda x: x.lower().replace(' ', '_'))
    return manager


@pytest.fixture
def mock_elo_system():
    """Mock de EloRatingSystem."""
    elo = MagicMock()
    elo.default_rating = 1500
    elo.calculate_expected_score = MagicMock(return_value=0.5)
    elo.update_rating = MagicMock(return_value=1520)
    return elo


@pytest.fixture
def mock_rivalry_analyzer():
    """Mock de RivalryAnalyzer."""
    analyzer = MagicMock()
    analyzer.calculate_base_opponent_weight = MagicMock(return_value=1.0)
    analyzer.calculate_elo_from_history = MagicMock(return_value=1600)
    analyzer.get_ranking_metrics = MagicMock(return_value={
        'pts': 5000,
        'prox_pts': 4800,
        'pts_max': 5200,
        'improvement_potential': 'Alta'
    })
    analyzer.analyze_rivalry = MagicMock(return_value={
        'player1_rank': 5,
        'player2_rank': 3,
        'player1_nationality': 'ESP',
        'player2_nationality': 'SUI',
        'common_opponents_count': 10,
        'p1_rivalry_score': 65.5,
        'p2_rivalry_score': 55.2,
        'prediction': {
            'favored_player': 'Rafael Nadal',
            'confidence': 65,
            'scores': {'p1_final_weight': 65.5, 'p2_final_weight': 55.2}
        },
        'p1_surface_stats': {},
        'p2_surface_stats': {},
        'p1_location_stats': {},
        'p2_location_stats': {},
        'common_opponents_detailed': []
    })
    return analyzer
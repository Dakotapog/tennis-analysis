"""
conftest.py - Fixtures y Configuración Global para Tests
Tennis Analysis Project
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Agregar directorio backend al path para imports
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))


# ============================================
# FIXTURES GLOBALES DE DATOS
# ============================================

@pytest.fixture
def sample_atp_rankings():
    """Rankings ATP de ejemplo"""
    return {
        "rankings": [
            {
                "name": "Carlos Alcaraz",
                "rank": 1,
                "ranking_position": 1,
                "points": 9000,
                "ranking_points": 9000,
                "PTS": 9000,
                "PROX": 9200,
                "MAX": 9500,
                "Potencial": 500,
                "defense_points": 1200,
                "nationality": "Spain",
                "tour": "ATP"
            },
            {
                "name": "Jannik Sinner",
                "rank": 2,
                "ranking_position": 2,
                "points": 8500,
                "ranking_points": 8500,
                "PTS": 8500,
                "PROX": 8600,
                "MAX": 8900,
                "Potencial": 400,
                "defense_points": 1000,
                "nationality": "Italy",
                "tour": "ATP"
            },
            {
                "name": "Novak Djokovic",
                "rank": 3,
                "ranking_position": 3,
                "points": 8200,
                "ranking_points": 8200,
                "PTS": 8200,
                "PROX": 8300,
                "MAX": 8700,
                "Potencial": 500,
                "defense_points": 1500,
                "nationality": "Serbia",
                "tour": "ATP"
            }
        ]
    }


@pytest.fixture
def sample_wta_rankings():
    """Rankings WTA de ejemplo"""
    return {
        "rankings": [
            {
                "name": "Iga Swiatek",
                "rank": 1,
                "ranking_position": 1,
                "points": 9500,
                "ranking_points": 9500,
                "PTS": 9500,
                "PROX": 9700,
                "MAX": 10000,
                "Potencial": 500,
                "defense_points": 1300,
                "nationality": "Poland",
                "tour": "WTA"
            },
            {
                "name": "Aryna Sabalenka",
                "rank": 2,
                "ranking_position": 2,
                "points": 8800,
                "ranking_points": 8800,
                "PTS": 8800,
                "PROX": 9000,
                "MAX": 9300,
                "Potencial": 500,
                "defense_points": 1100,
                "nationality": "Belarus",
                "tour": "WTA"
            }
        ]
    }


@pytest.fixture
def sample_match_complete():
    """Partido completo de ejemplo con toda la información"""
    return {
        "jugador1": "Carlos Alcaraz",
        "jugador2": "Jannik Sinner",
        "resultado": "2-1 (6-4, 3-6, 7-6)",
        "fecha": "15.10.24",
        "superficie": "Dura",
        "torneo": "ATP Finals",
        "ronda": "Semifinal",
        "pais": "Italy",
        "match_url": "https://www.flashscore.com/match/abc123",
        "jugador1_ranking": 1,
        "jugador2_ranking": 2,
        "duracion": "3h 24min"
    }


@pytest.fixture
def sample_player_history():
    """Historial completo de partidos de un jugador"""
    today = datetime.now()
    
    return [
        {
            "oponente": "Novak Djokovic",
            "resultado": "2-0 (6-3, 7-6)",
            "outcome": "ganó",
            "fecha": (today - timedelta(days=5)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "ATP Masters",
            "opponent_ranking": 3,
            "opponent_nationality": "Serbia"
        },
        {
            "oponente": "Daniil Medvedev",
            "resultado": "1-2 (7-6, 4-6, 5-7)",
            "outcome": "perdió",
            "fecha": (today - timedelta(days=12)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "ATP 500",
            "opponent_ranking": 5,
            "opponent_nationality": "Russia"
        },
        {
            "oponente": "Alexander Zverev",
            "resultado": "2-0 (6-2, 6-4)",
            "outcome": "ganó",
            "fecha": (today - timedelta(days=20)).strftime('%d.%m.%y'),
            "superficie": "Arcilla",
            "torneo": "ATP Masters",
            "opponent_ranking": 7,
            "opponent_nationality": "Germany"
        },
        {
            "oponente": "Stefanos Tsitsipas",
            "resultado": "2-1 (4-6, 6-3, 6-4)",
            "outcome": "ganó",
            "fecha": (today - timedelta(days=35)).strftime('%d.%m.%y'),
            "superficie": "Arcilla",
            "torneo": "ATP 1000",
            "opponent_ranking": 8,
            "opponent_nationality": "Greece"
        },
        {
            "oponente": "Andrey Rublev",
            "resultado": "2-0 (6-4, 6-3)",
            "outcome": "ganó",
            "fecha": (today - timedelta(days=50)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "ATP 500",
            "opponent_ranking": 6,
            "opponent_nationality": "Russia"
        }
    ]


@pytest.fixture
def sample_h2h_matches():
    """Enfrentamientos directos H2H de ejemplo"""
    today = datetime.now()
    
    return [
        {
            "ganador": "Carlos Alcaraz",
            "perdedor": "Jannik Sinner",
            "resultado": "2-1 (6-4, 3-6, 7-5)",
            "fecha": (today - timedelta(days=30)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "ATP Finals"
        },
        {
            "ganador": "Jannik Sinner",
            "perdedor": "Carlos Alcaraz",
            "resultado": "2-0 (7-6, 6-4)",
            "fecha": (today - timedelta(days=90)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "US Open"
        },
        {
            "ganador": "Carlos Alcaraz",
            "perdedor": "Jannik Sinner",
            "resultado": "2-0 (6-3, 6-2)",
            "fecha": (today - timedelta(days=180)).strftime('%d.%m.%y'),
            "superficie": "Arcilla",
            "torneo": "Roland Garros"
        },
        {
            "ganador": "Carlos Alcaraz",
            "perdedor": "Jannik Sinner",
            "resultado": "2-1 (6-7, 6-4, 7-6)",
            "fecha": (today - timedelta(days=300)).strftime('%d.%m.%y'),
            "superficie": "Dura",
            "torneo": "ATP 1000"
        }
    ]


# ============================================
# FIXTURES DE ARCHIVOS TEMPORALES
# ============================================

@pytest.fixture
def temp_ranking_file(tmp_path, sample_atp_rankings):
    """Crear archivo temporal de rankings ATP"""
    file_path = tmp_path / "atp_rankings_complete_20250113.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_atp_rankings, f, indent=2)
    return file_path


@pytest.fixture
def temp_matches_file(tmp_path):
    """Crear archivo temporal de partidos"""
    matches_data = {
        "ATP Finals": [
            {
                "jugador1": "Carlos Alcaraz",
                "jugador2": "Jannik Sinner",
                "resultado": "2-1",
                "fecha": "15.10.24",
                "match_url": "https://example.com/match1"
            }
        ]
    }
    
    file_path = tmp_path / "test_matches.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(matches_data, f, indent=2)
    return file_path


@pytest.fixture
def temp_data_directory(tmp_path, sample_atp_rankings, sample_wta_rankings):
    """Crear directorio temporal data/ con archivos de rankings"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Crear archivo ATP
    atp_file = data_dir / "atp_rankings_complete_20250113.json"
    with open(atp_file, 'w', encoding='utf-8') as f:
        json.dump(sample_atp_rankings, f)
    
    # Crear archivo WTA
    wta_file = data_dir / "wta_rankings_complete_20250113.json"
    with open(wta_file, 'w', encoding='utf-8') as f:
        json.dump(sample_wta_rankings, f)
    
    return data_dir


# ============================================
# FIXTURES DE MOCKS
# ============================================

@pytest.fixture
def mock_ranking_manager():
    """Mock de RankingManager con funcionalidad básica"""
    mock = Mock()
    
    # Configurar métodos comunes
    mock.normalize_name.side_effect = lambda x: x.lower().strip() if x else ""
    mock.get_player_ranking.return_value = 10
    mock.get_player_info.return_value = {
        'name': 'Test Player',
        'ranking_position': 10,
        'ranking_points': 5000,
        'nationality': 'Spain',
        'tour': 'ATP'
    }
    
    return mock


@pytest.fixture
def mock_elo_system():
    """Mock de EloRatingSystem"""
    mock = Mock()
    
    mock.default_rating = 1500
    mock.k_factor = 32
    mock.ratings = {}
    
    mock.get_rating.return_value = 1500
    mock.expected_score.return_value = 0.5
    mock.update_ratings.return_value = None
    
    return mock


@pytest.fixture
def mock_playwright_browser():
    """Mock de Playwright browser para tests de scraping"""
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_context = MagicMock()
    
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page
    mock_page.goto.return_value = None
    mock_page.wait_for_load_state.return_value = None
    
    return mock_browser


# ============================================
# FIXTURES DE CONFIGURACIÓN
# ============================================

@pytest.fixture(scope="session")
def test_config():
    """Configuración global para tests"""
    return {
        'max_test_duration': 30,  # segundos
        'temp_file_cleanup': True,
        'verbose_logging': False,
        'mock_external_calls': True
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Resetear singletons entre tests para evitar state leaking"""
    yield
    # Cleanup después de cada test
    # Agregar aquí cualquier limpieza necesaria


@pytest.fixture
def caplog_info(caplog):
    """Fixture para capturar logs de nivel INFO"""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# ============================================
# FIXTURES DE CONTEXTO DE PARTIDO
# ============================================

@pytest.fixture
def match_context_clay():
    """Contexto de partido en arcilla"""
    return {
        'surface': 'Arcilla',
        'country': 'Spain',
        'tournament': 'Roland Garros',
        'round': 'Semifinal',
        'date': '15.06.24'
    }


@pytest.fixture
def match_context_hard():
    """Contexto de partido en cancha dura"""
    return {
        'surface': 'Dura',
        'country': 'USA',
        'tournament': 'US Open',
        'round': 'Final',
        'date': '10.09.24'
    }


# ============================================
# FIXTURES DE FORM/MOMENTUM
# ============================================

@pytest.fixture
def player_form_good():
    """Forma/momentum bueno de un jugador"""
    return {
        'win_percentage': 75.0,
        'recent_wins': 6,
        'recent_losses': 2,
        'last_10_matches': 8,
        'streak': 'W-W-W-L-W-W',
        'momentum': 'positive'
    }


@pytest.fixture
def player_form_poor():
    """Forma/momentum pobre de un jugador"""
    return {
        'win_percentage': 40.0,
        'recent_wins': 2,
        'recent_losses': 3,
        'last_10_matches': 4,
        'streak': 'L-L-W-L-W',
        'momentum': 'negative'
    }


# ============================================
# HOOKS DE PYTEST
# ============================================

def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca tests que tardan más de 1 segundo"
    )
    config.addinivalue_line(
        "markers", "integration: tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: tests unitarios"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar items de tests recolectados"""
    for item in items:
        # Agregar marcador 'unit' por defecto si no tiene otros marcadores
        if not any(mark.name in ['integration', 'slow', 'performance'] 
                   for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="function", autouse=True)
def cleanup_temp_files(request, tmp_path):
    """Limpiar archivos temporales después de cada test"""
    yield
    
    # Cleanup después del test
    if tmp_path.exists():
        import shutil
        try:
            shutil.rmtree(tmp_path)
        except Exception:
            pass  # Ignorar errores de cleanup


# ============================================
# UTILIDADES PARA TESTS
# ============================================

@pytest.fixture
def assert_approximately():
    """Helper para comparaciones aproximadas"""
    def _assert_approximately(actual, expected, tolerance=0.1):
        """
        Assert que actual está dentro del tolerance% de expected
        """
        lower = expected * (1 - tolerance)
        upper = expected * (1 + tolerance)
        assert lower <= actual <= upper, \
            f"{actual} no está dentro del {tolerance*100}% de {expected} ({lower}-{upper})"
    
    return _assert_approximately


@pytest.fixture
def create_mock_match():
    """Factory para crear partidos mock"""
    def _create(winner, loser, surface="Dura", score="2-0", date=None):
        if date is None:
            date = datetime.now().strftime('%d.%m.%y')
        
        return {
            'ganador': winner,
            'perdedor': loser,
            'superficie': surface,
            'resultado': score,
            'fecha': date,
            'match_url': f'https://example.com/{winner}-vs-{loser}'
        }
    
    return _create
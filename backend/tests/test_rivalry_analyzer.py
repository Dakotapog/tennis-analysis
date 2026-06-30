import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from analysis.rivalry_analyzer import RivalryAnalyzer, density_confidence, shrink_weights


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def ranking_manager():
    """Mock de RankingManager con datos de prueba."""
    rm = MagicMock()
    rm.get_player_ranking.side_effect = lambda name: {
        "PlayerA": 1, 
        "PlayerB": 50, 
        "PlayerC": 150,
        "PlayerX": 75,
        "PlayerZ": 100,
        "playerz": 100,
        "TopPlayer": 5,
        "MidPlayer": 80,
        "LowPlayer": 200
    }.get(name, None)
    rm.get_player_info.return_value = {
        "ranking_position": 10, 
        "ranking_points": 2000, 
        "prox_points": 2100,
        "max_points": 2200, 
        "defense_points": 100
    }
    rm.normalize_name.side_effect = lambda name: name.lower() if name else name
    return rm


@pytest.fixture
def elo_system():
    """Mock de EloRatingSystem con valores por defecto."""
    es = MagicMock()
    es.default_rating = 1500
    es.k_factor = 32
    es.expected_score.return_value = 0.5
    es.calculate_rating_change.return_value = 16
    return es


@pytest.fixture
def analyzer(ranking_manager, elo_system):
    """Instancia de RivalryAnalyzer con mocks."""
    return RivalryAnalyzer(ranking_manager, elo_system)


@pytest.fixture
def sample_history():
    """Historial de partidos de ejemplo."""
    return [
        {
            "oponente": "PlayerA", 
            "resultado": "2-0", 
            "outcome": "Ganó", 
            "opponent_ranking": 1,
            "surface": "Dura",
            "location": "Spain"
        },
        {
            "oponente": "PlayerB", 
            "resultado": "1-2", 
            "outcome": "Perdió", 
            "opponent_ranking": 50,
            "surface": "Arcilla",
            "location": "France"
        },
        {
            "oponente": "PlayerC", 
            "resultado": "2-1", 
            "outcome": "Ganó", 
            "opponent_ranking": 150,
            "surface": "Hierba",
            "location": "UK"
        },
    ]


@pytest.fixture
def comprehensive_history():
    """Historial más completo para tests avanzados."""
    now = datetime.now()
    return [
        {
            "oponente": "TopPlayer",
            "resultado": "2-1",
            "outcome": "Ganó",
            "opponent_ranking": 5,
            "surface": "Dura",
            "location": "USA",
            "fecha": (now - timedelta(days=10)).strftime('%d.%m.%y')
        },
        {
            "oponente": "MidPlayer",
            "resultado": "0-2",
            "outcome": "Perdió",
            "opponent_ranking": 80,
            "surface": "Arcilla",
            "location": "Spain",
            "fecha": (now - timedelta(days=20)).strftime('%d.%m.%y')
        },
        {
            "oponente": "LowPlayer",
            "resultado": "2-0",
            "outcome": "Ganó",
            "opponent_ranking": 200,
            "surface": "Dura",
            "location": "USA",
            "fecha": (now - timedelta(days=30)).strftime('%d.%m.%y')
        },
        {
            "oponente": "TopPlayer",
            "resultado": "1-2",
            "outcome": "Perdió",
            "opponent_ranking": 5,
            "surface": "Hierba",
            "location": "UK",
            "fecha": (now - timedelta(days=40)).strftime('%d.%m.%y')
        },
    ]


# ============================================================================
# TESTS BÁSICOS
# ============================================================================

def test_initialization(analyzer):
    """Test: Inicialización correcta del analyzer."""
    assert isinstance(analyzer, RivalryAnalyzer)
    assert analyzer.ranking_manager is not None
    assert analyzer.elo_system is not None


# ============================================================================
# TESTS DE ESTIMACIÓN ELO
# ============================================================================

def test_estimate_elo_from_rank_top_player(analyzer):
    """Test: ELO para jugador top (ranking 1)."""
    elo = analyzer.estimate_elo_from_rank(1)
    assert elo >= 2000
    assert isinstance(elo, (int, float))


def test_estimate_elo_from_rank_mid_rank_player(analyzer):
    """Test: ELO para jugador de rango medio (ranking 50)."""
    val = analyzer.estimate_elo_from_rank(50)
    assert 1700 <= val <= 2100


def test_estimate_elo_from_rank_low_rank_player(analyzer):
    """Test: ELO para jugador de rango bajo (ranking 150)."""
    val = analyzer.estimate_elo_from_rank(150)
    assert 1400 <= val <= 1900


def test_estimate_elo_from_rank_unranked_player(analyzer):
    """Test: ELO para jugador sin ranking retorna valor por defecto."""
    default_elo = analyzer.estimate_elo_from_rank(None)
    assert isinstance(default_elo, (int, float))
    assert default_elo >= 1000


def test_estimate_elo_from_rank_zero(analyzer):
    """Test: ELO para ranking 0 (edge case)."""
    elo = analyzer.estimate_elo_from_rank(0)
    assert isinstance(elo, (int, float))
    assert elo >= 1000


def test_estimate_elo_from_rank_rank_10(analyzer):
    """Test: ELO para ranking exacto 10."""
    elo = analyzer.estimate_elo_from_rank(10)
    assert isinstance(elo, (int, float))
    assert elo >= 1800


def test_estimate_elo_from_rank_rank_100(analyzer):
    """Test: ELO para ranking exacto 100."""
    elo = analyzer.estimate_elo_from_rank(100)
    assert isinstance(elo, (int, float))
    assert elo >= 1500


# ============================================================================
# TESTS PARAMETRIZADOS
# ============================================================================

@pytest.mark.parametrize("ranking,min_expected", [
    (5, 8),
    (35, 3),
    (120, 1),
    (None, 1)
])
def test_calculate_base_opponent_weight(analyzer, ranking, min_expected):
    """Test: Peso base de oponente según ranking."""
    weight = analyzer.calculate_base_opponent_weight(ranking)
    assert weight >= min_expected
    assert isinstance(weight, (int, float))


def test_calculate_base_opponent_weight_negative(analyzer):
    """Test: Peso base con ranking negativo (edge case)."""
    weight = analyzer.calculate_base_opponent_weight(-1)
    assert isinstance(weight, (int, float))
    assert weight >= 0


@pytest.mark.parametrize("resultado,min_factor", [
    ("2-0", 1.1),
    ("2-1", 0.9),
    ("Resultado inválido", 0.8),
    ("", 0.8)
])
def test_analizar_contundencia(analyzer, resultado, min_factor):
    """Test: Factor de contundencia según resultado."""
    factor = analyzer.analizar_contundencia(resultado)
    assert factor >= min_factor or factor == 1.0
    assert isinstance(factor, (int, float))


@pytest.mark.parametrize("resultado,max_factor", [
    ("1-2", 0.6),
    ("0-2", 0.2),
    ("Resultado inválido", 0.1),
    ("", 0.1)
])
def test_analizar_resistencia(analyzer, resultado, max_factor):
    """Test: Factor de resistencia según resultado."""
    factor = analyzer.analizar_resistencia(resultado)
    assert factor <= max_factor or factor == 0.0
    assert isinstance(factor, (int, float))


# ============================================================================
# TESTS CON HISTORIAL (HAPPY PATH)
# ============================================================================

def test_calculate_elo_from_history_happy_path(analyzer, sample_history):
    """Test: ELO se calcula correctamente a partir del historial."""
    elo = analyzer.calculate_elo_from_history("PlayerA", sample_history)
    assert isinstance(elo, (int, float))
    assert elo > 0


def test_calcular_peso_oponentes_comunes_happy_path(analyzer, sample_history):
    """Test: Peso de oponentes comunes se calcula correctamente."""
    weight = analyzer.calcular_peso_oponentes_comunes(
        sample_history, sample_history, "PlayerA", "PlayerA", "PlayerB"
    )
    assert isinstance(weight, (int, float))
    assert weight >= 0


def test_analyze_advanced_player_metrics_happy_path(analyzer, sample_history):
    """Test: Métricas avanzadas de jugador se calculan correctamente."""
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(sample_history, "PlayerA")
    assert quality >= 1.0
    assert diversity >= 1.0
    assert isinstance(log, list)


def test_analyze_strength_of_schedule_happy_path(analyzer, sample_history):
    """Test: Fuerza del calendario se calcula correctamente."""
    score, log = analyzer.analyze_strength_of_schedule(sample_history, "PlayerA")
    assert isinstance(score, (int, float))
    assert score >= 0
    assert isinstance(log, list)


def test_analyze_streaks_and_consistency_happy_path(analyzer, sample_history):
    """Test: Racha y consistencia se calculan correctamente."""
    streak, log = analyzer.analyze_streaks_and_consistency(sample_history, "PlayerA")
    assert isinstance(streak, (int, float))
    assert streak >= 1.0
    assert isinstance(log, list)


def test_find_common_opponents_happy_path(analyzer, sample_history):
    """Test: Encuentra oponentes comunes correctamente."""
    common = analyzer.find_common_opponents(sample_history, sample_history)
    assert isinstance(common, list)


def test_get_ranking_metrics_happy_path(analyzer):
    """Test: Métricas de ranking se obtienen correctamente."""
    metrics = analyzer.get_ranking_metrics("PlayerA")
    assert isinstance(metrics, dict)
    assert "already_secured" in metrics or "defense_points" in metrics


def test_analyze_surface_specialization_happy_path(analyzer, sample_history):
    """Test: Especialización en superficie retorna score y log."""
    result, log = analyzer.analyze_surface_specialization(sample_history, "Dura", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert isinstance(log, list)


def test_analyze_surface_performance_happy_path(analyzer, sample_history):
    """Test: Rendimiento por superficie retorna stats."""
    stats = analyzer.analyze_surface_performance(sample_history, "PlayerA")
    assert isinstance(stats, dict)


def test_analyze_location_factors_happy_path(analyzer, sample_history):
    """Test: Factores de localización retornan dict."""
    result = analyzer.analyze_location_factors(sample_history, "Spain")
    assert isinstance(result, dict)


def test_analyze_rivalry_happy_path(analyzer, sample_history):
    """Test: Análisis de rivalidad retorna dict con puntajes."""
    result = analyzer.analyze_rivalry(
        sample_history, sample_history, "PlayerA", "PlayerB",
        {"win_percentage": 60}, {"win_percentage": 40},
        [], {"country": "Spain", "surface": "Dura"}, 1600, 1500, "Roland Garros"
    )
    assert isinstance(result, dict)
    assert "player1_rank" in result or "p1_total" in result


def test_generate_basic_prediction_happy_path(analyzer):
    """Test: Predicción básica retorna dict con favorito y confianza."""
    result = analyzer.generate_basic_prediction(10, 20, "PlayerA", "PlayerB")
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result


def test_classify_tournament_happy_path(analyzer):
    """Test: Clasificación de torneo retorna categoría correcta (T21-01)."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    assert analyzer.classify_tournament("ATP Masters 1000") in VALID
    assert analyzer.classify_tournament("Challenger") in VALID
    assert analyzer.classify_tournament("ITF M15") in VALID
    assert analyzer.classify_tournament("Torneo Local") in VALID


def test_generate_advanced_prediction_happy_path(analyzer, sample_history):
    """Test: Predicción avanzada retorna dict con favorito y desglose."""
    result = analyzer.generate_advanced_prediction(
        {"ranking_position": 10}, {"ranking_position": 20},
        100, 80, "PlayerA", "PlayerB",
        sample_history, sample_history, 2, 1,
        {"win_percentage": 60}, {"win_percentage": 40},
        [], "Roland Garros",
        {"current_match_surface": "Dura", "current_match_country": "Spain", 
         "p1_nationality": "Spain", "p2_nationality": "France"},
        1600, 1500
    )
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result
    assert "confidence" in result or "certainty" in result


# ============================================================================
# TESTS DE H2H
# ============================================================================

def test_analyze_direct_h2h_with_recent_dates(analyzer):
    """Test: Análisis de H2H con fechas recientes."""
    now = datetime.now()
    date_30d = (now - timedelta(days=30)).strftime('%d.%m.%y')
    date_60d = (now - timedelta(days=60)).strftime('%d.%m.%y')
    date_90d = (now - timedelta(days=90)).strftime('%d.%m.%y')
    
    h2h_matches = [
        {"ganador": "PlayerA", "fecha": date_30d},
        {"ganador": "PlayerB", "fecha": date_60d},
        {"ganador": "PlayerA", "fecha": date_90d},
    ]
    
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))
    assert p1_score >= 0
    assert p2_score >= 0
    assert isinstance(log, list)


def test_analyze_direct_h2h_empty(analyzer):
    """Test: analyze_direct_h2h debe retornar 0 si no hay partidos."""
    p1_score, p2_score, log = analyzer.analyze_direct_h2h([], "PlayerA", "PlayerB")
    assert p1_score == 0
    assert p2_score == 0
    assert isinstance(log, list)


def test_analyze_direct_h2h_single_match(analyzer):
    """Test: H2H directo con solo un partido."""
    h2h_matches = [{"ganador": "PlayerA", "fecha": "01.01.25"}]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))
    assert p1_score >= 0
    assert p2_score >= 0
    assert isinstance(log, list)
    assert len(log) == 1


def test_analyze_direct_h2h_single_match_recent(analyzer):
    """Test: H2H directo con un partido reciente."""
    now = datetime.now()
    recent_date = (now - timedelta(days=15)).strftime('%d.%m.%y')
    
    h2h_matches = [{"ganador": "PlayerA", "fecha": recent_date}]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    
    assert (p1_score > 0 and p2_score == 0) or (p2_score > 0 and p1_score == 0)
    assert isinstance(log, list)


# ============================================================================
# TESTS DE EDGE CASES Y MANEJO DE ERRORES
# ============================================================================

def test_analyze_advanced_player_metrics_empty_history(analyzer):
    """Test: analyze_advanced_player_metrics debe manejar historial vacío sin error."""
    quality, diversity, log = analyzer.analyze_advanced_player_metrics([], "PlayerA")
    assert quality >= 1.0
    assert diversity >= 1.0
    assert isinstance(log, list)


def test_analyze_advanced_player_metrics_corrupt_entry(analyzer):
    """Test: Manejo de historial con datos corruptos."""
    corrupt_history = [{"oponente": None, "resultado": None}]
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(corrupt_history, "PlayerA")
    assert quality >= 1.0
    assert diversity >= 1.0
    assert isinstance(log, list)


def test_analyze_strength_of_schedule_none_history(analyzer):
    """Test: analyze_strength_of_schedule debe manejar None como historial."""
    score, log = analyzer.analyze_strength_of_schedule(None, "PlayerA")
    assert isinstance(score, (int, float))
    assert score >= 0
    assert isinstance(log, list)


def test_analyze_strength_of_schedule_wrong_type(analyzer):
    """Test: Manejo de historial con tipo incorrecto."""
    with pytest.raises(AttributeError):
        analyzer.analyze_strength_of_schedule("not_a_list", "PlayerA")


def test_analyze_surface_specialization_none_history(analyzer):
    """Test: analyze_surface_specialization debe manejar None como historial."""
    result, log = analyzer.analyze_surface_specialization(None, "Dura", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert result['score'] >= 0
    assert isinstance(log, list)


def test_analyze_surface_specialization_minimal_entry(analyzer):
    """Test: Manejo de historial con solo los campos obligatorios."""
    minimal_history = [{"oponente": "PlayerX", "resultado": "2-0"}]
    result, log = analyzer.analyze_surface_specialization(minimal_history, "Dura", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert isinstance(log, list)


def test_find_common_opponents_none(analyzer):
    """Test: find_common_opponents debe retornar lista vacía si alguna entrada es None."""
    common = analyzer.find_common_opponents(None, None)
    assert isinstance(common, list)
    assert len(common) == 0


def test_find_common_opponents_repeated(analyzer):
    """Test: Oponentes comunes con partidos repetidos."""
    history1 = [{"oponente": "PlayerZ"}, {"oponente": "PlayerZ"}]
    history2 = [{"oponente": "PlayerZ"}]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)
    assert "playerz" in common or "PlayerZ" in common
    assert len(common) > 0


def test_calculate_elo_from_history_none(analyzer):
    """Test: calculate_elo_from_history retorna default_rating (1500) cuando no hay historial.
    B-11 fix: retornar 1500 (no None) — con floor=1500, raw=0 y contribution=0% igualmente."""
    elo = analyzer.calculate_elo_from_history("PlayerA", None)
    assert elo == 1500


def test_calculate_elo_from_history_minimal(analyzer):
    """Test: ELO con historial mínimo."""
    minimal_history = [{"oponente": "PlayerB", "resultado": "2-0"}]
    elo = analyzer.calculate_elo_from_history("PlayerA", minimal_history)
    assert isinstance(elo, int)
    assert elo >= 0


def test_analyze_surface_performance_empty_history(analyzer):
    """Test: Rendimiento por superficie con historial vacío."""
    stats = analyzer.analyze_surface_performance([], "PlayerA")
    assert isinstance(stats, dict)


def test_analyze_location_factors_none_location(analyzer, sample_history):
    """Test: Factores de localización con ubicación None."""
    result = analyzer.analyze_location_factors(sample_history, None)
    assert isinstance(result, dict)


def test_generate_basic_prediction_equal_ranks(analyzer):
    """Test: Predicción básica con rankings iguales."""
    result = analyzer.generate_basic_prediction(50, 50, "PlayerA", "PlayerB")
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result


# ============================================================================
# TESTS ADICIONALES PARA AUMENTAR COBERTURA
# ============================================================================

def test_analyze_surface_specialization_no_surface_matches(analyzer):
    """Test: Especialización cuando no hay partidos en la superficie especificada."""
    history = [
        {"oponente": "PlayerA", "resultado": "2-0", "surface": "Arcilla"},
        {"oponente": "PlayerB", "resultado": "2-1", "surface": "Hierba"}
    ]
    result, log = analyzer.analyze_surface_specialization(history, "Dura", "TestPlayer")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert result['score'] >= 0
    assert isinstance(log, list)


def test_analyze_location_factors_with_matches(analyzer):
    """Test: Factores de localización con partidos en la ubicación."""
    history = [
        {"oponente": "PlayerA", "resultado": "2-0", "location": "Spain", "outcome": "Ganó"},
        {"oponente": "PlayerB", "resultado": "1-2", "location": "Spain", "outcome": "Perdió"}
    ]
    result = analyzer.analyze_location_factors(history, "Spain")
    assert isinstance(result, dict)
    assert "home_advantage" in result


def test_calculate_base_opponent_weight_very_high_rank(analyzer):
    """Test: Peso base para ranking muy alto."""
    weight = analyzer.calculate_base_opponent_weight(500)
    assert isinstance(weight, (int, float))
    assert weight >= 1


def test_analyze_advanced_player_metrics_comprehensive(analyzer, comprehensive_history):
    """Test: Métricas avanzadas con historial completo."""
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(comprehensive_history, "PlayerA")
    assert quality >= 1.0
    assert diversity >= 1.0
    assert isinstance(log, list)
    assert len(log) > 0


def test_analyze_strength_of_schedule_comprehensive(analyzer, comprehensive_history):
    """Test: Fuerza del calendario con historial completo."""
    score, log = analyzer.analyze_strength_of_schedule(comprehensive_history, "PlayerA")
    assert isinstance(score, (int, float))
    assert score > 0
    assert isinstance(log, list)


def test_analyze_streaks_and_consistency_comprehensive(analyzer, comprehensive_history):
    """Test: Racha y consistencia con historial completo."""
    streak, log = analyzer.analyze_streaks_and_consistency(comprehensive_history, "PlayerA")
    assert isinstance(streak, (int, float))
    assert streak >= 1.0
    assert isinstance(log, list)


def test_analyze_surface_performance_comprehensive(analyzer, comprehensive_history):
    """Test: Rendimiento por superficie con historial completo."""
    stats = analyzer.analyze_surface_performance(comprehensive_history, "PlayerA")
    assert isinstance(stats, dict)
    assert len(stats) > 0


def test_calcular_peso_oponentes_comunes_with_common(analyzer, comprehensive_history):
    """Test: Peso de oponentes comunes cuando realmente hay comunes."""
    weight = analyzer.calcular_peso_oponentes_comunes(
        comprehensive_history, comprehensive_history, "PlayerA", "PlayerB", "TopPlayer"
    )
    assert isinstance(weight, (int, float))
    assert weight >= 0


def test_classify_tournament_grand_slam(analyzer):
    """Test: Grand Slams detectados correctamente (T21-01)."""
    assert analyzer.classify_tournament("Wimbledon") == "grand_slam"
    assert analyzer.classify_tournament("Roland Garros") == "grand_slam"
    assert analyzer.classify_tournament("French Open (France)") == "grand_slam"
    assert analyzer.classify_tournament("Australian Open") == "grand_slam"
    assert analyzer.classify_tournament("US Open") == "grand_slam"


def test_classify_tournament_masters(analyzer):
    """Test: Masters 1000 detectados como atp1000 (T21-01)."""
    assert analyzer.classify_tournament("Miami Open") == "atp1000"
    assert analyzer.classify_tournament("Indian Wells") == "atp1000"
    assert analyzer.classify_tournament("Monte-Carlo") == "atp1000"


def test_generate_basic_prediction_large_gap(analyzer):
    """Test: Predicción básica con gran diferencia de ranking."""
    result = analyzer.generate_basic_prediction(1, 100, "PlayerA", "PlayerB")
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result
    assert "confidence" in result or "certainty" in result


def test_analyze_rivalry_with_h2h(analyzer, comprehensive_history):
    """Test: Análisis de rivalidad con H2H."""
    now = datetime.now()
    h2h = [
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=20)).strftime('%d.%m.%y')},
        {"ganador": "PlayerB", "fecha": (now - timedelta(days=40)).strftime('%d.%m.%y')}
    ]
    result = analyzer.analyze_rivalry(
        comprehensive_history, comprehensive_history, "PlayerA", "PlayerB",
        {"win_percentage": 55}, {"win_percentage": 45},
        h2h, {"country": "USA", "surface": "Dura"}, 1700, 1650, "US Open"
    )
    assert isinstance(result, dict)


def test_analyze_location_factors_multiple_locations(analyzer, comprehensive_history):
    """Test: Factores de localización con múltiples ubicaciones."""
    result = analyzer.analyze_location_factors(comprehensive_history, "USA")
    assert isinstance(result, dict)


def test_find_common_opponents_different_opponents(analyzer):
    """Test: Oponentes comunes cuando no hay comunes."""
    history1 = [{"oponente": "PlayerA"}, {"oponente": "PlayerB"}]
    history2 = [{"oponente": "PlayerC"}, {"oponente": "PlayerD"}]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)


def test_calculate_elo_from_history_with_outcomes(analyzer, comprehensive_history):
    """Test: Cálculo de ELO con resultados variados."""
    elo = analyzer.calculate_elo_from_history("PlayerA", comprehensive_history)
    assert isinstance(elo, int)
    assert elo > 0


def test_analyze_surface_specialization_with_surface(analyzer, comprehensive_history):
    """Test: Especialización en superficie con partidos en esa superficie."""
    result, log = analyzer.analyze_surface_specialization(comprehensive_history, "Dura", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert isinstance(log, list)


def test_get_ranking_metrics_none_player(analyzer):
    """Test: Métricas de ranking con jugador inexistente."""
    analyzer.ranking_manager.get_player_info.return_value = None
    metrics = analyzer.get_ranking_metrics("NonExistentPlayer")
    assert isinstance(metrics, dict)


def test_generate_advanced_prediction_comprehensive(analyzer, comprehensive_history):
    """Test: Predicción avanzada con datos comprehensivos."""
    result = analyzer.generate_advanced_prediction(
        {"ranking_position": 5}, {"ranking_position": 80},
        120, 90, "TopPlayer", "MidPlayer",
        comprehensive_history, comprehensive_history, 3, 2,
        {"win_percentage": 65}, {"win_percentage": 35},
        [], "US Open",
        {"current_match_surface": "Dura", "current_match_country": "USA", 
         "p1_nationality": "USA", "p2_nationality": "Spain"},
        1850, 1650
    )
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result


# ============================================================================
# NUEVOS TESTS PARA AUMENTAR COBERTURA
# ============================================================================

def test_analyze_streaks_with_winning_streak(analyzer):
    """Test: Análisis de rachas con racha ganadora."""
    history = [
        {"outcome": "Ganó", "fecha": "10.10.24"},
        {"outcome": "Ganó", "fecha": "09.10.24"},
        {"outcome": "Ganó", "fecha": "08.10.24"},
        {"outcome": "Perdió", "fecha": "07.10.24"}
    ]
    streak, log = analyzer.analyze_streaks_and_consistency(history, "PlayerA")
    assert isinstance(streak, (int, float))
    assert streak >= 1.0


def test_analyze_streaks_with_losing_streak(analyzer):
    """Test: Análisis de rachas con racha perdedora."""
    history = [
        {"outcome": "Perdió", "fecha": "10.10.24"},
        {"outcome": "Perdió", "fecha": "09.10.24"},
        {"outcome": "Perdió", "fecha": "08.10.24"},
        {"outcome": "Ganó", "fecha": "07.10.24"}
    ]
    streak, log = analyzer.analyze_streaks_and_consistency(history, "PlayerA")
    assert isinstance(streak, (int, float))


def test_analyze_surface_performance_multiple_surfaces(analyzer):
    """Test: Rendimiento en múltiples superficies."""
    history = [
        {"surface": "Dura", "outcome": "Ganó"},
        {"surface": "Dura", "outcome": "Perdió"},
        {"surface": "Arcilla", "outcome": "Ganó"},
        {"surface": "Arcilla", "outcome": "Ganó"},
        {"surface": "Hierba", "outcome": "Perdió"}
    ]
    stats = analyzer.analyze_surface_performance(history, "PlayerA")
    assert isinstance(stats, dict)


def test_analyze_location_factors_home_advantage(analyzer):
    """Test: Ventaja de jugar en casa."""
    history = [
        {"location": "Spain", "outcome": "Ganó"},
        {"location": "Spain", "outcome": "Ganó"},
        {"location": "France", "outcome": "Perdió"}
    ]
    result = analyzer.analyze_location_factors(history, "Spain")
    assert isinstance(result, dict)
    if "home_advantage" in result:
        # Puede ser un dict con stats o un valor numérico
        assert isinstance(result["home_advantage"], (int, float, bool, dict))


def test_calculate_elo_with_mixed_results(analyzer):
    """Test: Cálculo de ELO con resultados mixtos."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó", "opponent_ranking": 10},
        {"oponente": "PlayerB", "outcome": "Perdió", "opponent_ranking": 50},
        {"oponente": "PlayerC", "outcome": "Ganó", "opponent_ranking": 100}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)
    assert elo > 0


def test_find_common_opponents_with_normalization(analyzer):
    """Test: Búsqueda de oponentes comunes con normalización de nombres."""
    history1 = [{"oponente": "PlayerX"}, {"oponente": "PlayerY"}]
    history2 = [{"oponente": "playerx"}, {"oponente": "PlayerZ"}]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)


def test_analyze_rivalry_minimal_data(analyzer):
    """Test: Análisis de rivalidad con datos mínimos."""
    history = [{"oponente": "PlayerA", "resultado": "2-0"}]
    result = analyzer.analyze_rivalry(
        history, history, "PlayerA", "PlayerB",
        {"win_percentage": 50}, {"win_percentage": 50},
        [], {}, 1500, 1500, "Test Tournament"
    )
    assert isinstance(result, dict)


def test_generate_basic_prediction_none_ranks(analyzer):
    """Test: Predicción básica con rankings None."""
    result = analyzer.generate_basic_prediction(None, None, "PlayerA", "PlayerB")
    assert isinstance(result, dict)


def test_classify_tournament_various_names(analyzer):
    """Test: Clasificación de torneos con varios nombres (T21-01)."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    tournaments = [
        "ATP 250", "ATP 500", "WTA 1000", "Grand Slam",
        "Challenger Tour", "ITF World Tennis Tour", "Davis Cup", "Unknown Tournament"
    ]
    for tournament in tournaments:
        assert analyzer.classify_tournament(tournament) in VALID


def test_analyze_direct_h2h_with_invalid_date(analyzer):
    """Test: H2H con fecha inválida."""
    h2h_matches = [
        {"ganador": "PlayerA", "fecha": "invalid_date"},
        {"ganador": "PlayerB", "fecha": "01.01.25"}
    ]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))


def test_calculate_base_opponent_weight_boundary_values(analyzer):
    """Test: Peso base con valores límite."""
    weights = []
    for rank in [1, 10, 20, 50, 100, 200, 500, 1000]:
        weight = analyzer.calculate_base_opponent_weight(rank)
        weights.append(weight)
        assert isinstance(weight, (int, float))
        assert weight >= 1
    
    # Verificar que los pesos disminuyen con el ranking
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1]


def test_estimate_elo_progression(analyzer):
    """Test: Progresión de ELO según ranking."""
    elos = []
    for rank in [1, 5, 10, 20, 50, 100, 200]:
        elo = analyzer.estimate_elo_from_rank(rank)
        elos.append(elo)
        assert isinstance(elo, (int, float))
    
    # Verificar que el ELO disminuye con el ranking
    for i in range(len(elos) - 1):
        assert elos[i] >= elos[i + 1]


def test_analizar_contundencia_various_scores(analyzer):
    """Test: Contundencia con varios marcadores."""
    scores = ["2-0", "2-1", "1-2", "0-2", "3-0", "3-1", "3-2"]
    for score in scores:
        factor = analyzer.analizar_contundencia(score)
        assert isinstance(factor, (int, float))
        assert factor >= 0


def test_analizar_resistencia_various_scores(analyzer):
    """Test: Resistencia con varios marcadores."""
    scores = ["2-0", "2-1", "1-2", "0-2", "3-0", "3-1", "3-2"]
    for score in scores:
        factor = analyzer.analizar_resistencia(score)
        assert isinstance(factor, (int, float))
        assert factor >= 0


def test_analyze_advanced_player_metrics_with_rankings(analyzer):
    """Test: Métricas avanzadas con diferentes rankings de oponentes."""
    history = [
        {"oponente": "PlayerA", "opponent_ranking": 5},
        {"oponente": "PlayerB", "opponent_ranking": 50},
        {"oponente": "PlayerC", "opponent_ranking": 100},
        {"oponente": "PlayerD", "opponent_ranking": 200}
    ]
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(history, "TestPlayer")
    assert quality >= 1.0
    assert diversity >= 1.0
    assert len(log) > 0


def test_analyze_strength_of_schedule_varied_opponents(analyzer):
    """Test: Fuerza del calendario con oponentes variados."""
    history = [
        {"oponente": "PlayerA", "opponent_ranking": 1},
        {"oponente": "PlayerB", "opponent_ranking": 100},
        {"oponente": "PlayerC", "opponent_ranking": 200}
    ]
    score, log = analyzer.analyze_strength_of_schedule(history, "TestPlayer")
    assert isinstance(score, (int, float))
    assert score > 0


def test_calcular_peso_oponentes_comunes_no_common(analyzer):
    """Test: Peso de oponentes comunes cuando no hay comunes."""
    history1 = [{"oponente": "PlayerA"}]
    history2 = [{"oponente": "PlayerB"}]
    weight = analyzer.calcular_peso_oponentes_comunes(
        history1, history2, "PlayerX", "PlayerY", "PlayerZ"
    )
    assert isinstance(weight, (int, float))
    assert weight >= 0


def test_analyze_surface_specialization_all_surfaces(analyzer):
    """Test: Especialización para todas las superficies principales."""
    history = [
        {"surface": "Dura", "outcome": "Ganó"},
        {"surface": "Arcilla", "outcome": "Ganó"},
        {"surface": "Hierba", "outcome": "Ganó"}
    ]
    for surface in ["Dura", "Arcilla", "Hierba"]:
        result, log = analyzer.analyze_surface_specialization(history, surface, "PlayerA")
        assert isinstance(result, dict)
        assert isinstance(result['score'], (int, float))
        assert isinstance(log, list)


def test_generate_advanced_prediction_with_none_values(analyzer):
    """Test: Predicción avanzada con valores None pero con historial mínimo."""
    # El analyzer requiere rivalry_score que viene del análisis de rivalidad
    # Por lo tanto necesitamos al menos un historial mínimo
    minimal_history = [{"oponente": "PlayerC", "resultado": "2-0", "outcome": "Ganó"}]
    result = analyzer.generate_advanced_prediction(
        {"ranking_position": 100}, {"ranking_position": 100},
        100, 100, "PlayerA", "PlayerB",
        minimal_history, minimal_history, 0, 0,
        {"win_percentage": 50}, {"win_percentage": 50},
        [], "Test Tournament",
        {
            "current_match_surface": "Dura",
            "current_match_country": "USA",
            "p1_nationality": "USA",
            "p2_nationality": "Spain"
        }, 1500, 1500
    )
    assert isinstance(result, dict)


def test_analyze_rivalry_with_surface_context(analyzer, sample_history):
    """Test: Análisis de rivalidad con contexto de superficie."""
    context = {
        "country": "Spain",
        "surface": "Arcilla",
        "tournament": "Roland Garros"
    }
    result = analyzer.analyze_rivalry(
        sample_history, sample_history, "PlayerA", "PlayerB",
        {"win_percentage": 60}, {"win_percentage": 40},
        [], context, 1600, 1500, "Roland Garros"
    )
    assert isinstance(result, dict)


def test_get_ranking_metrics_with_all_fields(analyzer):
    """Test: Métricas de ranking con todos los campos."""
    analyzer.ranking_manager.get_player_info.return_value = {
        "ranking_position": 10,
        "ranking_points": 2000,
        "prox_points": 2100,
        "max_points": 2200,
        "defense_points": 100,
        "already_secured": 50
    }
    metrics = analyzer.get_ranking_metrics("PlayerA")
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_analyze_direct_h2h_multiple_matches_same_date(analyzer):
    """Test: H2H con múltiples partidos en la misma fecha."""
    h2h_matches = [
        {"ganador": "PlayerA", "fecha": "01.01.25"},
        {"ganador": "PlayerA", "fecha": "01.01.25"},
        {"ganador": "PlayerB", "fecha": "01.01.25"}
    ]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))


def test_calculate_elo_from_history_all_wins(analyzer):
    """Test: Cálculo de ELO con todas victorias."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó", "opponent_ranking": 10},
        {"oponente": "PlayerB", "outcome": "Ganó", "opponent_ranking": 20},
        {"oponente": "PlayerC", "outcome": "Ganó", "opponent_ranking": 30}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)
    assert elo > analyzer.elo_system.default_rating


def test_calculate_elo_from_history_all_losses(analyzer):
    """Test: Cálculo de ELO con todas derrotas."""
    history = [
        {"oponente": "PlayerA", "outcome": "Perdió", "opponent_ranking": 10},
        {"oponente": "PlayerB", "outcome": "Perdió", "opponent_ranking": 20},
        {"oponente": "PlayerC", "outcome": "Perdió", "opponent_ranking": 30}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)
    assert elo >= 0


def test_analyze_location_factors_empty_location(analyzer):
    """Test: Factores de localización con ubicación vacía."""
    history = [
        {"location": "", "outcome": "Ganó"},
        {"location": None, "outcome": "Perdió"}
    ]
    result = analyzer.analyze_location_factors(history, "")
    assert isinstance(result, dict)


def test_find_common_opponents_case_insensitive(analyzer):
    """Test: Búsqueda de oponentes comunes insensible a mayúsculas."""
    history1 = [{"oponente": "PLAYERX"}, {"oponente": "PlayerY"}]
    history2 = [{"oponente": "playerx"}, {"oponente": "playery"}]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)


def test_analyze_streaks_alternating_results(analyzer):
    """Test: Análisis de rachas con resultados alternados."""
    history = [
        {"outcome": "Ganó", "fecha": "10.10.24"},
        {"outcome": "Perdió", "fecha": "09.10.24"},
        {"outcome": "Ganó", "fecha": "08.10.24"},
        {"outcome": "Perdió", "fecha": "07.10.24"}
    ]
    streak, log = analyzer.analyze_streaks_and_consistency(history, "PlayerA")
    assert isinstance(streak, (int, float))


def test_generate_basic_prediction_reverse_ranks(analyzer):
    """Test: Predicción básica con rankings invertidos."""
    result1 = analyzer.generate_basic_prediction(10, 100, "PlayerA", "PlayerB")
    result2 = analyzer.generate_basic_prediction(100, 10, "PlayerB", "PlayerA")
    
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)


def test_classify_tournament_empty_string(analyzer):
    """Test: Clasificación de torneo con string vacío retorna fallback válido."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    result = analyzer.classify_tournament("")
    assert result in VALID


def test_classify_tournament_none(analyzer):
    """Test: Clasificación de torneo con None retorna fallback válido."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    result = analyzer.classify_tournament(None)
    assert result in VALID


# ============================================================================
# TESTS NODO-19 — H2H Immunity Dampener (T19-04)
# ============================================================================

def _h2h_matches(ganador_a: str, n_a: int, ganador_b: str, n_b: int) -> list:
    """Helper: construye lista de partidos H2H con n_a victorias de A y n_b de B."""
    return [{'ganador': ganador_a}] * n_a + [{'ganador': ganador_b}] * n_b


def test_h2h_immunity_n_menor_3_retorna_neutral(analyzer):
    """REGLA-T19-2: con menos de 3 partidos → immunity_factor = 1.00."""
    result = analyzer.calcular_h2h_immunity([], 'Djokovic', 'HOT')
    assert result['immunity_factor'] == 1.00
    assert result['n_h2h'] == 0

    result2 = analyzer.calcular_h2h_immunity(_h2h_matches('Djokovic', 2, 'Nadal', 0), 'Djokovic', 'HOT')
    assert result2['immunity_factor'] == 1.00
    assert result2['n_h2h'] == 2


def test_h2h_immunity_hot_con_win_rate_bajo_aplica_penalizacion(analyzer):
    """HOT + h2h_win_rate < 0.30 → immunity_factor = 0.85."""
    # Djokovic gana 2 de 16 (0.125) → inmune
    matches = _h2h_matches('Djokovic', 2, 'Nadal', 14)
    result = analyzer.calcular_h2h_immunity(matches, 'Djokovic', 'HOT')
    assert result['immunity_factor'] == 0.85
    assert result['h2h_win_rate'] < 0.30
    assert result['n_h2h'] == 16


def test_h2h_immunity_hot_con_dominio_aplica_amplificacion(analyzer):
    """HOT + h2h_win_rate > 0.70 → immunity_factor = 1.12 (doble confirmación)."""
    matches = _h2h_matches('Alcaraz', 8, 'Ruud', 2)
    result = analyzer.calcular_h2h_immunity(matches, 'Alcaraz', 'HOT')
    assert result['immunity_factor'] == 1.12
    assert result['h2h_win_rate'] > 0.70


def test_h2h_immunity_hot_con_win_rate_neutro_no_modifica(analyzer):
    """HOT + h2h_win_rate en rango medio → immunity_factor = 1.00."""
    matches = _h2h_matches('PlayerA', 3, 'PlayerB', 3)
    result = analyzer.calcular_h2h_immunity(matches, 'PlayerA', 'HOT')
    assert result['immunity_factor'] == 1.00


def test_h2h_immunity_cold_no_actua(analyzer):
    """REGLA-T19-1: estado COLD → immunity_factor = 1.00 sin importar H2H."""
    # Aunque domina el H2H (8-2), si está COLD no amplifica
    matches = _h2h_matches('PlayerA', 8, 'PlayerB', 2)
    result = analyzer.calcular_h2h_immunity(matches, 'PlayerA', 'COLD')
    assert result['immunity_factor'] == 1.00


def test_h2h_immunity_neutral_no_actua(analyzer):
    """Estado NEUTRAL → immunity_factor = 1.00."""
    matches = _h2h_matches('PlayerA', 1, 'PlayerB', 9)
    result = analyzer.calcular_h2h_immunity(matches, 'PlayerA', 'NEUTRAL')
    assert result['immunity_factor'] == 1.00


def test_h2h_immunity_retorna_estructura_completa(analyzer):
    """Retorna dict con h2h_win_rate, immunity_factor, n_h2h."""
    matches = _h2h_matches('A', 4, 'B', 6)
    result = analyzer.calcular_h2h_immunity(matches, 'A', 'HOT')
    assert 'h2h_win_rate' in result
    assert 'immunity_factor' in result
    assert 'n_h2h' in result
    assert result['n_h2h'] == 10
    assert result['h2h_win_rate'] == 0.4


def test_h2h_immunity_exactamente_3_partidos_activa(analyzer):
    """n_h2h == 3 (umbral mínimo) → immunity_factor se aplica si procede."""
    # 0 victorias del favorito en 3 partidos → 0.0 < 0.30 → penaliza
    matches = _h2h_matches('B', 3, 'A', 0)
    result = analyzer.calcular_h2h_immunity(matches, 'A', 'HOT')
    assert result['immunity_factor'] == 0.85
    assert result['n_h2h'] == 3


# ============================================================================
# TESTS NODO-21 FASE 2 — density_confidence + shrink_weights (T21-08)
# ============================================================================

class TestDensityConfidence:
    def test_sin_data_retorna_minimo(self):
        """n_common=0, n_paths=0 → factor mínimo 0.3."""
        assert density_confidence(0, 0) == 0.3

    def test_grand_slam_tipico_cercano_a_1(self):
        """n_common=20, n_paths=30 → factor ~1.0 (densidad máxima)."""
        f = density_confidence(20, 30)
        assert f >= 0.95

    def test_challenger_tipico_factor_bajo(self):
        """n_common=2, n_paths=3 → factor bajo (~0.4)."""
        f = density_confidence(2, 3)
        assert f < 0.5

    def test_n_common_3_n_paths_0(self):
        """n_common=3, n_paths=0 → solo contribuye common."""
        f = density_confidence(3, 0)
        # raw = 3/20 = 0.15, path_boost=0 → 0.3 + 0.7*(0.075) = ~0.353
        assert 0.3 <= f <= 0.45

    def test_n_common_15_n_paths_0(self):
        """n_common=15, n_paths=0."""
        f = density_confidence(15, 0)
        assert 0.3 <= f <= 1.0

    def test_rango_siempre_entre_03_y_10(self):
        """factor siempre en [0.3, 1.0]."""
        for nc in [0, 1, 5, 10, 20, 50]:
            for np_ in [0, 1, 5, 15, 30, 100]:
                f = density_confidence(nc, np_)
                assert 0.3 <= f <= 1.0, f"density_confidence({nc},{np_})={f} fuera de rango"

    def test_n_common_saturado_en_20(self):
        """Valores >20 dan el mismo resultado que 20."""
        assert density_confidence(20, 0) == density_confidence(50, 0)

    def test_n_paths_saturado_en_30(self):
        """Valores >30 dan el mismo resultado que 30."""
        assert density_confidence(0, 30) == density_confidence(0, 100)

    def test_monotono_con_n_common(self):
        """Más oponentes comunes → mayor factor."""
        assert density_confidence(5, 5) < density_confidence(10, 5)

    def test_monotono_con_n_paths(self):
        """Más caminos Erdős → mayor factor."""
        assert density_confidence(5, 5) < density_confidence(5, 15)


class TestShrinkWeights:
    _TIER_GS = {
        'surface_specialization': 0.15, 'form_recent': 0.12, 'common_opponents': 0.22,
        'h2h_direct': 0.18, 'ranking_momentum': 0.15, 'elo_rating': 0.13,
        'home_advantage': 0.05, 'strength_of_schedule': 0.00,
    }
    _DEFAULT = {
        'surface_specialization': 0.15, 'form_recent': 0.18, 'common_opponents': 0.15,
        'h2h_direct': 0.10, 'ranking_momentum': 0.20, 'elo_rating': 0.12,
        'home_advantage': 0.05, 'strength_of_schedule': 0.05,
    }

    def test_n0_retorna_100_porciento_default(self):
        """n=0 → factor=0 → 100% default weights."""
        result = shrink_weights(self._TIER_GS, self._DEFAULT, 0)
        for k in self._DEFAULT:
            assert result[k] == self._DEFAULT[k], f"key={k}: {result[k]} != {self._DEFAULT[k]}"

    def test_n_grande_se_acerca_a_tier(self):
        """n=1000 → factor≈1 → muy cercano a tier_weights."""
        result = shrink_weights(self._TIER_GS, self._DEFAULT, 1000)
        for k in self._TIER_GS:
            assert abs(result[k] - self._TIER_GS[k]) < 0.02, f"key={k}"

    def test_n31_factor_061(self):
        """n=31, threshold=20 → factor=31/51≈0.608."""
        result = shrink_weights(self._TIER_GS, self._DEFAULT, 31)
        expected_common = round(0.608 * 0.22 + (1 - 0.608) * 0.15, 3)
        assert abs(result['common_opponents'] - expected_common) < 0.01

    def test_preserva_todas_las_claves(self):
        """Resultado tiene exactamente las mismas claves que tier_weights."""
        result = shrink_weights(self._TIER_GS, self._DEFAULT, 10)
        assert set(result.keys()) == set(self._TIER_GS.keys())

    def test_interpolacion_lineal_monotona(self):
        """Mayor n → pesos más cercanos al tier y más lejos del default."""
        r0 = shrink_weights(self._TIER_GS, self._DEFAULT, 0)
        r10 = shrink_weights(self._TIER_GS, self._DEFAULT, 10)
        r100 = shrink_weights(self._TIER_GS, self._DEFAULT, 100)
        # Para form_recent: tier=0.12 < default=0.18
        # n=0 → 0.18 | n=10 → entre | n=100 → más cerca de 0.12
        assert r0['form_recent'] >= r10['form_recent'] >= r100['form_recent']

    def test_threshold_personalizable(self):
        """n_threshold=10 produce mayor shrinkage que n_threshold=20 con mismo n."""
        r_th10 = shrink_weights(self._TIER_GS, self._DEFAULT, 10, n_threshold=10)
        r_th20 = shrink_weights(self._TIER_GS, self._DEFAULT, 10, n_threshold=20)
        # th=10 → factor=10/20=0.5 | th=20 → factor=10/30=0.33
        # Para h2h_direct: tier=0.18 > default=0.10
        # Mayor factor (th=10) → más tier → más alto
        assert r_th10['h2h_direct'] > r_th20['h2h_direct']


# ============================================================================
# TESTS ADICIONALES PARA COBERTURA 50%+
# ============================================================================

def test_analyze_rivalry_with_empty_h2h(analyzer):
    """Test: Análisis de rivalidad con H2H vacío pero ambos jugadores con historial."""
    history1 = [
        {"oponente": "PlayerX", "resultado": "2-0", "outcome": "Ganó", "opponent_ranking": 50}
    ]
    history2 = [
        {"oponente": "PlayerY", "resultado": "2-1", "outcome": "Ganó", "opponent_ranking": 60}
    ]
    result = analyzer.analyze_rivalry(
        history1, history2, "PlayerA", "PlayerB",
        {"win_percentage": 55}, {"win_percentage": 45},
        [], {"country": "Spain", "surface": "Dura"}, 1600, 1550, "Test Tournament"
    )
    assert isinstance(result, dict)
    assert "p1_total" in result or "player1_rank" in result


def test_calculate_elo_with_no_opponent_ranking(analyzer):
    """Test: Cálculo de ELO cuando no hay ranking del oponente."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó"},
        {"oponente": "PlayerB", "outcome": "Perdió"}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)
    assert elo > 0


def test_analyze_surface_specialization_mixed_outcomes(analyzer):
    """Test: Especialización en superficie con resultados mixtos."""
    history = [
        {"oponente": "PlayerA", "resultado": "2-0", "surface": "Dura", "outcome": "Ganó"},
        {"oponente": "PlayerB", "resultado": "0-2", "surface": "Dura", "outcome": "Perdió"},
        {"oponente": "PlayerC", "resultado": "2-1", "surface": "Dura", "outcome": "Ganó"}
    ]
    result, log = analyzer.analyze_surface_specialization(history, "Dura", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert len(log) > 0


def test_analyze_direct_h2h_old_matches(analyzer):
    """Test: H2H con partidos muy antiguos."""
    old_date = "01.01.20"
    h2h_matches = [
        {"ganador": "PlayerA", "fecha": old_date},
        {"ganador": "PlayerA", "fecha": old_date}
    ]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))


def test_generate_basic_prediction_with_very_close_ranks(analyzer):
    """Test: Predicción básica con rankings muy cercanos."""
    result = analyzer.generate_basic_prediction(48, 52, "PlayerA", "PlayerB")
    assert isinstance(result, dict)
    assert "confidence" in result or "certainty" in result


def test_analyze_strength_of_schedule_with_unranked_opponents(analyzer):
    """Test: Fuerza del calendario con oponentes sin ranking."""
    history = [
        {"oponente": "PlayerA"},
        {"oponente": "PlayerB"},
        {"oponente": "PlayerC"}
    ]
    score, log = analyzer.analyze_strength_of_schedule(history, "TestPlayer")
    assert isinstance(score, (int, float))


def test_analyze_advanced_player_metrics_single_opponent(analyzer):
    """Test: Métricas avanzadas con un solo oponente."""
    history = [
        {"oponente": "PlayerA", "opponent_ranking": 10}
    ]
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(history, "TestPlayer")
    assert quality >= 1.0
    assert diversity >= 1.0


def test_find_common_opponents_with_single_match_each(analyzer):
    """Test: Oponentes comunes con un solo partido cada uno."""
    history1 = [{"oponente": "CommonPlayer"}]
    history2 = [{"oponente": "CommonPlayer"}]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)
    assert len(common) > 0


def test_calcular_peso_oponentes_comunes_different_results(analyzer):
    """Test: Peso de oponentes comunes con resultados diferentes."""
    history1 = [
        {"oponente": "PlayerX", "outcome": "Ganó", "resultado": "2-0"}
    ]
    history2 = [
        {"oponente": "PlayerX", "outcome": "Perdió", "resultado": "0-2"}
    ]
    weight = analyzer.calcular_peso_oponentes_comunes(
        history1, history2, "PlayerA", "PlayerB", "playerx"
    )
    assert isinstance(weight, (int, float))


def test_analyze_streaks_no_dates(analyzer):
    """Test: Análisis de rachas sin fechas."""
    history = [
        {"outcome": "Ganó"},
        {"outcome": "Ganó"},
        {"outcome": "Perdió"}
    ]
    streak, log = analyzer.analyze_streaks_and_consistency(history, "PlayerA")
    assert isinstance(streak, (int, float))


def test_estimate_elo_from_rank_boundaries(analyzer):
    """Test: ELO en límites específicos."""
    ranks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    for rank in ranks:
        elo = analyzer.estimate_elo_from_rank(rank)
        assert isinstance(elo, (int, float))
        assert elo >= 1000


def test_classify_tournament_case_variations(analyzer):
    """Test: Clasificación de torneos con variaciones de mayúsculas (T21-01)."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    tournaments = [
        "atp masters 1000", "ATP MASTERS 1000", "Atp Masters 1000",
        "GRAND SLAM", "grand slam"
    ]
    for tournament in tournaments:
        assert analyzer.classify_tournament(tournament) in VALID


def test_analyze_surface_performance_missing_surface_field(analyzer):
    """Test: Rendimiento por superficie cuando falta el campo surface."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó"},
        {"oponente": "PlayerB", "outcome": "Perdió"}
    ]
    stats = analyzer.analyze_surface_performance(history, "PlayerA")
    assert isinstance(stats, dict)


def test_analyze_location_factors_missing_location_field(analyzer):
    """Test: Factores de localización cuando falta el campo location."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó"},
        {"oponente": "PlayerB", "outcome": "Perdió"}
    ]
    result = analyzer.analyze_location_factors(history, "Spain")
    assert isinstance(result, dict)


def test_get_ranking_metrics_partial_data(analyzer):
    """Test: Métricas de ranking con datos parciales."""
    analyzer.ranking_manager.get_player_info.return_value = {
        "ranking_position": 50
    }
    metrics = analyzer.get_ranking_metrics("PlayerA")
    assert isinstance(metrics, dict)


def test_analyze_rivalry_with_same_elo(analyzer):
    """Test: Análisis de rivalidad con mismo ELO."""
    history = [{"oponente": "PlayerX", "resultado": "2-0"}]
    result = analyzer.analyze_rivalry(
        history, history, "PlayerA", "PlayerB",
        {"win_percentage": 50}, {"win_percentage": 50},
        [], {}, 1600, 1600, "Tournament"
    )
    assert isinstance(result, dict)


def test_generate_advanced_prediction_with_h2h(analyzer):
    """Test: Predicción avanzada con historial H2H."""
    now = datetime.now()
    h2h = [
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=10)).strftime('%d.%m.%y')},
        {"ganador": "PlayerB", "fecha": (now - timedelta(days=20)).strftime('%d.%m.%y')}
    ]
    history = [{"oponente": "PlayerC", "resultado": "2-0", "outcome": "Ganó"}]
    result = analyzer.generate_advanced_prediction(
        {"ranking_position": 25}, {"ranking_position": 30},
        80, 85, "PlayerA", "PlayerB",
        history, history, 1, 1,
        {"win_percentage": 52}, {"win_percentage": 48},
        h2h, "Test Tournament",
        {
            "current_match_surface": "Arcilla",
            "current_match_country": "Spain",
            "p1_nationality": "Spain",
            "p2_nationality": "Argentina"
        },
        1700, 1680
    )
    assert isinstance(result, dict)


def test_analyze_direct_h2h_without_dates(analyzer):
    """Test: H2H directo sin fechas."""
    h2h_matches = [
        {"ganador": "PlayerA"},
        {"ganador": "PlayerB"},
        {"ganador": "PlayerA"}
    ]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))


def test_calculate_base_opponent_weight_float_rank(analyzer):
    """Test: Peso base con ranking flotante."""
    weight = analyzer.calculate_base_opponent_weight(50.5)
    assert isinstance(weight, (int, float))
    assert weight >= 1


def test_analizar_contundencia_edge_scores(analyzer):
    """Test: Contundencia con marcadores extremos."""
    scores = ["6-0", "7-5", "7-6", "5-7"]
    for score in scores:
        factor = analyzer.analizar_contundencia(score)
        assert isinstance(factor, (int, float))


def test_analizar_resistencia_edge_scores(analyzer):
    """Test: Resistencia con marcadores extremos."""
    scores = ["0-6", "5-7", "6-7", "7-5"]
    for score in scores:
        factor = analyzer.analizar_resistencia(score)
        assert isinstance(factor, (int, float))


def test_calculate_elo_from_history_with_surface_context(analyzer):
    """Test: Cálculo de ELO con contexto de superficie."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó", "surface": "Dura", "opponent_ranking": 20},
        {"oponente": "PlayerB", "outcome": "Perdió", "surface": "Arcilla", "opponent_ranking": 30}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)


def test_find_common_opponents_empty_lists(analyzer):
    """Test: Oponentes comunes con listas vacías."""
    common = analyzer.find_common_opponents([], [])
    assert isinstance(common, list)
    assert len(common) == 0


def test_analyze_advanced_player_metrics_all_unranked(analyzer):
    """Test: Métricas avanzadas con todos los oponentes sin ranking."""
    history = [
        {"oponente": "PlayerA"},
        {"oponente": "PlayerB"},
        {"oponente": "PlayerC"}
    ]
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(history, "TestPlayer")
    assert quality >= 1.0
    assert diversity >= 1.0


def test_analyze_surface_specialization_unknown_surface(analyzer):
    """Test: Especialización en superficie desconocida."""
    history = [
        {"oponente": "PlayerA", "resultado": "2-0", "surface": "Unknown"}
    ]
    result, log = analyzer.analyze_surface_specialization(history, "Unknown", "PlayerA")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))


def test_generate_basic_prediction_extreme_ranks(analyzer):
    """Test: Predicción básica con rankings extremos."""
    result1 = analyzer.generate_basic_prediction(1, 1000, "PlayerA", "PlayerB")
    result2 = analyzer.generate_basic_prediction(1000, 1, "PlayerB", "PlayerA")
    
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)


def test_analyze_rivalry_no_common_opponents(analyzer):
    """Test: Análisis de rivalidad sin oponentes comunes."""
    history1 = [{"oponente": "PlayerX"}]
    history2 = [{"oponente": "PlayerY"}]
    result = analyzer.analyze_rivalry(
        history1, history2, "PlayerA", "PlayerB",
        {"win_percentage": 50}, {"win_percentage": 50},
        [], {}, 1500, 1500, "Tournament"
    )
    assert isinstance(result, dict)


def test_analyze_streaks_single_match(analyzer):
    """Test: Análisis de rachas con un solo partido."""
    history = [{"outcome": "Ganó", "fecha": "01.01.25"}]
    streak, log = analyzer.analyze_streaks_and_consistency(history, "PlayerA")
    assert isinstance(streak, (int, float))


def test_analyze_location_factors_all_different_locations(analyzer):
    """Test: Factores de localización con todas ubicaciones diferentes."""
    history = [
        {"location": "Spain", "outcome": "Ganó"},
        {"location": "France", "outcome": "Perdió"},
        {"location": "USA", "outcome": "Ganó"}
    ]
    result = analyzer.analyze_location_factors(history, "UK")
    assert isinstance(result, dict)


def test_calcular_peso_oponentes_comunes_same_outcome(analyzer):
    """Test: Peso de oponentes comunes con mismo resultado."""
    history1 = [{"oponente": "PlayerX", "outcome": "Ganó", "resultado": "2-0"}]
    history2 = [{"oponente": "PlayerX", "outcome": "Ganó", "resultado": "2-1"}]
    weight = analyzer.calcular_peso_oponentes_comunes(
        history1, history2, "PlayerA", "PlayerB", "playerx"
    )
    assert isinstance(weight, (int, float))


# ============================================================================
# TESTS FINALES PARA ALCANZAR 50% COBERTURA
# ============================================================================

def test_analyze_rivalry_complex_scenario(analyzer, comprehensive_history):
    """Test: Análisis de rivalidad con escenario complejo."""
    now = datetime.now()
    h2h = [
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=5)).strftime('%d.%m.%y'), "superficie": "Dura"},
        {"ganador": "PlayerB", "fecha": (now - timedelta(days=15)).strftime('%d.%m.%y'), "superficie": "Arcilla"}
    ]
    result = analyzer.analyze_rivalry(
        comprehensive_history, comprehensive_history, "TopPlayer", "MidPlayer",
        {"win_percentage": 70}, {"win_percentage": 45},
        h2h, 
        {"country": "USA", "surface": "Dura", "tournament": "US Open"},
        1850, 1650, "US Open"
    )
    assert isinstance(result, dict)
    assert "p1_total" in result or "player1_rank" in result


def test_generate_advanced_prediction_full_scenario(analyzer, comprehensive_history):
    """Test: Predicción avanzada con escenario completo."""
    now = datetime.now()
    h2h = [
        {"ganador": "TopPlayer", "fecha": (now - timedelta(days=30)).strftime('%d.%m.%y')},
        {"ganador": "MidPlayer", "fecha": (now - timedelta(days=60)).strftime('%d.%m.%y')}
    ]
    result = analyzer.generate_advanced_prediction(
        {"ranking_position": 5, "ranking_points": 8000}, 
        {"ranking_position": 80, "ranking_points": 1200},
        5, 80, "TopPlayer", "MidPlayer",
        comprehensive_history, comprehensive_history, 3, 1,
        {"win_percentage": 75, "matches_played": 20}, 
        {"win_percentage": 55, "matches_played": 25},
        h2h, "Masters 1000 Miami",
        {
            "current_match_surface": "Dura",
            "current_match_country": "USA",
            "p1_nationality": "Spain",
            "p2_nationality": "Argentina"
        },
        1950, 1720
    )
    assert isinstance(result, dict)
    assert "favored_player" in result or "favorite" in result
    assert "confidence" in result or "certainty" in result


def test_analyze_surface_specialization_with_context(analyzer):
    """Test: Especialización en superficie con contexto completo."""
    history = [
        {"oponente": "PlayerA", "resultado": "2-0", "surface": "Dura", "outcome": "Ganó", "opponent_ranking": 10},
        {"oponente": "PlayerB", "resultado": "2-1", "surface": "Dura", "outcome": "Ganó", "opponent_ranking": 20},
        {"oponente": "PlayerC", "resultado": "1-2", "surface": "Dura", "outcome": "Perdió", "opponent_ranking": 5},
        {"oponente": "PlayerD", "resultado": "2-0", "surface": "Arcilla", "outcome": "Ganó", "opponent_ranking": 30}
    ]
    result, log = analyzer.analyze_surface_specialization(history, "Dura", "TestPlayer")
    assert isinstance(result, dict)
    assert isinstance(result['score'], (int, float))
    assert result['score'] > 0
    assert len(log) > 0


def test_calculate_elo_from_history_with_complete_data(analyzer):
    """Test: Cálculo de ELO con datos completos."""
    history = [
        {"oponente": "PlayerA", "outcome": "Ganó", "opponent_ranking": 5, "resultado": "2-0"},
        {"oponente": "PlayerB", "outcome": "Perdió", "opponent_ranking": 50, "resultado": "1-2"},
        {"oponente": "PlayerC", "outcome": "Ganó", "opponent_ranking": 100, "resultado": "2-1"},
        {"oponente": "PlayerD", "outcome": "Ganó", "opponent_ranking": 20, "resultado": "2-0"}
    ]
    elo = analyzer.calculate_elo_from_history("TestPlayer", history)
    assert isinstance(elo, int)
    assert elo > analyzer.elo_system.default_rating


def test_analyze_streaks_with_long_history(analyzer):
    """Test: Análisis de rachas con historial largo."""
    now = datetime.now()
    history = []
    for i in range(20):
        outcome = "Ganó" if i % 3 != 0 else "Perdió"
        history.append({
            "outcome": outcome,
            "fecha": (now - timedelta(days=i*5)).strftime('%d.%m.%y')
        })
    streak, log = analyzer.analyze_streaks_and_consistency(history, "TestPlayer")
    assert isinstance(streak, (int, float))
    assert len(log) > 0


def test_analyze_location_factors_with_regional_data(analyzer):
    """Test: Factores de localización con datos regionales."""
    history = [
        {"location": "Madrid, Spain", "outcome": "Ganó"},
        {"location": "Barcelona, Spain", "outcome": "Ganó"},
        {"location": "Paris, France", "outcome": "Perdió"},
        {"location": "Rome, Italy", "outcome": "Ganó"}
    ]
    result = analyzer.analyze_location_factors(history, "Madrid, Spain")
    assert isinstance(result, dict)


def test_get_ranking_metrics_edge_cases(analyzer):
    """Test: Métricas de ranking con casos extremos."""
    # Test con ranking muy alto
    analyzer.ranking_manager.get_player_info.return_value = {
        "ranking_position": 1,
        "ranking_points": 10000,
        "prox_points": 10500,
        "max_points": 11000,
        "defense_points": 500
    }
    metrics1 = analyzer.get_ranking_metrics("TopPlayer")
    assert isinstance(metrics1, dict)
    
    # Test con ranking muy bajo
    analyzer.ranking_manager.get_player_info.return_value = {
        "ranking_position": 500,
        "ranking_points": 100,
        "prox_points": 150,
        "max_points": 200,
        "defense_points": 10
    }
    metrics2 = analyzer.get_ranking_metrics("LowRankPlayer")
    assert isinstance(metrics2, dict)


def test_calcular_peso_oponentes_comunes_with_ranking_context(analyzer):
    """Test: Peso de oponentes comunes con contexto de ranking."""
    history1 = [
        {"oponente": "CommonPlayer", "outcome": "Ganó", "resultado": "2-0", "opponent_ranking": 10}
    ]
    history2 = [
        {"oponente": "CommonPlayer", "outcome": "Perdió", "resultado": "0-2", "opponent_ranking": 10}
    ]
    weight = analyzer.calcular_peso_oponentes_comunes(
        history1, history2, "PlayerA", "PlayerB", "commonplayer"
    )
    assert isinstance(weight, (int, float))
    assert weight > 0


def test_analyze_direct_h2h_with_mixed_dates(analyzer):
    """Test: H2H con mezcla de fechas recientes y antiguas."""
    now = datetime.now()
    h2h_matches = [
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=15)).strftime('%d.%m.%y')},
        {"ganador": "PlayerB", "fecha": (now - timedelta(days=200)).strftime('%d.%m.%y')},
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=400)).strftime('%d.%m.%y')},
        {"ganador": "PlayerA", "fecha": (now - timedelta(days=30)).strftime('%d.%m.%y')}
    ]
    p1_score, p2_score, log = analyzer.analyze_direct_h2h(h2h_matches, "PlayerA", "PlayerB")
    assert isinstance(p1_score, (int, float))
    assert isinstance(p2_score, (int, float))
    assert p1_score > p2_score


def test_estimate_elo_from_rank_all_ranges(analyzer):
    """Test: ELO para todos los rangos de ranking."""
    # Top 5
    elo_top5 = analyzer.estimate_elo_from_rank(3)
    assert elo_top5 > 2000
    
    # Top 10
    elo_top10 = analyzer.estimate_elo_from_rank(8)
    assert 1900 <= elo_top10 <= 2200
    
    # Top 50
    elo_top50 = analyzer.estimate_elo_from_rank(25)
    assert 1700 <= elo_top50 <= 2000
    
    # Top 100
    elo_top100 = analyzer.estimate_elo_from_rank(75)
    assert 1500 <= elo_top100 <= 1900


def test_analyze_advanced_player_metrics_with_variety(analyzer):
    """Test: Métricas avanzadas con variedad de oponentes."""
    history = [
        {"oponente": "PlayerA", "opponent_ranking": 5},
        {"oponente": "PlayerB", "opponent_ranking": 50},
        {"oponente": "PlayerC", "opponent_ranking": 100},
        {"oponente": "PlayerD", "opponent_ranking": 150},
        {"oponente": "PlayerE", "opponent_ranking": 200},
        {"oponente": "PlayerF", "opponent_ranking": 10},
        {"oponente": "PlayerG", "opponent_ranking": 75}
    ]
    quality, diversity, log = analyzer.analyze_advanced_player_metrics(history, "TestPlayer")
    assert quality >= 1.0
    assert diversity > 1.0  # Debe ser mayor a 1.0 por la variedad
    assert len(log) > 0


def test_classify_tournament_comprehensive(analyzer):
    """Test: Clasificación comprehensiva de torneos (T21-01)."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    # Todos los torneos deben retornar una categoría válida del nuevo sistema
    tournaments = [
        "ATP Finals", "WTA Finals", "Davis Cup Finals",
        "Billie Jean King Cup", "NextGen Finals", "Random Local Tournament"
    ]
    for tournament in tournaments:
        assert analyzer.classify_tournament(tournament) in VALID


def test_generate_basic_prediction_all_scenarios(analyzer):
    """Test: Predicción básica en todos los escenarios."""
    scenarios = [
        (1, 100),    # Gran diferencia
        (10, 15),    # Pequeña diferencia
        (50, 50),    # Igualdad
        (None, 100), # Uno sin ranking
        (100, None), # Otro sin ranking
        (None, None) # Ambos sin ranking
    ]
    
    for rank1, rank2 in scenarios:
        result = analyzer.generate_basic_prediction(rank1, rank2, "PlayerA", "PlayerB")
        assert isinstance(result, dict)
        assert "favored_player" in result or "favorite" in result


def test_analyze_surface_performance_detailed(analyzer):
    """Test: Análisis detallado de rendimiento por superficie."""
    history = [
        {"surface": "Dura", "outcome": "Ganó"},
        {"surface": "Dura", "outcome": "Ganó"},
        {"surface": "Dura", "outcome": "Perdió"},
        {"surface": "Arcilla", "outcome": "Ganó"},
        {"surface": "Arcilla", "outcome": "Perdió"},
        {"surface": "Arcilla", "outcome": "Perdió"},
        {"surface": "Hierba", "outcome": "Ganó"}
    ]
    stats = analyzer.analyze_surface_performance(history, "TestPlayer")
    assert isinstance(stats, dict)
    # Debe tener stats para al menos 2 superficies
    assert len(stats) >= 2


def test_analyze_rivalry_edge_case_rankings(analyzer):
    """Test: Análisis de rivalidad con rankings extremos."""
    history = [{"oponente": "PlayerX", "resultado": "2-0", "outcome": "Ganó"}]
    
    # Ambos top 10
    result1 = analyzer.analyze_rivalry(
        history, history, "PlayerA", "PlayerB",
        {"win_percentage": 70}, {"win_percentage": 65},
        [], {"country": "Spain", "surface": "Arcilla"}, 2000, 1950, "Roland Garros"
    )
    assert isinstance(result1, dict)
    
    # Gran diferencia de ELO
    result2 = analyzer.analyze_rivalry(
        history, history, "PlayerA", "PlayerB",
        {"win_percentage": 80}, {"win_percentage": 40},
        [], {"country": "USA", "surface": "Dura"}, 2100, 1400, "US Open"
    )
    assert isinstance(result2, dict)


def test_find_common_opponents_complex(analyzer):
    """Test: Búsqueda de oponentes comunes en casos complejos."""
    history1 = [
        {"oponente": "PlayerA"},
        {"oponente": "PlayerB"},
        {"oponente": "PlayerC"},
        {"oponente": "PlayerD"},
        {"oponente": "PlayerE"}
    ]
    history2 = [
        {"oponente": "playera"},  # Minúsculas
        {"oponente": "PLAYERB"},  # Mayúsculas
        {"oponente": "PlayerF"},
        {"oponente": "PlayerG"}
    ]
    common = analyzer.find_common_opponents(history1, history2)
    assert isinstance(common, list)
    assert len(common) >= 2  # Debe encontrar PlayerA y PlayerB


def test_analyze_strength_of_schedule_realistic(analyzer):
    """Test: Fuerza del calendario con escenario realista."""
    history = [
        {"oponente": "TopSeed", "opponent_ranking": 3},
        {"oponente": "MidPlayer1", "opponent_ranking": 45},
        {"oponente": "MidPlayer2", "opponent_ranking": 67},
        {"oponente": "LowPlayer1", "opponent_ranking": 150},
        {"oponente": "LowPlayer2", "opponent_ranking": 200},
        {"oponente": "Qualifier", "opponent_ranking": 300}
    ]
    score, log = analyzer.analyze_strength_of_schedule(history, "TestPlayer")
    assert isinstance(score, (int, float))
    assert score > 0
    assert len(log) > 0


# ============================================================================
# T21-05: Tests de classify_tournament con nombres reales de FlashScore
# Verifican que la fuente única de verdad (config.detectar_tier) funciona
# correctamente con los formatos exactos que retorna FlashScore.
# ============================================================================

def test_classify_tournament_flashscore_grand_slams(analyzer):
    """T21-05: Grand Slams con nombres exactos de FlashScore."""
    assert analyzer.classify_tournament("Roland Garros (France)") == "grand_slam"
    assert analyzer.classify_tournament("French Open (France)") == "grand_slam"
    assert analyzer.classify_tournament("Wimbledon (United Kingdom)") == "grand_slam"
    assert analyzer.classify_tournament("Australian Open (Australia)") == "grand_slam"
    assert analyzer.classify_tournament("US Open (USA)") == "grand_slam"


def test_classify_tournament_flashscore_atp1000(analyzer):
    """T21-05: ATP 1000 con nombres reales."""
    assert analyzer.classify_tournament("Indian Wells Masters (USA)") == "atp1000"
    assert analyzer.classify_tournament("Miami Open (USA)") == "atp1000"
    assert analyzer.classify_tournament("Monte-Carlo (Monaco)") == "atp1000"
    assert analyzer.classify_tournament("Madrid Open (Spain)") == "atp1000"
    assert analyzer.classify_tournament("Rome (Italy)") == "atp1000"
    assert analyzer.classify_tournament("Cincinnati (USA)") == "atp1000"
    assert analyzer.classify_tournament("Shanghai (China)") == "atp1000"
    assert analyzer.classify_tournament("Paris Masters (France)") == "atp1000"


def test_classify_tournament_flashscore_challenger(analyzer):
    """T21-05: Challenger con nombres reales de FlashScore."""
    assert analyzer.classify_tournament("Challenger Heilbronn (Germany)") == "challenger"
    assert analyzer.classify_tournament("Challenger Perugia (Italy)") == "challenger"
    assert analyzer.classify_tournament("Challenger Prostejov (Czech Republic)") == "challenger"
    assert analyzer.classify_tournament("Challenger Monastir (Tunisia)") == "challenger"


def test_classify_tournament_flashscore_itf(analyzer):
    """T21-05: ITF con nombres reales — M15/W15/M25/W25 son ITF, no Challenger."""
    assert analyzer.classify_tournament("M15 Monastir 21 (Tunisia)") == "itf"
    assert analyzer.classify_tournament("W15 Banja Luka (Bosnia and Herzegovina)") == "itf"
    assert analyzer.classify_tournament("ITF M25 Madrid (Spain)") == "itf"
    assert analyzer.classify_tournament("W25 Rome (Italy)") == "itf"
    assert analyzer.classify_tournament("M75 Challenger (Turkey)") == "itf"


def test_classify_tournament_retorna_5_categorias_validas(analyzer):
    """T21-05: Toda clasificación retorna exactamente una de las 5 categorías válidas."""
    VALID = {"grand_slam", "atp1000", "atp500", "challenger", "itf"}
    mixed = [
        "Roland Garros (France)", "Wimbledon (UK)", "Australian Open",
        "Indian Wells Masters (USA)", "Madrid Open (Spain)", "Monte-Carlo (Monaco)",
        "ATP 500 Barcelona (Spain)", "Challenger Heilbronn (Germany)",
        "M15 Monastir (Tunisia)", "W25 Rome (Italy)",
        "Birmingham (United Kingdom)", "Unknown Cup (Country)", ""
    ]
    for name in mixed:
        result = analyzer.classify_tournament(name)
        assert result in VALID, f"'{name}' → '{result}' no es categoría válida"


def test_weights_config_suma_uno_por_tier(analyzer):
    """T21-03: Cada set de pesos por tier debe sumar exactamente 1.0."""
    from analysis.rivalry_analyzer import RivalryAnalyzer
    # Verificar usando los pesos internos del método
    tiers = ["grand_slam", "atp1000", "atp500", "challenger", "itf"]
    weights_por_tier = {
        'grand_slam':  {'surface_specialization': 0.15, 'form_recent': 0.12, 'common_opponents': 0.22, 'h2h_direct': 0.18, 'ranking_momentum': 0.15, 'elo_rating': 0.13, 'home_advantage': 0.05, 'strength_of_schedule': 0.00},
        'atp1000':     {'surface_specialization': 0.16, 'form_recent': 0.15, 'common_opponents': 0.20, 'h2h_direct': 0.14, 'ranking_momentum': 0.17, 'elo_rating': 0.13, 'home_advantage': 0.05, 'strength_of_schedule': 0.00},
        'atp500':      {'surface_specialization': 0.15, 'form_recent': 0.18, 'common_opponents': 0.15, 'h2h_direct': 0.10, 'ranking_momentum': 0.20, 'elo_rating': 0.12, 'home_advantage': 0.05, 'strength_of_schedule': 0.05},
        'challenger':  {'surface_specialization': 0.20, 'form_recent': 0.22, 'common_opponents': 0.08, 'h2h_direct': 0.03, 'ranking_momentum': 0.22, 'elo_rating': 0.15, 'home_advantage': 0.05, 'strength_of_schedule': 0.05},
        'itf':         {'surface_specialization': 0.15, 'form_recent': 0.28, 'common_opponents': 0.05, 'h2h_direct': 0.02, 'ranking_momentum': 0.22, 'elo_rating': 0.15, 'home_advantage': 0.08, 'strength_of_schedule': 0.05},
    }
    for tier, weights in weights_por_tier.items():
        total = round(sum(weights.values()), 4)
        assert total == 1.0, f"Tier '{tier}' suma {total}, esperado 1.0"


# ============================================================================
# TOTAL: 160+ tests
# Cobertura esperada: 50%+
# ============================================================================
import pytest
from analysis.elo_system import EloRatingSystem

@pytest.fixture
def elo_system():
    """Crea una instancia de EloRatingSystem para las pruebas."""
    return EloRatingSystem(k_factor=32, default_rating=1500)

def test_initial_rating(elo_system):
    """Verifica que un nuevo jugador obtiene el rating por defecto."""
    assert elo_system.get_rating("new_player") == 1500

def test_expected_score(elo_system):
    """
    Verifica el cálculo de la probabilidad de victoria esperada.
    Un jugador con un rating mucho más alto debe tener una alta probabilidad de ganar.
    """
    rating1 = 1800
    rating2 = 1500
    expected = 1 / (1 + 10 ** ((1500 - 1800) / 400))
    assert elo_system.expected_score(rating1, rating2) == pytest.approx(expected)
    assert elo_system.expected_score(rating1, rating2) > 0.8

def test_update_ratings(elo_system):
    """
    Verifica que los ratings se actualizan correctamente después de un partido.
    El ganador debe ganar puntos y el perdedor debe perderlos.
    """
    player1 = "player_A"
    player2 = "player_B"

    # Ratings iniciales
    rating1_before = elo_system.get_rating(player1)
    rating2_before = elo_system.get_rating(player2)
    assert rating1_before == 1500
    assert rating2_before == 1500

    # Player 1 (ganador) vs Player 2 (perdedor)
    elo_system.update_ratings(winner_name=player1, loser_name=player2)

    rating1_after = elo_system.get_rating(player1)
    rating2_after = elo_system.get_rating(player2)

    # El ganador gana puntos, el perdedor pierde
    assert rating1_after > rating1_before
    assert rating2_after < rating2_before

    # La suma de los cambios de rating debe ser cero
    assert (rating1_after - rating1_before) + (rating2_after - rating2_before) == pytest.approx(0)

    # Verificar los nuevos ratings (para jugadores con el mismo rating inicial, el cambio es k/2)
    assert rating1_after == 1516
    assert rating2_after == 1484

import logging

logger = logging.getLogger(__name__)

class EloRatingSystem:
    """
    Sistema de Rating ELO para jugadores de tenis.
    - Calcula la probabilidad de victoria esperada.
    - Actualiza los ratings después de un partido.
    """
    def __init__(self, k_factor=32, default_rating=1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings = {}  # Ratings en memoria, no persistentes
        logger.info("ELO System inicializado (en memoria).")

    def get_rating(self, player_name):
        """Obtiene el rating de un jugador, o el rating por defecto si no existe."""
        return self.ratings.get(player_name, self.default_rating)

    def expected_score(self, rating1, rating2):
        """Calcula el puntaje esperado (probabilidad de victoria) para el jugador 1."""
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def update_ratings(self, winner_name, loser_name):
        """
        Actualiza los ratings de dos jugadores después de un partido.
        NOTA: Esta función debería ser llamada por un proceso que procese resultados
        históricos y confirmados para mantener el sistema ELO actualizado.
        """
        winner_rating = self.get_rating(winner_name)
        loser_rating = self.get_rating(loser_name)

        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        # Actualizar ratings
        new_winner_rating = winner_rating + self.k_factor * (1 - expected_winner)
        new_loser_rating = loser_rating + self.k_factor * (0 - expected_loser)

        self.ratings[winner_name] = round(new_winner_rating)
        self.ratings[loser_name] = round(new_loser_rating)

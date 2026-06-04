import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# T21-09 (Nodo-21 Fase 3) — K-factor por tier (Kalman gain analogy)
# ─────────────────────────────────────────────────────────────────────────────

K_FACTOR_BY_TIER = {
    'grand_slam': 24,   # señal densa → cambios pequeños pero confiables
    'atp1000':    28,
    'atp500':     32,   # base clásica — fallback conservador
    'challenger': 40,   # señal ruidosa → ELO se mueve más (captura volatilidad)
    'itf':        48,   # campo desconocido, cada partido muy informativo
}


def k_factor_efectivo(tier: str, recencia_pelt: int = None) -> int:
    """
    T21-09/T21-10 (Nodo-21 Fase 3): K-factor adaptivo por tier + reset post-PELT.

    Analogía Kalman: K-factor = gain del filtro.
      Grand Slam K=24: señal confiable → ELO se mueve poco por partido.
      Challenger K=40: señal ruidosa → ELO se mueve más.
      ITF K=48:        campo desconocido, cada partido muy informativo.

    REGLA-T21-5: Post-PELT fresco (recencia≤5) → K×1.5 (incertidumbre reiniciada).
      Conexión con Nodo-18: si el jugador cambió de régimen hace ≤5 partidos,
      el ELO debe ser más reactivo para capturar el nuevo nivel.

    Args:
        tier:          'grand_slam' | 'atp1000' | 'atp500' | 'challenger' | 'itf'
        recencia_pelt: partidos desde el último cambio PELT (None = sin cambio)
    """
    k_base = K_FACTOR_BY_TIER.get(tier, 32)
    if recencia_pelt is not None and recencia_pelt <= 5:
        return int(k_base * 1.5)  # régimen nuevo = incertidumbre reiniciada
    return k_base


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

    def update_ratings(self, winner_name, loser_name, tier: str = None, recencia_pelt: int = None):
        """
        Actualiza los ratings de dos jugadores después de un partido.

        T21-09/T21-10 (Nodo-21 Fase 3): Si se provee `tier`, usa K-factor adaptivo
        vía k_factor_efectivo(). Si `recencia_pelt` ≤ 5, K se multiplica ×1.5
        (régimen nuevo = incertidumbre reiniciada, conexión Nodo-18).

        Args:
            winner_name:   nombre del ganador
            loser_name:    nombre del perdedor
            tier:          tier del torneo — usa K adaptivo si se provee
            recencia_pelt: partidos desde cambio PELT — amplifica K si fresco
        """
        k = k_factor_efectivo(tier, recencia_pelt) if tier is not None else self.k_factor

        winner_rating = self.get_rating(winner_name)
        loser_rating = self.get_rating(loser_name)

        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        new_winner_rating = winner_rating + k * (1 - expected_winner)
        new_loser_rating = loser_rating + k * (0 - expected_loser)

        self.ratings[winner_name] = round(new_winner_rating)
        self.ratings[loser_name] = round(new_loser_rating)

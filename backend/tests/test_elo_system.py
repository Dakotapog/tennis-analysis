import pytest
from analysis.elo_system import EloRatingSystem, K_FACTOR_BY_TIER, k_factor_efectivo

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


# ─────────────────────────────────────────────────────────────────────────────
# TESTS NODO-21 FASE 3 — K-factor por tier + reset post-PELT (T21-11)
# ─────────────────────────────────────────────────────────────────────────────

class TestKFactorByTier:
    def test_constante_tiene_5_tiers(self):
        assert set(K_FACTOR_BY_TIER.keys()) == {'grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'}

    def test_orden_ascendente_gs_a_itf(self):
        """Grand Slam tiene K más bajo que ITF (señal más limpia → menos reactivo)."""
        assert K_FACTOR_BY_TIER['grand_slam'] < K_FACTOR_BY_TIER['atp1000'] < K_FACTOR_BY_TIER['atp500'] \
               < K_FACTOR_BY_TIER['challenger'] < K_FACTOR_BY_TIER['itf']

    def test_atp500_es_base_clasica(self):
        assert K_FACTOR_BY_TIER['atp500'] == 32

    def test_grand_slam_k24(self):
        assert K_FACTOR_BY_TIER['grand_slam'] == 24

    def test_itf_k48(self):
        assert K_FACTOR_BY_TIER['itf'] == 48


class TestKFactorEfectivo:
    def test_grand_slam_sin_pelt_retorna_24(self):
        assert k_factor_efectivo('grand_slam') == 24

    def test_challenger_sin_pelt_retorna_40(self):
        assert k_factor_efectivo('challenger') == 40

    def test_itf_sin_pelt_retorna_48(self):
        assert k_factor_efectivo('itf') == 48

    def test_atp500_fallback_sin_tier(self):
        """Tier desconocido → fallback K=32."""
        assert k_factor_efectivo('desconocido') == 32

    def test_recencia_none_no_amplifica(self):
        """recencia_pelt=None → K base sin amplificación."""
        assert k_factor_efectivo('grand_slam', recencia_pelt=None) == 24

    def test_recencia_5_amplifica_k_x15(self):
        """recencia_pelt=5 (límite inclusivo) → K × 1.5."""
        assert k_factor_efectivo('grand_slam', recencia_pelt=5) == int(24 * 1.5)   # 36
        assert k_factor_efectivo('challenger', recencia_pelt=5) == int(40 * 1.5)   # 60

    def test_recencia_3_amplifica(self):
        """recencia_pelt=3 (FRESCO) → K × 1.5."""
        assert k_factor_efectivo('atp1000', recencia_pelt=3) == int(28 * 1.5)   # 42

    def test_recencia_6_no_amplifica(self):
        """recencia_pelt=6 (> umbral de 5) → K base."""
        assert k_factor_efectivo('grand_slam', recencia_pelt=6) == 24

    def test_recencia_1_fresco_amplifica(self):
        """recencia_pelt=1 → K × 1.5."""
        assert k_factor_efectivo('itf', recencia_pelt=1) == int(48 * 1.5)   # 72

    def test_resultado_es_entero(self):
        """k_factor_efectivo siempre retorna int."""
        for tier in K_FACTOR_BY_TIER:
            assert isinstance(k_factor_efectivo(tier), int)
            assert isinstance(k_factor_efectivo(tier, recencia_pelt=3), int)


class TestUpdateRatingsConTier:
    def test_sin_tier_usa_k_instancia(self):
        """Sin tier → usa self.k_factor (comportamiento original)."""
        elo = EloRatingSystem(k_factor=32)
        elo.update_ratings('A', 'B')
        assert elo.get_rating('A') == 1516

    def test_grand_slam_menor_cambio_que_challenger(self):
        """Grand Slam K=24 → menor cambio de rating que Challenger K=40."""
        elo_gs = EloRatingSystem()
        elo_ch = EloRatingSystem()
        elo_gs.update_ratings('A', 'B', tier='grand_slam')
        elo_ch.update_ratings('A', 'B', tier='challenger')
        # GS: ganador sube menos (señal más pequeña pero confiable)
        assert elo_gs.get_rating('A') < elo_ch.get_rating('A')

    def test_pelt_fresco_amplifica_cambio(self):
        """recencia_pelt=3 → K×1.5 → mayor cambio de rating."""
        elo_normal = EloRatingSystem()
        elo_pelt = EloRatingSystem()
        elo_normal.update_ratings('A', 'B', tier='grand_slam')
        elo_pelt.update_ratings('A', 'B', tier='grand_slam', recencia_pelt=3)
        assert elo_pelt.get_rating('A') > elo_normal.get_rating('A')

    def test_suma_cambios_es_cero_con_tier(self):
        """La suma de cambios de rating debe ser cero (juego de suma cero)."""
        elo = EloRatingSystem()
        before_a = elo.get_rating('A')
        before_b = elo.get_rating('B')
        elo.update_ratings('A', 'B', tier='challenger')
        delta = (elo.get_rating('A') - before_a) + (elo.get_rating('B') - before_b)
        assert abs(delta) <= 1   # redondeo puede introducir ±1

    def test_itf_mayor_cambio_que_atp500(self):
        """ITF K=48 > ATP500 K=32 → mayor cambio."""
        elo_itf = EloRatingSystem()
        elo_500 = EloRatingSystem()
        elo_itf.update_ratings('A', 'B', tier='itf')
        elo_500.update_ratings('A', 'B', tier='atp500')
        assert elo_itf.get_rating('A') > elo_500.get_rating('A')

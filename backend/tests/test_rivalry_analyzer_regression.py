"""
Tests de regresión para RivalryAnalyzer — movidos desde screenshots/

ORIGEN: screenshots/test_extraer_historh2h.py
MOTIVO DEL TRASLADO:
  - Los tests de análisis pertenecen a tests/, no a screenshots/
  - Los valores hardcodeados fueron corregidos para coincidir con
    la implementación real de rivalry_analyzer.py

CORRECCIONES APLICADAS:
  - estimate_elo_from_rank(1): 2400 → 2200 (código real: 2200 - (1-1)*20 = 2200)
  - estimate_elo_from_rank(50): 2005 → 1825 (código real: 2020 - (50-11)*5 = 1825)
  - estimate_elo_from_rank(150): 1751 → 1671 (código real: 1720 - (150-101)*1 = 1671)
  Los tests originales en screenshots/ tenían valores INCORRECTOS que no
  correspondían a la implementación real.
"""

import pytest
from unittest.mock import Mock
from analysis.rivalry_analyzer import RivalryAnalyzer


@pytest.fixture
def mock_ranking_manager():
    return Mock()


@pytest.fixture
def mock_elo_system():
    return Mock()


@pytest.fixture
def analyzer(mock_ranking_manager, mock_elo_system):
    return RivalryAnalyzer(
        ranking_manager=mock_ranking_manager,
        elo_system=mock_elo_system
    )


# ─────────────────────────────────────────────────────────────────────────────
# estimate_elo_from_rank — valores verificados contra el código fuente
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateEloFromRank:
    """
    Fórmula de rivalry_analyzer.py:
      rank <= 10:  2200 - (rank-1) * 20
      rank <= 50:  2020 - (rank-11) * 5
      rank <= 100: 1820 - (rank-51) * 2
      rank <= 200: 1720 - (rank-101) * 1
      else:        1600
    """

    def test_rank_1_top_del_mundo(self, analyzer):
        """Rank 1 → 2200 - (1-1)*20 = 2200."""
        assert analyzer.estimate_elo_from_rank(1) == 2200

    def test_rank_10_limite_top(self, analyzer):
        """Rank 10 → 2200 - (10-1)*20 = 2200 - 180 = 2020."""
        assert analyzer.estimate_elo_from_rank(10) == 2020

    def test_rank_11_inicio_segundo_tramo(self, analyzer):
        """Rank 11 → 2020 - (11-11)*5 = 2020."""
        assert analyzer.estimate_elo_from_rank(11) == 2020

    def test_rank_50_limite_segundo_tramo(self, analyzer):
        """Rank 50 → 2020 - (50-11)*5 = 2020 - 195 = 1825."""
        assert analyzer.estimate_elo_from_rank(50) == 1825

    def test_rank_51_inicio_tercer_tramo(self, analyzer):
        """Rank 51 → 1820 - (51-51)*2 = 1820."""
        assert analyzer.estimate_elo_from_rank(51) == 1820

    def test_rank_100_limite_tercer_tramo(self, analyzer):
        """Rank 100 → 1820 - (100-51)*2 = 1820 - 98 = 1722."""
        assert analyzer.estimate_elo_from_rank(100) == 1722

    def test_rank_150_cuarto_tramo(self, analyzer):
        """Rank 150 → 1720 - (150-101)*1 = 1720 - 49 = 1671."""
        assert analyzer.estimate_elo_from_rank(150) == 1671

    def test_rank_200_limite_cuarto_tramo(self, analyzer):
        """Rank 200 → 1720 - (200-101)*1 = 1720 - 99 = 1621."""
        assert analyzer.estimate_elo_from_rank(200) == 1621

    def test_rank_sobre_200_retorna_1600(self, analyzer):
        """Rank > 200 → 1600 (valor fijo)."""
        assert analyzer.estimate_elo_from_rank(201) == 1600
        assert analyzer.estimate_elo_from_rank(500) == 1600

    def test_rank_none_retorna_default_1500(self, analyzer):
        """Sin ranking → 1500 (default ELO del sistema)."""
        assert analyzer.estimate_elo_from_rank(None) == 1500

    def test_elo_decrece_con_rank_mayor(self, analyzer):
        """A mayor ranking (peor posición) → menor ELO."""
        elos = [analyzer.estimate_elo_from_rank(r) for r in [1, 10, 50, 100, 200, 300]]
        assert elos == sorted(elos, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# calculate_base_opponent_weight
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateBaseOpponentWeight:
    """
    Fórmula de rivalry_analyzer.py:
      rank <= 10:  10
      rank <= 15:  8
      rank <= 30:  6
      rank <= 50:  4
      rank <= 100: 2
      else:        1
      None:        1
    """

    def test_top_10_peso_maximo(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(5) == 10
        assert analyzer.calculate_base_opponent_weight(1) == 10
        assert analyzer.calculate_base_opponent_weight(10) == 10

    def test_rank_11_a_15(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(11) == 8
        assert analyzer.calculate_base_opponent_weight(15) == 8

    def test_rank_16_a_30(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(16) == 6
        assert analyzer.calculate_base_opponent_weight(30) == 6

    def test_rank_31_a_50(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(45) == 4
        assert analyzer.calculate_base_opponent_weight(50) == 4

    def test_rank_51_a_100(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(75) == 2
        assert analyzer.calculate_base_opponent_weight(100) == 2

    def test_rank_sobre_100_peso_minimo(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(150) == 1
        assert analyzer.calculate_base_opponent_weight(500) == 1

    def test_sin_ranking_peso_minimo(self, analyzer):
        assert analyzer.calculate_base_opponent_weight(None) == 1

    def test_peso_decrece_con_rank_mayor(self, analyzer):
        """A mayor ranking → menor peso."""
        pesos = [analyzer.calculate_base_opponent_weight(r) for r in [5, 12, 20, 40, 80, 200]]
        assert pesos == sorted(pesos, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# analizar_contundencia y analizar_resistencia
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalizarContundencia:

    @pytest.mark.parametrize("resultado,factor_esperado", [
        ("2-0", 1.2),           # Victoria en 2 sets
        ("2-1", 1.0),           # Victoria en 3 sets
        ("Resultado inválido", 1.0),
        ("", 1.0),
    ])
    def test_factores_de_contundencia(self, analyzer, resultado, factor_esperado):
        assert analyzer.analizar_contundencia(resultado) == factor_esperado


class TestAnalizarResistencia:

    @pytest.mark.parametrize("resultado,factor_esperado", [
        ("1-2", 0.5),    # Perdió pero ganó 1 set
        ("0-2", 0.0),    # Perdió sin ganar sets
        ("Resultado inválido", 0.0),
        ("", 0.0),
    ])
    def test_factores_de_resistencia(self, analyzer, resultado, factor_esperado):
        assert analyzer.analizar_resistencia(resultado) == factor_esperado


# ─────────────────────────────────────────────────────────────────────────────
# determine_match_winner
# ─────────────────────────────────────────────────────────────────────────────

class TestDetermineMatchWinner:

    @pytest.mark.parametrize("match_data,esperado", [
        ({'outcome': 'Ganó el partido'}, True),
        ({'outcome': 'Player win'}, True),
        ({'outcome': 'Perdió el partido'}, False),
        ({'outcome': 'perdió'}, False),
        ({'resultado': 'WO'}, True),
        ({}, True),
    ])
    def test_determinar_ganador(self, analyzer, match_data, esperado):
        assert analyzer.determine_match_winner(match_data, 'player_name') == esperado

"""
Tests para Nodo-29 Fase 4 (FIX-5) — Integración circuit_asymmetry en edge_calculator

Validar que:
- circuit_asymmetry_signal se propaga desde H2H al edge_report
- circuit_asymmetry_ratio se propaga desde H2H al edge_report
- circuit_warning se activa correctamente (favorito == player_deflated + asimetría)
- edge y kelly_kl NO cambian por presencia de circuit_warning (campo informativo)
- defaults correctos cuando circuit_asymmetry ausente
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from edge_calculator import calcular_edge_completo


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — Partidos mock con variaciones de circuit_asymmetry
# ─────────────────────────────────────────────────────────────────────────────

def _create_partido_base(
    jugador1="Player A",
    jugador2="Player B",
    cuota1=1.80,
    cuota2=2.10,
    favored=None,
    confidence=60.0,
    circuit_signal="SYMMETRIC",
    circuit_ratio=1.0,
    player_deflated=None,
):
    """
    Constructor de partido mock con circuit_asymmetry en la estructura esperada:
    partido['ranking_analysis']['prediction']['circuit_asymmetry']
    """
    if favored is None:
        favored = jugador1

    partido = {
        "jugador1": jugador1,
        "jugador2": jugador2,
        "cuota1": cuota1,
        "cuota2": cuota2,
        "torneo_nombre": "Roland Garros",
        "torneo_completo": "French Open 2026",
        "superficie": "clay",
        "match_url": "https://example.com/match",
        "match_id": "test_match_001",
        "enfrentamientos_directos": [
            {"winner": jugador1, "loser": jugador2, "superficie": "clay"},
            {"winner": jugador2, "loser": jugador1, "superficie": "hard"},
        ],
        "ranking_analysis": {
            "prediction": {
                "favored_player": favored,
                "confidence": confidence,
                "score_breakdown": {
                    "player1": {
                        "surface_specialization": {"raw_score": "50.0", "normalized_score": "0.14", "weight": "15%", "weighted_score": "0.02", "contribution": "14.0%"},
                        "form_recent":            {"raw_score": "150.0", "normalized_score": "0.50", "weight": "12%", "weighted_score": "0.06", "contribution": "12.0%"},
                        "common_opponents":       {"raw_score": "80.0",  "normalized_score": "0.20", "weight": "22%", "weighted_score": "0.04", "contribution": "22.0%"},
                        "h2h_direct":             {"raw_score": "100.0", "normalized_score": "0.29", "weight": "18%", "weighted_score": "0.05", "contribution": "18.0%"},
                        "ranking_momentum":       {"raw_score": "200.0", "normalized_score": "0.44", "weight": "15%", "weighted_score": "0.07", "contribution": "15.0%"},
                        "elo_rating":             {"raw_score": "200.0", "normalized_score": "0.80", "weight": "13%", "weighted_score": "0.10", "contribution": "13.0%"},
                        "home_advantage":         {"raw_score": "0.0",   "normalized_score": "0.00", "weight": "5%",  "weighted_score": "0.00", "contribution": "5.0%"},
                        "strength_of_schedule":   {"raw_score": "50.0",  "normalized_score": "0.25", "weight": "0%",  "weighted_score": "0.00", "contribution": "0.0%"},
                        "Penalizacion_Inactividad": "0.00 pts",
                        "Puntaje_Final": "0.55",
                    },
                    "player2": {
                        "surface_specialization": {"raw_score": "40.0", "normalized_score": "0.11", "weight": "15%", "weighted_score": "0.02", "contribution": "14.0%"},
                        "form_recent":            {"raw_score": "130.0", "normalized_score": "0.43", "weight": "12%", "weighted_score": "0.05", "contribution": "12.0%"},
                        "common_opponents":       {"raw_score": "60.0",  "normalized_score": "0.15", "weight": "22%", "weighted_score": "0.03", "contribution": "22.0%"},
                        "h2h_direct":             {"raw_score": "80.0",  "normalized_score": "0.23", "weight": "18%", "weighted_score": "0.04", "contribution": "18.0%"},
                        "ranking_momentum":       {"raw_score": "180.0", "normalized_score": "0.40", "weight": "15%", "weighted_score": "0.06", "contribution": "15.0%"},
                        "elo_rating":             {"raw_score": "190.0", "normalized_score": "0.76", "weight": "13%", "weighted_score": "0.10", "contribution": "13.0%"},
                        "home_advantage":         {"raw_score": "0.0",   "normalized_score": "0.00", "weight": "5%",  "weighted_score": "0.00", "contribution": "5.0%"},
                        "strength_of_schedule":   {"raw_score": "40.0",  "normalized_score": "0.20", "weight": "0%",  "weighted_score": "0.00", "contribution": "0.0%"},
                        "Penalizacion_Inactividad": "0.00 pts",
                        "Puntaje_Final": "0.45",
                    },
                },
                # NUEVO: circuit_asymmetry anidado en prediction
                "circuit_asymmetry": {
                    "signal": circuit_signal,
                    "asymmetry_ratio": circuit_ratio,
                    "player_deflated": player_deflated,
                    "p1_circuit_tier_index": 0.5,
                    "p2_circuit_tier_index": 0.3,
                    "deflactor_applied": 1.0,
                },
            }
        },
    }
    return partido


@pytest.fixture
def calibracion_mock():
    """Calibración mínima para edge_calculator"""
    return {
        "global": {"wins": 100, "losses": 100},
        "por_superficie": {
            "clay": {"wins": 60, "losses": 40},
            "grass": {"wins": 30, "losses": 20},
            "hard": {"wins": 20, "losses": 30},
            "unknown": {"wins": 10, "losses": 10},
        },
        "por_zona": {
            "heavy_favorite": {"wins": 20, "losses": 10},
            "moderate_favorite": {"wins": 30, "losses": 20},
            "slight_underdog": {"wins": 30, "losses": 40},
            "underdog": {"wins": 20, "losses": 30},
        },
        "por_superficie_y_tier": {
            "clay_grand_slam": {"wins": 31, "losses": 12},
            "grass_grand_slam": {"wins": 15, "losses": 10},
            "hard_grand_slam": {"wins": 12, "losses": 8},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# TF5-01: circuit_asymmetry_signal presente en cada pick
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_01_CircuitSignalPresent:
    """circuit_asymmetry_signal está en el output del edge_report"""

    def test_TF5_01_signal_present_symmetric(self, calibracion_mock):
        """Campo circuit_asymmetry_signal debe existir con valor 'SYMMETRIC'"""
        partido = _create_partido_base(
            circuit_signal="SYMMETRIC",
            circuit_ratio=1.0,
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_signal" in resultado
        assert resultado["circuit_asymmetry_signal"] == "SYMMETRIC"

    def test_TF5_01_signal_present_moderate(self, calibracion_mock):
        """Campo debe existir con valor 'MODERATE_ASYMMETRY'"""
        partido = _create_partido_base(
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Player B",
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_signal" in resultado
        assert resultado["circuit_asymmetry_signal"] == "MODERATE_ASYMMETRY"

    def test_TF5_01_signal_present_strong(self, calibracion_mock):
        """Campo debe existir con valor 'STRONG_ASYMMETRY'"""
        partido = _create_partido_base(
            circuit_signal="STRONG_ASYMMETRY",
            circuit_ratio=6.0,
            player_deflated="Player A",
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_signal" in resultado
        assert resultado["circuit_asymmetry_signal"] == "STRONG_ASYMMETRY"


# ─────────────────────────────────────────────────────────────────────────────
# TF5-02: circuit_asymmetry_ratio presente en cada pick
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_02_CircuitRatioPresent:
    """circuit_asymmetry_ratio está en el output del edge_report"""

    def test_TF5_02_ratio_present_default(self, calibracion_mock):
        """Ratio debe existir con valor 1.0 (default SYMMETRIC)"""
        partido = _create_partido_base(
            circuit_signal="SYMMETRIC",
            circuit_ratio=1.0,
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_ratio" in resultado
        assert resultado["circuit_asymmetry_ratio"] == 1.0

    def test_TF5_02_ratio_present_moderate(self, calibracion_mock):
        """Ratio debe existir con valor 2.5 (MODERATE)"""
        partido = _create_partido_base(
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Player B",
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_ratio" in resultado
        assert resultado["circuit_asymmetry_ratio"] == 2.5

    def test_TF5_02_ratio_present_strong(self, calibracion_mock):
        """Ratio debe existir con valor alto (STRONG)"""
        partido = _create_partido_base(
            circuit_signal="STRONG_ASYMMETRY",
            circuit_ratio=6.0,
            player_deflated="Player A",
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_asymmetry_ratio" in resultado
        assert resultado["circuit_asymmetry_ratio"] == 6.0


# ─────────────────────────────────────────────────────────────────────────────
# TF5-03: circuit_warning = True cuando favorito == player_deflated Y MODERATE
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_03_WarningModerateAsymmetry:
    """circuit_warning = True solo cuando favorito== player_deflated y signal==MODERATE"""

    def test_TF5_03_warning_true_moderate_favorito_deflated(self, calibracion_mock):
        """Favorito es el deflated player + MODERATE_ASYMMETRY → circuit_warning=True"""
        partido = _create_partido_base(
            jugador1="Alcaraz",
            jugador2="Sinner",
            favored="Alcaraz",  # favorito
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Alcaraz",  # deflated == favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_warning" in resultado
        assert resultado["circuit_warning"] is True

    def test_TF5_03_warning_false_when_deflated_is_rival(self, calibracion_mock):
        """Favorito es A, deflated es B + MODERATE_ASYMMETRY → circuit_warning=False"""
        partido = _create_partido_base(
            jugador1="Alcaraz",
            jugador2="Sinner",
            favored="Alcaraz",  # favorito
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Sinner",  # deflated != favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_warning" in resultado
        assert resultado["circuit_warning"] is False


# ─────────────────────────────────────────────────────────────────────────────
# TF5-04: circuit_warning = True cuando favorito == player_deflated Y STRONG
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_04_WarningStrongAsymmetry:
    """circuit_warning = True cuando favorito== player_deflated y signal==STRONG"""

    def test_TF5_04_warning_true_strong_favorito_deflated(self, calibracion_mock):
        """Favorito es el deflated player + STRONG_ASYMMETRY → circuit_warning=True"""
        partido = _create_partido_base(
            jugador1="TopPlayer",
            jugador2="UnderPlayer",
            favored="TopPlayer",  # favorito
            circuit_signal="STRONG_ASYMMETRY",
            circuit_ratio=6.0,
            player_deflated="TopPlayer",  # deflated == favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_warning" in resultado
        assert resultado["circuit_warning"] is True

    def test_TF5_04_warning_false_when_deflated_is_other_player(self, calibracion_mock):
        """Favorito es A, deflated es B + STRONG_ASYMMETRY → circuit_warning=False"""
        partido = _create_partido_base(
            jugador1="TopPlayer",
            jugador2="UnderPlayer",
            favored="TopPlayer",  # favorito
            circuit_signal="STRONG_ASYMMETRY",
            circuit_ratio=6.0,
            player_deflated="UnderPlayer",  # deflated != favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert "circuit_warning" in resultado
        assert resultado["circuit_warning"] is False


# ─────────────────────────────────────────────────────────────────────────────
# TF5-05: circuit_warning = False cuando signal = SYMMETRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_05_NoWarningSymmetric:
    """circuit_warning = False (o ausente) cuando signal = SYMMETRIC"""

    def test_TF5_05_no_warning_symmetric_even_with_deflated(self, calibracion_mock):
        """SYMMETRIC + player_deflated=favorito → aún False (signal es lo que importa)"""
        partido = _create_partido_base(
            circuit_signal="SYMMETRIC",
            circuit_ratio=1.0,
            player_deflated="Player A",  # aunque lo marque deflated
            favored="Player A",  # y sea favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        # circuit_warning debe estar ausente o ser False
        warning = resultado.get("circuit_warning")
        assert warning is False or warning is None

    def test_TF5_05_symmetric_default(self, calibracion_mock):
        """SYMMETRIC sin player_deflated → circuit_warning False/absent"""
        partido = _create_partido_base(
            circuit_signal="SYMMETRIC",
            circuit_ratio=1.0,
            player_deflated=None,
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        warning = resultado.get("circuit_warning")
        assert warning is False or warning is None


# ─────────────────────────────────────────────────────────────────────────────
# TF5-06: circuit_warning = False cuando favorito != player_deflated
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_06_NoWarningDifferentPlayers:
    """circuit_warning = False cuando favorito != player_deflated (incluso con asimetría)"""

    def test_TF5_06_no_warning_favorable_is_different_moderate(self, calibracion_mock):
        """MODERATE_ASYMMETRY pero favorito != deflated → False"""
        partido = _create_partido_base(
            jugador1="Djokovic",
            jugador2="Nadal",
            favored="Djokovic",  # favorito
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Nadal",  # deflated != favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert resultado["circuit_warning"] is False

    def test_TF5_06_no_warning_favorable_is_different_strong(self, calibracion_mock):
        """STRONG_ASYMMETRY pero favorito != deflated → False"""
        partido = _create_partido_base(
            jugador1="Federer",
            jugador2="Murray",
            favored="Federer",  # favorito
            circuit_signal="STRONG_ASYMMETRY",
            circuit_ratio=5.5,
            player_deflated="Murray",  # deflated != favorito
        )
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert resultado["circuit_warning"] is False


# ─────────────────────────────────────────────────────────────────────────────
# TF5-07: edge y kelly_kl NO cambian por circuit_warning (campo informativo)
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_07_EdgeKellyUnchanged:
    """edge y kelly_kl deben ser idénticos independientemente de circuit_warning"""

    def test_TF5_07_same_edge_and_kelly_with_and_without_warning(self, calibracion_mock):
        """
        Calcular edge para mismo partido:
        - Caso 1: signal=SYMMETRIC (sin warning)
        - Caso 2: signal=MODERATE, favorito==deflated (con warning)
        → edge y kelly_kl deben ser IDÉNTICOS
        """
        base_kwargs = {
            "jugador1": "Sinner",
            "jugador2": "Alcaraz",
            "cuota1": 1.95,
            "cuota2": 1.90,
            "favored": "Sinner",
            "confidence": 58.0,
        }

        # Caso 1: sin warning
        partido_no_warning = _create_partido_base(
            **base_kwargs,
            circuit_signal="SYMMETRIC",
            circuit_ratio=1.0,
            player_deflated=None,
        )
        resultado_no_warning = calcular_edge_completo(partido_no_warning, calibracion_mock)

        # Caso 2: con warning
        partido_con_warning = _create_partido_base(
            **base_kwargs,
            circuit_signal="MODERATE_ASYMMETRY",
            circuit_ratio=2.5,
            player_deflated="Sinner",  # favorito == deflated → warning=True
        )
        resultado_con_warning = calcular_edge_completo(partido_con_warning, calibracion_mock)

        # Ambos deben tener resultados válidos
        assert resultado_no_warning is not None
        assert resultado_con_warning is not None

        # edge y kelly_kl deben ser iguales
        assert abs(resultado_no_warning["edge"] - resultado_con_warning["edge"]) < 1e-6
        assert abs(resultado_no_warning["kelly_kl"] - resultado_con_warning["kelly_kl"]) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TF5-08: defaults correctos cuando circuit_asymmetry ausente
# ─────────────────────────────────────────────────────────────────────────────

class TestTF5_08_DefaultsWhenAbsent:
    """signal='SYMMETRIC', ratio=1.0 cuando circuit_asymmetry no existe en pick"""

    def test_TF5_08_default_symmetric_when_absent(self, calibracion_mock):
        """Si prediction sin circuit_asymmetry → default signal='SYMMETRIC'"""
        partido = {
            "jugador1": "Player1",
            "jugador2": "Player2",
            "cuota1": 1.80,
            "cuota2": 2.10,
            "torneo_nombre": "ATP 250",
            "torneo_completo": "ATP 250 2026",
            "superficie": "hard",
            "match_url": "https://example.com",
            "match_id": "test_002",
            "enfrentamientos_directos": [],
            "ranking_analysis": {
                "prediction": {
                    "favored_player": "Player1",
                    "confidence": 60.0,
                    "score_breakdown": {
                        "player1": {
                            "surface_specialization": {"raw_score": "50.0", "normalized_score": "0.14", "weight": "15%", "weighted_score": "0.02", "contribution": "14.0%"},
                            "form_recent":            {"raw_score": "150.0", "normalized_score": "0.50", "weight": "12%", "weighted_score": "0.06", "contribution": "12.0%"},
                            "common_opponents":       {"raw_score": "80.0",  "normalized_score": "0.20", "weight": "22%", "weighted_score": "0.04", "contribution": "22.0%"},
                            "h2h_direct":             {"raw_score": "100.0", "normalized_score": "0.29", "weight": "18%", "weighted_score": "0.05", "contribution": "18.0%"},
                            "ranking_momentum":       {"raw_score": "200.0", "normalized_score": "0.44", "weight": "15%", "weighted_score": "0.07", "contribution": "15.0%"},
                            "elo_rating":             {"raw_score": "200.0", "normalized_score": "0.80", "weight": "13%", "weighted_score": "0.10", "contribution": "13.0%"},
                            "home_advantage":         {"raw_score": "0.0",   "normalized_score": "0.00", "weight": "5%",  "weighted_score": "0.00", "contribution": "5.0%"},
                            "strength_of_schedule":   {"raw_score": "50.0",  "normalized_score": "0.25", "weight": "0%",  "weighted_score": "0.00", "contribution": "0.0%"},
                            "Penalizacion_Inactividad": "0.00 pts", "Puntaje_Final": "0.55",
                        },
                        "player2": {
                            "surface_specialization": {"raw_score": "40.0", "normalized_score": "0.11", "weight": "15%", "weighted_score": "0.02", "contribution": "14.0%"},
                            "form_recent":            {"raw_score": "130.0", "normalized_score": "0.43", "weight": "12%", "weighted_score": "0.05", "contribution": "12.0%"},
                            "common_opponents":       {"raw_score": "60.0",  "normalized_score": "0.15", "weight": "22%", "weighted_score": "0.03", "contribution": "22.0%"},
                            "h2h_direct":             {"raw_score": "80.0",  "normalized_score": "0.23", "weight": "18%", "weighted_score": "0.04", "contribution": "18.0%"},
                            "ranking_momentum":       {"raw_score": "180.0", "normalized_score": "0.40", "weight": "15%", "weighted_score": "0.06", "contribution": "15.0%"},
                            "elo_rating":             {"raw_score": "190.0", "normalized_score": "0.76", "weight": "13%", "weighted_score": "0.10", "contribution": "13.0%"},
                            "home_advantage":         {"raw_score": "0.0",   "normalized_score": "0.00", "weight": "5%",  "weighted_score": "0.00", "contribution": "5.0%"},
                            "strength_of_schedule":   {"raw_score": "40.0",  "normalized_score": "0.20", "weight": "0%",  "weighted_score": "0.00", "contribution": "0.0%"},
                            "Penalizacion_Inactividad": "0.00 pts", "Puntaje_Final": "0.45",
                        },
                    },
                    # circuit_asymmetry AUSENTE
                }
            },
        }
        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert resultado["circuit_asymmetry_signal"] == "SYMMETRIC"
        assert resultado["circuit_asymmetry_ratio"] == 1.0

    def test_TF5_08_default_ratio_when_absent(self, calibracion_mock):
        """Si circuit_asymmetry present pero sin asymmetry_ratio → default 1.0"""
        partido = _create_partido_base()
        # Remover el campo ratio (simular legacy data)
        del partido["ranking_analysis"]["prediction"]["circuit_asymmetry"]["asymmetry_ratio"]

        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert resultado["circuit_asymmetry_ratio"] == 1.0

    def test_TF5_08_default_signal_when_absent_in_dict(self, calibracion_mock):
        """Si circuit_asymmetry present pero sin signal → default 'SYMMETRIC'"""
        partido = _create_partido_base()
        # Remover el campo signal (simular legacy data)
        del partido["ranking_analysis"]["prediction"]["circuit_asymmetry"]["signal"]

        resultado = calcular_edge_completo(partido, calibracion_mock)

        assert resultado is not None
        assert resultado["circuit_asymmetry_signal"] == "SYMMETRIC"


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

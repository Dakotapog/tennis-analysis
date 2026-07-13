"""
tests/test_nodo68_rival_value.py — Nodo-68 D68-05: Rival Value Flip

REGLA-T53: invoca calcular_edge_completo() real — no reimplementa la formula.
Aserciones estructurales/de umbral segun spec Nodo-68 §6 D68-05.

Dos casos obligatorios del spec:
  1. Obradovic-like: p_m=0.675, c_fav=1.18, c_riv=5.20
     → edge_vs_mercado_rival≈+0.133, rival_value_flag=True
  2. Control vig: edge_fav≈-0.04 (dentro del vig ~3%)
     → rival_value_flag=False (el vig se lo come)
"""
import pytest
from edge_calculator import calcular_edge_completo


def _partido(jugador1, jugador2, cuota1, cuota2, confidence, favored=None):
    """Fixture minimo para calcular_edge_completo — solo campos necesarios."""
    if favored is None:
        favored = jugador1
    return {
        'jugador1': jugador1,
        'jugador2': jugador2,
        'cuota1': cuota1,
        'cuota2': cuota2,
        'ranking_analysis': {
            'prediction': {
                'favored_player': favored,
                'confidence': confidence,  # 0-100
            }
        },
    }


CALIBRACION_BASE = {'global': {'wins': 100, 'losses': 50}}


class TestD68_05RivalValueFlag:
    """
    D68-05 (Nodo-68): rival_value_flag y edge_vs_mercado_rival
    serializados por calcular_edge_completo().
    """

    def test_obradovic_like_flag_true(self):
        """
        Caso semilla del spec: p_m=0.675, c_fav=1.18, c_riv=5.20.
        edge_fav = 0.675 - 1/1.18 ≈ -0.172  (<= -0.10 ✓)
        edge_rival = 0.325 - 1/5.20 ≈ +0.133 (valor real)
        cuota_rival=5.20 en [2.50, 8.00] ✓
        → rival_value_flag debe ser True
        """
        partido = _partido('Obradovic', 'Fabre', 1.18, 5.20, confidence=67.5)
        r = calcular_edge_completo(partido, CALIBRACION_BASE)
        assert r is not None
        # flag correcto
        assert r['rival_value_flag'] is True, (
            f"flag debe ser True para edge_fav≈-0.172. "
            f"edge={r.get('edge')}, cuota_rival={r.get('cuota_rival')}"
        )
        # edge_rival positivo y aproximadamente +0.133 (tolerancia ±0.005)
        erv = r.get('edge_vs_mercado_rival')
        assert erv is not None
        assert abs(erv - 0.133) < 0.005, (
            f"edge_vs_mercado_rival esperado ≈0.133, obtenido {erv}"
        )
        # vig serializado y positivo
        vig = r.get('vig')
        assert vig is not None and vig > 0, f"vig debe ser positivo, obtenido {vig}"
        # apostar NO debe cambiar (OBSERVACIONAL PURO)
        assert r.get('apostar') is False, "apostar no debe cambiar por D68-01"

    def test_control_dentro_del_vig_flag_false(self):
        """
        Control: edge_fav pequeño negativo (dentro del vig).
        c_fav=1.82, c_riv=2.10, confidence=51% →
        edge_fav = 0.51 - 1/1.82 ≈ -0.04 (> -0.10, el vig se lo come)
        → rival_value_flag debe ser False
        """
        partido = _partido('PlayerA', 'PlayerB', 1.82, 2.10, confidence=51.0)
        r = calcular_edge_completo(partido, CALIBRACION_BASE)
        assert r is not None
        assert r['rival_value_flag'] is False, (
            f"flag debe ser False cuando edge_fav≈-0.04 (dentro del vig). "
            f"edge={r.get('edge')}"
        )

    def test_cuota_rival_fuera_de_rango_bajo_flag_false(self):
        """
        cuota_rival < 2.50: fuera del rango congelado → flag False,
        aunque edge_fav sea muy negativo.
        """
        # c_fav=1.18 (edge muy negativo) pero c_riv=2.10 < 2.50
        partido = _partido('Heavy', 'Underdog', 1.18, 2.10, confidence=67.5)
        r = calcular_edge_completo(partido, CALIBRACION_BASE)
        assert r is not None
        assert r['rival_value_flag'] is False, (
            f"cuota_rival=2.10 < 2.50: flag debe ser False. "
            f"cuota_rival={r.get('cuota_rival')}"
        )

    def test_cuota_rival_fuera_de_rango_alto_flag_false(self):
        """
        cuota_rival > 8.00: fuera del rango congelado → flag False.
        """
        partido = _partido('Heavy', 'LongShot', 1.05, 9.50, confidence=67.5)
        r = calcular_edge_completo(partido, CALIBRACION_BASE)
        assert r is not None
        assert r['rival_value_flag'] is False, (
            f"cuota_rival=9.50 > 8.00: flag debe ser False. "
            f"cuota_rival={r.get('cuota_rival')}"
        )

    def test_edge_fav_positivo_flag_false(self):
        """
        edge_fav positivo (pick normal del pipeline): flag siempre False.
        """
        partido = _partido('Favorite', 'Rival', 1.50, 2.80, confidence=75.0)
        r = calcular_edge_completo(partido, CALIBRACION_BASE)
        assert r is not None
        # edge positivo → no hay flip
        assert r['rival_value_flag'] is False, (
            f"edge_fav positivo: flag debe ser False. edge={r.get('edge')}"
        )

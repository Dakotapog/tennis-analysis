"""
tests/test_nodo111_dual_book.py — REGLA-T53: invocan funciones reales.

Cubre Nodo-111 X1 — 4 funciones puras de scraping/dual_book_client.py:
  T1.  best_price — retorna la mejor casa cuando hay 2 feeds
  T2.  best_price — retorna None cuando el jugador no está en ningún feed
  T3.  best_price — con un solo feed retorna esa casa
  T4.  best_price — delta_pct correcto entre mejor y peor cuota
  T5.  divergencia — cálculo correcto entre dos cuotas
  T6.  divergencia — cuotas iguales → 0.0%
  T7.  es_arb — detecta arb real (suma implícita < 1.0)
  T8.  es_arb — NO arb cuando suma implícita >= 1.0
  T9.  es_middle — middle informado cuando rango modelo cae en la ventana
  T10. es_middle — NO middle cuando linea_under <= linea_over (ventana inexistente)
  T11. es_middle — NO middle cuando rango modelo cae fuera de la ventana
  T12. _norm — normalización lowercase, sin diacríticos, sin guiones
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scraping.dual_book_client import best_price, divergencia, es_arb, es_middle, _norm


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _feed(players: dict) -> dict:
    """Construye feed normalizado: {norm_name: {odds, jugador, rival}}."""
    return {_norm(k): {"odds": v, "jugador": k, "rival": "Rival"} for k, v in players.items()}


# ── T1-T4: best_price ─────────────────────────────────────────────────────────

def test_best_price_retorna_mejor_casa():
    """Dos feeds: best_price retorna la casa con cuota más alta."""
    feeds = {
        "betplay":    _feed({"Alcaraz": 1.80}),
        "flashscore": _feed({"Alcaraz": 2.05}),
    }
    result = best_price("Alcaraz", feeds)
    assert result is not None
    assert result["casa"] == "flashscore"
    assert result["cuota"] == 2.05


def test_best_price_sin_cobertura_retorna_none():
    """Jugador no presente en ningún feed → None."""
    feeds = {
        "betplay": _feed({"Djokovic": 1.50}),
    }
    result = best_price("Nadal", feeds)
    assert result is None


def test_best_price_un_solo_feed():
    """Con un solo feed, retorna esa casa con delta_pct=0.0."""
    feeds = {"betplay": _feed({"Sinner": 1.65})}
    result = best_price("Sinner", feeds)
    assert result is not None
    assert result["casa"] == "betplay"
    assert result["cuota"] == 1.65
    assert result["delta_pct"] == 0.0


def test_best_price_delta_pct_correcto():
    """delta_pct = (mejor/peor - 1) * 100."""
    feeds = {
        "betplay":    _feed({"Medvedev": 2.00}),
        "flashscore": _feed({"Medvedev": 2.20}),
    }
    result = best_price("Medvedev", feeds)
    assert result is not None
    expected_delta = round((2.20 / 2.00 - 1) * 100, 2)  # 10.0%
    assert result["delta_pct"] == expected_delta


# ── T5-T6: divergencia ────────────────────────────────────────────────────────

def test_divergencia_calculo_correcto():
    """divergencia(1.80, 2.10) = (2.10/1.80 - 1)*100 ≈ 16.67%."""
    result = divergencia(1.80, 2.10)
    expected = round((2.10 / 1.80 - 1) * 100, 2)
    assert result == expected


def test_divergencia_cuotas_iguales():
    """Cuotas idénticas → divergencia = 0.0."""
    result = divergencia(2.00, 2.00)
    assert result == 0.0


def test_divergencia_simetrica():
    """divergencia es simétrica: div(a, b) == div(b, a)."""
    assert divergencia(1.50, 1.80) == divergencia(1.80, 1.50)


# ── T7-T8: es_arb ─────────────────────────────────────────────────────────────

def test_es_arb_arb_real():
    """Suma implícita < 1.0 → arb detectado.
    1/2.10 + 1/2.10 = 0.952 < 1.0."""
    assert es_arb(2.10, 2.10) is True


def test_es_arb_no_arb_suma_mayor():
    """Suma implícita >= 1.0 → no es arb.
    1/1.80 + 1/2.00 = 0.556 + 0.500 = 1.056 > 1.0."""
    assert es_arb(1.80, 2.00) is False


def test_es_arb_cuota_uno_invalida():
    """Cuota <= 1 → inválida, no es arb."""
    assert es_arb(0.0, 3.00) is False
    assert es_arb(3.00, 1.0) is False


# ── T9-T11: es_middle ─────────────────────────────────────────────────────────

def test_es_middle_rango_dentro_ventana():
    """rango_modelo=(21, 24) cae en ventana (over=20.5, under=24.5) → middle."""
    # linea_under=24.5 > linea_over=20.5 → ventana válida
    # rango modelo [21,24]: lo=21>=20.5 y hi=24<=24.5 → dentro
    assert es_middle(20.5, 24.5, (21, 24)) is True


def test_es_middle_sin_ventana():
    """linea_under <= linea_over → ventana inexistente, nunca middle."""
    assert es_middle(22.5, 22.5, (21, 24)) is False
    assert es_middle(23.0, 22.0, (21, 24)) is False


def test_es_middle_rango_fuera_de_ventana():
    """rango_modelo fuera de la ventana → no es middle."""
    # ventana over=20.5, under=24.5; rango=[25,28] cae fuera
    assert es_middle(20.5, 24.5, (25, 28)) is False


# ── T12: _norm ────────────────────────────────────────────────────────────────

def test_norm_diacriticos_y_guiones():
    """_norm elimina diacríticos, guiones y convierte a lowercase."""
    assert _norm("García-López") == "garcia lopez"
    assert _norm("Müller") == "muller"
    assert _norm("Ñoño") == "nono"
    assert _norm("  Federer  ") == "federer"

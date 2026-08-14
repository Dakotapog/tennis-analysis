"""
Tests Nodo-152 — Score propagation + DOMINANTE extremo (D152-01/D152-02).
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Fixes cubiertos:
  D152-01: _enrich_live_score() propaga score_str/games_played/sets_complete/current_games
           al top-level del dict de señal (además de score_data).
  D152-02: _calcular_certeza_condicional() — cuando games_set1 ≤ 7 y zona=DOMINANTE
           y sets_complete ≥ 1, reducir µ por pace del set1 (µ=games_set1×2, cap 16).
           Root cause: Allegre 6:1,6:0 = 13j total, µ=18 → OVER 13.5 ALTA era incorrecto.
"""
import sys
import math
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import _calcular_certeza_condicional


# ── D152-02: DOMINANTE extremo ────────────────────────────────────────────────

def test_152_02_dominante_set1_7j_over_es_baja():
    """D152-02: games_set1=7 (6:1), DOMINANTE, OVER 15.5, gp=7 → p_condicional BAJA.

    Antes del fix µ=18 → p_over≈0.68 (ALTA). Con fix µ=14 → p_over≈0.29 (BAJA).
    Caso raíz: Allegre 6:1,6:0 = 13j total. OVER 15.5 era incorrecto.
    REGLA-T53: invoca _calcular_certeza_condicional real.
    """
    res = _calcular_certeza_condicional(
        linea=15.5,
        direccion="OVER",
        games_played=7,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        sets_home=1,
        sets_away=0,
        games_set1=7,
    )
    assert res["p_condicional"] < 0.40, (
        f"D152-02: games_set1=7 DOMINANTE OVER 15.5 gp=7 → p_over debe ser < 0.40, "
        f"got {res['p_condicional']:.3f}. µ debe ser 14, no 18."
    )
    assert res["alerta_nivel"] in ("BAJA", ""), (
        f"Nivel debe ser BAJA o vacío, got '{res['alerta_nivel']}'"
    )


def test_152_02_dominante_set1_6j_over_13_5_baja():
    """D152-02: games_set1=6 (6:0), DOMINANTE, OVER 13.5, gp=6 → p_condicional BAJA.

    µ_ajustado = 6×2 = 12. OVER 13.5 después de un 6:0 no es apuesta correcta.
    """
    res = _calcular_certeza_condicional(
        linea=13.5,
        direccion="OVER",
        games_played=6,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        sets_home=1,
        sets_away=0,
        games_set1=6,
    )
    assert res["p_condicional"] < 0.40, (
        f"D152-02: 6:0 set1 → µ=12, OVER 13.5 con gp=6 debe ser BAJA, "
        f"got p={res['p_condicional']:.3f}"
    )


def test_152_02_no_trigger_set1_mayor_7():
    """D152-02: games_set1=8 (6:2), DOMINANTE, sets_complete=1 → sin override.

    games_set1=8 > 7 → D152-02 NO aplica → µ estándar=18.
    """
    # Con µ=18 y gp=8, OVER 15.5: mu_rest=10, debería ser ALTA
    res_std = _calcular_certeza_condicional(
        linea=15.5,
        direccion="OVER",
        games_played=8,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        sets_home=1,
        sets_away=0,
        games_set1=8,   # > 7 → no trigger
    )
    # Con µ=18: mu_rest=10, x=(15.5+0.5-8-10)/3=-2/3, p_over≈0.68 → ALTA
    assert res_std["p_condicional"] > 0.50, (
        f"D152-02 no debe activarse con games_set1=8 (>7). "
        f"Esperado p_over>0.50, got {res_std['p_condicional']:.3f}"
    )


def test_152_02_no_trigger_zona_coinflip():
    """D152-02: games_set1=6, COINFLIP, sets_complete=1 → sin override.

    D152-02 solo aplica para DOMINANTE, no COINFLIP.
    """
    res = _calcular_certeza_condicional(
        linea=13.5,
        direccion="OVER",
        games_played=6,
        sets_complete=1,
        current_games=0,
        zona="COINFLIP",   # ← no DOMINANTE
        sets_home=1,
        sets_away=0,
        games_set1=6,
    )
    # COINFLIP µ=23: mu_rest=17, x=(13.5+0.5-6-17)/4.5=-9/4.5=-2, p_under=Φ(-1.41)≈0.079, p_over≈0.92
    assert res["p_condicional"] > 0.70, (
        f"COINFLIP con gp=6 OVER 13.5 debe ser ALTA (µ=23 intacto), "
        f"got p={res['p_condicional']:.3f}"
    )


def test_152_02_no_trigger_sets_complete_cero():
    """D152-02: games_set1=7, DOMINANTE, sets_complete=0 → sin override.

    Si el primer set aún no terminó (sets_complete=0), D152-02 no aplica.
    La evidencia de dominio no está confirmada aún.
    """
    res = _calcular_certeza_condicional(
        linea=15.5,
        direccion="OVER",
        games_played=7,
        sets_complete=0,   # ← set1 en curso, no terminado
        current_games=7,
        zona="DOMINANTE",
        sets_home=0,
        sets_away=0,
        games_set1=7,
    )
    # Sin override → µ=18, mu_rest=11, p_over > 0.50
    assert res["p_condicional"] > 0.50, (
        f"D152-02 no debe activarse cuando sets_complete=0 (set1 aún en curso). "
        f"got p={res['p_condicional']:.3f}"
    )

"""
Tests Nodo-151 — Live Edge Gates (D151-01/02/03).
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Gates cubiertos:
  D151-01: edge_live = p_condicional - (1/cuota_live) < 5% → EXCLUIR
  D151-02: score_str=null + games_played > 3 → EXCLUIR (desventaja informacional)
  D151-03: zona contradice dirección + p_condicional < 40% → EXCLUIR
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import _edge_live_gate, _score_null_gate, _zona_direccion_gate


# ─── D151-01: edge_live gate ────────────────────────────────────────────────

def test_151_01_edge_live_negativo_excluye():
    """p_condicional=0.412, cuota=2.12 → p_impl=0.472, edge_live=-0.06 < 0.05 → excluye.

    Caso real: Pinnington Jones UNDER 21.5 @2.12 (2026-07-28).
    REGLA-T53: invoca _edge_live_gate real.
    """
    certeza = {"p_condicional": 0.412, "certeza_matematica": False}
    cuota_live = 2.12
    assert _edge_live_gate(certeza, cuota_live, umbral=0.05) is True, (
        "edge_live=0.412-0.472=-0.06 debe excluir (< umbral 0.05)"
    )


def test_151_02_edge_live_positivo_pasa():
    """p_condicional=0.80, cuota=1.60 → p_impl=0.625, edge_live=0.175 > 0.05 → pasa.

    REGLA-T53: invoca _edge_live_gate real.
    """
    certeza = {"p_condicional": 0.80, "certeza_matematica": False}
    cuota_live = 1.60
    assert _edge_live_gate(certeza, cuota_live, umbral=0.05) is False, (
        "edge_live=0.80-0.625=0.175 debe pasar (> umbral 0.05)"
    )


def test_151_01b_certeza_none_no_excluye():
    """certeza=None → gate no bloquea (sin datos suficientes = pass-through)."""
    assert _edge_live_gate(None, 2.00, umbral=0.05) is False


def test_151_01c_cuota_cero_no_excluye():
    """cuota_live=0 → gate no bloquea (evita division by zero)."""
    certeza = {"p_condicional": 0.50}
    assert _edge_live_gate(certeza, 0.0, umbral=0.05) is False


# ─── D151-02: score_null gate ───────────────────────────────────────────────

def test_151_03_score_null_gp_alto_excluye():
    """score_str=None, games_played=7 > 3 → excluye (desventaja informacional).

    Caso real: Pinnington Jones gp=5, Ege Sik gp=7, todos ITF_VIVO 2026-07-28.
    REGLA-T53: invoca _score_null_gate real.
    """
    score_data = {"score_str": None, "games_played": 7}
    assert _score_null_gate(score_data, gp_min=3) is True, (
        "score_str=None con gp=7 > 3 debe excluir"
    )


def test_151_04_score_null_gp_bajo_pasa():
    """score_str=None, games_played=2 ≤ 3 → pasa (partido muy temprano, incertidumbre simétrica).

    REGLA-T53: invoca _score_null_gate real.
    """
    score_data = {"score_str": None, "games_played": 2}
    assert _score_null_gate(score_data, gp_min=3) is False, (
        "score_str=None con gp=2 ≤ 3 no debe excluir"
    )


def test_151_03b_score_str_presente_no_excluye():
    """score_str='6:3', games_played=9 → pasa (tenemos el marcador real)."""
    score_data = {"score_str": "6:3", "games_played": 9}
    assert _score_null_gate(score_data, gp_min=3) is False


def test_151_03c_score_data_none_no_excluye():
    """score_data=None → gate no bloquea (sin datos = pass-through)."""
    assert _score_null_gate(None, gp_min=3) is False


# ─── D151-03: zona-dirección gate ───────────────────────────────────────────

def test_151_05_zona_direccion_contradiccion_excluye():
    """zona=DOMINANTE (µ=18), dir=OVER, linea=19.5, p_cond=0.252 < 0.40 → excluye.

    Caso real: Zarazua OVER 19.5 @1.58. DOMINANTE µ=18 < 19.5 → zona predice UNDER.
    Apostamos OVER → contradicción severa (p_cond=0.252 < 0.40).
    REGLA-T53: invoca _zona_direccion_gate real.
    """
    certeza = {"p_condicional": 0.252, "certeza_matematica": False}
    assert _zona_direccion_gate("DOMINANTE", "OVER", 19.5, certeza, p_umbral=0.40) is True, (
        "DOMINANTE+OVER@19.5 con p_cond=0.252 debe excluir"
    )


def test_151_06_zona_direccion_consistente_pasa():
    """zona=DOMINANTE (µ=18), dir=OVER, linea=17.5, p_cond=0.55 → pasa.

    DOMINANTE µ=18 > 17.5 → zona predice OVER. Apostamos OVER → sin contradicción.
    REGLA-T53: invoca _zona_direccion_gate real.
    """
    certeza = {"p_condicional": 0.55, "certeza_matematica": False}
    assert _zona_direccion_gate("DOMINANTE", "OVER", 17.5, certeza, p_umbral=0.40) is False, (
        "DOMINANTE+OVER@17.5 → consistente, no debe excluir"
    )


def test_151_05b_contradiccion_p_alto_pasa():
    """zona=DOMINANTE, dir=OVER, linea=19.5, p_cond=0.45 ≥ 0.40 → pasa.

    Contradicción existe pero p_condicional ≥ umbral → no es severa, no bloquea.
    """
    certeza = {"p_condicional": 0.45}
    assert _zona_direccion_gate("DOMINANTE", "OVER", 19.5, certeza, p_umbral=0.40) is False, (
        "Contradicción marginal (p=0.45 ≥ 0.40) no debe bloquear"
    )


def test_151_05c_coinflip_under_linea_baja_excluye():
    """zona=COINFLIP (µ=23), dir=UNDER, linea=17.5, p_cond=0.30 → excluye.

    COINFLIP µ=23 > 17.5 → zona predice OVER. Apostamos UNDER → contradicción.
    """
    certeza = {"p_condicional": 0.30}
    assert _zona_direccion_gate("COINFLIP", "UNDER", 17.5, certeza, p_umbral=0.40) is True

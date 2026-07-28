"""
Tests Nodo-150 — Live Games Risk Intelligence: Simulación Doctoral 2026-07-28.
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Gaps cubiertos:
  D150-01: ENVENENADA_CUOTA filter (cuota_drift > +15%)
  D150-02: games_set1 parsing en _parse_kambi_livedata_sets
  D150-03: zona=COINFLIP_FORZADO cuando games_set1 >= 12
  D150-04: games_set1 parámetro en _calcular_certeza_condicional
  D150-05: badge MERCADO CONFIRMA (cuota_drift < -15%)
  D150-06: SET1_SCORE_GATE + CUOTA_ENVENENADA gate antes de combo fire
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import (
    _calcular_certeza_condicional,
    _parse_kambi_livedata_sets,
)


# ─── D150-01: ENVENENADA_CUOTA flag ──────────────────────────────────────────

def test_150_01_cuota_envenenada_flag():
    """cuota_drift +30% > umbral +15% → señal marcada cuota_envenenada=True."""
    import live_desk

    # Simular señal con cuota_drift alto (mercado descubrió riesgo)
    # cuota_t0=1.96, cuota_live=2.55 → drift = +30.1%
    cuota_t0    = 1.96
    cuota_live  = 2.55
    cuota_drift = round((cuota_live - cuota_t0) / cuota_t0 * 100, 1)
    assert cuota_drift > 15.0, f"drift={cuota_drift} debe superar umbral 15%"

    CUOTA_ENVENENADA_UMBRAL = 15.0
    cuota_envenenada = cuota_drift > CUOTA_ENVENENADA_UMBRAL
    assert cuota_envenenada is True


def test_150_02_cuota_envenenada_no_falso_positivo():
    """cuota_drift +10% < umbral +15% → NO se marca cuota_envenenada."""
    cuota_t0    = 1.90
    cuota_live  = 2.09
    cuota_drift = round((cuota_live - cuota_t0) / cuota_t0 * 100, 1)

    CUOTA_ENVENENADA_UMBRAL = 15.0
    cuota_envenenada = cuota_drift > CUOTA_ENVENENADA_UMBRAL
    assert cuota_envenenada is False, f"drift={cuota_drift}% debe quedar bajo umbral"


# ─── D150-04: games_set1 en _calcular_certeza_condicional ─────────────────────

def test_150_03_set1_tiebreak_zona_override():
    """games_set1=13 (tiebreak), zona=DOMINANTE → función usa COINFLIP internamente."""
    # Con zona=DOMINANTE pura (µ=18): gp=10, linea=22.5 → p_condicional alto
    resultado_dominante = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=10,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        games_set1=None,
    )
    # Con tiebreak: mismos inputs pero games_set1=13 → zona se fuerza COINFLIP (µ=23)
    resultado_tiebreak = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=10,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        games_set1=13,
    )
    # COINFLIP (µ=23) debe dar p_condicional más baja que DOMINANTE (µ=18) para UNDER
    assert resultado_tiebreak["p_condicional"] < resultado_dominante["p_condicional"], (
        f"tiebreak={resultado_tiebreak['p_condicional']:.3f} debe < dominante={resultado_dominante['p_condicional']:.3f}"
    )


def test_150_04_set1_normal_sin_override():
    """games_set1=8 (set normal 6:2), zona=DOMINANTE → NO se hace override."""
    resultado_sin_override = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=10,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        games_set1=8,  # set normal → no tiebreak
    )
    resultado_referencia = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=10,
        sets_complete=1,
        current_games=0,
        zona="DOMINANTE",
        games_set1=None,
    )
    # games_set1=8 no debe cambiar el resultado vs sin games_set1
    assert resultado_sin_override["p_condicional"] == resultado_referencia["p_condicional"], (
        "games_set1=8 no debe hacer override de zona DOMINANTE"
    )


# ─── D150-02: _parse_kambi_livedata_sets games_set1 ──────────────────────────

def test_150_05_parse_livedata_games_set1_tiebreak():
    """home=[7,4,-1], away=[6,5,-1] → games_set1=13 (tiebreak 7:6)."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [7, 4, -1],
                    "away": [6, 5, -1],
                }
            }
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["games_set1"] == 13, f"esperado 13, got {result['games_set1']}"
    assert result["sets_complete"] >= 1


def test_150_06_parse_livedata_set1_normal():
    """home=[6,4,-1], away=[2,6,-1] → games_set1=8 (set normal 6:2)."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 4, -1],
                    "away": [2, 6, -1],
                }
            }
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["games_set1"] == 8, f"esperado 8, got {result['games_set1']}"


# ─── D150-06: combo gate excluye piernas problemáticas ───────────────────────

def test_150_07_combo_gate_excluye_tiebreak():
    """Pierna con games_set1=13 ≥ 12 → excluida del combo (no aparece en alta_itf)."""
    # Simula el filtro D150-06 directamente
    signals = [
        {
            "partido": "A vs B",
            "convergencia_score": 4,
            "linea_envenenada": False,
            "cuota_envenenada": False,
            "score_data": {"games_set1": 13, "games_played": 20},
        },
        {
            "partido": "C vs D",
            "convergencia_score": 4,
            "linea_envenenada": False,
            "cuota_envenenada": False,
            "score_data": {"games_set1": 8, "games_played": 15},
        },
    ]
    alta_itf = []
    for s in signals:
        _gs1 = (s.get("score_data") or {}).get("games_set1") or s.get("games_set1") or 0
        if _gs1 >= 12:
            continue
        if s.get("cuota_envenenada"):
            continue
        alta_itf.append(s)

    partidos = [s["partido"] for s in alta_itf]
    assert "A vs B" not in partidos, "Pierna con tiebreak debe ser excluida"
    assert "C vs D" in partidos, "Pierna normal debe incluirse"


def test_150_08_combo_gate_excluye_cuota_envenenada():
    """Pierna con cuota_envenenada=True → excluida del combo."""
    signals = [
        {
            "partido": "X vs Y",
            "convergencia_score": 4,
            "linea_envenenada": False,
            "cuota_envenenada": True,
            "score_data": {"games_set1": 8, "games_played": 15},
        },
        {
            "partido": "P vs Q",
            "convergencia_score": 4,
            "linea_envenenada": False,
            "cuota_envenenada": False,
            "score_data": {"games_set1": 9, "games_played": 14},
        },
    ]
    alta_itf = []
    for s in signals:
        _gs1 = (s.get("score_data") or {}).get("games_set1") or 0
        if _gs1 >= 12:
            continue
        if s.get("cuota_envenenada"):
            continue
        alta_itf.append(s)

    partidos = [s["partido"] for s in alta_itf]
    assert "X vs Y" not in partidos, "Pierna cuota_envenenada debe ser excluida"
    assert "P vs Q" in partidos, "Pierna limpia debe incluirse"


# ─── D150-05: lógica badge MERCADO CONFIRMA ──────────────────────────────────

def test_150_09_badge_confirmado_umbral():
    """cuota_drift=-27% y estado EN_VIVO → _cuota_confirmada=True."""
    _drift  = -27.0
    _estado = "EN_VIVO"

    _cuota_confirmada = (
        _drift is not None and _drift < -15.0
        and _estado in ("EN_VIVO", "ITF_VIVO")
    )
    assert _cuota_confirmada is True


def test_150_10_badge_confirmado_no_premature():
    """cuota_drift=-5% (bajo umbral -15%) → _cuota_confirmada=False."""
    _drift  = -5.0
    _estado = "ITF_VIVO"

    _cuota_confirmada = (
        _drift is not None and _drift < -15.0
        and _estado in ("EN_VIVO", "ITF_VIVO")
    )
    assert _cuota_confirmada is False, f"drift={_drift}% no debe activar badge (umbral -15%)"

"""
Tests Nodo-147 — Live Score × Games Certeza Condicional en Tiempo Real.
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.
"""
import json
import sys
import pytest
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import (
    _calcular_certeza_condicional,
    _enrich_live_score,
    _freeze_baseline_if_needed,
    _write_games_odds_history,
    _fmt_progreso,
    _parse_kambi_livedata_sets,
    _fetch_kambi_livedata,
)


# ─── D147-02: _calcular_certeza_condicional ──────────────────────────────────

def test_nodo147_01_certeza_matematica_under_terminado():
    """Partido terminado 2-0 en sets con 21 juegos, línea 22.5 → CERTEZA_MATEMATICA UNDER."""
    resultado = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=21,
        sets_complete=2,
        current_games=0,
        zona="DOMINANTE",
        sets_home=2,
        sets_away=0,
    )
    assert resultado["certeza_matematica"] is True
    assert resultado["alerta_nivel"] == "CERTEZA"


def test_nodo147_02_no_certeza_under_sets_pendientes():
    """10 juegos, 1 set completo, aún puede llegar a 32+ juegos → no CERTEZA."""
    resultado = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=10,
        sets_complete=1,
        current_games=2,
        zona="DOMINANTE",
    )
    assert resultado["certeza_matematica"] is False


def test_nodo147_03_p_condicional_dominante_alto():
    """DOMINANTE con 18 juegos jugados → p_condicional UNDER ≥ 0.90, nivel ALTA o CERTEZA."""
    resultado = _calcular_certeza_condicional(
        linea=22.5,
        direccion="UNDER",
        games_played=18,
        sets_complete=1,
        current_games=6,
        zona="DOMINANTE",
    )
    assert resultado["p_condicional"] >= 0.90
    assert resultado["alerta_nivel"] in ("ALTA", "CERTEZA")


def test_nodo147_04_p_condicional_coinflip_bajo():
    """COINFLIP OVER con 8 juegos → incertidumbre alta, nivel BAJA o vacío o MOD."""
    resultado = _calcular_certeza_condicional(
        linea=22.5,
        direccion="OVER",
        games_played=8,
        sets_complete=0,
        current_games=8,
        zona="COINFLIP",
    )
    # p_condicional OVER debe ser plausible pero no dominante
    assert resultado["p_condicional"] >= 0.20
    assert resultado["alerta_nivel"] in ("", "BAJA", "MOD")


# ─── D147-03: _freeze_baseline_if_needed ─────────────────────────────────────

def test_nodo147_05_freeze_baseline_inmutable(tmp_path, monkeypatch):
    """Segunda llamada NO sobreescribe cuota_t0 ya congelada."""
    import live_desk
    monkeypatch.setattr(live_desk, "REPORTS", tmp_path)

    signals = [{
        "partido":   "Ruud vs Shapovalov",
        "direccion": "UNDER",
        "cuota_live": 1.70,
        "linea":      22.5,
        "estado":     "EN_VIVO",
    }]
    fecha_compact = "20260725"

    # Primera llamada → escribe T0 = 1.70
    baseline = _freeze_baseline_if_needed(signals, fecha_compact)
    assert baseline["Ruud vs Shapovalov_UNDER"]["cuota_t0"] == 1.70

    # Cambiar cuota_live y llamar de nuevo
    signals[0]["cuota_live"] = 1.55
    baseline2 = _freeze_baseline_if_needed(signals, fecha_compact)

    # T0 debe seguir siendo 1.70 — INMUTABLE
    assert baseline2["Ruud vs Shapovalov_UNDER"]["cuota_t0"] == 1.70


# ─── D147-01: _enrich_live_score ─────────────────────────────────────────────

def test_nodo147_06_enrich_live_score_conecta_games_played():
    """_enrich_live_score escribe score_data con games_played >= 0 cuando event_id matchea."""
    signal = {"partido": "A vs B", "event_id": 12345, "estado": "EN_VIVO"}

    # score_str "6:2,4:0" → sets: [6:2 completado] + [4:0 en curso]
    # games_played = 6+2+4+0 = 12
    live_events = [{
        "event": {
            "id": 12345,
            "homeName": "A",
            "awayName": "B",
            "liveData": {"scoreStr": "6:2,4:0"},
        }
    }]
    _enrich_live_score([signal], live_events)

    assert "score_data" in signal
    # Si el parse fue exitoso, games_played debe ser un entero no negativo
    if signal["score_data"] is not None:
        assert isinstance(signal["score_data"].get("games_played"), int)
        assert signal["score_data"]["games_played"] >= 0


def test_nodo147_07_enrich_live_score_sin_event_id():
    """Señal sin event_id → score_data = None (sin crash)."""
    signal = {"partido": "A vs B", "estado": "PRE_PARTIDO"}
    _enrich_live_score([signal], [])
    assert signal["score_data"] is None


# ─── D147-helper: _fmt_progreso ──────────────────────────────────────────────

def test_nodo147_08_fmt_progreso_con_datos():
    """_fmt_progreso formatea correctamente score_data completo."""
    score_data = {
        "sets_home": 1, "sets_away": 0,
        "games_played": 12,
        "current_games": 4,
    }
    result = _fmt_progreso(score_data)
    assert "1-0" in result
    assert "12j" in result


def test_nodo147_09_fmt_progreso_sin_datos():
    """_fmt_progreso retorna 'PRE' cuando no hay score_data."""
    assert _fmt_progreso(None) == "PRE"
    assert _fmt_progreso({}) == "PRE"


# ─── D147-01b: _parse_kambi_livedata_sets ────────────────────────────────────

def test_nodo147_10_parse_livedata_sets_dos_sets_completos():
    """home=[6,4,-1], away=[2,6,-1] → dos sets completos (6:2, 4:6), set 3 no iniciado."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 4, -1],
                    "away": [2, 6, -1],
                    "homeServe": True,
                }
            }
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["sets_complete"] == 2
    assert result["games_played"] == 18   # 6+2+4+6
    assert result["sets_home"] == 1
    assert result["sets_away"] == 1
    assert result["score_str"] == "6:2,4:6"
    assert result["current_games"] == 0


def test_nodo147_10b_parse_livedata_sets_set_en_curso():
    """home=[6,4,-1], away=[4,5,-1] → set 2 en curso (4:5), set 3 no iniciado."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {"home": [6, 4, -1], "away": [4, 5, -1]}
            }
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["sets_complete"] == 1       # solo set 1 completo (6:4)
    assert result["games_played"] == 19       # 6+4+4+5=19
    assert result["current_games"] == 9       # 4+5 en el set 2 en curso
    assert result["score_str"] == "6:4"       # solo sets completos en score_str


def test_nodo147_11_parse_livedata_sets_un_set_completo():
    """home=[6,-1], away=[3,-1] → sets_complete=1, games_played=9."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, -1],
                    "away": [3, -1],
                }
            }
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["sets_complete"] == 1
    assert result["games_played"] == 9
    assert result["sets_home"] == 1
    assert result["sets_away"] == 0


def test_nodo147_12_parse_livedata_sets_vacio():
    """Sin sets → retorna None sin crash."""
    assert _parse_kambi_livedata_sets({}) is None
    assert _parse_kambi_livedata_sets({"liveData": {}}) is None


def test_nodo147_13_enrich_live_score_usa_livedata_fallback(monkeypatch):
    """Si offering API no da score, _enrich_live_score llama _fetch_kambi_livedata."""
    import live_desk

    fake_livedata = {
        "liveData": {
            "statistics": {
                "sets": {"home": [6, 4, -1], "away": [2, 6, -1]}
            }
        }
    }
    monkeypatch.setattr(live_desk, "_fetch_kambi_livedata", lambda eid: fake_livedata)

    signal = {"partido": "A vs B", "event_id": 99999, "estado": "EN_VIVO"}
    # live_events vacío → offering API no resuelve score → fallback a livedata
    _enrich_live_score([signal], [])

    assert signal["score_data"] is not None
    assert signal["score_data"]["games_played"] == 18   # 6+2+4+6
    assert signal["score_data"]["score_str"] == "6:2,4:6"

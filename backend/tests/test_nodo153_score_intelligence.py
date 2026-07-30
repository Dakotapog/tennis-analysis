"""
Tests Nodo-153 — Score Intelligence: serve/break/current game (D153-01→D153-04).
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Gates cubiertos:
  D153-01: current_set_home/away extraídos desde statistics.sets.home/away (set en curso)
  D153-02: serving = home/away desde statistics.sets.homeServe
  D153-03: game_score = "30:15" desde liveData.score.home/away
  D153-04: break_situation = no-servidor lidera
"""
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_desk import _parse_kambi_livedata_sets, _fmt_progreso


# ── D153-01 + D153-02 + D153-03: Extracción livedata ─────────────────────────

def test_153_01_current_set_split_extraction():
    """D153-01: extrae current_set_home/away desde sets array (set en curso).

    Entrada: home=[7,1,2], away=[6,6,1]
    Parse: set1 7:6 completo, set2 1:6 completo, set3 2:1 en curso.
    Debe guardar current_set_home=2, current_set_away=1.
    """
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [7, 1, 2],
                    "away": [6, 6, 1],
                    "homeServe": True,
                }
            },
            "score": {"home": "30", "away": "15"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result is not None
    assert result["current_set_home"] == 2, f"Expected 2, got {result['current_set_home']}"
    assert result["current_set_away"] == 1, f"Expected 1, got {result['current_set_away']}"
    assert result["score_str"] == "7:6,1:6", "Sets completados mal parseados"


def test_153_02_serving_home():
    """D153-02: extrae serving='home' desde homeServe=True."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 3, 1],
                    "away": [4, 2, 0],
                    "homeServe": True,
                }
            },
            "score": {"home": "15", "away": "0"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["serving"] == "home", f"Expected 'home', got {result['serving']}"


def test_153_02_serving_away():
    """D153-02: extrae serving='away' desde homeServe=False."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 1, 2],
                    "away": [4, 6, 1],
                    "homeServe": False,
                }
            },
            "score": {"home": "0", "away": "30"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["serving"] == "away", f"Expected 'away', got {result['serving']}"


def test_153_03_game_score_extraction():
    """D153-03: extrae game_score desde liveData.score (ej: '30:15')."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [7, 3, 2],
                    "away": [6, 1, 1],
                    "homeServe": True,
                }
            },
            "score": {"home": "40", "away": "15"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["game_score"] == "40:15", f"Expected '40:15', got {result['game_score']}"


def test_153_03_game_score_zero_ignored():
    """D153-03: score '0:0' devuelve game_score=None (inicio de game)."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 2, 1],
                    "away": [4, 3, 0],
                    "homeServe": True,
                }
            },
            "score": {"home": "0", "away": "0"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["game_score"] is None, f"0:0 debe ser None, got {result['game_score']}"


# ── D153-04: Break Situation ───────────────────────────────────────────────────

def test_153_04_break_situation_home_serve_away_leads():
    """D153-04: break_situation=True cuando away lidera y home sirve.

    Set3: home sirve (homeServe=True), marcador 1:3 → away lidera.
    """
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 1, 1],
                    "away": [4, 6, 3],
                    "homeServe": True,
                }
            },
            "score": {"home": "30", "away": "15"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["break_situation"] is True, "Away lidera 3:1 con home sirviendo = break"


def test_153_04_break_situation_away_serve_home_leads():
    """D153-04: break_situation=True cuando home lidera y away sirve."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 2, 3],
                    "away": [4, 6, 1],
                    "homeServe": False,
                }
            },
            "score": {"home": "40", "away": "0"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["break_situation"] is True, "Home lidera 3:1 con away sirviendo = break"


def test_153_04_no_break_situation_holds():
    """D153-04: break_situation=False cuando el puntaje es el esperado por paridad.

    2:1◄ (away sirve, N=3 impar) → first_server=home → esperado home+1 → real=+1 → hold.
    Home ganó G1 (su saque) y G3 (saque de away = break... wait no:
    home=[6,2,2], away=[4,6,1] → set3: home=2, away=1, away sirve, N=3.
    N=3 impar, current=away → first_server=home → esperado=+1, real=+1 → NO quiebre ✓
    Traza sin quiebre: G1 home sirve → home gana (1:0), G2 away sirve → away gana (1:1),
    G3 home sirve → home gana (2:1), G4 away sirve ahora (◄).
    """
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 2, 2],
                    "away": [4, 6, 1],
                    "homeServe": False,   # away sirve ahora (G4), N=3 impar → home sirvió G1
                }
            },
            "score": {"home": "0", "away": "15"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["break_situation"] is False, (
        "2:1◄ N=3: home sirvió primero, esperado home+1 → real=+1 → HOLD (no quiebre)"
    )


def test_153_04_no_break_equal_score():
    """D153-04: break_situation=False cuando score está igualado."""
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 2, 3],
                    "away": [4, 6, 3],
                    "homeServe": True,
                }
            },
            "score": {"home": "30", "away": "30"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["break_situation"] is False, "3:3 igualado, no break"


def test_153_04_no_break_score_0_1_home_serves():
    """D153-04: ►0:1 | N=1 → NO quiebre (bug del usuario: Charlie Pade vs Scott Jones).

    Away sirvió G1 (N=1 impar, home sirve ahora → first_server=away).
    Away ganó su saque = HOLD. Esperado: away+1 → 0:1. Real=0:1 → SIN quiebre.
    """
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 0],
                    "away": [4, 1],
                    "homeServe": True,   # home sirve G2 → away sirvió G1
                }
            },
            "score": {"home": "0", "away": "0"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["current_set_home"] == 0
    assert result["current_set_away"] == 1
    assert result["break_situation"] is False, (
        f"►0:1 N=1: away sirvió G1 y ganó = HOLD, no quiebre. "
        f"got break_situation={result['break_situation']}"
    )


def test_153_04_real_break_diff_one_parity():
    """D153-04: ►2:1 | N=3 → QUIEBRE real con diff=1 (paridad confirma).

    Away sirvió G1 (N=3 impar, home sirve ahora → first_server=away).
    Sin quiebre esperado: away+1 → home:away = 1:2. Real = 2:1 → QUIEBRE.
    Traza: G1 away sirve → home gana (BREAK 1:0), G2 home sirve → home gana (hold 2:0),
           G3 away sirve → away gana (hold 2:1), G4 home sirve ahora (►).
    """
    livedata = {
        "liveData": {
            "statistics": {
                "sets": {
                    "home": [6, 2],
                    "away": [4, 1],
                    "homeServe": True,   # home sirve G4 → away sirvió G1
                }
            },
            "score": {"home": "15", "away": "0"}
        }
    }
    result = _parse_kambi_livedata_sets(livedata)
    assert result["current_set_home"] == 2
    assert result["current_set_away"] == 1
    assert result["break_situation"] is True, (
        f"►2:1 N=3: esperado 1:2 (away first), real=2:1 → QUIEBRE. "
        f"got break_situation={result['break_situation']}"
    )


# ── _fmt_progreso display (D153) ───────────────────────────────────────────────

def test_153_fmt_progreso_serve_arrow_home():
    """_fmt_progreso: muestra ► cuando home sirve."""
    score_data = {
        "score_str": "7:6,1:6",
        "current_set_home": 2,
        "current_set_away": 2,
        "serving": "home",
        "game_score": "15:30",
        "games_played": 26,
        "break_situation": False,
    }
    result = _fmt_progreso(score_data)
    assert "►2:2" in result, f"Expected ►2:2 in '{result}'"
    assert "[15:30]" in result, f"Expected [15:30] in '{result}'"


def test_153_fmt_progreso_serve_arrow_away():
    """_fmt_progreso: muestra ◄ cuando away sirve."""
    score_data = {
        "score_str": "6:4,2:6",
        "current_set_home": 1,
        "current_set_away": 2,
        "serving": "away",
        "game_score": "40:15",
        "games_played": 20,
        "break_situation": False,
    }
    result = _fmt_progreso(score_data)
    assert "1:2◄" in result, f"Expected 1:2◄ in '{result}'"
    assert "[40:15]" in result, f"Expected [40:15] in '{result}'"


def test_153_fmt_progreso_break_indicator():
    """_fmt_progreso: muestra QUIEBRE cuando break_situation=True."""
    score_data = {
        "score_str": "3:6",
        "current_set_home": 1,
        "current_set_away": 4,
        "serving": "home",
        "game_score": "30:0",
        "games_played": 14,
        "break_situation": True,
    }
    result = _fmt_progreso(score_data)
    assert "QUIEBRE" in result, f"Expected QUIEBRE in '{result}'"

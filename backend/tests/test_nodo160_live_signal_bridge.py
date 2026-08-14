"""
tests/test_nodo160_live_signal_bridge.py — REGLA-T53 Nodo-160 D160-05

Invoca la función real reconciliar_senales_partido() — nunca hardcodea la
clasificación. Casos: convergencia fuerte, divergencia (drift contra el
favorito pese a games dominante), y neutro (datos insuficientes en
cualquiera de los dos lados).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.live_signal_bridge import reconciliar_senales_partido as reconciliar


_GAMES_DOMINANTE = {
    "direccion": "UNDER", "certeza_matematica": True, "p_condicional": 0.92,
    "zona": "DOMINANTE", "break_situation": True, "serving": "home",
}


def test_160_60_convergencia_fuerte():
    winner_state = {"score_directo": 4, "break_state": "BREAK_CONFIRMADO", "drift_pct": -0.10,
                     "direccion_favorito": "home"}
    r = reconciliar("P1", _GAMES_DOMINANTE, winner_state)
    assert r["estado"] == "CONVERGENCIA_FUERTE"


def test_160_61_divergencia_drift_contra_favorito():
    winner_state = {"score_directo": 4, "break_state": "NORMAL", "drift_pct": 0.12,
                     "direccion_favorito": "home"}
    r = reconciliar("P1", _GAMES_DOMINANTE, winner_state)
    assert r["estado"] == "DIVERGENCIA"


def test_160_62_neutro_games_sin_certeza():
    games_state = {"direccion": "UNDER", "certeza_matematica": False, "zona": "COINFLIP",
                    "break_situation": False, "serving": "home"}
    winner_state = {"score_directo": 4, "break_state": "BREAK_CONFIRMADO", "drift_pct": -0.10}
    r = reconciliar("P1", games_state, winner_state)
    assert r["estado"] == "NEUTRO"


def test_160_63_neutro_score_directo_bajo():
    winner_state = {"score_directo": 1, "break_state": "BREAK_CONFIRMADO", "drift_pct": -0.10}
    r = reconciliar("P1", _GAMES_DOMINANTE, winner_state)
    assert r["estado"] == "NEUTRO"


def test_160_64_neutro_break_state_normal_sin_drift():
    winner_state = {"score_directo": 4, "break_state": "NORMAL", "drift_pct": 0.0}
    r = reconciliar("P1", _GAMES_DOMINANTE, winner_state)
    assert r["estado"] == "NEUTRO"


def test_160_65_neutro_estados_vacios_no_lanza():
    r = reconciliar("P1", {}, {})
    assert r["estado"] == "NEUTRO"

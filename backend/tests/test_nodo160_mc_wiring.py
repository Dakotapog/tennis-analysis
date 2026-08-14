"""
tests/test_nodo160_mc_wiring.py — REGLA-T53 Nodo-160 D160-02 (wiring)

_attach_mc_conditional() invoca la función real del módulo
(simular_total_juegos_condicionado + estimar_p_hold) — nunca hardcodea la
fórmula. Verifica: (1) mutación in-place con datos suficientes, (2) skip
silencioso sin score_data, (3) skip silencioso con score_data incompleto.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


_H2H_IDX = {
    "home": {"j1": "home", "j2": "away", "ranking1": 30, "ranking2": 500, "superficie": "Dura"},
    "away": {"j1": "home", "j2": "away", "ranking1": 30, "ranking2": 500, "superficie": "Dura"},
}


def test_160_50_attach_mc_muta_signals_con_datos_suficientes():
    sig = {
        "partido": "Home vs Away",
        "direccion": "UNDER",
        "linea": 22.5,
        "score_data": {
            "serving": "home", "current_set_home": 3, "current_set_away": 3,
            "sets_home": 0, "sets_away": 1, "games_played": 10,
        },
    }
    with patch("live_desk._load_h2h_index_for_games", return_value=_H2H_IDX):
        ld._attach_mc_conditional([sig], "20260801")

    assert "mc_p_condicional" in sig
    assert 0.0 <= sig["mc_p_condicional"] <= 1.0
    assert sig["mc_media_total_juegos"] is not None


def test_160_51_attach_mc_skip_sin_score_data():
    sig = {"partido": "Home vs Away", "direccion": "UNDER", "linea": 22.5}
    with patch("live_desk._load_h2h_index_for_games", return_value=_H2H_IDX):
        ld._attach_mc_conditional([sig], "20260801")

    assert "mc_p_condicional" not in sig


def test_160_52_attach_mc_skip_score_data_incompleto():
    sig = {
        "partido": "Home vs Away",
        "direccion": "UNDER",
        "linea": 22.5,
        "score_data": {"serving": "home", "games_played": 10},  # faltan sets/current_set
    }
    with patch("live_desk._load_h2h_index_for_games", return_value=_H2H_IDX):
        ld._attach_mc_conditional([sig], "20260801")

    assert "mc_p_condicional" not in sig


def test_160_53_resolve_player_rankings_asigna_por_j1_j2():
    r_home, r_away, sup = ld._resolve_player_rankings("Home", "Away", _H2H_IDX)
    assert r_home == 30
    assert r_away == 500
    assert sup == "Dura"

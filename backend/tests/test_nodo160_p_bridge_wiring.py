"""
tests/test_nodo160_p_bridge_wiring.py — REGLA-T53 Nodo-160 D160-05 (wiring)

_build_p_bridge() invoca la función real reconciliar_senales_partido() —
nunca hardcodea la clasificación. Verifica que el join games_live↔p2_break↔
p3_convergence produce el estado correcto y que NEUTRO se filtra del output
(REPORTE_SOLO: solo se muestra evidencia accionable, no ruido).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


_GAMES_DOMINANTE_SIGNAL = {
    "partido": "Home vs Away",
    "direccion": "UNDER",
    "zona": "DOMINANTE",
    "certeza": {"certeza_matematica": True, "p_condicional": 0.92},
    "score_data": {"serving": "home", "break_situation": True},
}


def _state_with(breaks=None, picks=None):
    return {
        "p2_break": {"breaks": breaks or []},
        "p3_convergence": {"picks": picks or []},
    }


def test_160_70_convergencia_fuerte_end_to_end():
    state = _state_with(
        breaks=[{"partido": "Home vs Away", "jugador": "Home", "estado": "BREAK_CONFIRMADO", "drift_pct": -10.0}],
        picks=[{"jugador": "Home", "score_directo": 4}],
    )
    with patch("live_desk._load_json", return_value={"signals_alta": [_GAMES_DOMINANTE_SIGNAL]}):
        result = ld._build_p_bridge("2026-08-02", state)

    reconciliaciones = result["reconciliaciones"]
    assert len(reconciliaciones) == 1
    assert reconciliaciones[0]["estado"] == "CONVERGENCIA_FUERTE"
    assert reconciliaciones[0]["partido_key"] == "Home vs Away"
    assert reconciliaciones[0]["favorito"] == "Home"
    assert result["por_partido"]["Home vs Away"]["estado"] == "CONVERGENCIA_FUERTE"


def test_160_71_neutro_filtrado_del_output():
    sig = dict(_GAMES_DOMINANTE_SIGNAL)
    sig["zona"] = "COINFLIP"  # rompe games_dominante -> NEUTRO
    state = _state_with(
        breaks=[{"partido": "Home vs Away", "jugador": "Home", "estado": "BREAK_CONFIRMADO", "drift_pct": -10.0}],
        picks=[{"jugador": "Home", "score_directo": 4}],
    )
    with patch("live_desk._load_json", return_value={"signals_alta": [sig]}):
        result = ld._build_p_bridge("2026-08-02", state)

    assert result["reconciliaciones"] == []
    assert result["por_partido"]["Home vs Away"]["estado"] == "NEUTRO"


def test_160_72_sin_match_en_p2_break_degrada_neutro_y_se_filtra():
    state = _state_with(breaks=[], picks=[])
    with patch("live_desk._load_json", return_value={"signals_alta": [_GAMES_DOMINANTE_SIGNAL]}):
        result = ld._build_p_bridge("2026-08-02", state)

    assert result["reconciliaciones"] == []


def test_160_73_sin_reporte_games_live_no_lanza():
    state = _state_with()
    with patch("live_desk._load_json", return_value=None):
        result = ld._build_p_bridge("2026-08-02", state)

    assert result == {"reconciliaciones": [], "por_partido": {}}

"""
tests/test_nodo160_x3_visibility_wiring.py — REGLA-T53 Nodo-160 (wiring visibilidad X3)

_build_x3_games() debe propagar mc_p_condicional/mc_media_total_juegos y
steam_z/steam_signal/steam_confirmado desde games_live_{fecha}.json hasta las
señales que consume el render del panel X3 — en ambas rutas: señales que
también están en games_signal_report (ruta "confirmada") y señales ITF_VIVO
inyectadas directo desde games_live (ruta D-ITF-LIVE-01). Antes de este fix
ambos campos se calculaban/persistían pero nunca llegaban a `signals`, dejando
la evidencia D160-02/D160-03 invisible en el dashboard pese a estar wireada.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


def _write_gsr(reports_dir, fecha_compact, partido):
    gsr = reports_dir / f"games_signal_report_{fecha_compact}.json"
    gsr.write_text(json.dumps({
        "metadata": {"n_partidos": 1},
        "apostar": [{
            "partido": partido,
            "hora": "2026-08-02T10:00:00Z",
            "games_range": "20-24",
            "señales_optimas": [{
                "apostar": True, "mercado": "Total de juegos", "direccion": "UNDER",
                "linea": 22.5, "cuota": 1.90, "gap_juegos": 2.0, "confianza_señal": "ALTA",
            }],
        }],
    }), encoding="utf-8")


def _write_games_live(reports_dir, fecha_compact, partido, extra):
    gl = reports_dir / f"games_live_{fecha_compact}.json"
    sig = {
        "partido": partido, "estado": "EN_VIVO", "cuota_live": 1.80, "drift_pct": -5.0,
        "score_data": {"games_played": 12}, "certeza": {},
    }
    sig.update(extra)
    gl.write_text(json.dumps({
        "signals_alta": [sig], "en_vivo_count": 1, "convergencia_activa": False,
    }), encoding="utf-8")


def test_160_80_mc_y_steam_propagan_ruta_confirmada(tmp_path):
    partido = "Home vs Away"
    _write_gsr(tmp_path, "20260802", partido)
    _write_games_live(tmp_path, "20260802", partido, {
        "mc_p_condicional": 0.62, "mc_media_total_juegos": 21.3,
        "steam_z": 2.4, "steam_signal": "ACELERANDO", "steam_confirmado": True,
    })
    with patch.object(ld, "REPORTS", tmp_path):
        result = ld._build_x3_games("2026-08-02")

    sig = result["signals"][0]
    assert sig["mc_p_condicional"] == 0.62
    assert sig["mc_media_total_juegos"] == 21.3
    assert sig["steam_confirmado"] is True
    assert sig["steam_z"] == 2.4


def test_160_81_mc_y_steam_propagan_ruta_itf_vivo(tmp_path):
    partido = "Solo vs EnGamesLive"
    gl = tmp_path / "games_live_20260802.json"
    gl.write_text(json.dumps({
        "signals_alta": [{
            "partido": partido, "estado": "ITF_VIVO", "direccion": "UNDER", "linea": 22.5,
            "cuota_pre": 1.90, "cuota_live": 1.80, "gap": 2.0, "confianza": "ALTA",
            "mc_p_condicional": 0.58, "mc_media_total_juegos": 20.1,
            "steam_z": 1.1, "steam_signal": "NEUTRO", "steam_confirmado": False,
        }],
        "en_vivo_count": 1, "convergencia_activa": False,
    }), encoding="utf-8")
    with patch.object(ld, "REPORTS", tmp_path):
        result = ld._build_x3_games("2026-08-02")

    sig = [s for s in result["signals"] if s["partido"] == partido][0]
    assert sig["mc_p_condicional"] == 0.58
    assert sig["steam_confirmado"] is False
    assert sig["steam_z"] == 1.1


def test_160_82_sin_mc_ni_steam_no_lanza(tmp_path):
    partido = "Home vs Away"
    _write_gsr(tmp_path, "20260802", partido)
    _write_games_live(tmp_path, "20260802", partido, {})
    with patch.object(ld, "REPORTS", tmp_path):
        result = ld._build_x3_games("2026-08-02")

    sig = result["signals"][0]
    assert sig.get("mc_p_condicional") is None
    assert sig.get("steam_confirmado") is False


def test_160_83_linea_actual_cuota_actual_propagan_ruta_confirmada(tmp_path):
    """D158-01 fix: linea_actual/cuota_actual se calculaban y persistían en
    games_live_*.json (_check_games_convergencia L4307-4331) pero _build_x3_games
    nunca los copiaba a `signals` — columnas LínAct/CuotaAct quedaban en '—' pese
    a tener dato real en el JSON. Verificado en producción 2026-08-02: Duckworth
    J. vs O'Connell C. tenía linea_actual=24.5/cuota_actual=1.98 en games_live
    pero la fila renderizada mostraba '—' en ambas columnas."""
    partido = "Home vs Away"
    _write_gsr(tmp_path, "20260802", partido)
    _write_games_live(tmp_path, "20260802", partido, {
        "linea_actual": 24.5, "cuota_actual": 1.98, "linea_drift": 3.0,
        "oc_id_actual": 4282176076,
    })
    with patch.object(ld, "REPORTS", tmp_path):
        result = ld._build_x3_games("2026-08-02")

    sig = result["signals"][0]
    assert sig["linea_actual"] == 24.5
    assert sig["cuota_actual"] == 1.98
    assert sig["linea_drift"] == 3.0
    assert sig["oc_id_actual"] == 4282176076


def test_160_84_linea_actual_cuota_actual_propagan_ruta_itf_vivo(tmp_path):
    partido = "Solo vs EnGamesLive"
    gl = tmp_path / "games_live_20260802.json"
    gl.write_text(json.dumps({
        "signals_alta": [{
            "partido": partido, "estado": "ITF_VIVO", "direccion": "UNDER", "linea": 22.5,
            "cuota_pre": 1.90, "cuota_live": 1.80, "gap": 2.0, "confianza": "ALTA",
            "linea_actual": 21.5, "cuota_actual": 1.75, "oc_id_actual": 999,
        }],
        "en_vivo_count": 1, "convergencia_activa": False,
    }), encoding="utf-8")
    with patch.object(ld, "REPORTS", tmp_path):
        result = ld._build_x3_games("2026-08-02")

    sig = [s for s in result["signals"] if s["partido"] == partido][0]
    assert sig["linea_actual"] == 21.5
    assert sig["cuota_actual"] == 1.75
    assert sig["oc_id_actual"] == 999

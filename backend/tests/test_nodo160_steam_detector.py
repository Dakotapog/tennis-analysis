"""
tests/test_nodo160_steam_detector.py — REGLA-T53 Nodo-160 D160-03

D160-03: velocity_zscore() (analysis/velocity_monitor.py, Nodo-71) estaba
completa pero huérfana (REPORTE_SOLO, H52-05 nunca conectada). Este test
verifica que _write_games_odds_history() en live_desk.py invoca la función
real y anota steam_confirmado/steam_z en el dict de la señal — sin tocar
ningún gate de disparo (campo puramente informativo).
"""
import json
import sys
from datetime import datetime as _real_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


class _FixedDatetime(_real_datetime):
    """Congela datetime.now() para que el punto nuevo caiga en 10:20 exacto,
    manteniendo el espaciado de 5 min usado para calibrar el fixture de abajo."""
    @classmethod
    def now(cls, tz=None):
        return _real_datetime(2026, 8, 2, 10, 20)


def test_160_20_steam_confirmado_con_caida_fuerte(tmp_path, monkeypatch):
    """4+ puntos (3 velocidades → std real) con caída final brusca → steam_confirmado=True.

    velocity_zscore necesita >=2 velocidades de referencia (>=3 puntos previos)
    para calcular std — con solo 3 puntos totales z_last siempre es None
    (1 sola velocidad de referencia, sin varianza). Por eso el fixture aquí
    usa 4 puntos previos con ligera varianza + 1 punto nuevo con caída fuerte.
    """
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    monkeypatch.setattr(ld, "datetime", _FixedDatetime)
    fecha_compact = "20260802"

    hist_path = tmp_path / f"games_odds_history_{fecha_compact}.json"
    pk = "A vs B_UNDER"
    hist_path.write_text(json.dumps({
        pk: [
            {"ts": "10:00", "cuota": 1.95, "games_played": 2},
            {"ts": "10:05", "cuota": 1.94, "games_played": 4},
            {"ts": "10:10", "cuota": 1.955, "games_played": 6},
            {"ts": "10:15", "cuota": 1.94, "games_played": 7},
        ]
    }), encoding="utf-8")

    sig = {
        "partido": "A vs B", "direccion": "UNDER", "estado": "EN_VIVO",
        "cuota_live": 1.55, "score_data": {"games_played": 8},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    assert sig.get("steam_z") is not None
    assert sig["steam_confirmado"] is True
    assert sig["steam_signal"] == "STEAM"


def test_160_21_sin_historial_suficiente_no_anota_steam(tmp_path, monkeypatch):
    """Menos de 3 puntos totales → no se anota steam_confirmado (sin crash)."""
    monkeypatch.setattr(ld, "REPORTS", tmp_path)
    fecha_compact = "20260802"

    sig = {
        "partido": "C vs D", "direccion": "OVER", "estado": "EN_VIVO",
        "cuota_live": 1.90, "score_data": {"games_played": 3},
    }
    ld._write_games_odds_history([sig], fecha_compact)

    assert "steam_confirmado" not in sig

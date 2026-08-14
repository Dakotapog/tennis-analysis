"""
tests/test_nodo160_winner_market_wiring.py — REGLA-T53 Nodo-160 D160-01

D160-01: live_edge_monitor.run() estaba huérfano (ningún proceso lo invocaba).
_winner_market_refresh() en live_desk.py lo conecta al thread daemon del
servicio tennis-live-desk. Test verifica que el loop realmente llama a
live_edge_monitor.run() con los kwargs correctos — no mockea el resultado,
invoca la función real (patchea solo time.sleep para cortar el while True).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


def test_160_01_winner_market_refresh_calls_live_edge_monitor_run():
    calls = []

    def _fake_run(reports_dir=None, ahora=None):
        calls.append({"reports_dir": reports_dir, "ahora": ahora})
        raise KeyboardInterrupt  # corta el while True tras la 1ra iteración real

    with patch("time.sleep", return_value=None), \
         patch.dict(sys.modules, {"live_edge_monitor": type(sys)("live_edge_monitor")}):
        sys.modules["live_edge_monitor"].run = _fake_run
        try:
            ld._winner_market_refresh(lambda: "2026-08-02")
        except KeyboardInterrupt:
            pass

    assert len(calls) == 1
    assert calls[0]["reports_dir"] == "reports"
    assert calls[0]["ahora"] is not None

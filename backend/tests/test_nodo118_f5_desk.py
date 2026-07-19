"""
REGLA-T53: Tests Nodo-118 F5 — Panel DATA en live_desk + embudo crosswalk.

Cubre: _build_data_panel() retorna campos cobertura_pct + fuga_nominal cuando
el ledger existe; HTML del desk contiene el panel DATA con el % de cobertura.
"""

import json
from pathlib import Path

from scraping.match_ledger import fusionar_dia
from live_desk import _build_data_panel, build_desk_state, render_html


def _make_ledger(tmp_path):
    """Crea un ledger real vía fusionar_dia() en tmp_path/data (donde el desk lo busca)."""
    import core.player_registry as _mod
    _mod._CROSSWALK_FILE = tmp_path / "crosswalk.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    kambi = [
        {"jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek",
         "cuota1": 1.27, "cuota2": 3.50, "hora": "14:00",
         "torneo_nombre": "Wimbledon", "superficie": "grass",
         "match_id": "abc123", "kambi_event_id": "K001"},
        {"jugador1": "Carlos Alcaraz", "jugador2": "Novak Djokovic",
         "cuota1": 1.55, "cuota2": 2.40, "hora": "16:00",
         "torneo_nombre": "Wimbledon", "superficie": "grass",
         "match_id": None, "kambi_event_id": "K002"},
    ]
    fs = [
        {"jugador1": "P. Badosa", "jugador2": "T. Zidansek",
         "cuota1": None, "match_id": "abc123", "match_url": "https://x.com/1",
         "hora_partido": "14:00", "superficie": "grass"},
    ]
    fusionar_dia(kambi, fs, "2026-07-19", output_dir=str(data_dir))


class TestDataPanelF5:

    def test_build_data_panel_sin_ledger_retorna_no_disponible(self, tmp_path, monkeypatch):
        """
        REGLA-T53: _build_data_panel() sin ledger retorna disponible=False (no crash).
        """
        import live_desk as ld
        monkeypatch.setattr(ld, "BASE_DIR", tmp_path)
        result = _build_data_panel("2099-01-01")
        assert result["disponible"] is False

    def test_build_data_panel_directo(self, tmp_path, monkeypatch):
        """
        REGLA-T53: _build_data_panel() con ledger real retorna campos requeridos.
        """
        import live_desk as ld
        monkeypatch.setattr(ld, "BASE_DIR", tmp_path)

        _make_ledger(tmp_path)

        result = _build_data_panel("2026-07-19")

        assert result["disponible"] is True, f"Debe estar disponible: {result}"
        assert "cobertura_pct" in result, "Debe tener cobertura_pct"
        assert 0.0 <= result["cobertura_pct"] <= 100.0
        assert "fuga_nominal" in result
        assert isinstance(result["fuga_nominal"], list)
        assert result["joins"] >= 1, "Debe haber al menos 1 join (match_id abc123)"

    def test_render_html_contiene_panel_data(self, tmp_path, monkeypatch):
        """
        REGLA-T53: render_html() incluye el panel DATA con cobertura_pct visible.
        """
        import live_desk as ld
        monkeypatch.setattr(ld, "BASE_DIR", tmp_path)

        _make_ledger(tmp_path)

        state = {"fecha": "2026-07-19", "ts": "2026-07-19T10:00:00",
                 "p0_ncal": {}, "p4_risk": {}, "p1_tape": {}, "p2_break": {},
                 "p3_convergence": {}, "p5_execution": {}, "p6_pnl": {},
                 "p7_clock": {}, "p8_books": {}, "p9_que_falta": [],
                 "p10_odds_history": {}, "p11_combo_live": {}, "p12_conformal": {},
                 "p_data": _build_data_panel("2026-07-19")}

        html = render_html(state)

        assert "DATA" in html, "HTML debe contener panel DATA"
        assert "Embudo" in html or "crosswalk" in html.lower() or "Crosswalk" in html, \
            "HTML debe mencionar crosswalk/embudo"
        assert "cobertura" in html.lower() or "%" in html, \
            "HTML debe mostrar porcentaje de cobertura"

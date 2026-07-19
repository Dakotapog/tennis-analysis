"""
REGLA-T53: Tests Nodo-118 F3 — Bootstrap retroactivo del crosswalk.

Cubre: extrae par verificado de fixture histórico real (zita con cuotas + sin cuotas);
bootstrap_desde_edge_reports popula crosswalk; reporte de cobertura incluye pct.
"""

import json
import pytest
from pathlib import Path

from core.player_registry import PlayerRegistry, normalize_player_name


def _make_reg():
    return PlayerRegistry(normalize_fn=normalize_player_name)


def _make_zita_con_cuotas(tmp_path, nombre="api.json") -> Path:
    data = {
        "Wimbledon": [
            {"jugador1": "P. Badosa", "jugador2": "T. Zidansek",
             "cuota1": 1.27, "cuota2": 3.50, "hora": "14:00",
             "torneo_nombre": "Wimbledon"},
            {"jugador1": "C. Alcaraz", "jugador2": "N. Djokovic",
             "cuota1": 1.55, "cuota2": 2.40, "hora": "16:00",
             "torneo_nombre": "Wimbledon"},
        ]
    }
    p = tmp_path / nombre
    p.write_text(json.dumps(data))
    return p


def _make_zita_sin_cuotas(tmp_path, nombre="playwright.json") -> Path:
    data = [
        {"jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek",
         "cuota1": None, "match_url": "https://x.com/1", "match_id": "abc",
         "hora": "14:00", "torneo_fs": "Wimbledon"},
        {"jugador1": "Carlos Alcaraz", "jugador2": "Novak Djokovic",
         "cuota1": None, "match_url": "https://x.com/2", "match_id": "def",
         "hora": "16:00", "torneo_fs": "Wimbledon"},
    ]
    p = tmp_path / nombre
    p.write_text(json.dumps(data))
    return p


class TestBootstrapF3:

    def test_bootstrap_zita_extrae_joins_verificados(self, tmp_path, monkeypatch):
        """
        REGLA-T53: bootstrap_desde_zita() con par API+Playwright del mismo día
        llama a fusionar_dia() y registra los joins como VERIFIED en el crosswalk.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        # Crear archivos zita del mismo día
        (tmp_path / "data").mkdir()
        api_file = tmp_path / "data" / "zita_tennis_matches_20260718_090000.json"
        pw_file = tmp_path / "data" / "zita_tennis_matches_20260718_085932.json"

        _make_zita_con_cuotas(tmp_path / "data", "zita_tennis_matches_20260718_090000.json")
        _make_zita_sin_cuotas(tmp_path / "data", "zita_tennis_matches_20260718_085932.json")

        # Monkeypatch glob para apuntar a tmp_path/data
        import glob as _glob
        orig_glob = _glob.glob
        def mock_glob(pattern):
            if "zita_tennis_matches" in pattern:
                return [str(api_file), str(pw_file)]
            return orig_glob(pattern)

        from scripts import build_crosswalk_bootstrap as bsmod
        monkeypatch.setattr(bsmod, "glob", type("g", (), {"glob": staticmethod(mock_glob)})())

        # Usar patch más simple: monkeypatch glob.glob directamente
        monkeypatch.setattr("scripts.build_crosswalk_bootstrap.glob.glob", mock_glob)

        reg = _make_reg()
        stats = bsmod.bootstrap_desde_zita(reg, dry_run=False)

        assert stats["fechas_con_pares"] >= 1, \
            f"Debe encontrar al menos 1 fecha con par API+Playwright: {stats}"

    def test_bootstrap_edge_reports_popula_crosswalk(self, tmp_path, monkeypatch):
        """
        REGLA-T53: bootstrap_desde_edge_reports() extrae favorito_predicho de
        los edge_reports y los registra como VERIFIED en el crosswalk.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        # Crear edge_report sintético
        edge = {
            "apostar": [
                {"favorito_predicho": "Paula Badosa", "partido": "Paula Badosa vs Tamara Zidansek",
                 "cuota_favorito": 1.27},
            ],
            "watchlist": [
                {"favorito_predicho": "Carlos Alcaraz", "partido": "Carlos Alcaraz vs Novak Djokovic",
                 "cuota_favorito": 1.55},
            ]
        }
        edge_file = tmp_path / "edge_report_20260718.json"
        edge_file.write_text(json.dumps(edge))

        from scripts import build_crosswalk_bootstrap as bsmod
        monkeypatch.setattr("scripts.build_crosswalk_bootstrap.glob.glob",
                            lambda p: [str(edge_file)] if "edge_report" in p else [])

        reg = _make_reg()
        stats = bsmod.bootstrap_desde_edge_reports(reg, dry_run=False)

        assert stats["picks"] >= 2, f"Debe procesar 2 picks: {stats}"
        assert stats["aliases_nuevos"] >= 2, \
            f"Debe crear al menos 2 aliases (Badosa + Alcaraz): {stats}"

        # Verificar que resolve_crosswalk encuentra los jugadores
        result = reg.resolve_crosswalk("Paula Badosa")
        assert result is not None, "Paula Badosa debe estar en el crosswalk"

    def test_estimar_cobertura_retorna_pct(self, tmp_path, monkeypatch):
        """
        REGLA-T53: estimar_cobertura() retorna dict con cobertura_estimada_pct.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "player_crosswalk.json"

        # Crear archivo zita con jugadores
        zita = [
            {"jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek"},
            {"jugador1": "Carlos Alcaraz", "jugador2": "Unknown Player"},
        ]
        zita_file = tmp_path / "zita_tennis_matches_20260718_090000.json"
        zita_file.write_text(json.dumps(zita))

        from scripts import build_crosswalk_bootstrap as bsmod
        monkeypatch.setattr("scripts.build_crosswalk_bootstrap.glob.glob",
                            lambda p: [str(zita_file)] if "zita_tennis_matches" in p else [])

        reg = _make_reg()
        # Registrar solo Badosa
        reg.add_alias("Paula Badosa", "Paula Badosa", source="test", confidence="VERIFIED")

        cov = bsmod.estimar_cobertura(reg, n_dias=7)

        assert "cobertura_estimada_pct" in cov
        assert 0.0 <= cov["cobertura_estimada_pct"] <= 100.0
        assert cov["total_partidos_muestra"] == 2
        # Al menos 1 partido tiene jugador resuelto (Badosa)
        assert cov["resueltos_crosswalk"] >= 1

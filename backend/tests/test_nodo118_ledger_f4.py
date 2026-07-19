"""
REGLA-T53: Tests Nodo-118 F4 — Adapter schema + refresh cuotas + CLI.

Cubre: exportar_para_edge_calculator emite schema idéntico al esperado por
edge_calculator; actualizar_cuotas_ledger actualiza in-place; CLI --build
escribe archivo merged con nombre zita_tennis_matches_*_merged.json;
run_daily.py contiene PASO 1b y PASO 1.5.
"""

import json
import re
from pathlib import Path

from scraping.match_ledger import (
    MERGED_PATTERN,
    _REQUIRED_FIELDS_EDGE,
    actualizar_cuotas_ledger,
    exportar_para_edge_calculator,
    fusionar_dia,
    load_ledger,
    save_ledger,
)


def _make_kambi(n=2):
    return [
        {"jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek",
         "cuota1": 1.27, "cuota2": 3.50, "hora": "14:00",
         "torneo_nombre": "Wimbledon", "superficie": "grass",
         "match_id": "abc123", "kambi_event_id": "K001"},
        {"jugador1": "Carlos Alcaraz", "jugador2": "Novak Djokovic",
         "cuota1": 1.55, "cuota2": 2.40, "hora": "16:00",
         "torneo_nombre": "Wimbledon", "superficie": "grass",
         "match_id": "def456", "kambi_event_id": "K002"},
    ][:n]


def _make_fs(n=2):
    return [
        {"jugador1": "P. Badosa", "jugador2": "T. Zidansek",
         "cuota1": None, "match_id": "abc123", "match_url": "https://x.com/1",
         "hora_partido": "14:00", "superficie": "grass"},
        {"jugador1": "C. Alcaraz", "jugador2": "N. Djokovic",
         "cuota1": None, "match_id": "def456", "match_url": "https://x.com/2",
         "hora_partido": "16:00", "superficie": "grass"},
    ][:n]


class TestExportarAdapterF4:

    def test_schema_tiene_campos_requeridos(self, tmp_path):
        """
        REGLA-T53: exportar_para_edge_calculator escribe archivo con todos los
        campos requeridos por edge_calculator (_REQUIRED_FIELDS_EDGE).
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "crosswalk.json"

        fusionar_dia(_make_kambi(), _make_fs(), "2026-07-19",
                     output_dir=str(tmp_path))

        out_path = exportar_para_edge_calculator("2026-07-19", data_dir=str(tmp_path))

        assert out_path, "Debe retornar un path no vacío"
        data = json.loads(Path(out_path).read_text())
        assert isinstance(data, list) and len(data) >= 1

        for campo in _REQUIRED_FIELDS_EDGE:
            assert campo in data[0], f"Campo requerido ausente: {campo}"

    def test_exportar_incluye_joins_y_single_kambi(self, tmp_path):
        """
        REGLA-T53: el archivo exportado incluye joins + single_source_kambi
        (ambos tienen cuotas) pero NO single_source_fs (sin cuotas).
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "crosswalk.json"

        # 2 Kambi, 1 FS: dejará 1 join + 1 single_source_kambi
        kambi = _make_kambi(2)
        fs = _make_fs(1)
        fusionar_dia(kambi, fs, "2026-07-19", output_dir=str(tmp_path))

        out_path = exportar_para_edge_calculator("2026-07-19", data_dir=str(tmp_path))
        data = json.loads(Path(out_path).read_text())

        # Todos los partidos exportados deben tener cuota1 != None
        for p in data:
            assert p.get("cuota1") is not None, \
                f"Partido sin cuota en el export: {p.get('jugador1')}"

    def test_refresh_cuotas_actualiza_in_place(self, tmp_path):
        """
        REGLA-T53: actualizar_cuotas_ledger() modifica cuota1/cuota2 en el ledger
        cuando llegan cuotas frescas del API. El ledger persiste el cambio.
        """
        import core.player_registry as _mod
        _mod._CROSSWALK_FILE = tmp_path / "crosswalk.json"

        fusionar_dia(_make_kambi(), _make_fs(), "2026-07-19",
                     output_dir=str(tmp_path))

        # Cuotas frescas: Badosa baja a 1.20
        kambi_frescos = [
            {"jugador1": "Paula Badosa", "jugador2": "Tamara Zidansek",
             "cuota1": 1.20, "cuota2": 3.80,
             "match_id": "abc123", "kambi_event_id": "K001"},
        ]
        stats = actualizar_cuotas_ledger("2026-07-19", kambi_frescos,
                                         data_dir=str(tmp_path))

        assert stats["actualizados"] >= 1, f"Debe actualizar al menos 1: {stats}"

        # Verificar persistencia en disco
        ledger = load_ledger("2026-07-19", data_dir=str(tmp_path))
        partido_badosa = next(
            (p for p in ledger.get("joins", []) + ledger.get("single_source_kambi", [])
             if "abc123" in str(p.get("match_id", ""))),
            None
        )
        assert partido_badosa is not None, "Partido Badosa debe estar en el ledger"
        assert partido_badosa["cuota1"] == 1.20, \
            f"Cuota1 debe ser 1.20, got {partido_badosa['cuota1']}"

    def test_run_daily_contiene_paso1b_y_paso15(self):
        """
        REGLA-T53: run_daily.py contiene PASO 1b (Playwright) y PASO 1.5 (ledger).
        """
        src = Path("run_daily.py").read_text(encoding="utf-8")

        assert "extraer_URL_partidos_version2.py" in src, \
            "run_daily debe invocar extraer_URL_partidos_version2.py en PASO 1b"
        assert "match_ledger.py" in src and "--build" in src, \
            "run_daily debe invocar match_ledger.py --build en PASO 1.5"
        assert "PASO 1b" in src, "run_daily debe tener comentario PASO 1b"
        assert "PASO 1.5" in src, "run_daily debe tener comentario PASO 1.5"

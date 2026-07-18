"""
REGLA-T53: Tests para D117-02 — select_best_json_file prioriza archivos con cuotas Kambi.

Bug: B117-03 en Nodo-117: selector elegía por tamaño/recencia → siempre seleccionaba
el archivo Playwright (130 partidos, cuota1=null) sobre el API (66 partidos, con cuotas).
Fix D117-02: matches_with_cuotas > 0 es criterio primario sobre mtime y match count.
"""

import json
import time
import pytest
from pathlib import Path
import tempfile
import os


def _write_json(dir_path: Path, filename: str, data: dict, mtime_offset: float = 0) -> Path:
    """Helper: escribe JSON y ajusta mtime."""
    fpath = dir_path / filename
    with open(fpath, "w") as f:
        json.dump(data, f)
    # Ajustar mtime para simular orden de creación
    t = time.time() + mtime_offset
    os.utime(fpath, (t, t))
    return fpath


def _make_partido(con_cuota: bool) -> dict:
    """Partido mínimo con o sin cuota Kambi."""
    p = {
        "jugador1": "A",
        "jugador2": "B",
        "match_url": "https://example.com/match/1",
    }
    if con_cuota:
        p["cuota1"] = 1.80
        p["cuota2"] = 2.10
    else:
        p["cuota1"] = None
        p["cuota2"] = None
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectBestJsonFileD11702:
    """D117-02: archivo con cuotas siempre gana sobre archivo sin cuotas."""

    def test_con_cuotas_gana_sobre_archivo_mas_grande_sin_cuotas(self, tmp_path):
        """
        REGLA-T53: archivo API (66 partidos, cuotas) debe ganar sobre Playwright
        (130 partidos, sin cuotas), aunque Playwright sea más grande y más reciente.
        """
        from scraping.file_utils import select_best_json_file

        # Playwright: 130 partidos SIN cuotas, más reciente
        playwright_data = {"Torneo A": [_make_partido(con_cuota=False) for _ in range(130)]}
        _write_json(tmp_path, "playwright_130.json", playwright_data, mtime_offset=10)

        # API: 66 partidos CON cuotas, más antiguo
        api_data = {"Torneo A": [_make_partido(con_cuota=True) for _ in range(66)]}
        _write_json(tmp_path, "api_66.json", api_data, mtime_offset=0)

        selected = select_best_json_file(str(tmp_path), "*.json", auto_select=True)

        assert selected is not None
        assert "api_66" in selected, (
            f"D117-02: esperado api_66 (con cuotas), seleccionado: {selected}"
        )

    def test_entre_archivos_con_cuotas_gana_el_mas_reciente(self, tmp_path):
        """
        REGLA-T53: cuando ambos archivos tienen cuotas, debe ganar el más reciente.
        """
        from scraping.file_utils import select_best_json_file

        # Archivo más antiguo con cuotas
        old_data = {"Torneo A": [_make_partido(con_cuota=True) for _ in range(20)]}
        _write_json(tmp_path, "old_20.json", old_data, mtime_offset=0)

        # Archivo más reciente con cuotas
        new_data = {"Torneo A": [_make_partido(con_cuota=True) for _ in range(30)]}
        _write_json(tmp_path, "new_30.json", new_data, mtime_offset=10)

        selected = select_best_json_file(str(tmp_path), "*.json", auto_select=True)

        assert selected is not None
        assert "new_30" in selected, (
            f"D117-02: entre archivos con cuotas, esperado new_30 (más reciente), seleccionado: {selected}"
        )

    def test_sin_cuotas_en_todos_vuelve_al_criterio_original(self, tmp_path):
        """
        REGLA-T53: si ningún archivo tiene cuotas, el criterio original (recencia) aplica.
        """
        from scraping.file_utils import select_best_json_file

        # Ambos sin cuotas — debe ganar el más reciente
        old_data = {"Torneo A": [_make_partido(con_cuota=False) for _ in range(50)]}
        _write_json(tmp_path, "old_50.json", old_data, mtime_offset=0)

        new_data = {"Torneo A": [_make_partido(con_cuota=False) for _ in range(20)]}
        _write_json(tmp_path, "new_20.json", new_data, mtime_offset=10)

        selected = select_best_json_file(str(tmp_path), "*.json", auto_select=True)

        assert selected is not None
        assert "new_20" in selected, (
            f"Sin cuotas: esperado new_20 (más reciente), seleccionado: {selected}"
        )

    def test_analyze_json_structure_cuenta_matches_with_cuotas(self, tmp_path):
        """
        REGLA-T53: analyze_json_structure reporta matches_with_cuotas correctamente.
        """
        from scraping.file_utils import analyze_json_structure

        data = {
            "Torneo A": [
                _make_partido(con_cuota=True),
                _make_partido(con_cuota=True),
                _make_partido(con_cuota=False),  # sin cuota
            ]
        }
        fpath = tmp_path / "test.json"
        with open(fpath, "w") as f:
            json.dump(data, f)

        analysis = analyze_json_structure(fpath)

        assert analysis["matches_with_cuotas"] == 2, (
            f"D117-02: esperado 2 partidos con cuotas, obtenido: {analysis['matches_with_cuotas']}"
        )
        assert analysis["match_count"] == 3

    def test_analyze_json_structure_lista_cuenta_cuotas(self, tmp_path):
        """
        REGLA-T53: analyze_json_structure en formato lista también cuenta cuotas.
        """
        from scraping.file_utils import analyze_json_structure

        data = [
            _make_partido(con_cuota=True),
            _make_partido(con_cuota=False),
            _make_partido(con_cuota=False),
        ]
        fpath = tmp_path / "lista.json"
        with open(fpath, "w") as f:
            json.dump(data, f)

        analysis = analyze_json_structure(fpath)

        assert analysis["matches_with_cuotas"] == 1, (
            f"Formato lista: esperado 1 con cuota, obtenido: {analysis['matches_with_cuotas']}"
        )

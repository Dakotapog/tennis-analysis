"""
tests/test_nodo110_favoritos_builder.py — Nodo-110 Favoritos Compuestos

REGLA-T53: Tests invocan función real del módulo — nunca hardcodean la fórmula.

Bug Fix Nodo-110 (2026-07-17): _leer_edge_report() estaba buscando claves
"picks"/"results" que no existen en el schema real de edge_report.json.
Esto causaba universo=[] silenciosamente. Tests verifican contrato correcto.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from favoritos_combo_builder import _leer_edge_report


class TestLeerEdgeReport:
    """REGLA-T53: _leer_edge_report() debe merguear las 3 listas reales del schema."""

    def test_merge_trois_listas_full(self):
        """Función retorna todos los candidatos: apostar + watchlist + sin_edge."""
        edge_report = {
            "apostar": [
                {"favorito_predicho": "Gaines", "cuota_favorito": 1.23, "confianza": "STRONG"},
            ],
            "watchlist": [
                {"favorito_predicho": "McNeil", "cuota_favorito": 1.32, "confianza": "MOD"},
            ],
            "sin_edge": [
                {"favorito_predicho": "Forbes", "cuota_favorito": 1.45, "confianza": "LOW"},
            ],
            "no_data": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(edge_report, f)
            f.flush()
            temp_path = f.name

        try:
            result = _leer_edge_report(temp_path)

            # Assertion: len(result) == len(apostar) + len(watchlist) + len(sin_edge)
            assert len(result) == 3, f"esperado 3, obtuve {len(result)}"
            nombres = [p.get("favorito_predicho") for p in result]
            assert "Gaines" in nombres, f"Gaines no está en {nombres}"
            assert "McNeil" in nombres, f"McNeil no está en {nombres}"
            assert "Forbes" in nombres, f"Forbes no está en {nombres}"
        finally:
            Path(temp_path).unlink()

    def test_leer_edge_report_vacío(self):
        """Edge case: todas las listas están vacías."""
        edge_report = {
            "apostar": [],
            "watchlist": [],
            "sin_edge": [],
            "no_data": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(edge_report, f)
            f.flush()
            temp_path = f.name

        try:
            result = _leer_edge_report(temp_path)
            assert len(result) == 0, f"vacío: esperado 0, obtuve {len(result)}"
        finally:
            Path(temp_path).unlink()

    def test_leer_edge_report_parcial(self):
        """Solo algunos campos tienen candidatos."""
        edge_report = {
            "apostar": [
                {"favorito_predicho": "A", "cuota_favorito": 1.5, "confianza": "STRONG"},
                {"favorito_predicho": "B", "cuota_favorito": 1.6, "confianza": "STRONG"},
            ],
            "watchlist": [],  # vacío
            "sin_edge": [
                {"favorito_predicho": "C", "cuota_favorito": 1.7, "confianza": "LOW"},
            ],
            "no_data": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(edge_report, f)
            f.flush()
            temp_path = f.name

        try:
            result = _leer_edge_report(temp_path)
            assert len(result) == 3, f"parcial: esperado 3, obtuve {len(result)}"
            nombres = [p.get("favorito_predicho") for p in result]
            assert nombres == ["A", "B", "C"], f"orden: esperado ['A','B','C'], obtuve {nombres}"
        finally:
            Path(temp_path).unlink()

    def test_leer_edge_report_lista_directa(self):
        """Si el JSON es una lista (esquema heredado), retorna como está."""
        edge_report = [
            {"favorito_predicho": "X", "cuota_favorito": 1.5},
            {"favorito_predicho": "Y", "cuota_favorito": 1.6},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(edge_report, f)
            f.flush()
            temp_path = f.name

        try:
            result = _leer_edge_report(temp_path)
            assert len(result) == 2, f"lista: esperado 2, obtuve {len(result)}"
            assert result[0]["favorito_predicho"] == "X"
        finally:
            Path(temp_path).unlink()

    def test_leer_edge_report_mantiene_orden(self):
        """Merge respeta orden: apostar primero, luego watchlist, luego sin_edge."""
        edge_report = {
            "apostar": [{"favorito_predicho": f"A{i}", "cuota_favorito": 1.2} for i in range(2)],
            "watchlist": [{"favorito_predicho": f"W{i}", "cuota_favorito": 1.5} for i in range(2)],
            "sin_edge": [{"favorito_predicho": f"S{i}", "cuota_favorito": 1.8} for i in range(2)],
            "no_data": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(edge_report, f)
            f.flush()
            temp_path = f.name

        try:
            result = _leer_edge_report(temp_path)
            nombres = [p.get("favorito_predicho") for p in result]
            assert nombres[:2] == ["A0", "A1"], "apostar debe ir primero"
            assert nombres[2:4] == ["W0", "W1"], "watchlist debe ir segundo"
            assert nombres[4:] == ["S0", "S1"], "sin_edge debe ir tercero"
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

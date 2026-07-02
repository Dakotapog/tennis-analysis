"""
tests/test_nodo51_f5.py — Nodo-51 F5: Framework de Validación Pre-Registrada

Tests:
  T51-F5-01: preregistered_hypotheses.json existe y tiene estructura válida
  T51-F5-02: Los 8 umbrales WAS congelados están presentes y no modificables
  T51-F5-03: Calibration epochs tienen los campos obligatorios
  T51-F5-04: Nodo-46 case count es correcto (n=1, Watanuki)
  T51-F5-05: nodo46_unlocked() = False con n=1 (n<5 → BLOQUEADO)
  T51-F5-06: Las 8 hipótesis (H52-01 a H52-08) están todas presentes

Detectan mutación real:
  T51-F5-02 FALLA si se eliminan umbrales congelados de H52-01 (WAS).
  T51-F5-04 FALLA si el contador de Nodo-46 cambia sin añadir un caso atribuible real.
  T51-F5-05 FALLA si se modifica el umbral de desbloqueo (n≥5) sin evidencia empírica.
"""
import json
import pytest
from pathlib import Path

HYPOTHESES_FILE = Path(__file__).parent.parent / "validation" / "preregistered_hypotheses.json"


@pytest.fixture(scope="module")
def hyp_data():
    """Carga el JSON una vez por módulo."""
    assert HYPOTHESES_FILE.exists(), "validation/preregistered_hypotheses.json debe existir"
    return json.loads(HYPOTHESES_FILE.read_text(encoding="utf-8"))


class TestF5HypothesesFile:

    def test_t51_f5_01_file_valid_json(self, hyp_data):
        """T51-F5-01: El archivo existe y tiene estructura top-level correcta."""
        assert "_meta" in hyp_data
        assert "hypotheses" in hyp_data
        assert isinstance(hyp_data["hypotheses"], dict)

    def test_t51_f5_06_all_eight_hypotheses_present(self, hyp_data):
        """T51-F5-06: Las 8 hipótesis H52-01 a H52-08 están todas presentes."""
        hypotheses = hyp_data["hypotheses"]
        required = [f"H52-0{i}" for i in range(1, 9)]
        for h_id in required:
            assert h_id in hypotheses, f"Hipótesis {h_id} no encontrada"

    def test_t51_f5_02_was_thresholds_frozen(self, hyp_data):
        """T51-F5-02: WAS (H52-01) tiene umbrales congelados obligatorios.
        FALLA si se eliminan los umbrales — su integridad es el contrato del pre-registro."""
        h01 = hyp_data["hypotheses"]["H52-01"]
        umbrales = h01.get("umbrales_congelados", {})
        assert "edge_min" in umbrales, "edge_min debe estar congelado en WAS"
        assert "cuota_min" in umbrales, "cuota_min debe estar congelado en WAS"
        assert umbrales["edge_min"] == 0.10, "edge_min debe ser 0.10 (D44-03)"
        assert umbrales["cuota_min"] == 2.0, "cuota_min debe ser 2.0 (D44-03)"
        assert h01.get("n_stop") == 30, "n_stop debe ser 30 para WAS"

    def test_t51_f5_03_calibration_epochs(self, hyp_data):
        """T51-F5-03: Los epochs de calibración tienen los campos obligatorios."""
        epochs = hyp_data["_meta"]["calibration_epochs"]
        assert "epoch_1" in epochs
        assert "epoch_2" in epochs
        e1 = epochs["epoch_1"]
        assert "corte_hasta" in e1
        assert "n_observaciones" in e1
        assert e1["n_observaciones"] >= 0
        e2 = epochs["epoch_2"]
        assert "corte_desde" in e2

    def test_t51_f5_04_nodo46_case_count_is_one(self, hyp_data):
        """T51-F5-04: Nodo-46 tiene exactamente 1 caso atribuible (Watanuki).
        FALLA si el contador cambia sin añadir evidencia real verificada."""
        h04 = hyp_data["hypotheses"]["H52-04"]
        assert h04.get("n_casos_atribuibles") == 1, (
            "Nodo-46 tiene n=1 caso atribuible (Watanuki). "
            "Solo modificar al añadir un caso que cumpla las 3 condiciones."
        )
        assert h04.get("estado") == "BLOQUEADO_ACUMULANDO"
        casos = h04.get("casos", [])
        assert len(casos) == 1
        assert casos[0]["jugador"] == "Watanuki"

    def test_t51_f5_05_nodo46_not_unlocked(self):
        """T51-F5-05: Con n=1 caso, Nodo-46 está BLOQUEADO (necesita n≥5).
        FALLA si se modifica el umbral de desbloqueo sin evidencia empírica."""
        from validation.hypothesis_tracker import nodo46_unlocked, get_nodo46_case_count
        assert get_nodo46_case_count() == 1
        assert nodo46_unlocked() is False, (
            "D46-07 debe estar BLOQUEADO hasta n≥5 casos atribuibles. "
            "Con n=1 (Watanuki), las constantes min_floor y THRESHOLD NO se calibran."
        )

    def test_t51_f5_07_tracker_api_works(self):
        """T51-F5-07: La API del hypothesis_tracker funciona correctamente."""
        from validation.hypothesis_tracker import (
            load_hypotheses, get_hypothesis, was_thresholds, get_calibration_epochs
        )
        data = load_hypotheses()
        assert "hypotheses" in data

        h01 = get_hypothesis("H52-01")
        assert h01 is not None

        thresholds = was_thresholds()
        assert thresholds.get("edge_min") == 0.10

        epochs = get_calibration_epochs()
        assert "epoch_1" in epochs

    def test_t51_f5_08_h52_04_three_conditions_documented(self, hyp_data):
        """T51-F5-08: H52-04 tiene las 3 condiciones de atribución documentadas."""
        h04 = hyp_data["hypotheses"]["H52-04"]
        criterio = h04.get("criterio_atribucion", [])
        assert len(criterio) == 3, (
            "Las 3 condiciones de atribución de Nodo-46 deben estar documentadas"
        )

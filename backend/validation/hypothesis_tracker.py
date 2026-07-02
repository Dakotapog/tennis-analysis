"""
validation/hypothesis_tracker.py — Nodo-51 F5

Acceso de lectura/escritura al registro de hipótesis pre-registradas.
READ-ONLY sobre calibracion_edge.json (nunca modifica datos de calibración).
"""
import json
from pathlib import Path
from typing import Dict, Optional

_HYPOTHESES_FILE = Path(__file__).parent / "preregistered_hypotheses.json"


def load_hypotheses() -> Dict:
    """Carga el registro de hipótesis. Lanza FileNotFoundError si no existe."""
    return json.loads(_HYPOTHESES_FILE.read_text(encoding="utf-8"))


def get_hypothesis(hypothesis_id: str) -> Optional[Dict]:
    """Retorna una hipótesis por ID (e.g. 'H52-01') o None si no existe."""
    data = load_hypotheses()
    return data.get("hypotheses", {}).get(hypothesis_id)


def get_nodo46_case_count() -> int:
    """
    Retorna el número de casos atribuibles a Nodo-46 (Surface Context Discount).
    Al llegar a n=5 → desbloquear D46-07 (calibración de constantes).
    """
    h = get_hypothesis("H52-04")
    if h is None:
        return 0
    return h.get("n_casos_atribuibles", 0)


def nodo46_unlocked() -> bool:
    """True si n_casos_atribuibles >= 5 → D46-07 se puede ejecutar."""
    return get_nodo46_case_count() >= 5


def get_calibration_epochs() -> Dict:
    """Retorna los cortes de época de calibración para calibracion_edge.json."""
    data = load_hypotheses()
    return data.get("_meta", {}).get("calibration_epochs", {})


def was_thresholds() -> Dict:
    """Retorna los umbrales congelados del WAS (H52-01). No modificar hasta n=30."""
    h = get_hypothesis("H52-01")
    if h is None:
        return {}
    return h.get("umbrales_congelados", {})

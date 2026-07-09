"""
validation/hypothesis_tracker.py — Nodo-51 F5

Acceso de lectura/escritura al registro de hipótesis pre-registradas.
READ-ONLY sobre calibracion_edge.json (nunca modifica datos de calibración).
"""
import json
import math
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


# ── Nodo-64: SPRT — Sequential Probability Ratio Test (Wald 1945) ─────────────

def llr_update(llr_prev: float, outcome: int, p0: float, p1: float) -> float:
    """
    Nodo-64: Actualiza el log-likelihood ratio con una nueva observación Bernoulli.
    outcome: 1 (hit) o 0 (miss)
    """
    if outcome == 1:
        return llr_prev + math.log(p1 / p0)
    else:
        return llr_prev + math.log((1 - p1) / (1 - p0))


def sprt_verdict(n: int, hits: int, p0: float, p1: float,
                 alpha: float = 0.05, beta: float = 0.05) -> dict:
    """
    Nodo-64: Sequential Probability Ratio Test (Wald 1945).

    H0: tasa real = p0 (breakeven / baseline)
    H1: tasa real = p1 (breakeven + delta)

    Fronteras pre-registradas:
      A = ln((1-beta)/alpha)  -> rechazar H0 (aceptar H1)
      B = ln(beta/(1-alpha))  -> rechazar H1 (aceptar H0)

    Returns dict con: llr, verdict ('ACEPTA_H1'|'ACEPTA_H0'|'CONTINUA'),
                       boundary_A, boundary_B, n, hits, p0, p1
    """
    boundary_A = math.log((1 - beta) / alpha)
    boundary_B = math.log(beta / (1 - alpha))

    # LLR se calcula desde cero: hits contribuciones +1, (n-hits) contribuciones 0
    # Para Bernoulli i.i.d. el orden no importa
    llr = hits * math.log(p1 / p0) + (n - hits) * math.log((1 - p1) / (1 - p0))

    if llr >= boundary_A:
        verdict = 'ACEPTA_H1'
    elif llr <= boundary_B:
        verdict = 'ACEPTA_H0'
    else:
        verdict = 'CONTINUA'

    return {
        'llr': llr,
        'verdict': verdict,
        'boundary_A': boundary_A,
        'boundary_B': boundary_B,
        'n': n,
        'hits': hits,
        'p0': p0,
        'p1': p1,
        'alpha': alpha,
        'beta': beta,
    }


def sprt_from_hypothesis(hypothesis_id: str) -> dict:
    """
    Nodo-64: Calcula veredicto SPRT para una hipótesis pre-registrada.
    Lee n_actual y hits del JSON. p0 = 1/cuota_media_segmento o breakeven.
    Usa alpha=beta=0.05 como defaults.
    Returns sprt_verdict(...) o {'error': motivo} si faltan campos.

    Para hipótesis sin cuota_media, usa p0=0.50 como prior conservador.
    """
    h = get_hypothesis(hypothesis_id)
    if h is None:
        return {'error': f'Hipótesis {hypothesis_id} no encontrada'}

    n = h.get('n_actual')
    hits = h.get('hits')

    if n is None or hits is None:
        return {'error': f'Hipótesis {hypothesis_id} no tiene n_actual/hits'}

    # p0: usa 1/cuota_media si disponible, si no p0=0.50 (prior conservador)
    cuota_media = h.get('cuota_media')
    if cuota_media and cuota_media > 1.0:
        p0 = 1.0 / cuota_media
    else:
        p0 = 0.50

    # p1: breakeven + delta (por defecto δ=0.10)
    p1 = min(0.99, p0 + 0.10)

    return sprt_verdict(n=n, hits=hits, p0=p0, p1=p1)

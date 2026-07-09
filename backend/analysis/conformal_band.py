"""
analysis/conformal_band.py — Nodo-68: Banda Conformal

INSTRUMENTO DE MEDICIÓN — READ-ONLY.
No modifica ninguna constante de producción.

Calcula la banda conformal desde el shadow book settled.
Muestra la banda junto al umbral fijo 54% sin cambiar decisiones de apuesta.

Nodo-68 constantes:
  _CONFORMAL_ALPHA = 0.10      # cobertura 90% (1-α)
  _CONFORMAL_N_MIN = 50        # gate global antes de modo activo
  _CONFORMAL_N_MIN_TIER = 30   # gate por tier
"""

import os
import json
import math
import glob as glob_mod
from typing import List, Optional, Dict

# Nodo-68 constantes
_CONFORMAL_ALPHA = 0.10
_CONFORMAL_N_MIN = 50
_CONFORMAL_N_MIN_TIER = 30


def conformal_quantile(residuals: List[float], alpha: float = _CONFORMAL_ALPHA) -> float:
    """
    Calcula el cuantil (1-α) de los residuos |outcome - p_modelo|.
    residuals: lista de valores absolutos |y - p̂|
    Retorna q tal que P(|residual| ≤ q) ≥ (1-α)
    """
    if not residuals:
        return 1.0  # máximo posible si sin datos
    n = len(residuals)
    sorted_r = sorted(residuals)
    # Índice de cuantil (1-alpha): ceil((1-alpha) * n) - 1, clipped
    idx = min(n - 1, math.ceil((1 - alpha) * n) - 1)
    # Conformal quantile estándar: ceil((n+1)*(1-alpha)) / n
    # Usamos la forma conservadora: percentil ceil((1-alpha)*(n+1)) - 1
    idx_conservative = min(n - 1, math.ceil((1 - alpha) * (n + 1)) - 1)
    return sorted_r[idx_conservative]


def is_no_bet_conformal(p_modelo: float, q: float) -> bool:
    """
    Retorna True si el intervalo conformal [p-q, p+q] contiene 0.5
    (el modelo no distingue el pick de una moneda).
    """
    return (p_modelo - q) <= 0.5 <= (p_modelo + q)


def conformal_report(shadow_dir: str = None) -> dict:
    """
    Calcula la banda conformal desde el shadow book settled.
    Lee |outcome - p_modelo| por registro, calcula q global y por tier.

    Returns: {
        'q_global': float or None,
        'q_por_tier': {'grand_slam': float, ...},
        'n_settled': int,
        'gate_ok': bool,   # n_settled >= _CONFORMAL_N_MIN
        'ejemplo': 'pick p=0.56 con q=0.08 → NO-BET (0.56-0.08=0.48 < 0.50)',
        'nota': 'REPORTE_SOLO — banda se muestra junto a umbral fijo 54%, sin cambiar decisiones'
    }
    """
    if shadow_dir is None:
        shadow_dir = os.path.join('reports', 'shadow_book')

    # Acumular residuos por tier
    residuals_global: List[float] = []
    residuals_by_tier: Dict[str, List[float]] = {}
    tiers = ('grand_slam', 'atp1000', 'atp500', 'challenger', 'itf')
    for t in tiers:
        residuals_by_tier[t] = []

    pattern = os.path.join(shadow_dir, 'sb_*.jsonl')
    for fpath in sorted(glob_mod.glob(pattern)):
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get('_type') == 'session_meta':
                        continue
                    if 'resolucion' not in rec:
                        continue
                    resultado = rec.get('resolucion', {}).get('resultado')
                    if resultado not in ('WON', 'LOST'):
                        continue
                    outcome = 1.0 if resultado == 'WON' else 0.0
                    snap = rec.get('pick_snapshot', {})
                    p_modelo = snap.get('p_modelo')
                    if p_modelo is None or not isinstance(p_modelo, (int, float)):
                        continue
                    residual = abs(outcome - float(p_modelo))
                    residuals_global.append(residual)
                    tier = snap.get('tier', 'unknown')
                    if tier in residuals_by_tier:
                        residuals_by_tier[tier].append(residual)
        except Exception:
            continue

    n_settled = len(residuals_global)
    gate_ok = n_settled >= _CONFORMAL_N_MIN

    q_global = conformal_quantile(residuals_global) if residuals_global else None

    q_por_tier: Dict[str, Optional[float]] = {}
    for t in tiers:
        res_t = residuals_by_tier[t]
        if len(res_t) >= _CONFORMAL_N_MIN_TIER:
            q_por_tier[t] = conformal_quantile(res_t)
        else:
            q_por_tier[t] = None

    # Ejemplo ilustrativo con q_global (o valor hipotético si no hay datos)
    q_ejemplo = round(q_global, 2) if q_global is not None else 0.08
    p_ej = 0.56
    no_bet = is_no_bet_conformal(p_ej, q_ejemplo)
    lo = round(p_ej - q_ejemplo, 2)
    veredicto = 'NO-BET' if no_bet else 'BET'
    ejemplo = (
        f"pick p={p_ej} con q={q_ejemplo} → {veredicto} "
        f"({p_ej}-{q_ejemplo}={lo} {'<' if lo < 0.5 else '>='} 0.50)"
    )

    return {
        'q_global': round(q_global, 4) if q_global is not None else None,
        'q_por_tier': {t: (round(v, 4) if v is not None else None) for t, v in q_por_tier.items()},
        'n_settled': n_settled,
        'gate_ok': gate_ok,
        'ejemplo': ejemplo,
        'nota': 'REPORTE_SOLO — banda se muestra junto a umbral fijo 54%, sin cambiar decisiones',
    }

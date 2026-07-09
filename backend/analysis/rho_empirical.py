"""
analysis/rho_empirical.py — Nodo-65: Bootstrap ρ Empírico Inter-Pick

INSTRUMENTO DE MEDICIÓN — READ-ONLY.
No modifica ninguna constante de producción.

Calcula la correlación inter-pick dentro de sesión por tier usando block bootstrap.
El bloque es la sesión entera (preserva dependencia intra-sesión).

Nodo-65 constantes:
  _BOOTSTRAP_B = 2000           # réplicas bootstrap
  _RHO_MIN_SESSIONS = 15        # gate: sesiones mínimas con ≥3 picks/tier
  _RHO_MIN_PICKS_PER_SESSION = 3
"""

import os
import json
import math
import glob as glob_mod
import random
from typing import List, Optional

# Nodo-65 constantes
_BOOTSTRAP_B = 2000
_RHO_MIN_SESSIONS = 15
_RHO_MIN_PICKS_PER_SESSION = 3


def _pairwise_correlation_session(outcomes: List[int]) -> float:
    """
    Correlación promedio de pares Bernoulli en una sesión.
    outcomes: lista de 0/1 de picks del mismo tier en esa sesión.
    Si len < 2: retorna 0.
    ρ̂ = mean de (x_i - μ)(x_j - μ) / var para todos los pares i<j
    """
    n = len(outcomes)
    if n < 2:
        return 0.0

    mean = sum(outcomes) / n
    var = sum((x - mean) ** 2 for x in outcomes) / n
    if var == 0.0:
        # Todos iguales: correlación perfecta (+1 si todos 1, +1 si todos 0)
        # Con Bernoulli binaria, varianza=0 → correlación perfecta positiva
        return 1.0

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((outcomes[i] - mean) * (outcomes[j] - mean) / var)

    return sum(pairs) / len(pairs) if pairs else 0.0


def block_bootstrap_rho(
    sessions: List[List[int]],
    seed: int = 42,
    B: int = _BOOTSTRAP_B,
) -> dict:
    """
    Block bootstrap de ρ inter-pick dentro de sesión.
    El bloque es la sesión entera (preserva dependencia intra-sesión).

    Returns: {
        'rho_hat': float,          # ρ̂ puntual
        'ci_90_lo': float,         # percentil 5
        'ci_90_hi': float,         # percentil 95
        'n_sessions': int,
        'gate_ok': bool,           # n_sessions >= _RHO_MIN_SESSIONS
        'nota': str
    }
    Si gate_ok=False: calcula igualmente pero nota 'n insuficiente — modo reporte'
    """
    n_sessions = len(sessions)
    gate_ok = n_sessions >= _RHO_MIN_SESSIONS

    if n_sessions == 0:
        return {
            'rho_hat': 0.0,
            'ci_90_lo': 0.0,
            'ci_90_hi': 0.0,
            'n_sessions': 0,
            'gate_ok': False,
            'nota': 'n insuficiente — modo reporte',
        }

    # ρ̂ puntual: media de correlaciones por sesión
    session_rhos = [_pairwise_correlation_session(s) for s in sessions if len(s) >= 2]
    rho_hat = sum(session_rhos) / len(session_rhos) if session_rhos else 0.0

    # Block bootstrap: resamplear sesiones enteras
    rng = random.Random(seed)
    boot_rhos = []
    for _ in range(B):
        boot_sessions = [rng.choice(sessions) for _ in range(n_sessions)]
        boot_session_rhos = [
            _pairwise_correlation_session(s) for s in boot_sessions if len(s) >= 2
        ]
        boot_rho = sum(boot_session_rhos) / len(boot_session_rhos) if boot_session_rhos else 0.0
        boot_rhos.append(boot_rho)

    boot_rhos_sorted = sorted(boot_rhos)
    lo_idx = max(0, int(0.05 * B) - 1)
    hi_idx = min(B - 1, int(0.95 * B))
    ci_90_lo = boot_rhos_sorted[lo_idx]
    ci_90_hi = boot_rhos_sorted[hi_idx]

    nota = (
        'n insuficiente — modo reporte'
        if not gate_ok
        else 'REPORTE_SOLO — recalibración solo en ventana mensual pre-agendada'
    )

    return {
        'rho_hat': round(rho_hat, 4),
        'ci_90_lo': round(ci_90_lo, 4),
        'ci_90_hi': round(ci_90_hi, 4),
        'n_sessions': n_sessions,
        'gate_ok': gate_ok,
        'nota': nota,
    }


def rho_report(shadow_dir: str = None) -> dict:
    """
    Lee shadow book, agrupa por (fecha, tier), calcula ρ por tier vía block bootstrap.

    Returns: {
        'por_tier': {
            'grand_slam': {...},  # block_bootstrap_rho output
            'atp1000': {...},
            ...
        },
        'n_total_sessions': int,
        'nota': 'REPORTE_SOLO — recalibración solo en ventana mensual pre-agendada'
    }
    """
    if shadow_dir is None:
        shadow_dir = os.path.join('reports', 'shadow_book')

    # Agrupa outcomes settled por (fecha, tier)
    # estructura: {(fecha, tier): [outcome, ...]}
    sessions_by_tier: dict = {}

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
                    outcome = 1 if resultado == 'WON' else 0
                    snap = rec.get('pick_snapshot', {})
                    tier = snap.get('tier', 'unknown')
                    # fecha desde sb_id o nombre de archivo
                    fname = os.path.basename(fpath)
                    # sb_YYYY-MM-DD.jsonl
                    fecha = fname[3:13] if len(fname) >= 13 else 'unknown'
                    key = (fecha, tier)
                    if key not in sessions_by_tier:
                        sessions_by_tier[key] = []
                    sessions_by_tier[key].append(outcome)
        except Exception:
            continue

    # Agrupar por tier: lista de sesiones (cada sesión = lista de outcomes)
    tiers = ('grand_slam', 'atp1000', 'atp500', 'challenger', 'itf')
    por_tier: dict = {}
    n_total_sessions = 0

    for tier in tiers:
        tier_sessions = [
            outcomes
            for (fecha, t), outcomes in sessions_by_tier.items()
            if t == tier and len(outcomes) >= _RHO_MIN_PICKS_PER_SESSION
        ]
        n_total_sessions += len(tier_sessions)
        por_tier[tier] = block_bootstrap_rho(tier_sessions)

    return {
        'por_tier': por_tier,
        'n_total_sessions': n_total_sessions,
        'nota': 'REPORTE_SOLO — recalibración solo en ventana mensual pre-agendada',
    }

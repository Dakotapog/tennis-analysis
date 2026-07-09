"""
analysis/velocity_monitor.py — Nodo-71: Kyle's λ Velocity Monitor

INSTRUMENTO DE MEDICIÓN — READ-ONLY.
No modifica ninguna constante de producción.

Monitorea la velocidad de movimiento de línea para detectar STEAM
(acortamiento anómalo de cuota que indica información privilegiada).

Nodo-71 constantes:
  _VELOCITY_Z_THRESH = -2.0  # z-score umbral para STEAM (acortamiento anómalo)
"""

import math
from typing import List, Optional

# Nodo-71 constantes
_VELOCITY_Z_THRESH = -2.0


def velocity_zscore(
    odds_series: List[float],
    times_minutes: List[float],
) -> dict:
    """
    Nodo-71: Kyle's λ — velocidad de movimiento de línea.

    velocity_i = (odds[i] - odds[i-1]) / (times[i] - times[i-1])
    Normaliza por volatilidad típica de la banda de cuota.

    z_score = (velocity_t - mean(velocity)) / std(velocity)
    donde velocity y stats se calculan sobre TODA la serie (excepto el último punto).

    El z_score de la ÚLTIMA observación es el indicador.

    STEAM = z_score < _VELOCITY_Z_THRESH (acortamiento anómalo = información)

    Returns: {
        'velocities': [...],   # lista de velocidades
        'z_scores': [...],     # lista de z-scores
        'z_last': float,       # z-score del último punto
        'steam': bool,         # z_last < _VELOCITY_Z_THRESH
        'signal': 'STEAM'|'DRIFT'|'FLAT',  # STEAM si z<-2, DRIFT si z>2, FLAT si |z|<2
        'nota': 'REPORTE_SOLO — solo reporta hasta que H52-05 concluya'
    }
    Si len(odds_series) < 3: returns {'z_last': None, 'steam': False, 'signal': 'FLAT', 'nota': ...}
    """
    _nota = 'REPORTE_SOLO — solo reporta hasta que H52-05 concluya'

    if len(odds_series) < 3 or len(times_minutes) < 3:
        return {
            'velocities': [],
            'z_scores': [],
            'z_last': None,
            'steam': False,
            'signal': 'FLAT',
            'nota': _nota,
        }

    n = min(len(odds_series), len(times_minutes))
    odds = odds_series[:n]
    times = times_minutes[:n]

    # Calcular velocidades (diferencias consecutivas en puntos de cuota / minuto)
    velocities: List[float] = []
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            dt = 1.0  # evitar división por cero, asumir 1 minuto
        v = (odds[i] - odds[i - 1]) / dt
        velocities.append(v)

    if len(velocities) < 2:
        return {
            'velocities': velocities,
            'z_scores': [],
            'z_last': None,
            'steam': False,
            'signal': 'FLAT',
            'nota': _nota,
        }

    # Calcular z-scores: para cada punto i, z usando stats de todos los puntos anteriores
    # Para el último punto: stats sobre todos los puntos menos el último
    z_scores: List[Optional[float]] = []

    for i in range(len(velocities)):
        # Stats sobre los puntos [0, i-1] (excluye el punto actual)
        # Mínimo 1 punto anterior para calcular z-score
        if i == 0:
            z_scores.append(None)
            continue
        ref = velocities[:i]
        mean_ref = sum(ref) / len(ref)
        if len(ref) < 2:
            # Con solo 1 punto de referencia, no hay std
            z_scores.append(None)
            continue
        var_ref = sum((v - mean_ref) ** 2 for v in ref) / (len(ref) - 1)
        std_ref = math.sqrt(var_ref) if var_ref > 0 else None
        if std_ref is None or std_ref == 0:
            z_scores.append(0.0)
            continue
        z = (velocities[i] - mean_ref) / std_ref
        z_scores.append(round(z, 4))

    # z del último punto
    z_last_raw = z_scores[-1] if z_scores else None

    # Si el último z es None (insuficientes datos), retornar FLAT
    if z_last_raw is None:
        return {
            'velocities': [round(v, 6) for v in velocities],
            'z_scores': [z if z is not None else None for z in z_scores],
            'z_last': None,
            'steam': False,
            'signal': 'FLAT',
            'nota': _nota,
        }

    z_last = float(z_last_raw)
    steam = z_last < _VELOCITY_Z_THRESH

    if z_last < _VELOCITY_Z_THRESH:
        signal = 'STEAM'
    elif z_last > abs(_VELOCITY_Z_THRESH):
        signal = 'DRIFT'
    else:
        signal = 'FLAT'

    return {
        'velocities': [round(v, 6) for v in velocities],
        'z_scores': [round(z, 4) if z is not None else None for z in z_scores],
        'z_last': round(z_last, 4),
        'steam': steam,
        'signal': signal,
        'nota': _nota,
    }

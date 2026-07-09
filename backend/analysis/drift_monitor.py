"""
analysis/drift_monitor.py — Nodo-67: CUSUM + PSI drift monitoring

READ-ONLY module: no modifica edge_report, calibracion_edge.json, shadow_book ni
ningun output del pipeline. Solo observacion estadistica.

Constantes marcadas PROVISIONAL — no cambiar sin n>=30 settled con alarma real.
"""

import os
import json
import math
import glob as glob_mod
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constantes — PROVISIONALES (etiquetadas per spec Nodo-67)
# ══════════════════════════════════════════════════════════════════════════════

_CUSUM_K = 0.005   # PROVISIONAL: drift permitido por observacion
_CUSUM_H = 0.05    # PROVISIONAL: threshold de alarma
_N_MIN_CUSUM = 10  # minimo de observaciones para calcular CUSUM con sentido

_NOTE_PROVISIONAL = "REPORTE_SOLO — constantes provisionales"


# ══════════════════════════════════════════════════════════════════════════════
# CUSUM sobre Brier score
# ══════════════════════════════════════════════════════════════════════════════

def cusum_brier(brier_series: List[float], k: float = _CUSUM_K, h: float = _CUSUM_H) -> dict:
    """
    CUSUM one-sided (detecta DEGRADACION = aumento de Brier).
    S_t = max(0, S_{t-1} + (brier_t - brier_ref - k))
    donde brier_ref = media de los primeros min(10, n) valores (periodo de referencia)

    Returns:
        cusum_series: S_t por observacion
        alarm_t:      primer t donde S_t > h (1-indexed), None si no hay alarma
        brier_ref:    media de referencia
        alarma:       bool
        k:            parametro k usado
        h:            threshold usado
    """
    n = len(brier_series)
    if n == 0:
        return {
            'cusum_series': [],
            'alarm_t': None,
            'brier_ref': 0.0,
            'alarma': False,
            'k': k,
            'h': h,
        }

    ref_n = min(10, n)
    brier_ref = sum(brier_series[:ref_n]) / ref_n

    cusum_series: List[float] = []
    alarm_t: Optional[int] = None
    s_prev = 0.0

    for i, bt in enumerate(brier_series):
        s_t = max(0.0, s_prev + (bt - brier_ref - k))
        cusum_series.append(round(s_t, 8))
        if alarm_t is None and s_t > h:
            alarm_t = i + 1  # 1-indexed
        s_prev = s_t

    return {
        'cusum_series': cusum_series,
        'alarm_t': alarm_t,
        'brier_ref': round(brier_ref, 6),
        'alarma': alarm_t is not None,
        'k': k,
        'h': h,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PSI — Population Stability Index
# ══════════════════════════════════════════════════════════════════════════════

def psi_score(dist_ref: List[float], dist_new: List[float], bins: int = 5) -> float:
    """
    Population Stability Index entre dos distribuciones.
    Usa quantile-based binning del dist_ref con fallback a rango combinado
    cuando la distribucion de referencia es degenerada (todos los valores iguales).
    PSI = Sigma (p_new - p_ref) * ln(p_new / p_ref)

    Valores de referencia:
      PSI < 0.1   -> estable
      0.1-0.25    -> cambio menor
      > 0.25      -> cambio mayor (alarma)

    Returns: float PSI (0 si distribuciones identicas)
    """
    if not dist_ref or not dist_new:
        return 0.0

    n_ref = len(dist_ref)
    n_new = len(dist_new)

    sorted_ref = sorted(dist_ref)
    min_ref = sorted_ref[0]
    max_ref = sorted_ref[-1]

    # Calcular breaks por quantiles del dist_ref
    raw_breaks: List[float] = []
    for i in range(bins + 1):
        idx = int(round(i * (n_ref - 1) / bins))
        raw_breaks.append(sorted_ref[idx])

    if min_ref == max_ref:
        # Distribucion degenerada: todos los valores de referencia son iguales.
        # Usar rango combinado (ref + new) para definir los bins.
        all_vals = sorted_ref + sorted(dist_new)
        combined_min = all_vals[0]
        combined_max = all_vals[-1]
        spread = combined_max - combined_min
        if spread == 0:
            # Todas las distribuciones son identicas y constantes -> PSI = 0
            return 0.0
        breaks = [combined_min + i * spread / bins for i in range(bins + 1)]
    else:
        breaks = raw_breaks

    # Ajustar bordes extremos para capturar todos los valores (inclusive)
    breaks[0] = -math.inf
    breaks[-1] = math.inf

    def _count_in_bins(data: List[float]) -> List[int]:
        counts = [0] * bins
        for v in data:
            placed = False
            # Iterar todos los bins excepto el ultimo; ultimo captura el resto
            for b in range(bins - 1):
                if breaks[b] <= v < breaks[b + 1]:
                    counts[b] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
        return counts

    counts_ref = _count_in_bins(dist_ref)
    counts_new = _count_in_bins(dist_new)

    _EPSILON = 1e-9  # evitar log(0) y division por cero

    psi = 0.0
    for i in range(bins):
        p_ref = counts_ref[i] / n_ref
        p_new = counts_new[i] / n_new
        p_ref = max(p_ref, _EPSILON)
        p_new = max(p_new, _EPSILON)
        psi += (p_new - p_ref) * math.log(p_new / p_ref)

    return round(max(0.0, psi), 6)


# ══════════════════════════════════════════════════════════════════════════════
# Lectura del shadow book
# ══════════════════════════════════════════════════════════════════════════════

def _default_shadow_dir() -> str:
    """Retorna el directorio shadow_book relativo al script."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "reports", "shadow_book")


def _load_all_settled(shadow_dir: str) -> List[dict]:
    """
    Lee todos los JSONL del shadow book y retorna unicamente registros settled
    (aquellos con campo 'resolucion' y outcome determinado).
    """
    records: List[dict] = []
    pattern = os.path.join(shadow_dir, "sb_*.jsonl")
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
                    res = rec.get('resolucion', {})
                    resultado = res.get('resultado')
                    if resultado not in ('WON', 'LOST'):
                        continue
                    records.append(rec)
        except Exception as e:
            logger.debug(f"[DriftMonitor] Error leyendo {fpath}: {e}")
    return records


def _brier_from_record(rec: dict) -> Optional[float]:
    """
    Extrae Brier score de un registro settled.
    Brier = (outcome - p_modelo)^2
    outcome: 1 si WON, 0 si LOST
    p_modelo: de pick_snapshot['probabilidad_modelo'] o pick_snapshot['p_modelo']
    """
    res = rec.get('resolucion', {})
    resultado = res.get('resultado')
    if resultado == 'WON':
        outcome = 1.0
    elif resultado == 'LOST':
        outcome = 0.0
    else:
        return None

    snap = rec.get('pick_snapshot', {})
    p_modelo = snap.get('probabilidad_modelo') or snap.get('p_modelo')
    if p_modelo is None:
        return None

    try:
        p_float = float(p_modelo)
    except (TypeError, ValueError):
        return None

    if not (0.0 <= p_float <= 1.0):
        return None

    return (outcome - p_float) ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Reporte diario de drift
# ══════════════════════════════════════════════════════════════════════════════

def daily_drift_report(shadow_dir: str = None) -> dict:
    """
    Lee el shadow book (JSONL) y calcula:
    1. Brier diario -> CUSUM
    2. PSI de distribucion de n_partidos por jugador (input distribution)
    3. PSI de mezcla de history_provenance (deteccion de regime shift tipo Nodo-47)

    Returns dict con:
        cusum:            output de cusum_brier()
        psi_n_partidos:   PSI de dist de n_partidos
        psi_provenance:   PSI de mezcla provenance (0.0 si sin datos suficientes)
        alarma_cusum:     bool
        alarma_psi:       bool — PSI > 0.25 en cualquier input
        n_settled:        int
        note:             str — siempre 'REPORTE_SOLO — constantes provisionales'

    Si shadow_dir es None: usa reports/shadow_book/ relativo al directorio del script.
    Si no hay suficientes datos (n<10): retorna {'note': 'n insuficiente', 'n_settled': n}
    """
    if shadow_dir is None:
        shadow_dir = _default_shadow_dir()

    settled = _load_all_settled(shadow_dir)
    n_settled = len(settled)

    if n_settled < _N_MIN_CUSUM:
        return {
            'note': 'n insuficiente',
            'n_settled': n_settled,
        }

    # 1. Serie de Brier (en orden de aparicion en los archivos)
    brier_series: List[float] = []
    for rec in settled:
        b = _brier_from_record(rec)
        if b is not None:
            brier_series.append(b)

    cusum_result = cusum_brier(brier_series) if brier_series else cusum_brier([])

    # 2. PSI de n_partidos (distribucion del input: riqueza de historial)
    # Mitad mas antigua (referencia) vs mitad mas reciente
    n_partidos_dist: List[float] = []
    for rec in settled:
        snap = rec.get('pick_snapshot', {})
        np_val = snap.get('n_partidos') or snap.get('n_h2h')
        if np_val is not None:
            try:
                n_partidos_dist.append(float(np_val))
            except (TypeError, ValueError):
                pass

    psi_n_partidos = 0.0
    if len(n_partidos_dist) >= 2:
        mid = len(n_partidos_dist) // 2
        ref_half = n_partidos_dist[:mid]
        new_half = n_partidos_dist[mid:]
        if ref_half and new_half:
            psi_n_partidos = psi_score(ref_half, new_half)

    # 3. PSI de history_provenance (detecta shifts tipo Nodo-47: API vs Playwright)
    # Codifica como numerico para PSI: playwright=1.0, api=0.0, other=0.5
    _prov_map = {
        'playwright': 1.0,
        'flashscore_playwright': 1.0,
        'api': 0.0,
        'ninja_api': 0.0,
        'ninja': 0.0,
    }

    prov_dist: List[float] = []
    for rec in settled:
        snap = rec.get('pick_snapshot', {})
        prov = (snap.get('history_provenance') or '').lower().strip()
        if not prov:
            continue
        val = _prov_map.get(prov, 0.5)
        prov_dist.append(val)

    psi_provenance = 0.0
    if len(prov_dist) >= 2:
        mid_p = len(prov_dist) // 2
        ref_p = prov_dist[:mid_p]
        new_p = prov_dist[mid_p:]
        if ref_p and new_p:
            psi_provenance = psi_score(ref_p, new_p)

    alarma_psi = (psi_n_partidos > 0.25) or (psi_provenance > 0.25)

    return {
        'cusum': cusum_result,
        'psi_n_partidos': round(psi_n_partidos, 6),
        'psi_provenance': round(psi_provenance, 6),
        'alarma_cusum': cusum_result.get('alarma', False),
        'alarma_psi': alarma_psi,
        'n_settled': n_settled,
        'note': _NOTE_PROVISIONAL,
    }

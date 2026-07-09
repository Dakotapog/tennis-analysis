"""
analysis/flb_curve.py — Nodo-66: Favorite-Longshot Bias (FLB) Curve

Estima la curva empírica de FLB por banda de cuota usando el shadow book.

READ-ONLY: no modifica ningún archivo del pipeline.
Flag-OFF en producción: solo reportes, no cambia decisiones de apuesta.
Constantes PROVISIONALES etiquetadas.
"""

import os
import json
import glob as glob_mod
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constantes — etiquetadas por spec Nodo-66
# ══════════════════════════════════════════════════════════════════════════════

# Bandas estándar de cuota (borde inferior inclusivo, superior exclusivo salvo última)
_CUOTA_BANDS = [
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.5),
    (3.5, 5.0),
    (5.0, 100.0),
]

_N_MIN_BANDA = 10   # mínimo por banda para breakeven ajustado (PROVISIONAL)

_NOTA = "REPORTE_SOLO — en modo report, no cambia decisiones"


# ══════════════════════════════════════════════════════════════════════════════
# Lectura del shadow book (reutiliza misma lógica que drift_monitor)
# ══════════════════════════════════════════════════════════════════════════════

def _default_shadow_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "reports", "shadow_book")


def _load_settled_for_flb(shadow_dir: str) -> List[dict]:
    """
    Carga registros settled del shadow book.
    Retorna solo aquellos con outcome determinado (WON o LOST) y cuota_favorito > 0.
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
                    if res.get('resultado') not in ('WON', 'LOST'):
                        continue
                    snap = rec.get('pick_snapshot', {})
                    cuota = snap.get('cuota_favorito', 0) or 0
                    if cuota <= 0:
                        continue
                    records.append(rec)
        except Exception as e:
            logger.debug(f"[FLBCurve] Error leyendo {fpath}: {e}")
    return records


def _banda_label(low: float, high: float) -> str:
    """Genera etiqueta legible para una banda de cuota."""
    lo_str = f"{low:.1f}" if low != int(low) else f"{int(low)}"
    hi_str = f"{high:.1f}" if high != int(high) else f"{int(high)}"
    return f"{lo_str}-{hi_str}"


def _in_banda(cuota: float, low: float, high: float, is_last: bool) -> bool:
    """Determina si cuota pertenece a la banda [low, high). Última banda: [low, high]."""
    if is_last:
        return low <= cuota <= high
    return low <= cuota < high


# ══════════════════════════════════════════════════════════════════════════════
# FLB Curve principal
# ══════════════════════════════════════════════════════════════════════════════

def flb_curve(shadow_dir: str = None) -> dict:
    """
    Nodo-66: Estima curva FLB empírica por banda de cuota.

    Para cada banda en _CUOTA_BANDS:
    - Filtra registros settled con cuota_favorito en la banda
    - Calcula hit% real = hits / n_banda
    - Calcula p_implicita_media = mean(1/cuota_favorito)
    - Calcula breakeven_ingenuo = mean(1/cuota_favorito)
    - Calcula breakeven_ajustado = hit% real (si n>=_N_MIN_BANDA, si no: None)
    - Calcula flb_delta = hit% - p_implicita_media
      (positivo = el modelo gana sobre la p implícita en esa banda)

    Returns dict con:
        bandas:  lista de dicts por banda (ver estructura abajo)
        n_total: int total de registros settled analizados
        nota:    'REPORTE_SOLO — en modo report, no cambia decisiones'

    Estructura de cada banda:
        rango              str          ej. '1.5-2.0'
        n                  int
        hit_pct            float        hit% en [0, 1]
        p_implicita_media  float        media de 1/cuota
        breakeven_ingenuo  float        igual que p_implicita_media
        breakeven_ajustado float|None   None si n < _N_MIN_BANDA
        flb_delta          float        hit_pct - p_implicita_media
        n_suficiente       bool         n >= _N_MIN_BANDA
    """
    if shadow_dir is None:
        shadow_dir = _default_shadow_dir()

    settled = _load_settled_for_flb(shadow_dir)
    n_total = len(settled)

    bandas = []
    last_idx = len(_CUOTA_BANDS) - 1

    for idx, (low, high) in enumerate(_CUOTA_BANDS):
        is_last = (idx == last_idx)
        banda_recs = [
            r for r in settled
            if _in_banda(r['pick_snapshot']['cuota_favorito'], low, high, is_last)
        ]
        n = len(banda_recs)

        if n == 0:
            bandas.append({
                'rango': _banda_label(low, high),
                'n': 0,
                'hit_pct': 0.0,
                'p_implicita_media': 0.0,
                'breakeven_ingenuo': 0.0,
                'breakeven_ajustado': None,
                'flb_delta': 0.0,
                'n_suficiente': False,
            })
            continue

        hits = sum(
            1 for r in banda_recs
            if r['resolucion']['resultado'] == 'WON'
        )

        cuotas = [r['pick_snapshot']['cuota_favorito'] for r in banda_recs]
        p_implicitas = [1.0 / c for c in cuotas if c > 0]

        p_impl_media = sum(p_implicitas) / len(p_implicitas) if p_implicitas else 0.0
        hit_pct = hits / n

        n_suficiente = n >= _N_MIN_BANDA
        breakeven_ajustado = hit_pct if n_suficiente else None
        flb_delta = hit_pct - p_impl_media

        bandas.append({
            'rango': _banda_label(low, high),
            'n': n,
            'hit_pct': round(hit_pct, 4),
            'p_implicita_media': round(p_impl_media, 4),
            'breakeven_ingenuo': round(p_impl_media, 4),
            'breakeven_ajustado': round(breakeven_ajustado, 4) if breakeven_ajustado is not None else None,
            'flb_delta': round(flb_delta, 4),
            'n_suficiente': n_suficiente,
        })

    return {
        'bandas': bandas,
        'n_total': n_total,
        'nota': _NOTA,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helper de lookup por cuota
# ══════════════════════════════════════════════════════════════════════════════

def breakeven_ajustado_para_cuota(cuota: float, flb_report: dict) -> float:
    """
    Retorna el breakeven ajustado para una cuota dada usando la banda correspondiente.
    Si la banda no tiene n suficiente (breakeven_ajustado es None), retorna 1/cuota
    (breakeven ingenuo).

    Args:
        cuota:      cuota decimal del favorito (ej. 1.80)
        flb_report: dict retornado por flb_curve()

    Returns: float en [0, 1] — probabilidad de breakeven
    """
    if cuota <= 0:
        return 0.0

    bandas = flb_report.get('bandas', [])
    last_idx = len(_CUOTA_BANDS) - 1

    for idx, ((low, high), banda) in enumerate(zip(_CUOTA_BANDS, bandas)):
        is_last = (idx == last_idx)
        if _in_banda(cuota, low, high, is_last):
            be_ajustado = banda.get('breakeven_ajustado')
            if be_ajustado is not None:
                return float(be_ajustado)
            # n insuficiente: fallback ingenuo
            return 1.0 / cuota

    # cuota fuera de rango: fallback ingenuo
    return 1.0 / cuota

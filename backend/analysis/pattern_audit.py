"""
analysis/pattern_audit.py — Nodo-69: Auditoría de Cohortes Emparejadas

INSTRUMENTO DE MEDICIÓN — READ-ONLY.
No modifica ninguna constante de producción.

Para cada pick settled con pattern_field == pattern_value:
- Busca 1-3 controles settled emparejados (misma tier, banda cuota ±0.3,
  banda p_modelo ±0.05, misma superficie, mismo epoch)
- Compara hit% patrón vs hit% controles
- McNemar test (chi² pareado) cuando n≥5 pares
"""

import os
import json
import math
import glob as glob_mod
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# TOLERANCIAS DE EMPAREJAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOLERANCIAS = {
    'cuota_band': 0.3,
    'p_modelo_band': 0.05,
}


def _normalize_superficie(s: str) -> str:
    """Normaliza nombre de superficie para comparación."""
    if not s:
        return ''
    s = s.lower().strip()
    if s in ('hierba', 'grass'):
        return 'grass'
    if s in ('clay', 'tierra', 'arcilla', 'polvo de ladrillo'):
        return 'clay'
    if s in ('hard', 'dura', 'indoor hard'):
        return 'hard'
    return s


def _match_control(pick: dict, candidate: dict, tolerancias: dict = None) -> bool:
    """
    Verifica si candidate es un buen control para pick (emparejamiento por propensión).
    tolerancias default: tier=exact, cuota_band=0.3, p_modelo_band=0.05, superficie=exact
    """
    if tolerancias is None:
        tolerancias = _DEFAULT_TOLERANCIAS

    cuota_band = tolerancias.get('cuota_band', 0.3)
    p_band = tolerancias.get('p_modelo_band', 0.05)

    p_snap = pick.get('pick_snapshot', {})
    c_snap = candidate.get('pick_snapshot', {})

    # Tier: exacto
    if p_snap.get('tier') != c_snap.get('tier'):
        return False

    # Superficie: exacta (normalizada)
    p_sup = _normalize_superficie(p_snap.get('superficie', ''))
    c_sup = _normalize_superficie(c_snap.get('superficie', ''))
    if p_sup and c_sup and p_sup != c_sup:
        return False

    # Cuota: banda ±cuota_band
    p_cuota = p_snap.get('cuota_favorito')
    c_cuota = c_snap.get('cuota_favorito')
    if p_cuota is not None and c_cuota is not None:
        if abs(float(p_cuota) - float(c_cuota)) > cuota_band:
            return False

    # p_modelo: banda ±p_band
    p_pmod = p_snap.get('p_modelo')
    c_pmod = c_snap.get('p_modelo')
    if p_pmod is not None and c_pmod is not None:
        if abs(float(p_pmod) - float(c_pmod)) > p_band:
            return False

    # Epoch: exacto (si existe)
    p_epoch = p_snap.get('calibration_epoch')
    c_epoch = c_snap.get('calibration_epoch')
    if p_epoch and c_epoch and p_epoch != c_epoch:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CARGA SHADOW BOOK
# ─────────────────────────────────────────────────────────────────────────────

def _load_settled_records(shadow_dir: str) -> List[dict]:
    """Carga todos los registros settled del shadow book."""
    records: List[dict] = []
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
                    records.append(rec)
        except Exception:
            continue
    return records


def _get_outcome(rec: dict) -> int:
    """Retorna 1 (WON) o 0 (LOST) para un registro settled."""
    return 1 if rec.get('resolucion', {}).get('resultado') == 'WON' else 0


def _get_field_value(rec: dict, field: str) -> Any:
    """Lee campo del pick_snapshot o del registro raíz."""
    snap = rec.get('pick_snapshot', {})
    if field in snap:
        return snap[field]
    return rec.get(field)


# ─────────────────────────────────────────────────────────────────────────────
# MCNEMAR TEST
# ─────────────────────────────────────────────────────────────────────────────

def _mcnemar_p(b: int, c: int) -> Optional[float]:
    """
    McNemar test chi² pareado.
    b = pares donde patrón=WON, control=LOST
    c = pares donde patrón=LOST, control=WON
    Retorna p-value (continuidad de Yates) o None si n_discordantes < 2
    """
    n_disc = b + c
    if n_disc < 2:
        return None
    # Chi² con corrección de Yates: (|b - c| - 1)^2 / (b + c)
    numerator = max(0.0, abs(b - c) - 1.0) ** 2
    chi2 = numerator / n_disc
    # p-value chi² con 1 grado de libertad (distribución chi² = Gamma(0.5, 0.5))
    # Implementación sin scipy: p = 1 - CDF_chi2(chi2, df=1)
    # CDF_chi2(x, 1) = erf(sqrt(x/2))
    p = 1.0 - math.erf(math.sqrt(chi2 / 2.0))
    return round(p, 4)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def audit_pattern(
    pattern_field: str,
    pattern_value: Any,
    shadow_dir: str = None,
) -> dict:
    """
    Nodo-69: Auditoría de cohortes emparejadas.

    Para cada pick settled con pattern_field == pattern_value:
    - Busca 1-3 controles settled emparejados (misma tier, banda cuota ±0.3,
      banda p_modelo ±0.05, misma superficie, mismo epoch)
    - Compara hit% patrón vs hit% controles
    - McNemar test (chi² pareado) cuando n≥5 pares

    Returns: {
        'patron': {'campo': pattern_field, 'valor': str(pattern_value)},
        'n_patron': int,
        'n_controles': int,
        'n_pares': int,
        'hit_pct_patron': float,
        'hit_pct_controles': float,
        'diferencia': float,          # hit% patrón - hit% controles
        'mcnemar_p': float or None,   # None si n_pares < 5
        'significativo': bool,        # p < 0.10
        'plantilla_hipotesis': str,   # texto pre-formateado para pre-registrar
        'nota': str
    }
    """
    if shadow_dir is None:
        shadow_dir = os.path.join('reports', 'shadow_book')

    all_settled = _load_settled_records(shadow_dir)

    # Separar patrón vs universo de controles candidatos
    patron_records: List[dict] = []
    control_candidates: List[dict] = []

    for rec in all_settled:
        val = _get_field_value(rec, pattern_field)
        # Comparación flexible: bool vs int, str vs str
        if _values_match(val, pattern_value):
            patron_records.append(rec)
        else:
            control_candidates.append(rec)

    n_patron = len(patron_records)

    # Emparejar: para cada pick patrón, buscar hasta 3 controles
    pares: List[Tuple[int, int]] = []  # (outcome_patron, outcome_control)
    controles_usados = set()

    for p_rec in patron_records:
        matched = 0
        for i, c_rec in enumerate(control_candidates):
            if i in controles_usados:
                continue
            if _match_control(p_rec, c_rec):
                p_out = _get_outcome(p_rec)
                c_out = _get_outcome(c_rec)
                pares.append((p_out, c_out))
                controles_usados.add(i)
                matched += 1
                if matched >= 3:
                    break

    n_pares = len(pares)
    n_controles = len(controles_usados)

    # Hit% patrón (sobre todos los registros patrón, no solo los emparejados)
    hit_patron = sum(_get_outcome(r) for r in patron_records)
    hit_pct_patron = round(hit_patron / n_patron * 100, 1) if n_patron > 0 else 0.0

    # Hit% controles (sobre los controles emparejados)
    if pares:
        hit_controles = sum(c_out for _, c_out in pares)
        hit_pct_controles = round(hit_controles / n_pares * 100, 1)
    else:
        hit_pct_controles = 0.0

    diferencia = round(hit_pct_patron - hit_pct_controles, 1)

    # McNemar: solo si n_pares >= 5
    mcnemar_p = None
    significativo = False
    if n_pares >= 5:
        b = sum(1 for p_out, c_out in pares if p_out == 1 and c_out == 0)
        c = sum(1 for p_out, c_out in pares if p_out == 0 and c_out == 1)
        mcnemar_p = _mcnemar_p(b, c)
        if mcnemar_p is not None:
            significativo = mcnemar_p < 0.10

    plantilla = (
        f"H_PATRON_{pattern_field.upper()}: {pattern_field}={pattern_value} → "
        f"hit%={hit_pct_patron}% vs controles={hit_pct_controles}% "
        f"(diff={diferencia:+.1f}pp, McNemar_p={mcnemar_p}, n_pares={n_pares})"
    )

    return {
        'patron': {'campo': pattern_field, 'valor': str(pattern_value)},
        'n_patron': n_patron,
        'n_controles': n_controles,
        'n_pares': n_pares,
        'hit_pct_patron': hit_pct_patron,
        'hit_pct_controles': hit_pct_controles,
        'diferencia': diferencia,
        'mcnemar_p': mcnemar_p,
        'significativo': significativo,
        'plantilla_hipotesis': plantilla,
        'nota': 'REPORTE_SOLO — resultado de auditoría, no modifica decisiones de producción',
    }


def _values_match(val: Any, pattern_value: Any) -> bool:
    """
    Comparación flexible entre valor del registro y valor buscado.
    Soporta: bool vs bool, bool vs int, str vs str (case-insensitive), int vs int.
    """
    if val is None:
        return pattern_value is None
    if isinstance(pattern_value, bool) or isinstance(val, bool):
        # Comparar como bool explícitamente
        return bool(val) == bool(pattern_value)
    if isinstance(pattern_value, str) and isinstance(val, str):
        return val.strip().lower() == pattern_value.strip().lower()
    return val == pattern_value

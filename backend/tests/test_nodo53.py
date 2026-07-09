"""
tests/test_nodo53.py — Nodo-53: Auditoría Corazón Predicción

Ciclo FAIL→PASS por test:
  ImportError  = función no extraída a nivel módulo → volver al Paso 1
  AssertionError = función existe y reproduce el bug → proceder al fix
  PASS = contrato cumplido después del fix

REGLA-T53: ningún test hardcodea la fórmula buggy. Siempre invoca la función del módulo real.
"""

import pytest
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# T53-01 — D53-01: Fechas con año de 4 dígitos crashean silenciosamente
# ─────────────────────────────────────────────────────────────────────────────

def test_t53_01_date_format_handles_4digit_year():
    """D53-01: '%d.%m.%y' descartaba H2H con año 4 dígitos (ej. '09.10.2024').

    Ciclo esperado:
      Paso 2 (bug intacto): AssertionError o ValueError propagado
      Paso 3 (fix '%d.%m.%Y'): PASS — fecha parseada con año correcto

    Llama a _parse_match_date() del módulo real (REGLA-T53).
    """
    from analysis.rivalry_analyzer import _parse_match_date

    date_str = "09.10.2024"
    # Con bug ('%d.%m.%y'): ValueError unconverted data remains: 24 → partido descartado
    # Con fix ('%d.%m.%Y'): datetime(2024, 10, 9)
    parsed = _parse_match_date(date_str)

    assert parsed.year == 2024, (
        f"D53-01: _parse_match_date('{date_str}') devolvió año={parsed.year}, esperado 2024."
    )
    assert parsed.month == 10
    assert parsed.day == 9


# ─────────────────────────────────────────────────────────────────────────────
# T53-06 — D53-06: surface_specialization escala ~10-20x menor que form_recent
# ─────────────────────────────────────────────────────────────────────────────

def test_t53_06_surface_normalizes_to_same_scale_as_form():
    """D53-06: surface_specialization debe normalizar a escala comparable a form_recent.

    Ciclo esperado:
      Paso 1 (antes de extraer normalize_scores): ImportError
      Paso 2 (extraída, bug intacto): AssertionError — ratio=0.1295 < 0.40
      Paso 3 (fix _LINEAR_COMPONENTS=set()): PASS — ratio=0.8176 > 0.40

    Distinción crítica:
      ImportError   = normalize_scores no extraída → volver al Paso 1
      AssertionError = bug confirmado → proceder al fix
    """
    from analysis.rivalry_analyzer import normalize_scores

    p1 = {'surface_specialization': 33.49, 'form_recent': 75.0}
    p2 = {'surface_specialization': 10.89, 'form_recent': 150.0}
    norm_p1, _ = normalize_scores(p1, p2)

    ratio = norm_p1['surface_specialization'] / norm_p1['form_recent']

    # Con bug (lineal/350): surface=0.5608, form=4.3307 → ratio=0.1295
    # Con fix (log1p):      surface=3.5407, form=4.3307 → ratio=0.8176
    # Valores de referencia: log1p(33.49)=3.5407, log1p(75.0)=4.3307
    assert ratio > 0.40, (
        f"D53-06 activo: surface/form ratio={ratio:.4f} (esperado >0.40). "
        f"_LINEAR_COMPONENTS incluye 'surface_specialization' con MAX_RAW=350. "
        f"Fix: _LINEAR_COMPONENTS = set() en normalize_scores()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T53-07 — D53-07: ELO cap=250 colapsa todo el top-200 al mismo raw score
# ─────────────────────────────────────────────────────────────────────────────

def test_t53_07_elo_differentiates_within_top200():
    """D53-07: _compute_raw_elo debe producir valores distintos para ELO distintos en top-200.

    Ciclo esperado:
      Paso 1 (antes de extraer _compute_raw_elo): ImportError
      Paso 2 (extraída, bug intacto): AssertionError — 250 == 250
      Paso 3 (fix sin cap): PASS — 900 != 257

    Distinción crítica:
      ImportError   = función no extraída → volver al Paso 1
      AssertionError = bug confirmado (cap colapsa ELO≥1750) → proceder al fix
    """
    from analysis.rivalry_analyzer import _compute_raw_elo

    raw_sinner = _compute_raw_elo(2400)    # ELO estimado Sinner (rank ~1)
    raw_dimitrov = _compute_raw_elo(1757)  # ELO estimado Dimitrov (rank ~10)

    # Con bug (cap=250): min(max(0,900),250)=250 == min(max(0,257),250)=250
    # Con fix (sin cap): max(0,900)=900 != max(0,257)=257
    assert raw_sinner != raw_dimitrov, (
        f"D53-07 activo: raw_elo(2400)={raw_sinner} == raw_elo(1757)={raw_dimitrov}. "
        f"Cap=250 colapsa todo ELO>=1750. "
        f"Fix: return max(0, elo - 1500)  # eliminar min(..., 250)"
    )

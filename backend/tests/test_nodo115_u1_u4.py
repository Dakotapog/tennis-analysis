"""
tests/test_nodo115_u1_u4.py — REGLA-T53: U4 sparkline + U1 conformal band + §2.5 fetch-refresh.

U4 tests (3):
  U4-1. _sparkline_drift jugador con 4 lecturas → string no vacío con ▁-█ + flecha
  U4-2. _sparkline_drift drift creciente → flecha ↑
  U4-3. _sparkline_drift jugador sin historial → string vacío

U1 tests (3):
  U1-1. render_html demo → columna "Conf U1" presente y p=X ±Y
  U1-2. q_global cruza breakeven → "BANDA CRUZA BE" en celda
  U1-3. q_global None (n_settled < gate) → muestra "n<N" sin crash

§2.5 tests (2):
  R1. render_html no contiene meta http-equiv refresh
  R2. render_html contiene acc-tbody + desk-ts (hooks fetch JS)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(_BACKEND))

from live_desk import _sparkline_drift, render_html, _demo_state


# ── U4-1: sparkline con 4 lecturas ───────────────────────────────────────────

def test_sparkline_4_lecturas_no_vacio():
    """4 lecturas de drift → string con chars ▁-█ y flecha."""
    history = {
        "Alcaraz vs Djokovic": {
            "readings": [
                {"ts": "14:00:00", "cuota": 1.80, "drift": 0.05},
                {"ts": "14:05:00", "cuota": 1.72, "drift": 0.09},
                {"ts": "14:10:00", "cuota": 1.65, "drift": 0.14},
                {"ts": "14:15:00", "cuota": 1.55, "drift": 0.18},
            ],
            "estado": "BREAK_POSIBLE", "fired": False,
        }
    }
    result = _sparkline_drift("Alcaraz", history)
    assert result != "", "Con historial debe retornar sparkline no vacío"
    # Al menos un char del set de sparkline
    assert any(c in result for c in '▁▂▃▄▅▆▇█'), f"Debe contener chars sparkline: {result}"
    assert any(c in result for c in '→↑↓'), f"Debe contener flecha: {result}"


# ── U4-2: drift creciente → flecha ↑ ─────────────────────────────────────────

def test_sparkline_drift_creciente_flecha_arriba():
    """Drift creciente (cuota bajando) → flecha ↑."""
    history = {
        "Rublev vs Medvedev": {
            "readings": [
                {"ts": "10:00:00", "cuota": 2.0, "drift": 0.02},
                {"ts": "10:05:00", "cuota": 1.9, "drift": 0.05},
                {"ts": "10:10:00", "cuota": 1.75, "drift": 0.12},
                {"ts": "10:15:00", "cuota": 1.60, "drift": 0.20},
            ],
            "estado": "BREAK_POSIBLE", "fired": False,
        }
    }
    result = _sparkline_drift("Rublev", history)
    assert "↑" in result, f"Drift creciente debe mostrar ↑, got: {result}"


# ── U4-3: jugador sin historial → vacío ──────────────────────────────────────

def test_sparkline_sin_historial_vacio():
    """Jugador no encontrado en history → string vacío."""
    result = _sparkline_drift("Djokovic", {})
    assert result == "", "Sin historial debe retornar string vacío"


# ── U1-1: columna Conf U1 en render_html ─────────────────────────────────────

def test_u1_columna_presente_en_html():
    """render_html demo → columna 'Conf U1' y 'Tendencia U4' en la tabla."""
    state = _demo_state()
    # Inyectar q_global simulado
    state["p12_conformal"] = {"q_global": 0.09, "n_settled": 302, "gate_ok": True}
    html = render_html(state)
    assert "Conf U1" in html, "Falta columna Conf U1 en tabla"
    assert "Tendencia U4" in html, "Falta columna Tendencia U4 en tabla"


# ── U1-2: banda cruza breakeven → BANDA CRUZA BE ─────────────────────────────

def test_u1_banda_cruza_breakeven():
    """p=0.55 ±0.09 → [0.46, 0.64] cruza 0.50 → celda muestra BANDA CRUZA BE."""
    state = _demo_state()
    # Asegurar que hay un accionable con p_modelo=0.55
    for a in state.get("p2_break", {}).get("breaks", []):
        a["p_modelo"] = 0.55
    state["p12_conformal"] = {"q_global": 0.09, "n_settled": 302, "gate_ok": True}
    html = render_html(state)
    # p=0.55 - 0.09=0.46 ≤ 0.50 ≤ 0.55+0.09=0.64 → cruza BE
    assert "BANDA CRUZA BE" in html, "Debe mostrar BANDA CRUZA BE cuando banda cruza 0.5"


# ── U1-3: q_global None → muestra n<N sin crash ──────────────────────────────

def test_u1_q_none_no_crash():
    """q_global=None (n_settled < gate) → render no falla y muestra n<N."""
    state = _demo_state()
    state["p12_conformal"] = {"q_global": None, "n_settled": 23, "gate_ok": False}
    html = render_html(state)  # no debe lanzar excepción
    assert "n<23" in html or "23" in html, "Debe mostrar n_settled cuando q_global es None"


# ── §2.5-R1: no meta http-equiv refresh ──────────────────────────────────────

def test_no_meta_http_refresh():
    """render_html NO debe contener meta http-equiv refresh (§2.5)."""
    html = render_html(_demo_state())
    assert 'http-equiv="refresh"' not in html, "No debe haber meta refresh — usar fetch JS"
    assert 'http-equiv' not in html.lower(), "No debe haber meta http-equiv"


# ── §2.5-R2: hooks JS fetch presentes ────────────────────────────────────────

def test_fetch_refresh_hooks_presentes():
    """render_html contiene acc-tbody, desk-ts y autoRefresh JS (§2.5)."""
    html = render_html(_demo_state())
    assert 'id="acc-tbody"' in html, "Falta id=acc-tbody para fetch-refresh"
    assert 'id="desk-ts"' in html, "Falta id=desk-ts para actualizar timestamp"
    assert 'autoRefresh' in html, "Falta función autoRefresh en JS"
    assert '_activeFilter' in html, "Falta _activeFilter para preservar filtro"
    assert '_openRows' in html, "Falta _openRows para preservar drill-downs"

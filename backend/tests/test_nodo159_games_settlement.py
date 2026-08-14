"""
tests/test_nodo159_games_settlement.py — REGLA-T53 Nodo-159 S1 (D159-01 + D159-04)

Alcance de esta sesión: solo settlement automático de picks games_live +
fillability check pre-dispatch. D159-02/03/05 diferidos a sesión futura.

5 tests:
  1. test_159_01_settle_over_above
       settle_games_outcome("OVER", 32.5, 35) → (True, ...)
  2. test_159_02_settle_under_below
       settle_games_outcome("UNDER", 24.5, 22) → (True, ...)
  3. test_159_03_settle_no_snapshot_skips
       shadow_book.settle() real sobre un pick games_live sin
       games_final_score_*.json correspondiente → NO añade 'resolucion'
       (fix del bug P1: LOST falso silencioso).
  4. test_159_10_fillable_within_threshold
       validate_fillability() con drift <5% → (True, "fillable")
  5. test_159_11_drift_exceeds_aborts
       validate_fillability() con drift >5% → (False, "cuota_drift_...")

REGLA-T53: todos invocan funciones reales de los módulos — nunca hardcodean la fórmula.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.games_settlement import settle_games_outcome
import shadow_book as sb
import live_desk as ld


# ─── Tests 1-2: settle_games_outcome (función pura) ──────────────────────────

def test_159_01_settle_over_above():
    """OVER 32.5 con 35 juegos finales → gana (35 > 32.5)."""
    win, razon = settle_games_outcome("OVER", 32.5, 35)
    assert win is True
    assert "OVER" in razon and "35" in razon


def test_159_02_settle_under_below():
    """UNDER 24.5 con 22 juegos finales → gana (22 < 24.5)."""
    win, razon = settle_games_outcome("UNDER", 24.5, 22)
    assert win is True
    assert "UNDER" in razon and "22" in razon


# ─── Test 3: settle() real — no forzar LOST sin snapshot ─────────────────────

def test_159_03_settle_no_snapshot_skips(tmp_path, monkeypatch):
    """Pick games_live sin games_final_score_*.json correspondiente → settle()
    NO le añade 'resolucion' (permanece abierto). Verifica el fix del bug P1:
    antes, este pick caía en la resolución por nombre/match_key y podía
    marcarse LOST falso porque games_live nunca trae favorito_predicho."""
    monkeypatch.chdir(tmp_path)

    fecha = "2026-08-01"
    pick = {
        "partido": "Jamie Mackenzie vs Max Dahlin",
        "direccion": "UNDER",
        "linea": 24.5,
        "torneo": "ITF M15 Test",
    }
    sb_id = sb.log_games_live_pick(pick, cuota_trigger=1.90, fecha=fecha)
    assert sb_id is not None

    count = sb.settle(fecha, resultados_map={})
    assert count == 0

    records = sb._load_jsonl(sb._jsonl_path(fecha))
    rec = records[sb_id]
    assert 'resolucion' not in rec


# ─── Tests 4-5: validate_fillability (D159-04) ────────────────────────────────

def test_159_10_fillable_within_threshold():
    """Cuota fresca con drift <5% respecto a la usada en los gates → fillable."""
    sig = {"partido": "A vs B", "direccion": "UNDER", "linea": 24.5,
           "cuota_live": 1.90, "event_id": 123}

    with patch("live_desk._fetch_live_games_all",
               return_value={"cuota_under": 1.885, "cuota_over": None}) as mock_fetch:
        ok, razon = ld.validate_fillability(sig)

    assert ok is True
    assert razon == "fillable"
    mock_fetch.assert_called_once_with(123, bypass_cache=True)


def test_159_11_drift_exceeds_aborts():
    """Cuota fresca con drift >5% respecto a la usada en los gates → aborta."""
    sig = {"partido": "A vs B", "direccion": "UNDER", "linea": 24.5,
           "cuota_live": 1.90, "event_id": 123}

    with patch("live_desk._fetch_live_games_all",
               return_value={"cuota_under": 2.036, "cuota_over": None}):
        ok, razon = ld.validate_fillability(sig)

    assert ok is False
    assert razon.startswith("cuota_drift_")

"""
tests/test_nodo109_live_desk.py — REGLA-T53: invocan funciones reales.

Cubre Nodo-109:
  T1. build_desk_state — tolera archivos ausentes (retorna dict con 7 claves)
  T2. build_desk_state — tolera fecha sin archivos (cero crashes)
  T3. accionable_ahora — governor BLOCK → retorna []
  T4. accionable_ahora — KGR<0 → retorna []
  T5. accionable_ahora — BREAK_CONFIRMADO ∩ PASS → retorna señal
  T6. render_html — contiene los 7 paneles
  T7. render_html — governor BLOCK incluye banner HALT
  T8. render_html — estado completo → HTML válido (no vacío, tiene P4 primero)
  T9. accionable_ahora — GCS pick → color=green (GRADUADA)
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from live_desk import build_desk_state, accionable_ahora, render_html


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state(gov_code=0, kgr=1.0, breaks=None, conv_picks=None):
    """Construye un state mínimo para testing."""
    return {
        "fecha": "2026-07-14",
        "ts": "2026-07-14T10:00:00",
        "p4_risk": {
            "governor_code": gov_code,
            "kgr_sesion": kgr,
            "bankroll": 125000,
            "stake_total": 5000,
            "kill_switches": {"MOTOR_DEFENSIVE": True},
            "exposicion": [],
        },
        "p1_tape": {"entries": []},
        "p2_break": {"breaks": breaks or []},
        "p3_convergence": {"picks": conv_picks or []},
        "p5_execution": {"picks": [], "clv_median": None},
        "p6_pnl": {"segmentos": []},
        "p7_clock": {"partidos": []},
    }


# ── T1: build_desk_state tolera archivos ausentes ────────────────────────────

def test_build_desk_state_sin_archivos():
    """Fecha sin archivos → retorna dict con las 7 claves, sin crash."""
    with patch("live_desk.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        state = build_desk_state("1900-01-01")  # fecha que no tiene archivos
    assert isinstance(state, dict)
    for key in ("fecha", "ts", "p1_tape", "p2_break", "p3_convergence",
                "p4_risk", "p5_execution", "p6_pnl", "p7_clock"):
        assert key in state, f"Falta clave: {key}"


# ── T2: build_desk_state con fecha hoy — no crashea ─────────────────────────

def test_build_desk_state_hoy_no_crashea():
    """Fecha de hoy → puede tener o no archivos, pero nunca debe crashear."""
    with patch("live_desk.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        state = build_desk_state()  # usa hoy
    assert "p4_risk" in state
    assert "governor_code" in state["p4_risk"]


# ── T3: accionable_ahora — governor BLOCK → [] ───────────────────────────────

def test_accionable_governor_block():
    """governor_code=2 (BLOCK) → accionable_ahora retorna []."""
    state = _state(gov_code=2, breaks=[{
        "estado": "BREAK_CONFIRMADO",
        "jugador": "Alcaraz",
        "pick": "Alcaraz",
        "drift_pct": 18.0,
        "hipotesis": "H100-01",
        "n_actual": 2,
    }])
    result = accionable_ahora(state)
    assert result == []


# ── T4: accionable_ahora — KGR<0 → [] ───────────────────────────────────────

def test_accionable_kgr_negativo():
    """KGR<0 → accionable_ahora retorna [] (REGLA-HF-5)."""
    state = _state(gov_code=0, kgr=-0.05)
    result = accionable_ahora(state)
    assert result == []


# ── T5: accionable_ahora — BREAK_CONFIRMADO + PASS → señal ──────────────────

def test_accionable_break_confirmado_pass():
    """BREAK_CONFIRMADO + governor PASS → retorna la señal."""
    state = _state(gov_code=0, kgr=0.8, breaks=[{
        "estado": "BREAK_CONFIRMADO",
        "jugador": "Sinner",
        "pick": "Sinner",
        "drift_pct": 16.0,
        "hipotesis": "H100-01",
        "n_actual": 1,
    }])
    result = accionable_ahora(state)
    # Nodo-114: FAVORITOS_ZERO siempre aparece → al menos 1 BREAK_CONFIRMADO
    breaks = [a for a in result if a["tipo"] == "BREAK_CONFIRMADO"]
    assert len(breaks) == 1, f"Debe haber exactamente 1 BREAK_CONFIRMADO, got {result}"
    assert breaks[0]["jugador"] == "Sinner"
    assert breaks[0]["color"] == "amber"  # pre-graduacion siempre amber


# ── T6: render_html — contiene los 7 paneles ─────────────────────────────────

def test_render_html_contiene_7_paneles():
    """render_html incluye etiquetas de los 7 paneles."""
    state = _state()
    html = render_html(state)
    for label in ["P1 TAPE", "P2 BREAK", "P3 CONVERGENCE", "P4 RISK",
                  "P5 EXECUTION", "P6 P", "P7 CLOCK"]:
        assert label in html, f"Falta panel {label} en HTML"


# ── T7: render_html — governor BLOCK → banner HALT ───────────────────────────

def test_render_html_governor_block_banner_halt():
    """governor_code=2 → HTML contiene 'HALT' y 'BLOCK'."""
    state = _state(gov_code=2)
    html = render_html(state)
    assert "HALT" in html
    assert "BLOCK" in html


# ── T8: render_html — P4 aparece antes que P2 en el HTML ─────────────────────

def test_render_html_p4_antes_que_p2():
    """P4 RISK debe aparecer antes que P2 BREAK en el HTML (P4 manda — §2 regla 3)."""
    state = _state()
    html = render_html(state)
    pos_p4 = html.find("P4 RISK")
    pos_p2 = html.find("P2 BREAK")
    assert pos_p4 < pos_p2, "P4 RISK debe aparecer antes que P2 BREAK en el HTML"


# ── T9: accionable_ahora — GCS → color=green ────────────────────────────────

def test_accionable_gcs_color_green():
    """Pick con gcs_active=True → accionable con color=green (H60-01 GRADUADA)."""
    state = _state(gov_code=0, kgr=1.0, conv_picks=[{
        "jugador": "Djokovic",
        "score_directo": 3,
        "confidence_flag": "STRONG",
        "markov_favorito": "HOT",
        "rival_value_flag": False,
        "rival": "",
        "gcs_active": True,
    }])
    result = accionable_ahora(state)
    gcs_signals = [a for a in result if a["tipo"] == "GCS"]
    assert len(gcs_signals) >= 1
    assert gcs_signals[0]["color"] == "green"
    assert gcs_signals[0]["hipotesis"] == "H60-01"

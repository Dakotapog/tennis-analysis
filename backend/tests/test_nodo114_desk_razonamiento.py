"""
tests/test_nodo114_desk_razonamiento.py — REGLA-T53: Nodo-114 live_desk v2.

§2 linea_razonamiento(), §4 FAVORITOS primera clase, §3 P8 dual-book.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_desk import linea_razonamiento, accionable_ahora, build_desk_state


# ── T1: pick sintético con 4 señales → string contiene las 4 en orden ────────

def test_linea_razonamiento_4_senales_en_orden():
    """T53: pick con 4 señales activas → string contiene las 4 en orden."""
    pick = {
        "tipo": "BREAK_CONFIRMADO",
        "hipotesis": "H100-01",
        "n_actual": 5,
        "n_stop": 20,
        "drift_pct": -18.0,
        "meta_score": 4,
        "señales_activas": ["HOT", "STRONG", "ELO_DOM", "RFI"],
        "clv": 2.3,
        "n_h2h": 3,
        "governor_code": 0,
    }
    line = linea_razonamiento(pick)
    assert "[H100-01 5/20]" in line, "Gate debe estar al frente"
    assert "BREAK_CONFIRMADO" in line
    assert "meta_score=4" in line
    assert "HOT" in line and "STRONG" in line
    assert "CLV+2.3%" in line
    assert "n_h2h=3" in line
    assert "governor PASS" in line
    # Orden: gate primero, luego tipo, luego meta_score, luego CLV
    idx_gate = line.index("[H100-01")
    idx_break = line.index("BREAK_CONFIRMADO")
    idx_meta = line.index("meta_score=4")
    idx_clv = line.index("CLV")
    assert idx_gate < idx_break < idx_meta < idx_clv, "Orden de señales incorrecto"


# ── T2: governor WARN → línea empieza con el gate de riesgo ─────────────────

def test_linea_razonamiento_governor_warn_al_frente():
    """T53: pick con governor WARN → gate H-XX primero, luego 'governor WARN'."""
    pick = {
        "tipo": "GCS",
        "hipotesis": "H60-01",
        "n_actual": 54,
        "n_stop": 54,
        "governor_code": 1,  # WARN
        "meta_score": 0,
        "señales_activas": [],
    }
    line = linea_razonamiento(pick)
    assert line.startswith("[H60-01 54/54]"), f"Gate debe ser primero: {line}"
    assert "governor WARN" in line


# ── T3: governor BLOCK → "governor BLOCK" en la línea ───────────────────────

def test_linea_razonamiento_governor_block():
    """governor BLOCK reflejado en la línea de razonamiento."""
    pick = {
        "tipo": "RIVAL_VALUE",
        "hipotesis": "H88-01",
        "n_actual": 3,
        "n_stop": 30,
        "governor_code": 2,
        "señales_activas": [],
    }
    line = linea_razonamiento(pick)
    assert "governor BLOCK" in line


# ── T4: mejor_precio → aparece al final con → ────────────────────────────────

def test_linea_razonamiento_mejor_precio():
    """Mejor precio P8 aparece al final con flecha →."""
    pick = {
        "tipo": "GCS",
        "hipotesis": "H60-01",
        "n_actual": 54,
        "n_stop": 54,
        "governor_code": 0,
        "señales_activas": [],
        "mejor_precio": {"casa": "flashscore", "cuota": 2.35, "gain_pct": 5.4},
    }
    line = linea_razonamiento(pick)
    assert "→ mejor precio: flashscore @2.35 (+5.4% vs plan)" in line
    # → debe ser lo último
    assert line.index("→") > line.index("governor")


# ── T5: FAVORITOS_ZERO → línea descriptiva de instrucción ────────────────────

def test_linea_razonamiento_favoritos_zero():
    """FAVORITOS_ZERO → línea muestra instrucción de correr el builder."""
    pick = {
        "tipo": "FAVORITOS_ZERO",
        "hipotesis": "H110-01",
        "n_actual": 8,
        "n_stop": 30,
        "governor_code": 0,
        "señales_activas": [],
        "nota": "python3 favoritos_combo_builder.py --bankroll 125000",
    }
    line = linea_razonamiento(pick)
    assert "FAVORITOS: sin correr" in line
    assert "favoritos_combo_builder.py" in line


# ── T6: FAVORITOS_COMPUESTOS → aparece en accionable_ahora ───────────────────

def test_favoritos_primera_clase_en_accionables():
    """FAVORITOS_COMPUESTOS siempre aparece en accionable_ahora (con o sin combos)."""
    state = {
        "fecha": "2026-07-17",
        "p4_risk": {"governor_code": 0, "kgr_sesion": 1.0},
        "p2_break": {"breaks": []},
        "p3_convergence": {"picks": []},
        "p8_books": {"picks": {}},
    }
    acc = accionable_ahora(state)
    tipos = [a["tipo"] for a in acc]
    assert "FAVORITOS_COMPUESTOS" in tipos or "FAVORITOS_ZERO" in tipos, \
        "FAVORITOS debe estar siempre en accionables (primera clase)"


# ── T7: P4 BLOCK → lista vacía (incluso sin FAVORITOS exception) ─────────────

def test_favoritos_bloqueado_con_governor_block():
    """Con governor BLOCK, accionable_ahora retorna [] (P4 MANDA)."""
    state = {
        "fecha": "2026-07-17",
        "p4_risk": {"governor_code": 2, "kgr_sesion": 1.0},
        "p2_break": {"breaks": []},
        "p3_convergence": {"picks": []},
        "p8_books": {"picks": {}},
    }
    acc = accionable_ahora(state)
    assert acc == [], "Governor BLOCK debe vaciar los accionables"


# ── T8: P8 fixture 2 feeds → pick tiene mejor_precio ─────────────────────────

def test_p8_mejor_precio_aparece_en_accionable():
    """Con p8_books poblado, el accionable BREAK recibe mejor_precio."""
    state = {
        "fecha": "2026-07-17",
        "p4_risk": {"governor_code": 0, "kgr_sesion": 1.0},
        "p2_break": {"breaks": [{
            "estado": "BREAK_CONFIRMADO",
            "jugador": "Alcaraz",
            "pick": "Alcaraz",
            "drift_pct": -18.0,
            "n_actual": 2,
        }]},
        "p3_convergence": {"picks": []},
        "p8_books": {
            "picks": {
                "alcaraz": {
                    "jugador": "Alcaraz",
                    "casa": "flashscore",
                    "cuota": 2.10,
                    "gain_pct": 5.0,
                    "divergencia_pct": 5.0,
                }
            }
        },
    }
    acc = accionable_ahora(state)
    breaks = [a for a in acc if a["tipo"] == "BREAK_CONFIRMADO"]
    assert breaks, "Debe haber un BREAK_CONFIRMADO"
    assert breaks[0].get("mejor_precio") is not None, "mejor_precio debe estar enriquecido"
    assert breaks[0]["mejor_precio"]["casa"] == "flashscore"


# ── T9: P8 divergencia > 8% → linea_razonamiento no aplica badge (es P8 render) ─

def test_p8_alta_divergencia_en_mejor_precio():
    """P8 con divergencia >8% → gain_pct positivo aparece en linea_razonamiento."""
    pick = {
        "tipo": "BREAK_CONFIRMADO",
        "hipotesis": "H100-01",
        "n_actual": 3,
        "n_stop": 20,
        "drift_pct": -15.0,
        "governor_code": 0,
        "señales_activas": [],
        "mejor_precio": {"casa": "flashscore", "cuota": 2.50, "gain_pct": 9.0},
    }
    line = linea_razonamiento(pick)
    assert "+9.0%" in line, "Ganancia en mejor precio debe aparecer en la línea"

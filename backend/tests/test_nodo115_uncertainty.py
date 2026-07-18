"""
tests/test_nodo115_uncertainty.py — REGLA-T53: Nodo-115 incertidumbre visible.

T1. _peso_evidencia — n=4 → pct=17, label=PRIOR MANDA, color rojo
T2. _peso_evidencia — n=33 → pct=62, color verde
T3. _gate_barra — H110-01 8/30 → "22 faltan" en string
T4. _gate_barra — n_actual >= n_stop → "GRADUADA"
T5. _build_que_falta — pick cuota_fav=2.35 → condicion cuota_rango, detalle contiene "2.10"
T6. render_html — demo state → data-tipo attrs + barra evidencia + panel QUÉ FALTA
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from live_desk import _peso_evidencia, _gate_barra, render_html, _demo_state


# ── T1: n=4 → 17% PRIOR MANDA ────────────────────────────────────────────────

def test_peso_evidencia_n4():
    """n=4 → shrinkage=4/24≈17%, etiqueta PRIOR MANDA, color rojo."""
    ev = _peso_evidencia(4)
    assert ev["pct"] == 17
    assert ev["label"] == "PRIOR MANDA"
    assert ev["color"] == "#f85149"
    assert ev["n"] == 4
    assert "█" in ev["bar"]


# ── T2: n=33 → 62% peso propio ───────────────────────────────────────────────

def test_peso_evidencia_n33():
    """n=33 → shrinkage=33/53≈62%, color verde."""
    ev = _peso_evidencia(33)
    assert ev["pct"] == 62
    assert ev["color"] == "#3fb950"
    assert "PRIOR" not in ev["label"]


# ── T3: _gate_barra H110-01 8/30 → "22 faltan" ───────────────────────────────

def test_gate_barra_8_de_30():
    """8/30 → string contiene '22 faltan'."""
    s = _gate_barra(8, 30)
    assert "22 faltan" in s
    assert "8/30" in s
    assert "█" in s


# ── T4: n_actual >= n_stop → GRADUADA ────────────────────────────────────────

def test_gate_barra_graduada():
    """n_actual=54 >= n_stop=30 → 'GRADUADA'."""
    assert _gate_barra(54, 30) == "GRADUADA"
    assert _gate_barra(20, 20) == "GRADUADA"


# ── T5: _build_que_falta — cuota_fav > 2.10 → condicion cuota_rango ──────────

def test_build_que_falta_cuota_techo(tmp_path):
    """Pick con cuota_fav=2.35 > 2.10 → condicion=cuota_rango, detalle contiene '2.10'."""
    import json
    from live_desk import _build_que_falta, REPORTS

    edge = {
        "watchlist": [{
            "favorito_predicho": "Rublev",
            "p_modelo": 0.65,
            "cuota_favorito": 2.35,
            "cuota_rival": 1.55,
            "confidence_flag": "MOD",
            "ranking_favorito": 10,
            "ranking_rival": 25,
            "n_calibracion": 8,
        }]
    }
    fecha = "2099-01-01"
    fecha_compact = "20990101"
    er_path = tmp_path / f"edge_report_{fecha_compact}_test.json"
    er_path.write_text(json.dumps(edge))

    with patch("live_desk.REPORTS", tmp_path):
        result = _build_que_falta(fecha)

    assert len(result) == 1
    assert result[0]["condicion"] == "cuota_rango"
    assert "2.10" in result[0]["detalle"]
    assert result[0]["jugador"] == "Rublev"


# ── T6: render_html demo → data-tipo + barras + QUÉ FALTA ────────────────────

def test_render_html_demo_nodo115():
    """Estado demo → HTML contiene data-tipo, barras evidencia U2/U3, panel QUÉ FALTA."""
    state = _demo_state()
    html = render_html(state)

    # data-tipo attrs para facetas
    assert 'data-tipo=' in html, "Falta data-tipo en filas accionables"

    # U2 barra evidencia (█ en la celda de evidencia)
    assert "█" in html, "Falta barra U2 evidencia"
    assert "PRIOR" in html or "%" in html, "Falta etiqueta U2"

    # U3 gate barra
    assert "faltan" in html or "GRADUADA" in html, "Falta U3 gate barra"

    # Panel QUÉ FALTA
    assert "QUÉ FALTA" in html, "Falta panel QUÉ FALTA"
    assert "favorito_claro" in html or "cuota_rango" in html, \
        "Falta condición en QUÉ FALTA"

    # JS embebido
    assert "filtrarTipo" in html, "Falta JS filtrarTipo"
    assert "toggleDetalle" in html, "Falta JS toggleDetalle"

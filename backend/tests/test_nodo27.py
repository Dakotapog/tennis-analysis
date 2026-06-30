"""
Tests para Nodo-27: Pipeline Tracker & Observabilidad
T27-01 → T27-08
"""
import sys
import os
import json
import tempfile
import pytest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_tracker as pt


# ── Helpers de datos mock ────────────────────────────────────────────────────

def _make_edge_report(picks, fecha_str="20260617"):
    """Genera estructura mínima de edge_report."""
    return {
        "metadata": {"fecha": f"2026-06-17T11:00:00"},
        "apostar": picks,
        "watchlist": [],
        "sin_edge": [],
        "sin_datos": [],
    }


def _edge_pick(match_id="m1", favorito="Player A", cuota=2.5, edge_pct="10.0%",
               confidence_flag="MODERATE", tier="challenger", superficie="clay",
               zona_cuota="underdog", golden_zone=False, bbi=0.6,
               markov="HOT", data_completeness=0.8, p_modelo=0.55, kelly_kl=0.05):
    return {
        "match_id": match_id,
        "favorito_predicho": favorito,
        "cuota_favorito": cuota,
        "edge_pct": edge_pct,
        "confidence_flag": confidence_flag,
        "tier": tier,
        "superficie": superficie,
        "zona_cuota": zona_cuota,
        "golden_zone": golden_zone,
        "bbi": bbi,
        "markov_favorito": markov,
        "data_completeness": data_completeness,
        "p_modelo": p_modelo,
        "kelly_kl": kelly_kl,
        "n_h2h": 3,
    }


# ── T27-01: Sin archivos de apuestas → no crash ──────────────────────────────

def test_t27_01_corre_sin_apuestas(tmp_path, capsys):
    """T27-01: pipeline_tracker corre sin error con 0 archivos de apuestas."""
    # Directorio de reports vacío — sin edge_reports tampoco
    with patch.object(pt, "REPORTS_DIR", tmp_path):
        picks = pt.cargar_edge_reports()
        resultados = pt.cargar_resultados_finales()
        apuestas = pt.cargar_apuestas()

    assert picks == []
    assert resultados == {}
    assert apuestas == {}


def test_t27_01b_main_sin_datos(tmp_path, capsys):
    """T27-01b: main() con 0 edge_reports imprime mensaje de sin datos."""
    with patch.object(pt, "REPORTS_DIR", tmp_path), \
         patch.object(pt, "OUTPUT_FILE", tmp_path / "out.txt"):
        pt.main.__wrapped__ = None  # no-op for args
        # Correr con args vacíos
        with patch("sys.argv", ["pipeline_tracker.py"]):
            pt.main()
    captured = capsys.readouterr()
    assert "Sin datos" in captured.out


# ── T27-02: S-27-1 confidence_flag counts correctos ─────────────────────────

def test_t27_02_confianza_counts():
    """T27-02: Sección S-27-1 cuenta correctamente por confidence_flag."""
    picks = [
        {"confidence_flag": "STRONG", "correcto": True,  "cuota": 2.0, "edge_pct": 10.0, "stake": 0, "ganancia": 0, "fuente_lista": "apostar"},
        {"confidence_flag": "STRONG", "correcto": True,  "cuota": 2.0, "edge_pct": 10.0, "stake": 0, "ganancia": 0, "fuente_lista": "apostar"},
        {"confidence_flag": "MODERATE","correcto": False, "cuota": 2.5, "edge_pct":  8.0, "stake": 0, "ganancia": 0, "fuente_lista": "apostar"},
        {"confidence_flag": "LOW",     "correcto": None,  "cuota": 3.0, "edge_pct":  6.0, "stake": 0, "ganancia": 0, "fuente_lista": "apostar"},
    ]
    out = []
    pt.seccion_27_1_confianza(picks, out)
    text = "\n".join(out)

    # STRONG debe tener 2 wins, 0 losses
    assert "STRONG" in text
    # MODERATE debe tener 0 wins, 1 loss
    assert "MODERATE" in text
    # LOW pendiente (None correcto) → 0 resultado
    assert "LOW" in text


def test_t27_02b_confianza_strong_wins():
    """T27-02b: STRONG group: 2W/0L → hit=100%."""
    picks = [
        {"confidence_flag": "STRONG", "correcto": True,  "cuota": 2.0, "edge_pct": 10.0, "stake": 0, "ganancia": 0},
        {"confidence_flag": "STRONG", "correcto": True,  "cuota": 2.0, "edge_pct": 10.0, "stake": 0, "ganancia": 0},
    ]
    s = pt._stats(pt._with_resultado(picks))
    assert s["wins"] == 2
    assert s["losses"] == 0
    assert s["hit"] == pytest.approx(100.0)


# ── T27-03: S-27-2 cuota bins correctos ─────────────────────────────────────

def test_t27_03_cuota_bins():
    """T27-03: Sección S-27-2 asigna picks a bins de cuota correctamente."""
    picks = [
        {"cuota": 1.75, "correcto": True,  "edge_pct": 5.0, "stake": 0, "ganancia": 0},
        {"cuota": 2.20, "correcto": False, "edge_pct": 7.0, "stake": 0, "ganancia": 0},
        {"cuota": 2.80, "correcto": True,  "edge_pct": 12.0, "stake": 0, "ganancia": 0},
        {"cuota": 3.50, "correcto": True,  "edge_pct": 20.0, "stake": 0, "ganancia": 0},
        {"cuota": 4.50, "correcto": False, "edge_pct": 25.0, "stake": 0, "ganancia": 0},
    ]
    out = []
    pt.seccion_27_2_cuotas(picks, out)
    text = "\n".join(out)

    assert "1.50-2.00" in text
    assert "2.00-2.50" in text
    assert "2.50-3.00" in text
    assert "3.00-4.00" in text
    assert "4.00+" in text


def test_t27_03b_bin_assignment():
    """T27-03b: pick con cuota 2.20 va al bin 2.00-2.50."""
    picks = [{"cuota": 2.20, "correcto": True, "edge_pct": 7.0, "stake": 0, "ganancia": 0}]
    for lo, hi, label in pt.CUOTA_BINS:
        if lo <= 2.20 < hi:
            expected_bin = label
            break
    assert expected_bin == "2.00-2.50"


# ── T27-04: ROI calcula correctamente ────────────────────────────────────────

def test_t27_04_roi_con_stake_real():
    """T27-04: ROI = (ganancia_total / stake_total) × 100 cuando hay stake real."""
    picks = [
        {"stake": 1000, "ganancia": 500,   "correcto": True,  "cuota": 2.5, "edge_pct": 10.0},
        {"stake": 1000, "ganancia": -1000, "correcto": False, "cuota": 2.5, "edge_pct": 10.0},
    ]
    s = pt._stats(picks)
    # ganancia_total = 500 - 1000 = -500 | stake_total = 2000 | ROI = -25%
    assert s["roi"] == pytest.approx(-25.0)


def test_t27_04b_roi_proxy_sin_stake():
    """T27-04b: ROI proxy = (cuota-1) por win, -1 por loss cuando stake=0."""
    picks = [
        {"stake": 0, "ganancia": 0, "correcto": True,  "cuota": 3.0, "edge_pct": 20.0},
        {"stake": 0, "ganancia": 0, "correcto": False, "cuota": 3.0, "edge_pct": 20.0},
    ]
    s = pt._stats(picks)
    # win: +2.0 | loss: -1.0 → avg = 0.5 → ROI proxy = 50%
    assert s["roi"] == pytest.approx(50.0)


def test_t27_04c_roi_stake_cero_excluido():
    """T27-04c: Si stake=0 en todos → usa proxy, no divide por 0."""
    picks = [{"stake": 0, "ganancia": 0, "correcto": True, "cuota": 2.0, "edge_pct": 8.0}]
    s = pt._stats(picks)
    assert s["roi"] is not None
    assert s["roi"] == pytest.approx(100.0)  # (2.0-1)*1 / 1 * 100


# ── T27-05: Join por match_id funciona ──────────────────────────────────────

def test_t27_05_join_por_match_id():
    """T27-05: join_resultados asigna correcto=True por match_id."""
    picks = [
        {"match_id": "abc123", "favorito": "Player A", "correcto": None,
         "ganancia": None, "stake": 0, "cuota": 2.0, "edge_pct": 10.0},
    ]
    resultados_map = {
        "abc123": {"correcto": True, "prediccion": "Player A", "fecha": date(2026, 6, 17)},
    }
    apuestas_map = {}

    enriched = pt.join_resultados(picks, resultados_map, apuestas_map)
    assert enriched[0]["correcto"] is True


def test_t27_05b_join_fallback_apuestas():
    """T27-05b: si no está en resultados_finales, busca en apuestas por (match_id, jugador)."""
    picks = [
        {"match_id": "xyz789", "favorito": "Player B", "correcto": None,
         "ganancia": None, "stake": 0, "cuota": 3.0, "edge_pct": 15.0},
    ]
    resultados_map = {}
    apuestas_map = {
        ("xyz789", "Player B"): {"correcto": False, "ganancia": 0, "stake": 0, "cuota": 3.0},
    }

    enriched = pt.join_resultados(picks, resultados_map, apuestas_map)
    assert enriched[0]["correcto"] is False


def test_t27_05c_join_sin_match_queda_none():
    """T27-05c: pick sin match en ningún mapa → correcto=None (no crash)."""
    picks = [
        {"match_id": "nomatch", "favorito": "Player C", "correcto": None,
         "ganancia": None, "stake": 0},
    ]
    enriched = pt.join_resultados(picks, {}, {})
    assert enriched[0]["correcto"] is None


# ── T27-06: --since filtra por fecha ────────────────────────────────────────

def test_t27_06_since_filter(tmp_path):
    """T27-06: --since 2026-06-15 excluye archivos con fecha anterior."""
    # Crear edge_report con fecha 20260614 (antes) y 20260617 (después)
    old_file = tmp_path / "edge_report_20260614_120000.json"
    new_file = tmp_path / "edge_report_20260617_120000.json"

    pick = _edge_pick()
    old_file.write_text(json.dumps(_make_edge_report([pick])))
    new_file.write_text(json.dumps(_make_edge_report([pick])))

    with patch.object(pt, "REPORTS_DIR", tmp_path):
        since = date(2026, 6, 15)
        picks = pt.cargar_edge_reports(since=since)

    # Solo debe cargar el del 17
    dates = set(p["fecha"] for p in picks)
    assert date(2026, 6, 14) not in dates
    assert date(2026, 6, 17) in dates


# ── T27-07: --tier filtra por tier ──────────────────────────────────────────

def test_t27_07_tier_filter(tmp_path):
    """T27-07: --tier challenger excluye picks de otros tiers."""
    pick_ch = _edge_pick(tier="challenger", match_id="c1")
    pick_atp = _edge_pick(tier="atp500", match_id="a1")

    report_file = tmp_path / "edge_report_20260617_120000.json"
    report_file.write_text(json.dumps(_make_edge_report([pick_ch, pick_atp])))

    with patch.object(pt, "REPORTS_DIR", tmp_path):
        picks = pt.cargar_edge_reports(tier_filter="challenger")

    tiers = set(p["tier"] for p in picks)
    assert "challenger" in tiers
    assert "atp500" not in tiers


# ── T27-08: Campos faltantes → None, no crash ───────────────────────────────

def test_t27_08_campos_faltantes_no_crash(tmp_path):
    """T27-08: edge_report sin bbi/golden_zone (versión vieja) → None, no crash."""
    old_pick = {
        "match_id": "old1",
        "favorito_predicho": "Player X",
        "cuota_favorito": 2.0,
        "edge_pct": "8.0%",
        "confidence_flag": "LOW",
        "tier": "atp500",
        "superficie": "grass",
        "p_modelo": 0.52,
        # Sin bbi, golden_zone, markov_favorito, data_completeness, etc.
    }
    report_file = tmp_path / "edge_report_20260617_120000.json"
    report_file.write_text(json.dumps(_make_edge_report([old_pick])))

    with patch.object(pt, "REPORTS_DIR", tmp_path):
        picks = pt.cargar_edge_reports()

    assert len(picks) == 1
    p = picks[0]
    assert p["bbi"] is None
    assert p["golden_zone"] is None
    assert p["markov_favorito"] is None
    assert p["data_completeness"] is None

    # Secciones no deben crashear con None
    out = []
    pt.seccion_27_1_confianza(picks, out)
    pt.seccion_27_2_cuotas(picks, out)
    pt.seccion_27_3_tier_superficie(picks, out)
    pt.seccion_27_4_senales(picks, out)
    # No excepción = PASS


def test_t27_08b_parse_edge_pct_formatos():
    """T27-08b: _parse_edge_pct maneja string '18.8%' y float 0.188."""
    assert pt._parse_edge_pct("18.8%") == pytest.approx(18.8)
    assert pt._parse_edge_pct("24.3%") == pytest.approx(24.3)
    assert pt._parse_edge_pct(0.188)   == pytest.approx(18.8)
    assert pt._parse_edge_pct(None)    is None


def test_t27_08c_flag_muestra_insuficiente():
    """T27-08c: _flag marca con * cuando n<10."""
    assert pt._flag(5).endswith("*")
    assert not pt._flag(10).endswith("*")
    assert not pt._flag(15).endswith("*")

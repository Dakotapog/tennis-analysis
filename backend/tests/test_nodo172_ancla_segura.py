"""REGLA-T53 — Nodo-172: Ancla Segura (1 pierna STRONG + fillers STRONG/MODERATE).

Origen: combo ganador 2026-08-03 (Castellanos/Mejia/Udvardy/Ruse @49.33x) auditado
a pedido del usuario — solo Castellanos era confidence_flag=STRONG (p_modelo=0.801);
Mejia/Udvardy/Ruse eran LOW (p_modelo 0.51-0.54, casi coinflip) que acertaron por
varianza. Estos tests invocan build_ancla_segura_combos() real — nunca hardcodean
la fórmula de cuota_combo/p_todas, solo verifican el filtro de confidence_flag y
el formato de coupon (REGLA-BAT-1).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import betplay_combo_builder as bcb
from edge_calculator import GATE_VERSION


def _write_edge_report(tmp_path, picks):
    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": [],
        "watchlist": picks,
    }
    p = tmp_path / "edge_report_test.json"
    p.write_text(json.dumps(edge_data), encoding="utf-8")
    return str(p)


def _pick(name, cuota, p_modelo, confidence_flag, tier="itf"):
    return {
        "favorito_predicho": name,
        "cuota_favorito": cuota,
        "p_modelo": p_modelo,
        "confidence_flag": confidence_flag,
        "tier": tier,
        "kambi_disponible": True,
    }


def _fake_outcomes_for(names):
    """fetch_kambi_outcomes()/find_outcome() mock: cada jugador resuelve a un
    outcome_id único y misma cuota que en el edge_report."""
    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        if jugador in names:
            idx = names.index(jugador)
            return {"outcome_id": 1000 + idx, "odds": cuota}, "ok"
        return None, "no encontrado"
    return _fake_find_outcome


def _run_build(tmp_path, picks, **kwargs):
    edge_path = _write_edge_report(tmp_path, picks)
    names = [p["favorito_predicho"] for p in picks]
    with patch.object(bcb, "_find_latest_edge_report", return_value=edge_path), \
         patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_outcomes_for(names)):
        return bcb.build_ancla_segura_combos(**kwargs)


def test_172_01_excluye_fillers_low(tmp_path):
    """Réplica del caso real 2026-08-03: 1 STRONG + 3 LOW no debe generar combo."""
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Mejia N.", 3.00, 0.513, "LOW"),
        _pick("Udvardy P.", 2.43, 0.51, "LOW"),
        _pick("Ruse G.", 2.02, 0.541, "LOW"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=3)
    assert links == []
    assert meta == {}


def test_172_02_acepta_fillers_strong_moderate(tmp_path):
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.65, "STRONG"),
        _pick("Ruse G.", 1.80, 0.58, "MODERATE"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=3)
    assert len(links) == 1
    combo = links[0]
    assert combo["piernas"] == 4
    confidences = [leg["confidence_flag"] for leg in combo["legs"]]
    assert "LOW" not in confidences


def test_172_03_ancla_es_siempre_strong(tmp_path):
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Otro X.", 2.80, 0.70, "STRONG"),
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.65, "MODERATE"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=2)
    assert len(links) == 1
    ancla_legs = [l for l in links[0]["legs"] if l["tipo"] == "ancla"]
    assert len(ancla_legs) == 1
    assert ancla_legs[0]["confidence_flag"] == "STRONG"
    assert meta["ancla_confidence"] == "STRONG"


def test_172_04_sin_ancla_strong_no_genera_combo(tmp_path):
    picks = [
        _pick("Mejia N.", 3.00, 0.55, "MODERATE"),
        _pick("Udvardy P.", 2.43, 0.56, "MODERATE"),
        _pick("Ruse G.", 2.02, 0.54, "MODERATE"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=2)
    assert links == []


def test_172_05_pocos_fillers_calificados_no_genera_combo(tmp_path):
    """1 STRONG ancla + solo 1 filler STRONG/MODERATE disponible, pero n_fillers=3."""
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.51, "LOW"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=3)
    assert links == []


def test_172_06_coupon_format_regla_bat_1(tmp_path):
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.65, "STRONG"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=2)
    assert len(links) == 1
    url = links[0]["url"]
    assert "|ML" not in url
    assert url.endswith("||replace")
    ids = links[0]["outcome_ids"]
    assert ",".join(ids) in url


def test_172_07_cuota_bajo_1_50_excluida_regla_hf1(tmp_path):
    picks = [
        _pick("Castellanos Y.", 3.35, 0.801, "STRONG"),
        _pick("Barato X.", 1.20, 0.90, "STRONG"),  # REGLA-HF-1: cuota<1.50
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.65, "STRONG"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=2)
    assert len(links) == 1
    names = [l["jugador"] for l in links[0]["legs"]]
    assert "Barato X." not in names


def test_172_08_ancla_cuota_min_respetado(tmp_path):
    """STRONG con cuota debajo del umbral de ancla no califica como ancla."""
    picks = [
        _pick("Castellanos Y.", 1.80, 0.801, "STRONG"),  # STRONG pero cuota baja
        _pick("Mejia N.", 2.10, 0.62, "MODERATE"),
        _pick("Udvardy P.", 1.90, 0.65, "MODERATE"),
    ]
    links, meta = _run_build(tmp_path, picks, n_fillers=2, ancla_cuota_min=2.50)
    assert links == []

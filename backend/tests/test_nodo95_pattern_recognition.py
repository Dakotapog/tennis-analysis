"""
Tests REGLA-T53 — D90-09 PatternRecognition engine (Nodo-95 Sprint 4)
Invocan funciones reales del módulo. Nunca hardcodean la fórmula.
"""

import json
import math
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pattern_recognition as pr


# ── Wilson IC95 ───────────────────────────────────────────────────────────────

def test_wilson_ic95_perfect():
    """n=100, wins=100 → lower bound debe ser alto."""
    lo, hi = pr._wilson_ic95(100, 100)
    assert lo > 0.96
    assert hi >= 0.9999  # float precision


def test_wilson_ic95_zero_wins():
    lo, hi = pr._wilson_ic95(0, 20)
    assert lo == 0.0
    assert hi < 0.20


def test_wilson_ic95_zero_n():
    lo, hi = pr._wilson_ic95(0, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ic95_half():
    """50% win rate, n=100 → IC cerca de [41%, 59%]."""
    lo, hi = pr._wilson_ic95(50, 100)
    assert 0.40 < lo < 0.50
    assert 0.50 < hi < 0.60


def test_wilson_ic95_symmetric():
    """p=0.5 debe ser simétrico alrededor de 0.5."""
    lo, hi = pr._wilson_ic95(50, 100)
    mid = (lo + hi) / 2
    assert abs(mid - 0.5) < 0.02


def test_wilson_ic95_contained():
    lo, hi = pr._wilson_ic95(30, 80)
    assert 0.0 <= lo <= hi <= 1.0


# ── Segment key ───────────────────────────────────────────────────────────────

def test_segment_key_normal():
    snap = {"tier": "itf", "superficie": "clay"}
    assert pr._segment_key(snap, "tier") == "itf"
    assert pr._segment_key(snap, "superficie") == "clay"


def test_segment_key_none_returns_question_mark():
    snap = {"markov_favorito": None}
    assert pr._segment_key(snap, "markov_favorito") == "?"


def test_segment_key_missing_returns_question_mark():
    snap = {}
    assert pr._segment_key(snap, "confidence_flag") == "?"


# ── Fixture builder ───────────────────────────────────────────────────────────

def _make_record(tier="itf", superficie="clay", zona="underdog",
                 markov="HOT", confidence="STRONG", apostar=True,
                 cuota=2.0, resultado="WON"):
    return {
        "pick_snapshot": {
            "tier": tier,
            "superficie": superficie,
            "zona_cuota": zona,
            "markov_favorito": markov,
            "confidence_flag": confidence,
            "apostar": apostar,
            "cuota_favorito": cuota,
        },
        "resolucion": {"resultado": resultado},
    }


def _make_jsonl(records: list[dict]) -> str:
    """Write records to a temp JSONL file, return path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        return f.name


# ── load_settled ──────────────────────────────────────────────────────────────

def test_load_settled_reads_settled_only():
    """Records sin 'resolucion' deben ser ignorados."""
    recs = [
        _make_record(resultado="WON"),
        {"pick_snapshot": {"tier": "itf"}, "note": "no resolucion"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sb_2026-07-01.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        loaded = pr.load_settled(td)
    assert len(loaded) == 1
    assert loaded[0]["resolucion"]["resultado"] == "WON"


def test_load_settled_apostar_only():
    recs = [
        _make_record(apostar=True, resultado="WON"),
        _make_record(apostar=False, resultado="LOST"),
        _make_record(apostar=True, resultado="LOST"),
    ]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sb_2026-07-01.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        loaded = pr.load_settled(td, apostar_only=True)
    assert len(loaded) == 2
    assert all(r["pick_snapshot"]["apostar"] is True for r in loaded)


def test_load_settled_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        loaded = pr.load_settled(td)
    assert loaded == []


# ── compute_segment_stats ─────────────────────────────────────────────────────

def test_segment_stats_basic():
    """n=10, wins=8, cuota=3.0 → breakeven=33.3%, IC_low≈49% > 33% → candidato True."""
    recs = [_make_record(cuota=3.0, resultado="WON")] * 8 + \
           [_make_record(cuota=3.0, resultado="LOST")] * 2
    rows = pr._compute_segment_stats(recs, "confidence_flag", min_n=5)
    strong_row = next(r for r in rows if r["value"] == "STRONG")
    assert strong_row["n"] == 10
    assert strong_row["wins"] == 8
    assert abs(strong_row["hit_pct"] - 0.8) < 0.001
    assert strong_row["candidate"] is True  # IC_low≈49% > break=33%


def test_segment_stats_no_candidate_below_min_n():
    """n=3 < min_n=5 → candidato=False incluso con hit% alta."""
    recs = [_make_record(cuota=2.0, resultado="WON")] * 3
    rows = pr._compute_segment_stats(recs, "confidence_flag", min_n=5)
    strong_row = next(r for r in rows if r["value"] == "STRONG")
    assert strong_row["candidate"] is False


def test_segment_stats_no_candidate_below_breakeven():
    """hit% < breakeven → candidato=False."""
    # cuota=1.5 → breakeven=66.7%, hit=40% → NO candidato
    recs = [_make_record(cuota=1.5, resultado="WON")] * 4 + \
           [_make_record(cuota=1.5, resultado="LOST")] * 6
    rows = pr._compute_segment_stats(recs, "confidence_flag", min_n=5)
    strong_row = next(r for r in rows if r["value"] == "STRONG")
    assert strong_row["candidate"] is False


def test_segment_stats_multiple_values():
    """Dos valores de tier → dos filas."""
    recs = [_make_record(tier="itf")] * 6 + [_make_record(tier="challenger")] * 4
    rows = pr._compute_segment_stats(recs, "tier", min_n=5)
    vals = {r["value"] for r in rows}
    assert "itf" in vals
    assert "challenger" in vals


# ── cross stats ───────────────────────────────────────────────────────────────

def test_cross_stats_basic():
    """n=7, wins=6, cuota=3.0 → breakeven=33%, IC_low≈49% > 33% → candidato True."""
    recs = [_make_record(tier="itf", confidence="STRONG", cuota=3.0, resultado="WON")] * 6 + \
           [_make_record(tier="itf", confidence="STRONG", cuota=3.0, resultado="LOST")] * 1
    rows = pr._compute_cross_stats(recs, "tier", "confidence_flag", min_n=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["dim"] == "tier×confidence_flag"
    assert row["value"] == "itf|STRONG"
    assert row["n"] == 7
    assert row["candidate"] is True


def test_cross_stats_dim_format():
    recs = [_make_record(superficie="clay", markov="HOT", cuota=2.2, resultado="WON")] * 5
    rows = pr._compute_cross_stats(recs, "superficie", "markov_favorito", min_n=5)
    assert rows[0]["dim"] == "superficie×markov_favorito"
    assert rows[0]["value"] == "clay|HOT"


# ── overall stats ─────────────────────────────────────────────────────────────

def test_overall_stats_hit_pct():
    recs = [_make_record(cuota=2.0, resultado="WON")] * 4 + \
           [_make_record(cuota=2.0, resultado="LOST")] * 6
    stats = pr._overall_stats(recs, min_n=5)
    assert stats["n"] == 10
    assert stats["wins"] == 4
    assert abs(stats["hit_pct"] - 0.4) < 0.001
    assert abs(stats["breakeven"] - 0.5) < 0.001


def test_overall_stats_empty():
    stats = pr._overall_stats([], min_n=5)
    assert stats["n"] == 0
    assert stats["hit_pct"] == 0


# ── run_pattern_recognition (integration) ────────────────────────────────────

def test_run_writes_json_file():
    """Engine debe escribir pattern_candidates_*.json."""
    recs = [_make_record(cuota=2.0, resultado="WON")] * 8 + \
           [_make_record(cuota=2.0, resultado="LOST")] * 2
    with tempfile.TemporaryDirectory() as td:
        shadow_dir = os.path.join(td, "shadow_book")
        os.makedirs(shadow_dir)
        with open(os.path.join(shadow_dir, "sb_2026-07-01.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        reports_dir = os.path.join(td, "reports")
        result = pr.run_pattern_recognition(shadow_dir, reports_dir, min_n=5)
        assert result  # no vacío
        report, out_path = result
        assert os.path.exists(out_path)  # check within with-block (dir still alive)
        assert "pattern_candidates_" in os.path.basename(out_path)


def test_run_report_structure():
    """Output JSON debe tener campos mandatorios."""
    recs = [_make_record(cuota=2.0, resultado="WON")] * 8 + \
           [_make_record(cuota=2.0, resultado="LOST")] * 2
    with tempfile.TemporaryDirectory() as td:
        shadow_dir = os.path.join(td, "shadow_book")
        os.makedirs(shadow_dir)
        with open(os.path.join(shadow_dir, "sb_2026-07-01.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        reports_dir = os.path.join(td, "reports")
        report, _ = pr.run_pattern_recognition(shadow_dir, reports_dir, min_n=5)

    assert "overall" in report
    assert "segments_1way" in report
    assert "segments_cross" in report
    assert "candidates_1way" in report
    assert "candidates_cross" in report
    assert "n_candidates" in report
    assert "note" in report
    assert "REPORTE_SOLO" in report["note"]


def test_run_no_preregistered_hypotheses_written():
    """El engine NO debe tocar preregistered_hypotheses.json."""
    recs = [_make_record(cuota=2.0, resultado="WON")] * 10
    with tempfile.TemporaryDirectory() as td:
        shadow_dir = os.path.join(td, "shadow_book")
        os.makedirs(shadow_dir)
        with open(os.path.join(shadow_dir, "sb_2026-07-01.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        reports_dir = os.path.join(td, "reports")
        pr.run_pattern_recognition(shadow_dir, reports_dir, min_n=5)
        # preregistered_hypotheses.json NO debe existir en td
        hyp_path = os.path.join(td, "validation", "preregistered_hypotheses.json")
        assert not os.path.exists(hyp_path)


def test_run_empty_shadow_dir():
    with tempfile.TemporaryDirectory() as td:
        shadow_dir = os.path.join(td, "shadow_book")
        os.makedirs(shadow_dir)
        reports_dir = os.path.join(td, "reports")
        result = pr.run_pattern_recognition(shadow_dir, reports_dir, min_n=5)
    assert not result  # vacío dict


def test_run_params_stored_in_report():
    recs = [_make_record(cuota=2.0, resultado="WON")] * 6
    with tempfile.TemporaryDirectory() as td:
        shadow_dir = os.path.join(td, "shadow_book")
        os.makedirs(shadow_dir)
        with open(os.path.join(shadow_dir, "sb_2026-07-01.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        reports_dir = os.path.join(td, "reports")
        report, _ = pr.run_pattern_recognition(shadow_dir, reports_dir, min_n=7, apostar_only=False)

    assert report["params"]["min_n"] == 7
    assert report["params"]["apostar_only"] is False

"""
tests/test_nodo58.py — Nodo-58: Dashboard de Observabilidad

T58-01: loaders con archivo ausente/corrupto → sin datos, no crash
T58-02: paridad de métricas — report_dict() y report() producen los mismos números
T58-03: Panel 6 — _decision_status con n=6 y n_stop=30 → NO_AUTORIZADO
T58-04: CLV con provenances distintas → columnas separadas, nunca agregado
T58-05: registro kambi_inplay → excluded=True, visible en contador aparte

REGLA-T53: nunca hardcodear fórmulas — siempre invocar funciones del módulo real.
"""

import json
import os
import re
import sys
import pytest

# ── path ──
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard import (
    _decision_status,
    _clv_by_provenance,
    load_shadow_report,
    load_edge_report,
    load_trader_plan,
)
from shadow_book import report_dict, report, wilson_ci


# ══════════════════════════════════════════════════════════════════════════════
# T58-01: Loaders con archivo ausente/corrupto → sin datos, no crash
# ══════════════════════════════════════════════════════════════════════════════

class TestT5801LoadersDegradation:

    def test_load_shadow_report_no_files(self, tmp_path, monkeypatch):
        """T58-01a: shadow_book.py ausente/vacío → {} sin crash."""
        # Parchamos SHADOW_DIR a un directorio vacío
        import shadow_book
        monkeypatch.setattr(shadow_book, "SHADOW_DIR", str(tmp_path))
        result = load_shadow_report(desde="2026-01-01", hasta="2026-01-01")
        # Puede ser {} (excepción) o dict vacío de report_dict
        assert isinstance(result, dict)

    def test_load_edge_report_missing(self, monkeypatch):
        """T58-01b: edge_report inexistente → {} sin crash."""
        import glob as glob_mod
        monkeypatch.setattr(glob_mod, "glob", lambda *a, **kw: [])
        result = load_edge_report()
        assert isinstance(result, dict)

    def test_load_trader_plan_missing(self, monkeypatch):
        """T58-01c: trader_plan inexistente → {} sin crash."""
        import glob as glob_mod
        monkeypatch.setattr(glob_mod, "glob", lambda *a, **kw: [])
        result = load_trader_plan()
        assert isinstance(result, dict)

    def test_load_shadow_report_corrupt_file(self, tmp_path, monkeypatch):
        """T58-01d: archivo .jsonl corrupto → {} o summary vacío, no crash."""
        import shadow_book
        sb_dir = tmp_path / "shadow_book"
        sb_dir.mkdir()
        corrupt = sb_dir / "sb_2026-01-01.jsonl"
        corrupt.write_text("NOT_JSON\n{corrupt}\n", encoding="utf-8")
        monkeypatch.setattr(shadow_book, "SHADOW_DIR", str(sb_dir))
        result = load_shadow_report(desde="2026-01-01", hasta="2026-01-01")
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# T58-02: Paridad de métricas — report_dict() == report() string
# ══════════════════════════════════════════════════════════════════════════════

class TestT5802MetricParity:

    @pytest.fixture()
    def shadow_dir_with_data(self, tmp_path, monkeypatch):
        """Crea un shadow book mínimo con un registro settled."""
        import shadow_book as sb

        sb_dir = tmp_path / "shadow_book"
        sb_dir.mkdir()
        monkeypatch.setattr(sb, "SHADOW_DIR", str(sb_dir))

        # Un registro settled: pick WON
        record = {
            "sb_id": "2026-07-03_test_a_b_ML",
            "logged_at": "2026-07-03T10:00:00",
            "match_key": "a_b",
            "_type": None,
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": {
                "apostar": True,
                "watchlist": False,
                "no_data": False,
                "edge": 0.12,
                "cuota_favorito": 2.10,
                "markov_favorito": "HOT",
                "tier": "challenger",
                "p_modelo": 0.62,
                "n_h2h": 2,
                "alignment_flag": "STRUCTURAL_ALPHA",
                "confidence_flag": "STRONG",
            },
            "resolucion": {
                "resultado": "WON",
                "cuota_cierre": 2.05,
                "cuota_cierre_provenance": "flashscore_ref",
                "clv_pct": 2.4,
                "pnl_flat_1u": 1.10,
                "settled_at": "2026-07-03T20:00:00",
            },
        }
        line = json.dumps(record, ensure_ascii=False)
        (sb_dir / "sb_2026-07-03.jsonl").write_text(line + "\n", encoding="utf-8")
        return sb_dir

    def test_n_settled_matches(self, shadow_dir_with_data):
        """T58-02a: n_settled en report_dict == n en report() string."""
        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        rstr = report(desde="2026-07-03", hasta="2026-07-03")
        assert rd['summary']['n_settled'] >= 1
        # La string tiene el conteo
        assert f"Settled: {rd['summary']['n_settled']}" in rstr

    def test_hit_pct_challenger_matches(self, shadow_dir_with_data):
        """T58-02b: hit% tier=challenger en report_dict tiene mismo valor que report() string."""
        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        rstr = report(desde="2026-07-03", hasta="2026-07-03")

        challenger_seg = next(
            (s for s in rd['segments'] if s.get('label') == 'tier=challenger'),
            None,
        )
        if challenger_seg is None:
            pytest.skip("No hay segmento challenger en datos de prueba")

        hit_pct = challenger_seg['hit_pct']
        # El string del report debe contener ese número
        assert str(hit_pct) in rstr, (
            f"hit_pct={hit_pct} de report_dict no aparece en report() string"
        )

    def test_wilson_ci_shared(self, shadow_dir_with_data):
        """T58-02c: IC Wilson del dict == wilson_ci(n, hits) — misma función."""
        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        for seg in rd['segments']:
            n = seg['n']
            hits = seg['hits']
            expected_ic = list(wilson_ci(n, hits))
            assert seg['ic'] == expected_ic, (
                f"Segmento {seg['label']}: IC del dict {seg['ic']} != wilson_ci({n},{hits}) = {expected_ic}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# T58-03: Panel 6 — _decision_status con n=6 y n_stop=30 → NO_AUTORIZADO
# ══════════════════════════════════════════════════════════════════════════════

class TestT5803DecisionStatus:

    def test_n_insuficiente(self):
        """T58-03a: n=6 < n_stop=30 → NO_AUTORIZADO."""
        status = _decision_status(n=6, n_stop=30)
        assert status == "NO_AUTORIZADO"

    def test_n_exacto_sin_ic(self):
        """T58-03b: n=n_stop sin IC → AUTORIZADO (sin criterio IC)."""
        status = _decision_status(n=30, n_stop=30)
        assert status == "AUTORIZADO"

    def test_ic_lower_debajo_breakeven(self):
        """T58-03c: n≥30 pero IC_lower ≤ breakeven → NO_AUTORIZADO."""
        status = _decision_status(n=30, n_stop=30, ic_lower=28.0, breakeven=31.1)
        assert status == "NO_AUTORIZADO"

    def test_clv_negativo(self):
        """T58-03d: n≥30 + IC ok pero CLV_median≤0 → NO_AUTORIZADO."""
        status = _decision_status(n=30, n_stop=30, ic_lower=45.0, breakeven=35.0, clv_median=-0.5)
        assert status == "NO_AUTORIZADO"

    def test_todos_criterios_ok(self):
        """T58-03e: n≥30 + IC_lower>breakeven + CLV>0 → AUTORIZADO."""
        status = _decision_status(n=35, n_stop=30, ic_lower=40.0, breakeven=33.0, clv_median=1.5)
        assert status == "AUTORIZADO"

    def test_extra_gate_false(self):
        """T58-03f: extra_gate=False → NO_AUTORIZADO aunque n y IC sean correctos."""
        status = _decision_status(n=50, n_stop=30, extra_gate=False)
        assert status == "NO_AUTORIZADO"


# ══════════════════════════════════════════════════════════════════════════════
# T58-04: CLV con provenances distintas → columnas separadas
# ══════════════════════════════════════════════════════════════════════════════

class TestT5804CLVProvenance:

    def _build_shadow_with_provenances(self, tmp_path, monkeypatch):
        """Construye shadow book con dos provenances distintas."""
        import shadow_book as sb
        sb_dir = tmp_path / "sb"
        sb_dir.mkdir()
        monkeypatch.setattr(sb, "SHADOW_DIR", str(sb_dir))

        records = []
        for i, (prov, clv) in enumerate([
            ("kambi_close", 3.5),
            ("kambi_close", 1.2),
            ("flashscore_ref", -0.8),
        ]):
            records.append({
                "sb_id": f"2026-07-03_test_{i}",
                "match_key": f"p{i}_q{i}",
                "es_qualifying": False,
                "season_transition_flag": False,
                "pick_snapshot": {
                    "apostar": True,
                    "edge": 0.08,
                    "cuota_favorito": 1.90,
                    "tier": "itf",
                },
                "resolucion": {
                    "resultado": "WON",
                    "cuota_cierre": 1.85,
                    "cuota_cierre_provenance": prov,
                    "clv_pct": clv,
                    "pnl_flat_1u": 0.90,
                    "settled_at": "2026-07-03T20:00:00",
                },
            })

        lines = "\n".join(json.dumps(r) for r in records)
        (sb_dir / "sb_2026-07-03.jsonl").write_text(lines, encoding="utf-8")
        return sb_dir

    def test_provenances_separadas(self, tmp_path, monkeypatch):
        """T58-04: dos provenances → dos claves separadas en clv_by_provenance."""
        self._build_shadow_with_provenances(tmp_path, monkeypatch)
        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        clv_prov = _clv_by_provenance(rd)
        assert "kambi_close" in clv_prov
        assert "flashscore_ref" in clv_prov
        # No deben estar agregadas en una sola clave
        assert "mixed" not in clv_prov
        assert "unknown" not in clv_prov or clv_prov.get("unknown", {}).get("n", 0) == 0

    def test_n_por_provenance_correcto(self, tmp_path, monkeypatch):
        """T58-04b: n por provenance es correcto."""
        self._build_shadow_with_provenances(tmp_path, monkeypatch)
        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        clv_prov = _clv_by_provenance(rd)
        assert clv_prov["kambi_close"]["n"] == 2
        assert clv_prov["flashscore_ref"]["n"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# T58-05: kambi_inplay → excluded=True en clv_by_provenance
# ══════════════════════════════════════════════════════════════════════════════

class TestT5805KambiInplayExcluded:

    def test_kambi_inplay_excluded(self, tmp_path, monkeypatch):
        """T58-05: registro kambi_inplay → excluded=True, visible en contador."""
        import shadow_book as sb
        sb_dir = tmp_path / "sb"
        sb_dir.mkdir()
        monkeypatch.setattr(sb, "SHADOW_DIR", str(sb_dir))

        record = {
            "sb_id": "2026-07-03_inplay_test",
            "match_key": "a_b",
            "es_qualifying": False,
            "season_transition_flag": False,
            "pick_snapshot": {
                "apostar": True,
                "edge": 0.07,
                "cuota_favorito": 1.80,
                "tier": "grand_slam",
            },
            "resolucion": {
                "resultado": "WON",
                "cuota_cierre": 1.60,
                "cuota_cierre_provenance": "kambi_inplay",
                "clv_pct": 12.5,
                "pnl_flat_1u": 0.80,
                "settled_at": "2026-07-03T20:00:00",
            },
        }
        (sb_dir / "sb_2026-07-03.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

        rd = report_dict(desde="2026-07-03", hasta="2026-07-03")
        clv_prov = _clv_by_provenance(rd)

        assert "kambi_inplay" in clv_prov, "kambi_inplay debe aparecer en clv_by_provenance"
        inplay = clv_prov["kambi_inplay"]
        assert inplay["excluded"] is True, "kambi_inplay debe tener excluded=True"
        assert inplay["n"] == 1, "kambi_inplay debe tener n=1"

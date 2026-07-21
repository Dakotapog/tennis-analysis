"""
Tests Nodo-125 — EvalGames Bridge + Dashboard X4 + Combo Time-Window
REGLA-T53: invocan función real del módulo, nunca hardcodean lógica.

Covers:
  D125-01 pick_snapshot includes match_id + hora
  D125-02 bridge derives diff_abs from cuota + output format
  D125-03 time-window grouping + combo gate
  D125-04 X4 panel graceful empty + enrichment
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── D125-02: diff_abs from cuota ──────────────────────────────────────────────

class TestD125_02_DiffAbs:
    """Test _diff_abs_from_cuota() — proxy de dominancia desde cuota favorito."""

    def test_cuota_1_10_yields_approx_0_82(self):
        from scripts.evaluar_games_bridge import _diff_abs_from_cuota
        result = _diff_abs_from_cuota(1.10)
        # p = 1/1.10 ≈ 0.909; diff = (0.909 - 0.5) * 2 ≈ 0.818
        assert abs(result - 0.818) < 0.005

    def test_cuota_1_20_yields_approx_0_667(self):
        from scripts.evaluar_games_bridge import _diff_abs_from_cuota
        result = _diff_abs_from_cuota(1.20)
        # p = 1/1.20 ≈ 0.833; diff = (0.833 - 0.5) * 2 ≈ 0.667
        assert abs(result - 0.667) < 0.005

    def test_cuota_1_28_yields_above_0_35_dominante(self):
        from scripts.evaluar_games_bridge import _diff_abs_from_cuota
        result = _diff_abs_from_cuota(1.28)
        # All cuota<1.30 must be in DOMINANTE zone (diff_abs >= 0.35)
        assert result >= 0.35

    def test_cuota_zero_returns_safe_default(self):
        from scripts.evaluar_games_bridge import _diff_abs_from_cuota
        result = _diff_abs_from_cuota(0)
        assert result == 0.5  # safe default


# ── D125-02: output format schema ─────────────────────────────────────────────

class TestD125_02_OutputFormat:
    """Test _save_report() produces schema consumible por betplay_combo_builder --evaluar."""

    def test_output_has_required_top_level_keys(self, tmp_path):
        from scripts.evaluar_games_bridge import _save_report
        import scripts.evaluar_games_bridge as bridge
        # Patch REPORTS_DIR to tmp_path so we don't write to real reports/
        orig = bridge.REPORTS_DIR
        bridge.REPORTS_DIR = tmp_path
        try:
            resultados = [
                {
                    'partido': 'Djokovic N. vs Murray A.',
                    'zona_diff': 'DOMINANTE',
                    'diff_abs': 0.82,
                    'predicted_sets': 2,
                    'games_range': '16-19',
                    'hora': '10:00',
                    'cuota_ml': 1.10,
                    'confidence': 0.84,
                    'señales_optimas': [],
                    'tiene_mercados': False,
                    '_source': 'evaluar_games',
                    '_sb_id': 'EVAL_20260721_001',
                }
            ]
            out_path = _save_report(resultados, '2026-07-21')
            data = json.loads(out_path.read_text())
            assert 'metadata' in data
            assert 'apostar' in data
            assert 'detalle_completo' in data
        finally:
            bridge.REPORTS_DIR = orig

    def test_metadata_includes_fuente_nodo125(self, tmp_path):
        from scripts.evaluar_games_bridge import _save_report
        import scripts.evaluar_games_bridge as bridge
        orig = bridge.REPORTS_DIR
        bridge.REPORTS_DIR = tmp_path
        try:
            out_path = _save_report([], '2026-07-21')
            data = json.loads(out_path.read_text())
            assert 'Nodo-125' in data['metadata']['fuente']
        finally:
            bridge.REPORTS_DIR = orig

    def test_apostar_filters_to_tiene_mercados_true(self, tmp_path):
        from scripts.evaluar_games_bridge import _save_report
        import scripts.evaluar_games_bridge as bridge
        orig = bridge.REPORTS_DIR
        bridge.REPORTS_DIR = tmp_path
        try:
            resultados = [
                {'partido': 'A vs B', 'tiene_mercados': True,
                 'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.85, 'linea': 21.5, 'confianza_señal': 'ALTA'}],
                 '_source': 'evaluar_games', '_sb_id': 'EVAL_001',
                 'zona_diff': 'DOMINANTE', 'diff_abs': 0.8, 'predicted_sets': 2,
                 'games_range': '16-19', 'hora': '09:00', 'cuota_ml': 1.10, 'confidence': 0.90},
                {'partido': 'C vs D', 'tiene_mercados': False,
                 'señales_optimas': [],
                 '_source': 'evaluar_games', '_sb_id': 'EVAL_002',
                 'zona_diff': 'DOMINANTE', 'diff_abs': 0.7, 'predicted_sets': 2,
                 'games_range': '16-19', 'hora': '11:00', 'cuota_ml': 1.20, 'confidence': 0.75},
            ]
            out_path = _save_report(resultados, '2026-07-21')
            data = json.loads(out_path.read_text())
            assert len(data['apostar']) == 1
            assert data['apostar'][0]['partido'] == 'A vs B'
            assert len(data['detalle_completo']) == 2
        finally:
            bridge.REPORTS_DIR = orig


# ── D125-03: time-window grouping ─────────────────────────────────────────────

class TestD125_03_TimeWindow:
    """Test _group_by_time_window() — greedy 90-min clustering."""

    def test_picks_within_90min_form_single_group(self):
        from betplay_combo_builder import _group_by_time_window
        signals = [
            {'partido': 'A vs B', 'hora': '09:00', 'cuota': 1.85},
            {'partido': 'C vs D', 'hora': '10:00', 'cuota': 1.80},
            {'partido': 'E vs F', 'hora': '10:30', 'cuota': 1.90},
        ]
        groups = _group_by_time_window(signals, window_min=90)
        # 09:00 to 10:30 = 90 min → all in one group
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_picks_outside_90min_split_into_separate_groups(self):
        from betplay_combo_builder import _group_by_time_window
        signals = [
            {'partido': 'A vs B', 'hora': '09:00', 'cuota': 1.85},
            {'partido': 'C vs D', 'hora': '11:00', 'cuota': 1.80},  # 120 min later → separate
        ]
        groups = _group_by_time_window(signals, window_min=90)
        assert len(groups) == 2

    def test_pick_without_hora_forms_own_group(self):
        from betplay_combo_builder import _group_by_time_window
        signals = [
            {'partido': 'A vs B', 'hora': '09:00', 'cuota': 1.85},
            {'partido': 'C vs D', 'hora': None, 'cuota': 1.80},   # no hora → own group
            {'partido': 'E vs F', 'hora': '09:30', 'cuota': 1.90},
        ]
        groups = _group_by_time_window(signals, window_min=90)
        # {A,E} together, {C} alone
        assert len(groups) == 2
        group_sizes = sorted(len(g) for g in groups)
        assert group_sizes == [1, 2]


# ── D125-03: combo gate ────────────────────────────────────────────────────────

class TestD125_03_ComboGate:
    """Test build_evaluar_games_combos() gate: cuota_combo >= 2.50."""

    def _make_signal_file(self, tmp_path, signals_apostar):
        """Write a fake evaluar_games_signal_*.json and return its path."""
        payload = {
            'metadata': {'fecha': '2026-07-21', 'generado': '2026-07-21T00:00:00',
                         'fuente': 'test', 'n_picks': len(signals_apostar), 'n_con_under': len(signals_apostar), 'nodo': 'test'},
            'apostar': signals_apostar,
            'detalle_completo': signals_apostar,
        }
        p = tmp_path / 'evaluar_games_signal_20260721_000000.json'
        p.write_text(json.dumps(payload))
        return p

    def test_combo_skips_single_pick_windows(self, tmp_path):
        from betplay_combo_builder import build_evaluar_games_combos
        signals = [
            {'partido': 'Solo A vs B', 'hora': '09:00', 'cuota_ml': 1.10, 'confidence': 0.84,
             'tiene_mercados': True,
             'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.85,
                                  'linea': 21.5, 'confianza_señal': 'ALTA', 'outcome_id': 9999}]},
        ]
        sig_file = self._make_signal_file(tmp_path, signals)
        combos, meta = build_evaluar_games_combos(stake_per_combo=1000, signal_file=sig_file)
        # Single-pick window: no combo possible (need ≥2 legs)
        assert combos == []

    def test_combo_cuota_gate_requires_min_2_50(self, tmp_path):
        from betplay_combo_builder import build_evaluar_games_combos
        # Two signals, product = 1.40 * 1.40 = 1.96 < 2.50 → no combo
        signals = [
            {'partido': 'A vs B', 'hora': '09:00', 'cuota_ml': 1.08, 'confidence': 0.80,
             'tiene_mercados': True,
             'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.40,
                                  'linea': 20.5, 'confianza_señal': 'MEDIA', 'outcome_id': 1001}]},
            {'partido': 'C vs D', 'hora': '09:30', 'cuota_ml': 1.10, 'confidence': 0.82,
             'tiene_mercados': True,
             'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.40,
                                  'linea': 21.5, 'confianza_señal': 'MEDIA', 'outcome_id': 1002}]},
        ]
        sig_file = self._make_signal_file(tmp_path, signals)
        combos, meta = build_evaluar_games_combos(stake_per_combo=1000, signal_file=sig_file)
        assert combos == []

    def test_combo_passes_gate_when_cuota_above_2_50(self, tmp_path):
        from betplay_combo_builder import build_evaluar_games_combos
        # Two signals, product = 1.85 * 1.90 = 3.515 > 2.50 → combo generated
        signals = [
            {'partido': 'Djokovic N. vs Murray A.', 'hora': '10:00', 'cuota_ml': 1.06, 'confidence': 0.90,
             'tiene_mercados': True,
             'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.85,
                                  'linea': 21.5, 'confianza_señal': 'ALTA', 'outcome_id': 2001}]},
            {'partido': 'Nadal R. vs Federer R.', 'hora': '10:30', 'cuota_ml': 1.08, 'confidence': 0.88,
             'tiene_mercados': True,
             'señales_optimas': [{'apostar': True, 'direccion': 'UNDER', 'cuota': 1.90,
                                  'linea': 20.5, 'confianza_señal': 'ALTA', 'outcome_id': 2002}]},
        ]
        sig_file = self._make_signal_file(tmp_path, signals)
        combos, meta = build_evaluar_games_combos(stake_per_combo=1000, signal_file=sig_file)
        assert len(combos) >= 1
        assert combos[0]['cuota_combo'] >= 2.50


# ── D125-04: X4 panel ─────────────────────────────────────────────────────────

class TestD125_04_X4Panel:
    """Test _build_x4_evaluar_games() — panel X4 del live_desk."""

    def test_returns_empty_dict_gracefully_when_no_sb_file(self, tmp_path):
        import live_desk
        orig = live_desk.BASE_DIR
        live_desk.BASE_DIR = tmp_path
        # Ensure shadow_book dir exists but is empty
        (tmp_path / 'reports' / 'shadow_book').mkdir(parents=True, exist_ok=True)
        try:
            result = live_desk._build_x4_evaluar_games('2099-01-01')
            assert result.get('disponible') is False
            assert result.get('picks') == []
            assert result.get('n') == 0
        finally:
            live_desk.BASE_DIR = orig

    def test_reads_evaluar_games_picks_from_shadow_book(self, tmp_path):
        import live_desk
        import shadow_book as sb
        # Write a minimal sb_FECHA.jsonl with one EVAL_ evaluar_games pick
        sb_dir = tmp_path / 'reports' / 'shadow_book'
        sb_dir.mkdir(parents=True, exist_ok=True)
        fecha = '2099-12-01'
        rec = {
            'EVAL_20991201_001': {
                'pick_snapshot': {
                    'pick_type': 'evaluar_games',
                    'partido': 'Test A vs Test B',
                    'cuota_favorito': 1.10,
                    'confidence': 0.84,
                    'hora': '10:00',
                    'favorito_predicho': 'Test A',
                },
                'status': 'pending',
                'cuota_trigger': 1.10,
                'ts': '2099-12-01T08:00:00',
            }
        }
        # JSONL format: one record per line, each with sb_id at top level
        sb_path = sb_dir / f'sb_{fecha}.jsonl'
        line = dict(rec['EVAL_20991201_001'])
        line['sb_id'] = 'EVAL_20991201_001'
        sb_path.write_text(json.dumps(line) + '\n')

        orig = live_desk.BASE_DIR
        live_desk.BASE_DIR = tmp_path
        try:
            result = live_desk._build_x4_evaluar_games(fecha)
            assert result['n'] == 1
            assert result['picks'][0]['partido'] == 'Test A vs Test B'
            assert abs(result['picks'][0]['conf'] - 0.84) < 0.01
        finally:
            live_desk.BASE_DIR = orig

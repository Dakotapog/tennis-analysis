"""
tests/test_nodo59.py — Nodo-59: Motor Agéntico — Odómetro de Tokens + Dream M2

REGLA-T53: ningún test hardcodea fórmulas. Todos invocan funciones del módulo real.

T59-01: parse_sessions con fixture JSONL → totales por modelo correctos
T59-02: sesión sin tag → agrupa en 'untagged', no crash
T59-03: costo estimado usa MODEL_COSTS (única fuente de verdad), no valores inline
T59-04: Dream M2 — secuencia 2 sesiones → NO propone; 3 sesiones → propone
"""
import json
import os
import pytest
from datetime import datetime, timezone

from token_odometer import (
    MODEL_COSTS,
    MODEL_RATIOS,
    _compute_cost,
    _extract_tag,
    _get_model_costs,
    detect_dream_sequences,
    parse_sessions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para crear fixtures JSONL
# ─────────────────────────────────────────────────────────────────────────────

def _make_assistant_line(model: str, inp: int, out: int,
                         cache_r: int = 0, cache_c: int = 0,
                         session_id: str = 's1',
                         ts: str = '2026-07-01T10:00:00.000Z') -> str:
    obj = {
        'type': 'assistant',
        'sessionId': session_id,
        'timestamp': ts,
        'message': {
            'model': model,
            'usage': {
                'input_tokens': inp,
                'output_tokens': out,
                'cache_read_input_tokens': cache_r,
                'cache_creation_input_tokens': cache_c,
            },
        },
    }
    return json.dumps(obj)


def _make_user_line(text: str, session_id: str = 's1',
                    ts: str = '2026-07-01T09:59:00.000Z') -> str:
    obj = {
        'type': 'user',
        'sessionId': session_id,
        'timestamp': ts,
        'message': {'content': text},
    }
    return json.dumps(obj)


def _write_jsonl(tmp_path, filename: str, lines: list) -> str:
    path = tmp_path / filename
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# T59-01 — parse_sessions con JSONL fixture → totales por modelo correctos
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSessionsT5901:
    """T59-01: parse_sessions con JSONL de fixture → totales por modelo correctos."""

    def test_t59_01_totals_by_model(self, tmp_path):
        """T59-01: dos sesiones con modelos distintos → totales acumulados correctos."""
        # Sesión 1: Sonnet, 1000 in, 200 out, 5000 cache_r
        _write_jsonl(tmp_path, 'session1.jsonl', [
            _make_user_line('# TAG: impl nodo-57', session_id='s1'),
            _make_assistant_line('claude-sonnet-4-6', 1000, 200, cache_r=5000, session_id='s1'),
            _make_assistant_line('claude-sonnet-4-6', 500, 100, cache_r=2000, session_id='s1'),
        ])
        # Sesión 2: Haiku, 2000 in, 400 out, 1000 cache_r
        _write_jsonl(tmp_path, 'session2.jsonl', [
            _make_user_line('# TAG: test nodo-57', session_id='s2',
                            ts='2026-07-01T11:00:00.000Z'),
            _make_assistant_line('claude-haiku-4-5-20251001', 2000, 400, cache_r=1000,
                                 session_id='s2', ts='2026-07-01T11:01:00.000Z'),
        ])

        data = parse_sessions(str(tmp_path))

        assert data['n_sessions'] == 2

        # Verificar acumulados Sonnet
        sonnet = data['models']['claude-sonnet-4-6']
        assert sonnet['input'] == 1500   # 1000 + 500
        assert sonnet['output'] == 300   # 200 + 100
        assert sonnet['cache_r'] == 7000  # 5000 + 2000

        # Verificar acumulados Haiku
        haiku = data['models']['claude-haiku-4-5-20251001']
        assert haiku['input'] == 2000
        assert haiku['output'] == 400
        assert haiku['cache_r'] == 1000

    def test_t59_01_total_cost_uses_model_costs(self, tmp_path):
        """T59-01b: el costo total es la suma de costos calculados con MODEL_COSTS."""
        _write_jsonl(tmp_path, 'sess.jsonl', [
            _make_user_line('implementar nodo-59', session_id='s1'),
            _make_assistant_line('claude-sonnet-4-6', 1_000_000, 0, session_id='s1'),
        ])
        data = parse_sessions(str(tmp_path))
        # 1M tokens input Sonnet = $3.00 según MODEL_COSTS
        expected = MODEL_COSTS['claude-sonnet']['input']  # $3.00
        assert abs(data['totals']['cost'] - expected) < 0.01

    def test_t59_01_empty_dir_no_crash(self, tmp_path):
        """T59-01c: directorio sin JSONL → n_sessions=0, costo=0, no crash."""
        data = parse_sessions(str(tmp_path))
        assert data['n_sessions'] == 0
        assert data['totals']['cost'] == 0.0

    def test_t59_01_corrupt_line_skipped(self, tmp_path):
        """T59-01d: línea corrupta en JSONL → skip silencioso, no crash."""
        _write_jsonl(tmp_path, 'corrupt.jsonl', [
            'esto no es json {{{',
            _make_assistant_line('claude-sonnet-4-6', 100, 50, session_id='s1'),
        ])
        data = parse_sessions(str(tmp_path))
        # La línea corrupta se salta; la válida se procesa
        assert data['n_sessions'] == 1
        assert data['models']['claude-sonnet-4-6']['input'] == 100


# ─────────────────────────────────────────────────────────────────────────────
# T59-02 — sesión sin tag → agrupa en 'untagged', no crash
# ─────────────────────────────────────────────────────────────────────────────

class TestUntaggedT5902:
    """T59-02: sesiones sin tag → 'untagged', sin crash."""

    def test_t59_02_no_tag_in_first_message(self, tmp_path):
        """T59-02a: primer mensaje sin '# TAG:' → tag='untagged'."""
        _write_jsonl(tmp_path, 'sess.jsonl', [
            _make_user_line('hola, cómo estás?', session_id='s1'),
            _make_assistant_line('claude-sonnet-4-6', 100, 50, session_id='s1'),
        ])
        data = parse_sessions(str(tmp_path))
        assert 'untagged' in data['tags']
        assert data['tags']['untagged']['input'] == 100

    def test_t59_02_extract_tag_explicit(self):
        """T59-02b: '# TAG: impl nodo-59' → tag='impl'."""
        assert _extract_tag('# TAG: impl nodo-59\nresto del mensaje') == 'impl'

    def test_t59_02_extract_tag_implicit(self):
        """T59-02c: primera línea contiene 'settle' → tag='settle'."""
        assert _extract_tag('settle partido de ayer') == 'settle'

    def test_t59_02_extract_tag_untagged(self):
        """T59-02d: texto sin palabras clave → 'untagged'."""
        assert _extract_tag('cuéntame sobre el partido de hoy') == 'untagged'

    def test_t59_02_empty_text_untagged(self):
        """T59-02e: texto vacío → 'untagged', no crash."""
        assert _extract_tag('') == 'untagged'
        assert _extract_tag(None) == 'untagged'

    def test_t59_02_mixed_sessions(self, tmp_path):
        """T59-02f: mezcla de sesiones con y sin tag → separadas correctamente."""
        _write_jsonl(tmp_path, 'tagged.jsonl', [
            _make_user_line('# TAG: test nodo-57', session_id='s1'),
            _make_assistant_line('claude-haiku-4-5-20251001', 500, 100, session_id='s1'),
        ])
        _write_jsonl(tmp_path, 'untagged.jsonl', [
            _make_user_line('sin categoría', session_id='s2',
                            ts='2026-07-01T12:00:00.000Z'),
            _make_assistant_line('claude-haiku-4-5-20251001', 300, 60, session_id='s2',
                                 ts='2026-07-01T12:01:00.000Z'),
        ])
        data = parse_sessions(str(tmp_path))
        assert 'test' in data['tags']
        assert 'untagged' in data['tags']
        assert data['tags']['test']['input'] == 500
        assert data['tags']['untagged']['input'] == 300


# ─────────────────────────────────────────────────────────────────────────────
# T59-03 — costo estimado usa MODEL_COSTS (única fuente de verdad)
# ─────────────────────────────────────────────────────────────────────────────

class TestCostsT5903:
    """T59-03: _compute_cost usa MODEL_COSTS, no valores inline."""

    def test_t59_03_costs_from_model_costs_dict(self):
        """T59-03a: _compute_cost('claude-haiku-4-5', usage) usa MODEL_COSTS['claude-haiku']."""
        usage = {'input_tokens': 1_000_000, 'output_tokens': 0,
                 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
        cost = _compute_cost(usage, 'claude-haiku-4-5')
        expected = MODEL_COSTS['claude-haiku']['input']  # $0.80 por MTok
        assert abs(cost - expected) < 0.001

    def test_t59_03_sonnet_cost_ratio_4x_haiku(self):
        """T59-03b: Sonnet cuesta ~4× más que Haiku en input (MODEL_RATIOS)."""
        usage = {'input_tokens': 1_000_000, 'output_tokens': 0,
                 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
        cost_haiku = _compute_cost(usage, 'claude-haiku-4-5')
        cost_sonnet = _compute_cost(usage, 'claude-sonnet-4-6')
        ratio = cost_sonnet / cost_haiku
        # Ratio esperado: MODEL_COSTS['claude-sonnet']['input'] / MODEL_COSTS['claude-haiku']['input']
        expected_ratio = MODEL_COSTS['claude-sonnet']['input'] / MODEL_COSTS['claude-haiku']['input']
        assert abs(ratio - expected_ratio) < 0.01

    def test_t59_03_model_costs_has_all_required_fields(self):
        """T59-03c: MODEL_COSTS tiene todas las claves requeridas para cada modelo."""
        required_fields = {'input', 'output', 'cache_read', 'cache_creation'}
        for prefix, costs in MODEL_COSTS.items():
            assert required_fields == set(costs.keys()), \
                f"MODEL_COSTS['{prefix}'] falta campos: {required_fields - set(costs.keys())}"

    def test_t59_03_model_ratios_haiku_is_1(self):
        """T59-03d: Haiku es el baseline (ratio=1). Sonnet=4, Opus=20."""
        assert MODEL_RATIOS['haiku'] == 1
        assert MODEL_RATIOS['sonnet'] == 4
        assert MODEL_RATIOS['opus'] == 20

    def test_t59_03_opus_cost_is_20x_haiku(self):
        """T59-03e: Opus cuesta ~20× más que Haiku en input."""
        usage = {'input_tokens': 1_000_000, 'output_tokens': 0,
                 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
        cost_haiku = _compute_cost(usage, 'claude-haiku-4-5')
        cost_opus = _compute_cost(usage, 'claude-opus-4-6')
        ratio = cost_opus / cost_haiku
        expected_ratio = MODEL_COSTS['claude-opus']['input'] / MODEL_COSTS['claude-haiku']['input']
        assert abs(ratio - expected_ratio) < 0.01

    def test_t59_03_unknown_model_falls_back_to_sonnet(self):
        """T59-03f: modelo desconocido → fallback a Sonnet (conservador)."""
        costs_unknown = _get_model_costs('claude-unknown-future-model')
        costs_sonnet = _get_model_costs('claude-sonnet-4-6')
        assert costs_unknown == costs_sonnet


# ─────────────────────────────────────────────────────────────────────────────
# T59-04 — Dream M2: regla n≥3
# ─────────────────────────────────────────────────────────────────────────────

class TestDreamT5904:
    """T59-04: Dream M2 — secuencia en 2 sesiones → NO propone; 3 → propone."""

    def _make_session_with_commands(self, tmp_path, filename, commands, session_id):
        """Helper: crea JSONL con comandos de usuario dados."""
        lines = []
        for i, cmd in enumerate(commands):
            ts = f'2026-07-01T{10+i:02d}:00:00.000Z'
            lines.append(_make_user_line(cmd, session_id=session_id, ts=ts))
            lines.append(_make_assistant_line('claude-sonnet-4-6', 10, 5,
                                              session_id=session_id,
                                              ts=f'2026-07-01T{10+i:02d}:01:00.000Z'))
        _write_jsonl(tmp_path, filename, lines)

    def test_t59_04_sequence_in_2_sessions_not_proposed(self, tmp_path):
        """T59-04a: secuencia que aparece en 2 sesiones → NO se propone (n<3)."""
        repeated_seq = [
            'python3 extraer_partidos_api.py',
            'python3 extraer_historh2h.py --api-mode',
            'python3 edge_calculator.py',
        ]
        # Secuencia aparece solo en 2 sesiones
        self._make_session_with_commands(tmp_path, 's1.jsonl', repeated_seq, 's1')
        self._make_session_with_commands(tmp_path, 's2.jsonl', repeated_seq, 's2')

        candidates = detect_dream_sequences(str(tmp_path), min_sessions=3, min_seq_len=3)
        assert len(candidates) == 0, \
            f"No debería proponer skill con n=2 sesiones: {candidates}"

    def test_t59_04_sequence_in_3_sessions_proposed(self, tmp_path):
        """T59-04b: secuencia que aparece en 3 sesiones → SÍ se propone."""
        repeated_seq = [
            'python3 extraer_partidos_api.py',
            'python3 extraer_historh2h.py --api-mode',
            'python3 edge_calculator.py',
        ]
        self._make_session_with_commands(tmp_path, 's1.jsonl', repeated_seq, 's1')
        self._make_session_with_commands(tmp_path, 's2.jsonl', repeated_seq, 's2')
        self._make_session_with_commands(tmp_path, 's3.jsonl', repeated_seq, 's3')

        candidates = detect_dream_sequences(str(tmp_path), min_sessions=3, min_seq_len=3)
        assert len(candidates) >= 1
        # Verificar que la secuencia correcta está en el candidato
        seqs = [c['sequence'] for c in candidates]
        found = any(
            'python3 extraer_partidos_api.py' in s[0]
            for s in seqs
        )
        assert found, f"Secuencia esperada no encontrada en candidatos: {seqs}"

    def test_t59_04_min_sessions_configurable(self, tmp_path):
        """T59-04c: min_sessions=2 → acepta secuencias con 2 sesiones."""
        repeated_seq = [
            'python3 shadow_book.py --settle 2026-07-01',
            'python3 shadow_book.py --report',
            'python3 pipeline_tracker.py --section shadow',
        ]
        self._make_session_with_commands(tmp_path, 's1.jsonl', repeated_seq, 's1')
        self._make_session_with_commands(tmp_path, 's2.jsonl', repeated_seq, 's2')

        # Con min_sessions=2 → debe proponer
        candidates_2 = detect_dream_sequences(str(tmp_path), min_sessions=2, min_seq_len=3)
        assert len(candidates_2) >= 1

        # Con min_sessions=3 → NO debe proponer (solo hay 2 sesiones)
        candidates_3 = detect_dream_sequences(str(tmp_path), min_sessions=3, min_seq_len=3)
        assert len(candidates_3) == 0

    def test_t59_04_unique_sequences_per_session(self, tmp_path):
        """T59-04d: secuencia distinta en cada sesión → 0 candidatos."""
        self._make_session_with_commands(tmp_path, 's1.jsonl', [
            'comando único sesión 1a', 'comando único sesión 1b', 'comando único sesión 1c',
        ], 's1')
        self._make_session_with_commands(tmp_path, 's2.jsonl', [
            'comando único sesión 2a', 'comando único sesión 2b', 'comando único sesión 2c',
        ], 's2')
        self._make_session_with_commands(tmp_path, 's3.jsonl', [
            'comando único sesión 3a', 'comando único sesión 3b', 'comando único sesión 3c',
        ], 's3')

        candidates = detect_dream_sequences(str(tmp_path), min_sessions=3, min_seq_len=3)
        assert len(candidates) == 0

    def test_t59_04_empty_sessions_no_crash(self, tmp_path):
        """T59-04e: sesiones sin mensajes de usuario → no crash, 0 candidatos."""
        # JSONL solo con mensajes de asistente (sin user)
        lines = [_make_assistant_line('claude-sonnet-4-6', 100, 50, session_id='s1')]
        _write_jsonl(tmp_path, 's1.jsonl', lines)
        candidates = detect_dream_sequences(str(tmp_path), min_sessions=3, min_seq_len=3)
        assert candidates == []

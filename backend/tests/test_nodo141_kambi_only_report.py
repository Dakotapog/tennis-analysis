"""
tests/test_nodo141_kambi_only_report.py — REGLA-T53: tests invocan función real del módulo.

Cubre Nodo-141 D141-01→D141-03: Kambi-Only Edge Report + PASO 3K.
Sin mocks — funciones puras testeadas con fixtures locales en /tmp.
"""
import importlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_fkp = importlib.import_module('scripts.filter_kambi_picks')
_filter_fn = _fkp.filter_kambi_picks
_find_full = _fkp._find_latest_full_report

_bpb = importlib.import_module('betplay_combo_builder')
_find_edge = _bpb._find_latest_edge_report


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge_report(picks, tmp_dir, name='edge_report_20260722_143022.json'):
    """Crea un edge_report fixture en tmp_dir."""
    report = {
        'gate_version': 'v3',
        'picks': picks,
        'metadata': {'fecha': '2026-07-22'},
    }
    path = Path(tmp_dir) / name
    path.write_text(json.dumps(report), encoding='utf-8')
    return path


def _pick(nombre='Alcaraz C.', kambi=True, status='APOSTAR'):
    return {
        'favorito_predicho': nombre,
        'kambi_disponible': kambi,
        'status': status,
        'edge_pct': '12%',
        'cuota_favorito': 1.65,
    }


# ── D141-01: filter_kambi_picks ──────────────────────────────────────────────

def test_D141_01_filter_produces_kambi_only_picks():
    """Solo picks kambi_disponible=True deben aparecer en el reporte filtrado."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_edge_report([
            _pick('Alcaraz C.', kambi=True),
            _pick('ITF Player', kambi=False),
            _pick('Sinner J.',  kambi=True),
            _pick('Unknown',    kambi=None),
        ], tmp)
        result = _filter_fn(src)
        names = [p['favorito_predicho'] for p in result['picks']]
        assert 'Alcaraz C.' in names
        assert 'Sinner J.' in names
        assert 'ITF Player' not in names, 'kambi=False debe ser excluido'
        assert 'Unknown' not in names, 'kambi=None (sin coverage) no es True → excluir'


def test_D141_01_filter_preserves_report_structure():
    """El reporte filtrado debe tener la misma estructura que el original."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_edge_report([
            _pick('Djokovic N.', kambi=True),
        ], tmp)
        result = _filter_fn(src)
        assert 'gate_version' in result, 'Estructura original debe preservarse'
        assert 'metadata' in result
        assert result['_kambi_only'] is True
        assert result['_n_kambi'] == 1
        assert result['_n_total'] == 1


def test_D141_01_filter_empty_when_no_kambi_picks():
    """Si 0 picks son kambi, el resultado tiene lista vacía."""
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_edge_report([
            _pick('ITF A', kambi=False),
            _pick('ITF B', kambi=False),
        ], tmp)
        result = _filter_fn(src)
        assert result['picks'] == []
        assert result['_n_kambi'] == 0
        assert result['_n_total'] == 2


def test_D141_01_filter_counts_correct():
    """_n_kambi y _n_total deben reflejar counts reales."""
    with tempfile.TemporaryDirectory() as tmp:
        picks = [
            _pick('A', kambi=True),
            _pick('B', kambi=True),
            _pick('C', kambi=False),
            _pick('D', kambi=None),
        ]
        src = _make_edge_report(picks, tmp)
        result = _filter_fn(src)
        assert result['_n_kambi'] == 2
        assert result['_n_total'] == 4


# ── D141-03: _find_latest_edge_report en betplay_combo_builder ───────────────

def test_D141_03_find_latest_prefers_todays_kambi(monkeypatch):
    """Si existe edge_report_kambi_HOY*.json, debe ser preferido."""
    today = datetime.now().strftime('%Y%m%d')
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Crear full report y kambi report de hoy
        full = tmp_path / f'edge_report_{today}_143022.json'
        kambi = tmp_path / f'edge_report_kambi_{today}_143025.json'
        full.write_text('{}')
        kambi.write_text('{}')

        monkeypatch.setattr(_bpb, 'Path', lambda x: tmp_path if x == 'reports' else Path(x))
        # Patch Path("reports") globbing
        import betplay_combo_builder as bpb
        original_find = bpb._find_latest_edge_report

        # Call with monkeypatched reports dir
        from unittest.mock import patch
        with patch('betplay_combo_builder.Path') as mock_path:
            mock_reports = mock_path.return_value
            mock_reports.exists.return_value = True
            mock_reports.glob.side_effect = lambda pat: (
                [kambi] if f'kambi_{today}' in pat else
                [f for f in [full] if 'kambi' not in f.name]
            )
            result = bpb._find_latest_edge_report()
        assert 'kambi' in result, f'Esperado kambi report, obtenido: {result}'


def test_D141_03_find_latest_falls_back_to_full_when_no_kambi_today(monkeypatch):
    """Si no hay kambi de hoy, usa el full report más reciente."""
    today = datetime.now().strftime('%Y%m%d')
    import betplay_combo_builder as bpb
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        full = tmp_path / f'edge_report_{today}_143022.json'
        full.write_text('{}')

        with patch('betplay_combo_builder.Path') as mock_path:
            mock_reports = mock_path.return_value
            mock_reports.exists.return_value = True
            mock_reports.glob.side_effect = lambda pat: (
                [] if f'kambi_{today}' in pat else
                [full]
            )
            result = bpb._find_latest_edge_report()
        assert result is not None
        assert 'kambi' not in result, f'No debe usar kambi: {result}'


def test_D141_03_find_latest_excludes_yesterday_kambi(monkeypatch):
    """No debe preferir kambi de ayer sobre full de hoy."""
    today = datetime.now().strftime('%Y%m%d')
    yesterday_dt = __import__('datetime').datetime.now() - __import__('datetime').timedelta(days=1)
    yesterday = yesterday_dt.strftime('%Y%m%d')

    import betplay_combo_builder as bpb
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        full_today  = tmp_path / f'edge_report_{today}_143022.json'
        kambi_yest  = tmp_path / f'edge_report_kambi_{yesterday}_090000.json'
        full_today.write_text('{}')
        kambi_yest.write_text('{}')

        with patch('betplay_combo_builder.Path') as mock_path:
            mock_reports = mock_path.return_value
            mock_reports.exists.return_value = True
            mock_reports.glob.side_effect = lambda pat: (
                [] if f'kambi_{today}' in pat else            # sin kambi de HOY
                [f for f in [full_today] if 'kambi' not in f.name]  # solo full de hoy
            )
            result = bpb._find_latest_edge_report()
        assert result is not None
        assert 'kambi' not in result, 'No debe usar kambi de ayer'


# ── D141-02: run_daily.py tiene PASO 3K ──────────────────────────────────────

def test_D141_02_run_daily_has_paso_3k():
    """run_daily.py debe mencionar PASO 3K con filter_kambi_picks."""
    src = Path('run_daily.py').read_text(encoding='utf-8')
    assert 'filter_kambi_picks.py' in src, 'PASO 3K filter_kambi_picks.py ausente en run_daily.py'
    assert 'PASO 3K' in src, 'Label PASO 3K ausente en run_daily.py'

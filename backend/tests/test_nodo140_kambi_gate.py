"""
tests/test_nodo140_kambi_gate.py — REGLA-T53: tests invocan función real del módulo.

Cubre Nodo-140 D140-01→D140-04: Kambi Gate + Coverage Fresca.
Sin mocks HTTP — funciones puras testeadas con fixtures locales.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
_bpb = importlib.import_module('betplay_combo_builder')
_ccb = importlib.import_module('combo_confianza_builder')

_filter_kambi_available = _bpb._filter_kambi_available


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _pick(nombre='Alcaraz C.', kambi_disponible=True, **kw):
    base = {
        'favorito_predicho': nombre,
        'kambi_disponible': kambi_disponible,
        'tier': 'atp500',
        'edge_pct': '12%',
        'cuota_favorito': 1.65,
        'status': 'APOSTAR',
        'markov_favorito': 'HOT',
        'markov_conf_fav': 0.65,
        'markov_rival': 'COLD',
        'markov_conf_rival': 0.70,
        'markov_wr_rec_fav': 0.72,
        'markov_wr_rec_rival': 0.31,
    }
    base.update(kw)
    return base


# ── D140-01: PASO 1c en run_daily.py ────────────────────────────────────────

def test_D140_01_run_daily_has_paso_1c():
    """run_daily.py debe mencionar fetch_kambi_coverage ANTES de edge_calculator."""
    src = Path('run_daily.py').read_text(encoding='utf-8')
    idx_coverage = src.find('fetch_kambi_coverage.py')
    idx_edge     = src.find('edge_calculator.py')
    assert idx_coverage != -1, 'PASO 1c fetch_kambi_coverage.py ausente en run_daily.py'
    assert idx_edge     != -1, 'edge_calculator.py ausente en run_daily.py'
    assert idx_coverage < idx_edge, (
        'fetch_kambi_coverage.py debe aparecer ANTES de edge_calculator.py en run_daily.py'
    )


def test_D140_01_paso_1c_label_correcto():
    """PASO 1c debe tener label identificable para logs."""
    src = Path('run_daily.py').read_text(encoding='utf-8')
    assert 'PASO 1c' in src, 'Label PASO 1c ausente en run_daily.py'
    assert 'Kambi Coverage' in src or 'kambi_coverage' in src.lower(), (
        'Descripción Kambi Coverage ausente en PASO 1c'
    )


# ── D140-02/03: _filter_kambi_available ─────────────────────────────────────

def test_D140_02_filter_excluye_disponible_false():
    """kambi_disponible=False debe excluir el pick."""
    picks = [
        _pick('Alcaraz C.',  kambi_disponible=True),
        _pick('ITF Player',  kambi_disponible=False),
        _pick('Sinner J.',   kambi_disponible=True),
    ]
    result = _filter_kambi_available(picks, 'TEST')
    names = [p['favorito_predicho'] for p in result]
    assert 'ITF Player' not in names, 'kambi_disponible=False debe ser excluido'
    assert 'Alcaraz C.' in names
    assert 'Sinner J.' in names


def test_D140_02_filter_permite_none():
    """kambi_disponible=None (sin coverage) = pass-through — no bloquear."""
    picks = [
        _pick('Djokovic N.', kambi_disponible=None),
        _pick('Nadal R.',    kambi_disponible=None),
    ]
    result = _filter_kambi_available(picks, 'TEST')
    assert len(result) == 2, 'None debe ser pass-through — no excluir'


def test_D140_02_filter_permite_true():
    """kambi_disponible=True = pick apostable, debe pasar."""
    picks = [_pick('Medvedev D.', kambi_disponible=True)]
    result = _filter_kambi_available(picks, 'TEST')
    assert len(result) == 1


def test_D140_02_filter_lista_vacia():
    """Lista vacía no genera error."""
    result = _filter_kambi_available([], 'TEST')
    assert result == []


def test_D140_02_filter_todos_false():
    """Todos False → lista vacía (sin combos ITF)."""
    picks = [
        _pick('ITF A', kambi_disponible=False),
        _pick('ITF B', kambi_disponible=False),
        _pick('ITF C', kambi_disponible=False),
    ]
    result = _filter_kambi_available(picks, 'TEST')
    assert result == [], 'Todos False debe retornar lista vacía'


# ── D140-03: betplay_combo_builder tiene _filter_kambi_available ────────────

def test_D140_03_betplay_builder_tiene_helper():
    """_filter_kambi_available debe existir en betplay_combo_builder."""
    assert hasattr(_bpb, '_filter_kambi_available'), (
        '_filter_kambi_available ausente en betplay_combo_builder.py'
    )


# ── D140-04: combo_confianza_builder importa coverage gate ──────────────────

def test_D140_04_confianza_builder_usa_kambi_gate():
    """_extract_and_categorize debe contener referencia a fetch_kambi_coverage."""
    import inspect
    src = inspect.getsource(_ccb._extract_and_categorize)
    assert 'fetch_kambi_coverage' in src or '_kc_cov' in src, (
        'D140-04: _extract_and_categorize no tiene gate fetch_kambi_coverage'
    )
    assert '_kc_available' in src or 'is_player_available' in src, (
        'D140-04: falta llamada a is_player_available en _extract_and_categorize'
    )

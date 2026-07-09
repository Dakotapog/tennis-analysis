"""
tests/test_nodo62.py — Nodo-62 Signal Bridge

T62-01: _compute_alpha_score con triple=0.661 -> score incluye +15 (TRIPLE_HIGH)
T62-02: _compute_alpha_score con markov=HOT -> score incluye +10
T62-03: _compute_alpha_score con markov=COLD -> score incluye -15
T62-04: _compute_alpha_score con gcs_bonus=True -> score incluye +12
T62-05: _compute_alpha_score con edge_pct=23.6% (>=15%) -> score incluye +10 (EDGE_HIGH)
T62-06: _compute_alpha_score con phantom data -> score incluye -25
T62-07: combo_priority = confianza + alpha_score en pick enriquecido
T62-08: constantes de gate Cat-C1 alpha correctas (edge>=5%, triple>=0.2)
T62-09: _load_edge_report_index importable y retorna dict
T62-10: pick con markov=COLD tiene combo_priority < confianza

REGLA-T53: todos los tests invocan funciones reales del modulo — ningun hardcode de formula.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combo_confianza_builder import (
    _compute_alpha_score,
    _load_edge_report_index,
    _ALPHA_TRIPLE_HIGH,
    _ALPHA_TRIPLE_MED,
    _ALPHA_MARKOV_HOT,
    _ALPHA_MARKOV_COLD,
    _ALPHA_GCS,
    _ALPHA_EDGE_HIGH,
    _ALPHA_EDGE_MED,
    _ALPHA_PHANTOM,
    _ALPHA_C1_EDGE_MIN,
    _ALPHA_C1_TRIPLE_MIN,
)


def test_t62_01_triple_high():
    """T62-01: triple_alignment>=0.5 -> +15 en alpha_score"""
    edge_data = {'triple_alignment': 0.661, 'markov_favorito': 'NEUTRAL'}
    score, senales = _compute_alpha_score(edge_data)
    assert score >= _ALPHA_TRIPLE_HIGH, (
        f"Esperado >={_ALPHA_TRIPLE_HIGH}, got {score}"
    )
    assert any('triple' in s for s in senales), (
        f"Senal triple no encontrada en: {senales}"
    )


def test_t62_02_markov_hot():
    """T62-02: markov=HOT -> +10 en alpha_score"""
    edge_data = {'triple_alignment': 0.0, 'markov_favorito': 'HOT'}
    score, senales = _compute_alpha_score(edge_data)
    assert score >= _ALPHA_MARKOV_HOT, (
        f"Esperado >={_ALPHA_MARKOV_HOT}, got {score}"
    )
    assert any('HOT' in s for s in senales), (
        f"Senal HOT no encontrada en: {senales}"
    )


def test_t62_03_markov_cold():
    """T62-03: markov=COLD -> -15 en alpha_score"""
    edge_data = {'triple_alignment': 0.0, 'markov_favorito': 'COLD'}
    score, senales = _compute_alpha_score(edge_data)
    assert score <= _ALPHA_MARKOV_COLD, (
        f"Esperado <={_ALPHA_MARKOV_COLD}, got {score}"
    )
    assert any('COLD' in s for s in senales), (
        f"Senal COLD no encontrada en: {senales}"
    )


def test_t62_04_gcs_bonus():
    """T62-04: gcs_bonus=True -> +12 en alpha_score"""
    edge_data = {'gcs_bonus': True, 'triple_alignment': 0.0, 'markov_favorito': 'NEUTRAL'}
    score, senales = _compute_alpha_score(edge_data)
    assert score >= _ALPHA_GCS, (
        f"Esperado >={_ALPHA_GCS}, got {score}"
    )
    assert any('gcs' in s for s in senales), (
        f"Senal gcs no encontrada en: {senales}"
    )


def test_t62_05_edge_high():
    """T62-05: edge_pct=23.6% (>=15%) -> +10 en alpha_score (EDGE_HIGH)"""
    edge_data = {'edge_pct': '23.6%', 'triple_alignment': 0.0, 'markov_favorito': 'NEUTRAL'}
    score, senales = _compute_alpha_score(edge_data)
    assert score >= _ALPHA_EDGE_HIGH, (
        f"Esperado >={_ALPHA_EDGE_HIGH}, got {score}"
    )
    assert any('edge' in s for s in senales), (
        f"Senal edge no encontrada en: {senales}"
    )


def test_t62_06_phantom_data():
    """T62-06: history_provenance=EMPTY -> -25 en alpha_score"""
    edge_data = {
        'triple_alignment': 0.0,
        'markov_favorito': 'NEUTRAL',
        'history_provenance': {'p1': 'EMPTY', 'p2': 'ninja_api'},
    }
    score, senales = _compute_alpha_score(edge_data)
    assert score <= _ALPHA_PHANTOM, (
        f"Esperado <={_ALPHA_PHANTOM}, got {score}"
    )
    assert any('phantom' in s for s in senales), (
        f"Senal phantom no encontrada en: {senales}"
    )


def test_t62_07_combo_priority_composition():
    """T62-07: combo_priority = confianza + alpha_score (positivo cuando senales fuertes)"""
    # Simula Hoeyeraal: conf=55.8, triple=0.661, HOT, surface=1.0
    edge_data = {
        'triple_alignment': 0.661,
        'markov_favorito': 'HOT',
        'surface_signal': 1.0,
        'bbi': 0.461,
    }
    confianza = 55.8
    alpha, _ = _compute_alpha_score(edge_data)
    combo_priority = confianza + alpha
    # triple_high(+15) + hot(+10) + surface_high(+8) = +33 -> priority ~88.8
    assert combo_priority > confianza, (
        "combo_priority debe ser mayor que confianza sola cuando hay senales alpha"
    )
    assert combo_priority >= confianza + 30, (
        f"Con triple+HOT+surface esperado >=+30, got alpha={alpha}"
    )


def test_t62_08_cat_c1_alpha_gate_constants():
    """T62-08: constantes de gate Cat-C1 alpha correctas (edge>=5%, triple>=0.2)"""
    assert _ALPHA_C1_EDGE_MIN == 5.0, (
        f"_ALPHA_C1_EDGE_MIN debe ser 5.0, got {_ALPHA_C1_EDGE_MIN}"
    )
    assert _ALPHA_C1_TRIPLE_MIN == 0.2, (
        f"_ALPHA_C1_TRIPLE_MIN debe ser 0.2, got {_ALPHA_C1_TRIPLE_MIN}"
    )


def test_t62_09_load_edge_report_index_importable():
    """T62-09: _load_edge_report_index importable y retorna dict (puede ser vacio)"""
    result = _load_edge_report_index()
    assert isinstance(result, dict), (
        f"_load_edge_report_index debe retornar dict, got {type(result)}"
    )


def test_t62_10_cold_reduces_priority():
    """T62-10: markov=COLD hace combo_priority < confianza"""
    edge_data = {'triple_alignment': 0.0, 'markov_favorito': 'COLD'}
    confianza = 65.0
    alpha, _ = _compute_alpha_score(edge_data)
    combo_priority = confianza + alpha
    assert combo_priority < confianza, (
        f"COLD debe reducir combo_priority por debajo de confianza. "
        f"confianza={confianza}, combo_priority={combo_priority}, alpha={alpha}"
    )

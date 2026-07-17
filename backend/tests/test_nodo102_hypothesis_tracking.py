"""
Tests Nodo-102 — Hypothesis Tracking: H98-01 (score_directo>=3) + H100-01 (Triple Convergencia).
REGLA-T53: invocan funciones reales del módulo.

D102-01: segmento score_directo>=3 visible en report() cuando hay picks settled
D102-02: H98-01 y H100-01 en bloque HIPÓTESIS de report()
D102-03: H100-01 registrada en preregistered_hypotheses.json
"""
import json
import os
import sys
import tempfile
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import shadow_book as sb


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_pick(score_directo: int, partido: str = 'A vs B', pick_type: str = None) -> dict:
    p = {
        'partido':           partido,
        'favorito_predicho': 'A',
        'cuota_favorito':    2.20,
        'p_modelo':          0.58,
        'edge':              0.04,
        'tier':              'challenger',
        'score_directo':     score_directo,
    }
    if pick_type:
        p['pick_type'] = pick_type
        p['cuota_trigger'] = 2.20
    return p


def _write_settled(tmpdir: str, fecha: str, picks: list, ganador: str = 'A') -> None:
    """Log, cierra y settle picks en tmpdir."""
    sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
    os.makedirs(sb.SHADOW_DIR, exist_ok=True)

    for i, pick in enumerate(picks):
        if pick.get('pick_type') == 'live':
            sb.log_live_pick(pick, cuota_trigger=pick.get('cuota_trigger', 2.20), fecha=fecha)
        else:
            sb.log_picks({'apostar': [pick], 'watchlist': [], 'no_data': []},
                         {'fecha': fecha})

    # Settle todos con WON
    path = os.path.join(sb.SHADOW_DIR, f'sb_{fecha}.jsonl')
    records = sb._load_jsonl(path)
    for sb_id, rec in records.items():
        if rec.get('_type') == 'session_meta' or 'resolucion' in rec:
            continue
        snap = rec.get('pick_snapshot', {})
        p1, p2 = sb._pick_partido_parts(snap)
        mk = rec['match_key']
        sb.settle(fecha, resultados_map={
            mk: {'ganador': ganador, 'cuota_cierre': 2.10, 'provenance': 'manual', 'void': False}
        })


# ─── Test 1: score_directo>=3 aparece en segmento ────────────────────────────

def test_score_directo_3_aparece_en_segmento():
    """D102-01: pick con score_directo=3 aparece en segmento; score=2 no aparece."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = sb.SHADOW_DIR
        try:
            fecha = '2026-07-14'
            picks = [
                _make_pick(score_directo=3, partido='Djokovic vs Alcaraz'),
                _make_pick(score_directo=2, partido='Ruud vs Zverev'),
            ]
            _write_settled(tmpdir, fecha, picks)

            txt = sb.report(desde=fecha, hasta=fecha)
            assert 'score_directo>=3' in txt
            # El segmento debe mostrar n=1 (solo el de score=3)
            # Buscar sección del segmento
            lines = txt.split('\n')
            score3_line = [l for l in lines if 'score_directo>=3' in l and 'n=' in l]
            assert len(score3_line) == 1
            assert 'n=1' in score3_line[0]
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 2: H98-01 visible en bloque HIPÓTESIS ──────────────────────────────

def test_h9801_en_hipotesis_continuar():
    """D102-02: H98-01 aparece en sección HIPÓTESIS con estado CONTINUAR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = sb.SHADOW_DIR
        try:
            fecha = '2026-07-14'
            # 1 pick con score=3 → H98-01 CONTINUAR (n=1/30)
            picks = [_make_pick(score_directo=3, partido='Sinner vs Medvedev')]
            _write_settled(tmpdir, fecha, picks)

            txt = sb.report(desde=fecha, hasta=fecha)
            assert 'H98-01' in txt
            assert 'score_directo>=3 supera breakeven' in txt
            assert 'CONTINUAR' in txt
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 3: H100-01 visible en bloque HIPÓTESIS ─────────────────────────────

def test_h10001_en_hipotesis_continuar():
    """D102-02: H100-01 aparece en sección HIPÓTESIS con estado CONTINUAR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = sb.SHADOW_DIR
        try:
            fecha = '2026-07-14'
            # 1 live pick → H100-01 CONTINUAR (n=1/20)
            picks = [_make_pick(score_directo=2, partido='Boogaard vs Onclin', pick_type='live')]
            _write_settled(tmpdir, fecha, picks)

            txt = sb.report(desde=fecha, hasta=fecha)
            assert 'H100-01' in txt
            assert 'BREAK_CONFIRMADO picks superan breakeven live' in txt
            assert 'CONTINUAR' in txt
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 4: H100-01 registrada en el JSON ───────────────────────────────────

def test_h10001_registrada_en_json():
    """D102-03: H100-01 presente en preregistered_hypotheses.json con campos completos."""
    json_path = os.path.join(
        os.path.dirname(__file__), '..', 'validation', 'preregistered_hypotheses.json'
    )
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)

    hs = d['hypotheses']
    assert 'H100-01' in hs, "H100-01 debe estar en preregistered_hypotheses.json"

    h = hs['H100-01']
    assert h.get('n_stop') == 20
    assert h.get('estado') == 'ACUMULANDO'
    assert 'umbrales_congelados' in h
    assert h['umbrales_congelados'].get('break_state') == 'BREAK_CONFIRMADO'
    assert h['umbrales_congelados'].get('pick_type') == 'live'

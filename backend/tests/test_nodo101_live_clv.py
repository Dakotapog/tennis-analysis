"""
Tests Nodo-101 — Shadow Book Live CLV: Pick-Type Live Tracking (D99-02).
REGLA-T53: invocan funciones reales del módulo, nunca hardcodean la lógica.

D101-01: pick_type='live' en pick_snapshot del registro JSONL
D101-02: cuota_trigger = cuota en momento del break (≠ cuota_pre)
D101-03: settle() calcula pnl y CLV_live usando cuota_trigger
D101-04: report() muestra sección LIVE PICKS cuando hay registros settled live
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import shadow_book as sb


# ─── Pick mínimo válido para log_live_pick ────────────────────────────────────

def _pick_live(partido='Boogaard vs Onclin', favorito='Boogaard'):
    return {
        'partido':           partido,
        'favorito_predicho': favorito,
        'cuota_favorito':    3.55,      # cuota_pre
        'p_modelo':          0.62,
        'edge':              0.087,
        'break_state':       'BREAK_CONFIRMADO',
        'drift_pct':         18.3,
        'pick_type':         'live',
    }


# ─── Test 1: log_live_pick escribe JSONL con pick_type='live' ────────────────

def test_log_live_escribe_jsonl():
    """D101-01: log_live_pick crea entrada pick_type=live en el JSONL del día."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            sb_id = sb.log_live_pick(_pick_live(), cuota_trigger=2.90, fecha='2026-07-14')
            assert sb_id is not None
            assert sb_id.startswith('LIVE_')

            # Leer JSONL y verificar que el registro existe
            path = os.path.join(sb.SHADOW_DIR, 'sb_2026-07-14.jsonl')
            assert os.path.exists(path)
            records = sb._load_jsonl(path)
            assert sb_id in records
            rec = records[sb_id]
            snap = rec.get('pick_snapshot', {})
            assert snap.get('pick_type') == 'live'
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 2: campos obligatorios presentes en el registro ────────────────────

def test_log_live_campos_completos():
    """D101-02: cuota_trigger, trigger_ts, partido, favorito_predicho presentes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            sb_id = sb.log_live_pick(_pick_live(), cuota_trigger=2.90, fecha='2026-07-14')
            path = os.path.join(sb.SHADOW_DIR, 'sb_2026-07-14.jsonl')
            records = sb._load_jsonl(path)
            snap = records[sb_id]['pick_snapshot']

            assert snap['cuota_trigger'] == 2.90        # D101-02: cuota live al trigger
            assert snap['pick_type'] == 'live'          # D101-01: campo live
            assert 'trigger_ts' in snap                 # timestamp del break
            assert snap['partido'] == 'Boogaard vs Onclin'
            assert snap['favorito_predicho'] == 'Boogaard'
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 3: settle usa cuota_trigger para CLV_live ──────────────────────────

def test_settle_live_won_usa_cuota_trigger():
    """D101-03: settle() usa cuota_trigger como cuota_tomada para CLV_live."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            fecha = '2026-07-14'
            sb_id = sb.log_live_pick(_pick_live(), cuota_trigger=2.90, fecha=fecha)

            # Settle manualmente con resultados_map
            # match_key = boogaard_onclin (nombres del partido)
            path = os.path.join(sb.SHADOW_DIR, f'sb_{fecha}.jsonl')
            records = sb._load_jsonl(path)
            rec = records[sb_id]
            mk = rec['match_key']

            resultados_map = {
                mk: {
                    'ganador':        'Boogaard',
                    'cuota_cierre':   3.10,
                    'provenance':     'manual',
                    'void':           False,
                }
            }

            n = sb.settle(fecha, resultados_map=resultados_map)
            assert n == 1

            records2 = sb._load_jsonl(path)
            resol = records2[sb_id]['resolucion']
            assert resol['resultado'] == 'WON'
            # CLV_live = (cuota_trigger / cuota_cierre - 1) × 100 = (2.90/3.10-1)*100 ≈ -6.45
            # pnl_flat_1u = cuota_trigger - 1 = 1.90
            assert abs(resol['pnl_flat_1u'] - 1.90) < 0.01
            assert resol['clv_pct'] is not None          # CLV_live calculado
        finally:
            sb.SHADOW_DIR = original_dir


# ─── Test 4: report() muestra sección LIVE PICKS cuando hay settled live ─────

def test_report_muestra_seccion_live():
    """D101-04: --report incluye línea LIVE PICKS H100-01 cuando hay picks live settled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            fecha = '2026-07-14'

            # Log live pick
            sb_id = sb.log_live_pick(_pick_live(), cuota_trigger=2.90, fecha=fecha)

            # Settle
            path = os.path.join(sb.SHADOW_DIR, f'sb_{fecha}.jsonl')
            records = sb._load_jsonl(path)
            mk = records[sb_id]['match_key']
            sb.settle(fecha, resultados_map={
                mk: {'ganador': 'Boogaard', 'cuota_cierre': 3.10, 'provenance': 'manual', 'void': False}
            })

            # Generar report y verificar sección live
            txt = sb.report(desde=fecha, hasta=fecha)
            assert 'LIVE PICKS H100-01' in txt
            assert 'pick_type=live' in txt
            assert 'n=1' in txt
        finally:
            sb.SHADOW_DIR = original_dir

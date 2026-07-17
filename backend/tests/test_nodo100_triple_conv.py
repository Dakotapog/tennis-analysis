"""
Tests Nodo-100 — Triple Convergencia Live: Break State Machine + Dashboard (REGLA-T53).
Invocan funciones reales del módulo. Nunca hardcodean la lógica de estado.

D100-01: BREAK_POSIBLE = 1er drift >= 15%
D100-02: BREAK_CONFIRMADO = 2do ciclo drift >= 12%
D100-07: NORMAL recovery si drift < 10%
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from live_edge_monitor import (
    detect_break_state,
    load_odds_history,
    save_odds_history,
)
from live_dashboard_generator import generar_dashboard_html


# ─── Test 1: primer drift alto → BREAK_POSIBLE ───────────────────────────────

def test_break_posible_en_primer_drift():
    """D100-01: 1er ciclo con drift >= 15% → BREAK_POSIBLE."""
    history = {}
    estado = detect_break_state(
        partido_key='Boogaard_vs_Onclin',
        current_drift=0.183,   # 18.3% >= 15%
        current_cuota=2.90,
        history=history,
    )
    assert estado == 'BREAK_POSIBLE'
    assert history['Boogaard_vs_Onclin']['estado'] == 'BREAK_POSIBLE'
    assert len(history['Boogaard_vs_Onclin']['readings']) == 1


# ─── Test 2: segundo ciclo con drift >= 12% → BREAK_CONFIRMADO ───────────────

def test_break_confirmado_en_segundo_ciclo():
    """D100-02: 2do ciclo consecutivo drift >= 12% → BREAK_CONFIRMADO."""
    history = {}

    # Ciclo 1: BREAK_POSIBLE
    detect_break_state('A_vs_B', 0.183, 2.90, history)
    assert history['A_vs_B']['estado'] == 'BREAK_POSIBLE'

    # Ciclo 2: drift >= 12% → CONFIRMADO
    estado = detect_break_state('A_vs_B', 0.161, 2.85, history)
    assert estado == 'BREAK_CONFIRMADO'
    assert history['A_vs_B']['estado'] == 'BREAK_CONFIRMADO'
    assert len(history['A_vs_B']['readings']) == 2


# ─── Test 3: recovery cancela BREAK_POSIBLE → NORMAL ─────────────────────────

def test_recovery_cancela_break_posible():
    """D100-07: drift < 10% después de BREAK_POSIBLE = fluctuación, volver a NORMAL."""
    history = {}

    # Ciclo 1: BREAK_POSIBLE
    detect_break_state('C_vs_D', 0.183, 2.90, history)
    assert history['C_vs_D']['estado'] == 'BREAK_POSIBLE'

    # Ciclo 2: drift < 10% → recovery → NORMAL
    estado = detect_break_state('C_vs_D', 0.07, 3.31, history)
    assert estado == 'NORMAL'
    assert history['C_vs_D']['estado'] == 'NORMAL'


# ─── Test 4: no re-disparar si fired=True ─────────────────────────────────────

def test_no_refire_si_fired_true():
    """D100-03: si history[partido]['fired']=True, detect retorna BREAK_CONFIRMADO sin cambiar estado."""
    history = {
        'E_vs_F': {
            'readings': [{'ts': '14:05:00', 'cuota': 2.90, 'drift': 0.183}],
            'estado': 'BREAK_CONFIRMADO',
            'fired': True,
        }
    }
    # Llamar de nuevo — debe retornar BREAK_CONFIRMADO y no añadir lecturas (fired=True, sale antes)
    estado = detect_break_state('E_vs_F', 0.14, 2.94, history)
    assert estado == 'BREAK_CONFIRMADO'
    # No debe agregar nueva lectura (retorna temprano por fired)
    assert len(history['E_vs_F']['readings']) == 1


# ─── Test 5: dashboard HTML contiene campos clave ─────────────────────────────

def test_dashboard_html_contiene_campos_clave():
    """D100-05: HTML generado incluye tabla, auto-refresh y campos de estado."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear live_edge mock para que el generador tenga datos
        snap = {
            'ts': '2026-07-14T14:07:00',
            'picks_monitoreados': 1,
            'picks_chequeados_data': [
                {
                    'partido': 'Boogaard vs Onclin',
                    'favorito': 'Boogaard',
                    'cuota_pre': 3.55,
                    'cuota_live': 2.90,
                    'drift_pct': 18.3,
                    'edge_live': 0.087,
                    'trigger': True,
                    'senales': ['STRONG', 'HOT'],
                    'break_state': 'BREAK_CONFIRMADO',
                }
            ],
            'n_triggers': 1,
            'break_confirmados': 1,
            'stake_permitido': True,
        }
        snap_path = os.path.join(tmpdir, 'live_edge_20260714_140700.json')
        with open(snap_path, 'w') as f:
            json.dump(snap, f)

        html_path = generar_dashboard_html(reports_dir=tmpdir)

        assert os.path.exists(html_path)
        html = open(html_path, encoding='utf-8').read()

    # Campos clave en el HTML
    assert 'meta http-equiv="refresh"' in html       # auto-refresh
    assert 'Boogaard vs Onclin' in html              # partido en tabla
    assert 'QUIEBRE CONFIRMADO' in html              # estado label
    assert 'STRONG' in html                          # señal
    assert 'break_confirmados' not in html           # no exponer campo interno
    assert 'Triple Convergencia' in html             # título

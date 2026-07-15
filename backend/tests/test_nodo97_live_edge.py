"""
Tests Nodo-97 — Live Edge Monitor (REGLA-T53).
Invocan funciones reales del módulo. Nunca hardcodean la fórmula.

Ventana ASIMÉTRICA verificada: [-30min pre, +45min post] (D99-05).
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from live_edge_monitor import (
    KambiLiveClientMock,
    calc_drift,
    calc_edge_live,
    en_ventana,
    es_trigger,
    filtrar_picks_monitoreados,
    get_p_modelo,
    run,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _pick(**kwargs):
    base = {
        'favorito_predicho': 'Boogaard',
        'partido': 'Boogaard vs Onclin',
        'confidence_flag': 'STRONG',
        'markov_favorito': 'HOT',
        'cuota_fav': 3.55,
        'edge': 0.22,
        'p_modelo': 0.432,
        'rfi_tier': 0,
        'irp_rival': {},
        'rival_value_flag': False,
        'status': 'APROBADO',
        'phantom_data': False,
    }
    base.update(kwargs)
    return base


def _edge_report(apostar=None, watchlist=None):
    return {
        'metadata': {},
        'apostar': apostar or [],
        'watchlist': watchlist or [],
        'sin_edge': [],
        'sin_datos': [],
    }


# ─── Test 1: TRIGGER cuando drift >= 15% y edge_live > 5% ───────────────────

def test_trigger_cuando_drift_supera_umbral():
    # drift = (3.55 - 2.90) / 3.55 = 18.3% >= 15%
    # edge_live = 0.432 - 1/2.90 = 0.432 - 0.345 = +0.087 > 5%
    drift     = calc_drift(cuota_pre=3.55, cuota_live=2.90)
    edge_live = calc_edge_live(p_modelo=0.432, cuota_live=2.90)
    assert drift >= 0.15
    assert edge_live > 0.05
    assert es_trigger(drift, edge_live) is True


# ─── Test 2: NO trigger si edge_live negativo ─────────────────────────────────

def test_no_trigger_si_edge_negativo():
    # Cuota bajó pero p_modelo no cubre (bajo p_modelo)
    drift     = calc_drift(cuota_pre=3.55, cuota_live=2.90)
    edge_live = calc_edge_live(p_modelo=0.20, cuota_live=2.90)
    assert drift >= 0.15           # drift OK
    assert edge_live <= 0.05       # pero edge negativo/insuficiente
    assert es_trigger(drift, edge_live) is False


# ─── Test 3: NO trigger si drift insuficiente ─────────────────────────────────

def test_no_trigger_si_drift_insuficiente():
    # Solo bajó 5% — no llega al umbral del 15%
    drift     = calc_drift(cuota_pre=3.55, cuota_live=3.37)
    edge_live = calc_edge_live(p_modelo=0.432, cuota_live=3.37)
    assert drift < 0.15
    assert es_trigger(drift, edge_live) is False


# ─── Test 4: solo picks STRONG o HOT monitoreados (D97-08) ──────────────────

def test_solo_picks_strong_hot_monitoreados():
    picks = [
        _pick(confidence_flag='STRONG', markov_favorito='NEUTRAL'),  # STRONG → incluido
        _pick(confidence_flag='LOW',    markov_favorito='HOT'),       # HOT → incluido
        _pick(confidence_flag='LOW',    markov_favorito='NEUTRAL'),   # ninguno → excluido
        _pick(confidence_flag='MODERATE', markov_favorito='COLD'),    # ninguno → excluido
        _pick(confidence_flag='STRONG', status='NO_DATA'),            # NO_DATA → excluido
    ]
    monitoreados = filtrar_picks_monitoreados(picks)
    assert len(monitoreados) == 2
    assert all(
        p['confidence_flag'] == 'STRONG' or p['markov_favorito'] == 'HOT'
        for p in monitoreados
    )


# ─── Test 5: ventana ASIMÉTRICA [-30min, +45min] (D99-05) ───────────────────

def test_ventana_horaria_asintorica():
    inicio = datetime(2026, 7, 14, 14, 0, 0)

    # 60 min ANTES → fuera de ventana (>30min pre)
    ahora_60_antes = inicio - timedelta(minutes=60)
    assert en_ventana(inicio, ahora_60_antes) is False

    # 20 min ANTES → dentro de ventana
    ahora_20_antes = inicio - timedelta(minutes=20)
    assert en_ventana(inicio, ahora_20_antes) is True

    # 50 min DESPUÉS → fuera de ventana (>45min post)
    ahora_50_despues = inicio + timedelta(minutes=50)
    assert en_ventana(inicio, ahora_50_despues) is False

    # 30 min DESPUÉS → dentro de ventana
    ahora_30_despues = inicio + timedelta(minutes=30)
    assert en_ventana(inicio, ahora_30_despues) is True

    # Exactamente en el inicio → dentro
    assert en_ventana(inicio, inicio) is True


# ─── Test 6: combo live construido con 2 triggers ────────────────────────────

def test_combo_live_construido_con_2_triggers():
    # Ambos picks idénticos: cuota_fav=3.55, p_modelo=0.432
    # cuota_live=2.90 → drift=18.3% ✓, edge_live=0.432-1/2.90=0.087 ✓ → TRIGGER
    picks = [
        _pick(favorito_predicho='Boogaard', partido='Boogaard vs Onclin',
              cuota_fav=3.55, p_modelo=0.432),
        _pick(favorito_predicho='Dodig', partido='Dodig vs Rival',
              cuota_fav=3.55, p_modelo=0.432),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'edge_report_20260714_000000.json')
        with open(report_path, 'w') as f:
            json.dump(_edge_report(apostar=picks), f)

        # Mock retorna 2.90 para todos: ambos picks generan trigger
        resultado = run(
            reports_dir=tmpdir,
            cliente=KambiLiveClientMock(cuota=2.90),
            ahora=datetime(2026, 7, 14, 14, 5, 0),
            observe_only=True,
            telegram=False,
        )

    assert resultado['n_triggers'] == 2
    assert resultado['combo_sugerido']['patas'] == 2


# ─── Test 7: output JSON escrito en reports/ ─────────────────────────────────

def test_output_json_escrito_en_reports():
    pick = _pick(cuota_fav=3.55, p_modelo=0.432)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'edge_report_20260714_000000.json')
        with open(report_path, 'w') as f:
            json.dump(_edge_report(apostar=[pick]), f)

        resultado = run(
            reports_dir=tmpdir,
            cliente=KambiLiveClientMock(cuota=2.90),  # drift=18.3%, trigger
            ahora=datetime(2026, 7, 14, 14, 5, 0),
            observe_only=True,
            telegram=False,
        )

        # Verificar que el archivo existe dentro del tmpdir
        live_files = [f for f in os.listdir(tmpdir) if f.startswith('live_edge_')]
        assert len(live_files) == 1
        out_path = os.path.join(tmpdir, live_files[0])
        with open(out_path) as f:
            data = json.load(f)

    assert 'triggers' in data
    assert 'ts' in data
    assert 'picks_monitoreados' in data
    assert data['n_triggers'] == 1


# ─── Test 8: fórmula edge_live correcta (función real, no hardcode) ──────────

def test_edge_live_formula_correcta():
    # edge_live = p_modelo - 1/cuota_live
    # Con p_modelo=0.432, cuota_live=2.90 → 0.432 - 0.3448 = 0.0872
    resultado = calc_edge_live(p_modelo=0.432, cuota_live=2.90)
    esperado  = 0.432 - 1.0 / 2.90
    assert abs(resultado - esperado) < 0.001

    # Verificar que es la FUNCIÓN real, no constante hardcodeada
    resultado2 = calc_edge_live(p_modelo=0.50, cuota_live=2.50)
    esperado2  = 0.50 - 1.0 / 2.50
    assert abs(resultado2 - esperado2) < 0.001
    assert resultado != resultado2   # varía con los inputs

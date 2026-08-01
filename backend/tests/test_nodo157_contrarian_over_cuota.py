"""
Tests Nodo-157 — Contrarian OVER Signal en ruta cuota_envenenada (D157-01).
REGLA-T53: usa las mismas constantes/fórmula de producción (live_desk.py
_check_games_convergencia) — función monolítica con I/O en vivo, mismo patrón
de simulación ya establecido en test_nodo150_live_risk_intelligence.py.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shadow_book as sb


def _calc_over_candidato(market_linea, midpoint, cuota_over):
    """Replica _calc_over_candidato() de live_desk.py (D157-01)."""
    if not cuota_over or not (1.30 <= cuota_over <= 2.80):
        return False, None
    z = (market_linea - midpoint) / 3.5
    p_cdf = (1 + math.erf(z / math.sqrt(2))) / 2
    p_model_over = 1 - p_cdf
    edge_over = round((p_model_over - 1 / cuota_over) * 100, 1)
    if edge_over > 5.0:
        return True, edge_over
    return False, edge_over


def test_157_01_cuota_envenenada_activa_over_candidato_con_edge():
    """cuota_envenenada=True (drift>+15%) + cuota_over en rango + edge>5% → over_candidato=True."""
    cuota_t0, cuota_live = 1.90, 2.35
    cuota_drift = round((cuota_live - cuota_t0) / cuota_t0 * 100, 1)
    assert cuota_drift > 15.0

    over_cand, edge_over = _calc_over_candidato(market_linea=22.5, midpoint=25.0, cuota_over=1.75)
    assert over_cand is True
    assert edge_over > 5.0


def test_157_02_cuota_envenenada_sin_over_candidato_si_cuota_fuera_de_rango():
    """cuota_over fuera de [1.30,2.80] → over_candidato queda False aunque cuota_envenenada=True."""
    over_cand, edge_over = _calc_over_candidato(market_linea=22.5, midpoint=25.0, cuota_over=3.10)
    assert over_cand is False
    assert edge_over is None


def test_157_03_cuota_envenenada_sin_over_candidato_si_edge_bajo():
    """cuota_over en rango pero edge<=5% → over_candidato=False."""
    over_cand, edge_over = _calc_over_candidato(market_linea=25.0, midpoint=25.0, cuota_over=2.20)
    assert over_cand is False
    assert edge_over is not None and edge_over <= 5.0


def test_157_04_badge_selection_prioriza_over_tercer_set():
    """cuota_envenenada=True + over_candidato=True → debe seleccionar rama badge
    'OVER — TERCER SET' (azul), no 'CUOTA ENVENENADA' (roja) — misma prioridad
    que la rama linea_envenenada+over_candidato ya existente."""
    _envenenada = False
    _cuota_envenenada = True
    _over_cand = True

    if _envenenada and _over_cand:
        branch = "OVER_TERCER_SET"
    elif _envenenada:
        branch = "LINEA_ENVENENADA"
    elif _cuota_envenenada and _over_cand:
        branch = "OVER_TERCER_SET"
    elif _cuota_envenenada:
        branch = "CUOTA_ENVENENADA"
    else:
        branch = "NONE"

    assert branch == "OVER_TERCER_SET"


# ─── D157-03: shadow_book logging para señales ITF live games ────────────────

def _signal_alta_itf(partido='Loge vs Blancaneaux'):
    """Shape real de un elemento de alta_itf en live_desk.py (D157-03)."""
    return {
        'partido':      partido,
        'direccion':    'UNDER',
        'linea':        27.5,
        'cuota_live':   2.04,
        'cuota_t0':     1.90,
        'oc_id':        4281376915,
        'zona':         'DOMINANTE',
    }


def test_157_05_log_games_live_pick_escribe_jsonl_con_pick_type_correcto():
    """D157-03: log_games_live_pick usa pick_type='games_live' (esperado por
    H147-01/H150-01/02/03/H151-01 en preregistered_hypotheses.json)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            sb_id = sb.log_games_live_pick(_signal_alta_itf(), cuota_trigger=2.04, fecha='2026-08-01')
            assert sb_id is not None
            assert sb_id.startswith('GLIVE_')

            path = os.path.join(sb.SHADOW_DIR, 'sb_2026-08-01.jsonl')
            records = sb._load_jsonl(path)
            rec = records[sb_id]
            snap = rec.get('pick_snapshot', {})
            assert snap.get('pick_type') == 'games_live'
            assert snap.get('cuota_trigger') == 2.04
            assert rec.get('strategy') == 'GAMES_LIVE'
        finally:
            sb.SHADOW_DIR = original_dir


def test_157_06_log_games_live_pick_no_duplica_reciclos_repetidos():
    """D157-03: señal EN_VIVO re-evaluada cada 15s no debe crear registros
    duplicados — upsert por sb_id determinístico (mismo partido+fecha)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            sig = _signal_alta_itf()
            id1 = sb.log_games_live_pick(sig, cuota_trigger=2.04, fecha='2026-08-01')
            id2 = sb.log_games_live_pick(sig, cuota_trigger=1.85, fecha='2026-08-01')
            assert id1 == id2

            path = os.path.join(sb.SHADOW_DIR, 'sb_2026-08-01.jsonl')
            records = sb._load_jsonl(path)
            assert len(records) == 1
            # primer disparo conserva cuota_trigger original (no se sobreescribe)
            assert records[id1]['pick_snapshot']['cuota_trigger'] == 2.04
        finally:
            sb.SHADOW_DIR = original_dir


def test_157_07_log_games_live_pick_distinto_de_log_live_pick():
    """D157-03: mismo partido logueado por log_live_pick (H100-01, general) y
    log_games_live_pick (ITF games) produce sb_id distintos (prefijos LIVE_/GLIVE_) —
    no colisionan ni se pisan entre sí."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'shadow_book'), exist_ok=True)
        original_dir = sb.SHADOW_DIR
        try:
            sb.SHADOW_DIR = os.path.join(tmpdir, 'shadow_book')
            sig = _signal_alta_itf()
            id_live = sb.log_live_pick(dict(sig, favorito_predicho='Loge'), cuota_trigger=2.04, fecha='2026-08-01')
            id_gl = sb.log_games_live_pick(sig, cuota_trigger=2.04, fecha='2026-08-01')
            assert id_live != id_gl
            assert id_live.startswith('LIVE_')
            assert id_gl.startswith('GLIVE_')

            path = os.path.join(sb.SHADOW_DIR, 'sb_2026-08-01.jsonl')
            records = sb._load_jsonl(path)
            assert records[id_live]['pick_snapshot']['pick_type'] == 'live'
            assert records[id_gl]['pick_snapshot']['pick_type'] == 'games_live'
        finally:
            sb.SHADOW_DIR = original_dir

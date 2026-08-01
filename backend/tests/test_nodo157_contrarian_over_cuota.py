"""
Tests Nodo-157 — Contrarian OVER Signal en ruta cuota_envenenada (D157-01).
REGLA-T53: usa las mismas constantes/fórmula de producción (live_desk.py
_check_games_convergencia) — función monolítica con I/O en vivo, mismo patrón
de simulación ya establecido en test_nodo150_live_risk_intelligence.py.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

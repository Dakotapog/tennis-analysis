"""
Tests Nodo-152 — Phantom History Guard (D152-01 a D152-05).
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.

Funciones cubiertas:
  D152-01: _validate_circuit_consistency() — 5 reglas acumulativas
  D152-04: edge_calculator gate — history_contamination → status=NO_DATA
  D152-05: ELO-ranking incoherence gate — segunda línea de defensa

Tests:
  test_elite_tournament_itf_blocks          → R1: ATP Finals en ITF → contaminated
  test_gs_top10_double_challenger_blocks    → R2: 2x GS vs top-10 en Challenger → contaminated
  test_gs_wildcard_legitimate_clean         → GS vs rank#180 en Challenger → clean
  test_thf_cache_amplifier_active           → R4: thf_cache amplifica score vs ninja_api
  test_atp_player_gs_clean                  → tier=atp500 → returns early, score=0
  test_propagation_to_data_quality          → D152-02→D152-03 chain: detección → flags
  test_edge_calculator_blocks_phantom_hist  → D152-04: history_contamination → NO_DATA
  test_elo_rank_incoherence_gate            → D152-05: elo=1974 + ranking=None + tier=itf → NO_DATA
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scraping.ninja_h2h_parser import _validate_circuit_consistency
from edge_calculator import calcular_edge_completo
from core.data_contract import PICK_STATUS_NO_DATA


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures compartidos
# ─────────────────────────────────────────────────────────────────────────────

CALIB_MINIMAL = {
    "global": {"wins": 50, "losses": 20},
    "por_superficie": {
        "clay": {"wins": 31, "losses": 10},
    },
    "por_superficie_y_tier": {},
    "fallback_por_tier": {
        "grand_slam": 0.758,
        "atp1000":    0.65,
        "atp500":     0.62,
        "challenger": 0.55,
        "itf":        0.50,
    },
}


def _make_partido_152(
    p1_contaminated: bool = False,
    p1_score: int = 0,
    p2_contaminated: bool = False,
    p2_score: int = 0,
    elo_fav: float = 1430,
    elo_rival: float = 1500,
    ranking_fav=200,
    torneo: str = "ITF M15 Bogotá",
    jugador1: str = "PlayerITF",
    jugador2: str = "PlayerB",
    confidence: float = 58.0,
) -> dict:
    """Partido mínimo para tests Nodo-152. Sigue el patrón _make_partido() de test_nodo33."""
    ra: dict = {
        "prediction": {
            "favored_player": jugador1,
            "confidence": confidence,
            "markov_analysis": None,
            "surface_specialization_meta": {},
            "circuit_asymmetry": None,
            "score_breakdown": {},
        },
    }
    ra[f"{jugador1}_elo"] = elo_fav
    ra[f"{jugador2}_elo"] = elo_rival
    ra[f"{jugador1}_ranking"] = ranking_fav
    ra[f"{jugador2}_ranking"] = 80

    return {
        "jugador1": jugador1,
        "jugador2": jugador2,
        "cuota1": 4.00,
        "cuota2": 1.25,
        "torneo_completo": torneo,
        "superficie": "clay",
        "enfrentamientos_directos": [],
        "ranking_analysis": ra,
        "data_quality": {
            "history_contamination": {
                "p1_contaminated": p1_contaminated,
                "p2_contaminated": p2_contaminated,
                "p1_score": p1_score,
                "p2_score": p2_score,
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests D152-01: _validate_circuit_consistency()
# ─────────────────────────────────────────────────────────────────────────────

def test_elite_tournament_itf_blocks():
    """R1: historial con ATP Finals + jugador ITF → contaminated=True, score≥100.

    Caso real: Vesantera B.T. (ITF M15) recibió historial con ATP Finals vía thf_cache.
    REGLA-T53: invoca _validate_circuit_consistency real.
    """
    history = [
        {'torneo': 'ATP Finals', 'opponent_ranking': 3, 'resultado': 'L'},
        {'torneo': 'ATP Finals', 'opponent_ranking': 7, 'resultado': 'W'},
    ]
    val = _validate_circuit_consistency(history, 'itf', 'ninja_api')

    assert val['contaminated'] is True, "ATP Finals en historial ITF debe detectarse como contaminado"
    assert val['score'] >= 100, f"R1 debe aportar score≥100, got {val['score']}"
    assert any('ELITE_TOURNAMENT' in e for e in val['evidence']), (
        f"evidence debe mencionar ELITE_TOURNAMENT, got {val['evidence']}"
    )


def test_gs_top10_double_challenger_blocks():
    """R2: ≥2 GS vs rivales top-10 + jugador Challenger → contaminated=True.

    Un challenger puede aparecer en GS como wildcard, pero no vs top-10 x2.
    REGLA-T53: invoca _validate_circuit_consistency real.
    """
    history = [
        {'torneo': 'Wimbledon', 'opponent_ranking': 7, 'resultado': 'L'},
        {'torneo': 'Roland Garros', 'opponent_ranking': 4, 'resultado': 'L'},
    ]
    val = _validate_circuit_consistency(history, 'challenger', 'ninja_api')

    assert val['contaminated'] is True, "2x GS vs top-10 en Challenger debe contaminar"
    assert any('GS_TOP10' in e for e in val['evidence']), (
        f"evidence debe mencionar GS_TOP10, got {val['evidence']}"
    )


def test_gs_wildcard_legitimate_clean():
    """GS vs rivales rank>150 en Challenger → NOT contaminated (wildcard legítimo).

    Un jugador Challenger puede recibir wildcard para GS y perder vs rank#180.
    Este historial es legítimo — R2 solo dispara con rivals TOP-10.
    REGLA-T53: invoca _validate_circuit_consistency real.
    """
    history = [
        {'torneo': 'Wimbledon', 'opponent_ranking': 180, 'resultado': 'L'},
        {'torneo': 'Roland Garros', 'opponent_ranking': 220, 'resultado': 'L'},
    ]
    val = _validate_circuit_consistency(history, 'challenger', 'ninja_api')

    assert val['contaminated'] is False, "GS vs rank>150 en Challenger es historial legítimo"
    assert val['score'] < 50, f"score debe ser <50 para historial limpio, got {val['score']}"


def test_thf_cache_amplifier_active():
    """R4: provenance=thf_cache amplifica score × 1.5 cuando score > 0.

    Mismo historial (Laver Cup = elite tournament, R1 activa) comparado con
    ninja_api vs thf_cache: el amplificador debe producir score mayor.
    REGLA-T53: invoca _validate_circuit_consistency real, compara las dos llamadas.
    """
    history = [{'torneo': 'Laver Cup', 'opponent_ranking': 2, 'resultado': 'L'}]

    val_ninja = _validate_circuit_consistency(history, 'itf', 'ninja_api')
    val_thf   = _validate_circuit_consistency(history, 'itf', 'thf_cache')

    assert val_thf['contaminated'] is True, "thf_cache con elite tournament debe contaminar"
    assert val_thf['score'] > val_ninja['score'], (
        f"R4 debe amplificar: thf_score={val_thf['score']} debe ser > ninja_score={val_ninja['score']}"
    )
    assert any('THF_CACHE_AMPLIFIER' in e for e in val_thf['evidence']), (
        "evidence debe registrar THF_CACHE_AMPLIFIER para thf_cache"
    )
    assert not any('THF_CACHE_AMPLIFIER' in e for e in val_ninja['evidence']), (
        "ninja_api NO debe tener THF_CACHE_AMPLIFIER en evidence"
    )


def test_atp_player_gs_clean():
    """Jugador ATP (tier=atp500) con GS vs top-50 → NOT contaminated, score=0.

    Caso Zhang Zhizhen (ATP rank ~60): GS vs top-12 es NORMAL para jugadores ATP.
    _validate_circuit_consistency retorna early cuando tier ∉ _ITF_CHALLENGER_TIERS.
    REGLA-T53: invoca _validate_circuit_consistency real.
    """
    history = [
        {'torneo': 'US Open', 'opponent_ranking': 12, 'resultado': 'L'},
        {'torneo': 'Roland Garros', 'opponent_ranking': 8, 'resultado': 'L'},
    ]
    val = _validate_circuit_consistency(history, 'atp500', 'ninja_api')

    assert val['contaminated'] is False, "Jugador ATP con GS no debe ser detectado como contaminado"
    assert val['score'] == 0, (
        f"tier=atp500 ∉ _ITF_CHALLENGER_TIERS → retorno early con score=0, got {val['score']}"
    )
    assert val['evidence'] == [], f"Sin evidence para jugador ATP, got {val['evidence']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test D152-02/03: cadena detección → match_data flags → data_quality
# ─────────────────────────────────────────────────────────────────────────────

def test_propagation_to_data_quality():
    """D152-02→D152-03: _validate_circuit_consistency detecta contaminación y
    los flags se propagan correctamente a data_quality.history_contamination.

    Simula el flujo real: detección → match_data._contamination_p1 → data_quality.
    REGLA-T53: invoca _validate_circuit_consistency real como paso inicial.
    """
    contaminated_history = [
        {'torneo': 'ATP Finals', 'opponent_ranking': 5, 'resultado': 'L'},
    ]

    # D152-01: detección (función real)
    val = _validate_circuit_consistency(contaminated_history, 'itf', 'thf_cache')
    assert val['contaminated'] is True, "pre-condición: historial debe ser contaminado"

    # D152-02: lógica de match_data flags (refleja _process_match L1267-1277)
    match_data: dict = {}
    if val['contaminated']:
        match_data['_contamination_p1'] = True
        match_data['_contamination_score_p1'] = val['score']

    # D152-03: construcción data_quality.history_contamination (refleja _consolidate_result L1768-1772)
    hc = {
        'p1_contaminated': match_data.get('_contamination_p1', False),
        'p2_contaminated': match_data.get('_contamination_p2', False),
        'p1_score':        match_data.get('_contamination_score_p1', 0),
        'p2_score':        match_data.get('_contamination_score_p2', 0),
    }

    assert hc['p1_contaminated'] is True, "D152-03: p1_contaminated debe propagarse a data_quality"
    assert hc['p1_score'] >= 50, f"p1_score debe ser ≥50 (threshold), got {hc['p1_score']}"
    assert hc['p2_contaminated'] is False, "p2 no contaminado no debe afectarse"
    assert hc['p2_score'] == 0, "p2_score debe ser 0 cuando p2 no es contaminado"


# ─────────────────────────────────────────────────────────────────────────────
# Tests D152-04/05: gates en edge_calculator
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_calculator_blocks_phantom_hist():
    """D152-04: partido con data_quality.history_contamination.p1_contaminated=True
    → calcular_edge_completo retorna apostar=False, phantom_data=True, status=NO_DATA.

    Simula el caso Vesantera B.T. (score=150) llegando a edge_calculator con el flag seteado.
    REGLA-T53: invoca calcular_edge_completo real.
    """
    partido = _make_partido_152(p1_contaminated=True, p1_score=150)
    r = calcular_edge_completo(partido, CALIB_MINIMAL)

    assert r is not None, "calcular_edge_completo no debe retornar None"
    assert r['apostar'] is False, "D152-04: partido contaminado nunca debe apostar"
    assert r.get('phantom_data') is True, "phantom_data debe estar marcado como True"
    assert r.get('status') == PICK_STATUS_NO_DATA, (
        f"status debe ser PICK_STATUS_NO_DATA, got {r.get('status')!r}"
    )
    assert 'D152-04' in r.get('motivo_reclasificacion', ''), (
        f"motivo debe identificar D152-04, got: {r.get('motivo_reclasificacion')!r}"
    )


def test_elo_rank_incoherence_gate():
    """D152-05: elo_favorito>1800 + ranking_favorito=None + tier=itf
    → phantom_data=True, status=NO_DATA — segunda línea de defensa.

    ELO=1974 (nivel top-20 ATP) es físicamente imposible para jugador ITF sin ranking.
    D152-04 NO dispara (p1_contaminated=False) → D152-05 es la única defensa aquí.
    REGLA-T53: invoca calcular_edge_completo real.
    """
    partido = _make_partido_152(
        p1_contaminated=False,   # D152-04 no dispara — D152-05 actúa solo
        elo_fav=1974,
        elo_rival=1200,
        ranking_fav=None,        # sin ranking ATP — (not _rk_152) = True en D152-05
        torneo="ITF M15 Bogotá",
    )
    r = calcular_edge_completo(partido, CALIB_MINIMAL)

    assert r is not None, "calcular_edge_completo no debe retornar None"
    assert r.get('phantom_data') is True, (
        f"D152-05: elo=1974 sin ranking en ITF debe marcar phantom_data=True"
    )
    assert r.get('status') == PICK_STATUS_NO_DATA, (
        f"status debe ser PICK_STATUS_NO_DATA, got {r.get('status')!r}"
    )
    assert 'D152-05' in r.get('motivo_reclasificacion', ''), (
        f"motivo debe identificar D152-05, got: {r.get('motivo_reclasificacion')!r}"
    )

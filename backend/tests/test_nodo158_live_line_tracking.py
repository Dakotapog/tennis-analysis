"""
Tests Nodo-158 — Línea/Cuota Actual en vivo (D158-01/D158-02).
REGLA-T53: D158-02 invoca build_games_combos_live() real. D158-01 replica la
fórmula de _check_games_convergencia (función monolítica con I/O en vivo vía
_fetch_live_games_all — mismo patrón de simulación de test_nodo150/157).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import betplay_combo_builder as bcb


def _linea_drift(linea_actual, linea_t0, linea_frozen):
    """Replica el cálculo de sig['linea_drift'] en live_desk.py (D158-01, L4045-46)."""
    base = linea_t0 if linea_t0 is not None else linea_frozen
    return round(linea_actual - base, 1) if base is not None else None


def test_158_01_linea_drift_contra_baseline_t0_no_frozen_actual():
    """linea_drift se calcula contra linea_t0 (baseline inmutable), no contra
    la última 'linea' mutable — caso del usuario: base=21.5, mercado ahora=23.5."""
    drift = _linea_drift(linea_actual=23.5, linea_t0=21.5, linea_frozen=21.5)
    assert drift == 2.0


def test_158_02_linea_drift_fallback_a_linea_si_sin_t0():
    """Si aún no se congeló baseline T0 (primer ciclo), usa 'linea' como base."""
    drift = _linea_drift(linea_actual=23.5, linea_t0=None, linea_frozen=22.0)
    assert drift == 1.5


def _certeza_linea_input(linea_actual, linea_frozen):
    """Replica la selección de línea para _calcular_certeza_condicional (D158-01, L4100):
    prioriza linea_actual sobre la congelada."""
    return float(linea_actual) if linea_actual is not None else float(linea_frozen or 0)


def test_158_03_certeza_usa_linea_actual_cuando_disponible():
    """La certeza condicional debe evaluarse contra la línea REALMENTE tradeable
    (23.5), no la congelada (21.5) — si no, el modelo mide contra un mercado muerto."""
    linea_para_certeza = _certeza_linea_input(linea_actual=23.5, linea_frozen=21.5)
    assert linea_para_certeza == 23.5


def test_158_04_certeza_cae_a_linea_congelada_sin_actual():
    """Sin mercado 'Total de juegos' confirmado (linea_actual=None), certeza usa
    la línea congelada como único dato disponible."""
    linea_para_certeza = _certeza_linea_input(linea_actual=None, linea_frozen=21.5)
    assert linea_para_certeza == 21.5


def _signal_en_vivo(linea_actual=23.5, cuota_actual=1.90, oc_id_actual=999888,
                     direccion="UNDER", zona="DOMINANTE",
                     p_condicional=0.75, score_str="6:4,3:2", games_played=15):
    return {
        "partido": "Jamie Mackenzie vs Max Dahlin",
        "direccion": direccion,
        "estado": "EN_VIVO",
        "linea_actual": linea_actual,
        "cuota_actual": cuota_actual,
        "oc_id_actual": oc_id_actual,
        "zona": zona,
        "certeza": {"p_condicional": p_condicional, "alerta_nivel": "ALTA",
                    "certeza_matematica": False},
        "score_data": {"score_str": score_str, "games_played": games_played},
    }


def _write_games_live(tmp_path, signals):
    p = tmp_path / "games_live_20260801.json"
    p.write_text(json.dumps({"signals_alta": signals}), encoding="utf-8")
    return p


def test_158_05_build_games_combos_live_usa_linea_actual_no_congelada(tmp_path):
    """D158-02: el combo generado debe usar linea_actual/cuota_actual/oc_id_actual
    (mercado real), no la línea congelada — end-to-end contra build_games_combos_live real."""
    path = _write_games_live(tmp_path, [_signal_en_vivo()])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert len(combos) == 1
    leg = combos[0]["legs"][0]
    assert leg["linea"] == 23.5
    assert leg["cuota"] == 1.90
    assert leg["outcome_id"] == "999888"
    assert leg["mercado"] == "Total de juegos"
    assert meta["n_candidatos"] == 1


def test_158_06_build_games_combos_live_excluye_por_score_null_gate(tmp_path):
    """D151-02 vía D158-02: score_str=None con games_played>3 excluye la señal
    del combo live-aware, igual que en la ruta ITF pura."""
    sig = _signal_en_vivo(score_str=None, games_played=10)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_158_07_build_games_combos_live_excluye_por_edge_live_gate(tmp_path):
    """D151-01 vía D158-02: p_condicional bajo frente a la cuota actual (edge<5%)
    excluye la señal — el edge se recalcula contra la línea/cuota ACTUAL, no la T0."""
    sig = _signal_en_vivo(p_condicional=0.30, cuota_actual=1.50)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_158_08_build_games_combos_live_requiere_campos_actual_completos(tmp_path):
    """Sin linea_actual/cuota_actual/oc_id_actual (mercado 'Total de juegos' aún
    no confirmado en este ciclo) la señal se descarta — nunca dispara con datos
    parciales."""
    sig = _signal_en_vivo(linea_actual=None, cuota_actual=None, oc_id_actual=None)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_158_09_build_games_combos_live_sin_archivo_retorna_vacio():
    combos, meta = bcb.build_games_combos_live(games_live_file="reports/no_existe_xyz.json")
    assert combos == []
    assert meta == {}

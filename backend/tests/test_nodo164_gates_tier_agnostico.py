"""
Tests Nodo-164 — Gates D150 (cuota_envenenada + set1-tiebreak) tier-agnósticos.

D164-01: cuota_envenenada para alta_signals (ATP/WTA/Challenger/ATP1000/ATP500) —
replica la fórmula de _check_games_convergencia (función monolítica con I/O en
vivo), mismo patrón de simulación que test_nodo150/157/158.
D164-02: build_games_combos_live() real ahora excluye EN_VIVO (no solo ITF_VIVO)
por cuota_envenenada o set1-tiebreak.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import betplay_combo_builder as bcb


# ── D164-01: cuota_envenenada para alta_signals ──────────────────────────────

def _cuota_envenenada_alta_signals(drift_pct):
    """Replica el cálculo D164-01 en live_desk.py (bloque de match alta_signals,
    mismo umbral CUOTA_ENVENENADA_UMBRAL=15.0 que D150-01 usa para itf_live_signals)."""
    if drift_pct is not None and drift_pct > 15.0:
        return True
    return False


def test_164_01_alta_signals_calcula_cuota_envenenada_igual_que_itf():
    """drift_pct>15% en una señal EN_VIVO no-ITF (ATP1000/Challenger/etc.) debe
    marcar cuota_envenenada=True — mismo umbral D150-01 ya aplicado a ITF."""
    assert _cuota_envenenada_alta_signals(drift_pct=22.5) is True


def test_164_01b_alta_signals_sin_drift_significativo_no_marca_envenenada():
    """Control negativo: drift dentro de rango normal no dispara el flag."""
    assert _cuota_envenenada_alta_signals(drift_pct=4.0) is False
    assert _cuota_envenenada_alta_signals(drift_pct=None) is False


# ── D164-02: build_games_combos_live() excluye EN_VIVO por D150, no solo ITF_VIVO ──

def _signal_en_vivo(estado="EN_VIVO", linea_actual=23.5, cuota_actual=1.90,
                     oc_id_actual=999888, direccion="UNDER", zona="DOMINANTE",
                     p_condicional=0.75, score_str="6:4,3:2", games_played=15,
                     games_set1=None, cuota_envenenada=False):
    return {
        "partido": "Alexander Zverev vs Cameron Norrie",
        "direccion": direccion,
        "estado": estado,
        "linea_actual": linea_actual,
        "cuota_actual": cuota_actual,
        "oc_id_actual": oc_id_actual,
        "zona": zona,
        "cuota_envenenada": cuota_envenenada,
        "certeza": {"p_condicional": p_condicional, "alerta_nivel": "ALTA",
                    "certeza_matematica": False},
        "score_data": {"score_str": score_str, "games_played": games_played,
                        "games_set1": games_set1},
    }


def _write_games_live(tmp_path, signals):
    p = tmp_path / "games_live_20260803.json"
    p.write_text(json.dumps({"signals_alta": signals}), encoding="utf-8")
    return p


def test_164_02_build_games_combos_live_excluye_en_vivo_con_cuota_envenenada(tmp_path):
    """D150-01 ahora tier-agnóstico: una pierna EN_VIVO (no ITF_VIVO) con
    cuota_envenenada=True se excluye del coupon, igual que ya pasaba con ITF."""
    sig = _signal_en_vivo(estado="EN_VIVO", cuota_envenenada=True)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_164_03_build_games_combos_live_excluye_en_vivo_con_set1_tiebreak(tmp_path):
    """D150-06 ahora tier-agnóstico: set1 tiebreak (>=12 juegos) excluye una
    pierna EN_VIVO (no ITF_VIVO), igual que ya pasaba con ITF."""
    sig = _signal_en_vivo(estado="EN_VIVO", games_set1=13)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_164_04_build_games_combos_live_itf_vivo_sigue_excluido_por_cuota_envenenada(tmp_path):
    """Regresión: ITF_VIVO con cuota_envenenada=True sigue excluido — el nuevo
    check no rompe el comportamiento previo de la ruta ITF."""
    sig = _signal_en_vivo(estado="ITF_VIVO", cuota_envenenada=True)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert combos == []


def test_164_05_build_games_combos_live_admite_en_vivo_limpia(tmp_path):
    """Control positivo: EN_VIVO sin cuota_envenenada ni set1-tiebreak sigue
    generando combo normalmente (el nuevo gate no bloquea señales sanas)."""
    sig = _signal_en_vivo(estado="EN_VIVO", cuota_envenenada=False, games_set1=6)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    assert len(combos) == 1
    assert combos[0]["legs"][0]["outcome_id"] == "999888"


# ── D164-05 (fix post-verificación 2026-08-03): leg dict live debe traer las
# keys que _mostrar_games_combos() accede sin .get() (zona_diff/gap_juegos/
# confianza_señal) — bug real encontrado al correr build_games_combos_live()
# de punta a punta por primera vez (KeyError: 'zona_diff' línea 2021). ──

def test_164_06_leg_dict_live_incluye_keys_que_mostrar_games_combos_requiere(tmp_path):
    """El leg dict de build_games_combos_live() debe traer zona_diff/gap_juegos/
    confianza_señal — _mostrar_games_combos() las lee con [] no .get()."""
    sig = _signal_en_vivo(estado="EN_VIVO", zona="DOMINANTE", games_set1=6,
                           games_played=15, linea_actual=23.5)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    leg = combos[0]["legs"][0]
    assert "zona_diff" in leg
    assert "gap_juegos" in leg
    assert "confianza_señal" in leg
    # zona="DOMINANTE" (mayúscula, D147 schema) debe normalizarse a minúscula
    # para calzar con la comparación == "dominante" en _mostrar_games_combos().
    assert leg["zona_diff"] == "dominante"
    assert leg["gap_juegos"] == abs(15 - 23.5)


def test_164_07_mostrar_games_combos_no_crashea_con_combo_live(tmp_path, capsys):
    """Regresión directa del crash reportado: _mostrar_games_combos() real,
    sin mocks, corriendo sobre un combo producido por build_games_combos_live()
    — antes crasheaba con KeyError: 'zona_diff' en la primera pierna live."""
    sig = _signal_en_vivo(estado="EN_VIVO", zona="DOMINANTE", games_set1=6)
    path = _write_games_live(tmp_path, [sig])
    combos, meta = bcb.build_games_combos_live(games_live_file=str(path))
    bcb._mostrar_games_combos(combos, meta)  # no debe lanzar KeyError
    out = capsys.readouterr().out
    assert "DOMIN" in out

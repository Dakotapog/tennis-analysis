"""
Tests Nodo-180 D180-09 — Tripwire: el mercado SETS no puede convertirse en
ruta de apuesta sin auditoría.

§9.1 del Nodo-180 establece que SETS hoy no apuesta dinero por accidente de
cableado (Combo C = print() a stdout, sin .bat/HTML/Telegram/ComboRegistry/
shadow_book) — no por diseño. Comparte F4 (_P_3SETS_POR_ZONA es un lookup
plano pre-partido, jamás condicionado al marcador de sets en curso). Este
test es el rastro de papel que impide que ese cableado accidental se rompa
en silencio el día que alguien conecte el Combo C a un cupón real.

REGLA-T53: invoca build_games_combos() y build_games_combos_live() reales
del módulo — no reimplementa el filtro que está verificando.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import betplay_combo_builder as bcb


# ── Helpers (estático — build_games_combos) ──────────────────────────────────

def _señal_juegos(direccion="UNDER", linea=22.5, cuota=1.85, gap=4.0,
                   outcome_id=123456, confianza="ALTA"):
    return {
        "mercado": "Total de juegos",
        "mercado_tipo": "JUEGOS",
        "linea": linea,
        "direccion": direccion,
        "cuota": cuota,
        "outcome_id": outcome_id,
        "gap_juegos": gap,
        "razon": "test",
        "confianza_señal": confianza,
        "apostar": True,
    }


def _señal_sets(direccion="UNDER", linea=2.5, cuota=1.65,
                 outcome_id=789012, confianza="MEDIA"):
    return {
        "mercado": "Total de sets",
        "mercado_tipo": "SETS",
        "linea": linea,
        "direccion": direccion,
        "cuota": cuota,
        "outcome_id": outcome_id,
        "gap_juegos": None,
        "razon": "test sets",
        "confianza_señal": confianza,
        "apostar": True,
    }


def _write_games_signal_report(tmp_path, partidos):
    p = tmp_path / "games_signal_report_20260812_120000.json"
    p.write_text(json.dumps({"apostar": partidos, "metadata": {"calibracion_n": 50}}),
                 encoding="utf-8")
    return p


# ── Helpers (live — build_games_combos_live) ──────────────────────────────────

def _signal_en_vivo(partido, linea_actual=23.5, cuota_actual=1.90,
                     oc_id_actual=999888, direccion="UNDER", zona="DOMINANTE",
                     p_condicional=0.75, score_str="6:4,3:2", games_played=15,
                     mercado_tipo=None):
    sig = {
        "partido": partido,
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
    if mercado_tipo is not None:
        sig["mercado_tipo"] = mercado_tipo
    return sig


def _write_games_live(tmp_path, signals):
    p = tmp_path / "games_live_20260812.json"
    p.write_text(json.dumps({"signals_alta": signals}), encoding="utf-8")
    return p


# ── D180-09: test_180_80 ──────────────────────────────────────────────────────

def test_180_80_sets_nunca_alcanza_artefacto_de_apuesta(tmp_path):
    """Tripwire Nodo-180 D180-09. Si este test falla, SETS se convirtió en
    ruta de apuesta y arrastra F4 (prior incondicional `_P_3SETS_POR_ZONA`).
    NO parchear el test: auditar SETS primero (nodo aparte, con el mismo
    rigor que Nodo-180)."""

    # ── Parte 1: build_games_combos() — reporte estático ──────────────────
    # Partido A: JUEGOS puro → debe generar leg normalmente (control positivo).
    partido_juegos = {
        "partido": "Normal Juegos Match",
        "zona_diff": "dominante",
        "diff_abs": 4.0,
        "señales_optimas": [_señal_juegos()],
    }
    # Partido B: JUEGOS + SETS mezclados en la misma señales_optimas — el
    # primer filtro (D149-05) ya debería quedarse solo con JUEGOS, pero lo
    # cubrimos explícitamente como regresión de Nodo-149.
    partido_mixto = {
        "partido": "Mixto Match",
        "zona_diff": "coinflip",
        "diff_abs": 3.0,
        "señales_optimas": [_señal_sets(outcome_id=555111), _señal_juegos(outcome_id=555222)],
    }
    # Partido C: señales_optimas contiene SOLO SETS (mercado_tipo="SETS",
    # mercado="Total de sets" — no matchea el filtro D149-05 → señales_juegos
    # queda vacío → dispara la rama de retrocompatibilidad
    # `if not señales_juegos: señales_juegos = señales` (betplay_combo_builder.py
    # ~2044-2045). Esta es la rama que el D180-09 audita: antes del guard
    # agregado en este mismo nodo, la señal SETS se colaba por aquí sin que
    # nadie la filtrara. outcome_id distintivo para poder afirmarlo con certeza.
    partido_solo_sets = {
        "partido": "All Sets Match",
        "zona_diff": "ajustada",
        "diff_abs": 1.0,
        "señales_optimas": [_señal_sets(outcome_id=777333)],
    }

    path = _write_games_signal_report(
        tmp_path, [partido_juegos, partido_mixto, partido_solo_sets]
    )
    combos, _meta = bcb.build_games_combos(games_file=str(path))

    assert combos, "esperaba al menos 1 combo desde el partido JUEGOS puro"
    # build_games_combos() conserva el tipo original de outcome_id (int, tal
    # como llega en la señal) — legs = signals[:max_legs] sin stringificar
    # (solo ids_str los convierte a str para la URL). Comparar contra ints,
    # no strings, o la aserción pasaría trivialmente sin probar nada.
    ids_sets_prohibidos = {555111, 777333}
    partidos_sets_prohibidos = {"All Sets Match"}
    for combo in combos:
        for leg in combo["legs"]:
            assert leg["outcome_id"] not in ids_sets_prohibidos, (
                f"leg con outcome_id de señal SETS ({leg['outcome_id']}) "
                f"alcanzó un combo apostable — {leg}"
            )
            assert leg["partido"] not in partidos_sets_prohibidos, (
                f"partido cuya única señal era SETS alcanzó un combo apostable "
                f"vía la rama de retrocompatibilidad — {leg}"
            )

    # ── Parte 2: build_games_combos_live() — pipeline en vivo ─────────────
    sig_juegos_live = _signal_en_vivo("Normal Live Match", oc_id_actual=999888)
    sig_sets_live = _signal_en_vivo(
        "Sets Live Match", oc_id_actual=777333, mercado_tipo="SETS"
    )
    path_live = _write_games_live(tmp_path, [sig_juegos_live, sig_sets_live])
    combos_live, meta_live = bcb.build_games_combos_live(games_live_file=str(path_live))

    assert combos_live, "esperaba al menos 1 combo desde la señal JUEGOS en vivo"
    assert meta_live["n_candidatos"] == 1, (
        "la señal SETS no debe contar como candidato — "
        f"n_candidatos={meta_live['n_candidatos']}"
    )
    for combo in combos_live:
        for leg in combo["legs"]:
            assert leg["outcome_id"] != "777333", (
                f"leg con outcome_id de señal SETS en vivo alcanzó un combo "
                f"apostable — {leg}"
            )
            assert leg["partido"] != "Sets Live Match", (
                f"partido SETS en vivo alcanzó un combo apostable — {leg}"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

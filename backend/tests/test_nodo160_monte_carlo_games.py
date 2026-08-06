"""
tests/test_nodo160_monte_carlo_games.py — REGLA-T53 Nodo-160 D160-02

Invoca la función real simular_total_juegos_condicionado() — nunca hardcodea
la fórmula del modelo. Casos: determinismo con seed, sensibilidad direccional
(UNDER vs OVER), y n_sims<=0 (guard).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.monte_carlo_games import simular_total_juegos_condicionado as sim
from core.monte_carlo_games import estimar_p_hold


def test_160_40_determinista_con_seed():
    kwargs = dict(
        games_played=10, current_set_home=3, current_set_away=3, serving="home",
        sets_home=0, sets_away=1, p_hold_home=0.6, p_hold_away=0.6,
        linea=22.5, direccion="UNDER", n_sims=500, seed=99,
    )
    r1 = sim(**kwargs)
    r2 = sim(**kwargs)
    assert r1 == r2


def test_160_41_partido_casi_decidido_under_alto_gana_casi_siempre():
    """Match 1-1 sets, set actual 5-4, línea UNDER holgada (40.5) — el total
    final no puede alejarse mucho de games_played + juegos del set decisivo,
    así que p_condicional_mc para UNDER 40.5 debe ser prácticamente 1."""
    r = sim(
        games_played=20, current_set_home=5, current_set_away=4, serving="home",
        sets_home=1, sets_away=1, p_hold_home=0.65, p_hold_away=0.65,
        linea=40.5, direccion="UNDER", n_sims=2000, seed=42,
    )
    assert r["p_condicional_mc"] > 0.95
    assert r["media_total_juegos"] < 40.5


def test_160_42_over_muy_bajo_temprano_partido_gana_casi_siempre():
    """Partido recién empezando (0 sets, set actual 2-1) con línea OVER muy
    baja (15.5) — el total final va a superarla casi siempre."""
    r = sim(
        games_played=5, current_set_home=2, current_set_away=1, serving="away",
        sets_home=0, sets_away=0, p_hold_home=0.65, p_hold_away=0.65,
        linea=15.5, direccion="OVER", n_sims=2000, seed=7,
    )
    assert r["p_condicional_mc"] > 0.95


def test_160_43_n_sims_cero_no_crashea():
    r = sim(
        games_played=10, current_set_home=0, current_set_away=0, serving="home",
        sets_home=0, sets_away=0, p_hold_home=0.6, p_hold_away=0.6,
        linea=22.5, direccion="UNDER", n_sims=0,
    )
    assert r["n_sims"] == 0
    assert r["p_condicional_mc"] is None


def test_160_44_percentiles_ordenados():
    r = sim(
        games_played=8, current_set_home=1, current_set_away=1, serving="home",
        sets_home=0, sets_away=0, p_hold_home=0.62, p_hold_away=0.58,
        linea=22.5, direccion="UNDER", n_sims=1000, seed=3,
    )
    assert r["p10_total_juegos"] <= r["media_total_juegos"] <= r["p90_total_juegos"]


# ─── estimar_p_hold() — proxy ranking/superficie (spec §3.3.1) ────────────────

def test_160_45_hierba_sube_clay_baja():
    """Mismo ranking, distinta superficie → hierba > dura > arcilla (coherente
    con el boost de GCS en hierba, CLAUDE.md §5)."""
    p_hierba = estimar_p_hold(100, "Hierba")
    p_dura = estimar_p_hold(100, "Dura")
    p_arcilla = estimar_p_hold(100, "Arcilla")
    assert p_hierba > p_dura > p_arcilla


def test_160_46_mejor_ranking_sube_p_hold():
    assert estimar_p_hold(20, "Dura") > estimar_p_hold(600, "Dura")


def test_160_47_sin_datos_usa_base_062():
    assert estimar_p_hold(None, None) == 0.62


def test_160_48_clamp_050_085():
    assert 0.50 <= estimar_p_hold(1, "Hierba") <= 0.85
    assert 0.50 <= estimar_p_hold(2000, "Arcilla") <= 0.85

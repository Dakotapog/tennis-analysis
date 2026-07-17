"""
tests/test_nodo110_favoritos_combo.py — REGLA-T53: invocan funciones reales.

Cubre Nodo-110 D110-01/D110-04/H110-01:
  T1. Filtro seguridad (NO_DATA, phantom, historial_incompleto)
  T2. Filtro favorito claro — p_modelo>=0.62
  T3. Filtro favorito claro — cuota<=1.40 con conf!=LOW
  T4. Filtro cuota rango [1.15, 2.10]
  T5. Filtro model=bookie (cuota_fav < cuota_rival)
  T6. Diversificación por torneo (máx 2 piernas por torneo)
  T7. Solape <=2 piernas entre combos seleccionados
  T8. Fixture jul-16: reproduce >=3 piernas reales del operador
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from favoritos_combo_builder import seleccionar_favoritos, armar_combos, LEG_MIN_CUOTA, LEG_MAX_CUOTA


# ── Fixtures helpers ──────────────────────────────────────────────────────────

def _pick(favorito="Jugador A", torneo="ITF-M10-BuenosAires", p_modelo=0.70,
          cuota_favorito=1.55, cuota_rival=2.40, confidence_flag="STRONG",
          status="APOSTAR", phantom=False, historial=False, no_data=False):
    s = "NO_DATA" if no_data else status
    return {
        "favorito": favorito,
        "torneo": torneo,
        "p_modelo": p_modelo,
        "cuota_favorito": cuota_favorito,
        "cuota_rival": cuota_rival,
        "confidence_flag": confidence_flag,
        "status": s,
        "phantom_data": phantom,
        "historial_incompleto": historial,
        "ranking_favorito": 100,
        "ranking_rival": 200,
    }


# ── T1: Filtro seguridad ──────────────────────────────────────────────────────

def test_filtro_no_data_excluido():
    """Picks con status=NO_DATA no entran al universo."""
    picks = [_pick(no_data=True), _pick(favorito="B")]
    validos, conteos = seleccionar_favoritos(picks)
    nombres = [p["favorito"] for p in validos]
    assert "Jugador A" not in nombres
    assert conteos["descartados_NO_DATA"] >= 1


def test_filtro_phantom_excluido():
    """Picks con phantom_data=True son descartados por seguridad."""
    picks = [_pick(phantom=True), _pick(favorito="B")]
    validos, conteos = seleccionar_favoritos(picks)
    nombres = [p["favorito"] for p in validos]
    assert "Jugador A" not in nombres
    assert conteos["descartados_phantom"] >= 1


def test_filtro_historial_incompleto():
    """Picks con historial_incompleto=True son descartados."""
    picks = [_pick(historial=True), _pick(favorito="B")]
    validos, conteos = seleccionar_favoritos(picks)
    assert "Jugador A" not in [p["favorito"] for p in validos]
    assert conteos["descartados_historial"] >= 1


# ── T2: p_modelo>=0.62 ────────────────────────────────────────────────────────

def test_filtro_p_modelo_alto_pasa():
    """p_modelo=0.75 pasa el filtro de favorito claro."""
    picks = [_pick(p_modelo=0.75, cuota_favorito=1.60, cuota_rival=2.20)]
    validos, _ = seleccionar_favoritos(picks)
    assert len(validos) == 1


def test_filtro_p_modelo_bajo_sin_alternativa():
    """p_modelo=0.50 sin cuota_clara ni ranking_gap → descartado."""
    picks = [_pick(p_modelo=0.50, cuota_favorito=1.80, cuota_rival=2.10,
                   confidence_flag="LOW")]
    # ranking_favorito=100, ranking_rival=200 → gap=100 < 300
    validos, conteos = seleccionar_favoritos(picks)
    assert len(validos) == 0
    assert conteos["descartados_no_favorito"] >= 1


# ── T3: cuota_fav<=1.40 con conf!=LOW ────────────────────────────────────────

def test_filtro_cuota_clara_strong_pasa():
    """cuota_fav=1.30, confidence=STRONG → pasa aunque p_modelo<0.62."""
    picks = [_pick(p_modelo=0.58, cuota_favorito=1.30, cuota_rival=3.20,
                   confidence_flag="STRONG")]
    validos, _ = seleccionar_favoritos(picks)
    assert len(validos) == 1


def test_filtro_cuota_clara_low_excluido():
    """cuota_fav=1.30 pero confidence=LOW → no pasa filtro favorito_claro si p_modelo<0.62."""
    picks = [_pick(p_modelo=0.55, cuota_favorito=1.30, cuota_rival=3.20,
                   confidence_flag="LOW")]
    # p_modelo=0.55 < 0.62; cuota<=1.40 pero conf=LOW; ranking_gap=100 < 300
    validos, _ = seleccionar_favoritos(picks)
    assert len(validos) == 0


# ── T4: Cuota rango [LEG_MIN_CUOTA, LEG_MAX_CUOTA] ───────────────────────────

def test_filtro_cuota_por_debajo_del_piso():
    """cuota_fav=1.10 < LEG_MIN_CUOTA=1.15 → descartado."""
    picks = [_pick(p_modelo=0.90, cuota_favorito=1.10, cuota_rival=6.00)]
    validos, conteos = seleccionar_favoritos(picks)
    assert len(validos) == 0
    assert conteos["descartados_cuota_rango"] >= 1


def test_filtro_cuota_por_encima_del_techo():
    """cuota_fav=2.50 > LEG_MAX_CUOTA=2.10 (sin --mega) → descartado."""
    picks = [_pick(p_modelo=0.65, cuota_favorito=2.50, cuota_rival=1.55)]
    validos, conteos = seleccionar_favoritos(picks)
    assert len(validos) == 0
    assert conteos["descartados_cuota_rango"] >= 1


def test_filtro_cuota_spice_pasa_con_mega():
    """cuota_fav=2.50 pasa si mega=True (techo=5.00)."""
    picks = [_pick(p_modelo=0.65, cuota_favorito=2.50, cuota_rival=1.55)]
    # cuota_fav < cuota_rival → FALSO: 2.50 < 1.55 es False → descartado por model!=bookie
    # Ajustar: cuota_fav debe ser < cuota_rival
    picks = [_pick(p_modelo=0.65, cuota_favorito=2.50, cuota_rival=3.20)]
    validos, _ = seleccionar_favoritos(picks, mega=True)
    assert len(validos) == 1


# ── T5: Filtro model=bookie ───────────────────────────────────────────────────

def test_filtro_model_neq_bookie_excluido():
    """cuota_fav >= cuota_rival → fav del modelo NO es fav del bookmaker → descartado."""
    # cuota_fav=1.80 en rango [1.15,2.10] pero cuota_fav > cuota_rival=1.60
    picks = [_pick(p_modelo=0.70, cuota_favorito=1.80, cuota_rival=1.60)]
    validos, conteos = seleccionar_favoritos(picks)
    assert len(validos) == 0
    assert conteos["descartados_model_neq_bookie"] >= 1


# ── T6: Diversificación por torneo ───────────────────────────────────────────

def test_diversificacion_max_2_por_torneo():
    """Combo no puede tener más de 2 piernas del mismo torneo."""
    picks = [
        _pick(favorito=f"Jugador {i}", torneo="ITF-M10-BuenosAires",
              p_modelo=0.70, cuota_favorito=1.55, cuota_rival=2.40)
        for i in range(4)
    ]
    combos = armar_combos(picks)
    for combo in combos:
        torneo_count = {}
        for leg in combo["legs"]:
            t = leg.get("torneo", "")
            torneo_count[t] = torneo_count.get(t, 0) + 1
        assert max(torneo_count.values()) <= 2, f"Combo viola diversificacion: {torneo_count}"


# ── T7: Solape <=2 piernas entre combos seleccionados ────────────────────────

def test_solape_maximo_2_piernas():
    """Cualquier par de combos seleccionados comparte <=2 piernas."""
    from favoritos_combo_builder import _normalize_name
    picks = [
        _pick(favorito=f"Jugador {i}", torneo=f"ITF-Torneo-{i%3}",
              p_modelo=0.68 - i * 0.01, cuota_favorito=1.50 + i * 0.05,
              cuota_rival=2.20 + i * 0.05)
        for i in range(8)
    ]
    combos = armar_combos(picks)
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            jugs_i = {_normalize_name(p.get("favorito", "")) for p in combos[i]["legs"]}
            jugs_j = {_normalize_name(p.get("favorito", "")) for p in combos[j]["legs"]}
            solape = len(jugs_i & jugs_j)
            assert solape <= 2, f"Combos {i+1} y {j+1} tienen solape={solape} > 2"


# ── T8: Fixture jul-16 reproduce piernas reales ──────────────────────────────

def test_fixture_jul16_piernas_reales():
    """
    Fixture basado en el reporte jul-16 del operador.
    Los 8 combos reales usaron favoritos con p>=0.70 y cuota [1.15, 2.10].
    El test verifica que la función de selección retorna >=3 de estos patrones.
    (Fixture sintético con los parámetros documentados en Nodo-110 §1)
    """
    # Picks representativos de los combos reales del operador (jul-14/16)
    fixture = [
        _pick("McNeil",    "Challenger-WTA",  p_modelo=0.82, cuota_favorito=1.23, cuota_rival=4.10),
        _pick("Grubor",    "ITF-M15",         p_modelo=0.78, cuota_favorito=1.32, cuota_rival=3.50),
        _pick("Galarneau", "Challenger-ATP",  p_modelo=0.80, cuota_favorito=1.15, cuota_rival=5.00),
        _pick("Batin",     "ITF-M25",         p_modelo=0.76, cuota_favorito=1.15, cuota_rival=4.80),
        _pick("Forbes",    "ITF-W15",         p_modelo=0.72, cuota_favorito=1.45, cuota_rival=2.80),
        _pick("Gaines",    "ITF-W25",         p_modelo=0.74, cuota_favorito=1.38, cuota_rival=3.20),
        _pick("Bynoe",     "ITF-W10",         p_modelo=0.71, cuota_favorito=1.52, cuota_rival=2.60),
        _pick("Zarazua",   "WTA-ITF",         p_modelo=0.68, cuota_favorito=1.68, cuota_rival=2.10),
        # Piernas spice del operador (para fixture con mega=False deben estar en rango)
        _pick("Mikulskyte","ITF-W15-B",       p_modelo=0.62, cuota_favorito=2.05, cuota_rival=1.75),
    ]
    # La última (Mikulskyte) tiene cuota_fav>cuota_rival → descartada por model!=bookie
    validos, conteos = seleccionar_favoritos(fixture)
    # Debe recuperar >=5 de las 8 piernas reales que están en rango [1.15, 2.10]
    # (Galarneau y Batin @1.15 están en el límite exacto del piso D110-01)
    assert len(validos) >= 5, (
        f"Solo {len(validos)} piernas validas — esperadas >=5. Conteos: {conteos}"
    )
    nombres_validos = {p["favorito"] for p in validos}
    piernas_reales = {"McNeil", "Grubor", "Galarneau", "Batin", "Forbes", "Gaines", "Bynoe"}
    recuperadas = nombres_validos & piernas_reales
    assert len(recuperadas) >= 5, (
        f"Solo {len(recuperadas)} piernas reales recuperadas: {recuperadas}"
    )


# ── T extra: governor BLOCK → no emite (integración argparse) ────────────────

def test_favoritos_acepta_arg_override():
    """favoritos_combo_builder.py acepta --override-governor sin crash de argparse."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "favoritos_combo_builder.py", "--override-governor", "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    assert "error: unrecognized arguments" not in result.stderr
    assert "unrecognized" not in result.stderr.lower()

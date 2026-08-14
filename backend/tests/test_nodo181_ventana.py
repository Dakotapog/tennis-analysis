"""
Tests Nodo-181 D181-01/D181-02 — reloj del sismografo + ledger unificado.

Cubre SOLO lo implementado hasta ahora de D181-09 (5 de los 12 tests
especificados: test_181_01/02/03 sobre lead_time_report, test_181_09/10
sobre fire_ledger). Los 7 restantes (p_wave, quorum_sensores, tier ACCION,
_estimar_ventana_restante) quedan pendientes hasta que D181-03..D181-08 se
implementen — no se fabrican aquí contra código que no existe (REGLA-T53).

REGLA-T53: invoca las funciones reales de los módulos, no reimplementa
la fórmula que está verificando.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.lead_time_report as ltr
import core.fire_guard as fire_guard
import core.fire_ledger as fire_ledger
import core.games_arithmetic as ga
import core.row_coherence as rc


def test_181_01_mov_despues_pct_correcto_serie_sintetica():
    """lead_time_report calcula mov_despues_pct correctamente sobre una serie sintética."""
    from datetime import datetime
    fecha_base = datetime(2026, 8, 1)
    serie = [
        {"ts": "10:00", "cuota": 2.00, "games_played": 10},
        {"ts": "10:05", "cuota": 1.50, "games_played": 12},  # disparo aquí
        {"ts": "10:10", "cuota": 1.20, "games_played": 14},  # final
    ]
    r = ltr.procesar_disparo("2026-08-01T10:05:00", serie, fecha_base)
    assert r["cuota_fire"] == 1.50
    assert r["cuota_final"] == 1.20
    assert abs(r["mov_despues_pct"] - 20.0) < 0.01
    assert r["direccion_movimiento"] == "A_FAVOR"


def test_181_02_serie_cruza_medianoche_no_produce_ventana_negativa():
    """Guard de rollover: serie 23:5x -> 00:0x no debe dar ventana_min negativa (§D181-01)."""
    from datetime import datetime
    fecha_base = datetime(2026, 8, 1)
    serie = [
        {"ts": "23:50", "cuota": 2.00, "games_played": 20},
        {"ts": "23:58", "cuota": 1.80, "games_played": 21},  # disparo aquí
        {"ts": "00:03", "cuota": 1.60, "games_played": 22},
        {"ts": "00:10", "cuota": 1.40, "games_played": 23},
    ]
    r = ltr.procesar_disparo("2026-08-01T23:58:00", serie, fecha_base)
    assert r["ventana_min"] >= 0
    assert r["ventana_min"] == 12.0  # 23:58 -> 00:10 del dia siguiente = 12 min


def test_181_03_sin_serie_cuotas_cae_en_sin_historial_y_cuenta_en_total():
    """Disparo sin historial de cuotas se reporta como SIN_HISTORIAL, nunca se excluye del total."""
    tmp_dir = ROOT / "reports"
    fecha = "20991231"  # fecha sintética que no colisiona con datos reales
    fired_path = tmp_dir / f"certeza_fired_{fecha}.json"
    odds_path = tmp_dir / f"games_odds_history_{fecha}.json"
    assert not fired_path.exists() and not odds_path.exists(), "fecha sintética debe estar libre"
    try:
        fired_path.write_text(json.dumps({"PartidoX_UNDER": "2099-12-31T10:00:00"}), encoding="utf-8")
        # deliberadamente NO se crea odds_path -> sin historial
        resultados = ltr.procesar_dia(fecha)
        assert len(resultados) == 1
        assert resultados[0]["categoria"] == "SIN_HISTORIAL"
        agregados = ltr.calcular_agregados(resultados)
        assert agregados["n_total"] == 1
        assert agregados["n_sin_historial"] == 1
        assert agregados["n_medidos"] == 0
    finally:
        fired_path.unlink(missing_ok=True)
        odds_path.unlink(missing_ok=True)


def test_181_09_registrar_disparo_no_rompe_fire_guard_should_fire():
    """fire_ledger es aditivo: escribir en el ledger no interfiere con el contrato de fire_guard."""
    tmp_dir = ROOT / "reports"
    fecha = "20991230"
    guard_path = tmp_dir / f"__test_guard_{fecha}.json"
    ledger_path = tmp_dir / f"fire_ledger_{fecha}.jsonl"
    try:
        assert fire_guard.should_fire(guard_path, ["A", "B"]) is True
        fire_ledger.registrar_disparo(fecha, "A_UNDER", "CERTEZA", cuota_al_disparo=1.5)
        # el ledger no toca guard_path en absoluto
        assert fire_guard.should_fire(guard_path, ["A", "B"]) is True
        fire_guard.mark_fired(guard_path, ["A", "B"])
        assert fire_guard.should_fire(guard_path, ["A", "B"]) is False
        assert ledger_path.exists()
        linea = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
        assert linea["clave"] == "A_UNDER"
        assert linea["tipo"] == "CERTEZA"
    finally:
        guard_path.unlink(missing_ok=True)
        ledger_path.unlink(missing_ok=True)


def test_181_10_fallo_escritura_ledger_no_impide_disparo(monkeypatch):
    """Best-effort: si escribir el ledger lanza excepción, registrar_disparo no propaga (no bloquea el disparo real)."""
    def _boom(*a, **kw):
        raise OSError("disco lleno (simulado)")
    monkeypatch.setattr(fire_ledger.Path, "open", _boom)
    # No debe lanzar excepción pese al fallo de escritura simulado.
    fire_ledger.registrar_disparo("20991229", "C_OVER", "COMBO", cuota_al_disparo=1.9)


def test_181_12_caso_nally_kessler_juegos_restantes_min_y_estado_linea():
    """Caso exacto §3.B.1: Nally C. vs Kessler M., 6:3 5:2 break point a favor de Nally.

    juegos_restantes_min debe dar 1 (Nally gana el juego actual y cierra el partido
    2 sets a 0), total_min=17 (16 ya jugados + 1), y estado_linea debe distinguir
    una línea aún VIVA (21.5) de una ya RESUELTA (15.5, superada por los 16 juegos
    ya disputados) — sin fabricar el "26-32+" de una distribución de partido completo.
    """
    minimo = ga.juegos_restantes_min(
        sets_ganados_home=1, sets_ganados_away=0,
        juegos_home=5, juegos_away=2, sets_a_ganar=2,
    )
    assert minimo == 1

    juegos_jugados_total = 16  # set1 6:3=9 + set2 en curso 5:2=7
    total_min, total_max_set_actual = ga.total_alcanzable(
        juegos_jugados_total, sets_ganados_home=1, sets_ganados_away=0,
        juegos_home=5, juegos_away=2, sets_a_ganar=2,
    )
    assert total_min == 17
    assert total_min < 22

    assert ga.estado_linea(21.5, "OVER", juegos_jugados_total, total_min,
                            total_max_set_actual) == "VIVO"
    assert ga.estado_linea(15.5, "OVER", juegos_jugados_total, total_min,
                            total_max_set_actual) == "RESUELTO"


def test_181_13_caso_nally_kessler_direccion_contradice_banner_es_incoherente():
    """Caso exacto §3.B.1: dirección apostada OVER pero banner dice CONFIRMAR UNDER.

    Esta fila nunca debió salir como pick — debe caer en INCOHERENTE con el
    motivo explícito, no en OK (D181-13 condición 1, primera en el orden).
    """
    estado, motivo = rc.evaluar_coherencia_fila(
        direccion="OVER", banner_direccion="CONFIRMAR UNDER",
    )
    assert estado == "INCOHERENTE"
    assert motivo == rc.MOTIVO_DIRECCION_CONTRADICE_BANNER


def test_181_13_direccion_alineada_con_banner_no_dispara_por_si_sola():
    """Cuando dirección y banner coinciden, la condición 1 no dispara."""
    estado, motivo = rc.evaluar_coherencia_fila(
        direccion="UNDER", banner_direccion="CONFIRMAR UNDER",
    )
    assert (estado, motivo) == ("OK", "")


def test_181_13_edge_negativo_es_incoherente():
    """Condición 2: p_propia vs 1/cuota da edge negativo -> INCOHERENTE."""
    estado, motivo = rc.evaluar_coherencia_fila(edge_pct=-44.0)
    assert estado == "INCOHERENTE"
    assert motivo == rc.MOTIVO_EDGE_INSUFICIENTE


def test_181_13_zona_inalcanzable_usa_techo_real_de_d181_12():
    """Condición 3: zona recomendada por debajo del mínimo real reusa
    ga.total_alcanzable() en vez de reimplementar la aritmética (REGLA-T53).

    Partido 6:4 6:5 (2 sets a 0 ya, set en curso con ventaja) — el mínimo
    absoluto de juegos ya deja fuera de rango una zona baja mal calculada.
    """
    juegos_jugados_total = 21  # set1 6:4=10 + set2 en curso 6:5=11
    total_min, total_max = ga.total_alcanzable(
        juegos_jugados_total, sets_ganados_home=1, sets_ganados_away=0,
        juegos_home=6, juegos_away=5, sets_a_ganar=2,
    )
    estado, motivo = rc.evaluar_coherencia_fila(zona_lo=100.0, total_max=total_max)
    assert estado == "INCOHERENTE"
    assert motivo == rc.MOTIVO_ZONA_INALCANZABLE


def test_181_13_etiqueta_no_corresponde_al_numero_es_incoherente():
    """Condición 4: §3.B.1 muestra "2/5 ALTA" — 2/5 corresponde a MEDIA, no ALTA."""
    estado, motivo = rc.evaluar_coherencia_fila(
        label_cualitativa="ALTA", score_num=2, score_max=5,
    )
    assert estado == "INCOHERENTE"
    assert motivo == rc.MOTIVO_ETIQUETA_NO_CORRESPONDE


def test_181_13_sin_datos_suficientes_falla_cerrado():
    """Fail-closed: fila sin ningún dato verificable se trata como INCOHERENTE."""
    estado, motivo = rc.evaluar_coherencia_fila()
    assert estado == "INCOHERENTE"
    assert motivo == rc.MOTIVO_SIN_DATOS


def test_181_14_explicacion_contiene_numero_derivado_del_marcador_en_vivo():
    """Tripwire D181-14: la explicación debe traer al menos un número que
    venga del marcador en vivo (games_played), no solo reformular la apuesta
    ("UNDER 26.5 juegos") ni quedarse en una estimación aislada del modelo.
    """
    from live_desk import _construir_explicacion_plana
    texto = _construir_explicacion_plana(
        "UNDER", 26.5, games_played=23, max_remaining=3,
        total_estimado=25.4, p_modelo=None, cuota_mercado=None,
    )
    assert "23" in texto  # games_played real, no fabricado


def test_181_14_explicacion_contrasta_modelo_vs_mercado_cuando_hay_cuota():
    """D181-14 parte 3: número del modelo con su contraste (1/cuota), no
    "el modelo estima X" aislado — eso solo reformula, no contrasta.
    """
    from live_desk import _construir_explicacion_plana
    texto = _construir_explicacion_plana(
        "OVER", 21.5, games_played=16, max_remaining=None,
        total_estimado=None, p_modelo=0.12, cuota_mercado=1.79,
    )
    assert "12.0%" in texto
    assert "mercado implica" in texto

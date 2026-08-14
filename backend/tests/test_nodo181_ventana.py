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
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.lead_time_report as ltr
import core.fire_guard as fire_guard
import core.fire_ledger as fire_ledger
import core.games_arithmetic as ga
import core.row_coherence as rc
import core.p_wave as pw
import live_desk
import shadow_book


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


def test_181_04_mov_acumulado_sobre_15pct_no_detecta_onda_p():
    """Criterio 2 de D181-03: si el movimiento dentro de la ventana ya supera
    15%, es onda S consumada, no el onset — detectar_onda_p NO debe marcar
    detectada aunque el z-score de velocidad sea alto."""
    serie = [
        {"ts": "10:00", "cuota": 2.20, "games_played": 5},
        {"ts": "10:05", "cuota": 2.15, "games_played": 6},
        {"ts": "10:10", "cuota": 2.10, "games_played": 7},
        {"ts": "10:15", "cuota": 2.05, "games_played": 8},
        {"ts": "10:20", "cuota": 1.50, "games_played": 9},
    ]
    r = pw.detectar_onda_p(serie, linea=26.5, direccion="UNDER", z_min=2.0, n_min=4)
    assert r["magnitud_acumulada_pct"] > 15.0
    assert r["detectada"] is False


def test_181_05_games_played_ya_avanzado_no_detecta_onda_p():
    """Criterio 3 de D181-03: si games_played del ultimo punto ya alcanzo
    linea*0.6, ya no queda suficiente partido para que valga la pena
    capturar el onset — detectar_onda_p NO debe marcar detectada."""
    serie = [
        {"ts": "10:00", "cuota": 2.20, "games_played": 5},
        {"ts": "10:05", "cuota": 2.15, "games_played": 6},
        {"ts": "10:10", "cuota": 2.10, "games_played": 7},
        {"ts": "10:15", "cuota": 2.05, "games_played": 8},
        {"ts": "10:20", "cuota": 1.85, "games_played": 9},
    ]
    linea = 15.0  # umbral = 9.0, games_played final (9) ya lo alcanza
    r = pw.detectar_onda_p(serie, linea=linea, direccion="UNDER", z_min=2.0, n_min=4)
    assert serie[-1]["games_played"] >= linea * 0.6
    assert r["detectada"] is False


def test_181_06_onset_temprano_con_z_alto_si_detecta_onda_p():
    """Con z alto, mov acumulado <15% y games_played lejos del umbral de
    linea*0.6, detectar_onda_p SI debe marcar detectada y confirmar
    direccion_implicita == direccion pedida."""
    serie = [
        {"ts": "10:00", "cuota": 2.20, "games_played": 5},
        {"ts": "10:05", "cuota": 2.15, "games_played": 6},
        {"ts": "10:10", "cuota": 2.10, "games_played": 7},
        {"ts": "10:15", "cuota": 2.05, "games_played": 8},
        {"ts": "10:20", "cuota": 1.85, "games_played": 9},
    ]
    r = pw.detectar_onda_p(serie, linea=26.5, direccion="UNDER", z_min=2.0, n_min=4)
    assert r["detectada"] is True
    assert r["direccion_implicita"] == "UNDER"
    assert r["magnitud_acumulada_pct"] < 15.0
    assert serie[-1]["games_played"] < 26.5 * 0.6


def test_181_07_tres_sensores_misma_familia_no_da_quorum():
    """Razon de ser de D181-04 (§1.5 punto 1): tres sensores activos pero
    todos de la familia MERCADO cuentan como UNA sola familia independiente
    — no deben poder simular quorum entre si (n_familias==1, quorum_ok False)."""
    senal = {
        "p_wave_detectada": True,
        "steam_confirmado": True,
        "drift_pct": 8.0,
    }
    r = pw.quorum_sensores(senal)
    assert r["n_familias"] == 1
    assert r["familias_activas"] == ["MERCADO"]
    assert r["quorum_ok"] is False


def test_181_08_un_sensor_por_familia_da_quorum_ok():
    """Un sensor activo en cada una de las tres familias independientes
    (MERCADO/MODELO/ESTADO) debe dar quorum_ok True — la barra minima
    exigida por D181-04 es >=2 familias distintas."""
    senal = {
        "p_wave_detectada": True,
        "mc_p_condicional": 0.62,
        "break_situation": True,
    }
    r = pw.quorum_sensores(senal)
    assert r["n_familias"] == 3
    assert set(r["familias_activas"]) == {"MERCADO", "MODELO", "ESTADO"}
    assert r["quorum_ok"] is True


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


def test_181_11_gate_accion_apagado_no_produce_stake_ni_marca_oportunidad_real():
    """D181-07: con _P_VENTANA_ACCION_ENABLED=False (su valor por defecto,
    nace apagado), una señal que cumple todos los criterios de ACCIÓN se
    calcula y se muestra pero con etiqueta SIMULADO — nunca badge OPORTUNIDAD
    en rojo con stake real. fire_ledger.registrar_disparo no envía Telegram
    ni genera cupón por sí solo (best-effort, D181-02); este test verifica
    que el panel P_VENTANA tampoco ofrece un stake sugerido bajo el gate."""
    assert live_desk._P_VENTANA_ACCION_ENABLED is False

    serie = [
        {"ts": "10:00", "cuota": 2.20, "games_played": 5},
        {"ts": "10:05", "cuota": 2.15, "games_played": 6},
        {"ts": "10:10", "cuota": 2.10, "games_played": 7},
        {"ts": "10:15", "cuota": 2.05, "games_played": 8},
        {"ts": "10:20", "cuota": 1.85, "games_played": 9},
    ]
    fecha_compact = "20991227"  # fecha sintética que no colisiona con datos reales
    odds_path = live_desk.REPORTS / f"games_odds_history_{fecha_compact}.json"
    fired_path = live_desk.REPORTS / f"ventana_fired_{fecha_compact}.json"
    ledger_path = live_desk.REPORTS / f"fire_ledger_{fecha_compact}.jsonl"
    lead_time_path = live_desk.REPORTS / f"lead_time_report_{fecha_compact}.json"
    assert not odds_path.exists(), "fecha sintética debe estar libre"
    assert not fired_path.exists(), "fecha sintética debe estar libre"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert not lead_time_path.exists(), "fecha sintética debe estar libre"
    try:
        odds_path.write_text(json.dumps({"PartidoY_UNDER": serie}), encoding="utf-8")
        # Ventana calibrada propia (n_medidos>=20, mediana>3min) para que la señal
        # realmente cumpla los criterios de ACCIÓN dentro de este test — sin esto,
        # _estimar_ventana_restante cae al lead_time_report real más reciente del
        # repo, que puede no cumplir el umbral, y el nivel nunca llega a ACCIÓN.
        lead_time_path.write_text(json.dumps({
            "agregados": {"n_medidos": 25, "ventana_min_mediana": 6.5},
        }), encoding="utf-8")
        state = {
            "fecha": "2099-12-27", "ts": "x",
            "p4_risk": {"governor_code": 0, "kgr_sesion": 1.0},
            "p1_tape": {}, "p2_break": {"breaks": []}, "p3_convergence": {"picks": []},
            "p5_execution": {}, "p6_pnl": {}, "p7_clock": {}, "p8_books": {},
            "p9_que_falta": {}, "p10_odds_history": {}, "p11_combo_live": {},
            "p12_conformal": {}, "p_drift": {}, "p_memoria": {}, "p_f8": {},
            "p_data": {}, "p_evaluar_games": {}, "p0_ncal": {},
            "p_games": {"signals": [{
                "partido": "PartidoY", "direccion": "UNDER", "linea": 26.5,
                "games_played": 9, "max_remaining": 12, "total_estimado": 22.0,
                "cuota_actual": 1.85, "cuota_live": 1.85, "cuota": 2.20,
                "steam_confirmado": True, "drift_pct": 15.0,
                "mc_p_condicional": 0.62, "certeza": {"p_condicional": 0.7},
                "edge_pct": 8.0, "games_set1": 6, "score_data": True,
            }]},
        }
        html = live_desk.render_html(state)
        assert "SIMULADO" in html
        assert "por confirmar" not in html  # stake real solo con el gate encendido
        assert "OPORTUNIDAD" not in html or "SIMULADO" in html
        assert fired_path.exists(), "D181-08 necesita el disparo registrado aunque el gate esté apagado"
        assert ledger_path.exists(), "D181-08 necesita el disparo registrado aunque el gate esté apagado"
    finally:
        odds_path.unlink(missing_ok=True)
        fired_path.unlink(missing_ok=True)
        ledger_path.unlink(missing_ok=True)
        lead_time_path.unlink(missing_ok=True)


def test_181_12_ventana_restante_sin_calibrar_devuelve_none_con_n_menor_20():
    """Tripwire Nodo-181 D181-07. Si este test falla, el nivel ACCIÓN se encendió
    sin que H181-01 graduara — se está enviando dinero a una señal cuya anticipación
    nunca se demostró (§1.2: 4 de 6 disparos medidos tenían ventana = 0%).
    NO parchear el test: graduar H181-01 primero, o dejar el gate apagado."""
    fecha = "20991230"  # fecha sintética que no colisiona con datos reales
    report_path = live_desk.REPORTS / f"lead_time_report_{fecha}.json"
    assert not report_path.exists(), "fecha sintética debe estar libre"
    try:
        report_path.write_text(json.dumps({
            "agregados": {"n_medidos": 5, "ventana_min_mediana": 8.2},
        }), encoding="utf-8")
        r = live_desk._estimar_ventana_restante(fecha)
        assert r is None
    finally:
        report_path.unlink(missing_ok=True)


def test_181_12_ventana_restante_calibrada_devuelve_mediana_con_n_suficiente():
    """Con n_medidos>=20, _estimar_ventana_restante devuelve la mediana
    empírica de D181-01 (no un valor inventado ni recalculado localmente)."""
    fecha = "20991229"  # fecha sintética que no colisiona con datos reales
    report_path = live_desk.REPORTS / f"lead_time_report_{fecha}.json"
    assert not report_path.exists(), "fecha sintética debe estar libre"
    try:
        report_path.write_text(json.dumps({
            "agregados": {"n_medidos": 24, "ventana_min_mediana": 6.5},
        }), encoding="utf-8")
        r = live_desk._estimar_ventana_restante(fecha)
        assert r == 6.5
    finally:
        report_path.unlink(missing_ok=True)


def test_181_13_calcular_stats_ventana_h181_cruza_fire_ledger_con_odds_y_certeza():
    """D181-08: con un disparo VENTANA sintético + su serie de cuotas + su
    disparo CERTEZA emparejado, calcular_stats_ventana_h181 debe contar 1 hit
    en H181-01 (mov_despues_pct>=5% A_FAVOR) y 1 hit en H181-02
    (ts_onda_p < ts_certeza) — invoca la función real, no reimplementa
    procesar_disparo (REGLA-T53)."""
    fecha_compact = "20991225"  # fecha sintética que no colisiona con datos reales
    ledger_path = live_desk.REPORTS / f"fire_ledger_{fecha_compact}.jsonl"
    odds_path = live_desk.REPORTS / f"games_odds_history_{fecha_compact}.json"
    certeza_path = live_desk.REPORTS / f"certeza_fired_{fecha_compact}.json"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert not odds_path.exists(), "fecha sintética debe estar libre"
    assert not certeza_path.exists(), "fecha sintética debe estar libre"
    try:
        entrada = {
            "ts_iso": "2099-12-25T10:07:00", "clave": "PartidoZ_UNDER", "tipo": "VENTANA",
            "cuota": 2.10, "linea": 26.5, "games_played": 8,
            "contexto": {"ts_onda_p": "10:05", "n_familias": 3, "direccion": "UNDER"},
        }
        ledger_path.write_text(json.dumps(entrada, ensure_ascii=False) + "\n", encoding="utf-8")
        odds_path.write_text(json.dumps({
            "PartidoZ_UNDER": [
                {"ts": "10:00", "cuota": 2.20, "games_played": 5},
                {"ts": "10:05", "cuota": 2.10, "games_played": 8},
                {"ts": "10:20", "cuota": 1.80, "games_played": 10},
            ],
        }), encoding="utf-8")
        certeza_path.write_text(json.dumps({
            "PartidoZ_UNDER": {"ts": "2099-12-25T10:15:00", "direccion": "UNDER"},
        }), encoding="utf-8")

        r = shadow_book.calcular_stats_ventana_h181("2099-12-25", "2099-12-25")
        assert r["H181-01"] == {"n": 1, "hits": 1}
        assert r["H181-02"] == {"n": 1, "hits": 1}
    finally:
        ledger_path.unlink(missing_ok=True)
        odds_path.unlink(missing_ok=True)
        certeza_path.unlink(missing_ok=True)


def test_181_14_calcular_stats_ventana_h181_sin_disparos_devuelve_ceros():
    """Rango sin fire_ledger VENTANA -> n=0 en ambas hipótesis, no un error
    ni un valor inventado."""
    r = shadow_book.calcular_stats_ventana_h181("2099-12-24", "2099-12-24")
    assert r == {"H181-01": {"n": 0, "hits": 0}, "H181-02": {"n": 0, "hits": 0}}


def test_181_15_shadow_book_report_incluye_segmento_ventana_h181():
    """El segmento VENTANA H181 debe aparecer en el texto de shadow_book.report()
    y marcar explícitamente H181-03 como NO MEDIBLE (D181-08) — nunca inventar
    un número para la cohorte de quorum<=2 que no existe."""
    out = shadow_book.report("2026-08-01", "2026-08-14")
    assert "VENTANA H181" in out
    assert "H181-01" in out and "H181-02" in out
    assert "NO MEDIBLE" in out


def test_181_16_procesar_alerta_a_apuesta_medido_calcula_minutos_friccion():
    """D181-11: un disparo VENTANA real (fire_ledger) con una apuesta registrada
    (reports/apuestas_*.json, betslip_registrar.py D87-09) del mismo partido
    y con ts_registro posterior al disparo debe matchear MEDIDO con los
    minutos de fricción reales entre alerta y apuesta."""
    fecha_compact = "20991226"  # fecha sintética que no colisiona con datos reales
    ledger_path = ltr.REPORTS_DIR / f"fire_ledger_{fecha_compact}.jsonl"
    apuestas_path = ltr.REPORTS_DIR / f"apuestas_{fecha_compact}_120500.json"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert not apuestas_path.exists(), "fecha sintética debe estar libre"
    try:
        fire_ledger.registrar_disparo(fecha_compact, "PartidoZ_OVER", "VENTANA", cuota_al_disparo=1.90)
        entrada = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
        ts_alerta = datetime.fromisoformat(entrada["ts_iso"])
        ts_registro = ts_alerta + timedelta(minutes=5)
        apuestas_path.write_text(json.dumps({
            "ts_registro": ts_registro.isoformat(),
            "picks": [{"partido": "PartidoZ", "outcome_id": 999}],
        }), encoding="utf-8")

        resultados = ltr.procesar_alerta_a_apuesta(fecha_compact)
        assert len(resultados) == 1
        assert resultados[0]["categoria"] == "MEDIDO"
        assert resultados[0]["minutos_friccion"] == 5.0

        agregados = ltr.calcular_agregados_friccion(resultados)
        assert agregados["n_total"] == 1
        assert agregados["n_medidos"] == 1
        assert agregados["minutos_friccion_mediana"] == 5.0
    finally:
        ledger_path.unlink(missing_ok=True)
        apuestas_path.unlink(missing_ok=True)


def test_181_17_procesar_alerta_a_apuesta_sin_match_devuelve_sin_apuesta_registrada():
    """D181-11: un disparo VENTANA sin ninguna apuesta registrada del mismo
    partido (usuario no confirmó la apuesta, o la registró para otro partido)
    debe categorizarse SIN_APUESTA_REGISTRADA — nunca inventar minutos_friccion."""
    fecha_compact = "20991228"  # fecha sintética que no colisiona con datos reales
    ledger_path = ltr.REPORTS_DIR / f"fire_ledger_{fecha_compact}.jsonl"
    apuestas_path = ltr.REPORTS_DIR / f"apuestas_{fecha_compact}_120500.json"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert not apuestas_path.exists(), "fecha sintética debe estar libre"
    try:
        fire_ledger.registrar_disparo(fecha_compact, "PartidoW_UNDER", "VENTANA", cuota_al_disparo=1.80)
        apuestas_path.write_text(json.dumps({
            "ts_registro": datetime.now().isoformat(),
            "picks": [{"partido": "OtroPartido", "outcome_id": 111}],
        }), encoding="utf-8")

        resultados = ltr.procesar_alerta_a_apuesta(fecha_compact)
        assert len(resultados) == 1
        assert resultados[0]["categoria"] == "SIN_APUESTA_REGISTRADA"
        assert "minutos_friccion" not in resultados[0]

        agregados = ltr.calcular_agregados_friccion(resultados)
        assert agregados["n_total"] == 1
        assert agregados["n_medidos"] == 0
        assert agregados["n_sin_apuesta_registrada"] == 1
        assert agregados["minutos_friccion_mediana"] is None
    finally:
        ledger_path.unlink(missing_ok=True)
        apuestas_path.unlink(missing_ok=True)


def test_181_18_procesar_alerta_a_apuesta_sin_fire_ledger_devuelve_lista_vacia():
    """D181-11: si no hay fire_ledger_{fecha}.jsonl para esa fecha (nunca se
    disparó ninguna VENTANA ese día), procesar_alerta_a_apuesta no debe
    fallar ni inventar disparos — devuelve []."""
    fecha_compact = "20991201"  # fecha sintética que no colisiona con datos reales
    ledger_path = ltr.REPORTS_DIR / f"fire_ledger_{fecha_compact}.jsonl"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert ltr.procesar_alerta_a_apuesta(fecha_compact) == []


def test_181_19_nota_operativa_sesion_caliente_aparece_en_fila_accion():
    """D181-11: cuando el nivel llega a ACCIÓN, el panel P_VENTANA debe incluir
    la nota operativa de sesión caliente ('confirma que estás logueado en
    Betplay antes de abrir') en el texto de la fila — vive ahí ahora, dormida
    bajo D181-07, y viajará automáticamente por Telegram el día que D181-06
    se encienda, sin necesitar un dispatcher nuevo."""
    assert live_desk._P_VENTANA_ACCION_ENABLED is False

    serie = [
        {"ts": "10:00", "cuota": 2.20, "games_played": 5},
        {"ts": "10:05", "cuota": 2.15, "games_played": 6},
        {"ts": "10:10", "cuota": 2.10, "games_played": 7},
        {"ts": "10:15", "cuota": 2.05, "games_played": 8},
        {"ts": "10:20", "cuota": 1.85, "games_played": 9},
    ]
    fecha_compact = "20991205"  # fecha sintética que no colisiona con datos reales
    odds_path = live_desk.REPORTS / f"games_odds_history_{fecha_compact}.json"
    fired_path = live_desk.REPORTS / f"ventana_fired_{fecha_compact}.json"
    ledger_path = live_desk.REPORTS / f"fire_ledger_{fecha_compact}.jsonl"
    lead_time_path = live_desk.REPORTS / f"lead_time_report_{fecha_compact}.json"
    assert not odds_path.exists(), "fecha sintética debe estar libre"
    assert not fired_path.exists(), "fecha sintética debe estar libre"
    assert not ledger_path.exists(), "fecha sintética debe estar libre"
    assert not lead_time_path.exists(), "fecha sintética debe estar libre"
    try:
        odds_path.write_text(json.dumps({"PartidoQ_UNDER": serie}), encoding="utf-8")
        lead_time_path.write_text(json.dumps({
            "agregados": {"n_medidos": 25, "ventana_min_mediana": 6.5},
        }), encoding="utf-8")
        state = {
            "fecha": "2099-12-05", "ts": "x",
            "p4_risk": {"governor_code": 0, "kgr_sesion": 1.0},
            "p1_tape": {}, "p2_break": {"breaks": []}, "p3_convergence": {"picks": []},
            "p5_execution": {}, "p6_pnl": {}, "p7_clock": {}, "p8_books": {},
            "p9_que_falta": {}, "p10_odds_history": {}, "p11_combo_live": {},
            "p12_conformal": {}, "p_drift": {}, "p_memoria": {}, "p_f8": {},
            "p_data": {}, "p_evaluar_games": {}, "p0_ncal": {},
            "p_games": {"signals": [{
                "partido": "PartidoQ", "direccion": "UNDER", "linea": 26.5,
                "games_played": 9, "max_remaining": 12, "total_estimado": 22.0,
                "cuota_actual": 1.85, "cuota_live": 1.85, "cuota": 2.20,
                "steam_confirmado": True, "drift_pct": 15.0,
                "mc_p_condicional": 0.62, "certeza": {"p_condicional": 0.7},
                "edge_pct": 8.0, "games_set1": 6, "score_data": True,
            }]},
        }
        html = live_desk.render_html(state)
        assert "confirma que estás logueado en Betplay antes de abrir" in html
        assert "SIMULADO" in html
    finally:
        odds_path.unlink(missing_ok=True)
        fired_path.unlink(missing_ok=True)
        ledger_path.unlink(missing_ok=True)
        lead_time_path.unlink(missing_ok=True)

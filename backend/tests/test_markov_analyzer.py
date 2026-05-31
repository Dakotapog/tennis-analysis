"""
Tests para analysis/markov_analyzer.py (Nodo-02)
Cubre: PELT change-point, estados HOT/COLD/NEUTRAL, factor Markov, extracción binaria.
"""
import pytest
from analysis.markov_analyzer import (
    detectar_cambio_regimen,
    calcular_factor_markov,
    extraer_resultados_binarios,
)


# ─────────────────────────────────────────────────────────────────────────────
# detectar_cambio_regimen — estados HOT / COLD / NEUTRAL
# ─────────────────────────────────────────────────────────────────────────────

class TestEstados:

    def test_jugador_hot_detectado(self):
        """Mejora clara al final → HOT, momentum positivo."""
        resultados = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        r = detectar_cambio_regimen(resultados)
        assert r['estado_actual'] == 'HOT'
        assert r['momentum'] > 0

    def test_jugador_cold_detectado(self):
        """Declive claro al final → COLD, momentum negativo."""
        resultados = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r = detectar_cambio_regimen(resultados)
        assert r['estado_actual'] == 'COLD'
        assert r['momentum'] < 0

    def test_jugador_neutral(self):
        """Win rate uniforme 50% → NEUTRAL."""
        resultados = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        r = detectar_cambio_regimen(resultados)
        assert r['estado_actual'] == 'NEUTRAL'

    def test_ultimos_5_determinan_estado(self):
        """Los últimos 5 partidos determinan HOT/COLD, no el historial global."""
        # 10 derrotas seguidas de 5 victorias → HOT por últimos 5
        resultados = [0] * 10 + [1] * 5
        r = detectar_cambio_regimen(resultados)
        assert r['estado_actual'] == 'HOT'

    def test_todos_victorias_hot(self):
        """100% victorias → HOT."""
        r = detectar_cambio_regimen([1] * 15)
        assert r['estado_actual'] == 'HOT'
        assert r['win_rate_reciente'] == 1.0

    def test_todos_derrotas_cold(self):
        """100% derrotas → COLD."""
        r = detectar_cambio_regimen([0] * 15)
        assert r['estado_actual'] == 'COLD'
        assert r['win_rate_reciente'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# detectar_cambio_regimen — PELT changepoint
# ─────────────────────────────────────────────────────────────────────────────

class TestPELT:

    def test_changepoint_detectado(self):
        """PELT encuentra el punto de cambio en una secuencia con cambio obvio."""
        # Primeros 8: 0% victorias | Últimos 8: 100% victorias
        resultados = [0] * 8 + [1] * 8
        r = detectar_cambio_regimen(resultados, min_size=4)
        assert r['change_point'] is not None
        # El changepoint debe estar cerca del índice 8
        assert 4 <= r['change_point'] <= 12

    def test_sin_changepoint_secuencia_uniforme(self):
        """Secuencia uniforme → sin changepoint significativo."""
        resultados = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        r = detectar_cambio_regimen(resultados, umbral_cambio=0.20)
        # Secuencia alternada sin cambio claro → change_point puede ser None
        # o la confianza es baja
        if r['change_point'] is not None:
            assert r['confianza'] < 0.50

    def test_confianza_alta_cambio_brusco(self):
        """Cambio brusco (0→1) produce confianza alta."""
        resultados = [0] * 10 + [1] * 10
        r = detectar_cambio_regimen(resultados)
        assert r['confianza'] >= 0.80

    def test_confianza_baja_secuencia_alternada(self):
        """Secuencia perfectamente alternada (50-50 estable) → sin changepoint claro."""
        # [1,0,1,0,1,0,1,0,1,0,1,0] — ningún split produce diferencia grande
        resultados = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        r = detectar_cambio_regimen(resultados)
        assert r['confianza'] < 0.60

    def test_datos_insuficientes_neutral(self):
        """Con menos de 2×min_size datos → NEUTRAL y sin changepoint."""
        r = detectar_cambio_regimen([1, 0, 1], min_size=5)
        assert r['estado_actual'] == 'NEUTRAL'
        assert r['change_point'] is None
        assert r['confianza'] == 0.0

    def test_lista_vacia_neutral(self):
        """Lista vacía → NEUTRAL con zeros."""
        r = detectar_cambio_regimen([])
        assert r['estado_actual'] == 'NEUTRAL'
        assert r['momentum'] == 0.0

    def test_momentum_positivo_cuando_mejora(self):
        """Momentum > 0 cuando la segunda mitad tiene más victorias."""
        resultados = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        r = detectar_cambio_regimen(resultados)
        assert r['momentum'] > 0

    def test_momentum_negativo_cuando_empeora(self):
        """Momentum < 0 cuando la segunda mitad tiene menos victorias."""
        resultados = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        r = detectar_cambio_regimen(resultados)
        assert r['momentum'] < 0

    def test_win_rate_reciente_mayor_cuando_hot(self):
        """win_rate_reciente > win_rate_anterior cuando el jugador mejora."""
        resultados = [0] * 8 + [1] * 8
        r = detectar_cambio_regimen(resultados)
        assert r['win_rate_reciente'] > r['win_rate_anterior']


# ─────────────────────────────────────────────────────────────────────────────
# calcular_factor_markov
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorMarkov:

    def test_hot_vs_cold_mayor_que_1(self):
        """P1 HOT vs P2 COLD → factor > 1 (P1 amplificado)."""
        hot  = {'estado_actual': 'HOT'}
        cold = {'estado_actual': 'COLD'}
        assert calcular_factor_markov(hot, cold) > 1.0

    def test_cold_vs_hot_menor_que_1(self):
        """P1 COLD vs P2 HOT → factor < 1 (P1 reducido)."""
        cold = {'estado_actual': 'COLD'}
        hot  = {'estado_actual': 'HOT'}
        assert calcular_factor_markov(cold, hot) < 1.0

    def test_neutral_vs_neutral_es_1(self):
        """P1 NEUTRAL vs P2 NEUTRAL → factor = 1.0."""
        neutral = {'estado_actual': 'NEUTRAL'}
        assert calcular_factor_markov(neutral, neutral) == 1.0

    def test_hot_vs_hot_es_1(self):
        """Ambos HOT → empate de momentum → factor = 1.0."""
        hot = {'estado_actual': 'HOT'}
        assert calcular_factor_markov(hot, hot) == 1.0

    def test_cold_vs_cold_es_1(self):
        """Ambos COLD → factor = 1.0."""
        cold = {'estado_actual': 'COLD'}
        assert calcular_factor_markov(cold, cold) == 1.0

    def test_simetria_inversa(self):
        """factor(P1, P2) × factor(P2, P1) ≈ 1 (simétrico)."""
        hot  = {'estado_actual': 'HOT'}
        cold = {'estado_actual': 'COLD'}
        f1 = calcular_factor_markov(hot, cold)
        f2 = calcular_factor_markov(cold, hot)
        assert abs(f1 * f2 - 1.0) < 0.05

    def test_factor_rango_valido(self):
        """Factor siempre en [0.85, 1.15]."""
        estados = ['HOT', 'NEUTRAL', 'COLD']
        for e1 in estados:
            for e2 in estados:
                f = calcular_factor_markov({'estado_actual': e1}, {'estado_actual': e2})
                assert 0.84 <= f <= 1.16, f"factor={f} fuera de rango para ({e1},{e2})"

    def test_hot_vs_neutral_entre_1_y_1_15(self):
        """P1 HOT vs P2 NEUTRAL → factor entre 1.0 y 1.15."""
        hot     = {'estado_actual': 'HOT'}
        neutral = {'estado_actual': 'NEUTRAL'}
        f = calcular_factor_markov(hot, neutral)
        assert 1.0 < f <= 1.15

    def test_factor_maximo_hot_vs_cold(self):
        """El factor máximo (1.15) ocurre cuando P1=HOT y P2=COLD."""
        hot  = {'estado_actual': 'HOT'}
        cold = {'estado_actual': 'COLD'}
        assert calcular_factor_markov(hot, cold) == 1.15

    def test_factor_minimo_cold_vs_hot(self):
        """El factor mínimo (0.85) ocurre cuando P1=COLD y P2=HOT."""
        cold = {'estado_actual': 'COLD'}
        hot  = {'estado_actual': 'HOT'}
        assert calcular_factor_markov(cold, hot) == 0.85


# ─────────────────────────────────────────────────────────────────────────────
# extraer_resultados_binarios
# ─────────────────────────────────────────────────────────────────────────────

class TestExtraerResultados:

    def test_outcome_gano_es_1(self):
        history = [{'outcome': 'ganó'}, {'outcome': 'perdió'}]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        assert r == [0, 1]  # reversed → perdió primero (más viejo), ganó después

    def test_outcome_win_es_1(self):
        history = [{'outcome': 'win'}, {'outcome': 'win'}, {'outcome': 'loss'}]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        assert r == [0, 1, 1]  # reversed

    def test_outcome_perdio_es_0(self):
        history = [{'outcome': 'perdió'}]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        assert r == [0]

    def test_ret_en_resultado_es_1(self):
        """RET en resultado → victoria para el jugador objetivo."""
        history = [{'outcome': '', 'resultado': '2-0 RET'}]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        assert r == [1]

    def test_wo_en_resultado_es_1(self):
        """WO en resultado → victoria."""
        history = [{'outcome': '', 'resultado': 'WO'}]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        assert r == [1]

    def test_historial_vacio(self):
        assert extraer_resultados_binarios([], 'Jugador A') == []

    def test_respeta_limite_n(self):
        """Solo toma los primeros n del historial (más recientes)."""
        history = [{'outcome': 'ganó'}] * 30
        r = extraer_resultados_binarios(history, 'Jugador A', n=10)
        assert len(r) == 10

    def test_orden_cronologico(self):
        """El resultado más viejo debe ser el primero (reversed)."""
        # newest-first: [win, loss, win] → reversed (oldest-first): [win, loss, win]
        history = [
            {'outcome': 'ganó'},   # más reciente
            {'outcome': 'perdió'}, # medio
            {'outcome': 'ganó'},   # más viejo
        ]
        r = extraer_resultados_binarios(history, 'Jugador A', n=3)
        assert r == [1, 0, 1]  # oldest-first: ganó(viejo), perdió(medio), ganó(reciente)

    def test_sin_outcome_claro_no_incluye(self):
        """Partidos sin outcome ni RET/WO se omiten para no introducir ruido."""
        history = [
            {'outcome': 'ganó'},
            {'outcome': '', 'resultado': 'desconocido'},  # sin info → omitir
            {'outcome': 'perdió'},
        ]
        r = extraer_resultados_binarios(history, 'Jugador A', n=5)
        # Solo 2 resultados claros
        assert len(r) == 2
        assert set(r) == {0, 1}


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN — pipeline completo
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineCompleto:

    def test_jugador_hot_vs_cold_amplifica_form(self):
        """
        Si P1 está HOT y P2 COLD, el factor Markov debe amplificar
        el form_recent de P1 por encima del de P2.
        """
        # Simular un form_recent base igual para ambos
        form_base = 150.0
        hot  = {'estado_actual': 'HOT'}
        cold = {'estado_actual': 'COLD'}

        factor_p1 = calcular_factor_markov(hot, cold)
        factor_p2 = calcular_factor_markov(cold, hot)

        form_p1_ajustado = form_base * factor_p1
        form_p2_ajustado = form_base * factor_p2

        assert form_p1_ajustado > form_p2_ajustado

    def test_ambos_neutral_no_cambia_nada(self):
        """Ambos NEUTRAL → factor = 1.0 → form_recent no cambia."""
        form_base = 150.0
        neutral = {'estado_actual': 'NEUTRAL'}
        factor = calcular_factor_markov(neutral, neutral)
        assert form_base * factor == form_base

    def test_secuencia_real_roland_garros(self):
        """
        Simulación de un jugador con mala racha que mejora en arcilla.
        Debe detectarse el changepoint y retornar HOT al final.
        """
        # 10 partidos perdidos seguidos → 8 victorias en Roland Garros
        resultados = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        r = detectar_cambio_regimen(resultados)
        assert r['estado_actual'] == 'HOT'
        assert r['momentum'] > 0
        assert r['change_point'] is not None

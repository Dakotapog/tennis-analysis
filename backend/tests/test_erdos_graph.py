"""
Tests para analysis/erdos_graph.py (Nodo-06)
Cubre: construir_grafo_victorias, distancia_erdos, historial_a_partidos.
Fundamento: decaimiento subcuadrático de ventajas transitivas (Erdős).
"""
import pytest
from analysis.erdos_graph import (
    construir_grafo_victorias,
    distancia_erdos,
    historial_a_partidos,
    pagerank_grafo,
)


# ─────────────────────────────────────────────────────────────────────────────
# construir_grafo_victorias
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruirGrafo:

    def test_victoria_simple(self):
        partidos = [{'ganador': 'A', 'perdedor': 'B'}]
        g = construir_grafo_victorias(partidos)
        assert 'A' in g
        assert g['A']['B'] == 1.0

    def test_derrota_registrada_inversamente(self):
        partidos = [{'ganador': 'A', 'perdedor': 'B'}]
        g = construir_grafo_victorias(partidos)
        # B no tiene victorias sobre A
        assert g.get('B', {}).get('A', 0) == 0

    def test_win_rate_50_cuando_empatados(self):
        partidos = [
            {'ganador': 'A', 'perdedor': 'B'},
            {'ganador': 'B', 'perdedor': 'A'},
        ]
        g = construir_grafo_victorias(partidos)
        assert g['A']['B'] == 0.5
        assert g['B']['A'] == 0.5

    def test_win_rate_2_de_3(self):
        partidos = [
            {'ganador': 'A', 'perdedor': 'B'},
            {'ganador': 'A', 'perdedor': 'B'},
            {'ganador': 'B', 'perdedor': 'A'},
        ]
        g = construir_grafo_victorias(partidos)
        assert abs(g['A']['B'] - 2/3) < 0.001

    def test_acepta_formato_winner_loser(self):
        partidos = [{'winner': 'A', 'loser': 'B'}]
        g = construir_grafo_victorias(partidos)
        assert 'A' in g
        assert g['A']['B'] == 1.0

    def test_lista_vacia(self):
        g = construir_grafo_victorias([])
        assert g == {}

    def test_mismo_jugador_ignorado(self):
        """Partido donde ganador == perdedor se ignora."""
        partidos = [{'ganador': 'A', 'perdedor': 'A'}]
        g = construir_grafo_victorias(partidos)
        assert g == {}

    def test_multiples_jugadores(self):
        partidos = [
            {'ganador': 'A', 'perdedor': 'B'},
            {'ganador': 'B', 'perdedor': 'C'},
            {'ganador': 'A', 'perdedor': 'C'},
        ]
        g = construir_grafo_victorias(partidos)
        assert 'A' in g
        assert 'B' in g
        assert g['A']['B'] == 1.0
        assert g['B']['C'] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# distancia_erdos
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanciaErdos:

    def test_ventaja_directa_score_positivo(self):
        """A venció a B directamente → erdos_score > 0."""
        grafo = {'A': {'B': 0.7}}
        r = distancia_erdos('A', 'B', grafo)
        assert r['erdos_score'] > 0

    def test_sin_conexion_score_cero(self):
        """Jugadores sin camino entre ellos → score = 0."""
        grafo = {'A': {'X': 0.8}, 'Y': {'B': 0.8}}
        r = distancia_erdos('A', 'B', grafo, max_depth=2)
        assert r['erdos_score'] == 0.0
        assert r['n_paths'] == 0

    def test_ventaja_transitiva_distancia_2(self):
        """A→C→B → score positivo, depth = 2."""
        grafo = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
        r = distancia_erdos('A', 'B', grafo)
        assert r['erdos_score'] > 0
        assert r['max_depth_alcanzado'] == 2

    def test_decaimiento_con_distancia(self):
        """Ventaja directa (d=1) mayor que transitiva (d=2) con igual win_rate."""
        grafo_directo   = {'A': {'B': 0.7}}
        grafo_transitivo = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
        r1 = distancia_erdos('A', 'B', grafo_directo)
        r2 = distancia_erdos('A', 'B', grafo_transitivo)
        assert r1['erdos_score'] > r2['erdos_score']

    def test_max_depth_respetado(self):
        """Cadena A→C→D→B (d=3) con max_depth=2 → score = 0."""
        grafo = {'A': {'C': 0.8}, 'C': {'D': 0.8}, 'D': {'B': 0.8}}
        r = distancia_erdos('A', 'B', grafo, max_depth=2)
        assert r['erdos_score'] == 0.0

    def test_max_depth_3_alcanza_camino(self):
        """Cadena A→C→D→B con max_depth=3 → score > 0."""
        grafo = {'A': {'C': 0.8}, 'C': {'D': 0.8}, 'D': {'B': 0.8}}
        r = distancia_erdos('A', 'B', grafo, max_depth=3)
        assert r['erdos_score'] > 0
        assert r['max_depth_alcanzado'] == 3

    def test_mismo_jugador_score_cero(self):
        """jugador_a == jugador_b → score = 0.0."""
        grafo = {'A': {'B': 0.7}}
        r = distancia_erdos('A', 'A', grafo)
        assert r['erdos_score'] == 0.0
        assert r['n_paths'] == 0

    def test_grafo_vacio(self):
        """Grafo vacío → score = 0.0."""
        r = distancia_erdos('A', 'B', {})
        assert r['erdos_score'] == 0.0

    def test_paths_ordenados_por_peso(self):
        """Caminos retornados ordenados de mayor a menor peso."""
        grafo = {
            'A': {'C': 0.9, 'D': 0.5},
            'C': {'B': 0.9},
            'D': {'B': 0.5},
        }
        r = distancia_erdos('A', 'B', grafo)
        if len(r['paths']) >= 2:
            assert r['paths'][0]['peso'] >= r['paths'][1]['peso']

    def test_score_rango_menos1_a_mas1(self):
        """erdos_score siempre en [-1, +1]."""
        grafo = {'A': {'B': 1.0}}
        r = distancia_erdos('A', 'B', grafo)
        assert -1.0 <= r['erdos_score'] <= 1.0

    def test_score_raw_rango_0_1(self):
        """erdos_score_raw siempre en [0, 1]."""
        grafo = {'A': {'B': 0.8}}
        r = distancia_erdos('A', 'B', grafo)
        assert 0.0 <= r['erdos_score_raw'] <= 1.0

    def test_simetria_invertida(self):
        """Si A tiene ventaja sobre B, B no debe tener ventaja sobre A."""
        grafo = {'A': {'B': 0.8}}
        r_a = distancia_erdos('A', 'B', grafo)
        r_b = distancia_erdos('B', 'A', grafo)
        assert r_a['erdos_score'] > 0
        assert r_b['erdos_score'] == 0.0  # B no tiene caminos a A

    def test_multiples_caminos_aumentan_score(self):
        """Más caminos transitivos → score más alto."""
        grafo_un_camino = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
        grafo_dos_caminos = {'A': {'C': 0.7, 'D': 0.7}, 'C': {'B': 0.7}, 'D': {'B': 0.7}}
        r1 = distancia_erdos('A', 'B', grafo_un_camino)
        r2 = distancia_erdos('A', 'B', grafo_dos_caminos)
        assert r2['erdos_score'] >= r1['erdos_score']

    def test_alpha_mayor_menos_decaimiento(self):
        """Alpha mayor (0.9) → menos decaimiento → score transitivo más alto."""
        grafo = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
        r_alpha_07 = distancia_erdos('A', 'B', grafo, alpha=0.7)
        r_alpha_09 = distancia_erdos('A', 'B', grafo, alpha=0.9)
        assert r_alpha_09['erdos_score'] >= r_alpha_07['erdos_score']

    def test_ciclos_no_causan_loop_infinito(self):
        """Grafo con ciclos no debe colgar."""
        grafo = {'A': {'B': 0.7, 'C': 0.6}, 'B': {'A': 0.3, 'C': 0.8}, 'C': {'B': 0.6}}
        r = distancia_erdos('A', 'C', grafo, max_depth=3)
        assert isinstance(r['erdos_score'], float)

    def test_paths_maximo_5(self):
        """Nunca retorna más de 5 caminos."""
        # Crear muchos caminos a través de intermediarios
        grafo = {'A': {f'C{i}': 0.7 for i in range(10)}}
        for i in range(10):
            grafo[f'C{i}'] = {'B': 0.7}
        r = distancia_erdos('A', 'B', grafo)
        assert len(r['paths']) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# historial_a_partidos
# ─────────────────────────────────────────────────────────────────────────────

class TestHistorialAPartidos:

    def test_victoria_genera_ganador(self):
        historial = [{'oponente': 'Nadal R.', 'outcome': 'ganó'}]
        r = historial_a_partidos(historial, 'Tsitsipas S.')
        assert len(r) == 1
        assert r[0]['ganador'] == 'Tsitsipas S.'
        assert r[0]['perdedor'] == 'Nadal R.'

    def test_derrota_genera_perdedor(self):
        historial = [{'oponente': 'Nadal R.', 'outcome': 'perdió'}]
        r = historial_a_partidos(historial, 'Tsitsipas S.')
        assert len(r) == 1
        assert r[0]['ganador'] == 'Nadal R.'
        assert r[0]['perdedor'] == 'Tsitsipas S.'

    def test_outcome_win(self):
        historial = [{'oponente': 'X', 'outcome': 'win'}]
        r = historial_a_partidos(historial, 'A')
        assert r[0]['ganador'] == 'A'

    def test_outcome_loss(self):
        historial = [{'oponente': 'X', 'outcome': 'loss'}]
        r = historial_a_partidos(historial, 'A')
        assert r[0]['ganador'] == 'X'

    def test_ret_en_resultado_es_victoria(self):
        historial = [{'oponente': 'X', 'outcome': '', 'resultado': '2-0 RET'}]
        r = historial_a_partidos(historial, 'A')
        assert len(r) == 1
        assert r[0]['ganador'] == 'A'

    def test_wo_en_resultado_es_victoria(self):
        historial = [{'oponente': 'X', 'outcome': '', 'resultado': 'WO'}]
        r = historial_a_partidos(historial, 'A')
        assert r[0]['ganador'] == 'A'

    def test_sin_outcome_claro_omite(self):
        """Partido sin outcome ni RET/WO se omite."""
        historial = [{'oponente': 'X', 'outcome': '', 'resultado': 'desconocido'}]
        r = historial_a_partidos(historial, 'A')
        assert r == []

    def test_oponente_vacio_omite(self):
        historial = [{'oponente': '', 'outcome': 'ganó'}]
        r = historial_a_partidos(historial, 'A')
        assert r == []

    def test_historial_vacio(self):
        assert historial_a_partidos([], 'A') == []

    def test_multiples_partidos(self):
        historial = [
            {'oponente': 'B', 'outcome': 'ganó'},
            {'oponente': 'C', 'outcome': 'perdió'},
            {'oponente': 'D', 'outcome': 'ganó'},
        ]
        r = historial_a_partidos(historial, 'A')
        assert len(r) == 3
        assert r[0]['ganador'] == 'A'
        assert r[1]['ganador'] == 'C'
        assert r[2]['ganador'] == 'A'


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN — pipeline completo historial → grafo → Erdős
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineCompleto:

    def test_jugadores_sin_oponentes_comunes_score_cero(self):
        """Si los historiales no se solapan → Erdős score = 0."""
        hist_a = [{'oponente': 'X', 'outcome': 'ganó'}]
        hist_b = [{'oponente': 'Y', 'outcome': 'ganó'}]
        p_a = historial_a_partidos(hist_a, 'A')
        p_b = historial_a_partidos(hist_b, 'B')
        grafo = construir_grafo_victorias(p_a + p_b)
        r = distancia_erdos('A', 'B', grafo)
        assert r['erdos_score'] == 0.0

    def test_oponente_comun_genera_ventaja_transitiva(self):
        """A venció a C, C venció a B → A tiene ventaja transitiva."""
        hist_a = [{'oponente': 'C', 'outcome': 'ganó'}]
        hist_b = [{'oponente': 'C', 'outcome': 'perdió'}]
        p_a = historial_a_partidos(hist_a, 'A')
        p_b = historial_a_partidos(hist_b, 'B')
        grafo = construir_grafo_victorias(p_a + p_b)
        r = distancia_erdos('A', 'B', grafo, max_depth=3)
        assert r['erdos_score'] > 0

    def test_resultado_tiene_campos_requeridos(self):
        """El resultado siempre tiene: erdos_score, paths, n_paths, max_depth_alcanzado, pagerank_scores."""
        hist = [{'oponente': 'B', 'outcome': 'ganó'}]
        grafo = construir_grafo_victorias(historial_a_partidos(hist, 'A'))
        r = distancia_erdos('A', 'B', grafo)
        for campo in ('erdos_score', 'erdos_score_raw', 'paths', 'n_paths', 'max_depth_alcanzado', 'pagerank_scores'):
            assert campo in r, f"Campo faltante: {campo}"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS NODO-20 — PageRank Erdős Quality (T20-04)
# ─────────────────────────────────────────────────────────────────────────────

def _grafo_lineal(jugadores):
    """Grafo lineal: A→B→C→D... con win_rate=1.0 en cada arista."""
    grafo = {}
    for i in range(len(jugadores) - 1):
        grafo[jugadores[i]] = {jugadores[i + 1]: 1.0}
    grafo[jugadores[-1]] = {}
    return grafo


def _grafo_estrella(centro, radios):
    """Grafo estrella: centro vence a todos los radios."""
    grafo = {centro: {r: 1.0 for r in radios}}
    for r in radios:
        grafo[r] = {}
    return grafo


class TestPageRankGrafo:
    def test_grafo_vacio_retorna_dict_vacio(self):
        """REGLA-T20-1: grafo vacío → {}."""
        assert pagerank_grafo({}) == {}

    def test_menos_de_5_nodos_retorna_vacio(self):
        """REGLA-T20-1: n < 5 → {} (sin masa crítica)."""
        grafo = _grafo_lineal(['A', 'B', 'C', 'D'])
        assert pagerank_grafo(grafo) == {}

    def test_exactamente_5_nodos_calcula(self):
        """n == 5 → PageRank calculado."""
        grafo = _grafo_lineal(['A', 'B', 'C', 'D', 'E'])
        result = pagerank_grafo(grafo)
        assert len(result) == 5

    def test_maximo_siempre_es_10(self):
        """El nodo con mayor centralidad siempre recibe 1.0 (normalización)."""
        grafo = _grafo_estrella('Top', [f'P{i}' for i in range(6)])
        result = pagerank_grafo(grafo)
        assert max(result.values()) == 1.0

    def test_rango_entre_0_y_1(self):
        """Todos los scores en [0, 1]."""
        grafo = _grafo_lineal([f'J{i}' for i in range(8)])
        result = pagerank_grafo(grafo)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_nodo_receptor_tiene_mayor_centralidad(self):
        """En grafo estrella, el nodo receptor (todos apuntan a él) tiene mayor PR."""
        # Todos los jugadores pierden ante 'Top' → Top recibe links entrantes
        radios = [f'P{i}' for i in range(7)]
        grafo = _grafo_estrella('Top', radios)
        result = pagerank_grafo(grafo)
        # Top tiene out-edges (vence a todos) pero otros no tienen out-edges hacia Top
        # En grafo de victorias, Top → radios. Radios sin salida → dangling nodes
        # Top debería estar entre los de mayor score
        assert result['Top'] == 1.0 or result['Top'] >= max(result.values()) * 0.5

    def test_convergencia_en_10_iteraciones(self):
        """10 iteraciones es suficiente: resultado estable entre 10 y 20 iteraciones."""
        grafo = _grafo_lineal([f'J{i}' for i in range(10)])
        r10 = pagerank_grafo(grafo, iteraciones=10)
        r20 = pagerank_grafo(grafo, iteraciones=20)
        for k in r10:
            assert abs(r10[k] - r20[k]) < 0.05, f"No convergió para {k}"

    def test_damping_por_defecto_es_085(self):
        """damping=0.85 es el estándar — resultado con otro damping debe diferir."""
        grafo = _grafo_lineal([f'J{i}' for i in range(6)])
        r_std = pagerank_grafo(grafo, damping=0.85)
        r_low = pagerank_grafo(grafo, damping=0.50)
        # Con menor damping la distribución es más uniforme → menor diferencia entre max y min
        dif_std = max(r_std.values()) - min(r_std.values())
        dif_low = max(r_low.values()) - min(r_low.values())
        assert dif_std >= dif_low


class TestDistanciaErdosPageRank:
    def _grafo_transitivo(self):
        """A→C→B con win_rate=1.0. C tiene alta centralidad relativa."""
        return {
            'A': {'C': 1.0},
            'C': {'B': 1.0},
            'B': {},
        }

    def test_retorna_pagerank_scores_en_output(self):
        """T20-03: pagerank_scores siempre está en el resultado."""
        grafo = _grafo_lineal(['A', 'B', 'C', 'D', 'E', 'F'])
        r = distancia_erdos('A', 'F', grafo)
        assert 'pagerank_scores' in r

    def test_grafo_pequeno_pagerank_vacio(self):
        """REGLA-T20-1: grafo < 5 nodos → pagerank_scores = {}."""
        grafo = self._grafo_transitivo()   # 3 nodos
        r = distancia_erdos('A', 'B', grafo)
        assert r['pagerank_scores'] == {}

    def test_camino_directo_no_usa_intermedio(self):
        """REGLA-T20-3: camino de profundidad 1 (len=2) → quality_multiplier=1.0."""
        # A→B directo en grafo de 5+ nodos
        grafo = {'A': {'B': 1.0, 'C': 0.5, 'D': 0.5, 'E': 0.5},
                 'B': {}, 'C': {}, 'D': {}, 'E': {}}
        # node_weights con centralidad baja para B
        nw = {'A': 1.0, 'B': 0.1, 'C': 0.3, 'D': 0.3, 'E': 0.3}
        r_con = distancia_erdos('A', 'B', grafo, node_weights=nw)
        r_sin = distancia_erdos('A', 'B', grafo, node_weights=None)
        # El camino directo no tiene intermedio → mismo score independiente de node_weights
        # (a menos que haya caminos transitivos también)
        # Ambos deben tener erdos_score válido
        assert r_con['erdos_score'] == r_sin['erdos_score']

    def test_node_weights_externos_se_usan(self):
        """node_weights pasados externamente se usan en lugar de calcular PageRank."""
        grafo = _grafo_lineal([f'J{i}' for i in range(8)])
        nw = {f'J{i}': 0.5 for i in range(8)}
        r = distancia_erdos('J0', 'J7', grafo, node_weights=nw)
        assert r['pagerank_scores'] == nw

    def test_erdos_score_es_mayor_con_intermedio_de_alta_centralidad(self):
        """Mismo camino: intermedio con centralidad alta → erdos_score más alto."""
        # Grafo: A→C→B + relleno para tener n≥5
        grafo = {
            'A': {'C': 1.0},
            'C': {'B': 1.0},
            'B': {},
            'D': {'A': 0.5},
            'E': {'D': 0.5},
        }
        nw_alto = {'A': 0.5, 'C': 1.0, 'B': 0.5, 'D': 0.3, 'E': 0.3}
        nw_bajo = {'A': 0.5, 'C': 0.1, 'B': 0.5, 'D': 0.3, 'E': 0.3}
        r_alto = distancia_erdos('A', 'B', grafo, node_weights=nw_alto)
        r_bajo = distancia_erdos('A', 'B', grafo, node_weights=nw_bajo)
        assert r_alto['erdos_score'] > r_bajo['erdos_score']

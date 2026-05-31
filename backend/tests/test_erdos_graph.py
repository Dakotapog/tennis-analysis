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
        """El resultado siempre tiene: erdos_score, paths, n_paths, max_depth_alcanzado."""
        hist = [{'oponente': 'B', 'outcome': 'ganó'}]
        grafo = construir_grafo_victorias(historial_a_partidos(hist, 'A'))
        r = distancia_erdos('A', 'B', grafo)
        for campo in ('erdos_score', 'erdos_score_raw', 'paths', 'n_paths', 'max_depth_alcanzado'):
            assert campo in r, f"Campo faltante: {campo}"

"""
Erdős Distance Graph — Ventaja Transitiva en Tenis (Nodo-06)

Fundamento científico:
  Inspirado en el Problema de Distancia Unitaria de Paul Erdős:
  ¿Cuántos pares de puntos en el plano pueden estar exactamente a distancia 1?
  La conjetura dice que el número es subcuadrático — menos de lo que parece intuitivamente.

Aplicación en tenis:
  Dos jugadores conectados a través de oponentes comunes forman un grafo de victorias.
  La ventaja transitiva válida es subcuadrática en el número de oponentes:
  más allá de distancia 3, los caminos son ruido estadístico.

  - Distancia 1: A venció a B directamente → máxima confianza
  - Distancia 2: A venció C, C venció a B → ventaja transitiva válida
  - Distancia 3: A→C→D→B → señal débil, peso mínimo

Factor de decaimiento α=0.7:
  Si A gana 60% contra C, y C gana 60% contra B:
    Cadena directa:  0.6 × 0.6 = 0.36
    Con Erdős:       0.6 × 0.6 × 0.7 = 0.252
  El 0.7 captura que victorias en el grafo NO son independientes
  (torneo, superficie y época del año son variables latentes compartidas).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Conversión de formato historial → entradas de grafo
# ──────────────────────────────────────────────────────────────────────────────

def historial_a_partidos(player_history: list, player_name: str) -> List[dict]:
    """
    Convierte el formato de historial del pipeline (oponente + outcome)
    al formato de entrada del grafo ({ganador, perdedor}).

    Formato entrada:
        [{'oponente': 'Nadal R.', 'outcome': 'ganó', ...}, ...]
    Formato salida:
        [{'ganador': 'Tsitsipas S.', 'perdedor': 'Nadal R.'}, ...]
    """
    partidos = []
    for match in player_history:
        oponente = match.get('oponente', '').strip()
        if not oponente:
            continue
        outcome = match.get('outcome', '').lower()
        if 'ganó' in outcome or 'win' in outcome:
            partidos.append({'ganador': player_name, 'perdedor': oponente})
        elif 'perdió' in outcome or 'loss' in outcome:
            partidos.append({'ganador': oponente, 'perdedor': player_name})
        else:
            # RET/WO → victoria para el jugador del historial
            resultado = match.get('resultado', '').upper()
            if 'RET' in resultado or 'WO' in resultado:
                partidos.append({'ganador': player_name, 'perdedor': oponente})
    return partidos


# ──────────────────────────────────────────────────────────────────────────────
# Construcción del grafo de victorias
# ──────────────────────────────────────────────────────────────────────────────

def construir_grafo_victorias(partidos: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Construye un grafo dirigido ponderado a partir de una lista de partidos.

    grafo[A][B] = win_rate de A contra B (float 0-1)

    Args:
        partidos: lista de dicts con {'ganador': str, 'perdedor': str}
                  También acepta {'winner': str, 'loser': str}.

    Returns:
        dict{jugador → dict{oponente → win_rate}}
    """
    # Acumular resultados: victorias[A][B] = número de veces que A venció a B
    victorias: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for partido in partidos:
        ganador = partido.get('ganador') or partido.get('winner', '')
        perdedor = partido.get('perdedor') or partido.get('loser', '')
        if ganador and perdedor and ganador != perdedor:
            victorias[ganador][perdedor] += 1

    # Convertir a win_rate: A vs B → A_wins / (A_wins + B_wins_against_A)
    # Bug fix: usar list() para evitar RuntimeError al iterar sobre un defaultdict
    # que se puede expandir si accedemos a claves no existentes vía .get()
    grafo: Dict[str, Dict[str, float]] = {}
    for jugador in list(victorias.keys()):
        grafo[jugador] = {}
        for oponente, wins_a in victorias[jugador].items():
            wins_b = victorias.get(oponente, {}).get(jugador, 0)
            total = wins_a + wins_b
            grafo[jugador][oponente] = wins_a / total if total > 0 else 0.5

    return grafo


# ──────────────────────────────────────────────────────────────────────────────
# BFS con decaimiento Erdős
# ──────────────────────────────────────────────────────────────────────────────

def distancia_erdos(
    jugador_a: str,
    jugador_b: str,
    grafo: Dict[str, Dict[str, float]],
    max_depth: int = 3,
    alpha: float = 0.7,
) -> dict:
    """
    Calcula la ventaja transitiva de jugador_a sobre jugador_b usando BFS
    con decaimiento Erdős por cada nivel de profundidad.

    Peso de un camino de profundidad d:
        peso = prod(win_rates en el camino) × α^(d-1)

    Args:
        jugador_a:   jugador cuya ventaja queremos medir
        jugador_b:   jugador contra el que se mide
        grafo:       salida de construir_grafo_victorias()
        max_depth:   profundidad máxima a explorar (por defecto 3)
        alpha:       factor de decaimiento por nivel (por defecto 0.7)

    Returns:
        erdos_score:         float (-1 a +1), >0 = A tiene ventaja
        erdos_score_raw:     float (0-1) antes de centrar en 0
        paths:               lista de hasta 5 caminos con más peso
        n_paths:             total de caminos encontrados
        max_depth_alcanzado: profundidad máxima de los caminos encontrados
    """
    if jugador_a == jugador_b:
        return {
            'erdos_score': 0.0,
            'erdos_score_raw': 0.5,
            'paths': [],
            'n_paths': 0,
            'max_depth_alcanzado': 0,
        }

    if not grafo:
        return {
            'erdos_score': 0.0,
            'erdos_score_raw': 0.5,
            'paths': [],
            'n_paths': 0,
            'max_depth_alcanzado': 0,
        }

    # BFS: cola de (nodo_actual, camino, peso_acumulado, profundidad)
    cola: deque = deque([(jugador_a, [jugador_a], 1.0, 0)])
    paths_encontrados: List[dict] = []
    # Evitar revisitar el mismo nodo al mismo nivel para no inflar el score
    visitados_por_nivel: Dict[int, set] = defaultdict(set)

    while cola:
        nodo, camino, peso, profundidad = cola.popleft()

        if profundidad > max_depth:
            continue

        if nodo in visitados_por_nivel[profundidad]:
            continue
        visitados_por_nivel[profundidad].add(nodo)

        vecinos = grafo.get(nodo, {})
        for vecino, win_rate in vecinos.items():
            if vecino in camino:  # evitar ciclos
                continue

            nuevo_peso = peso * win_rate * (alpha ** profundidad)
            nuevo_camino = camino + [vecino]
            nueva_profundidad = profundidad + 1

            if vecino == jugador_b:
                # Bug fix: respetar max_depth también cuando encontramos jugador_b
                if nueva_profundidad <= max_depth:
                    paths_encontrados.append({
                        'camino': nuevo_camino,
                        'peso': round(nuevo_peso, 4),
                        'profundidad': nueva_profundidad,
                    })
            elif nueva_profundidad <= max_depth:
                cola.append((vecino, nuevo_camino, nuevo_peso, nueva_profundidad))

    if not paths_encontrados:
        return {
            'erdos_score': 0.0,
            'erdos_score_raw': 0.5,
            'paths': [],
            'n_paths': 0,
            'max_depth_alcanzado': 0,
        }

    # Fórmula del score — ventaja absoluta respecto al baseline neutral con decaimiento.
    #
    # neutral(d) = (0.5^d) × α^(d-1)
    #   d=1 → 0.5                (win_rate=0.5 directo, sin decay)
    #   d=2 → 0.25 × 0.7 = 0.175 (win_rate=0.5 transitivo, un decay)
    #
    # advantage(path) = path_weight - neutral(d)
    #   > 0 si A tiene ventaja a través de este camino
    #
    # Ejemplo (win_rate=0.7, α=0.7):
    #   d=1: advantage = 0.700 - 0.500 = +0.200
    #   d=2: advantage = 0.343 - 0.175 = +0.168  (< directo ✓ decaimiento)
    advantages = []
    for path in paths_encontrados:
        d = path['profundidad']
        neutral_w = (0.5 ** d) * (alpha ** (d - 1))
        advantages.append(path['peso'] - neutral_w)

    erdos_score = sum(advantages) / len(advantages)
    erdos_score = max(-1.0, min(1.0, erdos_score))  # clamp a [-1, 1]

    # erdos_score_raw: ratio de pesos totales en [0, 1] para comparación rápida
    ventaja_total = sum(p['peso'] for p in paths_encontrados)
    n_paths = len(paths_encontrados)
    denominador = ventaja_total + n_paths * 0.5
    score_raw = ventaja_total / denominador if denominador > 0 else 0.5

    max_depth_real = max(p['profundidad'] for p in paths_encontrados)

    return {
        'erdos_score': round(erdos_score, 4),
        'erdos_score_raw': round(score_raw, 4),
        'paths': sorted(paths_encontrados, key=lambda x: -x['peso'])[:5],
        'n_paths': n_paths,
        'max_depth_alcanzado': max_depth_real,
    }

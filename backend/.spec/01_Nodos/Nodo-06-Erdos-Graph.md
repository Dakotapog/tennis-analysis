# Nodo-06: Erdős Distance Graph Enhancement

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]]
> ⚠️ PRODUCCIÓN: Erdős está implementado en `analysis/rivalry_analyzer.py` (✅ 37 tests) pero NO activo en producción hasta completar [[Nodo-07-Strangler-Fig]] Fase 1. Estado real: IMPLEMENTADO-NO-CONECTADO.
> **Fundamento científico:** Inspirado en la Conjetura de Distancia Unitaria de Paul Erdős

**Prioridad:** MEDIA — mejora `common_opponents` (20% del modelo, actualmente funcionando)
**Archivo objetivo:** `analysis/rivalry_analyzer.py` (sección `common_opponents_detailed`)

---

## La Conexión: Erdős → Tenis

El **Problema de Distancia Unitaria** de Erdős pregunta cuántos pares de puntos
en el plano pueden estar a distancia exactamente 1. La conjetura de Erdős dice que
el número de pares es O(n^(1+c/log log n)) — menos de lo que parece intuitivamente.

**Insight para tenis:** Dos jugadores que nunca se han enfrentado están
"conectados" a través de oponentes comunes — exactamente como en el grafo de distancias.
La **distancia Erdős en tenis** mide qué tan directamente podemos inferir una ventaja:

- Distancia 1: A venció a B directamente
- Distancia 2: A venció a C, C venció a B → A tiene ventaja transitiva sobre B
- Distancia 3: A → C → D → B → ventaja más débil, más ruido

**La conjetura aplicada:** El número de ventajas transitivas válidas decrece
súbitamente más allá de distancia 2 — igual que los pares a distancia unitaria.
Más allá de distancia 3, el grafo es ruido.

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  erdos_graph_score: float (0-1) para el par (jugador1, jugador2)
          erdos_paths: lista de caminos transitivos encontrados
          max_depth: profundidad máxima alcanzada en el grafo

CONSUME:  historial_jugador de h2h_results_enhanced (ya existe)
          common_opponents_detailed (campo existente en rivalry_analyzer.py)

PREREQUISITO: rivalry_analyzer.py debe estar importable
              common_opponents_detailed debe estar funcionando (✅ confirmado)
```

---

## Implementación

```python
# analysis/erdos_graph.py
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import math


def construir_grafo_victorias(historial: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Construye un grafo dirigido de victorias:
      grafo[A][B] = win_rate de A contra B (entre 0 y 1)
    
    historial: lista de partidos con {ganador, perdedor} o formato FlashScore
    """
    grafo = defaultdict(lambda: defaultdict(list))
    
    for partido in historial:
        ganador = partido.get('ganador') or partido.get('winner')
        perdedor = partido.get('perdedor') or partido.get('loser')
        if ganador and perdedor:
            grafo[ganador][perdedor].append(1)  # victoria directa
    
    # Convertir a win_rate (0-1)
    grafo_pesos = {}
    for jugador, oponentes in grafo.items():
        grafo_pesos[jugador] = {}
        for oponente, victorias in oponentes.items():
            total_encuentros = len(grafo[jugador][oponente]) + len(grafo[oponente][jugador])
            win_rate = len(victorias) / total_encuentros if total_encuentros > 0 else 0.5
            grafo_pesos[jugador][oponente] = win_rate
    
    return grafo_pesos


def distancia_erdos(
    jugador_a: str,
    jugador_b: str,
    grafo: Dict[str, Dict[str, float]],
    max_depth: int = 3,
    alpha: float = 0.7  # decaimiento por distancia
) -> dict:
    """
    Calcula la ventaja transitiva de A sobre B usando BFS en el grafo de victorias.
    
    La ventaja decrece con la distancia según A^depth:
      - Distancia 1 (directo):  peso = win_rate × alpha^0 = win_rate
      - Distancia 2 (transitivo): peso = prod(win_rates) × alpha^1
      - Distancia 3 (largo):    peso = prod(win_rates) × alpha^2
    
    α = 0.7 — calibrado en la conjetura de Erdős (ventajas transitivas
    pierden ~30% de fuerza por cada paso en el grafo)
    
    Retorna:
        erdos_score: float (-1 a +1), positivo = A tiene ventaja sobre B
        paths: lista de caminos encontrados con sus pesos
        max_depth_alcanzado: int
    """
    if jugador_a == jugador_b:
        return {'erdos_score': 0.0, 'paths': [], 'max_depth_alcanzado': 0}
    
    # BFS con límite de profundidad
    # Cola: (nodo_actual, camino, peso_acumulado, profundidad)
    cola = deque([(jugador_a, [jugador_a], 1.0, 0)])
    paths_encontrados = []
    visitados_por_nivel = defaultdict(set)
    
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
            
            if vecino == jugador_b:
                paths_encontrados.append({
                    'camino': nuevo_camino,
                    'peso': round(nuevo_peso, 4),
                    'profundidad': profundidad + 1
                })
            elif profundidad + 1 <= max_depth:
                cola.append((vecino, nuevo_camino, nuevo_peso, profundidad + 1))
    
    if not paths_encontrados:
        return {'erdos_score': 0.0, 'paths': [], 'max_depth_alcanzado': 0}
    
    # Agregar ventajas: suma ponderada normalizando por 0.5 (neutral)
    ventaja_total = sum(p['peso'] for p in paths_encontrados)
    n_paths = len(paths_encontrados)
    
    # Score normalizado: >0.5 = A tiene ventaja, <0.5 = B tiene ventaja
    score_normalizado = ventaja_total / (ventaja_total + n_paths * 0.5) if (ventaja_total + n_paths * 0.5) > 0 else 0.5
    
    # Centrar en 0: -1 a +1
    erdos_score = (score_normalizado - 0.5) * 2
    
    max_depth_real = max(p['profundidad'] for p in paths_encontrados)
    
    return {
        'erdos_score': round(erdos_score, 4),
        'erdos_score_raw': round(score_normalizado, 4),  # 0-1 para comparación
        'paths': sorted(paths_encontrados, key=lambda x: -x['peso'])[:5],  # top 5
        'n_paths': n_paths,
        'max_depth_alcanzado': max_depth_real
    }
```

---

## Integración en rivalry_analyzer.py

```python
# En analyze_rivalry(), enriquecer common_opponents_detailed:
from analysis.erdos_graph import construir_grafo_victorias, distancia_erdos

# Construir grafo desde historial combinado
historial_combinado = player1_history + player2_history
grafo = construir_grafo_victorias(historial_combinado)

# Calcular distancia Erdős
erdos_result = distancia_erdos(
    jugador_a=player1_name,
    jugador_b=player2_name,
    grafo=grafo,
    max_depth=3
)

# El erdos_score reemplaza/enriquece el common_opponents_score existente
# common_opponents actualmente = 20% del modelo total
# Integración conservadora: mezclar 50/50 con el score existente

erdos_contribution = erdos_result['erdos_score']  # -1 a +1
existing_common = resultado['common_opponents_score']  # ya normalizado

common_opponents_enhanced = (existing_common + erdos_contribution * 0.5) / 1.5

# Añadir al output JSON:
resultado['erdos_analysis'] = {
    'erdos_score': erdos_result['erdos_score'],
    'n_paths_transitivos': erdos_result['n_paths'],
    'max_depth': erdos_result['max_depth_alcanzado'],
    'caminos_top': erdos_result['paths'][:3]
}
```

---

## El Insight Matemático Profundo

```
Conjetura Erdős (plano): f(n) pares a distancia 1 de n puntos es subcuadrático
                          → pocos pares "realmente cercanos"

Aplicación tenis:        La ventaja transitiva válida es subcuadrática en n oponentes
                          → más allá de distancia 3, los paths son ruido estadístico

Por qué α = 0.7:
  Si A gana 60% contra C, y C gana 60% contra B:
    Probabilidad transitiva = 0.6 × 0.6 = 0.36 (cadena directa)
    Con decaimiento Erdős:   0.6 × 0.6 × 0.7 = 0.252 (más conservador)
  
  El factor 0.7 captura que las victorias en el grafo NO son independientes —
  los jugadores comparten un contexto de torneo, superficie, época del año.
  Erdős: la conjetura dice que las dependencias son más fuertes de lo que parecen.
```

---

## Conexiones Cross-Nodo (CX)

| CX | Conexión | Impacto |
|---|---|---|
| CX-02 | [[Nodo-02-Markov-Changepoint]] momentum | Si A está HOT y el grafo Erdős confirma ventaja transitiva → señal 2x más fuerte |
| CX-06 | [[Nodo-05-Validacion-API]] accuracy por superficie | Calibrar alpha=0.7 con datos reales (¿el decaimiento óptimo es 0.6 en clay?) |
| CX-07 | [[Nodo-01-Edge-Calculator]] edge final | erdos_score alto + edge > 5% = bet de máxima confianza |

---

## Output Esperado en JSON

```json
"erdos_analysis": {
    "erdos_score": 0.32,
    "n_paths_transitivos": 4,
    "max_depth": 2,
    "caminos_top": [
        {
            "camino": ["Alcaraz C.", "Zverev A.", "Sinner J."],
            "peso": 0.284,
            "profundidad": 2
        },
        {
            "camino": ["Alcaraz C.", "Ruud C.", "Sinner J."],
            "peso": 0.198,
            "profundidad": 2
        }
    ]
}
```

---

## Tests Requeridos

```python
# tests/test_erdos_graph.py
def test_ventaja_directa():
    """A venció a B directamente → score positivo."""
    grafo = {'A': {'B': 0.7}}
    r = distancia_erdos('A', 'B', grafo)
    assert r['erdos_score'] > 0

def test_ventaja_transitiva_distancia_2():
    """A venció a C, C venció a B → A tiene ventaja transitiva."""
    grafo = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
    r = distancia_erdos('A', 'B', grafo)
    assert r['erdos_score'] > 0
    assert r['max_depth_alcanzado'] == 2

def test_decaimiento_con_distancia():
    """Ventaja a distancia 2 debe ser menor que a distancia 1."""
    grafo_directo = {'A': {'B': 0.7}}
    grafo_transitivo = {'A': {'C': 0.7}, 'C': {'B': 0.7}}
    r1 = distancia_erdos('A', 'B', grafo_directo)
    r2 = distancia_erdos('A', 'B', grafo_transitivo)
    assert r1['erdos_score'] > r2['erdos_score']

def test_sin_conexion_score_cero():
    """Jugadores sin oponentes comunes → score = 0."""
    grafo = {'A': {'X': 0.8}, 'Y': {'B': 0.8}}
    r = distancia_erdos('A', 'B', grafo, max_depth=2)
    assert r['erdos_score'] == 0.0

def test_max_depth_respetado():
    """No explorar más allá de max_depth."""
    # Cadena A→C→D→B (distancia 3) con max_depth=2
    grafo = {'A': {'C': 0.8}, 'C': {'D': 0.8}, 'D': {'B': 0.8}}
    r = distancia_erdos('A', 'B', grafo, max_depth=2)
    assert r['erdos_score'] == 0.0  # no alcanzado con max_depth=2
```

---

## Ciclo de Vida

```
Estado:   POR CONSTRUIR (common_opponents_detailed ya funciona — este es un enhancement)
Prioridad relativa: después de Nodo-01, Nodo-02, Nodo-03, Nodo-04, Nodo-05
Construcción: ~4 horas (incluye integración en rivalry_analyzer.py)
Validación: comparar accuracy con/sin erdos_enhancement en dataset Jan 2026
Meta cuantitativa: accuracy mejora >1pp con erdos_enhancement activado
```

# Nodo-20 — PageRank Erdős Quality

> **Estado:** ✅ COMPLETADO — 2026-06-03
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-19-H2H-Immunity-Dampener]]
> **Tests:** 980 passed (13 tests nuevos T20-04)
> **Origen:** Test-Time Compute — 4 marcos expertos, sesión cerrada inesperadamente 2026-06-03
> **Orden de implementación recomendado:** 3 de 3 (refinamiento — después de Nodo-19 y Nodo-18)

---

## Problema — Nodos Intermedios sin Peso de Calidad

El grafo de victorias ya existe en `erdos_graph.py`. El problema está en cómo se calcula `distancia_erdos()`:

```python
# erdos_graph.py — peso de arista actual:
peso = wins / total_matches   # win rate de A contra B

# El problema: los NODOS INTERMEDIOS no tienen peso de centralidad
# "Parry ganó a X que ganó a Djokovic" vale IGUAL que
# "Parry ganó a Y que ganó a rank-300"
# (si las distancias de camino son las mismas)
```

El grafo distingue la calidad de las aristas (win rate), pero **no la centralidad de los nodos intermedios**. Un jugador que ganó a top-10 "transmite" el mismo valor Erdős que uno que ganó a rank-300.

---

## Fundamento — Marco 3: Arquitecto de Software (Conexiones en Código Existente)

### PageRank Power Iteration — 15 líneas sobre el grafo existente

```python
def pagerank_grafo(grafo: dict, damping: float = 0.85, iteraciones: int = 10) -> dict:
    """
    Calcula PageRank sobre el grafo de victorias ya construido.
    Retorna centrality score por jugador, normalizado [0, 1].
    """
    nodos = list(grafo.keys())
    n = len(nodos)
    if n == 0:
        return {}

    pr = {nodo: 1.0 / n for nodo in nodos}
    for _ in range(iteraciones):
        pr_nuevo = {}
        for nodo in nodos:
            suma = 0.0
            for origen, vecinos in grafo.items():
                if nodo in vecinos:
                    out_degree = sum(vecinos.values()) or 1
                    suma += pr[origen] * (vecinos[nodo] / out_degree)
            pr_nuevo[nodo] = (1 - damping) / n + damping * suma
        pr = pr_nuevo

    max_pr = max(pr.values()) or 1.0
    return {k: round(v / max_pr, 4) for k, v in pr.items()}
```

### Uso en `distancia_erdos()` (T20-02)

```python
# Ponderar advantage de cada camino por centralidad del nodo intermedio
centrality = pagerank_grafo(grafo)
for path in paths:
    intermediate = path[1] if len(path) > 2 else None
    quality_multiplier = centrality.get(intermediate, 0.5) if intermediate else 1.0
    advantage = path_weight * quality_multiplier
```

**Efecto concreto:** Parry ganando a alguien que ganó a top-10 vale **más** que Parry ganando a alguien que ganó a rank-200. Actualmente el sistema no distingue.

---

## Por qué es diferencial (Marco 4 — Hedge Fund Estratega)

La mayoría de sistemas de rating de tenis públicos usan win rate directo o ELO fijo. PageRank sobre el grafo transitivo añade una dimensión que no está en los modelos públicos: **la calidad de las conexiones indirectas**.

Bookmaker no tiene este insight en sus cuotas para partidos entre jugadores con H2H escaso (Challenger, jugadores jóvenes). Es donde Erdős + PageRank tiene más alpha.

---

## Conexión con el Pipeline Existente

| Componente | Rol en Nodo-20 | Estado |
|---|---|---|
| `analysis/erdos_graph.py` → `construir_grafo_victorias()` | Grafo ya construido — base para PageRank | ✅ existe |
| `analysis/erdos_graph.py` → `distancia_erdos()` | Punto de integración de `node_weights` | 🔴 modificar |
| `analysis/rivalry_analyzer.py` | Consume `erdos_score` — recibirá score mejorado automáticamente | — |
| `edge_calculator.py` | Recibe `erdos_score` vía `rivalry_analyzer` — sin modificación directa | — |

---

## Dependencias

| Prerequisito | Estado | Nodo |
|---|---|---|
| `construir_grafo_victorias()` en `erdos_graph.py` | ✅ ACTIVO | [[Nodo-06-Erdos-Graph]] |
| Grafo con n suficiente (≥20 jugadores) | ✅ activo en prod | — |
| Nodo-19 implementado (recomendado, no bloqueante) | 🟡 RECOMENDADO | [[Nodo-19-H2H-Immunity-Dampener]] |

---

## Tasks

| ID | Descripción | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T20-01 | `pagerank_grafo(grafo, damping=0.85, n=10)` → `dict {jugador: centrality_score}` normalizado [0,1] | `analysis/erdos_graph.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T20-02 | `distancia_erdos()` recibe opcionalmente `node_weights=None` → si se pasa, `advantage *= node_weights[intermediate]` | `analysis/erdos_graph.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T20-03 | Exportar `pagerank_scores` en output de `distancia_erdos()` (campo informativo) | `analysis/erdos_graph.py` | 🟢 BAJO | ✅ COMPLETADO |
| T20-04 | Tests: power iteration convergencia + casos edge (grafo vacío, un solo nodo, grafo desconectado) | `tests/test_erdos_graph.py` | — | ✅ COMPLETADO — 13 tests |

**Estimación:** ~15 líneas de lógica nueva (`pagerank_grafo`). Modificación de `distancia_erdos`: ~5 líneas. Tests: ~10 casos.

---

## Reglas Nuevas

**REGLA-T20-1: PageRank es enhancement, no reemplazo**
```
erdos_score actual sigue siendo válido.
PageRank añade quality_multiplier sobre el nodo intermedio.
Si grafo vacío o n<5 jugadores → quality_multiplier = 1.0 (sin cambio).
```

**REGLA-T20-2: damping=0.85 fijo hasta n≥50 jugadores en grafo**
```
damping=0.85 es el estándar (PageRank original).
No calibrar hasta tener suficiente muestra en el grafo.
```

**REGLA-T20-3: Caminos directos (longitud 2) no tienen nodo intermedio**
```
Si len(path) == 2 → quality_multiplier = 1.0 (sin ponderación por intermedio)
Solo caminos transitivos (longitud ≥ 3) aplican la ponderación por centralidad.
```

**REGLA-T20-4: Implementar después de Nodo-19 y Nodo-18**
```
Nodo-20 es refinamiento incremental de Erdős.
Requiere n suficiente en el grafo para que PageRank sea estable.
Mayor impacto en Challengers donde H2H directo es escaso y las cadenas transitivas son la señal principal.
```

# Nodo-104 — Graphify Faceted Filter + Ego-Network Opacity + Auditoría Wikilinks

> **Wikilinks:** [[Nodo-83-Graphify-Server]] | [[Nodo-84-Graphify-3D]] | [[Nodo-85-Graphify-Fase3]] | [[Nodo-88-Graphify-Fase4-Report]] | [[Nodo-75-Nodos-Index]]
> **Fecha:** 2026-07-16 | **Autor:** Sonnet 4.6
> **Contexto:** El filtro de tipo en graph3d.html era destructivo (rgba(0,0,0,0) + size 0.001).
> El sidebar desaparecía por canvas Three.js sin position:relative. 192 wikilinks rotos y
> 22 huérfanos identificados en auditoría sistemática de `.spec/01_Nodos/`.

---

## 1. AUDITORÍA WIKILINKS — Snapshot 2026-07-16

### 1.1 Métricas globales

| Métrica | Snapshot 2026-07-16 | Post-fix 2026-07-17 |
|---|---|---|
| Archivos `.md` en `.spec/01_Nodos/` | 108 | 111 (+3 MOCs) |
| Wikilinks totales | 627 | ~700 (nuevos MOCs) |
| Wikilinks rotos | 192 (30.6%) | **106 (estructurales únicamente)** |
| — Estructurales (vault Obsidian histórico) | 106 | 106 — intactos (historia inmutable) |
| — Fixables (nombre incorrecto / nodo inexistente) | 86 | **0 — RESUELTOS** |
| Huérfanos verdaderos (0 inbound) | 22 | ~12 (reducidos por MOCs nuevos) |

**Fix aplicado 2026-07-17:** 96 correcciones en 30 archivos — B1 short-form (63), B2 nombre incorrecto (25), B2-MISSING anotados (8). 0 wikilinks rotos no-estructurales restantes.

### 1.2 Categoría A — Estructurales (no fixables)

Nodos 01–30 referencian páginas del vault Obsidian original que nunca existieron como
`.md` en `.spec/01_Nodos/`. Son historia inmutable — no crear archivos vacíos para taparlos.

| Target roto | Apariciones | Decisión |
|---|---|---|
| `[[Mandatos-No-Negociables]]` | ~30 | STRUCTURAL — vault page |
| `[[Sprint-Pipeline]]` | ~30 | STRUCTURAL — vault page |
| `[[Pipeline-Arquitectura]]` | ~30 | STRUCTURAL — vault page |
| `[[Grafo-Dependencias-Datos]]` | ~30 | STRUCTURAL — vault page |
| `[[Fuentes-Datos]]` | ~30 | STRUCTURAL — vault page |
| `[[MOC-Principal]]` | ~20 | STRUCTURAL — vault page |
| `[[Inventario-Deuda-Tecnica]]` | ~10 | STRUCTURAL — vault page |
| `[[Sprint-Normalizacion-19jun]]` | 1 | STRUCTURAL — sprint page |

### 1.3 Categoría B — Fixables por sesión futura

**B1 — Short-form (nombre incompleto):**

| Archivo fuente | Wikilink roto | Corrección |
|---|---|---|
| Nodo-97, 98, 99, 103 | `[[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]]` | `[[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]]` |
| Nodo-98 | `[[Nodo-68-Rival-Value-Flip]]` | `[[Nodo-68-Rival-Value-Flip]]` |
| Nodo-98 | `[[Nodo-91-Sprint1-Capas-Fallback-Implementacion]]` | `[[Nodo-91-Sprint1-Capas-Fallback-Implementacion]]` |
| Nodo-99 | `[[Nodo-43-PELT-Cold-Rival-Promo-Filter]]` | `[[Nodo-43-PELT-Cold-Rival-Promo-Filter]]` |
| Nodo-42 | `[[Nodo-17-Calibracion-Por-Tier]]` | `[[Nodo-17-Calibracion-Por-Tier]]` |
| Nodo-32→57 | `[[Nodo-21-Pesos-Diferenciados-Por-Tier]]` | `[[Nodo-21-Pesos-Diferenciados-Por-Tier]]` |
| Nodo-32→57 | `[[Nodo-24-Bookmaker-Blindness-Scoring]]` | `[[Nodo-24-Bookmaker-Blindness-Scoring]]` |
| Nodo-32→57 | `[[Nodo-28-Conditional-Decomposition-Metamodel]]` | `[[Nodo-28-Conditional-Decomposition-Metamodel]]` |
| Nodo-32→57 | `[[Nodo-27-Pipeline-Tracker-Observabilidad]]` | `[[Nodo-27-Pipeline-Tracker-Observabilidad]]` |

**B2 — Nodos referenciados que no existen como archivo:**

| Wikilink | Estado | Decisión |
|---|---|---|
| `~~[[Nodo-70-CPPI]]~~ _(MISSING — CPPI no implementado como Nodo independiente)_` | No existe (en Nodo-87) | Marcar `[MISSING — Nodo-70 no implementado]` |
| `~~~~[[Nodo-22-API-Integration-Kambi-Ninja]]~~ _(MISSING)_~~ _(MISSING)_` | No existe | Marcar MISSING |
| `~~~~[[Nodo-43-Grass-Feature-Amplification]]~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_` | No existe | Marcar MISSING |
| `~~~~[[Nodo-14-Grass-Variance]]~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_` | No existe | Marcar MISSING |
| `[[Nodo-15-Portfolio-HedgeFund]]` | No existe | Marcar MISSING |
| `~~~~[[Nodo-31-Ronda-Futura-H2H]]~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_~~ _(MISSING — [[Nodo-31-Future-Match-Data-Leakage]] es diferente)_` | No existe (Nodo-31 tiene nombre diferente) | Verificar nombre real |
| `~~~~[[Nodo-28-Backtest-Limpio]]~~ _(MISSING)_~~ _(MISSING)_` | No existe | Marcar MISSING |
| `~~~~[[Nodo-32-Fase3-Markov-Postnorm]]~~ _(MISSING)_~~ _(MISSING)_` | No existe | Marcar MISSING |
| `[[Nodo-21-Pesos-Diferenciados-Por-Tier]]` | No existe | Marcar MISSING |
| `~~~~[[Nodo-32-Auditoria-Phantom-Edge]]~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_` | No existe | Marcar MISSING |
| `[[Nodo-93-Sprint2-Implementado]]` | No existe (archivo real: Nodo-93-Sprint2-PlayerDB-KambiLive) | Fix nombre |
| `[[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]]` | No existe (archivo real: Nodo-57-Penalizacion-Inactividad-Campeon-Validacion) | Fix nombre |
| `~~~~[[Nodo-38-Combo-Confianza]]~~ _(MISSING — [[Nodo-38-Portfolio-Aislamiento-Riesgo]] es diferente)_~~ _(MISSING — [[Nodo-38-Portfolio-Aislamiento-Riesgo]] cubre concepto diferente)_` | No existe (archivo real: Nodo-38-Portfolio-Aislamiento-Riesgo) | Fix nombre |
| `[[Nodo-02-Markov-Changepoint]]` | No existe (archivo real: Nodo-02-Markov-Changepoint) | Fix nombre |

### 1.4 Huérfanos verdaderos (0 inbound links desde otros specs)

```
FABLE_02_TENIS (1).md                          — doc raíz del sprint, no spec de nodo
FABLE_02_TENIS_DOCTORADO_SPEC.md               — doc raíz doctoral
PROMPT_PARA_FABLE_TENIS.md                     — prompt template
Nodo-09-API-Status-Keys.md                     — sin referencias cruzadas aún
Nodo-38B-Cobertura-Expandida-Sin-CatC.md       — variante de Nodo-38, sin citadores
Nodo-41-ML-Dataset-Cleanup-Trazabilidad.md     — ML suspendido, huérfano esperado
Nodo-42-Grass-Bootstrap.md                     — sin referencia desde Nodo-60/GCS
Nodo-52-ADDENDUM-Integracion-Contexto-Completo.md
Nodo-53-ADDENDUM-3-Fable-Final.md
Nodo-61-GCS-Season-Window-Fix.md               — citado en CLAUDE.md pero no en specs
Nodo-75-Nodos-Index.md                         — infraestructura, no es citada
Nodo-79-MinBet-Por-Tier.md
Nodo-80-Kambi-Name-Matching.md                 — PILOTO wikilinks cruzados (ver D104-06)
Nodo-81-Settlement-Name-Normalize.md           — PILOTO wikilinks cruzados (ver D104-06)
Nodo-82-Kambi-Match-ID-Structural.md
Nodo-84-Graphify-3D.md
Nodo-85-Graphify-Fase3.md
Nodo-88-Graphify-Fase4-Report.md               — este Nodo-104 lo cita
Nodo-94-Sprint3-PlayerIntelligence.md
Nodo-100-Taxonomia-Estrategias-Generacion-Combos.md
Nodo-102-Hypothesis-Tracking-H98-H100.md       — reciente, aún no citado
Nodo-103-Auditoria-Combo-Builder-Gates-n-h2h.md — reciente, aún no citado
```

**Huérfanos esperados** (docs de infraestructura/raíz): FABLE_02_*, PROMPT_*, Nodo-75
**Huérfanos accionables** (merecen al menos 1 wikilink desde nodo relacionado): 80, 81, 82, 84, 85, 88, 94, 100

---

## 2. GRAPHIFY FACETED FILTER — Plan de implementación

### Diagnóstico pre-implementación (ejecutado 2026-07-16)

| Check | Resultado |
|---|---|
| Filtro actual `hiddenTypes` | DESTRUCTIVO: `rgba(0,0,0,0)` + size `0.001` |
| Sidebar desaparece en 3D | Canvas Three.js sin `position:relative` tapa el sidebar |
| Campo `src` en graph.json | EXISTE en cada nodo (`src=archivo.py`) |
| `IntelligentMLEnhancer` en .graphifyignore | NO — god node de subsistema suspendido |
| Cross-wikilinks Nodo-72↔80↔81 | CERO — piloto wikilinks retroactivos confirmado |
| Nodos tipo `function` en grafo (post-update) | 1,008 de 4,532 totales |

### Decisiones

**D104-01 — Opacity tiers no-destructivos (IMPLEMENTADO)**

Reemplazar toggle binario por función `getOpacityTier(n)`:
- Nivel 1 — Foco: `return 1` (color original)
- Nivel 2 — Vecino directo del foco: `return 0.5` → `hexToRgba(color, 0.5)`
- Nivel 3 — Contexto: `return 0.07` → `hexToRgba(color, 0.07)`

NUNCA `rgba(0,0,0,0)` ni `display:none` — nodo sigue en DOM y en layout.
`nodeMap` (id→node) para O(1) lookup de vecinos.

**D104-02 — Layout fix sidebar (IMPLEMENTADO)**

Three.js inserta canvas con `position:absolute` relativo al viewport → tapa el sidebar.
Fix: `#graph { position: relative; overflow: hidden; min-width: 0; }` y `#sidebar { z-index: 100; position: relative; }`.

**D104-03 — Multi-facet filter chips (PENDIENTE)**

Estado del filtro como array de facetas activas combinables con AND:
```javascript
activeFilters = [
  { type: "nodeType", value: "function" },
  { type: "src",      value: "edge_calculator.py" },
  { type: "community", value: "37" }
]
```
UI: chips visuales en sidebar debajo del buscador. Cada chip tiene botón × para quitar.
Facetas: `nodeType` (Files/Classes/Functions) + `src` (archivo fuente) + `community`.
El campo `src` ya existe en graph.json — solo UI.

**D104-04 — Path-preserving BFS entre nodos de foco (PENDIENTE)**

Si dos nodos de foco no tienen arista directa, calcular camino BFS (reutiliza Nodo-84)
y mostrar nodos intermedios en Nivel 2 (vecino), aunque no coincidan con ningún filtro.
Evita que el grafo filtrado muestre islas desconectadas cuando sí existe un camino.

**D104-05 — Excluir ML suspendido de .graphifyignore (PENDIENTE)**

Pre-check ejecutado: cero archivos activos importan de estos módulos.
Agregar a `.graphifyignore`:
```
Intelligent_ml_enhancer.py
aplicar_enhancer.py
generar_dataset_plus.py
```
Elimina el god node `IntelligentMLEnhancer` del grafo de un subsistema apagado.

**D104-06 — Wikilinks retroactivos piloto Nodo-80↔81↔82 (PENDIENTE)**

Nodo-80 (Kambi Name Matching), Nodo-81 (Settlement Name Normalize) y Nodo-82 (Kambi Match ID)
son familia cohesiva de bugs de normalización — cero cross-links entre ellos.
Añadir referencia cruzada mínima en cada uno. Demuestra que wikilinks retroactivos reducen
huérfanos sin necesidad de tocar los 86 nodos con fixable broken links de una sola vez.

**D104-07 — Paso 5 del spec original (aclaración de numeración)**

El spec original del usuario numeró 8 pasos. El paso 5 era diagnóstico del campo `src`
en graph.json — ejecutado y confirmado (campo `src` existe en cada nodo).
La implementación del filtro por archivo vive en D104-03 (facetas). No hay paso fantasma:
paso 5 = diagnóstico ejecutado, implementación = D104-03.

---

## 3. ARCHIVOS MODIFICADOS / A MODIFICAR

| Archivo | Cambio | Estado |
|---|---|---|
| `graphify-out/graph3d.html` | D104-01: `hexToRgba()` + `getOpacityTier()` + `nodeMap` | HECHO |
| `graphify-out/graph3d.html` | D104-02: `position:relative` en `#graph` + `z-index:100` sidebar | HECHO |
| `graphify-out/graph3d.html` | D104-03: multi-facet filter chips | PENDIENTE |
| `graphify-out/graph3d.html` | D104-04: path-preserving BFS | PENDIENTE |
| `.graphifyignore` | D104-05: excluir ML suspendido | PENDIENTE |
| `Nodo-80/81/82.md` | D104-06: wikilinks cruzados piloto | PENDIENTE |

---

## 4. VERIFICACIÓN D104-01 + D104-02 (ejecutar en browser)

```
http://localhost:7779/graph3d.html

1. Ctrl+Shift+R → sidebar permanente visible (3 checkboxes Files/Classes/Functions)
2. Desmarcar "Functions" → 1,008 nodos función = niebla tenue fondo (0.07 opacity)
   Sus vecinos directos = semi-atenuados (0.5). Files+Classes = color pleno.
   El grafo NUNCA debe quedar vacío — nodos siempre en layout.
3. Desmarcar una Community desde la leyenda → mismo comportamiento de 3 tiers.
```

---

## 5. FLUJO POST-NODO-104 COMPLETO

```
graph3d.html (post D104-03+04):
  Sidebar:
    [x] Files  [x] Classes  [x] Functions     ← checkboxes tipo (existentes)
    [src: edge_calculator.py ×]               ← chip faceta archivo
    [community: 37 ×]                          ← chip faceta comunidad
    Buscador de nodos...
    Path Finder...
    Communities (leyenda)...

  Al filtrar edge_calculator.py + trader_ev_tenis.py sin arista directa:
    → BFS calcula camino rivalry_analyzer.py como intermediario
    → rivalry_analyzer aparece en Nivel 2 (0.5), aunque no coincida con filtro
    → no hay islas desconectadas
```

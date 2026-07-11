# Nodo-85 — Graphify Fase 3: Interactividad avanzada (graph3d.html)

**Fecha:** 2026-07-11
**Estado:** IMPLEMENTADO
**Rama:** main
**Spec:** `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` §Fase 3

---

## Features implementadas

### 3.1 SDD Coverage Overlay

`/nodos_index.json` servido por `graphify_server.py` vía ruta especial (no desde graphify-out/).
Al cargar, el HTML fetcha `graph.json` + `/nodos_index.json` en paralelo (`Promise.all`).

Lógica:
- Construye set de archivos cubiertos desde `nodos[].archivos_mencionados`
- Nodes `.py` cuyo `source_file` no aparece en ningún nodo → `sddOrphans` Set
- Nodos huérfanos: color `#FF6B6B` (rojo), badge `[sin Nodo SDD]` en tooltip e info panel
- Banner sidebar: `"SDD cobertura: 100% ✔"` (verde) o `"SDD huérfanos: N archivos"` (rojo)
- Estado actual (2026-07-11): 0 huérfanos — todos los archivos .py tienen cobertura Nodo

### 3.2 Path Finder (BFS)

Panel colapsable "Path Finder" en sidebar.
- Autocomplete en campos From/To (misma lógica que search)
- BFS sobre `adjacencyMap` pre-construido ANTES de que ForceGraph3D mute los links
- Path encontrado → `pathMode = { nodeSet, edgeSet }` → refresca colores:
  - Nodos en path: `#FFD700` (dorado), tamaño ampliado
  - Nodos fuera del path: `rgba(60,60,80,0.15)` (casi invisibles)
  - Edges en path: `#FFD700`, grosor 3px
  - Edges fuera: `rgba(40,40,40,0.04)` (invisibles)
- Muestra resultado: `N hops: A → B → C → D`
- "Clear" resetea pathMode y restaura colores normales
- Clic en nodo auto-rellena campo "From" si está vacío

### 3.3 Tooltip enriquecido en hover

Div flotante `#tooltip` posicionado en `mousemove`.
- Aparece con `onNodeHover`, desaparece al salir
- Contenido: `label` + tipo + grado + community + badge `[sin Nodo]` si es huérfano SDD
- No requiere clic — exploración rápida pasando el mouse

### 3.4 Filtro por tipo de nodo

3 checkboxes en la barra superior del sidebar:
- **Files**: `label.endsWith('.py')` → color `#76B7B2`
- **Classes**: CamelCase sin paréntesis → color `#F28E2B`
- **Functions**: `label.includes('()')` → color `#59A14F`

Toggle oculta/muestra nodos del tipo (`nodeVal → 0.001`, `nodeColor → transparent`).
Compatible con filtro de comunidades y path mode (todas las condiciones se combinan en `_nodeColor`/`_nodeVal`).

## Cambio en graphify_server.py

Ruta especial `GET /nodos_index.json` → sirve `{BACKEND_ROOT}/nodos_index.json`.
Patrón extensible: `BACKEND_ROUTES = {"/nodos_index.json": "nodos_index.json"}`.

## Criterio de aceptación (verificado 2026-07-11)

```
curl http://172.27.33.49:7779/nodos_index.json → 200, nodos:75, huerfanos:0
curl http://172.27.33.49:7779/graph3d.html     → 200, Cache-Control: no-cache
SDD banner: "SDD cobertura: 100% ✔" (verde)
Type filter: 3 checkboxes operativos
Path finder: colapsa/expande, autocomplete funciona
Tooltip: aparece en hover con info de nodo
```

## Vinculación

- Nodo-83: servidor :7779 (base) + `BACKEND_ROUTES` (nuevo endpoint)
- Nodo-84: graph3d.html (Fase 2, base de esta implementación)
- `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` §Fase 3
- `nodos_index.json` — fuente SDD (Nodo-75, generado por `rebuild_nodos_index.py`)
- Fase 4 (graphify report) — pendiente, requiere confirmación

# Nodo-84 — Graphify: Visualización 3D interactiva (graph3d.html)

**Fecha:** 2026-07-11
**Estado:** IMPLEMENTADO
**Rama:** main
**Spec:** `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` §Fase 2

---

## Contexto

Fase 2 de la spec Graphify v2. `graph.html` (2D, vis.js) sigue funcionando en paralelo —
`graph3d.html` es un archivo NUEVO, no reemplaza al 2D hasta confirmación explícita del usuario.

## Problema que resuelve

`graph.html` usa vis.js (2D). Para grafos con 1686 nodos la navegación 2D obliga a zoom
extremo para distinguir clusters. Una representación 3D con rotación orbital permite ver la
estructura de comunidades completa y navegar entre clusters sin perder el contexto global.

## Solución

### Librería: 3d-force-graph v1.80.0

- CDN: `https://unpkg.com/3d-force-graph@1.80.0/dist/3d-force-graph.min.js`
- Three.js + WebGL — 100% client-side, no backend nuevo
- Rango cómodo: 1686 nodos / 2753 edges (referencias con 10,000+ nodos)
- Compatible con `graph.json` existente (`nodes + links` con `source/target`)

### Archivo: `graphify-out/graph3d.html`

Características:
- `fetch('graph.json?t=' + Date.now())` — reutiliza servidor Fase 1 (:7779)
- Paleta Tableau-20 idéntica al 2D (`community % 20 → color`)
- Tamaño de nodo proporcional a `degree * 0.6`
- Links: `EXTRACTED` → azul #4E79A7 / grosor 1.2 | otros → gris / 0.4
- Sidebar: búsqueda, info de nodo al clic, leyenda de comunidades con toggle
- Botón "Reset view" — regresa cámara a posición inicial
- Link "Vista 2D" → `graph.html` (y viceversa en 2D)
- Adjacency list pre-computada ANTES de que ForceGraph3D mute los links

### Transformación de datos

`graph.json` nodos crudos → enriquecidos en JS:
```javascript
degree: degreeMap[n.id] || 0          // contado desde links
_color: PALETTE[n.community % 20]     // Tableau-20
_communityName: 'Community ' + n.community
nodeVal: degree * 0.6                 // tamaño esferas
```

## Criterio de aceptación (verificado 2026-07-11)

```
curl -si http://172.27.33.49:7779/graph3d.html → HTTP 200, Cache-Control: no-cache
graph.html 2D: sigue funcionando sin cambios ✓
graph3d.html: nuevo archivo, no elimina 2D ✓
```

Validación visual (usuario confirma en Chrome):
- Grafo 3D completo con rotación orbital / zoom / arrastre
- 1686 nodos coloreados por comunidad
- Búsqueda + clic → cámara vuela al nodo seleccionado
- Toggle de comunidades en sidebar

## Vinculación

- Nodo-83: Servidor HTTP :7779 — infraestructura base
- `graphify-out/graph3d.html` — implementación (en graphify-out/, gitignored como graph.html)
- `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` — spec completa Fases 1-4
- Fases 3-4 gateadas — requieren confirmación post-validación visual de Fase 2

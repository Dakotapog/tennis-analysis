# MOC — Graphify Visualización

> **Tipo:** Map of Content | **Creado:** 2026-07-16 (D105-02)
> **Wikilinks:** [[Nodo-83-Graphify-Server]] | [[Nodo-84-Graphify-3D]] | [[Nodo-85-Graphify-Fase3]] | [[Nodo-88-Graphify-Fase4-Report]] | [[Nodo-104-Graphify-Faceted-Filter-Wikilinks-Audit]] | [[Nodo-105-Knowledge-Graph-Navigation-Zettelkasten]]

---

## Nodos en este cluster

| Nodo | Tema | Estado |
|---|---|---|
| [[Nodo-83-Graphify-Server]] | Servidor HTTP `:7779` — `systemctl --user start graphify`, serve `graphify-out/` | completo |
| [[Nodo-84-Graphify-3D]] | `graph3d.html` Three.js/ForceGraph3D — visualización 3D orbital | completo |
| [[Nodo-85-Graphify-Fase3]] | SDD overlay, Path Finder, Tooltip, Type filter | completo |
| [[Nodo-88-Graphify-Fase4-Report]] | `graphify_report.py` CLI Markdown + Nodo-88 | completo |
| [[Nodo-104-Graphify-Faceted-Filter-Wikilinks-Audit]] | Faceted filter chips OR/AND, ego-network opacity 3 tiers, `.graphifyignore` | completo |
| [[Nodo-105-Knowledge-Graph-Navigation-Zettelkasten]] | Click-highlight ego-network, MOCs, PageRank, estado facet, bioluminiscencia neural | activo |

## Decisiones arquitecturales acumuladas

- `graphify update` NO sobreescribe `graph3d.html` — modificaciones manuales son seguras
- Campo en `graph.json` es `source_file`, no `src` — `_srcOf(n)` hace fallback
- `nodeThreeObjectExtend(false)` + `nodeOpacity(0)` = sprites reemplazan geometría default
- `renderer.render` override con `_inComposer` flag — evita recursión EffectComposer
- Links: `THREE.Color.set()` ignora alpha de rgba — usar hex colors + `linkOpacity()` global

## Acceso rápido

```bash
http://localhost:7779/graph3d.html   # 3D bioluminiscente (activo)
http://localhost:7779/graph.html     # 2D vis.js (fallback)
systemctl --user restart graphify    # si el servidor no responde
```

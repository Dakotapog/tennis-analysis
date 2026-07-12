# Nodo-88 — Graphify Fase 4: CLI de reporte Markdown

**Fecha:** 2026-07-12
**Estado:** IMPLEMENTADO
**Rama:** main
**Spec:** `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` §Fase 4

---

## Qué hace

`graphify_report.py` — script CLI que genera un reporte Markdown del grafo de código.
Sin dependencias externas (solo stdlib + `graphify-out/graph.json`).

```bash
python3 graphify_report.py
# → reports/graphify_report_YYYYMMDD.md

python3 graphify_report.py --output reports/graphify_report_custom.md
```

## Contenido del reporte

### 1. Header

```
# Graphify Report — YYYY-MM-DD
Nodos: N | Edges: E | Comunidades: C | Commit: `abc123`
```

### 2. Delta vs reporte anterior

Busca el `reports/graphify_report_*.md` más reciente (por fecha en nombre).
Extrae nodos/edges/comunidades del anterior y calcula delta:

| Métrica | Anterior | Actual | Delta |
|---|---|---|---|
| Nodos | 1680 | 1686 | +6 |
| Edges | 2740 | 2753 | +13 |
| Comunidades | 91 | 91 | +0 |

También detecta cambios de centralidad en nodos que aparecen en el Top 20 de ambos reportes.

### 3. Top 20 nodos por grado (centralidad)

Grado = `in-degree + out-degree` calculado desde `links[]` del grafo.
Tipo inferido: `file` si label termina en `.py`, `fn` si tiene `()`, `class` si no.

### 4. Huérfanos SDD

**Fuente única: `nodos_index.json`** — mismo archivo que `check_contradictions.py`.
No reimplementa la lógica de detección. Lee `.huerfanos[]` y `._meta`.

### 5. Resumen de comunidades

Top 5 nodos por grado dentro de cada comunidad. Ordenadas por tamaño (mayor → menor).

## Criterio de aceptación (verificado 2026-07-12)

```
python3 graphify_report.py
→ Reporte: reports/graphify_report_20260712.md
→ Nodos: 1686 | Edges: 2753 | Comunidades: 91

Contenido verificado:
- Top 20: betplay_combo_builder.py (50) → shadow_book.py (41) → IntelligentMLEnhancer (38)
- SDD: 62 archivos rastreados, 0 huérfanos, cobertura 100% ✓
- 91 comunidades, Community 0 con 64 nodos (IntelligentMLEnhancer, DataFrame, ...)
- Delta: "(No se encontró reporte anterior para delta.)" — primer reporte ✓
```

## Diseño

```
graphify_report.py
├── _previous_report(reports_dir)     → Path | None  (glob + sort por fecha)
├── _parse_prev(path)                 → dict {nodes, edges, communities, top, name}
│       extrae conteos vía regex + tabla top-20 vía parseo Markdown
└── generate(data, nodos_idx, prev)   → str Markdown
        ├── Degree computation desde links[]
        ├── Delta vs prev (si existe)
        ├── Top 20 por grado
        ├── Cambios de centralidad (si prev tiene top)
        ├── Huérfanos SDD desde nodos_idx["huerfanos"]
        └── Community summary (top 5 por grado, sorted por tamaño)
```

## Fuentes de datos

| Dato | Fuente | Responsable |
|---|---|---|
| Nodos / Edges / Comunidades | `graphify-out/graph.json` | graphify CLI |
| Commit de construcción | `graph.json .built_at_commit` | graphify CLI |
| Huérfanos SDD | `nodos_index.json .huerfanos[]` | `scripts/rebuild_nodos_index.py` (Nodo-75) |
| Meta SDD | `nodos_index.json ._meta` | Nodo-75 |

## Vinculación

- Nodo-83: `graphify_server.py` — infraestructura HTTP
- Nodo-84: `graph3d.html` — visualización 3D
- Nodo-85: interactividad avanzada (SDD overlay, path finder, tooltip, type filter)
- Nodo-75: `rebuild_nodos_index.py` — genera `nodos_index.json` (fuente SDD)
- `docs/specs/SPEC_GRAPHIFY_V2_3D_REPORTES.md` §Fase 4

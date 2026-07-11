# SPEC_GRAPHIFY_V2_3D_REPORTES.md
# Graphify — Especificación de mejoras v2: 3D + Reportes
# Audiencia: Claude Code (Sonnet) — ejecutar sin ambigüedad, fase por fase
# Fecha: 2026-07-11

---

## 0. Contexto arquitectónico

`graphify-out/graph.html` embebía el grafo completo como JSON inline en el momento de
`graphify update .`. Al abrirse via `file://`, el navegador bloquea `fetch()` a otros
archivos locales por política de mismo-origen (CORS) — de ahí el desfase de 20 nodos
encontrado en sesión 2026-07-11. Cualquier mejora (3D, interactividad, reportes)
construida sobre la arquitectura `file://` hereda la misma limitación.

**Decisión arquitectónica que desbloquea todo:** servir `graphify-out/` mediante un
servidor HTTP local. Puerto: **:7779** (evita colisión con Tamp :7778, dashboard :8501,
snapshot-bridge :8765).

---

## FASE 1 — Servidor local + fetch en vivo ✅ IMPLEMENTADA (2026-07-11)

### 1.1 graphify_server.py
Servidor HTTP estático stdlib, sirve `graphify-out/` en :7779.
- `Cache-Control: no-cache` en `.json` y `.html`
- Bind en `0.0.0.0` (accesible desde Windows Chrome via WSL2 IP)

### 1.2 graph.html — fetch() dinámico
- Datos cargados via `fetch('graph.json?t=' + Date.now())`
- LEGEND computado dinámicamente desde nodos (antes era `[]` vacío)
- Stats div actualizado con conteo real fetched
- `source/target` → `from/to` mapping para vis.js

### 1.3 systemd unit
`~/.config/systemd/user/graphify.service` — mismo patrón que tamp.service.
Estado: `active (running)`, enabled.

### 1.4 URL canónica
- WSL2: `http://localhost:7779/graph.html`
- Windows Chrome: `http://172.27.33.49:7779/graph.html`
- Verificar: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7779/graph.html`

### Criterio de aceptación Fase 1 — VERIFICADO
- HTTP 200, 14KB (vs 1.4MB embebido anterior)
- `Cache-Control: no-cache, no-store, must-revalidate` en JSON y HTML
- 1686 nodes · 2753 links en graph.json
- F5 post `graphify update .` → datos frescos sin Ctrl+Shift+R
- Nodo-83 creado. 1804 tests passed, 0 failed.

---

## FASE 2 — Migración a 3D interactivo

### 2.1 Librería: 3d-force-graph

Justificación técnica:
- JS client-side puro (Three.js + WebGL) — no requiere backend adicional
- 1,686 nodos / 2,753 edges está muy dentro del rango cómodo (referencias: 10,000+ nodos)
- Soporta out-of-the-box: rotación orbital, zoom, arrastre, clic para centrar, color/tamaño por atributo
- Compatible con el formato `graph.json` existente (nodes + links con source/target) — transformación mínima

### 2.2 Plan de migración — SIN romper 2D

- Crear `graphify-out/graph3d.html` como archivo **NUEVO** separado
- `graph.html` 2D se mantiene funcionando en paralelo hasta confirmación del usuario
- Reusar `fetch('/graph.json')` de Fase 1 — no duplicar lógica
- Mapeo de atributos:
  - `community % 20` → color (Tableau-20 palette, misma que 2D)
  - `degree` → tamaño de nodo (nodos más conectados = más grandes)
  - `confidence === 'EXTRACTED'` → link más grueso/brillante

### 2.3 Criterio de aceptación Fase 2

- `http://172.27.33.49:7779/graph3d.html` carga y renderiza grafo 3D completo (1686 nodos)
- Interactivo: rotar / zoom / arrastrar con mouse
- Sidebar: búsqueda funcional + info de nodo al clic
- `graph.html` 2D original sigue funcionando sin cambios
- Rendimiento: sin lag notable al rotar (objetivo: 30+ FPS)

**PAUSA aquí** — pedir confirmación antes de Fase 3, y antes de deprecar/eliminar graph.html 2D.

---

## FASE 3 — Interactividad avanzada (post-confirmación Fase 2)

Priorizadas por valor/esfuerzo:

### 3.1 Overlay cobertura SDD (ALTO valor, bajo esfuerzo)
Cruzar nodos "archivo" contra `nodos_index.json` (Nodo-75). Huérfanos → borde rojo.
Vista visual de archivos sin Nodo sin correr `check_contradictions.py` por separado.

### 3.2 Camino entre dos nodos (ALTO valor)
`graphify path "<A>" "<B>"` existe en CLI. Traducir a UI: dos campos búsqueda + botón,
resalta visualmente la ruta. Caso de uso real: `close_snapshot_server.py` → `shadow_book.py`.

### 3.3 Leyenda y tooltip enriquecido (medio valor, bajo esfuerzo)
Panel de leyenda permanente + tooltip al hover (no solo clic) con Type/Community/Degree.

### 3.4 Filtro por tipo de nodo (medio valor)
Checkboxes: mostrar/ocultar archivos, clases, funciones por separado.
Permite ver solo estructura de archivos sin 967 métodos individuales.

---

## FASE 4 — Generación de reportes (graphify report)

### Comando CLI
```bash
graphify report --output reports/graphify_report_$(date +%Y%m%d).md
```

### Contenido del reporte (Markdown, no HTML)
1. **Top 20 nodos por grado** — centralidad, "qué archivos merecen más tests"
2. **Huérfanos SDD** — misma fuente que `nodos_index.json` (reusar, no duplicar)
3. **Resumen de comunidades** — top 3-5 archivos por comunidad (~91 comunidades)
4. **Delta contra reporte anterior** — nodos/edges nuevos, cambios en centralidad

### Criterio de aceptación Fase 4
- Genera Markdown legible sin abrir navegador
- Huérfanos coinciden exactamente con `nodos_index.json` (misma fuente de verdad)

---

## No-objetivos explícitos

- NO exponer el servidor fuera de localhost
- NO reemplazar Obsidian como vault de Nodos
- NO tocar pipeline de apuestas, `analysis/`, `edge_calculator.py`, `trader_ev_tenis.py`, `combo_confianza_builder.py`, `shadow_book.py`
- NO autenticación/multi-usuario

---

## SDD

Cada fase implementada genera su Nodo retroactivo:
- Fase 1 → Nodo-83 ✅
- Fase 2 → Nodo-84 (pendiente)
- Fases 3-4 → Nodos posteriores (pendiente)

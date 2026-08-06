---
estado: activo
---

# Nodo-105 — Knowledge Graph Navigation: Click-Highlight + MOCs + PageRank + Estado Facet

> **Wikilinks:** [[Nodo-104-Graphify-Faceted-Filter-Wikilinks-Audit]] | [[Nodo-83-Graphify-Server]] | [[Nodo-84-Graphify-3D]] | [[Nodo-85-Graphify-Fase3]] | [[Nodo-88-Graphify-Fase4-Report]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-75-Nodos-Index]]
> **Fecha:** 2026-07-16 | **Autor:** Sonnet 4.6
> **Estado:** COMPLETADO 2026-07-16 — D105-01→06 implementados
> **Marcos:** Zettelkasten (Luhmann), MOC (Obsidian community), progressive summarization (Tiago Forte)
> **Contexto:** Extensión directa de D104-01→04. El filtro de archivos funciona con OR+AND.
> Quedan 3 gaps de navegación: (1) clic en nodo no ilumina aristas, (2) MOCs del vault
> histórico son referencias fantasma sin hogar real, (3) importancia estructural de Nodos
> medida solo por grado bruto, no por PageRank (quién cita a quién importa tanto como cuántos).

---

## 1. DECISIONES

### D105-01 — Click-highlight: ego-network al clic + aristas iluminadas

Al hacer clic en un nodo en graph3d.html:
- El nodo clickeado → tier 1.0 (foco)
- Vecinos directos → tier 0.5 (semi-atenuado)
- Aristas entre foco y vecinos → `#FFD700` (gold accent) + width 2
- Todo lo demás → tier 0.07 (contexto)
- Segundo clic en el mismo nodo = deselect (toggle)
- Clic en otro nodo = cambiar foco

Reutiliza `nodeMap`, `adjacencyMap`, `getOpacityTier()` ya existentes (D104-01).
Estado: `let clickFocusId = null` — combinable con chips de archivo (OR: foco = click ∪ chips).

### D105-02 — MOCs explícitos (3 archivos, 5-10 líneas c/u)

Los wikilinks `[[MOC-Principal]]`, `[[Atlas...]]` aparecen rotos en 20+ nodos porque
referencian páginas del vault Obsidian histórico que no existen en `.spec/01_Nodos/`.
Solución: crear 3 MOCs reales como archivos `.md` — no contenido nuevo, índices navegables.

```
MOC-Identidad-Jugador.md     — Nodo-72 | Nodo-80 | Nodo-81 | Nodo-82
MOC-GCS-Grass-Signal.md      — Nodo-60 | Nodo-61 | H60-01 GRADUADA
MOC-Graphify-Visualizacion.md — Nodo-83 | Nodo-84 | Nodo-85 | Nodo-88 | Nodo-104 | Nodo-105
```

### D105-03 — PageRank sobre grafo de Nodos

El mismo algoritmo de `analysis/erdos_graph.py` (Nodo-06/20) que calcula importancia
estructural de jugadores se aplica al grafo de Nodos: qué Nodo es crítico porque está
conectado a los Nodos que importan, no solo por tener muchas conexiones brutas.

Output: `nodos_pagerank.json` — ranking estructural para orientar qué Nodos revisar
primero en sesiones futuras. Un Nodo puede tener pocas conexiones pero alto PageRank
si conecta clusters que de otro modo estarían desconectados.

### D105-04 — Facet de estado (activo / gateado / suspendido / histórico)

Frontmatter `estado:` en nodos clave → chip en graph3d.html:

```yaml
estado: activo      # trabajo vigente (Nodo-100+)
estado: gateado     # implementado, esperando gate (Nodo-82, Nodo-90 D90-11)
estado: suspendido  # pausado explícitamente (ML, Nodo-39/41)
estado: historico   # resuelto, referencia solo (Nodo-32→39)
```

Permite filtrar "muéstrame solo lo activo hoy" con un chip.

### D105-05 — Ritual de revisión de huérfanos en check_contradictions.py

Añadir al cron lunes 9am: contar Nodos sin wikilinks entrantes vs semana anterior.
Si el número sube = señal de que se escribe más rápido de lo que se conecta.
Reutiliza el script de auditoría de Nodo-104.

---

## 2. ARCHIVOS MODIFICADOS / A CREAR

| Archivo | Cambio | Estado |
|---|---|---|
| `graphify-out/graph3d.html` | D105-01: ego-network click + neural BFS fire | COMPLETADO |
| `graphify-out/graph3d.html` | D105-04: estado facet buttons + _estadoMap | COMPLETADO |
| `graphify-out/graph3d.html` | D105-06: bioluminescent sprites + bloom + importmap THREE 0.167 | COMPLETADO |
| `.spec/01_Nodos/MOC-Identidad-Jugador.md` | D105-02 | COMPLETADO |
| `.spec/01_Nodos/MOC-GCS-Grass-Signal.md` | D105-02 | COMPLETADO |
| `.spec/01_Nodos/MOC-Graphify-Visualizacion.md` | D105-02 | COMPLETADO |
| `scripts/nodo_pagerank.py` | D105-03 — 101 nodos, 380 aristas, top: Nodo-02=1.0 | COMPLETADO |
| `scripts/rebuild_nodos_index.py` | D105-04: normalización taxonomía estado | COMPLETADO |
| `check_contradictions.py` | D105-05: BLOQUE D huérfanos, baseline semanal | COMPLETADO |

## 5. LECCIONES ARQUITECTURALES (D105-06)

- `build/three.min.js` no existe en three@0.167.0+ — solo ES module via importmap
- `THREE.Color.set()` ignora alpha de rgba — usar hex colors + `linkOpacity()` para transparencia
- `renderer.render` override para composer → stack overflow si RenderPass lo llama de vuelta — solución: flag `_inComposer`
- Script de módulo (`<script type="module">`) es diferido → init() necesita polling para esperar `window.THREE`
- Golden-ratio color distribution falla con IDs dispersos (todos caen en verde) → paleta 20 colores asignada por orden de inserción
- PageRank en grafo de nodos = métrica más útil que grado bruto para identificar nodos críticos

---

## 3. ORDEN DE IMPLEMENTACIÓN

```
D105-01 (bajo riesgo, alto valor visual) → confirmar
D105-02 (archivos .md, cero código)
D105-03 (script nuevo, reutiliza función existente)
D105-05 (añadir sección a script existente)
D105-04 (último — requiere frontmatter retroactivo en nodos)
```

---

## 4. VERIFICACIÓN

```bash
# Sintaxis
node -e "const html=require('fs').readFileSync('graphify-out/graph3d.html','utf8');
         const m=html.match(/<script>([\s\S]*?)<\/script>/);
         try{new Function(m[1]);console.log('OK')}catch(e){console.error(e.message)}"

# Tests
python -m pytest tests/ --no-cov -q 2>&1 | tail -3

# PageRank
python3 scripts/nodo_pagerank.py
cat nodos_pagerank.json | python3 -c "import json,sys; d=json.load(sys.stdin); [print(f'{v:.4f} {k}') for k,v in sorted(d.items(),key=lambda x:-x[1])[:10]]"

# Visual browser (Ctrl+Shift+R)
# → clic en edge_calculator.py → aristas iluminadas gold, vecinos 0.5, resto 0.07
# → clic de nuevo en mismo nodo → deselect
# → MOCs en graph.html como nodos navegables
```

---

## 5. Addendum — D174-12 (2026-08-06): decisión explícita RETIRAR de huérfano

`scripts/nodo_pagerank.py` (D105-03, arriba) apareció en [[Nodo-174]] como
"módulo huérfano" — sin PASO en `run_daily.py`. No corresponde: es una
herramienta de mantenimiento del vault `.spec/` (calcula PageRank sobre el grafo
de wikilinks entre Nodos, igual que `graphify update .` mantiene el grafo de
código) — se ejecuta manualmente cuando se audita la salud de la documentación,
no forma parte del pipeline de trading diario (extracción→edge→trader→combos).
Conectarlo a `run_daily.py` mezclaría una tarea de higiene documental con la
orquestación de apuestas, sin beneficio. **Decisión: RETIRAR de la lista de
huérfanos — standalone intencional, no pendiente.** Sin cambio de código.

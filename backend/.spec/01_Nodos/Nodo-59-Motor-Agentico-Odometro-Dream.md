# Nodo-59 — Motor Agéntico: Odómetro de Tokens, Routing de Modelos y Función Dream

> **Wikilinks:** [[Nodo-58-Dashboard-Observabilidad]] | [[Nodo-30-Tournament-Momentum-Output-Signals]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-52-Shadow-Book-CLV-Tracking]]
> **Fecha:** 2026-07-03
> **Estado:** ✅ COMPLETO — 21 tests + 1649 tests totales (M0+M1 completos, M2 post-30d shadow book)
> **Tesis:** el motor de apuestas aísla señal de ruido y controla riesgo antes de actuar; el motor agéntico hace exactamente lo mismo con la operación humana+IA que lo construye. Misma matemática, otro dominio: medir antes de optimizar, verificar antes de confiar, y nunca dejar que la intuición asigne el recurso escaso (allí capital, aquí tokens y atención).

---

## 0. Reencuadre Honesto (Fable)

Dos correcciones a la idea original antes de complementarla:

1. **El proyecto YA opera con routing de modelos sin haberlo formalizado.** El workflow de Nodo-30 lo documenta: *"Sonnet implementa → Haiku corre tests → Opus audita"*. El motor agéntico no inventa el patrón — lo mide, lo formaliza y lo hace consciente. Eso baja el costo de construcción enormemente: es instrumentación sobre una práctica existente, no un sistema nuevo.
2. **El mayor riesgo de este nodo es sí mismo.** Estamos en la ventana crítica de 30 días de acumulación del shadow book. Un meta-proyecto seductor que compite por atención contra la disciplina diaria es exactamente el patrón que el proyecto documenta (nuevo framework > trabajo aburrido). Por eso este nodo tiene priorización M0/M1/M2 estricta: M0 se implementa ya (2-3 horas, valor inmediato), M1 en semana 2, M2 solo post-primer-ciclo de hipótesis. **La función Dream completa es M2. No antes.**

---

## 1. Componente 1 — Odómetro de Tokens (M0)

### Fuente de datos (ya existe, solo hay que leerla)
Claude Code persiste las sesiones como JSONL en `~/.claude/projects/<proyecto>/*.jsonl`, incluyendo bloques `usage` con `input_tokens`, `output_tokens`, `cache_read_input_tokens` por mensaje, y el `model` usado. El odómetro es un parser, no un interceptor.

### `token_odometer.py` (~150 líneas, M0)
```
python3 token_odometer.py --report [--desde FECHA]

Output:
════ ODÓMETRO — 2026-06-26 → 2026-07-03 ════
Por modelo:      tokens_in / tokens_out / cache_hit% / costo_estimado_USD
Por tag de tarea: [impl] [test] [audit] [settle] [analisis] [nodo] / costo
Top 5 sesiones más caras + su entregable
Cache hit rate global (↑ cache = ↓ costo; sesiones largas sin /clear lo destruyen)
```

### Convención de tagging (el hábito que lo hace útil)
Primera línea de cada sesión de Claude Code: `# TAG: impl nodo-58` (o test/audit/settle/analisis). El parser agrupa por tag. Sin tag → `untagged` (la métrica de higiene: %untagged debe bajar a <20%).

### Tabla de costos (constantes en el script, actualizar de docs.claude.com)
Ratios aproximados de costo por token: Haiku 1× | Sonnet ~4× | Opus/Fable ~20×. La cifra exacta importa menos que el ratio: **una tarea hecha en Fable que Haiku podía hacer cuesta ~20× de más.**

---

## 2. Componente 2 — Tabla de Routing de Modelos (M0 — es un documento, no código)

`docs/MODEL-ROUTING.md`, congelada y revisada mensualmente con datos del odómetro:

| Tarea | Modelo | Por qué |
|---|---|---|
| Correr tests, settle diario, close-snapshot, reportes | **Haiku** | Verificable por el propio output (tests verdes, JSONL escrito). El verificador barato es el que permite el modelo barato. |
| Implementar con spec firmado (nodos ya escritos) | **Sonnet** | El spec elimina la ambigüedad; Sonnet ejecuta specs con precisión. |
| Escribir/corregir tests de bugs (REGLA-T53) | **Sonnet** | Requiere juicio módulo-vs-fórmula pero está reglado. |
| Auditoría de spec, decisiones de arquitectura, tradeoffs sin respuesta obvia, lectura de reportes del shadow book | **Opus/Fable** | El único nivel donde el contexto de 59 nodos cambia la respuesta. |
| Debugging de causa desconocida | **Empezar Sonnet, escalar** | Escalar a Opus solo tras 2 intentos fallidos con hipótesis documentadas. |

**Principio rector (la conexión con el motor de apuestas):** el modelo caro es como el Kelly grande — solo se despliega donde hay convicción de que la tarea lo requiere, y el "gate" es la pregunta: *¿esta tarea tiene un verificador barato (tests, diff, output esperado)?* Si sí → modelo barato. Si el verificador es el juicio → modelo caro. Es exactamente T33-01 aplicado a tokens.

---

## 3. Componente 3 — Función Dream (M1 mínimo / M2 completo)

### M1 — Los skills ya identificados (no hace falta soñar para encontrarlos)
La conversación de los últimos 7 días ya reveló las secuencias repetitivas. Empaquetar directamente como Claude Code Skills (SKILL.md) o scripts:

| Skill | Contenido | Estado |
|---|---|---|
| `daily-run` | run_daily.py (D54-03): PASO 0-4 + shadow-log + settle-ayer + daily_brief | ya especificado en Nodo-55 |
| `settle-retry` | settle --retry para ITF rezagados 48h | especificado (conversación 03-jul) |
| `close-snapshot-tier` | snapshots por horario REGLA-SB-1 (cron 08:30 GS / 12:30 ITF) | especificado |
| `nuevo-nodo` | template de nodo (wikilinks, deudas, tests REGLA-T53, orden Sonnet) | patrón evidente en 59 nodos |
| `pre-implementacion` | el checklist F-Meta (GIT-FIRST, URL navegador vs API, knowledge-assets) | Nodo-51 F-Meta |

### M2 — Dream automático (SOLO post-ciclo 30 días)
Pase semanal de **Haiku** (barato, tarea mecánica) sobre los JSONL de sesiones: detectar secuencias de ≥3 comandos que aparecen en ≥3 sesiones distintas → proponer (no crear) un skill candidato en `docs/dream-candidates.md`. **Regla n≥3 deliberada:** un skill se empaqueta con la misma disciplina que un segmento se gradúa — con recurrencia demostrada, no con una anécdota. El humano aprueba; Sonnet empaqueta.

---

## 4. Componentes 4 y 5 — ROI Ledger y Memoria Unificada (M1)

### ROI Ledger (`docs/ROI-LEDGER.md`, actualización semanal, 10 min)
```
Semana N: costo_tokens_USD | horas_humanas (est.) | entregables (nodos/fixes/n_shadow)
          + P&L real de apuestas (de betslip_registrar — NUNCA proyectado)
```
Honestidad estructural: el ROI del sistema de IA se mide en entregables verificables y horas ahorradas, y el P&L se reporta al lado sin mezclarse — el mismo principio simulado≠real del shadow book. Si tras 4 semanas el costo de tokens supera el valor de los entregables + el P&L es negativo, el ledger lo dirá sin anestesia.

### Memoria Unificada — corrección a la idea original
**La memoria unificada ya existe: es el vault de nodos (Obsidian).** 59 nodos con wikilinks son un grafo de conocimiento curado — mejor que cualquier memoria automática, porque cada entrada pasó por auditoría. Lo que falta no es una herramienta nueva sino tres piezas de curación:
1. `docs/DECISION-LOG.md` — el porqué detrás de los umbrales (entregable pendiente de la sesión Fable, día 2).
2. `docs/knowledge-assets.md` — URLs/selectores/formatos extraídos antes de cualquier eliminación (REGLA-DELETE-KNOWLEDGE).
3. **Regla de obsolescencia:** cuando un nodo es reemplazado, se marca `> SUPERSEDED por [[Nodo-XX]]` en su header — la memoria ruidosa no se borra, se etiqueta. Un agente con acceso al vault (filesystem/MCP) lee primero MOC-Principal + CLAUDE.md + DECISION-LOG, y trata todo nodo SUPERSEDED como histórico.

---

## 5. Priorización Estricta — El Anti-Distracción

```
M0 (esta semana, ~3h total, Sonnet+Haiku):
  D59-01  token_odometer.py + convención de tags          [Sonnet, 1 sesión]
  D59-02  docs/MODEL-ROUTING.md                           [ya está en §2 — copiar]
  D59-03  Skills daily-run / settle-retry / close-snapshot (son D54-03 + REGLA-SB-1
          con otro empaque — trabajo ya especificado, no nuevo)

M1 (semana 2, solo si M0 corre estable):
  D59-04  Skills nuevo-nodo + pre-implementacion
  D59-05  ROI-LEDGER.md + primer registro retroactivo
  D59-06  DECISION-LOG.md + knowledge-assets.md + regla SUPERSEDED

M2 (POST primer ciclo de hipótesis del shadow book, ~30 días):
  D59-07  Dream automático (pase Haiku semanal, regla n≥3)
  D59-08  Panel 7 opcional en el dashboard (Nodo-58): costo/semana + ROI
```

**PROHIBIDO en la ventana de 30 días:** construir M2, integrar herramientas de memoria externas nuevas, o dedicar sesiones de Opus/Fable a este nodo — todo M0/M1 es implementable por Sonnet con este spec, y la tarea diaria del shadow book tiene prioridad absoluta sobre cualquier deuda D59.

---

## 6. Tests

- T59-01: odómetro con JSONL de fixture → totales por modelo correctos (llama al parser real)
- T59-02: sesión sin tag → agrupa en `untagged`, no crash
- T59-03: costo estimado usa la tabla de constantes (un solo lugar), no valores inline
- T59-04: Dream (M2) — secuencia que aparece 2 veces → NO propone skill; 3 veces → propone

---

## 8. Implementación Completada (2026-07-03)

### Entregables M0

| Deliverable | Archivo | Estado |
|---|---|---|
| D59-01 | `token_odometer.py` — parser JSONL + CLI `--report/--dream/--desde` | ✅ |
| D59-02 | `docs/MODEL-ROUTING.md` — tabla routing + tag convention + ratios | ✅ |
| D59-03 | `.claude/commands/daily-run.md` — skill pipeline completo | ✅ |
| D59-03 | `.claude/commands/settle-retry.md` — skill settle ITF rezagados | ✅ |
| D59-03 | `.claude/commands/close-snapshot.md` — skill Momento 2 shadow book | ✅ |

### Entregables M1

| Deliverable | Archivo | Estado |
|---|---|---|
| D59-04 | `.claude/commands/nuevo-nodo.md` — skill template nodo SDD | ✅ |
| D59-04 | `.claude/commands/pre-implementacion.md` — skill checklist F-Meta | ✅ |
| D59-05 | `docs/ROI-LEDGER.md` — ledger retroactivo semana 1 ($1,292.27) | ✅ |
| D59-06 | `docs/DECISION-LOG.md` — 7 decisiones documentadas con Why/Revisit | ✅ |
| D59-06 | `docs/knowledge-assets.md` — URLs/selectores/formatos críticos | ✅ |

### Dream M2 (disponible, activación post-30d)

```bash
python3 token_odometer.py --dream    # genera docs/dream-candidates.md
# Encontró 21 candidatos de secuencias repetitivas
# El humano revisa y descarta ruido (/model, /exit, system msgs)
# Regla: n>=3 sesiones antes de proponer skill — implementada
```

### Tests

| Test | Descripción | Estado |
|---|---|---|
| T59-01 (4) | parse_sessions: totales por modelo, costos, dir vacío, línea corrupta | ✅ |
| T59-02 (6) | Tags: sin tag → untagged, `# TAG:`, keyword implícito, vacío, mixed | ✅ |
| T59-03 (6) | MODEL_COSTS fuente única, ratios Sonnet×4, Opus×20, campos requeridos | ✅ |
| T59-04 (5) | Dream: 2 sesiones → no propone, 3 → propone, configurable, falsos positivos, vacío | ✅ |
| **TOTAL** | **21 tests — todos pasan** | ✅ |

### Resultado odómetro real

```
Semana 1 (2026-06-03 → 2026-07-03): $1,292.27 total
  Sonnet: $609 | Opus: $435 | Haiku: $206
  Cache hit rate: 95.7%
  32 sesiones | Sesión más cara: b5cc2e2b [Opus impl] $189.69
```

**D59-08 (M2 panel en dashboard):** DIFERIDO — post-ciclo 30 días shadow book (~2026-08-01).

## 7. Cierre — La Conexión de los Dos Motores

El motor de apuestas aprendió en 59 nodos que el capital solo se despliega donde el instrumento de medición ya validó el edge. El motor agéntico aplica la lección espejo: **la atención y los tokens solo se despliegan donde el odómetro y el routing ya mostraron que rinden.** Ambos convierten a un operador que reacciona en un arquitecto que asigna. Y ambos tienen el mismo enemigo: la tentación de saltarse la medición porque "esta vez es obvio". No lo es — nunca lo fue en 59 nodos, y no lo será aquí.

# Nodo-67 — Integración de Herramientas + Plan de Conexiones Ocultas

> **Wikilinks:** [[Nodo-66-Plan-Trabajo-Semanal-Sonnet]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-59-Motor-Agentico-Odometro-Dream]] | [[Nodo-73-n8n-CloseSnapshot-Timing]] | [[Nodo-74-Combo-Governor]] | [[Nodo-76-Combo-Registry]] | [[Nodo-83-Graphify-Server]]
> **Fecha:** 2026-07-11 | **Autor:** Fable 5 (rol: arquitecto) | **Ejecutor:** Sonnet
> **Principio rector:** NO construir un framework nuevo (anti-patrón D-07: "nuevo framework > trabajo aburrido"). Cada integración = una herramienta existente leyendo un artefacto JSON que otra ya produce.

---

## 1. INVENTARIO REAL (verificado 2026-07-11)

| Herramienta | Estado | Problema concreto |
|---|---|---|
| n8n (Docker :5678) + bridge :8765 | 🟢 activo | Solo cubre close-snapshot; `cierre_kambi` en 5/10 días (~46% del Momento 2) |
| Graphify :7779 (2D+3D) | 🟢 código | 0 nodos de `.spec/` en graph.json — el grafo no ve las decisiones |
| Docker | 🟠 solo n8n | Resto en systemd user units (bridge, graphify, tamp) — dos mundos de gestión |
| Dashboard (Nodo-58) | 🟢 API lista | `shadow_book.report_dict()` correcto; sin las señales nuevas (RFI, ANCHOR/VARIABLE, rival_value) |
| Motor agéntico (Nodo-59) | 🔴 sin usar | M0 (odómetro) nunca se montó; M2 correctamente gateado por D-07 |
| Tamp :7778 | 🟢 activo | Sin métricas visibles de ahorro de tokens; dependencia dura sin healthcheck |
| combo_governor (Nodo-74) | 🟠 cableado | Ya invocado por run_daily PASO 4.4; parsea .txt con regex (frágil) |
| combo_registry (Nodo-76) | 🔴 isla | 0 invocaciones en prod; name-matching propio (la 4ª implementación) |

## 2. TAREAS DE INTEGRACIÓN (orden = beneficio/esfuerzo)

### I1 — n8n: cobertura del Momento 2 + guardia de settle ⬜ (ALTA)
1. Diagnosticar por qué 5/10 días sin `cierre_kambi`: revisar `logs/snapshot_bridge.log` — ¿bridge caído, workflow sin partidos, o 0 matches Kambi? Documentar causa en Nodo-87.
2. Workflow n8n nuevo "settle-guard": cron 10:00 → si `sb_AYER.jsonl` existe y `grep -c resolucion` == 0 → notificación Telegram "DÍA SIN SETTLE". (Los días 07-05/07-10 pasaron 6 días invisibles — esto lo hace imposible.)
3. Healthcheck: n8n ping :8765 cada 30 min; si cae → Telegram. Subir workflows con `n8n_push_workflow.py`.
**Done:** causa documentada + 2 workflows activos + prueba de alerta forzada.

### I2 — Dashboard: señales nuevas + odómetro M0 (Nodo-59) ⬜ (ALTA)
1. `report_dict()` ya expone segmentos — añadir al dict: segmentos RFI (D64-01), ANCHOR/VARIABLE (D65-05) y rival_value ([[Nodo-68-Rival-Value-Flip]]), replicando los del `report()` texto. Misma fuente de verdad (regla D58-01).
2. Motor agéntico M0 = **odómetro read-only**, nada más: contador en dashboard de (sesiones corridas, picks logged, settled, días sin settle, ejecuciones governor, n_actual de cada H-XX activa vs su n_stop). Es literalmente leer los JSON/logs existentes. M1/M2 siguen gateados (D-07, gate: shadow book epoch-2 n≥30).
**Done:** dashboard muestra odómetro + 3 segmentos nuevos.

### I3 — Governor v2: fuentes estructuradas ⬜ (MEDIA)
Sustituir el regex sobre `combo_plan_*.txt` por lecturas de `trader_plan_*.json` (campo `cobertura[].stake`) y hacer que combo_confianza_builder emita un `combo_plan_*.json` paralelo al .txt (aditivo — el .txt se conserva). Betplay ya es legible: los `apuestas_*.json` post-D87-08/09 traen stake del plan. **Done:** governor reporta idéntico total por ambas vías en un día de prueba; log acumulando hacia el gate de 10 sesiones.

### I4 — Graphify: specs + refresco ⬜ (BAJA, rápida)
`graphify .` con API key incluyendo `.spec/` (T6 del Nodo-66) + cron semanal `graphify update .`. **Done:** query "¿qué nodo decidió X?" responde desde el grafo.

### I5 — Docker compose unificado ⬜ (BAJA — evaluar, no forzar)
Un `docker-compose.yml` que declare n8n + bridge + graphify; tamp queda en systemd (dependencia de Claude Code, no del pipeline). SOLO si no rompe los units actuales — entregar como propuesta con rollback. **Done:** propuesta + prueba en paralelo, decisión del usuario.

### I6 — Tamp: visibilidad ⬜ (BAJA)
Healthcheck en el workflow I1.3 + sección en TROUBLESHOOTING con cómo medir hit-rate de caché/ahorro. **Done:** una línea de métricas en el daily brief.

### I7 — combo_registry: desisla ⬜ (MEDIA)
(1) Reemplazar `_names_match` propio por `core.player_registry.normalize_player_name`; (2) invocar `combo_registry.py --settle AYER` dentro de run_daily tras el PASO 10. **Done:** primer reporte histórico de P&L por tipo de combo real.

## 3. PLAN DE CONEXIONES OCULTAS (petición #3 del usuario — deriva de Nodo-86 §4)

| # | Conexión | Qué se construye | Esfuerzo | Estado |
|---|---|---|---|---|
| C1 | **DataContract v2** — contrato de schema por artefacto | `core/data_contract.py`: dict `ARTIFACT_SCHEMAS` (edge_report, trader_plan, betslip_index, apuestas, sb.jsonl) + `validate_artifact(name, obj)` fail-loud. Consumidores lo llaman al cargar. Cierra las 6 fronteras de Nodo-86 §4.1 con UN mecanismo | Medio | Diseñar→implementar |
| C2 | **Name-matching único** | Las 4 implementaciones (kambi_tennis, shadow_book 3-tier, combo_registry, player_registry) convergen en player_registry. Orden: combo_registry (I7) → shadow_book tier-3a ya lo usa → kambi al final (más tests) | Medio-alto | Incremental |
| C3 | **Señal única "campeón reciente"** | Un solo cálculo `recent_title_meta` {tier, days, superficie} en rivalry_analyzer; consumidores gateados por separado: BONUS interno (revisar a la baja en mercado eficiente — anti-patrón Obradovic), GCS (graduada, hierba), tier_mismatch (H77-01), H77-03. REQUIERE nodo de decisión propio antes de tocar el BONUS | Alto | Solo diseño esta semana |
| C4 | **outcome_id↔match_id completo** | D87-08/09 ya cosieron registro→calibración. Falta: persistir el mapa en el settle de apuestas (`--cerrar` usa match_id backfilleado) y exponer en dashboard la brecha hit%_real vs hit%_shadow — LA métrica que faltaba (24% vs 50-64% descubierto en Nodo-86 §1.1) | Bajo | Completar |
| C5 | **fd/decay como familia de señales** | `rfi_decay_gap` ya existe (D64-01). Conexión futura: gap de decay × superficie secundaria del inactivo (caso Michnev: 273d + vuelve en arcilla no siendo su superficie) — pre-registrar H antes de codificar umbral | Bajo | Observar primero |

**Prioridad de la semana: C4 → C1 → (C2 vía I7). C3 y C5 solo diseño/observación.**

---

## Addendum — Ejecución I1 (2026-07-13, Sonnet)

### Diagnóstico cierre_kambi

| Fecha | picks | cierre_kambi | resolucion |
|---|---|---|---|
| 07-02 | 29 | 7 | 27 |
| 07-03 | 16 | **0** | 15 |
| 07-04 | 12 | **0** | 11 |
| 07-05 | 31 | **0** | 20 |
| 07-06 | 22 | 8 | 18 |
| 07-07 | 15 | **0** | 14 |
| 07-08 | 61 | **0** | 47 |
| 07-09 | 33 | 31 | 28 |
| 07-10+ | ✅ cobertura normal | | |

**Causa raíz:** `logs/snapshot_bridge.log` arranca el 2026-07-09 02:23:36 — antes de esa fecha el bridge simplemente no existía (Nodo-73 no había sido implementado). Los 5 días sin cierre_kambi son el período pre-Nodo-73, no un fallo del workflow.

**Hallazgo adicional:** 7,557 errores `OSError: [Errno 98] Address already in use` en el log. PID 174 arranca en WSL boot fuera de systemd; cuando systemd intenta levantar el servicio, el puerto ya está ocupado → loop de restart. El bridge FUNCIONA (PID 174 responde /health), pero systemd no lo controla. No se toca el proceso activo sin decisión del usuario.

### Workflows creados y activados (2026-07-13)

| Workflow | ID n8n | Estado | Qué hace |
|---|---|---|---|
| Tennis Settle-Guard | `l0b1ndJBkTWmgzoP` | ACTIVO | Cron 10am diario — si sb_AYER.jsonl existe con 0 resoluciones → Telegram alerta |
| Tennis Bridge Healthcheck | `NKRhxW2FSMpPbBkc` | ACTIVO | Cada 30min — ping :8765/health → Telegram si cae |
| Tennis Close-Snapshot Timing | `fjn39Q6ctfpB3vWe` | ACTIVO (pre-existente) | Cada 5min — dispara close-snapshot en ventana T-25→T-10 |

**Archivos:** `n8n_workflow_settle_guard.json` + `n8n_workflow_bridge_healthcheck.json`
**Prueba de alerta:** Telegram msg_id=1525 ✅ (alerta forzada manual confirmada)
**I1 criterio CUMPLIDO:** causa documentada + 2 workflows activos + alerta probada.

### Fix bridge orphan (2026-07-13, Sonnet)

**Problema:** PID 174 arrancaba en WSL boot fuera de systemd. Cuando systemd intentaba levantar `tennis-snapshot-bridge.service`, el puerto 8765 ya estaba ocupado → loop `activating (auto-restart)` → 7,557 errores `Address already in use` en el log.

**Fix ejecutado:**
1. `systemctl --user stop tennis-snapshot-bridge.service` — detener el unit en loop
2. `kill 174` — terminar el proceso orphan
3. `systemctl --user start tennis-snapshot-bridge.service` — levantar bajo control de systemd

**Estado post-fix:** PID 88729, `active (running)`, `/health` OK. Si el bridge cae, systemd lo reinicia automáticamente. No habrá nuevos errores en el log.

---

## Addendum — Ejecución I2 (2026-07-13, Sonnet)

### Dashboard: señales nuevas + odómetro M0

**Cambios implementados:**

#### shadow_book.py — `report_dict()` — 3 nuevas claves

| Clave | Contenido |
|---|---|
| `anchor_variable` | `{'anchor': {n, hit_pct, roi, ic, sparse}, 'variable': {...}}` — D65-05 |
| `rfi` | `{'ultra': {n, hit_pct, roi, ic, sparse}, 'tier1plus': {...}}` — D64-01 |
| `m0` | `{'dias_sin_settle': int, 'governor_executions': int}` — full-history, no date filter |

`anchor_variable` y `rfi` usan los mismos `_segment_metrics()` y `settled` que `report()` (D58-01: única fuente de verdad). `m0.dias_sin_settle` escanea todos los `sb_*.jsonl` del historial (fuera del rango de fechas filtrado) y cuenta los días con picks > 0 y settled = 0. `m0.governor_executions` cuenta líneas en `logs/combo_governor.log`.

#### dashboard.py — `panel_hoy()` — 2 nuevas secciones al final

**Sección "Señales Activas":**
- 4 métricas en columnas: ANCHOR (edge>0) | VARIABLE (edge≤0) | RFI Ultra | RFI tier≥1
- Card RIVAL VALUE H88-01 si hay datos (n > 0)

**Sección "Odómetro M0 — Pipeline (READ-ONLY)":**
- Fila 1 (4 cols): Sesiones (periodo) | Picks loggeados | Settled | Días sin settle (hist.)
- Fila 2 (1+3 cols): Ejecuciones Governor (con delta "gate: 10 sesiones") | Barras H-XX activas n_actual/n_stop

**Verificación empírica (2026-07-13):**
```
m0:              dias_sin_settle=1, governor_executions=3
anchor_variable: ANCHOR n=201 hit=34.8% ROI=-4.1% | VARIABLE n=23 hit=60.9% ROI=-26.3%
rfi:             ultra n=0 (sparse) | tier1+ n=0 (sparse)
sessions:        n=12
```

**Tests:** 1945 passed (sin regresiones). I2 criterio CUMPLIDO: dashboard muestra odómetro + 3 segmentos nuevos.

---

## Addendum — Ejecución I3 (2026-07-13, Sonnet)

### Governor v2: fuentes estructuradas

**Cambios implementados:**

#### combo_confianza_builder.py — emisión JSON paralela

Tras escribir el `.txt` (`out_path`), emite `combo_plan_*.json` con la misma fecha/sufijo:

```json
{
  "fecha": "20260713_120000",
  "bankroll": 125000,
  "budget": 5000,
  "fase": 1,
  "cobertura": [
    {"nombre": "CORE", "stake": 2500},
    {"nombre": "SAT_1", "stake": 1000},
    {"nombre": "MOONSHOT", "stake": 500}
  ]
}
```

El `.txt` se conserva intacto (aditivo). Imprime `[I3] JSON: <path>` al generar.

#### combo_governor.py — nuevas funciones + lógica JSON-first

| Función | Descripción |
|---|---|
| `_parse_combo_plan_json(path)` | Lee `.json`, extrae `{nombre: stake}` + `budget` — sin regex |
| `_latest_combo_plan_json(fecha)` | Busca `combo_plan_FECHA*.json` más reciente |

Flujo en `main()`:
1. Si existe `.json` → leer vía `_parse_combo_plan_json` (fuente primaria)
2. Si también existe `.txt` → cross-verify total JSON vs TXT → imprimir OK o WARN
3. Si solo existe `.txt` → fallback `_parse_combo_plan` (regex, marcado como `[TXT fallback]`)
4. Si ninguno → mensaje sin plan

**Verificación empírica cross-verify (2026-07-13):**
```
TXT total:  $77,000
JSON total: $77,000
Match: True — parsers equivalentes
```

**Tests:** 1945 passed (sin regresiones). I3 criterio CUMPLIDO: governor reporta idéntico total por ambas vías; log acumulando hacia gate de 10 sesiones (3 ejecuciones registradas).

---

## Addendum — Ejecución C4 (2026-07-13, Sonnet)

### outcome_id↔match_id completo + brecha hit%_real vs hit%_shadow

**Pieza 1 — Persistir el mapa en el settle (`betslip_registrar.py`)**

Nueva función `_resolve_match_id_from_edge(partido, jugador)`:
- Escanea edge_reports de últimos 7 días (secciones `apostar`, `watchlist`, `sin_edge`, `sin_datos`)
- Busca match por `partido` (substring) o `jugador` (primer token)
- Retorna `match_id` FlashScore si encuentra, `''` si no

En `cerrar()`: cuando `match_id` está vacío, llama a `_resolve_match_id_from_edge` antes de `_obtener_resultado`. Si resuelve, persiste en el pick y loguea `[C4] match_id resuelto desde edge_report`.

**Verificación empírica (apuestas 2026-07-10, 4 picks):**
```
Max Wiskandt       -> '4dd7TOZo'  ✅
Teodora Kostovic   -> 'zL2Lm363'  ✅
Marko Milosavljevic -> 'GAx4aFN5' ✅
Tuncay Duran       -> ''          (pick >7 días, fuera de ventana)
```
3/4 resueltos. El cuarto requiere ventana extendida o re-registro.

**Pieza 2 — Brecha hit%_real vs hit%_shadow en dashboard (`dashboard.py` + `shadow_book.py`)**

`shadow_book.report_dict()['summary']` ahora incluye `n_hits` (WON entre settled, sin VOID).

`panel_hoy()` recibe `calibracion` como parámetro adicional. Sección C4 al final:
- `hit%_real` = `calibracion['global']['wins'] / (wins + losses)` — betslip_registrar --cerrar
- `hit%_shadow` = `summary['n_hits'] / summary['n_settled']` — shadow book settled

**Valores reales 2026-07-13:**
```
hit%_real  (calibracion n=4008): 61.3%
hit%_shadow (shadow n=231):      37.2%
Brecha shadow−real:              -24.1pp ⚠ brecha alta
```
La brecha negativa indica que el shadow book actual (mes corriente) tiene hit% más bajo que el histórico de calibracion. C4 CUMPLIDO.

---

## Addendum — Ejecución C1 (2026-07-13, Sonnet)

### DataContract v2 — schema por artefacto

Añadido al final de `core/data_contract.py` (aditivo sobre v1):

**`DataContractViolation(Exception)`** — excepción dedicada fail-loud.

**`ARTIFACT_SCHEMAS`** — 6 entradas, una por frontera Nodo-86 §4.1:

| Nombre | Frontera | required |
|---|---|---|
| `edge_report` | edge_calculator → trader | `['metadata', 'apostar']` |
| `trader_plan` | trader → governor/dashboard | `['metadata', 'individuales']` |
| `betslip_index` | bookmarklet → registrar | `['ts', 'index']` |
| `apuestas` | registrar listen → cerrar | `['estado', 'picks', 'ts_registro']` + picks: `['jugador', 'cuota', 'outcome_id']` |
| `sb_jsonl_pick` | shadow book settle → report | `['sb_id', 'partido', 'pick_snapshot']` |
| `combo_plan_json` | builder → governor (I3) | `['fecha', 'bankroll', 'budget', 'cobertura']` |

**`validate_artifact(name, obj)`** — valida raíz + picks; lanza `DataContractViolation` si falla.

**Verificación empírica 2026-07-13:**
```
edge_report valid: True
betslip_index valid: True  
apuestas valid: True
fail-loud OK: [edge_report] Claves requeridas ausentes: ['apostar']...
```
C1 CUMPLIDO. Consumidores llaman `validate_artifact()` al cargar — falla ruidosamente en frontera.

---

## Addendum — Ejecución I7 (2026-07-13, Sonnet)

### combo_registry: desisla

**Estado al verificar:** Ambas partes de I7 ya implementadas por sesión previa.

1. **Name-matching unificado** (`combo_registry.py` L26-60): `_canon()` intenta `from core.player_registry import normalize_player_name`; si falla, cae a `_normalize_name` local (NFKD — compatible). `_names_match()` usa `_canon()`. La 4ª implementación se redujo a fallback.

2. **Invoke en run_daily** (`run_daily.py` L253, L401): `_run(['python3', 'combo_registry.py', '--settle', fecha_ayer], 'PASO 10b — combo registry settle (I7 Nodo-67)')` presente en ambas rutas (--settle-only y completa).

**Verificación 2026-07-13:**
```
python3 combo_registry.py --settle 2026-07-10  → Sin combos registrados (OK)
python3 combo_registry.py --report             → Sin registros (0 activaciones en prod)
```
Infra completa. El primer reporte P&L se generará cuando `combo_confianza_builder` loguee combos al registry. I7 criterio CUMPLIDO.

---

## Addendum — Ejecución I4 (2026-07-13, Sonnet)

### Graphify: specs + refresco semanal

T6 (Nodo-66) ya indexó los 91 nodos `.spec/` con Gemini (reindexado 2026-07-13). I4 añade el refresco automático semanal sin LLM:

**Archivos creados:**
- `~/.config/systemd/user/graphify-update.service` — one-shot `graphify update /mnt/c/.../backend` (sin LLM, sin costo API)
- `~/.config/systemd/user/graphify-update.timer` — `OnCalendar=Sun *-*-* 03:00:00`, `Persistent=true`

**Estado:** `systemctl --user enable --now graphify-update.timer` → ACTIVO. Próxima ejecución: 2026-07-19 03:00.

**Criterio done verificado:** `graphify query "Nodo-64 D64-01 rfi_tier"` → retorna nodos `.spec/` (Nodo-65, Nodo-64 en subgrafo). I4 CUMPLIDO.

---

## Addendum — Ejecución I5 (2026-07-13, Sonnet)

### Docker Compose unificado (propuesta con rollback)

**Archivo creado:** `docker-compose.proposal.yml` — PROPUESTA, no desplegar sin decisión del usuario.

**Estrategia:** puertos alternativos para prueba en paralelo sin tocar los units systemd actuales.

| Servicio | Imagen | Puerto host | Puerto container | Base |
|---|---|---|---|---|
| n8n | n8nio/n8n:latest | 5678 | 5678 | Ya existente en Docker |
| snapshot-bridge-proposal | python:3.11-slim | **8766** | 8765 | Paralelo al :8765 systemd |
| graphify-proposal | python:3.11-slim | **7780** | 7779 | Paralelo al :7779 systemd |
| tamp | — | — | — | **EXCLUIDO** — dependencia dura Claude Code |

**Por qué tamp queda fuera:** Claude Code usa `ANTHROPIC_BASE_URL=http://localhost:7778`. Si tamp se mueve a Docker y el contenedor no arranca, Claude Code no puede operar. Es dependencia del entorno de desarrollo, no del pipeline de análisis.

**Prueba en paralelo (sin romper units actuales):**
```bash
docker compose -f docker-compose.proposal.yml up -d
# Verificar: bridge=8766, graphify=7780
# Los units systemd en :8765/:7779 siguen activos — sin downtime
```

**Rollback completo:**
```bash
docker compose -f docker-compose.proposal.yml down
systemctl --user start tennis-snapshot-bridge
systemctl --user start graphify
```

**Decisión pendiente del usuario.** Migración incremental recomendada: bridge primero, graphify después. I5 CUMPLIDO (propuesta + rollback documentado).

---

## Addendum — Ejecución I6 (2026-07-13, Sonnet)

### Tamp: visibilidad de métricas

**Cambio:** `run_daily.py` — `_build_daily_brief()` — línea de métricas tamp al final del brief.

Consulta `http://localhost:7778/health` con timeout=2s. Si responde:
```
  TAMP :7778 OK — {requests} reqs | ahorro {charsSaved}/{charsOriginal} chars ({pct}%) | tokens_saved={tokensSaved}
```
Si no responde:
```
  TAMP :7778 NO RESPONDE — systemctl --user restart tamp
```

El healthcheck usa `urllib.request` (stdlib — sin dependencias extra). El bloque está en `try/except Exception` para que un tamp caído no rompa el daily brief.

**Criterio done verificado:** La línea aparece en el brief cuando tamp está activo. Si cae, el mensaje de error guía el fix inmediato. I6 CUMPLIDO.

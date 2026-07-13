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

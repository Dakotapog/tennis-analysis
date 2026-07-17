# Nodo-66 — Plan de Trabajo Semanal para Sonnet (post-auditoría Fable 5)

> **Wikilinks:** [[Nodo-86-Auditoria-Fable5]] | [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]] | [[Nodo-67-Integracion-Herramientas]] | [[Nodo-68-Rival-Value-Flip]]
> **Fecha:** 2026-07-11 | **Autor:** Fable 5 | **Ejecutor:** Sonnet (semana 2026-07-13 → 07-19)
> **Nota numeración:** los archivos Nodo-66/67/68 no existían (el rango F4 "64-71" vive solo en código/tests). Números asignados por decisión del usuario 2026-07-11.
> **Estado verificado al escribir esto:** 1804 tests passed POST-fixes D87-01→D87-11 + D64-01. Cambios SIN COMMIT en main.

---

## 0. LECTURA OBLIGATORIA EN ORDEN (antes de tocar nada)

1. `CLAUDE.md` completo (es VISTA derivada — si contradice un nodo, el nodo gana)
2. `.spec/01_Nodos/Nodo-86-Auditoria-Fable5.md` → luego `docs/auditorias/AUDITORIA_FABLE5_2026-07-11.md` (6 secciones, evidencia archivo:línea)
3. `docs/DECISION-LOG.md` (D-01→D-09, E-01→E-05, C-01→C-07)
4. `validation/preregistered_hypotheses.json` — **PROHIBIDO modificar umbrales de hipótesis existentes** (añadir hipótesis NUEVAS sí está permitido)
5. Memoria Claude del proyecto: `auditoria-fable5-fixes-pendientes` (lista exacta de tests a escribir)

## 1. VACÍOS ANTICIPADOS — errores que Sonnet va a cometer si no lee esto

| # | Trampa | Regla |
|---|---|---|
| V1 | "Este `min(p_prior, p_modelo)` en `_p_blend` parece raro, lo arreglo" | **NO REVERTIR ningún fix D87-xx.** Cada uno tiene justificación en Nodo-86 con evidencia. Si algo parece mal → preguntar al usuario, no "corregir" |
| V2 | Escribir tests hardcodeando la fórmula esperada | REGLA-T53: el test invoca la función real del módulo, nunca reimplementa la fórmula |
| V3 | Tratar CLAUDE.md como fuente de verdad | Es vista derivada (§10 precedencia). La fuente son los nodos |
| V4 | Editar nodos existentes | Historia inmutable — se añade nodo nuevo o adendo fechado |
| V5 | Confundir los DOS sizings | `kelly_kl` (edge_calculator, 5 capas) solo decide `apostar`; el stake real usa `_kelly_quarter(p_blend)` en trader_ev_tenis. Son sistemas distintos — Nodo-86 §1.4 |
| V6 | Campos del pick_snapshot | Es el pick del edge_report: `favorito_predicho`, `ranking_favorito`, `cuota_favorito`, `cuota_rival`. NO existen `nombre`/`jugador`/`ranking` |
| V7 | Correr pytest desde PowerShell | Python solo existe en WSL (`venv` activado). PowerShell = solo lectura/edición |
| V8 | Migrar los buckets `?`/`?_?` de calibracion_edge.json | ~141 resultados de dinero real — SOLO preparar propuesta (T7), el usuario decide |
| V9 | "Limpiar" registros del shadow book | Append-only, inmutable (D-06). `settle()` solo AÑADE `resolucion` |
| V10 | Mezclar evidencia retrospectiva con prospectiva | Lección C-05. Todo backtest se etiqueta RETROSPECTIVO y no actualiza calibración |

## 2. CHECKLIST DE TAREAS (orden de ejecución, con criterio de done)

### T1 — Commit del trabajo verificado ⬜
Si el usuario no commiteó aún: un commit con todos los cambios D87+D64-01+nodos 66/67/68/86 + auditoría. Mensaje referenciando Nodo-86. **Done:** `git log` lo muestra; working tree limpio.

### T2 — Nodo-87: documentación de los fixes ⬜
Crear `.spec/01_Nodos/Nodo-87-Fixes-Auditoria-D87.md`: tabla D87-01→D87-11 + D64-01 con (decisión | archivo:línea | bug que cierra | referencia Nodo-86 §). Los IDs ya están en comentarios del código — grep `D87-` y `D64-01`. Añadir entrada `D-10` en DECISION-LOG.md resumiendo el lote. **Done:** archivo existe, wikilinks correctos, D-10 en el log.

### T3 — Tests REGLA-T53 para los caminos corregidos ⬜ (LA TAREA MÁS IMPORTANTE)
Los bugs vivieron meses porque nada los vigilaba — 1804 tests y CERO cubrían estos caminos. Crear `tests/test_nodo87_fixes.py`:
1. `_print_individuales`: kelly=0 → stake 0 (no $1,000); budget agotado → stake 0
2. `_p_blend(0.56, 0, 0.758)` → usa min(0.758, 0.56)=0.56; con prior<p_modelo comportamiento intacto
3. Gate GCS: pick con `motivo_reclasificacion` T33-01 + gcs_active + edge 0.20 → `apostar` sigue False
4. `update_alpha_flags` con snapshot `{'favorito_predicho': 'X'}` → marca `combo_flags.alpha_promoted`
5. Campos `rfi_*`: fixture con `form_decay_meta` p1 273d/p2 8d, cuotas 1.17/4.35, favorito=activo → `rfi_tier=2`, `rfi_ultra=True`, `rfi_decay_gap>1`
6. `settle()`: dos resultados del mismo día con el mismo favorito, distinto rival → settlea contra el partido cuyo RIVAL coincide
7. `pre_game_validator.validate_file`: edge_report real (claves apostar/watchlist/sin_edge) → valida los 3 pools
8. `_save_betslip_index`: pick cuota 1.20 SÍ entra al index; entry contiene p_modelo/kelly_kl
9. `_backfill_desde_edge`: pick superficie='?' + edge_report con el jugador → superficie/tier/p_modelo rellenados
**Done:** pytest verde con los nuevos tests incluidos.

### T4 — Re-sync CLAUDE.md ⬜
Conteo de tests nuevo, Nodo-87 en estado, fila de auditoría actualizada. **Done:** `check_contradictions.py --quick` limpio.

### T5 — `python3 scripts/rebuild_nodos_index.py` ⬜
**Done:** nodos_index.json contiene Nodo-83→87 + 66/67/68.

### T6 — Graphify con specs ⬜
`export ANTHROPIC_API_KEY=... && graphify .` incluyendo `.spec/`. **Done:** `grep -c '\.spec' graphify-out/graph.json` > 0.

### T7 — Propuesta migración bucket `?` (NO ejecutar) ⬜
Informe corto para el usuario: opciones (a) mover a bucket explícito `real_money_unknown`, (b) intentar re-atribuir por fecha+jugador contra h2h_results históricos, (c) dejar y solo excluir. Con pros/contras. **Done:** `docs/PROPUESTA_MIGRACION_BUCKET_Q.md` existe; nada ejecutado.

### T8 — Verificación empírica del loop nuevo ⬜
Correr PASO 3 sobre el h2h más reciente → confirmar que el edge_report contiene `rfi_tier`/`rfi_ultra`/`rfi_decay_gap` y que `shadow_book.py --report` muestra los segmentos RFI. **Done:** output pegado en Nodo-87 como evidencia.

### T9 — Deuda de settle ⬜
`shadow_book.py --settle 2026-07-05` y `--settle 2026-07-10` (+ `/settle-retry` si ITF rezagado). Hacer autoritativo el backfill de H54-01 (nota en el JSON: settle de Jul 3/8/9). **Done:** 0 días con settled=0 en el rango 07-02→07-11.

### T10 — Nodos 67 y 68 ⬜
Ejecutar sus checklists (ver [[Nodo-67-Integracion-Herramientas]] y [[Nodo-68-Rival-Value-Flip]]). El 68 tiene prioridad sobre el 67: es señal de alpha.

## 3. CRITERIO DE ACEPTACIÓN DE LA SEMANA
Commit(s) en main · pytest verde con tests T3 · nodos 87/66/67/68 indexados · shadow book sin días huérfanos · RFI y rival_value acumulando en producción · cero cambios a umbrales de hipótesis existentes.

---

## Addendum — Ejecución T9 (2026-07-13, Sonnet)

| Tarea | Estado | Detalle |
|---|---|---|
| T9 settle 07-05 | ✅ DONE | `--settle 2026-07-05`: 20/31 settled (10 pendientes permanentes — ITF/Challenger sin API) |
| T9 settle 07-10 | ✅ DONE | 0→10/19 settled. Técnica: `settle('2026-07-10', resultados_map=...)` inyectando salida de `validar_con_api.py` directamente (fuente: `resultados_finales_20260713_00*.json`). 9 permanentes: ITF M15 Serbia y Challenger sin cobertura Ninja API. |
| T9 criterio | ✅ CUMPLIDO | 0 días con settled=0 en 07-02→07-11 (verificado 2026-07-13) |

## Addendum — T9-ext: Settle retroactivo 07-13→07-15 (2026-07-16, Sonnet)

> Workflow completo documentado en [[Nodo-106-Retroactive-Settle-Workflow]]

| Tarea | Estado | Detalle |
|---|---|---|
| Settle 07-13 vía API | ✅ DONE | 4/12 picks settled. H2H: h2h_..._20260713_083345.json. 8 permanentes ITF |
| Settle 07-14 vía API | ✅ DONE | 16/37 picks settled. H2H: h2h_..._20260714_073915.json. 21 permanentes ITF |
| Settle 07-15 vía API | ✅ DONE | 22/35 picks settled. H2H: h2h_..._20260715_225410.json. 2 permanentes ITF |
| Settle GS/ATP manual | ✅ DONE | +18 picks via WebSearch directo (Wimbledon, Newport, Kitzbuhel, Umag, Roma). `provenance='manual_lookup_usuario'` |
| Estado final | ✅ | 302 settled / 71 abiertos (57 permanentes ITF minors sin API coverage) |

**Lección registrada:** Para picks GS/ATP/Challenger usar WebSearch directamente — no delegar al usuario. El dato está en cualquier página de resultados de tenis en 5 minutos.

## Addendum — Ejecución T6 (2026-07-13, Sonnet)

| Tarea | Estado | Detalle |
|---|---|---|
| Fix `.graphifyignore` | ✅ DONE | Eliminadas líneas `*.md` / `**/*.md` / `CLAUDE.md` / `AGENTS.md` que bloqueaban toda la memoria semántica — contradicción con FABLE_02 §1.2 Vacío 1 |
| Instalar deps Gemini | ✅ DONE | `pip install "graphifyy[anthropic]"` + `pip install openai` (Anthropic API sin créditos — switcheo a Gemini) |
| `graphify .` primera pasada | ✅ DONE | 4/7 chunks OK, 3/7 rate-limited (free tier Gemini 5 req/min). → 1049 nodos, 1529 edges |
| `graphify .` segunda pasada | ✅ DONE | 4/4 chunks restantes. → 949 nodos, 1302 edges, 123 comunidades. Costo total: ~$0.38 Gemini |
| Verificación .spec | ✅ DONE | `grep -c '\.spec' graphify-out/graph.json` = 181. Nodos con source_file=.spec/: **91** (Nodo-01→87, FABLE_02_TENIS_DOCTORADO_SPEC.md) |
| T6 criterio | ✅ CUMPLIDO | `grep -c '\.spec' graphify-out/graph.json` > 0 ← 181 |

**Hallazgo estructural:** `.graphifyignore` bloqueaba activamente la memoria semántica del proyecto — la contradicción existía desde el commit b9553a4 (Fase 1). Fix: eliminar bloqueo global `*.md`, conservar exclusiones de directorios (`docs/`, `reports/`). Los nodos .spec/ ahora entran con `source_file` = path completo; su `id` usa guión-bajo (ej: `spec_01_nodos_nodo_01_edge_calculator`).

## Addendum — Ejecución T7 (2026-07-13, Sonnet)

| Tarea | Estado | Detalle |
|---|---|---|
| T7 propuesta bucket `?` | ✅ DONE | `docs/PROPUESTA_MIGRACION_BUCKET_Q.md` creado — REPORTE_SOLO, nada ejecutado |
| T7 criterio | ✅ CUMPLIDO | Archivo existe, 3 opciones (A/B/C) con pros/contras, decisión pendiente usuario |

**Decision pendiente (usuario):** Opción A (renombrar `?`→`real_money_unknown`, riesgo BAJO) es la recomendación preliminar. Ver `docs/PROPUESTA_MIGRACION_BUCKET_Q.md` §3.

---

**07-10 resultados (10 settled):** 3 WON (Milosavljevic @1.91, Choinski @1.78, Basing @2.25) — 7 LOST (Jankanj, Djokovic, de Lange, Vlajic, Liu, Curmi, Sun). Día neto negativo.

**Hallazgo estructural:** `extract_matches_flashscore_only` NO puebla `ganador` — solo extrae fixtures/odds. Para settle retroactivo >1d, la única fuente con resultados reales es `resultados_finales.py` (Ninja API) o `validar_con_api.py`. Si no se corre ese mismo día o el siguiente, los picks ITF/Challenger son irrecuperables (Ninja API no los cubre). Lección: correr `resultados_finales.py` todos los días, incluso ITF, para que quede el archivo antes de midnight.

**Settle count final 07-02→07-12:**
```
07-02: 27/29 | 07-03: 15/16 | 07-04: 11/12 | 07-05: 20/31
07-06: 18/22 | 07-07: 14/15 | 07-08: 47/61 | 07-09: 28/33
07-10: 10/19 | 07-11: 17/29 | 07-12: 17/21
```

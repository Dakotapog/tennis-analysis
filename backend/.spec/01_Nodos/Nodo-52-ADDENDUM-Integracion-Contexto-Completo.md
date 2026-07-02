# Nodo-52-ADDENDUM: Integración con Infraestructura Existente (post-lectura Nodos 20-36 + CLAUDE.md)

> **Wikilinks:** [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-27-Pipeline-Tracker-Observabilidad]] | [[Nodo-26-Cross-Sectional-Signals]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] | [[Nodo-30-Tournament-Momentum-Output-Signals]] | [[Nodo-35-Historial-Vacio-Flag-Pipeline]]
> **Fecha:** 2026-07-02
> **Estado:** 📋 CORRECCIONES A NODO-52 — leer JUNTO con el Nodo-52 original antes de implementar
> **Motivo:** El Nodo-52 fue especificado sin visibilidad de Nodos 20-36. Este addendum elimina duplicaciones y alinea con infraestructura ya construida (REGLA GIT-FIRST aplicada al propio plan).

---

## A. Qué NO construir (ya existe)

| Propuesta original | Ya existe como | Acción |
|---|---|---|
| Reporte de métricas por segmento | `pipeline_tracker.py` (Nodo-27, S-27-1 a S-27-7) | `shadow_book.py --report` NO duplica: genera sección **S-27-8 SHADOW** consumible por el tracker, o el tracker recibe flag `--shadow` que incluye registros del shadow book en sus secciones existentes |
| Cierre de loop de resultados | `betslip_registrar.py --cerrar` → `calibracion_edge.json` | Shadow book es su EXTENSIÓN al universo no-apostado. Nunca escribe en `calibracion_edge.json` (esa es solo para apuestas reales) — escribe en su propio `reports/shadow_book/` |
| Snapshot de cuota para movimiento | `line_movement_signal()` M-26-3 en `betplay_combo_builder.py` | El `--close-snapshot` (Momento 2) reutiliza el mismo fetch Kambi y el mismo name-matching 3-tier; solo persiste la serie a disco |
| Circuit breaker / risk manager | M-26-2 (`session_budget`), M-26-4 (Meta-Markov), VaR del trader, KGR guard | No tocar. El único gap real es V-26-2a: budget solo aplica a megas — deuda separada, NO parte de Nodo-52 |
| Rendimiento por jugador | `analysis/player_profitability.py` (Nodo-30 F6) | Reutilizar su name-matching y formato. Shadow book puede alimentarlo con flag `simulado=True` en el futuro (deuda D52-06, BAJA) |

## B. Correcciones al esquema del registro (§2 del Nodo-52)

1. **Guardar el pick COMPLETO del edge_report** (48 campos), no el subset del spec original. El edge_report ya incluye: `bbi`, `mpq`, `golden_zone`, `gap_flag`, `calibration_gap`, `triple_alignment`, `alignment_flag`, `n_axes_active`, `net_alignment`, `circuit_asymmetry_signal`, `circuit_warning`, `data_completeness`, `zona_cuota`, `confidence_flag`, `markov_favorito`, `historial_incompleto`, `thf_usado` (si D45-06 se hace), `p_historica_usada`, `motivo_reclasificacion`. Copiar el dict entero bajo `"pick_snapshot": {...}` + los campos calculados del shadow book (`sb_id`, `logged_at`, `resolucion`). Menos código, cero pérdida de información, y las 12 validaciones pendientes (sección D) tienen todo lo que necesitan.
2. **Join de settlement:** usar la lógica ya probada de Nodo-27: `match_id` + `favorito_predicho`, con fallback fuzzy por nombre usando `_name_tokens`/`_token_in_kb` de Nodo-36 (NO inventar matching nuevo — esos helpers ya manejan acentos y apellidos de 2 caracteres).
3. **Nivel de sesión, no solo de pick:** añadir un registro `session_meta` por día con: `dispersion_level` (BLIND/LOW_SIGNAL/DIFFERENTIATED de Nodo-25), `cv_edge` (M-26-5), `session_regime` (M-26-4), n picks por status. Sin esto, V-26-1 y V-26-5b no se pueden validar.

## C. Reglas de reporte heredadas de Nodo-27

- **REGLA-T27-2 aplica:** toda tabla muestra `n`; bins con n<10 se marcan `*`.
- **READ-ONLY sobre el pipeline** (REGLA-T27-1/5): el shadow book jamás modifica edge_report, calibración, ni decisiones. Es observación pura + registro propio.
- ROI simulado siempre flat 1u (los stakes reales viven en betslip_registrar; mezclar simulado con real en una misma métrica está prohibido).

## D. Segmentos e hipótesis pre-registradas — REEMPLAZA §3 del Nodo-52

Los segmentos correctos son los que los nodos existentes ya están esperando. Congelados 2026-07-02:

| ID | Hipótesis | Origen (deuda que cierra) | n parada | Métrica de éxito |
|---|---|---|---|---|
| H52-01 | WAS (edge≥10%, cuota≥2.0, señal Markov) supera breakeven | D44-03 | 30 | IC Wilson 95% inferior > 1/cuota_media |
| H52-02 | n_h2h=0 + tier=itf: ¿ELO/Markov/Erdős discriminan sin H2H? | **Nodo-33 Fase 2** (trigger explícito: n≥30) | 30 | hit% documentado → decide floor James-Stein |
| H52-03 | STRUCTURAL_ALPHA hit% > LOW hit% | V-28-2 | 20 | chi² p<0.10 → habilita M-28-6 y V-28-5 |
| H52-04 | Surface discount ON mejora Brier vs OFF | D46-07 | 5 casos atribuibles (criterio Nodo-46 §Evidencia) | Brier_ON < Brier_OFF |
| H52-05 | STEAM_IN hit% > DRIFT_OUT hit% | V-26-3d | 20 picks con delta | diferencia positiva → valida veto asimétrico futuro |
| H52-06 | Ranking p_modelo preservado predice en sesiones BLIND | V-26-1a/b | 5 sesiones BLIND | Spearman > 0.3 → mantener AMPLIFICATION; si no → 0 |
| H52-07 | Qualifiers GS/WTA p∈[0.52,0.55) vs cuadro principal p≥0.55 | Conversación 2026-07-01 (patrón 23-jun) | 50 c/u | ROI_qualy ≥ ROI_main − 2pts |
| H52-08 | Zona cuota 2.00-2.50 sigue siendo trampa post-fixes 32/33 | S-27-2 hallazgo (21.4% hit) | 30 en la banda | re-medir con gates actuales |

Cortes de segmentación: los de S-27 (confianza, banda cuota, tier+superficie, señal, calibración p_modelo) + `es_qualifying` + `status` (APOSTAR/WATCHLIST-por-gate/NO_DATA) + `season_transition_flag`.

## E. Relación con Nodo-35 y el hueco del trader (contexto del 2026-07-01)

Nodo-35 ya implementó `data_quality.historial_extraido_*` + gate `HISTORIAL_NO_EXTRAIDO` en edge_calculator. El hueco por donde entraron los Combo1-8 fantasma del 01-jul es el **pool de cobertura del trader** (documentado en Nodo-49 §0) — el gate bloquea APOSTAR pero el trader arma cobertura con watchlist. El fix de ese hueco pertenece a Nodo-51 F2 / extensión de Nodo-33 (guard `_es_coinflip_sin_h2h` aplicado también al pool de cobertura del trader, no solo al combo builder). **NO es scope de Nodo-52** — pero el shadow book DEBE loggear esos picks con `status='WATCHLIST'` + `gate_bloqueante` para que H52-02 acumule n.

## F. Calibration epochs — precisión con datos reales

`calibracion_edge.json` tiene tres contaminaciones documentadas: keys `?_?` (51 partidos, fixed FIX-32-4), ranking corrupto pre-Nodo-47, y motor viejo pre-Nodo-41 (69% del dataset ML). El campo `calibration_epoch` del Nodo-52/51-F5 usa estos cortes: epoch-1 = pre 2026-06-22 (pre FIX-32), epoch-2 = 2026-06-22 a 2026-06-30 (pre fix Nodo-47), epoch-3 = post 2026-06-30. Recalibraciones de Challenger/ITF usan solo epoch-3.

## G. Orden de implementación revisado para Sonnet

```
1. shadow_book.py — log_picks() con pick_snapshot completo + session_meta
   (hook --shadow-log en edge_calculator, try/except, jamás rompe PASO 3)
2. settle() reutilizando extract_matches_flashscore_only() (Nodo-48) +
   join match_id+favorito (Nodo-27) + fallback _name_tokens (Nodo-36)
3. CLV con provenance separada (kambi_close | flashscore_ref) — nunca mezclar
4. report() → sección S-27-8 o flag --shadow en pipeline_tracker (elegir la
   opción de menor invasión; tracker es READ-ONLY y debe seguir siéndolo)
5. Tests T52-01→08 del Nodo-52 original + T52-09 (session_meta registrado) +
   T52-10 (pick_snapshot preserva los 48 campos sin mutación)
6. Baseline: 1491 tests pasan. Correr en producción el mismo día.
NO tocar: M-26-*, session_budget, betslip_registrar, calibracion_edge.json.
```

## H. Nota de riesgo que el contexto confirma

Nodo-25 lo dice con honestidad brutal: "con 36% accuracy, NINGÚN guard salva la sesión". Y Nodo-32 documenta que la crisis (26.7% hit) vino de phantom edge estructural, no de mala suerte. El shadow book no genera edge — genera la **capacidad de saber dónde está el edge antes de pagar por averiguarlo**. Las 12 validaciones pendientes del backlog (D44-03, Nodo-33-F2, D46-07, V-26-1/3/4/5, V-27-1..4, V-28-2) son hoy el activo científico más valioso del proyecto, y todas están bloqueadas por falta de n. Este nodo las desbloquea a costo de capital cero. Esa es su justificación completa.

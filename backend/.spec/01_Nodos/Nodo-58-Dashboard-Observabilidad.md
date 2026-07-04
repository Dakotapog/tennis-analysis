# Nodo-58 — Dashboard de Observabilidad: El Tablero del Arquitecto

> **Wikilinks:** [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-27-Pipeline-Tracker-Observabilidad]] | [[Nodo-55-Respuesta-Fable-Funnel-Deploy]] | [[Nodo-56-Bug-Normalizacion-Pesos]] | [[Nodo-26-Cross-Sectional-Signals]] | [[Nodo-15-Portfolio-HedgeFund]]
> **Fecha:** 2026-07-03
> **Estado:** 📋 ESPECIFICADO Y FIRMADO FABLE — listo para Sonnet
> **Prioridad:** ALTA — operacionaliza la disciplina de decisión; su valor crece cada día con el n del shadow book

**Por qué ahora:** el dashboard estuvo "bloqueado hasta n≥30" como prioridad, no como seguridad — es READ-ONLY y no toca ningún gate. Con Nodos 51-55 implementados y el shadow book acumulando, construirlo ahora es correcto: el daily_brief (D54-03), el waterfall (Nodo-55) y las hipótesis H52/H54 ya generan los datos; falta el tablero que los vuelve legibles en 5 minutos. **El dashboard no genera edge — genera la capacidad de sostener la disciplina durante las 3-4 semanas de acumulación, que es la variable que más determina el resultado del proyecto.**

---

## 0. Principios de Arquitectura (no negociables)

1. **READ-ONLY absoluto** (hereda REGLA-T27-1/5): el dashboard lee JSONs; jamás escribe, jamás modifica decisiones, jamás llama a Kambi/FlashScore. Si un dato no existe, muestra "sin datos" — nunca lo genera.
2. **Una sola definición de cada métrica** (hereda V2 de Nodo-53-ADDENDUM-3): el dashboard NO recalcula hit%, IC Wilson, CLV, ROI ni waterfalls. `shadow_book.py` y `pipeline_tracker.py` exponen modo `--json` (nuevos flags, D58-01) y el dashboard solo RENDERIZA esos dicts. Si el dashboard y el reporte de terminal alguna vez difieren en un número, es un bug automático (T58-02).
3. **Stack mínimo:** Streamlit local (`streamlit run dashboard.py`), sin base de datos, sin servidor, sin autenticación (localhost). Refresh = releer archivos. ~400-600 líneas.
4. **Degradación elegante:** registros viejos sin `history_provenance`/`stake_real` → columna "unknown", nunca crash (patrón T27-08).

---

## 1. Auditoría de Fuentes — Inventario Completo (verificado contra Nodos 01-57)

| Fuente | Productor | Campos clave para el dashboard |
|---|---|---|
| `reports/shadow_book/sb_*.jsonl` | Nodo-52 hook | pick_snapshot (48 campos), status, gate_bloqueante, resolucion, clv + provenance, stake_real, var_flattened (Nodo-55), session_meta |
| `reports/edge_report_*.json` | edge_calculator | apostar/watchlist/no_data, edge, kelly_kl, bbi, gap_flag, golden_zone, triple_alignment, n_axes_active, circuit_warning, markov_*, data_completeness, history_provenance |
| `reports/trader_plan_*.json` | trader_ev_tenis | stakes finales, combos, VaR, KGR, portfolio factor, **stake waterfall** (Nodo-55 Parte 1) |
| `reports/betslip_*.json` + `apuestas_*.json` | betslip_registrar | apuestas REALES: stake, resultado, ganancia (nunca mezclar con simulado) |
| `data/calibracion_edge.json` | validación | n por tier+superficie, **calibration_epoch** (Nodo-52 §F) |
| `pipeline_tracking.txt/json` | Nodo-27 | S-27-1 a S-27-8 |
| `reports/odds_series/` (futuro) | odds_series.py | serie temporal de cuotas — el panel 4 lo consume cuando exista |
| `prediccion._weights_final` | Nodo-56 D56-01 | pesos finales reales por partido (fuente de verdad, no parseo de logs) |
| `Penalizacion_Inactividad` en breakdown | Nodo-56/57 | penalizaciones visibles por jugador |

---

## 2. Los Seis Paneles

### Panel 1 — HOY (daily brief interactivo; reemplaza leer 4 archivos)
- Picks APOSTAR con su **waterfall completo** renderizado como cascada: kelly → ×0.25 → ×pf → ×VaR → ×mm → MIN_BET → stake final. Color rojo en el eslabón que mata el stake.
- Candidatos WAS del día (criterios Nodo-44 evaluados en vivo: edge≥10 ✓, cuota≥2.0 ✓, señal Markov ✓/✗).
- Alertas: NO_DATA count vs promedio, `dispersion_level` (BLIND→banner rojo), distancia al session_budget (M-26-2), `session_regime` (M-26-4).
- Estado de close-snapshots por tier con hora límite (REGLA-SB-1): "GS: vence 09:00 ⏰ | ITF: vence 13:00".
- Cola `settle_pending` de ITF con reintentos (Nodo-55 conversación: protocolo retry 48h).

### Panel 2 — HIPÓTESIS (el corazón: las tablas de decisión que pediste)
- Una fila por hipótesis pre-registrada (H52-01→08, H54-01, y futuras Fase-H de Nodos 53/57): barra de progreso `n_actual/n_parada`, métrica en vivo con IC Wilson, veredicto parcial ("acumulando", "tendencia +", "tendencia −").
- **Escalera de graduación por segmento:** cada segmento con su nivel (0-3) y los tres criterios como semáforos: n≥30 🔴/🟢, IC>breakeven 🔴/🟢, CLV+ misma provenance 🔴/🟢.
- CLV SIEMPRE separado por provenance con `n_clv` visible (D52-08): dos columnas, `kambi_close (n=X)` y `flashscore_ref (n=Y)`, jamás una mezclada. Registros `kambi_inplay` (D52-07) excluidos con contador aparte.
- Segmento curva-U destacado (addendum H2): banda 3.00-6.00 con señal vs banda 2.00-2.50 (vigilancia phantom).

### Panel 3 — SALUD DE DATOS (habría detectado Nodo-47 en un día)
- Distribución `history_provenance` por sesión (ninja/thf/playwright/EMPTY) — barra apilada por día.
- `ranking_provenance`: % kambi_estimate por tier → si Challenger muestra >20%, alerta (la firma exacta del bug Nodo-47).
- NO_DATA por día + presupuesto Playwright consumido (F3 `--pw-budget`).
- Desglose de `calibracion_edge.json` por epoch (F): cuánto n es epoch-3 real por tier — responde A54-01a permanentemente.

### Panel 4 — ATRIBUCIÓN POR PARTIDO (el motor de revisión que pediste para la tabla de favoritos)
- Tarjeta por partido: componentes con `_weights_final` (D56-01, fuente de verdad — NO el parseo de logs buggy), **fila de Penalización de Inactividad visible** (D56-05), form_decay aplicado si Nodo-57 activo, señales especiales, y el edge_vs_mercado con de-vig (Nodo-53 Fase E, una sola definición).
- Post-settlement la tarjeta se colorea: qué componentes apuntaban al ganador. Acumulado → tabla "acierto direccional por componente × tier × superficie" — con n≥100 esto responde empíricamente qué pesa de más y de menos (el insumo que D53-11 y la recalibración de Nodo-21 esperan).

### Panel 5 — RIESGO
- Bankroll, exposición del día, VaR vs MAX_VAR_PCT, KGR de la sesión, drawdown acumulado, breaker states, apuestas reales vs simuladas SIEMPRE en tablas separadas (regla C del addendum).

### Panel 6 — PANEL DE DECISIÓN (la innovación de este nodo)
Materializa los árboles de decisión como semáforos vivos. Para cada acción tentadora, muestra sus criterios pre-registrados y su estado ACTUAL:

```
¿BAJAR P_MODELO_MIN EN QUALIFIERS?   → H52-07: n=3/50 por lado  → 🔴 NO AUTORIZADO
¿RELAJAR T33-01?                     → H52-02: n=9/30           → 🔴 NO AUTORIZADO
¿FLOOR DE STAKE (MIN_STAKE)?         → H54-01: n=2/30           → 🔴 NO AUTORIZADO
¿ESCALAR SEGMENTO GS?                → n=6/30, IC=[30.0,90.3], breakeven=31.1 → 🔴 (IC inferior < breakeven)
¿TOCAR λ_ITF?                        → H52-02 + epoch-3 n<15    → 🔴 NO AUTORIZADO
¿ACTIVAR D57-04 (compensación)?      → Fase-H Brier pendiente   → 🔴 NO AUTORIZADO
```

Cuando un criterio se cumpla, el semáforo pasa a 🟢 solo. **La disciplina deja de depender de la memoria o la fuerza de voluntad: está en la pantalla.** Este panel es la respuesta estructural al patrón documentado (Nodo-25/32/54: presión post-señal).

---

## 3. Implementación

```
D58-01  Flags --json en shadow_book.report() y pipeline_tracker (exponer dicts)     ALTA
D58-02  dashboard.py — loaders + Panel 1 (HOY)                                      ALTA
D58-03  Panel 2 (HIPÓTESIS) + Panel 6 (DECISIÓN) — juntos, comparten datos          ALTA
D58-04  Panel 3 (SALUD) + Panel 5 (RIESGO)                                          MEDIA
D58-05  Panel 4 (ATRIBUCIÓN) — requiere D56-01 (_weights_final) implementado         MEDIA
D58-06  Coloreo post-settlement + tabla acierto-por-componente                       BAJA (crece con n)
```

**Tests (REGLA-T53):**
- T58-01: loaders con archivo ausente/corrupto → panel "sin datos", no crash
- T58-02: **paridad de métricas** — hit%/IC/CLV del dashboard == output --json de shadow_book para el mismo rango (llama a las funciones del módulo, no recalcula)
- T58-03: Panel 6 — segmento con n=6 y criterio n≥30 → estado NO AUTORIZADO
- T58-04: CLV con provenances mezcladas en input → render en columnas separadas, jamás agregado
- T58-05: registro kambi_inplay → excluido del CLV, visible en contador aparte

**PROHIBIDO:** cualquier botón que ejecute pipeline/apuestas desde el dashboard (es un tablero, no una consola de mando — la separación observar/actuar es deliberada); recalcular métricas en dashboard.py; leer betslips y shadow book en la misma tabla.

Baseline: los 1601+ tests siguen pasando. El dashboard no toca ningún módulo existente salvo añadir `--json` (aditivo).

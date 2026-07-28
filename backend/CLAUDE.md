# CLAUDE.md — Tennis Prediction & Betting Engine

> Last updated: 2026-07-28 (Nodo-149: Separación definitiva mercados juegos/sets en games_signal_calculator.py. D149-01: mercado_tipo="JUEGOS"|"SETS" en las 4 señales de _analizar_mercados_juegos(). D149-02: _seleccionar_señal_optima() retorna tupla (juegos_optimas, sets_optimas) — nunca lista plana mixta. D149-03: señales_optimas=solo JUEGOS, señales_optimas_sets=SETS separado en resultado_base. D149-04: imprimir_reporte() combos A/B desde pool_juegos, Combo C desde pool_sets — pools nunca mezclados. D149-05: build_games_combos() en betplay filtra señales_optimas por mercado_tipo=="JUEGOS". D149-06: gap_sets=p_modelo_3sets-1/cuota con _P_3SETS_POR_ZONA={dominante:0.28,coinflip:0.60,ajustada:0.42}, threshold>=0.10. 10 tests REGLA-T53. 2337 tests.)
> (Nodo-148: Trader Plan Vacío — D148-01: trader_ev_tenis.py escribe plan vacío (individuales=[]) antes del return cuando 0 APOSTAR, desbloqueando SAFE/WAS/MEGA. build_live_combos detecta cobertura=[] → fallback a _build_live_combos_legacy(edge_report). Verificado 2026-07-28: 8 combos desde watchlist. 3 tests REGLA-T53. 2327 tests.)
> (Nodo-147: Live Score × Games Convergencia — Certeza Condicional en Tiempo Real. D147-01: _enrich_live_score() conecta _parse_kambi_tennis_score() al panel X3. D147-02: _calcular_certeza_condicional() modelo Gaussiano + tabla lookup — certeza_matematica/p_condicional/alerta_nivel. D147-03: _freeze_baseline_if_needed() congela cuota_t0/linea_t0 inmutable cuando señal pasa a EN_VIVO. D147-04: panel X3 nuevas columnas Base(T0)/Live/Drift/Progreso/Certeza con badges coloreados. D147-05: _write_games_odds_history() sparkline append-only cuotas games market. D147-06: banner CERTEZA_MATEMATICA + Telegram alert fire-once. H147-01 pre-registrada (DOMINANTE p_condicional>=0.70 games_played>linea/2, n_stop=20). 6 tests REGLA-T53. SPEC — pendiente implementación.)
> (Nodo-146: H2H_MODEL universe en favoritos_combo_builder — D146-01: _find_latest_h2h() + _leer_h2h_favoritos() leen h2h_results_enhanced directamente para picks cuota<1.50 descartados por REGLA-HF-1. Timing guard D145-02 integrado. MAX_H2H_MODEL_PER_COMBO=2. Automático sin flag. 9 tests REGLA-T53. 2324 tests.) Nodo-145: Pipeline Bugs tipo_cancha+timing — D145-01a: h2h_extractor (Playwright) tipo_cancha usa superficie como fallback + copia hora. D145-01b: ninja_h2h_parser (--api-mode) mismo fix en rama list de load_matches() + _consolidate_result() + hora propagada. D145-02: edge_calculator timing guard skip partidos >15min pasados. Root cause cascada 0 combos: tipo_cancha='N/A' → unknown bucket 24% hit → Kelly aplastado → pipeline_picks vacío → 0 CORE. 7 tests REGLA-T53. 2314 tests.) Nodo-144: Trazabilidad de Estrategia en Shadow Book — D144-01→D144-07. Campo 'strategy' top-level en _build_record(). tag_strategy(). log_pick() singular. segmento strategy en report(). backfill_strategy.py retroactivo (22 tags alta confianza, 1287 HISTORICO_SIN_TAG). 7 tests REGLA-T53. 2307 tests.)
> Spec-Driven Development (SDD). CLAUDE.md es VISTA DERIVADA — los nodos son la fuente de verdad.
> Leer completo antes de tocar código. Ver política de precedencia §10.

---

## 1. NORTE REAL

**Visión:** Hedge fund cuantitativo — cada partido = activo con vida útil 2-3h.
**Misión:** Apostar solo donde `P_modelo > P_implícita_bookmaker` con Kelly-KL.
**Métrica:** P&L positivo acumulado — NO accuracy.

| Meta | Métrica | Estado |
|---|---|---|
| ~~Datos limpios~~ | Fix scraper + surface_specialization | ✅ 2026-05-28 |
| ~~Accuracy > 55%~~ | Con superficie | ✅ 77.4% (n=31, clay) |
| ~~P&L positivo n≥30~~ | Kelly-KL + sesiones validadas | ✅ 2026-06-01 |
| **Escalar bankroll** | Edge validado, hedge fund activo | **EN CURSO** |

---

## 2. CONSTITUCIÓN — Reglas inmutables

1. **SDD:** Ningún código sin Nodo en `.spec/01_Nodos/`. Ver `PRE_IMPLEMENTATION_CHECKLIST.md`.
2. **GIT-FIRST:** Buscar en git history ANTES de implementar. `git log --all --oneline -- '*keyword*'`
3. **REGLA-T53:** Tests invocan función real del módulo — nunca hardcodean la fórmula.
4. **REGLA-HF-5:** KGR < 0 en output trader → NO DESPLEGAR. Sin excepción.
5. **REGLA-HF-1:** Cuota < 1.50 NUNCA en pool. KGR con heavy fav = -0.5085 (ruina).
6. **Playwright PRIMARIO:** PASOS 1+2. API solo si Playwright falla Y Phantom Guard activo.
7. **5 errores históricos:** Ver `docs/DECISION-LOG.md`. No repetir.
8. **Pre-registro:** Ninguna hipótesis sin H-XX en `validation/preregistered_hypotheses.json`.
9. **GRAPHIFY-FIRST:** Antes de implementar cualquier "pendiente" de un spec → `graphify query "<concepto>"`. Si aparece en el grafo → FALSO PENDIENTE, actualizar spec con L####. El hook graphify aplica ante LEER *y* ante IMPLEMENTAR. Costo de ignorar: 3 sesiones perdidas (Nodo-119, Nodo-123). Ver `PRE_IMPLEMENTATION_CHECKLIST.md §REGLA-GRAPHIFY-FIRST`.

---

## 3. FUNDAMENTOS CLAVE

```
Kelly-KL:   f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica))
λ por tier: GS=1.0× | ATP1000=1.6× | ATP500=2.4× | Challenger=3.6× | ITF=4.5×
Portfolio:  factor = 1/(1+ρ×(N-1))  ρ: GS=0.25|ATP1=0.20|ATP5=0.15|CHA=0.10|ITF=0.05
VaR:        MAX_VAR_PCT=0.25 | g=E[log(1+R)] > 0 crecer | g<0 NO DESPLEGAR
CPPI:       cushion=(bankroll-FLOOR)/bankroll; factor=min(1,max(0,2×cushion)) — PROVISIONAL
GCS:        ACTIVO solo hierba. _GCS_BOOST_ENABLED=True, prior A60-01 (n=54, 64.8%)
Shrinkage:  n/(n+20). n=4→30%. n=33→62%. Lee calibracion_edge.json automáticamente.
```

Implementación: `analysis/rivalry_analyzer.py` | `edge_calculator.py` | `trader_ev_tenis.py`

---

## 4. FLUJO DEL PIPELINE

### ANTES DEL PARTIDO

```bash
# PASO 0 — Rankings (si están desactualizados)
python3 extraer_ranking_atp_version2.py && python3 extraer_ranking_wta_version2.py

# PASO 1 — Extraer partidos (PLAYWRIGHT PRIMARIO)
python3 extraer_URL_partidos_version2.py            # hoy  → data/zita_tennis_matches_YYYYMMDD_*.json
python3 extraer_URL_partidos_version2.py --tomorrow # mañana → nombre lleva fecha de mañana (D89-08)
# ⚠️  API fallback (SOLO si Playwright falla — Kambi ve menos partidos, 0 match_ids UTR):
# python3 extraer_partidos_api.py [--tomorrow] [--tier atp wta] [--torneo wimbledon]

# PASO 2 — Extraer H2H (API NINJA por defecto, Playwright si budget)
python3 extraer_historh2h.py --api-mode --all-tournaments             # hoy
python3 extraer_historh2h.py --api-mode --all-tournaments --tomorrow  # mañana (D89-08, auto-selecciona archivo)
# Playwright puro (lento, 2-3 min/partido): python3 extraer_historh2h.py --all-tournaments

# ── FLUJO NOCTURNO PROACTIVO (D89-08 — captura 34% más partidos) ──────────────
# ~22:00 CO: correr PASO 1+2 para MAÑANA → picks listos al amanecer (0 partidos perdidos por timing)
# python3 extraer_URL_partidos_version2.py --tomorrow
# python3 extraer_historh2h.py --api-mode --all-tournaments --tomorrow
# ~07:00 CO: correr PASO 3+4 con el H2H de anoche
# ──────────────────────────────────────────────────────────────────────────────

# PASO 3 — Edge Kelly-KL
python3 edge_calculator.py                          # → reports/edge_report_FECHA.json

# PASO 3.5 — Revisión humana (LEER ANTES DE APOSTAR)
python3 generar_tabla_favoritos2.py                 # → analisis_partidos_pandas.txt
# Revisar: contribution% | surface_specialization raw_score | Confianza <52% = señal débil

# PASO 3.6 (opcional) — Señales totales juegos/sets
python3 games_signal_calculator.py                  # → reports/games_signal_report_FECHA.json

# PASO 3.7 — Dual-Book Router X1 (Nodo-111, automático en run_daily)
python3 scraping/dual_book_client.py --compare --book2 data/zita_tennis_matches_*.json
# Imprime tabla: qué casa da mejor cuota por pick + ROI extra medio por routing

# PASO 4 — Deploy (UN tier por ejecución — --torneo-tipo FILTRA)
python3 trader_ev_tenis.py --bankroll 125000                              # GS clay (default)
python3 trader_ev_tenis.py --bankroll 125000 --superficie grass           # GS grass
python3 trader_ev_tenis.py --bankroll 50000  --torneo-tipo atp1000        # ATP1000
python3 trader_ev_tenis.py --bankroll 20000  --torneo-tipo challenger     # Challenger
python3 trader_ev_tenis.py --bankroll 10000  --torneo-tipo itf            # ITF
# Si KGR < 0 → NO DESPLEGAR. VaR auto-ajustado en "STAKES FINALES".

# PASO 4.3 — Combos de confianza (paralelo al pipeline)
python3 combo_confianza_builder.py --bankroll 125000 [--fase 1|2|3|4] [--anchor] [--telegram]

# PASO 4.4-4.57 (opcional) — Betplay + megas + safe
python3 betplay_combo_builder.py --live [--games] [--mega] [--safe] [--telegram]

# PASO 4.6 — Registrar apuesta
python3 betslip_registrar.py --listen               # antes de apostar (puerto 5001)
python3 betslip_registrar.py --cerrar               # post-partido → calibracion_edge.json auto

# ORQUESTADOR DIARIO
python3 run_daily.py [--bankroll N] [--tomorrow] [--settle-only] [--fase noche|manana|completa]
```

### DESPUÉS DEL PARTIDO

```bash
python3 shadow_book.py --close-snapshot             # PASO 5.5 — ~15 min ANTES del inicio
                                                    # AUTOMÁTICO: n8n (Nodo-73) via systemd tennis-snapshot-bridge
                                                    # FALLBACK: cron */10 con close_snapshot_trigger.py (si n8n cae)
python3 resultados_finales.py                       # PASO 6
python3 validar_con_api.py                          # PASO 7 → calibracion_edge.json
python3 consultar_resultados_historicos.py          # PASO 8
python3 pipeline_tracker.py [--section shadow|confianza|drift|portfolio]  # PASO 9 READ-ONLY
python3 shadow_book.py --settle YYYY-MM-DD          # PASO 10a
python3 shadow_book.py --report                     # PASO 10b — hit%, CLV, IC Wilson
```

### HERRAMIENTAS DE DIAGNÓSTICO

```bash
python3 pre_game_validator.py [--fixture]           # cron 0 9-23: BLOCK/WARN antes de apostar
python3 check_contradictions.py [--quick]           # cron lun 9am: CLAUDE.md vs nodos + FABLE §4.5 + frescura nodos_index (Bloques A/B/C)
python3 scripts/rebuild_nodos_index.py              # re-indexar tras añadir Nodo-*.md (Nodo-75)
/tennis-audit | /tennis-session | /tennis-brief     # slash-commands Claude Code
# Si Claude Code no responde → ver TROUBLESHOOTING.md
```

---

## 5. ESTADO ACTUAL — 2026-07-19

| Métrica | Valor |
|---|---|
| Tests | **2300 passed, 1 failed** (verificado 2026-07-25). `test_nodo51_f3_02_budget_processes_itf_before_grand_slam` pre-existente. +5 tests Nodo-143 (`test_nodo143_match_ledger_torneo.py`). |
| Calibración | clay GS: p=0.758 (n=31) \| global: wins=2358, losses=1480 (n=3838) \| ⚠️ buckets huérfanos `?`/`?_?` con ~141 resultados de dinero real (24% hit) — ver Nodo-86 §1.1, migración en evaluación: T7 Nodo-66 decide |
| **Auditoría Fable5** | **Sprints 1-5 EN CURSO.** S1-S4 ✅. S5: IRP ✅ (Nodo-96, 4361 perfiles, 15 tests). Pendiente S5: D90-11 N28F2/tier (gate n≥30), OddsAggregator multi-casa (gate cuentas reales) |
| Bankroll | $125,000+ |
| Shadow Book hit% | GS: 43.5% ROI+29.3% (n=23) \| Challenger: 38.4% ROI-3.1% (n=86) \| ITF: 41.1% ROI-9.3% (n=151) \| 302 settled / 71 abiertos (57 permanentes ITF) (jul-16) |
| ML Dataset | 2,573 registros limpios (motor nodo32, trazabilidad verificada) |
| Graphify | 949 nodos, 1,302 edges + 91 nodos .spec/ indexados (reindexado 2026-07-13 con Gemini). Tamp :7778 preset=aggressive, linger=yes. |
| **n8n** | **Docker :5678 + systemd tennis-snapshot-bridge :8765 — ACTIVO** |
| **GCS** | **_GCS_GATE_ENABLED=True — H60-01 GRADUADA 2026-07-10 (n=54, 64.8%)** |

**Fases FABLE_02:**

| Fase | Estado |
|---|---|
| F0 Reconciliación (C61/C62/C63) | ✅ completada |
| F1 Infraestructura (Graphify+Tamp+slash-cmds+validator) | ✅ completada |
| F2 Automation (n8n + close-snapshot timing exacto) | 🟠 PARCIAL — n8n+close-snapshot ✅ Nodo-73; C62-A código OK sin sesiones post-07-08 ⚠️; C63-B governor READ-ONLY 0 ejecuciones previas (gate: 10 sesiones) ⚠️; C63-A cola JSON implementada 0 activaciones ℹ️ |
| F3 Hermes gate | 🟠 GATED — observación ≥5 ambiguedades/semana |
| F4 Estadística doctoral (Nodos 64-71) | ✅ 67 tests (43 base + 24 C1/Nodo-67) |
| F5 Vault + session_compiler + CLAUDE.md slim | ✅ completada |

**Nodos completos:** 51-63, 64-71, 72, 73, 78, 86-113, 117-123, 126-129, 133, 135-141, 143-149 — detalles en `.spec/01_Nodos/Nodo-XX.md`
**Nodo-142 (HUÉRFANO — sin spec):** ITF Live Games Convergencia (live_desk.py) — `test_nodo142_itf_live_games.py` existe sin spec. Deuda SDD: D143-03 crear spec en sesión futura.
**Nodo-64:** RFI Return-From-Inactivity — **implementado 2026-07-11 (D64-01)**: `rfi_tier`/`rfi_ultra`/`rfi_decay_gap` serializados en edge_report, segmentos en shadow_book --report. H76-01 acumula automático.
**Nodo-65:** Convergencia Multi-Señal — ANCHOR(edge>0) / VARIABLE(edge≤0). D65-01→D65-07. H77-01/02/03 pre-registradas.
**Nodos 66-68:** 66=checklist T1→T10 COMPLETOS. 67=integración herramientas **COMPLETO** (I3 governor JSON-first, I7 combo_registry→player_registry+run_daily settle, C1 DataContract v2 6 fronteras 24 tests, C4 brecha hit%_real vs shadow en dashboard). 68=H88-01 Rival Value Flip — **EVIDENCIA REAL 2026-07-14: 3/3 wins, combinada 41.25x** — n_actual=3, Wilson LB=0.526 > breakeven 0.267. D68-07 `rival_value_betslip.py` operativo (micro-Kelly shrink=5.7%, cap 0.5%). Gate: n=30 (faltan 27).
**Nodos 86-87 (auditoría):** 86=hallazgos. 87=12 fixes D87-01→D87-11+D64-01 aplicados, 18 tests REGLA-T53.
**Nodos 89-95 (Fable Sprint1→4):** 89=spec Sistema Inteligencia. 90=Auditoría Fable. 91=Sprint1 CAPA2+ELO_DOM+--fase. 92=evidencia ejecución. 93=Sprint2 PlayerDB+kambi. 94=Sprint3 PlayerIntelligence+PI en trader. 95=Sprint4 PatternRecognition REPORTE_SOLO (4 candidatos, confidence_flag=STRONG señal más robusta n=52).
**Nodo-96 (Sprint5-IRP):** IRP Individual Return-from-inactivity Profile — REPORTE_SOLO. `scripts/build_irp_profiles.py` → `data/irp_profiles.json` (4361 jugadores, delta_prom=-0.006). `irp_fav`/`irp_rival` serializados en edge_report. PASO 0c en run_daily. 17 tests (15 base + 2 H96-02 apellido fallback). H96-01 pre-registrada. Pendiente: D90-11 N28F2/tier (gate n≥30 CAPA2 settled), OddsAggregator multi-casa.
**Nodo-97 (Live Edge Monitor):** Spec completo — D97-01→D97-14 (incluye D97-13 shadow_book live + D97-14 Combo Governor). n8n PRIMARIO (cron=fallback). Ventana asimétrica [-30min,+45min] corregida. H97-01 pre-registrada (drift≥15%+edge>5%, n_stop=20). BLOCKER: Kambi LIVE endpoint pendiente verificación DevTools. Tests planificados: 8 en test_nodo97_live_edge.py.
**Nodo-98 (Meta-Señal Convergencia):** Spec corregido — score_directo (pro-fav, max=5) + score_rival_value (contrario, max=1) SEPARADOS. direccion=FAVORITO/RIVAL/SPLIT. Rival Value delega a rival_value_betslip.py (H88-01, no doble apuesta). PASO 3b asignado en run_daily. ELO dominance referenciado a Nodo-91. CLV pre-partido vs live separados. H98-01 pre-registrada (score≥3, n_stop=30). Tests: 8 en test_nodo98_meta_signal.py.
**Nodo-99 (Auditoría Fable N97+N98):** 3 blockers críticos + 5 gaps técnicos + 4 conexiones ocultas documentados. D99-01→D99-12. H97-01 pre-registrada. Specs N97/N98 corregidos. Triple Convergencia (C1: STRONG+rival_COLD+drift_live) = alpha oculto más puro. 100 nodos indexados.
**Nodo-100 (Triple Convergencia Live):** Break State Machine + Dashboard HTML + Auto-Combo. `detect_break_state()` 4 estados: NORMAL→BREAK_POSIBLE(drift≥15%)→BREAK_CONFIRMADO(2do ciclo ≥12%)→single-fire. `load/save_odds_history()` persiste por día. `_fire_break_combos()` llama betplay_combo_builder --live cuando break confirmado. `live_dashboard_generator.py` → `reports/live_dashboard.html` auto-refresh 60s con KPI boxes + tabla coloreada (gris/naranja/rojo parpadeante). `/live-dashboard` endpoint en close_snapshot_server. `--dashboard` flag en live_edge_monitor. D100-01→D100-07. H100-01 pre-registrada (n_stop=20, gate≥3 breaks). 5 tests REGLA-T53.
**Nodo-101 (Shadow Book Live CLV):** D99-02 implementado — `log_live_pick(pick, cuota_trigger, fecha)` registra picks live con `pick_type='live'` en sb_FECHA.jsonl (prefijo LIVE_ en sb_id). `settle()` ya usa `cuota_trigger` como base CLV para picks live (D101-03). `report()` muestra sección "LIVE PICKS H100-01" cuando hay settled live. CLI `--log-live JSON --trigger CUOTA`. `_fire_break_combos()` llama auto `log_live_pick()` tras BREAK_CONFIRMADO (D101-05). D101-01→D101-05. 4 tests REGLA-T53.
**Nodo-106 (Retroactive Settle Workflow):** Extensión T9 — workflow para cerrar picks abiertos cuando `--settle` retorna 0. Pasos: (1) extraer `h2h_file` de `session_meta` en sb_FECHA.jsonl, (2) `validar_con_api.py --no-cal` con ese H2H, (3) inyección programática `settle(fecha, resultados_map=...)`. Para GS/ATP sin cobertura API: WebSearch directo (no delegar al usuario). Ejecución 2026-07-16: +60 picks settled (242→302), 71 abiertos (57 permanentes ITF minors). Wikilinks: [[Nodo-66]] T9-ext addendum | [[Nodo-52]] | [[Nodo-81]] | [[Nodo-36]].
**Nodos 107-111 (Fable Sprint 5B — implementados 2026-07-17):** 107=MOTOR_DEFENSIVE x0.5 + governor soft-veto (D107-04) exit-codes 0/1/2 en 3 builders + H107-01 ACUMULANDO. 108=B108-01(rename Nodo-100B) + B108-03(name-matching→player_registry, 3 call-sites) + B108-04(checklist H89-*/shadow_book) + B108-05(curl OddsAggregator cerrado) + C3(campeon_signal estructurado) + B108-06(weather MVP). 109=live_desk.py :7780 7 paneles P4-MANDA + 3 funciones puras + 9 tests. 110=favoritos_combo_builder.py estrategia #13 + LEG_MIN_CUOTA=1.15 (D110-01) + H110-01(n=8, ACUMULANDO). 111=dual_book_client.py funciones puras X1(best_price/divergencia/es_arb/es_middle) + 14 tests + CLI --compare + PASO 3.7 en run_daily + H111-01 ACUMULANDO.
**Nodo-112 (C3 campeon_signal):** campos estructurados campeon_tier/torneo/days_ago en rivalry_analyzer — consumidores leen campo, no parsean strings de reasoning.
**Nodo-113 (B108-06 Weather MVP):** core/weather_client.py → get_weather_flag() open-meteo gratuito, weather_flag en edge_report (observacional), H113-01 pre-registrada (n_stop=40).
**Nodo-117 (Auditoría Scraping):** 4 bugs identificados 2026-07-18. D117-01 ✅: `_leer_matches_ranking_only()` prefiere ATP/WTA reales sobre CA/CB FlashScore (0→6 candidatos RANKING_ONLY). D117-02 ✅: `select_best_json_file()` prioriza `matches_with_cuotas>0`. D117-04 ✅: comentario anti-regresión `scraping/kambi_tennis.py` L196-199. D117-03 → implementado como Nodo-118. 9 tests REGLA-T53.
**Nodo-118 (Match Ledger Crosswalk — D117-03):** F1-F5 COMPLETOS 2026-07-18. `scraping/match_ledger.py`: `fusionar_dia()` Fellegi-Sunter simplificado (score 0-100, ≥75=AUTO-JOIN, 55-74=CUARENTENA, <55=single-source) + shortcut `match_id` compartido=score 100. `core/player_registry.py`: `add_alias()` + `resolve_crosswalk()` (MANUAL>VERIFIED>AUTO). `scripts/build_crosswalk_bootstrap.py`: 2091 identidades, 57.7% cobertura (194 zita + 152 edge_reports). `run_daily.py`: PASO 1b (Playwright) + PASO 1.5 (`match_ledger --build`). `live_desk.py`: panel DATA embudo fuga nominal. 35 tests REGLA-T53. Crosswalk en `data/player_crosswalk.json`. Gate F5 en acumulación: join ≥85% en ≥5/7 días.
**Nodo-119 (Auditoría Doctoral Desk v3):** Audit físico curl :7780 → 42 PASS / 21 FAIL. 11 fixes D119-01→D119-08 implementados en `live_desk.py` (render-only, REPORTE_SOLO). **Bugs estructurales críticos:** (1) P3 SIEMPRE VACÍO — `data.get("picks")` vs `apostar+watchlist`; (2) P6 NUNCA PARSEABA — regex single-line vs output multi-línea shadow_book. **Hallazgo operativo:** MOTOR cuota≤2.5 hit=48.2% vs cuota>2.5 hit=26.7% (gap 21.5pp — justifica H107-01). P6 ahora muestra 16 segmentos reales. 10 gaps pendientes: Tasks #48/49/50/54/59/62/63/64/65/66.
**Nodo-120 (FS Single-Source Cuotas — F6 Nodo-118):** Compuerta qualifying abierta 2026-07-19. `exportar_para_edge_calculator()` en `scraping/match_ledger.py` ahora incluye `single_source_fs` con `cuota1>0` (qualifying rounds con cuotas FlashScore, fuera del catálogo Kambi/Betplay). Campo `_cuota_source='flashscore'` para trazabilidad. Impacto: ~67 → ~100 partidos/día exportados (+33 qualifying). NO_DATA 51.9% hit rate ahora llega al edge_calculator. H120-01 pre-registrada (n_stop=20). 3 tests REGLA-T53.
**Nodo-121 (OddsAggregator Cuota Enrichment — F7 Nodo-118):** Hallazgo 2026-07-20: ss_fs con cuota1=None (qualifying no en PASO 1) SÍ están en betplay+rushbet via odds_aggregator (474 outcomes vs 122 de PASO 1). `enriquecer_ss_fs_con_aggregator()` en `scraping/match_ledger.py`: post-fusión, enriquece ss_fs sin cuota usando `fetch_all_odds(['betplay','rushbet'])` con match por apellido. CLI: `--build --enrich`. Resultado primer día: 5/11 enriquecidos (Aksu/Costoulas betplay=3.0, etc). `run_daily.py` PASO 1.5 actualizado con `--enrich`. H121-01 pre-registrada (n_stop=20). 3 tests REGLA-T53.
**Nodo-136 (Tier Detection CTI Fallback — D136-01):** `edge_calculator.py` L932-948: cuando `torneo_nombre=''` (H2H combinado sin metadato), lee CTI de `ranking_analysis.prediction.circuit_asymmetry`. Umbrales: CTI_max<0.6→itf, <1.5→challenger, ≥1.5→atp500. Resultado: 12 picks todos-atp500 → 4 atp500 + 5 itf + 3 challenger. Mega desbloqueado. D136-02 (gap): fix en `extraer_historh2h.py` para propagar `torneo_nombre` — pendiente sesión futura.
**Nodo-137 (Governor MOTOR Exclusion — D137-01):** `combo_governor.py` L359: MOTOR excluido del gate de combos (`all_stakes` solo incluye confianza+rival+betplay). MOTOR muestra como referencia con WARN>40% bankroll. Fix colateral: `rival_value_betslip.py` OSError subprocess sin intérprete Python → `sys.executable` prefix. Impacto: governor pasa de BLOCK permanente → permite combos.
**Nodo-138 (G2 Gate Multi-Signal — D138-01+D138-02):** D138-01: G2 evoluciona de binario n_h2h a 3 reglas: Regla-1=triple convergencia (original), Regla-2=STRONG+edge≥20%+kelly>0+axes≥2, Regla-3=edge≥35%+kelly>0+axes≥2. Reducción: G2 bloquea 2/18 picks (era 11/18). Raíz: double-penalization — MOTOR ya incorpora n_h2h=0 en kelly_kl/shrinkage/p_blend. D138-02: `favoritos_combo_builder.py` torneo='Desconocido' usa `_match_{partido}` como clave única (no colapsar todos en mismo torneo). 9 tests REGLA-T53 — 9/9 PASS.
**Nodo-147 (Live Score × Games Convergencia — Certeza Condicional en Tiempo Real):** 4 gaps identificados entre señal y realidad operativa: (1) `_parse_kambi_tennis_score()` existe pero nunca conectado al panel X3; (2) baseline pre-partido se pierde cuando evento entra EN_VIVO (solo ITF_VIVO tenía D142-T0 freeze); (3) p_modelo estático sin actualización condicional P(UNDER | games_played=k); (4) sin historial de cuotas games para sparkline de tendencia. D147-01: `_enrich_live_score(signals, live_events)` — mutación in-place con score_data desde `_parse_kambi_tennis_score()`, índice O(1) por event_id. D147-02: `_calcular_certeza_condicional(linea, direccion, games_played, sets_complete, current_set_home, current_set_away, zona)` — certeza_matematica (peor_caso = sets_remaining*13), p_condicional Gaussiano (DOMINANTE µ=18 σ=3, COINFLIP µ=23 σ=4.5), alerta_nivel CERTEZA/ALTA/MOD/BAJA. D147-03: `_freeze_baseline_if_needed()` → `reports/games_baseline_{fecha_compact}.json` inmutable (primer ciclo EN_VIVO). D147-04: panel X3 nuevas columnas Base(T0)/Live/Drift/Progreso/Certeza con badges HTML coloreados + fila verde si certeza_matematica. D147-05: `_write_games_odds_history()` append-only sparkline → `reports/games_odds_history_{fecha_compact}.json`. D147-06: banner blink CERTEZA_MATEMATICA + Telegram alert fire-once (`reports/certeza_fired_{fecha_compact}.json`). H147-01 pre-registrada (DOMINANTE p_condicional≥0.70 games_played>linea/2, n_stop=20). 6 tests REGLA-T53. SPEC — pendiente implementación.
**Nodo-146 (H2H_MODEL Universe: Bridge REGLA-HF-1 Gap en FAVORITOS_COMPUESTOS):** Gap arquitectónico: Vancouver 23 partidos todos @1.10–1.49 → REGLA-HF-1 los descarta en edge_calculator → 0 picks Vancouver en edge_report → favoritos_combo_builder sin piernas. REGLA-HF-1 correcta para singles pero NO bloquea combos (4 fav @1.25 = @2.44x). D146-01: `_find_latest_h2h()` + `_leer_h2h_favoritos()` en `favoritos_combo_builder.py` leen `h2h_results_enhanced_*.json` directamente — universo H2H_MODEL independiente de REGLA-HF-1. MAX_H2H_MODEL_PER_COMBO=2 (máx 2 piernas H2H_MODEL por combo para no diluir). Timing guard D145-02 integrado. Automático sin flag. 9 tests REGLA-T53 — 9/9 PASS. 2324 tests.
**Nodo-145 (Pipeline Bugs tipo_cancha+timing):** Root cause cascada 0 combos 2026-07-27: `tipo_cancha='N/A'` → bucket `unknown` (24% hit rate) → Kelly aplastado → pipeline_picks vacío → 0 CORE/Confianza. D145-01a: `h2h_extractor.py` (Playwright) `tipo_cancha` usa `superficie` como fallback + copia `hora`. D145-01b: `ninja_h2h_parser.py` (--api-mode) mismo fix en rama `list` de `load_matches()` + `_consolidate_result()` + hora propagada. D145-02: `edge_calculator.py` timing guard — skip partidos con hora > 15min en el pasado (Boitan G.A. @2.38 ya había perdido cuando pipeline corrió). 7 tests REGLA-T53 — 7/7 PASS. 2314 tests.
**Nodo-144 (Trazabilidad de Estrategia en Shadow Book — D144-01→D144-07):** Brecha E3 confirmada: 24 JSONL sin tag de estrategia. D144-01: `_build_record()` añade `strategy` top-level con default `'SIN_TAG'` (inmutable: nunca toca `pick_snapshot`). D144-02: `tag_strategy(fecha, player_names, strategy)` — update top-level sin tocar snapshot (patrón `update_alpha_flags`). D144-03: `combo_confianza_builder.py` llama `tag_strategy()` tras construir el plan para CORE/COBERTURA/SATELITE/MOONSHOT/ANCHOR/GCS. D144-04: `log_pick(fecha, jugador, cuota, pick_snapshot)` — registra pick individual; fix `favoritos_combo_builder._registrar_shadow_book()` que llamaba función inexistente silenciosamente. D144-05: `report()` añade segmento ESTRATEGIA agrupando settled por strategy (excluye SIN_TAG/HISTORICO). D144-06: `scripts/backfill_strategy.py` — cruce combo_registry (3 días: 22/23/25-jul, alta confianza nombre+fecha+subtipo) → 22 registros tageados; restantes 1287 → HISTORICO_SIN_TAG. D144-07: 7 tests REGLA-T53. Valores válidos: MOTOR/CORE/SATELITE/COBERTURA/MOONSHOT/ANCHOR/GCS/SAFE/WAS/MEGA/GAMES/RIVAL_VALUE/FAVORITOS_COMPUESTOS/SIN_TAG/HISTORICO_SIN_TAG. Deuda: D144-08 extender a MOTOR/SAFE/WAS/MEGA/GAMES. 7 tests — 7/7 PASS. 2307 tests.
**Nodo-143 (Match Ledger Torneo Metadata Propagation — D143-01):** Root cause de torneo="Desconocido" en todos los picks Kambi. `fusionar_dia()` en `scraping/match_ledger.py` usaba FlashScore como base del join y descartaba silenciosamente `tier|torneo_nombre|torneo_completo|pais|ranking1|ranking2|tournament_context` de Kambi. Evidencia 2026-07-25 mañana: 40 joins → 0/40 con torneo. Fix: 7 líneas post-`partido_merged`, guard fill-gaps `not partido_merged.get(campo)`. Cadena completa verificada: ledger→h2h_results_enhanced→detectar_tier()→lambda_efectivo→Kelly-KL. NOTA: el fix requiere re-correr PASO 2 (extraer_historh2h) después del ledger — el h2h lleva torneo_nombre del merged file. Evidencia producción tarde 2026-07-25: 17/17 joins con torneo, Suresh D. tier=challenger/torneo=Bloomfield Hills (antes atp500/Desconocido), 5 combos $15k generados (CORE @16.44x $7,000 — antes 0 combos). Gap D143-04: combo_confianza_builder.py L658 gate log usa `torneo` no `torneo_nombre` → muestra '?' (cosmético). Nodo-142 HUÉRFANO sin spec — D143-03 pendiente. 5 tests REGLA-T53 — 5/5 PASS. Regresión Nodo-118: 23/23 PASS.
**Nodo-141 (Kambi-Only Edge Report — D141-01→D141-03):** Solución definitiva al problema de 0 combos apostables cuando los picks son mayoritariamente ITF. D141-01: `scripts/filter_kambi_picks.py` — lee `edge_report_FECHA.json` y produce `edge_report_kambi_FECHA.json` con SOLO picks `kambi_disponible=True`. D141-02: `run_daily.py` PASO 3K — corre `filter_kambi_picks.py` inmediatamente después de PASO 3 (edge_calculator). `optional=True` si sin coverage. D141-03: `betplay_combo_builder._find_latest_edge_report()` — prefiere `edge_report_kambi_HOY*.json` (picks 100% apostables en Betplay); fallback a full report si no existe kambi de hoy. `combo_confianza_builder` ya elige el kambi report automáticamente por mtime más reciente (sin modificar). Hallazgo: PASO 1 ya es `extraer_partidos_api.py` (Kambi primario); el problema era que PASO 1b Playwright (+228 ITF) inundaba el edge_report. PASO 3K es el filtro correcto: no toca la colección de datos, solo el reporting para combos. 8 tests REGLA-T53 — 8/8 PASS.
**Nodo-140 (Kambi Gate — D140-01→D140-04):** 3 cirugías para cerrar la desconexión entre `fetch_kambi_coverage.py` (Nodo-90 D90-01, existía pero nunca corría) y los combo builders. D140-01: `run_daily.py` PASO 1c — `fetch_kambi_coverage.py` corre diario ANTES de edge_calculator → `kambi_disponible` fresco en cada pick. D140-02/03: `_filter_kambi_available()` en `betplay_combo_builder.py` — pre-filtro en `build_safe_combos()` (edge_pick_map), `build_was_combos()` (watchlist), `build_mega_combos()` (edge_tier_map + pool). D140-04: gate `is_player_available()` en `combo_confianza_builder.py::_extract_and_categorize()` — excluye favoritos ITF/torneos sin Betplay antes de categorizar. `None` = pass-through (sin coverage). Hallazgo clave: campo `kambi_disponible` existía en edge_report desde Nodo-90 — nunca fue consumido. 9 tests REGLA-T53 — 9/9 PASS. Deuda catalogada: D141-01 unificar 5 normalizadores de nombres en silos.
**Nodo-139 (Kambi-First Combo Builder — D139-01→D139-07):** Inversión arquitectónica: en vez de buscar picks en Kambi post-hoc, parte del universo Kambi NOT_STARTED cuota [1.10,1.80] (136 events). D139-01: `_fetch_kambi_betting_universe()` — HTTP listView, filtra cuota, convierte UTC→Colombia HH:MM. D139-02: `_match_to_predictions()` — `_apellido_kambi()` (último token) + `_apellido_pick()` (quita iniciales ≤2 chars, toma primer token) + bigram Jaccard ≥0.70 → TIER_A (apostar/watchlist), TIER_B (sin_edge + p_modelo>p_implied). D139-03: gates G_EDGE (edge>0) + G_CONF (p≥0.55). D139-04: `_build_kambi_combos_kf()` — SIN cap de cuota (COMBO_MAX_CUOTA=7.0 era matemáticamente inválido: EV>0 cuando ∀leg edge>0 independiente del producto). EV gate EV_MIN_COMBO=2% + MIN_P_COMBO por n_legs. D139-05: half-Kelly con portfolio factor RHO=0.15, cap [500, 3% bankroll]. D139-06: KB_N.bat en Desktop. D139-07: `--kambi-first` flag en `betplay_combo_builder.py`. Fix colateral: `_find_outcome()` en `combo_confianza_builder.py` — si último token ≤2 chars (inicial), usa primer token como apellido (bug McFadzean L.→'l'). H139-01/02 pre-registradas. 15 tests REGLA-T53 — 15/15 PASS.
**Nodo-135 (EvalGames Live API Fix):** D135-01: `_extract_games_cuota_live(event_id, direccion, linea)` — HTTP a `betoffer/event/{id}.json` via `urllib.request` (no listView que devuelve `[]` para ITF). D135-02: filtro `" - Set "` y `"Juego"` excluye sub-mercados set/juego; solo acepta match-total "Total de juegos". D135-03: call-site `_check_games_convergencia()` actualizado a `matched["event"]["id"]`. Evidencia real 2026-07-21: listView 225 STARTED → 43 con Total de juegos (todos ATP/WTA/Challenger), 0 ITF. betoffer/event devolvió cuotas reales: Brockmann/Jacquemot OVER @ **1.55**, Suresh UNDER @ **1.65** (drift −6.8%). 4 tests REGLA-T53 — 4/4 PASS.
**Nodo-133 (Games Live Convergencia):** D133-01: `_kambi_started_events()` — 1 HTTP call a Kambi listView STARTED + liveEvents.json, sin Playwright. D133-02: match por `event_id` (primario) + apellido normalizado (fallback). D133-03: `_check_games_convergencia()` llamada en `_background_refresh()` cada 15s — clasifica señales ALTA como EN_VIVO/PRE_PARTIDO/TERMINADO. D133-04: si ≥2 ALTA EN_VIVO → anti-flood `_fired_games_{fecha}.json` (cap 10/día) → `subprocess.Popen(betplay_combo_builder --games --live)` fire-and-forget. D133-05: escribe `reports/games_live_YYYYMMDD.json`; `_build_x3_games()` enriquece señales con `estado_live`, `cuota_live`, `drift_pct`. D133-06: umbral ≥2 ALTA EN_VIVO. Banner CONVERGENCIA ACTIVA parpadeante + columna Estado en X3 panel. Sin cambios en live_edge_monitor, close_snapshot_server ni betplay_combo_builder. 5 tests REGLA-T53 — 5/5 PASS.
**Nodo-129 (LiveDesk AutoRefresh — 3 capas):** D129-01: `_STATE_CACHE` en memoria TTL 20s + thread daemon `_background_refresh()` cada 15s — primera request =60s, todas las demás =<1s. D129-02: `DeskHandler.do_POST()` en `live_desk.py` + `_notify_live_desk()` en `close_snapshot_server.py` tras `/check-and-close` y `/live-check` — n8n invalida cache explícitamente via POST a `:7780/api/refresh`. D129-03: `_data_freshness()` mtime real de edge_report/live_odds_history; header muestra "datos de hace 2m 15s" en lugar de timestamp de request. JS interval reducido 30s→12s. Latencia máxima real: 90s → ~12s. 3 tests REGLA-T53.
**Nodo-128 (Wplay P8 Multi-Book Alias):** D128-01: alias apellido en TODOS los feeds de `_build_p8_books()` (`"Botic Van De Zandschulp"` → alias `"van de zandschulp"`) para matchear picks con nombres abreviados. D128-02: games_signal_report players injected en P8 (strip inicial trailing `"B."`, seen_oids guard). Resultado: P8 pasa de 0 a **35 picks** con betplay+rushbet+wplay activos. Wplay SSR independiente de Kambi 429. 3 tests REGLA-T53.
**Nodo-127 (GamesSignal ITF OutcomeID Fix):** CERRADO 2026-07-21 (`582090c`). D126-04: `procesar_partidos()` Intento2 ahora filtra `state!="NOT_STARTED"` + usa `_apellido()` + excluye dobles. Intento3 desactivado (un solo apellido matcheaba partidos aleatorios). D126-05: `seen_outcome_ids: dict[int,str]` descarta IDs genéricos reutilizados en 2+ partidos (ID `4265916952` aparecía en 30+ matches ITF). Resultado: GAMES combos solo ATP250+ con IDs únicos. Primer combo válido: GamesA @2.64x (Van De Zandschulp UNDER 25.5 + Oliynykova OVER 19.5). 3 tests REGLA-T53 — 6/6 PASS (N126+N127).
**Nodo-126 (GamesSignal 3 Bugs Fix):** 3 bugs en `games_signal_calculator.py::_buscar_event_id_kambi()` detectados 2026-07-21. D126-01: `split()[-1]` extraía inicial en vez de apellido → nueva `_apellido()` toma último token no-inicial. D126-02: sin filtro dobles → `if "/" in name: continue`. D126-03: `o_mas["odds"]` KeyError → `.get("odds",0)`. 3 tests REGLA-T53. D126-04 pendiente: filtro `state=="NOT_STARTED"` para excluir odds live.
**Nodo-123 (Auditoría Dashboard Integraciones v2):** Auditoría 2026-07-20 — todos los 6 pendientes del spec eran falsos. D122-01→D122-04: `live_desk.py` ya tenía `_fetch_all_odds()` live (L1866), badges STEAM/ATN (L927-991), mensaje X3 accionable (L1051-1054), systemd service habilitado. D122-05: `_leer_matches_ranking_only()` ya existe vía `--matches`. D122-06: `KambiLiveClientReal` (`live_edge_monitor.py` L158) operativo desde D97-15 2026-07-14 — Fable encontró endpoint público `kambicdn.com` sin DevTools. Lección: auditar código real antes de spec.

---

## 6. MAPA DE ARCHIVOS CLAVE

```
── PIPELINE ─────────────────────────────────────────────────────────────────
extraer_URL_partidos_version2.py  ← PASO 1 PRIMARIO (Playwright entity IDs FlashScore)
extraer_partidos_api.py           ← PASO 1 FALLBACK (API — vulnerable a homónimos)
extraer_historh2h.py              ← PASO 2 (Playwright sin flags | --api-mode fallback)
edge_calculator.py                ← PASO 3: Kelly-KL 5 capas
generar_tabla_favoritos2.py       ← PASO 3.5: revisión humana
trader_ev_tenis.py                ← PASO 4: Hedge Fund Layer + CPPI
combo_confianza_builder.py        ← PASO 4.3: CORE/Satellite/Moonshot
betplay_combo_builder.py          ← PASO 4.4-4.57: links Betplay
rival_value_betslip.py            ← D68-07: micro-Kelly rival H88-01 (n=3/30, shrink=5.7%)
betslip_registrar.py              ← PASO 4.6: registro + loop calibración
run_daily.py                      ← Orquestador PASO 0→4.3 + settle

── SHADOW BOOK + OBSERVABILIDAD ─────────────────────────────────────────────
shadow_book.py                    ← CLV: log_picks | close_snapshot | settle | report
pipeline_tracker.py               ← READ-ONLY (--section shadow|confianza|drift|portfolio)
pre_game_validator.py             ← cron 0 9-23: kelly_kl=0.0 BLOCK | n<8 WARN
close_snapshot_server.py          ← HTTP :8765 bridge (Nodo-73) — timing exacto por partido
close_snapshot_trigger.py         ← cron */10 9-23h venv/bin/python3 FALLBACK (si n8n cae) — fix 2026-07-10
check_contradictions.py           ← cron lun 9am: CLAUDE.md vs nodos (Vacío 3)

── n8n AUTOMATION (Nodo-73, systemd) ─────────────────────────────────────────
n8n Docker :5678                  ← Tennis Close-Snapshot Timing workflow
tennis-snapshot-bridge.service    ← systemd, enabled, PID en logs/snapshot_bridge.log
n8n_push_workflow.py              ← sube/actualiza workflow via API REST

── DATOS CRÍTICOS ───────────────────────────────────────────────────────────
data/calibracion_edge.json              ← Thompson Beta priors (fuente de verdad)
reports/shadow_book/sb_YYYY-MM-DD.jsonl ← append-only, inmutable en predicción
validation/preregistered_hypotheses.json ← H52-01→H121-01 (19 hipótesis), NO modificar sin decisión
validation/hypothesis_tracker.py        ← sprt_verdict() + llr_update() (Nodo-64)
docs/DECISION-LOG.md                    ← D-01→D-10 + E-01→E-05 + C-01→C-07

── MOTOR DE PREDICCIÓN ──────────────────────────────────────────────────────
analysis/rivalry_analyzer.py      ← Erdős+Markov+GCS+PhantomGuard (núcleo)
analysis/markov_analyzer.py       ← PELT + surface_context_discount
config.py                         ← detectar_tier() — fuente única de tiers
core/data_contract.py             ← PICK_STATUS_NO_DATA + DataContract v2: validate_artifact() 6 fronteras (C1 Nodo-67)
core/player_registry.py           ← entity resolution canónica (Nodo-51)

── INSTRUMENTOS FASE 4 (REPORTE_SOLO, no cambian decisiones) ────────────────
analysis/drift_monitor.py | flb_curve.py | pattern_audit.py
analysis/conformal_band.py | rho_empirical.py | velocity_monitor.py

── ML (SUSPENDIDO hasta modelo > 78% held-out) ──────────────────────────────
generar_dataset_plus.py | aplicar_enhancer.py

── SUSPENDIDO (isla Flask) ───────────────────────────────────────────────────
app.py | routes/ | models/ | services/ | database.db
```

---

## 7. BUGS ACTIVOS

| Bug | Estado |
|---|---|
| Auditoría Nodo-86 (15 hallazgos) | ✅ 12 fixes D87-01→D87-11 + D64-01 aplicados 2026-07-11 — 18/18 tests REGLA-T53 verificados 2026-07-12 en WSL. Gap D87-07/D87-10: embebidos en `trader_ev_tenis.main()`, requieren refactor para aislar — candidato sesión futura |
| prediccion_ganador top-level=None | ✅ RESUELTO — usar `ranking_analysis.prediction.favored_player` |
| Edge falso historial corto (n<8) | ✅ RESUELTO — Nodo-63 `_MIN_HISTORY_FOR_DECAY=8` |
| Phantom Identity API homónimos | ✅ RESUELTO — Nodo-72 `_detect_phantom_identity()` + Playwright PRIMARIO |

---

## 8. PROTOCOLO DE TRABAJO

```bash
# 1. Buscar en git antes de implementar (GIT-FIRST — obligatorio)
git log --all --oneline -- '*keyword*'
git show COMMIT:backend/archivo.py    # recuperar si existe

# 2. Baseline antes de modificar
python -m pytest tests/ --no-cov -q  # 1986 passed

# 3. Syntax check después de editar
python -c "import ast; ast.parse(open('archivo.py').read()); print('OK')"

# 4. Graphify antes de grep (grafo existe)
graphify query "<pregunta>"   # orientarse primero, grep solo para líneas específicas
```

**SDD:** Ningún código sin Nodo en `.spec/01_Nodos/`. Ver `PRE_IMPLEMENTATION_CHECKLIST.md`.
**URLs:** `ninja` API ≠ `flashscore.com` DOM. NUNCA derivar URLs browser desde URLs API.

---

## 9. RECORDATORIOS CRÍTICOS

**Guards No-Ruina:** HF-1 (cuota<1.50 nunca en pool) | HF-5 (KGR<0 → NO DESPLEGAR) | VaR auto-ajustado (no calcular a mano) | `--torneo-tipo` filtra por tier — NO mezclar GS con ITF.

**Calibración:** p_prior automático (tier+superficie, `calibracion_edge.json`). confidence: STRONG≥0.60 | MOD 0.55-0.60 | LOW<0.55. Shrinkage n/(n+20): n<10 = revisar antes de apostar.

**Datos:** Predicción anidada en `ranking_analysis.prediction.favored_player`. Phantom alerta: ranking=None + n>20 + oldest>365d → LOG_PLAYWRIGHT_CANDIDATE. `_MIN_HISTORY_FOR_DECAY=8`.

**Testing:** REGLA-T53 (función real, nunca hardcodear fórmula). 2204 tests: no romper.

**Combo Builder:** correr trader POR TIER antes del combo builder. REGLA-KAMBI-1: `||replace` (no `||append`).

**Tamp (proxy :7778):** dependencia dura — si Claude Code no responde, ver `TROUBLESHOOTING.md`. Arreglo rapido: `systemctl --user restart tamp`.

---

## 10. POLÍTICA DE PRECEDENCIA (§1.2 Vacío 3, FABLE_02)

1. `.spec/01_Nodos/` es **historia inmutable** — no editar; añadir nueva entrada o marcar `SUPERSEDED por [[Nodo-XX]]`.
2. **CLAUDE.md es VISTA derivada** — si contradice al nodo más reciente, CLAUDE.md está desactualizado.
3. `python3 check_contradictions.py` (cron lunes 9am) compara CLAUDE.md vs últimos 10 nodos.

---

## 11. TAXONOMÍA DE ESTRATEGIAS — LAS 12 FORMAS DE GENERAR COMBOS

> Spec completo: [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]
> Todas leen `edge_report_*.json` como fuente única. Diferencia: qué hacen con la señal.

### Generadores y sus estrategias

```
edge_calculator.py  (PASO 3 — corre siempre, fuente única)
   │
   ├─ trader_ev_tenis.py          → (1) EL MOTOR
   ├─ combo_confianza_builder.py  → (2) CORE  (3) SATELITE  (4) MOONSHOT
   │                                 (5) COBERTURA  (6) ANCHOR  (7) GCS
   ├─ betplay_combo_builder.py    → (8) SAFE  (9) WAS  (10) MEGA  (11) GAMES
   └─ rival_value_betslip.py      → (12) RIVAL VALUE
```

| # | Nombre | Flag/Comando | Piernas | Cuota | Stake | Condición clave |
|---|--------|-------------|---------|-------|-------|-----------------|
| 1 | **EL MOTOR** | `trader_ev_tenis.py` | 1 | variable | Kelly-KL | edge>5% + KGR>0 + todos los gates |
| 2 | **CORE** | `combo_confianza_builder.py` | 4–7 | @2–5x | $2k–5k | Cat-A/B, P(win)≥25% |
| 3 | **SATELITE** | `--fase 2+` | 5 | @5–8x | $2k–3k | Cat-C1 disponible (conf≥60%) |
| 4 | **MOONSHOT** | `--fase 3+` | 5 | @15–35x | $1k–2k | ≥2 picks Cat-C conf≥57% |
| 5 | **COBERTURA** | `--fase 4` | 4–7 | @2–4x | $1k–2k | Fase 4 — hedge del CORE |
| 6 | **ANCHOR** | `--anchor` | 4–5 | @4–35x | $1.5k | prioridad≥75 + cuota≥1.65 |
| 7 | **GCS** | automático en hierba | 2–3 | @1.5–3x | $200–500 | gcs_active + tier≥ATP500 |
| 8 | **SAFE** | `betplay_combo_builder.py --safe` | 2 | @3–12x | $1k | P(ambos)≥25%, torneos distintos |
| 9 | **WAS** | `--live` (incluido) | 2–3 | @4–25x | $5k | edge≥10% + señal Markov explícita |
| 10 | **MEGA** | `--mega` | 6–10 | @100–1000x | $500 | Dispersión DIFF (std≥0.04) |
| 11 | **GAMES** | `--games` | 1 | @1.8–2.1x | $1k–2k | mercado Over/Under en Kambi |
| 12 | **RIVAL VALUE** | `rival_value_betslip.py` | 1 | @2.5–8x | $2k | edge_fav≤−10% (H88-01) |

### Hit rates reales — shadow book 2026-07-01→14 (picks individuales, n=231 settled)

| Segmento | Hit% | ROI flat | IC 95% | Estado |
|----------|------|----------|--------|--------|
| **GCS** (H60-01) | **64.8%** | — | — | GRADUADA n=54 — MEJOR ESTRATEGIA |
| **RIVAL VALUE** (H88-01) | **100%** | +275% | [52.6,100] | n=3 — pre-graduación, no significativo |
| **VARIABLE** (edge≤0) | **62.5%** | −24.7% | [42.7,78.8] | n=24 — pre-graduación (ROI neg por cuotas bajas) |
| **Grand Slam** | **47.4%** | **+40.2%** | [27.3,68.3] | n=19 — mejor ROI real |
| **GS+watchlist+edge≥20%** | 37.5% | **+53.8%** | [13.7,69.4] | n=8 — señal fuerte pero n pequeño |
| **season_transition** | 42.2% | +11.9% | [29.0,56.7] | n=45 |
| **ANCHOR** (edge>0) | 34.3% | −5.7% | [28.2,41.0] | n=207 |
| **Challenger** | 37.2% | −5.5% | [27.3,48.3] | n=78 |
| **WATCHLIST** | 34.9% | −5.2% | [27.7,42.8] | n=149 |
| **ITF** | 36.9% | −13.5% | [28.5,46.2] | n=111 |
| **EL MOTOR** (APROBADO) | 30.0% | −21.1% | [14.5,51.9] | n=20 — gates muy estrictos = picks de alta cuota |
| **WAS** (H52-01) | — | — | [18.6,49.9] | NO GRADUABLE |

**Lectura clave:** El ROI negativo en picks individuales NO indica que los combos fallen — los combos multiplican cuotas. Un pick CORE @1.40 con 34% hit rate en flat-1u puede ser correcto en kelly. La estrategia GCS es la única GRADUADA con hit% formal ≥breakeven.

**Hallazgo 2026-07-14:** RIVAL VALUE = discriminador `edge_fav ≤ −15%` suficiente aunque señales secundarias contradigan. Es la estrategia con mayor potencial de ROI (41.25x combinada hoy).

---

## graphify

Grafo de código en `graphify-out/` (949 nodos, 1302 edges + 91 nodos .spec/ — código Python + memoria semántica, reindexado 2026-07-13).

- **Visualización 2D:** http://localhost:7779/graph.html (Nodo-83 — vis.js, F5 = datos frescos)
- **Visualización 3D:** http://localhost:7779/graph3d.html (Nodo-84 — Three.js/ForceGraph3D, rotación orbital)
- **Verificar:** `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7779/graph.html` → debe retornar `200`
- Antes de grep: `graphify query "<pregunta>"` | `graphify path "<A>" "<B>"` | `graphify explain "<concepto>"`
- Actualizar tras cambios: `graphify update .` → F5 en browser muestra nuevos nodos sin regenerar HTML
- Para incluir `.spec/` docs: `export ANTHROPIC_API_KEY="sk-ant-..." && graphify .`
- Gestión servicio: `systemctl --user start|stop|restart graphify`

# CLAUDE.md — Tennis Prediction & Betting Engine

> Last updated: 2026-07-16 (Nodo-105 Knowledge Graph Navigation: Zettelkasten — bioluminescent sprites, ego-network click, MOCs, PageRank, estado facet, huérfanos ritual. 108 nodos indexados)
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

## 5. ESTADO ACTUAL — 2026-07-17

| Métrica | Valor |
|---|---|
| Tests | **2118 passed, 1 failed** (verificado 2026-07-17). `test_nodo51_f3_02_budget_processes_itf_before_grand_slam` pre-existente. `test_prior_bajo_no_se_ve_afectado` intermitente (estado global, pasa en aislamiento). +106 tests nuevos vs sesión anterior (Nodos 107-113). |
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

**Nodos completos:** 51-63, 64-71, 72, 73, 78, 86-113 — detalles en `.spec/01_Nodos/Nodo-XX.md`
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
validation/preregistered_hypotheses.json ← H52-01→H88-01 (17 hipótesis), NO modificar sin decisión
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

**Testing:** REGLA-T53 (función real, nunca hardcodear fórmula). 1827 tests: no romper.

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

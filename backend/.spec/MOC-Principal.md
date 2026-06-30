# MOC Principal — Tennis Prediction Engine

> **Última actualización:** 2026-06-30 (Nodo-47 ✅ RESUELTO — bug crítico: _inject_kambi_ranking sobreescribía rankings ATP reales por mismatch "Nombre Apellido" vs "Apellido Nombre". Causa directa de fallo Glinka/Mayo. Fix optimizado: O(1) key invertido + slow path. Nodo-45 ✅ THF activo | **Tests:** 1438 passed ✅)
> **Calibración global:** n=1353 | 850W/503L hit=62.8% | clay GS: 25W/8L p=75.8% | grass: 92W/73L=55.7%
> **ALERTA Nodo-32 (Jun 19-22):** APOSTAR hit%=26.7% (8W/22L) | Markov HOT=8.3% | Golden Zone=12.5% | 86% picks LOW | ROI=-21.1% | Phantom Edge + Markov decorativo + Golden Zone ciega → 5 fixes spec
> **Hallazgo Nodo-28 (18-jun):** 80.8% INFLADO por data leakage (H2H post-partido). Backtest limpio: **36/52=69.2%** (clay 76% | grass 67% | hard 64%). 6 predicciones cambiaron al quitar leak. 16 fallos → análisis post-mortem reveló Circuit Asymmetry → Nodo-29.
> **Fixes 2026-06-09 sesión 1:** ninja_h2h_parser.py (import re + match_id) ✅ | validar_con_api.py (superficie) ✅ | betplay_combo_builder.py (--live + combos parciales) ✅
> **Fixes 2026-06-11 sesión 3:** FlashScore feed tipo=13 (23→146 singles mañana, +6×) ✅ | combo builder multi-plan merge (stakes reales por tier, fallback legacy) ✅ | REGLA-FS-1/2 + REGLA-CB-1/2 documentadas
> **Fixes 2026-06-11 sesión 4:** markov_analysis path bug (edge_calculator:535 partido→pred) — Nodo-18/19 silenciados desde siempre ✅ | B-02 _load_p_prior(superficie,tier) jerarquía tier+superficie ✅ | B-03 header dinámico ✅ | _print_individuales usa p_historica_usada per-pick ✅
> **Fixes 2026-06-09 sesión 2:** Smart Combo Scoring (_score_combo) ✅ | Cobertura Exclusión (_select_with_cobertura + max_app) ✅ | Kambi `||replace` descubierto ✅ | GitHub Pages redirect botón + target=_blank ✅
> **Fixes 2026-06-10 sesión 1:** betslip_registrar.py ✅ (loop+stake+P&L) | ML pipeline desbloqueado ✅ (386 registros, leakage fix) | LR=71.7% no supera heurístico — conectar cuando >78%
> **Fixes 2026-06-10 sesión 3:** betslip_registrar --listen ✅ | bookmarklet v2 POST automático ✅ | cero copy-paste | pipeline completo ejecutado (228 matches, 43 picks APOSTAR, 8 combos)
> **Validación prod 2026-06-07:** 72.2% accuracy (57/79) multi-torneo | ITF 82.8% | ATP500 70.6% | Challenger 64.5%
> **MODO API 2026-06-07:** Kambi + FlashScore Ninja API → pipeline completo en ~45s (vs 40+ min Playwright) | cuota_es_real=True
> Este es el documento de entrada del vault. Todo lo demás cuelga de aquí.

---

## Norte Real (leer en 30 segundos)

**Misión:** Apostar únicamente donde `P_modelo > P_implícita_bookmaker + 5%`, con Kelly-KL.
**Métrica de éxito:** P&L positivo acumulado — no accuracy del modelo.
**Estado hoy:** Pipeline 100% operativo ✅ | **P&L acumulado: +$25,000** | Bankroll $125k | Hedge Fund Layer activo
- **Calibración global:** n=1241 | 784W/457L hit=63.2% | clay GS: 25W/8L p=75.8%
- **Validación prod 2026-06-18:** 52 partidos, **69.2% accuracy limpia** (36/52) — clay 76% | grass 67% | hard 64% (80.8% era inflado por data leakage)
- **Validación prod 2026-06-07:** 80 partidos, 72.2% accuracy (57/79) — ITF 82.8% | ATP500 70.6% | Challenger 64.5%
- **Señales validadas:** HOT Markov=84.6% (55W/10L) | COLD=0% (0W/10L) | cuota 4+=100% ROI+420% | cuota 1.50-2=100%
- **Features activas:** API Ninja + Kambi API (cuotas reales Betplay) + --all-tournaments + --torneo-tipo filtra + VaR auto-ajuste + shrinkage + density
- **Fixes 2026-06-09 sesión 1:** import re + match_id en ninja_h2h_parser.py ✅ | superficie en validar_con_api.py ✅ | betplay_combo_builder.py --live ✅
- **Fixes 2026-06-09 sesión 2:** _score_combo (combo_EV×diversity×regime×alpha) ✅ | _select_with_cobertura (max_app) ✅ | `||replace` vs `||append` Kambi ✅ | docs/bp/index.html target=_blank ✅
- **Fixes 2026-06-10:** betslip_registrar --listen ✅ | bookmarklet v2 POST automático a localhost:5001 ✅ | ML leakage fix ✅ | pipeline run completo ✅ | PASO 3.5 (tabla) movido antes de PASO 4 — revisión humana antes de apostar ✅
- **B-02 ✅ RESUELTO 2026-06-11:** _load_p_prior(superficie, tier) — jerarquía por_superficie_y_tier → fallback_por_tier → por_superficie → global. clay+challenger 0.697→0.590 | grass+atp500 0.569→0.650
- **Nodo-17/18/19/20/21 ✅** — pesos 5-tier + density + shrinkage + H2H Immunity + PELT Recency + PageRank Erdős

---

## Mapa del Vault

```
.spec/
├── MOC-Principal.md              ← ESTÁS AQUÍ — entry point
│
├── 00_Constitution/
│   └── Mandatos-No-Negociables.md    ← las reglas que nunca se rompen
│
├── 01_Nodos/                         ← todos al mismo nivel (plano en disco)
│   ├── [[Nodo-01-Edge-Calculator]]           (código ✅ | validación en prod pendiente)
│   ├── [[Nodo-02-Markov-Changepoint]]         (activo en prod desde 2026-05-29)
│   ├── [[Nodo-03-Scraper-Fix]]                ✅ RESUELTO
│   ├── [[Nodo-04-Dataset-Fix]]                ✅ RESUELTO
│   ├── [[Nodo-05-Validacion-API]]             (código ✅ | n≥30 pendiente)
│   ├── [[Nodo-06-Erdos-Graph]]                ✅ RESUELTO
│   ├── [[Nodo-07-Strangler-Fig]]              (Fase 1 ✅ | Fase 2 ✅ T07-09 ✅ SequentialH2HExtractor eliminado)
│   ├── [[Nodo-08-File-Selection-Bug]]         ✅ RESUELTO
│   ├── [[Nodo-09-API-Status-Keys]]            ✅ RESUELTO
│   ├── [[Nodo-10-Surface-Propagation]]        ✅ RESUELTO — surf_w 0.49–0.69 en prod (efecto Nodo-07)
│   ├── [[Nodo-11-Inventario-Scripts-Legado]]  ✅ CERRADO — decisiones ejecutadas, disco verificado 2026-05-30
│   ├── [[Nodo-12-Inventario-Infraestructura-Legado]] ✅ EJECUTADO — limpieza infraestructura 2026-05-30
│   ├── [[Nodo-13-Trader-EV-Tenis]]           ✅ IMPLEMENTADO v2.0 — deploy: individuales + combos + hedge fund layer (2026-06-01)
│   ├── [[Nodo-14-Validacion-Live-Conexiones]] ✅ PRIMERA VALIDACIÓN LIVE — Parry @ 4.50 ganó | 5 conexiones ocultas TTC (2026-05-30)
│   ├── [[Nodo-15-Portfolio-HedgeFund]]        ✅ IMPLEMENTADO — Sistema Cobertura Exclusión + Portfolio Kelly + VaR/CVaR (2026-06-01)
│   ├── [[Nodo-16-Multi-Torneo-Pipeline]]      ✅ IMPLEMENTADO — --max-matches 80 + --all-tournaments + Roland Garros filter fix (2026-06-02)
│   ├── [[Nodo-17-Calibracion-Por-Tier]]       ✅ FASE 1 — surface fix + prior estratificado + λ por tier (2026-06-03)
│   ├── [[Nodo-18-PELT-Recency-Alpha]]         ✅ COMPLETADO — calcular_recencia_regimen + factor_alpha_temporal → λ_efectivo (2026-06-03)
│   ├── [[Nodo-19-H2H-Immunity-Dampener]]      ✅ COMPLETADO — calcular_h2h_immunity: HOT×h2h_wr<0.30→0.85 | >0.70→1.12 (2026-06-03)
│   ├── [[Nodo-20-PageRank-Erdos-Quality]]     ✅ COMPLETADO — pagerank_grafo + quality_multiplier en caminos transitivos (2026-06-03)
│   ├── [[Nodo-21-Pesos-Diferenciados-Por-Tier]] ✅ COMPLETADO — 5 tiers SNR + density_confidence + shrinkage + K-factor ELO (2026-06-03)
│   ├── Nodo-22-API-Integration-Kambi-Ninja       ✅ COMPLETADO — Kambi+FlashScore API reemplaza Playwright (~45s vs 40+min) (2026-06-07) — sin spec standalone, documentado en Sprint-Pipeline Fase 24
│   ├── [[Nodo-23-Cross-Tier-Mega-Combos]]      🔧 IMPLEMENTADO — build_mega_combos() + --mega flag + ancla/satélite (2026-06-14)
│   ├── [[Nodo-24-Bookmaker-Blindness-Scoring]]  🔧 IMPLEMENTADO — BBI + gap_flag + MPQ + golden_zone en edge_calculator + mega scoring (2026-06-14)
│   ├── [[Nodo-25-Dispersion-Guard-Safe-Combos]] 🔧 IMPLEMENTADO — 4 Guards + Safe Combos Beta Book (2026-06-14)
│   ├── [[Nodo-26-Cross-Sectional-Signals]]     🔧 IMPLEMENTADO — Circuit Breaker + Line Movement + Ranking Preservation + Meta-Markov + CV Guard (2026-06-14)
│   ├── [[Nodo-27-Pipeline-Tracker-Observabilidad]] 🔧 IMPLEMENTADO — pipeline_tracker.py: 7 secciones READ-ONLY | Hallazgos: STRONG=100%, HOT Markov=64%, NEUTRAL=6.7% (ruido), cuota 3-4=75%, ATP500 grass=18% ⚠️ (2026-06-17)
│   ├── [[Nodo-28-Conditional-Decomposition-Metamodel]] ✅ COMPLETO — Fase 1: common_opponents filtrado por superficie | Fase 1.5: SkillFactor + Surface Alpha + Volume Confidence | Fase 2: Triple Alignment Score (STRUCTURAL_ALPHA + CONTESTED_ALPHA + net_alignment) — 1113 tests ✅ (2026-06-19)
│   ├── [[Nodo-29-Circuit-Asymmetry-Deflator]]    ✅ COMPLETO — CAD: deflactor form/ELO por asimetría de circuito competitivo + SoS dinámico. Backtest 38/52=73.1% (14 fallos, ninguno de tipo circuito). Schoen/Boogaard: predicción correcta 55.3%. circuit_warning activo 7/80 picks en prod (2026-06-28)
│   ├── [[Nodo-30-Tournament-Momentum-Output-Signals]] 🔧 IMPLEMENTADO — 30 tests T30-01/30 ✅ | F6 player_profitability.py ✅ | F7 JUGADOR RENTABLE ✅ | 1143 tests. Origen: Sprint-Normalizacion-19jun + caso Carnicella (2026-06-20)
│   ├── [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] 🚨 CRITICO — Phantom Edge (p_modelo~0.51 → "22% edge"), Markov decorativo (log1p comprime factor), Golden Zone ciega, ?_? calibracion, ITF sin fallback. 5 fixes, 29 tests. Origen: pipeline_tracker W25-26 hit%=26.7% (2026-06-22)
│   ├── [[Nodo-37-Combo-Confianza-Builder]] ⛔ SUPERSEDED — Progresión C5→C20 sin aislamiento de riesgo. Reemplazado por Nodo-38 (CORE/SAT/MOON). Ver Nodo-38 §1.1: Da Silva + Cardozo (2 Cat-C) en C11 destruyeron 9 picks ganadores.
│   ├── [[Nodo-38-Portfolio-Aislamiento-Riesgo]] ✅ COMPLETO — CORE/Satellite/Moonshot con aislamiento Cat-C. 28 tests T38-01/28 ✅. Evidencia fundacional verificada 26-jun: 15/15 picks genuinos. Mutación T38-26/27/28 confirma arquitectura. (2026-06-28)
│   ├── [[Nodo-38B-Cobertura-Expandida-Sin-CatC]] ✅ COMPLETO — Cuando no hay picks Cat-C, redistribuye SAT+MOON budget → 6 combos de cobertura con .bat. Opus análisis + Sonnet implementación. 25/25 tests pass (2026-06-27)
│   ├── [[Nodo-39-Kambi-Filtro-Fecha]] ✅ COMPLETO — Kambi devuelve eventos futuros sin filtro → PASO 1 ahora filtra por UTC date. 199→40 partidos reales. Data pipeline contaminada resuelta (2026-06-27)
│   ├── [[Nodo-40-Games-Sets-Signal-Layer]] ✅ COMPLETO — Alpha ortogonal al ganador: diff→sets/games, 5 fases. games_signal_calculator.py + --games flag + betslip ground truth + pipeline_tracker S-40 + auto-calibración. 37 tests ✅. calibracion_n dinámico. (2026-06-28)
│   ├── [[Nodo-43-PELT-Cold-Rival-Promo-Filter]] HALLAZGO — rival COLD conf≥0.60 = alpha para promo combos bloqueados por T33-01/FIX-3. n=2 descubrimiento 2026-06-29. Caso: Ilagan @2.05 + Mayo @2.18 = 4.47x promo.
│   ├── [[Nodo-44-Watchlist-Alpha-Signal]] HALLAZGO — framework unificado: watchlist edge≥10% + cuota≥2.0 + señal Markov. Bookmaker sobrevalora marca, modelo captura estado actual. Validado 2026-06-29: Carreno @3.30 (edge=21.1%) + Safiullin @2.65 (edge=12.8%) ambos GANARON. PCRS (Nodo-43) es subconjunto de WAS.
│   ├── [[Nodo-45-Temporal-History-Fallback]] ✅ IMPLEMENTADO — match_id=None → busca en h2h_results previos (7 días). D45-01 tests ✓ | D45-02 función ✓ | D45-03 refactor ✓ | D45-04 routing ✓
│   ├── [[Nodo-46-Markov-Surface-Context-Discount]] HALLAZGO — estado COLD/HOT mezclando superficies. Evidencia real: 1/3 fallos Cary (Watanuki). Pendiente más n antes de implementar discount.
│   └── [[Nodo-47-Inject-Kambi-Ranking-Guard-Bug]] ✅ RESUELTO — guard `rankings_data.get(normalized)` fallaba por formato "Apellido Nombre" vs "Nombre Apellido". Kambi estimate sobreescribía ATP real. Fix: fast path O(1) invertido + slow path intelligent match. 1438 tests ✅
│
├── 02_Sources/
│   └── Fuentes-Datos.md              ← contrato con FlashScore (Playwright + Ninja API) + Kambi API (Betplay)
│
├── 03_Atlas/
│   ├── Pipeline-Arquitectura.md      ← mapa de módulos y dependencias
│   └── Grafo-Dependencias-Datos.md   ← señales S1-S8 y su estado
│
├── 04_Pipeline/
│   └── Sprint-Pipeline.md            ← backlog vivo con estado por tarea
│
├── 05_Deuda/
│   └── Inventario-Deuda-Tecnica.md   ← D-01→D-13 eliminados ✅ | Nodo-10/11 abiertos
│
└── 06_Specs/
    └── Contratos-de-Senal-Maestro.md ← [[Contratos-de-Senal-Maestro]] JSON-Schema S1-S8
```

---

## Dashboard de Estado del Sistema

### Señales (S1-S8)

| Señal | Productor | Estado | Bloqueada por |
|---|---|---|---|
| S1_MATCH_LIST | extraer_partidos_api.py (API) / extraer_URL_partidos_v2 (Playwright) | ✅ 95% | **MODO API:** Kambi+FlashScore ~1.3s 111 partidos | cuota_es_real=True | **Playwright:** --max-matches 80, 8min |
| S2_H2H_DATA | extraer_historh2h.py --api-mode (Ninja) / default (Playwright) | ✅ 95% | **MODO API:** ~0.5s/partido Ninja API | **Playwright:** ~2-3 min/partido | --all-tournaments activo |
| S3_RANKINGS | extraer_ranking_atp/wta_v2 | ✅ 100% | — |
| S4_PREDICTION | rivalry_analyzer.py | ✅ 95% | 72.2% accuracy validado prod (57/79) | shrinkage + density activos |
| S5_EDGE | edge_calculator.py | ✅ 98% | λ por tier + PELT recency (path bug resuelto 2026-06-11) + p_historica por tier+superficie (B-02 resuelto) | — |
| S6_RESULTADO_REAL | resultados_finales.py | ✅ 95% | API Ninja funcional | 80 partidos verificados en ~73s |
| S7_MARKOV | markov_analyzer.py | ✅ ACTIVO | immunity + PELT recency integrados |
| S8_DATASET_ML | generar_dataset_plus.py | ⚠️ 40% | datos limpios de S1 en prod |

### Tests
```
1429 passed, 0 fallos — 2026-06-29 (Nodo-42: Grass Bootstrap | 9 tests T42-01→T42-07 | fix: superficie_filter)
Baseline mínimo: nunca bajar de 1429
```

### Archivos clave del pipeline diario

```
── MODO API (RECOMENDADO — ~45 segundos total) ────────────────────────────────
🚀 Reemplaza Playwright (40+ min) con APIs puras. Cuotas REALES de Betplay.
   Kambi API → jugadores + cuotas reales | FlashScore Ninja API → match_ids + rankings + superficie

PASO 0: python3 extraer_ranking_atp_version2.py
PASO 1: python3 extraer_partidos_api.py                    # partidos de hoy (~1.3s, 111+ partidos)
        python3 extraer_partidos_api.py --tomorrow          # partidos de mañana
        python3 extraer_partidos_api.py --tier atp wta      # solo ATP + WTA
PASO 2: python3 extraer_historh2h.py --api-mode --all-tournaments  # Ninja API (~0.5s/partido)
PASO 3: python3 edge_calculator.py
PASO 4: python3 trader_ev_tenis.py --bankroll 125000       # por tier (ver abajo)
PASO 4.5: python3 betplay_combo_builder.py                 # combos clásicos
          python3 betplay_combo_builder.py --live          # combos solo jugadores disponibles AHORA en Kambi
PASO 5: python3 generar_tabla_favoritos2.py

── MODO PLAYWRIGHT (fallback — ~40 minutos total) ─────────────────────────────
⏰ Usar si APIs están caídas. Ejecutar ~22:00 del día anterior.

PASO 0: python3 extraer_ranking_atp_version2.py
PASO 1: python3 extraer_URL_partidos_version2.py --tomorrow --max-matches 80   # Playwright 8min
PASO 2: python3 extraer_historh2h.py --all-tournaments                         # Playwright ~30min
PASO 3-5: iguales.

── POST-PARTIDO (después de que terminen los partidos) ────────────────────────
PASO 6: python3 resultados_finales.py reports/h2h_results_enhanced_FECHA.json
PASO 7: python3 validar_con_api.py --h2h reports/h2h_results_enhanced_FECHA.json
        (o workaround manual si match_id=None — ver CLAUDE.md PASO 7)

── MODO MULTI-TORNEO ──────────────────────────────────────────────────────────
Pasos 0-2 (API): extraer_partidos_api.py + extraer_historh2h.py --api-mode --all-tournaments
Pasos 0-2 (Playwright): --tomorrow + --max-matches 80 + --all-tournaments
PASO 3 (edge_calculator.py) estratifica per-match automáticamente (tier+superficie).

PASO 4: Correr UNA VEZ por tier que interese — --torneo-tipo FILTRA los picks:
  python3 trader_ev_tenis.py --bankroll 125000                                     # Grand Slam
  python3 trader_ev_tenis.py --bankroll 50000 --torneo-tipo atp1000 --superficie clay  # ATP 1000
  python3 trader_ev_tenis.py --bankroll 30000 --torneo-tipo atp500  --superficie grass # ATP 500
  python3 trader_ev_tenis.py --bankroll 20000 --torneo-tipo challenger --superficie clay # Challenger
  python3 trader_ev_tenis.py --bankroll 10000 --torneo-tipo itf --superficie hard       # ITF

PASO 5-6: iguales.
```

---

## Reglas de Integridad (leer antes de tocar código)

```
REGLA-1: predicción anidada
  ✅ partido['ranking_analysis']['prediction']['favored_player']
  ❌ partido['prediccion_ganador']  → siempre None

REGLA-2: Markov dentro de prediction
  ✅ partido['ranking_analysis']['prediction']['markov_analysis']['factor_markov']
  ❌ partido['ranking_analysis']['markov_analysis']  → no existe en S2 actual

REGLA-3: Erdős en ranking_analysis (post-fix línea 1256 de extraer_historh2h.py)
  ✅ partido['ranking_analysis']['erdos_analysis']['erdos_score']

REGLA-4: FlashScore API dc_1 — claves reales (Nodo-09)
  ✅ DJ='H'→jugador1 ganó | DJ='A'→jugador2 ganó | DJ=''→en curso
  ❌ ~AA, ~BH, ~BI → no existen en este endpoint

REGLA-5: file selection — recency first (Nodo-08)
  ✅ max(files, key=lambda x: (x['modified_time'], x['total_matches']))
  ❌ max(files, key=lambda x: (x['total_matches'], x['modified_time']))

REGLA-6: Kelly-KL cap
  p_historica clay = 0.758 Thompson Beta(24W,7L) — n=31 ✅ umbral cruzado
  Kelly-KL cap = 10% bankroll por apuesta individual

REGLA-7: Roland Garros filter (dev mode)
  'French Open' in torneo_completo AND 'Qualification' not in torneo_completo
  → 41 matches del cuadro principal (no calificación)

REGLA-HF-1: Solo underdogs en pool de combos
  cuota_favorito ≥ 1.50 para entrar al pool de cobertura (--min-cuota 1.50)
  Heavy favorites (cuota <1.50): sí en individuales si edge >5%, NUNCA en combos.
  Motivo empírico: 8 picks con heavy favorites → KGR = -0.5085 (ruina)
                   4 picks solo underdogs → KGR = +0.4142 (crecimiento)

REGLA-HF-2: Diversidad garantizada en selección top-N
  Para cada jugador en el pool, debe existir ≥1 combo en el plan que lo excluya.
  Sin diversidad: un solo fallo destruye todo el portfolio.
  Implementado: algoritmo greedy en _build_cobertura() (trader_ev_tenis.py)

REGLA-HF-3: VaR constraint
  Total en riesgo ≤ 25% bankroll ← MAX_VAR_PCT hardcoded
  Si se excede → stakes ajustados AUTOMÁTICAMENTE en output sección "STAKES FINALES" (T15-05 ✅).

REGLA-HF-4: Portfolio Kelly obligatorio + ρ calibrado por torneo (T15-04 ✅)
  factor = 1/(1 + ρ×(N-1))
  ρ: grand_slam=0.25 | atp1000=0.20 | atp500=0.15 | challenger=0.10 | itf=0.05
  N=4 grand_slam: reducir 42.9% | N=8 grand_slam: reducir 63.6%
  --torneo-tipo FILTRA picks por tier (2026-06-06) — no mezclar tiers en un pool

REGLA-HF-5: Growth Rate negativo = NO DESPLEGAR
  Si Kelly Growth Rate < 0 → el sistema está en régimen de ruina.
  Causas: demasiados picks, cuotas bajas, correlación alta.
  Solución: aumentar --min-cuota, reducir --piernas-max, reducir --top-n.

REGLA-8: FlashScore Ninja API — integración de resultados
  Endpoint: dc_1_{event_id} via config.FLASHSCORE_BASE
  Auth: X-Fsign: SW9D1eZo | Referer: flashscore.co
  Script: resultados_finales.py (PASO 6) — consulta resultado real post-partido
  Velocidad: ~1s/partido (vs 2-5min con Playwright)
  ⚠️ validar_con_api.py = PASO 7 (calibración), NO verificación de resultados

REGLA-9: Multi-torneo pipeline
  PASO 1 (API): python3 extraer_partidos_api.py --tomorrow --tier atp wta challenger
  PASO 1 (Playwright): python3 extraer_URL_partidos_version2.py --tomorrow --max-matches 80
  PASO 2 (API): python3 extraer_historh2h.py --api-mode --all-tournaments
  PASO 2 (Playwright): python3 extraer_historh2h.py --all-tournaments
  PASO 4: Correr trader UNA VEZ POR TIER (--torneo-tipo filtra)
  PASO 6: python3 resultados_finales.py archivo.json (verifica todos los tiers)
  ⚠️ NO mezclar tiers en un solo pool de trader
  ⚠️ API mode: Kambi + FlashScore Ninja (~45s) | Playwright mode: ~40 min (fallback)

REGLA-10: Kambi API (Betplay) — cuotas reales
  Base: https://us.offering-api.kambicdn.com/offering/v2018/betplay
  Endpoint: /listView/tennis.json (todos los eventos, sin auth, solo headers)
  Headers: Referer: https://www.betplay.com.co
  Cuotas: outcomes[].odds / 1000 → cuota decimal real
  Campo: cuota_es_real = True en JSON output (vs FlashScore cuotas promediadas)
  Módulo: scraping/kambi_tennis.py — extract_matches(day_offset, tiers)
  Name matching: 3-tier (exact surname+initial → surnames only → substring ≥5 chars)
  ⚠️ Kambi nombres completos ("Davidovich Fokina A.") vs FlashScore abreviados ("Davidovich A.")

REGLA-11: FlashScore Ninja H2H API — modo API para PASO 2
  Endpoint: df_hh_1_{match_id} via config.FLASHSCORE_BASE
  Auth: X-Fsign: SW9D1eZo | Referer: flashscore.co
  Formato: KC=timestamp, KD=surface, KF=tournament, KJ/KK=players, KL=score, CA/CB=rankings
  Ganador: prefijo * en KJ o KK (e.g., *Sinner J. → Sinner ganó)
  Módulo: scraping/ninja_h2h_parser.py — NinjaH2HExtractor.extract_all()
  Velocidad: ~0.5s/partido (vs 2-3 min con Playwright H2HExtractor)
  Output: MISMO formato JSON que H2HExtractor → downstream pipeline sin cambios (Strangler Fig)
```

---

## Histórico de Hitos (resumen)

```
2026-05-28: Fase 0-5 completadas (scraper fix + edge + markov + dataset + API)
2026-05-29: Fase 7-12 (Erdős + Strangler Fig + limpieza 9,400 líneas)
2026-05-30: Fase 13-16 (trader + validación live + P&L +$25k + T07-09)
2026-06-01: Fase 18 (Portfolio Hedge Fund, 8/8=100% RG R4, KGR=+0.4142)
2026-06-02: Nodo-16 multi-torneo (80 partidos, --all-tournaments)
2026-06-03: Sprint TTC — Nodo-17/18/19/20/21 (980 tests)
2026-06-05/06: Fixes operacionales (FlashScore DOM + markov persist + trader)
2026-06-07: Validación prod completa — 72.2% (57/79), 7 bugs documentados
2026-06-07: MODO API — Kambi + FlashScore Ninja (PASO 1: ~1.3s, PASO 2: ~0.5s/partido)
            Nuevos: kambi_tennis.py + ninja_h2h_parser.py + extraer_partidos_api.py
            Cross-domain bridge NBA→Tennis: name matching 3-tier, Kambi API idéntica
2026-06-09: Fixes ninja_h2h_parser.py (import re + match_id) + validar_con_api.py (superficie)
            betplay_combo_builder.py: --live mode + combos parciales (min 2 piernas) + started_map + find_outcome reasons
            Calibración actualizada: n=284 | validado Jun 7-8: 45/73=61.6%
2026-06-11: FlashScore feed tipo=13 → 1/201 match_url → 82/198 match_id (+6× cobertura mañana)
            combo builder multi-plan merge: build_live_combos lee cobertura de todos los trader_plans (24h)
            Stakes reales por tier: $1k ITF | $3k ATP500 grass | $2k Challenger — de $0 a stakes Kelly reales
            17 combos disponibles de 32 totales | Telegram enviado ✅
2026-06-18: Nodo-28 Conditional Decomposition (Fase 1 + Fase 1.5): common_opponents filtrado por superficie
            SkillFactor (wr/0.5)^1.5 + Surface Alpha + Volume Confidence — fix fórmula surface_specialization
            80.8% accuracy (42/52) — grass 88.9% | clay 85.7% | hard 72.7%
            Hallazgo: 10/10 fallos sin Markov resuelto — señal más potente del modelo (HOT=84.6%)
            Wikilinks vault: 0 huérfanos. Tests: 1050 passed. Calibración: n=1241
```

---

## Próximos Pasos (ordenados por impacto en P&L)

| # | Task | Impacto | Bloqueado por |
|---|---|---|---|
| 1 | **Markov "?" → INSUFFICIENT flag:** 10/10 fallos del 18-jun tienen Markov sin resolver. Regla: `Markov=? AND n_CO≤2 AND tier∈{itf,challenger}` → no predecir. HOT=84.6% vs ?=~65% | 🔴 CRÍTICO | Definir spec (candidato Nodo-29) |
| 2 | **Nodo-27 calibración real:** acumular n≥30 picks con correcto!=None para validar V-27-1→V-27-5 | 🔴 CRÍTICO | `betslip_registrar --cerrar` poblando apuestas_*.json con stake>0 |
| 3 | **Filtrar picks COLD Markov:** COLD=0W/10L=0% — nunca apostar en COLD. NEUTRAL=75% (mejoró vs 6.7% previo con n mayor) | 🔴 CRÍTICO | Confirmar con n≥20 |
| 4 | **B-01: Cap bankroll total** — VaR ajuste debe garantizar TOTAL ≤ 30% bankroll, no solo individual | 🟠 ALTO | — |
| 5 | **T17-06: window_size Markov por tier** — Challengers más corto | 🟡 MEDIO | Datos n≥10/tier |
| 6 | **T15-06: Backtesting formal n≥30 sesiones** | 🟠 ALTO | Más sesiones |
| 7 | **Nodo-10: Unificar ZitaScraper** → `scraping/url_scraper.py` | 🟡 MEDIO | Validación prod |
| 8 | **B-04: generar_tabla_favoritos2.py** — seleccionar archivo grande sobre pequeño | 🟡 MEDIO | — |

---

## Decisiones de Arquitectura (ADR)

| Fecha | Decisión | Alternativa descartada | Razón |
|---|---|---|---|
| 2026-05-28 | Strangler Fig para migración del monolito | Big-bang rewrite | APIs incompatibles en SequentialH2HExtractor vs H2HExtractor |
| 2026-05-29 | Roland Garros filter en pipeline de dev | Procesar todos los torneos | Velocidad: 41 partidos vs 235 |
| 2026-05-29 | modified_time como criterio primario en file selection | total_matches | Datos nuevos tienen menos partidos pero h2h_url válidas (post-Nodo-03) |
| 2026-05-30 | No ML — deploy via trader_ev_tenis.py (Bayesian blend + combos) | entrenar RandomForest primero | NBA demostró: combos + budget cascade > ML para bankroll growth con n pequeño |
| 2026-05-29 | DJ/DE/DF como claves de status en dc_1 API | ~AA/~BH/~BI | Evidencia empírica: 3 partidos reales confirmaron claves reales |
| 2026-05-30 | Buscar activamente odds 3.5–6.0 con señal superficie | apostar en cualquier underdog | EV es convexo: Parry @ 4.50 generó +134% EV vs Berrettini @ 1.45 con +16% (ver [[Nodo-14-Validacion-Live-Conexiones]]) |
| 2026-05-30 | Prior Bayesiano p=0.52 neutral hasta n≥30, luego derivar por superficie | usar p_modelo directo | Con n pequeño, el prior conservador protege contra ruina (Parry n_h2h=0 ganó a pesar del freno) |
| 2026-06-01 | Portfolio Kelly + min-cuota 1.50 en pool de combos | incluir heavy favorites en combos | Con 8 picks (inc. heavy fav @1.04-1.20): KGR=-0.5085 (ruina). Con 4 underdogs ≥1.50: KGR=+0.4142 (crecimiento). Ver [[Nodo-15-Portfolio-HedgeFund]] |
| 2026-06-01 | Sistema Cobertura por Exclusión (C(N,K) combos) en lugar de un parlay único | parlay único N piernas | Un parlay falla si cualquier pierna falla. Con cobertura: si 1 jugador falla, el combo que lo excluye sobrevive. P&L siempre positivo si falla ≤1 pick del pool de 4. |
| 2026-06-01 | ρ calibrado por categoría de torneo (--torneo-tipo) | ρ fijo 0.25 siempre | Picks en Challenger son más independientes que en Grand Slam — ρ fijo sobrepenalizaba portfolio Kelly fuera de Grand Slams |
| 2026-06-01 | p_prior derivado de calibracion_edge.json (--superficie) en lugar de 0.52 fijo | prior fijo 0.52 | n=31 cruzó umbral — el modelo tiene track record real (77% clay). Prior neutro 0.52 subestimaba la confianza validada |
| 2026-06-02 | Roland Garros filter desactivable con --all-tournaments | filter siempre activo | Con multi-torneo (Challenger+ATP+GS) el filter bloqueaba todos los Challengers silenciosamente. Ver [[Nodo-16-Multi-Torneo-Pipeline]] |
| 2026-06-02 | --max-matches N en scraper para controlar volumen de partidos | procesar todos los disponibles | Con >150 partidos disponibles, el filtro Kelly-KL es el guardián real — no el volumen. 80 es el balance óptimo velocidad/cobertura |
| 2026-06-02 | Calibración estratificada [tier][superficie] en lugar de prior global | prior único contaminado | Polmans @5.00 perdió por prior GS aplicado a Challenger en grass — el edge era espejismo de superficie incorrecta. Ver [[Nodo-17-Calibracion-Por-Tier]] |
| 2026-06-02 | λ_KL escalado por tier (0.5→1.8) en lugar de λ fijo | λ=0.5 para todos | Challenger tiene H2H escaso + mercado ineficiente → incertidumbre real es 3.6× mayor que GS. λ fijo subestima el riesgo real |
| 2026-06-07 | Kambi API (Betplay) para cuotas reales en lugar de FlashScore odds | FlashScore cuotas (no expuestas en API) | Kambi = fuente directa donde se apuesta. FlashScore no expone odds en Ninja API (verificado exhaustivamente). Cross-domain bridge NBA→Tennis. Ver Sprint-Pipeline Fase 24 (Nodo-22) |
| 2026-06-07 | FlashScore Ninja H2H API para PASO 2 (--api-mode) | Playwright H2HExtractor (30+ min) | ~0.5s/partido vs 2-3 min. Mismo output JSON → Strangler Fig sin cambios downstream. Playwright preservado como fallback |
| 2026-06-07 | Name matching 3-tier (NBA pattern) para cross-referencia Kambi↔FlashScore | match por nombre exacto | Kambi nombres completos vs FlashScore abreviados. 3-tier: exact surname+initial → surnames only → substring ≥5 chars. 71/111 matched (ATP+WTA 100%) |
| 2026-06-11 | FlashScore feed tipo=13 en vez de tipo=2 | tipo=2 (original) | tipo=13 es acumulativo: incluye todos los niveles de torneos. 23→146 singles mañana (+6×). Hoy: 378→582 matches. tipo=14+ cae. Verificado empíricamente tipos 1-14. |
| 2026-06-11 | combo builder lee cobertura de trader_plans (24h) en vez de recalcular | calcular stakes desde individuales con min() heurístico | El trader ya hizo Kelly/VaR/Cobertura Exclusión por tier. El combo builder es un mapper, no un calculador. Separación correcta de responsabilidades: trader=calcula, combo builder=mapea y filtra STARTED. Fallback legacy preservado cuando no hay planes. |

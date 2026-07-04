# CLAUDE.md — Tennis Prediction & Betting Engine

> Last updated: 2026-07-03 (Nodo-58 COMPLETO — 1649 tests)
> Spec-Driven Development (SDD) — no vibe coding.
> Este archivo es la fuente de verdad. Cualquier sesión futura de Claude debe leerlo completo antes de tocar código.

---

## 1. NORTE REAL — Leer antes de cualquier sesión

**Visión:** Sistema de apuestas deportivas en tenis que gana dinero de forma consistente y medible, operando como un **hedge fund cuantitativo de corta duración** — tratando cada partido como un activo financiero con vida útil de 2-3 horas.

**Misión:** Apostar únicamente donde `P_modelo > P_implícita_bookmaker`, con n suficiente para que no sea ruido estadístico, usando Kelly ajustado por incertidumbre (divergencia Kullback-Leibler) para evitar la ruina.

**Métrica de éxito:** P&L positivo acumulado — NO accuracy del modelo.

**Metas en orden — no saltarse pasos:**

| Meta | Métrica | Estado |
|---|---|---|
| ~~Datos limpios~~ | Fix scraper + surface_specialization funcional | ✅ RESUELTO 2026-05-28 |
| ~~Accuracy > 55%~~ | Con superficie funcionando | ✅ 77.4% (n=31, clay) — supera objetivo |
| ~~P&L positivo n≥30~~ | Edge calculado, Kelly-KL, sesiones validadas | ✅ n=31 cruzado 2026-06-01 |
| **Escalar bankroll** | Solo en señales con edge validado, hedge fund layer activo | **EN CURSO** |

---

## 2. Lo que pasó — No debe repetirse

- **Error 1:** 27 archivos sin SPEC → duplicados, pipeline roto, accuracy 47.37% → SDD obligatorio ✅
- **Error 2:** HTML garbage en tipo_cancha/torneo → surface_specialization=0% durante meses ✅
- **Error 3:** Sin edge calculator → apuestas sin ventaja cuantificada ✅
- **Error 4:** Labels corruptas en generar_dataset_plus.py → accuracy real 47.37% ✅
- **Error 5:** Kelly naive sin portfolio → KGR=-0.51 (ruina silenciosa) → Portfolio Kelly + Cobertura ✅

**Regla permanente:** Antes de proponer cualquier código, responder:
1. ¿Qué archivo lee, qué produce, qué P&L mejora?
2. ¿Hay datos disponibles que el modelo no usa?
3. ¿Qué ignora silenciosamente? (peor que errores visibles)

---

## 3. Fundamentos Científicos

### 3.1 Grafo de Rivalidad Transitiva (Erdős) + PageRank

- Peso: `peso = 1 / (distancia^α)` calibrado por superficie
- Implementado: `analysis/rivalry_analyzer.py` — `common_opponents_detailed` (clay: 28%)
- Clay calibrado: common_opponents 0.20→0.28 | ranking_momentum 0.20→0.12 (T14-03 ✅)
- **PageRank (Nodo-20 ✅):** `pagerank_grafo()` en `erdos_graph.py` — centralidad de nodos intermedios pondera ventaja transitiva. Caminos directos no afectados (REGLA-T20-3).

### 3.2 Kelly Ajustado por Incertidumbre (Kullback-Leibler)

```
f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica))
```
- Edge mínimo para apostar: P_modelo - P_implícita > 5%
- Implementado en `edge_calculator.py` ✅
- **λ por tier (Nodo-21 ✅):** GS=1.0× | ATP1000=1.6× | ATP500=2.4× | Challenger=3.6× | ITF=4.5×
- **λ PELT Recency Alpha (Nodo-18 ✅):** HOT fresco (≤3 partidos) → λ÷1.20 | COLD fresco → λ÷0.85

### 3.3 Portfolio Kelly Multi-Activo (Nodo-15)

```
factor = 1 / (1 + ρ × (N - 1))    ρ=0.25 (correlación Grand Slam misma sesión)
N=4: factor=0.571 | N=8: factor=0.364
```
- Implementado en `trader_ev_tenis.py` v2.0 como `_portfolio_kelly_factor()` ✅

### 3.4 Sistema Cobertura por Exclusión (Nodo-15)

Genera TODOS los combos C(N,K) para K en [piernas_min, piernas_max].
Cada combo excluye implícitamente (N-K) jugadores — si fallan, el combo paga.

```
Pool 4 picks → C(4,3)=4 + C(4,4)=1 = 5 combos
P&L siempre positivo si falla ≤1 pick del pool
```
- Diversidad garantizada: algoritmo greedy — cada jugador excluido en ≥1 combo top-N
- Implementado en `trader_ev_tenis.py` v2.0 como `_build_cobertura()` ✅

### 3.5 VaR/CVaR + Kelly Growth Rate (Nodo-15)

```
MAX_VAR_PCT = 0.25    # máximo 25% bankroll en VaR
g = E[log(1+R)]       # g>0 → crece | g<0 → ruina (NO DESPLEGAR)
```
- Implementado en `trader_ev_tenis.py` v2.0 como `_portfolio_risk_report()` ✅

### 3.6 Cadenas de Markov + PELT Change-Point

- Detecta cambios de régimen (buena racha → mala) con fecha exacta
- Implementado en `analysis/markov_analyzer.py` ✅ | integrado en `rivalry_analyzer.py`
- `factor_tardio` añadido 2026-05-30
- **`n_partidos` en output** (Nodo-18 ✅) — permite calcular recencia del cambio de régimen

### 3.7 Pesos Diferenciados por Tier + Densidad Local (Nodo-21)

```
# Pesos SNR: Grand Slam → H2H/common_opp altos | ITF → form_recent dominante
# James-Stein shrinkage: factor = n_tier / (n_tier + 20)
# density_confidence(n_common, n_paths) → [0.3, 1.0] modula common_opponents
```
- `detectar_tier()` en `config.py` — fuente única de verdad para clasificación (T21-02 ✅)
- 5 tiers: `grand_slam | atp1000 | atp500 | challenger | itf` en todos los módulos
- Shrinkage lee `calibracion_edge.json` automáticamente: clay_grand_slam n=31 → factor=0.61
- K-factor ELO por tier: GS=24, ATP1000=28, ATP500=32, Challenger=40, ITF=48 (Nodo-21 Fase 3 ✅)

### 3.8 H2H Immunity Dampener (Nodo-19)

```
# HOT × h2h_win_rate < 0.30 → immunity_factor = 0.85  (rival "inmune")
# HOT × h2h_win_rate > 0.70 → immunity_factor = 1.12  (doble confirmación)
# n_h2h < 3 → immunity_factor = 1.00                  (muestra insuficiente)
```
- `calcular_h2h_immunity()` en `rivalry_analyzer.py` — cruza estado Markov con H2H específico
- Previene sobreconfianza en favorito HOT contra rival que históricamente le gana

---

## 4. Flujo del Pipeline

### ANTES DEL PARTIDO

**PASO 0** — Actualizar rankings (opcional — ejecutar si los rankings están desactualizados)
```bash
python3 extraer_ranking_atp_version2.py   # rankings ATP → data/atp_rankings_complete_FECHA.json
python3 extraer_ranking_wta_version2.py   # rankings WTA → data/wta_rankings_complete_FECHA.json
```
→ Los rankings son leídos automáticamente por `extraer_historh2h.py` vía `analysis/ranking_manager.py`

**PASO 1** — Extraer partidos del día

```bash
# ── MODO API (RECOMENDADO — ~1.3 segundos, cuotas REALES Betplay) ──────────
python3 extraer_partidos_api.py                              # partidos de hoy
python3 extraer_partidos_api.py --tomorrow                   # partidos de mañana
python3 extraer_partidos_api.py --tier atp wta               # solo ATP + WTA
python3 extraer_partidos_api.py --tier atp wta --torneo wimbledon     # solo Wimbledon (Nodo-50)
python3 extraer_partidos_api.py --torneo wimbledon "us open"          # múltiples torneos (OR)

# ── MODO PLAYWRIGHT (fallback — ~8 min, DOM frágil) ────────────────────────
python3 extraer_URL_partidos_version2.py
```
→ `data/zita_tennis_matches_FECHA.json`

**PASO 2** — Extraer H2H

```bash
# ── MODO API (RECOMENDADO — ~0.5s/partido, FlashScore Ninja H2H) ──────────
python3 extraer_historh2h.py --api-mode --all-tournaments

# ── MODO PLAYWRIGHT (fallback — ~30 min para 80 partidos) ──────────────────
python3 extraer_historh2h.py --all-tournaments
```
→ `reports/h2h_results_enhanced_FECHA.json`

**PASO 3** — Calcular edge (Kelly-KL 5 capas)
```bash
python3 edge_calculator.py
```
→ `reports/edge_report_FECHA.json`

**PASO 3.5** — Tabla de análisis — Revisión humana antes de apostar
```bash
python3 generar_tabla_favoritos2.py
```
→ `analisis_partidos_pandas.txt`

Leer ANTES de correr el trader. Señales clave a revisar por pick APOSTAR:
- `contribution%` por componente — si `form_recent=0%` el modelo no tiene datos de forma
- `surface_specialization raw_score` — calidad real en la superficie del torneo
- `Confianza` baja (<52%) + cuota alta = señal débil, considerar bajar stake manualmente

**PASO 4** — Plan de deploy — Hedge Fund Layer

`--torneo-tipo` **FILTRA** los picks del edge report por tier. El trader procesa UN tier por ejecución.
`--superficie` define el p_prior fallback. Cada pick ya trae su propio `p_historica_usada` del edge_calculator.

```bash
# ── MODO MULTI-TORNEO (correr una vez por tier que interese) ──────
python3 trader_ev_tenis.py --bankroll 125000                                           # Grand Slam clay (default)
python3 trader_ev_tenis.py --bankroll 125000 --superficie grass                         # Grand Slam grass
python3 trader_ev_tenis.py --bankroll 50000  --torneo-tipo atp1000 --superficie clay    # ATP 1000
python3 trader_ev_tenis.py --bankroll 30000  --torneo-tipo atp500  --superficie grass   # ATP 500
python3 trader_ev_tenis.py --bankroll 20000  --torneo-tipo challenger --superficie clay  # Challenger
python3 trader_ev_tenis.py --bankroll 10000  --torneo-tipo itf --superficie hard         # ITF
```
→ `reports/trader_plan_FECHA.json` + `reports/trader_plan_FECHA.txt`
- `--torneo-tipo`: **FILTRA por tier** + aplica ρ (grand_slam=0.25 | atp1000=0.20 | atp500=0.15 | challenger=0.10 | itf=0.05)
- `--superficie`: p_prior fallback de calibracion_edge.json — cada pick usa su propio `p_historica_usada` per-match
- Defaults: cobertura, all-picks, watchlist, min-cuota 1.50, piernas 3-4, top-4, grand_slam, clay
- Stakes VaR-ajustados automáticamente — sección "STAKES FINALES" lista para usar
- Si KGR < 0 en output → **NO DESPLEGAR** (REGLA-HF-5)
- Si VaR > 25% → stakes ya ajustados automáticamente en output

**PASO 4.3** — Combos de Confianza con Aislamiento de Riesgo (Nodo-38 — paralelo al pipeline)

```bash
python3 combo_confianza_builder.py --bankroll 125000                    # Fase 4 default (todo)
python3 combo_confianza_builder.py --bankroll 125000 --fase 1           # solo CORE (validación)
python3 combo_confianza_builder.py --bankroll 125000 --fase 2           # CORE + 1 satellite
python3 combo_confianza_builder.py --bankroll 125000 --fase 3           # CORE + 3 SAT + moonshot
python3 combo_confianza_builder.py --bankroll 125000 --fase 4 --telegram # todo + Telegram
```
→ `reports/combo_plan_FECHA.txt`
- Capa **paralela** al pipeline Kelly-KL — opera cuando edge_calculator dice "0 apuestas"
- Arquitectura CORE/Satellite/Moonshot con aislamiento de riesgo:
  - **CORE:** Cat-A (cuota 1.15-1.59) + Cat-B (1.60-2.20), C4-C7. NUNCA Cat-C.
  - **SATELLITE:** 4×Cat-A/B + 1×Cat-C1 (cuota 2.21-3.50, conf≥60%). Aislado — si Cat-C falla, solo satellite muere.
  - **MOONSHOT:** 3×Cat-A + 2-3×Cat-C (conf≥57%). Baja prob, alto payout.
- VaR guard: budget diario = bankroll × fase_pct (2%→4%→7%→12%)
- Cross-reference con edge_report: picks pipeline promueven Cat-C2→Cat-C1 (Protocolo D)
- Parejo guard: conf<55% + cuota 1.55-1.70 = excluir
- Tournament guard: max 2 picks del mismo torneo por combo

**PASO 3.6** — Games/Sets Signal (Nodo-40 — corre después de edge_calculator)

```bash
python3 games_signal_calculator.py
```
→ `reports/games_signal_report_FECHA.json` — señales UNDER/OVER juegos/sets por partido
- DOMINANTE (|diff|>0.35): 2 sets → UNDER games | COINFLIP (|diff|<=0.18): 3 sets → OVER
- Solo señales con gap ≥ 2 juegos y cuota ≥ 1.50
- REGLA-G6: stake máx $2,000 hasta n≥50 observaciones calibradas (actualmente n=3)

**PASO 4.4** — Combos de Totales (Nodo-40 — opcional, requiere PASO 3.6)

```bash
python3 betplay_combo_builder.py --games                          # combos totales solo
python3 betplay_combo_builder.py --games --games-stake 2000       # stake explícito
python3 betplay_combo_builder.py --live --games                   # ganadores + totales
```
→ GamesA/B/C.bat en escritorio — alpha ortogonal al ganador

**PASO 4.5** — Combos Betplay (opcional — construir links de apuesta)

```bash
python3 betplay_combo_builder.py --live --strategy balanced --telegram
# Estrategias: balanced (default) | aggressive | conservative
```

**PASO 4.55** — Mega-Combos Cross-Tier (Nodo-23 — opcional)

```bash
python3 betplay_combo_builder.py --mega                                    # mega-combos solo
python3 betplay_combo_builder.py --live --mega --telegram                  # combos + megas
python3 betplay_combo_builder.py --mega --mega-stake 1000 --mega-min 7 --mega-max 9
```
→ Mega-combos: 6-10 piernas cruzando tiers. Requiere ≥2 tiers distintos. Stake fijo ($500 default).

**PASO 4.57** — Safe Combos Beta Book (Nodo-25 — opcional)

```bash
python3 betplay_combo_builder.py --safe                                    # safe combos solo
python3 betplay_combo_builder.py --live --mega --safe --telegram            # portafolio completo
```
→ Safe combos: 2 piernas, P(ambos)>25%, cuota 3-12, cross-tournament. Stake fijo ($1,000 default).
→ Two-Speed Portfolio: Alpha Book (megas) + Normal Book (combos) + Beta Book (safe)

**PASO 4.6** — Registrar apuesta + cerrar loop (betslip_registrar.py)

```bash
python3 betslip_registrar.py --listen    # levantar servidor ANTES de apostar (puerto 5001)
python3 betslip_registrar.py --estado    # ver pendientes
python3 betslip_registrar.py --cerrar    # post-partido → calibracion_edge.json auto
```
Flujo: `--live` genera betslip_index → `--listen` → bookmarklet POST → `--cerrar` cierra.

---

### DESPUÉS DEL PARTIDO

**PASO 5.5 — Shadow Book Momento 2** (correr ~15 min ANTES del inicio de cada partido)
```bash
python3 shadow_book.py --close-snapshot
```
→ Captura cuota de cierre Kambi para calcular CLV real (cuota_entrada vs cuota_cierre)
→ Sin este paso, CLV se calcula solo con cuota de entrada (menos preciso)

**PASO 6** — Registrar resultados
```bash
python3 resultados_finales.py
```
→ `reports/resultados_finales_FECHA.json`

**PASO 7** — Calibrar modelo
```bash
python3 validar_con_api.py
```
→ actualiza `data/calibracion_edge.json`

**PASO 8** — Consultar histórico
```bash
python3 consultar_resultados_historicos.py
```
→ lee `reports/resultados_finales_*.json` más reciente

**PASO 9** — Observabilidad (Nodo-27)
```bash
python3 pipeline_tracker.py                          # todo el histórico
python3 pipeline_tracker.py --since 2026-06-17       # solo hoy
python3 pipeline_tracker.py --tier challenger         # filtrar por tier
python3 pipeline_tracker.py --section confianza       # confianza|cuotas|tiers|senales|calibracion|drift|portfolio
python3 pipeline_tracker.py --save                    # guardar snapshot JSON
```
→ `pipeline_tracking.txt` (sobreescribe) | READ-ONLY — no modifica ningún dato

**PASO 10 — Shadow Book Momento 3** (después de conocer resultado del partido)
```bash
python3 shadow_book.py --settle YYYY-MM-DD    # AUTOMÁTICO — lee resultados_finales.json o scraping FlashScore
python3 shadow_book.py --report               # S-27-8: hit%, CLV median, IC Wilson por segmento
python3 pipeline_tracker.py --section shadow  # integrado en tracker general
```
→ `reports/shadow_book/sb_YYYY-MM-DD.jsonl` (append-only, inmutable)
→ **NO requiere input manual** — join automático por match_id, fallback fuzzy name.
→ Si ya corriste `resultados_finales.py` (PASO 6), lo usa. Si no, va directo a FlashScore.
→ El `--report` es el documento de validación más importante del proyecto: confirma si el edge es real o ruido.
→ Correr `--settle` ANTES de `--report`. Sin settled>0 el reporte no tiene métricas.

---

### PIPELINE ML — SUSPENDIDO (2026-06-10)
```bash
python3 generar_dataset_plus.py   # ML PASO A → 386 registros, 26 features
python3 aplicar_enhancer.py       # ML PASO B → ml_datasets/ (joblib, csv, npz)
```
Estado: RF=95.4%CV (overfit), LR=71.7% (honesto) — NO supera heurístico 75.8%.
Conectar a rivalry_analyzer.py SOLO cuando modelo supere 78% en held-out test.
Target: n≥1000 por superficie+tier. Acumular datos con betslip_registrar.py.

---

## 5. Estado Real — 2026-07-03

```
Tests:         1649 passed, 0 failed (1563→1588→1598→1601→1612→1649 tras Nodo-53/55/56/57/58)
Calibración:   clay GS: p=0.758 (n=31) | global: wins=467, losses=239, n=706 + 14 nuevos (Jun-28)
               calibration_epoch: epoch-1=pre-Nodo-47 (n=706, ranking parcial), epoch-2=post 2026-06-30
Bankroll:      $125,000+
Hedge Fund:    Portfolio Kelly + VaR auto-ajustado + Cobertura Exclusión — ACTIVO ✅
ML Dataset:    2,573 registros limpios (motor nodo32-fase3-markov-postnorm) — Nodo-41 ✅
               Trazabilidad: jugador1/jugador2/_trace_fecha/torneo_nombre verificada manualmente

Nodo-51 ✅ COMPLETO (2026-07-02):
  F0: PlayerRegistry — entity resolution canónica, absorbe clase entera de bugs tipo Nodo-47
  F1: TournamentContext — superficie + season_transition_flag viajan con cada partido
  F2: DataContract — picks con historial EMPTY → status=NO_DATA, excluidos de TODOS los pools
      El hueco por donde entraron combos fantasma (2026-07-01) está cerrado por construcción.
  F3: Playwright batch con presupuesto + memoria THF (Nodo-49 completado)
  F4: Surface Context Discount — _normalize_surface + surface_overlap_rate en markov_analyzer
      Constantes BLOQUEADAS (n=1, Watanuki) — flag --no-surface-discount para A/B
  F5: validation/preregistered_hypotheses.json — H52-01→H52-08 congeladas 2026-07-02
      validation/hypothesis_tracker.py — nodo46_unlocked() False hasta n≥5
  F-Meta: PRE_IMPLEMENTATION_CHECKLIST.md — GIT-FIRST como proceso, no como regla

Nodo-52 ✅ COMPLETO (2026-07-02):
  shadow_book.py — Libro Sombra: log_picks (Momento 1) + settle (Momento 3) + CLV + report (S-27-8)
  Hook --shadow-log en edge_calculator (default ON, try/except — PASO 3 nunca crashea)
  pipeline_tracker.py --section shadow — S-27-8 integrado (D52-05)
  Hipótesis H52-01→H52-08 pre-registradas y congeladas en validation/preregistered_hypotheses.json
  USO: python3 shadow_book.py --settle YYYY-MM-DD   # después del partido
       python3 shadow_book.py --report               # métricas por segmento
       python3 pipeline_tracker.py --section shadow  # integrado en tracker

Lecciones validadas:
  - Challenger/ITF: bookmaker tiene menos datos → mayor ventaja informacional (sesión 9/10 del 13-jun)
  - ATP500+: bookmaker conoce bien Top 100 → edge real menor que el calculado
  - Underdogs cuota ≥2.00 con edge >5% = alpha estructural (sesión 8/8 R4 01-jun)
  - Cobertura Exclusión demostrada: pick falla pero combo sin él sobrevive
  - Cross-tier mega-combos: ρ≈0.03 entre tiers → casi independientes
  - ML dataset DEBE tener trazabilidad; 69% contaminación por motor viejo = no entrenable (Nodo-41)
  - THF (Nodo-45): recupera historiales de sesiones anteriores cuando match_id=None o API vacía — D45-01 a D45-05 ✅
  - Nodo-48 ✅: `--flashscore-only` — 507 partidos + 233 con cuotas sin Kambi. Guard D48-05 activo.
  - edges fantasma 2026-07-01 (Arce/Vlajic/Guajardo/Cooper): causa raíz = historial vacío sin gate.
    FIX: F2 DataContract — status=NO_DATA por construcción, trader no los ve nunca.

Nodo-55 ✅ COMPLETO (2026-07-03):
  Respuesta Fable al brief Nodo-54: el embudo no está roto — es opaco y caro de operar.
  P54-01 CERRADO: λ_ITF=4.5 se mantiene. A54-01a confirma n=0 datos ITF post-Nodo-47 fix.
    ITF epoch_2 (post-2026-07-01): n=0. Hit 20% en shadow book con 2 stakes reales perdiendo = λ funciona.
  P54-02 IMPLEMENTADO: Stake Waterfall Log en trader_ev_tenis.py
    LOG_STAKE_WATERFALL: kelly_kl → q_kelly → ×portfolio_factor → ×var_factor → MIN_BET_CLIFF → $0
    var_flattened + stake_real escritos al shadow book via update_trader_stakes()
    H54-01 pre-registrado: APOSTAR stake=0 vs financiados — n=30 para decidir floor.
  P54-03 CERRADO: ruta WAS ya existe (Nodo-44). Sin gate nuevo. Liu/Zheng/Safiullin p≈0.51 = coin-flip.
    D54-02: sub-segmento WATCHLIST+grand_slam+edge>=20% visible en --report S-27-8.
  D54-03 IMPLEMENTADO: run_daily.py — pipeline completo en un comando, daily_brief_FECHA.txt
    PASO 0→4.3 + settle ayer + WAS check. Tiempo humano: 45 min → ~7 min/día.
  Tests: T55-01→T55-05 (5 tests, todos pasan)
  PROHIBIDO (doc):  λ_tier recalibración, GS_WATCHLIST_HIGH_EDGE gate, MIN_STAKE_APOSTAR incondicional.

Nodo-56 COMPLETO (2026-07-03):
  Bug A: get_weights_from_reasoning ignoraba LOG_SHRINKAGE → pesos 99%-103% en display.
    Fix D56-01: _weights_final retornado por rivalry_analyzer.py (fuente única de verdad).
    Fix D56-02: generar_tabla_favoritos2.py usa _weights_final en lugar de reconstruir desde logs.
    Fix D56-04: round(...,2) → round(...,4) en ajuste de superficie (clay/grass).
  Bug B: PUNTAJE FINAL TOTAL ≠ suma de componentes (caso Meligeni vs Pacheco: 3.77 vs 1.89).
    Causa: penalización de inactividad (days_since>30) se aplicaba pero NO se mostraba.
    Fix D56-05: generar_resumen_consolidado muestra fila Penalizacion_Inactividad cuando != 0.
    → con fila visible: sum(componentes) + penalización = PUNTAJE FINAL TOTAL.
  Scoring y predicciones: NUNCA estuvieron rotos. Solo el display era engañoso.
  Tests: T56-04→T56-06 (3 tests, todos pasan)

Nodo-57 COMPLETO (2026-07-03):
  Bug A: Penalización de inactividad global demasiado agresiva (50% a los 30d+).
    Fix D57-01: apply_weights_and_penalties simplificado — penalty siempre 0.0.
    Fix D57-02: form_decay_factor exponencial aplicado SOLO a norm_p['form_recent'].
      decay = max(0.35, exp(-0.025 × max(0, days-30))) | grace=30d | floor=0.35
      days=-1 → decay=0.70 (moderado fijo) | days≤30 → 1.0 (sin decay)
  Bug B: Gate de campeon de torneo no validaba tier — wins>=4 para cualquier torneo.
    Fix D57-03: _MIN_WINS_CHAMPION = {grand_slam:7, atp1000:6, atp500:5, challenger:5, itf:4}
    Caso real: Safiullin (3 qualifying + 2 main draw = 5 wins) recibía bonus GS champion.
    GS requiere 7 victorias consecutivas en draw principal.
  Bug C: TORNEO_COMPLETO_EXPIRADO no mostraba a quién pertenecía el campeonato.
    Fix D57-04: compensation bonus 90-180d → x1.15 | 180-365d → x1.05 | >365d sin bonus.
    Fix D57-05: LOG_FORM_DECAY añadido a SEÑALES ESPECIALES con atribución por jugador.
  Tests: T57-01→T57-09 + T30-10b + T30-10c (11 tests nuevos, todos pasan)
  PROHIBIDO: modificar shrinkage, kelly, ELO — solo form_recent decay y champion gate.

Nodo-58 ✅ COMPLETO (2026-07-03):
  Dashboard de Observabilidad — 6 paneles READ-ONLY con tema McLaren Electric Dark.
  D58-01: report_dict() en shadow_book.py + --json flags (shadow_book, pipeline_tracker)
    Expone métricas como dicts para el dashboard (nunca recalcula — fuente única shadow_book)
    T58-02: paridad garantizada — hit%/IC/CLV del dashboard == report_dict() para el mismo rango
  D58-02: Panel 1 (HOY) — cascada de stakes interactiva, WAS candidatos, alertas KGR/VaR/NO_DATA
  D58-03: Panel 2 (HIPÓTESIS) + Panel 6 (DECISIÓN) — semáforos vivos H52-01→08/H54-01
    Panel 6: 6 acciones tentadoras materializadas como criterios pre-registrados AUTORIZADO/NO_AUTORIZADO
    La disciplina deja de depender de la memoria
  D58-04: Panel 3 (SALUD) — distribución history_provenance por día, ranking_provenance alerta Nodo-47,
           Panel 5 (RIESGO) — KGR/VaR/Sharpe/drawdown, apuestas REALES vs SIMULADAS siempre separadas
  D58-05: Panel 4 (ATRIBUCIÓN) — componentes con _weights_final (D56-01), Penalizacion_Inactividad visible (D56-05)
  D58-06: Coloreo por componente post-settlement (✓ apuntó al ganador | ✗ no | ~ empate)
          Tabla acumulada acierto-por-señal × tier (con n≥100: insumo para Nodo-21 recalibración)
  Theme: McLaren Electric Dark — Naranja #FF8000 / Azul #00BFFF / Verde #00FF87 / Rojo #FF1E56
         Badges, cards, cascadas, progreso bars con bordes definidos
  CLI: streamlit run dashboard.py [--server.port PORT]
  Tests: T58-01→T58-05 (16 tests nuevos, todos pasan)
  PROHIBIDO: botones que ejecuten pipeline/apuestas, recalcular métricas, mezclar betslips+shadow en tabla
```

---

## 6. Mapa de Archivos

### NÚCLEO ACTIVO
```
── ANTES DEL PARTIDO ──────────────────────────────────────────────────────────
extraer_partidos_api.py           ← PASO 1 API (RECOMENDADO — Kambi + FlashScore, ~1.3s)
extraer_URL_partidos_version2.py  ← PASO 1 Playwright (fallback — ~8 min)
extraer_historh2h.py              ← PASO 2 (--api-mode: Ninja ~0.5s/p | default: Playwright ~2-3min/p)
edge_calculator.py                ← PASO 3: Kelly-KL 5 capas
trader_ev_tenis.py (v2.0)        ← PASO 4: Hedge Fund Layer
  Parámetros (todos con defaults — solo --bankroll es obligatorio):
    --bankroll N         bankroll en USD (sin default)
    --superficie         clay|grass|hard|unknown → p_prior desde calibracion [default: clay]
    --torneo-tipo        grand_slam|atp1000|atp500|challenger|itf → FILTRA por tier + ρ [default: grand_slam]
    --cobertura          Sistema Cobertura Exclusión C(N,K)       [default: ON]
    --all-picks          incluir sin_edge en pool                  [default: ON]
    --watchlist          incluir watchlist en pool                 [default: ON]
    --min-cuota 1.50     solo underdogs (REGLA-HF-1)              [default: 1.50]
    --piernas-min 3      piernas mínimas                          [default: 3]
    --piernas-max 4      piernas máximas                          [default: 4]
    --top-n 4            top N combos por tier                    [default: 4]
    --excluir "A,B"      excluir jugadores específicos
  Produce: reports/trader_plan_FECHA.json + .txt (stakes VaR-ajustados automáticamente)
generar_tabla_favoritos2.py       ← PASO 3.5: revisión humana ANTES de apostar (contribution%, overs, confianza)
combo_confianza_builder.py        ← PASO 4.3 (Nodo-38): CORE/Satellite/Moonshot con aislamiento de riesgo
  Parámetros:
    --bankroll N         bankroll total (default: 125000)
    --fase 1|2|3|4       fase de escalado (default: 4)
    --threshold 53       confianza mínima (default: 53)
    --telegram           enviar combos a Telegram
    --no-bat             no generar .bat en escritorio
    --file PATH          h2h_results_enhanced específico
  Categorías: Cat-A (1.15-1.59) | Cat-B (1.60-2.20) | Cat-C1 (2.21-3.50,conf≥60%) | Cat-C2 (>3.50)
  Produce: reports/combo_plan_FECHA.txt
games_signal_calculator.py        ← PASO 3.6 (Nodo-40): señales totales juegos/sets → reports/games_signal_report_FECHA.json
betplay_combo_builder.py          ← PASO 4.4 (--games) + 4.5 + 4.55 + 4.57: totales + combos (--live) + megas (--mega) + safe (--safe)
betslip_registrar.py              ← PASO 4.6: registra apuesta + cierra loop calibración
run_daily.py                      ← D54-03 (Nodo-55): orquestador PASO 0→4.3 + settle + daily_brief
                                    python3 run_daily.py [--bankroll N] [--tomorrow] [--settle-only]
                                   → reports/daily_brief_FECHA.txt (5 min lectura humana/día)

── DESPUÉS DEL PARTIDO ─────────────────────────────────────────────────────────
resultados_finales.py             ← PASO 6
validar_con_api.py                ← PASO 7
consultar_resultados_historicos.py← PASO 8
pipeline_tracker.py               ← PASO 9: observabilidad READ-ONLY (Nodo-27 + S-27-8 shadow)
shadow_book.py                    ← PASO 10: Libro Sombra CLV (Nodo-52)
  --settle FECHA    → settlement post-match, calcula CLV
  --report          → métricas S-27-8 por segmento (hit%, IC Wilson, CLV median)
  --close-snapshot  → Momento 2: captura cuota cierre Kambi ~15min antes del inicio

── DATOS ACTIVOS ────────────────────────────────────────────────────────────────
data/calibracion_edge.json        ← Thompson Beta: fuente de verdad para priors por tier+superficie
reports/shadow_book/sb_YYYY-MM-DD.jsonl  ← Shadow Book (append-only, inmutable en predicción)
validation/preregistered_hypotheses.json ← H52-01→H52-08 congeladas 2026-07-02 (NO modificar)
validation/hypothesis_tracker.py  ← nodo46_unlocked(), was_thresholds(), get_calibration_epochs()

── MÓDULOS COMPARTIDOS ──────────────────────────────────────────────────────────
config.py                         ← constantes API + browser + detectar_tier() (fuente única de tiers)
normalization.py                  ← MAX_RAW_SCORES + DEFAULT_WEIGHTS
core/player_registry.py           ← PlayerRegistry: entity resolution canónica (Nodo-51 F0)
core/tournament_context.py        ← TournamentContext: superficie+tier+season_flag (Nodo-51 F1)
core/data_contract.py             ← PICK_STATUS_NO_DATA + has_empty_history() (Nodo-51 F2)
scraping/kambi_tennis.py          ← Kambi API Betplay + FlashScore feed + name matching
scraping/ninja_h2h_parser.py      ← NinjaH2HExtractor — FlashScore Ninja H2H API + F3 Playwright batch
scraping/h2h_extractor.py         ← orquestador Playwright (fallback)
scraping/browser_manager.py       ← Playwright WSL-optimized (solo fallback)
scraping/data_parser.py           ← parser HTML corregido
scraping/file_utils.py            ← selección recency-first
analysis/rivalry_analyzer.py      ← motor de predicción + Erdős + Markov + Nodo-19/21 + surface discount
analysis/markov_analyzer.py       ← PELT + factor_tardio + recencia_regimen + surface_context_discount (F4)
analysis/erdos_graph.py           ← grafo transitivo + PageRank (Nodo-20)
analysis/elo_system.py            ← ELO + K-factor por tier + reset post-PELT
analysis/ranking_manager.py
utils/logger.py                   ← SmartLogger (fuente única)

── ML (NODO-41 ✅ LIMPIEZA DE DATASET COMPLETADA 2026-06-29) ──────────────────────
generar_dataset_plus.py           ← ML PASO A: 2,573 registros limpios (motor válido nodo32-fase3)
                                   Filtro rivalry_version + trazabilidad jugador1/jugador2/_trace_fecha
aplicar_enhancer.py               ← ML PASO B: entrena modelos → ml_datasets/ (LISTO PARA EJECUTAR)

── SUSPENDIDO (isla Flask/Selenium) ─────────────────────────────────────────────
app.py | routes/ | models/ | services/ | database.db
predecir_partidos.py  (importa informe_detallado.py que no existe)
```

---

## 7. Bugs Activos

| Bug | Archivo | Severidad | Estado |
|---|---|---|---|
| prediccion_ganador top-level = None | `extraer_historh2h.py` | 🟠 | Activo — usar `ranking_analysis.prediction.favored_player` |

> Bugs resueltos archivados en git history. Ver commits de sesiones 3-6 (2026-06-05 a 2026-06-16).

---

## 8. Protocolo de Trabajo

```bash
# PRIMERO — buscar en git si ya existe una implementacion previa (OBLIGATORIO)
git log --all --oneline -- '*keyword*'               # buscar archivos eliminados
git show COMMIT:backend/archivo.py                   # recuperar si existe

# Antes de cualquier edición
grep -n "texto_exacto" archivo.py

# Antes de cualquier PR — baseline obligatorio
python -m pytest tests/ --no-cov -q  # debe dar 1563 passed

# Verificar syntax
python -c "import ast; ast.parse(open('archivo.py').read()); print('OK')"

# Antes de eliminar cualquier archivo
grep -rn "nombre_archivo" --include="*.py"
```

### REGLA GIT-FIRST: Antes de implementar cualquier feature, buscar en git history si el usuario ya lo resolvio antes.
El usuario construyo soluciones que funcionan ANTES de usar IA. Ignorarlas y reinventar desde cero es el error mas costoso.
Caso real: extraer_cuotas_partidos.py (git: 23d2d91) resuelve Nodo-48. La IA lo declaro BLOQUEADO por usar URL incorrecta.

### Sobre URLs de scraping vs APIs:
- `global.flashscore.ninja/202/x/feed` = Ninja API (datos JSON, sin cuotas)
- `www.flashscore.com/tennis/` = sitio web real (DOM con cuotas, usar Playwright)
- NUNCA derivar URLs de browser desde URLs de API — son sistemas completamente distintos.

### Spec-Driven: ninguna línea de código nueva sin Nodo en `.spec/01_Nodos/`
### Para tareas spec-críticas (eliminaciones, migraciones): ver `.spec/TTC-Protocol.md`

---

## 9. Recordatorios Críticos

### Guards de No-Ruina (Quant)
- **REGLA-HF-1:** Heavy favorites (cuota <1.50) NUNCA en pool de combos. KGR con heavy fav = -0.5085 (ruina).
- **REGLA-HF-5:** Si KGR < 0 en output trader → NO DESPLEGAR. Subir --min-cuota o reducir --piernas-max.
- **VaR se ajusta automáticamente** — sección "STAKES FINALES" en output del trader. NO calcular a mano.
- **`--torneo-tipo` FILTRA por tier.** Correr el trader UNA vez por tier que interese. NO mezclar GS con ITF.
- **ρ varía por torneo** — grand_slam=0.25 | atp1000=0.20 | atp500=0.15 | challenger=0.10 | itf=0.05

### Calibración y Priors
- **p_prior calibrado automáticamente** por `tier+superficie` desde `calibracion_edge.json`. Jerarquía: `por_superficie_y_tier` n≥10 → `fallback_por_tier` clamped con `min(tier, superficie)` si divergen >0.03 → `por_superficie` → global.
- **confidence_flag:** STRONG (p>=0.60) / MODERATE (0.55<=p<0.60) / LOW (p<0.55). Picks LOW = edge por cuota extrema, no convicción.
- **calibration_confidence:** Kelly escalado por `n/(n+20)` (floor=0.30). n=4 → Kelly al 30%. n=33 → Kelly al 62%.
- **p_blend per-match** — cada pick usa su propio `p_historica_usada` del edge_calculator. `--superficie` es fallback.

### Estructura de Datos
- **La predicción está anidada:** `partido['ranking_analysis']['prediction']['favored_player']` — NO `partido['prediccion_ganador']` (siempre None).
- **markov_analysis está anidado:** `partido['ranking_analysis']['prediction']['markov_analysis']` — NO `partido['markov_analysis']` (siempre None).

### APIs y Scraping
- **MODO API es ruta primaria.** PASO 1: `extraer_partidos_api.py`. PASO 2: `--api-mode`. Playwright es fallback.
- **Kambi API = cuotas reales Betplay** donde se apuesta. Campo `cuota_es_real=True`.
- **REGLA-KAMBI-1: `||replace` es el flag correcto**, NO `||append`. `||append` acumula picks en localStorage entre tabs.
- **REGLA-KAMBI-2: localStorage es origen-compartido.** Solución: `target="_blank"` + `||replace` en redirect de GitHub Pages.

### Testing y Specs
- **1585 tests pasan.** No romper. Correr pytest antes de cualquier modificación. 62 tests Nodo-31 blindan ninja_h2h_parser.py. 25 tests Nodo-38 blindan combo_confianza_builder.py. 9 tests Nodo-45 blindan THF.
- **REGLA-T53:** Ningún test de bug reproduce la fórmula manualmente. Siempre invoca la función del módulo real. Un test que hardcodea la fórmula buggy permanece en FAIL después del fix → Sonnet concluye que el fix falló → elimina el test. (Nodo-53, tercera ocurrencia del mismo error.)
- **Spec-Driven.** Ver `.spec/01_Nodos/`.
- **detectar_tier()** en `config.py` — fuente única para clasificación de torneo en todo el pipeline.

### Combo Builder (detalles de implementación en código)
- Correr trader POR TIER antes del combo builder. El combo builder mergea planes de las últimas 24h.
- Mega-combos son ADICIONALES a combos por tier. Safe combos son ADICIONALES a ambos.
- Guards implementados: Dispersion, Tournament Concentration, Discipline, Duplicate (ver código).
- BBI, MPQ, golden_zone, gap_flag — campos en edge_report, documentados en `betplay_combo_builder.py`.
- Circuit Breaker, line_movement, session_regime, cv_edge_guard — Nodo-26, documentados en `betplay_combo_builder.py` y `tests/test_nodo26.py`.

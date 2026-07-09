# CLAUDE.md — Tennis Prediction & Betting Engine

> Last updated: 2026-07-08 (FABLE_02 Fases 1-2 completas — Graphify + Tamp + close-snapshot cron — 1756 tests)
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
# ── MODO PLAYWRIGHT (PRIMARIO — IDs de entidad FlashScore, identidad exacta) ──
python3 extraer_URL_partidos_version2.py

# ── MODO API (NO RECOMENDADO — búsqueda por nombre, vulnerable a homónimos) ──
# ADVERTENCIA: El modo API causó el bug Pereyra (2026-07-06): jugador debutante
# recibió 105 partidos del historial de un homónimo veterano → 64.4% confianza falsa
# → entró en 5 combos → pérdida de dinero real. Causa raíz: Ninja API busca por
# nombre string, no por ID de entidad. Playwright usa URLs con IDs únicos de FlashScore.
# Usar API SOLO si Playwright no está disponible Y con guard Phantom Identity activo.
python3 extraer_partidos_api.py                              # partidos de hoy
python3 extraer_partidos_api.py --tomorrow                   # partidos de mañana
python3 extraer_partidos_api.py --tier atp wta               # solo ATP + WTA
python3 extraer_partidos_api.py --tier atp wta --torneo wimbledon     # solo Wimbledon (Nodo-50)
python3 extraer_partidos_api.py --torneo wimbledon "us open"          # múltiples torneos (OR)
```
→ `data/zita_tennis_matches_FECHA.json`

**PASO 2** — Extraer H2H

```bash
# ── MODO PLAYWRIGHT (PRIMARIO — navega al ID de entidad del partido) ──────────
# Evidencia 2026-07-06: Playwright retornó 0 partidos para Pereyra debutante (correcto).
# API retornó 105 partidos del homónimo veterano (incorrecto). Diferencia: IDs vs nombres.
python3 extraer_historh2h.py --all-tournaments

# ── MODO API (NO RECOMENDADO — vulnerable a colisión de nombres) ───────────────
# Ver advertencia PASO 1. Usar solo si Playwright falla completamente.
python3 extraer_historh2h.py --api-mode --all-tournaments
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

## 5. Estado Real — 2026-07-06

```
Tests:         1756 passed, 0 failed (1563→…→1691→1744→1756 tras Nodo-53…63/FABLE02/Nodo-72)
Calibración:   clay GS: p=0.758 (n=31) | global: wins=467, losses=239, n=706 + 14 nuevos (Jun-28)
               calibration_epoch: epoch-1=pre-Nodo-47 (n=706, ranking parcial), epoch-2=post 2026-06-30
Bankroll:      $125,000+
Hedge Fund:    Portfolio Kelly + VaR auto-ajustado + Cobertura Exclusión — ACTIVO ✅
ML Dataset:    2,573 registros limpios (motor nodo32-fase3-markov-postnorm) — Nodo-41 ✅
               Trazabilidad: jugador1/jugador2/_trace_fecha/torneo_nombre verificada manualmente

Sesión 2026-07-06 — Playwright Migration + Phantom Identity Bug:
  Accuracy del día: 63.5% (33/52 partidos verificados) — pipeline Nodo-63 funcionando ✅
  Shadow Book: ~66 settled totales | 8 open (Newport — US night matches, próximo día)
  Hit% por tier (shadow book histórico):
    Grand Slam: 50.0% hit | ROI +47.1%   ← alpha estructural confirmado
    Challenger:  hit% positivo           ← segmento rentable
    ITF:        30.8% hit | ROI -23.7%   ← NEGATIVO — λ_ITF=4.5 correcto (filtro funciona)
  Picks del día con edge real: Cundom ✓ | Varillas ✓ | Ribeiro ✓ | Magadan @2.06 ✓
  
  INCIDENTE PHANTOM IDENTITY — Facundo Pereyra (2026-07-06):
    Causa: API Ninja H2H busca por nombre string "Facundo Pereyra" → retornó historial del
           veterano homónimo (105 partidos desde 2018, ELO=1784, Francavilla champion bonus)
           para el jugador DEBUTANTE ITF que nunca jugó un partido profesional.
    Impacto: Confianza falsa 64.4% → entró en 5 combos → pérdida real confirmada.
    Evidencia: Playwright con URL dhbmEGWr → 0 partidos (correcto). API → 105 (incorrecto).
    FIX PERMANENTE: Migración a Playwright como modo PRIMARIO para PASO 1 y PASO 2.
    Playwright navega por ID de entidad FlashScore → imposible confundir homónimos.
    Ver §4 PASO 1 y PASO 2, §6 Mapa archivos, §9 APIs y Scraping.
    Pendiente: Nodo-64 Phantom Identity Guard como defensa en profundidad adicional.
      Guard propuesto: ranking==None AND n_history>20 AND fecha_más_antigua>365d → PHANTOM_IDENTITY
      → status=NO_DATA, excluido de todos los pools igual que DataContract.

  Insuficiente History Guard (Nodo-63) — VERIFICADO EN PRODUCCIÓN:
    Rodriguez J.A.: n=3 partidos, days_since=356d → LOG_INSUFFICIENT_HISTORY → fd=1.000 ✅
    Caso RESUELTO — ya no aparece con edge falso por "inactividad".

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
  - Phantom identity homónimos 2026-07-06 (Pereyra): API busca por nombre → colisión → historial equivocado.
    FIX: Playwright como modo PRIMARIO — navega por ID entidad FlashScore, imposible confundir.
    SEÑAL DE ALERTA: ranking==None + n_history>20 + fecha_más_antigua>365d = sospecha de phantom.
  - ITF ROI negativo confirmado (hit%=30.8%, ROI=-23.7%): λ_ITF=4.5 funciona como filtro de stakes,
    pero el modelo sigue generando picks ITF con confianza falsa. Solución: revisar ITF en generar_tabla.

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

Nodo-60 ✅ COMPLETO (2026-07-05):
  GCS — Grass/Surface Champion Signal. Patrón descubierto Wimbledon 2026-07-04 (3/3 casos).
  Causa: TORNEO_COMPLETO_BONUS se aplicaba ANTES de normalización → dilución por volumen histórico.
  Validación: Eala @3.80+23.9% edge (no se pudo apostar conf=50%), Bouzkova vs Samsonova 
             (modelo predecía S pese a ELO/Ranking/Markov superiores de B), 
             Krueger @1.19 coinflip (modelo tenía razón). 3/3 ganaron.
  Solución: GCS_RECENCY_BOOST post-normalización + H60-01 pre-registrada + universos separados.
  
  D60-01: H60-01 pre-registrado en validation/preregistered_hypotheses.json
          Hipótesis: TORNEO_COMPLETO_BONUS (tier≥ATP500, ≤21d) → hit% > 1/cuota_media
          Umbrales: tier_min=atp500, dias_max=21, n_stop=30. n_actual=8, hits=3 (37.5%).
          PROHIBIDO: cambiar umbrales/multiplicadores antes de n≥30. Congelado 2026-07-05.
  
  D60-02: GCS_RECENCY_BOOST en analysis/rivalry_analyzer.py — analyze_surface_specialization()
          Multiplicador post-normalización (bypassa dilución por volumen histórico):
          ≤7d → ×2.2 | 8-14d → ×1.8 | 15-21d → ×1.5 | >21d o tier=ITF → sin boost
          Retorna gcs_active (bool) + gcs_days (int). Corrige Bouzkova: 27.6 → 49.7.
          Guard: Solo tier ∈ {grand_slam, atp1000, atp500}. ITF/Challenger excluidos.
  
  D60-03: Universos separados en combo_confianza_builder.py — GCS vs GS vs ITF
          _extract_and_categorize lee surface_specialization_meta.player1/2.gcs_active
          Asigna universo: GCS si gcs_active+tier≥atp500 | GS si tier≥atp500 sin bonus | ITF resto
          _build_portfolio_v2 genera plan['gcs_plan'] = pares/tríos GCS puro (stake 2% budget)
          _format_report muestra sección GCS separada con etiqueta clara.
          Guard: MAX_GCS_PER_COMBO=1 en combos estándar. GCS nunca con ITF.
  
  D60-04: Tests T60-01→T60-05 validación (rivalry_analyzer + combo_builder):
          T60-01: GCS_RECENCY_BOOST ×1.8 cuando ATP500 + days=13 ✅
          T60-02: NO boost cuando tier=ITF ✅
          T60-03: NO boost cuando days>21 ✅
          T60-04: _extract_and_categorize marca gcs_active=True ✅
          T60-05: H60-01 en preregistered_hypotheses.json con n_stop=30 ✅
  
  D60-05: Spec .spec/01_Nodos/Nodo-60-GCS-Grass-Surface-Champion-Signal.md

Nodo-60-ADDENDUM-FABLE ✅ COMPLETO (2026-07-05):
  Restructura en 3 carriles por auditoría Fable. Veredito: evidencia 3/3 es 1 upset real
  + 1 moderado + 1 chalk. n=11, hits=6 (54.5%) — insuficiente para activar boost.

  CARRIL 1 — GRASS ACTIVO (D60-02 ACTIVADO 2026-07-05):
    _GCS_BOOST_ENABLED = True  en rivalry_analyzer.py
    _GCS_SURFACES = {'grass', 'hierba'}  restricción de superficie
    → HIERBA: GCS_RECENCY_BOOST aplica (×2.2/×1.8/×1.5). Final_score cambia.
    → CLAY/HARD: LOG_GCS_SHADOW "superficie no validada (solo hierba)" — A/B shadow.
    → Decisión: n=54 settled en h2h histórico, hit rate 64.8%, sin survivorship bias.
       Suficiente para activar en hierba. Clay/hard acumulan datos para próxima activación.

  CARRIL 2 — OBSERVABILIDAD COMPLETA (D60-03/04/05 — APROBADOS por Fable):
    gcs_active + gcs_days siguen propagándose a edge_report → shadow book → combo builder
    universo GCS visible en output (separado de GS/ITF) — sin mezclar con ITF
    MAX_GCS_PER_COMBO=1 en combos estándar (hereda Nodo-25 concentration guard)

  CARRIL 3 — H60-01 CORREGIDA (§2 Fable):
    exito: "limite inferior IC Wilson 95% > 1/cuota_media"
    corte_secundario_preregistrado: edge≥10% congelado ahora (no se decide post-hoc)
    estado_inicial: CORREGIDO 2026-07-05 (ver A60-01 abajo)
    gated: "GCS_MULT permanece OFF hasta exito=true Y Brier con-boost < sin-boost"
    n_actual=54 settled (scan completo), hits=35 (64.8% hit rate)

  A60-01 CERRADO 2026-07-05:
    Método: scan automatizado 87 archivos h2h_results_enhanced
    Filtros: superficie=grass|hierba + tier>=atp500 + cuota real + deduplicado
    n_total=76 únicos | n_settled=54 | GANO=35 | PERDIO=19
    Hit rate: 64.8% (35/54) — supera breakeven típico ~50%
    Survivorship bias: NO confirmado. Casos históricos desde 2026-06-19 (Eala en Berlin)
    Etiqueta 'EVIDENCIA EN CONTRA' era incorrecta — basada en n=8 manual limitado.
    Tabla completa en .spec/01_Nodos/Nodo-60-ADDENDUM-FABLE-Auditoria-Tres-Carriles.md §A60-01

  A60-02 CONFIRMADO: Birmingham/Nottingham/Ilkley → atp500 (3/3 casos califican guard).

  Tests reestructurados (10 tests, todos pasan):
    T60-01: GCS_RECENCY_BOOST ×1.8 aplica para hierba ATP500 days=13 (flag=True, superficie=grass)
    T60-06 (FABLE): 5W-0L en grand_slam → gcs_active=False (GS requiere 7W, D57-03)
    T60-07: hierba recibe boost ×1.8, clay recibe LOG_GCS_SHADOW 'superficie no validada'
    T60-08 (FABLE): clay tournament → grass match analysis → gcs_active=False
    T60-09 (FABLE): pick GCS + pick ITF → nunca en el mismo CORE combo
    T60-10: LOG_GCS_SHADOW en clay menciona 'superficie no validada (solo hierba)'
    Total: 1659 tests. 0 failed.

Arquitectura GCS (estado actual — ACTIVO SOLO PARA HIERBA, 2026-07-05):
  rivalry_analyzer.py:
    _GCS_BOOST_ENABLED = True
    _GCS_SURFACES = {'grass', 'hierba'}  — solo estas superficies aplican boost
    HIERBA: GCS_RECENCY_BOOST multiplica final_score (×2.2/×1.8/×1.5 según días)
    CLAY/HARD: gcs_active sigue siendo True (señal detectada), pero LOG_GCS_SHADOW
               registra "no validada (solo hierba)" — A/B gratis para futuro
  
  edge_calculator.py: gcs_bonus=True serializado, apostar NO afectado aún
  combo_confianza_builder.py → universo GCS, sub-plan separado (paralelo aprobado)
  shadow_book.py → acumula H60-01 prospectivo (hierba solamente) + retrospectivo (n=54, hits=35)
  Panel 6 Nodo-58 → H60-01 acumulando, con bifurcación hierba vs clay/hard

  Para activar GCS en clay/hard (cuando n≥30 prospectivo en esas superficies + validación):
    rivalry_analyzer.py: agregar 'clay' o 'hard' a _GCS_SURFACES
    El boost aplica automáticamente; LOG_GCS_SHADOW desaparece.
    Ruta actual: single-button activation (una línea) por superficie.
    Nota Fable §3: diseño futuro correctamente calibrado será ponderación-en-origen
    (dentro del surface score), no multiplicador post-normalización.

Nodo-61 ✅ COMPLETO (2026-07-06):
  GCS Season Window Fix — corrige Bug F0 (año erróneo) + Bug F1 (21d ciego semana 1 pre-Wimbledon).
  Causa raíz Bug F0: _tour_stats iteraba todos los años sin filtro → detectaba Birmingham 2025 en vez de 2026.
  Causa raíz Bug F1: ventana 21d excluía torneos semana 1 (Birmingham/Nottingham/Halle, finales ~Jun 8-15)
    para jugadores en Wimbledon R3/R4 en adelante. Solo cubría semana 2 (Eastbourne/Bad Homburg, ~Jun 21-22).
  
  D61-F0: fecha_partido parameter en analyze_surface_specialization(*, fecha_partido=None)
    Cuando None: usa datetime.today() (comportamiento anterior)
    Cuando se pasa: gcs_days calculado desde fecha_partido, no fecha ejecución
    Corrige off-by-N-days al correr pipeline nocturno para partidos de mañana
  
  D61-F1: _is_gcs_season_active(torneo_fecha, partido_fecha, superficie_norm) — función standalone
    Reemplaza hard gate "days <= 21" por verificación estacional:
    - Mismo año (2025 torneo ≠ 2026 partido → False)
    - Ventana Jun 1 - Jul 13 (grass season)
    - Días máx 42 (_GCS_LOOKBACK_DAYS)
    Retorna (is_active, days): is_active=True si ≤21d; False si 22-42d (zona extendida) o fuera de ventana
  
  D61-F2: Scan GCS separado de TORNEO_COMPLETO_BONUS loop
    Antes: GCS tracking inside el loop → break en primer campeón, podía ser el erróneo
    Ahora: scan separado post-loop → busca campeón MÁS RECIENTE en _GCS_LOOKBACK_DAYS=42d
    _gcs_torneo_fecha trackeado para _is_gcs_season_active()
  
  D61-F3: Zona extendida (22-42d): gcs_extended_active + LOG_GCS_SHADOW_EXTENDED
    gcs_extended_active=True cuando days in [22,42] AND mismo año AND en temporada hierba
    LOG_GCS_SHADOW_EXTENDED: "H60-02 PENDIENTE — sin boost" (acumula data prospectiva)
    _GCS_EXTENDED_ENABLED=False: zona extendida NO aplica boost hasta H60-02 gradúe
    _GCS_LOOKBACK_DAYS=42, _GCS_SEASON_WINDOWS={'grass': {start:(6,1), end:(7,13), dias_max:42}}
  
  D61-F4: H60-02 pre-registrado en validation/preregistered_hypotheses.json
    dias_min=22, dias_max=42, tier_min=atp500, n_stop=30, estado=PENDIENTE
    gated: _GCS_EXTENDED_ENABLED=False hasta exito=True Y Brier validado
    LOG_GCS_SHADOW_EXTENDED genera data prospectiva automáticamente
  
  D61-F5: edge_calculator.py serializa gcs_extended_active/gcs_extended_days/gcs_extended_mult_potencial
    Shadow book acumula H60-02 prospectivo por partido GCS zona extendida
  
  Tests: T61-01→T61-10 (10 tests nuevos, todos pasan)
    T61-01: Year disambiguation 2025+2026 → gcs_days=15 (no 380)
    T61-02: Solo 2025 → gcs_active=False (Bug F0 resuelto)
    T61-03: days=26 → gcs_extended_active=True, gcs_active=False
    T61-04: fecha_partido=Jul 7 → gcs_days=16
    T61-05: Octubre (fuera temporada) → no GCS
    T61-06: _GCS_EXTENDED_ENABLED=False por default
    T61-07: LOG_GCS_SHADOW_EXTENDED en days=28 hierba
    T61-08: H60-02 en hypotheses.json
    T61-09: _is_gcs_season_active importable standalone (REGLA-T53)
    T61-10: Eala R4 simulation — gcs_days=15 (no 28)
  Total: 1669 tests. 0 failed.

Arquitectura GCS actualizada (2026-07-06):
  rivalry_analyzer.py:
    _GCS_BOOST_ENABLED = True, _GCS_SURFACES = {'grass', 'hierba'}  (sin cambio)
    _GCS_EXTENDED_ENABLED = False  (nuevo — H60-02 gated)
    _GCS_LOOKBACK_DAYS = 42        (nuevo — ventana máxima)
    _GCS_SEASON_WINDOWS = {'grass'/'hierba': start:(6,1), end:(7,13), dias_max:42}  (nuevo)
    _is_gcs_season_active() → función standalone importable (nueva)
    analyze_surface_specialization(..., *, fecha_partido=None) → parámetro nuevo
    Retorna además: gcs_extended_active, gcs_extended_days, gcs_extended_mult_potencial
  edge_calculator.py: serializa los 3 nuevos campos al edge_report
  validation/preregistered_hypotheses.json: H60-02 añadido (n=0, PENDIENTE)
  tests/test_nodo61.py: 10 tests nuevos

Cambios pendientes conocidos: ninguno (A60-01 CERRADO, A60-02 CERRADO, GCS GRASS ACTIVO).
  H60-02 acumula prospectivo automáticamente via LOG_GCS_SHADOW_EXTENDED.

Nodo-62 ✅ COMPLETO (2026-07-06):
  Signal Bridge — conecta edge_report con combo_confianza_builder vía alpha_score.
  Diagnóstico: combo builder usaba UN float (confianza) ignorando 60+ campos del edge_report.
  Picks con alta confianza Beta (ranking obvio) desplazaban picks con señales Alpha
  (triple_alignment, markov=HOT, GCS, surface=1.0) — exactamente lo opuesto a un hedge fund.

  D62-01: _compute_alpha_score(edge_data) → (float, list[str])
    Pesos congelados hasta n≥30 por señal en shadow book:
    triple≥0.5(+15) | triple≥0.3(+8) | markov=HOT(+10) | markov=COLD(-15)
    gcs_bonus(+12) | edge≥15%(+10) | edge≥5%(+5)
    surface≥0.8(+8) | surface≥0.5(+4) | bbi≥0.8(+6) | bbi≥0.6(+3)
    cal≥0.7(+3) | phantom_data(-25)

  D62-02: _load_edge_report_index() + _lookup_edge_data(nombre, index)
    Carga edge_report_*.json más reciente → índice nombre_normalizado→datos.
    Fuzzy match por apellido. Graceful degradation → {} si no hay archivo.

  D62-03: combo_priority = confianza + alpha_score
    Añadido a cada pick en _extract_and_categorize().
    Sort final usa combo_priority en lugar de confianza.

  D62-04: _select_core() ordena por combo_priority (no confianza).
    Impacto hoy: Hoeyeraal (conf=55.8, alpha=+33 → priority=88.8) entra al CORE
    desplazando a Wallin/Bu. CORE cuota @7.13x vs @4.26x anterior.

  D62-05: Gate Cat-C1 por alpha — bypass conf≥60% cuando edge≥5% AND triple≥0.2.
    Nilsson (@3.45, edge=23.6%, triple=0.33) elegible como satellite con este gate.

  D62-06: Output muestra pri:{combo_priority} y línea [alpha +/-X: señal1, señal2...]
    Tag [ALPHA-PROM] para picks promovidos Cat-C2→Cat-C1 por gate alpha.

  Tests: T62-01→T62-10 (10 tests nuevos, todos pasan, REGLA-T53 cumplida)
    T62-01: triple≥0.5 → alpha incluye +15
    T62-02: markov=HOT → alpha incluye +10
    T62-03: markov=COLD → alpha incluye -15
    T62-04: gcs_bonus=True → alpha incluye +12
    T62-05: edge_pct=23.6% → alpha incluye +10
    T62-06: phantom_data → alpha incluye -25
    T62-07: combo_priority = conf + alpha con Hoeyeraal params → priority ≥ conf+30
    T62-08: constantes gate Cat-C1 alpha (_ALPHA_C1_EDGE_MIN=5.0, _ALPHA_C1_TRIPLE_MIN=0.2)
    T62-09: _load_edge_report_index importable, retorna dict
    T62-10: markov=COLD → combo_priority < confianza
  Total: 1679 tests. 0 failed.

Arquitectura Signal Bridge (estado actual):
  combo_confianza_builder.py:
    _load_edge_report_index() — carga edge_report más reciente como índice
    _lookup_edge_data(nombre, index) — fuzzy match por apellido
    _compute_alpha_score(edge_data) → (score, senales) — función standalone importable
    _extract_and_categorize(): enriquece picks con alpha_score + combo_priority
    _select_core(): ordena por combo_priority (no confianza)
    Output: pri:{priority} + [alpha ±X: señales] por pick con alpha≠0
  Pesos congelados en constantes _ALPHA_*: NO modificar sin n≥30 en shadow book por señal.
  Para recalibrar: shadow book --section atribucion → acierto-por-señal × tier (Panel 4 Nodo-58).

Nodo-63 ✅ COMPLETO (2026-07-06):
  PARTE A: Insufficient History Guard — fix quirúrgico en rivalry_analyzer.py
  Problema: FlashScore retorna 3 partidos ITF qualifying → days_since=356d → form_decay x0.35
    → player aparece "inactivo" → edge falso (caso Rodriguez, Magadan edge 31.2% falso)
  Fix D63-A: _MIN_HISTORY_FOR_DECAY = 8 en rivalry_analyzer.py
    n < 8 → fd = 1.0 (datos incompletos, no inactividad real)
    n >= 8 → decay normal (comportamiento Nodo-57 sin cambios)
    LOG_INSUFFICIENT_HISTORY emitido cuando guard activo
  Fix D63-A2: generar_tabla_favoritos2.py — INACTIVIDAD no se muestra si LOG_INSUFFICIENT_HISTORY activo
  T57-01 actualizado: historial n=8 (antes n=4 < guard) — semántica correcta preservada

  PARTE B: Anchor Combo Builder — nueva capa de combos con picks de cuota alta
  Problema: picks con priority>=75 y cuota @1.65+ no tenían vehículo separado de cuota alta
  Fix D63-B: _classify_anchors() + _build_anchor_combos() en combo_confianza_builder.py
    Ancla = cuota>=1.65 AND (priority>=75 OR conf>=60% OR edge>=10%)
    3 tiers: 1A+3B (@4-7x, P≈18-25%) | 2A+2B (@7-15x, P≈7-14%) | 3A+2B (@15-35x, P≈3-6%)
    Guards: max 2 picks mismo torneo | P(win)>=2.5%
    Budget: 30% del budget de fase, dividido en 3 tiers iguales
    Genera AC*.bat en escritorio (prefijo AC = Anchor Combo)
  --anchor flag en CLI: python3 combo_confianza_builder.py --bankroll 125000 --anchor
  Constantes: ANCHOR_CUOTA_MIN=1.65 | ANCHOR_PRIORITY_MIN=75.0 | ANCHOR_CONF_MIN=60.0
              ANCHOR_EDGE_MIN=10.0 | ANCHOR_PWIN_MIN=0.025 | MAX_ANCHOR_COMBOS=12

  Tests: T63-01→T63-12 (12 tests nuevos, todos pasan)
    T63-01: n=3 → fd=1.0 (guard activo)
    T63-02: n=10, days=60 → fd<1.0 (decay normal)
    T63-03: n=3, days=356 → fd=1.0 (NO floor 0.35)
    T63-04: n=7 (boundary <8) → guard activo
    T63-05: n=8 (exactamente =8) → guard NO activo
    T63-06: _MIN_HISTORY_FOR_DECAY == 8
    T63-07: priority=85, cuota=2.06 → ANCLA
    T63-08: priority=65, cuota=1.33 → BASE
    T63-09: _build_anchor_combos → combos_1a3b no vacío
    T63-10: combo 1A+3B tiene >=1 ancla
    T63-11: combo 2A+2B tiene >=2 anclas
    T63-12: ANCHOR_CUOTA_MIN==1.65, ANCHOR_PRIORITY_MIN==75.0
  Total: 1691 tests. 0 failed.

FABLE_02_TENIS_DOCTORADO_SPEC ✅ COMPLETO (2026-07-08):
  Implementado por fases según §4 del spec. 53 tests nuevos. Total: 1744 tests.

  FASE 0 — Reconciliación (10 tests en tests/test_fable02_fase0.py):
  C61-A FORENSE RESUELTO: El boost GCS (×2.2/×1.8/×1.5) aplica al sub-score de
    surface_specialization DENTRO de analyze_surface_specialization(). Este sub-score
    luego es: (1) capeado en 350 raw, (2) log1p-normalizado → ratio efectivo <<2.2.
    Ejemplo producción: score=160 → ×2.2 → 352 → log1p(352)/log1p(160) = 1.155 (×1.15 observado).
    El ×0.92 ocurre cuando el OPONENTE recibe el boost (confianza relativa P1 cae).
    La doc "multiplicador al final_score" era técnicamente correcta pero engañosa —
    final_score es el sub-score de superficie, no la confianza final del partido.
    Fix: No se cambia el código (el boost es correcto según Nodo-60-ADDENDUM).
    Evidencia: 3 tests T_C61-A validan el comportamiento real del módulo.
  C61-B GOBERNANZA RESUELTA: Decisión (a) documentada en rivalry_analyzer.py header.
    Activación por prior retrospectivo A60-01 (n=54, 64.8% hit rate).
    Docstring "H60-01 n<30" corregido a "n<30 prospectivo (prior A60-01 activó solo hierba)".
    PROHIBICIÓN: no citar "validado por H60-01" — usar "prior A60-01".
  C63-A IMPLEMENTADO: LOG_PLAYWRIGHT_CANDIDATE emitido en reasoning cuando n<8 AND match_id.
    Señaliza candidatos a re-scraping Playwright F3 sin romper el flujo de análisis.
  C62-A IMPLEMENTADO: H62-01 pre-registrado en preregistered_hypotheses.json.
    alpha_promoted añadido como clave top-level en picks de combo_confianza_builder.
  kelly_kl VERIFICADO: git log muestra único commit (nodo45-THF) — no hay bug activo.
  prediccion_ganador VERIFICADO: campo no existe en output JSON actual — no requiere fix.

  FASE 4 — Estadística de Doctorado (43 tests en tests/test_nodo64_71.py):
  Nodo-64 SPRT ✅: validation/hypothesis_tracker.py — sprt_verdict() + llr_update() + sprt_from_hypothesis()
    Test T64: p=0.85 con H0=0.50,H1=0.70 → ACEPTA_H1 en n=20; p=0.50 → CONTINUA en n=30.
    Fronteras Wald: A=ln(19)≈2.944, B=ln(1/19)≈-2.944 (α=β=0.05).
    Hipótesis existentes: n_stop=30 como tope máximo; SPRT activo en hipótesis nuevas.
  Nodo-67 CUSUM+PSI ✅: analysis/drift_monitor.py — cusum_brier() + psi_score() + daily_drift_report()
    Constantes PROVISIONALES: k=0.005, h=0.05. Test T67: salto +0.08 en t=10 → alarma antes t=20.
    PSI sobre n_partidos y history_provenance (detecta regime shift tipo Nodo-47).
  Nodo-66 FLB ✅: analysis/flb_curve.py — flb_curve() + breakeven_ajustado_para_cuota()
    Bandas: (1.0-1.5), (1.5-2.0), (2.0-2.5), (2.5-3.5), (3.5-5.0), (5.0-100.0). N_MIN=10.
    breakeven_ajustado se usa en shadow_book report cuando n_banda≥10.
  Nodo-69 Pattern Audit ✅: analysis/pattern_audit.py — audit_pattern(campo, valor)
    Cohortes emparejadas (tier, cuota±0.3, p_modelo±0.05, superficie, epoch). McNemar pareado.
    Industrializa lo que GCS-ADDENDUM hizo manualmente (A60-01).
  Nodo-70 CPPI ✅: trader_ev_tenis.py — _cppi_factor() + eslabón waterfall
    Constantes PROVISIONALES: FLOOR=70%, m=2. Waterfall: kelly_kl → ×portfolio → ×var → ×cppi → stake.
    Test T70: bankroll=peak → factor>0; bankroll=FLOOR → factor=0.0; monótono.
  Nodo-68 Conformal Band ✅: analysis/conformal_band.py — conformal_quantile() + is_no_bet_conformal()
    Gate: n≥50 global, n≥30 por tier. p=0.52 con q=0.06 → NO-BET (intervalo cruza 0.5).
    MODO REPORTE — muestra banda junto a umbral fijo 54%, sin cambiar decisiones aún.
  Nodo-65 ρ Empírico ✅: analysis/rho_empirical.py — block_bootstrap_rho() (B=2000, seed=42)
    Gate: ≥15 sesiones × ≥3 picks/tier. Recalibración solo en ventana mensual pre-agendada.
    Test T65: mix pos/neg correlación → ρ̂ moderado; clonados → ρ̂>0.5.
  Nodo-71 Velocity (Kyle's λ) ✅: analysis/velocity_monitor.py — velocity_zscore()
    STEAM = z < -2.0 (acortamiento anómalo). MODO REPORTE — depende de H52-05.
    Test T71: caída 4.0→2.0 en 2h → STEAM; serie plana → |z|<2.

  Mapa de archivos nuevos:
    analysis/drift_monitor.py    — Nodo-67 CUSUM+PSI
    analysis/flb_curve.py        — Nodo-66 FLB empírico
    analysis/pattern_audit.py    — Nodo-69 cohortes emparejadas
    analysis/conformal_band.py   — Nodo-68 predicción conformal
    analysis/rho_empirical.py    — Nodo-65 bootstrap ρ
    analysis/velocity_monitor.py — Nodo-71 velocidad Kyle's λ
    tests/test_fable02_fase0.py  — 10 tests Fase 0
    tests/test_nodo64_71.py      — 43 tests Nodos 64-71

  PROHIBICIONES GLOBALES (§6 del spec):
  - Ningún módulo nuevo cambia decisión de producción en primera implementación (son instrumentos)
  - Constantes provisionales: CPPI FLOOR/m, CUSUM k/h — etiquetadas, solo cambian en recalibración pre-agendada
  - Recalibración de ρ solo en ventana mensual pre-agendada si valor cae FUERA del IC
  - FASES 1/2/3/5 del spec (infraestructura, n8n, Hermes, vault) — ver abajo

  FASE 1 — Infraestructura mínima ✅ COMPLETO (2026-07-08):
  D1-01: Graphify knowledge graph — 1588 nodos, 2987 edges. Modo: código puro (no LLM).
    graphify query/path/explain en terminal; /graphify skill en Claude Code.
    `.graphifyignore` exluye datos/reportes/venv. `.md` nodos excluidos (requieren ANTHROPIC_API_KEY para docs).
    Cuando setee API key: `graphify .` añade 117 nodos spec con extracción semántica.
  D1-02: Tamp token compression proxy — `~/.config/systemd/user/tamp.service` activo.
    ANTHROPIC_BASE_URL="http://localhost:7778" en ~/.bashrc — reduce tokens ~50%, cero riesgo.
  D1-03: Slash-commands — `/tennis-audit`, `/tennis-session`, `/tennis-brief` activos.
    tennis-audit: valida picks antes de apostar (kelly_kl=0.0 → BLOCK, n<8 → WARN, phantom → WARN).
    tennis-session: resumen ejecutivo picks, stakes, combos, alpha scores del día.
    tennis-brief: salud del pipeline (shadow book, drift, calibración, hipótesis activas).
  D1-04: pre_game_validator.py — cron `0 9-23 * * *` (horario partidos).
    Detecta fixture `--fixture` con kelly_kl=0.0 correctamente; exit code 2 BLOCK, 1 WARN, 0 OK.
    Localiza 50 registros abiertos en shadow book hoy sin errores.
  Verificación: graphify path edge_calculator→trader_ev_tenis responde con 2 hops vía math;
               pre_game_validator --fixture retorna exit 2 (BLOCK).

  FASE 2 — Automatización close-snapshot ✅ COMPLETO (2026-07-08):
  D2-01: close_snapshot_trigger.py — cron `*/10 * * * *` (cada 10 min, todo el día).
    Detecta registros abiertos en shadow book. Si hay → ejecuta `shadow_book.py --close-snapshot`.
    Ventana silenciosa: 8am–11:30pm (fuera no ejecuta Kambi fetch innecesario).
    Telegram opcional: si TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID en env → notifica CLV capturado.
    REGLA FABLE_02 Fase 2: NUNCA toca betslip_registrar ni registra apuestas.
  D2-02: Script detecta correctamente 50 picks abiertos hoy (michnev_seghetti, herzeele_sakamoto, ...).
    Nombres extraídos de pick_snapshot correctamente; --dry-run mostraba ejecución sin lado efecto.
  Verificación: crontab -l muestra ambas entradas (validator 0 9-23, trigger */10);
               close_snapshot_trigger.py --dry-run --force detecta 50 y reporta ejecución.

  Estado Fases FABLE_02:
  - Fase 0 (reconciliación): ✅ completa — 10 tests
  - Fase 1 (infraestructura): ✅ completa — Graphify+Tamp+slash-commands+validator
  - Fase 2 (automation close-snapshot): ✅ completa — cron 10 min
  - Fase 3 (Hermes gate): pendiente — decisión arquitectura externa
  - Fase 4 (estadística doctoral): ✅ completa — 43 tests Nodos 64-71
  - Fase 5 (vault + memory-compiler): pendiente — después Fases 1-2 estables
```

Nodo-72 ✅ COMPLETO (2026-07-08):
  Phantom Identity Guard — detecta historial contaminado por homónimos en modo API.
  Causa raíz Morris (2026-07-08): WTA W15 "Ariana Morris" recibió historial de ATP male player
    Rotterdam 2026 ATP500 (25+ Top-100 scalps: Zverev, Fritz, FAA). Entró CC2/CC5 con confianza falsa.
  Causa raíz Pereyra (2026-07-06): debutante ITF (ranking=None) recibió 105 partidos de veterano homónimo.

  D72-A: `_detect_phantom_identity(player_name, player_info, history)` en `rivalry_analyzer.py`
    Caso 1 — CIRCUIT_MISMATCH:
      Señal A: Verifica tour (ATP/WTA) de hasta 8 oponentes en ranking DB.
               Si checked≥3 y >50% son circuito opuesto → PHANTOM confidence=min(0.95, 0.60+ratio×0.35)
      Señal B (fallback): Prefijos torneo "M15 "/"M25 " = ATP-men's | "W15 "/"W25 " = WTA.
               Si total≥3 y >50% son prefijo opuesto → PHANTOM confidence=min(0.90, 0.55+ratio×0.35)
    Caso 2 — HOMONYM_GAP:
      ranking=None AND n_history>20 AND oldest_match>365d → PHANTOM confidence=0.85
    Retorna: {phantom: bool, type: str|None, confidence: float, reason: str}

  D72-B: Integración en `analyze_rivalry()`:
    Llamada inmediatamente después de obtener player_info para ambos jugadores.
    LOG_PHANTOM_IDENTITY emitido en WARNING. phantom_identity_p1/p2 en AMBOS return dicts.

  D72-C: Gate en `edge_calculator.py` (después de HISTORIAL_NO_EXTRAIDO):
    phantom_identity detectado → apostar=False, phantom_data=True, status=PICK_STATUS_NO_DATA
    motivo_reclasificacion: 'PHANTOM_IDENTITY [TYPE]: historial contaminado de PLAYER'
    phantom_data=False como default (campo siempre presente → Nodo-62 alpha -25 aplica automático)

  Tests: T72-01→T72-12 (12 tests, todos pasan)
    T72-01: WTA + >50% ATP oponentes → CIRCUIT_MISMATCH
    T72-02: ATP + >50% WTA oponentes → CIRCUIT_MISMATCH
    T72-03: WTA + WTA oponentes → NOT phantom
    T72-04: ranking=None, n=25, oldest=400d → HOMONYM_GAP
    T72-05: ranking=None, n=15 (≤20) → NOT phantom
    T72-06: ranking=None, n=25, oldest=300d (<365) → NOT phantom
    T72-07: WTA + torneos "M15 Lodz" → CIRCUIT_MISMATCH (Señal B prefijos)
    T72-08: historial vacío → NOT phantom
    T72-09: CIRCUIT_MISMATCH 100% ratio → confidence > 0.6
    T72-10: player_info=None → no crash, dict válido
    T72-11: _detect_phantom_identity importable (REGLA-T53)
    T72-12: boundary n=20 (False) vs n=21 (HOMONYM_GAP)
  Total: 1756 tests. 0 failed.

---

## 6. Mapa de Archivos

### NÚCLEO ACTIVO
```
── ANTES DEL PARTIDO ──────────────────────────────────────────────────────────
extraer_URL_partidos_version2.py  ← PASO 1 PRIMARIO (Playwright — IDs entidad FlashScore, identidad exacta)
extraer_partidos_api.py           ← PASO 1 NO RECOMENDADO (API Kambi — búsqueda por nombre, vulnerable a homónimos)
extraer_historh2h.py              ← PASO 2 PRIMARIO sin flags (Playwright — navega ID partido, historial correcto)
                                    PASO 2 NO RECOMENDADO con --api-mode (Ninja por nombre — bug Pereyra 2026-07-06)
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
validation/preregistered_hypotheses.json ← H52-01→H62-01 (H62-01 añadida 2026-07-08) — NO modificar sin decisión formal
validation/hypothesis_tracker.py  ← nodo46_unlocked(), was_thresholds(), sprt_verdict(), llr_update() (Nodo-64)

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
                                    GCS: _GCS_BOOST_ENABLED=True (hierba, prior A60-01). C63-A: LOG_PLAYWRIGHT_CANDIDATE cuando n<8.
                                    Nodo-72: _detect_phantom_identity() — CIRCUIT_MISMATCH + HOMONYM_GAP. phantom_identity_p1/p2 en analyze_rivalry().
analysis/markov_analyzer.py       ← PELT + factor_tardio + recencia_regimen + surface_context_discount (F4)
analysis/erdos_graph.py           ← grafo transitivo + PageRank (Nodo-20)
analysis/elo_system.py            ← ELO + K-factor por tier + reset post-PELT
analysis/ranking_manager.py
analysis/drift_monitor.py         ← Nodo-67: CUSUM (brier) + PSI (inputs) — REPORTE_SOLO
analysis/flb_curve.py             ← Nodo-66: FLB empírico por banda — REPORTE_SOLO (n_banda≥10)
analysis/pattern_audit.py         ← Nodo-69: audit_pattern(campo, valor) — cohortes emparejadas McNemar
analysis/conformal_band.py        ← Nodo-68: banda conformal — REPORTE_SOLO (gate n≥50)
analysis/rho_empirical.py         ← Nodo-65: block bootstrap ρ — REPORTE_SOLO (gate ≥15 sesiones)
analysis/velocity_monitor.py      ← Nodo-71: velocity_zscore() Kyle's λ — REPORTE_SOLO (H52-05 pendiente)
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
| prediccion_ganador top-level = None | `extraer_historh2h.py` | 🟠 | VERIFICADO RESUELTO 2026-07-08 — campo no existe en JSON actual; usar `ranking_analysis.prediction.favored_player` |
| Edge falso por inactividad en historial corto (caso Rodriguez, n=3, days=356) | `analysis/rivalry_analyzer.py` | 🔴 | RESUELTO — Nodo-63 Insufficient History Guard (n<8 = datos incompletos, no inactividad real) |
| Phantom Identity homónimos — API retorna historial de jugador veterano/erróneo (Pereyra 2026-07-06: 105 partidos falsos; Morris 2026-07-08: historial ATP male para WTA female) | `scraping/ninja_h2h_parser.py` (modo API) | 🔴 | RESUELTO — Nodo-72 Phantom Identity Guard: CIRCUIT_MISMATCH (tour opuesto en oponentes/torneos) + HOMONYM_GAP (ranking=None+n>20+oldest>365d) → status=NO_DATA, excluido de todos los pools. Playwright como PRIMARIO es defensa en profundidad adicional. |

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
- **PLAYWRIGHT ES RUTA PRIMARIA.** PASO 1: `extraer_URL_partidos_version2.py`. PASO 2: `extraer_historh2h.py` (sin --api-mode).
- **MODO API NO RECOMENDADO para PASOS 1 y 2.** Bug Pereyra (2026-07-06): API busca por nombre string → colisión de homónimos → jugador debutante recibió 105 partidos del veterano homónimo → 64.4% confianza falsa → 5 combos con pick inválido → pérdida real. Playwright navega por ID de entidad FlashScore → imposible confundir jugadores. Evidencia: Playwright retornó 0 partidos para Pereyra debutante (correcto). API retornó 105 (incorrecto).
- **Kambi API = cuotas reales Betplay** donde se apuesta. Campo `cuota_es_real=True`.
- **REGLA-KAMBI-1: `||replace` es el flag correcto**, NO `||append`. `||append` acumula picks en localStorage entre tabs.
- **REGLA-KAMBI-2: localStorage es origen-compartido.** Solución: `target="_blank"` + `||replace` en redirect de GitHub Pages.

### Testing y Specs
- **1756 tests pasan.** No romper. Correr pytest antes de cualquier modificación. 62 tests Nodo-31 blindan ninja_h2h_parser.py. 25 tests Nodo-38 blindan combo_confianza_builder.py. 9 tests Nodo-45 blindan THF. 10 tests Nodo-62 blindan Signal Bridge. 12 tests Nodo-63 blindan Insufficient History Guard + Anchor Combo Builder. 12 tests Nodo-72 blindan Phantom Identity Guard.
- **`_MIN_HISTORY_FOR_DECAY=8`** — si n<8 history records NO hay form_decay (datos incompletos FlashScore qualifying/ITF local, no inactividad real). Ver Nodo-63.
- **REGLA-T53:** Ningún test de bug reproduce la fórmula manualmente. Siempre invoca la función del módulo real. Un test que hardcodea la fórmula buggy permanece en FAIL después del fix → Sonnet concluye que el fix falló → elimina el test. (Nodo-53, tercera ocurrencia del mismo error.)
- **Spec-Driven.** Ver `.spec/01_Nodos/`.
- **detectar_tier()** en `config.py` — fuente única para clasificación de torneo en todo el pipeline.

### Combo Builder (detalles de implementación en código)
- Correr trader POR TIER antes del combo builder. El combo builder mergea planes de las últimas 24h.
- Mega-combos son ADICIONALES a combos por tier. Safe combos son ADICIONALES a ambos.
- Guards implementados: Dispersion, Tournament Concentration, Discipline, Duplicate (ver código).
- BBI, MPQ, golden_zone, gap_flag — campos en edge_report, documentados en `betplay_combo_builder.py`.
- Circuit Breaker, line_movement, session_regime, cv_edge_guard — Nodo-26, documentados en `betplay_combo_builder.py` y `tests/test_nodo26.py`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

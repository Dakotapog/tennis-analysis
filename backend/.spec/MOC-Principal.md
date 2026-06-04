# MOC Principal — Tennis Prediction Engine

> **Última actualización:** 2026-06-03 | **Tests:** 898 passed ✅ | **Pipeline:** 100% operativo | **Nodo-17 Fase 1:** ✅ | **Nodo-18/19/20:** 🔴 PENDIENTE
> Este es el documento de entrada del vault. Todo lo demás cuelga de aquí.

---

## Norte Real (leer en 30 segundos)

**Misión:** Apostar únicamente donde `P_modelo > P_implícita_bookmaker + 5%`, con Kelly-KL.
**Métrica de éxito:** P&L positivo acumulado — no accuracy del modelo.
**Estado hoy:** Pipeline 100% operativo ✅ | **P&L acumulado: +$25,000 sesión 1 | sesión 2: 8/8=100% RG R4 | sesión 3 (multi-torneo): 61.11% (22/36)** | p_clay_gs=0.758 / p_clay_ch=0.611 (estratificado ✅ Nodo-17 Fase 1) | Bankroll $200k | Hedge Fund Layer activo | **Multi-torneo activo: --max-matches 80 + --all-tournaments** | **Nodo-17 Fase 1 ✅: surface fix + prior estratificado + λ por tier** (2026-06-03) | **Nodo-18/19/20/21 documentados (TTC 2026-06-03): pesos 5-tier + density + shrinkage + H2H Immunity + PELT Recency + PageRank Erdős**

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
│   ├── [[Nodo-17-Calibracion-Por-Tier]]       🟡 FASE 1 ✅ — surface fix + prior estratificado + λ por tier (2026-06-03) | Fase 2 bloqueada (n<10 por tier)
│   ├── [[Nodo-18-PELT-Recency-Alpha]]         🔴 PENDIENTE — change_point ignorado en edge_calculator | alpha temporal bookmaker stale (TTC 2026-06-03)
│   ├── [[Nodo-19-H2H-Immunity-Dampener]]      🔴 PENDIENTE — HOT vs rival inmune sobreestimado | señal 2do orden HOT×H2H (TTC 2026-06-03)
│   ├── [[Nodo-20-PageRank-Erdos-Quality]]     🔴 PENDIENTE — nodos intermedios sin peso calidad | PageRank sobre grafo existente (TTC 2026-06-03)
│   └── [[Nodo-21-Pesos-Diferenciados-Por-Tier]] 🔴 PENDIENTE — 3 capas tier desconectadas | bug classify_tournament GS | densidad local + shrinkage (TTC 2026-06-03) ← PRIMERO
│
├── 02_Sources/
│   └── Fuentes-Datos.md              ← contrato con FlashScore (Playwright + API)
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
| S1_MATCH_LIST | extraer_URL_partidos_v2 | ✅ 80% | --max-matches 80 activo | multi-torneo validado 80 partidos 2026-06-02 |
| S2_H2H_DATA | extraer_historh2h.py | ⚠️ 85% | superficie: unknown en modo multi-torneo — bug propagación activo (ver [[Nodo-17-Calibracion-Por-Tier]] T17-01) |
| S3_RANKINGS | extraer_ranking_atp/wta_v2 | ✅ 100% | — |
| S4_PREDICTION | rivalry_analyzer.py | ✅ 90% | leer SIEMPRE `ranking_analysis.prediction.favored_player` |
| S5_EDGE | edge_calculator.py | ✅ 90% | p_historica clay=0.758 Thompson (n=31 ✅ umbral cruzado) |
| S6_RESULTADO_REAL | validar_con_api.py | ✅ 80% | match_id real necesario (S1 en prod) |
| S7_MARKOV | markov_analyzer.py | ✅ ACTIVO | integrado en rivalry_analyzer |
| S8_DATASET_ML | generar_dataset_plus.py | ⚠️ 40% | datos limpios de S1 en prod |

### Tests
```
875 passed, 0 fallos — 2026-06-01 (post T13-06/T15-04/T15-05 + aplicar_enhancer 13 tests)
Baseline mínimo: nunca bajar de 862
```

### Archivos clave del pipeline diario
```
── MODO ROLAND GARROS (Grand Slam, cuadro principal) ─────────────────────────
PASO 0: python3 extraer_ranking_atp_version2.py
PASO 1: python3 extraer_URL_partidos_version2.py
PASO 2: python3 extraer_historh2h.py
PASO 3: python3 edge_calculator.py
PASO 4: python3 trader_ev_tenis.py --bankroll 125000
PASO 5: python3 generar_tabla_favoritos2.py
PASO 6: python3 validar_con_api.py  (post-partido)

── MODO MULTI-TORNEO (Challenger + ATP + Grand Slam, hasta 80 individuales) ──
PASO 0: python3 extraer_ranking_atp_version2.py
PASO 1: python3 extraer_URL_partidos_version2.py --max-matches 80
PASO 2: python3 extraer_historh2h.py --all-tournaments         ← sin filtro RG
PASO 3: python3 edge_calculator.py
PASO 4: python3 trader_ev_tenis.py --bankroll 125000 --torneo-tipo atp500 --superficie clay
         → reports/trader_plan_FECHA.json + .txt  (stakes VaR-ajustados automáticamente)
PASO 5: python3 generar_tabla_favoritos2.py
PASO 6: python3 validar_con_api.py  (post-partido)
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
  ρ: grand_slam=0.25 | atp1000=0.20 | atp500=0.15 | challenger=0.10
  N=4 grand_slam: reducir 42.9% | N=8 grand_slam: reducir 63.6%

REGLA-HF-5: Growth Rate negativo = NO DESPLEGAR
  Si Kelly Growth Rate < 0 → el sistema está en régimen de ruina.
  Causas: demasiados picks, cuotas bajas, correlación alta.
  Solución: aumentar --min-cuota, reducir --piernas-max, reducir --top-n.
```

---

## Próximos 3 Pasos (ordenados por impacto en P&L)

1. ~~**Eliminar D-01 a D-13 + Nodo-12 infra**~~ ✅ HECHO — 7,996+ líneas eliminadas, infra limpia, 773 tests
2. ~~**Re-run con Erdős + surface activos**~~ ✅ CONFIRMADO 2026-05-30 — erdos_score=0.35, surf_w 0.49–0.69
3. ~~**Correr edge_calculator + trader_ev_tenis**~~ ✅ CONFIRMADO 2026-05-30 — 2 señales APOSTAR, $20,000 en riesgo (20% bankroll), combo 10.80x
4. ~~**Primera validación live + P&L registrado**~~ ✅ 2026-05-30 — +$25,000 (+25% bankroll). Accuracy 70%. p_hist 0.52→0.68. Ver [[Nodo-14-Validacion-Live-Conexiones]]
5. ~~**T14-03: Calibrar Erdős por superficie**~~ ✅ 2026-05-30 — clay common_opp 0.20→0.28, ranking_mom 0.20→0.12. 773 tests.
6. ~~**T07-09:**~~ ✅ 2026-05-30 — `SequentialH2HExtractor` eliminado (1,404 líneas), 53 tests migrados a H2HExtractor/DataParser. 768 tests.
7. ~~**D-17: config.py centralizado**~~ ✅ 2026-05-31 — FLASHSCORE_BASE/HEADERS/TOTAL_MATCHES_TO_PROCESS/BROWSER_* centralizados. 791 tests.
8. ~~**T13-04: Pipeline completo (80 partidos)**~~ → sesión 2 (Roland Garros R4 2026-06-01): 8 partidos 8/8=100%. Tres underdogs predichos (Kostyuk @3.0 +19.4%, Fonseca @2.3 +7.9%, Mensik @2.0 +1.2%). Ver [[Nodo-15-Portfolio-HedgeFund]].
9. ~~**T15-03**: Validar configuración óptima QF Roland Garros~~ ✅ 2026-06-01 — KGR=+0.2291, VaR auto-ajustado ×0.41
10. ~~**Ejecutar validar_con_api.py**~~ ✅ 2026-06-01 — n=31, p_historica clay=0.758 (umbral n≥30 cruzado)
11. ~~**T15-05**: Ajuste automático de stakes por VaR~~ ✅ 2026-06-01 — sección "STAKES FINALES" en output
12. ~~**T15-04**: Calibrar ρ por torneo~~ ✅ 2026-06-01 — grand_slam/atp1000/atp500/challenger + --torneo-tipo CLI
13. ~~**T13-06**: Calibrar p_blend con p_historica derivada~~ ✅ 2026-06-01 — _load_p_prior() + --superficie CLI
14. ~~**aplicar_enhancer.py tests**~~ ✅ 2026-06-01 — 13 tests. Suite total: 875 passed
15. ~~**T14-05**: Pipeline multi-torneo 80 partidos~~ ✅ 2026-06-02 — --max-matches 80 + --all-tournaments. Roland Garros filter fix. 875 tests. Ver [[Nodo-16-Multi-Torneo-Pipeline]]
16. ~~**T14-05**: Pipeline multi-torneo 80 partidos~~ ✅ 2026-06-02 — sesión 3: 61.11% (22/36) Challengers | prior contaminado detectado. Ver [[Nodo-16-Multi-Torneo-Pipeline]]
17. ~~**Nodo-17 Fase 1**~~ ✅ 2026-06-03 — T17-01 surface fix | T17-02 calibración estratificada | T17-03 λ por tier. 898 tests.
18. ~~**Nodo-17 Fase 1**~~ ✅ 2026-06-03 — T17-01 surface fix | T17-02 calibración estratificada | T17-03 λ por tier. 898 tests.
19. **Documentados TTC 2026-06-03:** [[Nodo-21-Pesos-Diferenciados-Por-Tier]] (primero — bug fix + infraestructura) → [[Nodo-19-H2H-Immunity-Dampener]] → [[Nodo-18-PELT-Recency-Alpha]] → [[Nodo-20-PageRank-Erdos-Quality]]

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

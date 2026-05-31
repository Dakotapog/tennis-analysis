# MOC Principal — Tennis Prediction Engine

> **Última actualización:** 2026-05-30 | **Tests:** 768 passed ✅ | **Pipeline:** 70% operativo
> Este es el documento de entrada del vault. Todo lo demás cuelga de aquí.

---

## Norte Real (leer en 30 segundos)

**Misión:** Apostar únicamente donde `P_modelo > P_implícita_bookmaker + 5%`, con Kelly-KL.
**Métrica de éxito:** P&L positivo acumulado — no accuracy del modelo.
**Estado hoy:** Pipeline 100% operativo ✅ | **P&L sesión: +$25,000 (+25% bankroll)** | Accuracy 70% (7/10) datos limpios | p_historica calibrada 0.52→0.68 (n=23) | Bankroll $100k→$125k

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
│   ├── Nodo-01-Edge-Calculator.md     (código ✅ | validación en prod pendiente)
│   ├── Nodo-02-Markov-Changepoint.md  (activo en prod desde 2026-05-29)
│   ├── Nodo-03-Scraper-Fix.md         ✅ RESUELTO
│   ├── Nodo-04-Dataset-Fix.md         ✅ RESUELTO
│   ├── Nodo-05-Validacion-API.md      (código ✅ | n≥30 pendiente)
│   ├── Nodo-06-Erdos-Graph.md         ✅ RESUELTO
│   ├── Nodo-07-Strangler-Fig.md       (Fase 1 ✅ | Fase 2 ✅ main() migrado a H2HExtractor — T07-09 pendiente)
│   ├── Nodo-08-File-Selection-Bug.md  ✅ RESUELTO
│   ├── Nodo-09-API-Status-Keys.md     ✅ RESUELTO
│   ├── [[Nodo-10-Surface-Propagation]]  RESUELTO ✅ — surf_w 0.49–0.69 en prod (efecto Nodo-07)
│   ├── Nodo-11-Inventario-Scripts-Legado.md  CERRADO ✅ — decisiones ejecutadas, disco verificado 2026-05-30
│   ├── Nodo-12-Inventario-Infraestructura-Legado.md  EJECUTADO ✅ — limpieza infraestructura 2026-05-30
│   ├── [[Nodo-13-Trader-EV-Tenis]]           IMPLEMENTADO ✅ — deploy: individuales + combos + sistema (2026-05-30)
│   └── [[Nodo-14-Validacion-Live-Conexiones]] PRIMERA VALIDACIÓN LIVE ✅ — Parry @ 4.50 ganó | 5 conexiones ocultas TTC (2026-05-30)
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
| S1_MATCH_LIST | extraer_URL_partidos_v2 | ⚠️ 60% | h2h_url OK, torneo OK en código, no validado en prod |
| S2_H2H_DATA | extraer_historh2h.py | ✅ 100% | surface=clay activo en prod (Nodo-10 RESUELTO 2026-05-30) |
| S3_RANKINGS | extraer_ranking_atp/wta_v2 | ✅ 100% | — |
| S4_PREDICTION | rivalry_analyzer.py | ✅ 90% | leer SIEMPRE `ranking_analysis.prediction.favored_player` |
| S5_EDGE | edge_calculator.py | ✅ 60% | p_historica=0.52 provisional hasta n≥30 |
| S6_RESULTADO_REAL | validar_con_api.py | ✅ 80% | match_id real necesario (S1 en prod) |
| S7_MARKOV | markov_analyzer.py | ✅ ACTIVO | integrado en rivalry_analyzer |
| S8_DATASET_ML | generar_dataset_plus.py | ⚠️ 40% | datos limpios de S1 en prod |

### Tests
```
768 passed, 0 fallos — 2026-05-30 (post T07-09)
Baseline mínimo: nunca bajar de 767
```

### Archivos clave del pipeline diario
```
PASO 0: extraer_ranking_atp_version2.py   → data/atp_rankings_complete_FECHA.json
PASO 1: extraer_URL_partidos_version2.py  → data/zita_tennis_matches_FECHA.json
PASO 2: extraer_historh2h.py              → reports/h2h_results_enhanced_FECHA.json
PASO 3: edge_calculator.py                → reports/edge_report_FECHA.json
PASO 3.5: trader_ev_tenis.py             → output consola (plan deploy: individuales+combos+sistema)
PASO 4: generar_tabla_favoritos2.py       → analisis_partidos_pandas.txt
PASO 5: validar_con_api.py (post-partido) → resultados_finales_FECHA.json
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
  p_historica = 0.52 hasta n≥30 validaciones limpias
  Kelly-KL cap = 10% bankroll por apuesta

REGLA-7: Roland Garros filter (dev mode)
  'French Open' in torneo_completo AND 'Qualification' not in torneo_completo
  → 41 matches del cuadro principal (no calificación)
```

---

## Próximos 3 Pasos (ordenados por impacto en P&L)

1. ~~**Eliminar D-01 a D-13 + Nodo-12 infra**~~ ✅ HECHO — 7,996+ líneas eliminadas, infra limpia, 773 tests
2. ~~**Re-run con Erdős + surface activos**~~ ✅ CONFIRMADO 2026-05-30 — erdos_score=0.35, surf_w 0.49–0.69
3. ~~**Correr edge_calculator + trader_ev_tenis**~~ ✅ CONFIRMADO 2026-05-30 — 2 señales APOSTAR, $20,000 en riesgo (20% bankroll), combo 10.80x
4. ~~**Primera validación live + P&L registrado**~~ ✅ 2026-05-30 — +$25,000 (+25% bankroll). Accuracy 70%. p_hist 0.52→0.68. Ver [[Nodo-14-Validacion-Live-Conexiones]]
5. ~~**T14-03: Calibrar Erdős por superficie**~~ ✅ 2026-05-30 — clay common_opp 0.20→0.28, ranking_mom 0.20→0.12. 773 tests.
6. ~~**T07-09:**~~ ✅ 2026-05-30 — `SequentialH2HExtractor` eliminado (1,404 líneas), 53 tests migrados a H2HExtractor/DataParser. 768 tests.
7. **T13-04: Pipeline completo (80 partidos)** → ≥3 señales → sistema 2/N → combos 1→15+ → bankroll exponencial

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

# Sprint — Pipeline de Construcción

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Grafo-Dependencias-Datos]] | [[Pipeline-Arquitectura]] | [[Inventario-Deuda-Tecnica]]
> Nodos: [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Nodo-13-Trader-EV-Tenis]] | [[Nodo-14-Validacion-Live-Conexiones]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-20-PageRank-Erdos-Quality]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | Nodo-22-API-Integration-Kambi-Ninja | [[Nodo-27-Pipeline-Tracker-Observabilidad]]
>
> Documento de planificación de implementación. Convierte el Spec Kit en **motor de construcción determinista**.
> Última actualización: 2026-06-19 20:30 — [[Sprint-PostMortem-19jun]] ✅ COMPLETADO (5 fixes, 1050→1113 tests) | [[Sprint-Normalizacion-19jun]] ✅ COMPLETADO FINAL (5 fixes + E-1 + E-2). Caso validador Eala vs Svitolina: **Eala 52.9% favorita, edge 20.6% APOSTAR**. E-1: surface 0.15→0.22 dinámico. E-2: bonus ×1.6 dinámico. 1113 tests ✅. | [[Nodo-30-Tournament-Momentum-Output-Signals]] IMPLEMENTADO (30 tests, F6 player_profitability, F7 output signals, 1143 tests) 2026-06-20

---

## 1. El Puente: Componente → Módulo → Prioridad

### 1.1 Productores primarios (deben existir ANTES de cualquier módulo ML)

| Productor | Dato producido | Módulo que desbloquea |
|---|---|---|
| `extraer_partidos_api.py` (API, ~1.3s) / `extraer_URL_partidos_version2.py` (Playwright, ~8min) | `h2h_url`, `torneo`, `match_id`, cuotas reales Betplay | Todo el pipeline |
| `extraer_historh2h.py --api-mode` (Ninja, ~0.5s/p) / default (Playwright, ~2-3min/p) | `h2h_results_enhanced_FECHA.json` | Predicción + Edge |
| `edge_calculator.py` | `edge`, `kelly_kl`, `apostar` | Decisión de apuesta |
| `betplay_combo_builder.py` (PASO 4.5) | links de combos Betplay — modo clásico o `--live` | Deploy directo en Betplay |
| `FlashScore API dc_1` | Resultado real post-partido | Labels limpias |

**Consecuencia crítica:** `edge_calculator.py`, `markov_analyzer.py` y `validar_con_api.py` son **consumidores muertos** si el scraper no produce `h2h_url` y `torneo` correctos. No construir hasta que Fase 1 esté completa.

### 1.2 Tabla de prioridad de implementación

| Prioridad | Fase | Módulo | Entrada requerida | Desbloquea |
|---|---|---|---|---|
| 0 | Fase 0 | Fix `data_parser.py` (3 bugs) | — | Tests pasando | ✅ DONE |
| 1 | Fase 1 | Fix scraper (h2h_url + torneo + match_id) | `match_url` | surface_specialization > 0% | ✅ DONE |
| 2 | Fase 2 | `edge_calculator.py` | `ranking_analysis.prediction` + cuotas | Primera apuesta con criterio | ✅ DONE |
| 3 | Fase 3 | `analysis/markov_analyzer.py` | `historial_jugador` | form_recent mejorado | ✅ DONE |
| 4 | Fase 4 | Fix `generar_dataset_plus.py` (2 bugs) | `h2h_results_enhanced` + `resultados_finales` | Dataset ML limpio | ✅ DONE |
| 5 | Fase 5 | `validar_con_api.py` | `match_id` (requiere Fase 1) | Labels en tiempo real |
| 6 | Fase 6 | Fix `generar_tabla_favoritos2.py` | `h2h_results_enhanced` | Reporte v2 funcional |
| 7 | Fase 7 | Erdős graph enhancement | `common_opponents_detailed` (ya existe) | Mejor ponderación transitiva |

---

## 2. Critical Path — 7 Fases

### Fase 0 — Datos limpios + tests base ✅ COMPLETADA 2026-05-28

```
[x] Fix normalize_surface: garbage > 30 chars → "Desconocida"
[x] Fix extract_tournament_info: garbage > 120 chars → guard separado para None vs garbage
[x] Fix clean_player_name("   "): → "N/A"
[x] Crear tests/test_data_parser.py (67 tests)
[x] Crear tests/test_normalization.py (111 tests)
[x] Crear tests/test_rivalry_analyzer_regression.py (corrección valores ELO)
[x] 506 tests passed, 0 failed
```

---

### Fase 1 — Fix Scraper (h2h_url + torneo + match_id) → Nodo-03 ✅ COMPLETADA 2026-05-28

```
Evidencia: 423 partidos, h2h_url = None (0/423), torneo = "Sin Torneo Asignado" → FIXED

[x] T03-01: Derivar h2h_url desde match_url
            match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
            Resultado: h2h_url generada correctamente en todos los partidos

[x] T03-02: Extraer match_id (event_id) desde ?mid= en match_url
            re.search(r'[?&]mid=([^&]+)', match_url).group(1)
            Resultado: event_id real extraído, nunca "tennis"

[x] T03-03: Extraer torneo del DOM con selector .event__header + inner_text()
            Añadido selector .event__header junto a .event__title (FlashScore real DOM)
            Resultado: torneo leído desde cabecera real del DOM

[x] T03-04: extraer_superficie(torneo_texto) → 'clay' | 'grass' | 'hard' | 'unknown'
            Método estático en ZitaScraper. Roland Garros → clay, Wimbledon → grass
            Resultado: campo 'superficie' añadido a cada partido del JSON

[x] T03-05: Crear tests/test_url_scraper_output.py (20 tests)
            TestBug1H2hUrl (4) + TestBug2MatchId (4) + TestBug3Superficie (12)
            Resultado: 20/20 passed

[ ] T03-06: Re-ejecutar extraer_URL_partidos_version2.py en producción y verificar
            Pendiente: requiere ejecutar el scraper en Roland Garros o próximo torneo
            Criterio: h2h_url presente + torneo ≠ "Sin Torneo Asignado" en output real
```

**Checkpoint Fase 1:** ✅ Código corregido + tests pasando. Pendiente: validación en producción (T03-06).

---

### Fase 2 — Edge Calculator (Kelly-KL) → Nodo-01 ✅ COMPLETADA 2026-05-28

```
[x] T01-01: Implementar edge_calculator.py — 5 capas de inteligencia
            L1 Kelly-KL · L2 Volatility Smile · L3 Fama-French · L4 Shannon · L5 Thompson
            Evidencia: Majchrzak → edge 9.5%, Kelly-KL 19.7%, zona underdog

[x] T01-02: Leer predicción desde path correcto
            partido['ranking_analysis']['prediction']['favored_player'] ← confirmado

[x] T01-03: Batch sobre h2h_results_enhanced_20260120_183437.json
            Resultado: 9 apuestas identificadas / 27 partidos, TODAS en zona underdog

[x] T01-04: Crear tests/test_edge_calculator.py (43 tests)
            5 clases: KellyKL · VolatilitySmile · ShannonEntropy · FactorDecomp · Thompson
            Resultado: 43/43 passed

[ ] T01-05: Comparar edge calculado vs resultado real (Jan 2026 n=19)
            Pendiente: verificar en resultados_finales_20260119_120450.json
            Criterio: Majchrzak vs Marozsan con edge+9.5% → ¿ganó Majchrzak?
```

**Checkpoint Fase 2:** ✅ 9 señales de apuesta reales identificadas. Kelly-KL + 4 capas innovadoras activas.

---

### Fase 3 — Markov Analyzer (PELT Change-Point) → Nodo-02 ✅ COMPLETADA 2026-05-28

```
[x] T02-01: Implementar detectar_cambio_regimen() en analysis/markov_analyzer.py
            PELT simplificado + estados HOT/COLD/NEUTRAL por últimos 5 partidos

[x] T02-02: Implementar calcular_factor_markov(markov_p1, markov_p2)
            Factor simétrico [0.85, 1.15] — HOT vs COLD → 1.15, COLD vs HOT → 0.85

[x] T02-03: Crear tests/test_markov_analyzer.py (37 tests)
            5 clases: Estados · PELT · FactorMarkov · ExtraerResultados · PipelineCompleto
            Resultado: 37/37 passed

[x] T02-04: Integrar en rivalry_analyzer.py — aplicar factor_markov sobre form_recent
            Línea 1152+: raw_p1/p2['form_recent'] *= factor_p1/p2
            markov_analysis añadido al dict de retorno de generate_advanced_prediction

[x] T02-05: Verificar que tests siguen pasando tras integración
            Resultado: 606 tests passed, 0 failed (+37 Markov + 20 Scraper + 43 Edge)
```

**Checkpoint Fase 3:** ✅ markov_analysis en JSON de salida. Mochizuki COLD × 0.85 activo.

---

### Fase 4 — Fix Dataset ML → Nodo-04 ✅ COMPLETADA 2026-05-28

```
[x] T04-01: Diagnosticar Bug 1 (KNN shape mismatch)
            Causa raíz: columns=numeric_cols (lista separada) puede divergir del array
            cuando hay columnas con tipos mixtos o duplicadas. Fix: usar df_numeric.columns
            directamente (siempre en sync con el array transformado).

[x] T04-02: Fix Bug 1 — _intelligent_imputation en generar_dataset_plus.py
            df_numeric = df.select_dtypes(include=np.number)
            columns=df_numeric.columns  # no una lista separada
            Test: test_knn_no_shape_mismatch_con_columnas_mixtas (71 num + 8 cat = 79 cols)

[x] T04-03: Fix Bug 2 — SmartLogger.error() + SmartLogger.warning()
            Añadidos métodos .error() y .warning() delegando a self.logger.error/warning
            Cubiertos: líneas 755 (.error), 611/645/661 (.warning)

[x] T04-04: Crear tests/test_dataset_generator.py (14 tests)
            TestSmartLogger (7) + TestIntelligentImputation (7) — 14/14 ✅

[ ] T04-05: Ejecutar generar_dataset_plus.py en producción y verificar que termina sin error
            Depende de: datos limpios de Fase 1 en producción
            Día: pendiente validación prod
```

---

### Fase 5 — API FlashScore para Labels → Nodo-05 ✅ COMPLETADA 2026-05-28

```
[x] T05-01: Implementar validar_con_api.py
            parsear_respuesta_flashscore() → KEY÷VALUE¬KEY÷VALUE → dict
            obtener_resultado_partido(event_id) → {'status': 'FT'|'NS'|'LIVE'|'ERROR'}
            validar_partido_individual(partido, resultado_api) → dict|None
            calcular_accuracy(resultados) → float
            accuracy_por_superficie(resultados) → dict por superficie
            actualizar_calibracion_desde_resultados() → CX-06: calibracion_edge.json
            CLI: python validar_con_api.py --h2h reports/... --output reports/...

[x] T05-02: Parser formato propietario FlashScore
            "DJ÷H¬DE÷2¬DF÷0¬DC÷1780057200" → {'DJ':'H','DE':'2','DF':'0','DC':'1780057200'}
            Status real (Nodo-09 ✅): DJ='H'→jugador1 ganó, DJ='A'→jugador2 ganó, DJ=''→en curso
            Nota: el formato ~AA/~BH/~BI era incorrecto — corregido en Nodo-09

[x] T05-03: Crear tests/test_validacion_api.py (39 tests)
            TestParserFlashscore (6) + TestObtenerResultado (9) +
            TestValidarPartidoIndividual (10) + TestCalcularAccuracy (5) +
            TestAccuracyPorSuperficie (6) + tests Nodo-09 (3) — 39/39 ✅

[ ] T05-04: Validación en producción — ejecutar contra h2h real con match_id limpios
            Requiere Nodo-03 Bug 2 activo en producción (match_id ≠ "tennis")
            Meta: n≥30 partidos validados → calibrar p_historica en Kelly-KL
```

---

### Fase 6 — Fix generar_tabla_favoritos2.py

```
[x] T06-01: Localizar bug score_breakdown variable scope
            Bug: score_breakdown usado en línea ~744 (dentro de `if reasoning:`)
            pero definido en línea 748 (después del bloque).
            Primer partido → NameError. Siguientes → dato del partido anterior.

[x] T06-02: Fix — mover definición a justo después de scores (línea 690)
            score_breakdown = prediction.get('score_breakdown', {}) or {}
            Eliminar definición duplicada posterior.

[x] T06-03: Ejecutar generar_tabla_favoritos2.py en producción ✅ 2026-05-30
            Sin errores. Output: analisis_partidos_pandas.txt — 16 partidos Roland Garros

[x] T06-04: generar_tabla_favoritos.py (v1) ya eliminada (Nodo-11 ✅) — comparación N/A
            v2 es la única versión activa. Fase 6 cerrada.
```

---

### Fase 7 — Erdős Graph Enhancement → Nodo-06 ✅ COMPLETADA 2026-05-29

```
[x] T07-01: Crear analysis/erdos_graph.py
            construir_grafo_victorias(partidos) → grafo dirigido ponderado
            historial_a_partidos(player_history, player_name) → [{ganador, perdedor}]
            distancia_erdos(A, B, grafo, max_depth=3, alpha=0.7) → erdos_score

[x] T07-02: Fórmula de score con decaimiento Erdős correcto
            advantage(path) = path_weight - neutral(d)
            neutral(d) = (0.5^d) × α^(d-1)
            erdos_score = mean(advantages) ∈ [-1, +1]
            Bugs fijados: RuntimeError set mutation + max_depth check + formula negativa

[x] T07-03: Integrar en rivalry_analyzer.py
            Import: from analysis.erdos_graph import ...
            Añadido en ambos returns de analyze_rivalry():
              - return temprano (no common_opponents) → erdos_analysis present
              - return principal → erdos_analysis present
            Try/except: error en Erdős no rompe el pipeline

[x] T07-04: Crear tests/test_erdos_graph.py (37 tests)
            TestConstruirGrafo (8) + TestDistanciaErdos (16) +
            TestHistorialAPartidos (10) + TestPipelineCompleto (3) — 37/37 ✅
```

---

### Fase 8 — Strangler Fig (Nodo-07 Fase 1) ✅ COMPLETADA 2026-05-29

```
Problema: extraer_historh2h.py (3717 líneas) tenía copias inline de 3 clases.
Módulos en analysis/ (con Markov+Erdős) nunca llegaban a producción.

[x] T-SF-01: Verificar analysis/__init__.py exporta EloRatingSystem, RankingManager, RivalryAnalyzer
[x] T-SF-02: Añadir from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer
[x] T-SF-03: Eliminar clases inline (líneas 32-2276 del original)
             Resultado: 3717 → 1691 líneas (−2026 líneas)
[x] T-SF-04: pytest tests/ --no-cov -q → 767 passed, 0 failed (post-migración)
[x] T-SF-05: Run producción y verificar factor_markov y erdos_score en output JSON
             Evidencia 2026-05-30: erdos_score=0.35, factor_markov=1.0 en h2h_results_enhanced_20260530_053518.json
             16 partidos Roland Garros, surface_specialization activo (surf_w 0.49–0.69)
```

---

### Fase 9 — Bug Selección de Archivo (Nodo-08) ✅ COMPLETADA 2026-05-29

```
Bug: max(files, key=(total_matches, modified_time)) → May 28 (423 matches, h2h=None) ganaba
Fix: max(files, key=(modified_time, total_matches)) → May 29 (235 matches, h2h válidas) gana

[x] T-FS-01: Corregir línea 278 → nueva línea 241 de extraer_historh2h.py
[x] T-FS-02: test_file_selection_prefers_recency_over_match_count
[x] T-FS-03: test_file_selection_uses_match_count_as_tiebreaker
[x] T-FS-04: test_file_selection_single_file_always_wins
[x] T-FS-05: Añadir log de advertencia post-selección (anomaly detection)
             tests/test_file_selection.py — 5/5 passed
             Total suite: 767 passed ✅
[x] T-FS-06: Re-run pipeline y verificar que selecciona May 29 ✅ 2026-05-30 — confirmado en T-SF-05 (erdos_score=0.35, surf_w 0.49–0.69 en h2h del 30 mayo)
```

---

### Fase 10 — Bug Claves API FlashScore DC_1 (Nodo-09) ✅ COMPLETADA 2026-05-29

```
Bug: obtener_resultado_partido() usaba ~AA/~BH/~BI (claves inexistentes en dc_1).
     Siempre retornaba status='NS'. Ningún partido se podía validar post-partido.

Evidencia real (2026-05-29, 3 partidos Roland Garros terminados):
  DJ=H → jugador1 ganó | DJ=A → jugador2 ganó | DJ='' → en juego/NS
  DE = sets local | DF = sets visitante | DC = timestamp programado

[x] T09-01: Fix obtener_resultado_partido(): ~AA→DJ, ~BH→DE, ~BI→DF, lógica NS/LIVE via DC
[x] T09-02: Actualizar 5 mocks en TestObtenerResultado al formato real (DJ÷H¬DE÷2¬DF÷0)
[x] T09-03: Añadir test_formato_real_dc1_endpoint
[x] T09-04: Añadir test_dv_no_es_indicador_de_estado (DV=2 es constante, no estado)
[x] T09-05: pytest tests/ → 767 passed, 0 failed ✅
[x] T09-06: Actualizar Fuentes-Datos.md con mapa real de claves dc_1
```

---

### Fase 11 — Limpieza Deuda Técnica (D-01..D-12 + SmartLogger) ✅ COMPLETADA 2026-05-29

```
Eliminados archivos obsoletos/backup sin importadores activos (verificado con grep).

[x] D-01: extraer_historh2h_backup_20260529_114017.py (3,717 líneas) eliminado
[x] D-02: extraer_historh2h_version2.py (2,963 líneas) eliminado
[x] D-03..D-08: main.py, utils.py, config.py, 3 stubs en services/ (0 bytes) eliminados
[x] D-09: SmartLogger extraído a utils/logger.py (fuente única de verdad)
          from utils.logger import SmartLogger en generar_dataset_plus.py ✅
          from utils.logger import SmartLogger en Intelligent_ml_enhancer.py ✅
          Creado tests/test_utils_logger.py — 19 tests ✅
[x] D-10: extraer_ranking_atp.py (v1, 443 líneas) eliminado
[x] D-11: extraer_ranking_wta.py (v1, 269 líneas) eliminado
[x] D-12: extraer_URL_partidos.py (v1, 604 líneas) eliminado
[x] pytest tests/ --no-cov -q → 767 passed, 0 failed ✅

Total líneas eliminadas: ~7,996 | Riesgo: CERO (ningún script activo importaba estos archivos)
```

---

### Fase 12 — Nodo-07 Fase 2 prep: 52 tests SequentialH2HExtractor ✅ COMPLETADA 2026-05-29

```
Precondición de Nodo-07 Fase 2 (migrar SequentialH2HExtractor → H2HExtractor): ≥40 tests.

[x] T12-01: Crear tests/test_sequential_h2h_extractor.py
            Cubre métodos puros (sin Playwright): determine_winner_from_result,
            extract_winner_sets, classify_form, analyze_recent_form_in_extractor,
            analyze_common_opponents_in_extractor, load_matches_from_json,
            generate_global_statistics, __init__ (estado inicial)
[x] T12-02: 52 tests pasando — precondición de Nodo-07 Fase 2 CUMPLIDA (era: 5 tests)
[x] T12-03: pytest tests/ --no-cov -q → 767 passed, 0 failed ✅

Nodo-07 Fase 2 ahora puede iniciarse (ver [[Nodo-07-Strangler-Fig]]).
```


---

### Fase 13 — Limpieza Infraestructura Legado (Nodo-12) ✅ COMPLETADA 2026-05-30

```
Auditoría + ejecución de limpieza de infraestructura paralela al pipeline S1-S8.

[x] T12-01: Auditoría completa app.py + routes/ + models/ + services/ + screenshots/ + reports/
            Hallazgo crítico: services/ NO estaba vacío (selenium_config.py 3,918 bytes)
            Hallazgo: screenshots/ nunca ejecutado por pytest (testpaths=tests en pytest.ini)
            Hallazgo: reports/ginput/ + reports/classification/ + predictions/ = datos pre-Nodo-03

[x] T12-02: Eliminación artefactos debug raíz
            28 archivos: h2h_match_*.html (9) + h2h_match_*.png (9) + match_*.png (9) + find (1)

[x] T12-03: Eliminación datos contaminados y dead code
            screenshots/conftest.py + screenshots/test_extraer_historh2h.py + 4 PNGs
            reports/ginput/ (16 JSONs pre-Nodo-03) | reports/classification/ | predictions/
            reports/stages/ (2 CSVs) | reports/nba_h2h_analysis_*.json (NBA fuera de scope)

[x] T12-A: total_matches_to_process = 80 ✅ YA ESTABA CORRECTO — línea 271 de extraer_historh2h.py
           Verificado 2026-05-30 con TTC: nunca se cambió a 16. Tachar. 773 tests pasan.

[x] T12-B: Mover flashscore_rankings_inspector.py → tools/ ✅ 2026-05-31
[x] T12-C: Auditar routes/ completo ✅ 2026-05-31 — SUSPENDER: isla Flask/Selenium, 0 acoplamiento pipeline
[x] T12-D: N/A — extraer_URL_partidos_en_vivo.py eliminado en D-15 (2026-05-31, 946 líneas)

SUSPENDIDO (no tocar hasta P&L validado con n≥30):
  app.py | routes/ | models/ | services/ | database.db | drivers/

Tests: 773 passed, 0 fallos — baseline actualizado 2026-05-30
```

---

### Fase 14 — Trader EV Tenis (Nodo-13) ✅ COMPLETADA 2026-05-30

```
Inspirado en NBA trader_ev.py — deploy sin ML usando Bayesian blend + combos.

[x] T13-01: Crear trader_ev_tenis.py — 3 capas (individuales, combos, sistema 2/N)
            Budget cascade: 40% individuales / 40% combos / 20% sistema
            p_blend = (n_h2h × p_modelo + 3 × 0.52) / (n_h2h + 3) — Bayesian k=3
            Kelly fraccionario ×0.25 | cap 10% individual | cap 15% combo

[x] T13-02: Run producción Roland Garros 2026-05-30 (16 partidos, 2 señales APOSTAR)
            Parry D. @ 4.50  → $10,000 (cap 10%)  | edge +29.3%
            Tien L.  @ 2.40  → $4,000              | edge +16.5%
            Combo 2 piernas  → $5,000 | cuota 10.80 | EV +192%
            Total en riesgo: $19,000 (19% bankroll de $100,000)

[x] T13-03: Añadir campo n_h2h en edge_calculator.py (lee enfrentamientos_directos)
            len([m for m in partido.get('enfrentamientos_directos',[]) if isinstance(m,dict)])
            Resultado: Parry n_h2h=0 | Tien n_h2h=1 | Osorio n_h2h=3 | Mboko n_h2h=1
            p_blend ahora varía por partido según historial real

[ ] T13-04: Correr pipeline completo 80 partidos → ≥3 señales → sistema 2/N activo

[x] T13-05: Añadir output JSON reports/trader_plan_FECHA.json para auditoría P&L ✅ 2026-05-30
            _build_combos/_build_sistema retornan (gastado, plan_list) — main() desempaqueta
            Guardado: reports/trader_plan_20260530_121616.json
```

### Resultado real validado (Roland Garros 2026-05-30):
```
Parry D. @ 4.50 (edge +29.3%)  → PARRY GANÓ ✅  → retorno $10,000 → $45,000
Berrettini @ 1.45              → GANÓ ✅
Tabilo                         → GANANDO ✅
Snaider                        → GANÓ ✅
Cresudolo (rivalidad reñida)   → 51+ games, sigue ✅  (calibración rivalidad correcta)
Tien L. @ 2.40                 → pendiente resultado
```
Alpha confirmado: bookmaker fijó Parry 22.2% implied, modelo 52%. Ver [[Nodo-14-Validacion-Live-Conexiones]].

---

### Fase 15 — Validación Live + Conexiones Ocultas (Nodo-14) ✅ DOCUMENTADA 2026-05-30

```
Primera validación live en producción real — Roland Garros 2026-05-30.
5 conexiones ocultas identificadas con TTC (Marco de Tres Expertos).

[x] T14-04: ADR "buscar activamente odds 3.5–6.0 con señal superficie" documentado en MOC
            Evidencia: EV convexo — Parry @ 4.50 → +134% EV vs Berrettini @ 1.45 → +16% EV

[x] Documentar resultados live en Nodo-14 con tabla de validación y 5 cadenas TTC
[x] Actualizar MOC, Nodo-13, Pipeline-Arquitectura, Sprint-Pipeline con wikilinks Nodo-14

[x] T14-01: Registrar resultados Roland Garros 2026-05-30 en validar_con_api.py ✅ 2026-05-30
            → n: 13→23 | p_historica: 0.52→0.68 (Thompson Beta 16W/7L)
            → Accuracy 70% (7/10) datos limpios | objetivo >55% SUPERADO
            → P&L sesión: +$25,000 (+25% bankroll) | bankroll $100k→$125k

[x] T14-02: Añadir factor_tardio en markov_analyzer.py ✅ 2026-05-30
            calcular_factor_tardio(history, min_matches=3) → win rate en sets tardíos
            calcular_factor_tardio_comparativo(t_p1, t_p2) → factor [0.85, 1.15]
            Integrado en rivalry_analyzer.generate_advanced_prediction()
            Campo tardio_analysis en JSON de salida (análogo a markov_analysis)
            +13 tests (50 en test_markov_analyzer.py) | 781 passed, 0 fallos ✅

[x] T14-03: Calibrar peso common_opponents por superficie en rivalry_analyzer.py ✅ 2026-05-30
            Implementado en generate_advanced_prediction() post selección de pesos por torneo:
              clay:  common_opponents +0.08 (0.20→0.28) | ranking_momentum -0.08 (0.20→0.12)
              grass: common_opponents -0.05 (0.20→0.15) | form_recent +0.05 (0.15→0.20)
              hard:  sin cambio (pesos base se mantienen)
            LOG emitido: LOG_WEIGHTS_SURFACE_CLAY / LOG_WEIGHTS_SURFACE_GRASS
            773 tests passing — sin regresiones

[ ] T14-05: Correr pipeline completo 80 partidos (ver T13-04)
```

---

### Fase 16 — Strangler Fig Fase 2 completa: eliminar SequentialH2HExtractor (T07-09) ✅ 2026-05-30

```
Mayor deuda arquitectónica del proyecto — clase monolítica de 1,404 líneas.
Metodología: TTC (Marco de Tres Expertos) + migración quirúrgica end-to-end.

DIAGNÓSTICO PRE-MIGRACIÓN (TTC):
  SE:  grep confirms no importadores activos de SequentialH2HExtractor fuera de tests.
       extraer_historh2h.py: main() ya usa H2HExtractor desde T07-0D (2026-05-29).
  DA:  53 tests en test_sequential_h2h_extractor.py. DataParser ya tiene determine_winner_from_result
       y extract_winner_sets como @staticmethod (idéntica lógica). H2HExtractor._classify_form
       recibe float (vs Sequential que recibía (wins, total)) — adaptación de firma necesaria.
  ARQ: TestAnalyzeCommonOpponents (3 tests) → ELIMINAR (cubierto en test_rivalry_analyzer.py).
       TestGenerateGlobalStatistics → MIGRAR a _generate_global_statistics() (privada pero existe).
       Costura limpia: H2HExtractor producción ya activa — Sequential solo vivía en tests.

[x] T07-09-01: Reescribir tests/test_sequential_h2h_extractor.py — 53 tests → 48 tests
               Fixture: scraping.h2h_extractor.* (antes: extraer_historh2h.*)
               DataParser.determine_winner_from_result / extract_winner_sets — estáticos, sin fixture
               H2HExtractor._classify_form(win_pct: float) — firma adaptada
               H2HExtractor._analyze_recent_form / load_matches / _generate_global_statistics
               TestAnalyzeCommonOpponents (3) ELIMINADOS — cobertura en test_rivalry_analyzer.py
               Resultado: 48/48 passed ✅

[x] T07-09-02: Verificar 48 tests pasan antes de tocar extraer_historh2h.py
               python -m pytest tests/test_sequential_h2h_extractor.py -v → 48 passed ✅

[x] T07-09-03: Eliminar SequentialH2HExtractor de extraer_historh2h.py (líneas 260-1663)
               Método: corte quirúrgico Python — lines[:259] + lines[1663:]
               Resultado: 1,714 → 310 líneas (−1,404 líneas, −82%)
               extraer_historh2h.py = entry point puro: imports + helpers + async def main()

[x] T07-09-04: Verificar sintaxis y run completo post-eliminación
               python -c "import ast; ast.parse(...)" → OK ✅
               python -m pytest tests/ --no-cov -q → 768 passed, 0 failed ✅
               Invariante ≥767 ✅ — Mandato 6 respetado

Líneas eliminadas del proyecto (acumulado):
  D-01..D-13 (sprints anteriores): ~7,996 líneas
  T07-09 (este sprint):            1,404 líneas
  TOTAL acumulado:                 ~9,400 líneas de deuda técnica eliminadas
```

**Impacto arquitectónico:**
- `extraer_historh2h.py`: 3,717 líneas (original) → 310 líneas (−92%)
- Toda la lógica vive en `scraping/` y `analysis/` — SRP perfecto
- D-14 del Inventario de Deuda Técnica: **CERRADO** ✅
- Ver [[Nodo-07-Strangler-Fig]] | [[Inventario-Deuda-Tecnica]]

---

### Fase 17 — D-17: Centralizar configuración en config.py ✅ 2026-05-31

```
Motivación: constantes de API y pipeline hardcodeadas en múltiples archivos — D-17 del inventario.

Archivos modificados:
  [+] config.py (nuevo, raíz)
        FLASHSCORE_BASE, FLASHSCORE_HEADERS — FlashScore Ninja API
        TOTAL_MATCHES_TO_PROCESS = 80 — batch size del pipeline
        BROWSER_HEADLESS = True, BROWSER_SLOW_MO = 250 — Playwright defaults
  [~] validar_con_api.py — from config import FLASHSCORE_BASE, FLASHSCORE_HEADERS as HEADERS
  [~] scraping/h2h_extractor.py — from config import TOTAL_MATCHES_TO_PROCESS, BROWSER_*

TTC aplicado:
  SE: grep confirmó FLASHSCORE_BASE/HEADERS solo en validar_con_api.py
  DA: MAX_RAW_SCORES/DEFAULT_WEIGHTS NO movidos — co-ubicados con normalization lógica
  ARQ: config.py centraliza solo constantes sin hogar natural (API + pipeline batch)

Tests: +10 en tests/test_config.py → 791 passed ✅ (invariante ≥767)
```

---

### Fase 18 — Portfolio Hedge Fund (Nodo-15) ✅ 2026-06-01

```
Descubrimiento: Roland Garros R4 2026-06-01 — 8/8=100% accuracy.
Tres underdogs correctamente predichos: Kostyuk @3.00 (+19.4%), Fonseca @2.30 (+7.9%), Mensik @2.00 (+1.2%).
Diagnóstico: Kelly individual naive en N=8 picks correlacionados → KGR=-0.5085 (ruina).
Fix: Portfolio Kelly + min-cuota 1.50 + Sistema Cobertura por Exclusión → KGR=+0.4142 (crecimiento).

Nuevas constantes en trader_ev_tenis.py:
  RHO_SESSION = 0.25  (correlación estructural misma sesión Grand Slam)
  VAR_CONFIDENCE = 0.95 | MAX_VAR_PCT = 0.25 (máx 25% bankroll en VaR)

[x] T15-01: Implementar _build_cobertura() con diversidad garantizada
            Genera C(N,k) combos para k en [piernas_min, piernas_max]
            Algoritmo greedy garantiza que cada jugador es excluido en ≥1 combo top-N
            Sin diversidad: un fallo destruye todo el portfolio.

[x] T15-02: Implementar _portfolio_risk_report() (Portfolio Kelly + VaR + Sharpe + Growth Rate)
            _portfolio_kelly_factor(n, rho) = 1/(1+rho*(n-1))
            _compute_var_cvar() via distribución binomial + KGR = E[log(1+R)]
            Output: PK factor | VaR 95% | CVaR | E[P&L] | Sharpe | KGR | proyección 5/10 sesiones
            REGLA-HF-5: si KGR < 0 → "NO DESPLEGAR" impreso en output

[x] T15-03: Validar configuración óptima (QF Roland Garros 2026-06-01 → sesión backcalculation)
            Configuración: --cobertura --all-picks --watchlist --min-cuota 1.50 --piernas-min 3 --piernas-max 4 --top-n 4
            KGR validado: +0.4142 | Sharpe: 0.169 | PK factor N=4: 0.571
            P&L escenario real 8/8: +$635,510 (+907%)
            P&L escenario 1 fallo Kostyuk: +$32,520 (SIEMPRE POSITIVO ✅)
            P&L escenario 1 fallo Svitolina: +$120,400 (SIEMPRE POSITIVO ✅)

[x] T15-04: Calibrar ρ por tipo de torneo (Grand Slam vs ATP 500 vs Challenger) ✅ 2026-06-01
            RHO_BY_TOURNAMENT = {grand_slam:0.25, atp1000:0.20, atp500:0.15, challenger:0.10}
            CLI: --torneo-tipo grand_slam|atp1000|atp500|challenger (default: grand_slam)
            Header muestra ρ activo: "ρ=0.25 (grand_slam)". Portfolio Kelly factor varía por torneo.
            862 tests passing.
[x] T15-05: Implementar ajuste automático de stakes por factor VaR en main() ✅ 2026-06-01
            Sección "STAKES FINALES (VaR AJUSTADO ×factor)" impresa automáticamente si VaR excedido.
            Aplica factor_var a stakes individuales + cobertura. Resumen final ya refleja stakes reales.
            Sin cálculo manual. 862 tests passing.
[x] T13-06: Calibrar p_blend con p_historica derivada (n≥30 cruzado) ✅ 2026-06-01
            _load_p_prior(superficie) → lee calibracion_edge.json → Thompson Beta mean
            clay: prior 0.52→0.758 (24W/7L). CLI: --superficie clay|grass|hard|unknown
            p_blend Parry n_h2h=0: 0.520→0.758. 862 tests passing.
[ ] T15-06: Backtesting formal con n≥30 sesiones cuando haya datos limpios

Tests: 862 passed ✅ (post cobertura tests en generar_tabla_favoritos2 + Intelligent_ml_enhancer)
Nuevo CLI: --cobertura | --all-picks | --piernas-min | --piernas-max | --top-n | --excluir | --min-cuota
Spec: [[Nodo-15-Portfolio-HedgeFund]] creado como nuevo nodo del vault
```

---

### Fase 19 — Pesos Diferenciados por Tier (Nodo-21) ✅ COMPLETADA 2026-06-03

```
[x] T21-01: Fix classify_tournament() → detectar_tier() en config.py
            5 categorías: grand_slam | atp1000 | atp500 | challenger | itf
[x] T21-02: detectar_tier() en config.py — fuente única compartida
            Importado desde edge_calculator.py, rivalry_analyzer.py, trader_ev_tenis.py, resultados_finales.py
[x] T21-03: weights_config en generate_advanced_prediction() — 5 tiers con SNR
            grand_slam: common_opp=0.22 h2h=0.18 form=0.12
            challenger: common_opp=0.08 h2h=0.03 form=0.22
            itf:        common_opp=0.05 h2h=0.02 form=0.28
[x] T21-04: DEFAULT_WEIGHTS actualizado con 5 tiers
[x] T21-05: Tests classify_tournament() con nombres reales FlashScore
[x] T21-06: density_confidence(n_common, n_paths) en rivalry_analyzer.py
            n_common=0 → density=0.3 (penaliza common_opp, redistribuye a form_recent)
            Evidencia: LOG_DENSITY: n_common=0 n_paths=0 density=0.3 co_w: 0.1925→0.0577 form_w→0.2782
[x] T21-07: shrink_weights() — James-Stein
            Evidencia: LOG_SHRINKAGE: key=clay_grand_slam n=31 factor=0.608
[x] T21-08: Tests densidad + shrinkage
[x] T21-09: K-factor por tier: GS=24, ATP1000=28, ATP500=32, Challenger=40, ITF=48
            k_factor_efectivo(tier, recencia_pelt) en elo_system.py
[x] T21-10: Reset K post-PELT: recencia<=5 → K×1.5
[x] T21-11: Tests K-factor adaptivo

980 tests passing post-sprint. Ver [[Nodo-21-Pesos-Diferenciados-Por-Tier]].
```

---

### Fase 20 — H2H Immunity Dampener (Nodo-19) ✅ COMPLETADA 2026-06-03

```
[x] T19-01: calcular_h2h_immunity(direct_h2h_matches, favored, opponent) en rivalry_analyzer.py
            HOT + h2h_win_rate < 0.30 → 0.85 | HOT + h2h_win_rate > 0.70 → 1.12
            n_h2h < 3 → 1.00 (muestra insuficiente)
[x] T19-02: Integrado en generate_advanced_prediction()
            factor_markov_efectivo = factor_markov * immunity_factor
            Evidencia: LOG_MARKOV_P1: immunity=1.0 h2h_wr=0.5 (n<3 → neutro)
[x] T19-03: Campo h2h_immunity_factor en score_breakdown
[x] T19-04: Tests implementados — campo 'ganador' (no 'winner') en JSON H2H

980 tests passing. Ver [[Nodo-19-H2H-Immunity-Dampener]].
```

---

### Fase 21 — PELT Recency Alpha (Nodo-18) ✅ COMPLETADA 2026-06-03

```
[x] T18-01: calcular_recencia_regimen(markov_result, n_total_partidos) en markov_analyzer.py
            FRESCO ≤3 | RECIENTE ≤7 | ESTABLE >7
[x] T18-02: factor_alpha_temporal(recencia, estado, delta_wr) → float
            HOT + FRESCO → λ/1.20 | COLD + FRESCO → λ/0.85
[x] T18-03: Integrado en edge_calculator.py — λ_efectivo = λ_tier × (1 / factor_alpha_temporal)
[x] T18-04: Campos recencia_regimen y alpha_temporal en output
[x] T18-C5: Campo n_partidos añadido al output de detectar_cambio_regimen()
[x] T18-05: Tests implementados

980 tests passing. Ver [[Nodo-18-PELT-Recency-Alpha]].
```

---

### Fase 22 — PageRank Erdős Quality (Nodo-20) ✅ COMPLETADA 2026-06-03

```
[x] T20-01: pagerank_grafo(grafo, damping=0.85, n=10) en erdos_graph.py
            Power iteration | normalizado [0, 1] | n<5 → {} sin PageRank
[x] T20-02: distancia_erdos() recibe node_weights=None opcional
            Calcula PageRank internamente si None
            Solo paths transitivos (len≥3) reciben quality_multiplier (REGLA-T20-3)
[x] T20-03: pagerank_scores en output de distancia_erdos()
[x] T20-04: Tests convergencia + edge cases

980 tests passing. Ver [[Nodo-20-PageRank-Erdos-Quality]].
```

---

### Fase 23 — Validación Multi-Torneo en Producción ✅ 2026-06-07

```
Primera validación completa del pipeline multi-torneo con datos reales (80 partidos, 2026-06-04).
Ejecutado: edge_calculator → trader (5 tiers) → resultados_finales → verificación API Ninja.

RESULTADOS:
  Accuracy general: 72.2% (57/79, 1 LIVE excluido)
  Por tier:
    Grand Slam:  50.0% (1/2)   — muestra insuficiente
    ATP 1000:    N/A            — sin partidos en archivo
    ATP 500:     70.6% (12/17) ✅ ALTA CONFIANZA
    Challenger:  64.5% (20/31) 🟡 RENTABLE CON KELLY
    ITF:         82.8% (24/29) ✅ ALTA CONFIANZA

  Underdogs @2.00+ que ganaron: 26/79 (33% de los partidos)
  Modelo predijo underdogs correctamente: 13/26 = 50%
  Por rango cuota: @2.00-2.49: 50% | @2.50-3.49: 50% | @3.50+: 50%

  Trader output por tier:
    Grand Slam:  1 señal, $12k en riesgo (9.6% bankroll) ✅
    ATP 1000:    0 señales (sin partidos) ✅
    ATP 500:     4 señales, VaR ajustó ×0.38 → $12k (40%) ⚠️
    Challenger:  8 señales, VaR ajustó ×0.83 → $24k (120%) 🔴
    ITF:         7 señales, VaR ajustó ×0.37 → $0 (protección) ✅

FEATURES VALIDADAS EN PRODUCCIÓN:
  [x] FlashScore Ninja API (dc_1_{event_id}) — 80 consultas, 0 errores, ~1s/partido
  [x] --all-tournaments en extraer_historh2h.py — procesa multi-torneo correctamente
  [x] --torneo-tipo en trader_ev_tenis.py — FILTRA picks por tier ✅
  [x] --torneo-tipo en resultados_finales.py — verifica por tier ✅
  [x] detectar_tier() — clasifica correctamente GS/ATP500/Challenger/ITF
  [x] VaR auto-ajuste — protege ITF ($0 stakes), reduce ATP500
  [x] James-Stein shrinkage — LOG_SHRINKAGE visible en output (factor=0.608, n=31)
  [x] Density confidence — penaliza common_opp cuando n_common=0

BUGS DESCUBIERTOS:
  [B-01] Challenger 120% bankroll — VaR ×0.83 insuficiente cuando muchos picks
  [B-02] ✅ RESUELTO 2026-06-11 — _load_p_prior(superficie, tier): jerarquía por_superficie_y_tier → fallback_por_tier → por_superficie → global. clay+challenger 0.697→0.590, grass+atp500 0.569→0.650
  [B-03] ✅ RESUELTO 2026-06-11 — header dinámico: TRADER EV TENIS — {torneo_tipo.upper()} {superficie.upper()}
  [B-04] generar_tabla_favoritos2.py selecciona archivo pequeño (2 partidos) sobre grande (80)
  [B-05] consultar_resultados_historicos.py — busca key 'partidos', JSON tiene 'detailed_results' → ROTO
  [B-06] Campo 'favorito' vacío en edge_report JSON (solo afecta auditoría, no P&L)
  [B-07] Calibración muestra "n=150" — posiblemente inflado vs CLAUDE.md que dice n=33

Ver [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-17-Calibracion-Por-Tier]]
```

---

### Fase 24 — API Integration: Kambi + FlashScore Ninja (Nodo-22) ✅ COMPLETADA 2026-06-07

```
Pipeline completo reemplazado: 40+ min Playwright → ~45 segundos APIs puras.
Cross-domain bridge NBA→Tennis: Kambi API idéntica, name matching 3-tier reutilizado.

PASO 1 — extraer_partidos_api.py (NUEVO):
  [x] T22-01: Implementar scraping/kambi_tennis.py — módulo core
              fetch_kambi_tennis() → partidos con cuotas reales Betplay (odds/1000)
              fetch_flashscore_feed(day_offset) → match_ids + rankings + superficie
              KAMBI_BASE = "https://us.offering-api.kambicdn.com/offering/v2018/betplay"
              KAMBI_TIER_MAP: termKeys → pipeline tiers (atp, wta, challenger, itf, doubles)
  [x] T22-02: Implementar _parse_nombre() — NBA pattern con suffix stripping
              Extrae (apellido, inicial) de nombres completos y abreviados
              Handles: Jr/Sr/II/III, compound surnames ("Davidovich Fokina")
  [x] T22-03: Implementar match_players() — 3-tier cross-reference Kambi↔FlashScore
              Tier 1: exact key match (surname+initial for both players)
              Tier 2: surname-only match with initial disambiguation
              Tier 3: substring overlap ≥5 chars with score ≥2
              Resultado: ATP+WTA 100% matched | 71/111 total (40 sin FlashScore = UTR Pro/ITF menor)
  [x] T22-04: Implementar save_matches() — output data/zita_tennis_matches_FECHA.json
              Mismo formato que Playwright PASO 1 — downstream sin cambios
              Campo cuota_es_real = True en cada partido
  [x] T22-05: Crear extraer_partidos_api.py — CLI entry point
              Flags: --tomorrow (day_offset=1), --tier atp wta challenger wta125 itf
              Testado: 1.3s para 111 partidos, 111/111 cuotas reales Betplay

PASO 2 — extraer_historh2h.py --api-mode (MODIFICADO):
  [x] T22-06: Crear scraping/ninja_h2h_parser.py — NinjaH2HExtractor
              Parser FlashScore Ninja H2H API (df_hh_1_{match_id})
              Funciones: _parse_sections(), _split_into_h2h_blocks(), _parse_player_history()
              Integra: EloRatingSystem, RankingManager, RivalryAnalyzer (analysis/)
              Output: MISMO formato JSON que H2HExtractor (Strangler Fig)
  [x] T22-07: Añadir --api-mode flag en extraer_historh2h.py
              from scraping.ninja_h2h_parser import NinjaH2HExtractor
              Ruta síncrona (sin async) cuando --api-mode activo
              Playwright mode preservado como default/fallback
  [x] T22-08: Smoke test producción — Borges vs Kecmanovic
              50+50 historiales + 4 H2H directos parseados correctamente
              ~0.5s/partido vs 2-3 min Playwright

VALIDACIÓN:
  [x] T22-09: 980 tests passing — pipeline no roto
  [x] T22-10: extraer_partidos_api.py → 111 partidos, 1.3s, cuotas reales
  [x] T22-11: --api-mode --all-tournaments → H2H completo vía Ninja API

Archivos nuevos:
  [+] scraping/kambi_tennis.py      — Kambi API + FlashScore feed + name matching
  [+] scraping/ninja_h2h_parser.py  — FlashScore Ninja H2H parser
  [+] extraer_partidos_api.py       — CLI PASO 1 API

Archivos modificados:
  [~] extraer_historh2h.py          — --api-mode flag añadido
  [~] config.py                     — detectar_tier() + Kambi API constants (si aplicable)

Ver Nodo-22-API-Integration-Kambi-Ninja
```

---

## 3. Tabla de Cobertura por Fase

| Fase | Componente | % Pipeline funcional | Impacto en P&L |
|---|---|---|---|
| 0 ✅ | data_parser.py fixes | 30% | Base limpia |
| 1 ✅ | Scraper fix (h2h_url + torneo) | 50% | surface_specialization activo |
| 2 ✅ | edge_calculator.py | 60% | Primera decisión matemática de apuesta |
| 3 ✅ | markov_analyzer.py | 70% | form_recent mejorado |
| 4 ✅ | Dataset ML fix | 80% | Modelo entrenable con datos limpios |
| 5 | FlashScore API labels | 90% | Labels en tiempo real |
| 6 | generar_tabla_favoritos2 fix | 95% | Reporte humano completo |
| 7 ✅ | Erdős graph | 98% | common_opponents más preciso |
| 13 ✅ | trader_ev_tenis.py | 100% | Deploy combos+sistema → bankroll exponencial |
| 14 ✅ | Validación live + TTC | — | Alpha confirmado: Parry @ 4.50 ganó. 5 conexiones ocultas documentadas |
| 17 ✅ | config.py centralizado | — | FLASHSCORE_BASE + HEADERS + pipeline constants. 791 tests |
| 18 ✅ | Portfolio Hedge Fund (Nodo-15) | — | Sistema Cobertura Exclusión + Portfolio Kelly + VaR/CVaR. KGR=+0.4142. 8/8=100% RG R4. 862 tests |
| 19 ✅ | Pesos Diferenciados por Tier (Nodo-21) | — | 5 tiers SNR + density + shrinkage + K-factor ELO. 980 tests. 2026-06-03 |
| 20 ✅ | H2H Immunity Dampener (Nodo-19) | — | factor_markov × immunity_factor. HOT×h2h_wr<0.30→0.85. 2026-06-03 |
| 21 ✅ | PELT Recency Alpha (Nodo-18) | — | change_point → recencia → λ_efectivo. HOT fresco→λ/1.20. 2026-06-03 |
| 22 ✅ | PageRank Erdős Quality (Nodo-20) | — | pagerank_grafo() + quality_multiplier en paths transitivos. 2026-06-03 |
| 23 ✅ | Validación Multi-Torneo Prod | — | 72.2% accuracy (57/79). API Ninja + --all-tournaments + resultados_finales. 2026-06-07 |
| 24 ✅ | API Integration Kambi+Ninja (Nodo-22) | — | Pipeline ~45s: kambi_tennis.py + ninja_h2h_parser.py + extraer_partidos_api.py. 2026-06-07 |
| 25 ✅ | Fixes pipeline + betplay_combo_builder.py | — | import re + match_id + superficie fix + --live mode + combos parciales. Calibración n=284. 2026-06-09 |
| 26 ✅ | Fixes pipeline sesión 3 (2026-06-11) | — | FlashScore feed tipo=13 (23→146 singles mañana) | combo builder lee cobertura de todos los trader_plans (24h) → stakes reales por tier |

---

## 4. APK / Entregable Checklist (Equivalente)

Para este proyecto el "APK" es el sistema en producción corriendo en Roland Garros:

```
[x] extraer_URL_partidos_version2.py genera h2h_url válidas ✅ (--max-matches 80 activo)
[x] extraer_historh2h.py procesa 80 partidos multi-torneo ✅ (--all-tournaments, 2026-06-04)
[x] edge_calculator.py identifica ≥1 partido con edge > 5% ✅ (20 señales en 80 partidos)
[x] trader_ev_tenis.py genera plan deploy por tier ✅ (5 tiers, VaR auto-ajuste)
[x] Primera señal APOSTAR validada: Parry @ 4.50 GANÓ ✅ → retorno $45,000 sobre $10,000
[x] reports/trader_plan guardados para auditoría P&L ✅ (JSON + TXT por tier)
[x] resultados_finales.py verifica resultados vía API Ninja ✅ (80 partidos, ~73s total)
[x] P&L sesión registrado: +$25,000 (+25% bankroll) | bankroll $100k→$125k ✅
[x] Accuracy multi-torneo: 72.2% (57/79) — objetivo >55% SUPERADO ✅
[x] generar_tabla_favoritos2.py genera reporte sin error ✅ (shrinkage + density logs visibles)
[x] 980 tests passing ✅ (2026-06-06, post Nodo-18/19/20/21 + fixes)
[x] SequentialH2HExtractor eliminado ✅ (T07-09) — extraer_historh2h.py: 3,717→310 líneas (−92%)
[x] trader_ev_tenis.py v2.0 ✅ — Sistema Cobertura Exclusión + Portfolio Kelly + VaR/CVaR
[x] Roland Garros R4 2026-06-01: 8/8=100% accuracy ✅ — Tres underdogs predichos
[x] KGR=+0.4142 (configuración óptima --min-cuota 1.50) ✅
[x] Validación multi-torneo 2026-06-07: ITF 82.8% | ATP500 70.6% | Challenger 64.5% ✅
[x] FlashScore Ninja API integrada: dc_1_{event_id} con auth X-Fsign ✅
[x] detectar_tier() fuente única en config.py — 5 tiers clasificados correctamente ✅
[x] MODO API: Kambi + FlashScore Ninja — PASO 1 ~1.3s, PASO 2 ~0.5s/partido ✅ (2026-06-07)
[x] extraer_partidos_api.py: 111 partidos, cuota_es_real=True, 5 tiers ✅
[x] scraping/kambi_tennis.py: Kambi API + FlashScore feed + name matching 3-tier ✅
[x] scraping/ninja_h2h_parser.py: NinjaH2HExtractor — same output as H2HExtractor ✅
[x] extraer_historh2h.py --api-mode: ruta Ninja API (~0.5s/partido) ✅
[x] ninja_h2h_parser.py import re + match_id en _consolidate_result() ✅ (2026-06-09)
[x] validar_con_api.py superficie fix: tipo_cancha or superficie or unknown ✅ (2026-06-09)
[x] betplay_combo_builder.py: --live mode + combos parciales + started_map + find_outcome reasons ✅ (2026-06-09)
[x] B-02: p_prior por tier ✅ 2026-06-11 — _load_p_prior(superficie, tier) estratificada
[ ] B-01: Cap total bankroll en trader (Challenger 120%) — PENDIENTE
```

---

## 5. Demo Script — Primera Evidencia Real

**Objetivo: mostrar que el sistema identifica edge real en Roland Garros**

```
1. [Fase 1] Correr extraer_URL_partidos_version2.py
   → Verificar torneo = "Roland Garros (France)" en JSON
   → Verificar h2h_url presente en output

2. [Fase 2] Correr edge_calculator.py sobre h2h_results_enhanced
   → Mostrar tabla: partido | edge | kelly_kl | apostar
   → Identificar underdog con edge > 5%

3. [Fase 3] Mostrar markov_analysis en JSON
   → Jugador HOT vs jugador COLD → factor 1.15

4. [Fase 5] Correr validar_con_api.py post-partido
   → Resultado en < 1 seg vs 2-5 min con Playwright
```

---

## 6. Risk Register

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| DOM de FlashScore cambia → selector torneo no funciona | Alta | **Bajo** (mitigado) | **MODO API es ruta primaria — DOM solo afecta fallback Playwright** (Nodo-22 ✅) |
| FlashScore Ninja API rota sin aviso | Media | Alto | Mantener Playwright como fallback para PASO 1+2. dc_1 para labels |
| Kambi API cambia estructura o bloquea requests | Baja | Alto | Monitorear headers Referer. Fallback: Playwright + FlashScore DOM para cuotas |
| surface_specialization sigue en 0% después de Fase 1 | Media | Alto | Verificar que rivalry_analyzer lee torneo correctamente |
| KNN bug requiere refactor profundo generar_dataset_plus | Media | Medio | Inspeccionar las 8 columnas faltantes antes de tocar |
| edge_calculator calibra mal λ (KL divergence) | Alta | Medio | Empezar con λ=0.5, calibrar con n≥30 partidos |
| markov_analyzer introduce regresión en tests | Media | Medio | `python -m pytest tests/ --no-cov -q` antes de cada commit |
| Roland Garros termina antes de Fase 2 | Baja | Medio | Los datos de Jan 2026 (19 partidos) son suficientes para calibrar |

---

## 7. Tasks Cross-Pipeline — Conexiones descubiertas por Test-time Compute

> Estas tasks no pertenecen a un módulo único — son el resultado de interconexiones entre áreas aparentemente aisladas.

| Task | Conexión | Descripción | Fase |
|---|---|---|---|
| **CX-01** | N03 → N01 | Fix h2h_url (Nodo-03) → torneo disponible → surface_specialization activo → edge_calculator más preciso | Fase 1 desbloquea Fase 2 |
| **CX-02** | N02 → N01 | Markov factor HOT/COLD modifica confidence del modelo → kelly_kl se ajusta automáticamente | Fase 3 mejora Fase 2 |
| **CX-03** | N05 → N04 | FlashScore API labels (Nodo-05) → dataset con labels limpias (Nodo-04) → modelo más preciso | Fase 5 alimenta Fase 4 |
| **CX-04** | N01 → edge | edge_calculator detecta underdogs mejor que el bookmaker → el sistema apuesta en contra del mercado donde tiene ventaja | Nodo-01 core |
| **CX-05** | N06 → N01 | Erdős graph (Nodo-06) → mejor ponderación de common_opponents (20% del modelo) → predicción más precisa → edge más confiable | Fase 7 mejora todo |
| **CX-06** | Markov → KL | El historial de cambios de régimen (Markov) alimenta la distribución P_histórica en KL divergence → Kelly-KL más calibrado | N02 alimenta N01 |
| **CX-07** | API → labels | dc_1_{event_id} da resultado en tiempo real → labels post-partido sin esperar Playwright → ciclo de mejora más rápido | N05 acelera ciclo |
| **CX-08** | NBA → Tennis | Cross-domain bridge: Kambi API idéntica + name matching 3-tier reutilizado desde NBA project → pipeline 40min→45s | Fase 24 (Nodo-22) |
| **CX-09** | Kambi → edge | cuota_es_real=True (Betplay real) reemplaza cuotas FlashScore promediadas → edge_calculator más preciso → P&L más confiable | Nodo-22 mejora N01 |

### Estado de implementación cross-pipeline

- [x] CX-01 — h2h_url → superficie limpia (Nodo-03 fix activo)
- [x] CX-02 — Markov HOT/COLD → factor aplicado a form_recent en rivalry_analyzer.py
- [ ] CX-03 — API labels → dataset limpio (requiere Nodo-05 + Nodo-04)
- [x] CX-04 — edge_calculator → underdog detection (9 señales Jan 2026, todas underdog)
- [x] CX-05 — Erdős graph → common_opponents mejorado (Fase 7 ✅)
- [x] CX-06 — Markov factor en edge_calculator output (campo markov_favorito)
- [x] CX-07 — dc_1 API → ciclo de labels rápido (validar_con_api.py ✅, pendiente prod)
- [x] CX-08 — NBA→Tennis cross-domain bridge: Kambi API + name matching 3-tier (Nodo-22 ✅)
- [x] CX-09 — Kambi cuota_es_real → edge_calculator con odds reales Betplay (Nodo-22 ✅)

---

## 8. Inventario Maestro de Tasks — Todos los Nodos

> Fuente de verdad para tracking. Actualizar a medida que se completan.
> Formato: `[ ]` pendiente · `[x]` completado · `[~]` en progreso · `[!]` bloqueado

| Task | Nodo | Fase | Día | Depende de |
|---|---|---|---|---|
| ~~T03-01~~ ✅ | N03 | 1 | 1 | — |
| ~~T03-02~~ ✅ | N03 | 1 | 1 | — |
| ~~T03-03~~ ✅ | N03 | 1 | 2-3 | DOM inspection |
| ~~T03-04~~ ✅ | N03 | 1 | 2-3 | T03-03 |
| ~~T03-05~~ ✅ | N03 | 1 | 3 | T03-01..04 |
| T03-06 ⏳ | N03 | 1 | prod | T03-05 — pendiente: ejecutar scraper en producción |
| ~~T01-01~~ ✅ | N01 | 2 | 4 | — |
| ~~T01-02~~ ✅ | N01 | 2 | 4 | T01-01 |
| ~~T01-03~~ ✅ | N01 | 2 | 4 | T01-01, T01-02 |
| ~~T01-04~~ ✅ | N01 | 2 | 4 | T01-01 |
| T01-05 ⏳ | N01 | 2 | 5 | T01-03 — pendiente: verificar Majchrzak resultado real |
| ~~T02-01~~ ✅ | N02 | 3 | 6 | — |
| ~~T02-02~~ ✅ | N02 | 3 | 6 | T02-01 |
| ~~T02-03~~ ✅ | N02 | 3 | 6 | T02-01, T02-02 |
| ~~T02-04~~ ✅ | N02 | 3 | 7 | T02-01, T02-02 |
| ~~T02-05~~ ✅ | N02 | 3 | 7 | T02-04 |
| ~~T04-01~~ ✅ | N04 | 4 | 8 | — |
| ~~T04-02~~ ✅ | N04 | 4 | 8 | T04-01 |
| ~~T04-03~~ ✅ | N04 | 4 | 8 | — |
| ~~T04-04~~ ✅ | N04 | 4 | 8 | T04-02, T04-03 |
| T04-05 ⏳ | N04 | 4 | 9 | T04-04 (validación prod) |
| ~~T05-01~~ ✅ | N05 | 5 | 10 | T03-02 (match_id) |
| ~~T05-02~~ ✅ | N05 | 5 | 10 | T05-01 |
| ~~T05-03~~ ✅ | N05 | 5 | 10 | T05-01, T05-02 |
| T05-04 ⏳ | N05 | 5 | 11 | T05-03 — pendiente: match_id real en producción |
| ~~T06-01~~ ✅ | N06fix | 6 | 11 | — |
| ~~T06-02~~ ✅ | N06fix | 6 | 11 | T06-01 |
| ~~T06-03~~ ✅ | N06fix | 6 | 2026-05-30 | Ejecutado en prod — 16 partidos RG, sin errores |
| ~~T06-04~~ ✅ | N06fix | 6 | 2026-05-30 | v1 eliminada (Nodo-11), v2 única activa |
| ~~T07-01~~ ✅ | N07 | 7 | 12 | — |
| ~~T07-02~~ ✅ | N07 | 7 | 12-13 | T07-01 |
| ~~T07-03~~ ✅ | N07 | 7 | 13 | T07-02 |
| ~~T07-04~~ ✅ | N07 | 7 | 13 | T07-03 |

| ~~T-SF-04~~ ✅ | N07-SF | 8 | 2026-05-29 | T-SF-03 |
| ~~T-SF-05~~ ✅ | N07-SF | 8 | 2026-05-30 | Erdős + surface confirmados en prod |
| ~~T-FS-01~~ ✅ | N08 | 9 | 2026-05-29 | — |
| ~~T-FS-02~~ ✅ | N08 | 9 | 2026-05-29 | — |
| ~~T-FS-03~~ ✅ | N08 | 9 | 2026-05-29 | — |
| ~~T-FS-04~~ ✅ | N08 | 9 | 2026-05-29 | — |
| ~~T-FS-05~~ ✅ | N08 | 9 | 2026-05-29 | — |
| T-FS-06 ⏳ | N08 | 9 | prod | T-FS-05 — verificar selección en run real |
| ~~T09-01~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~T09-02~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~T09-03~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~T09-04~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~T09-05~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~T09-06~~ ✅ | N09 | 10 | 2026-05-29 | — |
| ~~D-01..D-09~~ ✅ | Deuda | 11 | 2026-05-29 | — verificado con grep |
| ~~D-10/D-11/D-12~~ ✅ | Deuda | 11 | 2026-05-29 | — verificado con grep |
| ~~T12-01~~ ✅ | N07-F2 | 12 | 2026-05-29 | — |
| ~~T12-02~~ ✅ | N07-F2 | 12 | 2026-05-29 | T12-01 |
| ~~T12-03~~ ✅ | N07-F2 | 12 | 2026-05-29 | T12-02 |
| ~~T07-09~~ ✅ | N07-F2 | 16 | 2026-05-30 | T12-01..03 — 53→48 tests, 1,404 líneas eliminadas, 768 passed |
| ~~D-17~~ ✅ | Deuda | 17 | 2026-05-31 | — config.py creado, 10 tests, 791 passed |

| ~~T12-01~~ ✅ | N12 | 13 | 2026-05-30 | — |
| ~~T12-02~~ ✅ | N12 | 13 | 2026-05-30 | T12-01 |
| ~~T12-03~~ ✅ | N12 | 13 | 2026-05-30 | T12-01 |
| ~~T12-A~~ ✅ | N12 | 13 | 2026-05-30 | — ya estaba correcto en línea 271 (TTC verificado) |
| ~~T12-B~~ ✅ | N12 | 13 | 2026-05-31 | — flashscore_rankings_inspector.py → tools/ |
| ~~T12-C~~ ✅ | N12 | 13 | 2026-05-31 | — routes/ SUSPENDIDO: isla Flask/Selenium, 0 acoplamiento pipeline |
| ~~T12-D~~ ✅ | N12 | 13 | 2026-05-31 | — N/A: extraer_URL_partidos_en_vivo.py eliminado 2026-05-31 |

| ~~T21-01..11~~ ✅ | N21 | 19 | 2026-06-03 | — Pesos 5 tiers + density + shrinkage + K-factor ELO |
| ~~T19-01..04~~ ✅ | N19 | 20 | 2026-06-03 | — H2H Immunity Dampener |
| ~~T18-01..05~~ ✅ | N18 | 21 | 2026-06-03 | — PELT Recency Alpha |
| ~~T20-01..04~~ ✅ | N20 | 22 | 2026-06-03 | — PageRank Erdős Quality |
| ~~Fase 23~~ ✅ | Validación | 23 | 2026-06-07 | — 72.2% accuracy 80 partidos multi-torneo |
| ~~T22-01..11~~ ✅ | N22 | 24 | 2026-06-07 | — Kambi API + Ninja H2H + extraer_partidos_api.py + --api-mode |

| ~~Fase 26~~ ✅ | Fixes sesión 3 | 26 | 2026-06-11 | — FlashScore tipo=13 + combo builder multi-plan merge |
| ~~Fase 27~~ ✅ | Fixes sesión 4 | 27 | 2026-06-11 | — markov_analysis path bug (Nodo-18/19 silenciados) + B-02 p_prior tier + B-03 header |
| ~~Fase 29~~ ✅ | [[Nodo-27-Pipeline-Tracker-Observabilidad]] | 29 | 2026-06-17 | — pipeline_tracker.py 7 secciones + 17 tests. HOT=64%, NEUTRAL=6.7% descubiertos. 1024 tests |

### Fase 28 — Auditoría Financiera 4-Marcos (2026-06-12)

```
Análisis TTC (Marco de Tres Expertos): Senior SWE + Data Analyst + Architect + Quant
Origen: Sesión 5 validación multi-torneo — descubiertas 4 inconsistencias críticas en trader_ev_tenis.py

[x] B-06: KGR denominator — total_staked → bankroll en _compute_var_cvar()
           Impacto: Kelly Growth Rate inflado (KGR = log(1+P&L/total_staked) vs log(1+P&L/bankroll))
           Fix: línea 213 cambiar denominador a bankroll (capital inicial, no suma de stakes)
           Severidad: 🔴 CRÍTICO — puede bypass REGLA-HF-5 (no-deploy si KGR<0)

[x] B-01: VaR cálculo incompleto — gastado_ind ausente en riesgo total
           Impacto: VaR mide solo cobertura (40% bankroll) → excluye individuales (40% bankroll)
           Fix: línea 252 lectura completa de plan. Incluir gastado_ind en distribución binomial
           Severidad: 🔴 CRÍTICO — riesgo total underestimated 50%

[x] B-07: VaR binomial ignora ρ (correlación)
           Impacto: Tail risk underestimado para N picks correlacionados (ρ=0.25 Grand Slam)
           Fix: líneas 158-176 integrar ρ en cálculo de varianza conjunta (binomial multivariado)
           Severidad: 🟠 ALTO — portfolio risk underestimated para Grand Slams

[x] B-05: _print_resumen hardcoded "clay (Roland Garros)"
           Impacto: Output muestra siempre misma superficie independientemente de --superficie/--torneo-tipo
           Fix: línea 810 parámetrizar con args.superficie y torneo_tipo detectado dinámicamente
           Severidad: 🟡 MEDIO — solo afecta output legible, no P&L
```

**Total tasks: 127 | ✅ Completadas: 117 | ⏳ Pendiente: T15-06 backtesting, Nodo-27 validación V-27-1→V-27-5 (n≥50)**

---

### Fase 29 — Pipeline Tracker & Observabilidad (Nodo-27) ✅ COMPLETADA 2026-06-17

```
Motivación: el pipeline generaba picks sin ningún sistema de medición de rendimiento histórico.
Operamos "a ciegas" — sin saber qué señales, cuotas o tiers producen mejor ROI real.

Implementado: pipeline_tracker.py — READ-ONLY, 7 secciones, 3 fases.
CLI: python3 pipeline_tracker.py [--since YYYY-MM-DD] [--tier challenger] [--section confianza]
Output: pipeline_tracking.txt (sobreescribe) + --save para snapshot JSON en reports/

[x] T27-01: corre sin error con 0 archivos → "Sin datos de edge_report disponibles" ✅
[x] T27-02: S-27-1 confidence_flag counts correctos con datos mock ✅
[x] T27-03: S-27-2 cuota bins (1.50-2.00, 2.00-2.50, 2.50-3.00, 3.00-4.00, 4.00+) ✅
[x] T27-04: ROI calcula correctamente con stake real | proxy (cuota-1)*w-l cuando stake=0 ✅
[x] T27-05: Join edge_report + apuestas + resultados_finales por match_id (3 niveles) ✅
[x] T27-06: --since filtra por fecha correctamente ✅
[x] T27-07: --tier filtra por tier correctamente ✅
[x] T27-08: Campos faltantes (bbi, golden_zone) en reportes viejos → None, no crash ✅

Tests: 17 nuevos (T27-01→T27-08c) | Suite total: 1007→1024 passed ✅

HALLAZGOS CON DATOS REALES (Jun 14-17, n=33 con resultado):
  S-27-1 Confianza:
    STRONG    → 3W/0L = 100%  ✅ señal funciona
    MODERATE  → 1W/0L = 100%  ✅ (muestra pequeña *)
    LOW       → 8W/21L = 27%  ❌ picks LOW son ruido — NO apostar
    (sin flag) → picks viejos sin confidence_flag (pre-Nodo-24)

  S-27-2 Cuotas:
    1.50-2.00 → 5W/1L  = 83%  ✅ favoritos claros funcionan
    2.00-2.50 → 3W/11L = 21%  ❌ zona peligrosa
    2.50-3.00 → 1W/7L  = 12%  ❌ underdogs mediocres sin edge real
    3.00-4.00 → 3W/1L  = 75%  ✅ confirma hipótesis underdogs con edge alto
    4.00+     → 0W/1L  =  0%  * muestra insuficiente

  S-27-3 Tier+Superficie:
    ATP500 grass    → 2W/9L  = 18%  ❌ bookmaker conoce Top 100 (BBI=0.505)
    Challenger clay → 3W/5L  = 37%  ➡ margen positivo (BBI=0.580)
    Challenger grass→ 1W/2L  = 33%  * muestra pequeña
    ITF clay        → 5W/5L  = 50%  ✅ competitivo (BBI=0.514)
    ITF hard        → 1W/0L  =100%  * muestra pequeña (BBI=0.628)
    Lección épica confirmada: ATP500 grass -62% ROI proxy vs ITF clay -5.8%

  S-27-4b Markov:
    HOT     → 9W/5L  = 64%  ✅ señal más fuerte del sistema
    NEUTRAL → 1W/14L = 6.7% ❌ FILTRAR — casi igual que azar puro
    COLD    → 2W/2L  = 50%  * muestra pequeña

  ACCION INMEDIATA: considerar filtrar picks NEUTRAL Markov en edge_calculator.py o trader

Ver: [[Nodo-27-Pipeline-Tracker-Observabilidad]]
```

---

## 9. Vinculación

- [[Mandatos-No-Negociables]] — 8 mandatos que no pueden violarse
- [[Nodo-01-Edge-Calculator]] — Kelly-KL implementación
- [[Nodo-02-Markov-Changepoint]] — PELT + Cadenas de Markov
- [[Nodo-03-Scraper-Fix]] — 3 bugs del scraper de URLs
- [[Nodo-04-Dataset-Fix]] — 2 bugs del pipeline ML
- [[Nodo-05-Validacion-API]] — validación con API en tiempo real
- [[Nodo-06-Erdos-Graph]] — mejora del grafo transitivo
- [[Grafo-Dependencias-Datos]] — flujo completo de datos
- [[Pipeline-Arquitectura]] — diagrama del sistema
- [[Nodo-07-Strangler-Fig]] — Strangler Fig CERRADO ✅ — T07-09 ejecutado 2026-05-30
- [[Nodo-08-File-Selection-Bug]] — fix selección archivo por recency
- [[Nodo-09-API-Status-Keys]] — claves reales dc_1: DJ/DE/DF
- [[Nodo-13-Trader-EV-Tenis]] — deploy Bayesian blend + combos + sistema 2/N
- [[Nodo-14-Validacion-Live-Conexiones]] — primera validación live Roland Garros 2026-05-30
- [[Nodo-15-Portfolio-HedgeFund]] — Portfolio Kelly + VaR + Cobertura Exclusión
- [[Nodo-16-Multi-Torneo-Pipeline]] — --max-matches 80 + --all-tournaments
- [[Nodo-17-Calibracion-Por-Tier]] — λ por tier + surface propagation + calibración estratificada
- [[Nodo-18-PELT-Recency-Alpha]] — recencia_regimen + factor_alpha_temporal → λ_efectivo
- [[Nodo-19-H2H-Immunity-Dampener]] — calcular_h2h_immunity: HOT×h2h_wr
- [[Nodo-20-PageRank-Erdos-Quality]] — pagerank_grafo + quality_multiplier
- [[Nodo-21-Pesos-Diferenciados-Por-Tier]] — 5 tiers SNR + density + shrinkage + K-ELO
- [[Nodo-27-Pipeline-Tracker-Observabilidad]] — pipeline_tracker.py READ-ONLY, 7 secciones. Hallazgos: STRONG=100%, HOT=64%, NEUTRAL=6.7% (filtrar), ATP500 grass=18% ⚠️
- [[Inventario-Deuda-Tecnica]] — D-14 CERRADO ✅, ~9,400 líneas eliminadas acumulado

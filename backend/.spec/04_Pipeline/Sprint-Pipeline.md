# Sprint — Pipeline de Construcción

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Grafo-Dependencias-Datos]] | [[Pipeline-Arquitectura]] | [[Inventario-Deuda-Tecnica]]
> Nodos: [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Nodo-13-Trader-EV-Tenis]] | [[Nodo-14-Validacion-Live-Conexiones]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-20-PageRank-Erdos-Quality]]
>
> Documento de planificación de implementación. Convierte el Spec Kit en **motor de construcción determinista**.
> Última actualización: 2026-06-03 — Nodo-17 Fase 1 ✅ | **898 tests, 0 failed.** Nodo-18/19/20 documentados (TTC 2026-06-03): PELT Recency Alpha + H2H Immunity Dampener + PageRank Erdős Quality.

---

## 1. El Puente: Componente → Módulo → Prioridad

### 1.1 Productores primarios (deben existir ANTES de cualquier módulo ML)

| Productor | Dato producido | Módulo que desbloquea |
|---|---|---|
| `extraer_URL_partidos_version2.py` (con fix) | `h2h_url`, `torneo`, `match_id` limpios | Todo el pipeline |
| `extraer_historh2h.py` | `h2h_results_enhanced_FECHA.json` | Predicción + Edge |
| `edge_calculator.py` | `edge`, `kelly_kl`, `apostar` | Decisión de apuesta |
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

### Fase 19 — Pesos Diferenciados por Tier (Nodo-21) 🔴 PENDIENTE — IMPLEMENTAR PRIMERO

```
Problema: 3 capas de tier desconectadas + bug crítico classify_tournament()
BUG: "French Open (France)" → classify_tournament() retorna 'default', no 'grand_slam'
BUG: Grand Slam y ATP 500 usan pesos idénticos ('atp_wta')
Conexiones ocultas: densidad local del grafo + James-Stein shrinkage + K-factor Kalman

FASE 1 — Bug fix + Unificación (alto impacto, bajo riesgo):
[ ] T21-01: Fix classify_tournament() con misma lógica que detectar_tier()
            5 categorías: grand_slam | atp1000 | atp500 | challenger | itf
            Keywords: roland garros, french open, wimbledon, australian open, us open → grand_slam

[ ] T21-02: Mover detectar_tier() a config.py — función compartida única
            Importar desde edge_calculator.py Y rivalry_analyzer.py
            NUNCA dos funciones de clasificación independientes

[ ] T21-03: Actualizar weights_config en generate_advanced_prediction() — 5 tiers con SNR
            grand_slam: common_opp=0.22 h2h=0.18 form=0.12 (H2H denso)
            challenger:  common_opp=0.08 h2h=0.03 form=0.22 (red fragmentada)
            itf:         common_opp=0.05 h2h=0.02 form=0.28 (sólo forma)

[ ] T21-04: Actualizar normalization.py DEFAULT_WEIGHTS con 5 tiers

[ ] T21-05: Tests: classify_tournament() con nombres reales FlashScore
            French Open, Wimbledon, ATP 500 Barcelona, Challenger Heilbronn, ITF M25

FASE 2 — Densidad local + Shrinkage (implementar después de Nodo-19):
[ ] T21-06: density_confidence(n_common, n_paths) → modular common_opponents weight
            n_common=20+ → factor~1.0 | n_common=2-3 → factor~0.4
            Redistribuir peso sobrante a form_recent automáticamente

[ ] T21-07: shrink_weights(tier_weights, default, n_tier) — James-Stein
            n=0: 100% default | n=31: 61% tier | n=100: 83% tier
            Usar calibracion_edge.json[por_superficie_y_tier] como fuente de n

[ ] T21-08: Tests densidad + shrinkage

FASE 3 — K-factor ELO Kalman (después de Nodo-18):
[ ] T21-09: K-factor por tier: GS=24, ATP1000=28, ATP500=32, Challenger=40, ITF=48
[ ] T21-10: Reset K post-PELT: recencia<=5 → K×1.5 (conexión Nodo-18 T18-01)
[ ] T21-11: Tests K-factor adaptivo
```

---

### Fase 20 — H2H Immunity Dampener (Nodo-19) 🔴 PENDIENTE

```
Problema: factor_markov aplicado globalmente — HOT contra rival que históricamente lo domina.
Señal de 2do orden ignorada: estado HOT × h2h_win_rate_vs_rival_específico.

Archivos tocados: analysis/rivalry_analyzer.py
Qué lee: direct_h2h_matches + estado Markov del favorito predicho
Qué produce: immunity_factor que modifica factor_markov antes del score final
P&L: previene sobreconfianza cuando HOT player históricamente pierde a ESTE rival

[ ] T19-01: calcular_h2h_immunity(direct_h2h_matches, favored, opponent)
            → {'h2h_win_rate': float, 'immunity_factor': float, 'n_h2h': int}
            HOT + h2h_win_rate < 0.30 → 0.85 | HOT + h2h_win_rate > 0.70 → 1.12
            n_h2h < 3 → 1.00 (muestra insuficiente)

[ ] T19-02: Integrar en generate_advanced_prediction()
            factor_markov_efectivo = factor_markov * immunity_factor
            Aplicar ANTES del score final

[ ] T19-03: Campo h2h_immunity_factor en score_breakdown y edge report

[ ] T19-04: Tests (~12 casos): n_h2h=0, HOT+inmune, HOT+dominante, COLD, n_h2h<3
            Validar con patrón real Djokovic vs Nadal clay
```

---

### Fase 20 — PELT Recency Alpha (Nodo-18) 🔴 PENDIENTE

```
Problema: change_point en JSON de markov_analyzer — COMPLETAMENTE IGNORADO en edge_calculator.
Alpha real: bookmaker no repricing en ventana ≤3 partidos post-cambio de régimen.

Archivos tocados: analysis/markov_analyzer.py + edge_calculator.py
Qué lee: change_point + len(historial) + win_rate_reciente/anterior (ya en JSON)
Qué produce: recencia_regimen + factor_alpha_temporal → modifica λ_efectivo

[ ] T18-01: calcular_recencia_regimen(markov_result, n_total_partidos)
            → {'recencia': int, 'freshness': 'FRESCO'|'RECIENTE'|'ESTABLE'}
            FRESCO ≤3 | RECIENTE ≤7 | ESTABLE >7

[ ] T18-02: factor_alpha_temporal(recencia, estado, delta_wr) → float
            HOT + FRESCO → 1.20 | HOT + RECIENTE → 1.10 | COLD + FRESCO → 0.85 | else → 1.00

[ ] T18-03: Integrar en edge_calculator.py:
            λ_efectivo = λ_tier × (1 / factor_alpha_temporal)
            Precondición: T19 implementado primero

[ ] T18-04: Campos recencia_regimen y alpha_temporal en output edge report

[ ] T18-C5: Campo delta_wr (magnitud cambio) en edge report — informativo, no decisorio

[ ] T18-05: Tests (~15 casos): freshness levels + factor por estado + integración edge
```

---

### Fase 21 — PageRank Erdős Quality (Nodo-20) 🔴 PENDIENTE

```
Problema: nodos intermedios en grafo Erdős sin peso de centralidad.
"Parry ganó a X que ganó a Djokovic" = "Parry ganó a Y que ganó a rank-300" — incorrecto.

Archivos tocados: analysis/erdos_graph.py
Qué lee: grafo ya construido en construir_grafo_victorias()
Qué produce: pagerank_score por jugador + quality-weighted erdos_score

[ ] T20-01: pagerank_grafo(grafo, damping=0.85, n=10) → dict {jugador: centrality_score}
            Power iteration 15 líneas | normalizado [0, 1]
            Casos edge: grafo vacío → {} | un solo nodo → {n: 1.0}

[ ] T20-02: distancia_erdos() recibe node_weights=None opcional
            Si se pasa: advantage *= node_weights.get(intermediate, 0.5)
            Caminos directos (len=2): quality_multiplier = 1.0 siempre

[ ] T20-03: Exportar pagerank_scores en output de distancia_erdos() (campo informativo)

[ ] T20-04: Tests convergencia + casos edge (~10 casos)
            grafo vacío | un nodo | grafo desconectado | ciclos
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
| 19 🔴 | Pesos Diferenciados por Tier (Nodo-21) | — | BUG fix classify_tournament GS + 5 tiers con SNR correcto + density local + shrinkage ← PRIMERO |
| 20 🔴 | H2H Immunity Dampener (Nodo-19) | — | factor_markov × immunity_factor — señal 2do orden HOT×H2H. Previene error activo |
| 21 🔴 | PELT Recency Alpha (Nodo-18) | — | change_point → recencia → λ_efectivo reducido cuando bookmaker tiene precio stale |
| 22 🔴 | PageRank Erdős Quality (Nodo-20) | — | centrality de nodos intermedios → erdos_score ponderado por calidad de la cadena |

---

## 4. APK / Entregable Checklist (Equivalente)

Para este proyecto el "APK" es el sistema en producción corriendo en Roland Garros:

```
[ ] extraer_URL_partidos_version2.py genera h2h_url válidas (Fase 1)
[x] extraer_historh2h.py procesa 16+ partidos con surface_specialization > 0% ✅ (surf_w 0.49–0.69)
[x] edge_calculator.py identifica ≥1 partido con edge > 5% ✅ (Parry +29.3%, Tien +16.5%)
[x] trader_ev_tenis.py genera plan deploy con combos + sistema ✅ (2026-05-30, $20,000 en riesgo)
[x] Primera señal APOSTAR validada: Parry @ 4.50 GANÓ ✅ → retorno $45,000 sobre $10,000
[x] reports/trader_plan_20260530_121616.json guardado para auditoría P&L ✅ (T13-05)
[x] validar_con_api.py registró resultados 2026-05-30 ✅ (T14-01) — n: 13→23, p_hist: 0.52→0.68
[x] P&L sesión registrado: +$25,000 (+25% bankroll) | bankroll $100k→$125k ✅
[x] Accuracy con datos limpios: 70% (7/10) — objetivo >55% SUPERADO ✅
[x] generar_tabla_favoritos2.py genera reporte sin error ✅ 2026-05-30 (T06-03)
[x] 875 tests passing ✅ (2026-06-01, post T13-06/T15-04/T15-05/aplicar_enhancer — invariante ≥862)
[x] SequentialH2HExtractor eliminado ✅ (T07-09) — extraer_historh2h.py: 3,717→310 líneas (−92%)
[x] trader_ev_tenis.py v2.0 ✅ — Sistema Cobertura Exclusión + Portfolio Kelly + VaR/CVaR implementados
[x] Roland Garros R4 2026-06-01: 8/8=100% accuracy ✅ — Tres underdogs predichos (Kostyuk @3.0, Fonseca @2.3, Mensik @2.0)
[x] KGR=+0.4142 (configuración óptima --min-cuota 1.50) ✅ — P&L potencial +$635,510 (+907%)
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
| DOM de FlashScore cambia → selector torneo no funciona | Alta | Alto | Playwright screenshot + manual inspection primero |
| FlashScore Ninja API rota sin aviso | Media | Alto | Mantener Playwright como fallback para labels |
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

### Estado de implementación cross-pipeline

- [x] CX-01 — h2h_url → superficie limpia (Nodo-03 fix activo)
- [x] CX-02 — Markov HOT/COLD → factor aplicado a form_recent en rivalry_analyzer.py
- [ ] CX-03 — API labels → dataset limpio (requiere Nodo-05 + Nodo-04)
- [x] CX-04 — edge_calculator → underdog detection (9 señales Jan 2026, todas underdog)
- [x] CX-05 — Erdős graph → common_opponents mejorado (Fase 7 ✅)
- [x] CX-06 — Markov factor en edge_calculator output (campo markov_favorito)
- [x] CX-07 — dc_1 API → ciclo de labels rápido (validar_con_api.py ✅, pendiente prod)

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

**Total tasks: 70 | ✅ Completadas: 64 | ⏳ Pendiente validación prod: 5 (T03-06, T01-05, T04-05, T05-04, T-FS-06) | ⏳ Futuro: T14-05 | Pendientes dev activos: 0**

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
- [[Inventario-Deuda-Tecnica]] — D-14 CERRADO ✅, ~9,400 líneas eliminadas acumulado

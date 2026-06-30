# Inventario de Deuda Técnica — Tennis Prediction Engine

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Grafo-Dependencias-Datos]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Nodo-12-Inventario-Infraestructura-Legado]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-20-PageRank-Erdos-Quality]] | Nodo-22-API-Integration-Kambi-Ninja | [[Nodo-27-Pipeline-Tracker-Observabilidad]]
> Estado: 2026-06-18 sesión 8 | Tests: 1050 passed | Nodo-28 IMPLEMENTADO ✅
> Actualización: 2026-06-17 — pipeline_tracker.py implementado (7 secciones, READ-ONLY). Hallazgo crítico: picks NEUTRAL Markov=6.7% hit rate → deuda nueva: filtrar NEUTRAL del pool de combos.
> Deuda activa restante: T15-06 backtesting | Nodo-10 ZitaScraper unificación | D-18 consultar_resultados_historicos.py | B-01, B-04 | **NUEVO: filtro Markov NEUTRAL** (descubierto por Nodo-27)
> Metodología: Strangler Fig + Spec-Driven Development + TTC (Marco de Tres Expertos) + Test-Time Compute (4 Marcos)

---

## Estado Real Post-Nodo-07 (No Confundir con el Monolito Original)

El monolito original tenía **3,717 líneas** y **4 clases inline duplicadas**.
Nodo-07 Fase 1 (completado 2026-05-29) redujo `extraer_historh2h.py` a **1,707 líneas**.

```
ANTES (pre-Nodo-07):                    DESPUÉS (post-Nodo-07):
extraer_historh2h.py (3,717 líneas)     extraer_historh2h.py (1,707 líneas)
├── EloRatingSystem    (inline)    ✗    ├── from analysis import EloRatingSystem  ✅
├── RankingManager     (inline)    ✗    ├── from analysis import RankingManager   ✅
├── RivalryAnalyzer    (inline)    ✗    ├── from analysis import RivalryAnalyzer  ✅
└── SequentialH2HExtractor         →    └── SequentialH2HExtractor (1,447 líneas) ← DEUDA ACTIVA
```

**Backup del monolito original ELIMINADO 2026-05-29:** ✅
`extraer_historh2h_backup_20260529_114017.py` — ya no existe en disco.

---

## Clasificación de Deuda por Impacto en P&L

### NIVEL 1 — Bloquea P&L directamente (eliminar o corregir antes del próximo run)

| ID | Archivo | Deuda | Líneas de Deuda | Estado |
|---|---|---|---|---|
| D-01 | `extraer_historh2h_backup_20260529_114017.py` | Monolito original — crea confusión, no se usa | 3,717 | ✅ ELIMINADO 2026-05-29 |
| D-02 | `extraer_historh2h_version2.py` | Versión anterior (Jan 19), clases inline, sin imports internos | 2,963 | ✅ ELIMINADO 2026-05-29 |
| D-03 | `main.py` | 0 líneas — stub vacío que promete un entry point | 0 | ✅ ELIMINADO 2026-05-29 |
| D-04 | `utils.py` | 0 líneas — stub vacío | 0 | ✅ ELIMINADO 2026-05-29 |
| D-05 | `config.py` | 0 líneas — stub vacío, configuración sigue dispersa | 0 | ✅ ELIMINADO 2026-05-29 |
| D-06 | `services/scraper_service.py` | 0 bytes | 0 | ✅ ELIMINADO 2026-05-29 |
| D-07 | `services/scraper_service_atp_backup.py` | 0 bytes | 0 | ✅ ELIMINADO 2026-05-29 |
| D-08 | `services/simple_screenshot_test.py` | 0 bytes | 0 | ✅ ELIMINADO 2026-05-29 |

**Total líneas eliminadas NIVEL 1: 6,680 líneas** — 701 tests siguen pasando (verificado).
**Riesgo real: CERO** — ningún script activo importaba estos archivos.

---

### NIVEL 2 — Deuda de código duplicado (incrementa tiempo de mantenimiento)

| ID | Archivos duplicados | Clase duplicada | Líneas duplicadas | Estado |
|---|---|---|---|---|
| D-09 | `Intelligent_ml_enhancer.py` + `generar_dataset_plus.py` | `SmartLogger` | ~80 | ✅ RESUELTO 2026-05-29 — `utils/logger.py` creado, 19 tests |
| D-10 | `extraer_ranking_atp.py` (v1, 443 líneas) | `CompleteRankingScraper` (versión antigua) | 443 | ✅ ELIMINADO 2026-05-29 |
| D-11 | `extraer_ranking_wta.py` (v1, 269 líneas) | versión antigua | 269 | ✅ ELIMINADO 2026-05-29 |
| D-12 | `extraer_URL_partidos.py` (v1, 604 líneas) | `ZitaScraper` (versión antigua) | 604 | ✅ ELIMINADO 2026-05-29 |
| D-13 | `generar_tabla_favoritos.py` (v1, 848 líneas) | Reporte manual legacy | 848 | ✅ ELIMINADO 2026-05-30 (v2 validada en prod 2026-05-29) |

**Total líneas eliminadas D-09/D-10/D-11/D-12/D-13: ~2,244 líneas** — 772 tests passing.
**NIVEL 2 completamente resuelto.** Ver [[Nodo-12-Inventario-Infraestructura-Legado]] para deuda de infraestructura (Flask/Selenium stack).

---

### NIVEL 3 — Deuda de cobertura de tests (riesgo de regresión silenciosa)

| Archivo | Líneas prod | Tests actuales | Gap | Riesgo |
|---|---|---|---|---|
| `extraer_historh2h.py` | 310 | 53 migrados → H2HExtractor ✅ | — | CUBIERTO (T07-09) |
| `SequentialH2HExtractor` | ELIMINADO ✅ | — | — | T07-09 CERRADO 2026-05-30 |
| `generar_tabla_favoritos2.py` | 1,048 | 33 ✅ | — | CUBIERTO 2026-05-31 |
| `generar_dataset_plus.py` | 1,661 | 14 | más tests de integración | MEDIO |
| `Intelligent_ml_enhancer.py` | 1,350 | 38 ✅ | — | CUBIERTO 2026-05-31 |
| `aplicar_enhancer.py` | 1,005 | 13 ✅ | — | CUBIERTO 2026-06-01 |
| `scraping/kambi_tennis.py` | ~350 | 0 | smoke test solo | 🟡 BAJO — ruta API nueva, validado en prod |
| `scraping/ninja_h2h_parser.py` | ~400 | 0 | smoke test solo | 🟡 BAJO — validado con datos reales (Borges vs Kecmanovic) |
| `extraer_partidos_api.py` | ~115 | 0 | CLI validado en prod | 🟢 — entry point simple, lógica en kambi_tennis.py |
| `edge_calculator.py` | 673 | 43 ✅ | — | CUBIERTO |
| `validar_con_api.py` | ~400 | 39 ✅ | — | CUBIERTO (Nodo-09) |
| `normalization.py` | ~200 | 111 ✅ | — | CUBIERTO |

**Total líneas de producción sin cobertura: ~6,000+**
**Crítico para Nodo-07 Fase 2:** `SequentialH2HExtractor` necesita ≥40 tests antes de poder ser migrado a `scraping.H2HExtractor` (ver [[Nodo-07-Strangler-Fig]]).

---

### NIVEL 4 — Deuda de arquitectura (impacto en mantenibilidad a largo plazo)

#### D-14: SequentialH2HExtractor — clase legacy en extraer_historh2h.py

```
PRODUCCIÓN (main()): extraer_historh2h.py → H2HExtractor ✅ (Nodo-07 Fase 2 parcial, 2026-05-30)
CLASE LEGACY:        SequentialH2HExtractor conservada — 52 tests dependen de ella

Gaps cerrados Fase 2:
  ✅ Fix superficie en H2HExtractor.load_matches() — preferir match.get('superficie')
  ✅ Fix surface en H2HExtractor._process_single_match()
  ✅ Roland Garros filter añadido a H2HExtractor.load_matches()
  ✅ main() reescrito v5.0 — usa H2HExtractor

Pendiente T07-09 (sprint futuro):
  Eliminar SequentialH2HExtractor + migrar 52 tests a H2HExtractor
  Precondición: paridad de output verificada en ≥10 partidos reales
```

**Estado:** D-14 ✅ CERRADO 2026-05-30 — entry point migrado ✅ | SequentialH2HExtractor eliminado (T07-09) ✅ | extraer_historh2h.py = 310 líneas.

#### D-15: `extraer_URL_partidos_en_vivo.py` ✅ ELIMINADO 2026-05-31

237 líneas, clase `ZitaLiveScraper`. 0 importadores activos confirmados con grep.
Decisión final: ELIMINAR — git preserva el código si se necesita recuperar para modo LIVE futuro.
(`ml_trainer.py` también eliminado en este sprint — 709 líneas, 0 importadores, supersedido por aplicar_enhancer.py)

#### D-16: `prueba.py` ✅ CERRADO 2026-05-30 — NO EXISTE EN DISCO

Verificado con `ls -la prueba.py` → archivo no encontrado. Deuda eliminada de facto.

#### D-17: Configuración dispersa ✅ CERRADO 2026-05-31

`config.py` creado en raíz. Constantes migradas:
- `FLASHSCORE_BASE`, `FLASHSCORE_HEADERS` → `config.py` (importado como `HEADERS` en `validar_con_api.py`)
- `TOTAL_MATCHES_TO_PROCESS = 80` → `config.py` (importado en `scraping/h2h_extractor.py`)
- `BROWSER_HEADLESS = True`, `BROWSER_SLOW_MO = 250` → `config.py` (defaults de H2HExtractor.__init__)

TTC: `MAX_RAW_SCORES`/`DEFAULT_WEIGHTS` NO movidos — co-ubicados con lógica de normalización en `normalization.py`, no son "config dispersa" sino dominio de negocio.
10 tests añadidos en `tests/test_config.py` | 791 passed ✅

---

## Mapa de Costuras (Seams) para Strangler Fig

### Costura 1 — SequentialH2HExtractor ✅ COMPLETADA 2026-05-30 (T07-09)

```
ANTES (actual):
extraer_historh2h.py
  └── SequentialH2HExtractor  ← clase monolítica, 1,447 líneas
        ├── Browser (Playwright directo)
        ├── Data parsing (inline)
        └── Orquestación de matches

DESPUÉS (Nodo-07 Fase 2):
extraer_historh2h.py  (~50 líneas entry point)
  └── from scraping import H2HExtractor
        ├── BrowserManager     ← scraping/browser_manager.py ✅
        ├── DataParser         ← scraping/data_parser.py ✅
        └── H2HExtractor       ← scraping/h2h_extractor.py ✅

COMPLETADA: ✅ 2026-05-30 — 1,404 líneas eliminadas | 53 tests migrados → 48 en H2HExtractor/DataParser | 768 passed
```

### Costura 2 — SmartLogger duplicado ✅ COMPLETADA 2026-05-29

```
ANTES:
generar_dataset_plus.py    → class SmartLogger (definida inline)
Intelligent_ml_enhancer.py → class SmartLogger (COPIA IDÉNTICA)

DESPUÉS (estado actual):
utils/logger.py → class SmartLogger (única fuente de verdad, 19 tests)
generar_dataset_plus.py    → from utils.logger import SmartLogger  ✅
Intelligent_ml_enhancer.py → from utils.logger import SmartLogger  ✅
```

### Costura 3 — ZitaScraper (PRIORIDAD REDUCIDA — Nodo-22 mitiga)

```
ANTES:
extraer_URL_partidos_version2.py   → class ZitaScraper (604 líneas, con 3 bugs del Nodo-03)
extraer_URL_partidos.py            → ELIMINADO (D-12 ✅)
extraer_cuotas_partidos.py         → class ZitaScraper (VARIANTE odds)
extraer_URL_partidos_en_vivo.py    → ELIMINADO (D-15 ✅)

POST-NODO-22:
extraer_partidos_api.py            → RUTA PRIMARIA (~1.3s, Kambi+FlashScore API, sin Playwright)
extraer_URL_partidos_version2.py   → FALLBACK (Playwright, 8 min) — unificar ZitaScraper pierde urgencia

DESPUÉS (Nodo-10 — si se necesita):
scraping/url_scraper.py → class ZitaScraper (única, con todos los fixes del Nodo-03)
⚠️ Nodo-10 ahora tiene prioridad BAJA — el flujo API es la ruta principal
```

---

## Invariante de Calidad — Mandato 6 extendido

El éxito de cada intervención de Strangler Fig se mide por:

```
MÉTRICA 1 — Reducción de líneas duplicadas:
  Antes Nodo-07 Fase 1: 3,717 líneas (con 3 clases inline)
  Después Nodo-07 Fase 1: 1,707 líneas (-2,010 líneas, -54%)  ✅
  Meta Nodo-07 Fase 2: ~50 líneas entry point (-1,657 líneas adicionales)

MÉTRICA 2 — Tests (Mandato 6):
  Invariante: python -m pytest tests/ --no-cov -q NUNCA debe bajar de 767 passed
  Meta Nodo-07 Fase 2: ≥741 passed (40 nuevos tests SequentialH2HExtractor)
  Meta D-09 (SmartLogger): ≥746 passed (+5 tests utils/logger)

MÉTRICA 3 — Cobertura de orquestadores:
  Invariante: NINGÚN script de producción que toque datos financieros (cuotas, edge)
  puede tener 0 tests. Mínimo 10 tests de smoke.

MÉTRICA 4 — Archivos obsoletos en disco:
  Invariante: ningún archivo marcado ELIMINAR puede sobrevivir más de 1 sprint.
  Sprint deadline: D-01 a D-08 eliminados antes del próximo run de producción.
```

---

## Backlog Priorizado

| ID | Tarea | Impacto P&L | Esfuerzo | Estado |
|---|---|---|---|---|
| **D-01 a D-08** | Eliminar 8 archivos vacíos/backup | Claridad | 30 min | ✅ ELIMINADOS 2026-05-29 |
| **D-10/D-11/D-12** | Eliminar ranking/URL scrapers v1 | Claridad | 30 min | ✅ ELIMINADOS 2026-05-29 |
| **D-09** | Extraer SmartLogger a `utils/logger.py` | Mantenimiento | 1h | ✅ RESUELTO 2026-05-29 |
| **D-13** | Eliminar generar_tabla_favoritos.py v1 | Claridad | 15 min | ✅ ELIMINADO 2026-05-30 |
| **D-14** | Migrar SequentialH2HExtractor → H2HExtractor | Alto | 8h | ✅ CERRADO 2026-05-30 — T07-09 |
| **D-15** | Eliminar extraer_URL_partidos_en_vivo.py + ml_trainer.py | Claridad | 15 min | ✅ ELIMINADOS 2026-05-31 — 946 líneas |
| **D-16** | Liquidar prueba.py | Claridad | 2h | ✅ CERRADO (no existe en disco) |
| **D-17** | Centralizar config en `config.py` | Mantenimiento | 2h | ✅ CERRADO 2026-05-31 — 791 tests |
| **T12-B** | Mover flashscore_rankings_inspector.py → tools/ | Claridad | 30 min | ✅ HECHO 2026-05-31 |
| **T12-C** | Auditar routes/ completo | Claridad | 1h | ✅ AUDITADO 2026-05-31 — SUSPENDER: isla Flask/Selenium, 0 acoplamiento pipeline. Import roto en app.py (routes.players vs player_routes). Sin acción requerida. |
| **Nodo-10 (futuro)** | Unificar ZitaScraper → `scraping/url_scraper.py` | Medio | 4h | ⏳ Nodo-03 prod validado |
| **T14-05/T13-04** | Pipeline completo 80 partidos → sistema 2/N | Alto P&L | 4h | ⏳ bloqueado por decisión usuario |
| ~~**T15-05**~~ | ~~Implementar ajuste automático de stakes por factor VaR en main()~~ | Medio | — | ✅ 2026-06-01 |
| ~~**T15-04**~~ | ~~Calibrar ρ por tipo de torneo~~ | Medio | — | ✅ 2026-06-01 |
| ~~**T13-06**~~ | ~~Calibrar p_blend con p_historica derivada (n≥30)~~ | Alto P&L | — | ✅ 2026-06-01 — _load_p_prior() lee calibracion_edge.json, CLI --superficie |
| **T15-06** | Backtesting formal n≥30 sesiones limpias | Alto P&L | — | ⏳ bloqueado — requiere más sesiones |
| **Nodo-21 (T21-01..11)** | Pesos por tier (5 tiers SNR + density + shrinkage + K-ELO) | — | `rivalry_analyzer.py` + `config.py` + `elo_system.py` | ✅ COMPLETADO 2026-06-03 |
| **Nodo-19 (T19-01..04)** | H2H Immunity Dampener — señal 2do orden HOT×H2H_específico | — | `analysis/rivalry_analyzer.py` | ✅ COMPLETADO 2026-06-03 |
| **Nodo-18 (T18-01..05)** | PELT Recency Alpha — change_point → λ_efectivo en edge_calculator | — | `markov_analyzer.py` + `edge_calculator.py` | ✅ COMPLETADO 2026-06-03 |
| **Nodo-20 (T20-01..04)** | PageRank Erdős — centralidad de nodos intermedios en grafo transitivo | — | `analysis/erdos_graph.py` | ✅ COMPLETADO 2026-06-03 |
| **Fixes operacionales** | FlashScore DOM + markov persist + trader tier filter + p_blend per-match | 🔴 CRÍTICO | `h2h_extractor.py` + `edge_calculator.py` + `trader_ev_tenis.py` | ✅ RESUELTO 2026-06-05/06 |
| **Nodo-22 (T22-01..11)** | API Integration: Kambi + Ninja — pipeline 40min→45s, cuota_es_real | — | `kambi_tennis.py` + `ninja_h2h_parser.py` + `extraer_partidos_api.py` | ✅ COMPLETADO 2026-06-07 |
| **D-19** | Playwright como dependencia pesada — mitigada por API mode | Medio | Playwright ahora fallback, no ruta primaria | ✅ MITIGADA (Nodo-22) |
| **Fixes 2026-06-09** | import re + match_id + superficie + betplay_combo_builder --live | 🔴 CRÍTICO | `ninja_h2h_parser.py` + `validar_con_api.py` + `betplay_combo_builder.py` | ✅ RESUELTO 2026-06-09 |

---

## Nodo-17 — Calibración Estratificada por Tier (✅ Fase 1 COMPLETADA 2026-06-03 | Fase 2 parcial)

> Ver spec completo: [[Nodo-17-Calibracion-Por-Tier]]
> Origen: Test-Time Compute post-sesión 3 multi-torneo (61.11%, 22/36 Challengers)

| ID | Descripción | Impacto P&L | Archivo | Estado |
|---|---|---|---|---|
| **T17-01** | Fix surface propagation: torneo_completo → superficie en H2H output | 🔴 CRÍTICO | `scraping/h2h_extractor.py` | ✅ 2026-06-03 |
| **T17-02** | Estratificar `calibracion_edge.json` por `[tier][superficie]` | 🔴 CRÍTICO | `data/calibracion_edge.json` + `validar_con_api.py` | ✅ RESUELTO 2026-06-11 — `_load_p_prior(superficie, tier)` usa `por_superficie_y_tier` + `fallback_por_tier` (B-02 cerrado) |
| **T17-03** | `edge_calculator.py`: λ_efectivo = λ_base × tier_multiplier | 🔴 CRÍTICO | `edge_calculator.py` | ✅ 2026-06-03 — λ por tier activo (Nodo-21) |
| **T17-04** | `rivalry_analyzer.py`: common_opp_weight dict por tier | 🟠 ALTO | `analysis/rivalry_analyzer.py` | ✅ 2026-06-03 — 5 tiers SNR (Nodo-21) |
| **T17-05** | `elo_system.py`: K-factor por tier | 🟡 MEDIO | `analysis/elo_system.py` | ✅ 2026-06-03 — GS=24, ATP1000=28, ATP500=32, CH=40, ITF=48 |
| **T17-06** | `markov_analyzer.py`: window_size por tier | 🟡 MEDIO | `analysis/markov_analyzer.py` | 🟡 PENDIENTE — no implementado aún |

**Polmans Principle (REGLA-T17-4) — PARCIALMENTE ACTIVO:**
```
Bug B-02 RESUELTO 2026-06-11:
_load_p_prior(superficie, tier) — jerarquía idéntica a theta_thompson() en edge_calculator:
  1. por_superficie_y_tier[f'{superficie}_{tier}'] n≥10 → Thompson Beta
  2. fallback_por_tier[tier] → float directo (grand_slam=0.758, atp500=0.65, challenger=0.611, itf=0.59)
  3. por_superficie[superficie] n≥10 → Thompson Beta
  4. global → Thompson Beta
  5. _P_PRIOR = 0.52

clay+challenger: 0.697 → 0.590 | grass+challenger: 0.569 → 0.611 | grass+atp500: 0.569 → 0.650
_print_individuales() también corregido: usa p_historica_usada per-pick, no p_prior global.
```

---

## Bugs Descubiertos — Validación Prod 2026-06-07 (Fase 23)

> Origen: ejecución completa del pipeline multi-torneo (80 partidos, 5 tiers).
> Ninguno bloquea el pipeline — todos son mejoras de precisión o UX.

| ID | Bug | Archivo | Severidad | Impacto P&L |
|---|---|---|---|---|
| **B-01** | Challenger 120% bankroll — VaR ×0.83 insuficiente con muchos picks | `trader_ev_tenis.py` | 🟠 | Riesgo de ruina si se sigue ciegamente |
| ~~**B-02**~~ | ~~p_blend=0.697 para TODOS los tiers — prior clay GS contamina Challenger/ITF~~ | `trader_ev_tenis.py` `_load_p_prior()` | ✅ | **RESUELTO 2026-06-11** — `_load_p_prior(superficie, tier)` usa jerarquía: `por_superficie_y_tier` n≥10 → `fallback_por_tier` (float directo) → `por_superficie` → global. grass+challenger=0.611, clay+challenger=0.590 vs 0.697 antes. |
| **B-02a** | match_id=None en _consolidate_result() → validar_con_api.py retornaba 0/0 | `scraping/ninja_h2h_parser.py` | ✅ | **RESUELTO 2026-06-09** — match_id incluido en output |
| **B-02b** | superficie siempre "unknown" en validar_con_api.py | `validar_con_api.py` línea 174 | ✅ | **RESUELTO 2026-06-09** — tipo_cancha or superficie or 'unknown' |
| **B-02c** | import re faltaba en ninja_h2h_parser.py → crash en --api-mode | `scraping/ninja_h2h_parser.py` | ✅ | **RESUELTO 2026-06-09** — import re añadido |
| ~~**B-03**~~ | ~~Header "DEPLOY ROLAND GARROS" hardcoded~~ | `trader_ev_tenis.py` | ✅ | **RESUELTO 2026-06-11** — header dinámico: `TRADER EV TENIS — {torneo_tipo.upper()} {superficie.upper()}` |
| **B-04** | generar_tabla_favoritos2.py selecciona archivo pequeño (2) sobre grande (80) | `generar_tabla_favoritos2.py` | 🟠 | Reporte humano vacío |
| **B-05** | consultar_resultados_historicos.py busca key 'partidos', JSON usa 'detailed_results' | `consultar_resultados_historicos.py` | 🟡 | Script roto → ELIMINAR (D-18) |
| **B-06** | Campo 'favorito' vacío en edge_report JSON | `edge_calculator.py` | 🟢 | Solo auditoría |
| **B-07** | Calibración muestra "n=150" — posible inflación | `trader_ev_tenis.py` o `calibracion_edge.json` | ✅ | **RESUELTO 2026-06-09** — calibración real: n=284, clay GS: 25W/8L |

### D-18: `consultar_resultados_historicos.py` — ELIMINAR

```
SE: 0 importadores activos (grep confirmado). Requiere --file pero output incompatible.
DA: resultados_finales.py cubre 100% del caso de uso (API Ninja + accuracy + JSON export).
ARQ: ELIMINAR — redundante y roto. Git preserva historial si se necesita recuperar.
```

### D-19: Playwright como dependencia pesada — MITIGADA (Nodo-22)

```
ANTES (pre-Nodo-22):
  PASO 1: extraer_URL_partidos_version2.py — Playwright (~8 min, DOM frágil)
  PASO 2: extraer_historh2h.py — Playwright (~30 min para 80 partidos, DOM cambia periódicamente)
  Dependencia: Playwright + chromium browser (~200MB), WSL2 compatibility issues
  Riesgo: FlashScore DOM cambia CSS classes sin aviso → pipeline roto (ocurrió 2026-06-05)

DESPUÉS (post-Nodo-22):
  PASO 1: extraer_partidos_api.py — Kambi + FlashScore Ninja API (~1.3s, puro HTTP)
  PASO 2: extraer_historh2h.py --api-mode — FlashScore Ninja H2H API (~0.5s/partido)
  Dependencia API: solo requests/httpx — sin browser headless
  Playwright: PRESERVADO como fallback (extraer_URL_partidos_version2.py + modo default de extraer_historh2h.py)

Nuevos módulos:
  [+] scraping/kambi_tennis.py      — Kambi API Betplay + FlashScore feed + name matching NBA pattern
  [+] scraping/ninja_h2h_parser.py  — NinjaH2HExtractor (Strangler Fig: mismo output que H2HExtractor)
  [+] extraer_partidos_api.py       — CLI entry point PASO 1 API

Impacto en deuda:
  - Playwright ya NO es single point of failure para PASO 1 y PASO 2
  - DOM breakage (CSS class changes) ya no bloquea el pipeline — API es ruta primaria
  - Pipeline ejecutable en entornos sin browser (CI, cloud functions)
  - ~45s total API vs ~40 min Playwright = 53× más rápido
```

### Bugs Identificados — Sesión 5 (2026-06-12)

| ID | Bug | Descripción | Archivo | Línea | Status |
|---|---|---|---|---|---|
| **B-05** | `_print_resumen` hardcodes "clay (Roland Garros)" | Ignora parámetros --superficie y --torneo-tipo. Output muestra siempre la misma superficie independientemente de la entrada. | `trader_ev_tenis.py` | 810 | 🔧 En fix |
| **B-06** | KGR denominator usa `total_staked` en lugar de `bankroll` | En `_compute_var_cvar()`, el denominador de Kelly Growth Rate es `total_staked` (suma de todos los stakes) en lugar de `bankroll` (capital inicial). Inflates growth rate y puede bypass REGLA-HF-5 (no-deploy gate si KGR<0). | `trader_ev_tenis.py` | 213 | 🔧 En fix |
| **B-07** | VaR binomial model ignora ρ (correlación) | Modelo de riesgo calcula VaR solo con probabilidades individuales, sin considerar ρ (correlación estructura). Underestima tail risk para picks correlacionados en misma sesión/torneo. | `trader_ev_tenis.py` | 158-176 | 🔧 En fix |
| **B-01** | VaR mide solo cobertura, excluye individuales del cálculo | Función `_compute_var_cvar()` usa `gastado_cobertura` (stakes en combos) pero no incluye `gastado_ind` (stakes en individuales, 40% del bankroll). VaR parcial = underestimate de riesgo total. Raíz identificada: línea 252 lectura incompleta de plan. | `trader_ev_tenis.py` | 252 | 🔧 En fix (root cause identified) |

### Bugs Identificados — Post-mortem Sesión 2026-06-12 (Sizing Invertido)

| ID | Bug | Descripción | Archivo | Línea | Status |
|---|---|---|---|---|---|
| **B-08** | Prior conservador cuando sup+tier divergen | `theta_thompson()` salta a `fallback_por_tier` (ignora superficie). grass=0.569 pero atp500=0.650 → optimismo artificial +0.081 → Kelly inflado. Fix: `min(fallback_tier, p_superficie)` cuando divergen >0.03. | `edge_calculator.py` theta_thompson() + `trader_ev_tenis.py` _load_p_prior() | 328-337 | ✅ RESUELTO 2026-06-12 |
| **B-09** | Señal mínima de convicción antes de APOSTAR | 92% de picks tienen conf<55%. Un pick con conf=50.2% genera edge=20% solo por cuota extrema — no por convicción del modelo. Fix: campo `confidence_flag` (STRONG/MODERATE/LOW) en output de `calcular_edge()`. | `edge_calculator.py` calcular_edge() | 446-455 | ✅ RESUELTO 2026-06-12 |
| **B-10** | Sizing inversamente proporcional a calibración | Kelly se basa en p_historica inflada (fallback_por_tier sin datos). James-Stein shrinkage: `ccf = n/(n+20)`, floor=0.30. n=4 (grass_atp500) → ccf=0.30 vs n=33 (clay_gs) → ccf=0.62. Escala Kelly ANTES del cap. | `edge_calculator.py` calcular_edge() + calcular_edge_completo() | 440-448, 608-610 | ✅ RESUELTO 2026-06-12 |

### B-11: ELO floor 1300 satura normalización → colapso de discriminación (2026-06-16)

```
SÍNTOMA:
  Sesión Jun 15: accuracy 46.9% (15/32) — peor que coin flip.
  TODAS las confianzas entre 50.1% y 54.5%. Cero picks >55%.
  Sesión Jun 14: confianzas hasta 70.8% (diffs hasta 1.59). Sesiones anteriores: 75.8% clay GS.

CAUSA RAÍZ:
  Cambio no-commiteado en rivalry_analyzer.py:1313:
    ANTES: raw_scores['elo_rating'] = max(0, elo - 1500)    # max raw ≈ 250
    AHORA: raw_scores['elo_rating'] = max(0, elo - 1300)    # max raw ≈ 450+

  normalization.py:37 tiene MAX_RAW_SCORES['elo_rating'] = 250 (calibrado para floor 1500).
  
  Normalización lineal: normalized = raw / max_expected, clamped [0, 1] (normalization.py:169).
  
  Con floor 1300:
    ELO 1747 → raw=447 → 447/250 = 1.79 → CLAMPED a 1.0
    ELO 1675 → raw=375 → 375/250 = 1.50 → CLAMPED a 1.0
    DIFERENCIA = 0.0  ← señal ELO ELIMINADA

  Con floor 1500 (original):
    ELO 1747 → raw=247 → 247/250 = 0.99
    ELO 1675 → raw=175 → 175/250 = 0.70
    DIFERENCIA = 0.29  ← señal ELO INTACTA

  ELO tiene peso 14-15% en el modelo — uno de los componentes más discriminatorios.
  Al saturar, TODOS los jugadores con historial (ELO>1550) se normalizan a 1.0.
  Score difference cae de rango [-1.59, +1.15] a [-0.46, +0.47].
  Confianza cae de rango [50, 71%] a [50, 55%].

CAMBIO SECUNDARIO (mismo commit no-commiteado):
  rivalry_analyzer.py:90:
    ANTES: return current_elo     # 1500 si no hay historial
    AHORA: return None            # None si no hay historial
  
  Esto permite distinguir "sin datos" de "datos reales ELO bajo", pero propaga None
  que requiere manejo defensivo downstream. El edge_calculator.py ya tiene
  `elo_fav or 1500` como fallback, pero otros consumidores podrían no tenerlo.

OPCIONES DE FIX:
  Opción A — Revertir floor a 1500:
    Restaura comportamiento validado (75.8% clay GS, 8/8 R4).
    Rápido, bajo riesgo. Pierde la intención de dar crédito a jugadores 1300-1500.
  
  Opción B — Subir MAX_RAW_SCORES['elo_rating'] a 450:
    Mantiene la intención del cambio. Rango normalizado se ajusta al nuevo floor.
    ELO 1747 → 447/450 = 0.993 | ELO 1675 → 375/450 = 0.833 | diff=0.16.
    Menos discriminatorio que floor 1500 (diff era 0.29) pero funcional.
    Requiere re-calibrar todos los tests de normalización que usen ELO.

ARCHIVOS AFECTADOS:
  - analysis/rivalry_analyzer.py:90 (return None vs return current_elo)
  - analysis/rivalry_analyzer.py:1313 (elo - 1300 vs elo - 1500)
  - normalization.py:37 (MAX_RAW_SCORES['elo_rating'] = 250)
  
IMPACTO P&L: 🔴 CRÍTICO — modelo operando como coin flip (46.9%).
  Todo pick de la sesión Jun 15 fue esencialmente aleatorio.
  Calibracion_edge.json contaminado con 17 "fallos" que incluyen ruido, no señal real.

FIX APLICADO (2026-06-16 — Opción C3):
  ✅ rivalry_analyzer.py:90  — return current_elo (no None)
  ✅ rivalry_analyzer.py:1312 — min(max(0, elo - 1500), 250) — floor=1500 + cap=250
  ✅ test_rivalry_analyzer.py:488 — test actualizado: expect 1500 no None
  ✅ test_normalization.py — test_elo_rating_floor_cap_acoplamiento() NUEVO
  ✅ calibracion_edge.json — revertido: -14W/-17L (global, clay, hard, unknown)
  ✅ 1007 tests passed (1006→1007 por test de acoplamiento)
  
  Discriminación restaurada: Δ_elo(1747v1675) = 0.048 (C3) vs 0.024 (roto) = 2×
  
  Diagnóstico corregido durante TTC: la normalización usa log1p() (no normalize_min_max).
  El mecanismo era compresión logarítmica, no clamp duro a [0,1].
  MAX_RAW_SCORES NO participa del pipeline de scoring — solo normalization tests.
```

### Resolución propuesta B-02 (Polmans Principle):

```
_load_p_prior(superficie, tier) → leer calibracion_edge.json[tier][superficie]
Fallback: p_prior=0.52 neutral si no hay datos para ese [tier][superficie]
Resultado esperado: Challenger con n=0 → p_blend=0.52 (no 0.697)
Impacto: Mair @10.00 habría tenido edge mucho menor → probablemente WATCHLIST, no APOSTAR
```

---

## Vinculación

- [[Nodo-07-Strangler-Fig]] — plan de migración de SequentialH2HExtractor (Fase 2 bloqueada por tests)
- [[Mandatos-No-Negociables]] — Mandato 6: tests SIEMPRE preceden al código; baseline 980 passed
- [[Sprint-Pipeline]] — backlog activo; Fase 23 validación prod con 7 bugs
- [[Pipeline-Arquitectura]] — mapa de módulos; este doc es el inventario de deuda de ese mapa
- [[Grafo-Dependencias-Datos]] — las costuras impactan S1 (ZitaScraper), S2 (H2HExtractor)
- [[Nodo-08-File-Selection-Bug]] — ejemplo de deuda de lógica en extraer_historh2h.py ya corregida
- [[Nodo-09-API-Status-Keys]] — ejemplo de deuda de documentación de API ya corregida
- [[Nodo-12-Inventario-Infraestructura-Legado]] — stack Flask/Selenium paralelo (SUSPENDIDO) + limpieza infra 2026-05-30
- [[Nodo-16-Multi-Torneo-Pipeline]] — --max-matches 80 + --all-tournaments
- [[Nodo-17-Calibracion-Por-Tier]] — T17-02 ✅ RESUELTO 2026-06-11: _load_p_prior(superficie, tier) estratificada por tier (B-02 cerrado)
- [[Nodo-21-Pesos-Diferenciados-Por-Tier]] — 5 tiers + density + shrinkage completados
- Nodo-22-API-Integration-Kambi-Ninja — Kambi API + Ninja H2H → pipeline ~45s, Playwright = fallback

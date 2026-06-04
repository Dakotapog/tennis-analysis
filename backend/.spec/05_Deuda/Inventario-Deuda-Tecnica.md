# Inventario de Deuda Técnica — Tennis Prediction Engine

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Grafo-Dependencias-Datos]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Nodo-12-Inventario-Infraestructura-Legado]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-20-PageRank-Erdos-Quality]]
> Estado: 2026-06-03 | Auditoría: completa (raíz + subdirectorios + reports/ + screenshots/)
> Actualización: 2026-06-03 — Nodo-17 Fase 1 ✅ | Nodo-18/19/20 documentados (TTC)
> Deuda activa restante: Nodo-19 (H2H Immunity) | Nodo-18 (PELT Recency) | Nodo-20 (PageRank) | T15-06 backtesting
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

### Costura 3 — ZitaScraper (FUTURO — después de validar scraper en prod)

```
ANTES:
extraer_URL_partidos_version2.py   → class ZitaScraper (604 líneas, con 3 bugs del Nodo-03)
extraer_URL_partidos.py            → class ZitaScraper (VERSIÓN ANTIGUA)
extraer_cuotas_partidos.py         → class ZitaScraper (VARIANTE odds)
extraer_URL_partidos_en_vivo.py    → class ZitaScraper (VARIANTE live — propósito unclear)

DESPUÉS (Nodo-10 pendiente):
scraping/url_scraper.py → class ZitaScraper (única, con todos los fixes del Nodo-03)
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
| **Nodo-21 (T21-01..11)** | Pesos por tier (bug fix classify_tournament GS + 5 tiers SNR + density + shrinkage + K-ELO) | 🔴 CRÍTICO — bug GS activo | `rivalry_analyzer.py` + `config.py` + `normalization.py` + `elo_system.py` | 🔴 PENDIENTE — implementar PRIMERO |
| **Nodo-19 (T19-01..04)** | H2H Immunity Dampener — señal 2do orden HOT×H2H_específico | 🔴 ALTO — previene error activo | `analysis/rivalry_analyzer.py` | 🔴 PENDIENTE — segundo |
| **Nodo-18 (T18-01..05)** | PELT Recency Alpha — change_point → λ_efectivo reducido en ventana bookmaker stale | 🟠 ALTO — amplifica alpha temporal | `markov_analyzer.py` + `edge_calculator.py` | 🔴 PENDIENTE — implementar segundo |
| **Nodo-20 (T20-01..04)** | PageRank Erdős — centralidad de nodos intermedios en grafo transitivo | 🟡 MEDIO — refinamiento Erdős | `analysis/erdos_graph.py` | 🔴 PENDIENTE — implementar tercero |

---

## Nodo-17 — Calibración Estratificada por Tier (🔴 ACTIVO 2026-06-02)

> Ver spec completo: [[Nodo-17-Calibracion-Por-Tier]]
> Origen: Test-Time Compute post-sesión 3 multi-torneo (61.11%, 22/36 Challengers)

| ID | Descripción | Impacto P&L | Archivo | Estado |
|---|---|---|---|---|
| **T17-01** | Fix surface propagation: torneo_completo → superficie en H2H output (multi-torneo) | 🔴 CRÍTICO | `scraping/h2h_extractor.py` | 🔴 PENDIENTE |
| **T17-02** | Estratificar `calibracion_edge.json` por `[tier][superficie]` — separar GS de Challenger | 🔴 CRÍTICO | `data/calibracion_edge.json` + `validar_con_api.py` | 🔴 PENDIENTE |
| **T17-03** | `edge_calculator.py`: λ_efectivo = λ_base × tier_multiplier (0.5→1.8 challenger) | 🔴 CRÍTICO | `edge_calculator.py` | 🔴 PENDIENTE |
| **T17-04** | `rivalry_analyzer.py`: common_opp_weight dict por tier (0.28 GS → 0.12 Challenger) | 🟠 ALTO | `analysis/rivalry_analyzer.py` | 🟡 BLOQUEADO (n<10 por tier) |
| **T17-05** | `elo_system.py`: K-factor por tier (32 GS / 16 Challenger) | 🟡 MEDIO | `analysis/elo_system.py` | 🟡 BLOQUEADO |
| **T17-06** | `markov_analyzer.py`: window_size por tier (más corto en Challengers) | 🟡 MEDIO | `analysis/markov_analyzer.py` | 🟡 BLOQUEADO |

**Polmans Principle (REGLA-T17-4):**
```
Underdog @5.00 en Challenger con surface desconocida = NO APOSTAR.
Condición mínima para apostar en Challenger:
  1. superficie confirmada (T17-01 resuelto)
  2. λ_challenger aplicado (T17-03 resuelto)
  3. p_prior[tier][superficie] (T17-02 resuelto)
```

---

## Vinculación

- [[Nodo-07-Strangler-Fig]] — plan de migración de SequentialH2HExtractor (Fase 2 bloqueada por tests)
- [[Mandatos-No-Negociables]] — Mandato 6: tests SIEMPRE preceden al código; baseline 767 passed
- [[Sprint-Pipeline]] — backlog activo; D-01 a D-08 son el próximo sprint
- [[Pipeline-Arquitectura]] — mapa de módulos; este doc es el inventario de deuda de ese mapa
- [[Grafo-Dependencias-Datos]] — las costuras impactan S1 (ZitaScraper), S2 (H2HExtractor)
- [[Nodo-08-File-Selection-Bug]] — ejemplo de deuda de lógica en extraer_historh2h.py ya corregida
- [[Nodo-09-API-Status-Keys]] — ejemplo de deuda de documentación de API ya corregida
- [[Nodo-12-Inventario-Infraestructura-Legado]] — stack Flask/Selenium paralelo (SUSPENDIDO) + limpieza infra 2026-05-30

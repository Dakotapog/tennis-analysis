# Inventario de Deuda Técnica — Tennis Prediction Engine

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Grafo-Dependencias-Datos]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Nodo-12-Inventario-Infraestructura-Legado]]
> Estado: 2026-05-30 | Auditoría: completa (raíz + subdirectorios + reports/ + screenshots/)
> Actualización: 2026-05-30 — D-13 ELIMINADO ✅ | T06-03/04 ✅ | T12-A ✅ | T14-03 ✅ | T07-09 ✅ (1,404 líneas eliminadas, 768 tests)
> Deuda principal activa: D-14 CERRADO ✅ — extraer_historh2h.py = 310 líneas (entry point puro) | Próximo: T14-02 factor_tardio
> Metodología: Strangler Fig + Spec-Driven Development + TTC (Marco de Tres Expertos)

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
| `generar_tabla_favoritos2.py` | 1,048 | 0 | ~15 tests | MEDIO |
| `generar_dataset_plus.py` | 1,661 | 14 | más tests de integración | MEDIO |
| `Intelligent_ml_enhancer.py` | 1,350 | 0 | ~20 tests | MEDIO — usada por aplicar_enhancer |
| `aplicar_enhancer.py` | 1,005 | 0 | ~10 tests | MEDIO |
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

#### D-15: `extraer_URL_partidos_en_vivo.py` — propósito sin documentar

Script de 237 líneas con `ZitaScraper` variante "en_vivo". No importado por nadie.
¿Es la versión para partidos LIVE? ¿Experimental? Debe documentarse o eliminarse.

#### D-16: `prueba.py` — 1,289 líneas de "trabajo en progreso"

Archivo de experimentación con código sin estructura. No importado, no testeado.
Decisión pendiente: extraer lo útil → eliminar.

#### D-17: Configuración dispersa (sin `config.py`)

Constantes críticas dispersas en múltiples archivos:
- `FLASHSCORE_BASE`, `HEADERS` → `validar_con_api.py`
- `MAX_RAW_SCORES`, `DEFAULT_WEIGHTS` → `normalization.py`
- `total_matches_to_process = 80` → `extraer_historh2h.py` línea 262 (hardcoded)
- `headless=True`, `slow_mo=50` → `extraer_historh2h.py` (hardcoded)

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

| ID | Tarea | Impacto P&L | Esfuerzo | Precondición |
|---|---|---|---|---|
| **D-01 a D-08** | Eliminar 8 archivos vacíos/backup | Claridad | 30 min | Ninguna |
| **D-10/D-11/D-12** | Eliminar ranking/URL scrapers v1 | Claridad | 30 min | Verificar imports |
| **D-09** | Extraer SmartLogger a `utils/logger.py` | Mantenimiento | 1h | Ninguna |
| **Nodo-07 Fase 2 prep** | Ampliar test_h2h_extractor.py: 5→40 tests | Alto | 4h | Ninguna |
| **Nodo-07 Fase 2** | Migrar SequentialH2HExtractor → H2HExtractor | Alto | 4-8h | 40 tests |
| **D-13** | Eliminar generar_tabla_favoritos.py v1 | Claridad | 15 min | Validar v2 en prod |
| **D-15** | Documentar o eliminar `extraer_URL_partidos_en_vivo.py` | Claridad | 1h | Ninguna |
| **D-16** | Liquidar `prueba.py` | Claridad | 2h | Ninguna |
| **D-17** | Centralizar config en `config.py` | Mantenimiento | 2h | Ninguna |
| **Nodo-10 (futuro)** | Unificar ZitaScraper → `scraping/url_scraper.py` | Medio | 4h | Nodo-03 prod validado |

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

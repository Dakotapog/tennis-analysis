# Atlas: Arquitectura del Pipeline

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Grafo-Dependencias-Datos]] | [[Sprint-Pipeline]] | [[Fuentes-Datos]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-13-Trader-EV-Tenis]] | [[Nodo-14-Validacion-Live-Conexiones]]
> Estado: 2026-05-30 | Roland Garros — Primera validación live ✅ | 768 tests passing | T07-09 ✅ SequentialH2HExtractor eliminado — extraer_historh2h.py 310 líneas

---

## Orden de Ejecución (Pipeline Completo)

```
PASO 0a: extraer_ranking_atp_version2.py   ← PRIMERO (rankings frescos)
         → data/atp_rankings_complete_FECHA.json
         Clase: CompleteRankingScraper | URL: /rankings/atp-live/

PASO 0b: extraer_ranking_wta_version2.py   ← PRIMERO (rankings frescos)
         → data/wta_rankings_complete_FECHA.json
         Clase: CompleteRankingScraper | URL: /rankings/wta-live/

         ✅ Solo v2 existe: extraer_ranking_atp.py y extraer_ranking_wta.py (v1) ELIMINADOS 2026-05-29

PASO 1:  extraer_URL_partidos_version2.py  ← partidos del día (~2 min)
         → data/zita_tennis_matches_FECHA.json
         Clase: ZitaScraper
         Fix T03-01 ✅: h2h_url = match_url.split('?')[0] + '/#/h2h/overall/'
         Fix T03-02 ✅: match_id = re.search(r'[?&]mid=([^&]+)', match_url).group(1)
         Fix T03-03 ✅: selector .headerLeague__title (no event__header/event__title)
         Fix T03-04 ✅: superficie extraída desde texto "..., arcilla" del header

PASO 2:  extraer_historh2h.py              ← orquestador principal (~30-60 min)
         → reports/h2h_results_enhanced_FECHA.json
         ⚠️ ARQUITECTURA CRÍTICA: ver sección debajo

PASO 3:  edge_calculator.py                ← Kelly-KL + señales de apuesta
         → reports/edge_report_FECHA.json

PASO 3.5: trader_ev_tenis.py              ← Capa deploy: individuales + combos + sistema
         Lee: edge_report_FECHA.json
         python trader_ev_tenis.py --bankroll 100000 --combos 3 --sistema 4
         → output consola con plan de apuesta (3 capas + budget cascade)
         → activa combos cuando ≥N señales APOSTAR (hoy: 2 señales → 1 combo)

PASO 3b: resultados_finales.py             ← labels post-partido (Playwright)
         O validar_con_api.py              ← labels en <1 seg (FlashScore dc_1)
         → reports/resultados_finales_FECHA.json

PASO 4:  generar_tabla_favoritos2.py       ← reporte humano (fix T06-02 ✅)
         → analisis_partidos_pandas.txt
```

---

## Arquitectura Actual: extraer_historh2h.py (post-Nodo-07 Fase 2 COMPLETA ✅)

**Estado 2026-05-30 post T07-09:** Entry point puro de 310 líneas. `SequentialH2HExtractor` ELIMINADA.

```
extraer_historh2h.py (v5.0 — Nodo-07 Fase 2 COMPLETA, 310 líneas):
  imports + helpers         → find_all_json_files, select_best_json_file (sin cambios)
  main()                    → usa H2HExtractor (scraping/) ✅

Importaciones activas:
  from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer  ✅ (Fase 1)
  from scraping.h2h_extractor import H2HExtractor  ✅ (Fase 2)
  from extraer_ranking_atp_version2 import CompleteRankingScraper

SequentialH2HExtractor: ELIMINADA 2026-05-30 (T07-09) — 1,404 líneas liquidadas
  tests/test_sequential_h2h_extractor.py: 53 tests migrados → 48 en H2HExtractor/DataParser
```

**Evolución del archivo (Strangler Fig completo):**
```
Original (pre-Nodo-07):     3,717 líneas  — 4 clases inline duplicadas
Post-Fase-1 (2026-05-29):   1,707 líneas  — clases inline → imports analysis/
Post-Fase-2 (2026-05-30):     310 líneas  — SequentialH2HExtractor eliminada (T07-09)
Reducción total:              −92%
```

Ver [[Nodo-07-Strangler-Fig]] para detalle completo de la migración.

---

## Mapa de Módulos y Tests

### Paquete `scraping/` — Refactorización modular

| Módulo | Responsabilidad | Test | n tests |
|---|---|---|---|
| `scraping/browser_manager.py` | Playwright WSL-optimized | `tests/extractor/test_browser_manager.py` | 10 |
| `scraping/data_parser.py` | Parse HTML → datos limpios | `tests/extractor/test_data_parser.py` + `tests/test_data_parser.py` | 31+67 |
| `scraping/file_utils.py` | Selección de JSON más reciente | `tests/extractor/test_file_utils.py` | 32 |
| `scraping/h2h_extractor.py` | Orquestador modular (refactorizado) | `tests/test_h2h_extractor.py` | 5 ⚠️ |
| `scraping/__init__.py` | Exports del paquete | — | — |

### Paquete `analysis/` — Motor de predicción

| Módulo | Responsabilidad | Test | n tests |
|---|---|---|---|
| `analysis/elo_system.py` | Rating ELO por jugador | `tests/test_elo_system.py` | 3 ⚠️ |
| `analysis/ranking_manager.py` | Rankings ATP/WTA + métricas | `tests/test_ranking_manager.py` | 90 |
| `analysis/rivalry_analyzer.py` | Motor de predicción (8 componentes) | `tests/test_rivalry_analyzer.py` + `tests/test_rivalry_analyzer_regression.py` | 138+22 |
| `analysis/erdos_graph.py` | Grafo transitivo Erdős (Nodo-06) | `tests/test_erdos_graph.py` | 37 |
| `analysis/markov_analyzer.py` | PELT + Cadenas de Markov (Nodo-02) | `tests/test_markov_analyzer.py` | 37 |
| `analysis/__init__.py` | Exports: EloRatingSystem, RankingManager, RivalryAnalyzer | — | — |

### Scripts principales (raíz)

| Script | Responsabilidad | Test | n tests |
|---|---|---|---|
| `extraer_ranking_atp_version2.py` | Rankings ATP live | — | 0 ⚠️ |
| `extraer_ranking_wta_version2.py` | Rankings WTA live | — | 0 ⚠️ |
| `extraer_URL_partidos_version2.py` | URLs + match_id + superficie | `tests/test_url_scraper_output.py` | 20 |
| `extraer_historh2h.py` | Orquestador H2H + predicción | `tests/test_sequential_h2h_extractor.py` | 52 ✅ |
| `edge_calculator.py` | Kelly-KL + edge (Nodo-01) | `tests/test_edge_calculator.py` | 43 |
| `trader_ev_tenis.py` | Deploy: individuales + combos + sistema (Nodo-13) | — | 0 ⚠️ |
| `validar_con_api.py` | Labels FlashScore API (Nodo-05) | `tests/test_validacion_api.py` | 37 |
| `normalization.py` | MAX_RAW_SCORES + DEFAULT_WEIGHTS | `tests/test_normalization.py` | 62 |
| `generar_dataset_plus.py` | Dataset ML (Nodo-04, fix ✅) | `tests/test_dataset_generator.py` | 14 |
| `utils/logger.py` | SmartLogger — fuente única de verdad (D-09) | `tests/test_utils_logger.py` | 19 ✅ |

**Total: 773 tests, 0 failed** (2026-05-30)

---

## Diagrama de Flujo de Datos

```
FUENTES DE DATOS
════════════════
FlashScore.com (Playwright)           FlashScore Ninja API
       │                                      │
  Paso 0: Rankings                     dc_1_{event_id}
  CompleteRankingScraper                → score en tiempo real
  atp/wta_rankings_complete_FECHA.json  → labels post-partido
       │
  Paso 1: URLs + cuotas + superficie
  ZitaScraper → .headerLeague__title ✅
  zita_tennis_matches_FECHA.json
  235 partidos | 33 torneos | superficie=clay/hard/grass ✅
       │
  Paso 2: H2H + Predicción (~60 min)
  extraer_historh2h.py (post-Nodo-07 Fase 1 ✅)
  ├── CompleteRankingScraper (ranking en vivo)
  └── RivalryAnalyzer desde analysis/ (Markov+Erdős activos)
  h2h_results_enhanced_FECHA.json
  prediccion en: ranking_analysis.prediction.favored_player
       │
       ├─────────────────────────┐
       │                         │
  Paso 3a: Edge                  Paso 3b: Labels
  edge_calculator.py             validar_con_api.py
  Kelly-KL + P_implícita         dc_1_{event_id} <1 seg
  → edge_report_FECHA.json       → accuracy por superficie
       │                         │
  Paso 3.5: Deploy               │
  trader_ev_tenis.py             │
  Individuales+Combos+Sistema    │
  Budget cascade 40/40/20        │
  → plan de apuesta por consola  │
       │                         │
       └─────────────────────────┘
                │
         Paso 4: Reporte
         generar_tabla_favoritos2.py
         → analisis_partidos_pandas.txt
```

---

## Componentes del Motor de Predicción (rivalry_analyzer.py)

```
Componente              Peso   Estado 2026-05-30
─────────────────────────────────────────────────────────────
surface_specialization  15%    ✅ ACTIVO en prod — surf_w 0.49–0.69 (Nodo-10 RESUELTO)
form_recent             15%    ✅ + factor Markov HOT/COLD (Nodo-02)
common_opponents        20%    ✅ + Erdős graph depth≥2 (erdos_score=0.35 en prod)
h2h_direct              15%    ✅ funcionando
ranking_momentum        20%    ✅ prox_points + max_points (v2 scraper)
elo_rating              10%    ✅ funcionando
home_advantage           5%    ✅ ACTIVO (torneo con país real)
strength_of_schedule     0%    ⚠️ calculado pero peso=0 (oportunidad futura)

TOTAL                  100%  — TODOS LOS COMPONENTES ACTIVOS ✅

✅ POST-NODO-07 + NODO-10: surface, Markov y Erdős confirmados en producción 2026-05-30.
   Run: 16 partidos Roland Garros | h2h_results_enhanced_20260530_053518.json
```

---

## Jerarquía de Lectura de Predicción

```python
# CORRECTO — donde vive la predicción
partido['ranking_analysis']['prediction']['favored_player']
partido['ranking_analysis']['prediction']['confidence']
partido['ranking_analysis']['erdos_analysis']['erdos_score']
partido['ranking_analysis']['markov_analysis']['factor_markov']

# INCORRECTO — siempre None
partido['prediccion_ganador']  # ← no usar
```

---

## Métricas de Salud del Sistema

| Métrica | Objetivo | 2026-05-30 |
|---|---|---|
| Accuracy general | > 55% | ✅ **70% (7/10)** datos limpios — Roland Garros 2026-05-30 (era 47.37% datos sucios) |
| surface_specialization activo | > 0% | ✅ surf_w 0.49–0.69 en 16/16 |
| Torneos detectados correctamente | > 1 | ✅ Roland Garros clay confirmado |
| match_id real | ≠ "tennis" | ✅ IDs reales |
| h2h_url generada | 100% | ✅ |
| erdos_score en output | presente | ✅ 0.35 en prod |
| factor_markov en output | presente | ✅ activo |
| Señales APOSTAR edge > 5% | ≥1 por sesión | ✅ Parry +29.3%, Tien +16.5% (2 en 16 partidos) |
| P&L primera sesión | > 0 | ✅ **+$25,000 (+25% bankroll)** — bankroll $100k→$125k |
| p_historica calibrada | derivar de datos reales | ✅ **0.52→0.68** (n=23, Thompson Beta 16W/7L) |
| Tests passing | ≥773 | ✅ 773 |

---

## Deudas Técnicas Priorizadas

1. ~~**CRÍTICA:** Migrar `extraer_historh2h.py` para importar `analysis/rivalry_analyzer.py`~~ ✅ RESUELTO — Nodo-07 Fase 1 (Strangler Fig). Markov + Erdős + surface activos en prod.

2. ~~**PRÓXIMA ACCIÓN:** Correr edge_calculator + trader_ev_tenis~~ ✅ HECHO — 2 señales APOSTAR, Parry @ 4.50 GANÓ ($45k retorno). Ver [[Nodo-14-Validacion-Live-Conexiones]].

3. **ALTA (T14-01):** Registrar resultados hoy en `validar_con_api.py` → n: 13→18+ → prior Bayesiano empieza a calibrarse hacia clay.

4. **ALTA (T13-04):** Pipeline completo 80 partidos → ≥3 señales → sistema 2/N activo → combos 1→15+.

5. **MEDIA (T14-03):** Calibrar peso `common_opponents` por superficie — clay: 20%→28-30%.

6. **MEDIA:** `tests/test_h2h_extractor.py` solo tiene 5 tests.

7. **BAJA:** Scripts de ranking (`extraer_ranking_*_version2.py`) sin tests.

---

## Vinculación

- [[Mandatos-No-Negociables]] — 8 mandatos que no pueden violarse
- [[Grafo-Dependencias-Datos]] — flujo completo de datos entre archivos
- [[Sprint-Pipeline]] — tareas completadas y pendientes
- [[Nodo-01-Edge-Calculator]] — Kelly-KL
- [[Nodo-02-Markov-Changepoint]] — PELT + factor HOT/COLD
- [[Nodo-03-Scraper-Fix]] — bugs del scraper de URLs
- [[Nodo-04-Dataset-Fix]] — bugs del pipeline ML
- [[Nodo-05-Validacion-API]] — FlashScore dc_1 labels
- [[Nodo-06-Erdos-Graph]] — grafo transitivo
- [[Nodo-13-Trader-EV-Tenis]] — capa deploy combos + sistema
- [[Nodo-14-Validacion-Live-Conexiones]] — primera validación live + 5 conexiones ocultas TTC

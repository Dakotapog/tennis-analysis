# Nodo-12: Inventario Infraestructura y Scripts de Diagnóstico

> **Wikilinks:** [[Inventario-Deuda-Tecnica]] | [[Pipeline-Arquitectura]] | [[Nodo-11-Inventario-Scripts-Legado]] | [[Sprint-Pipeline]] | [[Mandatos-No-Negociables]]
> **Estado:** 2026-05-30 — EJECUTADO ✅ | Auditoría + limpieza completadas
> **Metodología:** Leer → Documentar → Decidir → Ejecutar (SDD)

---

## Resumen Ejecutivo (2026-05-30)

Auditoría y limpieza de infraestructura legado completada. **772 tests passing, 1 fallo temporal conocido** (test_default_total_matches espera 80, encuentra 16 por cambio temporal de dev — restaurar post-run).

**Líneas/archivos eliminados esta sesión:**
- Debug artifacts raíz: 27 archivos (h2h_match_*.html/png + match_*.png + find)
- screenshots/ dead code: 3 archivos + 4 PNGs (fuera de scope pytest)
- Datos contaminados: reports/ginput/ (16 JSONs pre-Nodo-03) + reports/classification/ + predictions/ + reports/stages/ + reports/nba_h2h_analysis_*.json

**Hallazgo crítico vs spec original:** `services/` NO estaba vacío — contenía `selenium_config.py` (3,918 bytes). Decisión cambiada de "ELIMINAR si 0 bytes" a **SUSPENDER con stack Flask**.

---

## Grupo 1: Flask API (`app.py` + `routes/` + `models/` + `services/`)

### Qué hacen

**`app.py`** (201 líneas) — Servidor Flask con:
- Blueprint `/api/players/*` desde `routes/players.py`
- `init_db()` desde `models/database.py` → crea `tennis_analysis.db`
- Health check en `/health` y `/`
- **BUG activo:** importa `routes.players` pero el archivo es `routes/player_routes.py` → falla en startup
- No conectado al pipeline de predicción (S1-S8)

**`routes/player_routes.py`** — Blueprint Selenium para test de conectividad a `flashscore.co`
- Usa **Selenium + chromedriver** (no Playwright) — stack diferente al pipeline
- Endpoints: `/api/players/health`, `/api/players/test/connectivity`

**`models/database.py`** (25,758 bytes) — SQLite con `tennis_analysis.db`
- No conectado a ningún script del pipeline activo
- No importado por ningún script de predicción

**`services/`** — ⚠️ HALLAZGO: NO estaba vacío como decía spec anterior
- `__init__.py` (318 bytes)
- `instalar_selenium_wsl2.sh` (964 bytes) — script de instalación Selenium en WSL2
- `selenium_config.py` (3,918 bytes) — configuración Selenium (diferente a Playwright del pipeline)

### Diagnóstico

Stack Flask/SQLite/Selenium es un **stack paralelo** al pipeline de predicción (S1-S8). El pipeline opera 100% en modo batch (JSON). La API Flask sería el entry point para un frontend futuro, pero:
- `app.py` tiene bug de import (`routes.players` vs `routes/player_routes.py`) → no arranca
- Usa Selenium (diferente al Playwright del pipeline)
- `database.db` existe en raíz pero vacío (0 datos del pipeline)
- `services/` tiene lógica Selenium activa, no eliminable sin auditar `routes/`

### Decisión
**SUSPENDER** — ✅ Confirmado. Infraestructura de frontend futuro válida arquitecturalmente pero desconectada del pipeline actual. No eliminar. No integrar hasta que el pipeline P&L esté validado con n≥30.

---

## Grupo 2: Scripts de Diagnóstico

### `flashscore_rankings_inspector.py` (205 líneas)

**Qué hace:** `FlashscoreRankingsInspector` — Playwright para inspeccionar la página de rankings ATP y extraer selectores DOM. Herramienta de desarrollo, no pipeline.

**Valor:** Útil para debuggear selectores cuando FlashScore cambia su DOM.

**Decisión:** **MANTENER** ✅ — herramienta de diagnóstico. Mover a `tools/` en sprint de organización (pendiente).

### `flashs_revisa h2h_inspector.py`

**Decisión:** **ELIMINADO** ✅ — ya no existe en disco (verificado 2026-05-29). Nombre con espacio → no importable; NBA scraper fuera de scope tenis.

---

## Grupo 3: Carpetas de Datos Legado

### `advanced/` — JSONs h2h históricos (Jul-Ago 2025)

Resultados H2H de runs anteriores al fix de Nodo-03. Datos contaminados (superficie=0%, match_id="tennis").

**Decisión:** **ELIMINADO** ✅ — datos sucios pre-Nodo-03. Verificado en disco: directorio ya no existe.

### `input/` — features + labels (Ago-Sep 2025)

9 JSONs: 5 features + 4 labels pre-Nodo-03. Contaminados.

**Decisión:** **ELIMINADO** ✅ — directorio ya no existe en disco.

### `data_preprocessing/` y `model_training/`

Solo contenían `AGENTS.md` (instrucciones para agentes AI). Sin código Python activo.

**Decisión:** **ELIMINADOS** ✅ — verificado en disco.

---

## Grupo 4: Debug Artifacts en Raíz (2026-05-30)

Archivos `h2h_match_*.html`, `h2h_match_*.png`, `match_*.png` + archivo vacío `find` acumulados durante runs de prueba del scraper H2H.

**Eliminados 2026-05-30:** 9 HTML + 9 PNG (h2h) + 9 PNG (match) + 1 `find` = **28 archivos**

**Decisión:** **ELIMINADOS** ✅ — artefactos regenerables, no pertenecen a raíz.

---

## Grupo 5: Datos Contaminados en `reports/` (2026-05-30)

Directorios y archivos fuera del pipeline activo detectados en auditoría:

| Item | Contenido | Razón eliminación |
|---|---|---|
| `reports/ginput/features/` | 8 h2h_results_enhanced (Ago 2025) | Pre-Nodo-03: surface=0%, match_id="tennis" |
| `reports/ginput/labeles/` | 8 resultados_finales (Ago 2025) | Pre-Nodo-03: labels contaminadas |
| `reports/classification/` | directorio vacío | Sin función |
| `predictions/` | 1 h2h_results_enhanced Jan 2026 | Datos pre-fix en directorio incorrecto |
| `reports/stages/` | 2 CSVs intermedios dataset | Sin referencias activas, regenerables |
| `reports/nba_h2h_analysis_20251112_121929.json` | Análisis NBA | Fuera de scope del proyecto tenis |

**Eliminados 2026-05-30:** todas las rutas anteriores. ✅

`reports/` queda con solo: `edge_report_*.json`, `h2h_results_enhanced_*.json`, `resultados_finales_*.json`.

---

## Grupo 6: `screenshots/` — Dead Code (2026-05-30)

**Hallazgo crítico:** `pytest.ini` define `testpaths = tests`. La carpeta `screenshots/` NUNCA fue ejecutada por pytest — ni `conftest.py` ni `test_extraer_historh2h.py`.

| Archivo | Estado | Razón |
|---|---|---|
| `screenshots/conftest.py` | **ELIMINADO** ✅ | Fuera de testpaths, sys.path manipulation innecesaria |
| `screenshots/test_extraer_historh2h.py` | **ELIMINADO** ✅ | ELO values incorrectos, nunca ejecutado, reemplazado por tests en `tests/` |
| `screenshots/*.png` (4 archivos) | **ELIMINADOS** ✅ | Debug artifacts de FlashScore scraper |

---

## Grupo 7: `drivers/` — Binarios Chrome/Chromium

9 archivos de chromedriver/webdriver. El pipeline usa Playwright (no Selenium). Algunos pueden ser requeridos por `routes/player_routes.py` (Selenium).

**Decisión:** **MANTENER** ✅ — no eliminar sin auditar `routes/` completamente.

---

## Estado Post-Ejecución

### Tests
```
772 passed, 1 failed (temporal)

FALLO CONOCIDO: test_default_total_matches
  Causa: total_matches_to_process = 16 (cambio temporal dev, era 3, debería ser 80)
  Acción: restaurar a 80 post-run → test vuelve a pasar
  Sin relación con limpieza de Nodo-12
```

### Raíz del proyecto — estado final
```
SUSPENDER (stack Flask, no tocar):
  app.py | routes/ | models/ | services/ | database.db | drivers/

MANTENER (herramientas activas):
  flashscore_rankings_inspector.py  ← diagnóstico DOM
  consultar_resultados_historicos.py ← fallback histórico
  ml_trainer.py                     ← SUSPENDER hasta datos limpios S8
  extraer_URL_partidos_en_vivo.py   ← MANTENER + fix h2h_url pendiente
  logs/                             ← SmartLogger escribe aquí activamente

LIMPIO:
  Raíz, reports/, screenshots/ — sin artefactos de debug
```

---

## Tareas Pendientes Post-Nodo-12

| Task | Qué | Cuándo |
|---|---|---|
| T12-A | Restaurar `total_matches_to_process = 80` | Post-run de 16 partidos |
| T12-B | Mover `flashscore_rankings_inspector.py` → `tools/` | Sprint de organización |
| T12-C | Auditar `routes/` completo → definir si hay lógica salvable | Cuando pipeline P&L validado |
| T12-D | Fix h2h_url en `extraer_URL_partidos_en_vivo.py` | Antes de integrar pipeline LIVE |

---

## Vinculación

- [[Inventario-Deuda-Tecnica]] — D-06/D-07/D-08 (services/ vacíos ya eliminados); Nodo-12 agrega contexto de stack Flask completo
- [[Nodo-11-Inventario-Scripts-Legado]] — auditoría previa de scripts Python (D-13 generar_tabla_favoritos v1 eliminado)
- [[Pipeline-Arquitectura]] — app.py no está en el pipeline S1-S8
- [[Sprint-Pipeline]] — tasks T12-A..D en backlog
- [[Mandatos-No-Negociables]] — Mandato 6: 772 tests, 1 fallo temporal conocido

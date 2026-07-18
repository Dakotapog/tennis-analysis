---
estado: activo
---
# Nodo-117 — Auditoría Scraping: Rankings FlashScore ≠ ATP/WTA + Cobertura H2H 33%

> **Wikilinks:** [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | [[Nodo-49-Playwright-H2H-Fallback-n-h2h-0]] | [[Nodo-80-Kambi-Name-Matching]] | [[Nodo-82-Kambi-Match-ID-Structural]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-103-Auditoria-Combo-Builder-Gates-n-h2h]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]]
> **Fecha:** 2026-07-18 | **Autor:** Fable 5 / Sonnet 4.6 | **Prioridad:** ALTA — afecta cobertura diaria del universo de picks
> **Contexto:** Investigación post-pipeline 2026-07-18. Pipeline extrajo 130 partidos Playwright pero H2H procesó solo 43 (33%). RANKING_ONLY (D110-06) devolvió 0 candidatos del archivo API a pesar de Paula Badosa @1.27 en el universo.

---

## §1. DIAGNÓSTICO EJECUTIVO

Cuatro bugs estructurales que se combinan para reducir el universo diario de picks al 33% de su potencial:

| # | Bug | Archivo afectado | Línea | Impacto |
|---|-----|-----------------|-------|---------|
| B117-01 | Rankings `CA`/`CB` FlashScore ≠ ATP/WTA reales | `scraping/kambi_tennis.py` | L196-197 | RANKING_ONLY calcula gaps incorrectos |
| B117-02 | Playwright file: `cuota1=null` → H2H ve 0 partidos | `extraer_URL_partidos_version2.py` | — | 130 partidos → 0 para H2H |
| B117-03 | `select_best_json_file()` elige por tamaño, no calidad | `scraping/file_utils.py` | L137 | Siempre elige archivo sin cuotas |
| B117-04 | `_leer_matches_ranking_only` no activa fallback ATP/WTA | `favoritos_combo_builder.py` | ~L en `_leer_matches_ranking_only` | 0 candidatos RANKING_ONLY del archivo API |

---

## §2. BUG DETALLADO

### B117-01 — Rankings FlashScore `CA`/`CB` ≠ ATP/WTA oficiales

**Código fuente** (`scraping/kambi_tennis.py` L196-197):
```python
"ranking1": _safe_int(fields.get("CA")),   # feed FlashScore
"ranking2": _safe_int(fields.get("CB")),
```

Los campos `CA` y `CB` del feed binario de FlashScore son su sistema de ranking **propio** (posiblemente Race-to-Finals o ranking live), no las posiciones ATP/WTA oficiales.

**Evidencia cuantitativa (2026-07-18):**

| Jugadora/or | FlashScore CA/CB | ATP/WTA real (nuestro archivo) | Delta |
|---|---|---|---|
| Paula Badosa | 176 | 107 (WTA) | -69 |
| Andrey Rublev | 8 | 18 (ATP) | -10 |
| Tamara Zidansek | 172 | 145 (WTA) | -27 |
| Alejandro Tabilo | 51 | 31 (ATP) | -20 |
| Mayar Sherif | 69 | ~60 (WTA) | ~-9 |

**Causa raíz:** FlashScore publica en su feed comprimido (delimitador `¬÷`) rankings que reflejan su propia clasificación interna, actualizada en frecuencia y criterio diferentes a los tours oficiales.

**Impacto en D110-06:** `_leer_matches_ranking_only()` usa estos valores cuando `ranking1 != None`. El gap real Badosa-Zidansek en ATP/WTA es |107-145|=38 (< 300 → tampoco califica D110-06, pero por razón correcta). Para pares donde el gap FlashScore sí supera 300 pero el real no, se generarían falsos positivos. Para pares donde el gap real supera 300 pero el FlashScore no, se pierden candidatos válidos.

---

### B117-02 — Playwright captura match_ids pero no cuotas Kambi

**Flujo actual:**
```
extraer_URL_partidos_version2.py (Playwright)
  → scrapa FlashScore DOM
  → captura: jugador1/2, match_id, match_url, hora, superficie
  → NO captura: cuota1, cuota2 (estas vienen de Kambi/Betplay)
  → resultado: cuota1=null, cuota2=null en 130/130 partidos
```

**Consecuencia en `extraer_historh2h.py`:**
```
🎯 Seleccionado automáticamente: zita_tennis_matches_20260718_085932.json
   ✅ Válido: 130 partidos, 130 con URLs
   🌍 Modo multi-torneo: 130 válidos → 0 individuales con cuotas
   ✅ Cola: 0 partidos para API Ninja
```

El H2H extractor filtra la cola de partidos por `cuota1 != None`. Sin cuotas → cola vacía → 0 H2H extraídos del archivo Playwright de 130 partidos.

**Por qué las cuotas no están en Playwright:** `extraer_URL_partidos_version2.py` navega a FlashScore que muestra cuotas de múltiples casas, pero no extrae las de Kambi (Betplay). Las cuotas Kambi solo se obtienen via API Kambi (`extraer_partidos_api.py`).

---

### B117-03 — `select_best_json_file()` prioriza cantidad sobre calidad

**Comportamiento observado:**
```
✅ zita_tennis_matches_20260718_164237.json: 66 partidos, 20 con URLs   ← tiene cuotas
🎯 Seleccionado: zita_tennis_matches_20260718_085932.json: 130 partidos, 130 con URLs  ← sin cuotas
```

El selector elige el archivo más grande (`130 > 66`) sin verificar si tiene `cuota1 != None`. Resultado: siempre se usa el peor archivo para el pipeline H2H cuando Playwright corrió antes que el API.

**Archivo afectado:** `scraping/file_utils.py` función `select_best_json_file()` L137.

---

### B117-04 — Fallback ATP/WTA bloqueado cuando ranking FlashScore presente

**Código actual** (`favoritos_combo_builder.py`, función `_leer_matches_ranking_only`):
```python
r1 = partido.get("ranking1")   # ← si FlashScore puso 176, se usa 176
r2 = partido.get("ranking2")
if r1 is None:                 # ← 176 != None → fallback NUNCA activa
    r1 = _buscar_ranking(j1, ranking_map)   # rankings ATP/WTA reales
```

Cuando el archivo de partidos tiene `ranking1=176` (valor FlashScore incorrecto), el fallback a los rankings ATP/WTA reales (`data/atp_rankings_complete_*.json`) nunca se activa. El gap calculado con valores FlashScore es incorrecto, causando que candidatos válidos sean rechazados o aceptados incorrectamente.

---

## §3. IMPACTO EN EL PIPELINE DIARIO

**Flujo actual (roto):**
```
PASO 1 Playwright → 130 partidos, cuota1=null
PASO 1 API        → 43-66 partidos, con cuotas (Kambi)
PASO 2 H2H        → elige Playwright (más grande) → 0 H2H
                  → o elige API explícitamente → 43 H2H (33% del total)
PASO 3 Edge       → analiza 43 partidos de 130 (33%)
D110-06 RANKING_ONLY → usa ranking FlashScore incorrecto → 0 candidatos
```

**Flujo objetivo (con fixes):**
```
PASO 1 Playwright → 130 partidos + match_ids + URLs
PASO 1 API        → 43-66 partidos + cuotas Kambi
MERGE             → 130 partidos con cuotas (donde existe) + match_ids
PASO 2 H2H        → 100-130 H2H (cubiertos por match_id FS)
PASO 3 Edge       → analiza 100-130 partidos (77-100%)
D110-06 RANKING_ONLY → usa rankings ATP/WTA reales → candidatos correctos
```

**Pérdida estimada diaria:** ~60-65% del universo de partidos no llega al edge_calculator ni al RANKING_ONLY.

---

## §4. DECISIONES Y FIXES

### D117-01 — Fix `_leer_matches_ranking_only`: forzar rankings ATP/WTA (QUIRÚRGICO)

**Prioridad: INMEDIATA** — 1 línea en `favoritos_combo_builder.py`.

```python
# ANTES (bug):
r1 = partido.get("ranking1")
if r1 is None:
    r1 = _buscar_ranking(j1, ranking_map)

# DESPUÉS (fix D117-01):
# Siempre preferir rankings ATP/WTA reales; FlashScore CA/CB solo como último recurso
r1 = _buscar_ranking(j1, ranking_map) or partido.get("ranking1")
r2 = _buscar_ranking(j2, ranking_map) or partido.get("ranking2")
```

**Salvaguarda:** Si `_buscar_ranking()` no encuentra al jugador (ITF obscuro), cae back al valor FlashScore. Mejor que nada.

### D117-02 — Fix `select_best_json_file()`: priorizar archivos con cuotas (MEDIO)

Añadir criterio: un archivo con `n_con_cuotas > 0` siempre supera a uno con `n_con_cuotas = 0`, independientemente del total de partidos.

```python
# Criterio actual: max(partidos_con_urls)
# Criterio nuevo: max(partidos_con_cuotas) > 0 primero, luego max(partidos_con_urls)
```

**Archivo:** `scraping/file_utils.py` función `select_best_json_file()`.

### D117-03 — Merge Playwright + API antes del H2H (ESTRUCTURAL)

Crear función `_merge_zita_files(playwright_path, api_path)` que:
1. Toma el archivo Playwright (130 partidos, match_ids, urls, sin cuotas)
2. Toma el archivo API (43-66 partidos, con cuotas, con o sin match_ids)
3. Cruza por nombre normalizado → enriquece el Playwright con cuotas del API
4. Resultado: 130 partidos, ~66 con cuotas, todos con match_ids

**Precondición:** PASO 1 debe correr ambos extractores (`--playwright` y `--api`). El merge se ejecuta antes del PASO 2.

### D117-04 — Documentar campo `CA`/`CB` FlashScore (DOCUMENTACIÓN)

En `scraping/kambi_tennis.py` L196-197, añadir comentario:
```python
# NOTA B117-01: CA/CB son rankings del feed FlashScore (sistema interno),
# NO posiciones ATP/WTA oficiales. No usar para gaps de ranking — ver
# data/atp_rankings_complete_*.json y data/wta_rankings_complete_*.json.
"ranking1": _safe_int(fields.get("CA")),
"ranking2": _safe_int(fields.get("CB")),
```

---

## §5. PRIORIZACIÓN

| Fix | Impacto | Esfuerzo | Estado |
|-----|---------|----------|--------|
| D117-01: ranking fallback ATP/WTA | RANKING_ONLY usa datos correctos | 1 línea | ✅ APLICADO 2026-07-18 — `favoritos_combo_builder.py` `_leer_matches_ranking_only()` |
| D117-02: select_best con cuotas | H2H elige archivo correcto | ~10 líneas | ✅ APLICADO 2026-07-18 — `scraping/file_utils.py` + 5 tests REGLA-T53 |
| D117-04: comentario CA/CB | Evita regresión futura | 3 líneas | ✅ APLICADO 2026-07-18 — `scraping/kambi_tennis.py` L196-199 |
| D117-03: merge Playwright+API | +67% cobertura H2H | ~100 líneas | GATEADO — requiere refactor PASO 1 |

**Gate D117-03:** requiere que `run_daily.py` corra ambos extractores en PASO 1 y que el merge sea suficientemente robusto para no introducir duplicados por name-matching impreciso (ver [[Nodo-80-Kambi-Name-Matching]]).

---

## §6. RELACIÓN CON NODOS PREVIOS

| Nodo | Relación |
|------|----------|
| [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | Misma familia: cobertura H2H insuficiente → coinflip |
| [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | Historia del scraper FlashScore — cuotas vs match_ids |
| [[Nodo-49-Playwright-H2H-Fallback-n-h2h-0]] | Playwright como fallback H2H — problema estructural relacionado |
| [[Nodo-80-Kambi-Name-Matching]] | Name matching entre Playwright (nombre completo) y API (abreviado) — crítico para D117-03 |
| [[Nodo-82-Kambi-Match-ID-Structural]] | match_id como puente entre fuentes — D117-03 lo usa |
| [[Nodo-86-Auditoria-Fable5]] | Patrón recurrente: bug silencioso en capa de datos → pérdida de señal |
| [[Nodo-103-Auditoria-Combo-Builder-Gates-n-h2h]] | Gates n_h2h afectan mismo universo restringido por estos bugs |
| [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | D110-06 RANKING_ONLY: afectado por B117-01 y B117-04 |

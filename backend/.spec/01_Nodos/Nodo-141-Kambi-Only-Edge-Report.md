# Nodo-141 — Kambi-Only Edge Report (PASO 3K)

**Fecha:** 2026-07-22
**Estado:** IMPLEMENTADO
**Wikilinks:** [[Nodo-140]] [[Nodo-139]] [[Nodo-90]] [[Nodo-87]]

---

## 1. Problema

Después de Nodo-140 (Kambi Gate), el filtro kambi_disponible existe pero no resuelve el problema de
raíz cuando solo hay 1-2 picks apostables de 10 totales.

**Evidencia real 2026-07-22:**
- 10 picks APOSTAR → 8 ITF (kambi_disponible=False) → 0 combos safe/was/mega
- 2 picks kambi_disponible=True (Arakawa N., Suresh D.) → no alcanzan mínimo para combo (edge mismatch)
- Solo GamesB @3.01x generado (games builder parte DE Kambi directamente)

**Root cause persistente:**
PASO 1 API (extraer_partidos_api.py) ya corre como primario y fetcha ~27-180 eventos Kambi.
PASO 1b Playwright (FlashScore) agrega ~228 partidos ITF adicionales.
PASO 1.5 Match Ledger funde ambos → edge_calculator procesa 255+ matches → 80%+ picks ITF.
Nodo-140 filtra en combo time, pero si solo hay 2 picks apostables → 0 combos.

**La solución definitiva:**
Producir un segundo edge_report que contenga SOLO picks kambi_disponible=True.
Los combo builders (betplay) prefieren este reporte automáticamente → 100% picks apostables.

---

## 2. Hallazgo arquitectónico clave

`_find_latest_file(pattern)` en combo_confianza_builder.py usa **mtime** (st_mtime).
`_find_latest_edge_report()` en betplay_combo_builder.py usa **sort alfabético inverso**.

Si `edge_report_kambi_FECHA.json` se escribe DESPUÉS de `edge_report_FECHA.json`:
- combo_confianza: lo elige automáticamente por mtime más reciente
- betplay: 'kambi' > '2026' alfabéticamente → lo elige automáticamente

→ PASO 3K es suficiente. No se necesita cambiar lógica de selección en combo_confianza_builder.
→ betplay_combo_builder requiere fix explícito para solo preferir kambi de HOY (no de ayer).

---

## 3. Fixes implementados

### D141-01 — `scripts/filter_kambi_picks.py` (NUEVO)
Script ligero que:
- Lee el último `edge_report_FECHA.json` (excluye archivos kambi)
- Filtra picks donde `kambi_disponible=True`
- Escribe `reports/edge_report_kambi_FECHA.json` con misma estructura
- `--dry-run` muestra stats sin escribir

### D141-02 — `run_daily.py` PASO 3K
Después de PASO 3 (edge_calculator), añadir:
```
PASO 3K — Kambi-Only Report (Nodo-141): filtra picks apostables
python3 scripts/filter_kambi_picks.py
```
`optional=True` por si no hay coverage (graceful).

### D141-03 — `betplay_combo_builder._find_latest_edge_report()` prefer kambi HOY
Modificar para preferir `edge_report_kambi_HOYYYYYMMDD*.json` cuando existe.
Fallback a `edge_report_*.json` (excluye kambi) si no hay kambi de hoy.

### D141-04 (DEUDA TÉCNICA — DEFERRED) — Unificar 5 normalizadores de nombres
Catalogado en Nodo-140. Los 5 silos: kambi_tennis.py, match_ledger.py, combo_registry.py,
player_registry.py, fetch_kambi_coverage.py → unificar en `core/name_normalizer.py`.
Gate: estabilidad de picks durante 5 días consecutivos antes de refactorizar.

---

## 4. Impacto esperado

| Situación | Antes (Nodo-140) | Después (Nodo-141) |
|-----------|------------------|---------------------|
| 10 picks apostar (8 ITF) | 0 combos betplay | ≥1 combo de picks kambi |
| 2 picks kambi disponible | 0 combos (insuficiente) | Safe 2-leg si P(ambos)≥25% |
| 5+ picks kambi disponible | Filtrados correctamente | Safe + WAS + posible MEGA |
| 0 picks kambi | 0 combos | 0 combos (correcto) |

---

## 5. Hipótesis pre-registradas

H141-01: "edge_report_kambi produce ≥1 combo apostable en ≥70% de días con ≥2 picks kambi"
n_stop=20, umbral=70%

---

## 6. Tests — REGLA-T53 (`tests/test_nodo141_kambi_only_report.py`)

- test_D141_01_filter_produces_kambi_only_picks
- test_D141_01_filter_preserves_report_structure
- test_D141_01_filter_empty_when_no_kambi_picks
- test_D141_03_find_latest_prefers_todays_kambi
- test_D141_03_find_latest_falls_back_to_full_when_no_kambi_today
- test_D141_03_find_latest_excludes_yesterday_kambi

---

## 7. Decision log

D-11 (2026-07-22): No modificar el flujo de PASO 1/1b/1.5 — el problema no está en la
colección de datos sino en el reporting. Producir un segundo edge_report es más seguro y
reversible que cambiar la fuente de PASO 1.

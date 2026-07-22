# Nodo-140 — Kambi Gate: Coverage Fresca + Pre-Filtro Apostable

**Fecha:** 2026-07-22
**Estado:** IMPLEMENTADO
**Wikilinks:** [[Nodo-90]] [[Nodo-118]] [[Nodo-139]] [[Nodo-67]] [[fetch_kambi_coverage.py]] [[edge_calculator.py]]

---

## 1. Diagnóstico — 5 Hallazgos Graphify

### H-1: El Kambi Gate ya existía al 80% (Nodo-90 D90-01) — FALSO PENDIENTE

```
Nodo-90 D90-01 (implementado):
  scripts/fetch_kambi_coverage.py  →  reports/kambi_coverage_FECHA.json
       ↓
  edge_calculator.py L1385: _annotate_kambi()
       ↓
  edge_report: "kambi_disponible": true/false  ← por cada pick
```

El campo `kambi_disponible` existía en edge_report desde Nodo-90. **Nunca fue consumido por los builders.**

### H-2: Coverage OBSOLETA — 9 días sin actualizar

```
reports/kambi_coverage_20260713_111700.json  ← última: 13 julio (9 días atrás)
run_daily.py: 0 menciones a fetch_kambi_coverage   ← NUNCA en el pipeline
```

`_annotate_kambi()` leía datos de 9 días atrás → `kambi_disponible` incorrecto.

### H-3: Nodo-139 construyó la pieza que faltaba en Nodo-90

```
fetch_kambi_coverage.py:         players_normalized[]   (SI/NO — sin outcome_id)
_fetch_kambi_betting_universe(): outcome_id_fav + cuota + hora (completo)
```

Juntos forman el gate completo: disponibilidad + datos para betslip.

### H-4: `fetch_kambi_outcomes()` ya funciona — no es el problema

`fetch_kambi_outcomes()` (L110 betplay_combo_builder.py) ya filtra NOT_STARTED y crea
`outcomes_map` con outcome_ids correctos. El problema: recibe picks ITF que genuinamente
no existen en Kambi → retorna vacío → 0 combos.

La raíz es que el pre-filtro `kambi_disponible` nunca bloqueó esos picks **antes** de llegar a `fetch_kambi_outcomes()`.

### H-5: 5 normalizadores de nombres en silos separados (deuda técnica)

```
kambi_tennis.py L226:         _normalize_name()
match_ledger.py L41:          _normalizar_nombre()
combo_registry.py L45:        _normalize_name()
player_registry.py L46:       normalize_player_name()
fetch_kambi_coverage.py:      _normalize_name()
```

5 implementaciones sin coordinación — riesgo de divergencia de matching.
**No se toca en este nodo** — deuda catalogada para Nodo-141.

---

## 2. Solución — 3 Cirugías Quirúrgicas

### Cirugía 1 — D140-01: PASO 1c en run_daily.py

**Archivo:** `run_daily.py` — entre PASO 1.5 y PASO 2

```python
# ── PASO 1c — Kambi Coverage (Nodo-140) ──────────────────────────────────────
# Fetcha catálogo Kambi/Betplay NOT_STARTED → reports/kambi_coverage_HOY.json
# edge_calculator lo lee en _annotate_kambi() → kambi_disponible fresco por pick
_run(['python3', 'scripts/fetch_kambi_coverage.py'],
     'PASO 1c — Kambi Coverage (Nodo-140: catálogo apostable fresco)')
```

**Por qué aquí:** PASO 3 (edge_calculator) llama `_annotate_kambi()` que lee este archivo.
Debe correr después de PASO 1 (tenemos lista de partidos) y antes de PASO 3.

### Cirugía 2 — D140-02/03: Pre-filtro `kambi_disponible` en builders

**Archivo:** `betplay_combo_builder.py`

Nueva función helper (una vez, reutilizable):

```python
def _filter_kambi_available(picks: list, label: str = '') -> list:
    """D140-02: excluye picks kambi_disponible=False antes de fetch_kambi_outcomes().
    None = sin coverage aún = pass-through (no bloquear si PASO 1c no corrió)."""
    available = [p for p in picks if p.get('kambi_disponible') is not False]
    n_excl = len(picks) - len(available)
    if n_excl:
        logger.info(f'[D140-02] {label}: {n_excl}/{len(picks)} excluidos (ITF/sin Betplay)')
    return available
```

**En `build_safe_combos()`** (L1099-1108): añadir `kambi_disponible` al `edge_pick_map`.
**En `build_was_combos()`** (L1454): filtrar `watchlist` antes del loop.
**En `build_mega_combos()`** (L2276-2292): añadir `kambi_disponible` al `edge_tier_map` + filtrar pool.

### Cirugía 3 — D140-04: Gate en `combo_confianza_builder.py`

**Archivo:** `combo_confianza_builder.py` — función `_extract_and_categorize()` L522

```python
# D140-04: Kambi gate — cargar coverage fresca una sola vez
try:
    from scripts.fetch_kambi_coverage import load_coverage as _kc_load, \
                                             is_player_available as _kc_available
    _kc_cov = _kc_load()
except Exception:
    _kc_cov = None

# En el loop, después de obtener favorito:
if _kc_cov is not None and not _kc_available(favorito, _kc_cov):
    continue  # No disponible en Betplay — ITF/torneo sin catálogo Kambi
```

---

## 3. Impacto Esperado

| Métrica | Antes | Después |
|---------|-------|---------|
| `kambi_disponible` frescura | 9 días | diario (PASO 1c) |
| Picks ITF en pool safe/was/mega | ✓ incluidos → "Sin outcome" | ✗ excluidos early |
| Picks ITF en CORE/SAT/MOON | ✓ categorizados → .bat vacío | ✗ excluidos en _extract |
| "Sin outcome para X" warnings | múltiples | 0 (picks pre-filtrados) |
| Combos generados | 0 o .bat vacío | solo picks apostables |

---

## 4. Archivos Modificados

- `run_daily.py` — D140-01: PASO 1c fetch_kambi_coverage
- `betplay_combo_builder.py` — D140-02/03: `_filter_kambi_available()` + 3 call-sites
- `combo_confianza_builder.py` — D140-04: gate `is_player_available` en `_extract_and_categorize`
- `tests/test_nodo140_kambi_gate.py` — 8 tests REGLA-T53

---

## 5. Deuda Técnica Catalogada (no implementada en este nodo)

- **D141-01**: Unificar los 5 normalizadores de nombres en `core/name_normalizer.py`
- **D136-02**: Propagar `torneo_nombre` en `extraer_historh2h.py` (pending desde Nodo-136)
- **Kambi-First overlap**: Cuando PASO 1 scrape FlashScore torneos que Kambi cubre → overlap sube de 3/136 a ~20/136

---

## 6. Guards

- Si `kambi_disponible` es `None` → **pass-through** (coverage no existe aún = no bloquear)
- Si `kambi_disponible` es `False` → **excluir** (ITF/torneo genuinamente no en Betplay)
- Si `kambi_disponible` es `True` → **incluir** (apostable, seguir flujo normal)
- Si `fetch_kambi_coverage.py` falla en PASO 1c → warning, pipeline continúa (no fatal)

# Nodo-81 — Settlement Name Normalization: `_normalize_name_match` en settle()

> **Familia normalización Kambi:** [[Nodo-80-Kambi-Name-Matching]] (close_snapshot, parche táctico) | [[Nodo-82-Kambi-Match-ID-Structural]] (solución estructural gateada)

**Fecha:** 2026-07-11
**Estado:** IMPLEMENTADO 2026-07-11 — `_normalize_name_match` Tier 3a en `settle()` de shadow_book.py
**Rama:** main

---

## Problema

`settle()` en `shadow_book.py` hace join entre picks del shadow book y resultados de FlashScore
usando tres tiers: match_id exacto → match_key exacto → fuzzy `_name_tokens` (Nodo-36).

El Tier 3 (fuzzy `_name_tokens`) fallaba para jugadores con acentos, guiones o variaciones
de inicial:

| Pick favorito          | Resultado FlashScore  | Resultado |
|------------------------|-----------------------|-----------|
| "Carreno Busta"        | "Carreño Busta"       | NO MATCH  |
| "Vives-Marcos"         | "Vives Marcos"        | posible NO MATCH |
| "P. Ruud"              | "Casper Ruud"         | NO MATCH  |

Sin normalización canónica, el Tier 3 dependía de que `_name_tokens` fragmentara bien el
string — insuficiente para acentos y variaciones de nombre.

### Diferencia con Nodo-80

| Dimensión          | Nodo-80 (Kambi CLV)               | Nodo-81 (Settlement)              |
|--------------------|-----------------------------------|-----------------------------------|
| Punto del pipeline | PASO 5.5: close_snapshot          | PASO 10a: settle()                |
| Fuente A           | shadow book snapshot (pick)       | shadow book pick / favorito       |
| Fuente B           | Kambi outcomes_map (cuota cierre) | FlashScore resultados (ganador)   |
| Función            | `_apellido_candidates()` Tier 2   | `_normalize_name_match()` Tier 3a |
| Spec               | Nodo-80                           | **Este Nodo**                     |

Ambos son en la misma familia: identidad de jugador por string de nombre → falla en apellidos
compuestos, acentos, homónimos. Son puntos distintos del pipeline y fixes distintos.

---

## Solución implementada

### `_normalize_name_match` (Tier 3a, nuevo)

Se añadió ANTES del Tier 3b (`_fuzzy_name_match` / `_name_tokens`) en `settle()`:

```python
def _normalize_name_match(candidate: str, pick_name: str) -> bool:
    """
    Tier 3a: normalización canónica via core.player_registry (Nodo-51).
    Cubre acentos, guiones, variaciones de inicial.
    Retorna True si candidate y pick_name son el mismo jugador tras normalización.
    """
    try:
        from core.player_registry import normalize_player_name
        nc = normalize_player_name(candidate)
        np_ = normalize_player_name(pick_name)
        if not nc or not np_:
            return False
        return (nc == np_
                or (len(np_) >= 4 and np_ in nc)
                or (len(nc) >= 4 and nc in np_))
    except Exception:
        return False
```

**Diseño:**
- Reutiliza `normalize_player_name` de `core.player_registry` (Nodo-51) — fuente canónica.
- Igualdad exacta post-normalización: cubre acentos, casing, guiones.
- Substring bidireccional (≥4 chars): cubre "Ruud" ↔ "Casper Ruud" y apellido ↔ nombre completo.
- `except Exception: return False` — no rompe settle() si player_registry falla.

### Integración en `settle()` — Tier 3a

En `shadow_book.py`, `settle()` usa el nuevo tier antes del fuzzy:

```python
# Fallback nombre: Tier 3a normalize_player_name (Nodo-51) + Tier 3b _name_tokens (Nodo-36)
else:
    favorito = snap.get('favorito_predicho', '')
    if favorito:
        for res_candidate in resultados_map.values():
            p1_fs = res_candidate.get('p1', '') or ''
            p2_fs = res_candidate.get('p2', '') or ''
            if (_normalize_name_match(p1_fs, favorito)
                    or _normalize_name_match(p2_fs, favorito)
                    or _fuzzy_name_match(p1_fs, favorito)
                    or _fuzzy_name_match(p2_fs, favorito)):
                res = res_candidate
                break
```

### WON/LOST determination — también actualizado

El bloque de determinación WON/LOST también usa `_normalize_name_match` como primer tier:

```python
resultado = 'WON' if (
    _normalize_name_match(ganador, favorito) or _fuzzy_name_match(ganador, favorito)
) else 'LOST'
```

---

## Efecto observado

Antes del fix (re-settle julio 6 con solo Tier 3b):
- Settled: 13 de 21 picks

Después del fix con Tier 3a (re-settle julio 6):
- Settled: +5 adicionales → 18 de 21

Los 3 restantes sin settle corresponden a picks con `match_id` presente pero sin
`resultados_finales` del día (partidos suspendidos, WO, o fecha sin archivo de resultados).

---

## Tests

`tests/test_settlement_name_normalize.py` — 17 tests

| Clase | Tests | Cubre |
|-------|-------|-------|
| TestNormalizeNameMatchAccents | 4 | tildes, ñ, mayúsculas |
| TestNormalizeNameMatchDash | 3 | guiones → espacio, guion vs sin guion |
| TestNormalizeNameMatchSubstring | 4 | apellido en nombre completo, inicial + apellido |
| TestNormalizeNameMatchNegative | 6 | falsos positivos, substring corto (<4), nombres distintos |

Total tests módulo: 1804 passed (incluye Nodo-80 12 tests + Nodo-81 17 tests).

---

## Condición para implementar

- [x] `_normalize_name_match` implementada y conectada en settle() Tier 3a y WON/LOST
- [x] Tests en `tests/test_settlement_name_normalize.py` — 17 tests passed
- [x] Re-settle julio 6 confirmado: +5 picks adicionales
- [x] No rompe Tier 1 (match_id) ni Tier 2 (match_key) — solo agrega antes de fuzzy
- [ ] Nodo-82: Kambi match ID en PASO 4 (solución estructural — gateado)

## Vinculación

- Nodo-51: `core.player_registry.normalize_player_name` — función reutilizada
- Nodo-36: `_name_tokens` fuzzy — Tier 3b, sigue vigente después de Tier 3a
- Nodo-72: Phantom Identity Guard — misma familia (string de nombre ≠ identificador único)
- Nodo-80: `_apellido_candidates` Kambi CLV — mismo problema, PASO 5.5 (distinto)
- `shadow_book.py`: `_normalize_name_match()`, `settle()` Tier 3, WON/LOST determination
- `core/player_registry.py`: `normalize_player_name()` — fuente canónica de normalización

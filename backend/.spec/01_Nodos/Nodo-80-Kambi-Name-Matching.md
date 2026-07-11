# Nodo-80 — Kambi Name Matching: apellidos compuestos y normalización

**Fecha:** 2026-07-10
**Estado:** IMPLEMENTADO 2026-07-11 — Opción 1 (candidatos múltiples) en `close_snapshot()` de shadow_book.py
**Opción implementada:** Opción 1 (parche táctico) — NO Opción 3 (Kambi match ID, deuda estructural → Nodo-82)
**Rama:** main

---

## Problema

`_match_key()` en `shadow_book.py` extrae el apellido como el último token del nombre
completo (`nombre.split()[-1]`). Para apellidos compuestos este recorte falla:

| Nombre completo       | Apellido extraído | Nombre en Kambi      | Resultado |
|-----------------------|-------------------|----------------------|-----------|
| Pedro Vives Marcos    | "Marcos"          | "Vives Marcos"       | NO MATCH  |
| Diego Dedura-Palomero | "Dedura-Palomero" | "Dedura" o "Palomero"| posible NO MATCH |

El 3-tier matching (`nombre completo → apellido → fuzzy`) no alcanza porque "Marcos" es
substring de "Vives Marcos" pero no el token exacto que Kambi indexa en `outcomes_map`.

### Evidencia cronológica

- Primer registro en `close_snapshot_trigger.log`: 2026-07-08
- Total ocurrencias "0 matches con Kambi": 21 (al 2026-07-10)
- Patrón: **crónico**, no aislado de hoy
- Enmascarado por: n8n capturaba cierre_kambi exitosamente en el mecanismo primario,
  cron fallback fallaba silenciosamente (primero numpy, después matching) sin alertas

### Caso concreto confirmado hoy

Pick `jankanj_marcos` = "Pedro Vives Marcos vs Vlado Jankanj":
- `match_key = "jankanj_marcos"` — extrae "Marcos" de "Pedro Vives Marcos"
- Kambi indexa el jugador como "Vives Marcos" o "P. Vives"
- `cierre_kambi: None` — CLV incalculable para este pick

---

## Relación con Nodo-72 (Phantom Identity Guard)

Este es el **mismo problema estructural** que Nodo-72, en un punto diferente del pipeline.

| Dimensión           | Nodo-72 (Phantom Guard)          | Nodo-80 (Kambi Matching)         |
|---------------------|----------------------------------|----------------------------------|
| Punto del pipeline  | PASO 2: extracción H2H            | PASO 5.5: cierre de cuota Kambi  |
| Síntoma             | Historial de jugador homónimo     | Cuota de cierre no encontrada    |
| Causa raíz          | Comparación por string de nombre  | Comparación por string de nombre |
| Fuente de verdad    | FlashScore entity ID              | Kambi match ID                   |
| Fix en Nodo-72      | `_detect_phantom_identity()` + Playwright PRIMARIO | Pendiente en este Nodo |

**Conclusión estructural:** el problema de identidad por nombre es **transversal** al sistema,
no aislado a Phantom Guard. El modelo asume que el string del nombre de un jugador es un
identificador único — esa asunción falla en apellidos compuestos, homónimos y variaciones
de formato (Apellido, Nombre / Nombre Apellido / inicial + apellido).

---

## Plan de solución (no implementado)

### Opción 1 — Normalización mejorada de apellido (mínima)
Modificar `_match_key()` para intentar múltiples variantes de apellido:
`["Marcos", "Vives Marcos", "Vives"]` y buscar cualquiera en `outcomes_map`.

### Opción 2 — Fuzzy matching robusto (más complejo)
Usar `difflib.SequenceMatcher` o `rapidfuzz` para buscar el candidato más cercano
en `outcomes_map` cuando los tiers 1 y 2 fallan.

### Opción 3 — Kambi match ID (ideal, mayor esfuerzo)
Almacenar el Kambi match ID en el momento de log_pick (cuando se hace el scraping)
para buscar por ID en vez de por nombre. Requiere cambio en PASO 4 (trader) y PASO 1.

**Recomendación:** Opción 1 como fix mínimo para apellidos compuestos, Opción 3 como
solución estructural definitiva alineada con la filosofía de Nodo-72 (ID, no nombre).

---

## Implementación — 2026-07-11

### Lo que se hizo (Opción 1)

Gate de "≥5 ejemplos" revisado: `_normalize_name_match` ya existía como función probada en el
mismo módulo, y la Opción 1 es un parche táctico de bajo riesgo (retro-compatible, no cambia
`_parse_nombre` ni el formato de PASO 4). Se implementó sin esperar más ejemplos.

**`shadow_book.py`** — función helper + Tier 2 actualizado en `close_snapshot()`:
```python
def _apellido_candidates(norm_nombre: str) -> list:
    """Genera candidatos: último token, últimos 2, ... (Nodo-80 Opción 1)."""
    parts = norm_nombre.split()
    return [' '.join(parts[-i:]) for i in range(1, len(parts))]

# close_snapshot() Tier 2 — antes: solo parts[-1]
# ahora: itera candidatos hasta encontrar match en outcomes_map
for _cand in _apellido_candidates(norm_fav):
    found = outcomes_map.get(_cand)
    if found:
        break
```

**Caso Pedro Vives Marcos verificado (simulación):**
- Antes: `outcomes_map.get("marcos")` → None
- Ahora: candidatos `["marcos", "vives marcos"]` → `outcomes_map.get("vives marcos")` → match ✓

**Tests:** `tests/test_nodo80_kambi_matching.py` — 12 tests, todos passed.

### Lo que NO se hizo (gateado como Nodo-82)

**Opción 3** (Kambi match ID en PASO 4): solución estructural definitiva, requiere cambio en
`trader_ev_tenis.py` (PASO 4) y scraper (PASO 1). Documentado como deuda futura, NO ejecutar
sin decisión explícita.

## Condición para implementar

- [x] Tests de regresión en `tests/test_nodo80_kambi_matching.py` — 12 tests
- [x] Implementación Opción 1 verificada con caso Pedro Vives Marcos
- [ ] Opción 3 (match ID) → Nodo-82 (deuda estructural, gate abierto)

## Vinculación

- Nodo-72: Phantom Identity Guard — misma familia de problemas
- Nodo-73: close_snapshot timing — el mecanismo que usa este matching
- Regla 7 de Nodo-78: respaldo no verificado (el cron fallback enmascaró este bug)
- `shadow_book.py`: `_match_key()`, `close_snapshot()`, `_parse_apellido()`
- `scraping/kambi_tennis.py`: `_build_match_key()`, `_normalize_name()`

# Nodo-80 — Kambi Name Matching: apellidos compuestos y normalización

**Fecha:** 2026-07-10
**Estado:** DIAGNÓSTICO — sin implementación todavía
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

## Condición para implementar

- [ ] Al menos 5 ejemplos adicionales de mismatch catalogados (para validar el fix)
- [ ] Tests de regresión en `tests/test_nodo80_kambi_matching.py`
- [ ] Pre-registro de hipótesis si el fix afecta datos de CLV retroactivos

## Vinculación

- Nodo-72: Phantom Identity Guard — misma familia de problemas
- Nodo-73: close_snapshot timing — el mecanismo que usa este matching
- Regla 7 de Nodo-78: respaldo no verificado (el cron fallback enmascaró este bug)
- `shadow_book.py`: `_match_key()`, `close_snapshot()`, `_parse_apellido()`
- `scraping/kambi_tennis.py`: `_build_match_key()`, `_normalize_name()`

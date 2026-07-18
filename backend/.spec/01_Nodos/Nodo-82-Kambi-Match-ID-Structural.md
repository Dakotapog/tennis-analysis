# Nodo-82 — Kambi Match ID en PASO 4: solución estructural (GATEADO)

> **Familia normalización Kambi:** [[Nodo-80-Kambi-Name-Matching]] (close_snapshot, parche táctico, origen de esta deuda) | [[Nodo-81-Settlement-Name-Normalize]] (Tier 3a en settle) | [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] (D117-03 usa match_id como puente Playwright↔API)

**Fecha:** 2026-07-11
**Estado:** GATEADO — criterio de activación definido, NO implementar hasta cumplir gate
**Deuda estructural de:** Nodo-80 Opción 3
**Rama:** main

---

## Contexto

Nodo-80 implementó Opción 1 (candidatos múltiples de apellido) como fix táctico para
apellidos compuestos en el lookup Kambi CLV. Opción 3 (Kambi match ID en PASO 4) es la
solución estructural definitiva alineada con la filosofía de Nodo-72 (ID, no nombre).

## Problema que resolvería

`close_snapshot()` busca cuota de cierre Kambi por nombre (3 tiers: nombre completo →
candidatos apellido → fuzzy). Si Kambi cambia el formato del nombre (inicial, guion, orden)
el lookup falla silenciosamente → `cierre_kambi: None` → CLV incalculable.

Opción 3 almacenaría el `kambi_match_id` en el momento de `log_pick()` (PASO 4, trader)
para buscar por ID en `close_snapshot()` → inmune a variaciones de nombre.

## Por qué está gateado

Nodo-80 Opción 1 ya cubre el caso documentado (apellidos compuestos) con bajo riesgo.
Opción 3 requiere:
- Cambio en `trader_ev_tenis.py` (PASO 4): pasar `kambi_match_id` a `log_pick()`
- Cambio en scraper (PASO 1): almacenar match ID de Kambi en el momento de extracción
- Cambio en `shadow_book.py`: `log_pick()` y `close_snapshot()` usan ID
- Riesgo de regresión en el pipeline completo

## Criterio de activación (2026-07-11 — Decisión explícita)

**Activar Nodo-82 si y solo si:**

Después de **2 semanas de operación con Nodo-80 activo** (fecha referencia: 2026-07-25),
el bug de key truncada reaparece con **otro jugador de apellido compuesto distinto** —
evidencia de que candidatos múltiples (Opción 1) no es cobertura suficiente.

**No activar si:**
- Nodo-80 no ha producido `cierre_kambi: None` nuevos en ese período
- El problema reaparece en un jugador con nombre simple (indica otro bug, no este)
- No hay al menos 2 casos distintos documentados

## Revisión programada

Revisar el 2026-07-25 o antes si aparece evidencia de fallo de Nodo-80.

## Vinculación

- Nodo-80: Opción 1 táctica (implementada) — candidatos múltiples en Tier 2 close_snapshot
- Nodo-72: Phantom Identity Guard — filosofía ID > nombre (fuente estructural)
- `trader_ev_tenis.py`: PASO 4 — punto de cambio si se activa
- `scraping/kambi_tennis.py`: `_build_match_key()` — fuente del ID de Kambi
- `shadow_book.py`: `log_pick()`, `close_snapshot()` — consumidores del ID

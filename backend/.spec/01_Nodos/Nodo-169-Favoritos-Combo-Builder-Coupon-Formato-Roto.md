# Nodo-169 — favoritos_combo_builder.py: Formato de Coupon Betplay Roto Desde su Creación (D169-01)

**Estado:** IMPLEMENTADO
**Fecha:** 2026-08-04
**Módulo principal:** `favoritos_combo_builder.py`

---

## Contexto / Bug reportado

Usuario reportó que los combos `FavCombo101/102/103.bat` generados hoy (2026-08-04,
PASO 4.3b) abren Betplay pero sin ninguna pierna cargada — mismo síntoma que
[[Nodo-162]], y preguntó si ya estaba documentado cómo se soluciona.

## Root cause

`_build_betplay_url()` (`favoritos_combo_builder.py:330-332`) construía el coupon con:

```python
ids_str = "/".join(f"{oid}|ML" for oid in outcome_ids)
```

→ `combination|ID1|ML/ID2|ML||replace` — viola REGLA-BAT-1 (CLAUDE.md §9,
INMUTABLE). **A diferencia de Nodo-162** (que era una regresión introducida por
el commit `4ae668d` en `docs/bp/index.html`), esta función nació ya rota:
`git log -p -- favoritos_combo_builder.py` muestra el formato `|ML/` en el commit
que **creó** la función (Nodo-146, H2H_MODEL universe) — nunca tuvo el formato
correcto. La auditoría de Nodo-162 no cubrió este archivo porque no es uno de
los 3 consumidores de `REDIRECT_BASE`/`docs/bp/index.html` que se auditaron
entonces; `favoritos_combo_builder.py` tiene su propia implementación duplicada
de `_build_betplay_url()` en vez de reusar la de `combo_confianza_builder.py`
(fuente de verdad, L1675).

## Fix

D169-01: `ids_str = ",".join(outcome_ids)` — mismo patrón que
`combo_confianza_builder.py::_generar_bats()` L1675 (fuente de verdad
REGLA-BAT-1). Comentario anti-regresión citando REGLA-BAT-1 + Nodo-162 +
Nodo-169. FavCombo101/102/103 regenerados en caliente tras el fix — verificado
en `favcombo101.html`: `coupon=combination|4284658243,4284034676,4285037863||replace`
(sin `|ML/`, comma-joined).

## Tests

`tests/test_nodo169_favoritos_coupon_format.py` — 5 tests REGLA-T53, invocan
`_build_betplay_url()` real (no hardcodean el string esperado salvo en el test
de formato completo). 5/5 PASS + 3/3 Nodo-162 regresión sin romper = 8/8 PASS.

## Wikilinks

- [[Nodo-162]] — mismo síntoma/root cause class, archivo distinto, no regresión
  sino bug independiente nunca antes detectado
- [[Nodo-146]] — origen de `favoritos_combo_builder.py` y de la función rota
- CLAUDE.md §9 REGLA-BAT-1 — fuente de verdad del formato coupon

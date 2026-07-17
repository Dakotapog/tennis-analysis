# Nodo-112 — C3: campeon_signal estructurado (Nodo-108 §3)

> Creado: 2026-07-17
> Estado: IMPLEMENTADO
> Wikilinks: [[Nodo-108]] [[Nodo-57]] [[Nodo-01]]

---

## Problema

`edge_calculator.py` extraía el tier del campeón de hierba/arcilla parseando strings de
reasoning log (`_rlog.split('tier=')[1].split(',')[0]...`). Esto era frágil: cualquier
cambio de formato del mensaje rompía silenciosamente la extracción.

## Solución (C3)

### D112-01 — Campos estructurados en `_surface_specialization`

`analysis/rivalry_analyzer.py`: la función `_surface_specialization` ahora devuelve tres
campos adicionales junto con `campeon_signal` (renombrado de `torneo_completo`):

```python
'campeon_signal':   bool   # True si hay TORNEO_COMPLETO_BONUS activo
'campeon_tier':     str|None  # tier del torneo donde fue campeón ('grand_slam', etc.)
'campeon_torneo':   str|None  # nombre del torneo
'campeon_days_ago': int|None  # días desde que ganó ese torneo
```

Variables capturadas justo antes del `break` del loop TORNEO_COMPLETO_BONUS:
```python
_campeon_tier_val = _tier_champ
_campeon_torneo_val = _tname
_campeon_days_val = _days_ago
```

### D112-02 — Consumo en `edge_calculator.py`

Reemplaza el bloque de string-parsing (líneas ~1116-1127) por lectura directa del campo:

```python
if _surf_fav_65.get('campeon_signal', _surf_fav_65.get('torneo_completo')):
    _campeon_tier_nivel = _surf_fav_65.get('campeon_tier')
```

El fallback `_surf_fav_65.get('torneo_completo')` garantiza compatibilidad si un
caller antiguo pasa el dict sin `campeon_signal`.

## Tests

- `tests/test_rivalry_analyzer.py` — 240 tests pasan (17 de surface_specialization)
- `tests/test_edge_calculator.py` — 240 tests pasan

## Invariante

Nunca parsear reasoning strings para extraer datos estructurados. Si un dato es necesario
en un consumer downstream, debe ser un campo en el dict de retorno.

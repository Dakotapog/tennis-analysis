# Nodo-127 — games_signal_calculator: IDs genéricos ITF + filtro STARTED

> Estado: PENDIENTE — D126-04 + D126-05 por implementar
> Detectados: sesión operacional 2026-07-21 (continuación Nodo-126)
> Tests: REGLA-T53 — 3 tests en `tests/test_nodo127_games_outcome_ids.py`

---

## 0. Contexto

Tras los 3 fixes de Nodo-126, games_signal_calculator ya no mapea a dobles.
Sin embargo se detectaron 2 bugs nuevos en la misma sesión 2026-07-21:

1. **Partidos STARTED (en vivo):** Bueno vs Pereira OVER 2.5 @9.5 — imposible para
   un partido NOT_STARTED. El match ya había empezado cuando corrió el calculator.

2. **IDs outcome genéricos/template en ITF:** el ID `4265916952` aparecía en 30+
   partidos diferentes todos con cuota @1.73. Son outcome IDs de plantilla que Kambi
   reutiliza para mercados ITF con la misma línea y cuota — NO son específicos de
   un partido. Abrir ese .bat en Betplay carga un partido aleatorio, no el elegido.

---

## 1. Bugs y Fixes

### D126-04 — Sin filtro de partidos STARTED (en vivo)

**Bug:** `_buscar_event_id_kambi()` itera todos los eventos de `listView/tennis.json`
sin distinguir `NOT_STARTED` de `STARTED`. Si un partido ya empezó, Kambi devuelve
odds live distorsionadas (ej. OVER 2.5 @9.5 cuando el match va 2-0 en sets).

**Síntoma observado:** Bueno G. vs Pereira T. OVER 2.5 @9.5 en report 071929.
Combo resultante: @19x para 2 piernas (imposible con odds normales @1.7-2.2).

**Fix:**
```python
# En _buscar_event_id_kambi(), antes del chequeo de nombre:
if ev.get("event", {}).get("state") != "NOT_STARTED":
    continue
```

**Archivo:** `games_signal_calculator.py::_buscar_event_id_kambi()` — añadir tras
el filtro de dobles (`if "/" in name: continue`).

---

### D126-05 — Outcome IDs genéricos/template en ITF

**Bug:** Kambi reutiliza el mismo outcome ID para mercados idénticos en torneos ITF.
El ID `4265916952` (UNDER 2.5 sets @1.73) aparece en 30+ partidos distintos.
El ID `4265925873` (UNDER 2.5 sets @1.87) aparece en 20+ partidos distintos.

**Root cause:** Para torneos pequeños (ITF, algunos Challenger), Kambi no asigna
outcome IDs únicos por partido — usa plantillas de mercado compartidas.
Al abrir el .bat con ese ID, Betplay puede cargar cualquier evento que tenga
ese outcome, no el partido específico que el modelo eligió.

**Fix:** Tras resolver el outcome_id para una señal, verificar unicidad:
```python
def _es_outcome_unico(outcome_id: int, all_events: list) -> bool:
    """Retorna True si el outcome_id aparece en exactamente 1 evento del feed."""
    count = 0
    for ev_w in all_events:
        for bo in ev_w.get("betOffers", []):
            for oc in bo.get("outcomes", []):
                if oc.get("id") == outcome_id:
                    count += 1
                    if count > 1:
                        return False
    return count == 1
```

Si `_es_outcome_unico()` retorna False → descartar señal (log como NO_UNICO).

**Impacto esperado:** La mayoría de señales ITF serán descartadas. Solo quedará
ATP250+ donde Kambi asigna IDs únicos por partido. Esto es correcto — los combos
GAMES solo son apostables donde el ID es unívoco.

**Archivo:** `games_signal_calculator.py::_analizar_mercados_juegos()` + nueva
función `_es_outcome_unico()`. Requiere pasar `all_events` como parámetro.

---

## 2. Tests REGLA-T53

Archivo: `tests/test_nodo127_games_outcome_ids.py`

```
test_outcome_unico_detecta_duplicado()
    feed con 2 eventos que tienen el mismo outcome_id
    → _es_outcome_unico(outcome_id, feed) == False

test_outcome_unico_detecta_unico()
    feed con 2 eventos con outcome_ids distintos
    → _es_outcome_unico(id_del_primero, feed) == True

test_filtro_started_excluye_live()
    evento con state="STARTED" no debe ser retornado por _buscar_event_id_kambi()
    → mockear listView con 1 NOT_STARTED + 1 STARTED para mismo jugador
    → solo retorna el NOT_STARTED
```

---

## 3. Impacto operacional

| Situación | Antes de fix | Después de fix |
|-----------|-------------|----------------|
| Partido en vivo | Odds @9.5 → combo @19x | Excluido, no entra al combo |
| ITF UNDER 2.5 @1.73 | ID compartido → partido aleatorio en Betplay | Descartado como NO_UNICO |
| ATP500 OVER 2.5 @2.4 | ID único → partido correcto | Conservado ✓ |

**Conclusión:** GAMES combos solo son confiables en ATP250+ con IDs únicos.
Para ITF, el sistema correctamente no generará combos apostables.

---

## 4. Secuencia de implementación

```
Sesión futura:
  D126-04 — filtro state=="NOT_STARTED" en _buscar_event_id_kambi()
  D126-05 — _es_outcome_unico() guard + log NO_UNICO
  3 tests REGLA-T53 en test_nodo127_games_outcome_ids.py
```

---

## §WIKILINKS

### Forward links
- [[Nodo-126-GamesSignal-3Bugs-Fix]] — 3 fixes previos, continuación directa
- [[Nodo-40-Games-Sets-Signal-Layer]] — módulo afectado
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53

### Back links
- [[Nodo-126]] ← D126-04/D126-05 derivados de la misma sesión de debugging

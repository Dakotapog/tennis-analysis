# Nodo-127 — games_signal_calculator: IDs genéricos ITF + filtro STARTED

> Estado: CERRADO — D126-04 + D126-05 implementados 2026-07-21
> Detectados: sesión operacional 2026-07-21 (continuación [[Nodo-126-GamesSignal-3Bugs-Fix]])
> Commit: `582090c`
> Tests: 3/3 PASS — `tests/test_nodo127_games_outcome_ids.py`

---

## 0. Contexto

Tras los 3 fixes de [[Nodo-126-GamesSignal-3Bugs-Fix]], `games_signal_calculator`
ya no mapeaba a dobles. Sin embargo en la misma sesión 2026-07-21 se detectaron
2 bugs nuevos al verificar los combos generados:

**Bug 1 — Partidos STARTED:** Bueno vs Pereira OVER 2.5 @**9.5** (match ya 2-0 en
sets → OVER imposible → cuota distorsionada). Combo resultante: @19x para 2 piernas.

**Bug 2 — IDs genéricos ITF:** El ID `4265916952` (UNDER 2.5 @1.73) aparecía en
**30+ partidos distintos**. ID `4265925873` (UNDER 2.5 @1.87) en 20+ partidos.
Kambi reutiliza IDs template en torneos ITF — abrir ese .bat cargaba un partido
aleatorio en Betplay, no el partido específico del modelo.

---

## 1. Fixes implementados

### D126-04 — Filtro NOT_STARTED en procesar_partidos() ✅

**Archivo:** `games_signal_calculator.py::procesar_partidos()` — Intento 2 (L~527)

**Bug:** Intento 2 usaba `j1_parts[-1]` (inicial, mismo bug D126-01) sin filtro
`state` ni dobles. Intento 3 era peligrosamente amplio (busca solo por apellido
de j1, toma el primer resultado — causaba matches erróneos).

**Fix aplicado:**
```python
# Intento 2: D126-01/04: _apellido() + NOT_STARTED + sin dobles
if not ev_id:
    events = _get_listview()
    ap1 = _apellido(j1)
    ap2 = _apellido(j2)
    if ap1 and ap2:
        for ev in events:
            if ev.get("event", {}).get("state") != "NOT_STARTED":
                continue  # D126-04: excluir partidos en vivo
            ev_name = ev.get("event", {}).get("name", "").lower()
            if "/" in ev_name:
                continue  # excluir dobles (D126-02)
            if ap1 in ev_name and ap2 in ev_name:
                ev_id = ev.get("event", {}).get("id")
                break

# Intento 3: desactivado — un solo apellido es demasiado amplio (D126-05)
```

**Nota:** El fix se aplicó en `procesar_partidos()` (ruta real de procesamiento),
NO en `_buscar_event_id_kambi()` (función auxiliar no usada en producción).

---

### D126-05 — Guard seen_outcome_ids en procesar_partidos() ✅

**Archivo:** `games_signal_calculator.py::procesar_partidos()` — post-señales (L~570)

**Fix aplicado:** patrón `seen_outcome_ids: dict[int, str]` — más simple y eficiente
que la función `_es_outcome_unico()` descrita en el spec inicial (no requiere
re-iterar todos los eventos del feed; acumula en un solo pase secuencial):

```python
# Antes del loop principal:
seen_outcome_ids: dict[int, str] = {}  # outcome_id → primer partido que lo usó

# Tras _seleccionar_señal_optima(), para cada evento:
unicas = []
for s in optimas:
    oid = s.get("outcome_id")
    if oid is None:
        unicas.append(s)
    elif oid in seen_outcome_ids:
        logger.info(f"   ⚠️  outcome {oid} NO_UNICO (visto en {seen_outcome_ids[oid]}) — descartado")
    else:
        seen_outcome_ids[oid] = nombre
        unicas.append(s)
optimas = unicas
```

**Resultado operacional 2026-07-21:**
- Partidos ITF: excluidos (no encontrados en NOT_STARTED o IDs genéricos)
- Combos generados: **GamesA @2.64x** (Van De Zandschulp UNDER 25.5 + Oliynykova OVER 19.5)
  y **GamesB @4.09x** (+ Brockmann OVER 19.5) — IDs únicos ATP/WTA, todos NOT_STARTED

---

## 2. Tests REGLA-T53 — 3/3 PASS

Archivo: `tests/test_nodo127_games_outcome_ids.py`

| Test | Contrato | Resultado |
|------|----------|-----------|
| `test_outcome_unico_detecta_duplicado` | mismo outcome_id en 2 partidos → segundo descartado | PASS ✅ |
| `test_outcome_unico_detecta_unico` | outcome_ids distintos → ambos conservados | PASS ✅ |
| `test_filtro_started_via_apellido` | `_apellido()` produce strings >2 chars útiles para match | PASS ✅ |

Helper de test `_simular_filtro_unico()` replica la lógica de `seen_outcome_ids`
en aislamiento sin necesidad de red (REGLA-T53: función real del módulo).

---

## 3. Impacto operacional

| Situación | Antes | Después |
|-----------|-------|---------|
| Partido STARTED | Odds @9.5 → combo @19x | Excluido (NO encontrado en NOT_STARTED) |
| ITF UNDER 2.5 @1.73 | ID 4265916952 en 30+ partidos | Descartado como NO_UNICO |
| ATP/WTA OVER 19.5 juegos | ID único por evento | Conservado ✓ |
| Intento 3 (apellido simple) | Matcheaba partido aleatorio | Desactivado ✓ |

**Conclusión:** GAMES combos confiables solo en ATP250+ con IDs únicos por partido.
Para ITF el sistema retorna 0 combos — comportamiento correcto.

---

## 4. Secuencia de implementación

```
COMPLETADO (2026-07-21, commit 582090c):
  D126-04 ✅ filtro state=="NOT_STARTED" + _apellido() en procesar_partidos() Intento 2
  D126-04 ✅ Intento 3 desactivado (demasiado amplio)
  D126-05 ✅ seen_outcome_ids guard — descarta IDs genéricos ITF en un pase
  3 tests REGLA-T53 — 6/6 PASS (N126 + N127 combinados)
```

---

## §WIKILINKS COMPLETOS

### Forward links (este nodo depende de)
- [[Nodo-126-GamesSignal-3Bugs-Fix]] — 3 fixes previos, continuación directa (D126-01/02/03)
- [[Nodo-40-Games-Sets-Signal-Layer]] — módulo `games_signal_calculator.py` afectado
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53 aplicada
- [[Nodo-111-Dual-Book-Live-Intelligence]] — patrón `_norm()` apellido similar (referencia)

### Back links (nodos que deben conocer este)
- [[Nodo-126-GamesSignal-3Bugs-Fix]] ← D126-04/D126-05 son continuación de la misma sesión
- [[Nodo-40-Games-Sets-Signal-Layer]] ← `procesar_partidos()` modificada: Intento2 corregido, Intento3 desactivado

### Huérfanos operacionales
- `games_signal_calculator.py` — Intento 2 corregido (L~527), seen_outcome_ids (L~444+L~570), Intento 3 desactivado
- `tests/test_nodo127_games_outcome_ids.py` — 3 tests REGLA-T53
- `nodos_index.json` — reindexar con `python3 scripts/rebuild_nodos_index.py`
- `CLAUDE.md` — actualizar párrafo Nodo-127 (hecho en esta sesión)

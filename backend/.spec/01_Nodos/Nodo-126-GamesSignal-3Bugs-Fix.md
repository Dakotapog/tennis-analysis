# Nodo-126 — games_signal_calculator: 3 bugs críticos corregidos

> Estado: CERRADO — D126-01→D126-03 aplicados 2026-07-21
> Detectados: sesión operacional 2026-07-21 (combo generado era inválido)
> Tests: REGLA-T53 — 3 tests en `tests/test_nodo126_games_signal_bugs.py`

---

## 0. Contexto

El combo GAMES generado el 2026-07-21 apuntaba a partidos de **dobles** en Betplay en lugar de singles. Al abrir el .bat, Betplay mostraba:
- `S. Gonzalez/M. Reyes-Varela - F. Ferreira Silva/T. Pereira` (dobles)
- `P-H. Herbert/K. Krawietz - J. Paul/R. Seggerman` (dobles)

Además el sistema generaba OVER y UNDER del **mismo partido** en el mismo combo — contradicción total.

Root cause: tres bugs independientes en `games_signal_calculator.py::_buscar_event_id_kambi()`.

---

## 1. Bugs y Fixes

### D126-01 — Extracción de apellido rota (`split()[-1]` → inicial)

**Bug:** `_buscar_event_id_kambi()` extraía el apellido con `split()[-1]`:
```python
j1 = partido.get("jugador1", "").split()[-1].lower()  # "Choinski J." → "j."
j2 = partido.get("jugador2", "").split()[-1].lower()  # "De Jong J."  → "j."
```
Nombres tipo `"Apellido Inicial."` tienen la **inicial** como último token, no el apellido.
Con `j1="j."` y `j2="j."`, cualquier evento Kambi que contenga "j." dos veces matcheaba — incluyendo dobles.

**Fix:** Nueva función `_apellido()` que toma la última palabra que NO sea una inicial (`len≤2` y termina en `.`):
```python
def _apellido(nombre: str) -> str:
    words = (nombre or "").split()
    for w in reversed(words):
        if not (len(w) <= 2 and w.endswith(".")):
            return w.lower()
    return (words[0] if words else "").lower()
```
- `"Choinski J."` → `"choinski"` ✓
- `"De Jong J."` → `"jong"` ✓
- `"Ugo Carabelli C."` → `"carabelli"` ✓

**Archivo:** `games_signal_calculator.py` — función `_apellido()` añadida antes de `_buscar_event_id_kambi()`, L~260.

---

### D126-02 — Sin filtro de dobles en Kambi listView

**Bug:** La búsqueda iteraba todos los eventos sin distinguir singles de dobles. Kambi incluye dobles en `listView/tennis.json`. Un apellido como "Pereira" matcheaba `"G. Pereira De Aguiar"` en un dobles.

**Fix:** Saltar eventos cuyo nombre contenga `/` (formato dobles en Kambi: `"A/B - C/D"`):
```python
if "/" in name:  # excluir dobles (formato "A/B - C/D")
    continue
```

**Archivo:** `games_signal_calculator.py::_buscar_event_id_kambi()` L~273.

---

### D126-03 — KeyError `'odds'` en outcome sin clave

**Bug:** L356-357 usaban acceso directo `o_mas["odds"]` que lanzaba `KeyError` cuando un outcome de Kambi no incluye la clave `odds` (partidos en estados especiales o suspended):
```python
cuota_mas   = (o_mas["odds"]  / 1000) if o_mas   else None   # ← KeyError
cuota_menos = (o_menos["odds"] / 1000) if o_menos else None  # ← KeyError
```

**Fix:** Acceso seguro con `.get()` y guard `>0`, consistente con las líneas 359-360 que ya lo hacían correctamente:
```python
cuota_mas   = (o_mas.get("odds", 0)   / 1000) if o_mas   and o_mas.get("odds",0)   > 0 else None
cuota_menos = (o_menos.get("odds", 0) / 1000) if o_menos and o_menos.get("odds",0) > 0 else None
```

**Archivo:** `games_signal_calculator.py::_analizar_mercados_juegos()` L~356.

---

## 2. Bug conocido pendiente (D126-04 — sesión futura)

### Sin filtro de partidos STARTED (en vivo)

**Síntoma:** Si un partido ya empezó cuando corre games_signal_calculator, Kambi devuelve odds live (ej. Bueno vs Pereira OVER 2.5 @**9.5** — match en 2-0). El modelo calculó la señal para un partido NOT_STARTED pero el outcome ID encontrado es para odds live.

**Consecuencia:** Cuotas irreales en combo (@19x para 2 piernas que deberían ser @4-6x).

**Fix pendiente:** En `_buscar_event_id_kambi()`, filtrar solo eventos con `state == "NOT_STARTED"`:
```python
if ev.get("event", {}).get("state") != "NOT_STARTED":
    continue
```

**Gate:** Verificar que no elimina partidos válidos con odds pre-partido todavía disponibles.

---

## 3. Tests REGLA-T53

Archivo: `tests/test_nodo126_games_signal_bugs.py`

```
test_apellido_simple()
    _apellido("Choinski J.") == "choinski"
    _apellido("Bueno G.") == "bueno"

test_apellido_compuesto()
    _apellido("De Jong J.") == "jong"
    _apellido("Ugo Carabelli C.") == "carabelli"

test_apellido_sin_inicial()
    _apellido("Djokovic") == "djokovic"   # sin inicial no rompe
```

---

## 4. Secuencia de implementación

```
COMPLETADO (2026-07-21):
  D126-01 ✅ _apellido() extrae apellido real, no inicial
  D126-02 ✅ filtro "/" excluye dobles de Kambi listView
  D126-03 ✅ .get("odds", 0) elimina KeyError en outcomes sin clave

Sesión futura:
  D126-04 — filtro state=="NOT_STARTED" para excluir odds live
```

---

## §WIKILINKS

### Forward links
- [[Nodo-40-Games-Sets-Signal-Layer]] — módulo afectado `games_signal_calculator.py`
- [[Nodo-118-Match-Ledger-Crosswalk]] — REGLA-T53 aplicada
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53 aplicada

### Back links
- [[Nodo-40]] ← 3 fixes correctivos en `_buscar_event_id_kambi()` y `_analizar_mercados_juegos()`

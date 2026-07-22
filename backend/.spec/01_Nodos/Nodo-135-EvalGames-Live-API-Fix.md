# Nodo-135 — EvalGames Live API Fix: betoffer/event endpoint + match-level filter

> **Estado:** IMPLEMENTADO — 2026-07-21
> **Tipo:** FIX — post-implementación Nodo-133 (Games Live Convergencia)
> **Autor:** Sonnet 4.6 (análisis doctoral — evidencia real sesión 2026-07-21)
> **Trigger:** Ibraimi vs Pham (ITF Brisbane) tenía "Total de juegos" en UI de Betplay pero nuestro sistema devolvía `cuota_live=None` porque buscaba betOffers en el endpoint equivocado (listView)

---

## Wikilinks

| Link | Rol | Archivo |
|------|-----|---------|
| [[Nodo-133-Games-Live-Convergencia]] | Padre — `_extract_games_cuota_live()` definida aquí, bug detectado con evidencia real | ✅ existe |
| [[Nodo-134-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | Sibling — auditoría bridge que documentó D126-04 (skip ITF) como false positive | ✅ existe |
| [[Nodo-40-Games-Sets-Signal-Layer]] | games_signal_calculator.py — fuente de señales pre-game | ✅ existe |
| [[Nodo-97-Live-Edge-Monitor]] | KambiLiveClientReal — patrón HTTP Kambi reutilizado | ✅ existe |
| [[Nodo-129-LiveDesk-AutoRefresh-Fix]] | `_background_refresh()` daemon — ciclo 15s donde corre este fix | ✅ existe |

**Wikilinks totales: 5 | Huérfanos: 0**

---

## §1. Evidencia real — 2026-07-21 ~22:30 CO

### Secuencia de hechos

1. Usuario confirma UI Betplay: **"Total de juegos — Ymerali Ibraimi vs Derek Pham — Más de 31.5 @ 1.80"** visible en la app móvil (ITF Brisbane, en vivo)
2. Nuestro sistema (Nodo-133) marca ese partido como `markets: []` en listView → `_extract_games_cuota_live()` devuelve `None` → cuota_live ausente en panel X3
3. Diagnóstico con curl en tiempo real:

```
GET listView/tennis.json   → [1028465663] Ibraimi vs Pham | Brisbane | markets: []
GET betoffer/event/1028465663.json → 15 betOffers activos (Set 3 juego 8/9)
GET betoffer/live/event/1028465663.json → 0 betOffers (endpoint /live/ incorrecto)
```

### Distribución en el universo de 225 eventos STARTED (22:30 CO)

| Grupo | Eventos | Con Total de juegos (listView) |
|-------|---------|-------------------------------|
| ATP/WTA/Challenger (Kitzbuhel, Hamburgo, Praga, Palermo, Estoril…) | ~50 | 43 (~86%) |
| ITF (Brisbane, Bali, India, Newport Beach, Waco…) | ~175 | 0 (0%) |
| **Total** | **225** | **43** |

**El problema:** listView solo expone "mercados destacados". Para ATP250+ y WTA los incluye. Para ITF → siempre vacío. El endpoint `betoffer/event/{id}.json` devuelve todos los betOffers activos para cualquier tier.

### Evidencia D135-01 en producción (misma sesión)

Cruce apellido: señales ALTA del games_signal_report (10:26 CO) vs 225 STARTED events:

| Señal pre-game | Evento matched | `_extract_games_cuota_live()` |
|----------------|---------------|-------------------------------|
| Brockmann T.J. vs Jacquemot E. — OVER 19.5 @ 1.55 ALTA | `[1028442321]` Hamburgo ✅ | **1.55** — mercado estable |
| Suresh D. vs Harris B. — UNDER 23.5 @ 1.77 ALTA | `[1028450468]` India (Keerthivassan Suresh) ✅ | **1.65** — cuota bajó -6.8% (drift: mercado espera más juegos) |
| Kozlov S. vs Magadan A. | `[1028470687]` Zheng vs Magadan | None — apellido ambiguo, partido distinto |

**2 señales con cuota_live real extraída correctamente en producción.**  
Antes del fix ambas devolvían `None` porque `betoffer/event` no era llamado.

---

## §2. Bug D135-01 — `_extract_games_cuota_live()` buscaba en listView betOffers

### Causa raíz

```python
# ANTES (Nodo-133, live_desk.py):
def _extract_games_cuota_live(ev_wrapper, direccion, linea):
    for bo in ev_wrapper.get("betOffers", []):   # ← listView betOffers: vacío para ITF
        if "juego" in bo.get("criterion", {}).get("label", "").lower():
            ...
```

`ev_wrapper` viene de `listView/tennis.json`. Su `betOffers` solo contiene mercados
destacados. Para Brisbane, Bali, India → `betOffers = []` siempre.

### Fix D135-01

Nueva firma: `_extract_games_cuota_live(event_id, direccion, linea)`.
Llamada HTTP dedicada a `betoffer/event/{event_id}.json` usando `urllib.request`
(misma librería HTTP que usa el resto de `live_desk.py`):

```python
# DESPUÉS (Nodo-135, live_desk.py):
def _extract_games_cuota_live(event_id: int, direccion: str, linea: Optional[float]) -> Optional[float]:
    """
    D135-01: busca mercado match-level 'Total de juegos' via endpoint betoffer/event/{id}.
    El listView solo devuelve mercados destacados (vacío para ITF). Este endpoint retorna
    todos los betOffers del evento.
    D135-02: excluye mercados set-level ("Total de juegos - Set 3") y juego-level.
    """
    url = (f"{_KAMBI_BASE}/betoffer/event/{event_id}.json"
           f"?{_KAMBI_PARAMS}")
    try:
        req = urllib.request.Request(url, headers=_KAMBI_HDR)
        with urllib.request.urlopen(req, timeout=3) as r:
            offers = json.loads(r.read().decode()).get("betOffers", [])
    except Exception:
        return None

    dir_norm = direccion.upper()
    for bo in offers:
        label = bo.get("criterion", {}).get("label") or ""
        # D135-02: solo mercado match-level, excluir "Total de juegos - Set X"
        if not ("Total de juegos" in label
                and " - Set " not in label
                and "Juego" not in label):
            continue
        for oc in bo.get("outcomes", []):
            oc_label = (oc.get("label") or "").lower()
            oc_line  = oc.get("line", 0) / 1000 if oc.get("line") else None
            is_under = "menos" in oc_label or "under" in oc_label
            is_over  = "más" in oc_label or "over" in oc_label or "mas" in oc_label
            dir_match  = (dir_norm == "UNDER" and is_under) or (dir_norm == "OVER" and is_over)
            line_match = (oc_line is None or linea is None or abs(oc_line - linea) < 1.0)
            if dir_match and line_match:
                odds_raw = oc.get("odds")
                if odds_raw:
                    return round(odds_raw / 1000, 2)
    return None
```

---

## §3. Bug D135-02 — Filtro match-level vs set-level

### Problema

Cuando un partido avanza al Set 3, Kambi cierra el mercado match-total y lo reemplaza
por mercados granulares de set/juego:

| Label Kambi | ¿Es match total? | Acción |
|-------------|-----------------|--------|
| `Total de juegos` | ✅ SÍ — el que apostamos pre-game | incluir |
| `Total de juegos - Set 3` | ❌ NO — sub-mercado de set | excluir |
| `Total de juegos - Set 3, Juego 8` | ❌ NO — sub-mercado de juego | excluir |

Evidencia real: Ibraimi/Pham en Set 3 devolvió:
```
"Total de juegos - Set 3"  (Más de: 7.500 | Menos de: 1.040)
"Total de juegos - Set 3"  (Más de: 25.000 | ...)
"Total de juegos - Set 3"  (Más de: 5.300 | Menos de: 1.100)
```
Cero instancias de `"Total de juegos"` match-level — el mercado ya había cerrado.

### Fix D135-02 (embebido en D135-01)

```python
if not ("Total de juegos" in label
        and " - Set " not in label
        and "Juego" not in label):
    continue
```

---

## §4. Actualización call-site en `_check_games_convergencia()` (D135-03)

```python
# ANTES (Nodo-133):
cuota_live = _extract_games_cuota_live(matched, sig["direccion"], sig["linea"])

# DESPUÉS (Nodo-135):
cuota_live = _extract_games_cuota_live(
    matched["event"]["id"], sig["direccion"], sig["linea"]
)  # D135-01
```

`event_id` ya está en `matched["event"]["id"]` — sin cambios en la lógica de detección.

---

## §5. Hallazgo documental — Ciclo de vida del mercado "Total de juegos"

```
Pre-game    │ Total de juegos (match) disponible
            │ games_signal_calculator genera señal: UNDER 22.5 @ 2.10
            │
Set 1-2     │ Total de juegos (match) LIVE — cuota se mueve
En vivo     │ Ej: UNDER 22.5 @ 1.80 (mercado espera más juegos, cuota baja)
VENTANA     │ drift_pct = (1.80−2.10)/2.10×100 = −14.3%
TRADING     │ D135-01 extrae cuota_live → _check_games_convergencia calcula drift
            │
Set 3       │ Mercado match-level CERRADO → reemplazado por:
avanzado    │   "Total de juegos - Set 3" (D135-02 lo excluye)
            │   "Set 3 - Juego 8/9" (granular)
            │
Fin         │ 0 betOffers — betoffer/event → []
```

**Ventana de trading real:** ~60-90 minutos (Set 1 hasta ~2/3 del Set 2).
El daemon cada 15s tiene ~6 oportunidades de lectura por partido.

---

## §6. Hallazgo adicional — Torneos por timezone (candidato Nodo-136)

Brisbane, Bali, India corren en horario CO-noche (~9pm-2am CO).
El games_signal_report (PASO 3.6, ~10:26 CO) no los captura — PASO 1 Playwright
corre en CO-mañana y solo ve torneos europeos/americanos.

Ibraimi vs Pham **no estaba** en el games_signal_report de hoy (21 señales, todas
europeas/americanas). La ventana de trading existió pero sin señal pre-game no hay
baseline para calcular drift ni dirección de apuesta.

**D135-04 (NO implementado aquí — candidato Nodo-136):**
Segunda ejecución PASO 1 + PASO 3.6 a las ~18:00 CO para capturar Australia/Asia.
Estimación: +8-15 señales adicionales/día operativas en horario CO-noche.

---

## §7. Tests REGLA-T53 — 4/4

Archivo: `tests/test_nodo135_games_live_api.py`

Los tests mockean `urllib.request.urlopen` (no `requests.get`) con un context manager
`_FakeResponse` que simula la respuesta del endpoint `betoffer/event/{id}.json`.

| Test | Contrato | Estado |
|------|----------|--------|
| `test_D135_01_extrae_cuota_via_betoffer_endpoint` | urlopen al endpoint correcto → cuota 1.80 retornada | ✅ PASS |
| `test_D135_02_filtra_set_level_markets` | label "Total de juegos - Set 3" → None | ✅ PASS |
| `test_D135_02_acepta_match_level_market` | label "Total de juegos" (match-level) → cuota correcta | ✅ PASS |
| `test_D135_01_retorna_none_si_mercado_inexistente` | betOffers sin Total de juegos → None | ✅ PASS |

---

## §8. Archivos modificados

| Archivo | Tipo | Cambio |
|---------|------|--------|
| `live_desk.py` | MODIFY | `_extract_games_cuota_live()` nueva firma + `urllib.request` call. Call-site en `_check_games_convergencia()` actualizado. ~18 líneas netas. |
| `tests/test_nodo135_games_live_api.py` | NEW | 4 tests REGLA-T53 — mock `urllib.request.urlopen` |

**Cero cambios** en games_signal_calculator.py, betplay_combo_builder.py,
close_snapshot_server.py, live_edge_monitor.py.

---

## §9. Decisiones de diseño

### D135-01 — urllib.request (no requests)

`live_desk.py` usa `urllib.request` en toda la base (L34: `import urllib.request`).
No se añade dependencia `requests` — consistencia con el módulo.

### D135-02 — Tolerancia de línea: ±1.0 juegos

Pre-game: UNDER 22.5. Live: Kambi puede ajustar línea a 23.5 (raramente).
Tolerancia 1.0 captura mismo mercado con línea levemente movida sin confundir
mercados completamente distintos (ej: 22.5 vs 30.5).

### D135-03 — timeout=3s, llamadas secuenciales

El daemon corre cada 15s. Con N señales EN_VIVO simultáneas → N×3s overhead máximo
(secuencial). Para N≤5 típico: <15s total. asyncio no necesario por ahora.

### D135-04 — event_id=N/A en todas las señales hoy

El games_signal_report de hoy tiene 21 señales, todas con `event_id=N/A`.
`_buscar_event_id_kambi()` (games_signal_calculator.py) no encontró ninguno en Kambi
a las 10:26 CO. El sistema usa apellido como fallback — funciona pero es menos preciso.
Root cause pendiente: los event_ids pre-game en Kambi pueden haber cambiado entre
PASO 3.6 (10:26) y el inicio del partido. Candidato Nodo-136 §2.

---

## §10. Evidencia de producción post-deploy (22:45 CO)

Validación manual invocando `_extract_games_cuota_live()` directamente contra Kambi live:

```
python3 -c "
import sys; sys.path.insert(0,'.')
import live_desk as ld
# Brockmann vs Jacquemot [1028442321] — OVER 19.5 en Hamburgo
print(ld._extract_games_cuota_live(1028442321, 'OVER', 19.5))   # → 1.55
# Kevin Titus Suresh [1028450468] — India
print(ld._extract_games_cuota_live(1028450468, 'UNDER', 23.5))  # → 1.65
"
```

Output verificado:
```
1.55   ← Brockmann/Jacquemot OVER 19.5 (cuota pre-game = 1.55, mercado estable)
1.65   ← Suresh UNDER 23.5 (cuota pre-game = 1.77, bajó −6.8% drift)
```

El fix funciona en producción. El daemon en `:7780` tiene este código activo —
próximo ciclo `_background_refresh()` escribirá `games_live_20260721.json` con estas
cuotas_live reales.

---

**Wikilinks totales: 5 | Huérfanos: 0**

[[Nodo-133-Games-Live-Convergencia]] | [[Nodo-134-Auditoria-EvalGames-Bridge-Fugas-Fixes]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-129-LiveDesk-AutoRefresh-Fix]]

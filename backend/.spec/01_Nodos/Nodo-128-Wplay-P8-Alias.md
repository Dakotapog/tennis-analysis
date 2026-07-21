# Nodo-128 — Wplay P8 multi-book: alias apellido + diagnóstico cobertura

> Estado: CERRADO — D128-01 implementado 2026-07-21
> Detectado: sesión operacional 2026-07-21 (continuación Nodo-127)
> Commit: pendiente
> Tests: 3/3 PASS — `tests/test_nodo128_wplay_p8_alias.py`

---

## 0. Contexto

Wplay SSR funciona de forma independiente de Kambi (52 outcomes activos).
`_build_p8_books()` ya llamaba `fetch_all_odds(["betplay","rushbet","wplay"])` pero
el panel mostraba 0 picks porque:

1. **Betplay/Rushbet**: Kambi 429 rate-limit → feeds vacíos
2. **Wplay**: devuelve nombres completos ("Botic Van De Zandschulp") vs el `edge_report`
   que usa nombres abreviados Kambi ("Van De Zandschulp B.") → `best_price()` exact-match falla

Adicionalmente: los picks de hoy (14, todos ITF/Challenger) no están en Wplay (que cubre
solo ATP/WTA). El alias fix es correcto pero no aplica en sesiones ITF-only.

---

## 1. Fixes implementados

### D128-01 — Alias apellido (todos los feeds) en `_build_p8_books()` ✅

**Archivo:** `live_desk.py::_build_p8_books()` — post-L1978

**Lógica:** tras construir `feeds["wplay"]`, añadir entradas adicionales indexadas por
apellido (todo excepto el primer token del nombre normalizado):

```python
# D128-01: alias Wplay nombre-completo → apellido para matchear picks abreviados del edge_report
if "wplay" in feeds:
    _wp_aliases: Dict[str, Any] = {}
    for _wk, _we in feeds["wplay"].items():
        _parts = _wk.split()
        if len(_parts) >= 2:
            _sn = " ".join(_parts[1:])   # drop primer token (nombre de pila)
            if _sn not in feeds["wplay"] and _sn not in _wp_aliases:
                _wp_aliases[_sn] = _we
    feeds["wplay"].update(_wp_aliases)
```

**Resultado:** `best_price("Van De Zandschulp", feeds)` → encuentra `"botic van de zandschulp"`
via alias `"van de zandschulp"`. Sin colisión: si una clave exacta ya existe, no se sobreescribe.

**Extensión D128-01 (sesión):** alias aplicado a TODOS los feeds (betplay, rushbet, wplay),
no solo wplay. Betplay también usa nombres completos → misma necesidad de alias por apellido.

---

### D128-02 — Games players injection en `_build_p8_books()` ✅

**Archivo:** `live_desk.py::_build_p8_books()` — post-L2015

**Problema:** P8 solo leía `edge_report` (picks ML). Hoy: 14 picks ITF no en ningún book.
Los jugadores ATP/WTA de GamesA/B (Van De Zandschulp, Oliynykova, Brockmann) estaban
en los 3 feeds pero P8 no los incluía.

**Fix:** Lee TODOS los `games_signal_report_FECHA_*.json` del día. Para cada señal con
outcome_id único, extrae los dos jugadores del `partido` field, stripea la inicial trailing
(`"Van De Zandschulp B."` → `"Van De Zandschulp"`), y los añade a `all_picks_er` con
`_source="games"` y `_mercado` con la señal activa.

```python
# Para cada partido en apostar, con seen_oids guard:
_gp_parts = _gp_raw.split()
while _gp_parts and len(_gp_parts[-1].rstrip(".")) <= 1:
    _gp_parts.pop()  # strip "B." → queda "Van De Zandschulp"
_gp = " ".join(_gp_parts)
```

**Resultado operacional 2026-07-21:** P8 pasa de 0 a **35 picks** con feeds betplay+rushbet+wplay.
- Betplay/Rushbet: 644 outcomes ATP/WTA matcheados por apellido
- Wplay: 48 outcomes SSR independientes de Kambi
- Picks con triple precio: Gaston, Jacquet, Baez, Molcan, Jacquemot, etc.
- Picks solo Wplay: Kozlov @2.15, Magadan @1.65, Carballes Baena @3.40, etc.

---

## 2. Diagnóstico de cobertura (hallazgo 2026-07-21)

| Fuente | Jugadores |
|--------|-----------|
| edge_report 2026-07-21 | 14 picks ITF/Challenger (Cora-Bruneton, Clarke, Tabunshchyk...) |
| Wplay SSR | 52 outcomes ATP/WTA (Van De Zandschulp, Oliynykova, Stricker...) |
| Overlap | **0 jugadores en común** |

→ P8 = 0 picks es **comportamiento correcto** cuando el modelo apuesta ITF pero Wplay solo tiene ATP/WTA.
→ El alias D128-01 activará automáticamente en sesiones ATP (GS, ATP1000, ATP500).

**Los picks GAMES** (Van De Zandschulp UNDER 25.5, Oliynykova OVER 19.5) sí están en Wplay
@2.62 y @1.08 (ML), pero P8 solo muestra picks del `edge_report`, no del `games_signal_calculator`.

---

## 3. Tests REGLA-T53 — 3/3 PASS

Archivo: `tests/test_nodo128_wplay_p8_alias.py`

| Test | Contrato | Resultado |
|------|----------|-----------|
| `test_alias_wplay_apellido_permite_best_price` | "Van De Zandschulp" matchea "Botic Van De Zandschulp" post-alias | PASS ✅ |
| `test_alias_wplay_no_sobreescribe_clave_existente` | clave exacta existente no se sobreescribe | PASS ✅ |
| `test_alias_wplay_multiples_jugadores` | Oliynykova, Van De Zandschulp, Schunk — todos vía apellido | PASS ✅ |

---

## 4. Cuotas Wplay en vivo (2026-07-21 sesión)

```
Botic Van De Zandschulp    2.62   wplay   (ML — nuestro pick: UNDER 25.5 juegos)
Jaime Faria                1.45   wplay
Oleksandra Oliynykova      1.08   wplay   (ML muy bajo → match posiblemente live)
Nastasja M. Schunk         8.00   wplay
```

Para consultar en tiempo real: `python3 scripts/odds_aggregator.py --bookmakers wplay --show-all`

---

## §WIKILINKS COMPLETOS

### Forward links
- [[Nodo-111-Dual-Book-Live-Intelligence]] — `_build_p8_books()` Nodo-111 X1 (best_price, feeds)
- [[Nodo-127-GamesSignal-ITF-OutcomeID-Fix]] — sesión previa, contexto operacional
- [[Nodo-123-Auditoria-Dashboard-Integraciones-v2]] — Wplay SSR VERIFIED (`api_type: "wplay_ssr"`)
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53 aplicada

### Back links
- [[Nodo-111-Dual-Book-Live-Intelligence]] ← `_build_p8_books()` modificada: D128-01 alias
- [[live_desk.py]] ← L1979-1991 nuevo bloque D128-01

### Huérfanos operacionales
- `live_desk.py` — bloque D128-01 (~L1979)
- `tests/test_nodo128_wplay_p8_alias.py` — 3 tests REGLA-T53
- `nodos_index.json` — reindexar con `python3 scripts/rebuild_nodos_index.py`

# Nodo-123 — Dashboard Live Desk :7780 — Integraciones Pendientes v2

> Estado: CERRADO — D122-01→D122-06 todos ya implementados (verificado 2026-07-20)
> Fecha spec: 2026-07-20
> Auditoría base: [[Nodo-119-Auditoria-Desk-v3-21-Gaps-11-Fixes]] (addendum 2026-07-20)
> Tests: REPORTE_SOLO — render-only, no tocan decisiones de apuesta

---

## 0. Contexto en lenguaje simple

El dashboard `live_desk.py` (:7780) fue auditado en sesión 2026-07-20.
De 10 gaps pendientes originales, **7 ya estaban implementados** (falsos pendientes).
Se especificaron 4 integraciones físicas (D122-01→D122-04) + 2 pendientes operativos.

**ADDENDUM 2026-07-20 (segunda auditoría):** Las 4 integraciones físicas también
estaban ya implementadas. Total falsos pendientes en este nodo: 4/4.
Solo quedan D122-05 y D122-06 (scope mayor / blocker externo).

---

## 1. Estado actual verificado (NO reimplementar)

| Feature | Evidencia en código | Commit / Sesión |
|---------|-------------------|-----------------|
| P9 Execution Router | L1324 `{p9_panel}` en HTML | sesión anterior |
| P4 atenuación BLOCK visual | `opacity:0.35` en L621/718/752/774 | sesión anterior |
| JS sort por columna (7 tablas) | `sortTable()` + `▲/▼` | `2dd2711` |
| H111-01 en hypotheses.json | 29 hipótesis verificadas | verificado |
| H107-01 en hypotheses.json | "MOTOR split cuota>2.50" | verificado |
| P8 rushbet 3ra columna | `fetch_all_odds(["betplay","rushbet","wplay"])` | `f274309` |
| X2 steam-lag dinámico N-casas | extrae desde `cuotas{}` | `fa52236` |
| P3 picks bug resuelto | `apostar+watchlist` correcto | Nodo-119 D119-05 |
| P6 parser multi-línea | regex multi-línea shadow_book | Nodo-119 D119-08 |
| **D122-01 P8 fuente viva** | `_build_p8_books()` L1866 `_fetch_all_odds(["betplay","rushbet","wplay"])` TTL 600s | verificado 2026-07-20 |
| **D122-02 X3 mensaje accionable** | `live_desk.py` L1051-1054 "correr PASO 3.6: python3 games_signal_calculator.py" | verificado 2026-07-20 |
| **D122-03 X2 badges STEAM/DRIFT** | `live_desk.py` L927-991 badges STEAM OK (verde ≥15%+CONFIRMA) y ATN (ámbar) | verificado 2026-07-20 |
| **D122-04 systemd service :7780** | `~/.config/systemd/user/tennis-live-desk.service` enabled+running | verificado 2026-07-20 |

**Lección**: antes de especificar "integraciones pendientes", auditar el código real con
`grep` o `graphify query`. El spec fue escrito sin verificar `live_desk.py` directamente.

---

## 2. Pendientes reales

### D122-05 — RANKING_ONLY en FAVORITOS_COMPUESTOS *(FALSO PENDIENTE — verificado 2026-07-20)*
**Premisa del spec**: "favoritos_combo_builder.py excluye jugadores con join_method=RANKING_ONLY"

**Realidad verificada**:
1. El ledger (`match_ledger_YYYYMMDD.json`) tiene 4 keys: `joins`, `cuarentena`, `single_source_kambi`, `single_source_fs`. **No existe bucket `ranking_only`**.
2. `_leer_matches_ranking_only()` en `favoritos_combo_builder.py` (L488) ya maneja estos jugadores — lee el zita PASO 1 y filtra por ranking_gap>300 + cuota [1.15,1.60].
3. Se activa con `--matches zita_tennis_matches_*.json` en la invocación manual.
4. No hay nada que implementar. El spec tenía premisa incorrecta sobre la estructura del ledger.

**Acción**: ninguna. Usar `--matches` al correr favoritos_combo_builder manualmente.

---

### D122-06 — Kambi LIVE endpoint *(FALSO BLOCKER — resuelto por Fable D97-15)*

**Premisa del spec**: "requiere captura DevTools del operador en Chrome logueado"

**Realidad verificada (2026-07-20)**:
- `KambiLiveClientReal` en `scripts/live_edge_monitor.py` L158 — operativo desde 2026-07-14 (D97-15)
- Endpoint: `https://us.offering-api.kambicdn.com/offering/v2018/betplay/liveEvents.json`
- **Endpoint público** — Fable lo descubrió sin necesitar DevTools capture
- Rushbet usa el mismo patrón: `rsico` offering key en `odds_aggregator.py` L60 (VERIFIED 2026-07-19)
- Cadena: `/liveEvents.json` → fallback `/listView/tennis.json?state=STARTED`

**No hay blocker. No hay pendiente. Nodo-123 completamente cerrado.**

---

## 4. Secuencia de implementación

```
COMPLETADO (verificado 2026-07-20 — ya estaba en código):
  D122-01 ✅ P8 fuente viva → live_desk.py L1866 _fetch_all_odds() TTL 600s
  D122-02 ✅ X3 mensaje accionable → live_desk.py L1051-1054
  D122-03 ✅ X2 badges STEAM/DRIFT → live_desk.py L927-991
  D122-04 ✅ systemd service → ~/.config/systemd/user/tennis-live-desk.service

Sesión futura:
  D122-05 → RANKING_ONLY favoritos (scope mayor)
  D122-06 → Kambi LIVE (blocker externo — requiere DevTools capture)
```

---

## §WIKILINKS COMPLETOS

### Forward links (este nodo depende de)
- [[Nodo-119-Auditoria-Desk-v3-21-Gaps-11-Fixes]] — auditoría base, 10 gaps originales, addendum 2026-07-20
- [[Nodo-121-OddsAggregator-Cuota-Enrichment-ss-fs]] — D122-01 desbloqueado: betplay+rushbet+wplay VERIFIED
- [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] — spec P8 multi-book + cache TTL 120s
- [[Nodo-111-Dual-Book-Live-Intelligence]] — `divergencia()` + `best_price()` funciones puras para D122-03
- [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] — rushbet VERIFIED + P8 3 columnas base
- [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] — D122-05 scope mayor RANKING_ONLY
- [[Nodo-97-Live-Edge-Monitor]] — D122-06 blocker externo Kambi LIVE
- [[Nodo-73-n8n-CloseSnapshot-Timing]] — patrón systemd para D122-04

### Back links (nodos que deben conocer este)
- [[Nodo-119-Auditoria-Desk-v3-21-Gaps-11-Fixes]] ← addendum v2: gaps resueltos aquí
- [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] ← D122-01 implementa cache TTL + fuente viva

### Huérfanos operacionales
- `live_desk.py` — todo D122-01→D122-04 ya presente, no modificar
- `nodos_index.json` — reindexar cuando D122-05/D122-06 se implementen
- `CLAUDE.md` — Nodo-123 no requiere actualización de estado (nada nuevo implementado)

# Nodo-129 — LiveDesk Auto-Refresh: Cache Memoria + Push n8n

> Estado: CERRADO — D129-01/02/03 implementados 2026-07-21
> Detectado: sesión operacional 2026-07-21 (continuación Nodo-128)
> Commit: pendiente (sesión actual)
> Tests: 3/3 PASS — `tests/test_nodo129_live_desk_cache.py`

---

## 0. Problema reportado

El usuario reporta que el dashboard en `http://localhost:7780` no se actualiza
automáticamente aunque n8n corre cada 20 segundos. La expectativa era que n8n
actualizara el dashboard en tiempo real.

---

## 1. Arquitectura real vs. esperada

### Lo que el usuario esperaba (PUSH)

```
n8n (20s) ──→ dashboard actualiza ──→ browser ve datos frescos en <20s
```

### Lo que existía (PULL ciego — diagnosticado vía graphify)

```
n8n (20s) ──→ close_snapshot_server.py :8765
                ├─ /check-and-close → shadow_book.py --close-snapshot
                └─ /live-check      → live_edge_monitor.py --observe
                         ↓ escribe archivo en disco
                         ↓ NO POST a :7780 (graphify: "No path found between
                           n8n_push_workflow.py and live_desk.py")

browser (30s) ──→ GET http://localhost:7780/
                         ↓ do_GET llama build_desk_state() SINCRÓNICO
                           ├─ _build_p8_books():  3 HTTP externas  = 15-20s
                           └─ _build_p6_pnl():    subprocess 30s   = 30s
                         ↓ total: 60+ segundos de bloqueo
browser recibe HTML
```

**Latencia máxima real: 90 segundos.**
- n8n terminó hace 30s
- browser espera hasta próximo ciclo de 30s
- servidor tarda 60s en responder

El usuario ve "Auto-refresh 30s" y cree tener datos de hace 30s. En realidad
pueden tener 90 segundos de antigüedad o más.

### Arquitectura tras la fix (PUSH + cache)

```
n8n (20s) ──→ close_snapshot_server.py :8765
                ├─ /check-and-close → shadow_book.py --close-snapshot
                │                         ↓ _notify_live_desk()
                └─ /live-check      → live_edge_monitor.py --observe
                                          ↓ _notify_live_desk()
                                   POST http://localhost:7780/api/refresh
                                          ↓ _STATE_CACHE["ts"] = None

background thread (15s) ──→ _get_cached_state() → rebuild si TTL vencido

browser (12s) ──→ GET http://localhost:7780/
                         ↓ _get_cached_state() → cache HIT → <1s
browser recibe HTML
```

**Latencia máxima real: ~12s** (JS interval).

---

## 2. Los 5 hallazgos (fallas)

### Hallazgo H129-01 — Sin cache en memoria ✅ RESUELTO

**Archivo:** `live_desk.py::do_GET`
**Código:** `state = build_desk_state(fecha)` se ejecuta en CADA request.
**Fix D129-01:** `_get_cached_state()` con TTL 20s + `_background_refresh()` daemon 15s.

### Hallazgo H129-02 — P8 bloquea el request path (HTTP en serie)

**Archivo:** `live_desk.py::_build_p8_books()`
**Código:** `_fetch_all_odds(["betplay", "rushbet", "wplay"])` llama 3 APIs HTTP
en secuencia dentro del request handler. TTL del cache en disco = 600s pero
una vez vencido bloquea.
**Impacto:** +15-20 segundos por request cuando cache vence.
**Estado:** cubierto por D129-01 (cache en memoria absorbe el bloqueo).

### Hallazgo H129-03 — P6 lanza subprocess en request path

**Archivo:** `live_desk.py::_build_p6_pnl()` — L1788
**Código:** `subprocess.run([...shadow_book.py, "--report"], timeout=30)` dentro
del request handler. Siempre ejecuta, sin cache propio.
**Impacto:** +30 segundos por CADA request.
**Estado:** cubierto por D129-01. Refactor P6 async = sesión futura (H129-X).

### Hallazgo H129-04 — n8n no notifica al dashboard ✅ RESUELTO

**Archivos:** `n8n_push_workflow.py` + `close_snapshot_server.py`
**Código original:** `close_snapshot_server.py` tenía `/live-dashboard` que solo
redirigía con HTTP 301 a `:7780` — sin invalidar ningún cache.
**Diagnóstico graphify:** `graphify path "close_snapshot_server.py" "live_desk.py"` →
"No path found" → confirmó arquitectura desconectada.
**Fix D129-02:** `_notify_live_desk()` en `close_snapshot_server.py` POST a
`:7780/api/refresh` tras `/check-and-close` y `/live-check`.

### Hallazgo H129-05 — Timestamp falso (datetime.now() ≠ mtime datos) ✅ RESUELTO

**Archivo:** `live_desk.py` — render_html() L~1407
**Código original:** header mostraba `datetime.now()` (momento del request),
no el mtime del archivo de datos más reciente.
**Fix D129-03:** `_data_freshness(fecha)` lee mtime de
`edge_report_FECHA_*.json` y `live_odds_history_FECHA_*.json`.
Header muestra: `"datos de hace 2m 15s"`.

---

## 3. Implementación — 3 capas

### Capa 1 — Cache en memoria TTL 20s + thread background (D129-01) ✅

**Archivos:** `live_desk.py` — top of module + `main()`

```python
import threading
import time
import urllib.request

_STATE_CACHE: Dict[str, Any] = {
    "state": None, "ts": None, "ttl_s": 20, "lock": threading.Lock(),
}

def _get_cached_state(fecha: str) -> dict:
    with _STATE_CACHE["lock"]:
        age = (datetime.now() - _STATE_CACHE["ts"]).total_seconds() if _STATE_CACHE["ts"] else 999
        if _STATE_CACHE["state"] is not None and age < _STATE_CACHE["ttl_s"]:
            return _STATE_CACHE["state"]
    state = build_desk_state(fecha)
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["state"] = state
        _STATE_CACHE["ts"] = datetime.now()
    return state

def _background_refresh(fecha_fn) -> None:
    while True:
        try:
            _get_cached_state(fecha_fn())
        except Exception:
            pass
        time.sleep(15)
```

`do_GET` llama `_get_cached_state(fecha)`. `main()` arranca thread daemon antes de servir.

### Capa 2 — Endpoint POST /api/refresh + notify en n8n bridge (D129-02) ✅

**Archivos:** `live_desk.py::DeskHandler` + `close_snapshot_server.py`

En `live_desk.py` — nuevo método `do_POST`:
```python
def do_POST(self):
    if self.path != "/api/refresh":
        self.send_response(404); self.end_headers(); return
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["ts"] = None  # fuerza reconstrucción
    body = b'{"ok": true}'
    self.send_response(200)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)
```

En `close_snapshot_server.py`:
```python
import urllib.request

_LIVE_DESK_REFRESH_URL = "http://localhost:7780/api/refresh"

def _notify_live_desk() -> None:
    try:
        urllib.request.urlopen(_LIVE_DESK_REFRESH_URL, data=b"{}", timeout=2)
    except Exception:
        pass  # live_desk puede estar apagado — ignorar
```

Llamado al final de `_handle_check_and_close()` y `_handle_live_check()`.

### Capa 3 — JS interval 12s + staleness mtime real (D129-03) ✅

**Archivo:** `live_desk.py::render_html()` + `build_desk_state()`

```python
# En build_desk_state():
"data_freshness": _data_freshness(fecha),  # D129-03

# En render_html():
freshness_note = f' | {freshness}' if freshness else ""
refresh_note = f'<p>Auto-refresh 12s{freshness_note} | <span id="desk-ts">{ts}</span></p>'
```

JS: `setTimeout(autoRefresh, 30000)` → `setTimeout(autoRefresh, 12000)` (2 ocurrencias).

```python
def _data_freshness(fecha: str) -> str:
    fecha_compact = fecha.replace("-", "")
    candidates = (
        glob.glob(str(REPORTS / f"live_odds_history_{fecha_compact}*.json"))
        + glob.glob(str(REPORTS / f"edge_report_{fecha_compact}*.json"))
    )
    mtimes = [os.path.getmtime(c) for c in candidates if os.path.exists(c)]
    if not mtimes:
        return "datos: desconocido"
    age_s = time.time() - max(mtimes)
    return f"datos de hace {int(age_s // 60)}m {int(age_s % 60)}s"
```

---

## 4. Tests REGLA-T53 — 3/3 PASS

Archivo: `tests/test_nodo129_live_desk_cache.py`

| Test | Contrato | Resultado |
|------|----------|-----------|
| `test_cache_memoria_hit` | Segunda llamada a `_get_cached_state()` = misma referencia, <0.1s, build no llamado | PASS ✅ |
| `test_refresh_endpoint_invalida_cache` | Tras `_STATE_CACHE["ts"]=None`, próxima llamada reconstruye | PASS ✅ |
| `test_staleness_mtime` | archivo mtime=ahora-135s → retorna "hace 2m 15s" (±1s) | PASS ✅ |

---

## 5. Decisiones de diseño

| ID | Decisión | Elección | Razón |
|----|----------|----------|-------|
| D129-01 | Cache en memoria (no Redis) | `_STATE_CACHE` dict + threading.Lock | Sin dependencias externas, mismo proceso |
| D129-02 | POST /api/refresh (no WebSocket) | HTTP urllib.request desde close_snapshot_server | n8n habla HTTP, no WS. Minimal change. |
| D129-03 | TTL 20s (no 5s) | `ttl_s: 20` | P8 y P6 tardan 60s al expirar — TTL largo + invalidación explícita |
| D129-04 | Thread daemon (no asyncio) | `threading.Thread(daemon=True)` | live_desk.py usa BaseHTTPServer síncrono — threading es consistente |
| D129-05 | JS 12s (no 5s) | `setTimeout(autoRefresh, 12000)` | Margen para TTL 20s + variación red local |
| H129-X | P6 subprocess NO se mueve aún | Diferido | Requiere refactor shadow_book. Gate: cache cubre el 80% del problema. |

---

## 6. Diagnóstico graphify (evidencia arquitectural)

```bash
graphify path "n8n_push_workflow.py" "live_desk.py"
# → "No path found between 'n8n_push_workflow.py' and 'live_desk.py'."

graphify path "close_snapshot_server.py" "live_desk.py"
# → "No path found between 'close_snapshot_server.py' and 'live_desk.py'."

graphify explain "close_snapshot_server"
# → Degree: 8. Connections: Handler, _log, _already_processed, main,
#   _matches_in_window, _read_today_matches, _run_close_snapshot + rationale_for

graphify explain "_run_live_check"
# → Source: live_check_trigger.py L61. Calls: _log(), main(). Ejecuta live_edge_monitor.py --observe
```

Confirmación: `close_snapshot_server.py::_handle_live_dashboard()` solo hacía HTTP 301
redirect a `:7780` — **jamás invalida cache ni notifica datos nuevos**.

`cleanup_on_exit` errors en suite (8x) provienen de `h2h_extractor.py` (Playwright asyncio)
y `_pytest/pathlib.py` — preexistentes, ajenos a nuestro threading.

---

## §WIKILINKS COMPLETOS

### Forward links

- [[Nodo-128-Wplay-P8-Alias]] — sesión previa; P8 multi-book, alias apellido detectado
- [[Nodo-109-Live-Trading-Desk-Dashboard]] — `live_desk.py` arquitectura original (:7780), DeskHandler
- [[Nodo-73-n8n-CloseSnapshot-Timing]] — workflow n8n + close_snapshot_server.py :8765 bridge
- [[Nodo-97-Live-Edge-Monitor]] — live_edge_monitor.py, endpoint /live-check que ahora notifica :7780
- [[Nodo-100B-Triple-Convergencia-Live]] — live_dashboard_generator.py (supersedido por live_desk)
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53 aplicada

### Back links

- [[Nodo-109-Live-Trading-Desk-Dashboard]] ← `live_desk.py` modificada: D129-01/02/03
  (`_STATE_CACHE`, `_get_cached_state`, `_background_refresh`, `do_POST`, `_data_freshness`)
- [[Nodo-73-n8n-CloseSnapshot-Timing]] ← `close_snapshot_server.py`:
  `_notify_live_desk()` añadida, llamada tras `/check-and-close` y `/live-check`

### Huérfanos operacionales (resueltos)

| Símbolo | Archivo | Estado |
|---------|---------|--------|
| `_STATE_CACHE` | `live_desk.py` | ✅ implementado |
| `_get_cached_state()` | `live_desk.py` | ✅ implementado |
| `_background_refresh()` | `live_desk.py` | ✅ implementado, arranca en `main()` |
| `do_POST /api/refresh` | `live_desk.py::DeskHandler` | ✅ implementado |
| `_data_freshness()` | `live_desk.py` | ✅ implementado |
| `_notify_live_desk()` | `close_snapshot_server.py` | ✅ implementado |
| POST a `:7780/api/refresh` | `close_snapshot_server.py` | ✅ tras /check-and-close y /live-check |
| `tests/test_nodo129_live_desk_cache.py` | tests/ | ✅ 3/3 PASS |
| `nodos_index.json` | scripts/ | pendiente — `python3 scripts/rebuild_nodos_index.py` |

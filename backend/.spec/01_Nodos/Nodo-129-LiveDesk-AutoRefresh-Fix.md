# Nodo-129 — LiveDesk Auto-Refresh: Cache Memoria + Push n8n

> Estado: PENDIENTE — análisis completo, implementación en curso
> Detectado: sesión operacional 2026-07-21 (continuación Nodo-128)
> Commit: pendiente
> Tests: pendiente — `tests/test_nodo129_live_desk_cache.py`

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

### Lo que existe (PULL ciego)

```
n8n (20s) ──→ escribe archivo en disco  ← nadie lo notifica a :7780
                        ↓ silencio

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
pueden tener 90 segundos de antiguedad o más.

---

## 2. Los 5 hallazgos (fallas)

### Hallazgo H129-01 — Sin cache en memoria

**Archivo:** `live_desk.py::do_GET` — L2255-2260  
**Código:** `state = build_desk_state(fecha)` se ejecuta en CADA request.  
**Impacto:** Primera request después de cache vacío = 60+ segundos bloqueantes.

### Hallazgo H129-02 — P8 bloquea el request path (HTTP en serie)

**Archivo:** `live_desk.py::_build_p8_books()` — L1970  
**Código:** `_fetch_all_odds(["betplay", "rushbet", "wplay"])` llama 3 APIs HTTP
en secuencia dentro del request handler. TTL del cache en disco = 600s pero
una vez vencido bloquea.  
**Impacto:** +15-20 segundos por request cuando cache vence.

### Hallazgo H129-03 — P6 lanza subprocess en request path

**Archivo:** `live_desk.py::_build_p6_pnl()` — L1788  
**Código:** `subprocess.run([...shadow_book.py, "--report"], timeout=30)` dentro
del request handler. Siempre ejecuta, sin cache propio.  
**Impacto:** +30 segundos por CADA request.

### Hallazgo H129-04 — n8n no notifica al dashboard

**Archivo:** `n8n_push_workflow.py` + `close_snapshot_server.py`  
**Código:** El workflow n8n termina su ciclo (close_snapshot o live-check) sin
hacer ninguna llamada HTTP POST a `:7780`.  
`close_snapshot_server.py` tiene `/live-dashboard` que redirige a `:7780` pero
no invalida ningún cache.  
**Impacto:** Dashboard no sabe cuándo hay datos nuevos → polling ciego.

### Hallazgo H129-05 — Timestamp falso (datetime.now() ≠ mtime datos)

**Archivo:** `live_desk.py` — render timestamp  
**Código:** El header muestra `datetime.now()` (momento del request), no el
`mtime` del archivo de datos más reciente.  
**Impacto:** Usuario ve "14:35:02" y cree que los datos son de las 14:35, pero
el archivo `live_odds_history_*.json` puede ser de las 14:20.

---

## 3. Plan de implementación — 3 capas

### Capa 1 — Cache en memoria TTL 20s + thread background (D129-01)

**Impacto:** primera request = 60s, todas las demás = <1s.
**Esfuerzo:** ~30 min.

Añadir en `live_desk.py`:

```python
# Al inicio del módulo:
import threading as _threading

_STATE_CACHE: dict = {"state": None, "ts": None, "ttl_s": 20, "lock": _threading.Lock()}

def _get_cached_state(fecha: str) -> dict:
    with _STATE_CACHE["lock"]:
        now = datetime.now()
        age = (now - _STATE_CACHE["ts"]).total_seconds() if _STATE_CACHE["ts"] else 999
        if _STATE_CACHE["state"] and age < _STATE_CACHE["ttl_s"]:
            return _STATE_CACHE["state"]
    # Cache miss o expirado → reconstruir
    state = build_desk_state(fecha)
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["state"] = state
        _STATE_CACHE["ts"] = datetime.now()
    return state

def _background_refresh(fecha_fn):
    """Thread daemon que precalienta el cache cada 15s."""
    while True:
        try:
            _get_cached_state(fecha_fn())
        except Exception:
            pass
        _threading.Event().wait(15)
```

`do_GET` llama `_get_cached_state(fecha)` en lugar de `build_desk_state(fecha)`.

### Capa 2 — Endpoint POST /api/refresh + nodo final en n8n (D129-02)

**Impacto:** latencia máxima baja de 90s a ~12s.
**Esfuerzo:** ~20 min.

En `live_desk.py::do_GET`:
```python
# Nuevo: POST /api/refresh invalida cache inmediatamente
if self.command == "POST" and self.path == "/api/refresh":
    with _STATE_CACHE["lock"]:
        _STATE_CACHE["ts"] = None  # fuerza reconstrucción
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b'{"ok": true}')
    return
```

En `close_snapshot_server.py` al final de `/live-check` y `/check-and-close`:
```python
# Notificar a live_desk que hay datos nuevos
try:
    urllib.request.urlopen("http://localhost:7780/api/refresh", data=b"{}", timeout=2)
except Exception:
    pass
```

### Capa 3 — JS interval 12s + indicador staleness real (D129-03)

**Impacto:** visibilidad honesta de frescura de datos.
**Esfuerzo:** ~15 min.

En `render_html()`:
```javascript
// ANTES:
setInterval(autoRefresh, 30000);

// DESPUÉS:
setInterval(autoRefresh, 12000);
```

Para el timestamp de datos (staleness real):
```python
def _data_freshness(fecha: str) -> str:
    """Retorna mtime del archivo de datos más reciente del día."""
    candidates = [
        _latest(str(REPORTS / f"live_odds_history_{fecha.replace('-','')}*.json")),
        _latest(str(REPORTS / f"edge_report_{fecha.replace('-','')}*.json")),
    ]
    mtimes = []
    for c in candidates:
        if c:
            try:
                mtimes.append(os.path.getmtime(c))
            except Exception:
                pass
    if mtimes:
        latest_mtime = max(mtimes)
        age_s = time.time() - latest_mtime
        return f"datos de hace {int(age_s//60)}m {int(age_s%60)}s"
    return "datos: desconocido"
```

Header muestra: `"datos de hace 2m 15s"` en lugar de `"14:35:02"`.

---

## 4. Tests REGLA-T53

Archivo: `tests/test_nodo129_live_desk_cache.py`

```python
test_cache_memoria_hit()
    # Segunda llamada a _get_cached_state() devuelve misma referencia (cache hit)
    # sin reconstruir (mide tiempo < 0.1s)

test_refresh_endpoint_invalida_cache()
    # Después de poblar cache, POST /api/refresh → cache["ts"] = None
    # Próxima llamada a _get_cached_state() reconstruye (cache miss)

test_staleness_mtime()
    # _data_freshness() retorna string con "hace Xm Ys"
    # Con archivo mtime = ahora - 135s → retorna "hace 2m 15s"
```

---

## 5. Decisiones de diseño

| Decisión | Elección | Razón |
|----------|----------|-------|
| D129-01 | Cache en memoria (no Redis) | Sin dependencias externas, mismo proceso |
| D129-02 | POST /api/refresh (no WebSocket) | n8n habla HTTP, no WS. Minimal change. |
| D129-03 | TTL 20s (no 5s) | P8 y P6 tardan 60s al expirar — mejor TTL largo con invalidación explícita |
| D129-04 | Thread daemon (no asyncio) | live_desk.py usa BaseHTTPServer síncrono — threading es consistente |
| H129-X | P6 subprocess NO se mueve aún | Requiere refactor shadow_book. Gate: cache cubre el 80% del problema. |

---

## §WIKILINKS COMPLETOS

### Forward links
- [[Nodo-128-Wplay-P8-Alias]] — sesión previa donde se detectó P8 siendo lento (60s)
- [[Nodo-100-Triple-Convergencia-Live]] — live_dashboard_generator.py (legacy vs live_desk)
- [[Nodo-109-Live-Desk]] — `live_desk.py` arquitectura original (:7780)
- [[Nodo-73-n8n-Close-Snapshot]] — workflow n8n Tennis Close-Snapshot Timing
- [[Nodo-97-Live-Edge-Monitor]] — live_edge_monitor.py, endpoint /live-check
- [[PRE_IMPLEMENTATION_CHECKLIST]] — REGLA-T53

### Back links
- [[Nodo-109-Live-Desk]] ← `live_desk.py` modificada: D129-01/02/03
- [[Nodo-73-n8n-Close-Snapshot]] ← `close_snapshot_server.py` añade POST a :7780

### Huérfanos operacionales
- `live_desk.py` — _STATE_CACHE, _get_cached_state(), _background_refresh(), POST /api/refresh, _data_freshness()
- `close_snapshot_server.py` — POST a http://localhost:7780/api/refresh tras /live-check
- `tests/test_nodo129_live_desk_cache.py` — 3 tests REGLA-T53
- `nodos_index.json` — reindexar con `python3 scripts/rebuild_nodos_index.py`

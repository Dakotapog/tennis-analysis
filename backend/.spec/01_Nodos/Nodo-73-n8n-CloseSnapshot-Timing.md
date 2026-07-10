# Nodo-73 — n8n Close-Snapshot Precision Timing

**Fecha:** 2026-07-09
**Rama:** main
**Estado:** IMPLEMENTADO

---

## Contexto

El cron actual `*/10 9-23 * * *` dispara `close_snapshot_trigger.py` cada 10 minutos sin importar si hay partidos próximos. La deuda técnica documentada en FABLE_02 §4: "REGLA-SB-1 con horas fijas es aproximación; la hora real de cada partido varía. Cron fijo es la herramienta equivocada; n8n con lectura del feed es la correcta."

El campo `hora` en `data/zita_tennis_matches_FECHA.json` contiene el timestamp exacto ISO 8601 de inicio de cada partido (ej. `2026-07-09T06:00:00Z`). La señal existe — solo falta un scheduler que la use.

---

## Causa raíz

`close_snapshot_trigger.py` + cron fijo no puede dispararse en el momento exacto (T-15 min antes del partido) porque el cron no conoce el calendario de partidos. n8n como scheduler puede leer el campo `hora` y disparar con precisión.

---

## Arquitectura

```
n8n (Docker :5678)
    │
    │ cada 5 min
    ▼
[Schedule Trigger]
    │
    ▼
[HTTP Request] ──GET──► close_snapshot_server.py (:8765, host WSL2)
                              │
                              ├── lee zita_tennis_matches_FECHA.json
                              ├── filtra partidos en ventana T-10min a T-25min
                              ├── si hay partido → python3 shadow_book.py --close-snapshot
                              ├── registra en logs/n8n_snapshots.log
                              └── retorna {ok, snapshot_ran, matches_found}
    │
    ▼
[IF response.ok == false]
    │
    ▼
[Code] → logs/n8n_errors.log
```

**Por qué HTTP bridge y no Execute Command dentro de Docker:**
- El contenedor n8n (Alpine/Node) no tiene Python ni acceso al filesystem del backend.
- Un servidor HTTP ligero en el host desacopla el scheduler (n8n) de la ejecución (Python backend).
- `host.docker.internal` resuelve al host desde Docker Desktop WSL2.

---

## Archivos

| Archivo | Rol |
|---|---|
| `close_snapshot_server.py` | HTTP bridge en :8765 — lógica de timing + ejecución |
| `n8n_workflow_close_snapshot.json` | Definición del workflow (importado via API) |
| `logs/n8n_snapshots.log` | Registro append-only de cada snapshot ejecutado |
| `n8n_push_workflow.py` | Sube/actualiza workflow via API REST n8n (mantenimiento) |

> **Adendo 2026-07-09:** corrección de omisión en la tabla de archivos (no cambio de decisión) —
> `n8n_push_workflow.py` ya existía y se autodocumentaba como Nodo-73 desde su creación;
> faltaba en esta tabla. Detectado en auditoría SDD 2026-07-09.

---

## Ventana de disparo

- **Ventana:** T-25min a T-10min antes del `hora` del partido (zona UTC → local según offset del sistema)
- **Tolerancia:** si el script ya corrió en los últimos 8 min para el mismo partido → skip (deduplicación por match_id en log)
- **Fallback:** el cron existente `*/10` sigue activo como red de seguridad

---

## Decisiones

- **D73-01:** Puerto 8765 para el bridge — evita conflicto con Flask (5000), betslip_registrar (5001), n8n (5678).
- **D73-02:** Ventana T-25/T-10 en lugar de exacto T-15 — absorbe drift de 5 min del schedule trigger.
- **D73-03:** Cron existente NO se elimina — n8n es mejora, no reemplazo; si n8n cae, el cron sigue cubriendo.
- **D73-04:** Deduplicación por `match_id + fecha_dia` en `n8n_snapshots.log` — evita doble snapshot si n8n atrasa.

---

## Tests

- **T73-01:** `close_snapshot_server.py /health` → 200 OK
- **T73-02:** Partido en ventana T-20min → `snapshot_ran=true` en respuesta
- **T73-03:** Partido fuera de ventana (T-60min) → `snapshot_ran=false`
- **T73-04:** Sin archivo de partidos del día → `matches_found=0`, no crashea
- **T73-05:** Partido ya procesado (match_id en log) → `snapshot_ran=false` (deduplicación)

---

## Prohibido

- NO eliminar el cron `*/10` hasta que n8n tenga ≥7 días de uptime continuo.
- NO usar `Execute Command` dentro del contenedor n8n (no tiene Python).
- NO exponer el bridge en 0.0.0.0 sin firewall — solo localhost/Docker bridge.

---

## Auditoría de cierre

- [ ] `close_snapshot_server.py` responde /health → 200
- [ ] workflow activo en n8n UI
- [ ] `logs/n8n_snapshots.log` recibe entradas en horario de partidos
- [ ] 1756 tests siguen pasando

# Nodo-161 — Fire Guard Unificación (D161-01)

**Estado:** COMPLETO
**Fecha:** 2026-08-02
**Módulo principal:** `core/fire_guard.py` (nuevo), `live_desk.py`

---

## Contexto / Gap

Auditoría de conexiones ocultas (solicitada por el usuario, ver sesión 2026-08-02)
encontró 4 implementaciones del mismo patrón "guard de disparo-único" en el proyecto:

1. `certeza_fired_{fecha}.json` (`live_desk.py`, D147-06) — dict `pk -> timestamp`
   (pk = `partido_direccion`), **sin cap**, alerta única que nunca debe re-expirar.
2. `games_live_{fecha}_fired.json` (`live_desk.py`, D133-04) — lista de listas
   (conjunto ordenado de nombres de partido), **cap 10/día**.
3. `itf_live_games_{fecha}_fired.json` (`live_desk.py`, D150/D157) — misma forma
   que #2: lista de listas, **cap 10/día**.
4. `combos_live/{fecha}/_fired.json` (`scripts/live_edge_monitor.py`) — dict
   `event_id -> {fired_at, hora_inicio}`, usado ADEMÁS para TTL cleanup de `.bat`
   (D116-01) — semántica distinta, no solo forma de datos distinta.

Solo #2 y #3 son duplicación literal (misma forma, mismo cap, mismo propósito) —
candidatas reales a unificación. #1 y #4 tienen semántica propia y se dejan
deliberadamente sin tocar.

## Implementación

- **D161-01:** `core/fire_guard.py` — `should_fire(path, key, cap=10)` (solo
  lectura, `key not in fired` + `len(fired) < cap`) y `mark_fired(path, key)`
  (append + write, best-effort). Errores de I/O tratados como "sin historial",
  igual que el código original en ambos call-sites.
- Call-site D133-04 (`live_desk.py`, fire combo pre-game games_live): el guard-write
  (`mark_fired`) ocurre DENTRO del mismo `try` que el `subprocess.Popen()`, solo si
  el Popen no lanza excepción — preserva el comportamiento original exacto (no se
  marca disparado si el combo builder no pudo lanzarse).
- Call-site D150/D157 (`live_desk.py`, fire combo ITF live): `_fire_itf_live_games_combo()`
  (el escritor de `.bat`/`.html`) sigue corriendo **incondicionalmente cada ciclo
  de 15s** fuera del guard (D157-02 — los outcome_id expiran y deben refrescarse).
  El guard `fire_guard.should_fire(...)` sigue controlando únicamente el log
  `[ITF_LIVE] combo disparado`, el Telegram (D157-05) y el logging a shadow_book
  (D157-03) — sin cambio de comportamiento respecto al código pre-refactor.

## No-Goals

- No se unificó `certeza_fired_{fecha}.json` (D147-06) — dict sin cap con semántica
  de "una vez por siempre" distinta a "cap N/día".
- No se tocó el guard de `scripts/live_edge_monitor.py` (`_fired_path`/`_load_fired`/
  `_save_fired`) — comparte responsabilidad con TTL cleanup de `.bat` (D116-01),
  no es un guard puro.
- No se cambiaron umbrales, caps, ni la lógica de qué dispara cada combo — puramente
  extracción de la mecánica de persistencia del guard a un módulo compartido.

## Tests

`tests/test_nodo161_fire_guard.py` — 8 tests REGLA-T53 sobre `core/fire_guard.py`
directamente (sin historial dispara, key repetida no dispara, key nueva sí dispara,
cap alcanzado bloquea incluso key nueva, mark_fired persiste append, ciclo
should_fire→mark_fired→should_fire, JSON corrupto tratado como sin historial,
mark_fired no lanza si el directorio no existe). Regresión: 67 tests de los módulos
que consumen el guard (Nodo-133/147/150/151/157) — 67/67 PASS, incluyendo
`test_antiflood_no_refire` (D133-04) que ejercita el guard end-to-end.

## Wikilinks

- [[Nodo-133]] — `_check_games_convergencia()`, guard D133-04 original
- [[Nodo-150]] / [[Nodo-157]] — guard ITF live original (D150-06, D157-02/03/05)
- [[Nodo-147]] — `certeza_fired_{fecha}.json` (D147-06), deliberadamente NO unificado

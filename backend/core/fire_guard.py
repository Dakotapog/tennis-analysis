"""core/fire_guard.py — Nodo-161 D161-01.

Guard de disparo-único para señales live re-evaluadas cada ciclo (15s) que
solo deben notificar/loggear la PRIMERA vez que aparece un conjunto de
partidos dado. Extraído de la duplicación literal encontrada en
`live_desk.py` entre el guard de `games_live_{fecha}_fired.json` (D133-04)
y el de `itf_live_games_{fecha}_fired.json` (D150/D157) — mismo patrón:
lista de listas (conjunto ordenado de nombres de partido), cap de N
disparos/día, persistido en `reports/`.

No unifica (deliberadamente):
- `certeza_fired_{fecha}.json` (D147-06) — dict `pk -> timestamp`, SIN cap,
  una alerta única por partido+dirección que nunca debe re-expirar.
- El guard de `scripts/live_edge_monitor.py` (`_fired_path`/`_load_fired`/
  `_save_fired`) — dict `event_id -> {fired_at, hora_inicio}`, usado además
  para TTL cleanup de `.bat` (D116-01). Semántica distinta a un simple
  dedup de disparo, no solo forma de datos distinta.
"""
import json
from pathlib import Path
from typing import List


def should_fire(path: Path, key: List[str], cap: int = 10) -> bool:
    """True si `key` (ej. lista ordenada de partidos de un combo) todavía
    no fue registrada en `path` y el cap diario no se alcanzó. Solo lectura
    — no persiste nada. Errores de I/O se tratan como "sin historial" (fired=[]),
    igual que el código original."""
    try:
        fired: List[List[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        fired = []
    if len(fired) >= cap:
        return False
    return key not in fired


def mark_fired(path: Path, key: List[str]) -> None:
    """Registra `key` como disparada en `path` (append + write). Best-effort:
    igual que el código original, un fallo de escritura no propaga excepción."""
    try:
        fired: List[List[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        fired = []
    fired.append(key)
    try:
        path.write_text(json.dumps(fired, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

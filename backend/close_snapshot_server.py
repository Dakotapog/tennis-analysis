"""
close_snapshot_server.py — HTTP bridge para n8n Close-Snapshot Timing (Nodo-73)
Puerto 8765. n8n corre en Docker y no tiene acceso al Python del backend — este
servidor actúa como puente: n8n llama /check-and-close, el servidor decide si
hay partido próximo y ejecuta shadow_book.py --close-snapshot.

Endpoints:
  GET /health               → {"ok": true}
  GET /check-and-close      → {"ok", "snapshot_ran", "matches_found", "matches"}
  GET /status               → últimas 10 entradas del log

Uso:
  python3 close_snapshot_server.py           # puerto 8765
  python3 close_snapshot_server.py --port 8766
"""
import argparse
import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR  = Path(__file__).parent
LOG_FILE  = BASE_DIR / "logs" / "n8n_snapshots.log"
LOG_ERRORS = BASE_DIR / "logs" / "n8n_errors.log"

# Ventana de disparo: T-25min a T-10min antes del partido (D73-02)
_WINDOW_EARLY_MIN = 25
_WINDOW_LATE_MIN  = 10


# ─── Helpers ────────────────────────────────────────────────────────────────

def _log(msg: str, error: bool = False) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}\n"
    log_path = LOG_ERRORS if error else LOG_FILE
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip(), file=sys.stderr if error else sys.stdout)


def _read_today_matches() -> list[dict]:
    """Lee el archivo de partidos más reciente del día."""
    today = datetime.now().strftime("%Y%m%d")
    pattern = str(BASE_DIR / "data" / f"zita_tennis_matches_{today}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    matches = []
    if isinstance(data, dict):
        for torneo, partidos in data.items():
            for p in partidos:
                p["torneo_key"] = torneo
                matches.append(p)
    elif isinstance(data, list):
        matches = data
    return matches


def _matches_in_window(matches: list[dict]) -> list[dict]:
    """Filtra partidos cuyo 'hora' cae en la ventana T-25min a T-10min."""
    now_utc = datetime.now(timezone.utc)
    result = []
    for m in matches:
        hora = m.get("hora")
        if not hora:
            continue
        try:
            # ISO 8601 con Z
            match_time = datetime.fromisoformat(hora.replace("Z", "+00:00"))
            minutes_to_start = (match_time - now_utc).total_seconds() / 60
            if _WINDOW_LATE_MIN <= minutes_to_start <= _WINDOW_EARLY_MIN:
                m["minutes_to_start"] = round(minutes_to_start, 1)
                result.append(m)
        except (ValueError, TypeError):
            continue
    return result


def _already_processed(match_id: str) -> bool:
    """Deduplica: si el match_id ya aparece en el log de hoy, skip (D73-04)."""
    if not LOG_FILE.exists():
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    content = LOG_FILE.read_text(encoding="utf-8")
    return bool(re.search(rf"^{today}.*{re.escape(match_id)}", content, re.MULTILINE))


def _run_close_snapshot() -> tuple[int, str]:
    """Ejecuta shadow_book.py --close-snapshot. Retorna (returncode, output)."""
    r = subprocess.run(
        [sys.executable, str(BASE_DIR / "shadow_book.py"), "--close-snapshot"],
        capture_output=True, text=True, cwd=BASE_DIR, timeout=60
    )
    return r.returncode, (r.stdout + r.stderr).strip()


# ─── Handler ────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json(200, {"ok": True, "ts": datetime.now().isoformat()})

        elif path == "/check-and-close":
            self._handle_check_and_close()

        elif path == "/status":
            self._handle_status()

        else:
            self._send_json(404, {"ok": False, "error": f"ruta desconocida: {path}"})

    def _handle_check_and_close(self):
        matches = _read_today_matches()
        in_window = _matches_in_window(matches)

        if not in_window:
            self._send_json(200, {
                "ok": True, "snapshot_ran": False,
                "matches_found": len(matches),
                "in_window": 0,
                "msg": "sin partidos en ventana T-25/T-10"
            })
            return

        # Deduplicar: solo procesar match_ids no vistos hoy
        new_matches = [m for m in in_window
                       if not _already_processed(m.get("match_id", ""))]

        if not new_matches:
            self._send_json(200, {
                "ok": True, "snapshot_ran": False,
                "matches_found": len(matches),
                "in_window": len(in_window),
                "msg": "partidos en ventana ya procesados hoy"
            })
            return

        # Ejecutar close-snapshot
        rc, output = _run_close_snapshot()
        ok = rc == 0

        # Log
        ids = [m.get("match_id", "?") for m in new_matches]
        names = [f"{m.get('jugador1','?')} vs {m.get('jugador2','?')}" for m in new_matches]
        log_msg = (f"close-snapshot rc={rc} | "
                   f"partidos={','.join(ids)} | "
                   f"{' | '.join(names)}")
        _log(log_msg, error=not ok)

        status_code = 200 if ok else 500
        self._send_json(status_code, {
            "ok": ok,
            "snapshot_ran": True,
            "matches_found": len(matches),
            "in_window": len(in_window),
            "processed": ids,
            "rc": rc,
            "output": output[:500]
        })

    def _handle_status(self):
        lines = []
        if LOG_FILE.exists():
            all_lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
            lines = all_lines[-10:]
        self._send_json(200, {"ok": True, "log": lines})

    def log_message(self, fmt, *args):
        pass  # suprimir logs HTTP por defecto


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    LOG_FILE.parent.mkdir(exist_ok=True)
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    _log(f"[close-snapshot-server] escuchando en :{args.port}")
    _log(f"[close-snapshot-server] ventana disparo: T-{_WINDOW_EARLY_MIN}min a T-{_WINDOW_LATE_MIN}min")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("[close-snapshot-server] detenido")


if __name__ == "__main__":
    main()

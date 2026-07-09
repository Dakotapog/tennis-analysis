"""
close_snapshot_trigger.py — Trigger automático para Momento 2 del Shadow Book
Reemplaza el flujo n8n de FABLE_02 — corre localmente cada 10 min via cron.

Lógica (spec FABLE_02 §4 Fase 2):
  1. Lee shadow book de hoy → busca registros abiertos (sin cierre_kambi)
  2. Si hay registros abiertos → ejecuta shadow_book.py --close-snapshot
  3. Notifica por Telegram si TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID están en env
  4. NUNCA toca betslip_registrar ni registra apuestas (regla FABLE_02 Fase 2)

Uso (cron lo llama, también se puede correr manual):
  python3 close_snapshot_trigger.py          # modo normal
  python3 close_snapshot_trigger.py --dry-run # solo reporta, no ejecuta
  python3 close_snapshot_trigger.py --force   # ignora ventana horaria, siempre ejecuta
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR   = Path(__file__).parent
SHADOW_DIR = BASE_DIR / "reports" / "shadow_book"
LOG_FILE   = BASE_DIR / "logs" / "close_snapshot_trigger.log"

# Ventana de operación: 8am–11:30pm (fuera no ejecuta Kambi fetch para no desperdiciar llamadas)
HORA_INICIO = 8
HORA_FIN    = 23

# ─── Helpers ────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _jsonl_path(fecha: str) -> Path:
    return SHADOW_DIR / f"sb_{fecha}.jsonl"


def _open_records(fecha: str) -> list[dict]:
    """Devuelve registros sin cierre_kambi ni resolucion."""
    path = _jsonl_path(fecha)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_type") == "session_meta":
                continue
            if "resolucion" in rec:
                continue
            if "cierre_kambi" in rec:
                continue
            records.append(rec)
    return records


def _send_telegram(msg: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat, "text": msg, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        _log(f"[telegram] error: {e}")


def _run_close_snapshot(fecha: str, dry_run: bool) -> int:
    """Ejecuta shadow_book.py --close-snapshot. Retorna código de salida."""
    cmd = [sys.executable, str(BASE_DIR / "shadow_book.py"),
           "--close-snapshot", "--fecha", fecha]
    if dry_run:
        _log(f"[dry-run] would run: {' '.join(cmd)}")
        return 0
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(BASE_DIR), timeout=120)
        if result.stdout:
            _log(f"[snapshot] stdout: {result.stdout.strip()}")
        if result.stderr:
            _log(f"[snapshot] stderr: {result.stderr.strip()[:300]}")
        return result.returncode
    except subprocess.TimeoutExpired:
        _log("[snapshot] TIMEOUT — shadow_book.py tardó > 120s")
        return 1
    except Exception as e:
        _log(f"[snapshot] ERROR: {e}")
        return 1


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo reporta, no ejecuta shadow_book.py")
    parser.add_argument("--force", action="store_true",
                        help="Ignora ventana horaria, ejecuta siempre")
    args = parser.parse_args()

    now   = datetime.now()
    fecha = now.strftime("%Y-%m-%d")
    hora  = now.hour

    # ── 1. Ventana horaria ──────────────────────────────────────────────────
    if not args.force and not (HORA_INICIO <= hora <= HORA_FIN):
        # Silencioso fuera de ventana — no llenar logs con "nada que hacer"
        return

    # ── 2. Registros abiertos ───────────────────────────────────────────────
    open_recs = _open_records(fecha)
    if not open_recs:
        _log(f"[trigger] {fecha}: sin registros abiertos — nada que hacer")
        return

    def _nombre(r: dict) -> str:
        snap = r.get("pick_snapshot") or {}
        return (snap.get("jugador") or snap.get("player") or
                r.get("jugador") or r.get("player") or
                r.get("match_key", "?"))
    jugadores = [_nombre(r) for r in open_recs]
    _log(f"[trigger] {fecha}: {len(open_recs)} registros abiertos — "
         f"{', '.join(jugadores[:5])}{'...' if len(jugadores) > 5 else ''}")

    # ── 3. Ejecutar close-snapshot ──────────────────────────────────────────
    _log(f"[trigger] ejecutando shadow_book.py --close-snapshot ...")
    rc = _run_close_snapshot(fecha, args.dry_run)

    if rc == 0:
        msg = (f"CLOSE-SNAPSHOT OK  {now.strftime('%H:%M')}\n"
               f"Picks: {', '.join(jugadores[:5])}\n"
               f"CLV cierre capturado para {len(open_recs)} registro(s)")
        _log(f"[trigger] OK — {len(open_recs)} cierre(s) capturados")
        _send_telegram(msg)
    else:
        msg = (f"CLOSE-SNAPSHOT ERROR  {now.strftime('%H:%M')}\n"
               f"exit_code={rc} — revisar logs/close_snapshot_trigger.log")
        _log(f"[trigger] ERROR rc={rc}")
        _send_telegram(msg)


if __name__ == "__main__":
    main()

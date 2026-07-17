"""
live_check_trigger.py — Trigger automático para Live Edge Monitor (Nodo-97)
Fallback para cuando n8n no está disponible. Corre cada 2 min via cron.

Lógica:
  1. Solo opera en ventana horaria 8am-11pm
  2. Llama scripts/live_edge_monitor.py --observe
  3. Notifica por Telegram si hay triggers
  4. NUNCA toca betslip_registrar ni registra apuestas

Uso (cron lo llama, también se puede correr manual):
  python3 live_check_trigger.py           # modo normal
  python3 live_check_trigger.py --dry-run # solo reporta, no ejecuta
  python3 live_check_trigger.py --force   # ignora ventana horaria
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "logs" / "live_check_trigger.log"

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


def _run_live_check(dry_run: bool) -> tuple[int, str]:
    """Ejecuta live_edge_monitor.py --observe. Retorna (returncode, output)."""
    cmd = [sys.executable, str(BASE_DIR / "scripts" / "live_edge_monitor.py"), "--observe"]
    if dry_run:
        _log(f"[dry-run] would run: {' '.join(cmd)}")
        return 0, ""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(BASE_DIR), timeout=60)
        output = (result.stdout + result.stderr).strip()
        if output:
            _log(f"[live-check] {output[:300]}")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        _log("[live-check] TIMEOUT — live_edge_monitor.py tardó > 60s")
        return 1, "timeout"
    except Exception as e:
        _log(f"[live-check] ERROR: {e}")
        return 1, str(e)


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo reporta, no ejecuta live_edge_monitor.py")
    parser.add_argument("--force", action="store_true",
                        help="Ignora ventana horaria, ejecuta siempre")
    args = parser.parse_args()

    now  = datetime.now()
    hora = now.hour

    if not args.force and not (HORA_INICIO <= hora <= HORA_FIN):
        return  # Silencioso fuera de ventana

    _log("[live-check] iniciando chequeo de triggers live ...")
    rc, output = _run_live_check(args.dry_run)

    import re as _re
    n_triggers = 0
    m = _re.search(r'"n_triggers"\s*:\s*(\d+)', output)
    if m:
        n_triggers = int(m.group(1))

    if rc == 0:
        if n_triggers > 0:
            msg = (f"LIVE EDGE TRIGGER  {now.strftime('%H:%M')}\n"
                   f"n_triggers={n_triggers} — revisar Desktop/combos/ o Telegram")
            _log(f"[live-check] OK — {n_triggers} trigger(s) encontrados")
            _send_telegram(msg)
        else:
            _log(f"[live-check] OK — sin triggers live en este ciclo")
    else:
        msg = (f"LIVE-CHECK ERROR  {now.strftime('%H:%M')}\n"
               f"exit_code={rc} — revisar logs/live_check_trigger.log")
        _log(f"[live-check] ERROR rc={rc}")
        _send_telegram(msg)


if __name__ == "__main__":
    main()

"""
n8n_push_workflow.py — Sube workflows a n8n via API REST.
Nodo-73 (close-snapshot) + Nodo-97 (live-check).

Uso:
  python3 n8n_push_workflow.py --api-key <TU_API_KEY>               # ambos workflows
  python3 n8n_push_workflow.py --api-key <TU_API_KEY> --live-only   # solo live-check
  python3 n8n_push_workflow.py --api-key <TU_API_KEY> --snap-only   # solo close-snapshot
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

N8N_URL      = "http://localhost:5678"
WF_FILE      = Path(__file__).parent / "n8n_workflow_close_snapshot.json"
WF_LIVE_FILE = Path(__file__).parent / "n8n_workflow_live_check.json"


def _api(method: str, path: str, body: dict | None, api_key: str) -> dict:
    url = f"{N8N_URL}/api/v1{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-N8N-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"[ERROR] {method} {path} → {e.code}: {err[:300]}", file=sys.stderr)
        sys.exit(1)


def _push_workflow(wf_file: Path, api_key: str, dry_run: bool) -> None:
    """Sube o actualiza un workflow en n8n."""
    workflow = json.loads(wf_file.read_text(encoding="utf-8"))
    print(f"[n8n] Workflow: {workflow['name']} ({len(workflow['nodes'])} nodos)")

    if dry_run:
        print("[dry-run] No subido.")
        return

    existing = _api("GET", "/workflows?limit=50", None, api_key)
    wf_id = None
    for wf in existing.get("data", []):
        if wf.get("name") == workflow["name"]:
            wf_id = wf["id"]
            print(f"[n8n] Existente encontrado: id={wf_id}")
            break

    payload = {k: v for k, v in workflow.items() if k not in ("active", "tags")}

    if wf_id:
        result = _api("PUT", f"/workflows/{wf_id}", payload, api_key)
        print(f"[n8n] Actualizado: id={result.get('id')}")
    else:
        result = _api("POST", "/workflows", payload, api_key)
        wf_id = result.get("id")
        print(f"[n8n] Creado: id={wf_id}")

    _api("POST", f"/workflows/{wf_id}/activate", None, api_key)
    print(f"[n8n] ACTIVO → http://localhost:5678/workflow/{wf_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="API key de n8n")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-only", action="store_true",
                        help="Solo subir workflow live-check (Nodo-97)")
    parser.add_argument("--snap-only", action="store_true",
                        help="Solo subir workflow close-snapshot (Nodo-73)")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] Modo simulación — no se sube nada.")

    # Verificar que n8n responde
    try:
        _api("GET", "/workflows?limit=1", None, args.api_key)
    except SystemExit:
        print("[ERROR] No se puede conectar a n8n en localhost:5678", file=sys.stderr)
        sys.exit(1)

    push_snap = not args.live_only
    push_live = not args.snap_only

    if push_snap:
        print("\n── Close-Snapshot (Nodo-73) ─────────────────────────────")
        _push_workflow(WF_FILE, args.api_key, args.dry_run)
        print(f"[n8n] Dispara cada 5 min via :8765/check-and-close")

    if push_live:
        print("\n── Live Edge Monitor (Nodo-97) ──────────────────────────")
        _push_workflow(WF_LIVE_FILE, args.api_key, args.dry_run)
        print(f"[n8n] Dispara cada 2 min via :8765/live-check")


if __name__ == "__main__":
    main()

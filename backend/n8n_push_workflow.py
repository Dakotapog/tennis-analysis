"""
n8n_push_workflow.py — Sube el workflow de close-snapshot a n8n via API REST.
Nodo-73. Uso: python3 n8n_push_workflow.py --api-key <TU_API_KEY>
"""
import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

N8N_URL  = "http://localhost:5678"
WF_FILE  = Path(__file__).parent / "n8n_workflow_close_snapshot.json"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="API key de n8n")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workflow = json.loads(WF_FILE.read_text(encoding="utf-8"))
    print(f"[n8n] Workflow: {workflow['name']}")
    print(f"[n8n] Nodos: {len(workflow['nodes'])}")

    if args.dry_run:
        print("[dry-run] No subido.")
        return

    # Verificar que n8n responde
    try:
        _api("GET", "/workflows?limit=1", None, args.api_key)
    except SystemExit:
        print("[ERROR] No se puede conectar a n8n en localhost:5678", file=sys.stderr)
        sys.exit(1)

    # Buscar si ya existe el workflow
    existing = _api("GET", "/workflows?limit=50", None, args.api_key)
    wf_id = None
    for wf in existing.get("data", []):
        if wf.get("name") == workflow["name"]:
            wf_id = wf["id"]
            print(f"[n8n] Workflow existente encontrado: id={wf_id}")
            break

    # n8n no acepta 'active' ni 'tags' en el body de creación/actualización
    payload = {k: v for k, v in workflow.items() if k not in ("active", "tags")}

    if wf_id:
        # Actualizar
        result = _api("PUT", f"/workflows/{wf_id}", payload, args.api_key)
        print(f"[n8n] Workflow actualizado: id={result.get('id')}")
    else:
        # Crear nuevo
        result = _api("POST", "/workflows", payload, args.api_key)
        wf_id = result.get("id")
        print(f"[n8n] Workflow creado: id={wf_id}")

    # Activar via endpoint dedicado
    _api("POST", f"/workflows/{wf_id}/activate", None, args.api_key)
    print(f"[n8n] Workflow ACTIVO en http://localhost:5678/workflow/{wf_id}")
    print(f"[n8n] Listo. El workflow dispara cada 5 min via close_snapshot_server.py :8765")


if __name__ == "__main__":
    main()

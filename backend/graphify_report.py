#!/usr/bin/env python3
"""
graphify report — Reporte Markdown del grafo de código.

Uso:
    python3 graphify_report.py
    python3 graphify_report.py --output reports/graphify_report_20260712.md

Contenido:
    1. Top 20 nodos por grado (centralidad)
    2. Huérfanos SDD (fuente: nodos_index.json — NO reimplementa la lógica)
    3. Resumen de comunidades (top 5 nodos por comunidad)
    4. Delta contra reporte anterior si existe en reports/

Nodo-88: Graphify Fase 4 (2026-07-12)
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent
GRAPH_JSON   = BACKEND_ROOT / "graphify-out" / "graph.json"
NODOS_INDEX  = BACKEND_ROOT / "nodos_index.json"
REPORTS_DIR  = BACKEND_ROOT / "reports"


def _previous_report(reports_dir: Path) -> Path | None:
    pat = re.compile(r"graphify_report_(\d{8})\.md$")
    candidates = [(m.group(1), f) for f in reports_dir.glob("graphify_report_*.md") if (m := pat.match(f.name))]
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _parse_prev(path: Path) -> dict:
    try:
        text = path.read_text()
        def grep(pattern):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else None
        top = {}
        in_top = False
        for line in text.splitlines():
            if "Top 20" in line:
                in_top = True
            if in_top and line.startswith("| ") and "---" not in line and "Nodo" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    try:
                        top[parts[1]] = int(parts[2])
                    except ValueError:
                        pass
        return {
            "nodes": grep(r"\*\*Nodos:\*\* (\d+)"),
            "edges": grep(r"\*\*Edges:\*\* (\d+)"),
            "communities": grep(r"\*\*Comunidades:\*\* (\d+)"),
            "top": top,
            "name": path.name,
        }
    except Exception:
        return {}


def generate(data: dict, nodos_idx: dict, prev_path: Path | None) -> str:
    # Degree
    deg: dict[str, int] = defaultdict(int)
    for lnk in data["links"]:
        deg[lnk["source"]] += 1
        deg[lnk["target"]] += 1
    for n in data["nodes"]:
        n["_deg"] = deg.get(n["id"], 0)

    nodes = data["nodes"]
    num_n, num_e = len(nodes), len(data["links"])
    communities = {n["community"] for n in nodes}
    num_c = len(communities)
    built_at = data.get("built_at_commit", "?")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    out = [
        f"# Graphify Report — {datetime.now().strftime('%Y-%m-%d')}",
        f"\n_Generado: {now}_\n",
        f"**Nodos:** {num_n} | **Edges:** {num_e} | **Comunidades:** {num_c} | Commit: `{built_at}`\n",
    ]

    # --- Delta ---
    prev = _parse_prev(prev_path) if prev_path else {}
    if prev:
        dn = num_n - (prev.get("nodes") or 0)
        de = num_e - (prev.get("edges") or 0)
        dc = num_c - (prev.get("communities") or 0)
        out += [
            f"## Delta vs `{prev['name']}`\n",
            "| Métrica | Anterior | Actual | Delta |",
            "|---|---|---|---|",
            f"| Nodos | {prev.get('nodes','-')} | {num_n} | {dn:+d} |",
            f"| Edges | {prev.get('edges','-')} | {num_e} | {de:+d} |",
            f"| Comunidades | {prev.get('communities','-')} | {num_c} | {dc:+d} |",
            "",
        ]
    else:
        out.append("_(No se encontró reporte anterior para delta.)_\n")

    # --- Top 20 ---
    top20 = sorted(nodes, key=lambda n: n["_deg"], reverse=True)[:20]
    out += [
        "## Top 20 nodos por grado (centralidad)\n",
        "_Si este nodo se rompe, más componentes se ven afectados._\n",
        "| Nodo | Grado | Tipo | Archivo fuente |",
        "|---|---|---|---|",
    ]
    for n in top20:
        lbl = n["label"]
        typ = "file" if lbl.endswith(".py") else ("fn" if "()" in lbl else "class")
        out.append(f"| `{lbl}` | {n['_deg']} | {typ} | `{n.get('source_file') or '-'}` |")

    if prev.get("top"):
        changes = [(n["label"], prev["top"][n["label"]], n["_deg"]) for n in top20 if n["label"] in prev["top"] and prev["top"][n["label"]] != n["_deg"]]
        if changes:
            out.append("\n**Cambios de centralidad respecto al reporte anterior:**\n")
            for lbl, old_deg, new_deg in changes:
                out.append(f"- `{lbl}`: {old_deg} → {new_deg} ({new_deg-old_deg:+d})")
            out.append("")

    # --- Huérfanos SDD ---
    meta = nodos_idx.get("_meta", {})
    huerfanos = nodos_idx.get("huerfanos", [])
    out += [
        "\n## Huérfanos SDD (archivos sin Nodo)\n",
        "_Fuente única: `nodos_index.json` — misma que `check_contradictions.py`. No reimplementa lógica._\n",
        f"- Archivos .py rastreados: **{meta.get('total_py_files', '?')}**",
        f"- Huérfanos oficiales: **{meta.get('huerfanos_count', len(huerfanos))}**",
        f"- Índice generado: {meta.get('generado', '?')}\n",
    ]
    if huerfanos:
        out += ["| Archivo huérfano |", "|---|"]
        for h in sorted(huerfanos):
            out.append(f"| `{h}` |")
    else:
        out.append("_Cobertura SDD: 100% — ningún archivo sin Nodo._ ✓")

    # --- Comunidades ---
    comm_nodes: dict[int, list] = defaultdict(list)
    for n in nodes:
        comm_nodes[n["community"]].append(n)
    comm_sorted = sorted(comm_nodes.items(), key=lambda x: len(x[1]), reverse=True)

    out += [
        "\n\n## Resumen de comunidades\n",
        "_Top 5 nodos por grado dentro de cada comunidad (lectura rápida de qué trata el clúster)._\n",
        "| Comunidad | Nodos | Top representantes |",
        "|---|---|---|",
    ]
    for cid, cnodes in comm_sorted:
        top5 = sorted(cnodes, key=lambda n: n["_deg"], reverse=True)[:5]
        reprs = ", ".join(f"`{n['label']}`" for n in top5)
        out.append(f"| Community {cid} | {len(cnodes)} | {reprs} |")

    out.append(f"\n---\n_graphify report · {now}_")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Graphify Markdown report — Nodo-88")
    parser.add_argument("--output", help="Output path (default: reports/graphify_report_YYYYMMDD.md)")
    args = parser.parse_args()

    if not GRAPH_JSON.exists():
        raise SystemExit(f"ERROR: {GRAPH_JSON} not found. Run `graphify update .` first.")

    data = json.loads(GRAPH_JSON.read_text())
    nodos_idx = json.loads(NODOS_INDEX.read_text()) if NODOS_INDEX.exists() else {"nodos": [], "huerfanos": [], "_meta": {}}

    REPORTS_DIR.mkdir(exist_ok=True)
    prev = _previous_report(REPORTS_DIR)

    out_path = Path(args.output) if args.output else REPORTS_DIR / f"graphify_report_{datetime.now().strftime('%Y%m%d')}.md"
    report = generate(data, nodos_idx, prev)
    out_path.write_text(report)
    print(f"Reporte: {out_path}")
    print(f"Nodos: {len(data['nodes'])} | Edges: {len(data['links'])} | Comunidades: {len({n['community'] for n in data['nodes']})}")


if __name__ == "__main__":
    main()

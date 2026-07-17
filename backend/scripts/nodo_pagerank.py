"""
D105-03 — PageRank sobre grafo de Nodos (Nodo-105)

Aplica el mismo algoritmo de analysis/erdos_graph.py (T20-01/Nodo-20) al
grafo de Nodos del vault: qué Nodo es crítico porque está conectado a los
Nodos que importan, no solo por tener muchas conexiones brutas.

Un Nodo con pocas conexiones puede tener alto PageRank si conecta clusters
que de otro modo estarían desconectados — ese es el alpha del análisis.

Output: nodos_pagerank.json  {nodo_id: score_0_a_1}
        nodos_pagerank_report.md  (tabla top-20, opcional --report)

Uso:
    python3 scripts/nodo_pagerank.py
    python3 scripts/nodo_pagerank.py --report
    cat nodos_pagerank.json | python3 -c "
        import json,sys
        d=json.load(sys.stdin)
        [print(f'{v:.4f} {k}') for k,v in sorted(d.items(),key=lambda x:-x[1])[:10]]"
"""

import json
import re
import sys
from pathlib import Path

# ── rutas ──────────────────────────────────────────────────────────────────
BACKEND = Path(__file__).resolve().parent.parent
SPEC_DIR = BACKEND / ".spec" / "01_Nodos"
OUTPUT_JSON = BACKEND / "nodos_pagerank.json"
OUTPUT_REPORT = BACKEND / "reports" / "nodos_pagerank_report.md"

sys.path.insert(0, str(BACKEND))
from analysis.erdos_graph import pagerank_grafo  # T20-01, REGLA-T53


def extraer_wikilinks(texto: str) -> list[str]:
    """Extrae los IDs de Nodo de los wikilinks [[Nodo-XX-...]] en un archivo."""
    hits = re.findall(r'\[\[Nodo-(\d+)[^\]]*\]\]', texto)
    return list(set(hits))


def construir_grafo_nodos(spec_dir: Path) -> tuple[dict, dict]:
    """
    Escanea todos los Nodo-*.md y construye:
      - grafo: {nodo_id: {vecino_id: 1.0}} (dirigido por wikilink saliente)
      - meta:  {nodo_id: {'titulo': str, 'archivo': str}}

    Sólo incluye nodos que SÍ existen como archivos en spec_dir.
    Los wikilinks a nodos fantasma se ignoran silenciosamente.
    """
    archivos = sorted(spec_dir.glob("Nodo-*.md"))
    existentes: set[str] = set()
    contenidos: dict[str, tuple[str, str]] = {}  # id → (titulo, texto)

    for f in archivos:
        m = re.match(r'Nodo-(\d+)', f.stem)
        if not m:
            continue
        nodo_id = m.group(1).lstrip('0') or '0'
        texto = f.read_text(encoding='utf-8', errors='replace')
        titulo = f.stem
        existentes.add(nodo_id)
        contenidos[nodo_id] = (titulo, texto)

    # Grafo dirigido: wikilink A→B = A cita a B
    grafo: dict[str, dict[str, float]] = {}
    meta: dict[str, dict] = {}

    for nodo_id, (titulo, texto) in contenidos.items():
        vecinos_raw = extraer_wikilinks(texto)
        # Filtrar auto-referencias y nodos que no existen en el vault
        vecinos = [
            v.lstrip('0') or '0'
            for v in vecinos_raw
            if (v.lstrip('0') or '0') != nodo_id and (v.lstrip('0') or '0') in existentes
        ]
        grafo[nodo_id] = {v: 1.0 for v in vecinos}
        meta[nodo_id] = {'titulo': titulo, 'archivo': str(SPEC_DIR / f'Nodo-{nodo_id}*.md')}

    return grafo, meta


def run(report: bool = False) -> dict:
    """Calcula PageRank y escribe nodos_pagerank.json. Retorna el dict de scores."""
    grafo, meta = construir_grafo_nodos(SPEC_DIR)

    n_nodos = len(grafo)
    n_edges = sum(len(v) for v in grafo.values())
    print(f"[pagerank] {n_nodos} nodos, {n_edges} aristas wikilink")

    if n_nodos < 5:
        print("[pagerank] WARN: menos de 5 nodos — PageRank no calculable (REGLA-T20-1)")
        return {}

    scores = pagerank_grafo(grafo, damping=0.85, iteraciones=50)

    OUTPUT_JSON.write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"[pagerank] → {OUTPUT_JSON}")

    # Top-10 consola
    ranking = sorted(scores.items(), key=lambda x: -x[1])
    print("\n── Top-10 Nodos por PageRank ──────────────────────────────")
    for i, (nid, score) in enumerate(ranking[:10], 1):
        titulo = meta.get(nid, {}).get('titulo', f'Nodo-{nid}')
        print(f"  {i:2d}. {score:.4f}  {titulo}")

    if report:
        _escribir_report(ranking, meta, scores, n_nodos, n_edges)

    return scores


def _escribir_report(ranking, meta, scores, n_nodos, n_edges):
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Nodos PageRank Report (D105-03)",
        "",
        f"**Nodos:** {n_nodos} | **Aristas wikilink:** {n_edges}",
        f"**Algoritmo:** power iteration damping=0.85, 50 iter (reutiliza T20-01)",
        "",
        "## Top-20 Nodos por Importancia Estructural",
        "",
        "| Rank | Score | Nodo |",
        "|------|-------|------|",
    ]
    for i, (nid, score) in enumerate(ranking[:20], 1):
        titulo = meta.get(nid, {}).get('titulo', f'Nodo-{nid}')
        lines.append(f"| {i} | {score:.4f} | [[{titulo}]] |")

    lines += [
        "",
        "## Nodos Huérfanos (PageRank=0, sin wikilinks entrantes)",
        "",
    ]
    huerfanos = [nid for nid, s in scores.items() if s == 0]
    if huerfanos:
        for nid in sorted(huerfanos, key=lambda x: int(x)):
            titulo = meta.get(nid, {}).get('titulo', f'Nodo-{nid}')
            lines.append(f"- [[{titulo}]]")
    else:
        lines.append("_Ninguno — todos los nodos tienen al menos una cita._")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding='utf-8')
    print(f"[pagerank] report → {OUTPUT_REPORT}")


if __name__ == "__main__":
    report_flag = "--report" in sys.argv
    result = run(report=report_flag)
    if not result:
        sys.exit(1)

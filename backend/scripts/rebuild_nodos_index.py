#!/usr/bin/env python3
"""
scripts/rebuild_nodos_index.py — Nodo-75: Genera nodos_index.json

Lee todos los Nodo-*.md en .spec/01_Nodos/ y produce un índice JSON con:
  - id, archivo_py, estado, fecha, archivos_mencionados
  - huerfanos: archivos .py sin cobertura de nodo

Uso:
  python3 scripts/rebuild_nodos_index.py           # genera nodos_index.json
  python3 scripts/rebuild_nodos_index.py --dry-run # imprime sin escribir
"""
import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SPEC_DIR = BASE_DIR / ".spec" / "01_Nodos"
OUTPUT   = BASE_DIR / "nodos_index.json"

# Archivos .py con cobertura implícita (Nodo antiguo) o explícitamente exentos
# Valor = número de Nodo si hay cobertura; None = exento sin nodo (baja prioridad)
_EXCLUSIONES: dict[str, str | None] = {
    # Cobertura implícita en nodo antiguo — nombre de archivo no aparece literalmente
    "rivalry_analyzer.py":            "Nodo-32",
    "edge_calculator.py":             "Nodo-35",
    "trader_ev_tenis.py":             "Nodo-55",
    "shadow_book.py":                 "Nodo-27",
    "combo_confianza_builder.py":     "Nodo-62",
    "betplay_combo_builder.py":       "Nodo-26",
    "extraer_URL_partidos_version2.py": "Nodo-51",
    "extraer_historh2h.py":           "Nodo-49",
    # Fase 4 instrumentos REPORTE_SOLO — documentados en CLAUDE.md §6, no cambian decisiones
    "analysis/conformal_band.py":   "Nodos-64-71",
    "analysis/drift_monitor.py":    "Nodos-64-71",
    "analysis/flb_curve.py":        "Nodos-64-71",
    "analysis/pattern_audit.py":    "Nodos-64-71",
    "analysis/rho_empirical.py":    "Nodos-64-71",
    "analysis/velocity_monitor.py": "Nodos-64-71",
    # Exentos sin nodo (ML suspendido o utilidad sin decisión técnica)
    "extraer_ranking_atp_version2.py": None,   # PASO 0 — sin lógica KL
    "extraer_ranking_wta_version2.py": None,   # PASO 0 — sin lógica KL
    "session_compiler.py":            None,    # audit-trail Fase 5, sin motor propio
    "Intelligent_ml_enhancer.py":     None,    # ML suspendido hasta >78% held-out
    "feature_engineering.py":         None,    # ML suspendido
}

# Subdirectorios a excluir del barrido de huérfanos
_SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules", "migrations",
              "scripts", "tests", "analysis", "scraping", "validation",
              "core", "routes", "models", "services", "docs"}


def _parse_nodo(path: Path) -> dict | None:
    """Extrae metadatos de un Nodo-*.md. Retorna None si no es parseable."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # ID desde nombre de archivo: Nodo-63-Anchor-Combo-Builder.md → "63"
    m = re.match(r"Nodo-(\d+)", path.stem, re.IGNORECASE)
    if not m:
        return None
    nodo_id = m.group(1)

    # Fecha — primera línea que empiece con **Fecha:** o > Fecha:
    fecha = None
    for line in lines[:20]:
        mf = re.search(r'(?:\*\*Fecha:\*\*|>?\s*Fecha:)\s*(\d{4}-\d{2}-\d{2})', line)
        if mf:
            fecha = mf.group(1)
            break

    # Estado — YAML frontmatter `estado:` (D105-04) o cuerpo `**Estado:**`
    estado = None
    # 1) YAML frontmatter entre --- delimiters
    if lines and lines[0].strip() == '---':
        for line in lines[1:20]:
            if line.strip() == '---':
                break
            ms = re.match(r'\s*estado\s*:\s*(.+)', line, re.IGNORECASE)
            if ms:
                estado = ms.group(1).strip()
                break
    # 2) Cuerpo del doc: **Estado:** o > Estado:
    if estado is None:
        for line in lines[:30]:
            ms = re.search(r'(?:\*\*Estado:\*\*|>?\s*Estado:)\s*(.+)', line)
            if ms:
                estado = ms.group(1).strip().rstrip("*").strip()
                break
    # 3) Normalizar a taxonomía D105-04: activo/gateado/suspendido/historico
    nid = int(nodo_id) if nodo_id.isdigit() else 0
    if estado:
        e = estado.lower()
        if "activo" in e:
            estado = "activo"
        elif "gateado" in e or "gate" in e:
            estado = "gateado"
        elif "suspendido" in e:
            estado = "suspendido"
        else:
            # Derivar desde nodo_id — el cuerpo tiene texto verbose sin taxonomía D105-04
            if nid >= 100:
                estado = "activo"
            elif nid in (39, 41):
                estado = "suspendido"
            elif nid in (82, 90):
                estado = "gateado"
            else:
                estado = "historico"
    else:
        if nid >= 100:
            estado = "activo"
        elif nid in (39, 41):
            estado = "suspendido"
        elif nid in (82, 90):
            estado = "gateado"
        else:
            estado = "historico"

    # Archivos mencionados — cualquier token *.py encontrado en el texto
    archivos_mencionados = sorted(set(re.findall(r'\b[\w./]+\.py\b', text)))

    return {
        "nodo_id": nodo_id,
        "nodo_archivo": path.name,
        "fecha": fecha,
        "estado": estado,
        "archivos_mencionados": archivos_mencionados,
    }


def _collect_py_files() -> list[str]:
    """Recoge todos los .py del repo (raíz + subdirectorios seleccionados)."""
    files = []
    for p in BASE_DIR.iterdir():
        if p.suffix == ".py" and p.is_file():
            files.append(p.name)
    # Subdirectorios relevantes del pipeline
    for subdir_name in ("analysis", "scraping", "validation", "core"):
        subdir = BASE_DIR / subdir_name
        if subdir.is_dir():
            for p in subdir.glob("*.py"):
                files.append(f"{subdir_name}/{p.name}")
    return sorted(set(files))


def build_index(dry_run: bool = False) -> dict:
    if not SPEC_DIR.exists():
        print(f"[rebuild_nodos_index] ERROR: {SPEC_DIR} no existe", file=sys.stderr)
        sys.exit(1)

    # 1. Parsear todos los Nodo-*.md
    nodos = []
    for md in sorted(SPEC_DIR.glob("Nodo-*.md")):
        entry = _parse_nodo(md)
        if entry:
            nodos.append(entry)

    # 2. Construir conjunto de archivos cubiertos
    cubiertos: set[str] = set()
    for nodo in nodos:
        for f in nodo["archivos_mencionados"]:
            # Normalizar: quitar prefijos de path relativos
            cubiertos.add(Path(f).name)

    # Añadir cobertura implícita de _EXCLUSIONES
    for nombre_archivo in _EXCLUSIONES:
        cubiertos.add(Path(nombre_archivo).name)

    # 3. Detectar huérfanos
    py_files = _collect_py_files()
    huerfanos = []
    for f in py_files:
        basename = Path(f).name
        if basename.startswith("__") or basename == "conftest.py":
            continue
        if basename not in cubiertos:
            # Verificar por ruta completa también
            if f not in cubiertos:
                huerfanos.append(f)

    # 4. Construir índice final
    index = {
        "_meta": {
            "generado": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "total_nodos": len(nodos),
            "total_py_files": len(py_files),
            "huerfanos_count": len(huerfanos),
            "exclusiones_count": len(_EXCLUSIONES),
        },
        "exclusiones": _EXCLUSIONES,
        "nodos": nodos,
        "huerfanos": huerfanos,
    }

    # 5. Output
    json_str = json.dumps(index, indent=2, ensure_ascii=False)

    if dry_run:
        print("=" * 60)
        print("DRY-RUN — nodos_index.json NO escrito")
        print("=" * 60)
        print(json_str)
        print("=" * 60)
        print(f"Nodos parseados: {len(nodos)}")
        print(f"Archivos .py en repo: {len(py_files)}")
        print(f"Huérfanos detectados: {len(huerfanos)}")
        if huerfanos:
            print("Huérfanos:")
            for h in huerfanos:
                print(f"  - {h}")
    else:
        OUTPUT.write_text(json_str + "\n", encoding="utf-8")
        print(f"[rebuild_nodos_index] Escrito: {OUTPUT}")
        print(f"  Nodos: {len(nodos)} | py_files: {len(py_files)} | huerfanos: {len(huerfanos)}")
        if huerfanos:
            print("  Huérfanos:")
            for h in huerfanos:
                print(f"    - {h}")

    return index


def main():
    parser = argparse.ArgumentParser(description="Nodo-75: rebuild nodos_index.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Imprime sin escribir nodos_index.json")
    args = parser.parse_args()
    build_index(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
session_compiler.py — Compilador de sesión para audit-trail (Fase 5 FABLE_02)
Reemplaza claude-memory-compiler (no existe como paquete público).

Genera un artículo .md estructurado en .spec/01_Nodos/audit-trail/ con:
  - Commits de la sesión (desde --desde o últimas N horas)
  - Entradas nuevas en DECISION-LOG.md (desde --desde)
  - Estado de tests al cierre
  - Resumen de archivos modificados

Sin dependencias externas. Sin LLM. Solo git + regex.

Uso:
  python3 session_compiler.py                    # últimas 8 horas
  python3 session_compiler.py --horas 24         # últimas 24h
  python3 session_compiler.py --desde 2026-07-08 # desde fecha
  python3 session_compiler.py --tema "GCS forense + C63-B governor"
  python3 session_compiler.py --dry-run          # preview sin guardar
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR   = Path(__file__).parent
AUDIT_DIR  = BASE_DIR / ".spec" / "01_Nodos" / "audit-trail"
DLOG_PATH  = BASE_DIR / "docs" / "DECISION-LOG.md"
CLAUDE_MD  = BASE_DIR / "CLAUDE.md"


# ─── Git helpers ────────────────────────────────────────────────────────────

def _git(args: list[str]) -> str:
    r = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=BASE_DIR)
    return r.stdout.strip()


def _commits_since(since: str) -> list[dict]:
    """Commits desde fecha ISO (YYYY-MM-DD HH:MM) o relativo ('8 hours ago')."""
    log = _git(["log", f"--since={since}", "--oneline", "--no-merges"])
    commits = []
    for line in log.splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            commits.append({"hash": parts[0], "msg": parts[1]})
    return commits


def _files_changed(since: str) -> list[str]:
    """Archivos modificados desde fecha."""
    out = _git(["diff", "--name-only", f"HEAD@{{{since}}}..HEAD"])
    return [f for f in out.splitlines() if f.strip()]


def _current_branch() -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"])


def _last_commit_hash() -> str:
    return _git(["rev-parse", "--short", "HEAD"])


# ─── DECISION-LOG parser ────────────────────────────────────────────────────

def _decision_log_entries_since(since_date: str) -> list[dict]:
    """
    Extrae entradas D-XX, E-XX, C-XX del DECISION-LOG.md
    añadidas en o después de since_date (YYYY-MM-DD).
    Heurística: busca '(YYYY-MM-DD)' en el encabezado de la entrada.
    """
    if not DLOG_PATH.exists():
        return []

    text = DLOG_PATH.read_text(encoding="utf-8", errors="replace")
    entries = []
    # Extraer bloques ### D/E/C-XX
    blocks = re.split(r'\n(?=###\s+[DEC]-\d+)', text)
    for block in blocks:
        header = block.splitlines()[0] if block.splitlines() else ""
        m_id = re.search(r'###\s+([DEC]-\d+)', header)
        if not m_id:
            continue
        entry_id = m_id.group(1)
        # Buscar fecha en el header: (YYYY-MM-DD)
        m_date = re.search(r'\((\d{4}-\d{2}-\d{2})\)', header)
        if m_date:
            entry_date = m_date.group(1)
            if entry_date >= since_date:
                # Extraer primera línea de descripción (negrita o texto)
                desc_lines = [l.strip() for l in block.splitlines()[1:] if l.strip()]
                desc = desc_lines[0].lstrip('*').strip() if desc_lines else ""
                entries.append({"id": entry_id, "fecha": entry_date,
                                "header": header.strip(), "desc": desc})
    return entries


# ─── Test status ────────────────────────────────────────────────────────────

def _test_status() -> str:
    """Lee el conteo de tests del último commit de pytest o de CLAUDE.md."""
    # 1. Buscar en git log mensajes con "passed"
    log = _git(["log", "--oneline", "-20"])
    for line in log.splitlines():
        m = re.search(r'(\d{3,4})\s+passed', line)
        if m:
            return f"{m.group(1)} passed (desde historial de commits)"
    # 2. Fallback: buscar en CLAUDE.md
    if CLAUDE_MD.exists():
        text = CLAUDE_MD.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'(\d{3,4})\s+passed', text)
        if m:
            return f"{m.group(1)} passed (desde CLAUDE.md)"
    return "ver: python -m pytest tests/ --no-cov -q"


# ─── Inferir tema ────────────────────────────────────────────────────────────

def _infer_tema(commits: list[dict], entries: list[dict]) -> str:
    """Infiere el tema de la sesión desde los commits."""
    if not commits:
        return "sin-commits"
    # Tomar primeras palabras del commit más reciente
    msg = commits[0]["msg"]
    # Limpiar prefijo convencional (feat/fix/docs/...)
    msg = re.sub(r'^(feat|fix|docs|refactor|test|chore)\([^)]+\):\s*', '', msg)
    # Truncar y slug
    words = msg[:60].strip()
    slug = re.sub(r'[^a-zA-Z0-9\s_-]', '', words).strip().replace(' ', '-').lower()
    return slug[:50] or "sesion"


# ─── Generar artículo ────────────────────────────────────────────────────────

def _generate_article(
    commits: list[dict],
    files: list[str],
    dlog_entries: list[dict],
    tema: str,
    fecha: str,
    test_status: str,
    branch: str,
    last_hash: str,
) -> str:
    lines = [
        f"---",
        f"fecha: {fecha}",
        f"branch: {branch}",
        f"commit_cierre: {last_hash}",
        f"tema: {tema}",
        f"tipo: session_audit",
        f"---",
        "",
        f"# Sesión {fecha} — {tema}",
        "",
    ]

    # Commits
    lines.append("## Commits de la sesión")
    if commits:
        for c in commits:
            lines.append(f"- `{c['hash']}` {c['msg']}")
    else:
        lines.append("- _(sin commits en este período)_")
    lines.append("")

    # Archivos modificados
    if files:
        lines.append("## Archivos modificados")
        for f in sorted(set(files))[:30]:
            lines.append(f"- `{f}`")
        if len(files) > 30:
            lines.append(f"- _...y {len(files)-30} más_")
        lines.append("")

    # Entradas DECISION-LOG nuevas
    lines.append("## Decisiones / Incidentes (DECISION-LOG)")
    if dlog_entries:
        for e in dlog_entries:
            lines.append(f"- **{e['id']}** ({e['fecha']}): {e['desc']}")
    else:
        lines.append("- _(sin entradas nuevas en este período)_")
    lines.append("")

    # Estado de tests
    lines.append("## Estado de tests al cierre")
    lines.append(f"```")
    lines.append(test_status)
    lines.append(f"```")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"_Generado por session_compiler.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}_")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compilador de sesión → .spec/01_Nodos/audit-trail/"
    )
    parser.add_argument("--horas", type=int, default=8,
                        help="Ventana de tiempo en horas (default: 8)")
    parser.add_argument("--desde", type=str, default=None,
                        help="Fecha ISO desde la que buscar (YYYY-MM-DD), override --horas")
    parser.add_argument("--tema", type=str, default=None,
                        help="Tema del artículo (default: inferido de commits)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el artículo sin guardar")
    args = parser.parse_args()

    fecha = datetime.now().strftime("%Y-%m-%d")

    # Calcular ventana
    if args.desde:
        since_git = args.desde
        since_dlog = args.desde
    else:
        dt_since = datetime.now() - timedelta(hours=args.horas)
        since_git = dt_since.strftime("%Y-%m-%d %H:%M")
        since_dlog = dt_since.strftime("%Y-%m-%d")

    print(f"[session_compiler] Compilando sesión desde {since_git}...")

    commits   = _commits_since(since_git)
    files     = _files_changed(since_git)
    dlog_entries = _decision_log_entries_since(since_dlog)
    branch    = _current_branch()
    last_hash = _last_commit_hash()

    tema = args.tema or _infer_tema(commits, dlog_entries)

    print(f"[session_compiler] {len(commits)} commits | {len(files)} archivos | "
          f"{len(dlog_entries)} entradas DECISION-LOG | tema: {tema}")

    # Test status (colección rápida, no ejecución)
    print("[session_compiler] Contando tests...")
    test_status = _test_status()

    article = _generate_article(
        commits, files, dlog_entries, tema, fecha,
        test_status, branch, last_hash
    )

    if args.dry_run:
        print("\n" + "="*60)
        print(article)
        print("="*60)
        print("\n[dry-run] No guardado.")
        return

    # Guardar
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    # Nombre único: fecha + slug del tema
    slug = re.sub(r'[^a-z0-9-]', '-', tema.lower())[:40]
    slug = re.sub(r'-+', '-', slug).strip('-')
    ts   = datetime.now().strftime("%H%M")
    out  = AUDIT_DIR / f"{fecha}_{slug}_{ts}.md"

    out.write_text(article, encoding="utf-8")
    print(f"[session_compiler] Artículo guardado: {out.name}")
    print(f"  Vault: .spec/01_Nodos/audit-trail/{out.name}")


if __name__ == "__main__":
    main()

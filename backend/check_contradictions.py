"""
check_contradictions.py — Chequeo semanal de contradicciones (Haiku)
FABLE_02 §1.2 Vacío 3, Fase 5. Ver Nodo-78 para el protocolo de auditoría
y disciplina de doble verificación que guía este tipo de script.

Lee los headers de los últimos 10 nodos en .spec/01_Nodos/ y los compara
contra el estado declarado en CLAUDE.md. Reporta contradicciones.

Funcionamiento SIN API key (modo texto puro):
  - Extrae estado de cada nodo del header (✅ COMPLETO / ⚠️ / 🔴)
  - Verifica que CLAUDE.md no contradiga ese estado
  - Emite PASS / CONTRADICCION / WARN por nodo

Con ANTHROPIC_API_KEY seteado → usa Haiku para análisis semántico profundo.
Sin API key → análisis por regex (cero costo, cobertura básica pero suficiente).

Uso:
  python3 check_contradictions.py              # análisis completo
  python3 check_contradictions.py --quick      # solo regex, sin LLM
  python3 check_contradictions.py --nodos N    # últimos N nodos (default 10)

Cron semanal (añadir a crontab):
  0 9 * * 1 cd /mnt/c/.../backend && python3 check_contradictions.py >> logs/contradicciones.log 2>&1
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR  = Path(__file__).parent
SPEC_DIR  = BASE_DIR / ".spec" / "01_Nodos"
CLAUDE_MD = BASE_DIR / "CLAUDE.md"
LOG_DIR   = BASE_DIR / "logs"

# Patrones que indican estado en nodo header
_PAT_COMPLETO  = re.compile(r'✅\s*COMPLETO|estado.*completo|COMPLETO', re.I)
_PAT_PENDIENTE = re.compile(r'PENDIENTE|⏳|pendiente', re.I)
_PAT_BLOQUEADO = re.compile(r'🔴\s*BLOQUEADO', re.I)  # requiere emoji — evita falsos positivos

# Patrones de nodo en CLAUDE.md
_PAT_NODO_REF  = re.compile(r'Nodo-(\d+).*?(✅|⚠️|🔴|COMPLETO|PENDIENTE|BLOQUEADO)', re.I)


def _get_nodo_files(n: int) -> list[Path]:
    """Devuelve los últimos N archivos de nodo ordenados por número."""
    files = list(SPEC_DIR.glob("Nodo-*.md"))
    def _nnum(p: Path) -> int:
        m = re.search(r'Nodo-(\d+)', p.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=_nnum)[-n:]


def _extract_nodo_state(path: Path) -> dict:
    """Extrae estado de un nodo desde su header."""
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return {'nodo': path.stem, 'estado': 'UNREADABLE', 'estado_raw': ''}

    nnum = re.search(r'Nodo-(\d+)', path.name)
    nodo_id = f"Nodo-{nnum.group(1)}" if nnum else path.stem

    # Busca estado en las primeras 20 líneas (header)
    header = '\n'.join(content.splitlines()[:20])
    if _PAT_COMPLETO.search(header):
        estado = 'COMPLETO'
    elif _PAT_BLOQUEADO.search(header):
        estado = 'BLOQUEADO'
    elif _PAT_PENDIENTE.search(header):
        estado = 'PENDIENTE'
    else:
        estado = 'DESCONOCIDO'

    return {'nodo': nodo_id, 'estado': estado, 'path': str(path)}


def _check_claude_md(nodo_id: str, estado_nodo: str, claude_text: str) -> tuple[str, str]:
    """
    Verifica que CLAUDE.md sea consistente con el estado del nodo.
    Retorna (nivel, mensaje): PASS | WARN | CONTRADICCION.
    """
    # Busca referencia al nodo en CLAUDE.md
    refs = [m for m in _PAT_NODO_REF.finditer(claude_text)
            if m.group(0).startswith(nodo_id.split('-')[0]) or nodo_id in m.group(0)]

    # Búsqueda más simple: ¿aparece el nodo_id en CLAUDE.md?
    if nodo_id not in claude_text:
        if estado_nodo == 'COMPLETO':
            return ('WARN', f"{nodo_id}: COMPLETO en spec pero sin referencia en CLAUDE.md")
        return ('PASS', f"{nodo_id}: no referenciado en CLAUDE.md ({estado_nodo})")

    # Busca contexto del nodo en CLAUDE.md
    idx = claude_text.find(nodo_id)
    context = claude_text[max(0, idx-20):idx+80]

    claude_dice_completo = '✅' in context or 'COMPLETO' in context.upper()
    claude_dice_pendiente = 'pendiente' in context.lower() or '⏳' in context

    if estado_nodo == 'COMPLETO' and claude_dice_pendiente:
        return ('CONTRADICCION',
                f"{nodo_id}: nodo dice COMPLETO pero CLAUDE.md dice pendiente — actualizar CLAUDE.md")
    if estado_nodo == 'BLOQUEADO' and claude_dice_completo:
        return ('CONTRADICCION',
                f"{nodo_id}: nodo dice BLOQUEADO pero CLAUDE.md dice COMPLETO — revisar")

    return ('PASS', f"{nodo_id}: consistente ({estado_nodo})")


def _haiku_analysis(nodos_data: list[dict], claude_text: str) -> list[str]:
    """Análisis semántico profundo con Haiku (solo si ANTHROPIC_API_KEY disponible)."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return []

    try:
        import anthropic
    except ImportError:
        return []

    nodos_str = '\n'.join(f"- {d['nodo']}: {d['estado']}" for d in nodos_data)
    # Extracto relevante de CLAUDE.md (primera vez que aparece cada nodo)
    claude_snippet = '\n'.join(
        line for line in claude_text.splitlines()
        if any(d['nodo'] in line for d in nodos_data)
    )[:2000]

    prompt = f"""Eres un auditor de consistencia. Tienes:

ESTADO REAL de nodos (fuente de verdad):
{nodos_str}

CLAUDE.md (vista derivada — puede estar desactualizada):
{claude_snippet}

Identifica SOLO contradicciones reales: donde CLAUDE.md dice algo diferente al estado del nodo.
Responde con lista breve. Si no hay contradicciones, di "Sin contradicciones detectadas."
NO inventes contradicciones. Solo las evidentes."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return [f"[HAIKU] {line}" for line in msg.content[0].text.strip().splitlines() if line]
    except Exception as e:
        return [f"[HAIKU] Error: {e}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chequeo semanal de contradicciones CLAUDE.md vs nodos")
    parser.add_argument('--quick', action='store_true', help="Solo regex, sin LLM")
    parser.add_argument('--nodos', type=int, default=10, help="Últimos N nodos a revisar (default 10)")
    args = parser.parse_args()

    if not SPEC_DIR.exists():
        print(f"[contradictions] ERROR: {SPEC_DIR} no existe")
        sys.exit(1)
    if not CLAUDE_MD.exists():
        print(f"[contradictions] ERROR: {CLAUDE_MD} no existe")
        sys.exit(1)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"CHEQUEO DE CONTRADICCIONES  {ts}")
    print(f"Últimos {args.nodos} nodos vs CLAUDE.md")
    print(f"{'='*60}")

    nodo_files = _get_nodo_files(args.nodos)
    if not nodo_files:
        print("[contradictions] Sin archivos de nodo en .spec/01_Nodos/")
        sys.exit(0)

    claude_text = CLAUDE_MD.read_text(encoding='utf-8', errors='replace')
    nodos_data  = [_extract_nodo_state(f) for f in nodo_files]

    passes = 0
    warns  = 0
    contras = 0
    results = []

    for d in nodos_data:
        nivel, msg = _check_claude_md(d['nodo'], d['estado'], claude_text)
        results.append((nivel, msg))
        if nivel == 'PASS':
            passes += 1
        elif nivel == 'WARN':
            warns += 1
        else:
            contras += 1

    for nivel, msg in results:
        prefix = "  [PASS]          " if nivel == 'PASS' else \
                 "  [WARN]          " if nivel == 'WARN' else \
                 "  [CONTRADICCION] "
        print(f"{prefix}{msg}")

    # Análisis Haiku (si API key disponible y no --quick)
    haiku_lines = []
    if not args.quick:
        haiku_lines = _haiku_analysis(nodos_data, claude_text)
        for line in haiku_lines:
            print(f"  {line}")

    print(f"\nResumen: {passes} PASS | {warns} WARN | {contras} CONTRADICCION")
    if contras > 0:
        print("ACCION: Actualizar CLAUDE.md para alinearlo con el estado real de los nodos.")
        print("        Regla: CLAUDE.md es VISTA derivada — el nodo es siempre la fuente de verdad.")
    elif warns > 0:
        print("ACCION: Revisar WARNs. Puede ser que CLAUDE.md omita nodos recientes.")
    else:
        print("ACCION: Sin contradicciones detectadas. CLAUDE.md consistente con los nodos.")
    print(f"{'='*60}\n")

    # ── BLOQUE B: pendientes de FABLE_02 §4.5 ──────────────────────────────
    fable_path = SPEC_DIR / "FABLE_02_TENIS_DOCTORADO_SPEC.md"
    fable_pendientes = 0
    if fable_path.exists():
        fable_text = fable_path.read_text(encoding='utf-8', errors='replace')
        # Extraer sección §4.5 para reducir falsos positivos
        seccion = fable_text
        m_inicio = re.search(r'§4\.5', fable_text)
        m_fin    = re.search(r'^# §5', fable_text, re.MULTILINE)
        if m_inicio and m_fin:
            seccion = fable_text[m_inicio.start():m_fin.start()]
        print(f"\n{'─'*60}")
        print("BLOQUE B — FABLE_02 §4.5 pendientes no resueltos")
        print(f"{'─'*60}")
        for line in seccion.splitlines():
            stripped = line.strip()
            if ('🔴' in stripped or '🟠' in stripped) and stripped:
                print(f"  [FABLE_PENDIENTE] {stripped}")
                fable_pendientes += 1
        if fable_pendientes == 0:
            print("  [PASS] Sin pendientes activos en §4.5")
        print(f"  Total pendientes FABLE_02: {fable_pendientes}")
    else:
        print(f"\n[WARN] FABLE_02 spec no encontrado en {fable_path}")

    # Log a archivo
    LOG_DIR.mkdir(exist_ok=True)
    log_entry = f"[{ts}] {passes}P {warns}W {contras}C — {len(nodo_files)} nodos | FABLE_pendientes={fable_pendientes}\n"
    (LOG_DIR / "contradicciones.log").open("a").write(log_entry)

    sys.exit(1 if contras > 0 else 0)


if __name__ == "__main__":
    main()

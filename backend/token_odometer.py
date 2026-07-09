#!/usr/bin/env python3
"""
token_odometer.py — D59-01: Odómetro de tokens/costos de sesiones Claude Code.

Parsea sesiones JSONL en ~/.claude/projects/-mnt-c-users-hogar-tennis-analysis-backend/
y reporta uso de tokens, costos por modelo, por tag y top sesiones.

Uso:
    python3 token_odometer.py --report
    python3 token_odometer.py --report --desde 2026-06-26
    python3 token_odometer.py --report --proyecto /ruta/al/directorio
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# ── Tabla de costos — ÚNICA fuente de verdad (T59-03) ─────────────────────────
MODEL_COSTS = {
    # prefijo_modelo: {input, output, cache_read, cache_creation} en USD por millón de tokens
    'claude-haiku': {
        'input': 0.80,
        'output': 4.00,
        'cache_read': 0.08,
        'cache_creation': 1.00,
    },
    'claude-sonnet': {
        'input': 3.00,
        'output': 15.00,
        'cache_read': 0.30,
        'cache_creation': 3.75,
    },
    'claude-opus': {
        'input': 15.00,
        'output': 75.00,
        'cache_read': 1.50,
        'cache_creation': 18.75,
    },
}

MODEL_RATIOS = {'haiku': 1, 'sonnet': 4, 'opus': 20}  # costo relativo

DEFAULT_PROJECT_DIR = os.path.expanduser(
    '~/.claude/projects/-mnt-c-users-hogar-tennis-analysis-backend'
)

VALID_TAGS = {'impl', 'test', 'audit', 'settle', 'analisis', 'nodo'}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_model_costs(model: str) -> dict:
    """Devuelve el dict de costos para un modelo dado por prefijo."""
    if not model:
        return MODEL_COSTS['claude-sonnet']  # fallback razonable
    for prefix, costs in MODEL_COSTS.items():
        if model.startswith(prefix):
            return costs
    return MODEL_COSTS['claude-sonnet']


def _compute_cost(usage: dict, model: str) -> float:
    """Calcula costo USD a partir del dict usage y nombre de modelo."""
    costs = _get_model_costs(model)
    m = 1_000_000
    return (
        usage.get('input_tokens', 0) * costs['input'] / m
        + usage.get('output_tokens', 0) * costs['output'] / m
        + usage.get('cache_read_input_tokens', 0) * costs['cache_read'] / m
        + usage.get('cache_creation_input_tokens', 0) * costs['cache_creation'] / m
    )


def _extract_first_user_text(content) -> str:
    """Extrae el primer texto del usuario de un campo content (str o lista)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
    return ''


def _extract_tag(text: str) -> str:
    """
    Detecta tag en la primera línea que empiece con '# TAG:' o
    que contenga uno de los patrones válidos (impl/test/audit/etc).
    Devuelve 'untagged' si no se encuentra.
    """
    if not text:
        return 'untagged'
    first_line = text.split('\n')[0].strip()
    # Formato explícito: "# TAG: impl nodo-58"
    if first_line.lower().startswith('# tag:'):
        rest = first_line[6:].strip().lower()
        for tag in VALID_TAGS:
            if rest.startswith(tag):
                return tag
        # tag desconocido pero explícito → lo devolvemos limpio
        return rest.split()[0] if rest else 'untagged'
    # Formato implícito: primera línea contiene la palabra clave
    lower = first_line.lower()
    for tag in VALID_TAGS:
        if tag in lower:
            return tag
    return 'untagged'


def _parse_timestamp(ts: str):
    """Parsea timestamp ISO8601 → datetime UTC. Devuelve None si falla."""
    if not ts:
        return None
    try:
        # Python 3.7+ no soporta 'Z' directamente en fromisoformat en 3.10-
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


# ── Parser principal ───────────────────────────────────────────────────────────

def parse_sessions(project_dir: str, desde: datetime = None) -> dict:
    """
    Lee todos los JSONL en project_dir y devuelve un dict con:
      sessions: {session_id: {model, tag, cost, input, output, cache_r, cache_c, ts_first, ts_last}}
      models:   {model_name: {input, output, cache_r, cache_c, cost}}
      tags:     {tag: {cost, input, output, cache_r, cache_c}}
      totals:   {cost, input, output, cache_r, cache_c}
      date_min, date_max: datetime
      n_sessions: int
      n_days: int (días distintos con actividad)
    """
    jsonl_files = sorted(glob.glob(os.path.join(project_dir, '*.jsonl')))
    if not jsonl_files:
        print(f"WARN: No se encontraron archivos JSONL en {project_dir}", file=sys.stderr)

    # Estado por sesión durante el parse
    session_data = {}   # session_id → acumuladores
    session_tags = {}   # session_id → tag (primer turno de usuario)
    session_models = {} # session_id → modelo dominante

    all_timestamps = []

    for fpath in jsonl_files:
        try:
            with open(fpath, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"WARN: No se pudo leer {os.path.basename(fpath)}: {e}", file=sys.stderr)
            continue

        for lineno, raw in enumerate(lines, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # línea corrupta → skip silencioso
                continue

            ts = _parse_timestamp(obj.get('timestamp'))
            if ts and desde and ts < desde:
                continue

            session_id = obj.get('sessionId', 'unknown')
            entry_type = obj.get('type')

            # Inicializar sesión si es nueva
            if session_id not in session_data:
                session_data[session_id] = {
                    'input': 0, 'output': 0, 'cache_r': 0, 'cache_c': 0, 'cost': 0.0,
                    'ts_first': None, 'ts_last': None,
                }
                session_tags[session_id] = 'untagged'
                session_models[session_id] = ''

            sd = session_data[session_id]

            # Actualizar timestamps de sesión
            if ts:
                all_timestamps.append(ts)
                if sd['ts_first'] is None or ts < sd['ts_first']:
                    sd['ts_first'] = ts
                if sd['ts_last'] is None or ts > sd['ts_last']:
                    sd['ts_last'] = ts

            # Procesar mensajes de usuario → detectar tag
            if entry_type == 'user':
                if session_tags[session_id] == 'untagged':
                    content = obj.get('message', {}).get('content', '')
                    text = _extract_first_user_text(content)
                    tag = _extract_tag(text)
                    if tag != 'untagged':
                        session_tags[session_id] = tag

            # Procesar mensajes de asistente → acumular tokens
            elif entry_type == 'assistant':
                msg = obj.get('message', {})
                model = msg.get('model', '')
                usage = msg.get('usage', {})

                if model and not session_models[session_id]:
                    session_models[session_id] = model

                inp = usage.get('input_tokens', 0)
                out = usage.get('output_tokens', 0)
                cache_r = usage.get('cache_read_input_tokens', 0)
                cache_c = usage.get('cache_creation_input_tokens', 0)

                cost = _compute_cost(usage, model)

                sd['input'] += inp
                sd['output'] += out
                sd['cache_r'] += cache_r
                sd['cache_c'] += cache_c
                sd['cost'] += cost

    # ── Agregar por modelo y tag ────────────────────────────────────────────────
    models_agg = defaultdict(lambda: {'input': 0, 'output': 0, 'cache_r': 0, 'cache_c': 0, 'cost': 0.0})
    tags_agg = defaultdict(lambda: {'input': 0, 'output': 0, 'cache_r': 0, 'cache_c': 0, 'cost': 0.0})
    totals = {'input': 0, 'output': 0, 'cache_r': 0, 'cache_c': 0, 'cost': 0.0}

    active_sessions = []
    for sid, sd in session_data.items():
        # Filtrar sesiones sin ningún token (solo metadatos)
        if sd['input'] == 0 and sd['output'] == 0:
            continue

        model = session_models.get(sid, '')
        tag = session_tags.get(sid, 'untagged')

        ma = models_agg[model]
        ma['input'] += sd['input']
        ma['output'] += sd['output']
        ma['cache_r'] += sd['cache_r']
        ma['cache_c'] += sd['cache_c']
        ma['cost'] += sd['cost']

        ta = tags_agg[tag]
        ta['input'] += sd['input']
        ta['output'] += sd['output']
        ta['cache_r'] += sd['cache_r']
        ta['cache_c'] += sd['cache_c']
        ta['cost'] += sd['cost']

        totals['input'] += sd['input']
        totals['output'] += sd['output']
        totals['cache_r'] += sd['cache_r']
        totals['cache_c'] += sd['cache_c']
        totals['cost'] += sd['cost']

        active_sessions.append({
            'session_id': sid,
            'model': model,
            'tag': tag,
            'cost': sd['cost'],
            'input': sd['input'],
            'output': sd['output'],
            'cache_r': sd['cache_r'],
            'cache_c': sd['cache_c'],
            'ts_first': sd['ts_first'],
        })

    date_min = min(all_timestamps) if all_timestamps else None
    date_max = max(all_timestamps) if all_timestamps else None
    n_days = len({t.date() for t in all_timestamps}) if all_timestamps else 0

    return {
        'sessions': active_sessions,
        'models': dict(models_agg),
        'tags': dict(tags_agg),
        'totals': totals,
        'date_min': date_min,
        'date_max': date_max,
        'n_sessions': len(active_sessions),
        'n_days': n_days,
    }


# ── Renderizado ────────────────────────────────────────────────────────────────

def render_report(data: dict) -> str:
    lines = []
    d_min = data['date_min'].strftime('%Y-%m-%d') if data['date_min'] else '?'
    d_max = data['date_max'].strftime('%Y-%m-%d') if data['date_max'] else '?'

    lines.append(f"\n{'='*4} ODOMETRO -- {d_min} -> {d_max} {'='*4}")
    lines.append(f"SESIONES: {data['n_sessions']} total | {data['n_days']} dias")

    # ── Modelos ────────────────────────────────────────────────────────────────
    lines.append("MODELOS:")
    for model, ma in sorted(data['models'].items(), key=lambda x: -x[1]['cost']):
        label = model if model else '(desconocido)'
        lines.append(
            f"  {label:<30} {ma['input']:>12,} in | {ma['output']:>10,} out | "
            f"{ma['cache_r']:>10,} cache_r | ${ma['cost']:.2f}"
        )
    lines.append(f"TOTAL: ${data['totals']['cost']:.2f}")

    # ── Por tag ────────────────────────────────────────────────────────────────
    lines.append("\nPOR TAG:")
    total_cost = data['totals']['cost'] or 1e-9
    for tag, ta in sorted(data['tags'].items(), key=lambda x: -x[1]['cost']):
        pct = ta['cost'] / total_cost * 100
        lines.append(f"  [{tag:<10}] {pct:5.1f}% | ${ta['cost']:.2f}")

    # ── Cache hit rate ─────────────────────────────────────────────────────────
    total_in = data['totals']['input']
    total_cache_r = data['totals']['cache_r']
    total_cache_c = data['totals']['cache_c']
    eligible = total_in + total_cache_r + total_cache_c
    cache_hit_pct = (total_cache_r / eligible * 100) if eligible > 0 else 0.0
    lines.append(f"\nCACHE HIT RATE: {cache_hit_pct:.1f}% (sesiones largas sin /clear lo bajan a <20%)")

    # ── Top 5 sesiones ─────────────────────────────────────────────────────────
    lines.append("TOP 5 SESIONES MAS COSTOSAS:")
    top5 = sorted(data['sessions'], key=lambda x: -x['cost'])[:5]
    for i, s in enumerate(top5, 1):
        ts_str = s['ts_first'].strftime('%Y-%m-%d') if s['ts_first'] else '?'
        short_id = s['session_id'][:8] + '...'
        model_short = s['model'].replace('claude-', '') if s['model'] else '?'
        lines.append(
            f"  {i}. {short_id} | {ts_str} | {model_short} | [{s['tag']}] | ${s['cost']:.2f}"
        )

    # ── Alerta untagged ────────────────────────────────────────────────────────
    untagged_cost = data['tags'].get('untagged', {}).get('cost', 0.0)
    untagged_pct = untagged_cost / total_cost * 100
    lines.append(f"\n%untagged: {untagged_pct:.1f}% (objetivo: <20%)")

    return '\n'.join(lines)


# ── Dream M2 — Detección de secuencias repetitivas (Nodo-59 §3) ───────────────

def _extract_user_commands(session_lines: list) -> list:
    """
    De una lista de líneas JSONL de una sesión, extrae las secuencias de
    texto del usuario (normalizadas a minúsculas, sin espacios extra).
    Solo texto real — ignora tags, mensajes de sistema y adjuntos.
    """
    commands = []
    for raw in session_lines:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if obj.get('type') != 'user':
            continue
        content = obj.get('message', {}).get('content', '')
        text = _extract_first_user_text(content).strip()
        if text and not text.startswith('#'):
            # normalizar: primera línea, máx 80 chars
            cmd = text.split('\n')[0].strip()[:80].lower()
            if cmd:
                commands.append(cmd)
    return commands


def detect_dream_sequences(
    project_dir: str,
    min_sessions: int = 3,
    min_seq_len: int = 3,
) -> list:
    """
    D59-07 Dream M2 — Detecta secuencias de ≥min_seq_len comandos de usuario
    que aparecen en ≥min_sessions sesiones distintas.

    Regla n≥3: un skill se empaqueta con recurrencia demostrada, no con una anécdota.
    Devuelve lista de dicts: {sequence, sessions, count}
    """
    jsonl_files = sorted(glob.glob(os.path.join(project_dir, '*.jsonl')))

    # Acumular ngrams por sesión
    session_ngrams: dict = {}  # session_id → set of ngram tuples

    for fpath in jsonl_files:
        session_id = os.path.basename(fpath).replace('.jsonl', '')
        try:
            with open(fpath, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            continue

        commands = _extract_user_commands(lines)
        if len(commands) < min_seq_len:
            continue

        # Generar n-gramas de longitud min_seq_len
        ngrams = set()
        for i in range(len(commands) - min_seq_len + 1):
            ngram = tuple(commands[i:i + min_seq_len])
            ngrams.add(ngram)

        if ngrams:
            session_ngrams[session_id] = ngrams

    # Contar en cuántas sesiones aparece cada n-grama
    ngram_sessions: dict = defaultdict(set)
    for sid, ngrams in session_ngrams.items():
        for ng in ngrams:
            ngram_sessions[ng].add(sid)

    # Filtrar por min_sessions
    candidates = []
    for ngram, sids in ngram_sessions.items():
        if len(sids) >= min_sessions:
            candidates.append({
                'sequence': list(ngram),
                'sessions': sorted(sids),
                'count': len(sids),
            })

    # Ordenar por frecuencia descendente
    candidates.sort(key=lambda x: -x['count'])
    return candidates


def write_dream_candidates(candidates: list, output_path: str) -> None:
    """
    Escribe los candidatos Dream en docs/dream-candidates.md.
    El humano aprueba; Sonnet empaqueta — NUNCA auto-crear skills.
    """
    lines = [
        "# dream-candidates.md — Secuencias Repetitivas (Dream M2)",
        "",
        "> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]",
        "> **Generado por:** `token_odometer.py --dream`",
        "> **INSTRUCCIÓN:** El humano revisa y aprueba. Sonnet empaqueta. NUNCA auto-crear skills.",
        "> **Regla n≥3:** solo se propone si aparece en ≥3 sesiones distintas.",
        "",
        f"## Candidatos detectados ({len(candidates)})",
        "",
    ]

    if not candidates:
        lines.append("*(Sin candidatos — ninguna secuencia aparece en ≥3 sesiones)*")
    else:
        for i, c in enumerate(candidates, 1):
            lines.append(f"### Candidato {i} — {c['count']} sesiones")
            lines.append(f"**Secuencia:**")
            for step in c['sequence']:
                lines.append(f"- `{step}`")
            lines.append(f"**Sesiones:** {', '.join(s[:8] for s in c['sessions'])}")
            lines.append(f"**Acción sugerida:** revisar si merece un skill en `.claude/commands/`")
            lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Dream candidates → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Odómetro de tokens — sesiones Claude Code')
    parser.add_argument('--report', action='store_true', help='Generar reporte de uso')
    parser.add_argument('--dream', action='store_true',
                        help='Dream M2: detectar secuencias repetitivas y proponer skills')
    parser.add_argument('--dream-min-sessions', type=int, default=3, metavar='N',
                        help='Mínimo de sesiones para proponer un skill (default: 3)')
    parser.add_argument('--dream-min-len', type=int, default=3, metavar='N',
                        help='Longitud mínima de secuencia (default: 3 comandos)')
    parser.add_argument('--desde', metavar='FECHA', default=None,
                        help='Filtrar desde fecha (YYYY-MM-DD)')
    parser.add_argument('--proyecto', metavar='PATH', default=DEFAULT_PROJECT_DIR,
                        help='Directorio con archivos JSONL (default: proyecto backend)')
    args = parser.parse_args()

    if not args.report and not args.dream:
        parser.print_help()
        sys.exit(0)

    desde_dt = None
    if args.desde:
        try:
            desde_dt = datetime.fromisoformat(args.desde).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: Fecha inválida '{args.desde}'. Usar formato YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(args.proyecto):
        print(f"ERROR: Directorio no encontrado: {args.proyecto}", file=sys.stderr)
        sys.exit(1)

    if args.report:
        data = parse_sessions(args.proyecto, desde=desde_dt)
        print(render_report(data))

    if args.dream:
        candidates = detect_dream_sequences(
            args.proyecto,
            min_sessions=args.dream_min_sessions,
            min_seq_len=args.dream_min_len,
        )
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'docs', 'dream-candidates.md'
        )
        write_dream_candidates(candidates, output_path)
        print(f"Candidatos encontrados: {len(candidates)}")
        for c in candidates[:5]:
            print(f"  [{c['count']} sesiones] {' → '.join(c['sequence'][:2])}...")


if __name__ == '__main__':
    main()

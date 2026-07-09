"""
run_daily.py — D54-03 (Nodo-55): Orquestador diario del pipeline completo.

Ejecutar UNA vez al día:
    python3 run_daily.py                        # partidos de hoy
    python3 run_daily.py --tomorrow              # partidos de mañana
    python3 run_daily.py --bankroll 125000       # bankroll GS (default)
    python3 run_daily.py --settle-only           # solo settle de ayer (post-partidos)

Output clave: reports/daily_brief_FECHA.txt
Tiempo humano objetivo: <7 min/día (vs 45 min manual).

Secuencia:
    PASO 0  — rankings si >7d sin actualizar
    PASO 1  — extraer partidos (API)
    PASO 2  — extraer H2H (Ninja API)
    PASO 3  — edge_calculator + shadow-log
    PASO 3.5 — tabla análisis
    PASO 3.6 — games signal
    PASO 4  — trader por tier (GS grass + challenger + itf)
    PASO 4.3 — combo confianza
    SETTLE  — resultados de ayer → shadow book
    BRIEF   — genera daily_brief (solo lo que importa)
"""

import subprocess
import sys
import os
import json
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────

BANKROLL_GS         = 125000
BANKROLL_CHALLENGER = 20000
BANKROLL_ITF        = 20000
REPORTS_DIR         = 'reports'
DATA_DIR            = 'data'


def _run(cmd: list, step: str, capture: bool = False) -> tuple[int, str]:
    """Ejecuta un comando y retorna (returncode, output)."""
    print(f"\n{'='*70}")
    print(f"  {step}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(
        cmd, capture_output=capture, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
    )
    if capture:
        return result.returncode, result.stdout + result.stderr
    return result.returncode, ''


def _rankings_stale(max_days: int = 7) -> bool:
    """True si el ranking más reciente tiene más de max_days días."""
    atp_files = sorted(glob.glob(f'{DATA_DIR}/atp_rankings_complete_*.json'), reverse=True)
    if not atp_files:
        return True
    fname = Path(atp_files[0]).stem  # atp_rankings_complete_YYYYMMDD_HHMMSS
    parts = fname.split('_')
    try:
        fecha_str = parts[3]  # YYYYMMDD
        fecha = datetime.strptime(fecha_str, '%Y%m%d')
        return (datetime.now() - fecha).days >= max_days
    except Exception:
        return True


def _latest_report(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def _build_daily_brief(fecha: str, tier_results: dict, was_candidates: list,
                        yesterday: str) -> str:
    """
    Genera el daily_brief — solo lo que el humano necesita leer.
    Objetivo: 5 minutos de lectura.
    """
    lines = []
    sep = '─' * 72

    lines.append(sep)
    lines.append(f"  DAILY BRIEF — {fecha}")
    lines.append(f"  Generado: {datetime.now().strftime('%H:%M:%S')}")
    lines.append(sep)

    # ── Picks APOSTAR con stake>0 ──────────────────────────────────────────
    lines.append("")
    lines.append("  APOSTAR (stake real):")
    any_apostar = False
    for tier, data in tier_results.items():
        for ind in data.get('individuales', []):
            if ind.get('stake', 0) > 0:
                wf = ind.get('_waterfall', {})
                lines.append(f"    [{tier}] {ind['favorito']:<28} @{ind['cuota']:.2f}  "
                             f"edge={ind['edge_pct']}  stake=${ind['stake']:,}")
                any_apostar = True
        for cob in data.get('cobertura', []):
            if cob.get('stake', 0) > 0:
                legs = ' + '.join(f"{l['jugador']}@{l['cuota']:.2f}"
                                  for l in cob.get('legs', []))
                lines.append(f"    [{tier}] COMBO {cob.get('piernas_n','?')}p @{cob.get('cuota_combo','?'):.2f}  "
                             f"${cob['stake']:,}  → {legs}")
                any_apostar = True

    if not any_apostar:
        lines.append("    Sin apuestas con stake>0 hoy.")

    # ── Picks aplastados por VaR (var_flattened) ──────────────────────────
    var_flat = []
    for tier, data in tier_results.items():
        for ind in data.get('individuales', []):
            wf = ind.get('_waterfall', {})
            if wf.get('var_flattened'):
                var_flat.append((tier, ind, wf))

    if var_flat:
        lines.append("")
        lines.append("  APOSTAR aplastados a $0 por VaR (H54-01 acumulando):")
        for tier, ind, wf in var_flat:
            lines.append(f"    [{tier}] {ind['favorito']:<28} @{ind['cuota']:.2f}  "
                         f"edge={ind['edge_pct']}")
            lines.append(f"           WATERFALL: {wf.get('terminal_reason','?')}")

    # ── Candidatos WAS ────────────────────────────────────────────────────
    if was_candidates:
        lines.append("")
        lines.append("  WAS CANDIDATOS (stake mínimo promo — REGLA-WAS-1):")
        for c in was_candidates:
            lines.append(f"    {c.get('jugador','?'):<28} @{c.get('cuota','?'):.2f}  "
                         f"edge={c.get('edge_pct','?')}  [{c.get('señal_was','?')}]")
    else:
        lines.append("")
        lines.append("  WAS: 0 candidatos (sin señal Markov o edge insuficiente).")

    # ── Games signal ──────────────────────────────────────────────────────
    games_file = _latest_report(f'{REPORTS_DIR}/games_signal_report_*.json')
    if games_file:
        try:
            gdata = json.loads(Path(games_file).read_text(encoding='utf-8'))
            senales = gdata.get('senales', [])
            if senales:
                lines.append("")
                lines.append("  GAMES SIGNAL:")
                for s in senales[:3]:
                    lines.append(f"    {s.get('partido','?')[:45]}  {s.get('tipo','?')} @{s.get('cuota','?')}")
        except Exception:
            pass

    # ── Settlement de ayer ────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  SETTLE AYER ({yesterday}):")
    sb_file = f'{REPORTS_DIR}/shadow_book/sb_{yesterday}.jsonl'
    if Path(sb_file).exists():
        try:
            records = {}
            with open(sb_file) as f:
                for line in f:
                    r = json.loads(line)
                    records[r['sb_id']] = r
            n_settled = sum(1 for r in records.values()
                           if r.get('record_type') != 'session_meta'
                           and r.get('resolucion'))
            n_total = sum(1 for r in records.values()
                         if r.get('record_type') != 'session_meta')
            lines.append(f"    {n_settled}/{n_total} picks settled")
        except Exception:
            lines.append("    Error leyendo shadow book de ayer")
    else:
        lines.append(f"    Sin shadow book para {yesterday} — correr --settle-only después del partido")

    # ── Alertas ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("  ALERTAS:")
    edge_file = _latest_report(f'{REPORTS_DIR}/edge_report_*.json')
    if edge_file:
        try:
            edata = json.loads(Path(edge_file).read_text(encoding='utf-8'))
            n_no_data = len(edata.get('no_data', []))
            n_apostar = len(edata.get('apostar', []))
            n_watch   = len(edata.get('watchlist', []))
            lines.append(f"    Edge report: {n_apostar} APOSTAR | {n_watch} WATCHLIST | {n_no_data} NO_DATA")
            if n_no_data > 20:
                lines.append(f"    ALERTA: NO_DATA count={n_no_data} (>20) — revisar PASO 2 H2H")
        except Exception:
            pass

    # ── Hipótesis en progreso ─────────────────────────────────────────────
    lines.append("")
    lines.append("  HIPOTESIS (progreso):")
    try:
        hyp_data = json.loads(Path('validation/preregistered_hypotheses.json').read_text())
        for hid, h in hyp_data.get('hypotheses', {}).items():
            n_actual = h.get('n_actual', 0)
            n_stop   = h.get('n_stop', 30)
            estado   = h.get('estado', '?')
            if estado == 'ACUMULANDO' and n_actual > 0:
                lines.append(f"    {hid}: {n_actual}/{n_stop}  {h.get('nombre','?')[:45]}")
    except Exception:
        pass

    lines.append("")
    lines.append(sep)
    lines.append("  ACCIONES:")
    lines.append("  08:30  python3 shadow_book.py --close-snapshot  (antes sesión europea)")
    lines.append("  12:30  python3 shadow_book.py --close-snapshot  (antes sesión americana)")
    lines.append("  POST   python3 run_daily.py --settle-only       (después de los partidos)")
    lines.append(sep)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Orquestador diario del pipeline')
    parser.add_argument('--tomorrow',      action='store_true', help='Partidos de mañana')
    parser.add_argument('--bankroll',      type=int,   default=BANKROLL_GS)
    parser.add_argument('--settle-only',   action='store_true', help='Solo settle de ayer')
    parser.add_argument('--skip-rankings', action='store_true', help='Saltar PASO 0')
    parser.add_argument('--skip-h2h',      action='store_true', help='Saltar PASO 2')
    parser.add_argument('--tier',          nargs='+',
                        default=['grand_slam', 'challenger', 'itf'],
                        help='Tiers a procesar en PASO 4')
    args = parser.parse_args()

    fecha_hoy   = datetime.now().strftime('%Y-%m-%d')
    fecha_ayer  = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    tier_results = {}

    print(f"\n{'#'*72}")
    print(f"  PIPELINE DIARIO — {fecha_hoy}")
    print(f"{'#'*72}")

    # ── SETTLE ONLY ──────────────────────────────────────────────────────
    if args.settle_only:
        print("\n  MODO: solo settle")
        _run(['python3', 'resultados_finales.py'], 'PASO 6 — resultados de ayer')
        _run(['python3', 'shadow_book.py', '--settle', fecha_ayer], 'PASO 10 — shadow book settle')
        print(f"\n  Settle completado para {fecha_ayer}")
        return

    # ── PASO 0 — Rankings ────────────────────────────────────────────────
    if not args.skip_rankings and _rankings_stale():
        _run(['python3', 'extraer_ranking_atp_version2.py'], 'PASO 0a — Rankings ATP')
        _run(['python3', 'extraer_ranking_wta_version2.py'], 'PASO 0b — Rankings WTA')

    # ── PASO 1 — Extraer partidos ─────────────────────────────────────────
    cmd_paso1 = ['python3', 'extraer_partidos_api.py']
    if args.tomorrow:
        cmd_paso1.append('--tomorrow')
    _run(cmd_paso1, 'PASO 1 — Extraer partidos (API)')

    # ── PASO 2 — H2H ─────────────────────────────────────────────────────
    if not args.skip_h2h:
        _run(['python3', 'extraer_historh2h.py', '--api-mode', '--all-tournaments'],
             'PASO 2 — Extraer H2H (Ninja API)')

    # ── PASO 3 — Edge calculator ──────────────────────────────────────────
    _run(['python3', 'edge_calculator.py'], 'PASO 3 — Edge calculator + shadow-log')

    # ── PASO 3.5 — Tabla análisis ─────────────────────────────────────────
    _run(['python3', 'generar_tabla_favoritos2.py'], 'PASO 3.5 — Tabla análisis')

    # ── PASO 3.6 — Games signal ───────────────────────────────────────────
    _run(['python3', 'games_signal_calculator.py'], 'PASO 3.6 — Games signal')

    # ── PASO 4 — Trader por tier ──────────────────────────────────────────
    tier_config = {
        'grand_slam':  {'bankroll': args.bankroll,      'superficie': 'grass'},
        'atp1000':     {'bankroll': 50000,              'superficie': 'grass'},
        'atp500':      {'bankroll': 30000,              'superficie': 'grass'},
        'challenger':  {'bankroll': BANKROLL_CHALLENGER, 'superficie': 'clay'},
        'itf':         {'bankroll': BANKROLL_ITF,        'superficie': 'clay'},
    }

    for tier in args.tier:
        cfg = tier_config.get(tier)
        if not cfg:
            continue
        rc, out = _run(
            ['python3', 'trader_ev_tenis.py',
             '--bankroll', str(cfg['bankroll']),
             '--torneo-tipo', tier,
             '--superficie', cfg['superficie']],
            f'PASO 4 — Trader {tier}',
            capture=True,
        )
        # Leer el plan más reciente para el brief
        plan_file = _latest_report(f'{REPORTS_DIR}/trader_plan_*.json')
        if plan_file:
            try:
                plan = json.loads(Path(plan_file).read_text(encoding='utf-8'))
                tier_results[tier] = {
                    'individuales': plan.get('individuales', []),
                    'cobertura': plan.get('cobertura', []),
                }
            except Exception:
                pass

    # ── PASO 4.3 — Combo confianza ─────────────────────────────────────────
    _run(['python3', 'combo_confianza_builder.py', '--bankroll', str(args.bankroll)],
         'PASO 4.3 — Combo confianza')

    # ── WAS check ─────────────────────────────────────────────────────────
    was_candidates = []
    try:
        edge_file = _latest_report(f'{REPORTS_DIR}/edge_report_*.json')
        if edge_file:
            from betplay_combo_builder import _was_qualifies
            edata = json.loads(Path(edge_file).read_text(encoding='utf-8'))
            for pick in edata.get('watchlist', []):
                if _was_qualifies(pick):
                    was_candidates.append({
                        'jugador':   pick.get('favorito_predicho', '?'),
                        'cuota':     pick.get('cuota_favorito', 0),
                        'edge_pct':  pick.get('edge_pct', '?'),
                        'señal_was': 'señal Markov activa',
                    })
    except Exception:
        pass

    # ── SETTLE DE AYER ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SETTLE AYER ({fecha_ayer})")
    print(f"{'='*70}")
    rc_res, _ = _run(['python3', 'resultados_finales.py'], 'PASO 6 — resultados de ayer')
    if rc_res == 0:
        _run(['python3', 'shadow_book.py', '--settle', fecha_ayer], 'PASO 10 — shadow book settle')

    # ── DAILY BRIEF ───────────────────────────────────────────────────────
    brief = _build_daily_brief(fecha_hoy, tier_results, was_candidates, fecha_ayer)
    brief_path = f'{REPORTS_DIR}/daily_brief_{fecha_hoy}.txt'
    os.makedirs(REPORTS_DIR, exist_ok=True)
    Path(brief_path).write_text(brief, encoding='utf-8')

    print(f"\n{'#'*72}")
    print(brief)
    print(f"\n  Brief guardado: {brief_path}")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()

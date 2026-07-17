"""
signal_audit.py — Trazabilidad completa pick → señales → resultado (Nodo-101 D101-01).

Construye data/signal_audit.jsonl (append-only, inmutable).
Cada registro: fecha + partido + favorito + fingerprint_señales + resultado.

Permite responder:
  - ¿Qué señales tienen el mayor hit% cuando están solas?
  - ¿Qué combinaciones convergen mejor?
  - ¿En qué tier/superficie funcionan más?

Uso:
  python3 scripts/signal_audit.py --rebuild        # incorpora nuevos settled
  python3 scripts/signal_audit.py --rebuild --force # reconstruye completo
  python3 scripts/signal_audit.py --report [--min-n 5]
  python3 scripts/signal_audit.py --player "Hosogi"
"""
from __future__ import annotations
import glob
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
AUDIT_FILE = BASE_DIR / 'data' / 'signal_audit.jsonl'
SB_DIR     = BASE_DIR / 'reports' / 'shadow_book'
REPORTS    = BASE_DIR / 'reports'


# ─── Fingerprint de señales (espejo de _senales_activas en live_edge_monitor) ─

def extract_signal_fingerprint(pick: dict) -> list[str]:
    """
    Extrae el fingerprint completo de señales del pick.
    Incluye señales de generar_tabla_favoritos2.py (contribution, reasoning, special_signals).
    """
    s = []

    # Confianza base
    flag = pick.get('confidence_flag', '')
    if flag in ('STRONG', 'MOD'):
        s.append(flag)

    # Markov estado de forma
    markov = pick.get('markov_favorito', '')
    if markov in ('HOT', 'COLD', 'NEUTRAL'):
        s.append(f'MRK_{markov}')
    markov_r = pick.get('markov_rival', '')
    if markov_r == 'COLD':
        s.append('rival_COLD')

    # Superficie especialización
    surf = pick.get('surface_specialization') or {}
    surf_score = float(surf.get('raw_score') or surf.get('score') or 0.0)
    if surf_score >= 0.65:
        s.append('SURF_HIGH')
    elif surf_score >= 0.55:
        s.append('SURF_OK')

    # Score directo convergencia (Nodo-98)
    sd = int(pick.get('score_directo') or 0)
    if sd >= 4:
        s.append('SD_HIGH')
    elif sd >= 3:
        s.append('SD_MED')

    # ELO dominance
    if pick.get('elo_dominance') in ('STRONG', 'DOMINANT'):
        s.append('ELO_DOM')

    # Edge bruto
    edge = float(pick.get('edge') or 0.0)
    if edge >= 0.20:
        s.append('EDGE_HIGH')
    elif edge >= 0.10:
        s.append('EDGE_MED')

    # Contribution% al puntaje (de generar_tabla_favoritos2.py vía rivalry_analyzer)
    contrib = float(pick.get('contribution') or 0.0)
    if contrib >= 0.70:
        s.append('CONTRIB_HIGH')
    elif contrib >= 0.50:
        s.append('CONTRIB_MED')

    # RFI (Return from Inactivity)
    rfi = int(pick.get('rfi_tier') or 0)
    if rfi >= 2:
        s.append(f'RFI_T{rfi}')
    elif rfi == 1:
        s.append('RFI_T1')

    # IRP rival
    irp = pick.get('irp_rival') or {}
    delta = float(irp.get('delta_return') or 0.0)
    if delta < -0.15:
        s.append('IRP_rival--')
    elif delta < -0.10:
        s.append('IRP_rival-')

    # GCS (Grass Court Specialist)
    if pick.get('gcs_active'):
        s.append('GCS')

    # Señales especiales de generar_tabla_favoritos2.py / rivalry_analyzer
    # Pueden venir en 'special_signals' (lista) o en 'reasoning' (lista de strings)
    special_raw = list(pick.get('special_signals') or [])
    for reason in (pick.get('reasoning') or []):
        special_raw.append(str(reason))

    for raw in special_raw:
        ru = raw.upper()
        if 'CAMPEON' in ru and 'SUPERFICIE' in ru:
            if 'CAMPEON_SUPERF' not in s:
                s.append('CAMPEON_SUPERF')
        if 'CAMPEON DE TORNEO' in ru or 'CAMPEON_TORNEO' in ru:
            if 'CAMPEON_TORNEO' not in s:
                s.append('CAMPEON_TORNEO')
        if 'RACHA_CALIENTE' in ru or ('RACHA' in ru and 'CALIENTE' in ru):
            if 'MRK_HOT' not in s:
                s.append('MRK_HOT')
        if 'AJUSTE DINAMICO' in ru or 'AJUSTE_DINAMICO' in ru:
            if 'AJUSTE_DIN' not in s:
                s.append('AJUSTE_DIN')

    # Tier como contexto (no es señal predictiva pero ayuda a segmentar)
    tier = pick.get('tier', '')
    if tier:
        s.append(f'TIER_{tier.upper()}')

    return s


# ─── Carga de datos ────────────────────────────────────────────────────────────

def _load_edge_reports_by_date() -> dict:
    """dict fecha(YYYYMMDD) → dict partido → pick"""
    by_date: dict = {}
    for f in sorted(REPORTS.glob('edge_report_*.json')):
        try:
            parts = f.stem.split('_')  # edge_report_YYYYMMDD_HHMMSS
            date_key = parts[2] if len(parts) >= 3 else ''
            if not date_key or len(date_key) != 8:
                continue
            data = json.loads(f.read_text(encoding='utf-8'))
            if date_key not in by_date:
                by_date[date_key] = {}
            for sec in ('apostar', 'watchlist'):
                for pick in data.get(sec, []):
                    partido = pick.get('partido', '')
                    if partido and partido not in by_date[date_key]:
                        by_date[date_key][partido] = pick
        except Exception:
            continue
    return by_date


def _load_shadow_book_settled() -> list[dict]:
    """Lee todos los sb_YYYY-MM-DD.jsonl y retorna sólo picks settled.

    El resultado está en rec['resolucion']['resultado'] con valores WON/LOST/VOID.
    Se normaliza a WIN/LOSE para el resto del código.
    """
    settled = []
    if not SB_DIR.exists():
        return []
    for f in sorted(SB_DIR.glob('sb_*.jsonl')):
        try:
            for line in f.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                # resultado puede estar en resolucion (producción) o top-level (tests/legado)
                resolucion = rec.get('resolucion') or {}
                raw = (resolucion.get('resultado') or rec.get('resultado') or '').upper()
                if raw in ('WON', 'WIN', 'GANADO'):
                    rec['_resultado_norm'] = 'WIN'
                elif raw in ('LOST', 'LOSE', 'PERDIDO'):
                    rec['_resultado_norm'] = 'LOSE'
                elif raw == 'VOID':
                    rec['_resultado_norm'] = 'VOID'
                else:
                    continue  # pendiente / sin resultado
                settled.append(rec)
        except Exception:
            continue
    return settled


def _load_existing_ids() -> set:
    if not AUDIT_FILE.exists():
        return set()
    ids = set()
    for line in AUDIT_FILE.read_text(encoding='utf-8').splitlines():
        try:
            ids.add(json.loads(line).get('_id', ''))
        except Exception:
            pass
    return ids


# ─── D103-07: Trazabilidad de no-bets bloqueados por gates ───────────────────

NO_BET_FILE = BASE_DIR / 'data' / 'signal_audit_nobets.jsonl'


def ingest_gate_log(gate_log_path: str | None = None) -> int:
    """
    D103-07 (Nodo-103): Incorpora picks bloqueados por gates del combo builder
    al registro de no-bets. Permite aprender de lo que NO apostamos.

    Lee:  reports/combo_gate_log_*.json (el más reciente si no se especifica)
    Escribe: data/signal_audit_nobets.jsonl (append-only)
    Retorna: número de registros nuevos incorporados.
    """
    # Encontrar gate log
    if gate_log_path:
        log_path = Path(gate_log_path)
    else:
        candidates = sorted(REPORTS.glob('combo_gate_log_*.json'), reverse=True)
        if not candidates:
            return 0
        log_path = candidates[0]

    try:
        gate_records = json.loads(log_path.read_text(encoding='utf-8'))
    except Exception:
        return 0

    # IDs existentes en nobets
    existing_ids: set = set()
    if NO_BET_FILE.exists():
        for line in NO_BET_FILE.read_text(encoding='utf-8').splitlines():
            try:
                existing_ids.add(json.loads(line).get('_id', ''))
            except Exception:
                pass

    new_records = []
    for rec in gate_records:
        nombre  = rec.get('nombre', '')
        gate    = rec.get('gate', '?')
        ts      = (rec.get('ts') or '')[:10]
        motivo  = rec.get('motivo', '')
        _id     = f"NOBET|{ts}|{nombre}|{gate}".replace(' ', '_')
        if _id in existing_ids:
            continue
        new_records.append({
            '_id':      _id,
            'tipo':     'NO_BET',
            'fecha':    ts,
            'nombre':   nombre,
            'gate':     gate,
            'motivo':   motivo,
            'torneo':   rec.get('torneo', ''),
            'conf':     rec.get('conf', None),
            'cuota':    rec.get('cuota', None),
            'combo':    rec.get('combo', ''),
            'source':   str(log_path.name),
        })
        existing_ids.add(_id)

    if new_records:
        NO_BET_FILE.parent.mkdir(exist_ok=True)
        with NO_BET_FILE.open('a', encoding='utf-8') as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return len(new_records)


def report_nobets(min_n: int = 3) -> None:
    """Muestra picks bloqueados más frecuentes — aprender de no-bets."""
    if not NO_BET_FILE.exists():
        print('[signal_audit] Sin no-bets registrados aún.')
        return
    records = []
    for line in NO_BET_FILE.read_text(encoding='utf-8').splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    if not records:
        print('[signal_audit] Sin no-bets registrados aún.')
        return

    from collections import defaultdict
    gate_counts: dict = defaultdict(int)
    player_counts: dict = defaultdict(int)
    for r in records:
        gate_counts[r.get('gate', '?')] += 1
        player_counts[r.get('nombre', '?')] += 1

    print(f'\n{"="*55}')
    print(f' NO-BETS BLOQUEADOS — total: {len(records)}')
    print(f'{"="*55}')
    print(' Por gate:')
    for g, n in sorted(gate_counts.items(), key=lambda x: -x[1]):
        print(f'   {g}: {n}')
    print(' Jugadores más bloqueados (n>={min_n}):')
    for p, n in sorted(player_counts.items(), key=lambda x: -x[1]):
        if n >= min_n:
            print(f'   {p}: {n} veces')


# ─── Build audit ──────────────────────────────────────────────────────────────

def build_audit(force: bool = False) -> int:
    """
    Incorpora picks settled al audit. Retorna nro de registros nuevos.
    force=True: reconstruye desde cero (no append, sobreescribe).
    """
    existing_ids = set() if force else _load_existing_ids()
    edge_by_date = _load_edge_reports_by_date()
    settled      = _load_shadow_book_settled()

    new_records = []
    for sb in settled:
        resultado_norm = sb.get('_resultado_norm', '')
        if resultado_norm == 'VOID':
            continue  # VOID no aporta señal

        ganado = resultado_norm == 'WIN'

        # Fecha: del sb_id (YYYY-MM-DD_...) o logged_at
        sb_id     = sb.get('sb_id', '')
        fecha     = sb_id[:10] if sb_id and len(sb_id) >= 10 else (sb.get('logged_at', '') or '')[:10]

        # Partido y favorito: en pick_snapshot (producción) o top-level (tests)
        snap      = sb.get('pick_snapshot') or {}
        partido   = snap.get('partido') or sb.get('partido') or sb.get('match', '')
        favorito  = snap.get('favorito_predicho') or sb.get('favorito') or sb.get('pick', '')

        if not partido or not favorito:
            continue

        _id = f"{fecha}|{partido}|{favorito}".replace(' ', '_')
        if _id in existing_ids:
            continue

        # Buscar pick enriquecido en edge_report del mismo día
        date_key = fecha.replace('-', '')
        pick     = edge_by_date.get(date_key, {}).get(partido, {})
        senales  = extract_signal_fingerprint(pick) if pick else extract_signal_fingerprint(snap)

        resolucion = sb.get('resolucion') or {}
        record = {
            '_id':        _id,
            'fecha':      fecha,
            'partido':    partido,
            'favorito':   favorito,
            'tier':       pick.get('tier') or snap.get('tier') or sb.get('tier', ''),
            'superficie': pick.get('superficie') or snap.get('superficie') or sb.get('superficie', ''),
            'cuota':      float(pick.get('cuota_fav') or snap.get('cuota_fav') or
                               resolucion.get('cuota_cierre') or 0.0),
            'edge':       float(pick.get('edge') or snap.get('edge') or 0.0),
            'senales':    senales,
            'ganado':     ganado,
            'resultado':  resultado_norm,
        }
        new_records.append(record)
        existing_ids.add(_id)

    if new_records:
        AUDIT_FILE.parent.mkdir(exist_ok=True)
        mode = 'w' if force else 'a'
        with AUDIT_FILE.open(mode, encoding='utf-8') as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    return len(new_records)


def load_audit() -> list[dict]:
    if not AUDIT_FILE.exists():
        return []
    records = []
    for line in AUDIT_FILE.read_text(encoding='utf-8').splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


# ─── Reportes ─────────────────────────────────────────────────────────────────

def report_signal_convergence(min_n: int = 5) -> None:
    """Señales individuales y combinaciones con mayor hit%."""
    records = load_audit()
    if not records:
        print('[signal_audit] Sin datos — correr --rebuild primero.')
        return

    sig_stats:   dict = defaultdict(lambda: {'n': 0, 'wins': 0})
    combo_stats: dict = defaultdict(lambda: {'n': 0, 'wins': 0})

    for r in records:
        senales = [s for s in r.get('senales', []) if not s.startswith('TIER_')]
        ganado  = r.get('ganado', False)
        for sig in senales:
            sig_stats[sig]['n'] += 1
            if ganado:
                sig_stats[sig]['wins'] += 1
        if senales:
            key = '|'.join(sorted(senales))
            combo_stats[key]['n'] += 1
            if ganado:
                combo_stats[key]['wins'] += 1

    print(f'\n{"="*60}')
    print(f' SEÑALES INDIVIDUALES (n>={min_n})  —  total picks: {len(records)}')
    print(f'{"="*60}')
    rows = [(k, v['n'], v['wins'], v['wins']/v['n'])
            for k, v in sig_stats.items() if v['n'] >= min_n]
    for sig, n, wins, hit in sorted(rows, key=lambda x: -x[3]):
        bar = '█' * int(hit * 20)
        print(f'  {sig:<22} n={n:3d}  hit%={hit:.1%}  {bar}')

    print(f'\n{"="*60}')
    print(f' TOP COMBINACIONES (n>={min_n})')
    print(f'{"="*60}')
    rows = [(k, v['n'], v['wins'], v['wins']/v['n'])
            for k, v in combo_stats.items() if v['n'] >= min_n]
    for combo, n, wins, hit in sorted(rows, key=lambda x: -x[3])[:20]:
        print(f'  {hit:.1%}  n={n:3d}  {combo}')


def report_player(jugador: str, min_n: int = 3) -> None:
    """Historial de señales y resultados para un jugador."""
    records = [r for r in load_audit()
               if jugador.lower() in (r.get('favorito') or '').lower()]
    if not records:
        print(f'[signal_audit] Sin datos para "{jugador}"')
        return

    n    = len(records)
    wins = sum(1 for r in records if r.get('ganado'))
    print(f'\n=== {jugador} — n={n} hit%={wins/n:.1%} ===')

    sig_stats: dict = defaultdict(lambda: {'n': 0, 'wins': 0})
    for r in records:
        for sig in r.get('senales', []):
            if sig.startswith('TIER_'):
                continue
            sig_stats[sig]['n'] += 1
            if r.get('ganado'):
                sig_stats[sig]['wins'] += 1

    print('  Señales (n>=2):')
    for sig, v in sorted(sig_stats.items(),
                         key=lambda x: -(x[1]['wins'] / max(x[1]['n'], 1))):
        if v['n'] >= min_n:
            print(f'    {sig:<22} {v["wins"]}/{v["n"]} = {v["wins"]/v["n"]:.1%}')

    print('\n  Por superficie:')
    surf_stats: dict = defaultdict(lambda: {'n': 0, 'wins': 0})
    for r in records:
        surf = r.get('superficie', '?') or '?'
        surf_stats[surf]['n'] += 1
        if r.get('ganado'):
            surf_stats[surf]['wins'] += 1
    for surf, v in sorted(surf_stats.items()):
        print(f'    {surf:<15} {v["wins"]}/{v["n"]} = {v["wins"]/v["n"]:.1%}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Signal Audit — Trazabilidad señales Nodo-101')
    ap.add_argument('--rebuild',  action='store_true', help='Incorpora picks settled nuevos')
    ap.add_argument('--force',    action='store_true', help='Reconstruye completo')
    ap.add_argument('--report',   action='store_true', help='Convergencia de señales')
    ap.add_argument('--player',   type=str,            help='Historial de un jugador')
    ap.add_argument('--min-n',    type=int, default=5, help='N mínimo para reportes')
    ap.add_argument('--nobets',   action='store_true', help='Ingestar gate log + reporte no-bets (D103-07)')
    ap.add_argument('--gate-log', type=str, default=None, help='Path explícito al combo_gate_log_*.json')
    args = ap.parse_args()

    if args.rebuild or args.force:
        n = build_audit(force=args.force)
        print(f'[signal_audit] {n} registros nuevos → {AUDIT_FILE}')
    if args.nobets:
        n = ingest_gate_log(args.gate_log)
        print(f'[signal_audit] {n} no-bets incorporados → {NO_BET_FILE}')
        report_nobets(min_n=args.min_n)
    if args.report:
        report_signal_convergence(min_n=args.min_n)
    if args.player:
        report_player(args.player)
    if not any([args.rebuild, args.force, args.report, args.player, args.nobets]):
        ap.print_help()

"""
player_consistency.py — Consistencia, matchup y resiliencia por jugador (Nodo-101 D101-02).

Responde:
  1. ¿Qué jugadores son más consistentes para apostar? (hit% estable n>=10)
  2. ¿Cómo rinden contra tipos específicos de rival?
  3. ¿Sostienen las señales bajo adversidad (TOP_TIER) o se rinden?

Fuente: data/signal_audit.jsonl (construido por signal_audit.py)
Output: data/player_consistency.json

Uso:
  python3 scripts/player_consistency.py --rebuild
  python3 scripts/player_consistency.py --report [--min-n 10]
  python3 scripts/player_consistency.py --player "Hosogi"
  python3 scripts/player_consistency.py --matchup "Hosogi" vs "Kinoshita"
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

BASE_DIR     = Path(__file__).parent.parent
CONSIST_FILE = BASE_DIR / 'data' / 'player_consistency.json'


def _load_audit() -> list[dict]:
    audit_file = BASE_DIR / 'data' / 'signal_audit.jsonl'
    if not audit_file.exists():
        return []
    records = []
    for line in audit_file.read_text(encoding='utf-8').splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def _classify_rival_context(r: dict) -> str:
    """
    Clasifica el contexto del rival desde señales + tier.
    Permite segmentar 'contra quién ganamos' y 'contra quién fallamos'.
    """
    senales = r.get('senales', [])
    tier    = r.get('tier', '')

    if 'ELO_DOM' in senales:
        return 'DOMINADO'          # rival claramente inferior en ELO
    if 'rival_COLD' in senales:
        return 'RIVAL_COLD'        # rival en mal momento de forma
    if 'IRP_rival--' in senales or 'IRP_rival-' in senales:
        return 'RIVAL_IRP_NEG'     # rival vuelve de inactividad
    if tier in ('gs', 'atp1000'):
        return 'TOP_TIER'          # torneo de alto nivel = rival probable fuerte
    if tier in ('challenger',):
        return 'MID_TIER'
    if tier in ('itf',):
        return 'LOW_TIER'
    return 'UNKNOWN'


def build_consistency_profiles(records: list[dict]) -> dict:
    """
    Construye perfiles de consistencia por jugador.

    Cada perfil contiene:
      n, wins, hit_pct, consistent (n>=10 y hit>=55%)
      resilient: ganó ≥2 veces en TOP_TIER = sustenta la presión
      best_signals: señales que co-ocurren con hit% superior al promedio del jugador
      by_rival_context: rendimiento segmentado por tipo de rival
      by_surface: rendimiento por superficie
      signal_in_wins: frecuencia de cada señal en sus victorias
      signal_in_losses: frecuencia de cada señal en sus derrotas
    """
    raw: dict = defaultdict(lambda: {
        'n': 0, 'wins': 0,
        'by_rival': defaultdict(lambda: {'n': 0, 'wins': 0}),
        'by_surface': defaultdict(lambda: {'n': 0, 'wins': 0}),
        'by_signal': defaultdict(lambda: {'n': 0, 'wins': 0}),
        'wins_signals': [],
        'loss_signals': [],
    })

    for r in records:
        jugador  = (r.get('favorito') or '').strip()
        if not jugador:
            continue
        ganado   = r.get('ganado', False)
        senales  = [s for s in r.get('senales', []) if not s.startswith('TIER_')]
        rival_ctx = _classify_rival_context(r)
        surface   = (r.get('superficie') or 'unknown').lower()

        p = raw[jugador]
        p['n'] += 1
        if ganado:
            p['wins'] += 1
            p['wins_signals'].extend(senales)
        else:
            p['loss_signals'].extend(senales)

        p['by_rival'][rival_ctx]['n'] += 1
        if ganado:
            p['by_rival'][rival_ctx]['wins'] += 1

        p['by_surface'][surface]['n'] += 1
        if ganado:
            p['by_surface'][surface]['wins'] += 1

        for sig in senales:
            p['by_signal'][sig]['n'] += 1
            if ganado:
                p['by_signal'][sig]['wins'] += 1

    # Post-process → perfiles finales
    profiles: dict = {}
    for jugador, p in raw.items():
        n    = p['n']
        wins = p['wins']
        hit  = wins / n if n > 0 else 0.0

        # Resiliencia: ¿ganó ≥2 veces en torneos de alto nivel?
        top_wins = p['by_rival'].get('TOP_TIER', {}).get('wins', 0)
        resilient = top_wins >= 2

        # Señales que elevan su hit% ≥10pp sobre el promedio (n>=3)
        best_signals = []
        for sig, sv in p['by_signal'].items():
            if sv['n'] >= 3 and n > 0:
                sig_hit = sv['wins'] / sv['n']
                if sig_hit >= hit + 0.10:
                    best_signals.append({
                        'signal': sig,
                        'hit':    round(sig_hit, 3),
                        'n':      sv['n'],
                        'lift':   round(sig_hit - hit, 3),
                    })
        best_signals.sort(key=lambda x: -x['lift'])

        # Señales que aparecen en victorias pero NO en derrotas (discriminativas)
        wins_set  = set(p['wins_signals'])
        loss_set  = set(p['loss_signals'])
        exclusive_win_signals = sorted(wins_set - loss_set)

        # Serializar sub-dicts
        def _finalize(d: dict) -> dict:
            out = {}
            for k, v in d.items():
                vn = v['n'] if isinstance(v, dict) else 0
                vw = v.get('wins', 0) if isinstance(v, dict) else 0
                out[k] = {'n': vn, 'wins': vw,
                           'hit_pct': round(vw / vn, 3) if vn > 0 else 0.0}
            return out

        profiles[jugador] = {
            'n':                     n,
            'wins':                  wins,
            'hit_pct':               round(hit, 3),
            'consistent':            n >= 10 and hit >= 0.55,
            'resilient':             resilient,
            'top_tier_wins':         top_wins,
            'best_signals':          best_signals[:5],
            'exclusive_win_signals': exclusive_win_signals,
            'by_rival_context':      _finalize(dict(p['by_rival'])),
            'by_surface':            _finalize(dict(p['by_surface'])),
            'by_signal':             _finalize(dict(p['by_signal'])),
        }

    return profiles


def rebuild() -> None:
    """Reconstruye player_consistency.json desde signal_audit.jsonl."""
    import sys
    sys.path.insert(0, str(BASE_DIR / 'scripts'))
    from signal_audit import build_audit, load_audit
    build_audit()
    records  = load_audit()
    profiles = build_consistency_profiles(records)
    CONSIST_FILE.parent.mkdir(exist_ok=True)
    CONSIST_FILE.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'[player_consistency] {len(profiles)} perfiles → {CONSIST_FILE}')


def load_profiles() -> dict:
    if not CONSIST_FILE.exists():
        return {}
    try:
        return json.loads(CONSIST_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


# ─── Reportes ─────────────────────────────────────────────────────────────────

def report_top_consistent(min_n: int = 10) -> None:
    profiles = load_profiles()
    rows = [(k, v) for k, v in profiles.items() if v['n'] >= min_n]
    rows.sort(key=lambda x: -x[1]['hit_pct'])

    print(f'\n{"="*70}')
    print(f' JUGADORES MÁS CONSISTENTES (n>={min_n})')
    print(f'{"="*70}')
    print(f'  {"Jugador":<28} {"n":>4} {"hit%":>6} {"Top-tier":>8} {"Resiliente":>10} {"Mejor señal"}')
    print(f'  {"-"*65}')
    for jugador, v in rows:
        star = '★' if v['consistent'] else ' '
        res  = 'SI' if v['resilient'] else '  '
        best = v['best_signals'][0]['signal'] if v['best_signals'] else '—'
        print(f'  {star}{jugador:<27} {v["n"]:>4} {v["hit_pct"]:>6.1%} '
              f'{v["top_tier_wins"]:>8} {res:>10} {best}')

    print(f'\nTotal jugadores con n>={min_n}: {len(rows)}')


def report_player(jugador: str) -> None:
    profiles = load_profiles()
    key = next((k for k in profiles if jugador.lower() in k.lower()), None)
    if not key:
        print(f'[player_consistency] Sin perfil para "{jugador}"')
        return

    v = profiles[key]
    print(f'\n{"="*60}')
    print(f' {key}')
    print(f'{"="*60}')
    print(f'  n={v["n"]}  hit%={v["hit_pct"]:.1%}  '
          f'consistente={"SI" if v["consistent"] else "NO"}  '
          f'resiliente={"SI" if v["resilient"] else "NO"}  '
          f'top_tier_wins={v["top_tier_wins"]}')

    if v['best_signals']:
        print('\n  Señales que elevan hit% (lift vs promedio):')
        for bs in v['best_signals']:
            print(f'    {bs["signal"]:<22} n={bs["n"]}  '
                  f'hit%={bs["hit"]:.1%}  lift=+{bs["lift"]:.1%}')

    if v['exclusive_win_signals']:
        print(f'\n  Señales que aparecen SOLO en victorias: '
              f'{", ".join(v["exclusive_win_signals"])}')

    print('\n  Por contexto de rival:')
    for ctx, sv in sorted(v['by_rival_context'].items()):
        print(f'    {ctx:<20} {sv["wins"]}/{sv["n"]} = {sv["hit_pct"]:.1%}')

    print('\n  Por superficie:')
    for surf, sv in sorted(v['by_surface'].items()):
        print(f'    {surf:<15} {sv["wins"]}/{sv["n"]} = {sv["hit_pct"]:.1%}')


def report_matchup(jugador_a: str, jugador_b: str) -> None:
    """
    Compara los perfiles de dos jugadores para inferir quién tiene la señal
    más sólida en el contexto del partido.
    """
    profiles = load_profiles()
    key_a = next((k for k in profiles if jugador_a.lower() in k.lower()), None)
    key_b = next((k for k in profiles if jugador_b.lower() in k.lower()), None)

    print(f'\n{"="*60}')
    print(f' MATCHUP: {jugador_a} vs {jugador_b}')
    print(f'{"="*60}')
    for label, key in [(jugador_a, key_a), (jugador_b, key_b)]:
        if not key:
            print(f'  {label}: sin perfil (n<3)')
            continue
        v = profiles[key]
        best = v['best_signals'][0]['signal'] if v['best_signals'] else '—'
        print(f'  {key:<28} hit%={v["hit_pct"]:.1%}  n={v["n"]}  '
              f'res={"SI" if v["resilient"] else "NO"}  '
              f'mejor_senal={best}')

    if key_a and key_b:
        va = profiles[key_a]
        vb = profiles[key_b]
        # Ventaja simple: quien tiene mayor hit% consolidado
        if va['hit_pct'] > vb['hit_pct'] + 0.05:
            winner = key_a
        elif vb['hit_pct'] > va['hit_pct'] + 0.05:
            winner = key_b
        else:
            winner = None

        if winner:
            print(f'\n  >> Ventaja histórica de señales: {winner}')
        else:
            print(f'\n  >> Rendimiento histórico similar — señales extra deciden')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Player Consistency — Consistencia, matchup y resiliencia Nodo-101'
    )
    ap.add_argument('--rebuild',  action='store_true')
    ap.add_argument('--report',   action='store_true')
    ap.add_argument('--player',   type=str)
    ap.add_argument('--matchup',  type=str, nargs=2, metavar=('A', 'B'))
    ap.add_argument('--min-n',    type=int, default=10)
    args = ap.parse_args()

    if args.rebuild:
        rebuild()
    if args.report:
        report_top_consistent(min_n=args.min_n)
    if args.player:
        report_player(args.player)
    if args.matchup:
        report_matchup(args.matchup[0], args.matchup[1])
    if not any([args.rebuild, args.report, args.player, args.matchup]):
        ap.print_help()

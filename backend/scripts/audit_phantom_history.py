"""
D152-06 (Nodo-152): Auditoría retroactiva de historiales contaminados.

Escanea todos los h2h_results_enhanced_*.json de los últimos N días,
aplica _validate_circuit_consistency() a cada historial y cruza con
shadow_book para identificar picks apostados con datos falsos.

Uso:
    python3 scripts/audit_phantom_history.py [--days 30] [--verbose]
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Asegurar que el módulo scraping es encontrable
sys.path.insert(0, str(Path(__file__).parent.parent))
from scraping.ninja_h2h_parser import _validate_circuit_consistency

REPORTS_DIR = Path(__file__).parent.parent / "reports"
SB_DIR = REPORTS_DIR / "shadow_book"

_ITF_KW  = ('M15', 'M25', 'W15', 'W25', 'ITF')
_CHAL_KW = ('Challenger', 'M50', 'M75', 'W50', 'W75')


def _tier_from_torneo(torneo_full: str) -> str:
    for kw in _ITF_KW:
        if kw in torneo_full:
            return 'itf'
    for kw in _CHAL_KW:
        if kw in torneo_full:
            return 'challenger'
    return 'atp'


def _load_shadow_book(days: int) -> dict:
    """Carga registros shadow_book de los últimos N días → {match_key: record}"""
    sb_map = {}
    cutoff = datetime.now() - timedelta(days=days)
    for f in sorted(SB_DIR.glob("sb_*.jsonl")):
        try:
            fecha = datetime.strptime(f.stem.replace('sb_', ''), '%Y-%m-%d')
        except ValueError:
            continue
        if fecha < cutoff:
            continue
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    mk = r.get('match_key', '')
                    if mk:
                        sb_map[mk] = r
                except Exception:
                    pass
    return sb_map


def _match_key_from_players(j1: str, j2: str) -> str:
    def _norm(n):
        parts = n.lower().split()
        return parts[-1] if parts else n.lower()
    return f"{_norm(j1)}_{_norm(j2)}"


def audit(days: int = 30, verbose: bool = False) -> dict:
    cutoff = datetime.now() - timedelta(days=days)
    h2h_files = sorted(REPORTS_DIR.glob("h2h_results_enhanced_*.json"))

    sb_map = _load_shadow_book(days)

    contaminated_players = []
    total_partidos = 0
    total_picks_afectados = 0

    for h2h_path in h2h_files:
        # Filtrar por fecha del archivo
        try:
            mtime = datetime.fromtimestamp(h2h_path.stat().st_mtime)
            if mtime < cutoff:
                continue
        except Exception:
            continue

        try:
            with open(h2h_path) as f:
                raw = json.load(f)
        except Exception:
            continue

        partidos = raw.get('partidos', raw) if isinstance(raw, dict) else raw
        if not isinstance(partidos, list):
            continue

        for partido in partidos:
            torneo_full = partido.get('torneo_completo', '')
            tier = _tier_from_torneo(torneo_full)
            j1 = partido.get('jugador1', '')
            j2 = partido.get('jugador2', '')
            total_partidos += 1

            for player_key in [k for k in partido if k.startswith('historial_')]:
                player_name = player_key.replace('historial_', '').replace('_', ' ').strip()
                history = partido[player_key]
                if not history:
                    continue

                # Determinar provenance desde data_quality si existe
                dq = partido.get('data_quality', {})
                prov = dq.get('history_provenance', {})
                p_idx = 'p1' if player_name.replace(' ', '_') in player_key[:20] else 'p2'
                provenance = prov.get(p_idx, 'unknown')

                val = _validate_circuit_consistency(history, tier, provenance)
                if not val['contaminated']:
                    continue

                # Cruzar con shadow_book
                mk = _match_key_from_players(j1, j2)
                sb_record = sb_map.get(mk)
                resultado_real = None
                pnl = None
                if sb_record:
                    res = sb_record.get('resolucion') or {}
                    resultado_real = res.get('resultado')
                    pnl = res.get('pnl_flat_1u')
                    total_picks_afectados += 1

                entry = {
                    'archivo': h2h_path.name,
                    'jugador': player_name,
                    'rival': j2 if player_name.lower() in j1.lower() else j1,
                    'torneo': partido.get('torneo_nombre', ''),
                    'tier': tier,
                    'provenance': provenance,
                    'contamination_score': val['score'],
                    'evidence': val['evidence'],
                    'n_history': len(history),
                    'resultado_real': resultado_real,
                    'pnl_flat': pnl,
                    'shadow_book_hit': sb_record is not None,
                }
                contaminated_players.append(entry)

                if verbose:
                    print(f"[CONTAMINADO] {player_name} | {partido.get('torneo_nombre')} ({tier}) | "
                          f"score={val['score']} | {val['evidence'][:2]}")
                    if sb_record:
                        print(f"  → shadow_book: resultado={resultado_real} pnl={pnl}")

    report = {
        'generated_at': datetime.now().isoformat(),
        'days_scanned': days,
        'total_partidos_escaneados': total_partidos,
        'total_contaminados': len(contaminated_players),
        'total_picks_afectados_shadow_book': total_picks_afectados,
        'contaminados': contaminated_players,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description='Auditoría retroactiva phantom history (Nodo-152)')
    parser.add_argument('--days', type=int, default=30, help='Días a escanear (default 30)')
    parser.add_argument('--verbose', action='store_true', help='Output detallado')
    parser.add_argument('--out', type=str, default='', help='Archivo de salida (default auto)')
    args = parser.parse_args()

    print(f"[D152-06] Escaneando {args.days} días de h2h_results_enhanced...")
    report = audit(days=args.days, verbose=args.verbose)

    out_path = args.out or str(REPORTS_DIR / f"audit_phantom_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== AUDITORÍA PHANTOM HISTORY (Nodo-152) ===")
    print(f"  Días escaneados:          {report['days_scanned']}")
    print(f"  Partidos totales:         {report['total_partidos_escaneados']}")
    print(f"  Jugadores contaminados:   {report['total_contaminados']}")
    print(f"  Picks en shadow_book:     {report['total_picks_afectados_shadow_book']}")
    print(f"  Reporte guardado:         {out_path}")

    if report['contaminados']:
        print(f"\n  Jugadores afectados:")
        seen = set()
        for c in report['contaminados']:
            key = c['jugador']
            if key in seen:
                continue
            seen.add(key)
            sb_str = f"resultado={c['resultado_real']} pnl={c['pnl_flat']}" if c['shadow_book_hit'] else "sin shadow_book"
            print(f"    {c['jugador']:35s} score={c['contamination_score']:3d} | {sb_str}")


if __name__ == '__main__':
    main()

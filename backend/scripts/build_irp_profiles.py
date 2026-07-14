#!/usr/bin/env python3
"""
scripts/build_irp_profiles.py — Nodo-96: IRP Individual Return-from-inactivity Profile

Lee data/player_db.json y calcula para cada jugador sus estadísticas de rendimiento
en "partidos de retorno" (primer partido tras un gap > RETURN_THRESHOLD_DAYS).

Output: data/irp_profiles.json
  - profiles: { slug → { n_matches, n_retornos, win_rate_return, win_rate_normal,
                          delta_return, avg_gap_return, last_match_fecha, days_since_last } }
  - name_index: { normalized_name → slug }  ← para lookup rápido desde edge_calculator

REPORTE_SOLO — este archivo se usa solo para serializar irp_fav/irp_rival en el edge_report.
No afecta edge, kelly_kl, p_modelo ni ninguna decisión de apuesta (D96-01).

Precondición: data/player_db.json debe existir (Nodo-93, Sprint 2).
"""

from __future__ import annotations

import json
import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent.parent
PLAYER_DB_PATH  = _BACKEND / 'data' / 'player_db.json'
IRP_OUTPUT_PATH = _BACKEND / 'data' / 'irp_profiles.json'

# ─── Configuración ────────────────────────────────────────────────────────────
RETURN_THRESHOLD_DAYS: int = 30   # D96-02: mismo inflection point que form_decay (Nodo-57)
MIN_RETORNOS: int = 2             # D96-03: mínimo retornos para tener profile


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN DE NOMBRES (lookup desde edge_calculator)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_for_index(name: str) -> str:
    """
    Normaliza un nombre para el name_index:
    - NFKD → ASCII
    - lower
    - strip

    Ejemplo: 'Novak_Djokovic' → 'novak djokovic'
             'Hsu Yu-Hsiou'   → 'hsu yu-hsiou'
    """
    name = name.replace('_', ' ')
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CÓMPUTO IRP POR JUGADOR
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_fecha(s: str) -> date | None:
    """Parsea YYYY-MM-DD → date. Retorna None si falla."""
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def compute_player_irp(slug: str, rows: list[dict], build_date: date) -> dict | None:
    """
    Calcula el IRP para un jugador dado su historial de matches.

    Args:
        slug:       clave canónica del jugador (ej. 'Novak_Djokovic')
        rows:       lista de dicts de matches desde PlayerDB (puede estar en cualquier orden)
        build_date: fecha de referencia para `days_since_last`

    Returns:
        dict con campos IRP, o None si n_retornos < MIN_RETORNOS
    """
    # Parsear y ordenar por fecha ascendente
    dated: list[tuple[date, dict]] = []
    for row in rows:
        d = _parse_fecha(row.get('fecha', ''))
        if d is not None:
            dated.append((d, row))

    if not dated:
        return None

    dated.sort(key=lambda x: x[0])

    last_match_fecha = dated[-1][0]
    days_since_last  = (build_date - last_match_fecha).days

    # Clasificar cada match como return_match o normal
    return_wins   = 0
    return_losses = 0
    normal_wins   = 0
    normal_losses = 0
    gap_days_list: list[float] = []

    for i, (d, row) in enumerate(dated):
        won = bool(row.get('won', False))

        if i == 0:
            # Primer match: sin gap anterior → normal
            if won:
                normal_wins += 1
            else:
                normal_losses += 1
            continue

        prev_date = dated[i - 1][0]
        gap = (d - prev_date).days

        if gap > RETURN_THRESHOLD_DAYS:
            # Return match
            gap_days_list.append(float(gap))
            if won:
                return_wins += 1
            else:
                return_losses += 1
        else:
            if won:
                normal_wins += 1
            else:
                normal_losses += 1

    n_retornos = return_wins + return_losses
    if n_retornos < MIN_RETORNOS:
        return None

    n_normal = normal_wins + normal_losses
    win_rate_return = return_wins / n_retornos
    win_rate_normal = normal_wins / n_normal if n_normal > 0 else None
    delta_return    = (
        round(win_rate_return - win_rate_normal, 4)
        if win_rate_normal is not None else None
    )
    avg_gap_return = round(sum(gap_days_list) / len(gap_days_list), 1) if gap_days_list else None

    return {
        'slug':              slug,
        'n_matches':         len(dated),
        'n_retornos':        n_retornos,
        'win_rate_return':   round(win_rate_return, 4),
        'win_rate_normal':   round(win_rate_normal, 4) if win_rate_normal is not None else None,
        'delta_return':      delta_return,
        'avg_gap_return':    avg_gap_return,
        'last_match_fecha':  last_match_fecha.isoformat(),
        'days_since_last':   days_since_last,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def build_irp_profiles(
    player_db_path: Path = PLAYER_DB_PATH,
    output_path: Path    = IRP_OUTPUT_PATH,
    build_date: date | None = None,
) -> dict:
    """
    Lee player_db.json y construye irp_profiles.json.

    Returns el dict de resultado (también escrito en output_path).
    """
    if not player_db_path.exists():
        print(f'  [IRP] SKIP — {player_db_path} no existe (correr build_player_db.py primero)')
        return {}

    bd = build_date or date.today()

    print(f'  [IRP] Leyendo {player_db_path.name} …')
    with player_db_path.open(encoding='utf-8') as f:
        db = json.load(f)

    players: dict = db.get('players', {})
    print(f'  [IRP] {len(players):,} jugadores en PlayerDB — calculando profiles …')

    profiles: dict[str, dict] = {}
    name_index: dict[str, str] = {}

    for slug, entry in players.items():
        rows = entry.get('rows', [])
        profile = compute_player_irp(slug, rows, bd)
        if profile is not None:
            profiles[slug] = profile
            # H96-02: name_index con slug completo + apellido fallback
            # slug completo: 'novak djokovic' → 'Novak_Djokovic'
            _full_key = normalize_for_index(slug)
            name_index[_full_key] = slug
            # apellido (última palabra): 'djokovic' → 'Novak_Djokovic'
            # Colisión posible si dos jugadores comparten apellido — el último gana.
            # Aceptable: REPORTE_SOLO, colisión = dato observacional erróneo, nunca decisión.
            _parts = _full_key.split()
            if len(_parts) > 1:
                name_index[_parts[-1]] = slug

    n_with_irp = len(profiles)
    print(f'  [IRP] {n_with_irp:,} jugadores con IRP (n_retornos≥{MIN_RETORNOS})')

    result = {
        'built_at':              datetime.now().isoformat(),
        'build_date':            bd.isoformat(),
        'n_players_with_irp':    n_with_irp,
        'return_threshold_days': RETURN_THRESHOLD_DAYS,
        'min_retornos':          MIN_RETORNOS,
        'profiles':              profiles,
        'name_index':            name_index,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'  [IRP] Escrito: {output_path}')
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Build IRP profiles from PlayerDB (Nodo-96)')
    parser.add_argument(
        '--date',
        help='Fecha de referencia YYYY-MM-DD (default: hoy)',
        default=None,
    )
    parser.add_argument(
        '--player-db',
        help=f'Path alternativo a player_db.json (default: {PLAYER_DB_PATH})',
        default=str(PLAYER_DB_PATH),
    )
    parser.add_argument(
        '--output',
        help=f'Path de salida (default: {IRP_OUTPUT_PATH})',
        default=str(IRP_OUTPUT_PATH),
    )
    args = parser.parse_args()

    bd = date.fromisoformat(args.date) if args.date else None
    result = build_irp_profiles(
        player_db_path=Path(args.player_db),
        output_path=Path(args.output),
        build_date=bd,
    )

    if not result:
        sys.exit(0)

    # Estadísticas rápidas
    profiles = result.get('profiles', {})
    if profiles:
        deltas = [p['delta_return'] for p in profiles.values() if p.get('delta_return') is not None]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            n_peor    = sum(1 for d in deltas if d < -0.05)
            n_mejor   = sum(1 for d in deltas if d > 0.05)
            print(f'  [IRP] delta_return promedio: {avg_delta:+.3f} | '
                  f'peor en retorno (<-5%): {n_peor} | mejor en retorno (>+5%): {n_mejor}')


if __name__ == '__main__':
    main()

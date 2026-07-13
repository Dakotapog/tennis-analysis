#!/usr/bin/env python3
"""
build_player_db.py — D90-03 (Nodo-90)

Construye data/player_db.json + data/player_db_index.json desde los bloques
historial_* settled dentro de los archivos h2h_results_enhanced_*.json.

Reglas:
- Deduplica por (jugador_slug, fecha_iso, oponente_raw)
- Prefiere la fila del archivo más reciente (mayor fecha_extraccion)
- resolution_confidence siempre "exact" (fuente: clave historial_* del H2H)
- own_ranking tomado del ranking_analysis del archivo; ranking_asof = fecha_extraccion
- Filas con resultado "-" o "0-0" se descartan (partidos walkover / sin datos)

Uso:
    python3 scripts/build_player_db.py [--reports-dir reports/] [--out-dir data/]
    python3 scripts/build_player_db.py --incremental   # solo procesa archivos nuevos
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, date

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BACKEND_DIR)

from config import detectar_tier

# ── constants ─────────────────────────────────────────────────────────────────
SUPERFICIE_MAP = {
    'dura':    'dura',
    'arcilla': 'arcilla',
    'hierba':  'hierba',
    'n/a':     'unknown',
    '':        'unknown',
}

# Ranking gap brackets: rank_diff = own_ranking - opponent_ranking
# Lower rank number = better player → negative diff = we're better
RANKING_BRACKETS = [
    ('dominant',       lambda d: d < -50),   # own much better
    ('favored',        lambda d: -50 <= d < -10),
    ('even',           lambda d: -10 <= d <= 10),
    ('underdog_slight',lambda d: 10 < d <= 50),
    ('underdog_big',   lambda d: d > 50),    # opponent much better
]

# Resultados que indican walkover o datos inválidos (descartar)
_INVALID_RESULTADOS = {'-', '0-0', ''}


def _parse_filename_date(filename: str) -> str | None:
    """Extrae fecha_extraccion ISO de h2h_results_enhanced_YYYYMMDD_HHMMSS.json"""
    base = os.path.basename(filename).replace('.json', '')
    # h2h_results_enhanced_20260614_012628
    parts = base.split('_')
    # find YYYYMMDD part
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            try:
                dt = datetime.strptime(p, '%Y%m%d')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    return None


def _normalize_superficie(raw: str) -> str:
    return SUPERFICIE_MAP.get(raw.strip().lower(), 'unknown')


def _is_three_set_match(resultado: str) -> bool:
    """2-1 o 1-2 → partido a 3 sets (proxy para partidos tensos)."""
    return resultado in ('2-1', '1-2', '1-3', '3-1', '3-2', '2-3')


def _is_underdog(own_ranking, opponent_ranking) -> bool | None:
    """True si jugador tiene ranking numérico PEOR que rival (= underdog)."""
    if own_ranking is None or opponent_ranking is None:
        return None
    return own_ranking > opponent_ranking


def _ranking_bracket(rank_diff) -> str | None:
    if rank_diff is None:
        return None
    for name, pred in RANKING_BRACKETS:
        if pred(rank_diff):
            return name
    return 'unknown'


def _fecha_to_iso(fecha_raw: str) -> str | None:
    """DD.MM.YYYY → YYYY-MM-DD"""
    try:
        return datetime.strptime(fecha_raw, '%d.%m.%Y').strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def _init_stats_bucket():
    return {'n': 0, 'wins': 0, 'losses': 0}


def _add_result(bucket, won: bool):
    bucket['n'] += 1
    if won:
        bucket['wins'] += 1
    else:
        bucket['losses'] += 1


def _win_rate(bucket) -> float | None:
    if bucket['n'] == 0:
        return None
    return round(bucket['wins'] / bucket['n'], 4)


def _compute_player_stats(rows: list) -> dict:
    """Agrega estadísticas desde filas deduplicadas de un jugador."""
    surface_stats = defaultdict(_init_stats_bucket)
    tier_stats = defaultdict(_init_stats_bucket)
    ranking_gap_stats = defaultdict(_init_stats_bucket)
    prs_stats = {
        'three_set': _init_stats_bucket(),
        'two_set': _init_stats_bucket(),
        'underdog': _init_stats_bucket(),
        'favorite': _init_stats_bucket(),
    }

    for row in rows:
        won = row['won']
        sup = row['superficie']
        tier = row['tier']
        bracket = row.get('ranking_bracket')
        is_three_set = row.get('is_three_set', False)
        is_und = row.get('is_underdog')

        _add_result(surface_stats[sup], won)
        _add_result(tier_stats[tier], won)
        if bracket:
            _add_result(ranking_gap_stats[bracket], won)

        # PRS (Dim 4 proxy)
        if is_three_set:
            _add_result(prs_stats['three_set'], won)
        else:
            _add_result(prs_stats['two_set'], won)
        if is_und is True:
            _add_result(prs_stats['underdog'], won)
        elif is_und is False:
            _add_result(prs_stats['favorite'], won)

    # Finalize win_rates
    def finalize(d):
        return {k: {**v, 'win_rate': _win_rate(v)} for k, v in d.items()}

    return {
        'n_total': len(rows),
        'surface_stats': finalize(surface_stats),
        'tier_stats': finalize(tier_stats),
        'ranking_gap_stats': finalize(ranking_gap_stats),
        'prs_stats': {k: {**v, 'win_rate': _win_rate(v)} for k, v in prs_stats.items()},
    }


def process_files(h2h_files: list, verbose: bool = True) -> dict:
    """
    Lee todos los archivos H2H y retorna:
    {
        slug: {
            'slug': str,
            'own_ranking': int | None,
            'ranking_asof': str,
            'rows': [...],  # deduplicadas
            'stats': {...},
        }
    }
    """
    # Ordenar por fecha_extraccion: más reciente al final (para prefer-latest en dedup)
    h2h_files_sorted = sorted(h2h_files)

    # Acumulador de filas: key = (slug, fecha_iso, oponente_raw)
    rows_by_key: dict[tuple, dict] = {}

    # Ranking más reciente por slug
    best_ranking: dict[str, tuple] = {}  # slug → (ranking, fecha_extraccion)

    n_files = 0
    n_rows_raw = 0

    for filepath in h2h_files_sorted:
        fecha_extraccion = _parse_filename_date(filepath)
        if fecha_extraccion is None:
            if verbose:
                print(f'  [WARN] No se pudo extraer fecha de {os.path.basename(filepath)}')
            continue

        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if verbose:
                print(f'  [WARN] Error leyendo {os.path.basename(filepath)}: {e}')
            continue

        n_files += 1
        partidos = data.get('partidos', [])

        for partido in partidos:
            ranking_analysis = partido.get('ranking_analysis', {})

            # Iterar sobre todos los bloques historial_*
            for key, historial in partido.items():
                if not key.startswith('historial_'):
                    continue
                if not isinstance(historial, list) or not historial:
                    continue

                slug = key[len('historial_'):]  # e.g. "Gabriela_Knutson"

                # own_ranking desde ranking_analysis del partido
                own_ranking = ranking_analysis.get(f'{slug}_ranking')
                if isinstance(own_ranking, (int, float)) and own_ranking > 0:
                    # Actualizar si este archivo es más reciente
                    prev = best_ranking.get(slug)
                    if prev is None or fecha_extraccion >= prev[1]:
                        best_ranking[slug] = (int(own_ranking), fecha_extraccion)

                for entry in historial:
                    fecha_raw = entry.get('fecha', '')
                    fecha_iso = _fecha_to_iso(fecha_raw)
                    if fecha_iso is None:
                        continue

                    resultado = entry.get('resultado', '').strip()
                    if resultado in _INVALID_RESULTADOS:
                        continue

                    oponente_raw = entry.get('oponente', '').strip()
                    if not oponente_raw:
                        continue

                    dedup_key = (slug, fecha_iso, oponente_raw)
                    n_rows_raw += 1

                    outcome = entry.get('outcome', '')
                    won = (outcome == 'Ganó')

                    sup = _normalize_superficie(entry.get('superficie', ''))
                    torneo = entry.get('torneo', '')
                    tier = detectar_tier(torneo) or 'unknown'
                    opp_ranking = entry.get('opponent_ranking')
                    if isinstance(opp_ranking, (int, float)) and opp_ranking > 0:
                        opp_ranking = int(opp_ranking)
                    else:
                        opp_ranking = None

                    is_three_set = _is_three_set_match(resultado)
                    own_rk_snapshot = best_ranking.get(slug, (None, None))[0]

                    rank_diff = None
                    if own_rk_snapshot and opp_ranking:
                        rank_diff = own_rk_snapshot - opp_ranking

                    row = {
                        'fecha': fecha_iso,
                        'oponente': oponente_raw,
                        'resultado': resultado,
                        'won': won,
                        'superficie': sup,
                        'torneo': torneo,
                        'tier': tier,
                        'opponent_ranking': opp_ranking,
                        'opponent_weight': entry.get('opponent_weight', 1),
                        'is_three_set': is_three_set,
                        'is_underdog': _is_underdog(own_rk_snapshot, opp_ranking),
                        'ranking_bracket': _ranking_bracket(rank_diff),
                        'resolution_confidence': 'exact',
                        'source_file': os.path.basename(filepath),
                        'ranking_asof': fecha_extraccion,
                    }

                    # Prefer latest file for deduplication
                    existing = rows_by_key.get(dedup_key)
                    if existing is None or fecha_extraccion >= existing.get('ranking_asof', ''):
                        rows_by_key[dedup_key] = row

    if verbose:
        print(f'  Archivos procesados: {n_files}')
        print(f'  Filas brutas: {n_rows_raw}')
        print(f'  Filas deduplicadas: {len(rows_by_key)}')

    # Agrupar por slug
    players: dict[str, dict] = {}
    for (slug, fecha_iso, oponente_raw), row in rows_by_key.items():
        if slug not in players:
            rk_info = best_ranking.get(slug, (None, None))
            players[slug] = {
                'slug': slug,
                'own_ranking': rk_info[0],
                'ranking_asof': rk_info[1],
                'rows': [],
            }
        players[slug]['rows'].append(row)

    # Actualizar own_ranking y ranking_asof con la info más reciente
    for slug, pdata in players.items():
        rk_info = best_ranking.get(slug, (None, None))
        pdata['own_ranking'] = rk_info[0]
        pdata['ranking_asof'] = rk_info[1]
        # Ordenar filas por fecha descendente
        pdata['rows'].sort(key=lambda r: r['fecha'], reverse=True)
        # Calcular stats agregadas
        pdata['stats'] = _compute_player_stats(pdata['rows'])

    return players, n_files, n_rows_raw, len(rows_by_key)


def build_index(players: dict) -> dict:
    """Construye player_db_index.json (resumen sin filas crudas)."""
    index = {}
    for slug, pdata in players.items():
        stats = pdata['stats']
        # Win rates por superficie (compacto)
        surface_wr = {
            sup: s['win_rate']
            for sup, s in stats.get('surface_stats', {}).items()
            if s['n'] >= 3
        }
        tier_wr = {
            tier: s['win_rate']
            for tier, s in stats.get('tier_stats', {}).items()
            if s['n'] >= 3
        }
        index[slug] = {
            'slug': slug,
            'own_ranking': pdata.get('own_ranking'),
            'ranking_asof': pdata.get('ranking_asof'),
            'n_matches': stats.get('n_total', 0),
            'surface_win_rates': surface_wr,
            'tier_win_rates': tier_wr,
            'prs_three_set_win_rate': stats.get('prs_stats', {}).get('three_set', {}).get('win_rate'),
            'prs_underdog_win_rate': stats.get('prs_stats', {}).get('underdog', {}).get('win_rate'),
        }
    return index


def main():
    parser = argparse.ArgumentParser(description='Construye PlayerDB desde archivos H2H')
    parser.add_argument('--reports-dir', default='reports/', help='Directorio de archivos H2H')
    parser.add_argument('--out-dir', default='data/', help='Directorio de salida')
    parser.add_argument('--quiet', action='store_true', help='Sin output verbose')
    args = parser.parse_args()

    verbose = not args.quiet
    reports_dir = args.reports_dir
    out_dir = args.out_dir

    h2h_pattern = os.path.join(reports_dir, 'h2h_results_enhanced_*.json')
    h2h_files = sorted(glob.glob(h2h_pattern))

    if not h2h_files:
        print(f'[ERROR] No se encontraron archivos H2H en {reports_dir}')
        sys.exit(1)

    if verbose:
        print(f'[D90-03] build_player_db.py')
        print(f'  Archivos H2H encontrados: {len(h2h_files)}')

    players, n_files, n_raw, n_deduped = process_files(h2h_files, verbose=verbose)

    n_players = len(players)
    built_at = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # Construir player_db.json (con rows crudas para incremental futuro)
    player_db = {
        'version': '1.0',
        'built_at': built_at,
        'n_files_processed': n_files,
        'n_rows_raw': n_raw,
        'n_rows_deduped': n_deduped,
        'n_players': n_players,
        'players': players,
    }

    out_db = os.path.join(out_dir, 'player_db.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(out_db, 'w', encoding='utf-8') as f:
        json.dump(player_db, f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f'  Jugadores: {n_players}')
        print(f'  -> {out_db}')

    # Construir player_db_index.json (compacto, sin rows crudas)
    index = build_index(players)
    index_db = {
        'version': '1.0',
        'built_at': built_at,
        'n_players': n_players,
        'players': index,
    }

    out_index = os.path.join(out_dir, 'player_db_index.json')
    with open(out_index, 'w', encoding='utf-8') as f:
        json.dump(index_db, f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f'  -> {out_index}')
        print(f'[D90-03] DONE — {n_players} jugadores, {n_deduped} filas deduplicadas')

    # Alias table (subproducto: slug → set of opponent names seen)
    alias_table = {}
    for slug, pdata in players.items():
        oponentes = sorted({r['oponente'] for r in pdata['rows']})
        alias_table[slug] = oponentes

    out_alias = os.path.join(out_dir, 'player_alias_table.json')
    with open(out_alias, 'w', encoding='utf-8') as f:
        json.dump({'built_at': built_at, 'aliases': alias_table}, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f'  -> {out_alias} (alias table)')


if __name__ == '__main__':
    main()

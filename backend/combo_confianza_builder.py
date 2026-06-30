#!/usr/bin/env python3
"""
combo_confianza_builder.py — Portfolio de combos con aislamiento de riesgo (Nodo-38).

Lee:    reports/h2h_results_enhanced_FECHA.json  (el más reciente)
        reports/edge_report_FECHA.json           (cross-reference pipeline picks)
Genera: reports/combo_plan_FECHA.txt

Arquitectura CORE / SATELLITE / MOONSHOT:
  CORE:      Cat-A (1.15-1.59) + Cat-B (1.60-2.20) — NUNCA Cat-C
  SATELLITE: 4×Cat-A/B base + 1×Cat-C1 (2.20-3.50, conf≥60%) — aislado
  MOONSHOT:  3×Cat-A + 2-3×Cat-C (conf≥57%) — baja probabilidad, alto payout

Si un pick Cat-C falla, solo su satellite muere. El CORE sobrevive intacto.

Fases de escalado:
  --fase 1: solo CORE (2% bankroll)
  --fase 2: CORE + 1 satellite (4% bankroll)
  --fase 3: CORE + 3 SAT + moonshot (7% bankroll)
  --fase 4: todo + cobertura (12% bankroll)

Uso:
  python combo_confianza_builder.py --bankroll 125000
  python combo_confianza_builder.py --bankroll 125000 --fase 2
  python combo_confianza_builder.py --bankroll 125000 --fase 4 --telegram
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Constantes ─────────────────────────────────────────────────────────────────

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')

DESKTOP_WIN   = Path("/mnt/c/users/hogar/Desktop")
COMBOS_DIR    = DESKTOP_WIN / "combos"
CHROME_WIN    = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BETPLAY_URL_BASE = "https://betplay.com.co/apuestas#home?coupon=combination|"
BETPLAY_URL_TAIL = "||replace"
REDIRECT_BASE = "https://dakotapog.github.io/tennis-analysis/bp/?ids="
TG_TOKEN      = "8684706586:AAHv4zhjQKvxORf6bnbwCxZQPly9OA7unpY"
TG_CHAT       = "8520949513"

# ── Categorización (Nodo-38) ──────────────────────────────────────────────────

# Umbrales de cuota por categoría
CUOTA_MIN       = 1.15   # debajo no aporta odds suficiente al combo
CUOTA_CAT_B     = 1.60   # Cat-A: 1.15-1.59 | Cat-B: 1.60-2.20
CUOTA_CAT_C     = 2.21   # Cat-C empieza aquí
CUOTA_C1_MAX    = 3.50   # Cat-C1: 2.21-3.50 (satellite eligible)
CUOTA_PIPELINE_MAX = 4.50  # pipeline promotion: hasta 4.50
CONF_MIN        = 53.0   # confianza mínima global
CONF_C1         = 60.0   # confianza mínima para Cat-C1 (STRONG)
CONF_C1_PIPELINE = 57.0  # confianza mínima Cat-C1 con señal pipeline
CONF_MOONSHOT   = 57.0   # confianza mínima para moonshot (SILVER+)

# Parejo detection
PAREJO_CONF_MAX  = 55.0
PAREJO_CUOTA_MIN = 1.55
PAREJO_CUOTA_MAX = 1.70

# Portfolio constraints
CORE_MAX_SIZE = 7
CORE_MIN_SIZE = 4
CORE_MIN_PWIN = 0.25          # P(CORE wins) mínimo usando P_mercado
MAX_SAME_TOURNAMENT = 2       # Guard 2 de betplay_combo_builder.py
SAT_BASE_SIZE = 4             # picks Cat-A/B en cada satellite
MAX_SATELLITES = 3
MAX_MOONSHOT_CAT_C = 3
MAX_SINGLE_PICK_EXPOSURE = 0.05  # 5% bankroll máximo por pick

# Budget allocation by phase
PHASE_CONFIG = {
    1: {'max_daily_pct': 0.02, 'core': True,  'satellites': 0, 'moonshot': False, 'cobertura': False},
    2: {'max_daily_pct': 0.04, 'core': True,  'satellites': 1, 'moonshot': False, 'cobertura': False},
    3: {'max_daily_pct': 0.07, 'core': True,  'satellites': 3, 'moonshot': True,  'cobertura': False},
    4: {'max_daily_pct': 0.12, 'core': True,  'satellites': 3, 'moonshot': True,  'cobertura': True},
}

BUDGET_CORE_PCT     = 0.45
BUDGET_SAT_PCT      = 0.15  # per satellite
BUDGET_MOONSHOT_PCT = 0.05
BUDGET_COB_PCT      = 0.05
MAX_COBERTURA_COMBOS = 3
MAX_COBERTURA_COMBOS_EXPANDED = 6


# ── Selección de archivo ────────────────────────────────────────────────────────

def _find_latest_file(pattern: str) -> str | None:
    """Encuentra el archivo más reciente que matchea el pattern en reports/."""
    reports = Path(REPORTS_DIR)
    if not reports.exists():
        return None
    candidates = sorted(
        reports.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _load_partidos(filepath: str) -> list:
    """
    Carga partidos desde h2h_results_enhanced JSON.
    Soporta tanto formato dict {metadata, partidos, ...} como lista directa.
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        partidos = data.get('partidos', [])
        if partidos:
            return partidos
        result = []
        for key, val in data.items():
            if isinstance(val, list):
                result.extend(val)
        return result

    if isinstance(data, list):
        return data

    return []


def _load_pipeline_picks() -> set:
    """Carga nombres de picks del edge_report más reciente (apostar + watchlist)."""
    filepath = _find_latest_file('edge_report_*.json')
    if not filepath:
        return set()
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        picks = set()
        for category in ('apostar', 'watchlist'):
            for entry in data.get(category, []):
                nombre = entry.get('favorito', '').strip()
                if nombre:
                    picks.add(nombre.lower())
        return picks
    except (json.JSONDecodeError, OSError):
        return set()


# ── Categorización de picks (Nodo-38) ────────────────────────────────────────

def _categorizar_pick(cuota: float, confianza: float,
                      pipeline_picks: set | None = None,
                      nombre: str = '',
                      conf_min: float = CONF_MIN,
                      conf_c1: float = CONF_C1) -> dict | None:
    """
    Categoriza un pick para el portfolio Nodo-38.

    Retorna dict con categoria y combos_permitidos, o None si el pick se excluye.
    conf_min / conf_c1 pueden sobrescribirse desde _extract_and_categorize (Nodo-42 grass mode).
    """
    if confianza < conf_min:
        return None
    if cuota < CUOTA_MIN:
        return None

    # Parejo detection: confianza baja + cuota media = ambos dicen "es parejo"
    if confianza < PAREJO_CONF_MAX and PAREJO_CUOTA_MIN <= cuota <= PAREJO_CUOTA_MAX:
        return None

    # Pipeline flag
    pipeline_flag = False
    if pipeline_picks and nombre.lower().strip() in pipeline_picks:
        pipeline_flag = True

    # Cat-A: multiplicadores seguros
    if cuota < CUOTA_CAT_B:
        return {
            'categoria': 'CAT_A',
            'combos_permitidos': ['CORE', 'SATELLITE_BASE', 'MOONSHOT_BASE'],
            'pipeline_flag': pipeline_flag,
        }

    # Cat-B: valor
    if cuota < CUOTA_CAT_C:
        return {
            'categoria': 'CAT_B',
            'combos_permitidos': ['CORE', 'SATELLITE_BASE', 'MOONSHOT_BASE'],
            'pipeline_flag': pipeline_flag,
        }

    # Cat-C: alto valor — subdivisión
    # Protocolo D: señal doble promueve Cat-C2 → Cat-C1
    if pipeline_flag and cuota <= CUOTA_PIPELINE_MAX and confianza >= CONF_C1_PIPELINE:
        return {
            'categoria': 'CAT_C1',
            'combos_permitidos': ['SATELLITE', 'MOONSHOT'],
            'pipeline_flag': True,
        }

    if cuota <= CUOTA_C1_MAX and confianza >= conf_c1:
        return {
            'categoria': 'CAT_C1',
            'combos_permitidos': ['SATELLITE', 'MOONSHOT'],
            'pipeline_flag': pipeline_flag,
        }

    # Todo lo demás >2.20
    return {
        'categoria': 'CAT_C2',
        'combos_permitidos': ['MOONSHOT'],
        'pipeline_flag': pipeline_flag,
    }


def _get_cuota_favorito(partido: dict, favorito: str) -> float:
    """Devuelve la cuota del favorito predicho."""
    j1 = (partido.get('jugador1') or '').strip()
    j2 = (partido.get('jugador2') or '').strip()
    c1 = partido.get('cuota1') or 0.0
    c2 = partido.get('cuota2') or 0.0
    fav = (favorito or '').strip()

    if fav == j1:
        return float(c1) if c1 else 1.0
    if fav == j2:
        return float(c2) if c2 else 1.0

    fav_lower = fav.lower()
    if j1 and fav_lower in j1.lower():
        return float(c1) if c1 else 1.0
    if j2 and fav_lower in j2.lower():
        return float(c2) if c2 else 1.0

    if c1 and c2:
        return float(min(c1, c2))
    return float(c1 or c2 or 1.0)


def _get_rival(partido: dict, favorito: str) -> str:
    j1 = (partido.get('jugador1') or '').strip()
    j2 = (partido.get('jugador2') or '').strip()
    fav_lower = (favorito or '').lower()
    if j1 and fav_lower in j1.lower():
        return j2
    if j2 and fav_lower in j2.lower():
        return j1
    if j1 == favorito:
        return j2
    return j1


def _extract_and_categorize(partidos: list, threshold: float,
                             pipeline_picks: set | None = None,
                             conf_min: float = CONF_MIN,
                             conf_c1: float = CONF_C1,
                             superficie_filter: str | None = None) -> list:
    """
    Extrae y categoriza picks válidos.
    Retorna lista de dicts ordenada por confianza descendente.
    conf_min / conf_c1 sobrescriben los defaults cuando se pasan (Nodo-42 grass mode).
    superficie_filter: si se define, solo incluye partidos cuyo tipo_cancha coincide.
    """
    picks = []
    for partido in partidos:
        # Nodo-42: filtrar pool por superficie antes de evaluar confianza
        if superficie_filter is not None:
            tipo_cancha = (partido.get('tipo_cancha') or '').lower().strip()
            if tipo_cancha != superficie_filter.lower():
                continue

        ra = partido.get('ranking_analysis')
        if not ra:
            continue
        pred = ra.get('prediction')
        if not pred:
            continue

        favorito = pred.get('favored_player')
        confidence = pred.get('confidence')
        if not favorito or confidence is None:
            continue

        conf = float(confidence)
        if conf < threshold:
            continue

        cuota = _get_cuota_favorito(partido, favorito)

        cat = _categorizar_pick(cuota, conf, pipeline_picks, favorito,
                                conf_min=conf_min, conf_c1=conf_c1)
        if cat is None:
            continue

        torneo = (
            partido.get('torneo_completo')
            or partido.get('torneo_nombre')
            or partido.get('torneo')
            or 'Desconocido'
        )

        picks.append({
            'nombre':       favorito,
            'confianza':    conf,
            'cuota':        cuota,
            'p_modelo':     conf / 100.0,
            'torneo':       torneo,
            'rival':        _get_rival(partido, favorito),
            'cat':          cat,
        })

    picks.sort(key=lambda x: x['confianza'], reverse=True)
    return picks


# ── Cálculo de combos ───────────────────────────────────────────────────────────

def _calc_combo(picks_subset: list, stake: float, nombre: str,
                excluidos: list | None = None) -> dict:
    """Calcula métricas de un combo dado un subconjunto de picks."""
    odds_total = 1.0
    nombres = []
    confianzas = []
    cuotas = []
    categorias = []

    for p in picks_subset:
        odds_total *= p['cuota']
        nombres.append(p['nombre'])
        confianzas.append(p['confianza'])
        cuotas.append(p['cuota'])
        categorias.append(p['cat']['categoria'])

    # P(win) estimado usando probabilidad implícita del mercado (conservador)
    p_win_market = 1.0
    for p in picks_subset:
        p_win_market *= min(1.0 / p['cuota'], 0.95)

    retorno = stake * odds_total
    ev = stake * (odds_total * p_win_market - 1)

    return {
        'nombre':            nombre,
        'piernas':           nombres,
        'n_piernas':         len(nombres),
        'confianzas':        confianzas,
        'cuotas':            cuotas,
        'categorias':        categorias,
        'p_win':             round(p_win_market, 6),
        'odds_total':        round(odds_total, 2),
        'stake':             stake,
        'retorno_bruto':     round(retorno, 0),
        'retorno_esperado':  round(retorno * p_win_market, 0),
        'ev':                round(ev, 0),
        'pick_excluido':     excluidos,
    }


# ── Construcción del portfolio (Nodo-38) ────────────────────────────────────────

def _select_core(picks_ab: list, max_size: int = CORE_MAX_SIZE,
                 max_same_tournament: int = MAX_SAME_TOURNAMENT) -> list:
    """
    Selecciona picks para el CORE: top confianza con guard de concentración por torneo.
    Solo acepta Cat-A y Cat-B.
    """
    core = []
    torneo_count = Counter()
    for pick in picks_ab:
        if len(core) >= max_size:
            break
        t = pick.get('torneo', '')
        if torneo_count[t] >= max_same_tournament:
            continue
        core.append(pick)
        torneo_count[t] += 1
    return core


def _validate_core_pwin(core_picks: list) -> list:
    """Reduce el CORE si P(win) < CORE_MIN_PWIN usando P_mercado."""
    if not core_picks:
        return core_picks

    while len(core_picks) > CORE_MIN_SIZE:
        p_win = 1.0
        for p in core_picks:
            p_win *= min(1.0 / p['cuota'], 0.95)
        if p_win >= CORE_MIN_PWIN:
            break
        core_picks.pop()  # quitar el de menor confianza (último, ya ordenado desc)

    return core_picks


def _build_cobertura(pool: list, core_size: int, stake_total: float,
                     max_combos: int = MAX_COBERTURA_COMBOS) -> list:
    """
    Genera combos de cobertura para el CORE.
    Excluye 1 pick de menor confianza del CORE y mete 1 de reserva.
    """
    if len(pool) <= core_size:
        return []

    principal = pool[:core_size]
    reservas = pool[core_size:]

    if not reservas:
        return []

    expanded = max_combos > MAX_COBERTURA_COMBOS
    n_candidatos = min(max_combos, core_size) if expanded else min(3, core_size)
    candidatos = principal[core_size - n_candidatos:]

    combos = []
    ya_vistos = set()

    for excluido in candidatos:
        if len(combos) >= max_combos:
            break
        for reserva in reservas:
            if len(combos) >= max_combos:
                break
            subset = [p for p in principal if p is not excluido] + [reserva]
            subset_sorted = sorted(subset, key=lambda x: x['confianza'], reverse=True)
            if len(subset_sorted) != core_size:
                continue
            key = tuple(p['nombre'] for p in subset_sorted)
            if key in ya_vistos:
                continue
            ya_vistos.add(key)
            combos.append((subset_sorted, [excluido['nombre']]))
            if not expanded:
                break  # normal mode: 1 combo per candidate

    if not combos:
        return []

    stake_each = round(stake_total / len(combos) / 500) * 500
    stake_each = max(500, stake_each)

    result = []
    for i, (subset, excluidos) in enumerate(combos, 1):
        excl_label = '_'.join(e.split()[-1] for e in excluidos)
        name = f"COB{i}_excl_{excl_label}" if expanded else f"COB_excl_{excl_label}"
        result.append(_calc_combo(subset, stake_each, name, excluidos=excluidos))

    return result


def _build_portfolio_v2(picks: list, bankroll: float, fase: int = 4,
                        stake_max: float | None = None) -> dict:
    """
    Construye el plan de combos del día con aislamiento de riesgo.

    Arquitectura:
      CORE:      Cat-A + Cat-B, C4-C7, nunca Cat-C
      SATELLITE: 4×Cat-A/B + 1×Cat-C1, máx 3 (o 1 en fase 2)
      MOONSHOT:  3×Cat-A + 2-3×Cat-C (SILVER+)
      COBERTURA: exclusión sobre el CORE (solo fase 4)
    """
    config = PHASE_CONFIG[fase]
    budget = bankroll * config['max_daily_pct']

    # Separar por categoría
    cat_a  = [p for p in picks if p['cat']['categoria'] == 'CAT_A']
    cat_b  = [p for p in picks if p['cat']['categoria'] == 'CAT_B']
    cat_c1 = [p for p in picks if p['cat']['categoria'] == 'CAT_C1']
    cat_c2 = [p for p in picks if p['cat']['categoria'] == 'CAT_C2']
    cat_ab = sorted(cat_a + cat_b, key=lambda x: x['confianza'], reverse=True)

    plan = {
        'core': None,
        'satellites': [],
        'moonshot': None,
        'cobertura': [],
        'budget': budget,
        'fase': fase,
        'resumen': {},
    }

    # ═══ CORE ═══
    if config['core'] and len(cat_ab) >= CORE_MIN_SIZE:
        core_picks = _select_core(cat_ab)
        core_picks = _validate_core_pwin(core_picks)

        if len(core_picks) >= CORE_MIN_SIZE:
            core_stake = round(budget * BUDGET_CORE_PCT / 500) * 500
            core_stake = max(500, core_stake)
            plan['core'] = _calc_combo(core_picks, core_stake, 'CORE')

    # ═══ SATELLITES ═══
    max_sats = config['satellites']
    if max_sats > 0 and cat_c1:
        # Base para satellites: top Cat-A, o Cat-A+B si no hay suficientes Cat-A
        sat_base_pool = sorted(cat_a, key=lambda x: x['confianza'], reverse=True)
        if len(sat_base_pool) < SAT_BASE_SIZE:
            sat_base_pool = cat_ab[:SAT_BASE_SIZE]
        sat_base = sat_base_pool[:SAT_BASE_SIZE]

        for i, c1_pick in enumerate(cat_c1[:max_sats]):
            sat_picks = sat_base + [c1_pick]
            # Sort by confidence for display consistency
            sat_picks_sorted = sorted(sat_picks, key=lambda x: x['confianza'], reverse=True)
            sat_stake = round(budget * BUDGET_SAT_PCT / 500) * 500
            sat_stake = max(500, sat_stake)
            plan['satellites'].append(
                _calc_combo(sat_picks_sorted, sat_stake, f'SAT_{i+1}')
            )

    # ═══ MOONSHOT ═══
    if config['moonshot']:
        cat_c_all = cat_c1 + cat_c2
        cat_c_silver = [p for p in cat_c_all if p['confianza'] >= CONF_MOONSHOT]

        if len(cat_c_silver) >= 2:
            # Base: top 3 Cat-A (más seguros)
            moon_base = cat_a[:3] if len(cat_a) >= 3 else cat_ab[:3]
            moon_cats = cat_c_silver[:MAX_MOONSHOT_CAT_C]

            # Tournament guard on Cat-C within moonshot
            moon_torneos = Counter()
            moon_cats_filtered = []
            for p in moon_cats:
                t = p.get('torneo', '')
                if moon_torneos[t] < 1:  # max 1 Cat-C por torneo en moonshot
                    moon_cats_filtered.append(p)
                    moon_torneos[t] += 1

            if len(moon_cats_filtered) >= 2:
                moon_picks = sorted(
                    moon_base + moon_cats_filtered,
                    key=lambda x: x['confianza'], reverse=True
                )
                moon_stake = round(budget * BUDGET_MOONSHOT_PCT / 500) * 500
                moon_stake = max(500, moon_stake)
                plan['moonshot'] = _calc_combo(moon_picks, moon_stake, 'MOONSHOT')

    # ═══ COBERTURA ═══
    cobertura_expanded = (not plan['satellites'] and not plan.get('moonshot'))
    plan['cobertura_expanded'] = cobertura_expanded

    if config['cobertura'] and plan['core']:
        core_picks = _select_core(cat_ab)
        core_picks = _validate_core_pwin(core_picks)
        # Pool ampliado: CORE picks + reservas
        reserva = [p for p in cat_ab if p not in core_picks]

        if cobertura_expanded:
            # Nodo-38B: no Cat-C → redistribute SAT+MOON budget to cobertura
            core_stake = plan['core']['stake']
            cob_stake = round((budget - core_stake) / 500) * 500
            cob_stake = max(500, cob_stake)
            max_cob = min(MAX_COBERTURA_COMBOS_EXPANDED,
                          len(reserva) + min(3, len(core_picks)))
            pool_ampliado = core_picks + reserva[:max_cob]
        else:
            cob_stake = round(budget * BUDGET_COB_PCT / 500) * 500
            cob_stake = max(500, cob_stake)
            max_cob = MAX_COBERTURA_COMBOS
            pool_ampliado = core_picks + reserva[:3]

        plan['cobertura'] = _build_cobertura(
            pool_ampliado, len(core_picks), cob_stake, max_combos=max_cob
        )

    # ═══ VaR GUARD ═══
    total = _total_stakes(plan)
    if total > budget:
        factor = budget / total
        _scale_stakes(plan, factor)

    # ═══ GRASS BOOTSTRAP STAKE CAP (Nodo-42) ═══
    # stake_max fuerza un cap duro por combo — no negociable en grass mode
    if stake_max is not None:
        _cap_stakes(plan, stake_max)

    # ═══ RESUMEN ═══
    plan['resumen'] = _resumen_portfolio(plan)

    return plan


def _total_stakes(plan: dict) -> float:
    total = 0.0
    if plan.get('core'):
        total += plan['core']['stake']
    for sat in plan.get('satellites', []):
        total += sat['stake']
    if plan.get('moonshot'):
        total += plan['moonshot']['stake']
    for cob in plan.get('cobertura', []):
        total += cob['stake']
    return total


def _scale_stakes(plan: dict, factor: float):
    """Escala todos los stakes proporcionalmente."""
    def scale(combo):
        combo['stake'] = max(500, round(combo['stake'] * factor / 500) * 500)
        combo['retorno_bruto'] = round(combo['stake'] * combo['odds_total'], 0)
        combo['retorno_esperado'] = round(combo['retorno_bruto'] * combo['p_win'], 0)
        combo['ev'] = round(combo['stake'] * (combo['odds_total'] * combo['p_win'] - 1), 0)

    if plan.get('core'):
        scale(plan['core'])
    for sat in plan.get('satellites', []):
        scale(sat)
    if plan.get('moonshot'):
        scale(plan['moonshot'])
    for cob in plan.get('cobertura', []):
        scale(cob)


def _cap_stakes(plan: dict, stake_max: float):
    """Cap duro por combo — usado en grass bootstrap (Nodo-42). No redondea a 500."""
    def cap(combo):
        if combo['stake'] > stake_max:
            combo['stake'] = stake_max
            combo['retorno_bruto'] = round(combo['stake'] * combo['odds_total'], 0)
            combo['retorno_esperado'] = round(combo['retorno_bruto'] * combo['p_win'], 0)
            combo['ev'] = round(combo['stake'] * (combo['odds_total'] * combo['p_win'] - 1), 0)

    if plan.get('core'):
        cap(plan['core'])
    for sat in plan.get('satellites', []):
        cap(sat)
    if plan.get('moonshot'):
        cap(plan['moonshot'])
    for cob in plan.get('cobertura', []):
        cap(cob)


def _resumen_portfolio(plan: dict) -> dict:
    total_invertido = 0.0
    total_retorno_esp = 0.0
    total_ev = 0.0
    n_combos = 0

    for combo in _all_combos(plan):
        total_invertido += combo['stake']
        total_retorno_esp += combo['retorno_esperado']
        total_ev += combo['ev']
        n_combos += 1

    return {
        'total_invertido':   total_invertido,
        'total_retorno_esp': total_retorno_esp,
        'total_ev':          total_ev,
        'n_combos':          n_combos,
        'budget':            plan.get('budget', 0),
        'fase':              plan.get('fase', 0),
    }


def _all_combos(plan: dict):
    """Genera todos los combos del plan para iteración."""
    if plan.get('core'):
        yield plan['core']
    for sat in plan.get('satellites', []):
        yield sat
    if plan.get('moonshot'):
        yield plan['moonshot']
    for cob in plan.get('cobertura', []):
        yield cob


# ── Formateo de output ──────────────────────────────────────────────────────────

def _format_report(picks: list, plan: dict, threshold: float,
                   filepath: str, grass_mode: bool = False) -> str:
    today = datetime.now().strftime('%Y-%m-%d')
    lineas = []

    def sep(char='=', n=70):
        lineas.append(char * n)

    def add(txt=''):
        lineas.append(txt)

    fase = plan.get('fase', 4)
    budget = plan.get('budget', 0)

    sep()
    add(f'COMBO PLAN (Nodo-38) — {today}  |  Fase {fase}  |  Conf min: {threshold}%')
    if grass_mode:
        add(f'[GRASS BOOTSTRAP — umbral reducido {threshold}%, stake cap $500, Nodo-42]')
    add(f'Fuente: {os.path.basename(filepath)}')
    add(f'Budget diario: ${budget:,.0f}  |  Picks seleccionados: {len(picks)}')
    sep()
    add()

    # ── Picks por categoría ───────────────────────────────────────────────────
    for cat_name, cat_desc in [('CAT_A', 'MULTIPLICADORES (1.15-1.59)'),
                                ('CAT_B', 'VALOR (1.60-2.20)'),
                                ('CAT_C1', 'ALTO VALOR SATELLITE (2.21-3.50, conf>=60%)'),
                                ('CAT_C2', 'ALTO VALOR MOONSHOT (>3.50 o conf<60%)')]:
        cat_picks = [p for p in picks if p['cat']['categoria'] == cat_name]
        if not cat_picks:
            continue
        add(f'{cat_desc}: {len(cat_picks)} picks')
        for i, p in enumerate(cat_picks, 1):
            pipeline_str = ' [PIPELINE]' if p['cat'].get('pipeline_flag') else ''
            torneo_corto = (p['torneo'] or '')[:40]
            add(f'  {i:2d}. {p["nombre"]:28s}  @{p["cuota"]:.2f}  '
                f'conf:{p["confianza"]:.1f}%  {torneo_corto}{pipeline_str}')
        add()

    if not any([plan.get('core'), plan.get('satellites'), plan.get('moonshot')]):
        add('Sin picks suficientes para construir combos.')
        sep()
        return '\n'.join(lineas)

    # ── CORE ──────────────────────────────────────────────────────────────────
    if plan.get('core'):
        sep('-', 70)
        add()
        core = plan['core']
        add(f'CORE — {core["n_piernas"]} piernas (solo Cat-A + Cat-B)')
        piernas_str = ', '.join(
            f'{n} @{q:.2f}' for n, q in zip(core['piernas'], core['cuotas'])
        )
        add(f'  Picks: {piernas_str}')
        add(f'  Odds: {core["odds_total"]:.2f}x  |  '
            f'P(win): {core["p_win"]*100:.1f}%  |  '
            f'EV: ${core["ev"]:+,.0f}')
        add(f'  STAKE: ${core["stake"]:,.0f}  →  '
            f'Retorno: ${core["retorno_bruto"]:,.0f}  |  '
            f'Esperado: ${core["retorno_esperado"]:,.0f}')

    # ── SATELLITES ────────────────────────────────────────────────────────────
    if plan.get('satellites'):
        add()
        sep('-', 70)
        add()
        add(f'SATELLITES — {len(plan["satellites"])} combos aislados (Cat-A/B + 1 Cat-C)')
        for sat in plan['satellites']:
            add()
            cat_c_pick = None
            for n, c in zip(sat['piernas'], sat['categorias']):
                if c in ('CAT_C1', 'CAT_C2'):
                    cat_c_pick = n
                    break
            add(f'  [{sat["nombre"]}]  Cat-C: {cat_c_pick or "?"}')
            piernas_str = ', '.join(
                f'{n} @{q:.2f}' for n, q in zip(sat['piernas'], sat['cuotas'])
            )
            add(f'  Picks: {piernas_str}')
            add(f'  Odds: {sat["odds_total"]:.2f}x  |  '
                f'P(win): {sat["p_win"]*100:.1f}%  |  '
                f'EV: ${sat["ev"]:+,.0f}')
            add(f'  STAKE: ${sat["stake"]:,.0f}  →  '
                f'Retorno: ${sat["retorno_bruto"]:,.0f}')
        add()
        add('  Si el Cat-C falla, solo ESE satellite muere. CORE sobrevive.')

    # ── MOONSHOT ──────────────────────────────────────────────────────────────
    if plan.get('moonshot'):
        add()
        sep('-', 70)
        add()
        moon = plan['moonshot']
        cat_c_names = [
            n for n, c in zip(moon['piernas'], moon['categorias'])
            if c in ('CAT_C1', 'CAT_C2')
        ]
        add(f'MOONSHOT — {moon["n_piernas"]} piernas '
            f'({len(cat_c_names)} Cat-C: {", ".join(cat_c_names)})')
        piernas_str = ', '.join(
            f'{n} @{q:.2f}' for n, q in zip(moon['piernas'], moon['cuotas'])
        )
        add(f'  Picks: {piernas_str}')
        add(f'  Odds: {moon["odds_total"]:.2f}x  |  '
            f'P(win): {moon["p_win"]*100:.1f}%  |  '
            f'EV: ${moon["ev"]:+,.0f}')
        add(f'  STAKE: ${moon["stake"]:,.0f}  →  '
            f'Retorno: ${moon["retorno_bruto"]:,.0f}')

    # ── COBERTURA ─────────────────────────────────────────────────────────────
    if plan.get('cobertura'):
        add()
        sep('-', 70)
        add()
        if plan.get('cobertura_expanded'):
            add(f'COBERTURA EXPANDIDA (sin Cat-C → budget SAT+MOON redistribuido) '
                f'— {len(plan["cobertura"])} combos')
        else:
            add(f'COBERTURA CORE — {len(plan["cobertura"])} combos')
        for cob in plan['cobertura']:
            excl_str = ', '.join(cob.get('pick_excluido') or [])
            add(f'  [{cob["nombre"]}]  Excluye: {excl_str}')
            add(f'    Odds: {cob["odds_total"]:.2f}x  P(win): {cob["p_win"]*100:.1f}%  '
                f'STAKE: ${cob["stake"]:,.0f}')

    # ── RESUMEN ───────────────────────────────────────────────────────────────
    add()
    sep('=', 70)
    res = plan['resumen']
    add(f'RESUMEN PORTFOLIO — Fase {fase}')
    add(f'  Total combos:           {res["n_combos"]}')
    add(f'  Total invertido:        ${res["total_invertido"]:,.0f}  '
        f'(de ${res["budget"]:,.0f} budget)')
    add(f'  Retorno esperado:       ${res["total_retorno_esp"]:,.0f}')
    add(f'  EV total:               ${res["total_ev"]:+,.0f}')
    add()

    # ── Notas ─────────────────────────────────────────────────────────────────
    add('NOTAS:')
    add('  - Cat-C NUNCA entra al CORE (REGLA-ISO-1)')
    add('  - Max 1 Cat-C por satellite, max 2 picks mismo torneo por combo')
    add('  - P(win) calculado con P_mercado (1/cuota) — conservador')
    add('  - VaR guard: stakes escalados si total > budget diario')
    if fase < 4:
        add(f'  - Fase {fase}: funcionalidad limitada. '
            f'Escalar a fase {fase+1} cuando se cumplan gates.')
    sep()

    return '\n'.join(lineas)


# ── Kambi lookup + .bat generation ─────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    import unicodedata
    import re
    name = unicodedata.normalize("NFD", name.lower())
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


def _fetch_kambi_outcomes() -> dict:
    """Devuelve dict nombre_normalizado → {outcome_id, odds, jugador}."""
    try:
        import requests
        from scraping.kambi_tennis import KAMBI_BASE, KAMBI_PARAMS, KAMBI_HEADERS
    except ImportError:
        return {}
    try:
        url = f"{KAMBI_BASE}/listView/tennis.json?{KAMBI_PARAMS}"
        resp = requests.get(url, headers=KAMBI_HEADERS, timeout=15)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception:
        return {}

    outcomes_map = {}
    for ev in events:
        offers = ev.get("betOffers", [])
        for offer in offers:
            label_crit = offer.get("criterion", {}).get("label", "")
            if label_crit not in ("Match", "Cuotas del partido"):
                continue
            for oc in offer.get("outcomes", []):
                outcome_id = oc.get("id")
                label = oc.get("label") or oc.get("participant") or ""
                odds = round((oc.get("odds", 0) / 1000), 2)
                if not outcome_id or not label:
                    continue
                norm = _normalize_name(label)
                apellido = norm.split()[-1] if norm.split() else norm
                entry = {"outcome_id": str(outcome_id), "odds": odds, "jugador": label}
                outcomes_map[norm] = entry
                if apellido not in outcomes_map:
                    outcomes_map[apellido] = entry
    return outcomes_map


def _find_outcome(nombre: str, cuota: float, outcomes_map: dict):
    """Busca outcome_id para un pick. Retorna dict o None."""
    norm = _normalize_name(nombre)
    apellido = norm.split()[-1] if norm.split() else norm

    for key in [norm, apellido]:
        if key in outcomes_map:
            return outcomes_map[key]

    for key, oc in outcomes_map.items():
        if apellido and len(apellido) > 3 and apellido in key:
            return oc
    return None


def _generar_bats(plan: dict, prefix: str = "CC") -> int:
    """Genera .bat en escritorio para cada combo principal del portfolio."""
    outcomes_map = _fetch_kambi_outcomes()
    if not outcomes_map:
        print("  WARN: No se pudo conectar a Kambi — sin .bat")
        return 0

    COMBOS_DIR.mkdir(parents=True, exist_ok=True)

    for old in DESKTOP_WIN.glob(f"{prefix}*.bat"):
        old.unlink(missing_ok=True)
    for old in COMBOS_DIR.glob(f"{prefix.lower()}*.html"):
        old.unlink(missing_ok=True)

    generated = 0
    idx = 0

    for combo in _all_combos(plan):
        if combo.get('pick_excluido') and not plan.get('cobertura_expanded'):
            continue  # skip cobertura for .bat only in normal mode
        idx += 1
        piernas = combo['piernas']
        cuotas_list = combo['cuotas']

        outcome_ids = []
        legs_str_parts = []
        ok = True
        for nombre, cuota in zip(piernas, cuotas_list):
            oc = _find_outcome(nombre, cuota, outcomes_map)
            if oc:
                outcome_ids.append(oc["outcome_id"])
                legs_str_parts.append(f"{nombre}@{cuota:.2f}")
            else:
                ok = False
                print(f"  WARN: Sin outcome para {nombre} @{cuota} — {combo['nombre']} omitido")
                break

        if not ok or len(outcome_ids) < 2:
            continue

        ids_str  = ",".join(outcome_ids)
        url      = f"{BETPLAY_URL_BASE}{ids_str}{BETPLAY_URL_TAIL}"
        legs_str = " + ".join(legs_str_parts)

        html_content = (
            f"<html><head><title>{prefix}{idx}</title></head><body>\n"
            f'<script>window.location.replace("{url}");</script>\n'
            f"<p>Redirigiendo... {combo['nombre']}: {legs_str}</p>\n"
            f"</body></html>"
        )
        html_path = COMBOS_DIR / f"{prefix.lower()}{idx}.html"
        html_path.write_text(html_content, encoding="utf-8")

        html_win  = f"C:\\users\\hogar\\Desktop\\combos\\{prefix.lower()}{idx}.html"
        bat_path  = DESKTOP_WIN / f"{prefix}{idx}.bat"
        bat_path.write_text(
            f"@echo off\r\nstart \"\" \"{CHROME_WIN}\" \"file:///{html_win}\"\r\n",
            encoding="utf-8"
        )

        print(f"  {prefix}{idx}.bat — {combo['nombre']}: {legs_str}")
        generated += 1

    # ── Escribir betslip_index para betslip_registrar.py (Nodo-42) ──────────
    # Construir índice outcome_id → pick_info desde outcomes_map resueltos
    betslip_index: dict = {}
    for combo in _all_combos(plan):
        if combo.get('pick_excluido') and not plan.get('cobertura_expanded'):
            continue
        for nombre, cuota in zip(combo['piernas'], combo['cuotas']):
            oc = _find_outcome(nombre, cuota, outcomes_map)
            if oc:
                oid = str(oc['outcome_id'])
                if oid not in betslip_index:
                    betslip_index[oid] = {
                        'jugador':    nombre,
                        'cuota':      cuota,
                        'cuota_kambi': oc.get('odds', cuota),
                        'partido':    '',
                        'match_id':   '',
                        'match_url':  '',
                        'torneo':     'Wimbledon' if prefix.startswith('CC_GRASS') else '',
                        'superficie': 'grass' if prefix.startswith('CC_GRASS') else '?',
                        'tier':       'grand_slam' if prefix.startswith('CC_GRASS') else '?',
                        'edge':       '0%',
                        'p_modelo':   0.5,
                        'kelly_kl':   0.0,
                    }

    if betslip_index:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        idx_path = os.path.join(REPORTS_DIR, f'betslip_index_{ts}.json')
        with open(idx_path, 'w', encoding='utf-8') as f:
            json.dump({
                'ts':      datetime.now().isoformat(),
                'modo':    f'COMBO_CONFIANZA_{prefix}',
                'n_picks': len(betslip_index),
                'index':   betslip_index,
            }, f, ensure_ascii=False, indent=2)
        print(f'  betslip_index guardado: {os.path.basename(idx_path)} ({len(betslip_index)} picks)')

    return generated


def _enviar_telegram(plan: dict):
    """Envía combos del portfolio a Telegram con links de redirect."""
    try:
        import requests
    except ImportError:
        return

    outcomes_map = _fetch_kambi_outcomes()

    msgs = []
    for combo in _all_combos(plan):
        if combo.get('pick_excluido') and not plan.get('cobertura_expanded'):
            continue
        piernas = combo['piernas']
        cuotas_list = combo['cuotas']
        p_win = combo['p_win'] * 100

        outcome_ids = []
        for nombre, cuota in zip(piernas, cuotas_list):
            oc = _find_outcome(nombre, cuota, outcomes_map)
            if oc:
                outcome_ids.append(oc["outcome_id"])

        if len(outcome_ids) < 2:
            continue

        ids_str      = ",".join(outcome_ids)
        redirect_url = f"{REDIRECT_BASE}{ids_str}"
        legs_txt     = "\n".join(f"  - {n} @{c:.2f}" for n, c in zip(piernas, cuotas_list))

        msgs.append(
            f"<b>{combo['nombre']}</b> | P(win): {p_win:.1f}% | @{combo['odds_total']:.2f}x\n"
            f"{legs_txt}\n"
            f"Stake: ${combo['stake']:,.0f} -> ${combo['retorno_bruto']:,.0f}\n"
            f'<a href="{redirect_url}">ABRIR {combo["nombre"]}</a>'
        )

    if not msgs:
        return

    fase = plan.get('fase', '?')
    header = (f"<b>COMBO CONFIANZA Nodo-38 — {datetime.now().strftime('%Y-%m-%d')}</b>\n"
              f"Fase {fase} | {len(msgs)} combos\n\n")
    body   = "\n\n---\n\n".join(msgs)
    payload = {
        "chat_id":    TG_CHAT,
        "text":       header + body,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json=payload, timeout=10
        )
        if r.status_code == 200:
            print("  Telegram enviado")
        else:
            print(f"  WARN Telegram: {r.status_code} {r.text[:80]}")
    except Exception as e:
        print(f"  WARN Telegram: {e}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Portfolio de combos con aislamiento de riesgo (Nodo-38).'
    )
    parser.add_argument('--threshold',   type=float, default=None,
                        help='Confianza minima para incluir un pick (default: 53, o 50 en --superficie grass)')
    parser.add_argument('--bankroll',    type=float, default=125000.0,
                        help='Bankroll total en pesos/USD (default: 125000)')
    parser.add_argument('--fase',        type=int,   default=4, choices=[1, 2, 3, 4],
                        help='Fase de escalado: 1=CORE only, 2=+1 SAT, 3=+3 SAT+MOON, 4=todo')
    parser.add_argument('--file',        type=str,   default=None,
                        help='Ruta explicita al h2h_results_enhanced JSON')
    parser.add_argument('--superficie',  type=str,   default=None,
                        choices=['grass', 'clay', 'hard'],
                        help='Superficie: grass activa modo bootstrap (Nodo-42) con umbral=50 y stake cap $500')
    parser.add_argument('--telegram',    action='store_true',
                        help='Enviar combos a Telegram con links de redirect')
    parser.add_argument('--no-bat',      action='store_true',
                        help='No generar archivos .bat en el escritorio')
    args = parser.parse_args()

    # ── Grass bootstrap mode (Nodo-42) ───────────────────────────────���────────
    grass_mode = args.superficie == 'grass'
    if grass_mode:
        conf_min_efectivo = 50.0
        conf_c1_efectivo  = 55.0
        stake_max         = 500.0
        var_budget_grass  = args.bankroll * 0.01   # 1% bankroll máximo en grass bootstrap
    else:
        conf_min_efectivo = CONF_MIN
        conf_c1_efectivo  = CONF_C1
        stake_max         = None
        var_budget_grass  = None

    # --threshold sobrescribe el default (explícito tiene prioridad)
    threshold = args.threshold if args.threshold is not None else conf_min_efectivo

    # Seleccionar archivo
    if args.file:
        filepath = args.file
        if not os.path.exists(filepath):
            print(f'ERROR: No se encontro el archivo: {filepath}')
            sys.exit(1)
    else:
        filepath = _find_latest_file('h2h_results_enhanced_*.json')
        if not filepath:
            print(f'ERROR: No se encontraron archivos h2h_results_enhanced en {REPORTS_DIR}')
            sys.exit(1)

    print(f'Leyendo: {filepath}')

    # Cargar datos
    try:
        partidos = _load_partidos(filepath)
    except (json.JSONDecodeError, OSError) as e:
        print(f'ERROR leyendo archivo: {e}')
        sys.exit(1)

    if not partidos:
        print('ERROR: No se encontraron partidos en el archivo.')
        sys.exit(1)

    print(f'Partidos cargados: {len(partidos)}')

    # Cargar picks pipeline para cross-reference
    pipeline_picks = _load_pipeline_picks()
    if pipeline_picks:
        print(f'Pipeline picks (edge_report): {len(pipeline_picks)}')

    if grass_mode:
        print(f'[GRASS BOOTSTRAP] umbral={threshold}% | conf_c1={conf_c1_efectivo}% | stake_max=${stake_max:.0f} | VaR_max=${var_budget_grass:,.0f}')

    # Extraer y categorizar picks — grass mode filtra por tipo_cancha (Nodo-42)
    superficie_filter = args.superficie if grass_mode else None
    picks = _extract_and_categorize(partidos, threshold, pipeline_picks,
                                     conf_min=conf_min_efectivo, conf_c1=conf_c1_efectivo,
                                     superficie_filter=superficie_filter)

    if not picks:
        print(f'Sin picks con confianza >= {threshold}%. '
              f'Revisa el threshold o los datos.')
        sys.exit(0)

    # Resumen de categorías
    cat_counts = Counter(p['cat']['categoria'] for p in picks)
    print(f'Picks categorizados: {len(picks)} '
          f'(A:{cat_counts.get("CAT_A", 0)} B:{cat_counts.get("CAT_B", 0)} '
          f'C1:{cat_counts.get("CAT_C1", 0)} C2:{cat_counts.get("CAT_C2", 0)})')

    # Construir portfolio
    plan = _build_portfolio_v2(picks, args.bankroll, args.fase, stake_max=stake_max)

    # Grass VaR guard: si total > 1% bankroll, escalar
    if grass_mode and var_budget_grass is not None:
        total_grass = _total_stakes(plan)
        if total_grass > var_budget_grass:
            _scale_stakes(plan, var_budget_grass / total_grass)
            _cap_stakes(plan, stake_max)  # re-aplicar cap tras escalar

    # Generar reporte
    report = _format_report(picks, plan, threshold, filepath, grass_mode=grass_mode)

    print()
    print(report)

    # Guardar en archivo
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fecha = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = '_grass' if grass_mode else ''
    out_path = os.path.join(REPORTS_DIR, f'combo_plan{suffix}_{fecha}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nGuardado en: {out_path}')

    # Generar .bat en escritorio
    has_combos = any([plan.get('core'), plan.get('satellites'), plan.get('moonshot'),
                      plan.get('cobertura_expanded') and plan.get('cobertura')])
    if not args.no_bat and has_combos:
        bat_prefix = 'CC_GRASS_' if grass_mode else 'CC'
        print('\nGenerando .bat en escritorio...')
        n_bat = _generar_bats(plan, prefix=bat_prefix)
        if n_bat:
            print(f'  {n_bat} archivos {bat_prefix}*.bat en escritorio')
        else:
            print('  Sin .bat generados (picks no disponibles en Kambi)')

    # Telegram
    if args.telegram and has_combos:
        print('\nEnviando a Telegram...')
        _enviar_telegram(plan)


if __name__ == '__main__':
    main()

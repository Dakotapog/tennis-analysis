#!/usr/bin/env python3
"""
trader_ev_tenis.py — Capa trader sobre edge_calculator outputs (análogo NBA trader_ev.py)

Lee:    reports/edge_report_FECHA.json  (producido por edge_calculator.py)
Genera:
  1. Individuales  — Kelly-KL stake por señal APOSTAR (cap 10% bankroll)
  2. Combos        — parlays N piernas, cuota = ∏cuotas, HR = ∏p_modelo
  3. Sistema 2/N   — ganas si ≥2 de N aciertos (binomial coverage)
  4. Budget cascade — 40% individuales / 40% combos / 20% sistema

Uso:
  python trader_ev_tenis.py --bankroll 100000
  python trader_ev_tenis.py --bankroll 100000 --combos 3 --sistema 4 --ncombos 2 --nsistema 3
  python trader_ev_tenis.py --bankroll 100000 --file reports/edge_report_20260530_072115.json
"""

import argparse
import json
import os
import math
from datetime import datetime
from itertools import combinations

# ── Constantes ────────────────────────────────────────────────────────────────

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')

EDGE_MIN          = 0.05    # umbral mínimo para señal APOSTAR
KELLY_CAP_IND     = 0.10    # cap por apuesta individual (10%)
KELLY_CAP_COMBO   = 0.15    # cap por combo (15%)
KELLY_FRACTION    = 0.25    # Kelly fraccionario ×0.25 (conservador)
BUDGET_IND_PCT    = 0.40    # 40% bankroll para individuales
BUDGET_COMBO_PCT  = 0.40    # 40% para combos
BUDGET_SIS_PCT    = 0.20    # 20% para sistema
MIN_BET           = 1000    # apuesta mínima en COP/USD (ajustar)

# Bayesian prior: k=3 equivale a "3 partidos de referencia" histórica (0.52)
# Cuando n_h2h=5: blend = 62.5% p_modelo + 37.5% prior
# Cuando n_h2h=20: blend = 87% p_modelo + 13% prior
_K_PRIOR     = 3
_P_PRIOR     = 0.52   # prior neutral — se calibra con validar_con_api.py


# ── Estadísticos ──────────────────────────────────────────────────────────────

def _kelly_quarter(hr: float, cuota: float) -> float:
    """Kelly fraccionario ×0.25."""
    if cuota <= 1.0:
        return 0.0
    ev = hr * cuota - 1.0
    if ev <= 0:
        return 0.0
    k = ev / (cuota - 1.0)
    return round(k * KELLY_FRACTION, 4)


def _ev(hr: float, cuota: float) -> float:
    return round(hr * cuota - 1.0, 4)


def _p_blend(p_modelo: float, n_h2h: int = 0) -> float:
    """
    Bayesian blend: (n_h2h × p_modelo + K × prior) / (n_h2h + K)
    Cuando n_h2h=0: usa solo el prior (señal sin historial directo)
    Cuando n_h2h grande: converge a p_modelo
    """
    return (n_h2h * p_modelo + _K_PRIOR * _P_PRIOR) / (n_h2h + _K_PRIOR)


def _binom_prob_at_least_k(n: int, k: int, p: float) -> float:
    """P(X ≥ k) para X ~ Binomial(n, p) asumiendo legs independientes."""
    prob = 0.0
    for i in range(k, n + 1):
        c = math.comb(n, i)
        prob += c * (p ** i) * ((1 - p) ** (n - i))
    return prob


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_latest_edge_report(file_override: str = None) -> dict:
    if file_override and os.path.exists(file_override):
        with open(file_override, encoding='utf-8') as f:
            return json.load(f)
    files = sorted([
        f for f in os.listdir(REPORTS_DIR)
        if f.startswith('edge_report_') and f.endswith('.json')
    ])
    if not files:
        raise FileNotFoundError("No se encontró ningún edge_report_*.json en reports/")
    path = os.path.join(REPORTS_DIR, files[-1])
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ── Individuales ──────────────────────────────────────────────────────────────

def _print_individuales(senales: list, bankroll: float, budget: float) -> tuple:
    """
    Imprime tabla de individuales. Retorna (total_gastado, lista_enriched).
    """
    print()
    print("  ┌─ INDIVIDUALES " + "─" * 50)
    print(f"  │  Budget: ${budget:,.0f} ({BUDGET_IND_PCT*100:.0f}% bankroll)")
    print()

    gastado = 0.0
    enriched = []
    for s in senales:
        p_mod   = s['p_modelo']
        n_h2h   = s.get('n_h2h', 0)
        p_b     = _p_blend(p_mod, n_h2h)
        cuota   = s['cuota_favorito']
        kelly   = min(KELLY_CAP_IND, _kelly_quarter(p_b, cuota))
        stake   = round(min(kelly * bankroll, budget - gastado) / MIN_BET) * MIN_BET
        stake   = max(MIN_BET, stake)
        ev_ind  = _ev(p_b, cuota)
        retorno = round(stake * cuota, 0)

        print(f"  │  🎾 {s['partido']}")
        print(f"  │     Apostar: {s['favorito_predicho']}")
        print(f"  │     Cuota {cuota:.2f}  │  Edge {s['edge_pct']}  │  p_blend {p_b:.3f} (n_h2h={n_h2h})")
        print(f"  │     Kelly-KL {s['kelly_kl']:.1%} → stake ${stake:,.0f}  │  Retorno potencial ${retorno:,.0f}")
        print(f"  │     EV real {ev_ind:+.1%}  │  Zona: {s['zona_cuota']}  │  Sup: {s['superficie']}")
        print()

        gastado += stake
        enriched.append({**s, 'stake': stake, 'p_blend': p_b, 'kelly_usado': kelly})

    print(f"  │  Total individuales: ${gastado:,.0f}")
    print("  └" + "─" * 65)
    return gastado, enriched


# ── Combos ────────────────────────────────────────────────────────────────────

def _build_combos(senales: list, n_lines: int, n_combos: int,
                  bankroll: float, budget: float) -> tuple:
    """
    Genera las mejores n_combos combinadas de n_lines piernas.
    Retorna total gastado en combos.
    """
    if len(senales) < n_lines:
        print(f"\n  ⚠️  Solo {len(senales)} señales — se necesitan ≥{n_lines} para combos de {n_lines} piernas.")
        return 0.0, []

    print()
    print("  ┌─ COMBOS " + "─" * 56)
    print(f"  │  Budget: ${budget:,.0f} ({BUDGET_COMBO_PCT*100:.0f}% bankroll) │ {n_lines} piernas │ Top {n_combos}")
    print()

    # Generar todos los combos posibles
    candidates = []
    for combo in combinations(senales, n_lines):
        cuota_combo = 1.0
        hr_combo    = 1.0
        edge_sum    = 0.0
        for leg in combo:
            cuota_combo *= leg['cuota_favorito']
            hr_combo    *= leg['p_blend']
            edge_sum    += leg['edge']
        ev_combo    = _ev(hr_combo, cuota_combo)
        kelly_combo = min(KELLY_CAP_COMBO, _kelly_quarter(hr_combo, cuota_combo))
        candidates.append({
            'legs':        combo,
            'cuota_combo': round(cuota_combo, 2),
            'hr_combo':    round(hr_combo, 4),
            'ev_combo':    ev_combo,
            'kelly_combo': kelly_combo,
            'edge_sum':    round(edge_sum, 4),
        })

    # Ordenar por EV desc
    candidates.sort(key=lambda x: x['ev_combo'], reverse=True)

    gastado = 0.0
    combos_plan = []
    for i, c in enumerate(candidates[:n_combos]):
        stake   = round(min(c['kelly_combo'] * bankroll, budget / n_combos) / MIN_BET) * MIN_BET
        stake   = max(MIN_BET, stake)
        retorno = round(stake * c['cuota_combo'], 0)

        print(f"  │  🔗 COMBO {i+1} ({n_lines} piernas):")
        for leg in c['legs']:
            print(f"  │     • {leg['favorito_predicho']} @ {leg['cuota_favorito']:.2f}  [{leg['partido']}]")
        print(f"  │     Cuota combinada: {c['cuota_combo']:.2f}  │  HR conjunta: {c['hr_combo']:.1%}")
        print(f"  │     EV: {c['ev_combo']:+.1%}  │  Stake: ${stake:,.0f}  │  Retorno potencial: ${retorno:,.0f}")
        print()
        gastado += stake
        combos_plan.append({
            'piernas': [{'partido': l['partido'], 'favorito': l['favorito_predicho'],
                         'cuota': l['cuota_favorito'], 'p_blend': l['p_blend']} for l in c['legs']],
            'cuota_combo': c['cuota_combo'], 'hr_combo': c['hr_combo'],
            'ev': c['ev_combo'], 'stake': stake, 'retorno_potencial': retorno,
        })

    print(f"  │  Total combos: ${gastado:,.0f}")
    print("  └" + "─" * 65)
    return gastado, combos_plan


# ── Sistema 2/N ───────────────────────────────────────────────────────────────

def _build_sistema(senales: list, n_sistema: int, n_sistema_apostar: int,
                   bankroll: float, budget: float) -> tuple:
    """
    Sistema N piernas: ganas si ≥2 aciertos.
    Divide el budget entre las C(n,2) subcombinas de 2 piernas.
    """
    if len(senales) < n_sistema:
        print(f"\n  ⚠️  Solo {len(senales)} señales — se necesitan ≥{n_sistema} para sistema {n_sistema}.")
        return 0.0, []

    legs = senales[:n_sistema]

    # P(≥2 aciertos de N) asumiendo legs independientes con p_blend promedio
    p_avg    = sum(l['p_blend'] for l in legs) / len(legs)
    p_win    = _binom_prob_at_least_k(n_sistema, 2, p_avg)

    # EV del sistema: más complejo — usar aproximación
    # Retorno mínimo (≥2): suma de cuotas de pares ganadores × stake_por_par
    pares        = list(combinations(legs, 2))
    n_pares      = len(pares)
    stake_por_par = round(budget / n_pares / MIN_BET) * MIN_BET
    stake_por_par = max(MIN_BET, stake_por_par)

    # EV esperado por par
    ev_pares = []
    for a, b in pares:
        cuota_par = a['cuota_favorito'] * b['cuota_favorito']
        hr_par    = a['p_blend'] * b['p_blend']
        ev_par    = _ev(hr_par, cuota_par)
        ev_pares.append(ev_par)
    ev_sistema_avg = sum(ev_pares) / len(ev_pares)

    gastado = stake_por_par * n_pares
    sistema_plan = []

    print()
    print("  ┌─ SISTEMA 2/" + str(n_sistema) + " " + "─" * 50)
    print(f"  │  Budget: ${budget:,.0f} ({BUDGET_SIS_PCT*100:.0f}% bankroll)")
    print(f"  │  Piernas ({n_sistema}): " + " | ".join(f"{l['favorito_predicho']} @{l['cuota_favorito']:.2f}" for l in legs))
    print(f"  │  P(≥2 aciertos): {p_win:.1%}  │  EV promedio por par: {ev_sistema_avg:+.1%}")
    print(f"  │  {n_pares} pares × ${stake_por_par:,.0f}/par = ${gastado:,.0f} total")
    print()
    for j, (a, b) in enumerate(pares):
        cuota_par = a['cuota_favorito'] * b['cuota_favorito']
        hr_par    = a['p_blend'] * b['p_blend']
        retorno   = round(stake_por_par * cuota_par, 0)
        print(f"  │  Par {j+1}: {a['favorito_predicho']} + {b['favorito_predicho']}")
        print(f"  │          Cuota: {cuota_par:.2f}  │  HR: {hr_par:.1%}  │  Retorno: ${retorno:,.0f}")
        sistema_plan.append({
            'jugadores': [a['favorito_predicho'], b['favorito_predicho']],
            'cuota_par': round(cuota_par, 2), 'hr_par': round(hr_par, 4),
            'stake': stake_por_par, 'retorno_potencial': retorno,
        })
    print()
    print(f"  │  Total sistema: ${gastado:,.0f}")
    print("  └" + "─" * 65)
    return gastado, sistema_plan


# ── Resumen deploy ────────────────────────────────────────────────────────────

def _print_resumen(bankroll: float, ind: float, combos: float, sistema: float,
                   senales: list, reporte: dict) -> None:
    total      = ind + combos + sistema
    pct_riesgo = total / bankroll
    meta_ind   = sum(s['stake'] * s['cuota_favorito'] for s in senales)
    ganancia_ind = meta_ind - ind

    print()
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                   RESUMEN DEPLOY                             ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Bankroll total:      ${bankroll:>14,.0f}                       ║")
    print(f"  ║  Individuales:        ${ind:>14,.0f}  ({ind/bankroll:.1%} bankroll)      ║")
    print(f"  ║  Combos:              ${combos:>14,.0f}  ({combos/bankroll:.1%} bankroll)      ║")
    print(f"  ║  Sistema:             ${sistema:>14,.0f}  ({sistema/bankroll:.1%} bankroll)      ║")
    print(f"  ║  TOTAL EN RIESGO:     ${total:>14,.0f}  ({pct_riesgo:.1%} bankroll)      ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Señales APOSTAR:     {len(senales):>3}                                       ║")
    print(f"  ║  Fuente:              {reporte['metadata']['fuente'][:40]:<40}  ║")
    print(f"  ║  Partidos analizados: {reporte['metadata']['n_procesados']:>3}                                       ║")
    print(f"  ║  Superficie:          clay (Roland Garros)                    ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print()

    if pct_riesgo > 0.30:
        print(f"  ⚠️  ALERTA: {pct_riesgo:.1%} bankroll en riesgo — por encima del 30% recomendado.")
        print(f"      Reducir combos o sistema hasta n≥30 con datos limpios.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Trader EV Tenis — capa de deploy')
    parser.add_argument('--bankroll',  type=float, default=100000, help='Bankroll en COP/USD')
    parser.add_argument('--combos',    type=int,   default=2,      help='N piernas por combo (default 2)')
    parser.add_argument('--sistema',   type=int,   default=3,      help='N piernas para sistema 2/N (default 3)')
    parser.add_argument('--ncombos',   type=int,   default=3,      help='Cuántos combos mostrar (default 3)')
    parser.add_argument('--nsistema',  type=int,   default=2,      help='N señales para sistema (default 2)')
    parser.add_argument('--file',      type=str,   default=None,   help='Archivo edge_report específico')
    parser.add_argument('--watchlist', action='store_true',         help='Incluir watchlist en combos/sistema')
    args = parser.parse_args()

    # ── Cargar reporte ──
    try:
        reporte = _load_latest_edge_report(args.file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    senales_raw = reporte.get('apostar', [])
    watchlist   = reporte.get('watchlist', [])

    if not senales_raw:
        print("⚠️  Sin señales APOSTAR en el reporte. Corre edge_calculator.py primero.")
        return

    # Pool para combos: APOSTAR + watchlist si --watchlist
    pool = list(senales_raw)
    if args.watchlist:
        for w in watchlist:
            w['_es_watchlist'] = True
        pool += watchlist

    # ── Enriquecer con p_blend ──
    for s in pool:
        n_h2h    = s.get('n_h2h', 0)
        s['p_blend'] = _p_blend(s['p_modelo'], n_h2h)

    # Budgets
    bankroll      = args.bankroll
    budget_ind    = bankroll * BUDGET_IND_PCT
    budget_combo  = bankroll * BUDGET_COMBO_PCT
    budget_sis    = bankroll * BUDGET_SIS_PCT

    # ── Header ──
    print()
    print("═" * 67)
    print("  TRADER EV TENIS — DEPLOY ROLAND GARROS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  │  Bankroll: ${bankroll:,.0f}")
    print(f"  Señales APOSTAR: {len(senales_raw)}  │  Watchlist: {len(watchlist)}")
    print("═" * 67)

    # ── 1. Individuales ──
    gastado_ind, senales_enriched = _print_individuales(senales_raw, bankroll, budget_ind)

    # Agregar p_blend a senales_enriched (ya calculado arriba)
    for s in senales_enriched:
        s['p_blend'] = _p_blend(s['p_modelo'], s.get('n_h2h', 0))

    # ── 2. Combos ──
    gastado_combos, combos_plan = _build_combos(
        pool, args.combos, args.ncombos, bankroll, budget_combo
    )

    # ── 3. Sistema ──
    gastado_sistema, sistema_plan = _build_sistema(
        pool, args.sistema, args.nsistema, bankroll, budget_sis
    )

    # ── 4. Resumen ──
    _print_resumen(
        bankroll, gastado_ind, gastado_combos, gastado_sistema,
        senales_enriched, reporte
    )

    # ── Nota de calibración ──
    cal_n = reporte['metadata'].get('calibracion_n', 0)
    print(f"  📊 Calibración: n={cal_n} validaciones. "
          f"{'⚠️  Prior uniforme activo — recalibrar con n≥30.' if cal_n < 30 else '✅ Suficiente n para Kelly real.'}")
    print(f"  📊 p_blend usa Bayesian k=3 sobre prior {_P_PRIOR} — "
          f"converge a p_modelo cuando n_h2h≥10.")
    print()

    # ── 5. Guardar plan JSON (T13-05) ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = os.path.join(REPORTS_DIR, f"trader_plan_{timestamp}.json")
    plan = {
        "metadata": {
            "timestamp": timestamp,
            "bankroll": bankroll,
            "edge_report_fuente": reporte['metadata'].get('fuente', ''),
            "n_partidos_analizados": reporte['metadata'].get('n_procesados', 0),
            "calibracion_n": cal_n,
            "parametros": {
                "combos": args.combos,
                "sistema": args.sistema,
                "ncombos": args.ncombos,
                "nsistema": args.nsistema,
                "watchlist": args.watchlist,
                "kelly_fraction": KELLY_FRACTION,
                "budget_pct": {"individuales": BUDGET_IND_PCT, "combos": BUDGET_COMBO_PCT, "sistema": BUDGET_SIS_PCT},
            },
        },
        "individuales": [
            {
                "partido": s['partido'],
                "favorito": s['favorito_predicho'],
                "cuota": s['cuota_favorito'],
                "edge_pct": s['edge_pct'],
                "p_modelo": s['p_modelo'],
                "n_h2h": s.get('n_h2h', 0),
                "p_blend": s['p_blend'],
                "kelly_kl": s['kelly_kl'],
                "kelly_usado": s['kelly_usado'],
                "stake": s['stake'],
                "retorno_potencial": round(s['stake'] * s['cuota_favorito'], 0),
                "zona_cuota": s['zona_cuota'],
                "superficie": s['superficie'],
            }
            for s in senales_enriched
        ],
        "combos": combos_plan,
        "sistema": sistema_plan,
        "resumen": {
            "gastado_individuales": gastado_ind,
            "gastado_combos": gastado_combos,
            "gastado_sistema": gastado_sistema,
            "total_en_riesgo": gastado_ind + gastado_combos + gastado_sistema,
            "pct_bankroll_en_riesgo": round((gastado_ind + gastado_combos + gastado_sistema) / bankroll, 4),
            "n_senales_apostar": len(senales_raw),
            "alerta_riesgo": (gastado_ind + gastado_combos + gastado_sistema) / bankroll > 0.30,
        },
    }
    with open(plan_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"  💾 Plan guardado: {plan_file}")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
trader_ev_tenis.py — Capa trader sobre edge_calculator outputs (análogo NBA trader_ev.py)

Lee:    reports/edge_report_FECHA.json  (producido por edge_calculator.py)
Genera:
  1. Individuales  — Kelly-KL stake por señal APOSTAR (cap 10% bankroll)
  2. Combos        — parlays N piernas, cuota = ∏cuotas, HR = ∏p_modelo
  3. Sistema 2/N   — ganas si ≥2 de N aciertos (binomial coverage)
  4. Cobertura     — sistema exclusión: combos de 3-8 piernas con escenarios P&L
  5. Budget cascade — 40% individuales / 40% combos / 20% sistema

Uso:
  python trader_ev_tenis.py --bankroll 100000
  python trader_ev_tenis.py --bankroll 100000 --combos 3 --sistema 4 --ncombos 2 --nsistema 3
  python trader_ev_tenis.py --bankroll 100000 --file reports/edge_report_20260530_072115.json

  # Sistema cobertura con exclusión (el sistema completo):
  python trader_ev_tenis.py --bankroll 100000 --cobertura --watchlist
  python trader_ev_tenis.py --bankroll 100000 --cobertura --watchlist --all-picks --piernas-max 6
  python trader_ev_tenis.py --bankroll 100000 --cobertura --all-picks --excluir "Zverev A.,De Jong J."
  python trader_ev_tenis.py --bankroll 100000 --cobertura --all-picks --min-cuota 1.50
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

# Nodo-79: MIN_BET proporcional por tier — MODO SOMBRA (no afecta stake real)
# Activar en real solo cuando H54-01 gradúe (n≥30, hit% flattened ≥ hit% financiado)
_MIN_BET_BY_TIER = {
    'itf':        100,
    'challenger': 200,
    'atp500':     500,
    'atp1000':    750,
    'grand_slam': 1000,
}

# Bayesian prior: k=3 equivale a "3 partidos de referencia" histórica (0.52)
# Cuando n_h2h=5: blend = 62.5% p_modelo + 37.5% prior
# Cuando n_h2h=20: blend = 87% p_modelo + 13% prior
_K_PRIOR     = 3
_P_PRIOR     = 0.52   # prior neutral — fallback cuando no hay calibración

_CALIBRACION_PATH = os.path.join(os.path.dirname(__file__), 'data', 'calibracion_edge.json')

# Nodo-70: CPPI — Kelly con piso de supervivencia (Black-Perold)
# Constantes PROVISIONALES etiquetadas — NO modificar sin sesión de recalibración
_CPPI_FLOOR_PCT  = 0.70   # PROVISIONAL: piso = 70% del bankroll pico
_CPPI_MULTIPLIER = 2.0    # PROVISIONAL: m=2 (apalancamiento sobre cushion)


def _cppi_factor(bankroll: float, peak_bankroll: float) -> float:
    """
    Nodo-70: CPPI factor de sizing.

    cushion_t = (bankroll_t - FLOOR) / bankroll_t
    donde FLOOR = _CPPI_FLOOR_PCT * peak_bankroll

    factor = min(1.0, max(0.0, _CPPI_MULTIPLIER * cushion_t))

    A bankroll = peak  -> cushion = 1 - FLOOR_PCT  -> factor aprox 0.60 (FLOOR=70%, m=2)
    A bankroll = FLOOR -> cushion = 0              -> factor = 0.0 (no sizing)
    A bankroll > peak  (no ocurre en t=0): factor clampeado a 1.0

    IMPORTANTE: Este factor se aplica DESPUES del VaR en el waterfall log.
    Las constantes son PROVISIONALES y estan etiquetadas como tales.
    """
    floor = _CPPI_FLOOR_PCT * peak_bankroll
    if bankroll <= 0:
        return 0.0
    cushion_t = (bankroll - floor) / bankroll
    factor = _CPPI_MULTIPLIER * cushion_t
    return min(1.0, max(0.0, factor))


def _load_p_prior(superficie: str = 'unknown', tier: str = 'grand_slam') -> float:
    """
    Carga p_historica desde calibracion_edge.json usando Thompson Beta.
    Jerarquía: [superficie_tier] n≥10 → fallback_por_tier n≥10 → por_superficie n≥10 → global → _P_PRIOR.
    Thompson mean = (wins + 1) / (wins + losses + 2)
    """
    try:
        cal = json.load(open(_CALIBRACION_PATH, encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return _P_PRIOR

    def _thompson(data: dict) -> float:
        w, l = data.get('wins', 0), data.get('losses', 0)
        return (w + 1) / (w + l + 2)

    # 1. Más específico: superficie + tier
    key = f'{superficie}_{tier}'
    data = cal.get('por_superficie_y_tier', {}).get(key, {})
    # FIX-5: preferir era_v2 cuando era_v2_n >= 10 (datos post-normalización-fix 2026-06-19)
    _ev2_n = data.get('era_v2_wins', 0) + data.get('era_v2_losses', 0)
    if _ev2_n >= 10:
        return _thompson({'wins': data.get('era_v2_wins', 0), 'losses': data.get('era_v2_losses', 0)})
    if isinstance(data, dict) and (data.get('wins', 0) + data.get('losses', 0)) >= 10:
        return _thompson(data)

    # 2. Tier solo (fallback_por_tier) — valores pre-calculados como float o dict
    #    B-08: clamp con p_superficie si disponible y diverge > 0.03
    data = cal.get('fallback_por_tier', {}).get(tier)
    if data is not None:
        p_tier = round(float(data), 4) if not isinstance(data, dict) else _thompson(data)
        sup_data = cal.get('por_superficie', {}).get(superficie, {})
        if isinstance(sup_data, dict) and (sup_data.get('wins', 0) + sup_data.get('losses', 0)) >= 10:
            p_sup = _thompson(sup_data)
            if p_tier - p_sup > 0.03:
                return round(min(p_tier, p_sup), 4)
        return p_tier

    # 3. Superficie sola
    data = cal.get('por_superficie', {}).get(superficie, {})
    if isinstance(data, dict) and (data.get('wins', 0) + data.get('losses', 0)) >= 10:
        return _thompson(data)

    # 4. Global
    data = cal.get('global', {})
    if isinstance(data, dict) and (data.get('wins', 0) + data.get('losses', 0)) >= 10:
        return _thompson(data)

    return _P_PRIOR


# ── Portfolio Risk Management (Hedge Fund Layer) ──────────────────────────────
#
# Trata cada sesión de apuestas como un portafolio de activos de corta duración.
# Correlación estructural: picks del mismo torneo/superficie/ronda están correlacionados.
# El "colapso convexo" (ruina) ocurre cuando se asume independencia entre activos correlacionados.
#
# Modelo de correlación estructural:
#   ρ_base = 0.15 (misma superficie)
#   ρ_round = 0.10 (misma ronda del torneo)
#   ρ_total ≈ 0.25 para picks en el mismo Grand Slam round
#
# Portfolio Kelly: f_portfolio = f_individual / (1 + ρ × (N-1))
#   → Con N=8 picks y ρ=0.25: factor = 1/2.75 = 0.364
#   → Reduce exposición total un 63.6% vs Kelly individual naive
#
# VaR (Value at Risk) al 95%: máxima pérdida esperada en 1 de cada 20 sesiones
# CVaR (Conditional VaR): pérdida esperada DADO que estás en el peor 5%
# Sharpe Ratio: exceso de retorno / volatilidad del retorno
# Kelly Growth Rate: g = Σ p_i × log(1 + f_i × (cuota_i - 1)) + (1-p_i) × log(1 - f_i)

RHO_SURFACE     = 0.15   # correlación por misma superficie
RHO_ROUND       = 0.10   # correlación por misma ronda
RHO_SESSION     = 0.25   # ρ total para misma sesión Grand Slam (surface + round)
VAR_CONFIDENCE  = 0.95   # nivel de confianza VaR
MAX_VAR_PCT     = 0.25   # VaR máximo permitido: 25% del bankroll

# ρ calibrado por categoría de torneo (T15-04)
# Grand Slam: presión máxima, misma superficie/ronda → correlación alta
# ATP 1000:   circuito premier, correlación media-alta
# ATP 500:    correlación media
# Challenger: menor presión, picks más independientes
RHO_BY_TOURNAMENT = {
    'grand_slam':  0.25,
    'atp1000':     0.20,
    'atp500':      0.15,
    'challenger':  0.10,
    'itf':         0.05,
}


def _portfolio_kelly_factor(n_picks: int, rho: float = RHO_SESSION) -> float:
    """
    Factor de reducción para Portfolio Kelly multi-activo.
    Cuando N activos tienen correlación ρ entre sí, el Kelly óptimo
    del portafolio se reduce por: 1 / (1 + ρ × (N-1))

    N=1: factor=1.0 (sin reducción)
    N=8, ρ=0.25: factor=0.364 (reduce 63.6%)
    """
    if n_picks <= 1:
        return 1.0
    return 1.0 / (1.0 + rho * (n_picks - 1))


def _compute_var_cvar(combos_plan: list, total_staked: float, p_avg: float,
                      n_picks: int, bankroll: float = 0.0, rho: float = 0.0) -> dict:
    """
    Calcula VaR y CVaR del portafolio de combinadas.

    Modelo: simulación binomial con N picks y p_avg por pick.
    Para cada escenario de k fallos, calcula P&L.
    VaR_95 = peor P&L con 95% de confianza.
    CVaR_95 = E[P&L | P&L < VaR_95]
    """
    # Distribución de P&L por escenario (0 fallos a N fallos)
    scenarios = []
    for n_fallos in range(n_picks + 1):
        # P(exactamente n_fallos)
        p_exact = (math.comb(n_picks, n_fallos)
                   * ((1 - p_avg) ** n_fallos)
                   * (p_avg ** (n_picks - n_fallos)))

        # P&L en este escenario: combos que pagan son los que NO contienen fallidos
        # Aproximación: si k piernas ganadas, combos de ≤k piernas pagan
        n_ganados = n_picks - n_fallos
        retorno = 0.0
        for c in combos_plan:
            n_piernas = c['piernas_n']
            if n_piernas <= n_ganados:
                # Probabilidad de que ESTE combo específico contenga solo ganadores
                # = C(n_ganados, n_piernas) / C(n_picks, n_piernas)
                p_combo_gana = math.comb(n_ganados, n_piernas) / max(1, math.comb(n_picks, n_piernas))
                retorno += c['retorno_potencial'] * p_combo_gana

        pl = retorno - total_staked
        scenarios.append({'n_fallos': n_fallos, 'prob': p_exact, 'pl': pl})

    # B-07: Correlation adjustment — inflate tail probabilities for ρ>0
    # With positive correlation, extreme outcomes (all win or all lose) are more likely
    if rho > 0 and n_picks > 1:
        expected_fallos = n_picks * (1 - p_avg)
        for s in scenarios:
            deviation = abs(s['n_fallos'] - expected_fallos) / max(1, n_picks)
            inflation = 1.0 + rho * (deviation ** 0.5) * 2.0
            s['prob'] *= inflation
        # Re-normalize
        total_prob = sum(s['prob'] for s in scenarios)
        if total_prob > 0:
            for s in scenarios:
                s['prob'] /= total_prob

    # Ordenar por P&L ascendente
    scenarios.sort(key=lambda x: x['pl'])

    # VaR: encontrar el P&L en el percentil (1-confidence)
    cumprob = 0.0
    var_95 = scenarios[0]['pl']  # peor caso por defecto
    for s in scenarios:
        cumprob += s['prob']
        if cumprob >= (1 - VAR_CONFIDENCE):
            var_95 = s['pl']
            break

    # CVaR: E[P&L | P&L ≤ VaR]
    cvar_sum = 0.0
    cvar_prob = 0.0
    for s in scenarios:
        if s['pl'] <= var_95:
            cvar_sum += s['prob'] * s['pl']
            cvar_prob += s['prob']
    cvar_95 = cvar_sum / cvar_prob if cvar_prob > 0 else var_95

    # E[P&L] y std
    e_pl = sum(s['prob'] * s['pl'] for s in scenarios)
    e_pl2 = sum(s['prob'] * s['pl'] ** 2 for s in scenarios)
    std_pl = math.sqrt(max(0, e_pl2 - e_pl ** 2))
    sharpe = e_pl / std_pl if std_pl > 0 else 0.0

    # Kelly Growth Rate: g = E[log(1 + R)] donde R = P&L/bankroll
    # Aproximación usando escenarios
    _kgr_denom = bankroll if bankroll > 0 else total_staked
    growth_rate = 0.0
    for s in scenarios:
        if _kgr_denom > 0:
            r = s['pl'] / _kgr_denom  # retorno relativo al bankroll
            if 1 + r > 0:
                growth_rate += s['prob'] * math.log(1 + r)

    return {
        'var_95': var_95,
        'cvar_95': cvar_95,
        'expected_pl': e_pl,
        'std_pl': std_pl,
        'sharpe_ratio': sharpe,
        'kelly_growth_rate': growth_rate,
        'scenarios': scenarios,
    }


def _portfolio_risk_report(pool: list, cobertura_plan: list, total_staked: float,
                           bankroll: float, gastado_ind: float,
                           rho: float = RHO_SESSION) -> dict:
    """
    Reporte completo de riesgo del portafolio (Hedge Fund layer).

    Aplica:
    1. Portfolio Kelly (correlación-ajustado)
    2. VaR/CVaR constraint
    3. Sharpe Ratio
    4. Kelly Growth Rate (tasa de crecimiento del bankroll)
    5. Recomendación de scaling de bankroll
    """
    N = len(pool)
    p_avg = sum(p['p_blend'] for p in pool) / N if N > 0 else 0.5

    # 1. Factor de Portfolio Kelly
    pk_factor = _portfolio_kelly_factor(N, rho)

    # 2. Stake total ajustado por Portfolio Kelly
    total_en_riesgo = total_staked + gastado_ind
    total_ajustado = total_en_riesgo * pk_factor
    reduccion_pct = 1.0 - pk_factor

    # 3. VaR/CVaR
    risk = _compute_var_cvar(cobertura_plan, total_staked, p_avg, N,
                             bankroll=bankroll, rho=rho)

    # B-01: Include individuales in VaR — worst case all individuales lose
    if gastado_ind > 0:
        risk['var_95'] = risk['var_95'] - gastado_ind
        risk['cvar_95'] = risk['cvar_95'] - gastado_ind

    # 4. VaR constraint: si VaR > MAX_VAR_PCT × bankroll → reducir
    var_pct = abs(risk['var_95']) / bankroll if bankroll > 0 else 0
    var_excedido = var_pct > MAX_VAR_PCT
    factor_var = min(1.0, MAX_VAR_PCT * bankroll / abs(risk['var_95'])) if risk['var_95'] < 0 else 1.0

    # Bug fix: si stakes totales superan el bankroll → reducir proporcionalmente
    # (ocurre cuando VaR es positivo pero total_staked > bankroll)
    if total_en_riesgo > bankroll:
        factor_bankroll = bankroll / total_en_riesgo
        factor_var = min(factor_var, factor_bankroll)
        var_excedido = True

    # 5. Kelly Growth Rate y bankroll scaling
    # Si g > 0 y n≥30: bankroll puede crecer
    # Recomendación: new_bankroll = bankroll × exp(g × n_sessions)
    sessions_to_double = math.log(2) / risk['kelly_growth_rate'] if risk['kelly_growth_rate'] > 0 else float('inf')

    # ── Imprimir ──
    print()
    print("  ┌─ PORTFOLIO RISK MANAGEMENT (HEDGE FUND LAYER) " + "─" * 17)
    print(f"  │")
    print(f"  │  ── Correlación y Portfolio Kelly ──")
    print(f"  │  Picks en sesión:     {N}")
    print(f"  │  ρ estructural:       {rho:.2f} (misma superficie + ronda)")
    print(f"  │  Portfolio Kelly:     factor = {pk_factor:.3f}")
    print(f"  │  Reducción vs naive:  -{reduccion_pct:.1%}")
    print(f"  │")
    print(f"  │  Stake naive (sin ajuste):    ${total_en_riesgo:>12,.0f}")
    print(f"  │  Stake ajustado (PK):         ${total_ajustado:>12,.0f}")
    print(f"  │  Ahorro por correlación:      ${total_en_riesgo - total_ajustado:>12,.0f}")
    print(f"  │")
    print(f"  │  ── Value at Risk ──")
    print(f"  │  VaR 95%:            ${risk['var_95']:>12,.0f}  ({var_pct:.1%} bankroll)")
    print(f"  │  CVaR 95%:           ${risk['cvar_95']:>12,.0f}")
    print(f"  │  Max VaR permitido:  ${MAX_VAR_PCT * bankroll:>12,.0f}  ({MAX_VAR_PCT:.0%} bankroll)")
    if var_excedido:
        print(f"  │  ⚠️  VaR EXCEDIDO → reducir stakes ×{factor_var:.2f}")
    else:
        print(f"  │  ✅ VaR dentro de límites")
    print(f"  │")
    print(f"  │  ── Métricas de Retorno ──")
    print(f"  │  E[P&L] por sesión:  ${risk['expected_pl']:>12,.0f}")
    print(f"  │  σ(P&L):             ${risk['std_pl']:>12,.0f}")
    print(f"  │  Sharpe Ratio:       {risk['sharpe_ratio']:>12.3f}")
    print(f"  │  Kelly Growth Rate:  {risk['kelly_growth_rate']:>12.4f} (por sesión)")
    print(f"  │")
    print(f"  │  ── Proyección Bankroll ──")
    if risk['kelly_growth_rate'] > 0:
        br_5 = bankroll * math.exp(risk['kelly_growth_rate'] * 5)
        br_10 = bankroll * math.exp(risk['kelly_growth_rate'] * 10)
        br_30 = bankroll * math.exp(risk['kelly_growth_rate'] * 30)
        print(f"  │  Bankroll en  5 sesiones:  ${br_5:>12,.0f}")
        print(f"  │  Bankroll en 10 sesiones:  ${br_10:>12,.0f}")
        print(f"  │  Bankroll en 30 sesiones:  ${br_30:>12,.0f}")
        print(f"  │  Sesiones para duplicar:   {sessions_to_double:.1f}")
    else:
        print(f"  │  ❌ Growth rate negativo — NO escalar bankroll")
        print(f"  │     Sistema en fase de calibración, mantener bankroll actual")
    print(f"  │")

    # Recomendación
    print(f"  │  ── RECOMENDACIÓN DE TRADER ──")
    if risk['kelly_growth_rate'] > 0 and not var_excedido:
        print(f"  │  ✅ Sistema con edge positivo y riesgo controlado")
        print(f"  │  → MANTENER bankroll actual. Escalar +20% después de 5 sesiones validadas.")
        print(f"  │  → Stake por combo ajustado PK: ${total_ajustado/max(1,len(cobertura_plan)):,.0f}")
    elif risk['kelly_growth_rate'] > 0 and var_excedido:
        print(f"  │  ⚠️  Edge positivo PERO VaR excedido")
        print(f"  │  → REDUCIR stakes ×{factor_var:.2f} (de ${total_en_riesgo:,.0f} a ${total_en_riesgo*factor_var:,.0f})")
        print(f"  │  → O reducir --top-n a {max(2, int(len(cobertura_plan) * factor_var / 4))} combos/tier")
    else:
        print(f"  │  ❌ Sin edge o riesgo excesivo. NO desplegar capital.")

    print("  └" + "─" * 65)

    return {
        'pk_factor': pk_factor,
        'total_ajustado': total_ajustado,
        'var_95': risk['var_95'],
        'cvar_95': risk['cvar_95'],
        'expected_pl': risk['expected_pl'],
        'sharpe_ratio': risk['sharpe_ratio'],
        'kelly_growth_rate': risk['kelly_growth_rate'],
        'sessions_to_double': sessions_to_double,
        'var_excedido': var_excedido,
        'factor_var': factor_var,
    }


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


def _p_blend(p_modelo: float, n_h2h: int = 0, p_prior: float = _P_PRIOR) -> float:
    """
    Bayesian blend: (n_h2h × p_modelo + K × prior) / (n_h2h + K)
    Cuando n_h2h=0: usa solo el prior (señal sin historial directo)
    Cuando n_h2h grande: converge a p_modelo
    p_prior: derivado de calibracion_edge.json si n≥30 (T13-06), sino _P_PRIOR=0.52
    """
    return (n_h2h * p_modelo + _K_PRIOR * p_prior) / (n_h2h + _K_PRIOR)


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

def _print_individuales(senales: list, bankroll: float, budget: float,
                        p_prior: float = _P_PRIOR) -> tuple:
    """
    Imprime tabla de individuales. Retorna (total_gastado, lista_enriched).
    Usa p_historica_usada per-pick del edge_calculator si disponible; cae a p_prior como fallback.
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
        p_b     = _p_blend(p_mod, n_h2h, s.get('p_historica_usada') or p_prior)
        cuota   = s['cuota_favorito']
        kelly   = min(KELLY_CAP_IND, _kelly_quarter(p_b, cuota))
        raw_stake     = kelly * bankroll
        capped_stake  = min(raw_stake, budget - gastado)
        rounded_stake = round(capped_stake / MIN_BET) * MIN_BET
        stake         = max(MIN_BET, rounded_stake)
        ev_ind  = _ev(p_b, cuota)
        retorno = round(stake * cuota, 0)

        # Waterfall log — almacenado para diagnóstico VaR (D54-01, P54-02)
        waterfall = {
            'kelly_kl_report': s.get('kelly_kl', 0),
            'p_blend': round(p_b, 4),
            'kelly_quarter': round(kelly, 4),
            'raw_stake': round(raw_stake, 0),
            'capped_by_budget': capped_stake < raw_stake,
            'capped_stake': round(capped_stake, 0),
            'rounded_stake': rounded_stake,
            'stake_pre_var': stake,
            'terminal_reason': None,  # se rellena en VaR adjustment
        }

        print(f"  │  🎾 {s['partido']}")
        print(f"  │     Apostar: {s['favorito_predicho']}")
        print(f"  │     Cuota {cuota:.2f}  │  Edge {s['edge_pct']}  │  p_blend {p_b:.3f} (n_h2h={n_h2h})")
        print(f"  │     Kelly-KL {s['kelly_kl']:.1%} → stake ${stake:,.0f}  │  Retorno potencial ${retorno:,.0f}")
        print(f"  │     EV real {ev_ind:+.1%}  │  Zona: {s['zona_cuota']}  │  Sup: {s['superficie']}")
        print()

        gastado += stake
        enriched.append({**s, 'stake': stake, 'p_blend': p_b, 'kelly_usado': kelly,
                         '_waterfall': waterfall})

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


# ── Cobertura por Exclusión ─────────────────────────────────────────────────────

def _build_cobertura(pool: list, piernas_min: int, piernas_max: int,
                     bankroll: float, budget: float, top_n: int = 5) -> tuple:
    """
    Sistema de cobertura por exclusión.

    Genera TODOS los combos posibles de piernas_min a piernas_max piernas.
    Cada combo de K piernas implícitamente EXCLUYE (N-K) jugadores.
    Si esos jugadores excluidos fallan, el combo sigue pagando.

    Distribución de budget:
      - Combos cortos → más peso (protección: toleran más fallos)
      - Combos largos → menos peso (upside: pagan más si todo acierta)

    Escenario de P&L: para cada nivel de fallos (0, 1, 2, 3, 4),
    calcula cuántos combos ganan y cuál es el P&L neto.
    """
    N = len(pool)
    piernas_max = min(piernas_max, N)
    if N < piernas_min:
        print(f"\n  ⚠️  Solo {N} picks en pool — se necesitan ≥{piernas_min}.")
        return 0.0, []

    # Header
    print()
    print("  ┌─ COBERTURA POR EXCLUSIÓN " + "─" * 39)
    print(f"  │  Pool: {N} picks  │  Combos: {piernas_min} a {piernas_max} piernas")
    print(f"  │  Budget: ${budget:,.0f}")
    print(f"  │")
    print(f"  │  Picks en pool (ordenados por cuota):")
    sorted_pool = sorted(pool, key=lambda x: x['cuota_favorito'], reverse=True)
    for i, p in enumerate(sorted_pool):
        tag = " [W]" if p.get('_es_watchlist') else ""
        zona = p.get('zona_cuota', '?')
        print(f"  │    {i+1}. {p['favorito_predicho']:20s} @{p['cuota_favorito']:.2f}  "
              f"p={p['p_blend']:.3f}  zona={zona}{tag}")
    print()

    # Generar todas las combinaciones por tier
    tiers = {}
    for k in range(piernas_min, piernas_max + 1):
        tier = []
        for combo in combinations(pool, k):
            cuota_combo = 1.0
            hr_combo = 1.0
            for leg in combo:
                cuota_combo *= leg['cuota_favorito']
                hr_combo *= leg['p_blend']
            ev = _ev(hr_combo, cuota_combo)
            tier.append({
                'legs': combo,
                'cuota_combo': round(cuota_combo, 2),
                'hr_combo': round(hr_combo, 4),
                'ev_combo': ev,
                'excluidos': [p['favorito_predicho'] for p in pool if p not in combo],
            })
        tier.sort(key=lambda x: x['ev_combo'], reverse=True)
        tiers[k] = tier

    # Seleccionar combos con DIVERSIDAD: asegurar que cada jugador es excluido
    # en al menos un combo seleccionado. Esto garantiza que si cualquier jugador
    # falla, al menos algunos combos del portfolio sobreviven.
    for k in tiers:
        tier = tiers[k]
        if len(tier) <= top_n:
            continue  # no hay que filtrar

        selected = []
        players_excluded = set()  # jugadores ya excluidos en algún combo seleccionado
        all_players = {p['favorito_predicho'] for p in pool}

        # Fase 1: garantizar diversidad — seleccionar 1 combo que excluya cada jugador
        for player in sorted(all_players):
            if len(selected) >= top_n:
                break
            for c in tier:
                if c in selected:
                    continue
                if player in c['excluidos'] and player not in players_excluded:
                    selected.append(c)
                    players_excluded.update(c['excluidos'])
                    break

        # Fase 2: rellenar hasta top_n con los de mayor EV que no estén seleccionados
        for c in tier:
            if len(selected) >= top_n:
                break
            if c not in selected:
                selected.append(c)

        # Re-ordenar por EV
        selected.sort(key=lambda x: x['ev_combo'], reverse=True)
        tiers[k] = selected

    total_combos = sum(len(v) for v in tiers.values())

    # Distribución de budget: peso inversamente proporcional a piernas
    # 3-piernas recibe más budget que 8-piernas (más protección)
    tier_weights = {}
    for k in range(piernas_min, piernas_max + 1):
        tier_weights[k] = piernas_max - k + 1  # 3→peso_alto, 8→peso_bajo
    total_weight = sum(tier_weights[k] * len(tiers.get(k, [])) for k in tier_weights
                       if len(tiers.get(k, [])) > 0) or 1

    # Calcular stake por combo en cada tier
    tier_stakes = {}
    for k in range(piernas_min, piernas_max + 1):
        n_combos_tier = len(tiers.get(k, []))
        if n_combos_tier == 0:
            tier_stakes[k] = 0
            continue
        weight_tier = tier_weights[k] * n_combos_tier
        budget_tier = budget * (weight_tier / total_weight)
        stake_each = round(budget_tier / n_combos_tier / MIN_BET) * MIN_BET
        stake_each = max(MIN_BET, stake_each)
        tier_stakes[k] = stake_each

    # Imprimir cada tier
    gastado_total = 0.0
    all_plan = []

    for k in range(piernas_min, piernas_max + 1):
        tier = tiers.get(k, [])
        if not tier:
            continue
        stake = tier_stakes[k]
        n_tier = len(tier)
        budget_tier_real = stake * n_tier

        print(f"  │  ── {k} PIERNAS: TOP {n_tier} combos × ${stake:,.0f} = ${budget_tier_real:,.0f} ──")
        print(f"  │     (cada combo excluye {N-k} jugador{'es' if N-k > 1 else ''})")

        for i, c in enumerate(tier):
            retorno = round(stake * c['cuota_combo'], 0)
            legs_str = " + ".join(f"{l['favorito_predicho']}@{l['cuota_favorito']:.2f}"
                                  for l in c['legs'])
            excl_str = ", ".join(c['excluidos']) if c['excluidos'] else "ninguno"
            print(f"  │     [{k}p-{i+1}] {legs_str}")
            print(f"  │            @{c['cuota_combo']:.2f} HR:{c['hr_combo']:.1%} "
                  f"EV:{c['ev_combo']:+.1%} →${retorno:,.0f}  excluye:[{excl_str}]")

        gastado_total += budget_tier_real
        for c in tier:
            all_plan.append({
                'piernas_n': k,
                'legs': [{'jugador': l['favorito_predicho'], 'cuota': l['cuota_favorito']}
                         for l in c['legs']],
                'cuota_combo': c['cuota_combo'],
                'hr_combo': c['hr_combo'],
                'ev_combo': c['ev_combo'],
                'excluidos': c['excluidos'],
                'stake': stake,
                'retorno_potencial': round(stake * c['cuota_combo'], 0),
            })
        print()

    # ── Análisis de escenarios por fallos ──
    print(f"  │  ── ESCENARIOS DE P&L (¿qué pasa si fallan 0-{min(4, N-piernas_min)} picks?) ──")
    print(f"  │  Total apostado: ${gastado_total:,.0f}")
    print()

    for n_fallos in range(0, min(5, N - piernas_min + 1)):
        # Si fallan n_fallos: los combos que NO contengan ningún fallido PAGAN
        # Peor caso: fallan los de MAYOR cuota (los underdogs, los que más pagan)
        # Mejor caso: fallan los de MENOR cuota (los favoritos, pagan poco)

        # Calcular MEJOR caso (fallan los de menor cuota → favoritos pesados)
        worst_picks = sorted(pool, key=lambda x: x['cuota_favorito'])[:n_fallos]
        worst_names = {p['favorito_predicho'] for p in worst_picks}

        retorno_mejor = 0.0
        combos_ganados_mejor = 0
        for c in all_plan:
            combo_names = {l['jugador'] for l in c['legs']}
            if not combo_names.intersection(worst_names):
                retorno_mejor += c['retorno_potencial']
                combos_ganados_mejor += 1

        # Calcular PEOR caso (fallan los de mayor cuota → underdogs)
        best_picks = sorted(pool, key=lambda x: x['cuota_favorito'], reverse=True)[:n_fallos]
        best_names = {p['favorito_predicho'] for p in best_picks}

        retorno_peor = 0.0
        combos_ganados_peor = 0
        for c in all_plan:
            combo_names = {l['jugador'] for l in c['legs']}
            if not combo_names.intersection(best_names):
                retorno_peor += c['retorno_potencial']
                combos_ganados_peor += 1

        pl_mejor = retorno_mejor - gastado_total
        pl_peor = retorno_peor - gastado_total

        fallos_str = ", ".join(p['favorito_predicho'] for p in worst_picks) if worst_picks else "ninguno"
        fallos_str2 = ", ".join(p['favorito_predicho'] for p in best_picks) if best_picks else "ninguno"

        if n_fallos == 0:
            icon = "🏆"
            print(f"  │  {icon} 0 fallos ({N}/{N} ganan): "
                  f"retorno ${retorno_mejor:,.0f} → P&L +${pl_mejor:,.0f} "
                  f"(TODOS {total_combos} combos pagan)")
        else:
            icon_m = "✅" if pl_mejor > 0 else "❌"
            icon_p = "✅" if pl_peor > 0 else "❌"
            print(f"  │  {icon_m} {n_fallos} fallo{'s' if n_fallos>1 else ''} "
                  f"(mejor caso: fallan favoritos [{fallos_str}]):")
            print(f"  │       retorno ${retorno_mejor:,.0f} → P&L {'+' if pl_mejor>=0 else ''}${pl_mejor:,.0f} "
                  f"({combos_ganados_mejor}/{total_combos} combos pagan)")
            print(f"  │  {icon_p} {n_fallos} fallo{'s' if n_fallos>1 else ''} "
                  f"(peor caso: fallan underdogs [{fallos_str2}]):")
            print(f"  │       retorno ${retorno_peor:,.0f} → P&L {'+' if pl_peor>=0 else ''}${pl_peor:,.0f} "
                  f"({combos_ganados_peor}/{total_combos} combos pagan)")

    print()
    print(f"  │  Total cobertura: ${gastado_total:,.0f}  │  {total_combos} combos totales")
    print("  └" + "─" * 65)
    return gastado_total, all_plan


# ── Resumen deploy ────────────────────────────────────────────────────────────

_TIER_DISPLAY = {
    'grand_slam':  'Grand Slam',
    'atp1000':     'ATP 1000',
    'atp500':      'ATP 500',
    'challenger':  'Challenger',
    'itf':         'ITF',
}


def _print_resumen(bankroll: float, ind: float, combos: float, sistema: float,
                   cobertura: float, senales: list, reporte: dict, pool_size: int = 0,
                   superficie: str = 'clay', torneo_tipo: str = 'grand_slam') -> None:
    total      = ind + combos + sistema + cobertura
    pct_riesgo = total / bankroll

    tier_label = _TIER_DISPLAY.get(torneo_tipo, torneo_tipo.replace('_', ' ').title())
    _sup_str = f"{superficie} / {tier_label}"

    print()
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                   RESUMEN DEPLOY                             ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Bankroll total:      ${bankroll:>14,.0f}                       ║")
    print(f"  ║  Individuales:        ${ind:>14,.0f}  ({ind/bankroll:.1%} bankroll)      ║")
    if cobertura > 0:
        print(f"  ║  Cobertura:           ${cobertura:>14,.0f}  ({cobertura/bankroll:.1%} bankroll)      ║")
    else:
        print(f"  ║  Combos:              ${combos:>14,.0f}  ({combos/bankroll:.1%} bankroll)      ║")
        print(f"  ║  Sistema:             ${sistema:>14,.0f}  ({sistema/bankroll:.1%} bankroll)      ║")
    print(f"  ║  TOTAL EN RIESGO:     ${total:>14,.0f}  ({pct_riesgo:.1%} bankroll)      ║")
    print("  ╠═══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Señales APOSTAR:     {len(senales):>3}                                       ║")
    if pool_size > 0:
        print(f"  ║  Pool cobertura:      {pool_size:>3} picks                                    ║")
    print(f"  ║  Fuente:              {reporte['metadata']['fuente'][:40]:<40}  ║")
    print(f"  ║  Partidos analizados: {reporte['metadata']['n_procesados']:>3}                                       ║")
    print(f"  ║  Superficie / Tier:   {_sup_str:<42}  ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print()

    if pct_riesgo > 0.30:
        print(f"  ⚠️  ALERTA: {pct_riesgo:.1%} bankroll en riesgo — por encima del 30% recomendado.")
        print(f"      Reducir combos o sistema hasta n≥30 con datos limpios.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Trader EV Tenis — capa de deploy')
    parser.add_argument('--bankroll',     type=float, default=100000, help='Bankroll en COP/USD')
    parser.add_argument('--combos',       type=int,   default=2,      help='N piernas por combo (default 2)')
    parser.add_argument('--sistema',      type=int,   default=3,      help='N piernas para sistema 2/N (default 3)')
    parser.add_argument('--ncombos',      type=int,   default=3,      help='Cuántos combos mostrar (default 3)')
    parser.add_argument('--nsistema',     type=int,   default=2,      help='N señales para sistema (default 2)')
    parser.add_argument('--file',         type=str,   default=None,   help='Archivo edge_report específico')
    parser.add_argument('--watchlist',    action=argparse.BooleanOptionalAction, default=True,
                        help='Incluir watchlist en combos/sistema (default: activado, --no-watchlist para desactivar)')
    # ── Nuevos: sistema cobertura por exclusión ──
    parser.add_argument('--cobertura',    action=argparse.BooleanOptionalAction, default=True,
                        help='Activar sistema cobertura por exclusión (default: activado, --no-cobertura para desactivar)')
    parser.add_argument('--all-picks',    action=argparse.BooleanOptionalAction, default=True,
                        help='Incluir TODOS los picks en pool cobertura (default: activado, --no-all-picks para desactivar)')
    parser.add_argument('--piernas-min',  type=int, default=3,
                        help='Mínimo piernas en cobertura (default 3)')
    parser.add_argument('--piernas-max',  type=int, default=4,
                        help='Máximo piernas en cobertura (default 4, max 8)')
    parser.add_argument('--excluir',      type=str, default='',
                        help='Jugadores a excluir del pool, separados por coma')
    parser.add_argument('--min-cuota',    type=float, default=1.50,
                        help='Cuota mínima para entrar al pool de cobertura (default 1.50)')
    parser.add_argument('--top-n',       type=int, default=4,
                        help='Top N combos por tier (por EV, default 4)')
    parser.add_argument('--torneo-tipo', type=str, default='grand_slam',
                        choices=['grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'],
                        help='Tipo de torneo para calibrar ρ de correlación (default: grand_slam)')
    parser.add_argument('--superficie',  type=str, default='clay',
                        choices=['clay', 'grass', 'hard', 'unknown'],
                        help='Superficie para derivar p_prior de calibracion (default: clay)')
    parser.add_argument('--telegram',    action='store_true',
                        help='Enviar señales a Telegram después de generar el plan')
    args = parser.parse_args()

    # Validaciones
    piernas_max = min(max(args.piernas_max, 2), 8)
    piernas_min = max(args.piernas_min, 2)

    # ── Capturar todo el output de consola para exportar a .txt ──
    import sys, io, contextlib
    _buf = io.StringIO()
    _tee = contextlib.redirect_stdout(contextlib.nullcontext())  # placeholder
    class _Tee:
        """Escribe en stdout real Y en buffer."""
        def __init__(self, buf):
            self._buf = buf
            self._real = sys.stdout
        def write(self, s):
            self._real.write(s)
            self._buf.write(s)
        def flush(self):
            self._real.flush()
    sys.stdout = _Tee(_buf)

    # ── Cargar reporte ──
    try:
        reporte = _load_latest_edge_report(args.file)
    except FileNotFoundError as e:
        sys.stdout = sys.stdout._real
        print(f"❌ {e}")
        return

    senales_raw = reporte.get('apostar', [])
    watchlist   = reporte.get('watchlist', [])
    sin_edge    = reporte.get('sin_edge', [])

    # ── D48-05 (Nodo-48): Guard cuota_es_real ────────────────────────────────
    # Si algún pick tiene cuota_es_real=False → las cuotas son de referencia
    # (FlashScore bookmaker id=523), NO de Betplay. No desplegar capital real.
    todos_los_picks = senales_raw + watchlist + sin_edge
    picks_sin_cuota_real = [p for p in todos_los_picks if p.get('cuota_es_real') is False]
    if picks_sin_cuota_real:
        print("=" * 70)
        print("  GUARD D48-05 — CUOTAS NO REALES DETECTADAS")
        print("=" * 70)
        print(f"  {len(picks_sin_cuota_real)}/{len(todos_los_picks)} picks tienen cuota_es_real=False")
        print("  Origen: FlashScore odds de referencia (--flashscore-only), NO Betplay.")
        print("  Este reporte es SOLO para testing/validacion post-hoc.")
        print()
        print("  NO desplegar capital real con este reporte.")
        print("  Para apuestas reales: correr PASO 1 con Kambi antes del partido.")
        print("=" * 70)
        sys.stdout = sys.stdout._real
        return
    # ─────────────────────────────────────────────────────────────────────────

    # Filtrar por tier del torneo ANTES de construir pool — el trader procesa UN tier por ejecución
    _tier_filtro = args.torneo_tipo
    senales_raw = [p for p in senales_raw if p.get('tier', 'atp500') == _tier_filtro]
    watchlist   = [p for p in watchlist   if p.get('tier', 'atp500') == _tier_filtro]
    sin_edge    = [p for p in sin_edge    if p.get('tier', 'atp500') == _tier_filtro]

    if not senales_raw and not args.all_picks:
        print(f"⚠️  Sin señales APOSTAR para tier '{_tier_filtro}'. Tiers disponibles en reporte:")
        _all = reporte.get('apostar', []) + reporte.get('watchlist', [])
        for t in sorted(set(p.get('tier', '?') for p in _all)):
            n = sum(1 for p in _all if p.get('tier') == t)
            print(f"     {t}: {n} picks")
        return

    # ── Construir pool ──
    pool = list(senales_raw)
    if args.watchlist or args.cobertura:
        for w in watchlist:
            w['_es_watchlist'] = True
        pool += watchlist
    if args.all_picks:
        for s in sin_edge:
            s['_es_sin_edge'] = True
        pool += sin_edge

    # Filtrar por cuota mínima
    if args.min_cuota > 1.0:
        pool = [p for p in pool if p['cuota_favorito'] >= args.min_cuota]

    # Filtrar por exclusiones manuales
    if args.excluir:
        excluir_set = {n.strip() for n in args.excluir.split(',')}
        pool = [p for p in pool if p['favorito_predicho'] not in excluir_set]

    # Budgets
    bankroll      = args.bankroll
    budget_ind    = bankroll * BUDGET_IND_PCT
    budget_combo  = bankroll * BUDGET_COMBO_PCT
    budget_sis    = bankroll * BUDGET_SIS_PCT

    # En modo cobertura: budget combo+sistema va al portfolio de cobertura
    if args.cobertura:
        budget_cobertura = budget_combo + budget_sis
    else:
        budget_cobertura = 0.0

    # ρ por tipo de torneo (T15-04) — pool ya filtrado por tier
    rho_sesion = RHO_BY_TOURNAMENT.get(args.torneo_tipo, RHO_SESSION)

    # p_prior calibrado por superficie+tier (B-02) — fallback si pick no trae p_historica_usada
    p_prior_efectivo = _load_p_prior(args.superficie, args.torneo_tipo)

    # ── Enriquecer con p_blend usando p_historica per-match (tier+superficie estratificado)
    # Cada pick del edge_report ya trae p_historica_usada calibrado por su propio tier+superficie.
    # Si no está disponible, cae al p_prior_efectivo del CLI.
    for s in pool:
        _p_prior_pick = s.get('p_historica_usada') or p_prior_efectivo
        s['p_blend'] = _p_blend(s['p_modelo'], s.get('n_h2h', 0), _p_prior_pick)

    # ── Header ──
    print()
    print("═" * 67)
    print(f"  TRADER EV TENIS — {args.torneo_tipo.upper()} {args.superficie.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  │  Bankroll: ${bankroll:,.0f}")
    print(f"  Señales APOSTAR: {len(senales_raw)}  │  Watchlist: {len(watchlist)}"
          + (f"  │  Sin edge: {len(sin_edge)}" if args.all_picks else ""))
    if args.cobertura:
        print(f"  MODO: COBERTURA POR EXCLUSIÓN  │  Pool: {len(pool)} picks")
        print(f"  Piernas: {piernas_min}-{piernas_max}  │  Min cuota: {args.min_cuota:.2f}  │  ρ={rho_sesion:.2f} ({args.torneo_tipo})")
        if args.excluir:
            print(f"  Excluidos: {args.excluir}")
    print("═" * 67)

    # ── 1. Individuales (siempre, solo con señales APOSTAR) ──
    gastado_ind = 0.0
    senales_enriched = []
    if senales_raw:
        gastado_ind, senales_enriched = _print_individuales(senales_raw, bankroll, budget_ind,
                                                             p_prior=p_prior_efectivo)
        # p_blend ya calculado per-pick con p_historica_usada dentro de _print_individuales — no sobreescribir

    # ── 2/3. Modo normal: Combos + Sistema ──
    gastado_combos = 0.0
    gastado_sistema = 0.0
    combos_plan = []
    sistema_plan = []

    if not args.cobertura:
        gastado_combos, combos_plan = _build_combos(
            pool, args.combos, args.ncombos, bankroll, budget_combo
        )
        gastado_sistema, sistema_plan = _build_sistema(
            pool, args.sistema, args.nsistema, bankroll, budget_sis
        )

    # ── 2/3. Modo cobertura: sistema por exclusión ──
    gastado_cobertura = 0.0
    cobertura_plan = []

    if args.cobertura:
        gastado_cobertura, cobertura_plan = _build_cobertura(
            pool, piernas_min, piernas_max, bankroll, budget_cobertura, top_n=args.top_n
        )

    # ── 4. Portfolio Risk Management (Hedge Fund Layer) ──
    risk_metrics = {}
    if args.cobertura and cobertura_plan:
        risk_metrics = _portfolio_risk_report(
            pool, cobertura_plan, gastado_cobertura, bankroll, gastado_ind,
            rho=rho_sesion
        )

    # ── 4b. Auto-ajuste VaR (T15-05) ──────────────────────────────────────────
    # Si VaR excedido, aplica factor_var a todos los stakes y muestra plan final.
    if risk_metrics.get('var_excedido') and risk_metrics.get('factor_var', 1.0) < 1.0:
        fv = risk_metrics['factor_var']

        # Nodo-70: CPPI factor — piso de supervivencia Black-Perold
        # peak_bankroll = bankroll en t=0 (primera sesión siempre en peak)
        cppi_f = _cppi_factor(bankroll=bankroll, peak_bankroll=bankroll)
        floor_cppi = _CPPI_FLOOR_PCT * bankroll

        # Ajustar individuales
        for s in senales_enriched:
            stake_pre = s['stake']
            # Waterfall: kelly_kl → ×portfolio_factor → ×var_factor → ×cppi → MIN_BET_CLIFF
            stake_post_var  = stake_pre * fv
            stake_post_cppi = stake_post_var * cppi_f
            s['stake'] = round(stake_post_cppi / MIN_BET) * MIN_BET
            s['retorno_potencial'] = round(s['stake'] * s['cuota_favorito'], 0)
            # Waterfall: registrar causa terminal (D54-01)
            wf = s.get('_waterfall', {})
            if s['stake'] == 0 and stake_pre > 0:
                wf['terminal_reason'] = (
                    f'MIN_BET_CLIFF (stake_pre_var=${stake_pre:,.0f} × var_factor={fv:.2f}'
                    f' × cppi={cppi_f:.4f} = ${stake_post_cppi:,.0f} < MIN_BET={MIN_BET:,})'
                )
            elif s['stake'] > 0:
                wf['terminal_reason'] = (
                    f'OK (stake_pre_var=${stake_pre:,.0f} × {fv:.2f} × cppi={cppi_f:.4f}'
                    f' = ${s["stake"]:,.0f})'
                )
            wf['stake_final'] = s['stake']
            wf['var_factor'] = fv
            wf['cppi_factor'] = cppi_f
            wf['cppi_log'] = (
                f'cppi={cppi_f:.4f}(bankroll={bankroll}, peak={bankroll},'
                f' floor={floor_cppi:.0f})'
            )
            wf['var_flattened'] = (s['stake'] == 0 and stake_pre > 0)
            # Nodo-79: shadow mode — calcula stake con MIN_BET por tier, sin cambiar stake real
            _tier_key = s.get('tier', 'grand_slam')
            _min_bet_shadow = _MIN_BET_BY_TIER.get(_tier_key, MIN_BET)
            _stake_shadow = round(stake_post_cppi / _min_bet_shadow) * _min_bet_shadow
            wf['stake_final_shadow'] = _stake_shadow
            wf['min_bet_shadow_usado'] = _min_bet_shadow
            wf['shadow_survives_cliff'] = (_stake_shadow > 0 and wf['var_flattened'])
        gastado_ind = sum(s['stake'] for s in senales_enriched)

        # Ajustar cobertura
        for c in cobertura_plan:
            c['stake'] = round(c['stake'] * fv / MIN_BET) * MIN_BET
            c['retorno_potencial'] = round(c['stake'] * c['cuota_combo'], 0)
        gastado_cobertura = sum(c['stake'] for c in cobertura_plan)

        # Imprimir plan final ajustado
        print()
        print(f"  ┌─ STAKES FINALES (VaR AJUSTADO ×{fv:.2f}) " + "─" * 27)
        print(f"  │")
        if senales_enriched:
            print(f"  │  INDIVIDUALES:")
            for s in senales_enriched:
                print(f"  │    {s['favorito_predicho']:25s} @{s['cuota_favorito']:.2f}  →  ${s['stake']:>8,.0f}  (ret. ${s['retorno_potencial']:>10,.0f})")
                wf = s.get('_waterfall', {})
                if wf.get('var_flattened'):
                    print(f"  │    LOG_STAKE_WATERFALL: kelly_kl={wf['kelly_kl_report']:.3f} "
                          f"→ p_blend={wf['p_blend']:.4f} → kelly_q={wf['kelly_quarter']:.4f} "
                          f"→ raw=${wf['raw_stake']:,.0f} → pre_var=${wf['stake_pre_var']:,.0f} "
                          f"→ {wf['terminal_reason']}")
        print(f"  │")
        print(f"  │  COBERTURA ({len(cobertura_plan)} combos):")
        for c in cobertura_plan:
            legs = ' + '.join(f"{l['jugador']}@{l['cuota']:.2f}" for l in c['legs'])
            print(f"  │    [{c['piernas_n']}p @{c['cuota_combo']:.2f}]  ${c['stake']:>8,.0f}  →  {legs}")
        print(f"  │")
        total_ajustado_real = gastado_ind + gastado_cobertura
        print(f"  │  TOTAL EN RIESGO:  ${total_ajustado_real:>10,.0f}  ({total_ajustado_real/bankroll:.1%} bankroll)")
        print(f"  └" + "─" * 65)
        print()

    # ── 5. Resumen ──
    _print_resumen(
        bankroll, gastado_ind, gastado_combos, gastado_sistema, gastado_cobertura,
        senales_enriched, reporte, pool_size=len(pool) if args.cobertura else 0,
        superficie=args.superficie, torneo_tipo=args.torneo_tipo
    )

    # ── Nota de calibración ──
    cal_n = reporte['metadata'].get('calibracion_n', 0)
    print(f"  📊 Calibración: n={cal_n} validaciones. "
          f"{'⚠️  Prior uniforme activo — recalibrar con n≥30.' if cal_n < 30 else '✅ Suficiente n para Kelly real.'}")
    prior_src = f"calibrado {args.superficie}" if p_prior_efectivo != _P_PRIOR else "neutral (sin calibración)"
    print(f"  📊 p_blend usa Bayesian k=3 sobre prior {p_prior_efectivo:.3f} ({prior_src}) — "
          f"converge a p_modelo cuando n_h2h≥10.")
    print()

    # ── 5. Guardar plan JSON ──
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
                "modo": "cobertura" if args.cobertura else "normal",
                "torneo_tipo": args.torneo_tipo,
                "superficie": args.superficie,
                "combos": args.combos,
                "sistema": args.sistema,
                "ncombos": args.ncombos,
                "nsistema": args.nsistema,
                "watchlist": args.watchlist,
                "all_picks": args.all_picks,
                "piernas_min": piernas_min,
                "piernas_max": piernas_max,
                "excluir": args.excluir,
                "min_cuota": args.min_cuota,
                "kelly_fraction": KELLY_FRACTION,
                "budget_pct": {"individuales": BUDGET_IND_PCT, "combos": BUDGET_COMBO_PCT, "sistema": BUDGET_SIS_PCT},
            },
        },
        "individuales": [
            {
                "partido": s['partido'],
                "match_id": s.get('match_id', ''),
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
                "_waterfall": s.get('_waterfall', {}),
            }
            for s in senales_enriched
        ],
        "senales": senales_enriched,  # para update_trader_stakes (P54-02)
        "combos": combos_plan,
        "sistema": sistema_plan,
        "cobertura": cobertura_plan,
        "risk_management": risk_metrics if risk_metrics else {},
        "resumen": {
            "gastado_individuales": gastado_ind,
            "gastado_combos": gastado_combos,
            "gastado_sistema": gastado_sistema,
            "gastado_cobertura": gastado_cobertura,
            "total_en_riesgo": gastado_ind + gastado_combos + gastado_sistema + gastado_cobertura,
            "pct_bankroll_en_riesgo": round(
                (gastado_ind + gastado_combos + gastado_sistema + gastado_cobertura) / bankroll, 4),
            "n_senales_apostar": len(senales_raw),
            "pool_cobertura": len(pool) if args.cobertura else 0,
            "alerta_riesgo": (gastado_ind + gastado_combos + gastado_sistema + gastado_cobertura) / bankroll > 0.30,
        },
    }
    with open(plan_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"  💾 Plan guardado: {plan_file}")

    # ── P54-02 Parte 2: enriquecer shadow book con stakes reales ──
    try:
        from shadow_book import update_trader_stakes
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        n_upd = update_trader_stakes(fecha_hoy, {'senales': senales_enriched})
        if n_upd > 0:
            print(f"  📒 Shadow book actualizado: {n_upd} pick(s) con stake_real/var_flattened")
    except Exception as _sb_err:
        pass  # shadow book nunca bloquea el trader

    # ── Guardar reporte .txt (captura todo lo impreso en main) ──
    sys.stdout = sys.stdout._real
    _console_output = _buf.getvalue()
    txt_file = plan_file.replace('.json', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(_console_output)
    print(f"  📄 Reporte txt: {txt_file}")

    # ── 7. Enviar señales a Telegram ──
    if args.telegram:
        try:
            from utils.telegram import enviar_señales_trader
            ok = enviar_señales_trader(plan_file)
            if ok:
                print(f"  📱 Señales enviadas a Telegram ✅")
            else:
                print(f"  ⚠️ Error enviando señales a Telegram")
        except Exception as e:
            print(f"  ⚠️ Telegram no disponible: {e}")
    print()


if __name__ == '__main__':
    main()

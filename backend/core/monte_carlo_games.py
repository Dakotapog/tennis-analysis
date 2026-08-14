"""
core/monte_carlo_games.py — D160-02 (Nodo-160, completa D159-02)

Monte Carlo ligero condicionado a SERVICIO para el mercado de total de juegos.
Complementa (no reemplaza) el Gaussiano estático de
live_desk.py::_calcular_certeza_condicional() — se dispara solo cuando una
señal ya pasó los gates D151 y está a punto de disparar un combo real
(mismo patrón que validate_fillability en D159-04), no en cada refresh de 15s.

Refinamiento vs la firma propuesta en el spec (§3.3.4): se añaden
current_set_home/current_set_away — ya existen en score_data desde D153-01
(_parse_kambi_livedata_sets) — en vez de estimarlos por promedio, lo cual
habría introducido error innecesario cuando el dato real ya está disponible.

Riesgo declarado (§3.4 del nodo): p_hold_home/p_hold_away son proxies
(ranking+superficie), no estadísticas de saque en vivo — el payload de
livedata.json no fue confirmado a traer % de saque real. Si esa fuente
aparece en una sesión futura, debe preferirse sobre el proxy.
"""
import math
import random
from typing import Optional


def simular_total_juegos_condicionado(
    games_played: int,
    current_set_home: int,
    current_set_away: int,
    serving: str,
    sets_home: int,
    sets_away: int,
    p_hold_home: float,
    p_hold_away: float,
    linea: float,
    direccion: str,
    n_sims: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """
    Simula el resto del partido (best-of-3) desde el estado actual, juego a
    juego, muestreando cada hold como Bernoulli(p_hold del que sirve) y cada
    tiebreak como Bernoulli ajustado por fuerza relativa. `games_played` es
    el total de juegos YA jugados en el partido completo (mismo campo usado
    en score_data en todo el proyecto, ver D147/D150/D153) — se usa como
    base fija; solo los juegos restantes se simulan.

    Retorna p_condicional_mc (fracción de sims donde linea/direccion se
    cumple) + percentiles 10/90 del total final simulado. REPORTE_SOLO:
    no reemplaza el Gaussiano, no participa en ningún gate hasta que su
    propia hipótesis (H160-02) acumule evidencia.
    """
    direccion = (direccion or "UNDER").upper()
    if n_sims <= 0:
        return {
            "p_condicional_mc": None, "n_sims": 0, "media_total_juegos": None,
            "p10_total_juegos": None, "p90_total_juegos": None,
            "direccion": direccion, "linea": linea,
            "nota": "n_sims<=0 — sin simulación",
        }

    rng = random.Random(seed)
    p_hold_home = min(0.95, max(0.05, p_hold_home))
    p_hold_away = min(0.95, max(0.05, p_hold_away))
    totales = []

    for _ in range(n_sims):
        gh, ga = current_set_home, current_set_away
        sh, sa = sets_home, sets_away
        sirve_home = (serving == "home")
        extra_games = 0

        while sh < 2 and sa < 2:
            if sirve_home:
                home_gana_juego = rng.random() < p_hold_home
            else:
                home_gana_juego = rng.random() >= p_hold_away

            if home_gana_juego:
                gh += 1
            else:
                ga += 1
            extra_games += 1
            sirve_home = not sirve_home

            if gh == 6 and ga == 6:
                p_tb_home = min(0.95, max(0.05, 0.5 + (p_hold_home - p_hold_away) / 2.0))
                if rng.random() < p_tb_home:
                    sh += 1
                else:
                    sa += 1
                extra_games += 1  # el juego de tiebreak cuenta como 1 juego más
                gh = ga = 0
            elif (gh >= 6 or ga >= 6) and abs(gh - ga) >= 2:
                if gh > ga:
                    sh += 1
                else:
                    sa += 1
                gh = ga = 0

        totales.append(games_played + extra_games)

    n = len(totales)
    if direccion == "OVER":
        hits = sum(1 for t in totales if t > linea)
    else:
        hits = sum(1 for t in totales if t < linea)

    totales_sorted = sorted(totales)

    def _pct(p: float) -> int:
        idx = min(n - 1, max(0, int(p * n)))
        return totales_sorted[idx]

    # D167-04: error estándar e IC95% analíticos (CLT) — p_condicional_mc es
    # una proporción muestral de n_sims ensayos Bernoulli i.i.d., el error
    # estándar sale directo sin infraestructura nueva.
    p_hat = hits / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    ic95_low = round(max(0.0, p_hat - 1.96 * se), 4)
    ic95_high = round(min(1.0, p_hat + 1.96 * se), 4)

    return {
        "p_condicional_mc": round(p_hat, 4),
        "n_sims": n,
        "media_total_juegos": round(sum(totales) / n, 2),
        "p10_total_juegos": _pct(0.10),
        "p90_total_juegos": _pct(0.90),
        "direccion": direccion,
        "linea": linea,
        "se": round(se, 4),
        "ic95_low": ic95_low,
        "ic95_high": ic95_high,
        "nota": "MC ligero — p_hold proxy ranking/superficie, no serve-stats en vivo (D160-02 §3.4)",
    }


def estimar_p_hold(ranking: Optional[float], superficie: Optional[str]) -> float:
    """
    Proxy conservador de p_hold (spec §3.3.1): 0.62 base + ajuste_ranking +
    ajuste_superficie, acotado [0.50, 0.85]. NO es estadística de saque en
    vivo (ver riesgo declarado arriba) — si falta ranking o superficie, ese
    ajuste se omite en vez de fabricar el dato.
    """
    p = 0.62
    if superficie:
        s = superficie.strip().lower()
        if s.startswith("hierba"):
            p += 0.05
        elif s.startswith("arcilla"):
            p -= 0.04
    if ranking:
        r = float(ranking)
        if r <= 50:
            p += 0.05
        elif r <= 150:
            p += 0.02
        elif r > 400:
            p -= 0.03
    return min(0.85, max(0.50, round(p, 4)))

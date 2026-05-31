#!/usr/bin/env python3
"""
Edge Calculator — Kelly-KL Multidimensional (Nodo-01)

5 capas de inteligencia. Cada una conecta un campo distinto al tenis:

  L1 — Kelly-KL (Kullback-Leibler):
       Protección matemática contra ruina. KL(P_modelo || P_histórica) mide
       cuánto diverge el modelo de su historial — si diverge mucho, apuesta menos.

  L2 — Volatility Smile (Options Theory → Tennis):
       En opciones financieras, la vol implícita es mayor en OTM (underdogs).
       Aquí: lambda_aversion decrece cuando la cuota es alta (nuestro sweet spot).
       Evidencia: Tsitsipas 1.08 → edge -33% | Majchrzak 2.35 → edge +9.5%.

  L3 — Factor Decomposition (Fama-French → Tennis):
       El bookmaker ya tiene priced-in ranking+ELO. Nuestro alpha real está en
       surface_specialization + common_opponents + form_recent — factores que él
       NO modela eficientemente. phi_idiosincratico amplifica cuando el edge viene
       de esos factores ocultos.

  L4 — Shannon Entropy (Information Theory → Betting):
       H(bookmaker) = -p1*log(p1) - p2*log(p2). Alta entropía = bookmaker inseguro
       = más oportunidad. Implementa: "¿dónde no sabe el bookmaker?"

  L5 — Thompson Sampling (Reinforcement Learning → Bayesian Calibration):
       En lugar de p_historica fija (0.52), mantiene Beta(wins+1, losses+1) por
       superficie. Cada resultado validado por Nodo-05 actualiza la distribución.
       El sistema se auto-calibra → conecta Nodo-01 con Nodo-05 automáticamente.
"""

import json
import math
import re
import os
import sys
import argparse
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

CALIBRACION_FILE = "data/calibracion_edge.json"

CALIBRACION_DEFAULT = {
    "global": {"wins": 0, "losses": 0},
    "por_superficie": {
        "clay":    {"wins": 0, "losses": 0},
        "grass":   {"wins": 0, "losses": 0},
        "hard":    {"wins": 0, "losses": 0},
        "unknown": {"wins": 0, "losses": 0},
    },
    "por_zona": {
        "heavy_favorite":   {"wins": 0, "losses": 0},
        "moderate_favorite":{"wins": 0, "losses": 0},
        "slight_underdog":  {"wins": 0, "losses": 0},
        "underdog":         {"wins": 0, "losses": 0},
    },
    "nota": "Inicializado. Actualizar con validar_con_api.py (Nodo-05). Min n=30 para calibrar.",
    "ultima_actualizacion": None,
}

# Umbrales del edge decision
EDGE_MIN = 0.05          # 5% mínimo para considerar apuesta
KELLY_KL_MIN = 0.02      # 2% mínimo de Kelly-KL para confirmar
BANKROLL_CAP = 0.10      # máximo 10% por apuesta


# ─────────────────────────────────────────────────────────────────────────────
# L2 — VOLATILITY SMILE (Options Theory)
# ─────────────────────────────────────────────────────────────────────────────

def zona_cuota(cuota: float) -> str:
    """
    Clasifica la cuota del favorito predicho en zonas de oportunidad.
    Análogo al "moneyness" de las opciones financieras.

    Cuota baja = favorito obvio = bookmaker muy seguro = nuestra zona débil.
    Cuota alta = underdog = bookmaker incierto = nuestra zona fuerte.
    """
    if cuota < 1.30:
        return "heavy_favorite"    # ATM deep ITM option — bookmaker win
    elif cuota < 1.60:
        return "moderate_favorite" # ATM option — neutral
    elif cuota < 2.10:
        return "slight_underdog"   # Slightly OTM — slight edge possible
    else:
        return "underdog"          # OTM option — our sweet spot


def lambda_por_zona(zona: str) -> float:
    """
    Factor de aversión al riesgo (λ) ajustado por zona de cuota.

    Heavy favorite: λ=2.0 → Kelly-KL muy conservador (bookmaker sabe más que nosotros)
    Underdog:       λ=0.3 → Kelly-KL menos conservador (aquí vive nuestro edge real)

    Formula: f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica))
    """
    return {
        "heavy_favorite":    2.0,
        "moderate_favorite": 1.0,
        "slight_underdog":   0.5,
        "underdog":          0.3,
    }.get(zona, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# L4 — SHANNON ENTROPY (Information Theory)
# ─────────────────────────────────────────────────────────────────────────────

def bookmaker_entropy(cuota1: float, cuota2: float) -> float:
    """
    H(bookmaker) = -p1*log2(p1) - p2*log2(p2) en bits [0, 1]

    H→0: bookmaker muy seguro (ejemplo: Tsitsipas 1.08 vs Mochizuki 6.5 → H≈0.59)
    H→1: bookmaker incierto   (ejemplo: Majchrzak 2.35 vs Marozsan 1.62 → H≈0.97)

    Alta entropía = match coin-flip para el bookmaker = mayor oportunidad para un
    modelo que tiene información diferencial (Erdős transitivo, surface_spec).
    """
    if not cuota1 or not cuota2 or cuota1 <= 0 or cuota2 <= 0:
        return 0.5  # neutral sin datos
    eps = 1e-9
    p1_raw = 1.0 / cuota1
    p2_raw = 1.0 / cuota2
    total = p1_raw + p2_raw
    if total <= 0:
        return 0.5
    p1 = p1_raw / total  # normalizado sin vig
    p2 = p2_raw / total
    h = -(p1 * math.log2(p1 + eps) + p2 * math.log2(p2 + eps))
    return round(min(max(h, 0.0), 1.0), 4)


def psi_entropy_multiplier(entropy: float) -> float:
    """
    Convierte la entropía del bookmaker en un multiplicador del Kelly-KL.
    Rango: 0.85 (baja entropía, bookmaker seguro) → 1.15 (alta entropía, chance para nosotros).

    Lógica: cuando el bookmaker es incierto, nuestro modelo diferencial tiene más valor.
    """
    # Escalar linealmente: entropy=0 → 0.85, entropy=1 → 1.15
    return round(0.85 + entropy * 0.30, 4)


# ─────────────────────────────────────────────────────────────────────────────
# L3 — FACTOR DECOMPOSITION (Fama-French → Tennis)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_contribution(contrib_str) -> float:
    """'23.3%' → 0.233 | None/'' → 0.0"""
    if not contrib_str:
        return 0.0
    try:
        return float(str(contrib_str).replace('%', '').strip()) / 100.0
    except (ValueError, AttributeError):
        return 0.0


def phi_idiosincratico(score_breakdown: dict, player_key: str = 'player1') -> float:
    """
    Factor de alpha idiosincratico: qué fracción de la predicción viene de factores
    que el bookmaker NO modela eficientemente.

    Bookmaker-known  (ya priced-in):    elo_rating + ranking_momentum
    Bookmaker-unknown (nuestro edge):   surface_specialization + form_recent
                                        + common_opponents + h2h_direct

    phi = 0.80 (predicción 100% ranking-driven, bookmaker ya lo sabe)
    phi = 1.30 (predicción 100% factores no modelados por bookmaker)

    Majchrzak ejemplo:
      elo=19.3% + ranking=33.3% = 52.6% known
      form=23.2% + common=24.1% = 47.3% unknown
      phi = 0.80 + (0.473 / 1.0) * 0.50 = 1.037 (ligera amplificación)
    """
    player_data = score_breakdown.get(player_key, {})
    if not player_data:
        return 1.0  # neutral sin datos

    # Factores que el bookmaker modela bien
    known_factors = ['elo_rating', 'ranking_momentum']
    # Factores donde tenemos ventaja diferencial
    unknown_factors = ['surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct']

    weight_known = sum(
        _parse_contribution(player_data.get(f, {}).get('contribution'))
        for f in known_factors
    )
    weight_unknown = sum(
        _parse_contribution(player_data.get(f, {}).get('contribution'))
        for f in unknown_factors
    )

    total = weight_known + weight_unknown
    if total <= 0:
        return 1.0

    frac_unknown = weight_unknown / total
    phi = 0.80 + frac_unknown * 0.50
    return round(phi, 4)


def elo_win_prob(elo_p1: float, elo_p2: float) -> float:
    """
    P(p1 gana) usando la fórmula estándar ELO.
    1/(1 + 10^((elo2 - elo1)/400))

    Permite calcular la "baseline ELO" para comparar con p_modelo.
    La diferencia = alpha más allá del ELO puro.
    """
    if not elo_p1 or not elo_p2:
        return 0.5
    return round(1.0 / (1.0 + 10.0 ** ((elo_p2 - elo_p1) / 400.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# L5 — THOMPSON SAMPLING / BAYESIAN CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def cargar_calibracion() -> dict:
    """Carga el estado Beta(wins, losses) desde disco. Crea default si no existe."""
    if os.path.exists(CALIBRACION_FILE):
        try:
            with open(CALIBRACION_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return dict(CALIBRACION_DEFAULT)


def guardar_calibracion(state: dict):
    """Persiste el estado de calibración (llamado por validar_con_api.py)."""
    os.makedirs("data", exist_ok=True)
    state['ultima_actualizacion'] = datetime.now().isoformat()
    with open(CALIBRACION_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def thompson_p_historica(wins: int, losses: int) -> float:
    """
    Mean de Beta(α=wins+1, β=losses+1) = α/(α+β).

    Con 0 datos: Beta(1,1) → mean=0.5 (prior uniforme — máxima incertidumbre)
    Con 15/20:   Beta(16,6) → mean=0.727 (modelo aprendiendo)
    Con 9/19:    Beta(10,11) → mean=0.476 (datos sucios Jan 2026 — no usar)

    Nota: Thompson Sampling "real" samplea de la distribución (random.betavariate)
    para explorar. Aquí usamos la media (explotación pura) hasta n≥30.
    """
    alpha = wins + 1
    beta_param = losses + 1
    return round(alpha / (alpha + beta_param), 4)


def theta_thompson(calibracion: dict, superficie: str) -> float:
    """
    Obtiene p_historica desde el estado Bayesiano.
    Prioriza calibración por superficie si n≥10, si no usa global.
    """
    sup_state = calibracion.get('por_superficie', {}).get(superficie, {"wins": 0, "losses": 0})
    n_sup = sup_state['wins'] + sup_state['losses']

    if n_sup >= 10:
        return thompson_p_historica(sup_state['wins'], sup_state['losses'])

    # Fallback: calibración global
    glob = calibracion.get('global', {"wins": 0, "losses": 0})
    return thompson_p_historica(glob['wins'], glob['losses'])


def actualizar_calibracion(superficie: str, zona: str, correcto: bool):
    """
    Actualiza Beta(wins, losses) con un resultado nuevo.
    Llamar desde validar_con_api.py (Nodo-05) después de cada partido validado.
    """
    state = cargar_calibracion()

    # Update global
    if correcto:
        state['global']['wins'] += 1
    else:
        state['global']['losses'] += 1

    # Update por superficie
    if superficie not in state['por_superficie']:
        state['por_superficie'][superficie] = {"wins": 0, "losses": 0}
    if correcto:
        state['por_superficie'][superficie]['wins'] += 1
    else:
        state['por_superficie'][superficie]['losses'] += 1

    # Update por zona
    if zona not in state['por_zona']:
        state['por_zona'][zona] = {"wins": 0, "losses": 0}
    if correcto:
        state['por_zona'][zona]['wins'] += 1
    else:
        state['por_zona'][zona]['losses'] += 1

    guardar_calibracion(state)
    n_total = state['global']['wins'] + state['global']['losses']
    logger.info(f"Calibración actualizada: {state['global']} | n={n_total}")


# ─────────────────────────────────────────────────────────────────────────────
# L1 — KELLY-KL CORE
# ─────────────────────────────────────────────────────────────────────────────

def calcular_edge(
    p_modelo: float,
    cuota_favorito: float,
    p_historica: float = 0.50,
    lambda_aversion: float = 0.5,
    phi: float = 1.0,
    psi: float = 1.0,
) -> dict:
    """
    Core Kelly-KL con todas las capas de ajuste.

    Args:
        p_modelo:        confianza del modelo (0-1)
        cuota_favorito:  cuota decimal del jugador favorito predicho
        p_historica:     accuracy histórica del modelo — de Thompson Sampling (L5)
        lambda_aversion: ajustado por zona (L2) — cuánto penalizar la divergencia
        phi:             factor idiosincratico Fama-French (L3)
        psi:             multiplicador de entropía Shannon (L4)

    Formula:
        p_implicita = 1 / cuota_favorito
        edge        = p_modelo - p_implicita
        KL          = p_modelo×log(p_modelo/p_historica) + (1-p_modelo)×log((1-p_modelo)/(1-p_historica))
        f*_clásico  = edge / (1 - p_implicita)
        f*_KL       = f*_clásico × exp(-λ × max(0, KL)) × phi × psi
        fraccion    = min(f*_KL, 0.10)   [cap 10%]
        apostar     = edge>5% AND f*_KL>2%
    """
    eps = 1e-9
    p_implicita = 1.0 / max(cuota_favorito, 1.001)
    edge = p_modelo - p_implicita

    # KL divergence: KL(P_modelo || P_histórica)
    kl = (
        p_modelo * math.log((p_modelo + eps) / (p_historica + eps))
        + (1 - p_modelo) * math.log((1 - p_modelo + eps) / (1 - p_historica + eps))
    )

    # Kelly clásico
    denominador = 1 - p_implicita
    kelly_clasico = edge / denominador if denominador > 0 else 0.0

    # Kelly-KL: penalizar según divergencia del modelo
    kelly_kl = kelly_clasico * math.exp(-lambda_aversion * max(0.0, kl))

    # Aplicar factores idiosincratico (L3) y entropía (L4)
    kelly_kl_ajustado = kelly_kl * phi * psi

    # Decisión de apuesta
    apostar = edge > EDGE_MIN and kelly_kl_ajustado > KELLY_KL_MIN

    return {
        'p_modelo':        round(p_modelo, 4),
        'p_implicita':     round(p_implicita, 4),
        'edge':            round(edge, 4),
        'edge_pct':        f"{edge*100:.1f}%",
        'kl_divergencia':  round(kl, 6),
        'kelly_clasico':   round(kelly_clasico, 4),
        'kelly_kl_base':   round(kelly_kl, 4),
        'kelly_kl':        round(kelly_kl_ajustado, 4),
        'fraccion_bankroll': round(min(kelly_kl_ajustado, BANKROLL_CAP), 4),
        'apostar':         apostar,
        # capas de ajuste
        'phi_idiosincratico': round(phi, 4),
        'psi_entropia':       round(psi, 4),
        'lambda_aversion':    round(lambda_aversion, 4),
        'p_historica_usada':  round(p_historica, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: IDENTIFICAR DATOS DEL PARTIDO
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """'Tsitsipas S.' → 'Tsitsipas_S'"""
    return re.sub(r'[\s.]+', '_', name).strip('_')


def _get_elo(ranking_analysis: dict, player_name: str) -> Optional[float]:
    """Extrae ELO de ranking_analysis usando el nombre del jugador."""
    key = f"{_sanitize_name(player_name)}_elo"
    val = ranking_analysis.get(key)
    return float(val) if val is not None else None


def _get_ranking(ranking_analysis: dict, player_name: str) -> Optional[int]:
    """Extrae ranking ATP/WTA de ranking_analysis."""
    key = f"{_sanitize_name(player_name)}_ranking"
    val = ranking_analysis.get(key)
    return int(val) if val is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO POR PARTIDO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_edge_completo(partido: dict, calibracion: dict) -> Optional[dict]:
    """
    Aplica las 5 capas de inteligencia a un partido del h2h_results_enhanced.
    Retorna None si faltan datos mínimos (predicción o cuotas).
    """
    jugador1 = partido.get('jugador1', '')
    jugador2 = partido.get('jugador2', '')
    cuota1 = partido.get('cuota1')
    cuota2 = partido.get('cuota2')

    ra = partido.get('ranking_analysis', {})
    pred = ra.get('prediction', {})
    favored = pred.get('favored_player')
    confidence = pred.get('confidence')

    # Validación mínima
    if not favored or confidence is None:
        return None
    if not cuota1 or not cuota2:
        return None

    # Probabilidad del modelo (confidence viene en 0-100)
    p_modelo = confidence / 100.0

    # Determinar cuota del favorito predicho
    if favored == jugador1:
        cuota_fav = cuota1
        cuota_rival = cuota2
        player_key_sb = 'player1'
        elo_fav    = _get_elo(ra, jugador1)
        elo_rival  = _get_elo(ra, jugador2)
        rank_fav   = _get_ranking(ra, jugador1)
        rank_rival = _get_ranking(ra, jugador2)
    elif favored == jugador2:
        cuota_fav = cuota2
        cuota_rival = cuota1
        player_key_sb = 'player2'
        elo_fav    = _get_elo(ra, jugador2)
        elo_rival  = _get_elo(ra, jugador1)
        rank_fav   = _get_ranking(ra, jugador2)
        rank_rival = _get_ranking(ra, jugador1)
    else:
        return None  # nombre no coincide exactamente — skip

    # ─── L2: Volatility Smile ───────────────────────────────
    zona = zona_cuota(cuota_fav)
    lambda_av = lambda_por_zona(zona)

    # ─── L3: Factor Decomposition ───────────────────────────
    sb = pred.get('score_breakdown', {})
    phi = phi_idiosincratico(sb, player_key=player_key_sb)

    # ELO baseline para contexto
    p_elo_base = elo_win_prob(elo_fav or 1500, elo_rival or 1500)
    alpha_idiosincratico = p_modelo - p_elo_base  # lo que va más allá del ELO puro

    # ─── L4: Shannon Entropy ────────────────────────────────
    entropy = bookmaker_entropy(cuota1, cuota2)
    psi = psi_entropy_multiplier(entropy)

    # ─── L5: Thompson Sampling ──────────────────────────────
    superficie = partido.get('superficie', 'unknown')
    p_hist = theta_thompson(calibracion, superficie)

    # ─── L1: Kelly-KL Core ──────────────────────────────────
    resultado = calcular_edge(
        p_modelo=p_modelo,
        cuota_favorito=cuota_fav,
        p_historica=p_hist,
        lambda_aversion=lambda_av,
        phi=phi,
        psi=psi,
    )

    # ─── Metadata adicional ─────────────────────────────────
    torneo = partido.get('torneo_nombre') or partido.get('torneo') or 'Desconocido'

    resultado.update({
        'partido':              f"{jugador1} vs {jugador2}",
        'favorito_predicho':    favored,
        'cuota_favorito':       cuota_fav,
        'cuota_rival':          cuota_rival,
        'torneo':               torneo,
        'superficie':           superficie,
        'zona_cuota':           zona,
        'entropy_bookmaker':    round(entropy, 4),
        'p_elo_base':           p_elo_base,
        'alpha_vs_elo':         round(alpha_idiosincratico, 4),
        'elo_favorito':         elo_fav,
        'elo_rival':            elo_rival,
        'ranking_favorito':     rank_fav,
        'ranking_rival':        rank_rival,
        'match_url':            partido.get('match_url'),
        'match_id':             partido.get('match_id'),
        # Historial H2H directo — alimenta p_blend Bayesiano en trader_ev_tenis.py
        'n_h2h':                len([m for m in partido.get('enfrentamientos_directos', []) if isinstance(m, dict)]),
        # Contexto Markov (si ya está disponible en el partido)
        'markov_favorito':      partido.get('markov_analysis', {}).get('jugador1' if player_key_sb == 'player1' else 'jugador2', {}).get('estado_actual'),
    })

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def procesar_archivo_h2h(h2h_file: str, output_file: Optional[str] = None) -> dict:
    """
    Lee h2h_results_enhanced_FECHA.json y calcula el edge para todos los partidos.
    Genera un reporte de apuestas con las 5 capas activadas.
    """
    logger.info(f"📂 Cargando: {h2h_file}")
    with open(h2h_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Normalizar estructura (lista directa o dict con 'partidos')
    if isinstance(raw, list):
        partidos = raw
    elif isinstance(raw, dict) and 'partidos' in raw:
        partidos = raw['partidos']
    else:
        partidos = list(raw.values()) if raw else []
        if partidos and isinstance(partidos[0], list):
            # dict de listas (agrupado por torneo)
            partidos = [p for sublist in partidos for p in sublist]

    calibracion = cargar_calibracion()

    resultados = []
    sin_datos = []

    for p in partidos:
        r = calcular_edge_completo(p, calibracion)
        if r is None:
            sin_datos.append({
                'partido': f"{p.get('jugador1','?')} vs {p.get('jugador2','?')}",
                'razon': 'Sin predicción o sin cuotas'
            })
            continue
        resultados.append(r)

    # Separar por decisión
    apostar_lista = [r for r in resultados if r['apostar']]
    no_apostar_lista = [r for r in resultados if not r['apostar'] and r['edge'] > 0]
    edge_negativo = [r for r in resultados if r['edge'] <= 0]

    # Ordenar por kelly_kl descendente
    apostar_lista.sort(key=lambda x: -x['kelly_kl'])
    no_apostar_lista.sort(key=lambda x: -x['edge'])

    # Estadísticas
    n_total = len(resultados)
    edges_positivos = [r['edge'] for r in resultados if r['edge'] > 0]
    n_calibracion = calibracion['global']['wins'] + calibracion['global']['losses']

    output = {
        'metadata': {
            'fecha':          datetime.now().isoformat(),
            'fuente':         h2h_file,
            'n_procesados':   n_total,
            'n_sin_datos':    len(sin_datos),
            'n_edge_positivo': len(edges_positivos),
            'n_apostar':      len(apostar_lista),
            'calibracion_n':  n_calibracion,
            'calibracion_nota': 'Prior uniforme hasta n≥30' if n_calibracion < 30 else 'Calibrado',
        },
        'apostar': apostar_lista,
        'watchlist': no_apostar_lista[:10],   # edge positivo pero bajo threshold
        'sin_edge': edge_negativo[:5],         # sample de edge negativo
        'sin_datos': sin_datos[:5],
    }

    # Mostrar resumen
    _print_resumen(apostar_lista, no_apostar_lista, n_total, len(sin_datos))

    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Reporte guardado: {output_file}")

    return output


def _print_resumen(apostar: list, watchlist: list, n_total: int, n_sin_datos: int):
    print("\n" + "═"*60)
    print("  EDGE CALCULATOR — REPORTE")
    print("═"*60)
    print(f"  Partidos analizados: {n_total}  |  Sin datos: {n_sin_datos}")
    print(f"  Edge > 5% + Kelly-KL > 2%:  {len(apostar)} apuesta(s)")
    print(f"  Watchlist (edge+, bajo thresh): {len(watchlist)}")
    print()

    if apostar:
        print("  ✅ APOSTAR:")
        for r in apostar:
            print(f"     {r['favorito_predicho']} en {r['partido']}")
            print(f"       Edge: {r['edge_pct']} | Kelly-KL: {r['kelly_kl']*100:.1f}%")
            print(f"       Cuota: {r['cuota_favorito']} | φ: {r['phi_idiosincratico']} | H: {r['entropy_bookmaker']:.2f} | Zona: {r['zona_cuota']}")
            print(f"       Bankroll: {r['fraccion_bankroll']*100:.1f}% | Sup: {r['superficie']}")
            print()
    else:
        print("  ⚠️  Sin apuestas hoy (edge < 5% o Kelly-KL < 2% en todos los partidos)")
        print()

    if watchlist:
        print("  👀 WATCHLIST (edge positivo, bajo threshold):")
        for r in watchlist[:5]:
            print(f"     {r['favorito_predicho']}: edge={r['edge_pct']} kelly={r['kelly_kl']*100:.1f}% zona={r['zona_cuota']}")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _ultimo_h2h_file() -> Optional[str]:
    """Encuentra el h2h_results_enhanced más reciente en reports/."""
    import glob as glob_mod
    archivos = sorted(glob_mod.glob("reports/h2h_results_enhanced_*.json"))
    return archivos[-1] if archivos else None


def main():
    parser = argparse.ArgumentParser(
        description="Edge Calculator — Kelly-KL Multidimensional"
    )
    parser.add_argument(
        '--h2h', type=str, default=None,
        help="Ruta al h2h_results_enhanced_FECHA.json (default: más reciente en reports/)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Ruta del JSON de salida (default: reports/edge_report_HOY.json)"
    )
    parser.add_argument(
        '--actualizar-calibracion', action='store_true',
        help="Muestra instrucciones para actualizar la calibración"
    )
    args = parser.parse_args()

    if args.actualizar_calibracion:
        print("Para actualizar la calibración, llama desde validar_con_api.py:")
        print("  from edge_calculator import actualizar_calibracion")
        print("  actualizar_calibracion(superficie='clay', zona='underdog', correcto=True)")
        return

    h2h_file = args.h2h or _ultimo_h2h_file()
    if not h2h_file:
        logger.error("No se encontró ningún h2h_results_enhanced_*.json en reports/")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"reports/edge_report_{ts}.json"

    procesar_archivo_h2h(h2h_file, output_file)


if __name__ == "__main__":
    main()

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
from config import detectar_tier  # T21-02: fuente única de verdad para tier
from analysis.markov_analyzer import calcular_recencia_regimen, factor_alpha_temporal  # T18-03 (Nodo-18)
from analysis.rivalry_analyzer import RIVALRY_VERSION as _EXPECTED_RIVALRY_VERSION  # Nodo-32 Fase 3

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
# T32-01 (Nodo-32): umbral p_modelo para underdogs — evita phantom edge
# Un pick con p_modelo=0.503 y cuota=3.60 produce edge=22.5% matemático pero el
# modelo expresa convicción de moneda al aire. Cuota >= 2.10 requiere MODERATE+.
P_MODELO_MIN_UNDERDOG = 0.55  # alineado con confidence_flag MODERATE
# Nodo-32 Acción 3: versión del gate serializada en cada edge_report.
# Incrementar en cada cambio de gate (P_MODELO_MIN_UNDERDOG, EDGE_MIN, KELLY_KL_MIN,
# golden_zone conditions). betplay_combo_builder.py rechaza archivos con versión distinta.
GATE_VERSION = "nodo32-fase2"

# T17-03: λ escalado por tier (Nodo-17)
# Grand Slam: modelo calibrado n=31, señal limpia → λ base 0.5
# Challenger: H2H escaso + mercado ineficiente → incertidumbre 3.6× mayor
LAMBDA_TIER_MULTIPLIER = {
    "grand_slam":  1.0,   # λ efectivo = λ_zona × 1.0 (base)
    "atp1000":     1.6,
    "atp500":      2.4,
    "challenger":  3.6,
    "itf":         4.5,   # máxima incertidumbre: mercado casi inexistente
}


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


def data_completeness(score_breakdown: dict, player_key: str = 'player1') -> float:
    """
    Fracción de componentes con datos reales (contribution > 0) sobre el total.

    Los 8 componentes del rivalry_analyzer tienen peso distinto pero cada uno
    con contribution=0% indica hueco de datos, no edge cero.

    Componentes con peso alto que suelen ser 0 cuando faltan datos:
      form_recent (28%)  — sin partidos recientes en FlashScore
      surface_spec (15%) — sin historial en esa superficie
      common_opponents   — sin rivales comunes encontrados
      h2h_direct         — primer enfrentamiento

    Retorna:
      1.0  → todos los componentes tienen datos
      0.5  → la mitad sin datos → kelly_kl ya inflado artificialmente
      0.0  → sin score_breakdown (no debería pasar)

    Uso: kelly_kl_ajustado = kelly_kl * sqrt(data_completeness)
    No se aplica automáticamente — se expone como campo para que el trader
    pueda filtrar con --excluir o el usuario decida manualmente.
    """
    COMPONENTES = [
        'surface_specialization', 'form_recent', 'common_opponents',
        'h2h_direct', 'ranking_momentum', 'elo_rating',
        'home_advantage', 'strength_of_schedule',
    ]
    player_data = score_breakdown.get(player_key, {})
    if not player_data:
        return 0.0

    con_datos = sum(
        1 for c in COMPONENTES
        if _parse_contribution(player_data.get(c, {}).get('contribution')) > 0.0
    )
    return round(con_datos / len(COMPONENTES), 3)


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



def theta_thompson(calibracion: dict, superficie: str, tier: str = 'grand_slam') -> float:
    """
    Obtiene p_historica estratificada por [tier][superficie] (T17-02/T17-03).
    Jerarquía: [superficie_tier] n≥10 → fallback_por_tier → por_superficie n≥10 → global.

    B-08 fix: cuando fallback_por_tier se usa (paso 2) y por_superficie tiene n≥10,
    aplica min(fallback_tier, p_superficie) si divergen > 0.03.
    Previene optimismo artificial cuando el tier fallback ignora datos de superficie.
    Ejemplo: grass real=0.569, atp500 fallback=0.650 → min=0.569.
    """
    # 1. Intentar calibración estratificada [superficie_tier]
    key = f"{superficie}_{tier}"
    tier_state = calibracion.get('por_superficie_y_tier', {}).get(key, {"wins": 0, "losses": 0})
    # FIX-5: preferir era_v2 cuando era_v2_n >= 10 (datos post-normalización-fix 2026-06-19)
    _ev2_w = tier_state.get('era_v2_wins', 0)
    _ev2_l = tier_state.get('era_v2_losses', 0)
    _ev2_n = _ev2_w + _ev2_l
    if _ev2_n >= 10:
        return thompson_p_historica(_ev2_w, _ev2_l)
    n_tier = tier_state['wins'] + tier_state['losses']
    if n_tier >= 10:
        return thompson_p_historica(tier_state['wins'], tier_state['losses'])

    # 2. Fallback por tier (valores calibrados offline en Nodo-17)
    #    B-08: clamp con p_superficie si disponible y diverge > 0.03
    fallback = calibracion.get('fallback_por_tier', {}).get(tier)
    if fallback is not None:
        p_tier = round(float(fallback), 4) if not isinstance(fallback, dict) else thompson_p_historica(fallback.get('wins', 0), fallback.get('losses', 0))
        # B-08: check surface data for conservative clamping
        sup_state = calibracion.get('por_superficie', {}).get(superficie, {"wins": 0, "losses": 0})
        n_sup = sup_state.get('wins', 0) + sup_state.get('losses', 0)
        if n_sup >= 10:
            p_sup = thompson_p_historica(sup_state['wins'], sup_state['losses'])
            if p_tier - p_sup > 0.03:
                return round(min(p_tier, p_sup), 4)
        return p_tier

    # 3. Fallback por superficie (calibración anterior al Nodo-17)
    sup_state = calibracion.get('por_superficie', {}).get(superficie, {"wins": 0, "losses": 0})
    n_sup = sup_state['wins'] + sup_state['losses']
    if n_sup >= 10:
        return thompson_p_historica(sup_state['wins'], sup_state['losses'])

    # 4. Prior neutro
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
    n_calibracion: int = 0,
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
        n_calibracion:   B-10: n de datos en por_superficie_y_tier (para shrinkage)

    Formula:
        p_implicita = 1 / cuota_favorito
        edge        = p_modelo - p_implicita
        KL          = p_modelo×log(p_modelo/p_historica) + (1-p_modelo)×log((1-p_modelo)/(1-p_historica))
        f*_clásico  = edge / (1 - p_implicita)
        f*_KL       = f*_clásico × exp(-λ × max(0, KL)) × phi × psi × ccf
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

    # B-10: Calibration confidence factor (James-Stein shrinkage on Kelly)
    # n_calibracion = n from por_superficie_y_tier for this specific combo
    # When n=0 → ccf=0.30 (floor: only prior, high uncertainty → reduce Kelly 70%)
    # When n=20 → ccf=0.50
    # When n=40 → ccf=0.67
    # When n=100 → ccf=0.83
    # Floor 0.30 prevents total suppression — keeps signal alive at reduced size
    _CCF_KAPPA = 20  # half-life: n=20 → factor=0.50
    _CCF_FLOOR = 0.30
    calibration_confidence = max(_CCF_FLOOR, n_calibracion / (n_calibracion + _CCF_KAPPA))
    kelly_kl_ajustado = kelly_kl_ajustado * calibration_confidence

    # Decisión de apuesta
    # T32-01 (Nodo-32): underdogs (cuota >= 2.10) requieren p_modelo >= 0.55
    # Favoritos y slight_underdogs pasan sin restricción adicional — su edge no
    # puede ser "fantasma" porque cuota baja acota el gap p_modelo - p_implicita.
    # T33-01 (Nodo-33): el bloqueo n_h2h=0 se aplica en calcular_edge_completo()
    # después de que _n_h2h_v = resultado['n_h2h'] esté disponible — ver línea ~800.
    apostar = (
        edge > EDGE_MIN
        and kelly_kl_ajustado > KELLY_KL_MIN
        and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
    )

    # B-09: Confidence flag — classify conviction level of p_modelo
    # STRONG (p>=0.60): high conviction, full sizing
    # MODERATE (p>=P_MODELO_MIN_UNDERDOG): decent conviction, aligned with gate
    # LOW (p<P_MODELO_MIN_UNDERDOG): edge may come from extreme odds, not model conviction
    # T32-03: reutiliza P_MODELO_MIN_UNDERDOG para eliminar drift — si se recalibra el threshold,
    # ambos gates (apostar + confidence_flag) se actualizan automáticamente
    if p_modelo >= 0.60:
        confidence_flag = 'STRONG'
    elif p_modelo >= P_MODELO_MIN_UNDERDOG:
        confidence_flag = 'MODERATE'
    else:
        confidence_flag = 'LOW'

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
        'confidence_flag':    confidence_flag,  # B-09: STRONG/MODERATE/LOW
        'calibration_confidence': round(calibration_confidence, 4),  # B-10
        'n_calibracion':     n_calibracion,  # B-10: transparency
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodo-28 Fase 2: Triple Alignment Score — Information Asymmetry Metamodel
# ─────────────────────────────────────────────────────────────────────────────

# Umbrales calibrados con caso Eala @5.20 (alignment=0.861 → STRUCTURAL_ALPHA)
_SURFACE_SIGNAL_CAP  = 0.25   # alpha_vs_elo techo práctico (22.4% = 0.896 norm)
_BBI_CAP             = 0.70   # BBI techo práctico (Eala 0.673 = 0.961 norm)
_AXIS_THRESHOLD      = 0.50   # eje activo si norm > 0.50
_ALIGNMENT_STRONG    = 0.40   # triple alignment mínimo para STRUCTURAL_ALPHA
_ALIGNMENT_PARTIAL   = 0.20   # mínimo para PARTIAL_ALIGNMENT

def triple_alignment_score(pick: dict) -> dict:
    """
    M-28-4 (Nodo-28 Fase 2): detecta alineación de las 3 fuentes de
    information asymmetry que el bookmaker no puede ver simultáneamente.

    Ejes:
      Surface Blindness  — el modelo ve algo en superficie que el ELO no ve
      Regime Blindness   — Markov detecta momentum divergente
      Bookmaker Blindness — BBI mide ceguera directa del bookmaker

    Cuando los 3 se alinean → STRUCTURAL_ALPHA: el pick tiene alpha real
    aunque confidence_flag sea LOW (p_modelo < 0.55).

    Retroactivo Eala @5.20 vs Rybakina:
      surface_norm = 0.896 | regime_norm = 1.00 | bbi_norm = 0.961
      alignment = 0.861 → STRUCTURAL_ALPHA (3/3 ejes)
    """
    # Eje 1 — Surface Blindness
    alpha = abs(pick.get('alpha_vs_elo') or 0.0)
    surface_norm = min(alpha / _SURFACE_SIGNAL_CAP, 1.0)

    # Eje 2 — Regime Blindness (Markov HOT + delta win-rate divergence)
    regime_raw = 0.0
    if pick.get('markov_favorito') == 'HOT':
        regime_raw += 0.5
    if (pick.get('delta_wr_markov') or 0.0) > 0.15:
        regime_raw += 0.5
    regime_norm = min(regime_raw, 1.0)

    # Eje 3 — Bookmaker Blindness
    bbi = pick.get('bbi') or 0.0
    bbi_norm = min(bbi / _BBI_CAP, 1.0)

    alignment = round(surface_norm * regime_norm * bbi_norm, 4)

    n_axes = sum([
        surface_norm > _AXIS_THRESHOLD,
        regime_norm  > _AXIS_THRESHOLD,
        bbi_norm     > _AXIS_THRESHOLD,
    ])

    if n_axes == 3 and alignment >= _ALIGNMENT_STRONG:
        flag = 'STRUCTURAL_ALPHA'
    elif n_axes >= 2 and alignment >= _ALIGNMENT_PARTIAL:
        flag = 'PARTIAL_ALIGNMENT'
    else:
        flag = 'NO_ALIGNMENT'

    # FIX-4 (Nodo-28 Fase 2): CONTESTED_ALPHA — si el rival también está HOT,
    # la ventaja informacional de régimen es bilateral → no hay asimetría real.
    # Solo aplica cuando flag=STRUCTURAL_ALPHA (los 3 ejes activos del favorito).
    regime_raw_dog = 0.0
    if pick.get('markov_rival') == 'HOT':
        regime_raw_dog += 0.5
    if (pick.get('delta_wr_rival') or 0.0) > 0.15:
        regime_raw_dog += 0.5
    regime_norm_dog = min(regime_raw_dog, 1.0)

    # alignment del rival usa el mismo surface_norm y bbi_norm (match-level)
    # pero el regime_norm del rival en lugar del favorito
    alignment_dog = round(surface_norm * regime_norm_dog * bbi_norm, 4)
    net_alignment = round(alignment - alignment_dog, 4)

    # REGLA-N28-F2-2: STRUCTURAL_ALPHA solo si net_alignment > 0.25
    if flag == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
        flag = 'CONTESTED_ALPHA'

    return {
        'triple_alignment': alignment,
        'alignment_flag':   flag,
        'n_axes_active':    n_axes,
        'surface_signal':   round(surface_norm, 3),
        'regime_signal':    round(regime_norm, 3),
        'bbi_signal':       round(bbi_norm, 3),
        'net_alignment':    net_alignment,   # FIX-4
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

    # ─── T17-03: Escalar λ por tier (Nodo-17) ───────────────
    torneo_completo = partido.get('torneo_completo', '') or partido.get('torneo_nombre', '') or ''
    tier = detectar_tier(torneo_completo)
    lambda_av = lambda_av * LAMBDA_TIER_MULTIPLIER.get(tier, 1.0)

    # ─── T18-03: PELT Recency Alpha (Nodo-18) ────────────────
    # HOT fresco (≤3 partidos) → bookmaker stale → λ reducido (÷1.20) → más confiado
    # COLD fresco               → precaución amplificada → λ aumentado (÷0.85)
    _markov_key = 'jugador1' if player_key_sb == 'player1' else 'jugador2'
    _markov_fav = (pred.get('markov_analysis') or {}).get(_markov_key, {})
    _recencia_info = calcular_recencia_regimen(_markov_fav)
    _delta_wr = round(
        _markov_fav.get('win_rate_reciente', 0.5) - _markov_fav.get('win_rate_anterior', 0.5), 3
    )
    # FIX-4 (S-4): rival Markov state para CONTESTED_ALPHA check
    _markov_rival_key = 'jugador2' if player_key_sb == 'player1' else 'jugador1'
    _markov_rival = (pred.get('markov_analysis') or {}).get(_markov_rival_key, {})
    _delta_wr_rival = round(
        _markov_rival.get('win_rate_reciente', 0.5) - _markov_rival.get('win_rate_anterior', 0.5), 3
    )
    _alpha_factor = factor_alpha_temporal(
        _recencia_info['recencia'],
        _markov_fav.get('estado_actual', 'NEUTRAL'),
        _delta_wr,
    )
    lambda_av = lambda_av / _alpha_factor

    # ─── L3: Factor Decomposition ───────────────────────────
    sb = pred.get('score_breakdown', {})
    phi = phi_idiosincratico(sb, player_key=player_key_sb)
    completeness = data_completeness(sb, player_key=player_key_sb)

    # ELO baseline para contexto
    p_elo_base = elo_win_prob(elo_fav or 1500, elo_rival or 1500)
    alpha_idiosincratico = p_modelo - p_elo_base  # lo que va más allá del ELO puro

    # ─── L4: Shannon Entropy ────────────────────────────────
    entropy = bookmaker_entropy(cuota1, cuota2)
    psi = psi_entropy_multiplier(entropy)

    # ─── L5: Thompson Sampling estratificado (T17-02/T17-03) ─
    superficie = partido.get('superficie') or partido.get('tipo_cancha') or 'unknown'
    if superficie in ('N/A', 'Desconocida', None):
        superficie = 'unknown'
    p_hist = theta_thompson(calibracion, superficie, tier)

    # B-10: n_calibracion for James-Stein shrinkage on Kelly
    _cal_key = f"{superficie}_{tier}"
    _cal_state = calibracion.get('por_superficie_y_tier', {}).get(_cal_key, {"wins": 0, "losses": 0})
    _n_cal = _cal_state.get('wins', 0) + _cal_state.get('losses', 0)

    # ─── L1: Kelly-KL Core ──────────────────────────────────
    resultado = calcular_edge(
        p_modelo=p_modelo,
        cuota_favorito=cuota_fav,
        p_historica=p_hist,
        lambda_aversion=lambda_av,
        phi=phi,
        psi=psi,
        n_calibracion=_n_cal,
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
        'tier':                 tier,
        'lambda_efectivo':      round(lambda_av, 4),
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
        'cuota_es_real':        partido.get('cuota_es_real', True),
        # Historial H2H directo — alimenta p_blend Bayesiano en trader_ev_tenis.py
        'n_h2h':                len([m for m in partido.get('enfrentamientos_directos', []) if isinstance(m, dict)]),
        # Contexto Markov (si ya está disponible en el partido)
        'markov_favorito':      _markov_fav.get('estado_actual'),
        'markov_rival':         _markov_rival.get('estado_actual'),   # FIX-4
        'delta_wr_rival':       _delta_wr_rival,                      # FIX-4
        # T18-04 + T18-C5 (Nodo-18): PELT recency + delta win_rate
        'recencia_regimen':     _recencia_info['recencia'],
        'freshness_pelt':       _recencia_info['freshness'],
        'alpha_temporal':       round(_alpha_factor, 4),
        'delta_wr_markov':      _delta_wr,
        # D44-02 (Nodo-44): Markov confidence + win_rate_reciente para WAS filter
        'markov_conf_fav':      _markov_fav.get('confianza', 0),
        'markov_conf_rival':    _markov_rival.get('confianza', 0),
        'markov_wr_rec_fav':    _markov_fav.get('win_rate_reciente'),
        'markov_wr_rec_rival':  _markov_rival.get('win_rate_reciente'),
        # Data completeness: fracción de componentes con datos reales (0.0-1.0)
        # <0.5 = modelo apostando con huecos grandes → revisar antes de desplegar
        'data_completeness':    completeness,
    })

    # ─── Nodo-24: Bookmaker Blindness Scoring ───────────────────────────────
    # F-24-1: BBI — cuánto NO ve el bookmaker (0=ve todo, 1=ciego total)
    _n_h2h_v = resultado['n_h2h']
    _bbi = (1 - 1 / max(cuota_fav, 1.001)) * (1 / (1 + _n_h2h_v * 0.20))

    # F-24-2: Calibration Gap — gap entre p_blend (James-Stein) y p_modelo
    _js_factor = _n_cal / (_n_cal + 20) if _n_cal > 0 else 0.0
    _p_blend = _js_factor * p_modelo + (1 - _js_factor) * p_hist
    _gap = round(_p_blend - p_modelo, 4)
    if _gap > 0.12:
        _gap_flag = 'CALIBRATION_DRIVEN'
    elif _gap < 0.08:
        _gap_flag = 'MARKET_DRIVEN'
    else:
        _gap_flag = 'MIXED'

    # F-24-3: Mega Pick Quality (MPQ) — calidad de pick para mega-combos
    _edge_pct_float = resultado['edge'] * 100
    _mpq = resultado['kelly_kl'] * _bbi * (1 + _edge_pct_float / 100)

    resultado.update({
        'bbi':             round(_bbi, 4),
        'p_blend':         round(_p_blend, 4),
        'calibration_gap': _gap,
        'gap_flag':        _gap_flag,
        'mpq':             round(_mpq, 6),
        'golden_zone':     False,  # T32-03: se recalcula post-_tas (requiere n_axes_active)
    })

    # ─── Nodo-28 Fase 2: Triple Alignment Score ─────────────────────────────
    # M-28-4/5: calcula alineación de las 3 fuentes de information asymmetry.
    # Requiere que BBI y alpha_vs_elo ya estén en resultado (calculados arriba).
    _tas = triple_alignment_score(resultado)
    resultado.update(_tas)

    # F-24-4 (Nodo-32 T32-03): Golden Zone — asimetría informacional REAL a nuestro favor
    # ANTES: solo cuota alta + sin H2H → seleccionaba ceguera MUTUA (modelo también ciego)
    # AHORA: bookmaker ciego (BBI>=0.60) + modelo con señal (2+ axes) + convicción (p>=0.55)
    # Nodo-32 audit: apostar requerido — golden_zone es zona de apuesta, no zona de observación.
    # Un pick bloqueado por KL-penalty alto no merece golden_bonus en mega-combos.
    _golden_zone = (
        resultado.get('apostar', False)                    # T32-20: solo picks realmente apostados
        and tier in ('challenger', 'itf')
        and cuota_fav >= 2.50
        and _bbi >= 0.60                                    # bookmaker realmente ciego
        and _tas.get('n_axes_active', 0) >= 2              # modelo tiene ≥2 señales activas
        and p_modelo >= P_MODELO_MIN_UNDERDOG               # modelo tiene convicción mínima
    )
    resultado['golden_zone'] = _golden_zone

    # M-28-6: confidence_flag override — LOW + STRUCTURAL_ALPHA → LOW_STRUCTURAL
    # Señala "baja convicción del modelo PERO alpha estructural confirmado".
    # Solo informativo: NO modifica kelly_kl ni sizing (hasta validar V-28-2).
    if resultado.get('confidence_flag') == 'LOW' and _tas['alignment_flag'] == 'STRUCTURAL_ALPHA':
        resultado['confidence_flag'] = 'LOW_STRUCTURAL'

    # ─── FIX-2: data_insufficient_surface (Nodo-28 Fase 2) ─────────────────────
    # Detecta cuando alguno de los jugadores no tiene datos de superficie confiables.
    # Campo informativo — NO modifica edge ni Kelly.
    _surf_meta = pred.get('surface_specialization_meta', {})
    _surf_fav = (_surf_meta.get('player1') if player_key_sb == 'player1' else _surf_meta.get('player2')) or {}
    _surf_dog = (_surf_meta.get('player2') if player_key_sb == 'player1' else _surf_meta.get('player1')) or {}
    _vol_fav = _surf_fav.get('volume_confidence', 1.0)
    _vol_dog = _surf_dog.get('volume_confidence', 1.0)
    resultado['data_insufficient_surface'] = min(_vol_fav, _vol_dog) < 0.25

    # ─── Nodo-35: HISTORIAL_NO_EXTRAIDO — bloqueo en origen ─────────────────────
    # Si la extracción de historial falló para cualquiera de los dos jugadores,
    # la predicción está basada en datos incompletos → bloqueado sin importar el edge.
    # El flag viaja desde ninja_h2h_parser → rivalry_analyzer → aquí.
    _historial_incompleto = pred.get('historial_incompleto', {})
    _p1_sin_datos = _historial_incompleto.get('p1', False)
    _p2_sin_datos = _historial_incompleto.get('p2', False)
    if (_p1_sin_datos or _p2_sin_datos) and resultado.get('apostar'):
        _sin_datos_nombres = []
        if _p1_sin_datos:
            _sin_datos_nombres.append(partido.get('jugador1', 'jugador1'))
        if _p2_sin_datos:
            _sin_datos_nombres.append(partido.get('jugador2', 'jugador2'))
        resultado['apostar'] = False
        resultado['motivo_reclasificacion'] = (
            f'HISTORIAL_NO_EXTRAIDO: sin datos de {", ".join(_sin_datos_nombres)} '
            f'— predicción no confiable, bloqueada en origen'
        )

    # ─── FIX-3 / REGLA-N28-F2-1: n_axes_active < 2 → watchlist ────────────────
    # BBI sola (1 eje activo) tiene 29% hit rate histórico — peor que random.
    # Mover a watchlist evita apostar sin convergencia de señales.
    if _tas['n_axes_active'] < 2 and resultado.get('apostar'):
        resultado['apostar'] = False
        resultado['motivo_reclasificacion'] = 'N28F2: n_axes_active < 2 (BBI sola no predice)'

    # ─── FIX-6 / Markov×BBI: HOT sin BBI alto = trampa de mercado ───────────────
    # Pipeline tracker S-27-4b: HOT = 9.1% hit (1W/10L), NEUTRAL = 40% (4W/6L).
    # El bookmaker ya pricea el momentum (cuota baja). HOT sin BBI alto significa
    # que el bookmaker VE la racha → no hay information asymmetry → edge falso.
    # Solo suprimir cuando el favorito es HOT pero BBI < 0.50 (bookmaker lo ve).
    _markov_fav = resultado.get('markov_favorito')
    _bbi = resultado.get('bbi', 0.5)
    if _markov_fav == 'HOT' and _bbi < 0.50 and resultado.get('apostar'):
        resultado['apostar'] = False
        resultado['motivo_reclasificacion'] = resultado.get('motivo_reclasificacion', '') or 'HOT_sin_BBI: bookmaker ya pricea momentum (BBI<0.50)'

    # ─── T33-01 (Nodo-33): Bloqueo coin-flip n_h2h=0 ──────────────────────────
    # BUG-33-1: James-Stein con n_cal=0 colapsa p_blend → 0.50 (solo prior).
    # BUG-33-2: puerta lateral cuota<2.10 bypasseaba el check p_modelo≥0.55.
    # Sin H2H directo y sin convicción del modelo, el edge es shrinkage noise,
    # no señal real. Aplica incluso a favoritos (cuota<2.10) — no hay exception.
    # _n_h2h_v ya disponible desde línea ~789 (resultado['n_h2h']).
    if _n_h2h_v == 0 and p_modelo < P_MODELO_MIN_UNDERDOG and resultado.get('apostar'):
        resultado['apostar'] = False
        resultado['motivo_reclasificacion'] = (
            resultado.get('motivo_reclasificacion', '') or
            f'T33-01: n_h2h=0 + p_modelo={p_modelo:.3f}<{P_MODELO_MIN_UNDERDOG} (coin-flip bloqueado)'
        )

    # ─── FIX-5 / Nodo-29 Fase 4: circuit_asymmetry en edge_report ───────────────
    # Lee la señal de asimetría de circuito calculada en rivalry_analyzer y
    # agrega campos informativos. NO modifica edge ni Kelly.
    _circuit = pred.get('circuit_asymmetry') or {}
    _circuit_signal = _circuit.get('signal', 'SYMMETRIC')
    _circuit_warning = False
    if _circuit_signal in ('MODERATE_ASYMMETRY', 'STRONG_ASYMMETRY'):
        _deflated = _circuit.get('player_deflated')
        if _deflated and _deflated == favored:
            _circuit_warning = True

    resultado.update({
        'circuit_asymmetry_signal': _circuit_signal,
        'circuit_asymmetry_ratio':  round(float(_circuit.get('asymmetry_ratio', 1.0)), 3),
        'circuit_warning':          _circuit_warning,
    })

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def _validate_h2h_rivalry_version(raw: dict, path: str) -> None:
    """Nodo-32 Fase 3: Rechaza h2h_results_enhanced generado con Markov PRE-norm.

    Un h2h_results_enhanced sin rivalry_version o con versión antigua contiene valores
    de `confidence` calculados antes de que el factor Markov se aplicara POST-log1p.
    El delta de señal era ~0.072 (ruido). Regenerar garantiza señal real (~0.795).
    """
    actual = raw.get("metadata", {}).get("rivalry_version")
    if actual != _EXPECTED_RIVALRY_VERSION:
        msg = (
            f"\n{'='*70}\n"
            f"  ERROR: h2h_results_enhanced con rivalry_version desactualizada o ausente\n"
            f"  Archivo:  {path}\n"
            f"  Versión en archivo: {actual!r}\n"
            f"  Versión esperada:   {_EXPECTED_RIVALRY_VERSION!r}\n"
            f"\n"
            f"  El archivo contiene predicciones calculadas con Markov PRE-normalizacion.\n"
            f"  Los valores de confidence reflejan delta ~0.072 (señal decorativa).\n"
            f"  Regenera el h2h con el motor actualizado:\n"
            f"      python3 extraer_historh2h.py --api-mode --all-tournaments\n"
            f"{'='*70}\n"
        )
        raise SystemExit(msg)


def procesar_archivo_h2h(h2h_file: str, output_file: Optional[str] = None) -> dict:
    """
    Lee h2h_results_enhanced_FECHA.json y calcula el edge para todos los partidos.
    Genera un reporte de apuestas con las 5 capas activadas.
    """
    logger.info(f"📂 Cargando: {h2h_file}")
    with open(h2h_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    _validate_h2h_rivalry_version(raw, h2h_file)  # Nodo-32 Fase 3

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
            'gate_version':   GATE_VERSION,  # Nodo-32 Acción 3: versión del gate para validación
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

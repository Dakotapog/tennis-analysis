"""
Markov Analyzer — PELT Change-Point Detection (Nodo-02)

Detecta cuándo un jugador cambió de régimen (buena racha → mala racha)
usando una versión simplificada del algoritmo PELT (Pruned Exact Linear Time).

Conexión científica:
  El ranking ATP promedia 52 semanas. form_recent promedia 20 partidos.
  Ninguno captura el MOMENTO EXACTO del cambio de régimen.
  PELT sobre la secuencia binaria W=1/L=0 detecta ese momento en O(n).

Estados:
  HOT    → últimos 5 partidos: win_rate ≥ 70%
  COLD   → últimos 5 partidos: win_rate ≤ 30%
  NEUTRAL → entre 30% y 70%

Factor de integración:
  calcular_factor_markov(p1_hot, p2_cold) → 1.15 (P1 amplificado)
  calcular_factor_markov(p1_cold, p2_hot) → 0.85 (P1 reducido)
  calcular_factor_markov(neutral, neutral) → 1.0  (sin cambio)
"""

from typing import List, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# NODO-46 / F4: Markov Surface-Context Discount
# ══════════════════════════════════════════════════════════════════════════════

_SURFACE_MAP = {
    'hierba': 'grass', 'grass': 'grass', 'herb': 'grass', 'grama': 'grass',
    'dura': 'hard', 'hard': 'hard', 'hardcourt': 'hard', 'hard court': 'hard',
    'indoor hard': 'hard', 'carpet': 'hard', 'cemento': 'hard',
    'arcilla': 'clay', 'clay': 'clay', 'tierra': 'clay', 'tierra batida': 'clay',
}

# D46-07: constantes BLOQUEADAS para calibración hasta n≥5 casos atribuibles (hoy n=1, Watanuki)
_SURFACE_DISCOUNT_THRESHOLD = 0.40  # overlap mínimo para no descontar
_SURFACE_DISCOUNT_MIN_FLOOR = 0.70  # discount máximo (0% overlap → factor se acerca 30% a 1.0)


def _normalize_surface(s: str) -> str:
    """D46-02: Normaliza superficie a 'hard' | 'clay' | 'grass' | 'unknown'."""
    if not s:
        return 'unknown'
    return _SURFACE_MAP.get(s.lower().strip(), 'unknown')


def _surface_overlap_rate(recent_matches: list, current_surface: str, k: int = 10) -> float:
    """
    D46-03: Fracción de los últimos K partidos jugados en la misma superficie que current_surface.

    Args:
        recent_matches : historial del jugador (más reciente primero)
        current_surface: superficie del torneo actual (ya normalizada: 'hard'|'clay'|'grass')
        k              : ventana de partidos recientes

    Returns:
        float [0.0, 1.0] — 1.0 = todos recientes en misma superficie | 0.0 = ninguno
        0.0 también cuando: historial vacío, current_surface='unknown', n<5
    """
    if not recent_matches or current_surface == 'unknown':
        return 0.0
    window = recent_matches[:k]
    if len(window) < 5:
        return 0.0  # muestra muy pequeña — PELT ya tiene baja confianza (Nodo-46 §casos no-intervención)
    same = sum(
        1 for m in window
        if _normalize_surface(m.get('superficie', '')) == current_surface
    )
    return same / len(window)


def apply_surface_context_discount(
    factor_markov: float,
    confianza: float,
    surface_overlap_rate: float,
    estado: str,
    season_transition_flag: bool = False,
    apply_discount: bool = True,
) -> Tuple[float, float, float]:
    """
    D46-04: Ajusta factor_markov y confianza según overlap de superficie.

    Curva de descuento lineal:
      overlap = 1.0 → discount = 1.0  (sin cambio — racha en misma superficie)
      overlap = 0.5 → discount ≈ 0.875 (descuento moderado)
      overlap = 0.0 → discount = min_floor (racha en otra superficie)

    Solo aplica cuando:
      - apply_discount=True (flag --no-surface-discount desactivado)
      - estado != 'NEUTRAL' (NEUTRAL no tiene señal que distorsionar)
      - overlap < THRESHOLD (sin overlap suficiente en misma superficie)

    season_transition_flag=True → fuerza apply aunque overlap sea ambiguo (MM-4).

    Returns:
        (new_factor_markov, new_confianza, discount_applied)
    """
    if not apply_discount:
        return factor_markov, confianza, 1.0

    if estado == 'NEUTRAL':
        return factor_markov, confianza, 1.0

    threshold = _SURFACE_DISCOUNT_THRESHOLD
    min_floor = _SURFACE_DISCOUNT_MIN_FLOOR

    # season_transition_flag: usar el min_floor aunque overlap esté en zona ambigua
    effective_threshold = threshold if not season_transition_flag else min(threshold * 1.5, 0.60)

    if surface_overlap_rate >= effective_threshold:
        return factor_markov, confianza, 1.0

    # Interpolar linealmente: overlap=0 → discount=min_floor; overlap=threshold → discount=1.0
    discount = min_floor + (1.0 - min_floor) * (surface_overlap_rate / effective_threshold)
    discount = round(max(min_floor, min(1.0, discount)), 4)

    # factor_markov HOT > 1.0 → se acerca a 1.0 | COLD < 1.0 → se acerca a 1.0
    new_factor = round(1.0 + (factor_markov - 1.0) * discount, 4)
    new_confianza = round(confianza * discount, 4)

    return new_factor, new_confianza, discount


def detectar_cambio_regimen(
    resultados: List[int],
    min_size: int = 5,
    umbral_cambio: float = 0.20,
) -> dict:
    """
    PELT simplificado sobre secuencia binaria de resultados.

    Args:
        resultados:     Lista [1=victoria, 0=derrota] ordenada cronológicamente
                        (más viejo primero, más reciente al final).
        min_size:       Mínimo de partidos por segmento para detectar cambio.
        umbral_cambio:  Diferencia mínima entre segmentos para declarar changepoint.

    Returns:
        estado_actual:     'HOT' | 'COLD' | 'NEUTRAL'
        momentum:          float (-1 a +1) — positivo = mejorando
        change_point:      índice donde ocurrió el último cambio significativo, o None
        confianza:         float (0-1) — qué tan claro es el cambio
        win_rate_reciente: win_rate de la segunda mitad (más reciente)
        win_rate_anterior: win_rate de la primera mitad
    """
    n = len(resultados)

    if n < min_size * 2:
        return {
            'estado_actual':     'NEUTRAL',
            'momentum':          0.0,
            'change_point':      None,
            'confianza':         0.0,
            'win_rate_reciente': round(float(np.mean(resultados)), 3) if resultados else 0.0,
            'win_rate_anterior': round(float(np.mean(resultados)), 3) if resultados else 0.0,
            'n_partidos':        n,
        }

    arr = np.array(resultados, dtype=float)

    # Dividir en primera y segunda mitad para calcular momentum global
    mid = n // 2
    win_rate_anterior = float(np.mean(arr[:mid]))
    win_rate_reciente = float(np.mean(arr[mid:]))
    delta = win_rate_reciente - win_rate_anterior
    momentum = delta  # rango -1 a +1

    # PELT simplificado: barrer todos los puntos de corte posibles
    # Buscar el punto que maximiza la diferencia entre segmentos izquierdo y derecho
    mejor_punto = mid
    mejor_diferencia = abs(delta)

    for i in range(min_size, n - min_size):
        antes = float(np.mean(arr[:i]))
        despues = float(np.mean(arr[i:]))
        diferencia = abs(despues - antes)
        if diferencia > mejor_diferencia:
            mejor_diferencia = diferencia
            mejor_punto = i

    # Estado actual: basado en los últimos 5 partidos
    n_recientes = min(5, n)
    ultimos = float(np.mean(arr[-n_recientes:]))
    if ultimos >= 0.70:
        estado = 'HOT'
    elif ultimos <= 0.30:
        estado = 'COLD'
    else:
        estado = 'NEUTRAL'

    # change_point: solo declarar si la diferencia supera el umbral
    change_point = mejor_punto if mejor_diferencia >= umbral_cambio else None

    return {
        'estado_actual':     estado,
        'momentum':          round(momentum, 3),
        'change_point':      change_point,
        'confianza':         round(min(mejor_diferencia * 2, 1.0), 3),
        'win_rate_reciente': round(win_rate_reciente, 3),
        'win_rate_anterior': round(win_rate_anterior, 3),
        'n_partidos':        n,
    }


def calcular_recencia_regimen(markov_result: dict) -> dict:
    """
    T18-01 (Nodo-18): Calcula cuántos partidos han pasado desde el último cambio
    de régimen PELT detectado.

    Ejemplo:
      n_partidos=20, change_point=17 → recencia=3  → FRESCO (bookmaker no repriced)
      n_partidos=20, change_point=6  → recencia=14 → ESTABLE (bookmaker ya repriced)

    Retorna: {'recencia': int|None, 'freshness': 'FRESCO'|'RECIENTE'|'ESTABLE'}
    """
    change_point = markov_result.get('change_point')
    n_total = markov_result.get('n_partidos', 0)

    if change_point is None or n_total == 0:
        return {'recencia': None, 'freshness': 'ESTABLE'}

    recencia = n_total - change_point
    if recencia <= 3:
        freshness = 'FRESCO'
    elif recencia <= 7:
        freshness = 'RECIENTE'
    else:
        freshness = 'ESTABLE'

    return {'recencia': recencia, 'freshness': freshness}


def factor_alpha_temporal(recencia, estado: str, delta_wr: float) -> float:
    """
    T18-02 (Nodo-18): Multiplicador inverso de λ basado en frescura del régimen PELT.

    Aplicación en edge_calculator:
      λ_efectivo = λ_tier × (1 / factor_alpha_temporal)

    REGLA-T18-1:
      HOT + recencia ≤ 3 → 1.20 (λ reducido 17%: bookmaker stale, máximo alpha)
      HOT + recencia ≤ 7 → 1.10 (λ reducido 9%: alpha parcial)
      COLD + recencia ≤ 3 → 0.85 (λ aumentado 18%: precaución amplificada)
      resto              → 1.00 (bookmaker ya repriced, sin ajuste)
    """
    if recencia is None:
        return 1.00
    if estado == 'HOT' and recencia <= 3:
        return 1.20
    if estado == 'HOT' and recencia <= 7:
        return 1.10
    if estado == 'COLD' and recencia <= 3:
        return 0.85
    return 1.00


def calcular_factor_markov(markov_p1: dict, markov_p2: dict) -> float:
    """
    Multiplicador para el score de form_recent de P1, comparado con P2.

    Lógica:
      P1 HOT  + P2 COLD   → factor 1.15 (P1 amplificado)
      P1 COLD + P2 HOT    → factor 0.85 (P1 reducido)
      P1 HOT  + P2 HOT    → factor 1.0  (empate de momentum)
      P1 COLD + P2 COLD   → factor 1.0  (empate de mal momentum)
      cualquier NEUTRAL   → factor se acerca a 1.0

    Rango: [0.85, 1.15]
    """
    estados = {'HOT': 1, 'NEUTRAL': 0, 'COLD': -1}
    e1 = estados.get(markov_p1.get('estado_actual', 'NEUTRAL'), 0)
    e2 = estados.get(markov_p2.get('estado_actual', 'NEUTRAL'), 0)
    diferencia = e1 - e2  # rango: -2 a +2

    # Escalar: diferencia=2 → factor=1.15, diferencia=-2 → factor=0.85
    factor = 1.0 + diferencia * 0.075
    return round(factor, 3)


def calcular_factor_tardio(
    player_history: list,
    min_matches: int = 3,
) -> Optional[dict]:
    """
    T14-02 — Win rate del jugador cuando el partido llega al 4to o 5to set.

    Un partido es "extendido" cuando sets_ganador + sets_perdedor >= 4:
      3-1, 1-3 → 4 sets | 3-2, 2-3 → 5 sets

    Ningún bookmaker modela este dato. Alpha exclusivo del sistema.

    Args:
        player_history: Lista de partidos (newest-first). Cada partido debe
                        tener 'resultado' ("X-Y") y 'outcome' ('Ganó'/'Perdió').
        min_matches:    Mínimo de partidos extendidos para generar señal.

    Returns:
        dict con win_rate_tardio, n_partidos_extendidos, factor_tardio
        None si no hay suficientes partidos extendidos.
    """
    if not player_history:
        return None

    wins_tardio = 0
    total_tardio = 0

    for match in player_history:
        resultado = match.get('resultado')
        if not resultado or resultado == 'N/A':
            continue

        try:
            parts = str(resultado).split('-')
            if len(parts) != 2:
                continue
            s1, s2 = int(parts[0]), int(parts[1])
        except (ValueError, AttributeError):
            continue

        if s1 + s2 < 4:
            continue  # partido corto (2-0, 2-1, 1-2, 0-2) — no extendido

        total_tardio += 1
        outcome = match.get('outcome', '').lower()
        if 'ganó' in outcome or 'win' in outcome:
            wins_tardio += 1

    if total_tardio < min_matches:
        return None

    win_rate = round(wins_tardio / total_tardio, 3)
    return {
        'win_rate_tardio': win_rate,
        'n_partidos_extendidos': total_tardio,
    }


def calcular_factor_tardio_comparativo(
    tardio_p1: Optional[dict],
    tardio_p2: Optional[dict],
) -> float:
    """
    Multiplicador para el score de P1 basado en rendimiento en sets tardíos.

    Análogo a calcular_factor_markov pero usando win_rate_tardio.
    Rango: [0.85, 1.15]. Si alguno es None → factor = 1.0 (sin penalizar).

    Args:
        tardio_p1: dict de calcular_factor_tardio para jugador 1, o None.
        tardio_p2: dict de calcular_factor_tardio para jugador 2, o None.

    Returns:
        float: multiplicador en [0.85, 1.15]
    """
    if tardio_p1 is None or tardio_p2 is None:
        return 1.0

    wr1 = tardio_p1.get('win_rate_tardio', 0.5)
    wr2 = tardio_p2.get('win_rate_tardio', 0.5)

    # Diferencia normalizada: rango [-1, +1] → factor [0.85, 1.15]
    diferencia = wr1 - wr2  # rango [-1, +1]
    factor = 1.0 + diferencia * 0.15
    return round(max(0.85, min(1.15, factor)), 3)


def extraer_resultados_binarios(
    player_history: list,
    player_name: str,
    n: int = 20,
) -> List[int]:
    """
    Extrae la secuencia binaria [1=victoria, 0=derrota] de los últimos n partidos.
    Ordena cronológicamente (más viejo primero) para PELT.

    Compatible con el formato de match del pipeline:
      match['outcome'] → 'ganó' | 'win' | 'perdió' | 'loss'
    """
    if not player_history:
        return []

    # Tomar los n más recientes (asume history = newest-first)
    recientes = player_history[:n]

    resultados = []
    for match in reversed(recientes):  # reversed → oldest first = cronológico
        outcome = match.get('outcome', '').lower()
        if 'ganó' in outcome or 'win' in outcome:
            resultados.append(1)
        elif 'perdió' in outcome or 'loss' in outcome:
            resultados.append(0)
        else:
            # Inferencia por RET/WO/fallback
            resultado = match.get('resultado', '').upper()
            if 'RET' in resultado or 'WO' in resultado:
                resultados.append(1)
            else:
                # Sin información suficiente — omitir (no añadir ruido)
                pass

    return resultados

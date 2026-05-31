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

from typing import List, Optional
import numpy as np


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
    }


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

# Nodo-02: Markov Analyzer + PELT (Simplified) Change-Point Detection

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]]
> ⚠️ PRODUCCIÓN: Markov está implementado en `analysis/rivalry_analyzer.py` (✅ 37 tests) pero NO activo en producción hasta completar [[Nodo-07-Strangler-Fig]] Fase 1. Estado real: IMPLEMENTADO-NO-CONECTADO.
> **Fundamento:** PELT (Pruned Exact Linear Time) — detección de cambio de régimen

**Prioridad:** ALTA — mejora el componente form_recent (15% del modelo)
**Archivo objetivo:** `analysis/markov_analyzer.py`
**Dependencias:** `historial_jugador` en h2h_results_enhanced (ya existe)

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  S7_MARKOV → {estado_actual: HOT|COLD|NEUTRAL, momentum: float, factor_markov: float}

CONSUME:  S2_H2H_DATA → player1_history[-20:] (lista de partidos recientes)
          S2_H2H_DATA → player2_history[-20:]

PREREQUISITO: historial_jugador con ≥10 partidos para detección confiable
              rivalry_analyzer.py debe importar analysis.markov_analyzer
```

---

## Conexiones Cross-Nodo (CX)

| CX | De → A | Impacto |
|---|---|---|
| CX-02 | [[Nodo-06-Erdos-Graph]] erdos_score + HOT estado | Señal 2x fuerte cuando ambos confirman ventaja |
| CX-05 | [[Nodo-04-Dataset-Fix]] → S8_DATASET_ML | estado HOT/COLD como feature binaria en dataset ML |
| CX-07 | [[Nodo-01-Edge-Calculator]] → kelly_kl | factor_markov HOT aumenta confianza → mayor edge |

---

---

## Problema

`form_analysis.win_percentage` promedia los últimos 20 partidos por igual. Un jugador con 14 victorias en los primeros 10 y 6 en los últimos 10 tiene el mismo 50% que uno con 6 y 14. Son situaciones radicalmente diferentes — uno está en declive, el otro en ascenso.

El ranking ATP promedia 52 semanas. `form_analysis` promedia 20 partidos. Ninguno captura el **momento exacto del cambio de régimen**.

## Solución: PELT Algorithm

```python
# analysis/markov_analyzer.py
from typing import List, Tuple, Optional
import numpy as np

def detectar_cambio_regimen(resultados: List[int], min_size: int = 5) -> dict:
    """
    resultados: lista binaria [1=victoria, 0=derrota] ordenada cronológicamente (más viejo primero)
    min_size:   mínimo de partidos por segmento para detectar cambio
    
    Retorna:
        estado_actual: 'HOT' | 'COLD' | 'NEUTRAL'
        momentum:      float (-1 a +1), positivo = mejorando
        change_point:  índice donde ocurrió el último cambio significativo
        confianza:     float (0-1), qué tan claro es el cambio
    """
    n = len(resultados)
    if n < min_size * 2:
        return {'estado_actual': 'NEUTRAL', 'momentum': 0.0, 'change_point': None, 'confianza': 0.0}

    # Dividir en primera mitad y segunda mitad
    mid = n // 2
    primera_mitad = np.mean(resultados[:mid])
    segunda_mitad = np.mean(resultados[mid:])

    delta = segunda_mitad - primera_mitad
    momentum = delta  # -1 a +1

    # PELT simplificado: buscar el punto de cambio con mayor diferencia
    mejor_punto = mid
    mejor_diferencia = abs(delta)

    for i in range(min_size, n - min_size):
        antes = np.mean(resultados[:i])
        despues = np.mean(resultados[i:])
        diferencia = abs(despues - antes)
        if diferencia > mejor_diferencia:
            mejor_diferencia = diferencia
            mejor_punto = i

    # Calcular estado actual (últimos 5 partidos)
    ultimos_5 = np.mean(resultados[-5:]) if n >= 5 else np.mean(resultados)
    if ultimos_5 >= 0.70:
        estado = 'HOT'
    elif ultimos_5 <= 0.30:
        estado = 'COLD'
    else:
        estado = 'NEUTRAL'

    return {
        'estado_actual': estado,
        'momentum': round(momentum, 3),
        'change_point': mejor_punto if mejor_diferencia > 0.20 else None,
        'confianza': round(min(mejor_diferencia * 2, 1.0), 3),
        'win_rate_reciente': round(ultimos_5, 3),
        'win_rate_anterior': round(primera_mitad, 3)
    }


def calcular_factor_markov(markov_p1: dict, markov_p2: dict) -> float:
    """
    Retorna un factor multiplicador para el score de form_recent:
    - Si P1 está HOT y P2 COLD → factor favorable para P1
    - Si ambos NEUTRAL → factor 1.0 (sin cambio)
    """
    estados = {'HOT': 1, 'NEUTRAL': 0, 'COLD': -1}
    e1 = estados.get(markov_p1.get('estado_actual', 'NEUTRAL'), 0)
    e2 = estados.get(markov_p2.get('estado_actual', 'NEUTRAL'), 0)
    diferencia = e1 - e2  # rango -2 a +2

    # Factor entre 0.85 y 1.15
    factor = 1.0 + diferencia * 0.075
    return round(factor, 3)
```

## Integración en rivalry_analyzer.py

```python
# En analyze_rivalry(), antes de calculate_form_score():
from analysis.markov_analyzer import detectar_cambio_regimen, calcular_factor_markov

resultados_p1 = [1 if m.get('resultado','').startswith('2') else 0 
                 for m in player1_history[-20:]]
resultados_p2 = [1 if m.get('resultado','').startswith('2') else 0 
                 for m in player2_history[-20:]]

markov_p1 = detectar_cambio_regimen(resultados_p1)
markov_p2 = detectar_cambio_regimen(resultados_p2)
factor_markov = calcular_factor_markov(markov_p1, markov_p2)

# Aplicar factor al score de form_recent
form_score_p1 = form_score_p1 * factor_markov
```

## Output en el JSON de salida

```json
"markov_analysis": {
    "jugador1": {
        "estado_actual": "HOT",
        "momentum": 0.30,
        "change_point": 12,
        "confianza": 0.60,
        "win_rate_reciente": 0.80,
        "win_rate_anterior": 0.50
    },
    "jugador2": {
        "estado_actual": "COLD",
        "momentum": -0.25,
        "change_point": 8,
        "confianza": 0.50,
        "win_rate_reciente": 0.30,
        "win_rate_anterior": 0.55
    },
    "factor_markov": 1.15
}
```

## Tests Requeridos

```python
# tests/test_markov_analyzer.py
def test_jugador_hot_detectado():
    resultados = [0,0,1,0,1,0,1,1,1,1,1,1,1,1,1]  # mejora al final
    r = detectar_cambio_regimen(resultados)
    assert r['estado_actual'] == 'HOT'
    assert r['momentum'] > 0

def test_jugador_cold_detectado():
    resultados = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]  # declive al final
    r = detectar_cambio_regimen(resultados)
    assert r['estado_actual'] == 'COLD'
    assert r['momentum'] < 0

def test_factor_hot_vs_cold():
    hot = {'estado_actual': 'HOT'}
    cold = {'estado_actual': 'COLD'}
    assert calcular_factor_markov(hot, cold) > 1.0
    assert calcular_factor_markov(cold, hot) < 1.0

def test_datos_insuficientes_neutral():
    r = detectar_cambio_regimen([1, 0, 1])
    assert r['estado_actual'] == 'NEUTRAL'
    assert r['change_point'] is None
```

---

## Ciclo de Vida

```
Estado:   POR CONSTRUIR
Construcción: ~3 horas (función + integración en rivalry_analyzer.py + tests)
Deploy: importar en rivalry_analyzer.py antes de calculate_form_score()
Validación: comparar accuracy con/sin factor_markov en Jan 2026 data
Meta: factor_markov reduce errores en partidos donde form_recent era engañoso
```

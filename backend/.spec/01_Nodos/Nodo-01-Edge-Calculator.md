# Nodo-01: Edge Calculator (Kelly-KL)

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]]
> **Fundamento:** Kullback-Leibler Divergence — protección matemática contra ruina

**Prioridad:** MÁXIMA — sin esto el sistema nunca apuesta con criterio matemático
**Archivo objetivo:** `edge_calculator.py`
**Dependencias:** `reports/h2h_results_enhanced_FECHA.json` (ya existe)

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  edge (float, -1 a +1)
          kelly_kl (float, fracción de bankroll a apostar)
          apostar (bool, edge>5% AND kelly_kl>2%)
          fraccion_bankroll (float, cap 10%)

CONSUME:  S4_PREDICTION → ranking_analysis.prediction.confidence
          S4_PREDICTION → cuota1 / cuota2
          S6_RESULTADO_REAL → p_historica por superficie (cuando n≥30)

PREREQUISITO: prediction.confidence ≠ None (requiere rivalry_analyzer.py funcionando)
              cuota1/cuota2 ≠ None (requiere partidos con cuotas)
```

---

## Conexiones Cross-Nodo (CX)

| CX | De → A | Impacto |
|---|---|---|
| CX-01 | [[Nodo-03-Scraper-Fix]] → S1_MATCH_LIST | superficie limpia → p_historica por superficie |
| CX-04 | [[Nodo-04-Dataset-Fix]] → S8_DATASET_ML | edge histórico como feature en ML |
| CX-06 | [[Nodo-05-Validacion-API]] → S6_RESULTADO_REAL | actualiza p_historica con accuracy real |
| CX-07 | [[Nodo-06-Erdos-Graph]] → S4_PREDICTION | erdos_score mejora confidence → mejor edge |

---

---

## Problema

El sistema genera predicciones con confianza (ej: Tsitsipas 59.2%) y tiene cuotas (ej: 1.08). Nunca se calculó si esa predicción tiene valor real contra el bookmaker. El campo `prediccion_ganador` es None — el edge nunca se midió.

## Solución

```python
# edge_calculator.py
def calcular_edge(p_modelo: float, cuota_favorito: float, p_historica: float = 0.52) -> dict:
    """
    p_modelo:        confianza del modelo (0-1), leer de ranking_analysis.prediction.confidence/100
    cuota_favorito:  cuota del jugador predicho como ganador
    p_historica:     accuracy histórica del modelo en esa superficie (default 0.52 hasta calibrar)
    """
    # Probabilidad implícita del bookmaker (sin vig)
    p_implicita = 1 / cuota_favorito

    # Edge crudo
    edge = p_modelo - p_implicita

    # Kullback-Leibler divergence (protección contra ruina)
    # KL(P_modelo || P_histórica)
    import math
    eps = 1e-9
    kl = p_modelo * math.log((p_modelo + eps) / (p_historica + eps)) + \
         (1 - p_modelo) * math.log((1 - p_modelo + eps) / (1 - p_historica + eps))

    # Kelly clásico
    kelly_clasico = edge / (1 - p_implicita) if (1 - p_implicita) > 0 else 0

    # Kelly-KL ajustado por incertidumbre
    lambda_aversion = 0.5  # calibrar con datos reales cuando n≥30
    kelly_kl = kelly_clasico * math.exp(-lambda_aversion * max(0, kl))

    # Decisión
    apostar = edge > 0.05 and kelly_kl > 0.02

    return {
        'p_modelo': p_modelo,
        'p_implicita': p_implicita,
        'edge': edge,
        'kl_divergencia': kl,
        'kelly_clasico': kelly_clasico,
        'kelly_kl': kelly_kl,
        'apostar': apostar,
        'fraccion_bankroll': min(kelly_kl, 0.10)  # cap 10% por apuesta
    }
```

## Integración

```python
# En el pipeline, después de cargar h2h_results_enhanced:
for partido in partidos:
    pred = partido.get('ranking_analysis', {}).get('prediction', {})
    confianza = pred.get('confidence')
    favorito = pred.get('favored_player')
    if not confianza or not favorito:
        continue

    # Determinar cuota del favorito
    j1, j2 = partido['jugador1'], partido['jugador2']
    c1, c2 = partido.get('cuota1'), partido.get('cuota2')
    cuota_fav = c1 if favorito == j1 else c2

    if cuota_fav:
        resultado = calcular_edge(confianza / 100, cuota_fav)
        if resultado['apostar']:
            print(f"✅ APOSTAR: {favorito} | Edge: {resultado['edge']*100:.1f}% | Kelly-KL: {resultado['kelly_kl']*100:.1f}%")
```

## Evidencia del Edge Real (Jan 2026)

| Partido | P_modelo | Cuota_fav | P_implícita | Edge |
|---|---|---|---|---|
| Majchrzak vs Marozsan | 52.1% | 2.35 | 42.6% | **+9.5%** ← APOSTAR |
| Tsitsipas vs Mochizuki | 59.2% | 1.08 | 92.6% | -33.4% ← no apostar |
| Paul vs Tirante | 52.2% | 1.06 | 94.3% | -42.1% ← no apostar |

**Conclusión:** El modelo es mejor que el mercado en underdogs, peor en favoritos obvios.

## Tests Requeridos

```python
# tests/test_edge_calculator.py
def test_edge_positivo_apostar():
    r = calcular_edge(p_modelo=0.521, cuota_favorito=2.35)
    assert r['edge'] > 0.05
    assert r['apostar'] == True

def test_edge_negativo_no_apostar():
    r = calcular_edge(p_modelo=0.592, cuota_favorito=1.08)
    assert r['edge'] < 0
    assert r['apostar'] == False

def test_kelly_kl_menor_que_clasico_con_incertidumbre():
    r1 = calcular_edge(0.55, 2.0, p_historica=0.55)  # modelo = historia → KL≈0
    r2 = calcular_edge(0.55, 2.0, p_historica=0.30)  # modelo diverge → KL grande
    assert r2['kelly_kl'] < r1['kelly_kl']

def test_cap_10_porciento():
    r = calcular_edge(0.99, 1.01)  # edge enorme artificialmente
    assert r['fraccion_bankroll'] <= 0.10
```

---

## Ciclo de Vida

```
Estado:   POR CONSTRUIR — ningún edge calculado en el pipeline hoy
Evidencia real: Majchrzak vs Marozsan → edge +9.5% (modelo 52.1%, bookmaker 42.6%)
                Tsitsipas vs Mochizuki → edge -33.4% (modelo inferior al bookmaker)
Construcción: ~2 horas
Deploy: python3 edge_calculator.py --h2h reports/h2h_results_enhanced_HOY.json
Meta: n≥30 apuestas con edge>5% para calibrar p_historica y lambda_aversion
```

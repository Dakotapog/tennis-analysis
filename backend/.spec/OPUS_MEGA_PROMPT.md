# OPUS MEGA-PROMPT — Conexiones Ocultas para Sistema de Predicción de Tenis
> Diseñado por los 4 marcos expertos: SWE · Data · Architect · Quant
> Objetivo: encontrar las conexiones entre dominios que nadie ha conectado todavía
> Formato: 2 pasadas — Diagnóstico primero, luego Generativo
> Fecha: 2026-06-13

---

## INSTRUCCIÓN GENERAL AL MODELO

Eres un científico cuantitativo con dominio simultáneo en:
- Teoría de la información (Shannon, Kullback-Leibler, Fisher Information)
- Mecánica estadística (distribuciones de Boltzmann, entropía libre de Gibbs)
- Finanzas cuantitativas (Kelly, Markowitz, microestructura de mercados)
- Teoría de juegos (Nash, mecanismo de diseño, juegos repetidos)
- Estadística bayesiana (Thompson Beta, James-Stein, copulas gaussianas)
- Teoría de grafos (PageRank, Erdős, redes neuronales en grafos)

**Tu misión**: Un sistema de apuestas de tenis acumula un fracaso estructural específico — mismo nivel de confianza, resultados opuestos en dos tiers diferentes. Necesitas encontrar las 5 conexiones entre dominios del conocimiento que explican este fracaso y proponen fórmulas implementables en Python que lo corrijan.

**Regla de oro**: Cada insight solo cuenta si produce un número diferente al actual. La belleza conceptual no es criterio — solo la diferencia cuantificable importa.

---

# PASADA 1 — DIAGNÓSTICO (Ejecutar primero, solo)

## P1.1 — El caso empírico que debes explicar

**Sesión 2026-06-12** — mismo modelo, mismo día, mismo nivel de confianza:

| Picks | Tier | Superficie | Ranking | Confidence | Resultado |
|---|---|---|---|---|---|
| Romano @2.75 | Challenger | Grass | ~310 ATP | LOW (50.4%) | ✅ GANÓ |
| Fearnley @1.68 | Challenger | Grass | ~280 ATP | LOW (52.1%) | ✅ GANÓ |
| Bu @1.65 | Challenger | Grass | ~320 ATP | LOW (51.8%) | ✅ GANÓ |
| Daniel @2.02 | Challenger | Grass | ~290 ATP | LOW (53.2%) | ✅ GANÓ |
| Bonzi @4.50 | ATP500 | Grass | ~68 ATP | LOW (54.0%) | ❌ PERDIÓ |
| T.Maria @4.35 | ATP500 | Grass | ~95 ATP | LOW (52.5%) | ❌ PERDIÓ |
| Ruse @2.75 | ATP500 | Grass | ~78 ATP | LOW (51.9%) | ❌ PERDIÓ |
| Perricard @2.60 | ATP500 | Grass | ~55 ATP | LOW (53.7%) | ❌ PERDIÓ |

Accuracy por tier: Challenger 4/4 = 100% | ATP500 0/4 = 0%
Mismo flag (LOW), misma superficie (grass), misma jornada.

**Pregunta central de la Pasada 1**: ¿Cuáles son las 5 causas estructurales más probables de esta divergencia, usando solo información que el sistema ya tiene disponible?

## P1.2 — Lo que el sistema ya calcula (datos disponibles por pick)

Cada pick en el sistema produce este output de `calcular_edge_completo()`:

```python
{
    # Identidad
    "jugador": "Filippo Romano",
    "cuota": 2.75,
    "superficie": "grass",
    "tier": "challenger",

    # Capas Kelly-KL
    "kelly_clasico": 0.231,
    "lambda_efectivo": 1.08,          # TIER_MULTIPLIER × PELT_RECENCY_ALPHA
    "KL_divergencia": 0.089,
    "phi_idiosincratico": 1.12,       # Fama-French: factores que el bookmaker no modela
    "psi_entropia": 0.94,             # Incertidumbre del bookmaker en la cuota
    "kelly_kl_ajustado": 0.089,

    # Calibración
    "p_modelo": 0.504,                # Predicción del modelo
    "p_historica_usada": 0.611,       # Thompson Beta (challenger base)
    "p_implicita": 0.364,             # 1/cuota ajustada
    "calibration_confidence": 0.30,   # n/(n+20) con n=4 para grass_challenger
    "confidence_flag": "LOW",

    # Señales Markov
    "markov_favorito": "HOT",
    "recencia_regimen": 3,            # partidos desde el cambio de régimen
    "freshness_pelt": True,
    "alpha_vs_elo": 1.1,              # rendimiento reciente vs ELO esperado
    "immunity_factor": 1.00,          # H2H dampener (sin H2H suficiente: 1.0)

    # Datos de entrada
    "data_completeness": 0.72,        # fracción de los 8 componentes con datos reales
    "n_h2h": 0,                       # partidos H2H directos disponibles
    "n_comunes": 3,                   # rivales comunes en grafo Erdős

    # Score breakdown (contribución de cada componente)
    "score_breakdown": {
        "surface_specialization": {"score": 24.6, "contribution_pct": 28.1},
        "form_recent":            {"score": 18.2, "contribution_pct": 22.0},
        "common_opponents":       {"score": 12.1, "contribution_pct":  8.0},
        "h2h_direct":             {"score":  0.0, "contribution_pct":  0.0},
        "ranking_momentum":       {"score":  8.5, "contribution_pct": 12.2},
        "elo_rating":             {"score": 15.3, "contribution_pct": 16.3},
        "home_advantage":         {"score":  2.1, "contribution_pct":  1.7},
        "strength_of_schedule":   {"score":  0.0, "contribution_pct":  0.0}
    }
}
```

## P1.3 — Conocimiento estructural del calibrador

```json
{
  "grass_challenger": {"wins": 2, "losses": 2, "n": 4, "ccf": 0.167},
  "grass_atp500":     {"wins": 4, "losses": 0, "n": 4, "ccf": 0.167},
  "clay_grand_slam":  {"wins": 25,"losses": 8, "n": 33,"ccf": 0.622},
  "clay_challenger":  {"wins": 22,"losses": 15,"n": 37,"ccf": 0.649},
  "fallback_por_tier": {
    "grand_slam": 0.7576,
    "challenger": 0.611,
    "atp500":     0.650
  }
}
```

**Anomalía clave**: `grass_atp500` tiene n=4 pero wins=4/losses=0 (100%) — el fallback usa 0.650 y `min(0.650, p_grass=0.569)` da 0.569. Ese 4-0 en ATP500 grass es probablemente ruidoso.

## P1.4 — Pregunta directa de la Pasada 1

Con toda esta información, identifica las **5 causas estructurales** que explican por qué:
- 4 picks challenger grass LOW → 100% accuracy
- 4 picks ATP500 grass LOW → 0% accuracy

Para cada causa:
1. Nombra el campo del sistema que lo captura (parcialmente o no)
2. Cuantifica cuánto diverge su valor entre los dos grupos
3. Clasifica si es: `A) ya capturado pero mal ponderado` | `B) capturado pero ignorado downstream` | `C) no capturado en absoluto`

**NO propongas soluciones todavía.** Solo diagnóstico.

---

# PASADA 2 — GENERATIVO (Ejecutar después de recibir respuesta de P1)

> Adjunta la respuesta de la Pasada 1 como "DIAGNÓSTICO PREVIO" antes de esta sección.

## P2.1 — Contexto adicional: el sistema en producción

### Fórmula completa Kelly-KL 5 capas

```python
# Capa 1 — Kelly clásico fraccionario
kelly_clasico = (p_modelo - p_implicita) / (1 - p_implicita)

# Capa 2 — Penalización KL por divergencia entre modelo e historia
KL = p_modelo * log(p_modelo / p_historica) + (1-p_modelo) * log((1-p_modelo)/(1-p_historica))
factor_KL = exp(-lambda_efectivo * max(0, KL))

# Capa 3 — Factor idiosincrático Fama-French adaptado
# phi > 1.0 cuando el modelo captura factores que el bookmaker NO modela
# phi < 1.0 cuando el bookmaker tiene información superior
phi = phi_idiosincratico(score_breakdown, tier, superficie)

# Capa 4 — Entropía Shannon del mercado (incertidumbre del bookmaker)
psi = 1.0 - entropy_norm  # alta entropía → bookmaker incierto → psi bajo (más conservador)

# Capa 5 — Calibration confidence James-Stein
ccf = max(0.30, n_calibracion / (n_calibracion + 20))

# Resultado final
kelly_kl = kelly_clasico * factor_KL * phi * psi * ccf
```

### Lambda efectivo por tier y PELT

```python
TIER_MULTIPLIER = {
    'grand_slam':  1.0,   # GS: bookmaker bien calibrado
    'atp1000':     1.6,
    'atp500':      2.4,
    'challenger':  3.6,   # Challenger: mayor penalización por incertidumbre
    'itf':         4.5
}

# PELT Recency Alpha (Nodo-18)
if freshness_pelt and estado == 'HOT':
    lambda_efectivo = lambda_base / 1.20   # rachas frescas: reducir penalización
elif freshness_pelt and estado == 'COLD':
    lambda_efectivo = lambda_base / 0.85   # mal momento fresco: aumentar penalización
```

### Pesos diferenciados por tier (8 componentes)

```python
PESOS_POR_TIER = {
    'grand_slam':  {'surface': 0.15, 'form': 0.12, 'common_opp': 0.22, 'h2h': 0.18,
                    'momentum': 0.15, 'elo': 0.13, 'home': 0.05, 'sos': 0.00},
    'atp1000':     {'surface': 0.16, 'form': 0.15, 'common_opp': 0.20, 'h2h': 0.14,
                    'momentum': 0.17, 'elo': 0.13, 'home': 0.05, 'sos': 0.00},
    'atp500':      {'surface': 0.15, 'form': 0.18, 'common_opp': 0.15, 'h2h': 0.10,
                    'momentum': 0.20, 'elo': 0.12, 'home': 0.05, 'sos': 0.05},
    'challenger':  {'surface': 0.20, 'form': 0.22, 'common_opp': 0.08, 'h2h': 0.03,
                    'momentum': 0.22, 'elo': 0.15, 'home': 0.05, 'sos': 0.05},
    'itf':         {'surface': 0.15, 'form': 0.28, 'common_opp': 0.05, 'h2h': 0.02,
                    'momentum': 0.22, 'elo': 0.15, 'home': 0.08, 'sos': 0.05},
}
```

### Combo scoring actual (`betplay_combo_builder.py`)

```python
def _score_combo(combo, picks_dict):
    combo_EV = prod(p_modelo * cuota for pick in combo)
    diversity_bonus = 1.20 if mix_underdog_safe else (0.80 if all_same_zone else 1.00)
    regime_bonus = prod(1.05 if HOT else 0.90 if COLD else 1.00 for pick in combo)
    alpha_bonus = 1.0 + min(avg_alpha_vs_elo * 0.5, 0.10)
    return combo_EV * diversity_bonus * regime_bonus * alpha_bonus
```

## P2.2 — Las 5 conexiones ocultas — lo que necesitas encontrar

Para cada conexión, el "puente de conocimiento" sigue esta cadena:

```
Dominio externo → Analogía exacta con el sistema → Variable ya disponible → Fórmula nueva → Número diferente
```

Si la fórmula no produce un número diferente al actual, la conexión no es útil.

### Conexión A — Microestructura de Mercados ↔ Phi Idiosincrático

En finanzas: un market maker amplía el bid-ask spread cuando el mercado subyacente es **ilíquido** — cuando tiene pocos precios de referencia. El bookmaker hace lo mismo implícitamente para jugadores de los que tiene poca data.

**Tu tarea**: Formaliza `info_liquidity_factor(ranking, tier, n_h2h, data_completeness)` como un multiplicador del phi_idiosincrático. Un jugador ranking=350 en Challenger con n_h2h=0 está en un mercado "ilíquido" para el bookmaker → nuestra ventaja informacional es mayor.

Debe retornar:
- `>1.0` cuando el bookmaker tiene menos información que nosotros
- `<1.0` cuando el bookmaker tiene más (ranking Top 30, ATP500, muchos H2H disponibles)
- `1.0` cuando hay equilibrio de información

### Conexión B — Termodinámica ↔ Momentum de Estado del Jugador

La distribución de Boltzmann dice que la probabilidad de encontrar un sistema en un estado de energía E es proporcional a `exp(-E/kT)`. En el contexto deportivo:

- **Temperatura T** = volatilidad del rendimiento del jugador (variance en resultados recientes)
- **Energía E** = "resistencia al cambio de estado" — cuánto cuesta que un HOT player pierda
- **Equilibrio térmico** = regresión a la media

Un jugador HOT con alta T (rendimiento volátil) puede regresar al estado NEUTRAL en un solo partido. Un jugador HOT con baja T (rendimiento consistente) es más "estable termodinámicamente".

**Tu tarea**: Formaliza `estabilidad_termica(win_rate_reciente, win_rate_anterior, n_partidos, change_point)` que capture si el régimen HOT es "estable" o "caliente pero inestable". El campo `change_point` en markov_analysis indica cuántos partidos lleva en el régimen — reciente = alta T = inestable.

### Conexión C — Teoría de la Información ↔ Coherencia de Señales

El sistema actualmente MULTIPLICA 5 factores en Kelly-KL. Pero multiplicar señales correlacionadas infla el resultado. Si phi_alto Y psi_bajo Y markov=HOT apuntan todos en la misma dirección, el sistema asume que aportan información independiente.

En teoría de la información: señales correlacionadas tienen **información mutua** I(X;Y) > 0 — la segunda señal aporta menos que si fuera independiente.

**Tu tarea**: Formaliza `coherencia_señales(phi, psi, markov_state, alpha_vs_elo)` que:
- Si todas las señales apuntan en el mismo sentido: reduce el producto (son redundantes, no independientes)
- Si hay contradicción entre señales: puede aumentar o reducir según la señal más confiable
- Retorna un factor de corrección `[0.7, 1.2]` para aplicar al producto actual de capas

### Conexión D — Copulas Gaussianas ↔ Probabilidad Conjunta de Combos

El sistema usa `HR = prod(p_i)` para estimar la hit rate de un combo. Esto asume independencia entre picks. Es falso cuando dos picks están en el mismo torneo, misma superficie, mismo día.

Copula gaussiana: `P(X₁ > u₁, X₂ > u₂) = Φ₂(Φ⁻¹(u₁), Φ⁻¹(u₂), ρ)` donde ρ es la correlación.

**Tu tarea**: Propón la matriz de correlación estructural ρ entre picks:

```python
def rho_entre_picks(pick_a, pick_b):
    """
    Retorna correlación estructural entre dos picks del mismo combo.
    No correlación de resultados históricos (no tenemos esos datos),
    sino correlación de las CONDICIONES que determinan el resultado.
    """
    # picks del mismo torneo: condiciones de pista, árbitros, momento del torneo
    if pick_a['torneo'] == pick_b['torneo']:
        rho = ?
    # mismo tier, diferente torneo
    elif pick_a['tier'] == pick_b['tier']:
        rho = ?
    # diferente tier, misma superficie
    elif pick_a['superficie'] == pick_b['superficie']:
        rho = ?
    # todo diferente
    else:
        rho = ?
```

Y cómo esto modifica `HR_ajustado` en `_score_combo()` sin añadir librerías externas (scipy ya está disponible).

### Conexión E — Mecanismo de Diseño ↔ Zona Ciega del Bookmaker

El bookmaker tiene un modelo de pricing diseñado para jugadores Top 100. Para jugadores fuera de ese rango (200-400 en Challenger/ITF), está **fuera de su espacio de diseño** — usa datos escasos y modelos menos calibrados.

En teoría de mecanismos: cuando un agente opera fuera de su dominio de diseño, su comportamiento produce *precios sistemáticamente erróneos* — no aleatoriamente erróneos.

**Tu tarea**: Define el **Índice de Zona Ciega del Bookmaker** (IZCB):

```python
def indice_zona_ciega_bookmaker(ranking_favorito, tier, n_h2h_disponibles, cuota):
    """
    Estima qué tan lejos está el jugador del 'dominio de diseño' del bookmaker.
    Alto IZCB = bookmaker menos calibrado = nuestra ventaja informacional mayor.
    """
```

Criterios a considerar:
- Ranking: < 100 → dentro del dominio | > 200 → zona gris | > 350 → zona ciega
- Tier: GS/ATP1000 → bien cubierto | Challenger → parcial | ITF → mínimo
- n_h2h: < 3 → bookmaker sin referencia directa
- cuota: > 3.00 en tier bajo = bookmaker muy incierto del resultado

IZCB alto → aumentar phi_idiosincrático. ¿Cuánto? Propón la función completa.

---

## P2.3 — Los 5 entregables concretos

Proporciona código Python implementable para cada uno:

### Entregable 1 — Discriminador de picks LOW ganadores

```python
def discriminar_low_pick(pick_dict: dict) -> float:
    """
    Toma el output de calcular_edge_completo() para un pick con confidence_flag='LOW'.
    Retorna multiplicador [0.5, 1.5] sobre kelly_kl_ajustado.
    
    > 1.0: este LOW tiene señales que sugieren victoria real
    < 1.0: este LOW es ruido, el modelo no tiene ventaja real
    = 1.0: señal neutra, comportamiento actual preservado
    
    Debe usar SOLO campos ya presentes en pick_dict.
    """
    # Tu implementación aquí
```

**Restricción**: no puede requerir datos externos. Solo usa: `phi_idiosincratico`, `psi_entropia`, `data_completeness`, `alpha_vs_elo`, `n_h2h`, `tier`, `superficie`, `ranking_favorito`, `score_breakdown`, `calibration_confidence`.

### Entregable 2 — Bonus de asimetría informacional

```python
def info_asymmetry_bonus(ranking: int, tier: str, n_h2h: int,
                          data_completeness: float, cuota: float) -> float:
    """
    Retorna multiplicador [0.8, 1.4] que captura la ventaja informacional
    estructural cuando el bookmaker tiene menos datos que el modelo.
    
    Se aplica sobre phi_idiosincratico en edge_calculator.py.
    Debe retornar 1.0 para Top 30 en Grand Slam (bookmaker bien informado).
    """
```

Incluye tabla de sensibilidad: ¿qué retorna para ranking={50, 150, 300, 450} × tier={'grand_slam','atp500','challenger','itf'}?

### Entregable 3 — Factor de coherencia de señales

```python
def coherencia_señales(phi: float, psi: float, markov_state: str,
                        alpha_vs_elo: float, freshness_pelt: bool,
                        recencia_regimen: int) -> float:
    """
    Captura si las 5 señales son independientes (multiplican correctamente)
    o si están correlacionadas (el producto actual sobrevalora la confianza).
    
    Retorna factor de corrección [0.7, 1.2] para aplicar al producto Kelly-KL.
    Cuando todas las señales son neutrales, retorna 1.0.
    """
```

### Entregable 4 — Scoring extendido para combos 7-10 piernas

```python
def score_combo_largo(combo: list, picks_dict: dict,
                       n_piernas_objetivo: int = 7) -> float:
    """
    Versión extendida de _score_combo() para combos de 7-10 piernas.
    Diferencias vs versión actual:
    - HR ajustado por copula gaussiana (ρ estructural entre picks)
    - Gate de calibración: picks con ccf < 0.5 penalizados en combos largos
    - Peso adicional por IZCB (Índice Zona Ciega Bookmaker) en picks favorables
    
    Compatible con _select_with_cobertura() existente.
    """
```

### Entregable 5 — Thompson Beta conservador para n < 10

```python
def theta_thompson_conservador(wins: int, losses: int,
                                 percentil: float = 0.20) -> float:
    """
    Reemplaza el uso de la media Beta(wins+1, losses+1) cuando n < 10.
    Usa el percentil 20% de la distribución — estimación conservadora.
    
    Para n >= 10: comportamiento actual (media o sampling).
    Para n < 10: scipy.stats.beta.ppf(percentil, wins+1, losses+1)
    
    Resultado: grass_atp500 n=4 (4W/0L): media=0.833 → p20=0.548
               clay_grand_slam n=33 (25W/8L): prácticamente sin cambio
    """
    from scipy.stats import beta as beta_dist
    n = wins + losses
    if n < 10:
        return beta_dist.ppf(percentil, wins + 1, losses + 1)
    else:
        return (wins + 1) / (wins + losses + 2)  # comportamiento actual
```

Verifica que esta función, aplicada a la calibración del 2026-06-12, hubiera cambiado los stakes de la sesión ATP500 grass. ¿En cuánto?

---

## P2.4 — Protocolo de validación requerido

Para cada entregable, incluye:

**Caso de prueba obligatorio**: ¿Qué retorna la función para:
- Romano (challenger, grass, ranking 310, ccf=0.167, data_completeness=0.72, n_h2h=0)?
- Bonzi (ATP500, grass, ranking 68, ccf=0.167, data_completeness=0.88, n_h2h=2)?
- El ratio Romano/Bonzi debe ser > 1.0 para que el entregable sea útil.

**Hipótesis nula**: ¿Qué retorna cuando todos los inputs son neutros? Debe ser 1.0.

**Rango realista**: ¿En qué rango oscila el output con datos reales de producción?

---

## P2.5 — La trampa que debes evitar

Existe UNA fórmula que parece elegante pero no debes proponer porque requiere datos que el sistema NO tiene:

❌ Movimiento de línea intra-día: `(cuota_cierre - cuota_apertura) / cuota_apertura`
- Betplay no expone historial de cuotas. Solo tenemos cuota actual al momento del PASO 1.
- Una fórmula que requiera esto no es implementable hoy.

✅ Lo que SÍ está disponible: todo lo que `extraer_partidos_api.py` extrae via Kambi en tiempo real + H2H de Ninja FlashScore. El sistema captura cuota_es_real=True pero es un snapshot, no una serie temporal.

---

## P2.6 — El gran objetivo: los combos épicos como rutina

La jornada de 2026-06-13 demostró:
- Con 10 picks correctos a cuotas entre @1.65 y @2.95, se generan combos de @99 (7p), @269 (8p), @633 (9p), @1866 (10p).
- Con $500 por combo, el retorno potencial es $933,000.
- El desafío: esto ocurrió UNA VEZ. ¿Cómo hacer que sea más frecuente?

La respuesta no es "apostar más". La respuesta es:
1. Identificar qué perfil de picks genera estas cadenas (el discriminador del Entregable 1)
2. Optimizar la selección de piernas en los combos largos (Entregable 4)
3. Calibrar con más precisión los tiers con menos datos (Entregable 5)

**Pregunta final para el modelo**: ¿Cuál sería la frecuencia esperada de sesiones con 8+ picks correctos si el sistema implementara los 5 entregables, asumiendo calibración global p=0.661? ¿Y si se filtrara solo a picks con IZCB > 0.7 (zona ciega bookmaker alta)?

Usa la distribución binomial y los datos de calibracion_edge.json para dar un número.

---

## INSTRUCCIONES FINALES AL MODELO

1. **Ejecuta Pasada 1 primero** — diagnóstico sin soluciones. Luego espera.
2. Cuando recibas confirmación, **ejecuta Pasada 2** con el diagnóstico adjunto.
3. Para cada fórmula: código Python completo, no pseudocódigo.
4. Para cada conexión de dominio: nombra explícitamente el autor/teoría de origen (Boltzmann, Shannon, Nash, Markowitz, Sklar para copulas).
5. Ordena los 5 entregables por **impacto estimado × facilidad de implementación** al final.

**Criterio de éxito**: El sistema debe producir más jornadas donde todos los picks de Challenger/ITF ganan y los combos de 7-10 piernas se vuelven una estrategia repetible — no un evento de una vez.

---

*Generado por: Marco SWE + Marco Data + Marco Architect + Marco Quant*
*Sistema: Tennis EV Pipeline v2.0 | Bankroll: $125,000 | 706 sesiones calibradas*
*2026-06-13*

# FABLE — Análisis Profundo del Sistema de Predicción de Tenis
> Generado: 2026-06-13 | Sesión épica activa
> Propósito: encontrar conexiones ocultas, fórmulas nuevas y mejoras revolucionarias

---

## 0. CONTEXTO: LA SESIÓN QUE CASI FUE HISTÓRICA

Esta jornada (2026-06-12/13) produjo la mejor sesión del sistema hasta la fecha.
Con **$5,000 invertidos** en 10 combos de $500 cada uno, el resultado fue:

### Combos ya cobrados (en cuenta ahora)

| Combo | Picks | Cuota | Stake | Pago | Ganancia |
|---|---|---|---|---|---|
| Triple | Romano + Daniel + Fearnley | @9.33 | $500 | $2,039 | +$1,539 |
| Triple | Romano + Daniel + Bu | @9.17 | $500 | $1,956 | +$1,456 |
| Triple | Romano + Fearnley + Bu | @7.62 | $500 | $3,811 | +$3,311 |
| Triple | Daniel + Fearnley + Bu | @5.60 | $500 | $1,195 | +$695 |
| Cuádruple | Romano + Daniel + Fearnley + Bu | @15.40 | $500 | $3,208 | +$2,708 |
| Triple | Romano + Daniel + Fearnley | @9.33 | $500 | $1,944 | +$1,444 |
| **TOTAL COBRADO** | | | **$3,000** | **$14,153** | **+$11,153** |

**El usuario cobró anticipadamente** los combos de challenger puro (Romano, Daniel, Fearnley, Bu) antes de que terminaran los picks ITF. Decisión conservadora correcta — ya tenía ganancia asegurada.

### Combos abiertos (AÚN VIVOS — 6/N ganadas)

Los 6 picks ya confirmados ganadores en TODOS los combos abiertos:
- ✅ **Yunchaokete Bu** @1.65 — Ilkley Challenger (0-2)
- ✅ **Jacob Fearnley** @1.68 — Ilkley Challenger (0-2)
- ✅ **Filippo Romano** @2.75 — Ilkley Challenger (1-2)
- ✅ **Taro Daniel** @2.02 — Bratislava Challenger (2)
- ✅ **Pablo Martinez Gomez** @1.50 — ITF Martos (1)
- ✅ **Jenny Lim** @2.18 — ITF Niza (2)

Los picks pendientes que deciden el potencial:

| Pick | Cuota | p_blend | Torneo | Estado |
|---|---|---|---|---|
| **Stefano D'Agostino** | @1.98 | 0.590 | ITF Kursumlijska Banja | EN VIVO |
| **Luis Felipe Miguel** | @2.35 | 0.655 | ITF Brasilia | EN VIVO |
| **Kaitlyn Carnicella** | @2.70 | 0.590 | ITF Los Angeles | 14:30 |
| **Zuzanna Pawlikowska** | @2.95 | 0.590 | ITF Decatur | 13:00 |

### Escalera de retornos potenciales

```
Si gana D'Agostino solo    → +$16,716   (7-piernas @99.70)    P=59.0%
Si ganan D'Ag + Carnicella → +$134,593  (Óctuple @269.2)      P=34.8%
Si ganan 3 de 4            → +$316,294  (Nónupla @632.6)      P=20.5%
Si ganan LOS 4             → +$933,069  (Décupla @1866)       P=13.5%

POTENCIAL TOTAL acumulado si todos ganan: $1,400,672
Inversión total: $5,000
ROI máximo: 27,913%
```

### La lección épica

El usuario **cobró anticipado** los combos puros de challenger.
Los combos mezclados (challenger + ITF) siguen vivos con potenciales de $134k-$933k.
Con $500 por combo y 6 de N picks ya ganados — **esto es el sistema funcionando exactamente como fue diseñado**.

---

## 1. ARQUITECTURA DEL SISTEMA (para tu contexto)

### Pipeline completo

```
PASO 1: extraer_partidos_api.py        → cuotas reales Betplay (Kambi API)
PASO 2: extraer_historh2h.py --api     → H2H + rankings (FlashScore Ninja)
PASO 3: edge_calculator.py             → Kelly-KL 5 capas → APOSTAR/WATCHLIST/SIN_EDGE
PASO 4: trader_ev_tenis.py             → Portfolio Kelly + VaR + Cobertura Exclusión
PASO 4.5: betplay_combo_builder.py     → URLs Betplay + Telegram + .bat escritorio
PASO 4.6: betslip_registrar.py         → registra apuesta → cierra loop → calibración
```

### El corazón: `analysis/rivalry_analyzer.py` (1,564 líneas)

Calcula 8 componentes con pesos diferenciados por tier (SNR por estructura de mercado):

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

### Capa financiera: `edge_calculator.py`

```
f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica)) × φ × ψ

Donde:
  f*_clásico = (P_modelo - P_implícita) / (1 - P_implícita)
  λ = lambda_base(zona_cuota) × TIER_MULTIPLIER × PELT_RECENCY_ALPHA
  φ = phi_idiosincratico(score_breakdown)   [Fama-French adaptado]
  ψ = psi_entropia_shannon(cuota1, cuota2)  [bookmaker uncertainty]
  P_histórica = theta_thompson(calibracion, superficie, tier)  [Thompson Beta]

TIER_MULTIPLIER: GS=1.0x | ATP1000=1.6x | ATP500=2.4x | Challenger=3.6x | ITF=4.5x
```

### Módulos ya implementados

| Módulo | Qué hace | Archivo |
|---|---|---|
| PELT Change-Point | Detecta cambios de régimen HOT/COLD/NEUTRAL | `analysis/markov_analyzer.py` |
| H2H Immunity Dampener | HOT × h2h_wr < 0.30 → factor=0.85 | `analysis/rivalry_analyzer.py` |
| PageRank Erdős | Centralidad en grafo de victorias transitivas | `analysis/erdos_graph.py` |
| James-Stein Shrinkage | Pesos × n_tier/(n_tier+20) | `analysis/rivalry_analyzer.py` |
| density_confidence | Densidad local del grafo [0.3, 1.0] | `analysis/rivalry_analyzer.py` |
| K-factor ELO adaptivo | GS=24 / ATP1000=28 / CH=40 / ITF=48 | `analysis/elo_system.py` |
| Portfolio Kelly | f_portfolio = f_ind / (1 + ρ×(N-1)) | `trader_ev_tenis.py` |
| VaR/CVaR auto-ajuste | MAX_VAR_PCT=25% → reduce stakes | `trader_ev_tenis.py` |
| Cobertura Exclusión | C(N,K) combos → si X falla, ≥1 combo sobrevive | `trader_ev_tenis.py` |
| confidence_flag | STRONG/MODERATE/LOW por p_modelo | `edge_calculator.py` |
| calibration_confidence | James-Stein en Kelly: n/(n+20) | `edge_calculator.py` |
| prior conservador B-08 | min(fallback_tier, p_superficie) | `edge_calculator.py` |

### Calibración histórica (n=706 sesiones)

```json
{
  "global":           {"wins": 467, "losses": 239, "p": 0.661},
  "clay":             {"wins": 327, "losses": 138, "p": 0.702},
  "grass":            {"wins": 32,  "losses": 24,  "p": 0.569},
  "hard":             {"wins": 58,  "losses": 40,  "p": 0.590},
  "clay_grand_slam":  {"wins": 25,  "losses": 8,   "p": 0.758},
  "clay_challenger":  {"wins": 22,  "losses": 15,  "p": 0.595},
  "grass_atp500":     {"wins": 4,   "losses": 0,   "n": 4},
  "fallback_por_tier": {
    "grand_slam": 0.7576,
    "atp1000":    0.700,
    "atp500":     0.650,
    "challenger": 0.611
  }
}
```

---

## 2. LO QUE APRENDIMOS ESTA SEMANA

### La jornada épica revela el patrón ganador real

**Los 4 picks del challenger grass que ganaron HOY (100%):**
- Romano @2.75 | Fearnley @1.68 | Bu @1.65 | Daniel @2.02
- Todos: conf < 55%, flag=LOW, ccf=0.30
- Tier: challenger | Superficie: grass
- Lambda efectivo: 1.08 (penalización correcta)

**Los 4 picks ATP500 grass que perdieron AYER (0%):**
- Bonzi @4.50 | T.Maria @4.35 | Ruse @2.75 | Perricard @2.60
- También: conf < 55%, flag=LOW
- Tier: ATP500 | Superficie: grass
- Lambda efectivo: 0.72 (penalización insuficiente)

### La diferencia que el modelo no captura todavía

| Factor | Challenger ganador | ATP500 perdedor |
|---|---|---|
| Ranking promedio jugadores | ~200-400 ATP | Top 50-100 ATP |
| Varianza de rendimiento | Alta (más impredecible para bookmaker) | Baja (bookmaker los conoce bien) |
| Historial en superficie | Menos datos → más incertidumbre del mercado | Más datos → cuotas más precisas |
| Lambda | 1.08 ✅ | 0.72 ❌ (muy bajo) |
| p_historica | 0.611 (conservador) | 0.650 → 0.569 post B-08 |
| **P_modelo vs P_implícita** | 50-54% vs 36-60% | 50-54% vs 22-38% |

### El insight clave: el bookmaker sabe más de los Top 100 que de los jugadores 200-500

Cuando Betplay pone a Bonzi @4.50, tienen modelos sofisticados sobre sus últimos 50 partidos.
Cuando ponen a Romano @2.75 (ranking ~300), tienen menos datos internos.
**Nuestro modelo puede tener ventaja informacional mayor en picks de menor perfil.**

---

## 3. PREGUNTAS PARA FABLE — CONEXIONES OCULTAS

### A. Diagnóstico estructural profundo

1. Los pesos de los 8 componentes son estáticos dentro de cada tier. ¿Cómo debería cambiar el peso de `form_recent` si el jugador lleva 3 meses sin jugar vs 3 días desde su último partido? ¿Hay una función de decaimiento temporal que capture esto?

2. La `density_confidence` modula `common_opponents` pero no los otros 7 componentes. ¿Debería existir un factor global de "confianza en la predicción" que escale TODOS los componentes según la riqueza de datos disponibles?

3. `strength_of_schedule` tiene peso 0.00 en Grand Slam. La justificación implícita es que en GS todos los jugadores han pasado por calendarios similares. ¿Es esto correcto? ¿O hay información de calendario pre-torneo que predice rendimiento en GS?

### B. Conexiones entre áreas del conocimiento

4. **Física termodinámica ↔ Momentum deportivo**: ¿Se puede modelar el estado de un jugador como temperatura cinética?
   ```
   T_jugador = Σᵢ resultado_i × cuota_rival_i × exp(-días_desde_partido_i / τ)
   ```
   donde τ ≈ 14 días (vida media del momentum). Un jugador HOT con victorias recientes contra rivales fuertes tendría T alta. ¿Esto supera al Markov PELT actual?

5. **Teoría de información ↔ Edge financiero**: El ψ_entropía actual mide incertidumbre del bookmaker sobre el resultado. Pero la distribución de cuotas también codifica información sobre el "mercado de opiniones". Si cuota_A cambia de 2.00 a 1.75 en las últimas 6 horas, ¿qué dice eso sobre el flujo de apuestas? ¿Podemos capturar movimiento de línea como señal?

6. **Redes neuronales de grafos ↔ Erdős PageRank**: El PageRank actual trata todos los arcos del grafo de victorias como iguales. Un GNN (Graph Neural Network) aprendería pesos de aristas en función de: ranking del rival, superficie, recencia, marcador. ¿Qué features de las aristas ya tenemos disponibles en los datos H2H?

7. **Teoría de juegos ↔ H2H adaptativo**: Algunos jugadores tienen estrategias específicas contra ciertos estilos de juego que el win_rate puro no captura. ¿Se puede inferir "compatibilidad de estilos" desde el score_breakdown de partidos H2H? (Un jugador con surface_spec alto contra rival con sos alto = mismatch)

8. **Mercados financieros ↔ Cobertura por exclusión**: La Cobertura Exclusión actual ordena combos por EV esperado. La teoría de portafolios de Markowitz ordena activos por media-varianza. ¿Cuál es la frontera eficiente de combos (EV máximo para varianza dada)?

### C. El problema de los picks LOW que ganan

La jornada de ayer revela una paradoja:
- Romano conf=50.4% → GANÓ (challenger grass)
- Bonzi conf=54.0% → PERDIÓ (ATP500 grass)

Romano tiene MENOR confianza pero ganó. Bonzi tiene MAYOR confianza pero perdió.

**Pregunta central**: ¿Qué combinación de features en el score_breakdown diferencia un LOW ganador de un LOW perdedor? Hipótesis a explorar:
- ¿Los picks LOW que ganan tienen phi_idiosincratico > 1.05 (factores que el bookmaker no modela)?
- ¿Los picks LOW que ganan tienen data_completeness ≥ 0.75 (modelo tiene todos los datos)?
- ¿Los picks LOW que ganan provienen de tiers con mayor varianza de rendimiento (challenger, ITF)?
- ¿Existe un "score de informational advantage" = phi × (1 - P_implícita) que prediga LOW ganadores?

### D. La gran oportunidad: combos de bajo stake, alta cuota

La jornada de hoy demostró el potencial real:
```
$500 × @1866 = $933,069 potencial (décupla pendiente)
$500 × @269  = $134,593 potencial (óctuple pendiente)
```

La estrategia actual genera combos maximizando EV. Pero para combos de alto número de piernas (7-10), el objetivo debería ser maximizar P(≥K de N ganan) — no el EV esperado.

**Pregunta**: ¿Existe una fórmula para seleccionar las N piernas de un combo de K-piernas que maximice P(todas ganan) bajo correlación ρ entre piernas del mismo tier/superficie?

```
Hipótesis: usar copulas de Gaussian para modelar la correlación 
entre resultados de picks del mismo torneo/día vs picks de torneos distintos.
Picks de Ilkley Challenger mismo día: ρ ≈ 0.30
Picks de torneos distintos mismo día: ρ ≈ 0.05
```

---

## 4. DATOS DISPONIBLES QUE AÚN NO EXPLOTAMOS

### En cada partido del h2h_results_enhanced

```json
{
  "match_url": "...",
  "jugador1": "...", "jugador2": "...",
  "cuota1": 2.75, "cuota2": 1.45,
  "superficie": "grass", "torneo_nombre": "...",
  "ranking_analysis": {
    "prediction": {
      "confidence": 51.0,
      "markov_analysis": {
        "jugador1": {
          "estado_actual": "HOT",
          "win_rate_reciente": 0.75,
          "win_rate_anterior": 0.45,
          "change_point": 8,
          "n_partidos": 15,
          "factor_tardio": 0.95,
          "immunity_factor": 0.85
        }
      },
      "score_breakdown": {
        "player1": {
          "surface_specialization": {"score": 24.6, "contribution": "23.3%"},
          "form_recent":            {"score": 18.2, "contribution": "28.1%"},
          "common_opponents":       {"score": 12.1, "contribution": "15.4%"},
          "h2h_direct":             {"score":  0.0, "contribution": "0.0%"},
          "ranking_momentum":       {"score":  8.5, "contribution": "12.2%"},
          "elo_rating":             {"score": 15.3, "contribution": "19.3%"},
          "home_advantage":         {"score":  2.1, "contribution": "1.7%"},
          "strength_of_schedule":   {"score":  0.0, "contribution": "0.0%"}
        }
      }
    }
  }
}
```

**Campos sub-explotados:**
- `factor_tardio`: ajuste por fatiga acumulada en el torneo
- `immunity_factor`: el H2H dampener ya calculado pero ¿lo usamos en el selector de combos?
- `win_rate_reciente` vs `win_rate_anterior`: delta de rendimiento (¿tendencia?)
- `n_partidos`: cuántos partidos tiene el jugador en su historial Markov
- `contribution %` por componente: ¿hay un perfil de componentes que prediga mejor?

### En calibracion_edge.json (lo que necesitamos añadir)

Actualmente: wins/losses por superficie y tier.
Faltante: **wins/losses por rango de confianza + tier + superficie**.

Si supiéramos que "picks confidence 50-53% en challenger grass ganan el 68% del tiempo", 
podríamos usar p_historica mucho más granular.

---

## 5. INSTRUCCIONES PARA FABLE

Eres un científico cuántico con expertise en teoría de grafos, finanzas cuantitativas, física estadística, psicología deportiva y machine learning bayesiano.

**Tu objetivo**: analizar este sistema y encontrar las 5 conexiones más poderosas entre áreas del conocimiento que ningún analista convencional ha identificado.

**Organiza tu respuesta en 5 secciones:**

### A) DIAGNÓSTICO FUNDAMENTAL
¿Cuál es la limitación estructural más profunda? No síntomas — causa raíz.
¿Por qué el sistema genera 92% de picks con conf < 55%?

### B) CONEXIONES OCULTAS (3-5 hallazgos)
Para cada conexión: datos disponibles → señal nueva → impacto estimado en accuracy.
Prioriza conexiones que se puedan implementar sobre la arquitectura existente.

### C) FÓRMULAS MATEMÁTICAS NUEVAS
Propuestas concretas. Por ejemplo:
```
Temperatura_Jugador(t) = Σᵢ resultado_i × ranking_factor_i × e^(-Δt_i/τ)
```
No conceptos vagos — fórmulas implementables en Python.

### D) SELECTOR DE COMBOS REVOLUCIONARIO
Cómo pasar de "maximizar EV" a "maximizar P(≥K de N ganan) bajo correlación estructural".
La jornada de hoy demostró que un décupla @1866 con $500 puede valer $933k.
¿Qué teoría matemática optimiza la selección de piernas para estos combos épicos?

### E) ROADMAP DE IMPLEMENTACIÓN
Top 5 mejoras ordenadas por: impacto × facilidad.
Solo sobre la arquitectura existente. No reescribir desde cero.

**Criterio de éxito**: que el sistema produzca más jornadas como la de hoy — donde todos los picks de challenger/ITF aciertan y los combos de 7-10 piernas se vuelven rutina, no excepción.

---

## 6. ESTADO ACTUAL DEL SISTEMA (resumen ejecutivo)

```
Tests:          980 passed, 0 failed
Bankroll:       $125,000
Calibración:    467W/239L  n=706  p=0.661
Mejor tier:     clay_grand_slam p=0.758 (n=33)
Bugs activos:   0 críticos (B-01 a B-10 todos resueltos)
Sesión épica:   $5,000 invertidos → $14,153 cobrados + combos abiertos
                Potencial máximo si pendientes ganan: $1,400,672
Picks pendientes HOY:
  - Stefano D'Agostino @1.98 (EN VIVO)
  - Luis Felipe Miguel @2.35 (EN VIVO)
  - Kaitlyn Carnicella @2.70 (14:30)
  - Zuzanna Pawlikowska @2.95 (13:00)
```

---

*Este documento fue generado por el pipeline de análisis cuantitativo de tenis.*
*Sesión: 2026-06-13 | Modelo activo: Sonnet 4.6 + Opus 4.6 (subagentes)*

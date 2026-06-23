# Nodo-32 --- Calibracion Pipeline: Phantom Edge, Señales Rotas y Mapa Completo de Parametros

> **Fecha:** 2026-06-22
> **Severidad:** CRITICA — P&L negativo, 26.7% hit rate en picks APOSTAR vs 75.8% calibracion historica
> **Prerequisitos:** Nodo-01, Nodo-02, Nodo-18, Nodo-19, Nodo-21, Nodo-24, Nodo-27
> **Archivos afectados:** `edge_calculator.py`, `analysis/rivalry_analyzer.py`, `betplay_combo_builder.py`, `betslip_registrar.py`, `data/calibracion_edge.json`
> **Implementa:** Sonnet | **Tests:** Haiku

---

## 0. CONTEXTO DE EMERGENCIA

### Que paso

| Metrica | Esperado | Real | Delta |
|---------|----------|------|-------|
| Hit% picks APOSTAR | >55% | 26.7% (8W/22L) | -28pp |
| Markov HOT hit% | >65% | 8.3% (1W/11L) | -57pp |
| Golden Zone hit% | >60% | 12.5% (1W/7L) | -48pp |
| STRONG confidence hit% | >70% | 0.0% (0W/1L) | N/A (n=1) |
| LOW confidence % del pool | <30% | 86% (221/257) | +56pp |
| ROI proxy | >0% | -21.1% | Fatal |

### Que NO paso

El modelo general NO colapso. La accuracy global sigue en ~57-62%. Lo que fallo es el **mecanismo de seleccion de apuestas** — el edge calculator selecciona picks donde el modelo tiene ~51% de conviccion (moneda al aire) y los trata como edge >15% porque el bookmaker los pricea como underdogs.

**Esto es el Betting Paradox:** un modelo que predice 62% overall puede perder dinero si apuesta solo en el subconjunto donde discrepa con el mercado, y esa discrepancia viene de ruido, no de informacion.

---

## 1. MAPA COMPLETO DEL PIPELINE — Flujo de Señales

```
SCRAPING (Paso 1-2)
    |
    v
RIVALRY_ANALYZER (8 componentes raw → normalizado → ponderado → confidence)
    |
    v
EDGE_CALCULATOR (confidence → p_modelo → edge → Kelly-KL 5 capas → APOSTAR/WATCHLIST/SIN_EDGE)
    |
    v
TRADER (picks APOSTAR → Portfolio Kelly → VaR → Cobertura → Stakes)
    |
    v
COMBO_BUILDER (stakes → combos Betplay → betslip_index)
    |
    v
BETSLIP_REGISTRAR (resultado → calibracion_edge.json → Thompson priors)
```

---

## 2. INVENTARIO EXHAUSTIVO DE PARAMETROS

### 2.1 RIVALRY_ANALYZER.PY — Motor de Prediccion

#### 2.1.1 Pesos por Tier (DEFAULT_WEIGHTS)

Definidos en `rivalry_analyzer.py:1319-1340` y `normalization.py:46-117`. La copia en rivalry_analyzer es la que se usa en runtime.

| Componente | grand_slam | atp1000 | atp500 | challenger | itf | Rol en prediccion |
|------------|-----------|---------|--------|------------|-----|-------------------|
| surface_specialization | 0.15 | 0.16 | 0.15 | 0.20 | 0.15 | Calidad historica en superficie |
| form_recent | 0.12 | 0.15 | 0.18 | 0.22 | **0.28** | Forma reciente + rachas |
| common_opponents | **0.22** | 0.20 | 0.15 | 0.08 | 0.05 | Red transitiva Erdos |
| h2h_direct | **0.18** | 0.14 | 0.10 | 0.03 | 0.02 | Enfrentamientos directos |
| ranking_momentum | 0.15 | 0.17 | 0.20 | 0.22 | 0.22 | Puntos + momentum ranking |
| elo_rating | 0.13 | 0.13 | 0.12 | 0.15 | 0.15 | ELO calibrado por tier |
| home_advantage | 0.05 | 0.05 | 0.05 | 0.05 | 0.08 | Ventaja local |
| strength_of_schedule | 0.00 | 0.00 | 0.05 | 0.05 | 0.05 | Calendario oponentes |

**Suma = 1.00 por tier.** Todos hardcoded. Fallback tier = atp500.

**Insight critico:** En ITF (44% de picks APOSTAR), `form_recent=0.28` domina pero `h2h_direct=0.02` y `common_opponents=0.05` son casi irrelevantes. El modelo depende casi enteramente de forma + ranking — señales que el bookmaker tambien ve.

#### 2.1.2 Modificadores de Pesos (aplicados secuencialmente)

**A. James-Stein Shrinkage** (`rivalry_analyzer.py:1344-1363`)
```
factor = n_tier / (n_tier + 20)
peso_final = factor * peso_tier + (1 - factor) * peso_atp500
```
- n_threshold = 20 (hardcoded)
- Fuente de n_tier: `calibracion_edge.json` campo `por_superficie_y_tier`
- n=0 → factor=0.00 (100% default atp500)
- n=31 → factor=0.61
- n=100 → factor=0.83
- **ESTADO:** Calibrado desde datos

**B. Density Confidence** (`rivalry_analyzer.py:1365-1377`)
```
density = 0.3 + 0.7 * ((min(n_common,20)/20 + min(n_paths,30)/30) / 2)
```
- Rango: [0.3, 1.0]
- Floor: 0.3 (hardcoded)
- Efecto: `weights['common_opponents'] *= density`, excedente → `form_recent`
- **BUG RELACIONADO:** Con density=0.3 (sin common opponents), el peso de form_recent SUBE, amplificando una señal que el bookmaker ya pricea.

**C. Ajuste Superficie Clay** (`rivalry_analyzer.py:1384-1389`)
- common_opponents += 0.08
- ranking_momentum -= 0.08
- Hardcoded, solo clay.

**D. Ajuste Superficie Grass** (`rivalry_analyzer.py:1391-1397`)
- common_opponents -= 0.05
- form_recent += 0.05
- Hardcoded, solo grass.

**E. Torneo Completo Boost** (`rivalry_analyzer.py:1410-1421`)
- Trigger: `torneo_completo=True` AND `form_recent_weight > 0.10`
- surface_specialization += 0.07, form_recent -= 0.07
- Hardcoded.

**F. Circuit Asymmetry SoS Shift** (`rivalry_analyzer.py:1739-1747`)
- Solo cuando CAD dispara AND `strength_of_schedule` base > 0
- `sos_multiplier = 1.0 + math.log(asimetria)`
- Mueve peso de form_recent a strength_of_schedule

#### 2.1.3 Raw Score Caps (MAX_RAW_SCORES)

| Componente | Cap | Linea enforcement | Normalizacion |
|------------|-----|-------------------|---------------|
| home_advantage | 100 | 1458 | log1p |
| surface_specialization | 350 | 1462 | LINEAR (especial) |
| ranking_momentum | 450 | 1497 | log1p |
| form_recent | 300 | 1507, 1583, 1621, 1727 | log1p |
| common_opponents | 400 | 1511 | log1p |
| h2h_direct | 350 | 1514 | log1p |
| elo_rating | 250 | 1519, 1728-1736 | log1p |
| strength_of_schedule | 200 | 1522 | log1p |

#### 2.1.4 Normalizacion — EL BUG CRITICO

**Dos modos** (`rivalry_analyzer.py:1771-1796`):

1. **surface_specialization** usa normalizacion **LINEAR**:
   ```
   norm = min(raw / max_expected, 1.0) * math.log1p(max_expected)
   ```
   Preserva ratio entre jugadores. max_expected=350 → escala a ~5.86

2. **Todos los demas** usan **log1p**:
   ```
   norm = math.log1p(raw_value)
   ```

**POR QUE ESTO MATA A MARKOV:**
- Markov factor rango [0.85, 1.15] multiplica `form_recent` ANTES de normalizacion
- form_recent = 200 → HOT (×1.075) → 215
- Despues de log1p: log1p(200)=5.303 vs log1p(215)=5.375
- **Delta = 0.072** sobre un total ponderado de ~5.0
- Con peso form_recent = 0.22 → contribucion = 0.016
- Efecto en confidence: **< 0.35 puntos porcentuales**
- **Markov es DECORATIVO — no afecta la prediccion**

#### 2.1.5 Score Final y Formula de Confianza

```python
# rivalry_analyzer.py:1805-1813
weighted[k] = normalized[k] * weights[k]
final_score = sum(weighted) - inactivity_penalty

# rivalry_analyzer.py:1842-1844
confidence = 50 + (abs(score_diff) / total_score) * 50
# Clamped: [50, 95]
```

**POR QUE ESTO GENERA p_modelo ≈ 0.51:**
- Cuando ambos jugadores tienen datos similares, `score_diff` es pequeño
- Ejemplo: P1=5.2, P2=5.0 → diff=0.2, total=10.2
- confidence = 50 + (0.2/10.2)*50 = 50 + 0.98 = **50.98%**
- p_modelo = 0.5098 → "moneda al aire"
- Pero con cuota=3.0: edge = 0.5098 - 0.333 = **17.6% "edge fantasma"**

#### 2.1.6 Factores Markov / PELT

**PELT Change-Point** (`markov_analyzer.py:27-105`)
- min_size = 5, umbral_cambio = 0.20
- n_recientes = 5 (ultimos 5 partidos para estado)
- HOT: win_rate >= 0.70 | COLD: <= 0.30 | NEUTRAL: entre ambos
- Confianza: `min(mejor_diferencia * 2, 1.0)`

**Factor Markov** (`markov_analyzer.py:160-180`)
```
diferencia = estado_p1 - estado_p2    # HOT=1, NEUTRAL=0, COLD=-1
factor = 1.0 + diferencia * 0.075     # Rango: [0.85, 1.15]
```

**H2H Immunity Dampener** (`rivalry_analyzer.py:1251-1285`)
- n_h2h < 3 → immunity = 1.00
- HOT + h2h_win_rate < 0.30 → immunity = 0.85
- HOT + h2h_win_rate > 0.70 → immunity = 1.12

**Factor Tardio** (`markov_analyzer.py:183-267`)
```
factor = 1.0 + (wr1 - wr2) * 0.15    # Rango: [0.85, 1.15]
```
- Solo partidos extendidos (sets_won + sets_lost >= 4)

**PELT Recency Alpha** (`markov_analyzer.py:108-157`) — Usado en edge_calculator, NO en rivalry_analyzer
- HOT + recencia <= 3: alpha = 1.20 (lambda ÷ 1.20)
- HOT + recencia <= 7: alpha = 1.10
- COLD + recencia <= 3: alpha = 0.85
- Otros: alpha = 1.00

#### 2.1.7 Componentes Especiales

**Erdos Graph** (`erdos_graph.py`)
- alpha decay = 0.7, max_depth = 3
- PageRank: damping=0.85, iteraciones=10, min 5 nodos
- erdos_score rango [-1, +1]

**ELO System** (`elo_system.py`)
- K-factor: GS=24, ATP1000=28, ATP500=32, Challenger=40, ITF=48
- Post-PELT reset: K × 1.5 si recencia <= 5
- Default ELO = 1500

**ELO desde Ranking** (`rivalry_analyzer.py:68-81`)
- Rank 1-10: 2200 - (rank-1)*20
- Rank 11-50: 2020 - (rank-11)*5
- Rank 51-100: 1820 - (rank-51)*2
- Rank 101-200: 1720 - (rank-101)*1
- Rank >200: 1600 | None: 1500

**Surface Specialization** (`rivalry_analyzer.py:720-904`)
- Victory points: Top 10=50, Top 20=40, Top 50=25, Top 100=15, >100=5
- Contundencia: 2-0 strong=1.5, 2-0 tight=1.2, 3 sets=1.0
- Resistencia: 1-2=0.5, 0-2 tight=0.3, 0-2 contundente=0.0
- Torneo completo bonus: base 1.3, +0.2 reciente, +0.1 Top10, +0.1 final, cap 2.0
- Volume confidence: min(n/8, 1.0)

**Streaks & Consistency** (`rivalry_analyzer.py:386-517`)
- 5+ win streak + 2 Top50: ×1.50
- 3+ win streak + 2 Top30: ×1.30
- Consistency >75% vs Rank 31-50: ×1.25
- Consistency >80% vs same tier: ×1.20
- Momentum >70% last 10: ×1.20
- 3+ Top20 wins in last 20: ×1.15
- Cap total: 1.50

---

### 2.2 EDGE_CALCULATOR.PY — Motor de Decision

#### 2.2.1 Constantes Core

| Parametro | Valor | Linea | Rol |
|-----------|-------|-------|-----|
| EDGE_MIN | 0.05 (5%) | 72 | Minimo edge para APOSTAR |
| KELLY_KL_MIN | 0.02 (2%) | 73 | Minimo Kelly-KL para confirmar |
| BANKROLL_CAP | 0.10 (10%) | 74 | Max fraccion bankroll por bet |
| eps (KL) | 1e-9 | 143, 437 | Evita log(0) |

#### 2.2.2 Formula Kelly-KL — 5 Capas

```
L1: Kelly clasico     = edge / (1 - p_implicita)
L2: KL divergence     = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    Kelly-KL base     = kelly_clasico * exp(-lambda * max(0, KL))
L3: Factor Phi        = 0.80 + frac_unknown * 0.50    # [0.80, 1.30]
L4: Factor Psi        = 0.85 + entropy * 0.30          # [0.85, 1.15]
L5: Calibration Conf  = max(0.30, n / (n + 20))        # [0.30, 1.00]

Kelly final = Kelly-KL * phi * psi * CCF
Fraccion    = min(Kelly_final, 0.10)                    # BANKROLL_CAP
```

#### 2.2.3 Lambda — Cadena Completa

```
lambda = lambda_zona * tier_multiplier / alpha_temporal
```

**Volatility Smile (lambda_zona)**:
| Zona cuota | Lambda | Rango cuota |
|------------|--------|-------------|
| heavy_favorite | 2.0 | < 1.30 |
| moderate_favorite | 1.0 | 1.30 - 1.59 |
| slight_underdog | 0.5 | 1.60 - 2.09 |
| underdog | 0.3 | >= 2.10 |

**Tier Multiplier**:
| Tier | Multiplicador |
|------|---------------|
| grand_slam | 1.0 |
| atp1000 | 1.6 |
| atp500 | 2.4 |
| challenger | 3.6 |
| itf | 4.5 |

**PELT Alpha Temporal**: HOT fresco → ÷1.20 | COLD fresco → ÷0.85

**Ejemplo lambda para pick tipico (ITF underdog HOT fresco):**
```
lambda = 0.3 * 4.5 / 1.20 = 1.125
```
Comparado con GS underdog sin PELT:
```
lambda = 0.3 * 1.0 / 1.0 = 0.30
```
ITF tiene 3.75x mas aversion — PERO no compensa el phantom edge.

#### 2.2.4 Factor Phi (Fama-French Decomposition)

`edge_calculator.py:239-258`

- **Known (bookmaker pricea):** elo_rating, ranking_momentum
- **Unknown (nuestro edge):** surface_specialization, form_recent, common_opponents, h2h_direct
- frac_unknown = sum(contrib_unknown) / sum(all_contrib)
- phi = 0.80 + frac_unknown * 0.50

**Problema:** Cuando density_confidence es baja, common_opponents cae y form_recent sube. form_recent es mitad-conocido por el bookmaker. Phi puede sobreestimar el edge idiosincratico.

#### 2.2.5 Confidence Flag — INFORMATIVO, NO DECISIVO

| p_modelo | Flag | Usado en gate? |
|----------|------|----------------|
| >= 0.60 | STRONG | NO |
| >= 0.55 | MODERATE | NO |
| < 0.55 | LOW | NO |

**BUG CRITICO:** El gate en linea 464 solo checa `edge > 0.05 AND kelly_kl > 0.02`. confidence_flag se calcula pero **NUNCA** se usa en la decision de APOSTAR. El 86% de picks LOW pasan el gate.

#### 2.2.6 Gate APOSTAR — Logica Completa

```python
# Linea 464 — Gate primario
apostar = edge > EDGE_MIN and kelly_kl_ajustado > KELLY_KL_MIN

# Linea 820-822 — FIX-3: requiere 2+ axes activos
if n_axes_active < 2 and apostar:
    apostar = False  # "BBI sola no predice"

# Linea 831-833 — FIX-6: HOT sin BBI
if markov_favorito == 'HOT' and bbi < 0.50 and apostar:
    apostar = False  # "bookmaker ya pricea momentum"
```

**Lo que FALTA:** No hay gate de p_modelo minimo. Un pick con p_modelo=0.503 pasa si la cuota es suficientemente alta.

#### 2.2.7 BBI, MPQ, Golden Zone

**BBI** (`edge_calculator.py:765-766`):
```
BBI = (1 - 1/cuota) * (1 / (1 + n_h2h * 0.20))
```

**MPQ** (`edge_calculator.py:781`):
```
MPQ = kelly_kl * BBI * (1 + edge_pct / 100)
```

**Golden Zone** (`edge_calculator.py:784`):
```
golden_zone = tier in ('challenger', 'itf') AND cuota >= 2.50 AND n_h2h == 0
```

**BUG:** `n_h2h == 0` selecciona picks donde el MODELO tambien esta ciego (density_confidence = 0.3), no solo el bookmaker.

#### 2.2.8 Triple Alignment (Nodo-28)

| Parametro | Valor | Linea |
|-----------|-------|-------|
| _SURFACE_SIGNAL_CAP | 0.25 | 504 |
| _BBI_CAP | 0.70 | 505 |
| _AXIS_THRESHOLD | 0.50 | 506 |
| _ALIGNMENT_STRONG | 0.40 | 507 |
| _ALIGNMENT_PARTIAL | 0.20 | 508 |

Flags: STRUCTURAL_ALPHA | CONTESTED_ALPHA | PARTIAL_ALIGNMENT | NO_ALIGNMENT

#### 2.2.9 Thompson Sampling (p_historica)

Jerarquia:
1. `por_superficie_y_tier[key]` → prefer era_v2 if n>=10, else regular if n>=10
2. `fallback_por_tier[tier]` → con B-08 clamp (divergencia > 0.03 → min)
3. `por_superficie[superficie]` → if n>=10
4. `global` → wins/losses

Formula: `p = (wins + 1) / (wins + losses + 2)` (Beta mean)

---

### 2.3 TRADER_EV_TENIS.PY — Hedge Fund Layer

| Parametro | Valor | Linea | Rol |
|-----------|-------|-------|-----|
| KELLY_FRACTION | 0.25 | 39 | Quarter-Kelly (conservador) |
| KELLY_CAP_IND | 0.10 | 37 | Max Kelly individual |
| KELLY_CAP_COMBO | 0.15 | 38 | Max Kelly combo |
| BUDGET_IND_PCT | 0.40 | 40 | 40% bankroll individuales |
| BUDGET_COMBO_PCT | 0.40 | 41 | 40% bankroll combos |
| BUDGET_SIS_PCT | 0.20 | 42 | 20% bankroll sistema |
| MIN_BET | 1000 | 43 | Minimo por apuesta |
| VAR_CONFIDENCE | 0.95 | 127 | 95th percentile |
| MAX_VAR_PCT | 0.25 | 128 | Max 25% bankroll en VaR |
| _K_PRIOR | 3 | 48 | Pseudo-observaciones Bayesian |
| _P_PRIOR | 0.52 | 49 | Prior neutral fallback |
| min-cuota default | 1.50 | 877 | REGLA-HF-1 |

**Portfolio Kelly por tier (rho)**:
| Tier | rho | factor N=4 |
|------|-----|------------|
| grand_slam | 0.25 | 0.571 |
| atp1000 | 0.20 | 0.625 |
| atp500 | 0.15 | 0.690 |
| challenger | 0.10 | 0.769 |
| itf | 0.05 | 0.870 |

**KGR (Kelly Growth Rate):**
```
g = E[log(1 + R)]    # g > 0 → crece | g < 0 → ruina → NO DESPLEGAR
```

---

### 2.4 CALIBRACION_EDGE.JSON — Estado Actual

| Key | Wins | Losses | n | Thompson | Estado |
|-----|------|--------|---|----------|--------|
| global | 850 | 503 | 1353 | 0.629 | OK |
| clay | 522 | 259 | 781 | 0.668 | OK |
| grass | 92 | 73 | 165 | 0.557 | OK |
| hard | 179 | 103 | 282 | 0.634 | OK |
| ? | 12 | 39 | 51 | 0.245 | **BASURA** |
| clay_grand_slam | 25 | 8 | 33 | 0.743 | OK |
| clay_challenger | 22 | 15 | 37 | 0.590 | OK |
| grass_challenger | 2 | 2 | 4 | 0.500 | n<10 |
| grass_atp500 | 4 | 0 | 4 | 0.833 | n<10 |
| clay_itf | 0 | 1 | 1 | 0.333 | n<10 |
| ?_? | 12 | 39 | 51 | 0.245 | **BASURA** |

**fallback_por_tier:**
| Tier | Prior | Nota |
|------|-------|------|
| grand_slam | 0.7576 | OK |
| atp1000 | 0.7000 | OK |
| atp500 | 0.6500 | OK |
| challenger | 0.6110 | OK |
| itf | **AUSENTE** | **BUG — cae a superficie o global** |

---

## 3. DIAGNOSTICO — 5 BUGS RAIZ

### BUG-32-1: Phantom Edge (CRITICO)

**Sintoma:** 86% de picks APOSTAR tienen confidence_flag=LOW (p_modelo < 0.55). Hit rate 26.7%.

**Causa raiz:** El gate en `edge_calculator.py:464` no tiene threshold de p_modelo minimo.

```python
# ACTUAL — linea 464
apostar = edge > EDGE_MIN and kelly_kl_ajustado > KELLY_KL_MIN
```

Un pick con p_modelo=0.503 y cuota=3.60:
- p_implicita = 1/3.60 = 0.278
- edge = 0.503 - 0.278 = **0.225 (22.5%)** ← FANTASMA
- kelly_clasico = 0.225 / (1 - 0.278) = 0.312
- Pasa ambos gates facilmente

**El modelo dice "es una moneda al aire" pero la aritmetica dice "apuesta fuerte".**

**Fix:** Agregar threshold de p_modelo minimo al gate.

### BUG-32-2: Markov Decorativo (ALTO)

**Sintoma:** Markov HOT hits at 8.3%, NEUTRAL at 37.5%. HOT no predice nada.

**Causa raiz:** `log1p()` comprime el factor Markov a ruido.
- Factor Markov rango: [0.85, 1.15] → aplicado a form_recent ANTES de log1p
- form_recent=200 × 1.075 (HOT vs NEUTRAL) = 215
- log1p(200)=5.303 vs log1p(215)=5.375 → **delta = 0.072**
- Con peso 0.22 → contribucion = **0.016 puntos** sobre score total ~5.0
- Efecto en confidence: **< 0.35pp** — indistinguible de ruido

**Fix:** Aplicar factor Markov DESPUES de normalizacion (post-log1p), no antes.

### BUG-32-3: Golden Zone Ciega (ALTO)

**Sintoma:** Golden Zone 12.5% hit rate — peor señal del portfolio.

**Causa raiz:** `n_h2h == 0` en `edge_calculator.py:784` filtra por ceguera del bookmaker PERO tambien garantiza ceguera del modelo (density_confidence = 0.3, h2h_direct = 0).

**Fix:** Cambiar definicion para requerir alguna señal del modelo.

### BUG-32-4: ?_? en Calibracion (MEDIO)

**Sintoma:** 51 partidos (12W/39L) clasificados como `?_?` en calibracion_edge.json.

**Causa raiz:** `betplay_combo_builder.py:1399-1414` no copia `superficie` ni `tier` al betslip_index.

```python
# ACTUAL — betplay_combo_builder.py:1399-1414
all_picks.append({
    "jugador":     p["favorito_predicho"],
    "cuota":       cuota,
    "edge":        p.get("edge_pct", "0%"),
    "partido":     p.get("partido", ""),
    "match_id":    p.get("match_id", ""),
    "match_url":   p.get("match_url", ""),
    "torneo":      p.get("torneo", ""),
    # FALTA: superficie, tier
})
```

Luego `betslip_registrar.py:221-222` usa default `"?"`:
```python
"superficie":  info.get("superficie", "?"),
"tier":        info.get("tier", "?"),
```

**Fix:** Agregar campos faltantes en combo builder + cambiar default a "unknown" en registrar.

### BUG-32-5: ITF sin fallback_por_tier (MEDIO)

**Sintoma:** 44% de picks APOSTAR son ITF, pero no hay entrada en `fallback_por_tier`.

**Causa raiz:** `calibracion_edge.json:86-91` no incluye `"itf"`. El Thompson sampling cae a `por_superficie` o `global` — priors demasiado optimistas (0.629 global vs realidad ITF ~0.45).

**Fix:** Agregar `"itf": 0.50` a fallback_por_tier (conservador, basado en ?_? hit rate 23.5% y clay_itf 0W/1L).

---

## 4. FIXES — IMPLEMENTACION DETALLADA

### FIX-32-1: Gate p_modelo Minimo (edge_calculator.py)

**Archivo:** `edge_calculator.py`
**Linea:** 464

```python
# ANTES (linea 464):
apostar = edge > EDGE_MIN and kelly_kl_ajustado > KELLY_KL_MIN

# DESPUES:
P_MODELO_MIN_UNDERDOG = 0.55   # nueva constante, linea ~75
apostar = (
    edge > EDGE_MIN
    and kelly_kl_ajustado > KELLY_KL_MIN
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
)
```

**Logica:** Si es underdog (cuota >= 2.10), exigir p_modelo >= 0.55 (MODERATE o mejor). Favoritos y slight_underdogs pasan sin restriccion adicional porque su edge es inherentemente mas bajo y no produce phantom edge.

**Impacto estimado:** Elimina ~78% de picks LOW actuales. Los 7 picks MODERATE+STRONG mostraron 35% hit rate y ROI +0.1% — mucho mejor que el 26.7% global.

**Constante nueva:**
| Nombre | Valor | Linea | Justificacion |
|--------|-------|-------|---------------|
| P_MODELO_MIN_UNDERDOG | 0.55 | ~75 | Alineado con MODERATE threshold; evita coin-flip betting |

### FIX-32-2: Markov Post-Normalizacion (rivalry_analyzer.py)

**Archivo:** `analysis/rivalry_analyzer.py`
**Concepto:** Mover la aplicacion del factor Markov de PRE-log1p a POST-normalizacion.

```python
# ANTES (linea ~1582):
raw_p1['form_recent'] = min(raw_p1['form_recent'] * factor_p1, 300)
# ... luego log1p comprime todo

# DESPUES:
# 1. NO multiplicar form_recent por factor_markov antes de log1p
# 2. Despues de normalizacion (linea ~1800), aplicar:
if factor_markov != 1.0:
    normalized_p1['form_recent'] *= factor_markov  # rango [0.85, 1.15]
    # Ahora el delta es 0.85-1.15 sobre el valor normalizado
    # En lugar de 0.072, el delta es ~0.75 (15% de 5.0)
    # Con peso 0.22 → contribucion = 0.165 vs 0.016 antes (10x amplificacion)
```

**Impacto:** El factor Markov pasa de contribuir 0.016 puntos a ~0.165 puntos en el score. Esto mueve confidence ~1.5pp en lugar de ~0.35pp — todavia conservador pero ahora MEDIBLE.

**REGLA-T32-1:** Los tests existentes de Markov en `test_rivalry_analyzer.py` deben actualizarse para reflejar el nuevo punto de aplicacion. Los valores esperados cambiaran.

**REGLA-T32-2:** Factor tardio sigue la misma logica — mover a post-normalizacion junto con Markov.

### FIX-32-3: Golden Zone Redefinida (edge_calculator.py)

**Archivo:** `edge_calculator.py`
**Linea:** 784

```python
# ANTES:
golden_zone = (tier in ('challenger', 'itf') and cuota_fav >= 2.50 and n_h2h == 0)

# DESPUES:
golden_zone = (
    tier in ('challenger', 'itf')
    and cuota_fav >= 2.50
    and bbi >= 0.60                              # bookmaker realmente ciego
    and n_axes_active >= 2                        # modelo tiene al menos 2 señales
    and p_modelo >= P_MODELO_MIN_UNDERDOG         # modelo tiene conviccion minima
)
```

**Logica:** Golden Zone ahora requiere que:
1. El bookmaker este ciego (BBI >= 0.60)
2. El modelo tenga informacion (2+ axes activos)
3. El modelo tenga conviccion (p_modelo >= 0.55)

Esto cambia de "maximo desconocimiento mutuo" a "asimetria informacional a nuestro favor".

### FIX-32-4: Campos Faltantes en Betslip (betplay_combo_builder.py + betslip_registrar.py)

**Archivo 1:** `betplay_combo_builder.py:1399-1414`

```python
# Agregar a all_picks.append({...}):
    "superficie":  p.get("superficie", "unknown"),
    "tier":        p.get("tier", "unknown"),
```

**Archivo 2:** `betslip_registrar.py:364`

```python
# ANTES:
tier = pick.get("tier", "?")

# DESPUES:
tier = pick.get("tier", "unknown")
```

### FIX-32-5: ITF en fallback_por_tier (calibracion_edge.json)

```json
"fallback_por_tier": {
    "grand_slam": 0.7576,
    "atp1000": 0.7,
    "atp500": 0.65,
    "challenger": 0.611,
    "itf": 0.50
}
```

**Justificacion:** 0.50 es conservador. Con n=1 en clay_itf (0W/1L) y ?_? mostrando 23.5%, usar 0.50 como prior neutral hasta acumular datos era_v2.

---

## 5. ORDEN DE IMPLEMENTACION

| Fase | Fix | Riesgo | Dependencia | Tests Nuevos |
|------|-----|--------|-------------|-------------|
| **Fase 1** | FIX-32-4 (campos betslip) | Bajo | Ninguna | 4 |
| **Fase 1** | FIX-32-5 (ITF fallback) | Bajo | Ninguna | 2 |
| **Fase 2** | FIX-32-1 (gate p_modelo) | Medio | Ninguna | 8 |
| **Fase 2** | FIX-32-3 (golden zone) | Medio | FIX-32-1 (comparte constante) | 5 |
| **Fase 3** | FIX-32-2 (Markov post-norm) | Alto | Tests regression update | 10 |

---

## 6. ESPECIFICACION DE TESTS

### Nomenclatura: `tests/test_nodo32.py`

#### Fase 1 Tests (6 tests)

```
T32-01: test_betslip_index_includes_superficie_tier
    GIVEN edge_report pick con superficie="clay" y tier="challenger"
    WHEN betplay_combo_builder construye betslip_index
    THEN pick en betslip_index tiene superficie="clay" y tier="challenger"

T32-02: test_betslip_registrar_default_tier_not_question_mark
    GIVEN pick sin campo tier
    WHEN betslip_registrar procesa el pick
    THEN tier es "unknown", NO "?"

T32-03: test_betslip_registrar_default_superficie_not_question_mark
    GIVEN pick sin campo superficie
    WHEN betslip_registrar procesa el pick
    THEN superficie es "unknown", NO "?"

T32-04: test_calibracion_no_question_mark_keys
    GIVEN calibracion_edge.json actualizado post-fix
    WHEN se procesan picks con superficie y tier correctos
    THEN no hay keys "?" ni "?_?" en por_superficie ni por_superficie_y_tier

T32-05: test_fallback_por_tier_includes_itf
    GIVEN calibracion_edge.json
    WHEN theta_thompson busca prior para tier="itf"
    THEN encuentra fallback_por_tier["itf"] = 0.50

T32-06: test_theta_thompson_itf_uses_fallback
    GIVEN calibracion con itf n<10
    WHEN theta_thompson(superficie="clay", tier="itf")
    THEN usa fallback_por_tier["itf"] y no cae a global
```

#### Fase 2 Tests (13 tests)

```
T32-07: test_phantom_edge_blocked_low_confidence_high_cuota
    GIVEN p_modelo=0.503, cuota=3.60
    WHEN edge_calculator procesa
    THEN clasificacion != "APOSTAR" (phantom edge bloqueado)

T32-08: test_moderate_confidence_underdog_passes
    GIVEN p_modelo=0.57, cuota=2.80
    WHEN edge_calculator procesa
    THEN clasificacion == "APOSTAR" (edge real con conviccion)

T32-09: test_low_confidence_favorite_still_passes
    GIVEN p_modelo=0.52, cuota=1.80 (slight_underdog zona)
    WHEN edge_calculator procesa
    THEN gate no aplica restriccion extra (cuota < 2.10)

T32-10: test_strong_confidence_underdog_passes
    GIVEN p_modelo=0.65, cuota=3.00
    WHEN edge_calculator procesa
    THEN clasificacion == "APOSTAR"

T32-11: test_p_modelo_threshold_boundary_054
    GIVEN p_modelo=0.549, cuota=2.50
    WHEN edge_calculator procesa
    THEN clasificacion != "APOSTAR" (justo debajo del threshold)

T32-12: test_p_modelo_threshold_boundary_055
    GIVEN p_modelo=0.550, cuota=2.50
    WHEN edge_calculator procesa
    THEN clasificacion == "APOSTAR" (justo en el threshold)

T32-13: test_golden_zone_requires_bbi_060
    GIVEN tier="challenger", cuota=3.00, n_h2h=0, bbi=0.55
    WHEN edge_calculator evalua golden_zone
    THEN golden_zone == False (BBI < 0.60)

T32-14: test_golden_zone_requires_2_axes
    GIVEN tier="itf", cuota=3.00, bbi=0.70, n_axes_active=1
    WHEN edge_calculator evalua golden_zone
    THEN golden_zone == False

T32-15: test_golden_zone_requires_p_modelo_055
    GIVEN tier="challenger", cuota=2.80, bbi=0.65, n_axes=3, p_modelo=0.52
    WHEN edge_calculator evalua golden_zone
    THEN golden_zone == False

T32-16: test_golden_zone_all_conditions_met
    GIVEN tier="challenger", cuota=2.80, bbi=0.65, n_axes=2, p_modelo=0.58
    WHEN edge_calculator evalua golden_zone
    THEN golden_zone == True

T32-17: test_fix3_still_active_post_changes
    GIVEN n_axes_active=1, edge>5%, kelly>2%
    WHEN edge_calculator procesa
    THEN apostar == False (FIX-3 sigue activo)

T32-18: test_fix6_still_active_post_changes
    GIVEN markov="HOT", bbi=0.40, edge>5%, kelly>2%, p_modelo=0.58
    WHEN edge_calculator procesa
    THEN apostar == False (FIX-6 sigue activo)

T32-19: test_constant_P_MODELO_MIN_UNDERDOG_exists
    WHEN importar edge_calculator
    THEN P_MODELO_MIN_UNDERDOG == 0.55
```

#### Fase 3 Tests (13 tests) — Markov POST-normalizacion + Versionado

```
T32-21: test_t32_21_markov_applied_post_norm_in_real_code
    GIVEN P1=HOT (factor=1.075), P2=NEUTRAL, form_recent raw=200 aislado
    WHEN generate_advanced_prediction() calcula scores
    THEN norm_form = log1p(200) * 1.075 (POST-norm, no log1p(200*1.075))
    TEST: invoca pipeline real, inspeciona score_breakdown raw vs normalized

T32-22: test_t32_22_hot_vs_neutral_confidence_delta
    GIVEN dos jugadores identicos excepto P1=HOT, P2=NEUTRAL (weights form_recent=1.0 solo)
    WHEN generate_advanced_prediction() se ejecuta con ambos escenarios
    THEN confidence(HOT vs NEUTRAL) > 51.5% (delta >= 1.0pp vs NEUTRAL vs NEUTRAL)
    MEASURED: 54.70% (A) vs 51.00% (B) → delta=+3.70pp ✓ CUMPLE

T32-23: test_t32_23_cold_vs_hot_p2_favored
    GIVEN P1=COLD (factor_markov=0.85), P2=HOT (factor_markov=1.15)
    WHEN generate_advanced_prediction() calcula
    THEN P2 favorecido con confidence > 51.0%

T32-24: test_t32_24_neutral_vs_neutral_no_bias
    GIVEN dos jugadores identicos, ambos NEUTRAL
    WHEN generate_advanced_prediction() calcula
    THEN factor_markov = 1.0, no sesgo, confidence ~ 50%

T32-25: test_t32_25_markov_caps_at_log1p_300
    GIVEN factor_markov=1.15 aplicado post-norm con cap = log1p(300)
    WHEN norm_value * factor > log1p(300)
    THEN resultado capped correctamente a log1p(300)

T32-26: test_t32_26_immunity_dampener_post_markov
    GIVEN HOT player con h2h_win_rate < 0.30 (immunity=0.85)
    WHEN rivalry_analyzer aplica immunity POST-norm
    THEN factor efectivo = 1.075 * 0.85 = 0.914, reduce ventaja de HOT

T32-27: test_t32_27_post_norm_vs_pre_norm_amplification
    GIVEN form_recent=200, factor=1.075 (HOT vs NEUTRAL real)
    WHEN comparar pre-norm (log1p(215)) vs post-norm (log1p(200)*1.075)
    THEN post-norm delta=0.398, pre-norm delta=0.072, ratio=5.53x > 5.0 ✓
    NOTA: spec decía '>10x' pero eso era mezcla de 2 factores distintos (1.075 PRE, 1.15 POST).
    Con cualquier factor consistente el ratio es ~5.5-5.7x, que ES SUFICIENTE (confidence delta +3.70pp).

T32-28: test_t32_28_confidence_with_vs_without_markov
    GIVEN P1=HOT vs NEUTRAL (aislado form_recent), comparado vs NEUTRAL vs NEUTRAL
    WHEN generate_advanced_prediction() calcula confidence
    THEN conf_hot > 51.5%, conf_neutral ~ 50%, separacion medible en output

T32-29: test_t32_29_regression_baseline_documented
    DOCUMENTAL: Baseline esperado = 1244 tests passing (Nodo-32 Fases 1+2+3)
    VERIFICAR CON: pytest tests/ --no-cov -q   (externo, no desde pytest recursivo)
    No se invoca pytest dentro de pytest. Deixa constancia en código del baseline.

T32-32: test_t32_32_rivalry_version_validation
    GIVEN h2h_results_enhanced sin rivalry_version o con versión antigua
    WHEN edge_calculator.py carga el h2h
    THEN _validate_h2h_rivalry_version() rechaza con SystemExit NO SILENCIOSO
    Patrón análogo a T32-31 (GATE_VERSION): fuerza regeneración post-cambio.

T32-32b: test_t32_32b_h2h_enhanced_old_file_rejected
    GIVEN h2h_results_enhanced_20260622_081423.json (generado PRE-Fase3)
    WHEN edge_calculator.py intenta validar
    THEN rechaza con mensaje de 7 líneas, obliga regeneración

T32-33: test_t32_33_markov_infrastructure_intact_post_fase3
    GIVEN Fase 3 edits aplicados a rivalry_analyzer.py
    WHEN checks internos de factor range, caps, GATE_VERSION
    THEN calcular_factor_markov(HOT,COLD)=1.15, cap=log1p(300), GATE_VERSION="nodo32-fase2"
    Verifica que Markov infrastructure + Gate versioning intactos.
```

---

## 7. METRICAS DE EXITO POST-IMPLEMENTACION

### Validacion Inmediata (sin esperar partidos)

| Metrica | Antes | Esperado Post-Fix | Como medir |
|---------|-------|-------------------|------------|
| % picks LOW en APOSTAR | 86% | < 40% | Contar en edge_report |
| % picks MODERATE+ en APOSTAR | 14% | > 60% | Contar en edge_report |
| Golden Zone con n_axes < 2 | 100% | 0% | Verificar en edge_report |
| ?_? en calibracion | 51 | 0 (nuevos) | Inspeccionar JSON |
| ITF fallback presente | No | Si | Inspeccionar JSON |
| Tests passing | 1210 | >= 1239 | pytest |

### Validacion con Partidos (requiere n>=20 APOSTAR con resultado)

| Metrica | Antes | Target | Timeline |
|---------|-------|--------|----------|
| Hit% picks APOSTAR | 26.7% | > 45% | 2-3 semanas |
| Markov HOT hit% | 8.3% | > 40% | 2-3 semanas |
| ROI proxy | -21.1% | > 0% | 4 semanas |
| KGR | positivo (falso) | positivo (real) | 4 semanas |

---

## 8. REGLAS PERMANENTES DERIVADAS

**REGLA-T32-1:** Ningun factor multiplicativo debe aplicarse ANTES de log1p. Siempre post-normalizacion.

**REGLA-T32-2:** Todo pick APOSTAR con cuota >= 2.10 debe tener p_modelo >= 0.55. Sin excepciones.

**REGLA-T32-3:** Golden Zone requiere informacion asimetrica REAL (modelo sabe, bookmaker no), no ceguera mutua.

**REGLA-T32-4:** Todo campo usado en calibracion (superficie, tier) debe propagarse desde edge_report hasta betslip_registrar sin defaults silenciosos.

**REGLA-T32-5:** Cada tier en el pipeline debe tener un fallback_por_tier explicito en calibracion_edge.json.

---

## 9. CIERRE 2026-06-22 — Scope de Cada Fase e Incidentes de Remediación

### 9.1 Scope Original de Cada Fase

| Fase | Scope definido | Archivos modificados | Estado |
|------|----------------|---------------------|--------|
| **Fase 1** | Betslip fields propagados sin "?" default; ITF fallback prior = 0.50 | `betplay_combo_builder.py`, `betslip_registrar.py`, `calibracion_edge.json` | ✅ COMPLETO |
| **Fase 2** | Gate `P_MODELO_MIN_UNDERDOG` bloquea underdogs sin convicción; `golden_zone` redefinida (bbi≥0.60 + 2 ejes + p_modelo≥threshold) | `edge_calculator.py` | ✅ COMPLETO |
| **Fase 3** | Markov factor aplicado POST-normalizacion (`norm_form = log1p(raw) × factor`, no PRE); confidence delta medido y verificado | `analysis/rivalry_analyzer.py` | ✅ COMPLETO |

### 9.2 Incidentes de Remediación Descubiertos Durante el Audit

Descubiertos al ejecutar las fases, **no parte del scope original**. Documentados separadamente para trazabilidad.

---

**Incidente A — edge_report con gate viejo (post Fase 2)**

`reports/edge_report_20260622_082554.json` (generado 08:25 con gate viejo):
- Niels McDonald: `apostar=True golden_zone=True p_modelo=0.512` → phantom golden bonus
- Luciano Ambrogi: `apostar=True p_modelo=0.51` → pure phantom edge
- **Exposición:** Verificación exhaustiva confirmó que el archivo nunca fue consumido (no hay betslip_index, apuestas, ni combos posteriores a 08:25).

*Remediación:* `GATE_VERSION = "nodo32-fase2"` en `edge_calculator.py` serializado en metadata del edge_report; `_validate_edge_report_gate()` en 4 call sites de `betplay_combo_builder.py`; regeneración del edge_report a las 16:10 → todos los phantom picks a watchlist.

Archivo viejo preservado en `reports/edge_report_20260622_082554.json` como caso de prueba real.

---

**Incidente B — h2h_results_enhanced con Markov PRE-norm (post Fase 3)**

`reports/h2h_results_enhanced_20260622_081423.json` (generado 08:14, PRE-Fase 3):
- Contiene predicciones calculadas con Markov PRE-norm → confidence delta decorativo (~0.35pp)
- **Exposición:** No consumido por edge_calculator después de que Fase 3 fuera implementada (16:41).

*Remediación:* `RIVALRY_VERSION = "nodo32-fase3-markov-postnorm"` en `analysis/rivalry_analyzer.py` serializado en metadata del h2h; `_validate_h2h_rivalry_version()` en `edge_calculator.py`; rechaza archivos pre-Fase3 con SystemExit y mensaje que obliga a regenerar h2h.

---

**Incidente C — golden_zone podía ser True con apostar=False**

`golden_zone` se calculaba sobre `p_modelo >= threshold` sin verificar que el pick hubiera pasado el gate completo. Un pick bloqueado (apostar=False por algún otro criterio) podía recibir el bono golden (1.20×) en mega-combos.

*Remediación:* `edge_calculator.py:816` — añadido `resultado.get('apostar', False)` como primer requisito de `_golden_zone`.

---

### 9.3 Evidencia Cuantitativa — Delta de Confidence Markov POST-norm

Medido con `generate_advanced_prediction()` real, `optimized_weights={form_recent:1.0, otros:0.0}` para aislar el efecto Markov. Control = NEUTRAL vs NEUTRAL → confidence 51.00%.

| Escenario | factor_p1 | factor_p2 | confidence favored | Delta vs control |
|-----------|-----------|-----------|-------------------|-----------------|
| HOT vs NEUTRAL | 1.075 | 1.0 | 54.70% | **+3.70 pp** ✅ |
| COLD vs HOT | 0.85 | 1.15 | 58.40% | **+7.40 pp** ✅ |
| NEUTRAL vs NEUTRAL | 1.0 | 1.0 | 51.00% | 0.00 pp (control) |

**Meta: delta ≥ 1.0 pp** — CUMPLIDA en todos los escenarios medidos (factor mínimo 3.7×).

**Nota sobre ratio 5.53x vs spec "10x":** El ratio matemático POST/PRE con factor=1.075 consistente es 5.53x (no 10x). El spec mezclaba factor=1.075 para delta_pre (0.072) con factor=1.15 para delta_post (0.795), produciendo ratio artificial de ~11x. Con cualquier factor consistente el ratio es ~5.5-5.7x. Este ratio es irrelevante en la práctica: lo que importa es el delta de confidence en el output del pipeline, que es +3.70pp y +7.40pp respectivamente — ambos materiales.

### 9.4 Casos de Prueba Implementados

**34 tests Nodo-32 (T32-01 a T32-33) — 1244 passed, 0 failed (baseline 2026-06-22):**

| Rango | Scope | Cubre |
|-------|-------|-------|
| T32-01 a T32-06 | Fase 1 | betslip fields, ITF fallback |
| T32-07 a T32-19 | Fase 2 | gate p_modelo, golden_zone conditions |
| T32-20 | Incidente C | golden_zone requires apostar=True |
| T32-21 a T32-29 | Fase 3 | Markov POST-norm, orden de aplicación, confidence delta |
| T32-30 | Fase 2 | confidence_flag sync ↔ P_MODELO_MIN_UNDERDOG |
| T32-31 | Incidente A | Gate version validation (rechaza edge_reports viejos) |
| T32-32, T32-32b | Incidente B | Rivalry version validation (rechaza h2h pre-Fase3) |
| T32-33 | Fase 3 | Markov infrastructure intact post-migración |

### 9.5 Estado Final (Nodo-32 Fases 1+2+3 — CERRADO)

| Ítem | Estado | Evidencia |
|------|--------|-----------|
| Betslip fields propagados (superficie, tier) | ✅ | Fase 1 — sin "?" defaults |
| ITF fallback prior = 0.50 | ✅ | Fase 1 — calibracion_edge.json |
| Gate P_MODELO_MIN_UNDERDOG = 0.55 | ✅ | Fase 2 — 9 phantom picks → apostar=False |
| Golden_zone redefinida (apostar=True + bbi + ejes) | ✅ | Fase 2 + Incidente C |
| Markov aplicado POST-normalizacion | ✅ | Fase 3 — log1p(raw)×factor |
| Confidence delta HOT vs NEUTRAL | ✅ | +3.70 pp (meta ≥1.0 pp) |
| Confidence delta COLD vs HOT | ✅ | +7.40 pp (medido directamente) |
| GATE_VERSION en edge_report metadata | ✅ | Incidente A — "nodo32-fase2" |
| RIVALRY_VERSION en h2h metadata | ✅ | Incidente B — "nodo32-fase3-markov-postnorm" |
| Validación gate en 4 call sites | ✅ | Incidente A — betplay_combo_builder |
| Validación rivalry version en edge_calculator | ✅ | Incidente B — rechaza h2h pre-Fase3 |
| Tests passing | ✅ | 1244/1244 |

---

## 10. WIKILINKS

- [[Nodo-01-Edge-Calculator]] — formula Kelly-KL original
- [[Nodo-02-Markov-PELT]] — factores Markov originales
- [[Nodo-18-PELT-Recency-Alpha]] — lambda temporal
- [[Nodo-19-H2H-Immunity-Dampener]] — immunity factor
- [[Nodo-21-Pesos-Diferenciados-Tier]] — tier weights + shrinkage
- [[Nodo-24-Bookmaker-Blindness-BBI]] — BBI + golden_zone
- [[Nodo-27-Pipeline-Tracker]] — observabilidad que detecto el problema
- [[Nodo-28-Conditional-Decomposition]] — triple alignment
- [[MOC-Principal]] — indice de specs
- [[Sprint-Pipeline]] — estado del sprint

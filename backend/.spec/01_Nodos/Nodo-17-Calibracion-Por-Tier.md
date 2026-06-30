# Nodo-17 — Calibración Estratificada por Tier de Torneo

> **Estado:** 🟡 FASE 1 IMPLEMENTADA — 2026-06-03 | Tests: 898 passed
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Nodo-10-Surface-Propagation]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-16-Multi-Torneo-Pipeline]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-02-Markov-Changepoint]]
> **Tests:** 875 passed (baseline al momento del diagnóstico)
> **Origen:** Test-Time Compute — 4 marcos expertos aplicados post-sesión 2026-06-02

---

## Problema

El modelo fue calibrado exclusivamente en Roland Garros (Grand Slam, clay, top-50).
Al expandir a multi-torneo (Nodo-16), se descubrieron 3 fallas estructurales:

### Falla 1 — Prior contaminado post-sesión Challenger

```
Antes (solo GS):     wins=24, losses=7,  n=31  → p_clay = 0.758 ✅ limpio
Después (GS+Chall):  wins=46, losses=21, n=67  → p_clay = 0.687 ← contaminado

calibracion_edge.json mezcla poblaciones distintas:
  Grand Slam clay:    top-50, H2H denso, mercado eficiente, rankings estables
  Challenger clay:    top-150-300, H2H escaso, mercado ineficiente, rankings volátiles

Usar p=0.687 como prior para ambos = error de dominio
```

### Falla 2 — λ de KL divergencia fijo para todos los tiers

```
Kelly-KL: f*_KL = f* × exp(-λ × KL(P_modelo || P_histórica))

λ actual = 0.5 para todo (calibrado en Grand Slam)
λ en Challenger debería ser 1.5-1.8:
  → H2H escaso = P_modelo menos confiable
  → Odds menos eficientes = KL real es mayor
  → El sistema apuesta en Challengers con certeza de Grand Slam → pérdida estructural
```

### Falla 3 — Pesos del motor de predicción son constantes entre tiers

```
rivalry_analyzer.py — pesos calibrados para Roland Garros:
  common_opponents_weight = 0.28   ← OK para GS (H2H denso)
  ranking_momentum        = 0.12   ← OK para GS (rankings estables)
  
En Challenger:
  common_opponents_weight = 0.28   ← ALTO: pocos oponentes comunes → ruido amplificado
  ranking_momentum        = 0.12   ← BAJO: rankings más volátiles → señal más valiosa

normalization.py DEFAULT_WEIGHTS → uniforme para todos los tiers
```

---

## El Caso Polmans — Cadena Causal Completa

```
Polmans M. @5.00 → edge calculado: +30.3% → TOP SIGNAL del día → PERDIÓ 2-0

Cadena causal:
  1. superficie: unknown  ← bug propagación (45/45 partidos afectados, ver Nodo-10)
       ↓
  2. p_prior = fallback global (0.687, prior contaminado GS+Challenger)
       ↓
  3. Bayesian blend sobreestima Polmans asumiendo clay
       ↓
  4. PERO Birmingham (UK) = torneo de GRASS → Squire H. grass-native
       ↓
  5. λ=0.5 (Grand Slam) aplicado a Challenger → sin penalty de incertidumbre
       ↓
  6. common_opponents_weight=0.28 con H2H Challenger escaso → cadena Erdős ruidosa
       ↓
  7. Edge del +30.3% era espejismo: error de superficie + prior contaminado + λ bajo

Conclusión: mismo error de Nodo-03 (HTML garbage) pero en el dominio del prior.
El edge NO era alpha real — era error sistemático del modelo.
```

---

## Análisis Test-Time Compute — 4 Marcos Expertos

### Marco 1 — Analista Cuantitativo

**Diagnóstico de ruido por tier:**

| Variable | Grand Slam | ATP 500 | Challenger |
|---|---|---|---|
| Ranking variación/semana | <5 posiciones | 10-20 | 20-50 |
| Oponentes comunes (H2H) | 15-30 | 5-15 | 2-5 |
| Eficiencia de mercado (odds) | Alta | Media | Baja |
| Régimen Markov (duración) | Largo (meses) | Medio | Corto (semanas) |
| Señal surface | Constante (GS = 1 superficie) | Variable | Variable |
| p_historica confiable | ✅ n=31 | ⚠️ poca muestra | ⚠️ poca muestra |

**Calibración estratificada objetivo:**

```json
{
  "por_superficie_y_tier": {
    "clay_grand_slam":    {"wins": 24, "losses":  7, "p": 0.758},
    "clay_challenger":    {"wins": 22, "losses": 14, "p": 0.611},
    "grass_challenger":   {"wins":  0, "losses":  0, "p": null},
    "hard_atp500":        {"wins":  0, "losses":  0, "p": null},
    "clay_atp1000":       {"wins":  0, "losses":  0, "p": null}
  },
  "fallback_por_tier": {
    "grand_slam":  0.758,
    "atp1000":     0.700,
    "atp500":      0.650,
    "challenger":  0.611
  }
}
```

Regla de fallback: si `[superficie][tier]` tiene n<10 → usar `fallback_por_tier[tier]`.
Si `fallback_por_tier` tiene n<10 → usar 0.52 (prior neutro, no contaminar).

---

### Marco 2 — Trading / Kelly Financiero

**λ debe escalar con la incertidumbre del dominio:**

```
f*_KL = f* × exp(-λ × KL(P_modelo || P_histórica))

λ por tier:
  grand_slam   = 0.5   (modelo calibrado n=31, señal limpia)
  atp1000      = 0.8   (menos datos, más ruido)
  atp500       = 1.2   (intermedio)
  challenger   = 1.8   (mayor incertidumbre — penalty 3.6× mayor que GS)

Efecto en Polmans @5.00:
  f*_KL con λ=0.5  → stake alto (confianza de GS aplicada a Challenger) ← LO QUE PASÓ
  f*_KL con λ=1.8  → stake reducido ~60-70% (incertidumbre real Challenger)
```

**Regla financiera emergente:**

```
Mismo +30% edge en distintos tiers NO es el mismo asset:
  GS Challenger:   comprar agresivo (Parry @4.50 → +$35,000 ✅)
  Challenger:      comprar con descuento de incertidumbre (λ mayor)

Añadir al edge_calculator.py:
  λ_efectivo = λ_base × tier_multiplier[torneo_tipo]
```

**ρ de correlación (ya implementado en Nodo-15):**

```
grand_slam  ρ=0.25 | atp1000 ρ=0.20 | atp500 ρ=0.15 | challenger ρ=0.10
→ Continuar usando -- ya correctamente calibrado
→ λ es independiente de ρ: ambos se aplican
```

---

### Marco 3 — Arquitectura (Conexiones Ocultas en Código Existente)

**Lo que existe y puede explotarse sin código nuevo:**

```
analysis/elo_system.py
  → K-factor actualmente fijo para todos los torneos
  → Exploit: K_grand_slam=32, K_challenger=16
    Resultado: ELO de Challenger menos "informativo" → rivalry_analyzer
    lo recibe con señal atenuada automáticamente

analysis/markov_analyzer.py (PELT change-point)
  → window_size actual: calibrado para Grand Slam (regímenes largos)
  → Exploit: window_size_challenger = window_size_GS / 2
    Resultado: factor_tardio más reactivo en Challengers (régimen corto)

analysis/rivalry_analyzer.py
  → common_opponents_weight = 0.28 (clay GS)
  → Exploit: dict por tier
    common_opponents_weight = {
      "grand_slam": 0.28,
      "atp1000":    0.22,
      "atp500":     0.18,
      "challenger": 0.12   ← Erdős pesa menos donde la red es escasa
    }

normalization.py → DEFAULT_WEIGHTS
  → Actualmente uniforme para todos los torneos
  → Exploit: WEIGHTS_BY_TIER dict — misma estructura, pesos diferenciados

calibracion_edge.json
  → Actualmente: {"global": {...}, "por_superficie": {...}}
  → Exploit: añadir "por_superficie_y_tier" → mismo archivo, más granularidad
```

**El módulo que conecta todo:**

```
edge_calculator.py ← detecta torneo_tipo desde h2h_results_enhanced
  → ya tiene acceso a torneo_completo por partido
  → con torneo_completo → detectar tier (GS/1000/500/Challenger)
  → con tier → elegir λ correcto + p_prior estratificado
  → sin código nuevo: solo parametrizar lo que existe
```

---

### Marco 4 — Estratega / Hedge Fund (El Modelo Predictivo Emergente)

**La arquitectura no requiere modelo nuevo — requiere hiperparámetros conscientes del contexto:**

```
PIPELINE ACTUAL (ciego al tier):
  superficie (unknown) → p_prior_global → λ_fijo → Kelly → stake

PIPELINE OBJETIVO (tier-aware):
  torneo_url
    → detectar_tier()           # GS/1000/500/Challenger desde torneo_completo
    → detectar_superficie()     # clay/grass/hard (fix propagación)
    → p_prior[tier][superficie] # calibracion estratificada
    → λ[tier]                   # KL penalty por incertidumbre del dominio
    → K_factor[tier] (ELO)      # señal ELO atenuada en Challengers
    → common_opp_weight[tier]   # Erdős menos peso donde red es escasa
    → markov_window[tier]       # regímenes más cortos en Challengers
    → Kelly tier-aware          # stake correcto por dominio
```

**Por qué es diferencial:**

La mayoría de sistemas usan pesos fijos entrenados en su dataset principal.
El insight clave: **los hiperparámetros del modelo son variables de dominio, no constantes**.
Un Challenger no es un Grand Slam pequeño — es un mercado con estructura de ruido distinta.

---

## Dependencias y Prerequisitos

| Prerequisito | Estado | Nodo |
|---|---|---|
| Surface propagation en multi-torneo | 🔴 BUG ACTIVO — superficie: unknown | [[Nodo-10-Surface-Propagation]] — reabierto |
| `calibracion_edge.json` estratificado | 🔴 PENDIENTE — estructura nueva | Este nodo (T17-01) |
| λ por tier en edge_calculator.py | 🔴 PENDIENTE | Este nodo (T17-02) |
| K-factor por tier en elo_system.py | 🟡 BAJO IMPACTO — puede esperar | Este nodo (T17-03, fase 2) |
| common_opp_weight por tier | 🟡 MEDIO IMPACTO | Este nodo (T17-04, fase 2) |
| markov_window por tier | 🟡 BAJO IMPACTO — puede esperar | Este nodo (T17-05, fase 2) |

**Orden de implementación:**
```
FASE 1 (alto impacto, bajo riesgo):
  T17-01: Surface propagation fix (desbloquea todo lo demás)
  T17-02: calibracion_edge.json → estructura estratificada
  T17-03: edge_calculator.py → λ por tier + p_prior estratificado

FASE 2 (medio impacto, requiere más datos):
  T17-04: rivalry_analyzer.py → pesos por tier (necesita n≥10 por tier)
  T17-05: elo_system.py → K-factor por tier
  T17-06: markov_analyzer.py → window por tier
```

---

## Tasks

| ID | Descripción | Fase | Impacto P&L | Estado |
|---|---|---|---|---|
| T17-01 | Fix surface propagation: torneo_completo → superficie en H2H output | 1 | 🔴 CRÍTICO | ✅ 2026-06-03 — preservar tipo_cancha/torneo_nombre en _process_single_match |
| T17-02 | Estratificar calibracion_edge.json por [tier][superficie] | 1 | 🔴 CRÍTICO | ✅ 2026-06-03 — por_superficie_y_tier + fallback_por_tier añadidos |
| T17-03 | edge_calculator.py: λ_efectivo = λ_base × tier_multiplier[torneo_tipo] | 1 | 🔴 CRÍTICO | ✅ 2026-06-03 — detectar_tier() + LAMBDA_TIER_MULTIPLIER + theta_thompson(tier) |
| T17-04 | rivalry_analyzer.py: common_opp_weight dict por tier | 2 | 🟠 ALTO | 🟡 BLOQUEADO (n<10 por tier) |
| T17-05 | elo_system.py: K-factor por tier (32/24/20/16) | 2 | 🟡 MEDIO | 🟡 BLOQUEADO |
| T17-06 | markov_analyzer.py: window_size por tier | 2 | 🟡 MEDIO | 🟡 BLOQUEADO |

---

## Reglas Nuevas

**REGLA-T17-1: Prior estratificado obligatorio en multi-torneo**
```
Nunca mezclar calibración GS con Challenger en el mismo prior.
Si n_tier < 10 → usar fallback_por_tier, no fallback_global.
Si n_global < 10 → usar 0.52 (prior neutro).
```

**REGLA-T17-2: λ por tier**
```
grand_slam=0.5 | atp1000=0.8 | atp500=1.2 | challenger=1.8
Challenger con edge aparente = investigar superficie primero.
```

**REGLA-T17-3: Surface es el prerequisito de todo**
```
Sin superficie correcta → edge es espejismo.
Antes de calibrar por tier → resolver propagación de superficie (T17-01).
```

**REGLA-T17-4: Polmans Principle**
```
Un underdog @5.00 en Challenger con surface desconocida = NO APOSTAR.
Condición mínima: superficie confirmada + λ_challenger aplicado.
```

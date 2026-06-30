# Nodo-28 --- Conditional Decomposition Metamodel

> **Estado:** IMPLEMENTADO (Fase 1 + Fase 1.5 + Fase 2 + Post-Mortem 19-jun) --- 2026-06-19 | Tests: 1033→1113 passed
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-24-Bookmaker-Blindness-Scoring]] | [[Nodo-26-Cross-Sectional-Signals]] | [[Nodo-27-Pipeline-Tracker-Observabilidad]] | [[Nodo-15-Portfolio-HedgeFund]]
> **Origen:** Post-mortem Eala @5.20 vs Rybakina @1.16 (Berlín grass, 18-jun-2026) --- modelo acertó con confidence LOW y edge 33.7%. Análisis reveló que el alpha vino de descomposición condicional (superficie + régimen) vs agregación marginal del bookmaker.
> **Prioridad:** ALTA --- corrige contaminación de dominio en common_opponents (análogo a Error 2) + formaliza el metamodelo que unifica los 3 ejes de information asymmetry.

---

## Problema

### Fase 1 --- Common Opponents contaminados por superficie (BUG)

`find_common_opponents()` y `analyze_rivalry()` en `rivalry_analyzer.py` evalúan oponentes comunes **sin filtrar por la superficie del partido objetivo**. Esto produce:

```
Eala vs Rybakina en GRASS (Berlín ATP 500):
  Common opponents: 12 encontrados
  Rybakina score: 189.1 pts  (victorias sobre Andreeva, Fernández, Mertens — en clay/hard)
  Eala score:       8.5 pts
  
  Problema: el 90%+ de esos 189.1 puntos vienen de partidos en clay/hard.
  En GRASS, Rybakina tiene n=2 partidos totales (50% wr).
  El common_opponents score le regala 189 puntos de información IRRELEVANTE.
```

**Analogía directa con Error 2 (HTML garbage):** tipo_cancha contenía HTML → surface_specialization=0%. Aquí: common_opponents contiene todas las superficies → señal condicional contaminada con ruido marginal.

**Impacto medido:** Para Eala vs Rybakina, si common_opponents fuera condicional por superficie, el score_diff habría subido de 0.31 a ~0.45+, moviendo confidence de 52.9% (LOW) a ~57%+ (MODERATE). La señal estaba ahí pero el pipeline la diluyó.

### Fase 2 --- Triple Alignment no está formalizado

El modelo usa BBI, Markov y surface_specialization de forma independiente. No existe un detector de **alineación triple** que identifique cuándo las tres fuentes de information asymmetry convergen:

```
Para Eala @5.20:
  Surface Blindness:  87.5% grass vs 0% signal rival     ✓ (bookmaker no ve)
  Regime Blindness:   HOT vs NEUTRAL (momentum diverge)  ✓ (bookmaker no ve)
  BBI:                0.673 (bookmaker 67% ciego)         ✓ (confirmado)
  
  Triple alignment → edge 33.7% → ACERTÓ
  
  Pero confidence_flag = LOW (p=52.9%) → casi se descarta
```

El pick fue correcto **a pesar** de la flag LOW, no gracias a ella. Necesitamos un mecanismo que distinga LOW-con-alpha-estructural (Eala) de LOW-sin-alpha (coin flip).

---

## Principio Científico

### Paradoja de Simpson Aplicada a Betting

> Un jugador puede ser GLOBALMENTE inferior pero CONDICIONALMENTE superior.
> El bookmaker comete la Paradoja de Simpson cuando pone odds sin descomponer por condición.

```
BOOKMAKER (Marginal Aggregation):
  Inputs: ranking global, forma general, nombre/reputación
  Rybakina: #102, 75% form, ex-Wimbledon champ → P=86%
  
MODELO (Conditional Decomposition):
  Inputs: superficie-específico, régimen-específico, recencia-específica
  Rybakina ON GRASS: n=2, wr=50%, momentum=-0.3, NEUTRAL → P≈45%
  Eala ON GRASS:     n=8, wr=87.5%, momentum=+0.2, HOT    → P≈53%
```

El alpha surge cuando la distancia entre realidad condicional y percepción marginal es grande.

### Tres fuentes de Information Asymmetry

```
           ┌─────────────────────┐
           │   INFORMATION       │
           │   ASYMMETRY         │
           │   (BBI > 0.50)      │
           └──────┬──────────────┘
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│  TIER    │ │ SURFACE  │ │ REGIME   │
│ BLINDNESS│ │ BLINDNESS│ │ BLINDNESS│
│          │ │          │ │          │
│ CH/ITF   │ │ Grass    │ │ Markov   │
│ rank200+ │ │ specialist│ │HOT/COLD │
│ under-   │ │ hidden in│ │ change   │
│ covered  │ │ overall  │ │ not in   │
│          │ │ stats    │ │ rankings │
└──────────┘ └──────────┘ └──────────┘
  13-jun       18-jun        ambos
  9/10         Eala          amplifica
```

Cada eje fue validado independientemente:
- **Tier Blindness:** Sesión 13-jun, 9/10 picks Challenger+ITF ✅
- **Surface Blindness:** Eala 18-jun, grass specialist @5.20 ✅
- **Regime Blindness:** Markov HOT vs NEUTRAL (factor_markov amplifica ambos)

---

## Diseño

### Fase 1 --- Surface-Conditional Common Opponents

**Archivo:** `analysis/rivalry_analyzer.py`

**M-28-1: Filtro de superficie en evaluate_common_opponent()**

En `analyze_rivalry()` (línea ~916), al iterar common_opponents, el match más reciente (`p1_matches[0]`, `p2_matches[0]`) se selecciona sin considerar superficie. Cambio:

```python
# ANTES (línea 917-918):
p1_matches = [m for m in player1_history 
              if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]

# DESPUÉS:
target_surface = prediction_context.get('superficie', None)
p1_matches_all = [m for m in player1_history 
                  if self.ranking_manager.normalize_name(m.get('oponente', '')) == common_opponent]

# Preferir partidos en misma superficie; fallback a todos si no hay
if target_surface:
    p1_matches_surface = [m for m in p1_matches_all 
                          if normalize_surface(m.get('superficie', '')) == target_surface]
    p1_matches = p1_matches_surface if p1_matches_surface else p1_matches_all
else:
    p1_matches = p1_matches_all
```

**M-28-2: Surface relevance multiplier**

Cuando el partido del oponente común es en la misma superficie que el partido objetivo, el weight se multiplica. Cuando es en otra superficie, se reduce:

```python
# Después de calcular weight (línea 929):
opponent_match_surface = normalize_surface(p1_recent.get('superficie', ''))
if target_surface and opponent_match_surface:
    if opponent_match_surface == target_surface:
        surface_relevance = 1.30   # +30% si misma superficie
    else:
        surface_relevance = 0.60   # -40% si superficie diferente
else:
    surface_relevance = 1.00       # neutral si no hay datos de superficie

weight *= surface_relevance
```

**M-28-3: Propagación de `prediction_context` a `analyze_rivalry()`**

`prediction_context` ya existe y contiene `superficie`. Verificar que `analyze_rivalry()` lo recibe (actualmente no lo usa para common_opponents). Pasarlo al loop de common_opponents.

**Impacto esperado:**
- Rybakina common_opp en Eala match: 189.1 → ~90-110 (victorias clay/hard penalizadas ×0.60)
- Eala common_opp: 8.5 → ~10-12 (si Boulter fue en grass: ×1.30)
- score_diff: 0.31 → ~0.45+ 
- confidence: 52.9% → ~56-58% (LOW → MODERATE)

### Fase 1.5 --- SkillFactor + Surface Alpha + Volume Confidence (FIX surface_specialization)

**Archivo:** `analysis/rivalry_analyzer.py` — `calculate_surface_specialization()`

**Problema diagnosticado:** La fórmula original `(quality_score / n) × 2.5 × (1 + win_rate)` tenía 3 defectos:

1. **Compresión por `(1 + win_rate)`** — rango [1.0, 2.0], ratio máximo entre 85% y 60% WR = 1.15×. Casi inútil como discriminante.
2. **Threshold cliff** — `n < 3 → score = 0`. Jugadores elite con n=2 partidos en superficie (Zverev en grass) recibían 0 absoluto.
3. **División por n** — eliminaba la señal de volumen. 9 partidos al 89% se comprimía a rango similar a 3 partidos al 33%.

**Casos reales que fallaron:**
- **Zverev (#3, ganador RG) vs Hanfmann:** Zverev n=2 grass → score=0 | Hanfmann n=4 50% → score positivo → modelo predijo Hanfmann favorito (INCORRECTO)
- **Hardt vs Soto:** 85% vs 60.5% WR → gap de solo 0.08 en surface contribution (debería ser mucho mayor)
- **Eala vs Rybakina (post-rescrape):** confianza cayó 52.9%→50.5% cuando Rybakina adquirió n=3 grass al 33% WR

**M-28-7: SkillFactor — función convexa anclada en 50%**

```python
skill_factor = (max(win_rate, 0.01) / 0.5) ** 1.5
# 85% → 2.22x | 60% → 1.33x | 50% → 1.0x | 33% → 0.54x
# Ratio 85%/60% = 1.67x (vs 1.15x del antiguo)
```

Inspirado en Kelly: 50% = coin flip = neutral (factor 1.0). Exponente 1.5 crea convexidad que amplifica verdaderos especialistas.

**M-28-8: Surface Alpha — alpha_vs_elo aplicado a superficies**

```python
surface_alpha = win_rate_surface - overall_win_rate
alpha_bonus = 1.0 + max(surface_alpha, 0) * 2.0
# +10% alpha → 1.20x boost | +20% alpha → 1.40x boost | 0% → neutral
```

Analogía directa con `alpha_vs_elo` de Nodo-24 (BBI). Un jugador con 87% grass pero 60% overall tiene alpha_surface = +27% → bonus 1.54×. Captura la Paradoja de Simpson: globalmente inferior, condicionalmente superior.

**M-28-9: Volume Confidence — James-Stein shrinkage por n**

```python
volume_confidence = min(n_surface_matches / 8.0, 1.0)
# n=2 → 0.25 | n=3 → 0.38 | n=5 → 0.63 | n=8+ → 1.0
```

Inspirado en `calibration_confidence = n/(n+20)` de Nodo-21. Threshold reducido de n≥3 a n≥2 — VolConf maneja la incertidumbre gradualmente en lugar de cliff binario.

**M-28-10: Fórmula combinada**

```python
final_score = (quality_score / n × 2.5) × SkillFactor × AlphaBonus × VolConf
```

**Resultados post-fix:**
- Eala: 52.9% → 52.8% (estable, ya no cae con re-scrape)
- Zverev vs Hanfmann: Zverev predicts favored (correcto)
- Hardt vs Soto: gap surface amplificado (correcto)
- Arseneault vs Langmo: ~50% coin flip (correcto — ambos desconocidos)

### Fase 2 --- Triple Alignment Score

**Archivo:** `edge_calculator.py`

**M-28-4: `triple_alignment_score()` en edge_calculator**

```python
def triple_alignment_score(pick: dict) -> dict:
    """
    Detecta alineación de las 3 fuentes de information asymmetry.
    Retorna score + flag + override de confidence.
    """
    # Eje 1: Surface Blindness
    # ¿El modelo ve algo en superficie que el ELO no ve?
    surface_signal = abs(pick['alpha_vs_elo'])  # >0.15 = fuerte
    
    # Eje 2: Regime Blindness  
    # ¿El Markov detecta momentum divergente?
    regime_signal = 0.0
    if pick.get('markov_favorito') == 'HOT':
        regime_signal += 0.5
    if pick.get('delta_wr_markov', 0) > 0.15:
        regime_signal += 0.5
    # regime_signal ∈ [0.0, 1.0]
    
    # Eje 3: Bookmaker Blindness
    bbi = pick.get('bbi', 0.5)
    
    # Triple alignment: producto de las 3 señales normalizadas
    surface_norm = min(surface_signal / 0.25, 1.0)   # 0.25 = techo calibrado
    regime_norm = regime_signal                        # ya en [0, 1]
    bbi_norm = min(bbi / 0.70, 1.0)                    # 0.70 = techo calibrado
    
    alignment = surface_norm * regime_norm * bbi_norm
    
    # Clasificación
    n_axes_active = sum([
        surface_norm > 0.50,
        regime_norm > 0.50,
        bbi_norm > 0.50
    ])
    
    if n_axes_active == 3 and alignment > 0.40:
        flag = 'STRUCTURAL_ALPHA'
    elif n_axes_active >= 2 and alignment > 0.20:
        flag = 'PARTIAL_ALIGNMENT'
    else:
        flag = 'NO_ALIGNMENT'
    
    return {
        'triple_alignment': round(alignment, 4),
        'alignment_flag': flag,
        'n_axes_active': n_axes_active,
        'surface_signal': round(surface_norm, 3),
        'regime_signal': round(regime_norm, 3),
        'bbi_signal': round(bbi_norm, 3),
    }
```

**Retroactivo Eala:**
- surface_norm = min(0.224/0.25, 1.0) = 0.896
- regime_norm = 0.5 (HOT) + 0.5 (delta_wr=0.20 > 0.15) = 1.0
- bbi_norm = min(0.673/0.70, 1.0) = 0.961
- alignment = 0.896 × 1.0 × 0.961 = **0.861** → **STRUCTURAL_ALPHA**

**M-28-5: Campos en edge_report**

Cada pick en `apostar`, `watchlist`, `sin_edge` incluirá:
- `triple_alignment`: float [0, 1]
- `alignment_flag`: STRUCTURAL_ALPHA | PARTIAL_ALIGNMENT | NO_ALIGNMENT
- `n_axes_active`: int [0, 3]

**M-28-6: Confidence flag override (opcional, validar primero)**

```python
# En calcular_edge(), después de calcular confidence_flag:
if alignment_flag == 'STRUCTURAL_ALPHA' and confidence_flag == 'LOW':
    confidence_flag = 'LOW_STRUCTURAL'  # LOW por p_modelo, pero con alpha confirmado
```

`LOW_STRUCTURAL` señala al trader y al humano: "la confidence es baja pero las 3 fuentes de asymmetry están alineadas — no descartar automáticamente."

**NO modifica Kelly ni sizing** — eso requiere validación empírica (V-28-5).

---

## Dependencias

| ID | Dependencia | Estado |
|---|---|---|
| D-28-1 | `normalize_surface()` accesible en rivalry_analyzer | ✅ ya importado desde data_parser |
| D-28-2 | `prediction_context` contiene `superficie` | ✅ verificar propagación |
| D-28-3 | `alpha_vs_elo` disponible en edge_report | ✅ desde Nodo-24 |
| D-28-4 | `bbi` disponible en edge_report | ✅ desde Nodo-24 |
| D-28-5 | `markov_favorito` disponible en edge_report | ✅ desde Nodo-18 |

---

## Tests

### Fase 1 Tests

| Test | Descripción | Criterio |
|---|---|---|
| T28-01 | Common opponent en misma superficie recibe ×1.30 | weight_surface > weight_base |
| T28-02 | Common opponent en otra superficie recibe ×0.60 | weight_cross < weight_base |
| T28-03 | Sin datos de superficie → factor neutral (1.00) | weight_nodata == weight_base |
| T28-04 | Preferencia de match en misma superficie sobre otro | p1_matches[0].superficie == target |
| T28-05 | Fallback a todos los matches si no hay en misma superficie | len(p1_matches) > 0 siempre |
| T28-06 | Retroactivo: Rybakina common_opp score baja con filtro grass | score_new < 189.1 |
| T28-07 | Retroactivo: score_diff Eala sube con filtro grass | diff_new > 0.31 |
| T28-08 | prediction_context.superficie propagado a common_opp loop | superficie leída correctamente |

### Fase 1.5 Tests

| Test | Descripción | Criterio |
|---|---|---|
| T28-18 | SkillFactor(85%) > SkillFactor(60%) por ratio ≥1.5 | (0.85/0.5)^1.5 / (0.60/0.5)^1.5 ≥ 1.5 |
| T28-19 | SkillFactor(50%) = 1.0 (neutral) | (0.50/0.5)^1.5 == 1.0 |
| T28-20 | SkillFactor(33%) < 1.0 (penaliza) | (0.33/0.5)^1.5 < 0.60 |
| T28-21 | Surface alpha positivo → bonus > 1.0 | alpha_bonus > 1.0 |
| T28-22 | Surface alpha negativo → bonus = 1.0 (no penaliza) | max(alpha, 0) = 0 → bonus = 1.0 |
| T28-23 | Volume confidence n=2 → 0.25 | min(2/8, 1.0) == 0.25 |
| T28-24 | Volume confidence n=8+ → 1.0 (cap) | min(8/8, 1.0) == 1.0 |
| T28-25 | Threshold reducido: n=2 produce score > 0 | score != 0 con n=2 |
| T28-26 | n=1 sigue produciendo score = 0 | threshold mínimo respetado |

### Fase 2 Tests

| Test | Descripción | Criterio |
|---|---|---|
| T28-09 | Triple alignment Eala retroactivo = STRUCTURAL_ALPHA | alignment > 0.40, flag == STRUCTURAL_ALPHA |
| T28-10 | n_axes_active = 3 para Eala (surface + regime + bbi) | n_axes == 3 |
| T28-11 | Pick sin Markov HOT → n_axes < 3 | n_axes <= 2 |
| T28-12 | Pick con BBI < 0.30 → bbi_signal < 0.50 | bbi_norm < 0.50 |
| T28-13 | Pick con alpha_vs_elo < 0.05 → surface_signal < 0.50 | surface_norm < 0.20 |
| T28-14 | alignment_flag = NO_ALIGNMENT cuando 0 ejes activos | flag == NO_ALIGNMENT |
| T28-15 | Campos triple_alignment, alignment_flag en edge_report | campos presentes |
| T28-16 | LOW_STRUCTURAL solo cuando LOW + STRUCTURAL_ALPHA | flag correcto |
| T28-17 | STRUCTURAL_ALPHA no modifica kelly_kl (solo informativo) | kelly_before == kelly_after |

### Validación Empírica

| Test | Descripción | Criterio |
|---|---|---|
| V-28-1 | Retroactivo: picks con STRUCTURAL_ALPHA en edge_reports históricos | Identificar cuántos hubo y su hit rate |
| V-28-2 | STRUCTURAL_ALPHA win rate > LOW win rate | diferencia significativa (chi-squared p<0.10) |
| V-28-3 | PARTIAL_ALIGNMENT win rate > NO_ALIGNMENT | tendencia positiva |
| V-28-4 | Common_opp filtrado mejora accuracy global | accuracy_new >= accuracy_old |
| V-28-5 | Si V-28-2 confirmado con n≥20: habilitar Kelly override para STRUCTURAL_ALPHA | Sizing ×1.25 para STRUCTURAL_ALPHA |

---

## Caso de Estudio: Eala @5.20 vs Rybakina @1.16

### Señales que convergieron

```
                        BOOKMAKER VE          MODELO VE
Ranking:                102 vs 153            (igual)
ELO:                    2018 vs 1875          (igual)  
Forma general:          75% vs 60%            60% pero HOT vs 75% pero NEUTRAL
Grass specific:         (no descompone)       87.5% vs 50% (n=2, below threshold → 0.0)
Common opponents:       (no descompone)       189.1 vs 8.5 (contaminado ALL surfaces)
                                              → con fix Fase 1: ~100 vs ~11

Resultado bookmaker:    Rybakina 86%          
Resultado modelo:       Eala 52.9%            → con fix Fase 1: ~57%
Edge:                   —                     33.7% → con fix: ~38%
Resultado real:         EALA GANÓ ✅
```

### Triple Alignment retroactivo

```
Surface Blindness:  alpha_vs_elo = 0.224 → norm = 0.896  ✓
Regime Blindness:   HOT + delta_wr=0.20  → norm = 1.000  ✓  
Bookmaker Blindness: BBI = 0.673         → norm = 0.961  ✓

Triple Alignment = 0.861 → STRUCTURAL_ALPHA (3/3 ejes)
```

### Lección para el metamodelo

El bookmaker perdió porque:
1. Puso odds basadas en **marginal** (rankings, nombre, forma general)
2. No condicionó en **superficie** (Eala 87.5% grass oculto en 60% overall)
3. No detectó **régimen** (Rybakina declinando en grass, Eala subiendo)
4. La cuota extrema (5.20) **amplificó** el edge de 33.7% a retorno masivo

El modelo ganó porque descompuso condicionalmente. El fix de Fase 1 habría amplificado la señal al eliminar la contaminación cross-surface en common_opponents.

### Caso de Estudio 2: Zverev vs Hanfmann (Halle grass, 18-jun-2026)

**Bug pre-Fase 1.5:** Zverev (#3 mundo, ganador Roland Garros) tenía n=2 partidos en grass → `n < 3` → score=0. Hanfmann con n=4 al 50% → score positivo. Modelo predijo Hanfmann como favorito.

**Fix:** Threshold reducido a n≥2 + VolConf(2)=0.25 penaliza pero no anula. Zverev recupera score positivo, modelo corrige predicción.

**Lección:** El threshold binario es incompatible con jugadores elite que rotan superficies. VolConf maneja la incertidumbre sin cliff.

### Caso de Estudio 3: Hardt vs Soto (Challenger clay, 18-jun-2026)

**Bug pre-Fase 1.5:** Hardt 85% WR clay vs Soto 60.5% WR clay. Surface contribution gap = 0.08 (casi invisible). El `(1 + win_rate)` daba 1.85 vs 1.605 = ratio 1.15×.

**Fix:** SkillFactor `(wr/0.5)^1.5`: Hardt → 2.22× vs Soto → 1.33× = ratio 1.67×. Gap amplificado 4.5× respecto al antiguo.

**Lección:** La función convexa `(wr/0.5)^1.5` es el discriminante correcto — amplifica diferencias reales sin inventar señal.

### Caso de Estudio 4: Arseneault vs Langmo (ITF, 18-jun-2026)

**Resultado:** ~50% confidence (coin flip). Correcto — ambos jugadores sin historial significativo en la superficie.

**Lección:** VolConf baja + sin alpha surface → el modelo correctamente dice "no sé". No fuerza una predicción.

---

## Análisis Post-Mortem: 10 Fallos 18-jun-2026 (80.8% accuracy, 42/52)

### Hallazgo principal: Markov = "?" en 10/10 fallos

El estado Markov no se resolvió para **ninguno** de los 10 partidos fallidos. Todos muestran `estado_j1: ?`, `estado_j2: ?`, `factor_markov: 1.0` (neutro). Sin Markov, el modelo pierde su señal más potente — HOT Markov tiene 84.6% hit rate (55W/10L).

```
MARKOV RESUELTO vs NO RESUELTO — 18-jun-2026:
  Markov resuelto (HOT/COLD):  hit rate = 84.6% (datos globales, S-27-4)
  Markov no resuelto (?):      hit rate = ~65% (contiene los 10 fallos)
  
  Conclusión: cuando Markov no puede resolver el estado, 
  la predicción se degrada a coin flip con poca señal.
```

### Perfil de los 10 fallos

```
#   Partido                     Tier         Sup    Conf%  Cuotas       CO   Markov  Patron
─────────────────────────────────────────────────────────────────────────────────────────────
1   Rincon vs Gadamauri         challenger   clay   50.5   2.02/1.75    4    ?/1.075 COIN FLIP
2   O'Connell vs Bonzi          challenger   grass  51.7   2.70/1.44    7    ?/1.0   COIN FLIP
3   Ruggeri vs Sherif            challenger   clay   50.8   3.50/1.29    3    ?/1.0   COIN FLIP
4   Dussault vs Baris            itf          hard   55.0   5.10/1.13    0    ?/0.925 UPSET @5.10
5   Arseneault vs Langmo         itf          hard   51.0   2.43/1.49    2    ?/0.925 COIN FLIP
6   Fenty vs Aguiard             itf          hard   51.0   1.70/2.02    2    ?/1.075 COIN FLIP
7   Gogineni vs Nakashima        itf          hard   52.3   1.72/2.00    1    ?/1.0   COIN FLIP
8   Marton vs Rogozinska         itf          hard   56.8   2.70/1.40    1    ?/1.0   UPSET @2.70
9   Combs vs Rabman              itf          hard   50.5   1.87/1.83    0    ?/1.0   COIN FLIP
10  Wang vs Shi                  itf          hard   50.6   3.30/1.28    2    ?/1.0   COIN FLIP
```

### 6 patrones observados

1. **Markov "?" en 10/10 (100%)** — sin señal de régimen, la confianza colapsa a ~50%.
2. **Confidence ≤ 52.3% en 8/10** — el modelo sabe que no sabe. Solo Baris (55%) y Rogozinska (56.8%) superan.
3. **Tier bajo: 0 ATP500+ | 3 Challenger | 7 ITF** — donde menos datos hay, más falla.
4. **Hard court: 7/10** — superficie con mayor varianza en ITF.
5. **Common opponents escasos: 5/10 tienen 0-1** — grafo Erdős sin caminos transitivos.
6. **0/10 en edge report** — ninguno habría sido apuesta real. Pipeline los descartó correctamente.

### Casos notables

- **Fallo 1 (Rincon vs Gadamauri):** 3 de 4 common opponents favorecen a Gadamauri, pero sus victorias fueron en **Dura** vs partido en **Arcilla**. El Nodo-28 Fase 1 penaliza esto con ×0.60, pero sin Markov la señal no alcanza.
- **Fallo 4 (Dussault @5.10):** ELO gap 180pts, forma 38.5% vs 70.0%, 0 common opp. Noise puro en ITF — varianza irreducible.
- **Fallo 8 (Marton):** 20.8% WR hard (5W/19L) venció a 54.1% WR. ITF femenino = varianza máxima.
- **Fallo 10 (Wang vs Shi):** Gemelos estadísticos — #52/#52, ELO 1677/1674, forma 50%/50%, hard 64%/62.5%. Coin flip honesto.

### Implicación para el sistema

**El modelo no necesita "mejorar" en estos 10 fallos.** Necesita reconocer que sin Markov resuelto + pocos common opponents + tier bajo = **señal insuficiente para predecir**. El confidence flag LOW (50-52%) ya lo indica, pero podría reforzarse con un `data_completeness_flag` que alerte cuando Markov está ausente.

Posible regla futura: `Si Markov=? AND n_common_opponents ≤ 2 AND tier ∈ {itf, challenger} → confidence_flag = INSUFFICIENT (no predecir, no apostar).`

---

## Hallazgos Post-Mortem 19-jun-2026 (Sprint-PostMortem-19jun)

Cruce edge_report vs resultados 18-jun reveló 2 reglas adicionales para Fase 2:

### REGLA-N28F2-1: n_axes_active < 2 → watchlist

Picks con solo 1 eje activo (BBI sola, sin surface ni regime) tuvieron 29% hit rate (2/7) en APOSTAR el 18-jun. La señal BBI sola no predice.

```python
# Implementado en edge_calculator.py
if n_axes_active < 2 and resultado.get('apostar'):
    resultado['apostar'] = False
    resultado['motivo_reclasificacion'] = 'N28F2: n_axes_active < 2 (BBI sola no predice)'
```

Solo mueve APOSTAR → watchlist. No afecta picks ya en watchlist o sin_edge.

### REGLA-N28F2-2: CONTESTED_ALPHA cuando rival también HOT

Sziedat (STRUCTURAL_ALPHA, triple=0.49) perdió contra Stoyanov que también estaba HOT. STRUCTURAL_ALPHA ahora requiere `net_alignment > 0.25` — ventaja informacional **relativa**, no absoluta.

```python
# net_alignment = alignment_fav - alignment_dog
# alignment_dog calcula surface_norm × regime_norm_dog × bbi_norm del rival
if flag == 'STRUCTURAL_ALPHA' and net_alignment < 0.25:
    flag = 'CONTESTED_ALPHA'
```

`CONTESTED_ALPHA` se trata como `PARTIAL_ALIGNMENT` en clasificación (no suprime picks con n_axes≥2). Campo `net_alignment` presente en output de `triple_alignment_score()`.

### Nuevos campos en edge_report (por pick)

| Campo | Tipo | Descripción |
|---|---|---|
| `data_insufficient_surface` | bool | `min(vol_conf_fav, vol_conf_dog) < 0.25` — sin datos de superficie |
| `net_alignment` | float | `alignment_fav - alignment_dog` — ventaja informacional relativa |
| `motivo_reclasificacion` | str | Auditoría cuando APOSTAR → watchlist por REGLA-N28F2-1 |

---

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Over-filtering: pocos common_opponents en superficie específica | Fallback a todos cuando n_surface < 2 |
| Surface data ausente en historial antiguo | Factor neutral (1.00) cuando superficie desconocida |
| STRUCTURAL_ALPHA como falso positivo | Solo informativo hasta V-28-2 con n≥20 |
| Complejidad adicional en common_opponents | Un multiplicador, no un rediseño |
| Markov no resuelto → predicción degradada a coin flip | Detectar `estado=?` y escalar confidence_flag (hallazgo 18-jun-2026) |

---

## Orden de implementación

1. **Fase 1** (M-28-1, M-28-2, M-28-3) — fix common_opponents — ✅ IMPLEMENTADO
2. Tests Fase 1 (T28-01 a T28-08) — ✅ 1050 passed
3. **Fase 1.5** (M-28-7, M-28-8, M-28-9, M-28-10) — SkillFactor + AlphaBonus + VolConf — ✅ IMPLEMENTADO
4. Tests Fase 1.5 (T28-18 a T28-26) — ✅ 1050 passed
5. **Fase 2** (M-28-4, M-28-5) — triple alignment score — ✅ IMPLEMENTADO 2026-06-19
6. Tests Fase 2 (T28-09 a T28-17) + TF1-TF4 (post-mortem) — ✅ 1113 passed
7. M-28-6 (confidence override) — solo después de V-28-2 validado
8. Validación empírica (V-28-1 a V-28-5) — ongoing con cada sesión

# Nodo-21 — Pesos Diferenciados por Tier (Unificacion de Capas)

> **Estado:** ✅ COMPLETADO — Fases 1+2+3 implementadas — 2026-06-03
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-20-PageRank-Erdos-Quality]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]]
> **Tests:** 967 passed (Fase 3 añade 20 tests T21-11)
> **Origen:** Test-Time Compute — 4 marcos expertos, 5 conexiones ocultas, 2026-06-03
> **Prioridad:** ANTES de Nodo-19/18/20 — este nodo es la infraestructura que los otros tres necesitan

---

## Problema — Tres Capas de Tier Desconectadas

El sistema tiene 3 capas que clasifican torneos por tier, pero operan de forma independiente y con granularidades distintas:

```
CAPA 1 — rivalry_analyzer.py classify_tournament() → pesos del modelo
  Categorias: 'atp_wta' | 'challenger' | 'itf' | 'default'
  BUG: Grand Slam y ATP 500 son AMBOS 'atp_wta' → pesos identicos
  BUG: "French Open (France)" → NO matchea 'atp' ni 'grand slam' → cae a 'default'

CAPA 2 — edge_calculator.py detectar_tier() → lambda de Kelly-KL
  Categorias: 'grand_slam' | 'atp1000' | 'atp500' | 'challenger'
  CORRECTO: tiene keywords especificas por Grand Slam

CAPA 3 — trader_ev_tenis.py RHO_BY_TOURNAMENT → correlacion portfolio
  Categorias: 'grand_slam' | 'atp1000' | 'atp500' | 'challenger'
  CORRECTO: granularidad alineada con Capa 2

RESULTADO: el modelo PREDICE con 4 categorias burdas (Capa 1),
           el edge PENALIZA con 4 finas (Capa 2),
           y el portfolio REDUCE con 4 finas (Capa 3).
           Las capas no comparten logica de clasificacion.
```

---

## Bug Critico C5 — classify_tournament() no detecta Grand Slams

```python
# rivalry_analyzer.py line 1010:
if any(keyword in name_lower for keyword in ['atp', 'wta', 'grand slam', 'masters', 'olympic']):
    return 'atp_wta'

# FlashScore retorna:
#   "French Open (France)"         → NO contiene 'atp' ni 'grand slam' → cae a 'default'
#   "Roland Garros (France)"       → NO contiene 'atp' ni 'grand slam' → cae a 'default'
#   "Wimbledon (UK)"               → NO contiene ninguna keyword       → cae a 'default'
#   "Australian Open (Australia)"  → NO contiene ninguna keyword       → cae a 'default'

# PERO edge_calculator.py detectar_tier() SI funciona:
if any(gs in t for gs in ('roland garros', 'french open', 'wimbledon', 'australian open', 'us open')):
    return 'grand_slam'  # ← keywords especificas

# CONSECUENCIA: rivalry_analyzer usa pesos 'default' (= 'atp_wta')
# para TODOS los Grand Slams. Nunca ha usado pesos calibrados por GS.
# Sin embargo, como 'default' == 'atp_wta', el impacto P&L hasta ahora ha sido neutro.
# El problema se vuelve critico cuando queremos pesos diferenciados por tier.
```

---

## Analisis Test-Time Compute — 4 Marcos Expertos

### Marco 1 — Quant de Renta Fija: Estructura de Mercado como Variable

En renta fija, el spread de un bono depende del **mercado donde se negocia**, no solo del emisor. Un bono AAA en OTC tiene spread mayor que el mismo bono en mercado regulado.

**Analogia directa al tenis:**

| Dimension | Grand Slam | ATP 1000 | ATP 500 | Challenger | ITF |
|---|---|---|---|---|---|
| H2H directo disponible | ~70% parejas | ~50% | ~30% | ~5% | <1% |
| Red Erdos (oponentes comunes) | Densa (15-30) | Media (8-15) | Escasa (3-8) | Fragmentada (1-3) | Inexistente |
| Ranking volatilidad/semana | <5 pos | 5-10 | 10-20 | 30-50 | 50-100 |
| Mercado cuotas eficiente | Muy alto | Alto | Medio | Bajo | Muy bajo |
| Regimen Markov (duracion) | Meses | Semanas-Meses | Semanas | Dias-Semanas | Dias |

**Principio:** El peso de cada senal debe ser proporcional a su **signal-to-noise ratio** en ese mercado.

#### Pesos propuestos por tier

```python
WEIGHTS_BY_TIER = {
    'grand_slam': {
        'surface_specialization': 0.15,  # alta: torneo 1 superficie, jugadores conocidos
        'form_recent':            0.12,  # baja: regimenes largos, forma ya priced por bookmaker
        'common_opponents':       0.22,  # alta: red Erdos densa, senal confiable
        'h2h_direct':             0.18,  # alta: H2H disponible en ~70% parejas
        'ranking_momentum':       0.15,  # media: rankings estables, poco alpha
        'elo_rating':             0.13,  # media-alta: ELO calibrado con datos densos
        'home_advantage':         0.05,
        'strength_of_schedule':   0.00,
    },
    'atp1000': {
        'surface_specialization': 0.16,
        'form_recent':            0.15,
        'common_opponents':       0.20,
        'h2h_direct':             0.14,
        'ranking_momentum':       0.17,
        'elo_rating':             0.13,
        'home_advantage':         0.05,
        'strength_of_schedule':   0.00,
    },
    'atp500': {
        'surface_specialization': 0.15,
        'form_recent':            0.18,  # sube: regimenes mas cortos
        'common_opponents':       0.15,  # baja: red Erdos escasa
        'h2h_direct':             0.10,  # baja: H2H en ~30% parejas
        'ranking_momentum':       0.20,  # sube: rankings mas volatiles
        'elo_rating':             0.12,
        'home_advantage':         0.05,
        'strength_of_schedule':   0.05,  # aparece: schedule importa con campo heterogeneo
    },
    'challenger': {
        'surface_specialization': 0.20,  # sube: nichos de superficie marcados
        'form_recent':            0.22,  # sube mucho: regimenes de dias, senal principal
        'common_opponents':       0.08,  # baja mucho: red fragmentada, ruido > senal
        'h2h_direct':             0.03,  # casi nulo: <5% parejas tienen H2H
        'ranking_momentum':       0.22,  # sube: rankings volatiles = senal fresca
        'elo_rating':             0.15,  # sube: ELO da contexto donde H2H falta
        'home_advantage':         0.05,
        'strength_of_schedule':   0.05,
    },
    'itf': {
        'surface_specialization': 0.15,
        'form_recent':            0.28,  # dominante: unica senal confiable
        'common_opponents':       0.05,  # casi nulo: red inexistente
        'h2h_direct':             0.02,  # nulo: primer enfrentamiento >99%
        'ranking_momentum':       0.22,
        'elo_rating':             0.15,
        'home_advantage':         0.08,  # sube: en ITF, home advantage es real
        'strength_of_schedule':   0.05,
    },
}
```

---

### Marco 2 — Ecologista de Redes: Densidad Local como Meta-Variable

En ecologia de redes, el comportamiento de un nodo depende de la **densidad local del grafo** (clustering coefficient).

**Dato que el sistema ya tiene pero no usa:**

```python
# erdos_graph.py — distancia_erdos() ya retorna:
'n_paths': n_paths,           # caminos encontrados entre A y B

# rivalry_analyzer.py — analyze_common_opponents() ya conoce:
len(common_opponents)         # oponentes comunes entre A y B
```

`n_paths` y `len(common_opponents)` son proxies directos de la densidad local del grafo.

**Propuesta — density_confidence_factor:**

```python
def density_confidence(n_common_opponents: int, n_erdos_paths: int) -> float:
    """
    Modula confianza en senales transitivas segun densidad local.
    
    Grand Slam tipico: n_common ~15-30, n_paths ~20+ → factor ~1.0
    Challenger tipico: n_common ~2-3,  n_paths ~3-5  → factor ~0.4
    """
    raw = min(n_common_opponents, 20) / 20.0
    path_boost = min(n_erdos_paths, 30) / 30.0
    return 0.3 + 0.7 * ((raw + path_boost) / 2)  # rango [0.3, 1.0]

# Aplicar en generate_advanced_prediction():
density = density_confidence(len(common_opponents), erdos_result.get('n_paths', 0))
weights['common_opponents'] *= density
# Redistribuir peso sobrante a form_recent (no depende de la red)
redistribuido = weights_original['common_opponents'] * (1 - density)
weights['form_recent'] += redistribuido
```

**Ventaja sobre pesos fijos:** Los pesos por tier asumen que todos los Grand Slams tienen la misma densidad. Pero Parry vs Seyboth Wild en RG R1 tiene densidad baja (ambos jovenes) mientras que Djokovic vs Alcaraz tiene densidad alta. **Densidad local es variable continua, no categoria discreta.**

---

### Marco 3 — Bayesiano Empirico: James-Stein Shrinkage para Pesos

Cuando tienes multiples parametros (pesos por tier) estimados con poca data, el James-Stein estimator demuestra que "shrinking" cada estimacion hacia la media global **siempre** reduce el error total.

**Propuesta — Shrinkage adaptivo:**

```python
def shrink_weights(tier_weights: dict, default_weights: dict, n_tier: int, n_threshold: int = 20) -> dict:
    """
    Empirical Bayes shrinkage: pesos del tier se acercan al default
    cuando hay poca evidencia.
    
    n=31 (clay_gs):  factor = 31/51 = 0.61 → 61% tier-especifico, 39% default
    n=0  (hard_500): factor = 0/20  = 0.00 → 100% default (pesos seguros)
    n=100:           factor = 100/120 = 0.83 → 83% tier-especifico
    """
    factor = n_tier / (n_tier + n_threshold)
    return {
        k: round(factor * tier_weights[k] + (1 - factor) * default_weights[k], 3)
        for k in tier_weights
    }
```

**Fuente de n:** `calibracion_edge.json` → `por_superficie_y_tier` ya tiene los conteos.

```
clay_grand_slam: n=31 → factor=0.61 → confianza moderada en pesos GS
clay_challenger: n=36 → factor=0.64 → confianza moderada en pesos Challenger
hard_atp500:     n=0  → factor=0.00 → 100% default (nunca pesos ciegos)
```

---

### Marco 4 — Ingeniero de Senales: K-factor ELO como Kalman Gain

El K-factor en ELO es exactamente el **gain** de un Filtro de Kalman unidimensional:

```
Kalman gain: K_n = P_n / (P_n + R)
  P_n = incertidumbre sobre el rating real
  R   = ruido del resultado del partido
```

El sistema actual usa K=32 fijo para todo (`elo_system.py` line 11).

**Propuesta — K-factor por tier + reset post-PELT:**

```python
K_FACTOR_BY_TIER = {
    'grand_slam': 24,   # senal densa → cambios pequenos pero confiables
    'atp1000':    28,
    'atp500':     32,   # base clasica
    'challenger': 40,   # senal ruidosa → cambios mas agresivos
    'itf':        48,   # campo desconocido, cada partido muy informativo
}

# Conexion con Nodo-18 (PELT Recency):
def k_factor_efectivo(tier: str, recencia_pelt: int = None) -> int:
    k_base = K_FACTOR_BY_TIER.get(tier, 32)
    if recencia_pelt is not None and recencia_pelt <= 5:
        return int(k_base * 1.5)  # regimen nuevo = incertidumbre reiniciada
    return k_base
```

**Por que conecta:** Un ELO con K-factor tier-aware produce ratings que ya reflejan la fiabilidad de la senal. Cuando `rivalry_analyzer.py` lee `elo_rating`, recibe un ELO que "sabe cuanto confiar en si mismo".

---

## 5 Conexiones Ocultas Priorizadas

| # | Conexion | Alpha | Datos necesarios | Impacto |
|---|---|---|---|---|
| **C5** | `classify_tournament("French Open")` → `default` en vez de `grand_slam` — BUG | **Critico** | Texto torneo (ya existe) | Fix inmediato |
| **C1** | `classify_tournament()` no distingue GS de ATP 500 — ambos `atp_wta` | **Critico** | `detectar_tier()` ya resuelve esto | Unificar funciones |
| **C2** | Densidad local grafo como modulador continuo de peso common_opponents | **Alto** | `n_paths` + `n_common` (ya existen) | Reemplaza discreto por continuo |
| **C3** | James-Stein shrinkage de pesos segun n calibrado por tier | **Medio-Alto** | `calibracion_edge.json` (ya existe) | Pesos nunca ciegos |
| **C4** | K-factor ELO por tier + reset post-PELT (Kalman) | **Medio** | `tier` + `change_point` (ya existen) | ELO tier-aware |

---

## Tasks

### Fase 1 — Bug fix + Unificacion (alto impacto, bajo riesgo)

| ID | Descripcion | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T21-01 | **Fix classify_tournament()**: usar misma logica que `detectar_tier()` de edge_calculator. Retornar 5 categorias: `grand_slam`, `atp1000`, `atp500`, `challenger`, `itf` | `analysis/rivalry_analyzer.py` | 🔴 CRITICO (bug activo) | 🔴 PENDIENTE |
| T21-02 | Unificar en funcion compartida: mover `detectar_tier()` a `normalization.py` o `config.py`, importar desde ambos archivos | `config.py` + `edge_calculator.py` + `rivalry_analyzer.py` | 🟠 ALTO | 🔴 PENDIENTE |
| T21-03 | Actualizar `weights_config` en `generate_advanced_prediction()`: 5 tiers con pesos diferenciados (tabla de este nodo) | `analysis/rivalry_analyzer.py` | 🟠 ALTO | 🔴 PENDIENTE |
| T21-04 | Actualizar `normalization.py` DEFAULT_WEIGHTS para reflejar los 5 tiers | `normalization.py` | 🟡 MEDIO | 🔴 PENDIENTE |
| T21-05 | Tests: classify_tournament() con nombres reales de FlashScore (French Open, Wimbledon, ATP 500 Barcelona, Challenger Heilbronn, ITF M25, etc.) | `tests/test_rivalry_analyzer.py` | — | 🔴 PENDIENTE |

### Fase 2 — Densidad local + Shrinkage (medio impacto)

| ID | Descripcion | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T21-06 | Implementar `density_confidence(n_common, n_paths)` → modular `common_opponents` weight | `analysis/rivalry_analyzer.py` | 🟠 ALTO | ✅ COMPLETADO |
| T21-07 | Implementar `shrink_weights(tier_weights, default, n_tier)` con James-Stein | `analysis/rivalry_analyzer.py` o `normalization.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T21-08 | Tests: density_confidence con n_common=0,3,15,30 + shrinkage con n=0,10,31,100 | `tests/` | — | ✅ COMPLETADO — 16 tests |

### Fase 3 — K-factor ELO adaptivo (requiere Nodo-18)

| ID | Descripcion | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T21-09 | K-factor por tier en `elo_system.py`: K=24 GS, K=40 Challenger, K=48 ITF | `analysis/elo_system.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T21-10 | Reset K post-PELT: si `recencia_pelt <= 5` → K × 1.5 (conexion con Nodo-18 T18-01) | `analysis/elo_system.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T21-11 | Tests: K-factor por tier + reset post-PELT + convergencia ELO | `tests/` | — | ✅ COMPLETADO — 20 tests |

---

## Reglas Nuevas

**REGLA-T21-1: Una sola funcion de clasificacion de tier**
```
detectar_tier() en config.py o normalization.py — fuente unica de verdad.
classify_tournament() debe importar y delegar a detectar_tier().
NUNCA mantener dos funciones de clasificacion independientes.
```

**REGLA-T21-2: Pesos proporcionales al SNR del tier**
```
En Grand Slam: H2H y common_opponents ALTOS (data densa, senal limpia)
En Challenger:  form_recent y ranking_momentum ALTOS (data escasa, forma es lo unico)
En ITF:         form_recent DOMINANTE (unica senal confiable)
```

**REGLA-T21-3: Densidad local > Categoria discreta**
```
density_confidence() es un modulador CONTINUO de common_opponents weight.
Se aplica DESPUES de seleccionar pesos por tier.
Con densidad baja: peso common_opponents se redistribuye a form_recent.
```

**REGLA-T21-4: Pesos nunca ciegos — James-Stein obligatorio**
```
Si n_tier < 20 en calibracion_edge.json → shrinkage hacia default weights.
Con n=0: 100% default. Con n=31: 61% tier. Con n=100: 83% tier.
NUNCA usar pesos tier-especificos sin evidencia empirica.
```

**REGLA-T21-5: K-factor ELO refleja confianza del mercado**
```
Grand Slam K=24: senal confiable → ELO se mueve poco por partido
Challenger K=40: senal ruidosa → ELO se mueve mas (captura volatilidad)
Post-PELT fresco: K × 1.5 → incertidumbre reiniciada = ELO mas reactivo
```

---

## Orden de Implementacion Global (actualizado)

```
1. Nodo-21 Fase 1 (T21-01..05)  ← FIX BUG + unificar tier classification + pesos 5-tier
     ↓ desbloquea
2. Nodo-19 (H2H Immunity)       ← previene sobreconfianza HOT vs rival inmune
3. Nodo-18 (PELT Recency)       ← amplifica alpha temporal en ventana bookmaker stale
     ↓ desbloquea
4. Nodo-21 Fase 2 (T21-06..08)  ← densidad local + shrinkage
5. Nodo-20 (PageRank Erdos)     ← refinamiento calidad nodos intermedios
6. Nodo-21 Fase 3 (T21-09..11)  ← K-factor ELO adaptivo (requiere Nodo-18)
```

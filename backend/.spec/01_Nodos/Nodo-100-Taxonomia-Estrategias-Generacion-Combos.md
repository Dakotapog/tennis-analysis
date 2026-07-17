# Nodo-100 — Taxonomía Completa de Estrategias y Generación de Combos

> **Wikilinks:** [[Nodo-38-Portfolio-Aislamiento-Riesgo]] | [[Nodo-63-Anchor-Combo-Builder]] | [[Nodo-68-Rival-Value-Flip]] | [[Nodo-90-Auditoria-Fable-Nodo89]]
> **Fecha:** 2026-07-14 | **Origen:** Sesión de documentación — usuario pregunta cómo se generan cada tipo de combo
> **Estado:** REFERENCIA INMUTABLE — mapa del sistema tal como existe en 2026-07-14

---

## 1. VISIÓN GENERAL

El sistema genera picks desde **un único punto de entrada** (`edge_calculator.py`) y los distribuye a **4 generadores de combos** que producen **12 tipos distintos de apuestas**. Cada generador tiene su propia lógica, umbral de activación y stake.

```
edge_calculator.py  ←── FUENTE ÚNICA DE VERDAD
        │
        ├── NIVEL 1: TRADER FORMAL
        │       └── trader_ev_tenis.py
        │
        ├── NIVEL 2: PORTFOLIO POR CAPAS
        │       └── combo_confianza_builder.py
        │           ├── CORE + SATÉLITES + MOONSHOT
        │           ├── ANCHOR (Nodo-63)
        │           └── GCS (Nodo-61)
        │
        ├── NIVEL 3: LIBRO ALPHA/BETA
        │       └── betplay_combo_builder.py
        │           ├── SAFE (Beta Book)
        │           ├── WAS (Watchlist Alpha Signal)
        │           ├── MEGA (Alpha Book)
        │           └── GAMES (Over/Under juegos)
        │
        └── NIVEL 4: SEÑALES INVERSAS
                └── rival_value_betslip.py
                    └── RIVAL VALUE H88-01
```

---

## 2. NIVEL 1 — TRADER FORMAL ("El Motor")

**Archivo:** `trader_ev_tenis.py`
**Cómo generar:** `python3 trader_ev_tenis.py --bankroll 125000 --torneo-tipo atp1000`

### Lógica de generación

```
edge_calculator → edge_report
    → picks con apostar=True (edge>5%, kelly>2%)
    → Kelly-KL ajustado por tier (λ: GS=1.0× | ATP1000=1.6× | Challenger=3.6× | ITF=4.5×)
    → KGR > 0 (guard anti-ruina obligatorio)
    → VaR auto-ajustado (MAX_VAR_PCT=0.25)
    → stake individual por pick
```

### Cuándo dispara
Solo cuando el modelo ve edge REAL sobre el bookmaker Y todos los guards pasan. Es el más exigente del sistema — puede pasar sesiones sin disparar si el mercado está bien calibrado.

### Nombre propio: **"EL MOTOR"**
- Apuesta: el FAVORITO del modelo cuando tiene edge positivo confirmado
- Stake: Kelly-KL fraccionado por tier
- Output: `trader_plan_*.json` → alimenta a NIVEL 2 y NIVEL 3

---

## 3. NIVEL 2 — PORTFOLIO POR CAPAS ("El Satélite")

**Archivo:** `combo_confianza_builder.py`
**Cómo generar:** `python3 combo_confianza_builder.py --bankroll 125000 --fase 4 [--anchor] [--telegram]`

### 3.1 Categorización de picks

| Cat | Cuota | Confianza mínima | Rol en combos |
|-----|-------|-----------------|---------------|
| **Cat-A** | 1.15–1.59 | ≥53% | Base del CORE — multiplicador seguro |
| **Cat-B** | 1.60–2.20 | ≥53% | Valor real del modelo — CORE + SAT |
| **Cat-C1** | 2.21–3.50 | ≥60% (57% con pipeline) | Alpha — entra a SAT y MOON |
| **Cat-C2** | >3.50 ó conf<60% | — | Alpha puro — solo MOONSHOT |

**Pipeline Promotion:** Si pick está en `edge_report.apostar` + cuota≤4.50 + conf≥57% → Cat-C2 sube automáticamente a Cat-C1.

---

### 3.2 COMBO: CORE ("La Base")

**Nombre propio: "CORE"**
**Qué es:** Combo de 4–7 picks Cat-A y Cat-B únicamente. NUNCA Cat-C (REGLA-ISO-1).
**Cuota típica:** @2–5x
**Probabilidad de ganar:** ≥25%
**Budget:** 45% del budget de la fase

```
Activación: siempre que haya ≥4 picks Cat-A/B con P(win_combo) ≥ 25%
Stake: $2,000–$5,000 según fase
REGLA: el CORE es intocable — si muere es porque el modelo falló globalmente
```

**Selección de picks:** `combo_priority = confianza + alpha_score (Signal Bridge Nodo-62)`. Los picks con mayor prioridad entran primero.

---

### 3.3 COMBO: SATÉLITE ("El Aislado")

**Nombre propio: "SAT-1", "SAT-2", "SAT-3"**
**Qué es:** Combo de 4 picks Cat-A/B + 1 pick Cat-C1. El Cat-C es la "apuesta de valor" — si falla, el CORE y los otros satélites siguen vivos.
**Cuota típica:** @5–8x
**Probabilidad:** 12–20%
**Budget:** 15% c/u (hasta 3 satélites)

```
Activación: si hay picks Cat-C1 disponibles (conf≥60%)
Fases: SAT disponible desde Fase 2
Aislamiento: cada SAT tiene distinto pick Cat-C — nunca el mismo Cat-C en dos SAT
REGLA: Max 2 picks del mismo torneo por combo (Guard concentración)
```

---

### 3.4 COMBO: MOONSHOT ("El Jackpot")

**Nombre propio: "MOONSHOT"**
**Qué es:** Combo de 5 picks — 3 Cat-A/B + 2–3 Cat-C (conf≥57%). El objetivo es la sesión épica.
**Cuota típica:** @15–35x
**Probabilidad:** 4–10%
**Budget:** 5% del budget de fase

```
Activación: si hay ≥2 picks Cat-C con conf≥57%
Fases: MOONSHOT disponible desde Fase 3
Stake: $500–$2,000
REGLA: Cuota combo mínima @15x. Si no se alcanza, no generar.
```

---

### 3.5 COMBO: COBERTURA ("El Hedge")

**Nombre propio: "COB"**
**Qué es:** CORE omitiendo 1 pick de menor confianza + 1 pick de reserva. Hedgea el escenario donde 1 pierna del CORE falla.
**Cuota típica:** @2–4x
**Probabilidad:** 30–50%
**Budget:** 5% del budget de fase

```
Activación: solo Fase 4
Genera: 2 combos (excluye el pick más débil del CORE, lo reemplaza por 1 reserva)
Stake: $1,000–$2,000
```

---

### 3.6 ANCHOR COMBOS ("Las Anclas") — Nodo-63

**Nombre propio: "ANCHOR"**
**Flag:** `--anchor`
**Qué es:** Combos construidos alrededor de picks de ALTA prioridad (anclas) — cuota ≥1.65 + alpha_score alto. Tres tiers de riesgo creciente.

**¿Qué es una ancla?** Pick con AL MENOS UNO de:
- `combo_priority ≥ 75.0` AND `cuota ≥ 1.65`
- `confianza ≥ 60.0%` AND `cuota ≥ 1.65`
- `edge_pct ≥ 10.0%` AND `cuota ≥ 1.65`

| Tipo | Estructura | Cuota objetivo | P(win) | Budget |
|------|-----------|----------------|--------|--------|
| **ANCHOR-1A3B** | 1 ancla + 3 base | @4–7x | 18–25% | 10% fase |
| **ANCHOR-2A2B** | 2 anclas + 2 base | @7–15x | 7–14% | 10% fase |
| **ANCHOR-3A2B** | 3 anclas + 2 base | @15–35x | 3–6% | 10% fase |

```
Constantes congeladas:
  ANCHOR_CUOTA_MIN    = 1.65
  ANCHOR_PRIORITY_MIN = 75.0
  ANCHOR_CONF_MIN     = 60.0
  ANCHOR_EDGE_MIN     = 10.0%
  ANCHOR_PWIN_MIN     = 2.5%
  MAX_ANCHOR_COMBOS   = 12 por tier
Output: AC1.bat, AC2.bat, ... (prefijo AC en Desktop)
```

---

### 3.7 GCS COMBOS ("El Especialista de Hierba") — Nodo-61

**Nombre propio: "GCS"**
**Qué es:** Sub-plan INDEPENDIENTE para picks de jugadores con dominancia demostrada en hierba. NUNCA mezclar con ITF en el CORE.
**Cuota típica:** @1.5–3x
**Budget:** 2% del budget de fase

```
Activación:
  gcs_active = True en edge_report
  + tier ≥ ATP500
  + dentro de ventana 21 días hasta torneo
  + superficie = hierba

Combos: 2–3 piernas solo entre picks GCS
Stake: $200–$500 (conservador — H60-01 acumulando, n<30)
REGLA: Si CORE muere, GCS vive — universo separado
```

---

## 4. NIVEL 3 — LIBRO ALPHA/BETA ("Los Libros")

**Archivo:** `betplay_combo_builder.py`
**Cómo generar:** `python3 betplay_combo_builder.py --live [--mega] [--safe] [--games] [--telegram]`

---

### 4.1 SAFE COMBOS ("El Libro Beta") — Nodo-25

**Nombre propio: "SAFE"**
**Flag:** `--safe`
**Qué es:** Combos de EXACTAMENTE 2 piernas de torneos distintos. Máxima probabilidad, mínimo riesgo relativo.
**Cuota típica:** @3–12x
**Probabilidad:** 25–50%
**Stake:** $1,000 fijo

```
Activación: siempre que haya ≥2 picks en trader_plan con P(ambos) ≥ 25%
Algoritmo:
  1. Genera todos los pares (i,j) del pool
  2. Guard 1: P(ambos) ≥ 25% usando P_mercado
  3. Guard 2: torneos DISTINTOS (max 2 del mismo torneo)
  4. Guard 3: solo picks en trader_plan (APOSTAR o WATCHLIST)
  5. Guard 4: no combos duplicados
  6. Guard Dispersión: si std(p_blend)<0.015 (BLIND) → bloquear
  Score: P(ambos) + 0.01 × log(cuota_combo)
  Top N por score

Output: Safe1.bat, Safe2.bat, ... (prefijo Safe en Desktop)
Telegram: siempre con --telegram
```

**Guards críticos:**
- **Guard Dispersión (Nodo-25-1):** `std(p_blend) < 0.015` → BLIND, mega bloqueado
- **Guard Concentración (Nodo-25-2):** max 2 picks mismo torneo

---

### 4.2 WAS COMBOS ("El Watchlist Alpha") — Nodo-44

**Nombre propio: "WAS"**
**Flag:** incluido en `--live`
**Qué es:** Combos de 2–3 piernas construidos desde picks de la WATCHLIST (edge ≥10%, cuota ≥2.0) con señal Markov explícita. Alpha "invisible" que el bookmaker no modeló.
**Cuota típica:** @4–25x

```
Activación de pick WAS (los 3 requisitos):
  1. edge_pct ≥ 10%
  2. cuota_favorito ≥ 2.0
  3. AL MENOS UNA señal Markov:
     - markov_rival == COLD AND conf_rival ≥ 60%  (rival frío)
     - markov_favorito == HOT AND conf_fav ≥ 60%  (favorito caliente)
     - |wr_rec_fav - wr_rec_rival| > 0.35         (zona dominante)
     - COINFLIP + rival COLD conf ≥ 60%

Sin señal Markov = no WAS (coin-flip puro bloqueado por REGLA-WAS-1)
Stake: $5,000 (promo) hasta n≥30 observaciones
Output: WAS1.bat, WAS2.bat, ...
```

---

### 4.3 MEGA COMBOS ("El Libro Alpha") — Nodo-23

**Nombre propio: "MEGA"**
**Flag:** `--mega`
**Qué es:** Combos de 6–10 piernas cross-tier con cuotas @100–@1000+. El objetivo es la sesión histórica.
**Cuota típica:** @100–@1000+
**Probabilidad:** 0.5–3%
**Stake:** $500 fijo por combo

```
Activación: solo si Dispersión = DIFFERENTIATED (std ≥ 0.04)
Smart Scoring (Nodo-26-1):
  combo_score = EV × diversity_bonus × regime_bonus × alpha_bonus
  diversity_bonus: 1.20 (mix safe+risky) | 0.80 (homogéneo)
  regime_bonus: producto Markov por pick (HOT×1.05, COLD×0.90)
  alpha_bonus: 1.0 + min(avg_alpha_vs_elo × 0.5, 0.10)

Guards (5):
  G1: std(p_blend) < 0.015 → BLIND → NO mega
  G2: std(edges) < 0.15 → señal débil → NO mega
  G3: Session Regime (COLD_MODEL → stake×0.50)
  G4: Max session loss 4% bankroll → circuit breaker
  G5: Planes frescos ≤4h (D89-01)

Output: Combo1.bat, Combo2.bat, ... (prefijo Combo en Desktop)
Frecuencia: 1–2 días por semana cuando Dispersión DIFF
```

---

### 4.4 GAMES COMBOS ("El Over/Under") — Nodo-40

**Nombre propio: "GAMES"**
**Flag:** `--games`
**Qué es:** Apuestas sobre el TOTAL DE JUEGOS del partido (Over/Under), no sobre el ganador. Señal ORTOGONAL al pick ganador.
**Cuota típica:** @1.80–2.10
**Probabilidad:** 45–55%

```
Activación: match tiene mercado BET_GAMES_OVER_UNDER en Kambi
Modelo:
  - Lee histórico de juegos/sets entre jugadores
  - Fit log-normal para distribución esperada de juegos
  - Calibra thresholds por tier (ATP1000 vs ITF)
  - Auto-calibración: ajusta confidence gates con accuracia histórica

Output: picks OVER o UNDER por partido
Uso en combos: se puede combinar con pick ganador (partido X favorito + OVER juegos)
Estado: OPERATIVO pero pendiente acumulación n≥20 para gates de confianza
```

---

## 5. NIVEL 4 — SEÑALES INVERSAS ("El Inverso")

---

### 5.1 RIVAL VALUE ("El Inverso") — H88-01, Nodo-68

**Nombre propio: "RIVAL VALUE"**
**Archivo:** `rival_value_betslip.py`
**Cómo generar:** `python3 rival_value_betslip.py --bankroll 125000 [--telegram]`

**Qué es:** Apuesta al RIVAL del modelo cuando el FAVORITO tiene edge MUY negativo. El mercado sobre-pagó al favorito → el valor cruzó de lado.

```
Activación (rival_value_flag=True en edge_report):
  edge_fav ≤ −10%        (mercado sobre-valora al favorito masivamente)
  cuota_rival ∈ [2.50, 8.00]  (rango de valor pre-registrado)
  status ≠ NO_DATA
  phantom_data ≠ True

Stake (micro-Kelly pre-graduación):
  kelly_raw = edge_rival / (cuota_rival - 1)
  shrinkage = n_obs / (n_obs + 50) = 3/53 = 5.7%  (actual 2026-07-14)
  stake = min(kelly_raw × shrinkage, 0.5%) × bankroll
  mínimo: $2,000 | redondeado a $500

Output:
  - Links individuales por rival
  - Link COMBO con TODOS los rivales del día (ancla de cuota alta)
  - Telegram con stakes y links

Protocolo H88-01:
  n_actual = 3 (hits=3, Wilson LB=0.526, breakeven=0.267)
  Gate: n≥30 antes de incrementar stakes
  PROHIBIDO subir stakes antes de n=30
  Evidencia 2026-07-14: 3/3 wins, combinada 41.25x
```

---

### 5.2 RFI ULTRA ("El Regreso") — H76-01, Nodo-64

**Nombre propio: "RFI"**
**Dónde aparece:** `rfi_ultra=True` en edge_report; usado como señal auxiliar en SATELLITE y ANCHOR

**Qué es:** Pick donde el favorito regresa de INACTIVIDAD PROLONGADA (>30 días sin jugar). El bookmaker penaliza la cuota por inactividad; el modelo detecta que el historial previo era positivo.

```
Activación:
  rfi_ultra = True en pick (serializado por edge_calculator)
  rfi_tier: distancia de inactividad (días)
  rfi_decay_gap: qué tan severo fue el gap

Uso en combos:
  Pick con rfi_ultra=True puede entrar a SATELLITE aunque conf sea MOD
  En ANCHOR: +5 pts de alpha_score si rfi_ultra=True AND cuota≥1.65

Gate H76-01: acumulando n hacia n=30
```

---

## 6. FLUJO COMPLETO DE GENERACIÓN

### Mañana típica (9–12h CO)

```bash
# PASO 1: Datos
python3 extraer_URL_partidos_version2.py
python3 extraer_historh2h.py --all-tournaments

# PASO 2: Edge + señales
python3 edge_calculator.py
# → edge_report_*.json con:
#    apostar[], watchlist[], sin_edge[]
#    campos: confidence_flag, triple_alignment, markov, rfi_ultra, rival_value_flag
#    games_signal disponible si se corrió games_signal_calculator.py

# PASO 3: Revisión humana (OBLIGATORIO antes de apostar)
python3 generar_tabla_favoritos2.py

# PASO 4A: EL MOTOR (picks formales)
python3 trader_ev_tenis.py --bankroll 125000
# → trader_plan_*.json

# PASO 4B: EL SATÉLITE (portfolio por capas)
python3 combo_confianza_builder.py --bankroll 125000 --fase 4 --anchor --telegram
# → CC1.bat … CC11.bat (CORE/SAT/MOON/COB)
# → AC1.bat … AC12.bat (ANCHOR tiers)

# PASO 4C: LOS LIBROS (Alpha/Beta)
python3 betplay_combo_builder.py --live --safe --mega --telegram
# → Combo1.bat … ComboN.bat (MEGA)
# → Safe1.bat … SafeN.bat (SAFE)
# → WAS1.bat … WASN.bat (WAS si hay watchlist con markov)

# PASO 4D: EL INVERSO (rival value)
python3 rival_value_betslip.py --bankroll 125000 --telegram
# → Links individuales rival + combo H88-01
```

---

## 7. TABLA RESUMEN — LAS 12 ESTRATEGIAS

| # | Nombre Propio | Archivo | Piernas | Cuota | Stake | P(win) | Estado |
|---|--------------|---------|---------|-------|-------|--------|--------|
| 1 | **EL MOTOR** | `trader_ev_tenis.py` | 1 | variable | Kelly-KL | variable | OPERATIVO |
| 2 | **CORE** | `combo_confianza_builder.py` | 4–7 | @2–5x | $2k–$5k | 25–40% | OPERATIVO |
| 3 | **SATELITE** | `combo_confianza_builder.py` | 5 | @5–8x | $2k–$3k | 12–20% | OPERATIVO (F2+) |
| 4 | **MOONSHOT** | `combo_confianza_builder.py` | 5 | @15–35x | $1k–$2k | 4–10% | OPERATIVO (F3+) |
| 5 | **COBERTURA** | `combo_confianza_builder.py` | 4–7 | @2–4x | $1k–$2k | 30–50% | OPERATIVO (F4) |
| 6 | **ANCHOR** | `combo_confianza_builder.py` | 4–5 | @4–35x | $1.5k | 3–25% | OPERATIVO (--anchor) |
| 7 | **GCS** | `combo_confianza_builder.py` | 2–3 | @1.5–3x | $200–$500 | 35–65% | OPERATIVO (hierba) |
| 8 | **SAFE** | `betplay_combo_builder.py` | 2 | @3–12x | $1k | 25–50% | OPERATIVO (--safe) |
| 9 | **WAS** | `betplay_combo_builder.py` | 2–3 | @4–25x | $5k promo | 8–20% | OPERATIVO (--live) |
| 10 | **MEGA** | `betplay_combo_builder.py` | 6–10 | @100–@1000+ | $500 | 0.5–3% | OPERATIVO (--mega) |
| 11 | **GAMES** | `betplay_combo_builder.py` | 1 | @1.8–2.1x | $1k–$2k | 45–55% | OPERATIVO (--games) |
| 12 | **RIVAL VALUE** | `rival_value_betslip.py` | 1 | @2.5–8x | $2k | 20–40% | ACUMULANDO H88-01 (n=3/30) |

---

## 8. SEÑALES AUXILIARES QUE CRUZAN ESTRATEGIAS

Estas señales NO son estrategias propias — son flags del edge_report que los generadores usan para PRIORIZAR picks dentro de sus combos:

| Señal | Campo en edge_report | Efecto en combos |
|-------|---------------------|-----------------|
| **RFI Ultra** | `rfi_ultra=True` | +prioridad en SAT, ANCHOR |
| **Markov HOT** | `markov_favorito=HOT` | +alpha_score CORE/ANCHOR, +regime_bonus MEGA |
| **Markov COLD rival** | `markov_rival=COLD` | activa WAS, +descuento en MOONSHOT |
| **Triple Alignment** | `triple_alignment≥0.5` | +15pts Signal Bridge → Cat-C2→C1 |
| **GCS activo** | `gcs_bonus=True` | entra al sub-plan GCS |
| **Rival Value flag** | `rival_value_flag=True` | genera betslip invertido |
| **IRP Rival** | `irp_rival.delta_return` | señal auxiliar Rival Value (Nodo-96) |
| **confidence STRONG** | `confidence_flag=STRONG` | prioridad máxima en CORE |

---

## 9. LOS COMBOS QUE FUNCIONARON (Evidencia Real)

| Fecha | Estrategia | Resultado |
|-------|-----------|-----------|
| 2026-07-14 | **RIVAL VALUE** (3 picks) | 3/3 WINS — combinada 41.25x |
| 2026-06-29 | **WAS** (Carreno @3.30 + Safiullin @2.65) | 2/2 WINS |
| 2026-06-01 | **EL MOTOR** (Kelly-KL, n≥30) | P&L positivo validado |
| 2026-07-10 | **GCS** (H60-01 GRADUADA, n=54, 64.8%) | Hit% 64.8% — gate superado |

---

*Documento de referencia. Actualizar este Nodo cuando se añada una nueva estrategia o se cambie la lógica de generación de algún combo. No editar estrategias existentes — añadir entrada nueva marcada con fecha.*

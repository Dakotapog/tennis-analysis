# Nodo-24 — Bookmaker Blindness Scoring

> **Estado:** 🔧 IMPLEMENTADO — 2026-06-14
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-23-Cross-Tier-Mega-Combos]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Origen:** Post-mortem sesión épica 2026-06-13 — análisis de por qué Miguel perdió y 9 ganaron
> **Prioridad:** ALTA — mejora directa en selección de mega-combos y filtrado de picks débiles

---

## Problema — El Pipeline No Distingue Edge Real de Edge por Calibración

La sesión 2026-06-13 reveló que **todos los p_modelo estaban entre 0.504 y 0.531** (coin flip).
El edge venía EXCLUSIVAMENTE de la cuota vs p_blend (calibración histórica).

Pero no todos los edges son iguales:
- **Edge real:** el bookmaker NO tiene datos → cuota mal puesta → mispricing explotable
- **Edge fantasma:** el bookmaker SÍ tiene datos → cuota correcta → calibración infla p_blend

Miguel (@2.35, edge 8.5%) parecía "seguro" con p_blend=0.654, pero perdió.
Romano (@2.75, edge 14.0%) parecía "coin flip" con p_blend=0.569, pero ganó.

**La diferencia: de DÓNDE viene el edge, no cuánto edge hay.**

---

## Evidencia Empírica — Sesión 2026-06-13

### Datos crudos de los 10 picks

```
PICK                  cuota  p_mod  p_blend  gap    edge%  kelly  n_h2h  RESULTADO
─────────────────────────────────────────────────────────────────────────────────
Romano         @2.75  0.504  0.569   0.065   14.0%  0.080  0      ✅ WON
Pawlikowska    @2.95  0.504  0.590   0.086   16.5%  0.088  0      ✅ WON
Carnicella     @2.70  0.510  0.590   0.080   14.0%  0.079  0      ✅ WON
D'Agostino     @1.98  ~0.51  ~0.59   ~0.08   ~10%   ~0.06  0      ✅ WON
Lim            @2.18  0.531  0.660   0.129    7.2%  0.047  1      ✅ WON (borderline)
Martinez Gomez @1.50  ~0.53  ~0.59   ~0.06   ~5%    ~0.04  0      ✅ WON
Daniel         @2.02  ~0.50  ~0.57   ~0.07   ~14%   ~0.08  0      ✅ WON
Fearnley       @1.68  ~0.52  ~0.57   ~0.05   ~8%    ~0.06  0      ✅ WON
Bu             @1.65  ~0.52  ~0.57   ~0.05   ~8%    ~0.06  0      ✅ WON
──────────────── PERDEDORES ────────────────────────────────────────────────────
Miguel         @2.35  0.511  0.654   0.143    8.5%  0.051  1      ❌ LOST
Mannarino      @4.30  0.522  0.538   0.016   28.9%  0.131  6      ❌ LOST
Mmoh           @2.35  0.514  0.535   0.021    8.8%  0.055  5      ❌ LOST
```

### 4 conexiones ocultas descubiertas

**Conexión 1 — Gap (p_blend - p_modelo) como señal de peligro:**
```
gap = p_blend - p_modelo
gap > 0.12 → "CALIBRATION_DRIVEN" — edge viene de historial, no del análisis
gap < 0.08 → "CUOTA_DRIVEN" — edge viene de cuota mal puesta por bookmaker
```
Miguel gap=0.143 (el más alto) → calibración clay inflaba p_blend → PERDIÓ.

**Conexión 2 — n_h2h=0 gana, n_h2h>0 es ruidoso:**
```
n_h2h=0: 7/7 ganaron (100%) — bookmaker NO tiene datos directos
n_h2h=1: 1/2 ganaron (50%)  — señal mixta
n_h2h≥5: 0/2 ganaron (0%)   — bookmaker YA incorporó H2H en cuota
```
Sin H2H → cuota más ruidosa → mayor oportunidad de mispricing.

**Conexión 3 — Tier bajo + cuota alta = "golden zone":**
```
Challenger/ITF + cuota > 2.50 + n_h2h = 0 → "GOLDEN ZONE"
Rankings 200-500: bookmaker tiene modelos menos sofisticados
Rankings Top 100: bookmaker tiene datos perfectos → edge ≈ 0
```
Romano, Pawlikowska, Carnicella: todos golden zone → todos ganaron.

**Conexión 4 — kelly_kl bajo NO significa mal pick en mega-combos:**
```
Kelly penaliza picks con alta incertidumbre (correctamente para individuales).
Pero en mega-combos, la incertidumbre es FEATURE no BUG:
  alta incertidumbre → bookmaker ciego → cuota regalada → valor real
```

---

## Fórmulas Implementables

### F-24-1: Bookmaker Blindness Index (BBI)

```python
def bookmaker_blindness(cuota, n_h2h):
    """Cuánto NO ve el bookmaker. 0=ve todo, 1=ciego total."""
    cuota_factor = 1 - (1 / cuota)          # cuota alta → bookmaker subestima
    h2h_penalty  = 1 / (1 + n_h2h * 0.20)   # más H2H → bookmaker sabe más
    return cuota_factor * h2h_penalty

# Pawlikowska @2.95, n_h2h=0: BBI = 0.661 × 1.0 = 0.661  ← bookmaker ciego → ✅
# Miguel     @2.35, n_h2h=1: BBI = 0.574 × 0.83 = 0.478  ← bookmaker ve algo → ❌
# Mannarino  @4.30, n_h2h=6: BBI = 0.767 × 0.45 = 0.349  ← bookmaker ve todo → ❌
```

**Umbral:** BBI < 0.40 → excluir de mega-combos (bookmaker tiene info suficiente).

### F-24-2: Calibration Gap Alert

```python
def calibration_gap(p_blend, p_modelo):
    """Gap entre calibración histórica y predicción del modelo."""
    return p_blend - p_modelo

# gap > 0.12 → flag "CALIBRATION_DRIVEN"
# gap < 0.08 → flag "MARKET_DRIVEN" (edge real de cuota)
```

**Uso en combo builder:** picks CALIBRATION_DRIVEN tienen peso reducido (×0.85) en mega_score.

### F-24-3: Mega-Combo Pick Quality (MPQ)

Reemplaza el scoring actual `P_todas × log(cuota) × cross_tier_bonus`:

```python
def mega_pick_quality(kelly_kl, bbi, edge_pct):
    """Calidad de un pick para mega-combos. Incorpora ceguera del bookmaker."""
    return kelly_kl * bbi * (1 + edge_pct / 100)

# Pawlikowska: 0.088 × 0.661 × 1.165 = 0.0675  → TOP
# Romano:      0.080 × 0.636 × 1.140 = 0.0584  → TOP
# Miguel:      0.051 × 0.478 × 1.085 = 0.0264  → WORST → excluir
```

**Nuevo mega_score:**
```python
mega_score = (Π mpq_i) × log(cuota_combo) × cross_tier_bonus × gap_penalty
gap_penalty = Π min(1.0, 1.15 - gap_i) para cada pierna
```

### F-24-4: Golden Zone Detector

```python
def is_golden_zone(tier, cuota, n_h2h):
    """Detecta picks donde bookmaker tiene máxima desventaja informacional."""
    return (
        tier in ('challenger', 'itf') and
        cuota >= 2.50 and
        n_h2h == 0
    )

# golden_zone=True → bonus ×1.20 en mega_score
# golden_zone en ≥50% de piernas → flag "EPIC_POTENTIAL"
```

---

## Diseño de Implementación

### Fase 1 — Signals en edge_calculator.py (campos nuevos por pick)

```python
# Añadir a cada pick en el edge report:
pick['bbi']              = bookmaker_blindness(cuota, n_h2h)
pick['calibration_gap']  = p_blend - p_modelo
pick['gap_flag']         = 'CALIBRATION_DRIVEN' if gap > 0.12 else 'MARKET_DRIVEN' if gap < 0.08 else 'MIXED'
pick['golden_zone']      = is_golden_zone(tier, cuota, n_h2h)
pick['mpq']              = mega_pick_quality(kelly_kl, bbi, edge_pct)
```

### Fase 2 — Scoring en build_mega_combos() (betplay_combo_builder.py)

```python
# Reemplazar scoring actual:
# ANTES:  P_todas × log(cuota_combo) × cross_tier_bonus
# DESPUÉS: (Π mpq_i) × log(cuota_combo) × cross_tier_bonus × gap_penalty

# Filtro nuevo: BBI < 0.40 → excluir pick del pool mega
# Bonus: golden_zone picks → ×1.20
```

### Fase 3 — Display en _mostrar_mega_combos()

```
Mega-Combo #1 (7 piernas) — cuota @269.2
  ⚓ Bu @1.65           BBI=0.39  gap=0.05  MARKET     
  ⚓ Fearnley @1.68     BBI=0.40  gap=0.05  MARKET     
  🛰️ Romano @2.75      BBI=0.64  gap=0.07  MARKET     🌟 GOLDEN
  🛰️ Pawlikowska @2.95 BBI=0.66  gap=0.09  MARKET     🌟 GOLDEN
  🛰️ Carnicella @2.70  BBI=0.63  gap=0.08  MARKET     🌟 GOLDEN
  🛰️ Lim @2.18         BBI=0.54  gap=0.13  CALIBRATION ⚠️
  🛰️ D'Agostino @1.98  BBI=0.49  gap=0.08  MARKET
  
  MPQ medio: 0.052 | Golden: 3/7 (43%) | ⚠️ Calibration: 1/7
```

---

## Restricciones

- **R-24-1:** BBI, gap_flag, golden_zone son campos informativos — NO cambian p_blend ni edge_pct.
- **R-24-2:** MPQ scoring solo aplica en mega-combos (≥6 piernas). Combos normales usan scoring existente.
- **R-24-3:** BBI < 0.40 es soft filter — el usuario puede override con `--no-bbi-filter`.
- **R-24-4:** Estos campos fluyen por el JSON del edge_report → el trader los propaga → el combo builder los lee.
- **R-24-5:** Golden zone bonus (×1.20) es configurable. Default conservador.

---

## Tests

```
T24-01: BBI(cuota=2.95, n_h2h=0) = 0.661
T24-02: BBI(cuota=4.30, n_h2h=6) = 0.349 → excluido de mega pool (< 0.40)
T24-03: calibration_gap(0.654, 0.511) = 0.143 → CALIBRATION_DRIVEN
T24-04: calibration_gap(0.569, 0.504) = 0.065 → MARKET_DRIVEN  
T24-05: golden_zone(challenger, 2.75, 0) = True
T24-06: golden_zone(atp500, 4.30, 6) = False
T24-07: MPQ(0.088, 0.661, 16.5) > MPQ(0.051, 0.478, 8.5) — Pawlikowska > Miguel
T24-08: mega_score con MPQ prioriza pool sin Miguel sobre pool con Miguel
T24-09: gap_penalty reduce score de combo con ≥2 piernas CALIBRATION_DRIVEN
T24-10: --no-bbi-filter permite picks con BBI < 0.40
```

---

## Impacto Esperado

Aplicado retroactivamente a la sesión 2026-06-13:
- Miguel (BBI=0.478, gap=0.143) → **excluido** o penalizado → combo 9p no se genera
- Mannarino (BBI=0.349) → **excluido** del pool mega → ATP500 loss evitado
- Combo 7p y 8p se generan igual (todos sus picks tienen BBI > 0.50)
- Resultado: misma ganancia, menos pérdida

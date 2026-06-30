# Nodo-25 — Dispersion Guard + Safe Combos + Tournament Concentration

> **Estado:** 📋 SPEC COMPLETO — 2026-06-14
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-23-Cross-Tier-Mega-Combos]] | [[Nodo-24-Bookmaker-Blindness-Scoring]] | [[Nodo-15-Portfolio-HedgeFund]]
> **Origen:** Post-mortem sesión desastre 2026-06-14 — 25 combos, 0 vivos, -$12,500
> **Prioridad:** CRÍTICA — previene repetición de pérdida total por blindness del modelo

---

## Problema — 5 Fallas Estructurales del 14-Jun

### F-25-1: Model Blindness (James-Stein Collapse)

```
Milev     @2.75  p_blend=0.590  → PERDIÓ
Bocchi    @3.35  p_blend=0.590  → GANÓ
Seggerman @2.55  p_blend=0.590  → GANÓ
Hewitt    @3.55  p_blend=0.569  → GANÓ
```

Con `n_tier` bajo, James-Stein shrinkage: `factor = n/(n+20)` → 83% de p_blend viene del fallback_por_tier.
TODOS los picks del mismo tier colapsan al mismo p_blend.
El modelo se vuelve **CIEGO** — no distingue entre ganadores y perdedores.

**Analogía financiera (Factor Crowding):** Cuando el shrinkage colapsa todas las predicciones, el modelo apuesta por SECTOR, no por INDIVIDUO. Es como comprar "tech sector" sin distinguir entre Apple y una startup que quiebra mañana.

### F-25-2: Tournament Concentration (Parma = 75%)

```
Combo #12: Bocchi+Milev+Seggerman+Hewitt  → 3/4 piernas = Parma Challenger
Combo #13: Meliss+Bocchi+Milev+Seggerman  → 3/4 piernas = mismo torneo
```

Con ρ_mismo_torneo=0.25, la independencia asumida es falsa.
`n_effective = n / (1 + ρ × (n_same - 1))` → 4 picks del mismo torneo ≈ 2.3 picks independientes.

### F-25-3: Picks Fuera del Trader Plan (Disciplina)

6 picks EXTRA (Cerny, Cherubini, Vedder, Ribero, Damm, Romano@1.35) sin validación Kelly-KL.
El combo builder mezcló picks del trader con picks sin edge validado.

### F-25-4: Combo Duplicado

Combo #10 y #11 idénticos: Bautista+Sun+Damm @12.62 × 2 = $1,000 en el mismo resultado.
Viola REGLA-KAMBI-1 (||append acumula betslip).

### F-25-5: Sin Safe Combos (Solo Mega-Combos de Alta Varianza)

La sesión solo generó combos de 3-11 piernas con cuotas @10-@28000.
Con 36% accuracy (4/12), NINGÚN combo sobrevivió.
Sin combos de 2 piernas "safe" que generaran retorno parcial.

---

## Señales Disponibles para Detección

### Bookmaker Relative Signal

Cuando el modelo tiene dispersión CERO pero el bookmaker tiene variación de cuotas:

```
Parma Challenger — modelo: todos p_blend=0.590
                — bookmaker: Seggerman @2.55, Milev @2.75, Bocchi @3.35

Bookmaker ranking (menor cuota = más favorecido):
  1. Seggerman @2.55  → GANÓ ✅
  2. Milev @2.75      → PERDIÓ ❌
  3. Bocchi @3.35     → GANÓ ✅

Bookmaker acertó 2/3 del ranking relativo.
```

Cuando el modelo es ciego, la variación de cuotas del bookmaker CONTIENE información.
No la usamos para calcular edge (eso ya lo hace p_implícita), pero sí para ranking relativo dentro del tier.

---

## Diseño — 4 Guards + Safe Combos

### Guard 1: Dispersion Guard (F-25-1)

```python
def dispersion_index(picks_pool):
    """Mide si el modelo distingue entre picks del pool."""
    p_blends = [p['p_blend'] for p in picks_pool]
    return np.std(p_blends)

# Clasificación:
# BLIND:          std < 0.015 → modelo NO distingue → NO generar mega-combos
# LOW_SIGNAL:     0.015 ≤ std < 0.04 → señal débil → máx 3 piernas, stake reducido
# DIFFERENTIATED: std ≥ 0.04 → modelo distingue → mega-combos permitidos
```

**Acción por nivel:**

| Nivel | std(p_blend) | Mega-combos | Safe combos | Stake |
|---|---|---|---|---|
| BLIND | < 0.015 | ❌ BLOQUEADO | ✅ solo safe | 50% normal |
| LOW_SIGNAL | 0.015-0.04 | ⚠️ máx 3 piernas | ✅ prioritario | 75% normal |
| DIFFERENTIATED | ≥ 0.04 | ✅ normal | ✅ normal | 100% |

### Guard 2: Tournament Concentration Limit (F-25-2)

```python
def tournament_concentration(combo_picks):
    """Máximo 2 picks del mismo torneo en cualquier combo."""
    from collections import Counter
    torneos = Counter(p['torneo'] for p in combo_picks)
    max_same = max(torneos.values())
    return max_same <= 2  # True = OK, False = rechazar combo

def n_effective(n_picks, n_same_tournament, rho=0.25):
    """Picks efectivos descontando correlación intra-torneo."""
    return n_picks / (1 + rho * (n_same_tournament - 1))
```

**Regla:** `max_same_tournament = 2` en cualquier combo (mega o safe).
Con ρ=0.25, 2 del mismo torneo ≈ 1.6 independientes (aceptable).
3 del mismo torneo ≈ 2.0 independientes (inaceptable — 33% de correlación perdida).

### Guard 3: Discipline Guard (F-25-3)

```python
def discipline_check(pick, trader_plans):
    """Solo picks del trader plan (APOSTAR/WATCHLIST) entran a combos."""
    all_trader_picks = set()
    for plan in trader_plans:
        for p in plan.get('picks_apostar', []) + plan.get('picks_watchlist', []):
            all_trader_picks.add(p['jugador'])
    return pick['jugador'] in all_trader_picks
```

**Regla:** El combo builder NUNCA usa picks que no están en algún trader_plan de las últimas 24h.
Flag `--allow-extra` para override explícito (disabled by default).

### Guard 4: Duplicate Combo Guard (F-25-4)

```python
def is_duplicate(new_combo, existing_combos):
    """Detecta combos con picks idénticos (orden irrelevante)."""
    new_set = frozenset(p['jugador'] for p in new_combo)
    return any(frozenset(p['jugador'] for p in c) == new_set for c in existing_combos)
```

**Regla:** Combo duplicado → rechazado automáticamente con warning.

---

## Safe Combos — Beta Book

### Concepto

Combos de **2 piernas** con la mayor probabilidad de ganar ambas.
Cuota baja (3.0-12.0), P(ambos ganan) > 25%.
Objetivo: retorno consistente, no jackpot.

### Fórmulas

```python
def build_safe_combos(picks_pool, top_n=8, min_p_both=0.25, max_cuota=12.0):
    """Genera combos 2-piernas priorizando P(ambos ganan)."""
    safe_combos = []
    
    for i, p1 in enumerate(picks_pool):
        for p2 in picks_pool[i+1:]:
            # Guard 2: distintos torneos
            if p1['torneo'] == p2['torneo']:
                continue
            
            # Calcular P(ambos ganan)
            p_both = p1['p_blend'] * p2['p_blend']
            cuota_combo = p1['cuota'] * p2['cuota']
            
            if p_both < min_p_both or cuota_combo > max_cuota:
                continue
            
            # Scoring: P(both) domina, cuota es tiebreaker
            # gap_penalty reduce picks CALIBRATION_DRIVEN
            gap_penalty = min(1.0, 1.15 - max(p1.get('gap', 0), p2.get('gap', 0)))
            score = p_both + 0.01 * math.log(cuota_combo)
            score *= gap_penalty
            
            safe_combos.append({
                'picks': [p1, p2],
                'p_both': p_both,
                'cuota': cuota_combo,
                'score': score,
                'gap_penalty': gap_penalty
            })
    
    # Top N por score (P dominante)
    safe_combos.sort(key=lambda x: x['score'], reverse=True)
    return safe_combos[:top_n]
```

### Safe Combo Scoring — P Domina

```
safe_score = P(both) + 0.01 × log(cuota_combo)
safe_score × gap_penalty
```

P(both) en rango [0.25, 0.50] domina totalmente.
`0.01 × log(cuota)` en rango [0.011, 0.025] → solo desempata.

### Display

```
🛡️ SAFE COMBOS — Beta Book (2 piernas, P>25%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Safe #1 — cuota @4.22 | P(ambos)=34.8% | gap_penalty=1.00
  ⚓ Seggerman @2.55  (Challenger Parma)   p=0.590  gap=0.081  MARKET
  ⚓ Martinez  @2.14  (ITF Martos)         p=0.590  gap=0.060  MARKET
  → $1,000 × @4.22 = $4,220 si ambos ganan

Safe #2 — cuota @5.46 | P(ambos)=33.5% | gap_penalty=1.00  
  ⚓ Seggerman @2.55  (Challenger Parma)   p=0.590  gap=0.081  MARKET
  ⚓ Bocchi    @3.35  (Challenger Parma)   p=0.569  gap=0.074  MARKET
  ⚠️ mismo torneo — RECHAZADO por Guard 2
```

---

## CLI Integration

```bash
# Safe combos solamente:
python3 betplay_combo_builder.py --safe --safe-stake 1000 --safe-top-n 8

# Live + safe (combos normales + safe combos):
python3 betplay_combo_builder.py --live --safe --telegram

# Live + mega + safe (todo el portafolio):
python3 betplay_combo_builder.py --live --mega --safe --telegram

# Con guards deshabilitados (para testing):
python3 betplay_combo_builder.py --live --no-dispersion-guard --allow-extra
```

### Two-Speed Portfolio

```
┌─────────────────────────────────────────────────┐
│  ALPHA BOOK (mega-combos)                       │
│  • 6-10 piernas cross-tier                      │
│  • Cuota @100-@1000+                            │
│  • Stake fijo $500/combo                        │
│  • Objetivo: +$50k en días épicos               │
│  • Frecuencia: 1-2 días/semana pagan            │
├─────────────────────────────────────────────────┤
│  NORMAL BOOK (combos por tier)                  │
│  • 3-4 piernas mismo tier                       │
│  • Cuota @10-@50                                │
│  • Stake Kelly-VaR ajustado                     │
│  • Objetivo: retorno base del pipeline          │
│  • Frecuencia: mayoría de sesiones              │
├─────────────────────────────────────────────────┤
│  BETA BOOK (safe combos)                        │
│  • 2 piernas cross-tournament                   │
│  • Cuota @3-@12                                 │
│  • Stake fijo configurable                      │
│  • Objetivo: +$15k consistente en buenos días   │
│  • Frecuencia: cada sesión con accuracy >55%    │
└─────────────────────────────────────────────────┘
```

---

## Retroactive Analysis — 14-Jun con Guards

### Guard 1 (Dispersion): std(p_blend) para pool Parma = 0.000 → BLIND
→ **BLOQUEA mega-combos con pool Parma.** Combos #1-#19 (que dependen de Milev/Bocchi/Seggerman) → bloqueados o reducidos a máx 3 piernas.
→ Milev en 14 combos → nunca hubiera ocurrido.

### Guard 2 (Concentration): Combos #12-#15 tienen 3/4 picks Parma → RECHAZADOS
→ 4 combos eliminados directamente.

### Guard 3 (Discipline): Cerny, Cherubini, Vedder, Ribero, Damm, Romano@1.35 → RECHAZADOS
→ 6 picks eliminados del pool. Combos #5-#9 (que los incluyen) → no se generan.

### Guard 4 (Duplicate): Combo #10 = #11 → #11 RECHAZADO
→ $500 ahorrados en combo duplicado.

### Impacto estimado con todos los guards:
- Combos generados: ~8-10 (vs 25 sin guards)
- Inversión: ~$4,000-$5,000 (vs $12,500)
- Pérdida: ~-$4,000-$5,000 (vs -$12,500)
- **Ahorro: ~$7,500-$8,500** — no salva la sesión pero reduce pérdida al 40%

### Honestidad: con 36% accuracy (4/12), NINGÚN guard salva la sesión
Los guards reducen la MAGNITUD del daño, no lo eliminan.
La protección real es: accuracy histórica 66.1% → 1 día bueno (+$50k+) paga 4-6 días malos (-$5-10k).

---

## Restricciones

- **R-25-1:** Guards son PRE-filtros — se aplican ANTES de generar combos, no después.
- **R-25-2:** Dispersion Guard evalúa el pool COMPLETO, no combo por combo.
- **R-25-3:** Tournament Concentration = hard limit (max 2). NO configurable para más.
- **R-25-4:** Discipline Guard ON por default. `--allow-extra` para override.
- **R-25-5:** Duplicate Guard siempre ON. Sin override.
- **R-25-6:** Safe combos son ADICIONALES a combos normales y megas. No reemplazan.
- **R-25-7:** Safe combos SIEMPRE disponibles (no dependen de Dispersion Guard).
- **R-25-8:** Safe combo scoring usa `P(both) + 0.01 × log(cuota)` — P domina.
- **R-25-9:** Safe combos requieren picks de torneos distintos (Guard 2 obligatorio).

---

## Tests

```
T25-01: dispersion_index([0.590,0.590,0.590,0.569]) < 0.015 → BLIND
T25-02: dispersion_index([0.590,0.654,0.569,0.535]) ≥ 0.04 → DIFFERENTIATED
T25-03: BLIND pool → mega-combos bloqueados, safe combos permitidos
T25-04: tournament_concentration con 3 picks Parma → rechazado
T25-05: tournament_concentration con 2 picks Parma → aceptado
T25-06: n_effective(4, 3, 0.25) ≈ 2.67
T25-07: discipline_check filtra picks fuera del trader_plan
T25-08: --allow-extra permite picks sin trader_plan
T25-09: is_duplicate detecta combos #10 y #11 como idénticos
T25-10: build_safe_combos genera pares cross-tournament con P>0.25
T25-11: safe scoring prioriza P(both) sobre cuota (Meliss @4.30 NO domina)
T25-12: gap_penalty reduce score de pares con gap>0.12
T25-13: safe combos integrados con --safe en CLI
T25-14: --live --mega --safe genera los 3 libros (Alpha+Normal+Beta)
```

---

## Dependencias

- **Nodo-24 (BBI + gap):** Safe combos usan gap_penalty de Nodo-24. Si Nodo-24 no implementado, gap_penalty=1.0 para todos.
- **Nodo-23 (Mega-combos):** Dispersion Guard aplica a mega-combos de Nodo-23.
- **Nodo-21 (Tiers):** `detectar_tier()` clasifica picks para Guard 2 (torneo).
- **trader_ev_tenis.py:** Discipline Guard lee trader_plans para validar picks.

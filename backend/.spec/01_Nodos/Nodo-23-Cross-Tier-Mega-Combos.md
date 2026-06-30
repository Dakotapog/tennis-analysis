# Nodo-23 — Cross-Tier Mega-Combos

> **Estado:** 🔧 EN IMPLEMENTACIÓN — 2026-06-14
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Origen:** Sesión épica 2026-06-13 — 9/10 picks Challenger+ITF (Miguel perdió), mega-combos 7p+8p pagaron
> **Prioridad:** ALTA — funcionalidad que el usuario ya ejecuta manualmente con resultados probados

---

## Problema — El Pipeline No Genera Combos Cross-Tier

El trader corre **un tier por ejecución** (REGLA: no mezclar GS con ITF en un pool).
El combo builder lee combos del trader → máximo 4 piernas, siempre dentro del mismo tier.

El usuario armó **manualmente** combos de 7-8-9 piernas mezclando Challenger grass (Ilkley, Bratislava) con ITF (Martos, K.Banja, Niza, LA, Decatur, Brasilia).

```
Combo 7p @269.2  → $500 → $52,608  ✅ (4 Challenger + 3 ITF)
Combo 8p @632.6  → $500 → PAGÓ     ✅ (4 Challenger + 4 ITF)
Combo 9p @???    → $500 → PERDIÓ   ❌ (Miguel perdió — 1 de 9)
```

El pipeline NO tiene capacidad de generar estos combos. Esta es la brecha.

---

## Fundamento Científico

### Correlación Cross-Tier ≈ 0.03

```
ρ mismo_torneo      = 0.25  (Ilkley R1 vs Ilkley R1)
ρ mismo_tier_otro_t = 0.10  (Ilkley vs Bratislava)
ρ cross_tier        = 0.03  (Challenger Ilkley vs ITF Niza)
```

Con ρ=0.03, los picks cross-tier son casi independientes.
P(7 ganan) ≈ Π p_i (con ajuste copula mínimo).

### Piernas Ancla + Satélite

```
ANCLA:     cuota < 1.80, p_blend > 0.55 → baja cuota, sube P(todas ganan)
SATÉLITE:  cuota > 2.00, edge > 5%      → sube cuota combo, baja P marginalmente
```

La sesión épica tenía 3 anclas (Bu @1.65, Fearnley @1.68, Martinez @1.50) y 
6-7 satélites (Romano @2.75, Pawlikowska @2.95, Carnicella @2.70, etc.)

### Scoring: P(todas ganan) > EV

Para mega-combos (≥6 piernas), maximizar P(todas ganan) es mejor que maximizar EV.
EV alto con P baja = combo que casi nunca paga.

```
mega_score = P_todas × log(cuota_combo) × cross_tier_bonus
P_todas    = Π p_blend_i × copula_adj(ρ_matrix)
cross_tier_bonus = 1.0 + 0.05 × n_tiers_distintos
```

---

## Diseño

### PASO 4.7 — Cross-Tier Mega-Combo Builder

**Dónde:** Nueva función `build_mega_combos()` en `betplay_combo_builder.py`
**Cuándo:** Después de PASO 4 (trader por tier) y antes de PASO 4.5 (combo builder live)
**Flag:** `--mega` en `betplay_combo_builder.py`

### Flujo

```
1. Leer TODOS los trader_plans de las últimas 24h (ya existe en build_live_combos)
2. Extraer pool unificado: picks APOSTAR + watchlist de todos los tiers
3. Clasificar cada pick como ANCLA (cuota<1.80) o SATÉLITE (cuota≥1.80)
4. Generar escalera de mega-combos:
   - 6 piernas: C(N,6) → top 3 por mega_score
   - 7 piernas: C(N,7) → top 3 por mega_score
   - 8 piernas: C(N,8) → top 2 por mega_score
   - 9 piernas: C(N,9) → top 1 por mega_score
   - 10 piernas: C(N,10) → top 1 si existe
5. Filtro: cada mega-combo debe tener ≥1 ancla y ≥2 tiers distintos
6. Stake fijo por combo (configurable, default $500)
7. Mapear a Kambi → generar .bat como los combos normales
```

### Restricciones

- **R-23-1:** Mega-combos NO reemplazan los combos por tier. Son ADICIONALES.
- **R-23-2:** Stake fijo (no Kelly). Kelly no aplica a combos de 7+ piernas — son apuestas de alta varianza.
- **R-23-3:** Máximo 10 mega-combos por sesión (budget = 10 × stake_mega).
- **R-23-4:** ≥2 tiers distintos obligatorio (si no, es un combo normal, no mega).
- **R-23-5:** ≥1 pierna ancla (cuota<1.80) obligatorio para mantener P(todas) razonable.

---

## Tests

```
T23-01: build_mega_combos con 4 challenger + 6 ITF picks → genera combos 6-9p
T23-02: mega-combo sin ancla → rechazado (R-23-5)
T23-03: mega-combo con 1 solo tier → rechazado (R-23-4)
T23-04: mega_score prioriza P(todas ganan) sobre EV
T23-05: --mega flag en CLI funciona y genera .bat adicionales
```

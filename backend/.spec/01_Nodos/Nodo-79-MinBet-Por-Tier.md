# Nodo-79 — MIN_BET proporcional por tier (modo sombra)

**Fecha:** 2026-07-10
**Estado:** BORRADOR — modo sombra activo desde 2026-07-10
**Rama:** main

---

## Problema

`MIN_BET=$1,000` fijo en `trader_ev_tenis.py:43` elimina sistemáticamente picks ITF
válidos porque el bankroll ITF ($10,000) hace que MIN_BET represente el 10% del bankroll,
un umbral prohibitivo que no ocurre en GS ($125k → MIN_BET=0.8%).

### Traza del waterfall combinado cuando VaR está excedido

```
multiplicador_efectivo = var_factor(0.25) × cppi_factor(0.60) = 0.15

Umbral de supervivencia: stake_pre ≥ $6,667 para sobrevivir MIN_BET_CLIFF con ITF
kelly_kl necesario ITF:  6,667/10,000 = 66.7% — imposible con λ_ITF=4.5
kelly_kl necesario GS:   6,667/125,000 = 5.3%  — completamente razonable
```

### Casos reales documentados (PROPUESTA_VAR_2026-07-10.md)

| Pick             | Tier | Edge  | Cuota | kelly_kl | stake_shadow | Resultado |
|------------------|------|-------|-------|----------|-------------|-----------|
| Leyton Rivera    | itf  | 39.7% | 4.35  | 0.4857   | $728→$0     | WON       |
| Maria Sara Popa  | itf  | 6.4%  | 2.04  | 0.0843   | $126→$0     | WON       |
| Aziz Ouakaa      | itf  | 7.0%  | 1.74  | 0.1012   | —           | WON       |

8 picks WATCHLIST-VAR totales (julio 3–9): 3 WON, 5 LOST, hit%=37.5% > breakeven ITF.

### Relación con H54-01

La hipótesis H54-01 (pre-registrada 2026-07-03, n_stop=30) mide exactamente este segmento.
Con n=8/30, la evidencia es positiva pero estadísticamente no concluyente todavía.

---

## Solución: Opción A — MIN_BET proporcional por tier

```python
# trader_ev_tenis.py — reemplazar MIN_BET fijo por dict por tier
_MIN_BET_BY_TIER = {
    'itf':        100,   # 1% de bankroll $10k
    'challenger': 200,   # 1% de bankroll $20k
    'atp500':     500,
    'atp1000':    750,
    'grand_slam': 1000,
}
```

**Riesgo controlado:** picks ITF con edge<5% que hoy se eliminan podrían pasar.
Controlado por: (a) modo sombra durante H54-01, (b) gates de edge/confianza existentes.

---

## Implementación: modo sombra (esta fase)

- `stake_final` REAL: sigue usando `MIN_BET=1000`. **No cambia. No se apuesta según sombra.**
- Campos nuevos en `_waterfall` dict (se persisten en shadow_book via `update_trader_stakes`):
  - `stake_final_shadow`: stake que habría resultado con MIN_BET por tier
  - `min_bet_shadow_usado`: el MIN_BET del dict para ese tier
  - `shadow_survives_cliff`: True si pick habría sobrevivido con MIN_BET por tier

---

## Condición de graduación (propuesta)

| Condición | Valor |
|-----------|-------|
| Gate primario | H54-01: n_actual ≥ 30 con hit% flattened ≥ hit% financiado |
| Gate secundario | IC95 de hit% flattened no cruza cero por debajo del breakeven tier |
| Sesión requerida | Sesión explícita de recalibración antes de activar en real |
| Plazo estimado | ~30 días al ritmo actual (~1 pick/día ITF settled) |

**Prohibición:** `MIN_BET` real no se modifica hasta que H54-01 gradúe. Cualquier cambio
antes requiere anotación en DECISION-LOG con justificación explícita.

---

## Archivos afectados

- `trader_ev_tenis.py`: +`_MIN_BET_BY_TIER` dict, +shadow calc en waterfall loop
- `tests/test_nodo79_minbet_shadow.py`: casos Leyton Rivera + Popa
- `validation/preregistered_hypotheses.json`: H54-01 n_actual backfill

## Vinculación

- H54-01 — hipótesis pre-registrada que gatekea esta decisión
- D54-01 — origen de la deuda documentada en DECISION-LOG
- Nodo-70 — CPPI (cppi_factor PROVISIONAL que amplifica el problema)
- PROPUESTA_VAR_2026-07-10.md — diagnóstico completo

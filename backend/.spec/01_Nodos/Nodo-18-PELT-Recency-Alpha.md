# Nodo-18 — PELT Recency Alpha

> **Estado:** ✅ COMPLETADO — 2026-06-03
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-19-H2H-Immunity-Dampener]] | [[Nodo-20-PageRank-Erdos-Quality]]
> **Tests:** 931 passed (19 tests nuevos T18-05)
> **Origen:** Test-Time Compute — 4 marcos expertos, sesión cerrada inesperadamente 2026-06-03
> **Orden de implementación recomendado:** 2 de 3 (después de Nodo-19)

---

## Problema — Dato Ignorado Silenciosamente

`edge_calculator.py` lee el estado Markov del favorito así:

```python
# edge_calculator.py — campo markov_favorito actual:
'markov_favorito': partido.get('markov_analysis', {}).get('estado_actual')
#                                                          ↑ solo lee HOT/COLD/NEUTRAL
#                   pero en el JSON ya existe:
#   "change_point": 6   ← COMPLETAMENTE IGNORADO
```

`change_point` es un índice absoluto. Con `len(historial)` y `change_point`:

```
recencia = len(historial) - change_point

Ejemplo A: len=45, change_point=6  → recencia=39 → cambio hace 39 partidos → BOOKMAKER YA REPRICED
Ejemplo B: len=45, change_point=42 → recencia=3  → cambio hace 3 partidos  → BOOKMAKER NO HA REPRICED ← ALPHA TEMPORAL
```

**El caso Kostyuk @3.00 (2026-06-01):** el sistema predijo HOT pero apostó el mismo λ que si llevara HOT 40 partidos. El alpha temporal no se capturó.

---

## Fundamento — Marco 1: Senior Quant (HFT + Options)

### Decaimiento de Señal — Analogía al Mercado Financiero

En HFT, una señal tiene vida media. El alpha decae exponencialmente:

```
α(t) = α₀ × exp(-λ_decay × t)
```

El bookmaker es un market-maker. Cuando un jugador tiene un cambio de régimen (PELT), el bookmaker tarda **2-7 días** (1-3 partidos) en repricing. Pasado ese tiempo, el alpha temporal se ha absorbido en las cuotas.

### Implementación — PELT Freshness Factor

Los datos ya existen en el JSON de salida de `markov_analyzer.py`:

```python
# markov_analyzer.py ya retorna:
'change_point': 6           # índice absoluto donde cambió el régimen
'win_rate_reciente': 0.80   # win rate DESPUÉS del cambio
'win_rate_anterior': 0.60   # win rate ANTES del cambio

# Lo que necesitamos calcular (T18-01):
recencia = len(historial) - change_point
delta_wr = win_rate_reciente - win_rate_anterior  # magnitud del cambio (T18-C5)
```

### Lógica del Factor Alpha Temporal (T18-02)

```python
def factor_alpha_temporal(recencia: int, estado: str, delta_wr: float) -> float:
    """
    Multiplica λ_efectivo por (1/factor), reduciendo la penalización KL
    cuando el cambio de régimen es fresco (bookmaker todavía tiene precio stale).
    """
    if estado == 'HOT' and recencia <= 3:
        return 1.20   # máximo alpha — bookmaker tiene precio viejo, apuesta más confiada
    elif estado == 'HOT' and recencia <= 7:
        return 1.10   # alpha parcial
    elif estado == 'COLD' and recencia <= 3:
        return 0.85   # COLD fresco = señal de precaución amplificada
    else:
        return 1.00   # stale — bookmaker ya repriced, sin bonus/penalidad

# Aplicar en edge_calculator.py:
# λ_efectivo = λ_tier × (1 / factor_alpha_temporal(recencia, estado, delta_wr))
# Resultado: λ más bajo cuando señal HOT es FRESCA → apuesta más confiada
```

### C5 — Magnitud Delta Markov (bonus de bajo costo)

```python
# win_rate_reciente y win_rate_anterior ya existen en markov_result
delta_wr = win_rate_reciente - win_rate_anterior

# Un cambio de 0.20 (60%→80%) es más informativo que 0.10 (65%→75%).
# Bookmaker no modela la magnitud — solo el estado final.
# Añadir como campo informativo en edge report sin modificar λ directamente.
```

---

## Conexión con el Pipeline Existente

| Componente | Rol en Nodo-18 | Estado |
|---|---|---|
| `markov_analyzer.py` | Produce `change_point` + `win_rate_reciente/anterior` | ✅ ya existe en output |
| `edge_calculator.py` | Consume `change_point` — actualmente IGNORADO | 🔴 fix requerido |
| `analysis/rivalry_analyzer.py` | `markov_analysis` propagado al edge_report | ✅ ya fluye |
| `data/calibracion_edge.json` | No se modifica — solo λ_efectivo cambia | — |

---

## Dependencias

| Prerequisito | Estado | Nodo |
|---|---|---|
| `markov_analyzer.py` activo + produciendo `change_point` | ✅ ACTIVO | [[Nodo-02-Markov-Changepoint]] |
| λ por tier implementado en `edge_calculator.py` | ✅ ACTIVO | [[Nodo-17-Calibracion-Por-Tier]] T17-03 |
| `Nodo-19` H2H Immunity implementado | 🟡 RECOMENDADO PRIMERO | [[Nodo-19-H2H-Immunity-Dampener]] |

---

## Tasks

| ID | Descripción | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T18-01 | `calcular_recencia_regimen(markov_result, n_total_partidos)` → `{'recencia': int, 'freshness': 'FRESCO'\|'RECIENTE'\|'ESTABLE'}` | `analysis/markov_analyzer.py` | 🟠 ALTO | ✅ COMPLETADO |
| T18-02 | `factor_alpha_temporal(recencia, estado, delta_wr)` → float multiplicador de λ inverso | `analysis/markov_analyzer.py` | 🟠 ALTO | ✅ COMPLETADO |
| T18-03 | Integrar en `edge_calculator.py`: `λ_efectivo × (1 / factor_alpha_temporal)` cuando favorito es FRESCO | `edge_calculator.py` | 🟠 ALTO | ✅ COMPLETADO |
| T18-04 | Campo `recencia_regimen` y `alpha_temporal` en output del edge report | `edge_calculator.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T18-C5 | Campo `delta_wr` (magnitud cambio de régimen) en edge report — informativo, sin modificar λ | `edge_calculator.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T18-05 | Tests: `calcular_recencia_regimen` + `factor_alpha_temporal` + integración edge_calculator | `tests/` | — | ✅ COMPLETADO — 19 tests |

**Estimación:** ~20 líneas de lógica nueva. Tests: ~15 casos.

---

## Reglas Nuevas

**REGLA-T18-1: PELT Freshness — alpha temporal existe solo en ventana corta**
```
HOT con recencia ≤ 3 partidos → bookmaker todavía tiene precio stale → λ reducido 20%
HOT con recencia > 7 partidos → bookmaker ya repriced → λ sin cambio
COLD con recencia ≤ 3 partidos → precaución amplificada → λ aumentado 15%
```

**REGLA-T18-2: change_point NUNCA debe ignorarse**
```
Si change_point está en el JSON de markov_analyzer → SIEMPRE calcular recencia.
El estado HOT/COLD sin recencia = señal incompleta.
```

**REGLA-T18-3: Delta Markov es diagnóstico, no decisorio**
```
delta_wr = win_rate_reciente - win_rate_anterior
→ Añadir como campo informativo en edge report.
→ NO usar directamente en cálculo de λ hasta tener n≥30 por categoría de cambio.
```

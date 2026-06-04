# Nodo-19 — H2H Immunity Dampener

> **Estado:** ✅ COMPLETADO — 2026-06-03
> **Wikilinks:** [[MOC-Principal]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-14-Validacion-Live-Conexiones]] | [[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-20-PageRank-Erdos-Quality]]
> **Tests:** 912 passed (8 tests nuevos T19-04)
> **Origen:** Test-Time Compute — 4 marcos expertos, sesión cerrada inesperadamente 2026-06-03
> **Orden de implementación recomendado:** 1 de 3 (primero — previene error activo)

---

## Problema — Señal de 2do Orden Ignorada

El sistema aplica `factor_markov` globalmente: si el favorito está HOT, el factor se amplifica contra **todos** sus rivales por igual.

```python
# rivalry_analyzer.py — comportamiento actual:
# Si Djokovic HOT → factor_markov = 1.15 contra CUALQUIER rival
# Pero analyze_direct_h2h() ya retorna:
p1_score = 1.0   # jugador1 ganó 1 de los últimos H2H directos
p2_score = 3.0   # jugador2 ganó 3
# h2h_win_rate_p1 = 1/(1+3) = 0.25  ← jugador1 HOT pero históricamente pierde aquí
# Este cruce NUNCA ocurre.
```

**Dato ignorado silenciosamente:**

```python
# rivalry_analyzer.py line ~519: analyze_direct_h2h() retorna:
p1_score = 1.0   # jugador1 ganó 1 de los últimos H2H directos
p2_score = 3.0   # jugador2 ganó 3
# → h2h_win_rate_favorito = 0.25
# → Pero esto NUNCA se cruza con el estado Markov del jugador
```

---

## Fundamento — Marco 2: Sports Analytics Expert (Tennis-Specific)

> "Hay jugadores que simplemente no saben jugar contra ciertos rivales, sin importar su forma actual. Djokovic puede estar en racha de 10 victorias pero contra Nadal en clay sigue perdiendo."

La mayoría de modelos tratan "forma reciente" y "H2H histórico" como señales **independientes**. La interacción entre ellas (forma × específico rival) es una señal de 2do orden que requiere razonamiento cruzado — que el modelo actual no hace.

### H2H Immunity Dampener — Lógica

```python
def calcular_h2h_immunity(direct_h2h_matches: list, favored: str, opponent: str) -> dict:
    """
    Calcula si el favorito tiene un patrón de pérdida histórica contra este rival específico,
    incluso cuando está en estado HOT.

    Retorna: {'h2h_win_rate': float, 'immunity_factor': float, 'n_h2h': int}
    """
    wins_fav = sum(1 for m in direct_h2h_matches if m.get('winner') == favored)
    total = len(direct_h2h_matches)
    h2h_win_rate = wins_fav / max(total, 1)

    if estado_fav == 'HOT':
        if h2h_win_rate < 0.30:    # HOT pero pierde históricamente a ESTE rival
            immunity_factor = 0.85   # reducir confianza — señal de 2do orden negativa
        elif h2h_win_rate > 0.70:   # HOT Y domina históricamente
            immunity_factor = 1.12   # doble confirmación → mayor confianza
        else:
            immunity_factor = 1.00   # H2H neutro — no modifica
    else:
        immunity_factor = 1.00       # solo actúa cuando estado es HOT

    return {'h2h_win_rate': round(h2h_win_rate, 3), 'immunity_factor': immunity_factor, 'n_h2h': total}
```

### Integración en `generate_advanced_prediction()` (T19-02)

```python
# Antes de aplicar factor_markov al score final:
immunity = calcular_h2h_immunity(direct_h2h_matches, favored, opponent)
factor_markov_efectivo = factor_markov * immunity['immunity_factor']

# Añadir en score_breakdown:
'h2h_immunity_factor': immunity['immunity_factor'],
'h2h_win_rate_vs_rival': immunity['h2h_win_rate'],
'n_h2h_directo': immunity['n_h2h']
```

---

## Por qué ningún modelo actual lo tiene

La mayoría de sistemas calculan:
- Señal A = forma reciente (Markov HOT/COLD)
- Señal B = H2H histórico (win rate global)

Como señales **independientes** que se suman linealmente.

La interacción `estado_HOT × h2h_desfavorable_vs_rival_específico` requiere razonamiento cruzado de 2do orden. Bookmaker tampoco lo modela sistemáticamente — actualiza cuotas por forma reciente general, no por el patrón específico contra este rival.

---

## Caso de Validación — Djokovic vs Nadal (clay)

```
Djokovic estado: HOT (última racha 8-2)
Djokovic vs Nadal clay H2H: 2-14 (h2h_win_rate = 0.125)

Sin Nodo-19: factor_markov = 1.15 → modelo sobreestima a Djokovic
Con Nodo-19: factor_markov_efectivo = 1.15 × 0.85 = 0.978 → predicción corregida
```

Validar con casos reales disponibles en `reports/h2h_results_enhanced_*.json`.

---

## Conexión con el Pipeline Existente

| Componente | Rol en Nodo-19 | Estado |
|---|---|---|
| `rivalry_analyzer.py` → `analyze_direct_h2h()` | Produce `p1_score`, `p2_score` — ya disponible | ✅ existe |
| `rivalry_analyzer.py` → `generate_advanced_prediction()` | Punto de integración de `immunity_factor` | 🔴 modificar |
| `analysis/markov_analyzer.py` | Produce `estado_actual` del favorito | ✅ activo |
| `edge_calculator.py` | Recibe `h2h_immunity_factor` en `score_breakdown` | 🟡 consumir campo nuevo |

---

## Dependencias

| Prerequisito | Estado | Nodo |
|---|---|---|
| `analyze_direct_h2h()` en `rivalry_analyzer.py` | ✅ ACTIVO — retorna p1_score/p2_score | [[Nodo-06-Erdos-Graph]] |
| Markov `estado_actual` disponible en prediction | ✅ ACTIVO | [[Nodo-02-Markov-Changepoint]] |
| Sin prerequisitos de otros Nodos nuevos | ✅ INDEPENDIENTE | — |

---

## Tasks

| ID | Descripción | Archivo | Impacto P&L | Estado |
|---|---|---|---|---|
| T19-01 | `calcular_h2h_immunity(direct_h2h_matches, favored, opponent)` → `{'h2h_win_rate': float, 'immunity_factor': float, 'n_h2h': int}` | `analysis/rivalry_analyzer.py` | 🔴 ALTO | ✅ COMPLETADO |
| T19-02 | Integrar en `generate_advanced_prediction()`: antes de aplicar `factor_markov`, multiplicar por `immunity_factor` | `analysis/rivalry_analyzer.py` | 🔴 ALTO | ✅ COMPLETADO |
| T19-03 | Campo `h2h_immunity_factor` en `score_breakdown` y en output del edge report | `analysis/rivalry_analyzer.py` + `edge_calculator.py` | 🟡 MEDIO | ✅ COMPLETADO |
| T19-04 | Tests + validación con casos reales (patrón Djokovic vs Nadal clay) | `tests/` | — | ✅ COMPLETADO — 8 tests |

**Estimación:** ~30 líneas de lógica nueva. Tests: ~12 casos (incluyendo n_h2h=0, HOT+dominante, HOT+inmune, COLD).

---

## Reglas Nuevas

**REGLA-T19-1: H2H Immunity se activa solo cuando estado es HOT**
```
Si estado_favorito == HOT y h2h_win_rate < 0.30 → immunity_factor = 0.85
Si estado_favorito == HOT y h2h_win_rate > 0.70 → immunity_factor = 1.12
Otros casos → immunity_factor = 1.00 (sin modificación)
COLD no se amplifica por H2H dominante — evitar señales conflictivas.
```

**REGLA-T19-2: n_h2h mínimo para aplicar immunity**
```
Si n_h2h < 3 → immunity_factor = 1.00 (muestra insuficiente)
Con n_h2h ≥ 3 → aplicar factor según h2h_win_rate
Con n_h2h ≥ 6 → señal más confiable — podría aumentarse el rango de los factores en el futuro
```

**REGLA-T19-3: Implementar antes de Nodo-18 y Nodo-20**
```
Nodo-19 PREVIENE un error activo (sobreconfianza en HOT contra rival inmune).
Nodo-18 AMPLIFICA una señal. Nodo-20 REFINA Erdős.
Orden: prevenir > amplificar > refinar.
```

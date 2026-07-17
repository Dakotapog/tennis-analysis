# Nodo-56 — Bugs de Display en generar_tabla_favoritos2.py

> **Wikilinks:** [[Nodo-55-Respuesta-Fable-Funnel-Deploy]] | [[Nodo-54-Brief-Fable-Funnel-Deploy]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Fecha:** 2026-07-03
> **Estado:** ABIERTO
> **Severidad:** COSMÉTICA para Bugs A/B (scoring correcto) | CONFUSIÓN CRÍTICA para Bug C (penalización oculta)
> **Descubierto en:** revisión de 5 partidos por tier post-Nodo-55 (sesión 2026-07-03)

---

## 0. Síntomas

### Síntoma A — Pesos no suman 100%

La tabla `--- DISTRIBUCIÓN DE PESOS DEL ANÁLISIS ---` muestra sumas incorrectas:

```
# Casos observados (sesión 2026-07-03, 110 partidos):
Wimbledon GS grass (Safiullin, Liu, Sabalenka, Hurkacz):  99.0%   ← ALERTA D53-04
Challenger hard n=0 (Kennedy vs Watanuki, Cary):          103.0%  ← ALERTA D53-04 GRAVE
Challenger clay n=49 (Cecchinato, Kuzmanov, Giustino):    100.0%  ← OK
ITF clay (Maric, Ouakaa):                                 99.0–99.8% ← ALERTA
```

El alert `D53-04` se activa correctamente pero la raíz del bug es en `generar_tabla_favoritos2.py`,
no en `calibracion_edge.json` como indica el mensaje.

### Síntoma B (rounding) — Pérdidas de ±0.2-0.4% por precisión mixta

Ajuste de superficie usa `round(..., 2)` mientras densidad usa `round(..., 4)`.

### Síntoma C — PUNTAJE FINAL ≠ suma de componentes (CONFUSIÓN CRÍTICA)

```
Caso: Meligeni vs Pacheco — Challenger clay, Quito (sesión 2026-07-03)

CONSOLIDADO DE PUNTUACIÓN (tabla display):
| Componente                 | Puntos Felipe | Puntos Rodrigo |
|----------------------------|---------------|----------------|
| Especialización Superficie | 0.47          | 0.41           |
| Forma Reciente             | 1.03          | 1.26           |
| Rivales Comunes            | 0.63          | 0.42           |
| H2H Directo                | 0             | 0              |
| Ranking/Momentum           | 0.64          | 0.67           |
| Rating ELO                 | 0.64          | 0.75           |
| Ventaja Localía            | 0             | 0              |
| Fuerza Calendario          | 0.26          | 0.26           |
| PUNTAJE FINAL TOTAL        | 3.67          | 1.89           |

Suma de componentes Rodrigo = 0.41+1.26+0.42+0+0.67+0.75+0+0.26 = 3.77
PUNTAJE FINAL TOTAL Rodrigo = 1.89
Diferencia = 3.77 - 1.89 = 1.88  ← PENALIZACIÓN DE INACTIVIDAD OCULTA
```

El usuario vio "1.89 ≠ 3.77" y concluyó que la suma estaba rota. La suma sí funciona —
la `Penalizacion_Inactividad` no se muestra en la tabla pero SÍ se descuenta en el PUNTAJE FINAL.

---

## 1. Causa Raíz por Bug

### Bug A — `get_weights_from_reasoning()` ignora shrinkage

`get_weights_from_reasoning()` en `generar_tabla_favoritos2.py:140` reconstruye los pesos
**parseando los logs de reasoning**. El proceso:

1. Lee `LOG_WEIGHTS_STRATEGY` → base de pesos (ej. challenger: CO=0.08, form=0.22)
2. **IGNORA `LOG_SHRINKAGE`** → no aplica James-Stein
3. Lee `LOG_DENSITY` → actualiza CO y form con valores que vienen de CO **shrunkado**

El mismatch: la densidad se aplica en `rivalry_analyzer.py` sobre `CO_shrunk`, pero
`get_weights_from_reasoning` usa `CO_inicial` como punto de partida.

**Caso Kennedy (hard_challenger, n=0):**
```
CO_inicial   (challenger)     = 0.08
CO_shrunk    (n=0 → 100% atp500) = 0.15     ← IGNORADO por get_weights_from_reasoning
density      = 0.3525

# En rivalry_analyzer.py (CORRECTO):
CO_new   = round(0.15 × 0.3525, 4)         = 0.0529
form_add = round(0.15 × 0.6475, 4)         = 0.0971
form_new = round(0.18 + 0.0971, 4)         = 0.2771

# En get_weights_from_reasoning (BUGGY):
CO_display   = 0.0529  (correcto — lo lee del log)
form_display = 0.2771  (correcto — lo lee del log)
# Pero el RESTO (h2h, ranking, elo) viene de LOG_WEIGHTS_STRATEGY (pre-shrink):
h2h_display      = 0.03 (challenger)   vs  0.10 (atp500 shrunk)
ranking_display  = 0.22 (challenger)   vs  0.20 (atp500 shrunk)
elo_display      = 0.15 (challenger)   vs  0.12 (atp500 shrunk)

# Suma display = 0.20+0.2771+0.0529+0.03+0.22+0.15+0.05+0.05 = 1.03 (103%)
# Delta = (CO_shrunk - CO_inicial) × (1 - density) = (0.15 - 0.08) × 0.6475 ≈ +0.03
```

### Bug B — Precisión mixta en ajuste de superficie

El ajuste de superficie en `rivalry_analyzer.py:1442-1454` usa `round(..., 2)` (2 decimales)
mientras que densidad usa `round(..., 4)`. Mezclar precisiones introduce errores de redondeo:

```python
# Grass (rivalry_analyzer.py:1449-1450):
weights['common_opponents'] = round(weights['common_opponents'] - 0.05, 2)  # 2 dec
weights['form_recent']      = round(weights['form_recent']       + 0.05, 2)  # 2 dec

# Ejemplo Hurkacz: CO=0.1527 → round(0.1527-0.05, 2)=0.10  (pierde 0.0027)
#                  form=0.1816 → round(0.1816+0.05, 2)=0.23  (pierde 0.0016)
# Neto: -0.0043 por paso de superficie
```

### Bug C — `generar_resumen_consolidado()` oculta la penalización de inactividad

En `rivalry_analyzer.py:apply_weights_and_penalties()`:
```python
def apply_weights_and_penalties(normalized_scores, weights, days_since):
    weighted_scores = {k: normalized_scores[k] * weights[k] for k in weights}
    penalty = 0
    if days_since == -1: penalty = sum(weighted_scores.values()) * 0.3
    elif days_since > 30: penalty = min(sum(weighted_scores.values()) * 0.5, (days_since - 30) * 0.1)
    final_score = sum(weighted_scores.values()) - penalty
    return final_score, weighted_scores, penalty
```

En `get_breakdown()` (rivalry_analyzer.py:1907), el breakdown SÍ incluye:
```python
breakdown['Penalizacion_Inactividad'] = f"{-penalty:.2f} pts"
breakdown['Puntaje_Final'] = f"{final_score:.2f}"
```

Pero en `generar_resumen_consolidado()` (generar_tabla_favoritos2.py:597), `valid_keys` es:
```python
valid_keys = ['surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct',
              'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule']
```

`Penalizacion_Inactividad` NO está en `valid_keys` → la penalización se aplica al PUNTAJE FINAL
pero NO se muestra como fila en la tabla. El usuario ve:
- Suma de componentes = 3.77
- PUNTAJE FINAL TOTAL = 1.89 (= 3.77 - 1.88 penalización)
- Diferencia aparente = 1.88 ← parecé un error de suma cuando en realidad es inactividad

**Caso Meligeni vs Pacheco:** Pacheco último partido >30 días → penalty = min(3.77×0.5, (días-30)×0.1) ≈ 1.88.
La penalización de inactividad es CORRECTA. El modelo SÍ favorece correctamente a Meligeni por inactividad de Pacheco. Solo el display es engañoso.

---

## 2. Impacto

| Qué | Estado |
|---|---|
| Predicciones (p_modelo, confianza) | **CORRECTO** — `rivalry_analyzer.py` usa su propio `weights` dict |
| Kelly-KL, stakes, edge | **CORRECTO** — no usan `get_weights_from_reasoning` |
| Tabla de pesos (Síntoma A/B) | **INCORRECTO** — cosmético |
| Alerta D53-04 | **FALSA ALARMA** para A/B (el scoring es correcto) |
| PUNTAJE FINAL vs suma componentes (Síntoma C) | **ENGAÑOSO** — penalización oculta confunde al usuario |
| Decisión de Meligeni sobre Pacheco | **CORRECTO** — Pacheco inactivo >30 días justifica penalización |

---

## 3. Solución Propuesta

### Fix A+B (Pesos — Recomendado: fuente única de verdad)

`rivalry_analyzer.py` ya tiene el `weights` dict correcto al final de `analyze_prediction`.
Retornarlo directamente en el resultado del partido.

En `rivalry_analyzer.py` → añadir `_weights_final` al dict de predicción:
```python
# Al final de analyze_prediction, dentro del return dict:
'_weights_final': dict(weights),  # snapshot post-todas-las-modificaciones
```

En `generar_tabla_favoritos2.py` → usar `_weights_final` si está disponible:
```python
def get_weights_for_display(partido):
    pred = partido.get('ranking_analysis', {}).get('prediction', {})
    if '_weights_final' in pred:
        return pred['_weights_final']   # fuente de verdad
    return get_weights_from_reasoning(pred.get('reasoning', []))  # fallback
```

### Fix B adicional — Precisión de superficie

```python
# rivalry_analyzer.py:1442-1454 — cambiar round(..., 2) → round(..., 4):
weights['common_opponents'] = round(weights['common_opponents'] + 0.08, 4)
weights['ranking_momentum'] = round(weights['ranking_momentum'] - 0.08, 4)
# (y todos los ajustes de superficie análogos)
```

### Fix C (Penalización oculta — Crítico para UX)

En `generar_resumen_consolidado()` (generar_tabla_favoritos2.py:597), añadir la fila de
penalización cuando no es cero:

```python
# Después de construir summary_data, antes de añadir la fila PUNTAJE FINAL TOTAL:
p1_penalty_raw = (p1_breakdown.get('Penalizacion_Inactividad') or '0.00 pts')
p2_penalty_raw = (p2_breakdown.get('Penalizacion_Inactividad') or '0.00 pts')

def _parse_penalty(s):
    try: return float(str(s).replace(' pts', ''))
    except: return 0.0

p1_penalty = _parse_penalty(p1_penalty_raw)
p2_penalty = _parse_penalty(p2_penalty_raw)

if p1_penalty != 0.0 or p2_penalty != 0.0:
    summary_data.append({
        'Componente': 'Penalizacion Inactividad',
        f'Puntos {p1_name}': f"{p1_penalty:.4f}",
        f'Puntos {p2_name}': f"{p2_penalty:.4f}"
    })
```

Con esto la tabla muestra:
```
| Penalizacion Inactividad | 0.0000 | -1.8800 |
| PUNTAJE FINAL TOTAL      | 3.6700 |  1.8900 |
```
Y la suma de filas = PUNTAJE FINAL TOTAL. El usuario puede ver la razón del descuento.

### Fix D — Mensaje de alerta D53-04

```python
# generar_tabla_favoritos2.py — corregir el mensaje de alerta D53-04:
# ANTES: "Revisar calibracion_edge.json"
# DESPUÉS: "Los pesos se reconstruyen desde logs — ver Nodo-56 para fix definitivo"
```

---

## 4. Deudas

| Deuda | Descripción | Prioridad |
|---|---|---|
| D56-01 | Implementar Fix A: retornar `_weights_final` desde `rivalry_analyzer.py` | MEDIA |
| D56-02 | Actualizar `get_weights_for_display` en `generar_tabla_favoritos2.py` | MEDIA |
| D56-03 | Corregir mensaje de alerta D53-04 | BAJA |
| D56-04 | Corregir `round(..., 2)` → `round(..., 4)` en ajuste de superficie (Bug B) | BAJA |
| D56-05 | Mostrar fila `Penalizacion Inactividad` en `generar_resumen_consolidado` (Bug C) | ALTA — confunde al usuario |

---

## 5. Tests de Validación

Siguiendo REGLA-T53: los tests invocan funciones del módulo real, no hardcodean fórmulas.

- **T56-01:** Para un partido `hard_challenger` con `n=0` en calibración, los pesos del display
  suman `100% ± 0.5%`. (Caso Kennedy: actualmente 103%)
- **T56-02:** Para un partido GS grass con campeón de torneo, los pesos del display suman
  `100% ± 0.5%`. (Caso Safiullin/Liu: actualmente 99%)
- **T56-03:** Para un partido clay challenger con `n≥20`, los pesos del display suman
  `100% ± 0.5%`. (Caso Cecchinato: actualmente 100% — regresión guard)
- **T56-04:** `_weights_final` retornado por `analyze_prediction` suma `1.0 ± 0.001`.
- **T56-05:** Cuando `days_since > 30` para un jugador, `generar_resumen_consolidado` incluye
  fila `Penalizacion Inactividad` con valor negativo para ese jugador, y
  `sum(component_rows) + penalty_row ≈ PUNTAJE FINAL TOTAL ± 0.01`.
- **T56-06:** Cuando `days_since ≤ 30` para ambos jugadores, la fila `Penalizacion Inactividad`
  NO aparece en la tabla (penalty=0 → fila suprimida).

---

## 6. Orden de Implementación para Sonnet

```
1. D56-05 — mostrar Penalizacion Inactividad en generar_resumen_consolidado  [PRIMERO — UX crítico]
2. D56-01 — añadir _weights_final al return de analyze_prediction
3. D56-02 — get_weights_for_display usa _weights_final
4. D56-04 — round(..., 2) → round(..., 4) en ajuste de superficie
5. D56-03 — actualizar mensaje de alerta D53-04
6. Tests T56-01 a T56-06
Baseline: 1598 tests siguen pasando.
PROHIBIDO: modificar la lógica de scoring, kelly, o shrinkage — solo el display.
```

---

## 7. Registro

**Descubierto:** 2026-07-03, revisión post-Nodo-55 de 5 partidos × tier.

**Casos documentados (Síntoma A/B — pesos):**
- Kennedy vs Watanuki (hard_challenger n=0): **103%** ← caso más grave
- Safiullin/Liu/Sabalenka/Hurkacz (GS grass n=15): **99.0%**
- Ouakaa/Maric ITF clay (n=42): **~99.8%**
- Cecchinato/Kuzmanov Challenger clay (n=49): **100.0%** ← OK

**Caso documentado (Síntoma C — penalización oculta):**
- Meligeni vs Pacheco (Challenger clay, Quito): Pacheco componentes=3.77, PUNTAJE FINAL=1.89
  → penalización oculta = -1.88 por inactividad >30 días. Scoring CORRECTO, display ENGAÑOSO.
  El modelo favorece correctamente a Meligeni porque Pacheco lleva >30 días sin jugar.

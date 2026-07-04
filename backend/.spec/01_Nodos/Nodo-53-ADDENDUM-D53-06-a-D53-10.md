# Nodo-53 ADDENDUM — D53-06 a D53-10: Los Bugs que Decidieron el Partido

> **Wikilinks:** [[Nodo-53-Auditoria-Corazon-Prediccion]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-28-Backtest-Limpio]]
> **Fecha:** 2026-07-02
> **Estado:** 📋 ESPECIFICADO — auditoría Fable completa, causa raíz verificada con código y números reales
> **Contexto:** Fable identificó 5 problemas que el Nodo-53 original no recogía. Este addendum los documenta con bloques de código exactos, líneas, y aritmética verificada.

**Advertencia:** D53-06 es el bug más grave que este proyecto ha documentado. No porque rompa el código — el pipeline corre sin errores — sino porque hace que el componente más específico de la superficie valga 1.4% real cuando el modelo cree que vale 15%. Todo lo que el proyecto sabe sobre hierba está siendo ignorado silenciosamente.

---

## D53-06 — CRÍTICO: Normalización con escala inconsistente destruye surface_specialization

### Causa raíz exacta

**Archivo:** `analysis/rivalry_analyzer.py:1813-1837` + `normalization.py:30-39`

El modelo tiene dos rutas de normalización:

```python
# rivalry_analyzer.py línea 1816
_LINEAR_COMPONENTS = {'surface_specialization'}

def normalize_scores(p1_scores, p2_scores):
    for key in p1_scores:
        if key in _LINEAR_COMPONENTS:
            # Ruta LINEAL — surface_specialization
            max_expected = MAX_RAW_SCORES.get(key, 350)          # → 350
            norm = min(raw / max_expected, 1.0) * math.log1p(max_expected)  # → (raw/350) * 5.857
        else:
            # Ruta LOG — todos los demás
            norm = math.log1p(raw)
```

**`normalization.py:32`:**
```python
MAX_RAW_SCORES = {
    'surface_specialization': 350,   # ← cap teórico, nunca alcanzado en práctica
    'form_recent': 300,
    'elo_rating': 250,
    ...
}
```

### Por qué produce escala ~0.18 en lugar de ~5

La fórmula `(raw/350) * log1p(350)` normaliza correctamente **solo si raw puede llegar a 350**.

Pero los valores reales de `surface_specialization` están en el rango 10–50 (scores altos son excepcionales). En el partido Mensik vs Dimitrov:
- Mensik raw=10.89 → `(10.89/350) * 5.857 = 0.1824`
- Dimitrov raw=33.49 → `(33.49/350) * 5.857 = 0.5608`

Mientras que `form_recent` usa `log1p(raw)`:
- Mensik raw=150 → `log1p(150) = 5.017`
- Dimitrov raw=75 → `log1p(75) = 4.331`

**Diferencia de escala: 27x** entre surface y form.

### Impacto aritmético verificado

Con los datos reales del partido (pesos post-grass-adjustment: surface=15%, form=23%, elo=13%):

| Componente | Peso | Norm Mensik | Contrib Mensik | Norm Dimitrov | Contrib Dimitrov |
|---|---|---|---|---|---|
| surface_spec | 15% | **0.1824** | **0.0274 (1.4%)** | **0.5608** | **0.0841 (4.7%)** |
| form_recent | 23% | 5.017 | 1.154 (60.7%) | 4.331 | 0.996 (55.4%) |
| elo_rating | 13% | 5.526 | 0.718 (37.8%) | 5.526 | 0.718 (39.9%) |

**Peso efectivo real de superficie: 1.4% (Mensik) / 4.7% (Dimitrov)** — no el 15% nominal.

Dimitrov tiene **3.07x más calidad en hierba** (raw 33.49 vs 10.89), con 50 partidos en hierba (64% win rate) contra 20 de Mensik (45%). Esta ventaja, que debería ser la señal más fuerte del partido para Wimbledon, contribuyó 4.7% vs 1.4% = diferencia de 0.057 puntos ponderados. La diferencia final entre ambos jugadores fue 0.13 puntos — menos del 50% de la diferencia vino de superficie.

### La razón del comentario en el código (Nodo-28)

```python
# rivalry_analyzer.py:1813
# surface_specialization usa normalización LINEAL porque SkillFactor/VolConf
# (Nodo-28 Fase 1.5) ya controlan la escala. log1p aplasta la señal:
# raw 86 vs 142 → log 4.47 vs 4.97 (ratio 1.11x vs 1.65x real).
```

El comentario asume raw en rango 86–142. En práctica los valores son 10–50. El razonamiento era válido si el rango empírico coincidía con el cap=350, pero no coincide. El resultado es el inverso del objetivo: en lugar de preservar el ratio, la normalización lineal con cap=350 aplasta la contribución absoluta a escala insignificante.

### Fix propuesto

**Opción A — Ajustar cap a rango empírico real (mínimo invasivo):**
```python
# normalization.py línea 32
'surface_specialization': 70,   # max observado empíricamente ~50-70 (no 350)
```
Efecto: Dimitrov raw=33.49 → `(33.49/70) * log1p(70)` = `0.479 * 4.263 = 2.04` — sigue siendo menor que form (4.33) pero la diferencia es 2x, no 27x.

**Opción B — Pasar a log1p como los demás (preserva ratio relativo):**
```python
# rivalry_analyzer.py línea 1816
_LINEAR_COMPONENTS = set()   # vacío — surface usa log1p como todos
```
Efecto: raw 10.89 → `log1p(10.89) = 2.469` | raw 33.49 → `log1p(33.49) = 3.519`. Ratio 1.43x preservado, escala compatible con otros componentes.

**Opción C — Normalización min-max dinámica (más correcta, más compleja):**
Usar el percentil 95 observado en los últimos N partidos como `max_expected`, actualizado periódicamente. Requiere más infraestructura.

**Recomendación para Fable:** Opción B primero (1 línea, test FAIL→PASS claro), luego calibrar con Shadow Book si el impacto en accuracy es el esperado (H53-03).

### Test requerido (T53-06 — debe existir antes del fix)

```python
def test_t53_06_surface_normalizes_to_same_scale_as_form():
    """D53-06: surface_specialization debe normalizar a escala comparable a form_recent.
    Con raw_surface=33.49 y raw_form=75.0, el ratio norm_surface/norm_form debe ser > 0.5.
    Actualmente es 0.5608/4.331 = 0.13 (aplastado 8x bajo lo esperado).
    """
    import math
    raw_surface = 33.49
    raw_form = 75.0
    
    # Normalización actual (buggy)
    max_surface = 350
    norm_surface_actual = min(raw_surface / max_surface, 1.0) * math.log1p(max_surface)
    norm_form = math.log1p(raw_form)
    ratio_actual = norm_surface_actual / norm_form
    
    # El ratio debe ser razonable — si surface tiene peso 15% y form 23%,
    # no deben diferir en escala por más de 3x
    assert ratio_actual > 0.40, (
        f"surface_specialization normaliza a escala {ratio_actual:.2f}x de form_recent. "
        f"Debe ser >0.40. Bug D53-06: MAX_RAW_SCORES['surface_specialization']={max_surface} "
        f"es demasiado alto para los valores reales (~10-50)."
    )
```

---

## D53-07 — CRÍTICO: ELO calculado pero capped — 13% del modelo es constante para todo el top-200

### Causa raíz exacta

**Archivo:** `analysis/rivalry_analyzer.py:1524`

```python
raw_scores['elo_rating'] = min(max(0, elo - 1500), 250)
```

**`normalization.py:37`:**
```python
'elo_rating': 250,   # ELO normalizado: max(0, elo - 1500)
```

El cap es 250. La fórmula `max(0, elo-1500)` llega a 250 cuando `elo >= 1750`. Todo jugador con ELO ≥ 1750 recibe `raw_elo = 250` identico:

| Jugador | ELO real | raw_elo | norm_elo |
|---|---|---|---|
| Sinner #1 | 2400 | 250 | 5.526 |
| Alcaraz #2 | 2300 | 250 | 5.526 |
| Mensik #18 | 1942 | 250 | 5.526 |
| Dimitrov #211 | 1757 | 250 | 5.526 |
| ITF #500 | 1450 | 0 | 0.000 |

**Umbral de colapso: ELO ≥ 1750 → todos equivalentes.** Sinner y Dimitrov tienen el mismo ELO en el modelo. El 13% de peso asignado a ELO no diferencia entre ningún jugador del top-200 ATP.

El LOG confirma: `LOG_ELO_RATINGS: Jakub Mensik=1942, Grigor Dimitrov=1757` — calculado correctamente. Pero `raw_scores['elo_rating'] = 250` para ambos — descartado.

### Fix propuesto

```python
# Opción A — eliminar cap, usar raw directo
raw_scores['elo_rating'] = max(0, elo - 1500)
# Efecto: Mensik=442, Dimitrov=257 → ratio 1.72x preservado
# norm: log1p(442)=6.094 vs log1p(257)=5.554 → diferencia real

# Opción B — escala logarítmica desde 1000 (más estable para outliers)
raw_scores['elo_rating'] = math.log1p(max(0, elo - 1000)) * 30
# Comprime rangos extremos, preserva diferencias en zona media
```

### Test requerido (T53-07)

```python
def test_t53_07_elo_differentiates_within_top200():
    """D53-07: ELO debe producir raw_scores diferentes para jugadores con ELO distinto dentro del top-200."""
    # Sinner ELO=2400, Dimitrov ELO=1757 — deben tener raw_elo diferente
    elo_sinner = 2400
    elo_dimitrov = 1757
    raw_sinner = min(max(0, elo_sinner - 1500), 250)
    raw_dimitrov = min(max(0, elo_dimitrov - 1500), 250)
    assert raw_sinner != raw_dimitrov, (
        f"D53-07: Sinner (ELO={elo_sinner}) y Dimitrov (ELO={elo_dimitrov}) "
        f"producen el mismo raw_elo={raw_sinner}. El cap=250 colapsa todo el top-200."
    )
```

---

## D53-08 — SoS plano: sub-componente calculado pero con cap que iguala a todos

### Causa raíz exacta

**Archivo:** `analysis/rivalry_analyzer.py` — dentro de `calculate_raw_scores()`

```python
raw_scores['strength_of_schedule'] = min(schedule_score * diversity_mult, 200)
```

**`normalization.py:38`:**
```python
'strength_of_schedule': 200
```

En el partido Mensik vs Dimitrov:
- Mensik SoS raw = 1087.5, con diversity_mult aplicado → capped a 200
- Dimitrov SoS raw = 770.0 → capped a 200

Ambos llegan a 200. El log lo confirma: `raw_scores['strength_of_schedule'] = 200` para ambos.

**Agravante:** en Grand Slam el peso de SoS es 0% (`'strength_of_schedule': 0.00` en DEFAULT_WEIGHTS). Así que SoS tiene dos problemas encadenados: está capped (todos igual) **y** no contribuye al puntaje en el tier donde más datos hay.

Sin embargo, SoS **sí alimenta `diversity_mult`** que amplifica otros componentes — por lo que el bug de cap puede estar silenciando el efecto multiplicador.

### Fix

Mismo patrón que D53-07: eliminar o elevar el cap. La escala 1087 vs 770 es una diferencia real (41%) que el modelo colapsa a 200/200 = 0%.

```python
# Opción A — elevar cap a 1500
raw_scores['strength_of_schedule'] = min(schedule_score * diversity_mult, 1500)
# normalization.py: 'strength_of_schedule': 1500

# Opción B — log1p para comprimir outliers
raw_scores['strength_of_schedule'] = math.log1p(schedule_score * diversity_mult) * 40
```

---

## D53-09 — Data Layer: ranking del oponente es el ACTUAL, no el histórico

### Causa raíz exacta

**Archivo:** `scraping/ninja_h2h_parser.py`

**Paso 1 — extracción desde API (líneas 321-373):**
```python
# _parse_player_history()
opponent_ranking = None
# Lee campo CA o CB del feed Ninja (ranking en el momento del partido)
opponent_ranking = int(cb)  # o int(ca)
entry['opponent_ranking'] = opponent_ranking
```

**Paso 2 — enriquecimiento posterior (líneas 1558-1566):**
```python
def _enrich_history(self, history):
    for match in history:
        opponent = match.get('opponent', '')
        rank = self.ranking_manager.get_player_ranking(opponent)   # ← RANKING HOY
        enriched_match['opponent_ranking'] = rank                  # ← SOBREESCRIBE el histórico
```

`_enrich_history` sobreescribe el `opponent_ranking` original (que venía del momento del partido) con el ranking **actual** del oponente. Si Medvedev estaba en Top-5 en 2017 pero hoy está en #7, el sistema usa #7. Si Jarry estaba en #20 en 2023 pero hoy está en #764, el sistema usa #764.

### Impacto en el partido Mensik vs Dimitrov

La señal `SCALP TOP-10 EN SUPERFICIE` se activa porque "Dimitrov venció a Medvedev #7 en hierba". Pero esa victoria de Dimitrov sobre Medvedev fue en 2019 Wimbledon — cuando Medvedev estaba en un ranking diferente. El sistema usa el ranking actual de Medvedev (#7) que puede ser correcto por casualidad o incorrecto según el momento.

Más grave: en los logs de SoS aparecen rivales con `Rank 764` que en el momento del partido eran Top-50. Esto contamina:
- `analyze_surface_specialization()` — puntos por calidad del rival en superficie
- `analyze_strength_of_schedule()` — fuerza de calendario
- Señal SCALP TOP-10 — activa/desactiva según ranking **actual** del rival, no histórico

### Relación con Nodo-51

Este es un bug de **data layer** — conecta con Nodo-51 F0 (PlayerRegistry) y F1 (TournamentContext). La solución correcta no es modificar `rivalry_analyzer.py` sino asegurarse de que `_enrich_history` preserve el `opponent_ranking` original del momento del partido cuando está disponible:

```python
def _enrich_history(self, history):
    for match in history:
        # Preservar ranking histórico si ya existe (viene del feed Ninja)
        if match.get('opponent_ranking'):
            enriched_match['opponent_ranking'] = match['opponent_ranking']  # ← no sobreescribir
        else:
            rank = self.ranking_manager.get_player_ranking(opponent)
            enriched_match['opponent_ranking'] = rank
```

**GATE:** este fix requiere validar que el feed Ninja realmente proporciona ranking histórico en CA/CB y que no sean campos vacíos. Verificar con una muestra antes de implementar.

---

## D53-10 — Ranking momentum: sub-componentes son constantes por falta de datos

### Causa raíz — NO es fuga de variables entre P1 y P2

El agente inicial sospechó una fuga. La evidencia muestra que NO hay fuga: cada jugador llama a `get_ranking_metrics(full_name)` independientemente. El problema es diferente:

**Código:** `rivalry_analyzer.py:711-713`
```python
already_secured = max(0, prox_pts - pts)         # prox_pts - pts
improvement_potential = max(0, pts_max - pts)     # pts_max - pts
pressure_index = defense_points - already_secured  # defense_points es 0 para ambos
```

**Datos reales en `atp_rankings_complete_*.json`:**
```
Mensik:   pts=2205  prox=2255  max=4155  defense_points=0
Dimitrov: pts=257   prox=307   max=2207  defense_points=0
```

**Resultado calculado:**

| Sub-componente | Mensik | Dimitrov | Idénticos? |
|---|---|---|---|
| already_secured | 50 | 50 | SÍ — `prox-pts=50` para ambos |
| improvement_potential | 1950 | 1950 | SÍ — `max-pts` por coincidencia |
| pressure_index | -50 | -50 | SÍ — `0-50=-50` para ambos |
| **Momentum** | 58.98 | 58.98 | SÍ |
| **Potencial** | 75.76 | 75.76 | SÍ |
| **Presión** | 19.66 | 19.66 | SÍ |

Solo `Base` difiere (usa `pts`: 153.98 vs 111.06).

**La causa real:** `already_secured = prox_pts - pts`. El ranking manager guarda `prox_pts` como los puntos que el jugador ya tiene asegurados para el próximo periodo — en este caso ambos tienen exactamente 50 puntos más asegurados. Coincidencia numérica, no bug. Pero revela que `prox_points`, `max_points` y `defense_points` son campos que probablemente no están bien poblados en el scraper ATP — `defense_points=0` para todos sugiere que ese campo no se extrae.

**Conclusión D53-10:** No hay fuga. Hay datos de ranking incompletos: `defense_points` siempre 0, `prox_points` y `max_points` posiblemente genéricos. El 87% de `ranking_momentum` (Momentum+Potencial+Presión) es constante entre jugadores. Solo el `Base` (log1p(pts)*10) diferencia realmente.

---

## D53-11 — Cuestionamiento Nodo-14: ajuste de hierba refuerza el componente menos informativo

### El ajuste actual (líneas 1396-1402)

```python
elif _surf_adj == 'grass':
    weights['common_opponents'] -= 0.05   # baja rivales comunes
    weights['form_recent'] += 0.05        # sube forma reciente
    # "alta varianza césped, Nodo-14"
```

**El razonamiento de Nodo-14:** en hierba hay más varianza → bajar rivales comunes (menos predictivos), subir forma reciente (más fresca).

**El problema que Fable señala:** con D53-06 activo, `form_recent` normaliza a escala ~5 y `surface_specialization` normaliza a ~0.18-0.56. El ajuste de hierba sube el componente que ya domina (forma, peso efectivo ~55-60%) y mantiene igual el componente que casi no existe en el modelo (superficie, peso efectivo ~1-5%). 

En otras palabras: la lógica "alta varianza hierba → subir forma" puede ser correcta en teoría, pero mientras D53-06 exista, el ajuste de pesos opera sobre un modelo donde superficie ya vale casi 0. El ajuste de Nodo-14 no puede compensar D53-06 porque trabajan en capas distintas.

**Orden correcto de corrección:**
1. Fix D53-06 primero (normalización superficie)
2. Luego re-evaluar si el ajuste de Nodo-14 sigue siendo necesario o si la superficie ya discrimina suficiente por sí sola
3. No tocar Nodo-14 antes de que D53-06 esté corregido — los pesos ajustados no tienen sentido sobre una normalización rota

---

## Resumen: Prioridad de implementación revisada

| ID | Severidad | Descripción | Fix | Gate |
|---|---|---|---|---|
| **D53-06** | 🔴 CRÍTICO | Superficie normaliza a 0.18 vs escala ~5 del resto | `MAX_RAW_SCORES['surface_specialization'] = 70` o `_LINEAR_COMPONENTS = set()` | T53-06 FAIL→PASS |
| **D53-07** | 🔴 CRÍTICO | ELO constante para todo top-200 (cap=250, umbral ELO≥1750) | Eliminar cap o subir a 1000 | T53-07 FAIL→PASS |
| **D53-01** | 🔴 CRÍTICO | H2H date parsing `%y` vs `%Y` — silencia historial directo | `'%d.%m.%Y'` en líneas 655 y 1682 | T53-01 FAIL→PASS |
| **D53-09** | 🟠 IMPORTANTE | Ranking oponente = ranking actual, no histórico | Preservar campo original en `_enrich_history` | verificar feed Ninja |
| **D53-08** | 🟡 MEDIO | SoS capped a 200 para todos — diferencias reales eliminadas | Elevar cap a 1500 o usar log1p | T53-08 |
| **D53-02** | 🟡 MEDIO | H2H threshold 250 días demasiado agresivo | Ponderación decreciente | GATE n≥30 Shadow Book |
| **D53-10** | 🟡 MEDIO | Ranking momentum: 3 de 4 sub-componentes constantes por datos incompletos | Poblar defense_points en ATP scraper | verificar fuente |
| **D53-03** | 🟡 MEDIO | Rivales comunes no descuenta superficie | Usar `_surface_overlap_rate()` | GATE n≥30 |
| **D53-11** | ⚪ DIFERIDO | Nodo-14 grass adjustment opera sobre normalización rota | Re-evaluar DESPUÉS de D53-06 | depende de D53-06 |
| **D53-04** | ⚪ COSMÉTICO | Pesos suman 99% en display | Redondeo en output | ninguno |
| **D53-05** | ⚪ ORGANIZACIÓN | Señales enterradas en logs | Output format | ninguno |

### Orden de implementación obligatorio

```
Fase A: Tests T53-06, T53-07, T53-01 en estado FAIL (confirmar bugs antes de tocar código)
Fase B: Fix D53-06 → pytest PASS → correr pipeline → verificar contribución superficie
Fase C: Fix D53-07 → pytest PASS → correr pipeline → ELO diferencia top-200
Fase D: Fix D53-01 → pytest PASS → verificar H2H 2-0 Mensik en output
Fase E: Fix D53-09 → verificar con muestra de feed Ninja que CA/CB tienen ranking histórico
Fase F (GATED n≥30): D53-02, D53-03, D53-08
Fase G (DESPUÉS DE B): Re-evaluar D53-11 (Nodo-14 grass adjustment)
```

---

## Preguntas para Fable — Addendum

**F53-Q6:** ¿El valor `MAX_RAW_SCORES['surface_specialization'] = 350` fue elegido basado en datos empíricos o es el mismo cap que los otros componentes por defecto? Si es por defecto, D53-06 Opción B (log1p) es la corrección más limpia.

**F53-Q7:** ¿`prox_points`, `max_points`, `defense_points` se extraen del feed ATP o son campos calculados? Si `defense_points=0` para todos los jugadores, el sub-componente `pressure_index` es siempre negativo (bonus) para cualquier jugador con puntos asegurados — sesgo sistemático hacia jugadores con más puntos.

**F53-Q8:** ¿El campo CA/CB en el feed Ninja H2H es efectivamente el ranking del oponente en el momento del partido, o es el ranking actual en el momento de la consulta? Si es el ranking en el momento de la consulta, entonces `_enrich_history` no sobreescribe nada útil y D53-09 no tiene fix simple.

**F53-Q9:** ¿Con D53-06 y D53-07 corregidos, el modelo habría predicho Dimitrov sobre Mensik en Wimbledon? Esta pregunta tiene respuesta verificable — correr el modelo con los fixes aplicados sobre los mismos datos y comparar el puntaje final.

---

*Verificación aritmética completa disponible ejecutando:*
```bash
python3 -c "
import math
# D53-06: escala real con datos Mensik/Dimitrov
print('surface norm (actual):', min(10.89/350,1)*math.log1p(350), 'vs', min(33.49/350,1)*math.log1p(350))
print('form norm:', math.log1p(150), 'vs', math.log1p(75))
# D53-07: ELO cap
print('ELO cap Mensik:', min(max(0,1942-1500),250))
print('ELO cap Dimitrov:', min(max(0,1757-1500),250))
"
```

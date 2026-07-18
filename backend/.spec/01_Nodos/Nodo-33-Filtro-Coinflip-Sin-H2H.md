# Nodo-33 — Filtro Coinflip Sin H2H: Gate Lateral cuota<2.10 y Colapso James-Stein en ITF

> **Wikilinks:** [[Nodo-117-Auditoria-Scraping-Rankings-Cobertura-H2H]] (cobertura H2H 33% — misma familia: partidos sin H2H por bugs de pipeline)
> **Fecha:** 2026-06-24
> **Severidad:** ALTA — CONFIRMADA, pérdida real materializada en sesiones 2026-06-23 y 2026-06-24
> **Prerequisitos:** Nodo-32 (gate p_modelo, golden_zone, ITF fallback — completado 2026-06-22)
> **Archivos afectados:** `edge_calculator.py`, `betplay_combo_builder.py`
> **Implementa:** Sonnet | **Tests:** Haiku
>
> **Estado Fase 1:** ✅ CERRADA 2026-06-24 — 12 tests, 1256 passed, mutación detectada
> **Estado Fase 2:** ⏳ PENDIENTE — requiere n≥30 resultados en segmento n_h2h=0/itf

---

## 0. RESUMEN EJECUTIVO

Picks con `n_h2h=0` y `p_modelo` entre 0.509–0.514 (coin-flip) terminaron anclando 6 de 8 combos ITF en producción el 2026-06-24, porque el gate de `apostar` no evalúa `n_h2h` en ningún punto de su condición y porque el shrinkage James-Stein colapsa `p_blend` hacia 0.50 cuando la calibración ITF está vacía — aplastando la señal real de ELO/Markov/Erdős.

---

## 1. CONTEXTO DEL HALLAZGO

Descubierto en sesión de producción 2026-06-24. Tras correr el pipeline completo (Pasos 1-4), el combo builder generó 8 combos ITF. Inspección manual de los picks reveló que 3 de los 4 anclas de los combos eran picks con `p_modelo` entre 0.509 y 0.514 — literalmente coin-flip.

### Picks reales de hoy (evidencia primaria)

| Pick | p_modelo | p_blend | edge | n_h2h | cuota | tier | cat |
|---|---|---|---|---|---|---|---|
| Oliver Majdandzic | 0.514 | 0.501 | 36.6% | 0 | 6.75 | itf | watchlist |
| Karyna Fiadosik | 0.509 | 0.500 | 35.5% | 0 | 6.50 | itf | watchlist |
| Louay Makke | 0.509 | 0.500 | 34.2% | 0 | 6.00 | itf | watchlist |
| Giulia Alexandra Musat | 0.583 | 0.504 | 41.6% | 0 | 6.00 | itf | watchlist |

Los 4 llegaron a pool del combo builder como picks de cobertura. Los 3 primeros tienen `p_blend≈0.500` — el modelo no tiene convicción real. Aparecieron en los combos 1-8 (todos los combos ITF del día).

### Evidencia de pérdida real confirmada

**2026-06-23:** Resultado combos: 4 ganados / 8 perdidos (hit% 33.3%). Causa raíz confirmada en `betslip_index_20260623_004706.json`: Simona Cucu (`p_modelo=0.50`, `cuota=4.7`, `edge=61.4%`) y Gabriela Kawano Cho (`p_modelo=0.50`, `cuota=5.5`, `edge=35.0%`) — ambos con `p_modelo=0.50` hardcodeado, arrastraron 9 de los 12 combos. El campo `kelly_kl=0.0` en el betslip_index indica que los stakes no fueron serializado correctamente. **Monto exacto pendiente de reconciliación con registro de bankroll.**

**2026-06-24:** Mismo patrón observado en producción (Majdandzic, Fiadosik, Makke con p_modelo=0.509-0.514 anclando 6/8 combos ITF). Combos descartados manualmente antes de apostarse al detectar el patrón — pérdida evitada en esta sesión.

---

## 2. DIAGNÓSTICO TÉCNICO

### BUG-33-1: Colapso James-Stein + fallback ITF vacío aplasta señal ELO/Markov/Erdős

**Archivo:** `edge_calculator.py:786-788`

```python
# F-24-2: Calibration Gap — gap entre p_blend (James-Stein) y p_modelo
_js_factor = _n_cal / (_n_cal + 20) if _n_cal > 0 else 0.0   # línea 787
_p_blend = _js_factor * p_modelo + (1 - _js_factor) * p_hist   # línea 788
```

Cuando `_n_cal=0` (sin calibración para ese tier+superficie), `_js_factor=0.0` y `_p_blend = p_hist` — el blend ignora completamente `p_modelo`.

**Cadena de fallback cuando n_h2h=0 en ITF** (`edge_calculator.py:349, 368` — función `theta_thompson()`):

```
1. por_superficie_y_tier['clay_itf'] → n=1 (0W/1L) → n<10 → FALLA
2. fallback_por_tier['itf'] → 0.50 (añadido por Nodo-32 Fase 1) → RETORNA 0.50
3. [Si 2 no existiera] por_superficie['clay'] → pocas muestras → FALLA
4. [Último recurso] Thompson(wins_global, losses_global) → 0.629
```

**Resultado:** `p_hist=0.50`, `_js_factor≈0.0` cuando `n_cal` es bajo → `p_blend ≈ 0.50` sin importar lo que digan ELO/Markov/Erdős.

**Por qué Nodo-32 no cerró esto:** Nodo-32 Fase 1 añadió `fallback_por_tier['itf']=0.50` intencionalmente conservador. Eso es correcto para regularización cuando los datos son escasos, pero tiene el efecto secundario de hacer que picks con `p_modelo=0.509` produzcan `p_blend=0.500` — aún más cercano al coin-flip que el propio `p_modelo`.

**Mecanismo del aplastamiento con valores reales:**

```
p_modelo=0.509, p_hist=0.50 (fallback ITF), n_cal=1
_js_factor = 1 / (1 + 20) = 0.048
_p_blend = 0.048 × 0.509 + 0.952 × 0.50 = 0.024 + 0.476 = 0.500
```

ELO/Markov calculan `p_modelo=0.509` — pero el blend lo aplasta a `0.500`.

### BUG-33-2: Gate de apostar no evalúa n_h2h — puerta lateral cuota<2.10

**Archivo:** `edge_calculator.py:476-480`

```python
apostar = (
    edge > EDGE_MIN                                               # 5%
    and kelly_kl_ajustado > KELLY_KL_MIN                         # 2%
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)   # línea 479
)
```

El campo `n_h2h` se calcula en línea 783 (`_n_h2h_v = resultado['n_h2h']`) pero **nunca se usa en el gate**.

La condición de la línea 479 tiene dos ramas:
- **Rama convicción:** `p_modelo >= 0.55` — evalúa la calidad del modelo
- **Rama lateral:** `cuota_favorito < 2.10` — bypass completo de la convicción cuando la cuota es baja

**Zona gris desprotegida (cuota 2.10–2.75):**

Un pick con `p_modelo=0.509`, `cuota=2.75`, `n_h2h=0`:
- Rama convicción: 0.509 < 0.55 → FALLA
- Rama lateral: 2.75 > 2.10 → FALLA
- **Pero si `edge=14.5% > 5%` y `kelly_kl=8.1% > 2%` → apostar=True**

El edge "real" de 14.5% es artificial: viene de la diferencia entre `p_modelo=0.509` y `p_implicita=1/2.75=0.364`, pero `p_modelo=0.509` no es convicción real del modelo — es el resultado del colapso James-Stein cuando no hay calibración.

**Nota sobre watchlist (casos de hoy):** Los 4 picks de hoy llegaron como **watchlist**, no APOSTAR. El gate los bloqueó correctamente para apuesta individual. Sin embargo, el combo builder los usó como anclas de combos **sin filtro adicional de calidad**, lo que convierte este bug en un problema real incluso sin que el pick cruce a `apostar=True`.

### BUG-33-3: betplay_combo_builder.py:1620-1621 usa p_blend/p_modelo hardcodeados para picks watchlist en cobertura — INVESTIGACIÓN DE CAUSA

**Archivo:** `betplay_combo_builder.py:1609-1628`

```python
# Also grab watchlist picks from cobertura legs not in individuales
for combo in plan.get("cobertura", []):
    for leg in combo.get("legs", []):
        name = leg.get("jugador", "")
        if name and name not in seen_names:
            seen_names.add(name)
            pool.append({
                "jugador":    name,
                "cuota":      leg.get("cuota", 0),
                "p_blend":    0.55,   # línea 1620 — hardcoded
                "p_modelo":   0.50,   # línea 1621 — hardcoded
                "edge_pct":   "?",
                ...
            })
```

**Investigación del origen del hardcode:**

`git log --oneline -- betplay_combo_builder.py` no muestra historial detallado de estas líneas (5 commits totales en repo, todos recientes). Sin embargo, el análisis del código circundante confirma la causa técnica:

- Las `cobertura` en `trader_plan_*.json` almacenan legs con campos mínimos: `jugador`, `cuota`, `stake`. **No propagan `p_blend`, `p_modelo`, ni `n_h2h`** desde el edge_report original.
- El combo builder intenta compensar con `edge_tier_map` (construido a partir del edge_report vía lookup por nombre), pero los picks de cobertura que son **watchlist** pueden no estar en `edge_tier_map` con todos sus campos.
- Los valores `0.55` y `0.50` son **placeholders de desarrollo**: `0.55` es el threshold mínimo `P_MODELO_MIN_UNDERDOG` (señal de "convicción mínima asumida") y `0.50` es el prior neutro ITF.

**Conclusión BUG-33-3:** Es un **placeholder de desarrollo nunca reemplazado**, no una decisión intencional documentada. El código asume `p_blend=0.55` (MODERATE) para todos los watchlist de cobertura, independientemente de su convicción real. Picks con `p_modelo=0.509` entran al pool con el mismo peso visual que picks con `p_modelo=0.78`.

**Segunda instancia del mismo hardcode** (línea 1652-1653, para picks `sin_edge`):
```python
"p_blend":    0.55,   # mismo placeholder
"p_modelo":   0.50,
```

**Causa raíz estructural:** El trader_plan no serializa suficientes campos de los picks de cobertura. El fix correcto requiere lookup al edge_report real, no hardcode.

---

## 3. EVIDENCIA CUANTITATIVA

Análisis sobre los 10 últimos edge_reports (2026-06-14 a 2026-06-24), 30 picks APOSTAR total:

| Segmento | Picks | Coin-flip (p_modelo 0.49-0.55) | % coin-flip |
|---|---|---|---|
| n_h2h=0 | 25 (83.3%) | 6 picks | **24%** del segmento |
| n_h2h>0 | 5 (16.7%) | 0 picks | 0% |

**Media p_modelo:**
- n_h2h=0: 0.709 (pero con cola inferior coin-flip que arrastra)
- n_h2h>0: 0.570 (más conservador pero sin coin-flips)

**Nota explícita sobre muestra:** n=30 picks APOSTAR total, n=6 picks coin-flip con n_h2h=0. **Muestra insuficiente para confirmar hit rate del subsegmento específico** (n_h2h=0, p_modelo 0.49-0.55). La evidencia de daño es estructural (picks anclan combos) y por antecedente (2026-06-23: hit% 33.3% con mismo patrón), no estadística con n suficiente.

**Picks n_h2h=0 con p_modelo en zona gris [0.55-0.60) — riesgo del fix Fase 1:**

| Pick | p_modelo | tier | cat |
|---|---|---|---|
| Alexandra Eala | 0.569 | atp500 | apostar |
| Giulia Alexandra Musat | 0.583 | itf | watchlist |

Solo 2 picks en zona [0.55, 0.60) con n_h2h=0 en los últimos 10 reportes. El fix de Fase 1 (condición `not (n_h2h==0 and p_modelo < 0.55)`) **no los afecta** — ambos tienen `p_modelo >= 0.55` y pasan el threshold. El impacto del fix recae exclusivamente sobre el rango 0.49–0.549 con n_h2h=0.

---

## 4. PLAN DE FIX POR FASES

### Fase 1: Opción A — Cierre del síntoma en gate + fix combo builder (IMPLEMENTAR)

**Cambio 1 — `edge_calculator.py:476-480`:**

Agregar condición que bloquee picks con `n_h2h=0` y `p_modelo < 0.55`, independientemente de cuota:

```python
# ANTES (línea 476-480):
apostar = (
    edge > EDGE_MIN
    and kelly_kl_ajustado > KELLY_KL_MIN
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
)

# DESPUÉS:
apostar = (
    edge > EDGE_MIN
    and kelly_kl_ajustado > KELLY_KL_MIN
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
    and not (n_h2h == 0 and p_modelo < P_MODELO_MIN_UNDERDOG)
)
```

**Lógica:** La puerta lateral `cuota < 2.10` sigue existiendo para favoritos válidos, pero ya no puede bypassear la convicción del modelo cuando no hay historial H2H directo. Si el modelo no tiene H2H y tampoco tiene convicción (p_modelo < 0.55), bloquear sin importar cuota.

**Nota:** Esta condición es redundante para underdogs (cuota >= 2.10) donde ya aplica `p_modelo >= 0.55`. Su efecto real es en la zona de favoritos y slight_underdogs (cuota < 2.10) con n_h2h=0.

**Cambio 2 — `betplay_combo_builder.py:1617-1628` — Opción 1 + lookup real (bloqueo duro):**

**Decisión:** Se implementa **Opción 1** (bloqueo duro) más la corrección del hardcode. Opción 2 sola es insuficiente.

**Justificación con evidencia de código:** El único filtro individual en el loop `available_pool` (líneas 1679-1696) es el BBI guard (`BBI < 0.40`). BBI de Majdandzic @6.75 = `(1 - 1/6.75) × 1.0 = 0.852` — pasa sin problema. CV guard y Dispersion guard (líneas 1707-1716) operan sobre el pool completo, no sobre picks individuales. **No existe ningún filtro individual por `p_modelo` en todo el path de construcción del pool.** Corregir el hardcode de 0.50 a 0.514 real no impide que el pick entre — solo cambia el valor con el que entra. El bloqueo duro es necesario.

```python
# ANTES: hardcode sin lookup ni guard
pool.append({
    "p_blend":  0.55,
    "p_modelo": 0.50,
    ...
})

# DESPUÉS: lookup real + bloqueo duro si coin-flip sin H2H
_info = edge_tier_map.get(name, {})
_p_modelo_real = _info.get("p_modelo", 0.50)
_n_h2h_real    = _info.get("n_h2h", 0)

# BLOQUEO DURO (Opción 1): n_h2h=0 + p_modelo<0.55 = coin-flip garantizado
if _n_h2h_real == 0 and _p_modelo_real < P_MODELO_MIN_UNDERDOG:
    logger.info(f"  🚫 {name} — n_h2h=0 + p_modelo={_p_modelo_real:.3f} < 0.55 (coin-flip guard, excluido pool)")
    continue

pool.append({
    "jugador":    name,
    "cuota":      leg.get("cuota", 0),
    "p_blend":    _info.get("p_blend", 0.55),   # real si existe, placeholder si no
    "p_modelo":   _p_modelo_real,
    "n_h2h":      _n_h2h_real,
    "edge_pct":   _info.get("edge_pct", "?"),
    ...
})
```

El mismo fix (lookup + bloqueo duro) aplica a la instancia de líneas 1652-1653 (sin_edge picks).

**Comportamiento cuando el lookup falla** (pick no está en edge_tier_map): fallback `p_modelo=0.50` y `n_h2h=0` → bloqueo automático por la condición. Correcto: sin información real del pick, no entra al pool.

### Fase 2: Opción B — Revisión del shrinkage cuando n_cal=0 (PENDIENTE DE DATOS)

**Objetivo:** Cuando `n_cal=0`, no aplicar shrinkage hacia `p_hist=0.50` sino permitir que ELO/Markov/Erdős expresen su señal sin aplastamiento.

**Cambio candidato** (`edge_calculator.py:787-788`):
```python
# CANDIDATO (NO IMPLEMENTAR AÚN):
# Si n_cal=0, usar p_modelo directamente sin blend (o con factor mínimo)
_js_factor = max(_n_cal / (_n_cal + 20), 0.30) if _n_cal > 0 else 0.30  # floor en 0.30
_p_blend = _js_factor * p_modelo + (1 - _js_factor) * p_hist
```

**Por qué no implementar ahora:** Esta fase requiere validar que ELO/Markov/Erdős en ITF con `n_h2h=0` tienen poder discriminante real — dato que no tenemos con n suficiente. Implementar sin evidencia podría introducir sobreconfianza en la dirección opuesta.

**Requisito para Fase 2:** n≥30 picks con resultado conocido en el segmento `n_h2h=0, tier=itf` antes de modificar el shrinkage. Acumular con `betslip_registrar.py`.

---

## 5. TESTS REQUERIDOS

**Archivo:** `tests/test_nodo33.py`

### Fase 1 Tests

```
T33-01: test_gate_blocks_n_h2h0_coinflip_cuota_bajo
    GIVEN p_modelo=0.514, cuota=1.90 (cuota < 2.10), n_h2h=0, edge>5%, kelly>2%
    WHEN edge_calculator evalúa el gate
    THEN apostar == False (la puerta lateral cuota<2.10 no bypasea n_h2h=0 sin convicción)

T33-02: test_gate_blocks_n_h2h0_coinflip_cuota_alto
    GIVEN p_modelo=0.509, cuota=6.75, n_h2h=0, edge>5%, kelly>2%
    WHEN edge_calculator evalúa el gate
    THEN apostar == False (ya bloqueado por Nodo-32, confirmar no regresión)

T33-03: test_gate_allows_n_h2h0_strong_conviction
    GIVEN p_modelo=0.67, cuota=3.90, n_h2h=0, edge>5%, kelly>2%
    WHEN edge_calculator evalúa el gate
    THEN apostar == True (n_h2h=0 no bloquea cuando hay convicción real)

T33-04: test_gate_allows_n_h2h1_coinflip_cuota_bajo
    GIVEN p_modelo=0.514, cuota=1.90, n_h2h=1, edge>5%, kelly>2%
    WHEN edge_calculator evalúa el gate
    THEN resultado NO bloqueado por BUG-33-2 fix (hay H2H, aplica solo Nodo-32)
    NOTA: puede bloquearse por otras razones (axes, BBI) — verificar que n_h2h>0 no activa la condición nueva

T33-05: test_majdandzic_fixture — caso real 2026-06-24
    GIVEN Oliver Majdandzic: p_modelo=0.514, p_blend=0.501, edge=36.6%, n_h2h=0, cuota=6.75
    WHEN edge_calculator procesa
    THEN apostar == False (pre-fix: watchlist por razones previas — confirmar)
    THEN n_h2h=0 guard activo: pick excluido de pool combos (post-fix)

T33-06: test_fiadosik_fixture — caso real 2026-06-24
    GIVEN Karyna Fiadosik: p_modelo=0.509, p_blend=0.500, edge=35.5%, n_h2h=0, cuota=6.50
    WHEN edge_calculator procesa
    THEN apostar == False, pick excluido de pool combos post-fix

T33-07: test_makke_fixture — caso real 2026-06-24
    GIVEN Louay Makke: p_modelo=0.509, p_blend=0.500, edge=34.2%, n_h2h=0, cuota=6.00
    WHEN edge_calculator procesa
    THEN apostar == False, pick excluido de pool combos post-fix

T33-08: test_musat_fixture — caso real 2026-06-24
    GIVEN Giulia Alexandra Musat: p_modelo=0.583, p_blend=0.504, edge=41.6%, n_h2h=0, cuota=6.00
    WHEN edge_calculator procesa
    THEN apostar == False (p_modelo=0.583 >= 0.55 — no bloqueado por fix Fase 1)
    THEN pick PUEDE entrar a pool combos (tiene convicción)

T33-09: test_combo_builder_lookup_real_p_modelo
    GIVEN watchlist pick en cobertura: Majdandzic con p_modelo=0.514 en edge_tier_map
    WHEN combo builder construye pool
    THEN pick en pool tiene p_modelo=0.514 (real), NO 0.50 (hardcode)

T33-10: test_combo_builder_coinflip_guard_excluye_pick
    GIVEN watchlist pick: p_modelo=0.514, n_h2h=0 en edge_tier_map
    WHEN combo builder aplica guard n_h2h=0
    THEN pick NO entra al pool de combos

T33-11: test_combo_builder_coinflip_guard_permite_pick_conviccion
    GIVEN watchlist pick: p_modelo=0.67, n_h2h=0 en edge_tier_map
    WHEN combo builder aplica guard
    THEN pick SÍ entra al pool (tiene convicción, n_h2h=0 no lo bloquea)

T33-12: test_combo_builder_fallback_cuando_no_en_edge_map
    GIVEN watchlist pick en cobertura que NO está en edge_tier_map (nombre sin match)
    WHEN combo builder construye pool
    THEN pick usa hardcode p_modelo=0.50 como fallback
    THEN pick excluido por guard (p_modelo=0.50 < 0.55 y n_h2h=0 por defecto)
```

---

## 6. RIESGOS CONOCIDOS

### Riesgo 1 — Impacto en picks n_h2h=0, p_modelo [0.55, 0.60)

**Pregunta:** ¿Cuántos picks con n_h2h=0 pero p_modelo entre 0.55-0.60 existen históricamente? ¿El fix de Fase 1 los afecta?

**Respuesta con datos reales (query sobre últimos 10 edge_reports):**

| Pick | p_modelo | tier | cat |
|---|---|---|---|
| Alexandra Eala | 0.569 | atp500 | apostar |
| Giulia Alexandra Musat | 0.583 | itf | watchlist |

Solo 2 picks en zona [0.55, 0.60) con n_h2h=0 en los últimos 10 reportes. **El fix Fase 1 no los afecta** — ambos tienen `p_modelo >= 0.55` y pasan el guard. El impacto recae exclusivamente sobre el rango 0.49–0.549 con n_h2h=0.

**Nota explícita sobre tamaño de muestra:** Esta verificación se realizó sobre una ventana de 10 edge_reports (2026-06-14 a 2026-06-24), n=30 picks APOSTAR total. **No es una garantía estadística.** Es posible que con mayor volumen de datos aparezcan más picks en la zona [0.55, 0.60) con n_h2h=0, especialmente en tiers ITF y Challenger donde la señal H2H es escasa. El equipo debe re-verificar este punto cuando haya ≥100 picks APOSTAR acumulados post-fix, para confirmar que el umbral `p_modelo < 0.55` no está cortando señal real en esa zona gris.

### Riesgo 2 — Reducción de pool disponible para combos

Con el guard activo, los combos ITF podrían quedar con pools más pequeños (o vacíos) cuando todos los picks de cuota alta tienen n_h2h=0 y p_modelo < 0.55. Esto es comportamiento **correcto** — mejor no tener combos que tener combos anclados en coin-flips.

### Riesgo 3 — Interacción con golden_zone (Nodo-32)

`golden_zone` ya requiere `p_modelo >= P_MODELO_MIN_UNDERDOG` (Nodo-32 FIX-32-3). El fix de Fase 1 agrega una condición ortogonal sobre `n_h2h`. No hay conflicto — ambos gates son independientes y el más restrictivo aplica.

### Riesgo 4 — Hardcode residual en combo builder (BUG-33-3)

Si un pick de watchlist no está en `edge_tier_map`, el fallback sigue siendo `p_modelo=0.50` — y el guard lo excluirá automáticamente (0.50 < 0.55). Esto es el comportamiento correcto: picks sin información real no entran a combos.

---

## 7. REGLAS PERMANENTES DERIVADAS

**REGLA-T33-1:** Un pick con `n_h2h=0` y `p_modelo < 0.55` nunca puede ser APOSTAR ni ancla de combo, independientemente de la cuota. Sin H2H y sin convicción del modelo = coin-flip garantizado.

**REGLA-T33-2:** El combo builder no puede usar picks de cobertura sin lookear sus valores reales de `p_modelo` y `n_h2h` en el edge_report. Los hardcodes `0.55/0.50` son aceptables solo como fallback explícito cuando el lookup falla, y en ese caso aplica el guard de T33-1.

**REGLA-T33-3:** Antes de modificar los parámetros de shrinkage James-Stein (Fase 2), acumular n≥30 resultados en el segmento `n_h2h=0, tier=itf` con `betslip_registrar.py`. No implementar Fase 2 sin ese dato.

---

## 8. MÉTRICAS DE ÉXITO POST-IMPLEMENTACIÓN

### Validación inmediata (sin esperar partidos)

| Métrica | Antes | Post-fix | Estado |
|---|---|---|---|
| Picks coin-flip (p_modelo 0.49-0.55, n_h2h=0) en pool combos | presente | 0 | ✅ CONFIRMADO — `_es_coinflip_sin_h2h()` bloquea |
| Tests Nodo-33 passing | 0/12 | 12/12 | ✅ CONFIRMADO 2026-06-24 |
| Tests regresión total | 1244 | ≥1256 | ✅ 1256 passed 2026-06-24 |
| Mutación detectada T33-01 | — | apostar=True con fix comentado | ✅ CONFIRMADO |
| Mutación detectada T33-10 | — | False is True con fn mutada | ✅ CONFIRMADO |

### Validación con partidos (requiere n≥15 combos con resultado)

| Métrica | Baseline | Target | Timeline |
|---|---|---|---|
| Hit% combos ITF | 33.3% (2026-06-23) | >50% | 2-3 semanas |
| Picks coin-flip que arrastran combos | presente | 0 ocurrencias | cada sesión |

### Trigger para Fase 2

| Condición | Descripción |
|---|---|
| n≥30 resultados registrados | Picks con `n_h2h=0, tier=itf` en `betslip_registrar.py` |
| Hit% del segmento documentado | ¿ELO/Markov/Erdős tienen poder discriminante real en ITF sin H2H? |
| Acumulación vía pipeline normal | No requiere acción especial — el fix de Fase 1 no bloquea picks con p_modelo≥0.55 que sí se registran |

---

## 10. CIERRE FASE 1 — 2026-06-24

### 10.1 Alcance real de Fase 1

| Ítem | Descripción | Archivo | Líneas |
|---|---|---|---|
| T33-01 override | Post-call override en `calcular_edge_completo()` — bloquea cuando `n_h2h==0 and p_modelo<0.55 and apostar=True` | `edge_calculator.py` | ~867–878 |
| `_es_coinflip_sin_h2h()` | Función extraída que consolida el guard en combo builder — importable desde tests | `betplay_combo_builder.py` | ~1291–1303 |
| Instancia 1 (cobertura legs) | Reemplaza hardcode — lookup real de `p_modelo`/`n_h2h` en `edge_tier_map` + llamada a `_es_coinflip_sin_h2h()` | `betplay_combo_builder.py` | ~1638–1644 |
| Instancia 2 (sin_edge picks) | Mismo patrón para picks watchlist que llegan por ruta sin_edge | `betplay_combo_builder.py` | ~1679–1685 |
| 12 tests | T33-01 a T33-12 cubriendo gate, guard logic y combo builder | `tests/test_nodo33.py` | 1–284 |

**Nota sobre dónde NO está el override:** La spec original (Sección 4) describía el fix en `calcular_edge()` (línea 476-480). Durante la auditoría se detectó que `_n_h2h_v` no existe en ese scope — solo en `calcular_edge_completo()` a partir de línea 789. El override fue correctamente colocado en `calcular_edge_completo()` siguiendo el mismo patrón de FIX-3 y FIX-6.

### 10.2 Incidente de remediación: NameError descubierto en auditoría

Durante la auditoría Fase 1 se descubrió que la implementación original (sesión previa) había colocado la condición de bloqueo dentro de `calcular_edge()`:

```python
# BUGGY — dentro de calcular_edge() — _n_h2h_v NO existe aquí
apostar = (
    edge > EDGE_MIN
    and kelly_kl_ajustado > KELLY_KL_MIN
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
    and not (_n_h2h_v == 0 and p_modelo < P_MODELO_MIN_UNDERDOG)  # NameError!
)
```

`_n_h2h_v` se define en `calcular_edge_completo()` en la línea 789 (`_n_h2h_v = resultado['n_h2h']`). `calcular_edge()` no tiene acceso a ese scope. `ast.parse()` no detecta NameErrors — solo errores de sintaxis.

**Impacto en producción (análisis):** El short-circuit de `and` habría protegido la mayoría de los casos — cuando `edge <= EDGE_MIN`, Python nunca evalúa `_n_h2h_v`. Pero para picks con edge real (>5%), el NameError habría causado una excepción en runtime. `calcular_edge_completo()` atrapa excepciones y retorna `None` para el pick afectado — efecto silencioso: el pick desaparece del edge_report sin mensaje de error visible.

**Remediación:** Revertir `calcular_edge()` al estado pre-T33 limpio, mover el override a `calcular_edge_completo()` después de línea 789.

### 10.3 Tabla de cobertura de tests

| Test | Clase | Qué prueba | Detecta revertir fix en |
|---|---|---|---|
| T33-01 | Gate | n_h2h=0 + p_modelo=0.54 + cuota=2.08 → apostar=False; 'T33-01' in motivo | `calcular_edge_completo()` override |
| T33-02 | Gate | n_h2h=0 + p_modelo=0.514 + cuota=4.70 → apostar=False (regresión T32-01) | T32-01 gate sin regresión |
| T33-03 | Gate | n_h2h=0 + p_modelo=0.67 + cuota=2.08 → apostar=True (convicción real) | Falso positivo override |
| T33-04 | Gate | n_h2h=1 + p_modelo=0.54 + cuota=2.08 → apostar=True; no 'T33-01' in motivo | Falso positivo n_h2h=1 |
| T33-05 | GuardLogic | Majdandzic p_modelo=0.514, n_h2h=0 → `_would_block()=True` | Constante P_MODELO_MIN_UNDERDOG |
| T33-06 | GuardLogic | Fiadosik p_modelo=0.509, n_h2h=0 → `_would_block()=True` | Constante P_MODELO_MIN_UNDERDOG |
| T33-07 | GuardLogic | Makke p_modelo=0.509, n_h2h=0 → `_would_block()=True` | Constante P_MODELO_MIN_UNDERDOG |
| T33-08 | GuardLogic | Musat p_modelo=0.583, n_h2h=0 → `_would_block()=False` | Falso positivo umbral |
| T33-09 | ComboBuilder | `_es_coinflip_sin_h2h(0.514, 0)` → True (Majdandzic) | Función real en betplay_combo_builder |
| T33-10 | ComboBuilder | `_es_coinflip_sin_h2h(0.514, 0)` → True — DETECCIÓN MUTACIÓN | Función real en betplay_combo_builder |
| T33-11 | ComboBuilder | `_es_coinflip_sin_h2h(0.67, 0)` → False (convicción real) | Falso positivo función |
| T33-12 | ComboBuilder | `_es_coinflip_sin_h2h(0.50, 0)` → True (fallback p_modelo=0.50) | Caso fallback bloqueado |

**Por qué T33-09 y T33-10 parecen duplicados:** T33-09 documenta el caso real de Majdandzic con nombre explícito. T33-10 es el test de detección de mutación con docstring que explica qué falla si `_es_coinflip_sin_h2h()` es comentada o retorna siempre False. Ambos son necesarios — el primero documenta el caso real, el segundo prueba la función.

### 10.4 Evidencia de detección de mutaciones

**Mutación T33-01** — comentar el override en `calcular_edge_completo()`:

```
FAILED tests/test_nodo33.py::TestNodo33Fase1Gate::test_t33_01_lateral_door_blocked_by_n_h2h_zero
AssertionError: T33-01: n_h2h=0 + p_modelo=0.54 debe bloquearse incluso con cuota<2.10.
Si este test falla, el fix T33-01 fue revertido.
assert True is False
```

**Por qué falla con el fix revertido:**
- edge = 0.54 - 1/2.08 = 0.059 > 0.05 ✓
- (p_modelo=0.54 >= 0.55 OR cuota=2.08 < 2.10) = (False OR True) = True ← puerta lateral abierta
- n_axes_active = 2 (surface=0.556 + BBI=0.741) → FIX-3 no bloquea
- markov=None → NOT HOT → FIX-6 no bloquea
- Sin override T33-01: apostar=True ← BUG activo, test FALLA con `assert True is False`

**Mutación T33-10** — cambiar `_es_coinflip_sin_h2h()` a `return False`:

```
FAILED tests/test_nodo33.py::TestNodo33Fase1ComboBuilder::test_t33_10_combo_builder_blocks_coinflip_pick
AssertionError: p_modelo=0.514 (<0.55) + n_h2h=0 debe ser identificado como coin-flip (True=bloqueado)
assert False is True
```

**Por qué detecta la mutación:** T33-09 a T33-12 importan `_es_coinflip_sin_h2h` directamente desde `betplay_combo_builder`. No hay lógica replicada en el test — si la función en el módulo real retorna False, el test `assert resultado is True` falla. No hay forma de que T33-10 pase con una función mutada.

**Falso-positivo eliminado:** La versión original de T33-09 a T33-12 tenía un helper inline `_apply_guard()` que replicaba la condición dentro del test. Esto hacía que los tests pasaran aunque el código real en `betplay_combo_builder.py` fuera comentado. La refactorización a `_es_coinflip_sin_h2h()` + import directo resuelve esto — es el mismo patrón de falso-positivo detectado en Nodo-32 Fase 3.

### 10.5 Estado final Fase 1

| Métrica | Valor |
|---|---|
| Tests Nodo-33 | 12/12 passing |
| Tests regresión total | 1256 passed, 0 failed |
| Tests regresión Nodo-32 | 34/34 passing |
| Gate override | `calcular_edge_completo()` línea ~867 |
| Función extraída | `_es_coinflip_sin_h2h()` en `betplay_combo_builder.py` |
| Instancias del guard | 2 (cobertura legs + sin_edge picks) |
| Mutaciones detectadas | 2/2 (T33-01 + T33-10) |
| Fecha cierre Fase 1 | 2026-06-24 |
| Fase 2 estado | ⏳ PENDIENTE — requiere n≥30 resultados en n_h2h=0/itf |

### 10.6 Qué activa Fase 2

Fase 2 (revisar floor de James-Stein cuando `n_cal=0`) no debe implementarse hasta:

1. **n≥30 picks registrados** con resultado conocido en segmento `n_h2h=0, tier=itf` — acumulados con `betslip_registrar.py`
2. **Hit% del segmento documentado** — si ELO/Markov/Erdős tienen poder discriminante real en ITF sin H2H, el floor actual de `_js_factor=0` aplasta señal real. Si no tienen poder discriminante, Fase 2 introduciría sobreconfianza
3. **No requiere acción especial** — el fix de Fase 1 no bloquea picks con p_modelo≥0.55 (que sí se registran); la acumulación ocurre con pipeline normal

---

## 11. WIKILINKS

- [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] — gate P_MODELO_MIN_UNDERDOG, golden_zone, ITF fallback
- [[Nodo-01-Edge-Calculator]] — fórmula Kelly-KL original
- [[Nodo-21-Pesos-Diferenciados-Por-Tier]] — James-Stein shrinkage en pesos
- [[Nodo-24-Bookmaker-Blindness-Scoring]] — BBI, n_h2h como señal de ceguera bookmaker
- [[Nodo-27-Pipeline-Tracker-Observabilidad]] — observabilidad que detectó el patrón
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint

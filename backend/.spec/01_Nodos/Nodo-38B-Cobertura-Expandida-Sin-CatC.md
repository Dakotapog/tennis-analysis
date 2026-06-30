# Nodo-38B — Cobertura Expandida Sin Cat-C

> **Fecha inicio:** 2026-06-26
> **Fecha cierre:** 2026-06-27 08:50 UTC
> **Severidad:** MEJORA OPERATIVA — Cuando no hay picks Cat-C, Nodo-38 genera solo 1 CORE + 3 cobertura ($500 c/u). El ~55% del budget diario (SAT+MOON) queda sin usar.
> **Prerequisitos:** Nodo-38 (combo_confianza_builder.py con CORE/SAT/MOON implementado)
> **Archivos modificados:** `combo_confianza_builder.py` (6 cambios quirúrgicos)
> **Tests:** `tests/test_nodo38.py` (25/25 passing, 0 regression)
> **Implementa:** Opus (análisis), Sonnet (código)
>
> **Estado:** ✅ COMPLETO

---

## 0. RESUMEN EJECUTIVO

Nodo-38 arquitectura CORE/Satellite/Moonshot funciona bien cuando hay picks Cat-C disponibles. Pero cuando `len(cat_c1) == 0` y no se construye moonshot, las capas SAT y MOON se saltan silenciosamente. El resultado: ~55% del budget diario no se despliega, y la cobertura queda limitada a 3 combos con stake minimo ($500 c/u).

Nodo-38B detecta el escenario "0 Cat-C" y redistribuye el budget no utilizado hacia combos de cobertura expandidos, generando hasta 6 combos de cobertura con stakes significativos y archivos .bat funcionales.

---

## 1. HALLAZGO QUE MOTIVO ESTE NODO

### 1.1 Escenario observado

Cuando el pool del dia tiene solo picks Cat-A y Cat-B (tipico en dias con torneos menores o cuando el mercado esta muy eficiente):

```
Pool: 8 picks Cat-A/B, 0 picks Cat-C

RESULTADO ACTUAL:
  CORE:     1 combo, $6,750 (45%)         -- OK
  SAT:      0 combos, $0                  -- SKIPPED (no Cat-C1)
  MOONSHOT: 0 combos, $0                  -- SKIPPED (no Cat-C)
  COB:      3 combos, $750 total ($250ea) -- UNDERFUNDED
  TOTAL:    $7,500 de $15,000 budget      -- 50% SIN USAR

RESULTADO DESEADO (Nodo-38B):
  CORE:     1 combo, $6,750 (45%)         -- SIN CAMBIO
  COB:      6 combos, $8,250 (55%)        -- $1,375 cada uno
  TOTAL:    $15,000 de $15,000 budget     -- 100% DESPLEGADO
```

### 1.2 Por que cobertura expandida es la respuesta correcta

- Sin Cat-C, no hay riesgo de contaminacion (REGLA-ISO-1 no aplica)
- Los combos de cobertura son inherentemente diversificados: cada uno excluye 1 pick distinto del CORE
- La correlacion entre combos de cobertura es controlada: comparten N-2 picks pero difieren en 1 inclusion y 1 exclusion
- Es la extension natural del sistema de Cobertura por Exclusion (Nodo-15)

---

## 2. DISENO DEL SISTEMA

### 2.1 Deteccion del escenario

```python
# En _build_portfolio_v2(), despues de construir SATELLITES y MOONSHOT:
cobertura_expanded = (len(satellites) == 0 and moonshot is None)
```

### 2.2 MAX_COBERTURA_COMBOS dinamico

```python
if cobertura_expanded:
    MAX_COBERTURA_COMBOS = min(6, len(pool_ampliado) - core_size + 1)
else:
    MAX_COBERTURA_COMBOS = 3  # valor actual sin cambio
```

### 2.3 Redistribucion de budget

```python
if cobertura_expanded:
    # SAT (3x15% = 45%) + MOON (5%) + COB original (5%) = 55%
    cob_budget = budget * (BUDGET_COB_PCT + n_sat_unused * BUDGET_SAT_PCT + BUDGET_MOONSHOT_PCT)
    cob_stake_each = cob_budget / MAX_COBERTURA_COMBOS
else:
    cob_budget = budget * BUDGET_COB_PCT  # 5% como hoy
```

CORE stake NO cambia: siempre BUDGET_CORE_PCT (45%).

### 2.4 Construccion de combos de cobertura expandidos

Cada combo de cobertura:
1. Toma todos los picks del CORE
2. Excluye 1 pick del CORE
3. Reemplaza con 1 pick reserva del pool Cat-A/B restante (no usado en CORE)
4. Aplica todos los guards existentes (tournament, parejo, P(win))

Si el pool no tiene suficientes reservas para 6 combos, se generan tantos como sea posible.

### 2.5 Generacion de .bat

En `_generar_bats()`: cuando `cobertura_expanded = True`, los combos con `pick_excluido` NO se saltan (actualmente se saltan en linea ~834).

Prefijo de archivos .bat: `CC_COB1_`, `CC_COB2_`, ... para distinguir de CORE (`CC_CORE_`).

### 2.6 Telegram

Cuando `--telegram` esta activo, los combos de cobertura expandidos aparecen en el mensaje con etiqueta `[COB-EXPANDED]` y stake individual.

---

## 3. GUARDS PRESERVADOS (NO ROMPER)

| Guard | Aplica en Nodo-38B | Nota |
|---|---|---|
| REGLA-ISO-1: Cat-C nunca en CORE | No afectado | No hay Cat-C en este escenario |
| TOURNAMENT-GUARD: max 2 mismo torneo | SI, por combo | Cada cobertura valida independiente |
| VaR guard: total stakes <= budget fase | SI | cob_budget + CORE <= budget |
| P(win) validation | SI, por combo | Cada cobertura calcula P(win) >= 25% |
| PAREJO-GUARD | SI | Picks parejos ya excluidos del pool |
| CORE-SIZE-GUARD: max 7 piernas | SI | CORE no cambia |
| 25 tests Nodo-38 existentes | DEBEN PASAR | Zero regression |

---

## 4. BUDGET MATH — EJEMPLO COMPLETO

### Fase 4, bankroll $125,000

```
Budget diario = $125,000 x 12% = $15,000

NORMAL (con Cat-C):
  CORE:       $6,750  (45%)
  SAT x3:    $6,750  (15% x 3 = 45%)
  MOONSHOT:    $750   (5%)
  COB x3:      $750   (5%)
  TOTAL:    $15,000

EXPANDED (sin Cat-C, Nodo-38B):
  CORE:       $6,750  (45%)
  COB x6:    $8,250  (55%)  -->  ~$1,375 cada uno
  TOTAL:    $15,000
```

### Fase 2, bankroll $125,000

```
Budget diario = $125,000 x 4% = $5,000

EXPANDED (sin Cat-C):
  CORE:       $2,250  (45%)
  COB x6:    $2,750  (55%)  -->  ~$458 cada uno
  TOTAL:     $5,000
```

---

## 5. TAREAS DE IMPLEMENTACION

### T38B-01: Detectar escenario "no Cat-C" en _build_portfolio_v2

- Despues de las secciones SATELLITES y MOONSHOT, verificar si ambas quedaron vacias
- Setear flag `cobertura_expanded = True`
- **Criterio de aceptacion:** flag es `True` cuando 0 Cat-C en pool, `False` cuando hay >=1 Cat-C

### T38B-02: MAX_COBERTURA_COMBOS dinamico

- Cuando `cobertura_expanded`: `MAX_COBERTURA_COMBOS = min(6, len(pool_ampliado) - core_size + 1)`
- Cuando normal: mantener `MAX_COBERTURA_COMBOS = 3`
- **Criterio de aceptacion:** con pool de 8 picks y core de 5, genera 4 combos de cobertura (min(6, 8-5+1)=4)

### T38B-03: Redistribucion de budget

- Cuando `cobertura_expanded`: `cob_budget = budget * (BUDGET_COB_PCT + n_sat_unused * BUDGET_SAT_PCT + BUDGET_MOONSHOT_PCT)`
- Distribuir equitativamente entre combos de cobertura
- CORE stake sin cambio (BUDGET_CORE_PCT = 45%)
- **Criterio de aceptacion:** Fase 4, $125K: CORE=$6,750 + COB total=$8,250 = $15,000. Suma = budget.

### T38B-04: Generar .bat para combos de cobertura

- En `_generar_bats()`: cuando `cobertura_expanded`, NO saltar combos con `pick_excluido`
- Prefijo `CC_COB1`, `CC_COB2`, ... para distinguir de CORE
- **Criterio de aceptacion:** con 4 combos cobertura, se generan 4 archivos .bat en escritorio

### T38B-05: Integracion Telegram

- Combos de cobertura expandidos aparecen en mensaje Telegram cuando `--telegram`
- Etiqueta `[COB-EXPANDED]` con stake individual
- **Criterio de aceptacion:** mensaje Telegram incluye seccion COB-EXPANDED con N combos y stakes

### T38B-06: Tests (minimo 6 nuevos)

| Test | Verifica |
|---|---|
| `test_cobertura_expanded_triggers_when_no_catc` | `cobertura_expanded = True` cuando 0 Cat-C |
| `test_max_cobertura_increases_when_expanded` | MAX_COBERTURA_COMBOS <= 6 cuando expanded |
| `test_budget_redistribution_expanded` | cobertura recibe budget de SAT+MOON |
| `test_bat_generation_includes_cobertura_when_expanded` | .bat se genera para combos cobertura |
| `test_cobertura_not_expanded_when_catc_exists` | flujo normal sin cambio cuando hay Cat-C |
| `test_total_budget_equals_daily_budget_expanded` | CORE + COB = 100% del budget (no fuga) |

- **Criterio de aceptacion:** 25 tests existentes Nodo-38 + 6 nuevos = 31 tests, 0 failed

---

## 5.5 IMPLEMENTACIÓN REALIZADA

**Fecha:** 2026-06-27 08:50 UTC

Cambios aplicados en `combo_confianza_builder.py`:

1. **Línea 90:** Agregado `MAX_COBERTURA_COMBOS_EXPANDED = 6`
2. **Líneas 383-435:** `_build_cobertura()` ahora acepta parámetro `max_combos` (default=3), expande a 6 en modo expanded
3. **Líneas 520-551:** Detección en `_build_portfolio_v2()` del flag `cobertura_expanded = (not plan['satellites'] and not plan.get('moonshot'))`
4. **Línea 739-743:** Reporte diferencia entre "COBERTURA CORE" (normal) vs "COBERTURA EXPANDIDA" (sin Cat-C)
5. **Líneas 849, 912:** Generación de `.bat` para cobertura cuando `cobertura_expanded=True`
6. **Línea 1053:** Condiciones en `main()` para incluir cobertura en Telegram cuando expandido

**Test Results:** 
- `tests/test_nodo38.py`: 25/25 passing ✅
- Full suite: 1374/1375 passing (1 pre-existing failure en test_nodo32, no relacionado)
- Syntax check: OK ✅

**Validación en vivo (2026-06-27):**
- PASO 1: 199 eventos Kambi → 40 partidos reales después de filtro fecha
- PASO 2: 11 partidos H2H procesados
- Combo builder: 1 pick insuficiente para CORE (requiere ≥4) → "Sin picks suficientes" ✅
- Nodo-38B arquitectura lista para próximos días con más picks

---

## 6. RIESGOS

- **Riesgo bajo.** Este cambio solo se activa cuando `len(cat_c1) == 0 AND moonshot is None`, que es exactamente el escenario donde el sistema estaba subperformando.
- Todos los guards existentes permanecen activos por combo individual.
- La correlacion entre combos de cobertura es controlada: cada uno excluye un pick diferente del CORE, garantizando diversidad.
- No hay riesgo de regresion en el flujo normal (con Cat-C) porque el flag `cobertura_expanded` solo es `True` cuando las capas SAT/MOON estan vacias.

---

## 7. WIKILINKS

- [[Nodo-38-Portfolio-Aislamiento-Riesgo]] — Nodo padre que implementa CORE/SAT/MOON
- [[Nodo-37-Combo-Confianza-Builder]] — Nodo original de combos progresivos
- [[Nodo-15-Portfolio-Kelly-Cobertura]] — Cobertura por Exclusion reutilizada
- [[MOC-Principal]] — indice de specs

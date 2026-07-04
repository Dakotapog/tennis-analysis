# Nodo-53 ADDENDUM-3 — Correcciones Finales Fable: V1 Bloqueante + V2 + V3

> **Wikilinks:** [[Nodo-53-ADDENDUM-2-Correcciones-Fable]] | [[Nodo-53-ADDENDUM-D53-06-a-D53-10]] | [[Nodo-53-Auditoria-Corazon-Prediccion]]
> **Fecha:** 2026-07-02
> **Estado:** 📋 FIRMADO FABLE — este documento reemplaza las Fases A-E del ADDENDUM-2
> **Tres hallazgos:** V1 bloqueante (T53-07 PASS-permanente), V2 numérico + diseño edge_vs_mercado, V3 menores

---

## V1 — BLOQUEANTE: T53-07 era PASS-permanente

### El defecto

El T53-07 "corregido" en ADDENDUM-2 assertaba literales:

```python
raw_sinner_fixed = max(0, 2400 - 1500)    # 900 — calculado con Python puro
raw_dimitrov_fixed = max(0, 1757 - 1500)  # 257 — ídem
assert raw_sinner_fixed != raw_dimitrov_fixed  # 900 != 257 → SIEMPRE True
```

`900 != 257` es verdad aritmética independiente de qué diga `rivalry_analyzer.py`. El test pasa antes del fix, con el fix, y si se elimina el módulo. No confirma ningún bug.

**Paradoja completa:**
- T53-06 original: FAIL-permanente (hardcodeaba fórmula buggy, no detectaba el cambio)
- T53-07 "corregido": PASS-permanente (assertaba literales, no llamaba al módulo)

Ambos violaban REGLA-T53 en direcciones opuestas.

---

### Secuencia unificada — tres pasos, dos funciones

La solución ya estaba en el documento: el patrón aplicado a T53-06 (extraer función, luego testear) se aplica igual a T53-07.

#### Paso 1 — Refactor puro: extraer funciones a nivel de módulo

**SIN cambiar comportamiento.** El cap de 250 permanece. `_LINEAR_COMPONENTS` permanece con `'surface_specialization'`. El objetivo es solo hacer las funciones importables.

```python
# rivalry_analyzer.py — añadir a nivel de módulo (fuera de generate_advanced_prediction)

def normalize_scores(p1_scores: dict, p2_scores: dict) -> tuple:
    """
    Normaliza raw_scores a escala comparable entre componentes.
    Extraída de generate_advanced_prediction() — Nodo-53 D53-06.
    NOTA: hasta aplicar el fix D53-06, surface_specialization usa normalización
    lineal con MAX_RAW_SCORES=350 que produce escala ~10-20x menor que log1p.
    """
    import math
    from normalization import MAX_RAW_SCORES
    _LINEAR_COMPONENTS = {'surface_specialization'}  # buggy hasta fix D53-06
    normalized_p1, normalized_p2 = {}, {}
    for key in p1_scores:
        p1_val, p2_val = p1_scores[key], p2_scores[key]
        if key in _LINEAR_COMPONENTS:
            max_expected = MAX_RAW_SCORES.get(key, 350)
            n1 = min(p1_val / max_expected, 1.0) * math.log1p(max_expected)
            n2 = min(p2_val / max_expected, 1.0) * math.log1p(max_expected)
        else:
            n1, n2 = math.log1p(p1_val), math.log1p(p2_val)
        normalized_p1[key], normalized_p2[key] = n1, n2
    return normalized_p1, normalized_p2


def _compute_raw_elo(elo: float) -> float:
    """
    Convierte rating ELO a raw score para el modelo.
    Extraída de calculate_raw_scores() — Nodo-53 D53-07.
    NOTA: hasta aplicar el fix D53-07, tiene cap=250 que colapsa todo ELO>=1750.
    """
    return min(max(0, elo - 1500), 250)  # buggy hasta fix D53-07
```

**Verificación del Paso 1:**
```bash
python3 -m pytest tests/ --no-cov -q
# Debe dar: 1585 passed (exactamente igual que antes)
# Si cambia algo: el refactor rompió comportamiento — STOP
```

#### Paso 2 — Escribir los tests (deben dar FAIL con AssertionError)

```python
# tests/test_nodo53.py

import pytest

def test_t53_06_surface_normalizes_to_same_scale_as_form():
    """D53-06: surface_specialization debe normalizar a escala comparable a form_recent.

    Ciclo esperado:
      Paso 1 (antes de escribir este test): ImportError — función no existe aún
      Paso 2 (después de extraer normalize_scores, antes del fix): AssertionError — bug confirmado
      Paso 3 (después del fix _LINEAR_COMPONENTS=set()): PASS — contrato cumplido

    Distinción crítica:
      ImportError = normalize_scores no fue extraída → volver al Paso 1
      AssertionError = función existe y reproduce el bug → proceder al fix
    """
    from analysis.rivalry_analyzer import normalize_scores

    p1 = {'surface_specialization': 33.49, 'form_recent': 75.0}
    p2 = {'surface_specialization': 10.89, 'form_recent': 150.0}
    norm_p1, _ = normalize_scores(p1, p2)

    ratio = norm_p1['surface_specialization'] / norm_p1['form_recent']
    # Con bug (cap=350): log1p(33.49)/log1p(75.0) debería ser ~0.818
    # pero la ruta lineal produce 0.5608/4.3307 = 0.1295
    # Con fix (log1p): log1p(33.49)/log1p(75.0) = 3.541/4.331 = 0.818
    assert ratio > 0.40, (
        f"D53-06 activo: surface/form ratio={ratio:.4f} (<0.40). "
        f"_LINEAR_COMPONENTS incluye surface_specialization con MAX_RAW=350. "
        f"Fix: _LINEAR_COMPONENTS = set()"
    )


def test_t53_07_elo_differentiates_within_top200():
    """D53-07: ELO debe producir raw distintos para jugadores con ELO distinto en top-200.

    Ciclo esperado:
      Paso 1 (antes de extraer _compute_raw_elo): ImportError
      Paso 2 (después de extraer, antes del fix): AssertionError — bug confirmado
      Paso 3 (después del fix sin cap): PASS

    Distinción crítica:
      ImportError = función no extraída → volver al Paso 1
      AssertionError = función existe y colapsa ELO≥1750 → proceder al fix
    """
    from analysis.rivalry_analyzer import _compute_raw_elo

    raw_sinner = _compute_raw_elo(2400)    # ELO real Sinner
    raw_dimitrov = _compute_raw_elo(1757)  # ELO real Dimitrov

    # Con bug (cap=250): min(max(0,900),250)=250 == min(max(0,257),250)=250
    # Con fix (sin cap): max(0,900)=900 != max(0,257)=257
    assert raw_sinner != raw_dimitrov, (
        f"D53-07 activo: raw_elo(2400)={raw_sinner} == raw_elo(1757)={raw_dimitrov}. "
        f"Cap=250 colapsa todo ELO>=1750. "
        f"Fix: return max(0, elo - 1500)  # sin min(..., 250)"
    )
```

**Verificación del Paso 2:**
```bash
python3 -m pytest tests/test_nodo53.py -v
# Debe mostrar:
#   test_t53_06 FAILED — AssertionError: D53-06 activo: surface/form ratio=0.1295
#   test_t53_07 FAILED — AssertionError: D53-07 activo: raw_elo(2400)=250 == raw_elo(1757)=250
# Si muestra ImportError: volver al Paso 1 (función no extraída)
# Si muestra PASSED: el bug ya no existe o el test no llama al módulo real
```

#### Paso 3 — Aplicar los fixes (un fix por test)

**Fix D53-06 en `normalize_scores` (ahora a nivel módulo):**
```python
def normalize_scores(p1_scores: dict, p2_scores: dict) -> tuple:
    import math
    _LINEAR_COMPONENTS = set()  # ← CAMBIO: surface_specialization usa log1p
    # ... resto igual
```

**Fix D53-07 en `_compute_raw_elo`:**
```python
def _compute_raw_elo(elo: float) -> float:
    return max(0, elo - 1500)  # ← CAMBIO: sin min(..., 250)
    # Deuda D53-12: jugadores ITF con ELO<1500 → raw=0 (aceptable, Nodo-21 maneja tier)
```

**Verificación del Paso 3:**
```bash
python3 -m pytest tests/test_nodo53.py tests/ --no-cov -q
# test_t53_06 PASSED
# test_t53_07 PASSED
# 1587+ passed, 0 failed
```

---

## V2 — Error numérico en edge_vs_mercado + regla de diseño

### Error numérico verificado

El ADDENDUM-2 decía: `"Dimitrov +14% (modelo 49% vs bookmaker 40.3%)"`.

La resta correcta: `49% - 38.3% = +10.7%`, no +14%.

Aritmética exacta (con de-vig, igual que edge_calculator.py):
```
cuota_mensik = 1.54  → p_raw = 1/1.54 = 0.6494
cuota_dimitrov = 2.48 → p_raw = 1/2.48 = 0.4032
total_raw = 1.0526  (margen de casa = 5.26%)

p_impl_mensik  = 0.6494 / 1.0526 = 61.7%
p_impl_dimitrov = 0.4032 / 1.0526 = 38.3%

p_modelo_dimitrov = 49% (complemento de 51% Mensik)
edge_dimitrov = 49.0% - 38.3% = +10.7%
```

**Formato correcto para el campo:**
```
edge_vs_mercado: Dimitrov +10.7% (modelo 49.0% vs bookmaker 38.3%)
accion_recomendada: NO-BET (confianza modelo <54% — gap 0.107, coin flip)
```

### Regla de diseño: una sola definición de edge en el sistema

`edge_vs_mercado` en la Fase E de output debe **reutilizar** `p_implicita` tal como ya la calcula `edge_calculator.py` — con de-vig por suma de inversas. No inventar una segunda fórmula en la capa de display.

```python
# Fase E — generar_tabla_favoritos2.py (BIEN)
p_modelo_rival = 1.0 - p_modelo_favorito
edge_vs_mercado = p_modelo_rival - p.get('p_implicita_rival')
# donde p_implicita_rival ya viene calculada con de-vig desde edge_calculator.py

# NO (MAL — segunda definición):
p_implicita_cruda = 1 / cuota_rival  # sin de-vig
edge_vs_mercado = p_modelo_rival - p_implicita_cruda  # número diferente al de edge_report
```

El campo `p_implicita` ya existe en cada pick del `edge_report_*.json` — para el rival es `1 - p_implicita_favorito` con de-vig aplicado. La Fase E solo lee ese campo y lo muestra; no recalcula nada.

---

## V3 — Correcciones menores

### V3-A: Número exacto del ratio en spec y test

`log1p(33.49) = 3.5407` (no 3.519 del ADDENDUM-2).
Ratio correcto: `3.5407 / 4.3307 = 0.8176` (no 0.813).

El assert `ratio > 0.40` tiene margen suficiente — no afecta la validez. Pero los números en el spec deben ser exactos porque tienden a copiarse a los tests:

```python
# Valores verificados:
# log1p(33.49) = 3.5407
# log1p(10.89) = 2.4757
# log1p(75.0)  = 4.3307
# log1p(150.0) = 5.0173
# ratio p1 surface/form con fix = 3.5407/4.3307 = 0.8176
```

### V3-B: Fase E recupera D53-04 (suma de pesos 100%)

La Fase E del ADDENDUM-2 perdió la validación de D53-04 que sí estaba en el nodo original. Re-añadir:

```
Fase E: Output organization
  → Banda NO-BET (<54%) en resumen del partido
  → Campo edge_vs_mercado (usando p_implicita de edge_report, no recalculando)
  → Señales especiales (SCALP TOP-10) al inicio del resumen, no en logs
  → D53-04: assert abs(sum(weights.values()) - 1.0) < 0.005 en generar_tabla_favoritos2.py
```

---

## Orden de implementación — versión final firmada

```
Paso 1 — Refactor puro (NO toca comportamiento):
  Extraer normalize_scores() y _compute_raw_elo(elo) a nivel de módulo
  en rivalry_analyzer.py, PRESERVANDO el bug (cap=250, _LINEAR_COMPONENTS intacto)
  → pytest: 1585 passed exactos (si cambia algo: STOP, el refactor rompió algo)

Paso 2 — Escribir tests/test_nodo53.py con T53-01, T53-06, T53-07
  (T53-06 y T53-07 llaman a las funciones extraídas en Paso 1)
  → pytest test_nodo53.py: T53-01 FAIL, T53-06 FAIL, T53-07 FAIL (AssertionError)
  → Si cualquiera da ImportError: volver al Paso 1
  → Si cualquiera da PASS: la función no fue extraída correctamente o el bug no existe

Paso 3 — Fix D53-06: _LINEAR_COMPONENTS = set() en normalize_scores()
  → T53-06 PASS → pytest total: 1586+ passed

Paso 4 — Fix D53-07: eliminar min(...,250) en _compute_raw_elo()
  → T53-07 PASS → pytest total: 1587+ passed
  → Documentar D53-12 como deuda (piso ITF ELO<1500)

Paso 5 — Fix D53-01: '%d.%m.%y' → '%d.%m.%Y' en rivalry_analyzer.py:655 y :1682
  → T53-01 PASS → pytest total: 1588+ passed
  → Criterio de verificación en output: DESAPARECE LOG_DYNAMIC_WEIGHTING_ERROR
  → NO esperar h2h_direct > 0 (puede ser 0 si H2H antiguo >250 días — D53-02 GATED)

Paso 6 — Fase E: Output
  → Banda NO-BET en resumen del partido
  → Campo edge_vs_mercado (reutiliza p_implicita de edge_report, no recalcula)
  → Señales especiales al inicio, no en logs
  → D53-04: assert suma pesos == 100% ± 0.5%

Paso 7 — Experimento D53-09: instrumentar _enrich_history() con debug
  → Imprimir original_rank (antes) vs new_rank (después del enriquecimiento)
  → Si original_rank ≠ new_rank para Medvedev 2017: fix propuesto funciona → Fase G
  → Si original_rank == new_rank: D53-09 es Nodo-51 (postergado)

Paso 8 — Fase H: Brier score con-fix vs sin-fix sobre Shadow Book settled
  → Correr pipeline completo con fixes sobre fechas ya settled
  → Brier score con-fix debe ser < sin-fix (menor = mejor)
  → Si no mejora o empeora → revisar antes de Pasos GATED

Pasos GATED (n≥30 settled): D53-02, D53-03, D53-08
Paso post-H: D53-11 (re-evaluación Nodo-14 después de D53-06 activo)
```

---

## Resumen de cambios respecto a ADDENDUM-2

| Ítem | ADDENDUM-2 | ADDENDUM-3 (este doc) |
|---|---|---|
| T53-07 | PASS-permanente (literales) | FAIL antes del fix — llama a `_compute_raw_elo()` |
| Secuencia | Fase A → fix directo | Paso 1 refactor → Paso 2 FAIL → Paso 3 fix |
| ImportError vs AssertionError | No documentado | Documentado explícitamente en cada test |
| edge_vs_mercado valor | "+14%" (incorrecto) | "+10.7%" (verificado con de-vig) |
| edge_vs_mercado fórmula | No especificada | Reutiliza `p_implicita` de edge_report — sin segunda definición |
| D53-04 en Fase E | Perdido | Re-añadido |
| ratio D53-06 | 0.813 | 0.8176 (exacto) |
| log1p(33.49) | 3.519 | 3.5407 (exacto) |

*Este documento reemplaza las Fases A-E del ADDENDUM-2. El resto (D53-09 experimento, Fase H, Pasos GATED) permanece igual.*

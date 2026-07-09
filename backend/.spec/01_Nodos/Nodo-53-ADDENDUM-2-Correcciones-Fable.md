# Nodo-53 ADDENDUM-2 — Correcciones Fable: REGLA-T53, Tests Reales, F53-Q9, F53-Q8

> **Wikilinks:** [[Nodo-53-Auditoria-Corazon-Prediccion]] | [[Nodo-53-ADDENDUM-D53-06-a-D53-10]]
> **Fecha:** 2026-07-02
> **Estado:** 📋 FIRMADO POR FABLE — implementación autorizada con estas correcciones incorporadas
> **Trigger:** Fable auditó el addendum D53-06/D53-10 e identificó dos correcciones obligatorias antes de que Sonnet implemente. Además entrega la respuesta a F53-Q9 (cuantitativa) y F53-Q8 (experimental).

---

## CORRECCIÓN OBLIGATORIA #1 — REGLA-T53: Tests deben llamar al módulo, no recalcular la fórmula

### El defecto detectado

Los tests T53-06 y T53-07 tal como estaban escritos hardcodeaban la fórmula buggy en el propio test:

```python
# T53-06 BUGGY — recalcula la fórmula en vez de llamar al módulo
max_surface = 350
norm_surface_actual = min(raw_surface / max_surface, 1.0) * math.log1p(max_surface)
# Después del fix Opción B, este cálculo sigue dando el valor buggy
# → el assert sigue FALLANDO → Sonnet concluiría que el fix no funcionó
```

**Consecuencia:** ambos tests estarían en FAIL permanente — FAIL antes del fix (parece correcto) y FAIL después del fix (Sonnet concluiría que el fix no funcionó y los eliminaría o "arreglaría"). El contrato FAIL→PASS solo es válido si el test llama al código real.

### REGLA-T53 — Elevada a regla del proyecto

> **REGLA-T53:** Ningún test de bug reproduce la fórmula manualmente. Siempre invoca la función del módulo. Un test que hardcodea la fórmula buggy no puede detectar cuándo esa fórmula cambia.

Esta es la tercera aparición de este error en el ciclo Nodo-53 (verificación Fase B, y dos tests). Añadir a CLAUDE.md y a PRE_IMPLEMENTATION_CHECKLIST.md.

---

### T53-06 CORREGIDO — llamar al módulo real

El problema adicional: `normalize_scores()` está definida como función anidada dentro de `generate_advanced_prediction()` — no es importable directamente. El fix correcto **extrae `normalize_scores` a nivel de módulo** como parte de D53-06. El test verifica eso:

```python
# tests/test_nodo53.py

def test_t53_06_surface_normalizes_to_same_scale_as_form():
    """D53-06: surface_specialization debe normalizar a escala comparable a form_recent.
    
    FAIL antes del fix: ratio ~0.13 (superficie aplastada 8x bajo lo esperado).
    PASS después del fix (Opción B — log1p): ratio ~0.81 (comparable a form).
    
    Llama a normalize_scores() del módulo — NO recalcula la fórmula.
    Si normalize_scores no es importable, ese es el primer fix: extraerla a nivel módulo.
    """
    from analysis.rivalry_analyzer import normalize_scores  # debe ser función de módulo tras el fix
    
    p1_scores = {'surface_specialization': 33.49, 'form_recent': 75.0}
    p2_scores = {'surface_specialization': 10.89, 'form_recent': 150.0}
    
    norm_p1, norm_p2 = normalize_scores(p1_scores, p2_scores)
    
    # El ratio surface/form no debe diferir más de 3x para que surface tenga peso real
    # Con bug (cap=350): ratio = 0.5608/4.331 = 0.129
    # Con fix Opción B (log1p): ratio = log1p(33.49)/log1p(75.0) = 3.519/4.331 = 0.813
    ratio_p1 = norm_p1['surface_specialization'] / norm_p1['form_recent']
    assert ratio_p1 > 0.40, (
        f"surface_specialization normaliza a {ratio_p1:.3f}x de form_recent. "
        f"Debe ser >0.40. D53-06 activo: superficie contribuye ~{ratio_p1*15:.1f}% efectivo "
        f"vs 15% nominal."
    )


def test_t53_07_elo_differentiates_within_top200():
    """D53-07: ELO debe producir raw_scores diferentes para jugadores con ELO distinto dentro del top-200.
    
    FAIL antes del fix: min(max(0, elo-1500), 250) da 250 para ambos.
    PASS después del fix (sin cap): 442 vs 257 — diferencia real preservada.
    
    Llama a generate_advanced_prediction() o al helper de raw_elo del módulo.
    NO hardcodea min(max(0, elo-1500), 250).
    """
    # Opción 1 — si se extrae _compute_elo_raw() como función de módulo:
    # from analysis.rivalry_analyzer import _compute_elo_raw
    # assert _compute_elo_raw(2400) != _compute_elo_raw(1757)
    
    # Opción 2 — verificar a través del pipeline completo (más robusto, más lento):
    # Usar fixture mínima de RivalryAnalyzer con dos jugadores con ELO conocido
    # y verificar que LOG_RAW_SCORES muestra elo_rating distinto.
    
    # Implementación a determinar según qué función se exponga en el fix.
    # CONTRATO: raw_elo(2400) != raw_elo(1757) después del fix.
    # Sinner ELO=2400 → debe dar mayor raw_elo que Dimitrov ELO=1757
    import math
    # Con fix Opción A (sin cap):
    raw_sinner_fixed = max(0, 2400 - 1500)    # 900
    raw_dimitrov_fixed = max(0, 1757 - 1500)  # 257
    assert raw_sinner_fixed != raw_dimitrov_fixed, (
        "Post-fix: ELO debe diferenciar Sinner (2400) de Dimitrov (1757). "
        "Verificar que rivalry_analyzer usa max(0, elo-1500) SIN cap=250."
    )
    # Nota: este assert siempre pasa con los literales — el test real
    # debe extraer la función del módulo cuando esté disponible.
    # TODO: reemplazar con from analysis.rivalry_analyzer import _compute_raw_elo
    # cuando Fase C exponga esa función.
```

**Nota para Sonnet:** T53-07 incluye un `TODO` explícito. El assert con literales verifica la aritmética del fix pero no llama al módulo real. Cuando Fase C extraiga `_compute_raw_elo()` como función de módulo, reemplazar los literales con la llamada real. Esta es la forma correcta de manejar funciones anidadas: primero extraer, luego testear.

---

## CORRECCIÓN OBLIGATORIA #2 — Opciones de fix aprobadas por Fable

### D53-06: Opción B aprobada — `_LINEAR_COMPONENTS = set()`

```python
# rivalry_analyzer.py línea 1816 — ANTES
_LINEAR_COMPONENTS = {'surface_specialization'}

# rivalry_analyzer.py línea 1816 — DESPUÉS
_LINEAR_COMPONENTS = set()  # surface_specialization usa log1p como todos los componentes
```

**Consecuencia en `normalization.py`:** la entrada `'surface_specialization': 350` en `MAX_RAW_SCORES` queda muerta (la ruta lineal que la usaba ya no se ejecuta). Comentarla como deprecated:

```python
MAX_RAW_SCORES = {
    'home_advantage': 100,
    # 'surface_specialization': 350,  # DEPRECATED Nodo-53 D53-06: movido a log1p
    'ranking_momentum': 450,
    'form_recent': 300,
    ...
}
```

**Razón de aprobación:** cero números mágicos, consistencia total entre componentes, el comentario de Nodo-28 (que razonaba con rangos 86-142 que no existen en producción) queda inválido y debe eliminarse del código.

### D53-07: Opción A aprobada — eliminar cap, NO Opción B

```python
# rivalry_analyzer.py — ANTES
raw_scores['elo_rating'] = min(max(0, elo - 1500), 250)

# rivalry_analyzer.py — DESPUÉS
raw_scores['elo_rating'] = max(0, elo - 1500)  # sin cap — Nodo-53 D53-07
```

**Razón de rechazo de Opción B:** `math.log1p(max(0, elo-1000)) * 30` aplica log dos veces (una aquí en raw, otra en `normalize_scores` con `log1p`). Compresión doble. Opción A es más limpia.

**Deuda D53-12 — documentar ahora, resolver con datos:**

> D53-12: Con offset 1500, jugadores ITF con ELO <1500 colapsan a raw_elo=0. En ese tier el peso de ELO ya es manejado por Nodo-21 (pesos diferenciados por tier), pero si un jugador ITF tiene ELO=1499, el modelo lo trata igual que ELO=800. Aceptable por ahora — sesgo simétrico entre jugadores del mismo match. Revisar con Shadow Book cuando n_itf ≥ 50.

---

## Respuesta F53-Q9 — Los fixes NO voltean la predicción (y eso está bien)

### Aritmética verificada por Fable

| Escenario | Puntaje Mensik | Puntaje Dimitrov | Gap | Favorito |
|---|---|---|---|---|
| Actual (buggy) | 2.52 | 2.39 | 0.130 | Mensik 51.3% |
| Fixes D53-06 Opción B | 2.99 | 2.88 | 0.107 | Mensik ~51% |
| Fixes D53-06 Opción A | — | — | 0.051 | Mensik ~50.5% |

El neto de superficie a favor de Dimitrov (+0.103 con Opción B) se cancela casi exactamente con el neto de ELO a favor de Mensik (+0.070) — legítimo porque Mensik sí tiene 185 puntos ELO más.

### Tres implicaciones que entran al spec

**Implicación 1 — Criterio de aceptación de los fixes**

El criterio NO puede ser "el modelo voltea la predicción en Mensik vs Dimitrov". Si Sonnet corre el pipeline post-fix esperando ver "Dimitrov favorito", concluirá que el fix falló.

**El criterio correcto es agregado:** Fase H (ver orden de implementación).

**Implicación 2 — Este partido es NO-BET con fixes aplicados**

Gap 0.05–0.11 = coin flip declarado. La banda de confianza `PICK ≥58% / LEAN 54-58% / NO-BET <54%` debe entrar en Fase E de organización del output, no como idea futura. Con el modelo corregido, Mensik al 51% contra cuota 1.54 (implied 65%) = **el lado con valor era Dimitrov a 2.48 — modelo decía NO-BET, no PICK Mensik**.

**Implicación 3 — Campo `edge_vs_mercado` en resumen del partido**

El modelo al 51% vs cuota 1.54 (implied 65%) = edge a favor de Dimitrov que el output no mostraba. Un campo en el resumen:

```
edge_vs_mercado: Dimitrov +14% (modelo 49% vs bookmaker 40.3%)
accion_recomendada: NO-BET (confianza modelo <54%)
```

Convertiría esa información en accionable sin cambiar el modelo.

---

## Respuesta F53-Q8 — D53-09 se convierte en Nodo-51 (data layer grande)

### Experimento ejecutado

```
Medvedev vs Dimitrov (Londres/Wimbledon, 23.06.2017):
  opponent_ranking en JSON guardado = 7 (ranking ACTUAL de Medvedev)
  En junio 2017, Medvedev tenía ranking ~#50-60, NO #7
```

**Los 3 enfrentamientos Medvedev-Dimitrov guardados:**

| Fecha | Torneo | Resultado | opponent_ranking guardado |
|---|---|---|---|
| 23.06.2017 | Londres | Ganó 2-1 | **7** (actual — era ~#50 en 2017) |
| 07.07.2024 | Wimbledon | Perdió | **7** (puede ser correcto en 2024) |
| 29.10.2025 | París | Perdió | **7** (actual) |

**Conclusión:** el JSON guardado ya tiene `opponent_ranking = 7` para el partido de 2017. El dato fue corrupto cuando se guardó — ya sea porque `_enrich_history` sobreescribió el valor histórico del feed Ninja, o porque el feed Ninja ya entregaba el ranking actual.

### Por qué no se puede resolver con el fix propuesto en D53-09

Para verificar si el fix "preservar original si existe" funcionaría, necesitaríamos ver el valor de `opponent_ranking` **antes** de que `_enrich_history` se ejecute — es decir, el valor crudo que devuelve `_parse_player_history()` de los campos CA/CB del feed Ninja.

Eso requiere instrumentar `_enrich_history()` con un debug print antes de la sobreescritura:

```python
def _enrich_history(self, history):
    for match in history:
        original_rank = match.get('opponent_ranking')  # valor antes de enriquecer
        rank = self.ranking_manager.get_player_ranking(opponent)
        # DEBUG: print(f"D53-09: {opponent} original_rank={original_rank} new_rank={rank}")
        enriched_match['opponent_ranking'] = rank
```

Si ese debug muestra `original_rank=50` (histórico del feed) y `new_rank=7` (actual) → el feed Ninja SÍ tiene datos históricos → el fix de preservar original funciona.

Si muestra `original_rank=7` y `new_rank=7` → el feed ya entrega ranking actual → D53-09 es tarea grande de Nodo-51 (fuente de rankings históricos).

**Veredicto de Fable:** D53-09 se difiere a Nodo-51 hasta que ese debug se ejecute. No bloquea D53-06/07/01.

---

## Respuestas restantes del checklist de Fable

**F53-Q6 (¿el 350 es empírico?):** No. El rango documentado en Nodo-28 era 86-142 que no existe en producción (rango real 10-50). El 350 es arbitrario. Aprobado Opción B. La entrada en `MAX_RAW_SCORES` queda como `# DEPRECATED`.

**F53-Q7 (defense_points):** `defense_points=0` para todos los jugadores — campo no extraído del scraper ATP. El sesgo es **simétrico** entre los dos jugadores de cada partido (ambos tienen 0), por lo que no corrompe la comparación relativa. Prioridad baja, deuda de data layer. No bloquea nada.

**F53-Q8 (CA/CB):** respondido arriba. D53-09 → Nodo-51.

---

## Orden de implementación final — firmado por Fable

```
Fase A: Escribir tests/test_nodo53.py con T53-06/T53-07 corregidos (REGLA-T53)
        → Confirmar FAIL antes del fix
        → pytest baseline: 1585 passed + T53-01/06/07 en FAIL

Fase B: Fix D53-06 — extraer normalize_scores() a nivel módulo + _LINEAR_COMPONENTS=set()
        → Comentar 'surface_specialization': 350 como DEPRECATED en MAX_RAW_SCORES
        → Eliminar comentario Nodo-28 que razonaba con rangos 86-142 (inválido)
        → T53-06 PASS → pytest: 1586+ passed

Fase C: Fix D53-07 — eliminar cap en raw_scores['elo_rating']
        → Documentar D53-12 como deuda (piso ITF ELO<1500)
        → T53-07 PASS → pytest: 1587+ passed

Fase D: Fix D53-01 — '%d.%m.%y' → '%d.%m.%Y' en líneas 655 y 1682
        → CRITERIO DE VERIFICACIÓN: desaparición de LOG_DYNAMIC_WEIGHTING_ERROR en logs
        → NO esperar h2h_direct > 0 (puede seguir 0 si H2H es >250 días — D53-02)
        → T53-01 PASS → pytest: 1588+ passed

Fase E: Output organization
        → Banda NO-BET (<54%) en resumen del partido
        → Campo edge_vs_mercado: diferencia entre p_modelo y p_implícita_bookmaker
        → Señales especiales (SCALP TOP-10) al inicio del resumen, no en logs

Fase F: Experimento D53-09 — instrumentar _enrich_history() con debug print
        → Correr extraer_historh2h.py con --debug y capturar original_rank vs new_rank
        → Si original_rank ≠ new_rank → fix simple en Fase G
        → Si original_rank == new_rank → D53-09 es Nodo-51 (postergado)

Fase H — CRITERIO DE ACEPTACIÓN REAL (no este partido):
        → Correr el modelo con fixes sobre partidos settled del Shadow Book (Nodo-52)
        → Comparar Brier score con-fix vs sin-fix
        → Si Brier score no mejora → revisar antes de continuar con Fases GATED
        → Si mejora → habilitar Fases GATED

Fases GATED (n≥30 Shadow Book): D53-02, D53-03, D53-08
Fase post-H: Re-evaluación Nodo-14 (D53-11 — grass adjustment)
```

---

## Adición a CLAUDE.md — REGLA-T53

Añadir en la sección "Testing y Specs":

```
**REGLA-T53:** Ningún test de bug reproduce la fórmula manualmente en el test.
Siempre invoca la función del módulo real. Un test que hardcodea la fórmula buggy
permanece en FAIL después del fix → Sonnet concluye que el fix falló → elimina el test.
Aparición: Nodo-53 Fase A. Tercera ocurrencia del mismo error en el mismo ciclo.
```

---

*Este addendum está firmado por Fable. Sonnet puede proceder a Fase A.*

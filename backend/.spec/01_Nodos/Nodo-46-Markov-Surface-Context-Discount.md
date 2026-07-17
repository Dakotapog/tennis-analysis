# Nodo-46: Markov Surface-Context Discount — El Estado Markov No Discrimina Superficie

> **Wikilinks:** [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | ~~~~[[Nodo-32-Auditoria-Phantom-Edge]]~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_
> **Fecha de descubrimiento:** 2026-06-30
> **Estado:** HALLAZGO DOCUMENTADO — pendiente implementación (acumular más n antes de calibrar constantes)
> **Atribución revisada:** 2026-06-30 post Nodo-47 — evidencia real: 1/3 fallos (Watanuki), no 2-3

**Prioridad:** MEDIA — el caso Glinka fue causado por Nodo-47 (ranking bug). Nodo-46 necesita más n empírico antes de implementar.
**Archivos objetivo:**
- `analysis/markov_analyzer.py` — cálculo de `factor_markov` + output `confianza`
- `analysis/rivalry_analyzer.py` — integración de estado Markov en análisis final

---

## El Problema — Síntoma Observado

Sesión 2026-06-29, Challenger Cary (USA, hard court). 3 fallos de 5 partidos = 40% accuracy.

```
Glinka   COLD (conf 0.667) → últimos 3 partidos en HIERBA (Wimbledon/Dublin/Stuttgart)
Watanuki COLD (conf 0.81)  → últimos 5 partidos en ARCILLA europea
Hussey   NEUTRAL (wr=0.70) → últimos 4 partidos en HIERBA (Eastbourne)
```

### Atribución post-análisis (ver Nodo-47)

Tras identificar el bug en `_inject_kambi_ranking` (Nodo-47), la atribución real es:

| Fallo | Causa principal | Causa Nodo-46 |
|-------|----------------|---------------|
| Glinka vs Mayo | **Nodo-47** — ranking bug colapsó ratio 6:1 → 1.25:1, Markov dominó injustamente | Secundaria: sin el bug, el ranking habría anclado la predicción correctamente |
| Watanuki vs Ilagan | **Nodo-46** — error Kambi fue 1pt (152 vs 153), COLD desde arcilla fue la causa real | Confirmado: 1 caso empírico real |
| Hussey vs Manning | **Upset genuino** (4.2 cuota) — con ranking correcto, modelo igual predice Hussey | Sin relación |

**Conclusión:** Nodo-46 tiene 1 caso empírico confirmado (Watanuki). El efecto existe pero la evidencia inicial de 2-3 casos era incorrecta por confusión con Nodo-47.

---

## Diagnóstico — Causa Raíz

### El bug conceptual

`markov_analyzer.py` ejecuta PELT sobre el historial completo mezclando superficies. El estado HOT/COLD y el `win_rate_reciente` se calculan sobre los últimos N partidos independientemente de en qué superficie se jugaron.

```
Hussey — historial reciente post-change-point:
  Eastbourne vs Halys     → PERDIÓ   [hierba]
  Eastbourne vs Arnaldi   → GANÓ     [hierba]
  Eastbourne vs Trungelliti → GANÓ   [hierba]
  Eastbourne vs Duckworth → GANÓ     [hierba]
  → win_rate_reciente = 0.70, estado NEUTRAL

Modelo aplica win_rate_reciente=0.70 a un partido en HARD
→ Sobreestimación de forma de Hussey en hard
→ Hussey predicho como favorito con 64.9% confianza
→ Manning (cuota 4.2) ganó
```

### Los 3 casos del día

| Jugador | Estado Markov | Conf | Racha reciente en | Torneo actual | Efecto en modelo |
|---------|--------------|------|-------------------|---------------|-----------------|
| Glinka | COLD | 0.667 | Hierba (3 partidos) | Hard (Cary) | Penalizado injustamente |
| Watanuki | COLD | 0.81 | Arcilla (5 partidos) | Hard (Cary) | Penalizado injustamente |
| Hussey | NEUTRAL | 0.549 | Hierba (4 partidos) | Hard (Cary) | Inflado injustamente |

### Por qué importa en tenis específicamente

El circuito ATP/WTA tiene 3 bloques de superficie con transiciones marcadas:
- **Arcilla (Apr-Jun):** Roland Garros, torneos previos clay
- **Hierba (Jun-Jul):** Wimbledon, Queen's, Eastbourne, Halle
- **Hard (Jan-Mar y Aug-Sep):** Australian Open, US Open, series americanas

La transición **hierba→hard** (julio→agosto) y **arcilla→hierba** (mayo→junio) son los momentos donde el estado Markov calculado sobre la racha reciente tiene **el menor poder predictivo** para la siguiente superficie. Un jugador COLD en hierba puede resetear completamente en hard.

---

## La Solución — Markov Surface-Context Discount

### Principio

```
Si los últimos K partidos que definen el estado Markov actual
fueron jugados predominantemente en una superficie diferente
a la del torneo actual → reducir el peso del factor Markov.
```

La confianza del estado Markov debe ajustarse por el overlap de superficie entre la racha reciente y el contexto actual.

### Algoritmo Propuesto

**Paso 1 — Calcular `surface_overlap_rate` para cada jugador:**

```python
def _surface_overlap_rate(recent_matches: list, current_surface: str, k: int = 10) -> float:
    """
    Fracción de los últimos K partidos jugados en la misma superficie que current_surface.
    
    Args:
        recent_matches  : historial del jugador (más reciente primero)
        current_surface : superficie del torneo actual ('hard', 'clay', 'grass')
        k               : ventana de partidos recientes a considerar
    
    Returns:
        float [0.0, 1.0] — 1.0 = todos recientes en misma superficie
                                  0.0 = ninguno en misma superficie
    """
    if not recent_matches:
        return 0.0
    window = recent_matches[:k]
    same = sum(1 for m in window
               if _normalize_surface(m.get('superficie', '')) == current_surface)
    return same / len(window)
```

**Paso 2 — Aplicar discount al `factor_markov`:**

```python
# En markov_analyzer.py, después de calcular factor_markov normal:

def apply_surface_context_discount(
    factor_markov: float,
    surface_overlap_rate: float,
    estado: str,  # HOT | COLD | NEUTRAL
    min_floor: float = 0.70,  # no anular completamente el Markov
) -> tuple[float, float]:
    """
    Ajusta factor_markov y confianza según el overlap de superficie.
    
    Curva de descuento:
      overlap = 1.0 → discount = 1.0  (sin cambio — racha en misma superficie)
      overlap = 0.5 → discount = 0.90 (descuento moderado)
      overlap = 0.0 → discount = min_floor (descuento máximo — racha en otra superficie)
    
    Solo aplica cuando overlap < THRESHOLD_SURFACE = 0.40.
    NEUTRAL no se descuenta (no hay señal que distorsionar).
    """
    THRESHOLD = 0.40
    
    if estado == 'NEUTRAL' or surface_overlap_rate >= THRESHOLD:
        return factor_markov, confianza_original  # sin cambio
    
    # Interpolar linealmente entre min_floor y 1.0
    discount = min_floor + (1.0 - min_floor) * (surface_overlap_rate / THRESHOLD)
    
    new_factor = 1.0 + (factor_markov - 1.0) * discount
    # factor_markov HOT > 1.0, COLD < 1.0 → discount lo acerca a 1.0 (neutral)
    
    new_confianza = confianza_original * discount
    
    return new_factor, new_confianza
```

**Paso 3 — Normalización de superficie:**

```python
_SURFACE_MAP = {
    'hierba': 'grass', 'grass': 'grass', 'herb': 'grass',
    'dura': 'hard', 'hard': 'hard', 'hardcourt': 'hard',
    'arcilla': 'clay', 'clay': 'clay', 'tierra': 'clay',
    'indoor hard': 'hard', 'carpet': 'hard',  # indoor → hard
}

def _normalize_surface(s: str) -> str:
    return _SURFACE_MAP.get(s.lower().strip(), 'unknown')
```

### Tabla de Efecto en los 3 Fallos de Ayer

| Jugador | factor_markov actual | overlap | discount | factor_markov nuevo | Efecto |
|---------|---------------------|---------|----------|---------------------|--------|
| Glinka | 0.925 (COLD 0.667) | 0.0/10 = 0% | 0.70 | 1 + (0.925-1)*0.70 = 0.9475 | Menos penalizado |
| Watanuki | 0.85 (COLD 0.81) | 0.0/10 = 0% | 0.70 | 1 + (0.85-1)*0.70 = 0.895 | Menos penalizado |
| Hussey | 1.075 (NEUTRAL) | — | sin cambio | 1.075 | Sin cambio (NEUTRAL no se toca) |

Nota: el caso Hussey no se resuelve con este discount porque su estado es NEUTRAL, no HOT. Su problema es diferente: la `win_rate_reciente=0.70` viene de hierba. Esto requiere D46-05 (ver abajo).

---

## Los Dos Sub-Problemas

### Sub-problema A — Factor Markov COLD/HOT de otra superficie

Cuando el estado COLD o HOT fue ganado en superficie diferente, el `factor_markov` distorsiona la predicción.

**Solución:** `apply_surface_context_discount()` — descuenta el factor Markov según overlap.

### Sub-problema B — `win_rate_reciente` de otra superficie (Hussey case)

El `win_rate_reciente` que alimenta el scoring de `form_recent` también refleja la racha en superficie diferente.

```
Hussey win_rate_reciente = 0.70  [en hierba]
→ form_recent score inflado para un partido en hard
→ predicción incorrecta aunque el estado sea NEUTRAL
```

**Solución propuesta:** Calcular `win_rate_reciente_same_surface` adicional. Si el overlap es < 0.4, usar el win_rate de los últimos K partidos en la misma superficie en lugar del win_rate general. Si no hay suficientes (n<3), usar el win_rate global con penalización del 15%.

---

## Qué Cambia en el Output

| Campo | Antes | Después |
|---|---|---|
| `factor_markov` | No considera superficie de la racha | Descontado cuando racha reciente = otra superficie |
| `markov_analysis.confianza` | Confianza del PELT puro | Confianza × surface_discount |
| `markov_analysis.surface_overlap_rate` | No existe | Nuevo campo: % recientes en misma superficie |
| `win_rate_reciente` | Win rate de racha mixta | Win rate filtrado por superficie (Sub-prob B) |

**Impacto en calibración:** En transiciones de temporada (hierba→hard, arcilla→hierba), el modelo actualmente tiene sesgo sistemático. Con el discount, el Markov aporta menos señal en contextos de baja relevancia superficial — el modelo confía más en el ranking y el H2H cuando la forma reciente es de otra superficie.

---

## Casos de No-Intervención

El discount **no aplica** cuando:
- `estado == 'NEUTRAL'` — no hay señal directa que distorsionar
- `surface_overlap_rate >= 0.40` — suficiente historial reciente en misma superficie
- `n_partidos < 5` — muestra muy pequeña, el PELT ya tiene baja confianza
- Jugadores con `match_id=None` que usaron THF — el historial temporal puede ser de cualquier superficie, no inferir el overlap

---

## Deuda Técnica Generada

| ID | Tarea | Prioridad |
|---|---|---|
| D46-01 | Tests — `test_nodo46_surface_discount.py` (8 tests) | ALTA |
| D46-02 | Implementar `_normalize_surface()` en `markov_analyzer.py` | ALTA |
| D46-03 | Implementar `_surface_overlap_rate()` en `markov_analyzer.py` | ALTA |
| D46-04 | Aplicar discount al `factor_markov` y `confianza` en output Markov | ALTA |
| D46-05 | Sub-problema B: `win_rate_reciente_same_surface` para form scoring | MEDIA |
| D46-06 | Agregar `surface_overlap_rate` al output de `markov_analysis` | MEDIA |
| D46-07 | Calibrar `min_floor=0.70` y `THRESHOLD=0.40` con datos históricos | BAJA |

**Orden de implementación:** D46-01 → D46-02 → D46-03 → D46-04 (Sub-A) → D46-05 (Sub-B) → D46-06

---

## Tests Necesarios (D46-01)

```python
# test_nodo46_surface_discount.py

def test_normalize_surface_grass_variants():
    """'Hierba', 'Grass', 'hierba' → 'grass'"""

def test_normalize_surface_hard_variants():
    """'Dura', 'Hard', 'hard court' → 'hard'"""

def test_overlap_rate_all_same_surface():
    """10 partidos en hard, torneo hard → overlap = 1.0"""

def test_overlap_rate_all_different_surface():
    """5 partidos en arcilla, torneo hard → overlap = 0.0"""

def test_overlap_rate_mixed():
    """3 hard + 7 hierba, torneo hard → overlap = 0.3"""

def test_discount_cold_zero_overlap():
    """COLD, overlap=0.0, factor=0.85 → nuevo factor > 0.85 (menos penalizado)"""

def test_discount_hot_zero_overlap():
    """HOT, overlap=0.0, factor=1.15 → nuevo factor < 1.15 (menos inflado)"""

def test_discount_neutral_not_applied():
    """NEUTRAL → factor sin cambio independientemente del overlap"""
```

---

## Evidencia Empírica

| Sesión | Fallos Nodo-46 confirmados | Total fallos sesión | Nota |
|--------|---------------------------|---------------------|------|
| 2026-06-29 (Cary hard) | **1/3** (Watanuki COLD/arcilla→hard) | 3 | Glinka atribuido a Nodo-47; Hussey upset genuino |

**Criterio para sumar n:** un fallo es atribuible a Nodo-46 cuando:
1. El ranking bug (Nodo-47) NO es la causa (diferencia de pts Kambi vs ATP < 20pts)
2. El estado COLD/HOT fue ganado principalmente en otra superficie (≥60% partidos recientes)
3. El modelo habría predicho diferente con Markov descontado

**Prioridad de implementación:** acumular n≥5 casos confirmados antes de calibrar `min_floor` y `THRESHOLD`. Con n=1 cualquier constante es arbitraria.

La transición hierba→hard americano (agosto: Washington, Montréal, Cincinnati) es el próximo momento de riesgo. Monitorear sesiones de esa semana con este criterio.

---

## Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-18-PELT-Recency-Alpha]] | Nodo-18 ajusta λ por recencia del régimen; Nodo-46 ajusta la confianza por superficie del régimen |
| [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | Cuando Markov es descontado, otros componentes (ranking, H2H) deben pesar más — SNR implícito |
| ~~~~[[Nodo-32-Auditoria-Phantom-Edge]]~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_~~ _(MISSING — ver [[Nodo-86-Auditoria-Fable5]])_ | Phantom edge puede surgir cuando el Markov HOT en hierba se aplica incorrectamente a hard |
| [[Nodo-43-PELT-Cold-Rival-Promo-Filter]] | PCRS usa rival COLD para promo — si el COLD es de otra superficie, la señal es falsa positivo |

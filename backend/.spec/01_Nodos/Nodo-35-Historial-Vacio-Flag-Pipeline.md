# Nodo-35 — Propagación de Flag Historial Vacío: Bloqueo en Origen

> **Fecha:** 2026-06-25
> **Severidad:** ALTA — Historial vacío de un jugador produce predicciones fantasma que llegan a `apostar=True` sin ser bloqueadas. Causó pérdida real (caso Dimitrov vs Davidovich Fokina, 2026-06-25).
> **Prerequisitos:** Nodo-31 (anti-leakage), Nodo-34 (corrupción extracción H2H)
> **Archivos afectados:** `scraping/ninja_h2h_parser.py`, `analysis/rivalry_analyzer.py`, `edge_calculator.py`, `generar_tabla_favoritos2.py`
> **Implementa:** Sonnet | **Tests:** Haiku
>
> **Estado:** ⏳ PENDIENTE

---

## 0. RESUMEN EJECUTIVO

El extractor `ninja_h2h_parser.py` ya sabe en el momento de la extracción cuando un jugador tiene 0 partidos en el historial — lo imprime en log (`"historial vacío"`, `"Sin match_id_j2"`) pero **no lo serializa** en el JSON de salida. Los pasos posteriores del pipeline (`rivalry_analyzer`, `edge_calculator`) reciben una lista vacía `[]` y siguen calculando como si los datos fueran válidos, produciendo predicciones basadas en una sola mitad del partido.

**Impacto demostrado hoy (2026-06-25):**
- Davidovich Fokina: 0 partidos extraídos → modelo dio Dimitrov como 78.4% favorito → `edge=36.7%` → entró en watchlist con señal fantasma
- Julien Penzlin: 0 partidos extraídos → modelo dio Lazaro Juncadella con `apostar=True`, `edge=25.8%` → el único pick real del día era basura de datos
- 7 partidos afectados en total en el H2H de hoy

**El bloqueo debe nacer en el primer eslabón (extracción) y viajar como campo estructurado** a través de todo el pipeline. No debe reconstruirse adivinando en cada paso si el historial "se ve vacío".

---

## 1. DIAGNÓSTICO: DÓNDE NACE EL PROBLEMA

### 1.1 Punto de origen — `scraping/ninja_h2h_parser.py`

**Caso A — `_process_match()`, líneas 688 y 707:** cuando la asignación inteligente de bloques detecta que un jugador no está en la API proxy y no hay `match_id_j2`:

```python
# Línea 688 (p2 sin datos):
logger.info(f"   ⚠️ {p2} no en API proxy y sin match_id_j2 — historial vacío")
p2_records = []

# Línea 707 (p1 sin datos):
logger.info(f"   ⚠️ {p1} no en API proxy y sin match_id_j2 — historial vacío")
p1_records = []
```

**Caso B — `_process_ronda_futura()`, línea 872:** cuando `match_id_j2` no existe:

```python
logger.info(f"   ⚠️ Sin match_id_j2 para {p2} — historial vacío (solo ranking)")
```

**Caso C — `_analyze_form()`, línea 959:** cuando `history` es vacía, retorna `None`. Pero este caso ya es downstream del origen — el historial llega vacío porque Caso A o B ocurrieron antes.

**Lo que falta:** después de que `p1_history` o `p2_history` quedan como `[]`, en `_consolidate_result()` (línea 1002) se serializa el resultado al JSON sin ningún flag que indique que el historial está vacío. Solo queda implícito en `len(historial_X) == 0`.

### 1.2 Lo que llega a `_consolidate_result()` — línea 1028-1034

```python
return {
    ...
    f'historial_{p1_key}': p1_hist,      # puede ser []
    f'historial_{p2_key}': p2_hist,      # puede ser []
    'estadisticas': {
        f'partidos_{p1_key}': len(p1_hist),   # puede ser 0
        f'partidos_{p2_key}': len(p2_hist),   # puede ser 0
    },
    ...
}
```

No hay campo explícito `historial_extraido_p1` ni `historial_extraido_p2`. El JSON de salida no distingue entre "historial vacío porque el jugador no tiene partidos" y "historial vacío porque la extracción falló".

### 1.3 Cómo `rivalry_analyzer.py` recibe el historial vacío

En `generate_advanced_prediction()` (línea 1301), cuando `player1_history=[]` o `player2_history=[]`, los componentes que dependen del historial (`surface_specialization`, `form_recent`, `common_opponents`) retornan `0.5` o `0` silenciosamente:

- `find_common_opponents()` línea 629: `if not player1_history or not player2_history: return []`
- `calcular_surface_specialization()` línea 94: `if not player_history: return None`
- `calcular_form_reciente()` línea 284: `if not player_history: return None`

El modelo sigue con los componentes que sí tienen datos (ranking, ELO) y produce una predicción sesgada.

### 1.4 Cómo `edge_calculator.py` no bloquea

`data_completeness()` (línea 189) calcula qué fracción de componentes aportaron datos. Con p2 vacío:
- `surface_signal` = 0 o bajo
- `regime_signal` = 0
- Pero `bbi_signal` (derivado de cuota) sigue activo

El campo `data_completeness` existe en el edge_report pero **no hay gate que bloquee `apostar=True` cuando el historial de cualquiera de los dos jugadores es 0**. La guarda `n_axes_active < 2` (Nodo-28) bloqueó algunos casos pero no todos — Penzlin con 0 partidos pasó igualmente porque el eje de ranking (sin historial) siguió activo.

### 1.5 `generar_tabla_favoritos2.py` — línea 387

```python
if historial_df.empty:
    return ["- No hay datos históricos para analizar."]
```

Esta alarma existe pero está en el **último paso del pipeline** — para entonces ya se calculó, publicó en el edge_report y potencialmente se envió al trader.

---

## 2. FIX — FASE 1

### Fix 35-1: Serializar el flag en `ninja_h2h_parser.py`

**Archivo:** `scraping/ninja_h2h_parser.py`
**Función:** `_consolidate_result()`, línea 1015

Agregar dos campos al dict de retorno, inmediatamente después de `estadisticas`:

```python
# Después de 'estadisticas': { ... }, agregar:
'data_quality': {
    'historial_extraido_p1': len(p1_hist) > 0,
    'historial_extraido_p2': len(p2_hist) > 0,
    'n_partidos_p1': len(p1_hist),
    'n_partidos_p2': len(p2_hist),
},
```

Esto se escribe siempre — tanto cuando hay datos como cuando no — para que sea legible de forma estructurada por todos los pasos downstream.

### Fix 35-2: Propagar en `rivalry_analyzer.py`

**Archivo:** `analysis/rivalry_analyzer.py`
**Función:** `generate_advanced_prediction()`, línea 1301

Al inicio de la función, antes de cualquier cálculo, leer el flag desde `match_data` o desde el contexto que llega, y propagarlo al dict de predicción:

```python
# Al inicio de generate_advanced_prediction():
historial_incompleto = {
    'p1': len(player1_history) == 0,
    'p2': len(player2_history) == 0,
}
```

Y añadirlo al dict de retorno de `generate_advanced_prediction()`:

```python
# En el return de generate_advanced_prediction():
'historial_incompleto': historial_incompleto,
```

### Fix 35-3: Gate en `edge_calculator.py`

**Archivo:** `edge_calculator.py`
**Ubicación:** en la función que calcula el resultado por partido, junto a los gates existentes de `N28F2` y `HOT_sin_BBI` (líneas 853–876)

Agregar gate inmediatamente antes de los gates existentes:

```python
# Gate 35: bloquear si el historial del favorito o del rival está vacío
prediction = partido.get('ranking_analysis', {}).get('prediction', {})
historial_incompleto = prediction.get('historial_incompleto', {})
favorito_key = 'p1' if favorito_predicho == partido.get('jugador1') else 'p2'
rival_key = 'p2' if favorito_key == 'p1' else 'p1'

if historial_incompleto.get(favorito_key) or historial_incompleto.get(rival_key):
    jugador_sin_datos = []
    if historial_incompleto.get(favorito_key):
        jugador_sin_datos.append(favorito_predicho)
    if historial_incompleto.get(rival_key):
        jugador_sin_datos.append(nombre_rival)
    resultado['apostar'] = False
    resultado['motivo_reclasificacion'] = (
        f'HISTORIAL_NO_EXTRAIDO: sin datos de {", ".join(jugador_sin_datos)} '
        f'— predicción no confiable, bloqueada en origen'
    )
```

### Fix 35-4: Alerta visual en `generar_tabla_favoritos2.py`

**Archivo:** `generar_tabla_favoritos2.py`
**Ubicación:** en el bloque de escritura de cada partido, antes de mostrar el favorito predicho

```python
# Leer data_quality del partido
data_quality = partido.get('data_quality', {})
p1_sin_datos = not data_quality.get('historial_extraido_p1', True)
p2_sin_datos = not data_quality.get('historial_extraido_p2', True)

if p1_sin_datos or p2_sin_datos:
    jugadores_sin_datos = []
    if p1_sin_datos:
        jugadores_sin_datos.append(jugador1)
    if p2_sin_datos:
        jugadores_sin_datos.append(jugador2)
    f.write(f"\n{'='*80}\n")
    f.write(f"*** SIN HISTORIAL EXTRAIDO PARA: {', '.join(jugadores_sin_datos)} ***\n")
    f.write(f"*** BLOQUEADO EN extraer_historh2h.py. NO APOSTAR. ***\n")
    f.write(f"{'='*80}\n\n")
```

---

## 3. PLAN DE IMPLEMENTACIÓN — FASE 1

### Orden de implementación

1. Fix 35-1 (`ninja_h2h_parser.py`) — serializar el flag en el origen
2. Fix 35-2 (`rivalry_analyzer.py`) — propagar a prediction dict
3. Fix 35-3 (`edge_calculator.py`) — gate que fuerza `apostar=False`
4. Fix 35-4 (`generar_tabla_favoritos2.py`) — alerta visual

### Archivos a modificar

| Archivo | Función | Línea aproximada | Cambio |
|---|---|---|---|
| `scraping/ninja_h2h_parser.py` | `_consolidate_result()` | ~1031 | Agregar campo `data_quality` al dict de retorno |
| `analysis/rivalry_analyzer.py` | `generate_advanced_prediction()` | ~1301 | Leer len de historiales, añadir `historial_incompleto` al return |
| `edge_calculator.py` | gate section | ~850 | Gate `HISTORIAL_NO_EXTRAIDO` antes de gates existentes |
| `generar_tabla_favoritos2.py` | bloque por partido | ~bloque escritura | Leer `data_quality`, mostrar alerta si vacío |

---

## 4. TESTS — Nodo-35

**Archivo:** `tests/test_nodo35.py`

### T35-01: `test_flag_serializado_cuando_p2_vacio`
```
GIVEN _consolidate_result() con p2_hist=[]
WHEN se genera el dict resultado
THEN resultado['data_quality']['historial_extraido_p2'] == False
AND  resultado['data_quality']['n_partidos_p2'] == 0
AND  resultado['data_quality']['historial_extraido_p1'] == True  (p1 tiene datos)
```

### T35-02: `test_flag_serializado_cuando_p1_vacio`
```
GIVEN _consolidate_result() con p1_hist=[]
WHEN se genera el dict resultado
THEN resultado['data_quality']['historial_extraido_p1'] == False
AND  resultado['data_quality']['n_partidos_p1'] == 0
```

### T35-03: `test_flag_ausente_cuando_ambos_tienen_datos`
```
GIVEN _consolidate_result() con p1_hist=[...20 partidos...], p2_hist=[...15 partidos...]
WHEN se genera el dict resultado
THEN resultado['data_quality']['historial_extraido_p1'] == True
AND  resultado['data_quality']['historial_extraido_p2'] == True
AND  resultado['data_quality']['n_partidos_p1'] == 20
AND  resultado['data_quality']['n_partidos_p2'] == 15
```

### T35-04: `test_historial_incompleto_propagado_a_prediction`
```
GIVEN rivalry_analyzer.generate_advanced_prediction() con player2_history=[]
WHEN se genera la predicción
THEN prediccion['historial_incompleto']['p2'] == True
AND  prediccion['historial_incompleto']['p1'] == False
```

### T35-05: `test_historial_incompleto_ambos_propagado`
```
GIVEN generate_advanced_prediction() con player1_history=[] y player2_history=[]
WHEN se genera la predicción
THEN prediccion['historial_incompleto']['p1'] == True
AND  prediccion['historial_incompleto']['p2'] == True
```

### T35-06: `test_edge_calculator_bloquea_cuando_favorito_sin_datos`
```
GIVEN edge_calculator recibe partido con:
  - favorito_predicho = jugador1
  - historial_incompleto = {'p1': True, 'p2': False}
  - edge calculado = 30% (señal fuerte)
WHEN se aplican los gates
THEN resultado['apostar'] == False
AND  'HISTORIAL_NO_EXTRAIDO' in resultado['motivo_reclasificacion']
AND  nombre del jugador1 en motivo_reclasificacion
```

### T35-07: `test_edge_calculator_bloquea_cuando_rival_sin_datos`
```
GIVEN partido con:
  - favorito_predicho = jugador1
  - historial_incompleto = {'p1': False, 'p2': True}  ← rival sin datos
  - edge calculado = 25%
WHEN se aplican los gates
THEN resultado['apostar'] == False
AND  'HISTORIAL_NO_EXTRAIDO' in resultado['motivo_reclasificacion']
AND  nombre del jugador2 (rival) en motivo_reclasificacion
```
(Este es el caso Penzlin — el rival sin datos, no el favorito.)

### T35-08: `test_edge_calculator_no_bloquea_cuando_ambos_tienen_datos`
```
GIVEN partido con historial_incompleto = {'p1': False, 'p2': False}
  y edge real = 20%
WHEN se aplican los gates
THEN gate HISTORIAL_NO_EXTRAIDO NO se activa
AND  resultado['apostar'] puede ser True o False por otros gates
AND  'HISTORIAL_NO_EXTRAIDO' NOT in resultado.get('motivo_reclasificacion', '')
```
(Test de no-regresión: partidos con datos completos no son bloqueados por este gate.)

### T35-09: `test_tabla_favoritos_muestra_alerta_cuando_p2_vacio`
```
GIVEN partido con data_quality.historial_extraido_p2 = False
  y nombre del jugador2 = 'Julien Penzlin'
WHEN generar_tabla_favoritos2 escribe el bloque del partido
THEN output contiene 'SIN HISTORIAL EXTRAIDO'
AND  output contiene 'Julien Penzlin'
AND  output contiene 'NO APOSTAR'
AND  alerta aparece ANTES del bloque del favorito predicho
```

### T35-10: `test_tabla_favoritos_sin_alerta_cuando_datos_completos`
```
GIVEN partido con data_quality.historial_extraido_p1=True, historial_extraido_p2=True
WHEN generar_tabla_favoritos2 escribe el bloque
THEN output NO contiene 'SIN HISTORIAL EXTRAIDO'
AND  output NO contiene 'NO APOSTAR' (por este motivo)
```
(No-regresión: partidos normales no muestran la alerta.)

### T35-11: `test_flag_sobrevive_pipeline_completo`
```
GIVEN un h2h_results_enhanced.json con data_quality.historial_extraido_p2=False
  en un partido con edge aparente > 5%
WHEN edge_calculator.py procesa ese archivo
THEN en edge_report.json ese partido aparece en 'sin_edge' o 'watchlist' con motivo
AND  NO aparece en 'apostar'
AND  motivo_reclasificacion contiene 'HISTORIAL_NO_EXTRAIDO'
```
(Test de integración end-to-end del flag a través de dos archivos.)

---

## 5. CASOS LÍMITE

### CL-1: Jugador con historial vacío porque no tiene partidos recientes (legítimo)
El flag `historial_extraido=False` aplica cuando la extracción falló (0 partidos devueltos por la API). Si la API devuelve 0 partidos porque el jugador realmente no tiene historial en FlashScore, el comportamiento es idéntico — en ambos casos el modelo no tiene datos y no debe apostar. El gate es correcto en ambos casos.

### CL-2: Historial vacío en el jugador que NO es el favorito
Fix 35-3 bloquea si cualquiera de los dos tiene historial vacío. Razón: el edge se calcula comparando los dos jugadores — si uno de los dos es un punto ciego, la comparación es inválida independientemente de quién sea el favorito. Caso Penzlin confirmó esto.

### CL-3: Ambos jugadores con historial vacío
El gate bloquea igualmente. Motivo menciona ambos jugadores.

### CL-4: `data_quality` ausente en JSON legacy (archivos anteriores al fix)
Edge_calculator y tabla_favoritos deben tener fallback:
```python
data_quality = partido.get('data_quality', {})
historial_extraido_p1 = data_quality.get('historial_extraido_p1', True)  # assume OK si campo ausente
```
Los archivos legacy no tienen el campo → se asume que están OK → sin bloqueo → comportamiento anterior preservado.

---

## 6. MÉTRICAS DE ÉXITO

| Métrica | Antes | Post-fix |
|---|---|---|
| Partidos con historial vacío que llegan a `apostar=True` | Ocurrió hoy (Penzlin) | 0 |
| `data_quality` en todos los nuevos h2h JSON | Ausente | Presente (100%) |
| `historial_incompleto` en prediction dict | Ausente | Presente |
| Tests Nodo-35 | 0/11 | 11/11 |
| Tests totales (no regresión) | 1270 | ≥1281 |

---

## 7. WIKILINKS

- [[Nodo-34-Corrupcion-Datos-Extraccion-H2H]] — bugs de extracción en ninja_h2h_parser (mismo archivo)
- [[Nodo-33-Filtro-Coinflip-Sin-H2H]] — coin-flip gate (patrón relacionado: datos insuficientes)
- [[Nodo-31-Future-Match-Data-Leakage]] — anti-leakage (mismo extractor)
- [[Nodo-28-Conditional-Decomposition-Metamodel]] — n_axes_active gate (precursor de este gate)
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint

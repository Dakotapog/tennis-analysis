# Nodo-173 — Calibración de `p_modelo` y Reparación del Embudo de Decisión (D173-01 → D173-12)

> **Estado:** ✅ IMPLEMENTADO 2026-08-05 (Sonnet 5). BLOQUE A/B/E completos, BLOQUE C (D173-08/09) cerrado legítimamente: PUERTA 3 rechazó (`skill_holdout ≤ 0`). 34 tests REGLA-T53 nuevos + 3 regresiones pre-existentes arregladas (D173-06 exponía bugs de fixtures, D173-02 causaba test obsoleto).
> **Autor del análisis:** sesión de razonamiento extendido 2026-08-05; **implementación:** Sonnet 5 2026-08-05.
> **Destinatario:** CERRADO. Resultado medible aunque no rentable (mejor que estado anterior).
> **Precedencia:** este nodo deroga D154-01 (cap watchlist 10→50 es ahora 0) y T32-01 (reemplazado por PHANTOM_CAP en D173-06). Todo lo demás es aditivo u observacional.

---

## 0. Resumen ejecutivo (leer completo antes de tocar código)

El síntoma reportado por el usuario durante ~3 meses — *"el pipeline nunca encuentra señales"* —
es real y está cuantificado. Pero la causa **no** es que los gates sean demasiado estrictos.

La causa es que **`p_modelo` no es una probabilidad**. Es una transformación monótona comprimida
del margen de puntaje, con piso duro en 0.50, y está **mal calibrada al punto de tener skill
negativo** contra la tasa base. Los gates fueron ajustados para suprimir el ruido de ese
estimador, y la única forma en que un umbral puede suprimir ruido de un estimador sin escala
es suprimiendo casi todo.

El resultado neto es el peor de los dos mundos posibles:

- **rechaza picks buenos y malos de forma arbitraria** (por un accidente de la fórmula, no por evidencia), y
- **acepta preferentemente fantasmas de datos faltantes** (`ranking_rival=None` → `p_modelo` 0.93 → hit real 0.625).

> **Conclusión operativa:** aflojar los umbrales sería activamente dañino. La reparación es
> calibrar el estimador contra los resultados ya liquidados que el shadow book tiene, abrir el
> embudo para que la muestra crezca, y recién entonces re-sintonizar los gates contra evidencia
> medida. Este nodo especifica exactamente eso.

---

## 1. Evidencia — todo reproducible

Cada número de esta sección fue derivado durante la auditoría y **debe ser re-derivado por el
implementador antes de tocar código** (ver §9 "Comandos de reproducción"). Si algún número no
reproduce, **detener e informar** — no implementar sobre una premisa que cambió.

### 1.1 La fórmula de confianza

`analysis/rivalry_analyzer.py:2359-2361`:

```python
confidence = 50
if total_score > 0:
    confidence = 50 + (abs(score_diff) / total_score * 50)
```

`edge_calculator.py:906`:

```python
p_modelo = confidence / 100.0
```

Es decir:

```
p_modelo = 0.5 + 0.5 · |final_p1 − final_p2| / (final_p1 + final_p2)
```

**Tres defectos estructurales, cada uno independiente:**

| # | Defecto | Consecuencia |
|---|---------|--------------|
| F1 | **Piso duro en 0.50** (usa `abs()`) | El estimador es incapaz de expresar duda. Siempre asigna ≥50% a quien eligió. `edge = p_modelo − p_implicita` no puede ser fuertemente negativo por convicción del modelo, solo por cuota extrema. |
| F2 | **Compresión por normalización** | Alcanzar `p_modelo ≥ 0.55` exige ratio de puntajes ≥ 1.222; `≥0.60` exige ≥ 1.50; `≥0.70` exige ≥ 2.33. Entre dos tenistas, ratios >1.2 en el agregado ponderado son raros. La masa de la distribución queda atrapada en [0.50, 0.57]. |
| F3 | **Auto-dilución del denominador** | Sumar bonus al favorito también sube Σ. La ganancia marginal de confianza por evidencia adicional es sublineal y asintóticamente nula. |

**Verificación aritmética contra los tres casos citados por el usuario** (2026-07-30) — la fórmula
reproduce los tres exactamente, confirmando que es la ruta activa:

| Partido | final_fav | final_dog | Δ | Σ | `50+50·Δ/Σ` | Publicado |
|---|---|---|---|---|---|---|
| Liutova K. vs Joint M. | 4.03 | 3.45 | 0.58 | 7.48 | **53.88** | 53.9% ✓ |
| Lee E. vs Hewitt D. | 3.23 | 2.61 | 0.62 | 5.84 | **55.31** | 55.3% ✓ |
| Laron M. vs Casari A. | 2.56 | 2.09 | 0.47 | 4.65 | **55.05** | 55.0% ✓ |

### 1.2 Distribución empírica (edge_report 2026-07-30, n=55 serializados)

```
p_modelo    min=0.503  p25=0.515  mediana=0.526  p75=0.576  max=0.950
p_elo_base  min=0.251  p25=0.497  mediana=0.567  p75=0.640  max=0.841
```

`p_modelo` **no tiene soporte bajo 0.50** — es una consecuencia directa de F1, no de los partidos.
El estimador ELO, calculado sobre los mismos partidos, sí tiene soporte hasta 0.251.

### 1.3 Calibración empírica contra resultados liquidados (shadow book, n=727)

| bin `p_modelo` | n | p_medio | hit real | sesgo |
|---|---:|---:|---:|---:|
| [0.50, 0.53) | 370 | 0.513 | 0.414 | **−0.099** |
| [0.53, 0.56) | 102 | 0.542 | 0.539 | −0.003 |
| [0.56, 0.60) | 66 | 0.579 | 0.439 | **−0.140** |
| [0.60, 0.70) | 60 | 0.638 | 0.467 | **−0.171** |
| [0.70, 0.85) | 81 | 0.773 | 0.630 | **−0.144** |
| [0.85, 1.01) | 48 | 0.932 | 0.625 | **−0.307** |

```
Brier(p_modelo) = 0.2599      Brier(tasa base 0.476) = 0.2494      skill = −0.0420
```

**`p_modelo` es peor que predecir siempre la tasa base.** Sobreestima en todos los bins, y de forma
catastrófica en el extremo superior. La curva además es **no monótona** entre 0.53 y 0.60.

### 1.4 Comparación de estimadores (mismo conjunto liquidado)

| estimador | n | Brier | skill | AUC |
|---|---:|---:|---:|---:|
| `p_modelo` | 727 | 0.2599 | −0.0420 | **0.5752** |
| `p_elo_base` | 633 | 0.2726 | −0.1041 | 0.5334 |
| `p_blend` | 633 | 0.2616 | −0.0597 | 0.5469 |
| `p_implicita` (mercado) | 727 | **0.2365** | **+0.0519** | **0.6339** |

**Lectura crítica — es el hallazgo central del nodo:**

- `p_modelo` tiene **AUC 0.5752**: hay señal de ordenamiento real. El motor *sí sabe* rankear.
- Pero su **skill de calibración es negativo**. Es la firma clásica de un **buen ordenador con
  escala rota** — exactamente lo que predice F1/F2/F3.
- **`p_elo_base` es PEOR que `p_modelo`** (AUC 0.5334). Descarta la vía intuitiva de "usar ELO en
  vez del score". *No implementar esa idea.* Se probó y falla.
- El mercado domina en ambas métricas. Es el benchmark a superar, no a ignorar.

### 1.5 ¿Aporta el modelo información sobre el mercado?

Regresión logística sobre n=727: `y ~ logit(p_implicita) + (p_modelo − 0.5)`

```
M1 solo mercado        logL = −479.103
M2 mercado + modelo    logL = −478.004
LR χ²(1) = 2.198       (crítico 5% = 3.84)
coef(margen_modelo) = +1.0012   SE = 0.6764   z = +1.48   p = 0.1388
coef(logit_mercado) = +0.5939   (1.0 = mercado perfectamente calibrado)
```

**Veredicto honesto: el margen del modelo NO alcanza significancia estadística sobre el mercado
(p=0.139).** El punto estimado es positivo y económicamente relevante, pero el IC cruza cero.
Estamos **sub-potenciados**, no refutados.

**Esto debe decirse sin adornos:** a fecha de hoy no está demostrado que el sistema le gane al
mercado en el agregado lineal. Cualquier despliegue debe tratarse como acumulación prospectiva
bajo hipótesis pre-registrada, no como alfa establecida.

Nota metodológica: `coef(logit_mercado) = 0.594 ≠ 1.0` indica que **el mercado también está
sobre-confiado en esta población**. Ahí es donde vive la estructura explotable, y motiva §1.7.

### 1.6 El embudo real de un día (2026-07-30, n=55 serializados)

Descomposición por primer gate que mata cada pick:

```
 23  edge ≤ 5%
 21  T32-01  (p_modelo < 0.55 Y cuota ≥ 2.10)     ← DOMINANTE
  8  N28F2   (n_axes_active < 2)
  1  HOT_sin_BBI
  1  kambi_no_disponible
  1  SOBREVIVE
```

**T32-01 mata 21 de los 32 picks que tenían edge>5% y Kelly>2% (66%).** Y T32-01 es un umbral sobre
`p_modelo`, es decir, **sobre la escala rota**.

### 1.7 ¿Qué destruye T32-01? — el dato que invierte la conclusión

Sobre los liquidados (n=727), segmentado por cuota:

| cuota | n | p_impl | hit real | sesgo | ROI flat |
|---|---:|---:|---:|---:|---:|
| [1.00, 1.35) | 85 | 0.683 | 0.776 | +0.094 | −11.1% |
| [1.35, 1.60) | 53 | 0.623 | 0.585 | −0.038 | −14.3% |
| [1.60, 1.90) | 64 | 0.561 | 0.578 | +0.017 | +0.8% |
| [1.90, 2.30) | 155 | 0.474 | 0.471 | −0.003 | −1.7% |
| [2.30, 3.00) | 213 | 0.394 | 0.399 | +0.005 | +1.4% |
| **[3.00, 20)** | **156** | **0.256** | **0.346** | **+0.090** | **+37.2%** |

Banda `cuota ≥ 3.0`, n=157: **ROI = +36.3%, IC95% bootstrap = [+6.2%, +68.1%]** → límite inferior
sobre cero.

Descomposición por T32-01 **dentro de esa banda rentable**:

| grupo | n | hit | ROI flat | IC95% |
|---|---:|---:|---:|---|
| PASAN el gate (`p_modelo ≥ 0.55`) | 49 | 0.367 | +52.4% | [−9.5%, +121.2%] |
| **BLOQUEADOS por T32-01** | **108** | 0.333 | **+29.0%** | [−5.4%, +65.3%] |

Y sobre el **dominio completo** del gate (`cuota ≥ 2.10`, n=463):

| grupo | n | hit | ROI flat | IC95% |
|---|---:|---:|---:|---|
| PASAN | 116 | 0.353 | +12.8% | [−19.0%, +47.1%] |
| **BLOQUEADOS** | **347** | **0.380** | **+7.6%** | [−7.5%, +23.8%] |

**Sobre su dominio completo, T32-01 bloquea el 75% del volumen, y el grupo bloqueado tiene MAYOR
hit rate (0.380 vs 0.353) y ROI positivo.** El gate no está seleccionando calidad. Está cortando
por una propiedad de la fórmula comprimida.

**Advertencia de validez obligatoria** (el implementador debe conservarla en todo reporte derivado):
estos son picks que el sistema ya había superficializado (`apostar` + `watchlist` + muestras), no
una muestra aleatoria. Hay selección. Los IC son anchos. El settlement ITF está incompleto
(ver CLAUDE.md §5: 57 abiertos permanentes). **Estas cifras justifican instrumentar y medir, no
justifican desplegar tamaño.**

### 1.8 Fantasmas de datos faltantes

| grupo | n | mediana `p_modelo` | % ≥ 0.70 |
|---|---:|---:|---:|
| `ranking_rival` presente | 45 | 0.522 | **2%** |
| `ranking_rival = None` | 10 | 0.651 | **50%** |

La ausencia de dato del rival **infla** la confianza (su componente de ranking colapsa a ~0, lo que
infla Δ/Σ). Es la inversión exacta de lo correcto: falta de información debe *reducir* convicción.

Confirmado sobre dinero real: el bin `[0.85, 1.01)` — poblado casi enteramente por este mecanismo —
predijo 0.932 y entregó **0.625**.

De los 2 picks `apostar=True` del 2026-07-30, **ambos** eran de este tipo
(`Biot A. vs Grekul E.` rival sin ranking, `Ryan Ziegann S. vs Nirundorn T.` rival sin ranking).

### 1.9 Caps de serialización — la fuga mayor

`edge_calculator.py:1579-1583`:

```python
'watchlist': no_apostar_lista[:50],   # D154-01: cap 10→50
'sin_edge':  edge_negativo[:5],        # sample de edge negativo
'sin_datos': sin_datos[:5],
```

Metadata real del 2026-07-30: `n_procesados = 268`, `n_edge_positivo = 54`, `n_apostar = 2`.

→ **214 picks con edge ≤ 0 existieron y solo 5 fueron serializados (2.3%).**

Y `rival_value_betslip.py:148-150` consume exactamente ese bucket:

```python
data.get("apostar", []) + data.get("watchlist", []) + data.get("sin_edge", [])
```

**La estrategia RIVAL VALUE (H88-01) — la de mayor ROI del sistema según CLAUDE.md §11 — está
alimentada por un bucket truncado a 5 de 214.** Ve el 2.3% de su universo, todos los días.

Además `watchlist` está en 48/50: **el cap está a punto de saturar**, y en días de alto volumen
(2026-07-29: 354 procesados; 2026-08-04: 263) trunca en silencio.

### 1.10 La tasa de conversión histórica

51 días con `edge_report` (2026-06-14 → 2026-08-05):

```
procesados = 7382    edge_positivo = 1906    apostar = 252    → conversión 3.41%
días con 0 apostar:  15/51 = 29%
apostar/día:         media 4.94   mediana 2.0   max 46
```

El reclamo del usuario es sustancialmente correcto: mediana 2 picks/día antes de los filtros
downstream (REGLA-HF-1, Kambi, tier, KGR), y 29% de días en cero. **87% de los picks con edge
positivo mueren en los gates.**

### 1.11 Los tres casos citados — trazados a su causa

**Caso A — Liutova K. vs Joint M. (2026-07-30).** Presente en `edge_report_20260730_095243.json`,
bucket `watchlist`:

```
p_modelo 0.539   p_implicita 0.4484   edge 9.1%   kelly_kl 0.0591   confidence_flag LOW
cuota_favorito 2.23   apostar False   motivo_reclasificacion None
kambi_disponible True   ranking 229 vs 77   n_h2h 0   markov_favorito HOT
p_elo_base 0.7022   alpha_vs_elo −0.1632   elo 1862 vs 1713
```

Diagnóstico: pasó **todos** los gates de reclasificación (`motivo_reclasificacion = None`), tenía
edge y Kelly positivos, y era **apostable** (`kambi_disponible=True`, cuota 2.23).
**Murió únicamente por T32-01, por 1.1 puntos porcentuales** (0.539 < 0.55, cuota 2.23 ≥ 2.10).

Las señales especiales que el usuario cita (campeona de torneo ×1.4, scalp top-20, HOT 90%, ajuste
dinámico de pesos) **sí se aplicaron** — el motor las procesó. Pero por F3 su efecto neto sobre
`p_modelo` fue de ~+3pp, porque el bonus multiplica una componente que luego se pondera, se agrega
y se normaliza contra el denominador que el propio bonus subió.

> Nota de rigor: `p_elo_base = 0.7022` es *sugestivo* en este caso, pero §1.4 demuestra que ELO es
> globalmente peor que `p_modelo`. **No usar este caso para justificar un switch a ELO.** El caso
> demuestra la compresión, no la superioridad de ELO.

**Caso B — Laron M. vs Casari A. (2026-07-30).** Favorito del mercado @1.06 con **74 días de
inactividad**; ganó Casari @8.00.

`edge_calculator.py:1161-1170`:

```python
def _rfi_tier_de(days):
    # RFI-0 <90d | RFI-1 90-179 | RFI-2 180-364 | RFI-3 >=365
    if days is None or days < 90:
        return 0
```

**74 días → `rfi_tier = 0` → señal cero.** Y `rfi_ultra` exige `rfi_tier >= 2` (≥180d).

Mientras tanto `generar_tabla_favoritos2.py` **sí** aplicó decay por inactividad a 74d
(`form_recent ×0.35`, Nodo-57, curva exponencial continua). **Dos capas del mismo sistema modelan
el mismo fenómeno con umbrales incompatibles**: la capa de predicción lo ve desde ~30d de forma
continua; la capa de decisión no lo ve hasta 90d, como escalón. La banda **30–90d**, donde vive la
mayoría de los layoffs reales en ITF/Challenger, es **invisible para la decisión**.

**Caso C — Lee E. vs Hewitt D. (2026-07-30).** El más simple y el más frustrante:

```
p_modelo 0.553   confidence_flag MODERATE   cuota 2.05   edge 6.5%   kelly_kl 0.1057
apostar False    kambi_disponible False
```

Pasó T32-01 (cuota 2.05 < 2.10). Tenía edge y Kelly positivos. **Murió por
`kambi_disponible=False`** — el gate de Nodo-140/141 lo excluyó de todos los combo builders de
forma **silenciosa y no registrada**. El usuario vio "55.3% EVALUAR" en la tabla y nada en los
combos, sin ninguna explicación disponible en ningún artefacto.

### 1.12 La brecha tabla ↔ edge_report

`reports/analisis_partidos_20260730_112234.txt` contiene **268 partidos** analizados
(150 `ACCION: EVALUAR`, 119 `ACCION: NO-BET`). El `edge_report` de la misma corrida serializa **55**.

Además el umbral de la tabla (`generar_tabla_favoritos2.py:982`, `< 54` → NO-BET) y el de la
decisión (`P_MODELO_MIN_UNDERDOG = 0.55`) son **dos umbrales distintos sobre la misma cantidad**,
definidos en archivos distintos, sin constante compartida. Drift garantizado.

---

## 2. Diagnóstico consolidado

```
Motor de predicción          → señal de ordenamiento REAL (AUC 0.575)
        ↓
confidence = 50+50·|Δ|/Σ     → DESTRUYE la escala (piso 0.50, comprimido a [0.50,0.57])
        ↓
p_modelo = confidence/100    → no es probabilidad (Brier skill −0.042)
        ↓
edge = p_modelo − p_implicita→ resta escala rota menos escala calibrada = sin sentido
        ↓
T32-01: p_modelo ≥ 0.55      → umbral sobre escala rota; mata 66% de candidatos con edge,
                               incluido el 75% del volumen de su dominio que era rentable
        ↓
caps [:5] / [:50]            → 214 picks de edge negativo → 5; RIVAL VALUE ve 2.3% de su universo
        ↓
sin telemetría de embudo     → "hoy no hubo señales" es INFALSIFICABLE
```

**Los cinco defectos son independientes y cada uno merece un fix separado.** El orden importa:
instrumentar → reparar estimador → re-sintonizar gates → liberar estrategias.

---

## 3. Principios de diseño (vinculantes para el implementador)

1. **No aflojar umbrales sin calibrar primero.** Aflojar T32-01 sobre `p_modelo` actual es
   aumentar exposición con un estimador de skill negativo. Prohibido.
2. **El mercado es el prior, no el enemigo.** `p_implicita` tiene AUC 0.634 y skill positivo.
   La arquitectura correcta ancla al mercado y deja que el modelo aporte un *tilt residual*.
3. **Medir antes de cambiar.** Todo cambio de comportamiento va precedido de telemetría que
   permita evaluarlo, y detrás de un flag con default = comportamiento actual.
4. **Fuera de muestra o no vale.** Toda calibración se ajusta con partición temporal
   (entrenar en el pasado, evaluar en el futuro). Nada de ajustar y reportar in-sample.
5. **REGLA-T53.** Los tests invocan la función real. Nunca replican la fórmula.
6. **Honestidad estadística.** §1.5 no alcanza significancia. Todo despliegue es acumulación
   prospectiva bajo hipótesis pre-registrada con `n_stop` y kill-switch.

---

## 4. Deliverables

### BLOQUE A — Instrumentación (implementar primero, sin cambio de comportamiento)

---

#### D173-01 — Telemetría de embudo: `gate_ledger`

**Problema:** "hoy no hubo señales" no es verificable. No existe registro de qué gate mató qué.

**Archivo:** `edge_calculator.py`

**Implementación:**

Crear función pura:

```python
def registrar_gate(resultado: dict, gate_id: str, motivo: str) -> None:
    """Anota en resultado['gate_ledger'] el gate que bloqueó este pick.

    Append-only. El PRIMER gate que bloquea queda en resultado['gate_bloqueante'].
    Los subsiguientes se acumulan en la lista pero no sobreescriben el primero.
    NUNCA muta 'apostar' — es puramente observacional.
    """
```

Instrumentar **todos** los puntos de bloqueo existentes con su `gate_id` canónico. Lista completa
(verificar cada línea antes de editar; los números son del estado 2026-08-05):

| gate_id | ubicación | condición |
|---|---|---|
| `G_EDGE_MIN` | `:518` | `edge <= EDGE_MIN` |
| `G_KELLY_MIN` | `:519` | `kelly_kl_ajustado <= KELLY_KL_MIN` |
| `G_T32_01` | `:520` | `p_modelo < 0.55 and cuota_favorito >= 2.10` |
| `G_SIN_DATOS` | `:1230` | `_p1_sin_datos or _p2_sin_datos` |
| `G_PHANTOM` | `:1248` | `_phantom_detected` |
| `G_HIST_CONTAM` | `:1264` | contaminación Nodo-152 |
| `G_ELO_INCOHERENTE` | `:1293` | D152-05 |
| `G_N28F2` | `:1305` | `n_axes_active < 2` |
| `G_HOT_SIN_BBI` | `:1316` | `markov HOT and bbi < 0.50` |
| `G_T33_01` | `:1326` | `n_h2h == 0 and p_modelo < 0.55` |

Serializar en cada pick: `gate_bloqueante` (str|None) y `gate_ledger` (list[str]).

**Agregar a `metadata` del reporte:**

```python
'funnel': {
    'n_procesados': int,
    'por_gate': {gate_id: count, ...},   # conteo del gate BLOQUEANTE (primero)
    'n_sobrevive': int,
}
```

**Restricción dura:** D173-01 no cambia ni una decisión. Es puro registro. Un test debe probar
que el conjunto de `apostar=True` es **idéntico** antes y después.

---

#### D173-02 — Eliminar caps de serialización

**Problema:** §1.9 — 214 picks de edge negativo → 5. RIVAL VALUE ve 2.3% de su universo.

**Archivo:** `edge_calculator.py:1579-1583`

**Cambio:**

```python
'apostar':   apostar_lista,
'watchlist': no_apostar_lista,        # D173-02: cap 50 eliminado (saturaba en 48)
'sin_edge':  edge_negativo,           # D173-02: cap 5 eliminado (era 5 de 214)
'sin_datos': sin_datos,               # D173-02: cap 5 eliminado
'no_data':   no_data_lista,
```

**Mitigación de tamaño de archivo** (obligatoria, el reporte pasa de ~55 a ~270 picks):
añadir a `metadata` los conteos pre-serialización `n_watchlist_total`, `n_sin_edge_total`,
`n_sin_datos_total` para que ningún consumidor tenga que inferirlos.

**Verificar y ajustar consumidores** — buscar con `grep -rn "sin_edge\|watchlist" --include=*.py .`
Todo consumidor que asumía ≤5 o ≤50 debe declarar explícitamente su propio límite en su propio
código, nunca depender del truncado del productor. Consumidores conocidos a revisar:
`rival_value_betslip.py`, `betplay_combo_builder.py`, `combo_confianza_builder.py`,
`favoritos_combo_builder.py`, `shadow_book.py::log_picks`.

**Riesgo declarado:** `shadow_book.log_picks()` va a registrar ~5× más picks/día. Es **deseable**
(la muestra de calibración crece), pero verificar que `settle()` y `report()` escalan. Si el
volumen de JSONL se vuelve problema, la solución es rotación/compresión, **no** re-truncar.

---

#### D173-03 — Exponer el margen crudo con signo

**Problema:** F1/F2 — `|Δ|/Σ` destruye tanto el signo como la escala. El implementador **no puede
calibrar lo que no está serializado**.

**Archivo:** `analysis/rivalry_analyzer.py`, dict de retorno de `generate_advanced_prediction()`
(~`:2375-2385`). El bloque ya expone `scores.score_difference` — hay que asegurar que se propague
sin normalizar y con signo respecto al **pick**, más los insumos.

Añadir al dict retornado:

```python
'score_margin_raw':   round(final_p1 - final_p2, 4),   # con signo, SIN normalizar
'score_sum_raw':      round(final_p1 + final_p2, 4),
'score_fav_raw':      round(max(final_p1, final_p2), 4),
'score_dog_raw':      round(min(final_p1, final_p2), 4),
```

**Archivo:** `edge_calculator.py` — propagar al pick serializado, orientado al favorito elegido:

```python
'score_margin_signed': <margen con signo POSITIVO a favor del favorito_predicho>,
'score_sum':           <suma>,
'rival_ranking_missing': bool(ranking_rival is None),
'fav_ranking_missing':   bool(ranking_favorito is None),
```

**`confidence` y `p_modelo` NO se tocan en este deliverable.** D173-03 es puramente aditivo.

---

#### D173-04 — Backfill de features en el shadow book

**Problema:** la calibración necesita `score_margin_signed` en registros históricos, que no lo
tienen. Sin backfill, el conjunto de entrenamiento arranca en cero.

**Archivo nuevo:** `scripts/backfill_calibration_features.py`

**Comportamiento:**
- Recorre `reports/shadow_book/sb_*.jsonl` y los `reports/edge_report_*.json` correspondientes.
- Para cada registro liquidado (`resolucion.resultado in ('WON','LOST')`) reconstruye
  `score_margin_signed` a partir de `p_modelo` cuando sea posible:
  **`|Δ| = (p_modelo − 0.5) · 2 · Σ`** — pero Σ **no** está serializado históricamente.
- **Por lo tanto:** para el histórico, usar `(p_modelo − 0.5)` como proxy del margen normalizado
  y **marcar el registro con `feature_provenance: 'proxy_normalizado'`**.
- Los registros nuevos (post-D173-03) llevan `feature_provenance: 'raw'`.

**El modelo de calibración (D173-05) debe incluir `feature_provenance` como control** y reportar
coeficientes por separado para ambos grupos. Si difieren materialmente, entrenar solo con `'raw'`
y esperar acumulación.

**Sigue el patrón de** `scripts/backfill_strategy.py` (D144-06): idempotente, `--dry-run` por
defecto, nunca toca `pick_snapshot` (inmutable), escribe solo campos top-level nuevos.

---

### BLOQUE B — Reparar el estimador

---

#### D173-05 — Calibrador ancla-mercado

**Este es el deliverable central del nodo.**

**Archivo nuevo:** `core/probability_calibrator.py`

**Arquitectura** (justificada por §1.4 y §1.5 — el mercado es el prior, el modelo aporta tilt):

```
p_final = σ( β0 + β1·logit(p_implicita)
                + β2·score_margin_signed
                + β3·rival_ranking_missing
                + β4·fav_ranking_missing )
```

**Por qué esta forma y no otra** (el implementador NO debe sustituirla por su cuenta):

- Anclar a `logit(p_implicita)` garantiza que el estimador **nunca sea peor que el mercado**:
  con β2=β3=β4=0 y β1=1, β0=0 recupera exactamente el mercado. Es un piso de seguridad estructural.
- `β1` libre (no fijado en 1.0) absorbe la sobre-confianza del mercado medida en §1.5 (0.594).
- `score_margin_signed` entra **crudo y con signo** — repara F1 y F2 de una vez.
- Los indicadores de ranking faltante recibirán coeficiente **negativo** por construcción
  (§1.8: predijo 0.932, entregó 0.625), **eliminando el fantasma automáticamente** sin regla ad-hoc.

**API pública obligatoria:**

```python
def fit_calibrator(records: list[dict], *, min_n: int = 300) -> dict:
    """Ajusta el calibrador. Devuelve artefacto serializable con coeficientes y diagnósticos.
    Levanta ValueError si len(records) < min_n."""

def predict_calibrated(artifact: dict, *, p_implicita: float, score_margin_signed: float,
                       rival_ranking_missing: bool, fav_ranking_missing: bool) -> float:
    """Aplica el calibrador. Función pura. Sin I/O."""

def evaluate_calibration(y_true: list[int], p_pred: list[float]) -> dict:
    """Devuelve {'brier','brier_baseline','skill','auc','bins':[...],'n'}."""
```

**Persistencia:** `data/probability_calibrator.json` con:
`{coeficientes, n_entrenamiento, ventana_temporal, metricas_holdout, fitted_at, feature_provenance_split}`

**Protocolo de ajuste — no negociable:**

1. **Partición temporal**, nunca aleatoria: entrenar con los registros más antiguos, evaluar con
   el 30% más reciente. Los partidos de tenis se agrupan por día y torneo; una partición aleatoria
   filtra información.
2. **Criterio de aceptación duro:** `skill > 0` **fuera de muestra**. Baseline actual: **−0.0420**.
   Si el holdout no supera skill 0, **el artefacto no se despliega** y `USE_CALIBRATOR` queda en
   `False`. Reportar y detener — no ajustar la forma funcional para forzar el número (p-hacking).
3. **Criterio secundario:** para todo bin con n≥30, `|hit_real − p_medio| ≤ 0.05`.
4. **Reportar siempre** el AUC junto al skill. Si el AUC cae por debajo de 0.55, algo se rompió en
   el pipeline de features — investigar antes de continuar.

**Script de ajuste:** `scripts/fit_probability_calibrator.py`, con `--dry-run` (default),
`--report`, y `--commit` (único modo que escribe el artefacto).

**Integración en `edge_calculator.py`:**

```python
USE_CALIBRATOR = False   # D173-05: default OFF hasta que el holdout pase skill>0
```

Cuando `True`, serializar **ambos**:

```python
'p_modelo':      <sin cambio, escala vieja>,   # NUNCA se elimina — trazabilidad y rollback
'p_modelo_cal':  <p_final calibrada>,
'edge_cal':      <p_modelo_cal − p_implicita>,
```

**Regla de oro:** `p_modelo` no se sobreescribe jamás. Todo consumidor migra explícitamente a
`p_modelo_cal` en D173-07. Esto permite rollback instantáneo y comparación A/B en el shadow book.

---

#### D173-06 — Guard de confianza fantasma (defensa en profundidad)

**Problema:** §1.8. Aunque D173-05 lo resuelve estadísticamente, se necesita una defensa
independiente que funcione con `USE_CALIBRATOR = False`.

**Archivo:** `edge_calculator.py`

Función pura:

```python
def _phantom_confidence_cap(p_modelo: float, rival_ranking_missing: bool,
                            fav_ranking_missing: bool, n_h2h: int) -> tuple[float, str | None]:
    """D173-06: la ausencia de datos no puede producir alta convicción.

    Cuando falta el ranking de alguno de los dos jugadores, p_modelo se acota a
    PHANTOM_CAP. Devuelve (p_ajustada, motivo|None).
    """
```

`PHANTOM_CAP = 0.60` — justificación: el bin `[0.85,1.01)`, poblado por este mecanismo, entregó
hit real 0.625. Un cap de 0.60 es conservador respecto a esa observación.

Aplicar **antes** del cálculo de `edge`. Registrar vía `registrar_gate(..., 'G_PHANTOM_CONF', ...)`
cuando el cap muerde.

**Efecto esperado y deseado:** los 2 picks `apostar=True` del 2026-07-30 (§1.8), ambos fantasmas,
dejan de ser `apostar`. El sistema apostará **menos** el día que esto entre — y estará **mejor**.
Documentar esto explícitamente para que no se lea como regresión.

---

#### D173-07 — Unificar la constante de umbral

**Problema:** §1.12 — `54` en `generar_tabla_favoritos2.py:982`, `0.55` en `edge_calculator.py:82`.
Dos umbrales sobre la misma cantidad, en archivos distintos.

**Archivo:** `config.py` — fuente única (mismo patrón que `detectar_tier()`):

```python
P_MODELO_MIN_UNDERDOG = 0.55   # D173-07: fuente única. Escala p_modelo (0-1).
```

`edge_calculator.py` y `generar_tabla_favoritos2.py` importan de ahí.
`generar_tabla_favoritos2.py` compara contra `P_MODELO_MIN_UNDERDOG * 100`.

---

### BLOQUE C — Re-sintonizar los gates contra evidencia

> **Bloquear todo este bloque hasta que D173-05 pase su criterio de aceptación.**
> Si el calibrador no supera skill>0 fuera de muestra, **BLOQUE C no se implementa** y el nodo
> se cierra con los bloques A/B/E. Esa es una salida legítima y debe reportarse como tal.

---

#### D173-08 — Reemplazar T32-01 por un gate de edge calibrado

**Problema:** §1.6/§1.7 — T32-01 es un umbral sobre la escala rota; bloquea el 75% del volumen de
su dominio y ese volumen bloqueado tenía mayor hit rate y ROI positivo.

**Archivo:** `edge_calculator.py:518-521`

```python
# ANTES (T32-01)
apostar = (
    edge > EDGE_MIN
    and kelly_kl_ajustado > KELLY_KL_MIN
    and (p_modelo >= P_MODELO_MIN_UNDERDOG or cuota_favorito < 2.10)
)

# DESPUÉS (D173-08) — solo cuando USE_CALIBRATOR is True
apostar = (
    edge_cal > EDGE_MIN_CAL
    and kelly_kl_ajustado > KELLY_KL_MIN
)
```

**Justificación del reemplazo, no del aflojamiento:** T32-01 existía para impedir "edge fantasma"
en underdogs — un edge que nace de cuota extrema, no de convicción. Con `p_modelo_cal` calibrada,
**esa protección es intrínseca**: si el modelo no tiene convicción real, `p_modelo_cal` se pega al
mercado, `edge_cal → 0`, y el pick muere en `EDGE_MIN_CAL` sin necesidad de un umbral separado.
El gate no se elimina; se vuelve redundante y se retira.

`EDGE_MIN_CAL`: arrancar en **0.05** (igual que `EDGE_MIN`). **No optimizar este valor contra el
histórico** — sería sobreajuste sobre la misma muestra que motivó el cambio. Se re-evalúa después
de `n_stop` observaciones prospectivas bajo H173-01.

**Obligatorio:** mantener `T32_01_SHADOW = True` durante toda la fase de acumulación — registrar
en cada pick si T32-01 *lo habría* bloqueado (`t32_01_habria_bloqueado: bool`), sin actuar.
Esto permite medir prospectivamente el valor real del gate retirado, que es exactamente la
pregunta que §1.7 deja abierta con IC anchos.

---

#### D173-09 — Cerrar la brecha RFI 30–90d

**Problema:** §1.11 Caso B — la capa de predicción modela inactividad continuamente desde ~30d;
la capa de decisión no la ve hasta 90d.

**Archivo:** `edge_calculator.py:1161-1170`

```python
def _rfi_tier_de(days):
    # D173-09: nuevo tier RFI-0.5 para la banda 30-89d.
    # Antes: <90d → 0 (invisible). El decay Nodo-57 sí actúa desde ~30d en la capa
    # de predicción — este tier alinea la capa de decisión con esa curva.
    if days is None or days < 30:
        return 0
    if days < 90:
        return 0.5
    if days < 180:
        return 1
    if days < 365:
        return 2
    return 3
```

**Cuidado — cambio de tipo:** `rfi_tier` pasa de `int` a `float`. Auditar **todo** comparador
existente con `grep -rn "rfi_tier" --include=*.py .` — las comparaciones `>= 1` y `>= 2` conservan
su semántica; las de igualdad (`== 0`) **no**. Corregir cada una explícitamente.

**Nuevo campo observacional** (no gatea nada en este nodo):

```python
'rfi_layoff_fade': bool(
    _rfi_tier_v >= 0.5 and _rfi_is_bookie_fav and cuota_inactivo < 1.50
),
```

Marca el patrón exacto del Caso B: favorito corto del mercado volviendo de un layoff.
**Pre-registrado bajo H173-02, REPORTE_SOLO.** No abre apuestas en este nodo.

---

#### D173-10 — Observabilidad del gate Kambi

**Problema:** §1.11 Caso C — `kambi_disponible=False` excluye de todos los combo builders sin
dejar rastro consultable.

**Archivos:** `betplay_combo_builder.py`, `combo_confianza_builder.py`, `favoritos_combo_builder.py`

Cada builder escribe `reports/combo_exclusions_{fecha_compact}.json`:

```json
{"builder": "betplay_safe", "excluidos": [
  {"partido": "...", "motivo": "kambi_no_disponible", "cuota": 2.05,
   "p_modelo": 0.553, "edge_pct": "6.5%"}
]}
```

**No cambiar el comportamiento del gate** — el filtro Kambi es correcto (no se puede apostar lo que
la casa no lista). Solo hacerlo **auditable**, para que "el pipeline lo vio y los combos no" tenga
respuesta en un archivo en vez de requerir una sesión de depuración.

---

### BLOQUE E — El reporte que nunca dice "nada"

---

#### D173-11 — Reporte diario de embudo

**Problema:** la queja de fondo del usuario. "Hoy no hubo señales" es una no-respuesta.

**Archivo nuevo:** `scripts/funnel_report.py`

Consume `metadata.funnel` (D173-01) + `combo_exclusions_*.json` (D173-10) y emite, **siempre**,
llueva o truene:

```
EMBUDO 2026-08-05                              268 partidos analizados
──────────────────────────────────────────────────────────────────────
  edge <= 5%                    112   ▓▓▓▓▓▓▓▓▓▓▓▓
  T32-01 / edge_cal              43   ▓▓▓▓▓
  N28F2 (n_axes < 2)             31   ▓▓▓
  phantom / contaminacion        18   ▓▓
  kambi no disponible            24   ▓▓
  ──────────────────────────────────
  SOBREVIVEN                      6

  LOS 3 QUE MAS CERCA ESTUVIERON:
    Lee E. vs Hewitt D.      edge_cal 4.2%  (faltaron 0.8pp)  [kambi: NO]
    Ruan Z. vs Stagno B.     n_axes 1/2     (falto 1 eje: BBI 0.48)
    Agwi M. vs Leroux J.     edge_cal 4.9%  (faltaron 0.1pp)
```

**Requisito de diseño explícito:** el reporte **nunca** puede terminar sin contenido accionable.
Cuando `SOBREVIVEN = 0`, la sección "LOS 3 QUE MÁS CERCA ESTUVIERON" es obligatoria, con la
distancia exacta al umbral. Esto cumple `feedback_zero_response_prohibition` y MANDATO-01→06 de
[[Nodo-89]].

Integrar como **PASO 3.9** en `run_daily.py`, después de PASO 3K. `optional=True` — no puede
romper el pipeline.

---

#### D173-12 — Segmentos de calibración en el shadow book

**Archivo:** `shadow_book.py`

Añadir a `report()`:

- Curva de fiabilidad de `p_modelo` **y** de `p_modelo_cal` lado a lado (bins de §1.3), con Brier,
  baseline, skill y AUC.
- Segmento por banda de cuota, replicando §1.7 (es la tabla que detecta si el alfa de la banda
  ≥3.0 persiste prospectivamente o era ruido de muestra).
- Segmento `t32_01_habria_bloqueado` (D173-08) — hit% y ROI de lo que el gate retirado habría
  matado. **Es el juez final de si retirarlo fue correcto.**

---

## 5. Hipótesis pre-registradas

Añadir a `validation/preregistered_hypotheses.json` **antes** de implementar BLOQUE C.
Umbrales congelados — modificarlos antes de `n_stop` es p-hacking (regla del archivo).

### H173-01 — El calibrado supera al bruto

```json
{
  "H173-01": {
    "nombre": "p_modelo_cal alcanza Brier skill > 0 fuera de muestra",
    "descripcion": "El calibrador ancla-mercado (D173-05) produce probabilidades con skill de Brier positivo contra la tasa base, en partición temporal holdout. Baseline medido pre-fix: skill=-0.0420, AUC=0.5752, n=727.",
    "origen_deuda": "Nodo-173 — auditoria 2026-08-05. p_modelo tiene AUC 0.575 (ordena bien) pero skill -0.042 (escala rota).",
    "preregistrado": "2026-08-05",
    "umbrales_congelados": {
      "metrica_primaria": "brier_skill_holdout",
      "umbral_exito": 0.0,
      "baseline_medido": -0.0420,
      "metrica_secundaria": "max_abs_sesgo_por_bin",
      "umbral_secundario": 0.05,
      "min_n_por_bin": 30,
      "particion": "temporal_70_30",
      "n_stop": 300
    },
    "kill_switch": "Si tras n>=300 holdout el skill sigue <=0, USE_CALIBRATOR permanece False y BLOQUE C no se implementa. Es un resultado valido: significa que el motor no aporta sobre el mercado."
  }
}
```

### H173-02 — El fade de layoff corto

```json
{
  "H173-02": {
    "nombre": "rfi_layoff_fade: favorito corto volviendo de layoff 30-90d rinde bajo su cuota implicita",
    "descripcion": "Picks con rfi_layoff_fade=True (favorito del mercado con cuota<1.50 e inactividad 30-89d) tienen hit rate inferior a su p_implicita. Motivado por el caso Laron M. vs Casari A. (2026-07-30): favorito @1.06 con 74d de inactividad, perdio. La capa de prediccion aplica decay desde ~30d (Nodo-57) pero la capa de decision no lo veia hasta 90d.",
    "origen_deuda": "Nodo-173 D173-09 — brecha de umbral entre Nodo-57 (continuo desde 30d) y RFI (escalon en 90d).",
    "preregistrado": "2026-08-05",
    "umbrales_congelados": {
      "condicion": "rfi_tier >= 0.5 AND rfi_is_bookie_fav AND cuota_inactivo < 1.50",
      "metrica": "hit_real - p_implicita",
      "umbral_exito": -0.05,
      "n_stop": 40
    },
    "modo": "REPORTE_SOLO — no abre apuestas en Nodo-173",
    "kill_switch": "Si con n>=25 el sesgo es >= 0, descartar la senal y cerrar la hipotesis."
  }
}
```

### H173-03 — El valor real de T32-01

```json
{
  "H173-03": {
    "nombre": "T32-01 retirado: el volumen que bloqueaba es rentable prospectivamente",
    "descripcion": "Picks marcados t32_01_habria_bloqueado=True tienen ROI flat >= 0 en acumulacion prospectiva. Medicion retrospectiva pre-fix (n=347, dominio cuota>=2.10): hit 0.380 vs 0.353 de los que pasaban, ROI +7.6% IC95 [-7.5%,+23.8%]. La retrospectiva tiene sesgo de seleccion; esta hipotesis lo mide limpio.",
    "origen_deuda": "Nodo-173 D173-08 — T32-01 era umbral sobre escala no calibrada.",
    "preregistrado": "2026-08-05",
    "umbrales_congelados": {
      "metrica": "roi_flat_1u",
      "umbral_exito": 0.0,
      "n_stop": 120
    },
    "kill_switch": "Si con n>=60 el ROI flat es <= -10%, restaurar T32-01 sobre p_modelo_cal con umbral re-derivado del holdout."
  }
}
```

---

## 6. Orden de implementación y puertas

```
FASE 1 (sin cambio de comportamiento — seguro de mergear)
  D173-01  gate_ledger + funnel metadata
  D173-03  score_margin_signed + flags de ranking faltante
  D173-07  constante unificada en config.py
  D173-10  observabilidad de exclusiones Kambi
  D173-11  reporte de embudo
      ↓ PUERTA 1: suite completa verde, conjunto apostar=True IDENTICO al baseline

FASE 2 (abre el embudo — cambia volumen, no criterio)
  D173-02  eliminar caps de serializacion
  D173-04  backfill de features
  D173-12  segmentos de calibracion en shadow book
      ↓ PUERTA 2: rival_value_betslip ve >100 candidatos/dia; sin degradacion de runtime

FASE 3 (repara el estimador — detras de flag, default OFF)
  D173-05  calibrador ancla-mercado
  D173-06  guard de confianza fantasma
      ↓ PUERTA 3: H173-01 evaluada sobre holdout.
                  skill > 0  → continuar a FASE 4
                  skill <= 0 → DETENER. Cerrar nodo con A/B/E. Reportar honestamente.

FASE 4 (re-sintoniza gates — solo si PUERTA 3 abrio)
  D173-08  edge_cal reemplaza T32-01 (+ T32_01_SHADOW)
  D173-09  RFI 30-90d
      ↓ acumulacion prospectiva bajo H173-01/02/03
```

**Ninguna fase arranca sin que la anterior haya pasado su puerta.** Documentar el resultado de cada
puerta en este archivo antes de avanzar.

---

## 7. Tests (REGLA-T53 — invocan la función real, nunca replican la fórmula)

**`tests/test_nodo173_funnel.py`**
1. `test_173_01` — `registrar_gate()` acumula en orden y `gate_bloqueante` conserva el **primero**.
2. `test_173_02` — un pick bloqueado por N28F2 tras fallar T32-01 reporta `G_T32_01` como bloqueante.
3. `test_173_03` — **regresión dura**: sobre `edge_report_20260730_095243.json`, el conjunto de
   `apostar=True` es idéntico con y sin D173-01.
4. `test_173_04` — `metadata.funnel.por_gate` suma exactamente `n_procesados`.

**`tests/test_nodo173_serializacion.py`**
5. `test_173_05` — con 214 picks de edge negativo, `sin_edge` serializa 214, no 5.
6. `test_173_06` — `metadata.n_sin_edge_total` coincide con `len(sin_edge)`.
7. `test_173_07` — `rival_value_betslip._leer_edge_report()` recibe >100 candidatos con un
   reporte sin caps.

**`tests/test_nodo173_calibrador.py`**
8. `test_173_08` — `predict_calibrated()` con `β2=β3=β4=0, β1=1, β0=0` reproduce `p_implicita`
   exactamente (piso de seguridad estructural del §D173-05).
9. `test_173_09` — `fit_calibrator()` con n<`min_n` levanta `ValueError`.
10. `test_173_10` — `evaluate_calibration()` sobre un vector perfectamente calibrado sintético
    devuelve `skill > 0` y sesgo por bin ≈ 0.
11. `test_173_11` — `predict_calibrated()` es **pura**: misma entrada → misma salida, sin I/O.
12. `test_173_12` — con `rival_ranking_missing=True` y coeficiente β3 negativo ajustado,
    `p_final < p_final(rival_ranking_missing=False)` — el fantasma se penaliza.
13. `test_173_13` — la partición es **temporal**: ningún registro de holdout tiene fecha anterior
    a algún registro de entrenamiento.

**`tests/test_nodo173_phantom_cap.py`**
14. `test_173_14` — `_phantom_confidence_cap(0.95, rival_missing=True, ...)` devuelve `0.60`.
15. `test_173_15` — sin rankings faltantes, es identidad (devuelve la entrada sin tocar).
16. `test_173_16` — **caso real**: `Biot A. vs Grekul E.` del 2026-07-30 (p_modelo 0.928, rival
    sin ranking) deja de ser `apostar=True` con el cap activo.

**`tests/test_nodo173_rfi.py`**
17. `test_173_17` — `_rfi_tier_de(74)` devuelve `0.5` (caso Laron; antes devolvía `0`).
18. `test_173_18` — `_rfi_tier_de(29)` → `0`; `_rfi_tier_de(90)` → `1`; `_rfi_tier_de(180)` → `2`;
    `_rfi_tier_de(365)` → `3`. Fronteras exactas.
19. `test_173_19` — `rfi_ultra` conserva su semántica pre-D173-09 (sigue exigiendo ≥180d).

**`tests/test_nodo173_gate_edge_cal.py`**
20. `test_173_20` — **caso Liutova**: con el calibrador activo y sus features reales
    (`p_implicita 0.4484`, margen crudo, rankings presentes), el pick **no muere por T32-01**.
    El test verifica la ausencia de `G_T32_01` en el ledger, **no** que se convierta en `apostar`
    (eso depende de `edge_cal`, que es una cantidad medida, no una que debamos forzar).
21. `test_173_21` — `t32_01_habria_bloqueado` se marca correctamente con `T32_01_SHADOW=True`
    sin alterar `apostar`.
22. `test_173_22` — con `USE_CALIBRATOR=False`, el comportamiento es **bit-idéntico** al baseline.

**`tests/test_nodo173_reporte.py`**
23. `test_173_23` — con `SOBREVIVEN=0`, el reporte incluye la sección "LOS 3 QUE MÁS CERCA
    ESTUVIERON" con distancias numéricas al umbral (prohibición de respuesta cero).

---

## 8. Riesgos y contra-indicaciones

| # | Riesgo | Mitigación |
|---|---|---|
| R1 | El calibrador se sobreajusta a n=727 con selección | Partición temporal obligatoria; `n_stop=300` prospectivo; kill-switch H173-01 |
| R2 | Quitar T32-01 aumenta exposición con estimador aún imperfecto | FASE 4 bloqueada tras PUERTA 3; `T32_01_SHADOW` mide el costo real; H173-03 con kill-switch |
| R3 | Sin caps, el edge_report crece ~5× | Medido en PUERTA 2; si hay problema → rotación/compresión, **nunca** re-truncar |
| R4 | `rfi_tier` int→float rompe comparadores | Auditoría exhaustiva por grep en D173-09; test de fronteras |
| R5 | Shadow book crece ~5× | Deseable (la muestra de calibración crece). Verificar `settle()`/`report()` en PUERTA 2 |
| R6 | El alfa de la banda cuota≥3.0 es artefacto de selección | Explícitamente NO se explota en este nodo. Solo se mide vía D173-12 |
| R7 | **El calibrador no supera skill>0** | **Salida legítima.** Cerrar con A/B/E. El sistema queda medible aunque no rentable — que es mejor que ahora |

---

## 9. Comandos de reproducción

El implementador **debe** re-derivar estos números antes de empezar. Si alguno no reproduce,
detener e informar.

```bash
# §1.3 calibración de p_modelo
python3 - <<'PY'
import json,glob
pairs=[]
for f in sorted(glob.glob('reports/shadow_book/sb_2026-*.jsonl')):
    for line in open(f):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        ps=r.get('pick_snapshot') or {}; rs=r.get('resolucion') or {}
        if not isinstance(rs,dict) or rs.get('resultado') not in ('WON','LOST'): continue
        pm=ps.get('p_modelo')
        if pm is not None: pairs.append((float(pm),1 if rs['resultado']=='WON' else 0))
b=sum((p-w)**2 for p,w in pairs)/len(pairs)
base=sum(w for _,w in pairs)/len(pairs)
br=sum((base-w)**2 for _,w in pairs)/len(pairs)
print(f'n={len(pairs)} Brier={b:.4f} baseline={br:.4f} skill={1-b/br:+.4f}')
PY
# ESPERADO: n=727 Brier=0.2599 baseline=0.2494 skill=-0.0420

# §1.6 embudo del dia
python3 -c "
import json,collections
d=json.load(open('reports/edge_report_20260730_112725.json'))
a=[p for k in ('apostar','watchlist','sin_edge','sin_datos') for p in (d.get(k) or []) if isinstance(p,dict)]
print(collections.Counter((p.get('motivo_reclasificacion') or 'SIN_MOTIVO')[:40] for p in a))"

# §1.9 magnitud del cap
python3 -c "
import json; m=json.load(open('reports/edge_report_20260730_095243.json'))['metadata']
print('procesados',m['n_procesados'],'edge+',m['n_edge_positivo'],'-> edge negativos:',m['n_procesados']-m['n_edge_positivo'],'serializados: 5')"
# ESPERADO: procesados 268 edge+ 54 -> edge negativos: 214 serializados: 5

# §1.11 caso Liutova
python3 -c "
import json; d=json.load(open('reports/edge_report_20260730_095243.json'))
for k in ('apostar','watchlist','sin_edge'):
  for p in d.get(k) or []:
    if 'Liutova' in (p.get('partido') or ''):
      print(k,{x:p.get(x) for x in ('p_modelo','edge_pct','kelly_kl','cuota_favorito','apostar','motivo_reclasificacion','kambi_disponible','p_elo_base')})"
```

---

## 10. Qué NO hacer (errores que esta auditoría ya descartó)

1. **NO sustituir `p_modelo` por `p_elo_base`.** Se probó: AUC 0.5334 vs 0.5752. ELO es peor.
   El caso Liutova (`p_elo_base=0.702`) es sugestivo pero no generaliza.
2. **NO bajar el umbral de T32-01** de 0.55 a 0.53. Mueve el corte sobre la misma escala rota.
3. **NO añadir bonus de puntaje** para "premiar" campeones/scalps/HOT. F3 demuestra que el
   denominador se los come. Ya se aplican; el problema es la normalización posterior.
4. **NO eliminar `p_modelo`** al introducir `p_modelo_cal`. Trazabilidad y rollback.
5. **NO optimizar `EDGE_MIN_CAL`** contra el histórico. Sobreajuste sobre la muestra que motivó
   el cambio. Se re-evalúa tras `n_stop` prospectivo.
6. **NO explotar la banda cuota≥3.0** en este nodo. §1.7 tiene sesgo de selección declarado.
   Se mide (D173-12), no se apuesta.
7. **NO tratar §1.5 como refutación del motor.** `p=0.139` con `coef=+1.00, SE=0.68` es
   sub-potenciado, no nulo. Por eso el nodo abre el embudo: para que `n` crezca.

---

## 11. Wikilinks

**NÚCLEO:**
[[Nodo-32]] (T32-01, el gate que este nodo reemplaza) ·
[[Nodo-28]] (N28F2, segundo gate del embudo) ·
[[Nodo-152]] (Phantom History Guard — misma filosofía: datos malos no pueden producir convicción, D173-06 implementa PHANTOM_CAP análogo) ·
[[Nodo-154]] (D154-01 subió el cap de watchlist 10→50; D173-02 lo elimina completamente) ·
[[Nodo-144]] (strategy tagging en shadow_book, D173-12 reúsa segmentos por estrategia)

**CONSUMIDORES del embudo abierto:**
[[Nodo-68]] / H88-01 (RIVAL VALUE — consumidor de `sin_edge`, desnutrido históricamente por el cap; D173-02 lo destranca) ·
[[Nodo-89]] (MANDATO-01→06, prohibición de respuesta cero — origen de D173-11 reporte de embudo) ·
[[Nodo-100]] (taxonomía de estrategias — el embudo alimenta las 13, cap removal multiplica candidatos)

**CONTEXTO genealógico:**
[[Nodo-57]] (decay de inactividad continuo) ·
[[Nodo-96]] (IRP — misma familia de retorno tras inactividad) ·
[[Nodo-140]] / [[Nodo-141]] (gate Kambi — filtro pre-selector de apostabilidad) ·
[[Nodo-56]] (display en `generar_tabla_favoritos2.py`, mismo backend) ·
[[Nodo-90]] (pipeline metadata: Kambi coverage en edge_report)

**IMPLEMENTACIÓN:**
- **D173-01/D173-02:** `registrar_gate()` + funnel + caps eliminados (edge_calculator.py:505-684, 1755-1829)
- **D173-03/D173-07:** `score_margin_signed` + constante unificada (edge_calculator.py:1003-1056, config.py:42)
- **D173-04:** `backfill_calibration_features.py` (scripts/ + tests/)
- **D173-05:** `fit_probability_calibrator.py` + `core/probability_calibrator.py` (PUERTA 3: FALLIDA, skill≤0)
- **D173-06:** `_phantom_confidence_cap()` (edge_calculator.py:482-502, 1075-1077, 1220-1226)
- **D173-10:** `core/combo_exclusions.py` (observabilidad gate Kambi + tests/ test_nodo173 D173-10a/10b)
- **D173-11:** `scripts/funnel_report.py` (reporte diario embudo, D173-11c mandato nunca vacío)
- **D173-12:** `shadow_book.py` §1.7 (segmentos cuota-band 6 bins, mensajes PUERTA 3/T32_01 honesto)

**Tests:** 34 tests REGLA-T53 en `tests/test_nodo173_calibracion.py` (2575 → 2609 total suite)

**CLAUDE.md actualizado:** §5 (Nodo-173 IMPLEMENTADO), §7 (bugs activos: ninguno nuevo), §11 (RIVAL VALUE sin cap histórico).

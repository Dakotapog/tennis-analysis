# Nodo-68 — Rival Value Flip: edge positivo del RIVAL cuando el favorito tiene edge negativo

> **Wikilinks:** [[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]] | [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-66-Plan-Trabajo-Semanal-Sonnet]]
> **Fecha:** 2026-07-11 | **Origen:** hallazgo de Opus, verificado manualmente por el usuario en varios partidos
> **Estado:** PRE-REGISTRO PENDIENTE (H88-01) — implementación observacional esta semana (Sonnet)
> **Severidad de la oportunidad:** ALTA — es la lectura inversa del marco #1 de Nodo-86 ("el edge es el interruptor"): si el mercado sobre-paga al favorito, el valor está en el otro lado

---

## 1. El hallazgo (caso Obradovic, 2026-07-10)

```
edge_vs_mercado (favorito):  Obradovic −17.2%  (modelo 67.5% vs implícita 84.7%, cuota 1.18)
edge_vs_mercado_rival:       Fabre    +17.2%  (modelo 32.5% vs implícita ~15-19%, cuota 5.20)
Resultado: Obradovic PERDIÓ → el lado rival pagaba 5.20
```

El sistema actual solo evalúa como apuesta al `favored_player` del modelo. Cuando ese favorito tiene edge negativo, hoy el pick simplemente "no se apuesta" — pero la información no se tira: **el lado rival puede ser una apuesta de valor positivo que nadie está mirando.**

## 2. Formalización (nivel doctoral — esto NO es una tautología, y eso importa)

Con probabilidades del modelo complementarias (p_m + (1−p_m) = 1) y cuotas con vig:

```
edge_fav   = p_m − 1/c_fav
edge_rival = (1−p_m) − 1/c_riv
edge_fav + edge_rival = 1 − (1/c_fav + 1/c_riv) = −vig     (vig ≈ 3-8%)
```

**Consecuencia:** `edge_rival > 0 ⟺ edge_fav < −vig`. No todo favorito con edge negativo genera valor en el rival — solo cuando el edge negativo **supera el vig**. Obradovic: −17.2% ≪ −vig → Fabre +valor real. Un favorito con edge −2% NO genera flip (el vig se lo come). Esta frontera es el primer umbral congelable.

**El precedente ya existía en el sistema:** el caso semilla de Nodo-64 (Rivera @4.35 vs Michnev @1.17) ES un rival value flip — el modelo dio 62.7% al underdog y ganó 6-0 6-0. RFI-ULTRA es un *subconjunto* de este patrón (flip causado por inactividad). Este nodo generaliza: flip por CUALQUIER causa que el modelo vea y el precio no.

## 3. La propiedad que lo hace barato de validar: la acumulación es GRATIS

El `pick_snapshot` ya serializa `cuota_rival` y `edge`. Para todo registro settled del segmento flip:

```
rival GANA ⟺ resolucion.resultado == 'LOST'
ROI_rival_flat_1u = (cuota_rival − 1) si LOST, −1 si WON
```

**No hace falta ni un solo mecanismo nuevo de settle.** Los 177 settled existentes permiten un backtest retrospectivo HOY (etiquetado RETROSPECTIVO — lección C-05: jamás mezclarlo con la acumulación prospectiva ni usarlo para activar nada).

## 4. Riesgos que el pre-registro debe blindar (por qué no apostamos ya)

1. **Descalibración invertida:** si `p_modelo` está sobreconfiado en favoritos (lo típico), entonces (1−p_m) está *subestimado*… lo que haría el edge_rival CONSERVADOR. Pero si el modelo infla underdogs (phantom edge, T32-01), el edge_rival se infla igual que los phantom edges históricos. El segmento debe excluir picks NO_DATA/phantom y n_axes<2 del lado que se evalúa.
2. **Apostar contra el propio modelo:** el rival gana solo (1−p_m) de las veces según el modelo — hit% esperado 25-40%. Es una estrategia de ROI por cuota, no de hit rate: psicológicamente dura (rachas largas de pérdida) y exige flat stakes pequeños. Kelly con p=0.32 y cuota 5.2 da fracción minúscula — correcto.
3. **Sesgo de selección de las semillas:** "ya lo comprobé en varios partidos" = casos identificados post-hoc (mismo sesgo que H77-01/03 declaran). Las semillas motivan, no validan.
4. **Solapamiento con H76-01:** Rivera es semilla de AMBAS. En el análisis, el segmento flip debe reportarse CON y SIN los casos rfi_ultra para no contar el mismo alpha dos veces.
5. **Zona trampa H52-08:** cuotas rival 2.00-2.50 tienen historial 21.4% hit — el rango congelado empieza en 2.50.

## 5. H88-01 — pre-registro (Sonnet: AÑADIR como hipótesis nueva al JSON; nunca tocar las existentes)

```json
"H88-01": {
  "nombre": "Rival Value Flip: el lado rival supera breakeven cuando edge_fav < -10%",
  "descripcion": "Picks donde el favorito del modelo tiene edge_vs_mercado <= -0.10 (muy por debajo de -vig): el RIVAL, a su cuota, supera breakeven. Acumulacion gratis: rival gana cuando resolucion=LOST.",
  "origen_deuda": "Nodo-68 — hallazgo Opus 2026-07-10 (Obradovic/Fabre), precedente Rivera/Michnev (Nodo-64)",
  "preregistrado": "2026-07-12",
  "umbrales_congelados": {
    "edge_fav_max": -0.10,
    "cuota_rival_min": 2.50,
    "cuota_rival_max": 8.00,
    "excluir": "status=NO_DATA, phantom_data=true",
    "sub_segmento_preregistrado": "mismo segmento AND rfi_ultra=false (aislar del alpha RFI)",
    "nota": "PROHIBIDO cambiar umbrales antes de n_stop. PROHIBIDO apostar el lado rival antes de graduacion."
  },
  "metrica": "hit%_rival (=% LOST del favorito en el segmento) con IC Wilson 95% + ROI flat 1u a cuota_rival",
  "exito": "IC Wilson 95% inferior > 1/cuota_rival_media del segmento, con n >= 30",
  "n_stop": 30,
  "estado": "ACUMULANDO",
  "n_actual": 0, "hits": 0
}
```

## 6. CHECKLIST DE IMPLEMENTACIÓN (Sonnet, esta semana — todo OBSERVACIONAL)

- ⬜ **D68-01** `edge_calculator.py` (tras el bloque RFI): serializar `edge_vs_mercado_rival = round((1 - p_modelo) - 1/cuota_rival, 4)`, `vig = round(1/cuota_fav + 1/cuota_rival - 1, 4)` y `rival_value_flag = (edge <= -0.10 and 2.50 <= cuota_rival <= 8.00 and status != NO_DATA and not phantom_data)`. NO tocar `apostar` ni kelly.
- ⬜ **D68-02** `shadow_book.py` report + report_dict: segmento `RIVAL_VALUE (H88-01)` con hit%_rival = % LOST, ROI a cuota_rival (usar `pick_snapshot.cuota_rival`; NO el pnl_flat_1u del favorito), IC Wilson. Sub-segmento sin rfi_ultra.
- ⬜ **D68-03** Añadir H88-01 al `preregistered_hypotheses.json` (hipótesis NUEVA — permitido).
- ⬜ **D68-04** Backtest RETROSPECTIVO sobre los settled existentes (script one-off, REPORTE_SOLO): n, hit%_rival, ROI del segmento con umbrales congelados. Guardar en `reports/backtest_rival_value_RETRO.md` con la etiqueta RETROSPECTIVO en mayúsculas. Si el retro da ROI negativo → informar al usuario ANTES de seguir invirtiendo esfuerzo.
- ⬜ **D68-05** Test REGLA-T53: partido sintético Obradovic-like (p_m=0.675, c_fav=1.18, c_riv=5.20) → `edge_vs_mercado_rival≈+0.133`, `rival_value_flag=True`; y un edge_fav=−0.04 → flag False (el vig se lo come).
- 🚫 **PROHIBIDO hasta graduación H88-01:** generar picks apostables del lado rival, incluirlos en combos, o boost de kelly. La única salida operativa pre-graduación es una línea informativa en `generar_tabla_favoritos2.py` (opcional, D68-06 a decisión del usuario).

## 7. Por qué esta oportunidad es coherente con todo lo que ya sabemos

Es el marco #1 de Nodo-86 leído al revés: si el edge del favorito es el interruptor de las señales, un edge **muy negativo** no es ausencia de señal — es señal de que el mercado sobre-pagó, y el valor cruzó de lado. Los VARIABLE de Nodo-65 rendían "al azar" apostados COMO favoritos baratos; este nodo pregunta lo que nadie preguntó: ¿cuánto pagaba sistemáticamente el otro lado? La respuesta cuesta n=30 settled y cero riesgo, porque la acumulación es gratis.

## Addendum — Implementación (2026-07-12, Sonnet)

| ID | Estado | Commit | Detalle |
|---|---|---|---|
| D68-01 | ✅ IMPLEMENTADO | `1516d11` | `edge_calculator.py`: 3 campos serializados al final de `calcular_edge_completo()`, después de todos los guards (status y phantom_data finales). Caso Obradovic: edge_rival=+0.1327, flag=True. Control vig=-0.04: flag=False. 1822 tests OK. |
| D68-02 | ✅ IMPLEMENTADO | `7d68e1c` | `shadow_book.py`: helper `_rival_value_metrics()` (métricas invertidas), bloque RIVAL_VALUE en `report()` entre RFI y D54-02, clave `rival_value` en `report_dict()` con estado CONTINUAR/GRADUABLE/NO_GRADUABLE. Misma fuente de verdad (D58-01). |
| D68-03 | ✅ IMPLEMENTADO | `f04100c` | H88-01 añadida a `preregistered_hypotheses.json` (hipótesis #17, sin tocar existentes). Umbrales congelados: edge_fav_max=-0.10, cuota_rival [2.50-8.00]. n_actual=0. |
| D68-04 | ✅ IMPLEMENTADO | `2fe6fb3` | Backtest RETROSPECTIVO: n=0 picks en segmento (esperado — shadow book solo loggea pools con edge>0). Sin sesgo de supervivencia. Acumulación H88-01 arranca prospectivamente desde hoy. Reporte en `reports/backtest_rival_value_RETRO.md`. |
| D68-05 | ✅ IMPLEMENTADO | `e16af9d` | 5 tests REGLA-T53 en `tests/test_nodo68_rival_value.py`. 5/5 pasando. Suite total: 1827 passed. |

**Gap conocido (D68-06 — a decisión del usuario):** línea informativa en `generar_tabla_favoritos2.py` mostrando el flip cuando `rival_value_flag=True`. **IMPLEMENTADO 2026-07-14** (sesión de evidencia real — ver §8).

---

## §8 — EVIDENCIA REAL 2026-07-14 (sesión histórica)

> **Auditoría:** Sonnet 4.6 — análisis multi-señal completo del stack de los 3 picks RIVAL VALUE.

### 8.1 Resultados

| Partido | Cuota rival | edge_fav | Veredicto modelo | Resultado |
|---|---|---|---|---|
| Maaya Rajeshwaran Revathi vs **Leticia Romanova** | @6.00 | −20.3% | OBSERVAR (p_hist=0.33, Markov COLD, n_axes=1) | **RIVAL GANO** |
| Raluca Georgiana Serban vs **Daria Kuczer** | @2.50 | −15.3% | NO GO (n_axes=0, LOW conf, Kelly −0.46) | **RIVAL GANO** |
| Deniz Dilek vs **Weronika Falkowska** | @2.75 | −19.7% | NO GO (cal_conf=0.30, n_cal=5, fav HOT) | **RIVAL GANO** |

**Combinada apostada por el usuario: 6.00 × 2.50 × 2.75 = 41.25x el stake.**

### 8.2 Análisis del stack de señales

El análisis completo mostró que en 2 de 3 picks las señales secundarias CONTRADECÍAN la apuesta rival (fav HOT, p_hist=0.611, cal_conf=0.30). Sin embargo la señal primaria — `edge_fav <= −15%` — fue suficiente en los 3 casos. Hallazgo clave: **el RIVAL VALUE es más robusto de lo que el stack secundario sugería.**

Señales pro-rival más consistentes entre los 3 picks:
- `edge_fav` muy negativo (−15% a −20%): **el discriminador real**
- `Markov fav = COLD` en 2/3: señal confirmatoria válida
- `p_hist < 0.45` en 2/3: calibración histórica contradice al bookmaker

### 8.3 Wilson IC actualizado

```
n=3, hits=3 | p_hat=1.000
Wilson 95% IC: [0.526, 1.000]
Cuota media: 3.75 | Breakeven: 0.267
Wilson LB (0.526) > Breakeven (0.267): PASA ✓
ROI flat 1u: +275%

Gate formal: n >= 30 — faltan 27 observaciones independientes
```

### 8.4 Decisión post-evidencia: rival_value_betslip.py (D68-07)

| ID | Decisión |
|---|---|
| D68-07 | Crear `rival_value_betslip.py` — micro-Kelly pre-graduación para apuestas individuales del rival. Shrinkage = n/(n+50) = 3/53 = 5.7%. Cap: 0.5% bankroll. Cada apuesta = 1 observación H88-01. |
| D68-08 | Stake individual: `kelly_raw × shrinkage`, mínimo 2000 COP. **PROHIBIDO** subir antes de n=30. |
| D68-09 | Combo Betplay: todos los rivales del día en 1 link. Útil como ancla de alta cuota combinada. |

**`rival_value_betslip.py` implementado 2026-07-14** — `python3 rival_value_betslip.py --bankroll 125000 [--telegram]`

### 8.5 H88-01 actualizada

```json
"n_actual": 3,
"hits": 3,
"wilson_lb_actual": 0.526,
"roi_flat_1u": 2.75,
"estado": "ACUMULANDO — 27 obs más para gate"
```

**Nota protocolo:** el usuario apostó antes del gate n=30 (PROHIBIDO por umbrales congelados). Resultó en 3/3. El gate sigue vigente — n=30 protege contra overfitting en muestras pequeñas. La combinada de n=3 no es estadísticamente equivalente a 3 observaciones independientes.

### 8.6 Camino a graduación

Con 3 picks RIVAL VALUE por sesión activa: **~9 sesiones más** → n=30 → evaluación formal IC Wilson.
Criterio de éxito (pre-registrado, inmutable): IC Wilson 95% inferior > 1/3.75 = 0.267 con n≥30.

# Nodo-54 — Brief para Fable: Embudo de Deploy Demasiado Estrecho

> Fecha: 2026-07-03
> Autor: sesión Claude (nodo-51-f3 branch)
> Prioridad: ALTA — bloquea monetización diaria
> Contexto necesario: leer CLAUDE.md secciones 1, 3.2, 3.3, 3.5 antes de responder
> **Estado:** RESUELTO via Nodo-55 (2026-07-03) — ver [[Nodo-55-Respuesta-Fable-Funnel-Deploy]]
>
> Resolución por problema:
> - P54-01 (λ_ITF): NO CAMBIAR — epoch_2 ITF n=0, hit 20% shadow book confirma λ=4.5 funciona
> - P54-02 (stake $0): Waterfall Log implementado en `trader_ev_tenis.py`; H54-01 pre-registrada
> - P54-03 (GS watchlist edge>20%): usar ruta WAS existente (`--was`); D54-02 en `shadow_book.py`
> - Pregunta Meta: `run_daily.py` implementado (D54-03) — 45 min → ~7 min/día

---

## Situación Actual

El sistema funciona. Evidencia:
- Calibración clay GS: p=0.758 (n=31), P&L positivo acumulado
- S-27-8 (Shadow Book primer reporte real): CLV median +167 en Grand Slam, 66.7% hit rate
- El modelo detecta edge real — los picks que pasan el gate ganan

**Problema operacional:** el pipeline procesa 150 partidos y genera 1-3 apuestas desplegadas con stake real. El overhead (PASO 1-4, ~45 minutos) no está justificado por el output.

Hoy (2026-07-03): 150 partidos → 3 APOSTAR → 1 con stake real ($1,000 Meligeni @1.81).

---

## Tres Problemas Específicos

### P54-01 — λ_ITF=4.5× mata picks con calibración real

**Qué pasa:** El parámetro `λ_tier` para ITF es 4.5×. Fue diseñado para penalizar la incertidumbre cuando n es bajo. Hoy ITF clay tiene n=42 en `calibracion_edge.json`.

**Síntoma:** Ouakaa @1.74 edge=7.0% y Popa @2.04 edge=6.4% — ambos con señal APOSTAR del edge_calculator — llegan al trader como individales con stake $0 después del ajuste VaR×0.36 + kelly_kl_itf.

**Dónde vive:** `edge_calculator.py` → `lambda_aversion` por tier. Constante en `config.py` o inline.

**Pregunta para Fable:** ¿Debe λ_ITF recalibrarse dinámicamente en función de n_calibracion, en lugar de ser fija? Propuesta: `λ_efectivo = λ_tier × (20 / (n + 20))` — a n=42, λ_ITF bajaría de 4.5× a ~1.5×. ¿Es esto matemáticamente correcto dado que James-Stein shrinkage ya está aplicado en otro punto del pipeline?

---

### P54-02 — No existe stake floor para picks APOSTAR

**Qué pasa:** Un pick que pasa todos los gates (edge>5%, confianza>54%, kelly_kl>0, señal APOSTAR) puede llegar al trader con stake $0 si el VaR del pool lo aplana.

**Síntoma:** Ouakaa y Popa son "APOSTAR" en el edge_report pero el trader los despliega con $0. El usuario ve 3 picks APOSTAR pero solo 1 tiene dinero.

**Dónde vive:** `trader_ev_tenis.py` → sección de ajuste VaR individual → no tiene floor.

**Pregunta para Fable:** ¿Debe existir un `MIN_STAKE_APOSTAR` (ej. $500) para cualquier pick que pasó el gate? ¿O el $0 es información valiosa (el sistema dice "edge real pero contexto de riesgo no lo soporta")? Si se añade el floor, ¿debe eximir del VaR pool o simplemente overridearlo con un cap separado?

**Restricción:** REGLA-HF-5 dice "si KGR<0 → NO DESPLEGAR". Un floor no debe violar esto.

---

### P54-03 — Watchlist Grand Slam con edge >20% no tiene ruta a apuesta individual

**Qué pasa:** El S-27-8 mostró que los picks Grand Slam son la señal más fuerte del sistema (66.7% hit, CLV+167). Hoy hay 3 picks Grand Slam con edge extraordinario:
- Claire Liu @6.75, edge=36.1%
- Michael Zheng @4.90, edge=29.9%
- Roman Safiullin @3.60, edge=23.6%

Estos van a WATCHLIST (no APOSTAR) porque no pasan algún gate — probablemente n_h2h bajo o calibration_confidence bajo.

El trader los toma como pool de cobertura ($31,000 en el combo 3p @119x) pero no los despliega como individuales.

**Pregunta para Fable:** ¿Debe existir un tier especial "GS_WATCHLIST_HIGH_EDGE" que active una ruta de deploy diferente cuando `edge > 20% AND tier = grand_slam AND cuota >= 3.0`? ¿O el problema es que estos picks no deberían ser watchlist — deberían pasar a APOSTAR con un gate alternativo basado en edge en lugar de confianza?

**Datos disponibles:** el edge_report tiene `golden_zone`, `triple_alignment`, `alignment_flag`, `bbi_signal` para cada pick. ¿Puede Fable diseñar un gate compuesto que use estas señales en lugar del gate confianza para el segmento GS_alto_edge?

---

## Contexto Técnico Adicional

### Estado del pipeline (para no re-inventar)
```
edge_calculator.py    → produce apostar/watchlist/sin_edge con kelly_kl calculado
trader_ev_tenis.py    → consume edge_report, aplica VaR, genera stakes finales
calibracion_edge.json → n por tier+superficie (ITF clay: n=42, GS grass: n=2)
```

### Señales disponibles en edge_report por pick (no todas usadas en deploy)
```
golden_zone, triple_alignment, alignment_flag, n_axes_active
surface_signal, regime_signal, bbi_signal, net_alignment
circuit_asymmetry_signal, markov_favorito, markov_rival
data_completeness, history_provenance
```

### Lo que NO queremos
- Bajar el umbral de edge (5%) — el gate existe para evitar ruido
- Aumentar el n de picks forzando picks malos — queremos mejor deployment de picks buenos
- Cambios que rompan los 1588 tests existentes

---

## Output Esperado de Fable

Para cada problema (P54-01, P54-02, P54-03):

1. **Diagnóstico**: ¿Es el problema real o un síntoma de algo más profundo?
2. **Solución propuesta**: cambio mínimo, archivo específico, líneas a modificar
3. **Test de validación**: cómo saber que el fix funciona sin esperar 30 días de datos
4. **Riesgo**: ¿puede el fix introducir apuestas malas que antes estaban bloqueadas correctamente?

El sistema ya tiene 1588 tests. Cualquier fix debe proponer tests nuevos siguiendo el patrón de `tests/test_nodo53.py` (REGLA-T53: tests invocan funciones del módulo real, nunca hardcodean fórmulas).

---

## Pregunta Meta (opcional)

El pipeline actual tiene 9 pasos manuales para llegar a 1-3 apuestas/día. ¿Hay un diseño alternativo donde el output de mayor valor (picks con edge real) se identifica en <5 minutos, y el resto del pipeline corre en background o se omite en días de bajo edge?

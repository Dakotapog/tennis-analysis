# PROPUESTA — Migración Bucket `?` en calibracion_edge.json

> **Estado:** SOLO PROPUESTA — nada ejecutado. Decisión al usuario (T7 Nodo-66).
> **Fecha:** 2026-07-13 | **Autor:** Sonnet (T7 Nodo-66)
> **Referencia:** Nodo-86 §1.1, CLAUDE.md §5 (buckets huérfanos)

---

## 1. Diagnóstico

### ¿Qué es el bucket `?`?

`theta_thompson()` en `edge_calculator.py:L336` construye la clave de calibración como:

```python
key = f"{superficie}_{tier}"   # ej: "clay_grand_slam", "grass_challenger"
```

Cuando un pick se cierra con `superficie='?'` o `tier='?'` (dato ausente en el
momento del registro), el resultado se acumula en `por_superficie['?']` y
`por_superficie_y_tier['?_?']`.

### Estado actual

| Bucket | n | wins | losses | hit% | Buckets normales (ref) |
|---|---|---|---|---|---|
| `?` (por_superficie) | 96 | 23 | 73 | **24.0%** | clay=64.3%, hard=61.4%, grass=57.2% |
| `?_?` (por_superficie_y_tier) | 96 | 23 | 73 | **24.0%** | mismo conjunto |
| `unknown` (por_superficie) | 75 | 46 | 29 | 61.3% | ← este es DISTINTO: superficie="unknown" pero válida |

**Impacto en producción:** cuando se procesa un pick con superficie desconocida,
`theta_thompson()` cae eventualmente al `global` (wins=2358, losses=1480, hit=61.4%),
NO al bucket `?`. El bucket `?` solo afecta a la calibración Thompson histórica del
stack de calibraciones — no al p_prior inmediato de los picks nuevos.

### Por qué el 24% es problemático

El 24% hit sugiere que estos 96 picks fueron registrados en un período de mala
calibración (probablemente antes del fix de normalización 2026-06-19 documentado en
`_nota_era_v2`) o con datos de superficie incorrectos. Si se dejaran crecer,
contaminarían la calibración `global` cuando se haga un rebalanceo.

---

## 2. Opciones

### Opción A — Renombrar a bucket explícito `real_money_unknown`

**Qué hace:** En `calibracion_edge.json`, cambiar la clave `"?"` a
`"real_money_unknown"` en `por_superficie`, e `"?_?"` a `"real_money_unknown_?"`
en `por_superficie_y_tier`.

```python
# Migración one-off (NO ejecutar sin decisión):
cal['por_superficie']['real_money_unknown'] = cal['por_superficie'].pop('?')
cal['por_superficie_y_tier']['real_money_unknown_?'] = cal['por_superficie_y_tier'].pop('?_?')
```

**Pros:**
- Sin pérdida de datos — los 96 resultados de dinero real se conservan
- El nuevo nombre es explícito y auditeable
- `theta_thompson()` nunca construye la clave `real_money_unknown` → estos datos
  quedan aislados y no contaminan futuros lookups

**Contras:**
- Los datos siguen siendo de calidad desconocida (24% hit)
- No se sabe cuántos provienen de picks pre-normalización-fix (potencialmente todos)
- Requiere actualizar cualquier script que haga lookup por `"?"` explícitamente

**Riesgo:** BAJO. Solo renombra — no modifica los wins/losses.

---

### Opción B — Re-atribuir por fecha+jugador contra h2h_results históricos

**Qué hace:** Para cada uno de los 96 picks en el bucket `?`, buscar en los
archivos `reports/h2h_results_enhanced_*.json` y `reports/edge_report_*.json` el
partido por jugador+fecha para recuperar la superficie y tier reales, y
re-acumular en el bucket correcto.

**Pros:**
- Máxima integridad de datos — los resultados quedan en los buckets donde pertenecen
- Puede mejorar la calibración de `clay_challenger` o `hard_itf` si esos eran los
  tiers reales

**Contras:**
- El join por nombre+fecha es frágil (homónimos, abreviaciones, fechas after-midnight)
- No garantiza match para los 96 — muchos h2h históricos podrían no existir
- Riesgo de double-counting si el join falla silenciosamente y un resultado queda
  en AMBOS buckets (el `?` y el real)
- Complejidad alta para beneficio incierto: si la mayoría son pre-era-v2, re-atribuirlos
  contamina los buckets correctos con datos históricos malos

**Riesgo:** ALTO. Requiere validación cuidadosa resultado por resultado.

---

### Opción C — Dejar los datos y solo excluir del stack de fallback

**Qué hace:** Modificar `theta_thompson()` en `edge_calculator.py:L336` para
ignorar el bucket `?` en el lookup (ya lo hace implícitamente — verificar que no
haya path donde `?` se devuelva como prior).

```python
# Verificación: theta_thompson nunca devuelve datos del bucket "?"
# porque ningún pick activo tiene superficie="?" en el flujo normal
```

**Pros:**
- Cero riesgo de modificar datos
- Los 96 resultados quedan auditables en el JSON sin contaminar calibraciones activas
- Es lo que el sistema ya hace en la práctica

**Contras:**
- El bucket `?` sigue creciendo si persiste la causa raíz (nuevos picks con superficie desconocida)
- La causa raíz (¿por qué llegaron picks con `?`?) no se investiga
- Visualmente confuso al inspeccionar el JSON

**Riesgo:** NINGUNO (no-op). Pero no resuelve el problema de fondo.

---

## 3. Recomendación preliminar

**Opción A + investigación de causa raíz.**

1. Ejecutar Opción A (renombrar) — bajo riesgo, sin pérdida de datos, el bucket
   queda aislado con nombre explícito.
2. Antes de ejecutar: verificar si los 96 picks son todos pre-2026-06-19
   (era anterior al fix de normalización). Si sí → descartarlos está justificado
   doctrinalmente (eran datos de calibración corrupta, ya documentado en `_nota_era_v2`).
3. Investigar en `betslip_registrar.py` y `validar_con_api.py` por qué un pick
   llega con `superficie='?'` al momento del cierre — ¿es un bug que sigue ocurriendo?

**Frontera de decisión del usuario:**
- ¿Queremos conservar los 96 como `real_money_unknown` (opción A)?
- ¿O son todos pre-era-v2 y podemos eliminarlos (opción A simplificada: `pop` sin `add`)?
- ¿O el problema está resuelto en producción y solo necesitamos limpiar la vista (opción C)?

---

## 4. Checklist de ejecución (cuando el usuario decida)

- [ ] Backup de `calibracion_edge.json` antes de cualquier cambio
- [ ] Verificar que `git diff data/calibracion_edge.json` muestra solo el cambio esperado
- [ ] Correr `python -m pytest tests/ --no-cov -q` post-cambio (1846 tests deben pasar)
- [ ] Verificar que `theta_thompson(cal, '?', '?')` retorna el valor del `global` (fallback correcto)
- [ ] Documentar la decisión en `docs/DECISION-LOG.md` como D-11

---

*REPORTE_SOLO — no ejecuta nada. Decisión al usuario.*

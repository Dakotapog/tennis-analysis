# Nodo-42 — Grass Surface Bootstrap

> **Fecha:** 2026-06-29 | **Severidad:** ESTRUCTURAL — Cold-start en superficie hierba
> Resuelve: confianzas 50-54% en Wimbledon 2026 — modelo honestamente no distingue en hierba, pipeline necesita modo de desbloqueo temporal

---

## Problema

El pipeline tiene calibración grass_grand_slam funcional (era_v2_wins=9, era_v2_losses=6, n=15, theta_thompson=0.5882) pero el motor de predicción produce confianzas de 50-54% en todos los partidos de Wimbledon 2026 porque:

1. El ajuste de pesos en hierba es hardcodeado (±0.05 fijos, `rivalry_analyzer.py` L1396-1401) — no usa datos calibrados, no escala con n
2. `common_opponents` es débil en hierba (pocos torneos grass/año → caminos transitivos escasos)
3. `form_recent` mezcla victorias en clay/hard del mes previo — señal ruidosa para predicción grass
4. Markov PELT detecta regímenes en datos mixtos, sin sesgo por superficie

El `combo_confianza_builder.py` filtra correctamente con `CONF_MIN=53.0%` — no es un bug, es el síntoma correcto de un modelo que honestamente no distingue en hierba. El deadlock documentado en el contexto de la sesión (grass_grand_slam inexistente) es un falso diagnóstico: la key ya existe con n=15.

**El problema real**: confianza de predicción baja, no falta de calibración. El pipeline necesita un modo de desbloqueo temporal para acumular observaciones reales de Wimbledon mientras el modelo mejora.

---

## Diagnóstico Técnico

```
calibracion_edge.json — estado real 2026-06-29:
  grass (por_superficie):     wins=244, losses=184, n=428, Thompson=0.5698
  grass_grand_slam (era_v2):  wins=9,   losses=6,   n=15  → theta_thompson=0.5882 ✅
  grass_challenger (era_v2):  wins=2,   losses=2,   n=4   → FALLBACK (n<10)
  grass_atp500 (era_v2):      wins=4,   losses=0,   n=4   → FALLBACK (n<10)

theta_thompson para grass+grand_slam:
  era_v2_n=15 ≥ 10 → devuelve 0.5882 directamente (no cae a global)
  B-08 no aplica (sale en paso 1)

rivalry_analyzer.py L1396-1401 — ajuste grass hardcodeado:
  common_opponents -= 0.05  (fijo, no calibrado)
  form_recent += 0.05       (fijo, no calibrado)

combo_confianza_builder.py:
  CONF_MIN = 53.0  (constante global)
  --threshold existe pero baja globalmente (contamina clay/hard)
  --superficie NO existe

Wimbledon 2026 — 19 partidos extraídos 2026-06-29:
  confianza rango: 50.2% - 55.7%
  picks > 53%: solo 3 (Jodar @1.19 bajo CUOTA_MIN, Bencic @1.11 bajo CUOTA_MIN, Van Assche @2.95 sin conf≥60%)
  resultado: 0 picks de Wimbledon en combo plan
```

---

## Solución Implementada

### Fase 1 — Flag --superficie grass en combo_confianza_builder.py

**Archivo:** `combo_confianza_builder.py`

Nuevo argumento `--superficie {grass,clay,hard}`. Cuando `--superficie grass`:

- `conf_min_efectivo` = 50.0 (en lugar de 53.0)
- `conf_c1_efectivo` = 55.0 (en lugar de 60.0)
- `stake_max_grass` = 500 (fijo, no Kelly, cap duro)
- VaR grass: budget cap = min(budget_normal, 1% × bankroll) = $1,250 sobre $125,000
- Watermark `[GRASS BOOTSTRAP — umbral reducido, stake cap $500]` en header del reporte
- Archivos .bat con prefijo `CC_GRASS_` (no `CC`) para distinguir de combos normales

**Cambios de código (4 funciones):**

1. `_categorizar_pick()` — acepta `conf_min=CONF_MIN, conf_c1=CONF_C1` como parámetros opcionales
2. `_extract_and_categorize()` — acepta y propaga `conf_min`, `conf_c1`
3. `_build_portfolio_v2()` — acepta `stake_max=None`; cuando definido, clampea todos los stakes a `min(stake_calculado, stake_max)`
4. `main()` — lee `args.superficie`, setea variables locales, pasa a las funciones anteriores

**Constraint crítico**: `--superficie grass` y `--threshold` son compatibles (`--threshold` puede sobrescribir el 50.0 default grass si el usuario quiere). El `stake_max=$500` no es sobrescribible por CLI — es un guard de seguridad.

### Fase 2 — Activación automática post-n≥30 (sin código nuevo)

Cuando `grass_grand_slam era_v2_n ≥ 30` (meta post-Wimbledon 2026):
- `calibration_confidence = 30/(30+20) = 0.60` — comparable a clay_challenger actual
- `theta_thompson` ya usa era_v2 desde n≥10 — **ningún cambio de código requerido**
- El modelo sigue produciendo 50-54% hasta que Fase 3 mejore features

### Fase 3 — Grass feature amplification (fuera de scope Nodo-42, spec para Nodo-43)

Reemplazar ajuste hardcodeado de ±0.05 en `rivalry_analyzer.py` con ajuste calibrado por n_grass del jugador (n_grass ≥ 20 → surface_specialization_grass activo). Implementar solo cuando n≥10 jugadores tienen historial grass suficiente.

---

## Alcance

**EN SCOPE:**
- `combo_confianza_builder.py`: flag `--superficie` con lógica grass bootstrap
- `.spec/01_Nodos/Nodo-42-Grass-Bootstrap.md`: este spec

**FUERA DE SCOPE:**
- `rivalry_analyzer.py` — no tocar ajuste grass hardcodeado (Nodo-43)
- `edge_calculator.py` — no tocar theta_thompson (ya funciona)
- `betslip_registrar.py` — no agregar modo shadow (observaciones $0 no distinguibles de reales)
- Bajar CONF_MIN global — vibe coding explícitamente rechazado

---

## Tests

- T42-01: `--superficie grass` acepta picks con conf≥50% (antes filtrados con 53%)
- T42-02: sin `--superficie grass`, CONF_MIN sigue siendo 53% (no contaminación)
- T42-03: con `--superficie grass`, stake por combo ≤ $500
- T42-04: output incluye watermark `[GRASS BOOTSTRAP]` cuando flag activo
- T42-05: `--superficie clay` (u otro valor inválido) no activa grass mode
- T42-06: VaR guard: total invertido ≤ min(budget_fase, 1% bankroll) en modo grass

---

## Criterio de Éxito

- Wimbledon 2026: ≥1 combo grass ejecutable por día con exposición ≤ $1,250/sesión
- Post-Wimbledon: `grass_grand_slam era_v2_n ≥ 30` en `calibracion_edge.json`
- Wimbledon 2027: pipeline completo sin override, CONF_MIN=53% con picks naturalmente ≥53% (requiere Nodo-43)

---

## Wikilinks

- [[Nodo-38-Portfolio-Aislamiento-Riesgo]] — arquitectura CORE/Satellite sobre la que opera
- [[Nodo-17]] — theta_thompson y jerarquía de calibración
- [[Nodo-43-Grass-Feature-Amplification]] (futuro) — ajuste calibrado por n_grass

---

Estado: IMPLEMENTADO — 2026-06-29

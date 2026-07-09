# MODEL-ROUTING.md — Tabla de Routing de Modelos

> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]
> **Revisión:** 2026-07-03 (inicial) — revisar mensualmente con datos del odómetro
> **Principio rector:** el modelo caro es como el Kelly grande — solo se despliega donde hay convicción de que la tarea lo requiere. Gate: *¿esta tarea tiene un verificador barato (tests, diff, output esperado)?* Si sí → modelo barato. Si el verificador es el juicio → modelo caro.

---

## Tabla de Routing

| Tarea | Modelo | Por qué |
|---|---|---|
| Correr tests, settle diario, close-snapshot, reportes pipeline | **Haiku** | Verificable por el propio output (tests verdes, JSONL escrito). El verificador barato permite el modelo barato. |
| Implementar con spec firmado (nodos ya escritos) | **Sonnet** | El spec elimina la ambigüedad; Sonnet ejecuta specs con precisión. |
| Escribir/corregir tests de bugs (REGLA-T53) | **Sonnet** | Requiere juicio módulo-vs-fórmula pero está reglado. |
| Auditoría de spec, decisiones de arquitectura, tradeoffs sin respuesta obvia | **Opus/Fable** | El único nivel donde el contexto de 59 nodos cambia la respuesta. |
| Lectura de reportes del shadow book, análisis estadístico de hipótesis | **Opus/Fable** | Interpretación de señales con alta incertidumbre, no ejecución mecánica. |
| Debugging de causa desconocida | **Empezar Sonnet → escalar** | Escalar a Opus solo tras 2 intentos fallidos con hipótesis documentadas. |
| parse_sessions / token_odometer / run_daily.py | **Haiku** | Output verificable directamente. Nunca escalar. |
| Generar spec nuevo nodo (nueva investigación) | **Fable** | Contexto de 59 nodos + tradeoffs de diseño — imposible para Haiku/Sonnet sin perder matices. |

---

## Ejemplos Concretos

### ✅ Usar Haiku
```bash
python -m pytest tests/ --no-cov -q          # tests (Haiku en background)
python3 shadow_book.py --settle 2026-07-03   # settle post-partido
python3 shadow_book.py --close-snapshot      # snapshot pre-partido
python3 pipeline_tracker.py --section shadow # reporte observabilidad
python3 token_odometer.py --report           # odómetro de costos
```

### ✅ Usar Sonnet
```bash
# Implementar D57-03 del spec Nodo-57 (champion gate)
# Escribir tests T57-01 a T57-09 (REGLA-T53)
# Fix de bug con causa conocida y spec escrito
# PR de código con tests ya escritos
```

### ✅ Usar Opus/Fable
```bash
# "¿Debo cambiar λ_ITF de 4.5 a 3.5 dados estos 20 resultados?"
# "Analiza el shadow book de 30 días — ¿el edge es real o ruido?"
# "¿Cómo diseñar el gate de campeón para N rounds variable por torneo?"
# "Audita la implementación de Nodo-56 — ¿hay casos edge que faltan?"
```

---

## Convención de Tags (odómetro)

Primera línea de cada sesión de Claude Code:
```
# TAG: impl nodo-58      ← implementación de nodo
# TAG: test nodo-57      ← escritura de tests
# TAG: audit nodo-55     ← auditoría / spec
# TAG: settle 2026-07-03 ← settlement post-partido
# TAG: analisis h52-01   ← análisis de hipótesis
# TAG: nodo nuevo-nodo   ← creación de nuevo spec
```

`%untagged` objetivo: **< 20%** (medido por `token_odometer.py --report`).

---

## Ratios de Costo (referencia)

| Modelo | Ratio | Input USD/MTok | Output USD/MTok |
|--------|-------|----------------|-----------------|
| Haiku 4.5 | 1× | $0.80 | $4.00 |
| Sonnet 4.6 | ~4× | $3.00 | $15.00 |
| Opus/Fable 4.6 | ~20× | $15.00 | $75.00 |

Una tarea hecha en Opus que Haiku podía hacer cuesta **~20× de más**.
Una sesión de auditoría sin scope en Opus = decenas de dólares. Medir antes de desplegar.

---

## Historial de Revisiones

| Fecha | Cambio | Basado en |
|-------|--------|-----------|
| 2026-07-03 | Tabla inicial | Spec Nodo-59 + odómetro Jun-03→Jul-03 ($1,292 total, 95.7% cache hit) |

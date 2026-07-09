# Nodo-55 — Respuesta Fable a Nodo-54: El Embudo No Está Roto; Es Opaco y Caro de Operar

> **Wikilinks:** [[Nodo-54-Brief-Fable-Funnel-Deploy]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-44-Watchlist-Alpha-Signal]] | [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] | [[Nodo-25-Dispersion-Guard-Safe-Combos]] | [[Nodo-15-Portfolio-HedgeFund]] | [[Nodo-26-Cross-Sectional-Signals]]
> **Fecha:** 2026-07-03
> **Estado:** COMPLETO (2026-07-03) — 1598 tests pasan
> **Contexto crítico:** este brief llegó <24h después del primer reporte del shadow book. El reporte dijo "GS promete, esperar n=30; ITF sin edge de modelo". Los tres problemas del brief proponen formas de ensanchar el embudo. La respuesta usa la disciplina construida hace 48 horas.
>
> **Implementación:**
> - A54-01a: ITF epoch_2 (post-2026-07-01) n=0 — λ_ITF=4.5 se mantiene, discusión cerrada por datos
> - P54-02 Parte 1: Stake Waterfall Log en `trader_ev_tenis.py` (LOG_STAKE_WATERFALL por pick APOSTAR)
> - P54-02 Parte 2: `stake_real` + `var_flattened` en shadow book via `update_trader_stakes()`
> - H54-01: pre-registrada en `validation/preregistered_hypotheses.json`
> - D54-02: sub-segmento `WATCHLIST+grand_slam+edge>=20%` añadido al reporte S-27-8 en `shadow_book.py`
> - D54-03: `run_daily.py` — orquestador PASO 0→4.3 + settle + daily_brief (45 min → ~7 min/día)
> - Tests T55-01→T55-05: 5 pasan (REGLA-T53 cumplida)
> - PROHIBIDO implementado: sin λ_tier recalibración, sin GS_WATCHLIST_HIGH_EDGE, sin MIN_STAKE incondicional

---

## 0. El Diagnóstico del Diagnóstico

Antes de los tres problemas, la aritmética que reencuadra todo:

```
Los 3 picks "extraordinarios" de P54-03:
  Claire Liu     @6.75, edge=36.1%  →  p_implicita=0.148  →  p_modelo = 0.509
  Michael Zheng  @4.90, edge=29.9%  →  p_implicita=0.204  →  p_modelo = 0.503
  Roman Safiullin @3.60, edge=23.6% →  p_implicita=0.278  →  p_modelo = 0.514
```

**Los tres tienen p_modelo ≈ 0.50-0.51.** Esta es la firma EXACTA del phantom edge documentada en Nodo-32 (BUG-32-1: "el modelo dice moneda al aire, la aritmética dice apuesta fuerte"), el patrón de Majdandzic/Fiadosik (Nodo-33) y de Mario Arce @9.50 edge=60.4% (Nodo-49). El "edge extraordinario" no es convicción del modelo — es la distancia entre un coin-flip y una cuota larga.

**Y al mismo tiempo:** es el mismo patrón que WAS (Nodo-44) validó ganando el 29-jun con Carreno @3.30 y Safiullin @2.65 — el MISMO Safiullin que hoy aparece @3.60. La tensión no se resuelve con un gate nuevo hoy: se resuelve con H52-01 (ya pre-registrada, n=30) y la ruta WAS que ya existe.

**El embudo estrecho es correcto para el estado actual de conocimiento.** Lo que sí está roto: (a) el pipeline es opaco — nadie puede decir con precisión en qué paso un stake se vuelve $0; (b) cuesta 45 minutos manuales por día; (c) los picks aplastados por VaR se pierden como datos. Los tres problemas de abajo se responden desde ahí.

---

## P54-01 — λ_ITF=4.5× — VEREDICTO: NO RECALIBRAR

### Diagnóstico: la premisa está contradicha por los datos más nuevos del propio sistema

1. **El S-27-8 de ayer:** ITF hit%=20.0% (n=20, IC=[8.1%, 41.6%]), ROI −43.5%. Los 2 picks ITF con apostar=True y stake real (Waldner, Kovgan) **perdieron ambos**. El λ=4.5 aplastando a Ouakaa y Popa pudo literalmente haber ahorrado dinero ayer. Con la evidencia disponible, el freno ITF no es el problema — es el sistema funcionando.
2. **n=42 está contaminado por epochs.** El addendum de Nodo-52 §F es explícito: recalibraciones Challenger/ITF usan solo epoch-3 (post 2026-06-30, post fix del ranking Nodo-47). ¿Cuántas de las 42 observaciones son epoch-3? Probablemente <15. El "n real" para relajar λ no es 42.
3. **La propuesta duplica un mecanismo que ya existe.** La capa L5 del Kelly-KL (`calibration_confidence = max(0.30, n/(n+20))`, Nodo-32 §2.2.2) YA escala el Kelly con el n de calibración — a n=42 el Kelly ya opera al 68% en vez del 30%. λ_tier fue diseñado (Nodo-17/21) para capturar la incertidumbre ESTRUCTURAL del dominio: mercado menos eficiente, H2H escaso, rankings volátiles. Esa incertidumbre no desaparece con n=42 puntos de calibración — el hit 20% de ayer la confirma. Hacer λ_tier ∝ 1/n mezcla dos incertidumbres distintas (muestral y estructural) en un solo parámetro, y la muestral ya está en L5.

### Solución propuesta: ninguna modificación de λ. Dos acciones de datos:

- **A54-01a:** Sonnet reporta el desglose de `calibracion_edge.json` ITF por epoch (wins/losses en epoch-1/2/3). Si epoch-3 tiene n<15, la discusión se cierra sola.
- **A54-01b:** La decisión queda gobernada por H52-02 (n≥30 en shadow book, segmento itf). Si ITF gradúa, se revisa λ con evidencia; si no, λ=4.5 se queda.

### Test de validación: N/A (no hay cambio de código)
### Riesgo de NO hacer nada: perder algunos picks ITF buenos durante 3-4 semanas — a costo $0 porque el shadow book los registra igual. Riesgo del cambio propuesto: re-abrir el grifo del tier con hit 20% documentado. Asimetría obvia.

---

## P54-02 — Stake $0 en picks APOSTAR — VEREDICTO: NO FLOOR TODAVÍA; PRIMERO VISIBILIDAD, DESPUÉS RUTA GANADA

### Diagnóstico: es imposible decidir "floor sí/no" sin saber DÓNDE muere el stake

El pipeline de sizing tiene al menos 6 multiplicadores en cadena y un acantilado:

```
kelly_kl → ×0.25 (quarter) → ×portfolio_factor(ρ,N) → ×VaR_adjust → ×meta_markov → MIN_BET=$1000 (cliff)
```

Hipótesis probable (verificar): el stake calculado cae debajo de `MIN_BET=1000` y se convierte en $0 — un acantilado, no un ajuste. En microestructura de mercados esto es el problema clásico del *tick size mínimo*: la pregunta de diseño es zero-out vs round-up, y hoy ni siquiera sabemos cuál cliff está actuando. Segunda hipótesis: el VaR del pool está siendo consumido por el combo 3p @119x ($31,000) y aplasta a los individuales — si es así, el bug es de aislamiento de presupuestos (el budget cascade 40/40/20 de Nodo-13 debería proteger a los individuales de la exposición de combos), no de floor.

### Solución propuesta (cambio mínimo, dos partes):

**Parte 1 — Stake Waterfall Log (`trader_ev_tenis.py`, ~20 líneas):** para cada pick APOSTAR, loggear la cascada completa:

```
LOG_STAKE_WATERFALL: Ouakaa | kelly_kl=0.034 → q_kelly=$1,062 → ×pf(0.87)=$924
                     → ×var(0.36)=$333 → ×mm(1.0)=$333 → MIN_BET_CLIFF → $0
```

Esto responde la pregunta real (¿cliff MIN_BET, VaR compartido con combos, o kelly diminuto?) en UNA sesión de producción. Es el equivalente al P&L attribution de un fondo: no se arregla lo que no se ve.

**Parte 2 — Los aplastados se vuelven dataset:** el hook del shadow book añade a cada registro `stake_real` y `var_flattened: true/false`. Pre-registrar **H54-01** (congelada hoy): *"Los picks APOSTAR con stake_real=0 tienen hit% y CLV ≥ que los APOSTAR financiados"*, n=30, misma métrica de graduación de Nodo-52 §6. Si H54-01 gradúa → el floor se implementa como **NIVEL 1 de la escalera existente** (1 unidad flat, dentro del session_budget de M-26-2, fuera del pool VaR de combos). El floor no se decreta: se gana con datos. Y si el waterfall revela que el culpable es el combo consumiendo el VaR de los individuales, el fix es de aislamiento de presupuesto — deuda **D54-01**, prioridad ALTA, resuelve el síntoma sin ningún floor.

### Test de validación:
- T55-01: waterfall log presente para todo pick APOSTAR (llama al módulo real, REGLA-T53)
- T55-02: `var_flattened` y `stake_real` presentes en registros del shadow book
- T55-03: pick con stake calculado $800 y MIN_BET=1000 → waterfall muestra `MIN_BET_CLIFF` como causa terminal

### Riesgo: cero — Parte 1 y 2 son observabilidad pura, no cambian ninguna decisión de deploy. REGLA-HF-5 intacta.

---

## P54-03 — GS watchlist edge>20% — VEREDICTO: LA RUTA YA EXISTE (WAS); NO CREAR GATE NUEVO

### Diagnóstico: el gate compuesto propuesto es p-hacking con n=6

Crear "GS_WATCHLIST_HIGH_EDGE" hoy sería tallar un segmento **después** de ver un buen día (n=6 GS settled, un solo día de datos) — la violación exacta de la disciplina de pre-registro congelada hace 48 horas (Nodo-52 §3). Y la aritmética del §0 muestra que los tres picks son p_modelo≈0.51: el gate de confianza los bloquea porque debe bloquearlos para Kelly.

**Pero el sistema YA construyó la ruta para exactamente este perfil, y la validó:** Nodo-44 (WAS). Criterios: watchlist + edge≥10% + cuota≥2.0 + señal Markov. Deployment: stake fijo mínimo/promo (REGLA-WAS-1: nada de Kelly hasta n≥30). Flag ya implementado: `betplay_combo_builder.py --was` (D44-01 ✅). Safiullin ganó por esta ruta el 29-jun @2.65 y hoy reaparece @3.60 — el sistema ya conoció este caso y ya escribió su respuesta.

### Solución propuesta (cero código nuevo de gates):

1. Correr `--was` sobre los picks del día. Si Liu/Zheng/Safiullin cumplen los criterios WAS (verificar señal Markov de cada uno — sin señal Markov NO son WAS, son coin-flips puros y se quedan en shadow), se despliegan a **stake mínimo fijo** por la ruta WAS existente.
2. El shadow book ya los registra como WATCHLIST con `gate_bloqueante` — H52-01 acumula. A n=30, la hipótesis decide si este perfil merece ruta permanente con qué sizing.
3. **D54-02 (única mejora real detectada):** el reporte S-27-8 debe mostrar el sub-segmento `WATCHLIST ∩ tier=grand_slam ∩ edge≥20%` como corte visible (es intersección de cortes ya pre-registrados: status × tier × banda de cuota — NO es segmento nuevo, es visualización de uno existente). Así la pregunta de P54-03 se responde sola cada semana con el n que lleve.

### Test de validación:
- T55-04: pick con edge=36%, cuota=6.75, sin señal Markov → NO califica WAS (el filtro existente lo excluye)
- T55-05: pick con edge=23%, cuota=3.60, rival COLD conf≥0.60 → califica WAS, stake=mínimo fijo

### Riesgo: el riesgo real es el inverso — crear el gate compuesto hoy y que el perfil p≈0.51 + cuota alta repita el 12.5% de hit de la golden zone original (Nodo-32 BUG-32-3: el mismo razonamiento "el bookmaker no ve" ya falló una vez cuando no exigía convicción del modelo). La ruta WAS limita la exposición a stake mínimo mientras el n decide.

---

## Pregunta Meta — El rediseño operacional (aquí está el valor real del brief)

### El reencuadre: 150→3 no es un embudo roto; es el ratio normal de un negocio de edge

Un fondo de venture capital revisa 150 deals para financiar 2-3. Una mesa de trading escanea el universo entero para ejecutar un puñado de órdenes. **El ratio no es el problema — el costo manual por revisión sí.** La solución no es bajar la vara: es que revisar cueste casi nada.

### Diseño: `run_daily.py` — el pipeline como demonio, la atención humana como recurso escaso

```
06:00  run_daily.py (cron o un solo comando):
       PASO 0 (si rankings >7d) → PASO 1 → PASO 2 (con budget Playwright F3)
       → PASO 3 + shadow-log → PASO 3.5 → PASO 4 por tier del día
       → settle de AYER (--settle --retry, recoge ITF rezagados)
       → genera UN archivo: reports/daily_brief_FECHA.txt

08:30  close-snapshot GS (REGLA-SB-1, antes de sesión europea)
12:30  close-snapshot Challenger/ITF americano

HUMANO: lee daily_brief (5 min). Contiene SOLO:
  - Picks APOSTAR con stake>0 y su waterfall
  - Candidatos WAS del día (si hay promo activa)
  - Alertas: NO_DATA count anómalo, dispersion BLIND, breaker cerca
  - S-27-8 delta: qué hipótesis avanzó (ej. "H52-01: n=9/30")
```

Tiempo humano: 45 min/día → ~7 min/día. El pipeline completo SIGUE corriendo entero todos los días — no se puede recortar porque **el universo completo es el combustible del shadow book** (recortar partidos = recortar n = alargar las 3-4 semanas). Lo que se elimina es el costo de atención, no el cómputo. Y el calendario alfa (idea previa, ahora implementable trivialmente) marca qué días merecen atención humana ampliada: días de qualifying, lunes post-transición — el resto, el brief de 5 minutos basta.

**Deuda D54-03:** `run_daily.py` orquestador — prioridad ALTA, es el fix real del dolor que motivó el Nodo-54.

---

## Orden de implementación para Sonnet

```
1. A54-01a — desglose calibración ITF por epoch (solo query, 10 min)
2. Parte 1 P54-02 — Stake Waterfall Log + T55-01/03 (una sesión)
3. Parte 2 P54-02 — stake_real/var_flattened al shadow book + T55-02 + pre-registro H54-01
4. P54-03 — correr --was hoy; verificar señales Markov de Liu/Zheng/Safiullin;
   deploy solo los que califiquen, a stake mínimo; D54-02 (corte visible en S-27-8)
5. D54-03 — run_daily.py orquestador + daily_brief
6. D54-01 — SOLO si el waterfall muestra VaR compartido combos↔individuales:
   aislar presupuestos según cascade 40/40/20 original
PROHIBIDO: tocar λ_tier, crear GS_WATCHLIST_HIGH_EDGE, añadir MIN_STAKE incondicional.
Baseline: 1588 tests siguen pasando.
```

## Cierre — para el registro del proyecto

Este brief llegó un día después de que el primer reporte del shadow book recomendara esperar. Eso no es una crítica a quien lo escribió — es el patrón documentado del proyecto operando (Nodo-25: post-13-jun; Nodo-15: post-01-jun), y la razón por la que el árbol de decisión existe. La respuesta del sistema fue: no ensanchar el embudo, hacerlo transparente (waterfall), barato de operar (run_daily), y dejar que las rutas ya construidas (WAS, escalera de graduación) manejen los casos límite mientras el n decide. Si dentro de 30 días H52-01, H52-02 y H54-01 dicen que el embudo era demasiado estrecho, se ensancha — con intervalos de confianza, no con tres picks de un jueves.

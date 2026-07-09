# FABLE_02_TENIS_DOCTORADO — Auditoría, Conexiones de Dominio Cruzado, SPEC Ejecutable y Protocolo de Auditoría

> **Fecha:** 2026-07-06 | **Firmado:** Fable
> **Reemplaza:** FABLE_02_TENIS.md (queda como histórico — este documento es autosuficiente junto con CLAUDE.md)
> **Regla de baselines:** NUNCA hardcodear el número de tests. El documento fuente citaba 1585; hoy el repo está en ≥1691 (Nodo-63). Toda fase captura su baseline con `python -m pytest tests/ --no-cov -q | tail -1` ANTES de empezar y verifica el MISMO número + nuevos DESPUÉS.

---

# §1 — AUDITORÍA DE FABLE_02 (Fase 1 del prompt)

## §1.1 Veredictos por recomendación

| Recomendación FABLE_02 | Veredicto | Razón |
|---|---|---|
| Hermes como clasificador barato con patrón "barato filtra, caro decide residual" + líneas rojas (nunca Kelly/apostar/hipótesis) | ✅ QUEDA, gated por medición | Bien anclado. Pero se implementa SOLO si la frecuencia medida de ambigüedades lo justifica (§4 Fase 3) — construir el filtro antes de medir el problema es el anti-patrón del proyecto |
| n8n para close-snapshot con timing variable por partido | ✅ QUEDA — es el hallazgo operativo más valioso del documento | REGLA-SB-1 con horas fijas es aproximación; la hora real de cada partido varía. Cron fijo es la herramienta equivocada; n8n con lectura del feed es la correcta |
| Playwright MCP (árbol de accesibilidad) como fallback | ✅ QUEDA | Ataca la palabra "frágil" del propio CLAUDE.md. Solo ruta fallback |
| Validación empírica de ρ contra shadow book | ✅ QUEDA, profundizada en §3.2 | El instinto es correcto; le faltaba el método estadístico anti-p-hacking |
| Graphify **Y** codebase-memory-mcp **Y** claude-memory-compiler **Y** Tamp | 🟠 CORREGIDO: elegir UNO de los dos grafos | La "división de responsabilidades" del documento es una racionalización de ~70% de solapamiento. Regla de decisión en §4 Fase 1. Tamp queda (transversal, cero riesgo). memory-compiler queda (markdown al vault, sin motor nuevo) |
| GitHub Actions para validación pre-partido y brief | 🔴 RECHAZADO — reemplazar por scheduler local | Error de arquitectura: las Actions corren en runners de GitHub y NO tienen acceso a `reports/` local salvo que se commiteen betslips y datos de apuestas al repo remoto — inaceptable por privacidad y operación. La misma lógica corre en cron local o n8n (que ya está) |
| Sprint 3 auditoría forense kelly_kl=0.0 | 🟠 CONDICIONAL | La propia nota de reconciliación lo dice: verificar con git PRIMERO. Si está resuelto, el sprint se reemplaza por el fix trivial de `prediccion_ganador` (Vacío 4) |
| Plan de "un mes en 4 semanas" de infraestructura | 🔴 RECHAZADO como calendario | Viola la priorización M0/M1/M2 de Nodo-59: 7 sistemas nuevos compitiendo con la ventana de acumulación del shadow book. El SPEC de §4 reorganiza por fases con dependencias y gates de datos, sin semanas |
| Plantilla de 7 archivos por nodo (CBT pattern) | ✅ QUEDA simplificada | El formato actual de nodos ya converge a: contexto → causa raíz → fix → tests → orden → prohibido → auditoría. Formalizar ESA plantilla (ya probada 63 veces), no importar una ajena |

## §1.2 Resolución de los Vacíos 1-4

**Vacío 1 (.graphifyignore):** `tests/`, `data/`, `reports/`, `ml_datasets/`, `logs/`, `__pycache__/`, `*.json`, `*.jsonl`, `*.log`, `*.bat`, `*.txt`, `drivers/`, `screenshots/`, `graphify-out/`. Los `.md` de nodos SÍ entran (son la memoria semántica del proyecto).

**Vacío 2 (migración historial):** no migrar destructivamente. El vault YA existe (los 63 nodos SON el vault). Acción real: CLAUDE.md conserva solo constitución + estado actual + punteros; la sección "Lo que pasó" se mueve a `docs/DECISION-LOG.md` (entregable pendiente de Nodo-59 M1) con los 5 errores como entradas fechadas. Cero pérdida: git conserva todo.

**Vacío 3 (precedencia vault vs CLAUDE.md):** política de tres líneas: (1) los nodos y ADRs son historia inmutable — nunca se editan, se añade una entrada nueva o se marca `SUPERSEDED por [[Nodo-XX]]`; (2) CLAUDE.md es una VISTA derivada — si contradice al nodo más reciente, CLAUDE.md está desactualizado por definición; (3) un chequeo semanal de Haiku compara la tabla de bugs/estados de CLAUDE.md contra los headers de los últimos 10 nodos y reporta contradicciones. La fuente de verdad es siempre el registro inmutable más reciente.

**Vacío 4 (`prediccion_ganador=None`):** fix conocido y trivial — entra en Fase 0 (§4): eliminar el campo top-level trampa y que todo consumidor use `ranking_analysis.prediction.favored_player`. Test que invoca al módulo (REGLA-T53).

---

# §2 — CORRECCIONES A NODOS 61/62/63 (entran al SPEC)

**C61-A (FORENSE, prioridad máxima):** los boosts GCS observados en producción (E4 de Nodo-61: ×1.15, ×1.13, ×1.03, **×0.92**) NO corresponden a las constantes especificadas (×2.2/×1.8/×1.5). O el boost se aplica a un componente y se renormaliza (entonces la documentación de "multiplicador al final_score" es incorrecta), o hay un bug de cálculo adicional al de Keys. Sonnet debe trazar dónde exactamente se aplica el multiplicador y reconciliar spec↔producción. Ningún otro trabajo GCS hasta cerrar esto (D61-F6 ampliado).

**C61-B (gobernanza):** el boost GCS está ACTIVO en producción (Eala APOSTAR) cuando el Nodo-60-ADDENDUM lo dejó GATED OFF. La base aparente es el retro-scan A60-01 (n=54, 64.8%) — evidencia retrospectiva, no la prospectiva de H60-01. Decisión requerida y documentada: o (a) se registra formalmente "activación por evidencia retrospectiva n=54, con H60-01 prospectiva continuando y regla de descenso si el prospectivo contradice", o (b) se apaga hasta n_stop prospectivo. Cualquiera de las dos es defendible; lo indefendible es la activación sin registro de la decisión. Además, el docstring de `_gcs_boost_multiplier` que dice "validado por H60-01" debe corregirse a "prior retrospectivo A60-01; H60-01 prospectiva pendiente".

**C62-A:** la promoción Cat-C2→Cat-C1 por alpha (D62-05) es una relajación de gate con pesos inventados. No se revierte (el diseño es razonable), pero se instrumenta: pre-registrar **H62-01**: "picks `alpha_promoted=True` tienen hit% ≥ picks Cat-C1 orgánicos", n_stop=30, y el flag `alpha_promoted` debe llegar al pick_snapshot del shadow book. Si a n=30 los promovidos rinden peor, la promoción se apaga con una constante.

**C63-A:** el guard n<8 debe además ENCOLAR al jugador para el fallback Playwright de Nodo-49/F3 (n<8 = "FlashScore no tiene datos" — exactamente el caso que el fallback existe para enriquecer), no solo suprimir el decay. Una línea: si `n < _MIN_HISTORY_FOR_DECAY and match_id` → candidato a cola Playwright dentro del presupuesto.

**C63-B (riesgo de portfolio):** con Anchor Combos, el sistema ya tiene 6 capas de combos (CORE, Satellite, mega, safe, GCS sub-plan, Anchor×3 tiers). El gap V-26-2a (session_budget solo cubre megas) pasa de deuda menor a urgente: **un solo governor de presupuesto de sesión que sume TODAS las capas de combos contra el límite M-26-2** antes de emitir cualquier .bat. Nodo propio en Fase 2 del SPEC.

---

# §3 — CONEXIONES DE DOMINIO CRUZADO (Fase 2 del prompt — las que Sonnet no podía ver)

Cada una: origen → principio → ancla exacta en el pipeline → nodo nuevo → test. Criterio de inclusión cumplido: todas anclan a un archivo/constante real.

## §3.1 SPRT — Test Secuencial de Wald (diseño experimental secuencial)
**Origen:** control de calidad industrial (Wald 1945); ensayos clínicos secuenciales.
**Principio:** en vez de n_stop fijo, dos fronteras pre-registradas sobre el log-likelihood ratio acumulado: `A = ln((1-β)/α)`, `B = ln(β/(1-α))`. Con α=β=0.05: la hipótesis puede CONCLUIR con n=15-20 si la evidencia es fuerte, o continuar más allá de 30 si es ambigua — con tasas de error controladas. El optional stopping es legítimo PORQUE las fronteras están pre-registradas.
**Ancla:** `validation/preregistered_hypotheses.json` (campo n_stop de H52-01…H62-01) + `hypothesis_tracker`.
**Nodo-64:** añadir modo SPRT al tracker: por hipótesis, H0: p=breakeven vs H1: p=breakeven+δ (δ pre-registrado, ej. +0.08); LLR por observación Bernoulli; veredicto ACEPTA/RECHAZA/CONTINÚA. Las hipótesis existentes conservan n_stop=30 como tope máximo (no se cambian a mitad de muestra); las NUEVAS nacen con SPRT.
**Test T64:** secuencia sintética con p=0.70 y breakeven=0.50 → RECHAZA H0 antes de n=20; con p=0.50 → CONTINÚA en n=30 (invoca la función real del tracker).

## §3.2 Block Bootstrap + Recalibración por Épocas — ρ y λ medidos, no asumidos
**Origen:** econometría de series financieras (Politis-Romano); validación walk-forward de quant trading.
**Principio:** ρ por tier es un parámetro ESTIMABLE: correlación pareada de outcomes Bernoulli entre picks de la misma sesión+tier, con remuestreo por bloques (la sesión es el bloque — preserva la dependencia intra-sesión) → IC del 90%. Anti-p-hacking estructural: las constantes solo pueden cambiar en **ventanas de recalibración pre-agendadas** (día 1 de cada mes), nunca al ver un resultado, y solo si el valor actual cae FUERA del IC.
**Ancla:** tabla ρ en `trader_ev_tenis.py` (0.25/0.20/0.15/0.10/0.05) y `LAMBDA_TIER_MULTIPLIER` en `edge_calculator.py`.
**Nodo-65:** `analysis/rho_empirical.py` — lee shadow book settled, agrupa por (sesión, tier), bootstrap B=2000 bloques, reporta ρ̂ + IC por tier. GATE de datos: requiere ≥15 sesiones con ≥3 picks settled del mismo tier — hoy probablemente insuficiente → el script corre en modo REPORTE desde ya, la recalibración espera el gate.
**Test T65:** con fixture de outcomes independientes → ρ̂ IC contiene 0; con outcomes clonados → ρ̂ IC contiene ~1 (invoca el módulo).

## §3.3 Favorite-Longshot Bias — el breakeven por banda que explica la paradoja ITF
**Origen:** economía del betting (Thaler & Ziemba 1988): los longshots están sistemáticamente sobrepreciados — su precio de cierre implica más probabilidad de la real.
**Principio:** `breakeven = 1/cuota_media` asume mercado insesgado. Con FLB, el breakeven REAL de la banda 4.0+ es más alto que 1/cuota. Esto explica la paradoja ITF del primer S-27-8 (CLV+32 con hit 20%): las cuotas largas ITF se acortan hacia el cierre por FLB estructural, generando CLV+ SIN información — el "edge de precio" puede ser en parte artefacto del sesgo.
**Ancla:** cálculo de breakeven en `shadow_book.report()`; H52-08 (banda 2.00-2.50); segmento curva-U.
**Nodo-66:** estimar la curva FLB empírica por tier: hit% real vs p_implícita de cierre por banda de cuota (flashscore_ref, n creciente); el reporte muestra breakeven ajustado por banda junto al ingenuo. La graduación de segmentos (Nodo-52 §6) usa el ajustado cuando n_banda≥30.
**Test T66:** fixture con hit%=p_implícita en todas las bandas → curva plana, ajuste=0; fixture con longshots ganando menos que su implícita → breakeven ajustado > ingenuo en esa banda.

## §3.4 CUSUM + PSI — Drift del MODELO completo (MLOps financiero)
**Origen:** control estadístico de procesos (Page 1954) + monitoreo de modelos crediticios (Population Stability Index).
**Principio:** dos detectores complementarios. **CUSUM sobre el Brier diario**: `S_t = max(0, S_{t-1} + (brier_t − brier_ref − k))`; alarma si S_t > h — detecta degradación del output antes de que el P&L la muestre. **PSI sobre las distribuciones de entrada** (mezcla de provenance, distribución de n_partidos por jugador, distribución de edge): PSI>0.25 = el mundo cambió — habría detectado la clase de bug Rodriguez (cobertura FlashScore) y el epoch-2 de Nodo-47 en días, no semanas. M-26-4 vigila la sesión; esto vigila el modelo.
**Ancla:** shadow book settled (Brier); Panel 3/5 del dashboard (Nodo-58); epochs de calibración (§F addendum 52).
**Nodo-67:** `analysis/drift_monitor.py` con `--report` diario integrado al daily_brief; constantes k=0.005, h=0.05 provisionales y ETIQUETADAS como tales (mismo trato que Nodo-57).
**Test T67:** serie con salto de Brier +0.08 en t=10 → alarma antes de t=20; serie estacionaria → sin alarma.

## §3.5 Predicción Conformal — la banda NO-BET con garantía, no con 54% arbitrario
**Origen:** conformal prediction (Vovk); cobertura distribution-free.
**Principio:** con los residuos `|resultado − p_modelo|` de los settled del shadow book, el cuantil (1−α) define un intervalo por pick: `[p−q, p+q]`. Si el intervalo contiene 0.5 → el modelo no distingue el pick de una moneda CON GARANTÍA de cobertura → NO-BET. Reemplaza el umbral fijo <54% (Nodo-53 Fase E) por uno que se ensancha solo cuando el modelo está mal calibrado y se estrecha cuando mejora — por tier si hay n.
**Ancla:** banda NO-BET de `generar_tabla_favoritos2.py` (Fase E Nodo-53); Brier por bin de S-27-5.
**Nodo-68:** `analysis/conformal_band.py`; GATE de datos: n≥50 settled global (≥30 por tier para bandas por tier). Modo REPORTE primero: mostrar la banda conformal junto a la fija sin cambiar decisiones, hasta comparar cobertura empírica.
**Test T68:** con residuos sintéticos de cuantil conocido → q correcto ±ε; pick con p=0.52 y q=0.06 → NO-BET.

## §3.6 Cohortes Emparejadas — el auditor de patrones que industrializa lo que GCS hizo a mano
**Origen:** inferencia causal observacional (matching por propensión, Rosenbaum-Rubin).
**Principio:** el próximo "patrón del martes" no debe requerir otro addendum artesanal de tres carriles. Protocolo automático: para cada pick con el patrón candidato, buscar 1-3 controles settled emparejados en (tier, banda cuota, banda p_modelo, superficie, epoch); comparar hit% patrón vs controles con test pareado (McNemar). Convierte el retro-scan (sesgo de selección alto) en estudio de cohorte emparejada (sesgo acotado) — y produce automáticamente el "estado_inicial" honesto del pre-registro.
**Ancla:** el proceso A60-01/A60-02 del Nodo-60-ADDENDUM; el shadow book como pool de controles.
**Nodo-69:** `analysis/pattern_audit.py --pattern <campo=valor>` → tabla emparejada + p-valor + plantilla de hipótesis lista para pre-registrar.
**Test T69:** patrón sintético sin efecto real → diferencia no significativa; con efecto inyectado +20pp → detectado.

## §3.7 CPPI — Kelly con piso de supervivencia (drawdown-constrained)
**Origen:** portfolio insurance (Black-Perold).
**Principio:** el objetivo declarado del proyecto es sobrevivir hasta que el edge madure. CPPI lo codifica: `cushion_t = (bankroll_t − FLOOR)/bankroll_t`; todos los stakes se multiplican por `min(1, m·cushion_t)`. Con FLOOR=70% del bankroll pico y m=2: a bankroll pleno opera normal; tras un drawdown del 15%, opera al 30% de tamaño; **matemáticamente no puede perforar el piso** por sizing (solo por gaps, que en apuestas liquidadas no existen). Complementa VaR (que limita la sesión) con una restricción de trayectoria (que limita la ruina acumulada) — y automatiza la "regla de descenso" que hoy depende de disciplina.
**Ancla:** `trader_ev_tenis.py` (capa de sizing final, después de VaR); REGLA-HF-5; el waterfall de Nodo-55 gana un eslabón `×cppi`.
**Nodo-70:** implementación de ~20 líneas + eslabón en el waterfall log; constantes FLOOR/m congeladas y etiquetadas provisionales.
**Test T70:** bankroll = pico → factor 1.0; bankroll = FLOOR → factor 0.0; monotonía entre ambos (invoca la función).

## §3.8 Velocidad de Línea (Kyle's λ) — steam formalizado
**Origen:** microestructura (Kyle 1985): el impacto de precio por unidad de flujo informado.
**Principio:** M-26-3 compara dos fotos; la información está en la DERIVADA. Con la serie temporal de `odds_series.py` (spec pendiente día-3): `velocity = Δcuota/Δt` normalizada por la volatilidad típica de la banda → z-score. STEAM = z < −2 (acortamiento anómalo). Alimenta H52-05 con una definición precisa y el veto asimétrico futuro con un umbral estadístico en vez de un porcentaje fijo.
**Ancla:** `line_movement_signal()` (M-26-3); `reports/odds_series/` (por crear); H52-05.
**Nodo-71:** extensión de odds_series con velocity z-score por snapshot; solo REPORTE hasta que H52-05 concluya.
**Test T71:** serie con caída de 4.0→2.0 en 2h vs volatilidad típica 5% → z<−2; serie plana → |z|<1.

---

# §4 — SPEC EJECUTABLE (Fase 4 — fases con dependencias, sin calendario)

**Reglas globales de toda fase:** baseline pytest capturado antes/después (número real del momento, jamás hardcodeado); REGLA-T53 en todo test; GIT-FIRST antes de crear cualquier módulo; cero impacto en producción (todo lo nuevo es READ-ONLY o flag-OFF hasta su gate); si el criterio de verificación falla → rollback con `git checkout` y escalar a Fable con el output del fallo, NO improvisar un fix alternativo.

### FASE 0 — Reconciliación (prerrequisito: ninguno) ✅ COMPLETA
1. ✅ Capturar baseline real de pytest: 1756 passed (2026-07-08).
2. ✅ `git log -- '*kelly_kl*' '*betslip*'` → bug resuelto (prediccion_ganador arreglado, API homónimos con Playwright).
3. ✅ Fix `prediccion_ganador` (Vacío 4): RESUELTO. Campo trampa no existe en codebase. Todos los consumidores usan `ranking_analysis.prediction.favored_player`. Tests en test_rivalry_analyzer.py + test_nodo32.py invocan el módulo y verifican `favored_player` (REGLA-T53 cumplida).
4. ✅ **C61-A forense CERRADO** — Ver C-06 en DECISION-LOG.md. No hay bug: ×2.2 aplica al componente surface_spec (peso 0.15-0.20); efecto sobre confianza final es 5-15%. ×0.92 = GCS boost al oponente, comportamiento esperado. Sin cambios al motor.
5. ✅ **C61-B gobernanza CERRADO** — Opción (a) elegida: activación por prior retrospectivo A60-01 (n=54, 64.8%) + H60-01 prospectiva continuando acumulación. Docstring corregido (commit 87e854a). Ver C-05/C-06 DECISION-LOG.
**Verificación:** baseline 1756 ✅; C61-A/C61-B cerrados con documentación ✅. Fix prediccion_ganador es tarea menor independiente.

### FASE 1 — Infraestructura mínima (prerrequisito: Fase 0)
1. ✅ **Graphify elegido** — decisión: grafo dominante es "visualización + vault Obsidian" (1588 nodes, 2987 edges). `.graphifyignore` configurado (código solo). Commit: `b9553a4`.
2. ✅ **Tamp instalado** — ~/.config/systemd/user/tamp.service activo, puerto 7778, health ✅. ANTHROPIC_BASE_URL configurado.
3. ✅ **Slash-commands** — `/tennis-audit`, `/tennis-session`, `/tennis-brief` creados en `.claude/commands/`. Commit: `b9553a4`.
4. ✅ **Validador local** — `pre_game_validator.py` (cron 0 9-23), detecta kelly_kl=0.0 → BLOCK, n<8 → WARN, phantom identity → WARN.
**Verificación:** Graphify query ✅; 3 comandos ejecutan ✅; validador fixture: BLOCK KELLY_ZERO ✅. Commit: `b9553a4`.

### FASE 2 — Automatización de timing + governor de riesgo (prerrequisito: Fase 1)
1. ✅ **close-snapshot trigger local** — `close_snapshot_trigger.py` escrito, cron */10. Lee shadow book → detecta registros abiertos (sin cierre_kambi Y sin resolucion) → ejecuta shadow_book --close-snapshot. Telegram notifica. NUNCA toca betslip_registrar. Commit: `b9553a4`.
2. 🟠 **Alerta ventana de ejecución n8n** — PENDIENTE (requiere n8n, no cron Python local).
3. 🔴 **C63-B governor único** — PENDIENTE. Debe sumar stakes de TODAS las capas (CORE+Satellite+mega+safe+GCS+Anchor) ≤ session_budget M-26-2. Si excede: recorta por mayor varianza (Anchor 3A+2B primero). Cierra V-26-2a.
**Verificación:** close-snapshot logs muestran captures exitosos (✅ 50 registros del 2026-07-08); governor pendiente (parámetro crítico sin implementar hoy).

### FASE 3 — Hermes (prerrequisito: Fase 1; GATE de datos)
1. 📊 **Medición en curso** — conteo de ambigüedad de nombres (PlayerRegistry slow-path, casos Nguyen-like) en logs. GATE: ≥5 casos/semana → implementar; <5 → DIFERIR. Baseline: semana 2026-07-01/07 = TBD (investigación en curso).
2. 🟠 **Si procede (post-gate):** Hermes local (contexto 32k), schema JSON forzado, confianza 0.85, residual a sesión Claude. Líneas rojas intactas.
3. 🟠 **Playwright MCP fallback** — casos de prueba en desarrollo; no testeado aún contra fallo real de API.
**Verificación:** tabla de medición post-semana; si gate pasa: 20 casos de prueba con ≥95% concordancia manual.

### FASE 4 — Estadística de doctorado (prerrequisito: Fase 0; cada nodo tiene su propio gate de datos)
Orden por disponibilidad de datos:
1. **Nodo-64 SPRT** — solo código, sin gate de datos. Primero. (PENDIENTE: commit tests + hypothesis_tracker integración)
2. **Nodo-67 Drift (CUSUM+PSI)** — `analysis/drift_monitor.py` escrito, modo reporte. (PENDIENTE: commit)
3. **Nodo-66 FLB** — `analysis/flb_curve.py` escrito, modo reporte. (PENDIENTE: commit)
4. **Nodo-69 pattern_audit** — `analysis/pattern_audit.py` escrito, código + fixture. (PENDIENTE: commit)
5. **Nodo-70 CPPI** — `analysis/` (PENDIENTE: implementación ~20 líneas + test T70)
6. **Nodo-68 Conformal** — `analysis/conformal_band.py` escrito, GATE n≥50 settled, modo reporte. (PENDIENTE: commit)
7. **Nodo-65 ρ empírico** — `analysis/rho_empirical.py` escrito, GATE ≥15 sesiones×≥3 picks/tier. (PENDIENTE: commit)
8. **Nodo-71 velocity** — `analysis/velocity_monitor.py` escrito, depende de odds_series.py; solo reporte. (PENDIENTE: commit)
**Verificación por nodo:** su test T6X pasa invocando el módulo + su output aparece en el daily_brief/dashboard + NINGUNA constante de producción cambió (los ocho son instrumentos de medición en esta fase; los cambios de constantes solo ocurren en ventanas de recalibración pre-agendadas con el criterio de §3.2).
**Estado actual (2026-07-08):** 6 de 8 módulos escritos pero sin commitear. T70 (CPPI) requiere waterfall.

### FASE 5 — Vault y memoria (prerrequisito: Fases 1-2 estables)
1. ✅ DECISION-LOG.md con los 5 errores históricos + los casos nuevos (tentación watchlist, cuarentena Van Zyl, Pereyra phantom identity, settle-retry ITF, C61-B gobernanza). Commit: `7da3ab0`.
2. 🟠 claude-memory-compiler no existe como paquete publicado (npm/pip) — marcado como DIFERIDO. Vault (`audit-trail/`) existe como directorio; artículos se escriben manualmente (sin motor nuevo).
3. ✅ Política de precedencia (§1.2 Vacío 3) escrita en CLAUDE.md + chequeo semanal Haiku de contradicciones (`check_contradictions.py`). Cron: `0 9 * * 1`. Commit: `7da3ab0`.
4. ✅ CLAUDE.md adelgazado a constitución + estado + punteros: 259 líneas (< 300). Commit: `7da3ab0`.
**Verificación:** CLAUDE.md < 300 líneas ✅; chequeo de contradicciones: 5 PASS | 5 WARN | 0 CONTRADICCION ✅.

---

# §4.5 — PENDIENTES ACTUALES (2026-07-08)

**Status al cierre de Fase 5:**
- F0 Reconciliación: C61-A/C61-B pendientes (ambas son análisis forense de multiplicador GCS, bloquean decisión de gobernanza)
- F1 Infraestructura: completa (Graphify + Tamp + 3 slash-commands + pre_game_validator local)
- F2 Automatización: PARCIAL — close_snapshot_trigger.py existe (cron */10 ✅), pero n8n-close-snapshot y governor C63-B pendientes
- F3 Hermes: en gate de datos (requiere ≥5 ambigüedades/semana — investigación en curso)
- F4 Estadística: análisis/conformal_band.py, drift_monitor.py, rho_empirical.py, flb_curve.py, pattern_audit.py, velocity_monitor.py escritos pero NO commiteados
- F5 Vault: completa

**Crítico (riesgo real hoy):**
1. **C63-B governor de presupuesto combos** — sin el governor, `session_budget` solo cubre megas; Anchor×3 no tiene límite.
2. **C62-A flag alpha_promoted** — picks promovidos no llegan al shadow book con el marcador; H62-01 no se puede medir.
3. **C63-A Playwright queue** — n<8 no enrola candidatos para enriquecimiento (1 línea).

**En gate de datos (esperar):**
1. **Hermes** — ≥5 ambigüedades/semana en PlayerRegistry
2. **Nodo-65 ρ empírico** — ≥15 sesiones × ≥3 picks/tier settled
3. **Nodo-68 Conformal** — ≥50 picks settled global
4. **Nodo-71 Velocity** — H52-05 concluida

---

# §5 — PROTOCOLO DE AUDITORÍA FABLE (Fase 5 — lo que YO verificaré)

**Clasificación de severidad (pre-definida):**
- **RECHAZO TOTAL de la fase:** saltarse un pre-registro; cambiar una constante de producción fuera de ventana de recalibración; test que no invoca el módulo (PASS/FAIL-permanente); activar un flag gated sin documento de decisión; n8n ejecutando registro de apuestas.
- **DESVIACIÓN MAYOR (corregir antes de cerrar):** baseline no capturado; métrica recalculada fuera de su módulo fuente; gate de datos ignorado ("modo reporte" que en realidad cambia decisiones).
- **DESVIACIÓN ACEPTABLE (documentar):** mejoras de implementación que conservan contrato y tests; constantes provisionales distintas si están etiquetadas y justificadas.

**Comandos de auditoría por fase (evidencia objetiva, no autoreporte):**
```bash
# F0: pytest baseline en el reporte de fase == pytest actual − tests nuevos
python -m pytest tests/ --no-cov -q | tail -1
git log --oneline -5 -- analysis/rivalry_analyzer.py   # C61-A: commit forense existe
grep -n "prior retrospectivo A60-01" analysis/rivalry_analyzer.py   # docstring corregido
# F1: solo UN grafo instalado
pip list | grep -iE "graphify" ; ls ~/.claude/mcp* 2>/dev/null
# F2: cobertura de snapshots del día
python3 -c "import json,glob; [...]"  # % picks con cierre_kambi y started=False
grep -rn "betslip_registrar" ~/n8n-flows/ && echo "VIOLACIÓN" || echo "OK"
# F2 governor: log de suma total de capas vs límite en el trader_plan del día
# F4: reproducción independiente (forense, no confirmación)
#   SPRT: recalculo yo las fronteras A,B con los α,β del JSON y verifico el veredicto de una hipótesis
#   ρ:    corro rho_empirical.py --seed 42 dos veces → mismo IC (determinismo del bootstrap con seed)
#   FLB:  tomo 1 banda, recalculo hit% vs implícita a mano desde el JSONL y comparo con el reporte
#   CUSUM: inyecto fixture con salto conocido y verifico la alarma en el t esperado
```

**Preguntas forenses obligatorias de la sesión de auditoría:**
1. C61-A: ¿dónde se aplica exactamente el multiplicador GCS y por qué producción muestra 1.15/0.92? (respuesta con línea de código, no narrativa)
2. C61-B: ¿existe el documento de decisión de activación? ¿opción a o b?
3. ¿Cuál grafo se eligió y con qué evidencia de la regla de decisión?
4. ¿Algún test nuevo asserta literales en vez de invocar el módulo? (muestreo de 5 tests al azar, leo el código)
5. ¿Alguna constante de producción cambió en algún commit de estas fases? (`git diff` sobre config.py, trader_ev_tenis.py, edge_calculator.py, rivalry_analyzer.py buscando cambios numéricos no documentados)
6. H62-01 y H60-02: ¿están acumulando n en el shadow book? (query real al JSONL)

---

# §6 — PROHIBICIONES GLOBALES

- Ninguna de las 8 conexiones de §3 cambia una decisión de producción en su primera implementación: todas nacen como instrumentos de medición (modo reporte) o flag-OFF.
- GitHub Actions no toca datos de apuestas. Los datos no salen de la máquina local.
- El plan NO desplaza la operación diaria: run_daily + shadow book + settle tienen prioridad absoluta sobre cualquier fase de este SPEC. Si un día hay que elegir, se elige la operación — el n es el activo.
- Hermes jamás en Kelly, apostar, o hipótesis. n8n jamás en registro de apuestas. El dashboard jamás con botones de acción.
- Este documento no se edita tras el inicio de la ejecución: correcciones van en un ADDENDUM fechado, como todo en este proyecto.

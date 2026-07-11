# DECISION-LOG.md — Porqué detrás de los umbrales

> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]
> **Propósito:** Documenta el PORQUÉ de cada decisión de diseño no obvia. Sin este log, las decisiones parecen arbitrarias → se revierten → los bugs vuelven.
> **Formato:** Decisión → Alternativas consideradas → Por qué elegimos esta → Cuándo revisitar

---

## D-01 — λ_ITF = 4.5 (Nodo-21)

**Decisión:** El multiplicador de incertidumbre Kullback-Leibler para ITF es 4.5× (más restrictivo que GS=1.0×).

**Alternativas consideradas:**
- λ_ITF = 2.5× (más permisivo, más stakes)
- λ_ITF = 6.0× (casi prohibitivo)

**Por qué 4.5×:** El bookmaker tiene menos datos históricos en ITF → mayor ventaja informacional para el modelo → PERO también mayor varianza por n pequeño de calibración. 4.5× balancea explorar el alpha ITF sin ruina por sobreconfianza. Validado: sesión 9/10 del 13-jun, ITF hit 82.8%.

**Cuándo revisitar:** cuando n_ITF_shadow ≥ 30 eventos settled en shadow book. Actualmente epoch-2 post-2026-07-01: n=0.

---

## D-02 — James-Stein Shrinkage con factor n/(n+20) (Nodo-21)

**Decisión:** El floor de calibración es `n/(n+20)`. Con n=4 → 16.7% kelly. Con n=33 → 62.1%.

**Alternativas consideradas:**
- Floor fijo de 50% (simple pero ignorante del n real)
- n/(n+50) (más conservador en n bajos)

**Por qué n+20:** El "pseudo-count" de 20 equivale a la certeza de tener 20 datos históricos como prior. Clay GS con n=31 da factor=0.61 — razonable para apostar con convicción media. Si n=4, el factor 0.17 prácticamente zeroa el stake — correcto, es ruido estadístico.

**Cuándo revisitar:** nunca sin evidencia de que el floor actual produce P&L sistemáticamente diferente al Kelly completo.

---

## D-03 — Grace period inactividad = 30 días (Nodo-57)

**Decisión:** `form_decay_factor(days) = 1.0` para days ≤ 30. La inactividad no penaliza forma hasta 30 días.

**Alternativas consideradas:**
- Grace = 14 días (más agresivo, penaliza pausas transición arcilla→hierba)
- Grace = 45 días (demasiado permisivo con inactividades largas)

**Por qué 30 días:** Las pausas de transición ATP (arcilla→hierba: 2-3 semanas; hierba→dura: 3-4 semanas) caen dentro del grace. Penalizar por descanso programado es penar una característica del circuito, no un dato de forma real. Los 4 marcos de expertos (Nodo-57 §1) convergieron en este umbral.

**Cuándo revisitar:** si n≥50 observaciones de jugadores con days_since=25-35 muestran hit rate materialmente diferente al baseline.

---

## D-04 — _MIN_WINS_CHAMPION GS = 7 (Nodo-57)

**Decisión:** Un campeón de Grand Slam debe tener ≥7 victorias en el cuadro principal (sin contar qualifying).

**Alternativas consideradas:**
- 5 victorias (cuartos de final) — demasiado bajo, Safiullin demostró el bug
- 6 victorias (semifinalista) — semifinales ≠ campeón
- 7 victorias (CORRECTO: R1→R7 del cuadro principal)

**Por qué 7:** Wimbledon tiene 128 jugadores en el cuadro principal → 7 rondas para ser campeón. Safiullin tenía 5W (3 qualifying + 2 main draw) en Wimbledon 2026 cuando el sistema le asignó bonus×1.6 erróneamente. El gate `wins>=4` original era tier-agnóstico. Fix: `_MIN_WINS_CHAMPION['grand_slam']=7`.

**Cuándo revisitar:** si ATP cambia el formato de GS (improbable). No revisitar por temporada.

---

## D-05 — Density confidence [0.3, 1.0] para common_opponents (Nodo-21)

**Decisión:** La densidad local modula el peso de rivales comunes entre 30% y 100% de su peso base.

**Por qué 30% de floor:** Si n_common=0, los rivales comunes no aportan señal — pero el modelo no puede ignorarlos completamente porque el grafo transitivo puede tener caminos indirectos. El 30% mantiene la señal viva pero reducida.

**Cuándo revisitar:** cuando el PageRank Erdős (Nodo-20) tenga suficientes datos para calibrar la conversión de centralidad a probabilidad.

---

## D-06 — Shadow book como única fuente de verdad CLV (Nodo-52)

**Decisión:** El shadow book (`reports/shadow_book/sb_YYYY-MM-DD.jsonl`) es append-only e inmutable en predicciones.

**Por qué append-only:** La manipulación retroactiva de registros invalida el backtest estadístico. La hipótesis H52-01→H52-08 requieren que los picks estén pre-registrados antes de conocer el resultado. Sin append-only, el humano puede (inconscientemente) "limpiar" picks malos.

**Cuándo revisitar:** nunca. Es una regla estructural de integridad estadística, no un parámetro.

---

## D-07 — Routing M0/M1/M2 de Nodo-59 (Nodo-59)

**Decisión:** El Dream automático (detección de secuencias repetitivas) es M2 — solo post-primer ciclo de 30 días del shadow book.

**Por qué M2:** El mayor riesgo del motor agéntico es sí mismo — un meta-proyecto seductor que compite con la disciplina diaria. El shadow book necesita 30 días para tener n≥30 hipótesis con valor estadístico. Construir M2 ahora es exactamente el patrón "nuevo framework > trabajo aburrido" que el proyecto documenta.

**Cuándo revisitar:** cuando shadow book epoch-2 tenga n≥30 settled (estimado: ~2026-08-01).

---

---

## ERRORES HISTÓRICOS — Los 5 de CLAUDE.md §2

### E-01 — Sin SPEC
**Error:** 27 archivos sin especificación → duplicados, pipeline roto, accuracy 47.37%.
**Causa raíz:** Vibe coding. Cada sesión reimplementaba sin consultar historia.
**Fix:** SDD obligatorio. Ver `PRE_IMPLEMENTATION_CHECKLIST.md`.
**Lección:** El costo de no documentar supera el de documentar.

### E-02 — HTML garbage en tipo_cancha
**Error:** `surface_specialization=0%` durante meses. Scraper extraía HTML en lugar de texto.
**Fix:** `scraping/data_parser.py` corregido 2026-05-28.
**Lección:** Un campo que siempre devuelve 0 silenciosamente es peor que un error visible.

### E-03 — Sin edge calculator
**Error:** Apuestas sin ventaja cuantificada. Accuracy alta ≠ P&L positivo.
**Fix:** `edge_calculator.py` Kelly-KL 5 capas. Edge mínimo: P_modelo - P_implícita > 5%.
**Lección:** P(modelo) > P(implícita) + 5% es el único criterio válido para apostar.

### E-04 — Labels corruptas ML
**Error:** accuracy real 47.37% pese a 95.4% CV (overfit). Labels cruzadas en generar_dataset_plus.py.
**Fix:** Nodo-41 — filtro rivalry_version + trazabilidad jugador1/jugador2/_trace_fecha. 2,573 registros limpios.
**Lección:** Un dataset sin trazabilidad es un dataset contaminado.

### E-05 — Kelly naive sin portfolio
**Error:** KGR=-0.51 (ruina silenciosa). N picks a Kelly completo = ruina garantizada.
**Fix:** Portfolio Kelly multi-activo + VaR/CVaR + Cobertura por Exclusión. Ver `trader_ev_tenis.py` v2.0.
**Lección:** REGLA-HF-5: si KGR < 0 en output → NO DESPLEGAR sin importar la confianza individual.

---

## CASOS NUEVOS (post FABLE_02)

### C-01 — Tentación watchlist (2026-07-01)
**Incidente:** Picks WATCHLIST entraron en combos directamente sin revisión humana.
**Por qué es tentador:** La cuota alta del watchlist es atractiva; el combo builder los incluye si --watchlist activo.
**Regla:** Picks WATCHLIST requieren revisión explícita en PASO 3.5 (generar_tabla_favoritos2.py) antes de apostar.

### C-02 — Cuarentena Van Zyl / edges fantasma (2026-07-01)
**Incidente:** Arce/Vlajic/Guajardo/Cooper en combos con historial vacío → edges fantasma → pérdidas reales.
**Por qué pasó:** Pipeline no daba error — generaba confianza alta sin datos reales.
**Fix:** F2 DataContract (Nodo-51) — status=NO_DATA por construcción. `core/data_contract.py`.

### C-03 — Phantom Identity homónimos (2026-07-06)
**Incidente:** Facundo Pereyra debutante recibió 105 partidos del veterano homónimo vía API → 64.4% confianza falsa → 5 combos inválidos → pérdida real confirmada.
**Causa raíz:** API busca por nombre string; Playwright navega por entity ID de FlashScore.
**Fix:** Playwright migrado a PRIMARIO (PASO 1 y 2). Nodo-72 Phantom Identity Guard.
**Señal de alerta:** ranking=None + n_history>20 + fecha_más_antigua>365d = PHANTOM_IDENTITY_SIGNAL.

### C-04 — Settle-retry ITF (2026-07-06)
**Incidente:** Partidos ITF/Challenger no aparecían en FlashScore a tiempo → settle silenciosamente fallaba → picks permanecían open.
**Fix:** `close_snapshot_trigger.py` cron 10 min. Slash-command `/settle-retry` reintenta hasta 48h.

### C-05 — C61-B Gobernanza GCS (2026-07-06)
**Incidente:** Docstring decía "validado por H60-01" pero H60-01 es prospectiva (n<30). La activación real fue por A60-01 retrospectivo (n=54, 64.8%).
**Por qué importa:** Mezclar evidencia prospectiva con retrospectiva contamina el trail de decisión.
**Fix:** Docstring corregido. PROHIBICIÓN permanente: no citar "validado por H60-01" para GCS.

### C-06 — C61-A Forense multiplicador GCS (2026-07-08)
**Incidente:** Producción mostraba efectos ×1.15, ×1.13, ×1.03, ×0.92 cuando las constantes especifican ×2.2/×1.8/×1.5.
**Causa raíz:** NO hay bug de cálculo. Arquitectura correcta pero descripción incompleta en la spec:
  - `_gcs_mult` (×2.2/×1.8/×1.5) se aplica a `final_score` dentro de `analyze_surface_specialization()` — este `final_score` ES `surface_specialization.score`, un sub-componente.
  - `surface_specialization` tiene peso **0.15–0.20** en `generate_advanced_prediction()` (suma ponderada con ELO, H2H, form_recent, common_opponents, etc.).
  - Efecto real sobre confianza final ≈ weight × (mult − 1) × normalized_component / total_weighted ≈ 5–15%.
  - El ×0.92 es comportamiento ESPERADO: GCS boost al oponente → su surface_spec sube → confianza del pick evaluado baja.
**Lección:** "Multiplicador al final_score" en la spec significaba multiplicador al score del COMPONENTE surface_spec, no a la confianza global. Diferencia de semántica, no de cálculo.
**Cuándo revisitar:** Si se quiere que GCS afecte directamente la confianza global → requiere un Nodo nuevo con ponderación explícita post-suma. Hoy: sin cambios al motor.

---

### C-07 — Auditoría scraping/ Strangler Fig: huérfanos falsos + zombie cleanup (2026-07-09)

**Hallazgo (auditoría Nodo-75):** `scraping/browser_manager.py` y `scraping/data_parser.py`
aparecieron como huérfanos en el índice Nodo-75. Investigación completa reveló que NO son
huérfanos — son dependencias activas de `scraping/h2h_extractor.py` (H2HExtractor), que ES
el modo Playwright primario de `extraer_historh2h.py` hoy (línea 321: `extractor = H2HExtractor()`).
La migración T07-0D (Nodo-07 Fase 2) se completó en 2026-05-30 y nunca fue revertida.
d9dc90a (Nodo-61) solo tocó 7 líneas en extraer_historh2h.py, sin cambio de arquitectura.

**Causa raíz del falso huérfano:** Nodo-07 referenciaba `BrowserManager` y `DataParser`
por nombre de clase, no por nombre de archivo (`browser_manager.py`, `data_parser.py`).
El regex del parser Nodo-75 busca patrón `*.py` y no los capturaba.

**Fix aplicado:** Adendo 2026-07-09 en Nodo-07 — tabla de archivos explícita con nombres
`*.py`. No hubo cambio de decisión arquitectónica; solo corrección de omisión de trazabilidad.

**Identidad de jugador:** SequentialH2HExtractor (eliminado en bac389d) usaba
`RankingManager.normalize_name()` — string-matching. Sin entity IDs FlashScore.
La solución real al problema de homónimos llegó con Nodo-72 (julio 2026). H2HExtractor
tampoco implementa resolución por entity ID — el Phantom Identity Guard opera en capas
superiores (rivalry_analyzer.py + pre_game_validator.py).

**Zombie Chrome cleanup — gap real cerrado:** `BrowserManager._kill_zombie_chrome_processes()`
ya estaba portado a `extraer_historh2h.py` (psutil, líneas 39–53). Sin embargo,
`extraer_URL_partidos_version2.py` (PASO 1 — corre Playwright desatendido en n8n/cron)
no lo tenía. Sin evidencia histórica de impacto, pero gap real en sistema desatendido.
**Fix aplicado 2026-07-09:** `_kill_zombie_chrome_processes()` portado a PASO 1
(~15 líneas, sin cambio de arquitectura, referencia C-07).

---

## POLÍTICA DE PRECEDENCIA (§1.2 Vacío 3, FABLE_02)

1. Los nodos y ADRs son **historia inmutable** — nunca se editan. Se añade entrada nueva o se marca `SUPERSEDED por [[Nodo-XX]]`.
2. **CLAUDE.md es una VISTA derivada** — si contradice al nodo más reciente, CLAUDE.md está desactualizado por definición.
3. Un **chequeo semanal de Haiku** compara bugs/estados de CLAUDE.md contra headers de los últimos 10 nodos y reporta contradicciones. La fuente de verdad es siempre el registro inmutable más reciente.

---

## D-08 — Gap WO/retiro entre PASO 1 y settle (2026-07-11)

**Decisión:** No conectar la detección de walkover/retiro de PASO 1 (extraer_URL_partidos_version2.py, extraer_cuotas_partidos.py) con el mecanismo de settle() del shadow book. Gap documentado, no corregido.

**Contexto:** resultados_finales.py usa ninja `dc_1_` endpoint: devuelve FT si DJ='H'/'A' (incluye WO con ganador declarado), o 404→ERROR si el partido fue cancelado antes de empezar. ERROR queda fuera de `detailed_results` → settle() no lo ve. En sesión 2026-07-11 se encontraron 17 picks sin settle explicados por esta vía (ninja 404 = partido cancelado/abandonado antes del inicio).

**Alternativas consideradas:**
- Añadir rama `status='WO'/'CANCELLED'` en `obtener_resultado_api()` que detecte 404 y lo cierre como No_Contest en shadow book
- Propagarla señal del scraper PASO 1 hacia resultados_finales.py

**Por qué no ahora:** Solo 17/208 picks afectados (5.9%) en la ronda de análisis — tasa baja. Todos con `apostar=False` o `stake=0` (sin P&L real expuesto en esta instancia). Costo de implementación no justificado con n actual.

**Cuándo revisitar:** si en 4 semanas de operación la tasa de picks-sin-settle con 404 supera el 10% del total settled, o si aparece un pick con `apostar=True` y stake>0 con 404.

---

## Plantilla para nuevas decisiones

```
## D-XX — [Nombre de la decisión] (Nodo-YY)

**Decisión:** [qué se decidió]

**Alternativas consideradas:**
- [alt A] — [por qué no]
- [alt B] — [por qué no]

**Por qué esta:** [razonamiento concreto, con datos si existen]

**Cuándo revisitar:** [condición específica y medible, no "cuando sea necesario"]
```

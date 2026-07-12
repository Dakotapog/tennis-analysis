# AUDITORÍA FABLE 5 — Proyecto Completo — 2026-07-11

> **Auditor:** Claude Fable 5 (sesión PowerShell, análisis estático solo-lectura)
> **Alcance:** CLAUDE.md, 87 archivos en `.spec/01_Nodos/`, DECISION-LOG (D-01→D-09, E-01→E-05, C-01→C-07),
> `preregistered_hypotheses.json` (15 hipótesis), código núcleo (edge_calculator, trader_ev_tenis,
> shadow_book, combo_confianza_builder, combo_governor, combo_registry, pre_game_validator,
> betslip_registrar, rivalry_analyzer), shadow book completo (10 días, ~177 settled),
> `calibracion_edge.json`, `apuestas_*.json` reales, graphify-out, nodos_index.json.
> **Método:** lectura exhaustiva + verificación cruzada código↔datos. Cada hallazgo cita archivo:línea
> y, donde fue posible, confirmación empírica en los datos en disco. CERO cambios de código aplicados.
> **Exclusiones (ya cerrados esta semana, no repetidos):** governor read-only (Nodo-74), Kambi matching
> (Nodo-80/82), settlement name normalize (Nodo-81), cron numpy, systemd units, gap WO/retiro (D-08),
> Bergeron trader_deploy={} (D-09).

---

## SECCIÓN 1 — Bugs y errores reales no documentados

Ordenados por riesgo de pérdida de dinero real o corrupción silenciosa de datos.

### 1.1 [CRÍTICO — corrupción silenciosa] El loop de calibración con dinero real deposita en buckets huérfanos `?`/`?_?` que ningún lector consulta

**Evidencia:**
- Los picks reales del betslip (bookmarklet Kambi) llegan sin contexto del modelo:
  `reports/apuestas_20260710_021635.json` → cada pick tiene `superficie: "?"`, `tier: "?"`,
  `match_id: ""`, `partido: ""`, `p_modelo: 0.5`, `edge: "0%"`, `kelly_kl: 0.0`, `stake: 0`.
- `betslip_registrar.py:459-487` cierra el loop escribiendo la calibración con esos valores tal cual:
  `sup = pick.get("superficie", "unknown")` → clave `f"{sup}_{tier}"` = `"?_?"`.
- `calibracion_edge.json` ya acumula el daño: bucket `"?"` = 23W/73L (**24% hit**) y `"?_?"` con
  era_v2 = 11W/34L (**24.4%**). Total ~141 resultados de dinero real.
- Ningún lector consulta esas claves: `theta_thompson()` (`edge_calculator.py:325-370`) y
  `_load_p_prior()` (`trader_ev_tenis.py:93-140`) solo buscan claves reales (`clay_grand_slam`, etc.).
- La normalización de superficie en `edge_calculator.py:726-727` cubre `'N/A'`, `'Desconocida'`, `None`
  — pero **no** `'?'`.

**Consecuencia doble:** (a) la evidencia más valiosa del sistema — resultados con dinero real — no
calibra ningún prior, jamás; (b) un hit rate real del **24%** está estadísticamente invisible mientras
el shadow book (picks del modelo, flat 1u) reporta 50-64%. La brecha entre "lo que el modelo elige" y
"lo que se apuesta de verdad" no la está midiendo nadie.

**Causa raíz:** el betslip conoce `outcome_id` de Kambi pero no tiene puente al `match_id` de
FlashScore ni al pick del edge_report → no puede enriquecerse. Ver Sección 4 (es la siguiente
instancia del patrón nombre-vs-ID de Nodo-72/80/81/82).

### 1.2 [CRÍTICO — hipótesis muerta al nacer] `update_alpha_flags()` nunca marca ningún registro → H62-01 acumula n=0 para siempre

**Evidencia:**
- `shadow_book.py:633-634` busca el jugador con `snap.get('nombre') or snap.get('jugador') or
  snap.get('player')` — pero el `pick_snapshot` (que es el pick del edge_report, línea 192) usa el
  campo **`favorito_predicho`**. Ninguno de los tres nombres existe.
- El llamador `combo_confianza_builder.py:1752-1757` pasa correctamente los nombres — el fallo está
  solo en el lookup del lado shadow_book.
- **Confirmación empírica:** grep de `"alpha_promoted"` y `"combo_flags"` sobre los 10 archivos
  `sb_*.jsonl` = **0 ocurrencias**. Nunca se ha marcado un solo registro.

**Consecuencia:** la promoción D62-05 (Cat-C2→Cat-C1 por gate alpha) **está activa en producción**
metiendo picks a SATELLITE/MOONSHOT, mientras su hipótesis de control H62-01 — cuya nota dice
explícitamente "Si falla: apagar D62-05" — no puede acumular jamás. Es exactamente el escenario que
el pre-registro quería impedir: mecanismo activo sin monitoreo.

**Bug hermano latente:** `shadow_book.py:578` y `:630` filtran session_meta con
`rec.get('record_type') == 'session_meta'`, pero el campo real es `_type` (línea 214). Hoy es inocuo
por casualidad (los session_meta no tienen pick_snapshot), pero es el mismo patrón de campo fantasma.

### 1.3 [ALTO — dinero] `max(MIN_BET, …)` fuerza apuestas de $1,000 con EV negativo y por encima del budget

**Evidencia:** `trader_ev_tenis.py:490-491`:
```python
rounded_stake = round(capped_stake / MIN_BET) * MIN_BET
stake         = max(MIN_BET, rounded_stake)
```
- Si `_kelly_quarter()` retorna 0 (EV ≤ 0 bajo p_blend, línea 416-424), `raw_stake=0` → el floor
  igual asigna **$1,000**. Toda señal APOSTAR recibe ≥MIN_BET aunque el propio trader calcule EV negativo.
- Si el budget está agotado, `capped_stake = min(raw_stake, budget - gastado)` puede ser ≤0 → el
  floor apuesta $1,000 **por encima del budget**.
- Mismo patrón en `_build_combos` (línea 569) y `_build_cobertura` (líneas 767-768).

**Mitigación parcial existente:** el ajuste VaR (línea 1131+) puede aplanar después — pero solo corre
si `var_excedido`, y el floor se aplica de nuevo en el redondeo.

### 1.4 [ALTO — dinero] Con `n_h2h=0`, `p_blend` = accuracy histórica del tier → EV de combos ficticio

**Evidencia:** `trader_ev_tenis.py:431-438` + `:1064-1065`:
`_p_blend(p_modelo, n_h2h=0, p_prior)` = `p_prior` exactamente — ignora `p_modelo` por completo.
- Cuando `_P_PRIOR` era 0.52 (neutral), esto era conservador. Tras B-02/T13-06, `p_prior` es la
  accuracy Thompson del tier+superficie — p.ej. **0.758** para clay GS. Nadie revisó la interacción.
- Resultado: un pick clay GS con n_h2h=0 y cuota 2.50 recibe `p_blend=0.758` → EV = +89% → los
  combos de cobertura **se ordenan y financian por ese EV** (`_build_cobertura` ordena por `ev_combo`,
  línea 709). n_h2h=0 es el caso más común en ITF/Challenger.
- Nota agravante: `p_historica` mide "accuracy media del modelo" (dominada por favoritos), no
  "probabilidad de que ESTE pick gane". Usarla como probabilidad del pick sobreestima sistemáticamente
  los underdogs.

**Observación arquitectónica mayor (ver Sección 2):** el `kelly_kl` de 5 capas del edge_calculator
**no se usa para sizing** — el trader recalcula `_kelly_quarter(p_blend, cuota)` (línea 487). Toda la
sofisticación KL/φ/ψ/ccf influye solo en la decisión binaria `apostar`, no en el tamaño del stake.

### 1.5 [MEDIO-ALTO] El gate GCS revive picks bloqueados por los gates de seguridad

**Evidencia:** orden de ejecución en `calcular_edge_completo()`:
1. Bloqueos soft que ponen `apostar=False` **sin** status: `n_axes<2` (`edge_calculator.py:939-941`),
   `HOT_sin_BBI` (`:950-952`), coin-flip T33-01 (`:960-965`).
2. Después, el gate GCS (`:996-999`) hace `resultado['apostar'] = True` **incondicionalmente** si
   `gcs_active + edge≥0.15 + kelly_ef>0.02 + p_blend≥0.45` — sin verificar `motivo_reclasificacion`
   ni si un gate de seguridad ya bloqueó el pick.

Un pick de hierba con campeón reciente bloqueado por T33-01 (n_h2h=0, p_modelo<0.55) puede
reactivarse con `motivo_reclasificacion` contradictorio en el snapshot. Los NO_DATA (phantom,
historial vacío) sí quedan protegidos porque el pool filtra por `status` (`:1101`) — pero los tres
bloqueos soft no. La intención documentada de H60-01 era relajar el **umbral de edge** (50%→15%),
no puentear los guards anti-coin-flip.

### 1.6 [MEDIO] `pre_game_validator` solo valida el pool `apostar`, con campos que no existen

**Evidencia:** `pre_game_validator.py:107-120` — el edge_report real no tiene clave `picks`/`apuestas`/
`edge_picks`, así que cae al fallback "primera lista del root" = **solo `apostar`**. Pero el pool del
trader incluye `watchlist` y `sin_edge` **por defecto** (`trader_ev_tenis.py:929-935` — 
`BooleanOptionalAction default=True` — y `:1023-1031`). Los picks que más necesitan validación
(watchlist con edge bajo, sin_edge con edge negativo) nunca pasan por el validador.
Además `:42` (`pick.get("jugador")`) y `:57` (`pick.get("ranking")`) no existen en el pick real
(`favorito_predicho`, `ranking_favorito`) → nombre siempre `?` y el check phantom es inoperante
(mitigado porque Nodo-72 bloquea phantom en origen).

### 1.7 [MENORES pero reales]
- **Picks con edge negativo entran al pool de dinero por defecto:** `--all-picks` default=True mete
  `sin_edge` a la cobertura (`trader_ev_tenis.py:1028-1031`), donde el bug 1.4 les fabrica EV positivo.
- **CPPI solo se aplica a individuales:** el ajuste VaR multiplica `fv × cppi_f` en individuales
  (`:1134-1135`) pero solo `fv` en combos de cobertura (`:1174-1176`) — el piso de supervivencia
  Nodo-70 no protege la capa que más capital consume.
- **`p.get('tier', 'atp500')`** (`:1010-1012`): un pick sin campo tier entra silenciosamente solo a
  ejecuciones atp500.
- **`nodos_index.json` desactualizado:** no contiene Nodo-83/84/85 (generado 2026-07-10 01:50,
  anterior a esos commits). `scripts/rebuild_nodos_index.py` pendiente.
- **CLAUDE.md §5 desactualizado:** dice "global: n=706, wins=467" — el archivo real tiene
  wins=2307/losses=1452 (n=3759). Vista derivada vencida (la política §10 lo contempla, pero el delta
  es 5×, no cosmético).

---

## SECCIÓN 2 — ¿Están conectadas las señales correctas?

### Mapa de estado señal → decisión

| Señal | Se calcula | Influye en decisión | Acumula evidencia | Veredicto |
|---|---|---|---|---|
| Kelly-KL 5 capas (λ,φ,ψ,ccf) | ✅ | Solo en `apostar` binario | ✅ | **El sizing la ignora** (§1.4) |
| T33-01 / N28F2 / HOT_sin_BBI | ✅ | ✅ | ✅ | OK, pero GCS los puentea (§1.5) |
| GCS (H60-01 graduada) | ✅ | ✅ gate activo | ✅ | OK — única graduación limpia del sistema |
| tier_mismatch (D65-03) | ✅ `edge_calculator.py:862-892` | ❌ (correcto, gateado H77-01) | ✅ en pick_snapshot | OK |
| ANCHOR/VARIABLE (D65-05/06) | ✅ | ❌ (correcto, gateado H77-02) | ✅ segmento en report | OK |
| **RFI (Nodo-64)** | **❌ no existe código** | ❌ | **❌ solo manual** | D64-01 nunca implementado: `rfi_signal`/`rfi_tier` solo existen en el JSON de hipótesis y el nodo. H76-01 lleva n=1/30 desde 2026-07-09 y seguirá ahí |
| **alpha_promoted (D62-05)** | ✅ | **✅ ACTIVA en producción** | **❌ roto (§1.2)** | **Peor combinación posible: influye sin monitoreo** |
| data_completeness | ✅ | ❌ ("se expone como campo") | ✅ | Observacional permanente sin dueño — nadie lo consume ni tiene hipótesis asociada |
| circuit_asymmetry / circuit_warning | ✅ | ❌ informativo | ✅ | Ídem — sin hipótesis pre-registrada que lo gradúe |
| H52-05 (STEAM/DRIFT) | ✅ | ❌ | ⚠️ depende de `cierre_kambi` | Cobertura de cierre: solo 5 de 10 días tienen algún `cierre_kambi` (0 en 07-03/04/05/07/08). Con n8n "ACTIVO", la mitad de los días no captura cierre → H52-05 y el CLV avanzan a media velocidad |
| MPQ / golden_zone | ✅ | ✅ (betplay mega) | ✅ | OK |
| games_signal (Nodo-40) | ✅ | ✅ (betslip cierre) | ✅ | OK |

### Redundancias y contradicciones

1. **El evento "campeón reciente" está contado 4 veces sin dueño común:** TORNEO_COMPLETO_BONUS
   (multiplicador interno de quality_score, todas las superficies), GCS (gate graduado, solo hierba),
   tier_mismatch (H77-01), y el patrón semilla de H77-03. Nodo-65 §2 ya demostró que el BONUS interno
   es el anti-patrón Obradovic — duplica información que el precio ya refleja y **rebaja el peso de
   `form` de 0.30→0.23** justo cuando más se necesita — pero el bonus sigue operando en todas las
   superficies mientras solo la variante hierba (GCS) pasó por graduación. La señal subyacente es UNA
   ("título reciente en tier X hace D días") y debería calcularse una vez, con consumidores gateados
   por separado.
2. **Doble corrección bayesiana apilada con semánticas distintas:** `calibration_confidence` (B-10,
   shrinkage sobre kelly en edge_calculator:461-471) y `p_blend` (prior k=3 en trader:431-438)
   corrigen la misma incertidumbre de calibración dos veces, en capas distintas, con parámetros
   distintos (κ=20 vs k=3) — y luego el trader descarta la primera al no usar kelly_kl para sizing.
3. **Observacionales permanentes sin proceso de revisión:** data_completeness y circuit_asymmetry se
   calculan desde hace semanas sin hipótesis H-XX asociada. La constitución exige pre-registro para
   hipótesis, pero no existe el mecanismo inverso: "toda señal serializada debe tener o una hipótesis
   activa o una fecha de eliminación". Son las únicas dos señales verdaderamente huérfanas.

---

## SECCIÓN 3 — ML y dataset: ¿se puede conectar ya?

**Veredicto: la suspensión sigue siendo correcta para el predictor. Pero hay una vía de bajo riesgo
que NO es el predictor — y un prerequisito no negociable.**

1. **Prerequisito:** con el loop de calibración real roto (§1.1) y dos días sin settle (07-05, 07-10
   tienen 0 `resolucion` en el shadow book — y 07-10 es justamente la sesión semilla de
   H77-01/02/03), cualquier entrenamiento heredaría labels incompletas. E-04 (labels corruptas,
   47.37% real vs 95.4% CV) ocurrió exactamente por entrenar sobre datos con trazabilidad rota.
   Arreglar §1.1-1.2 y la disciplina de settle ANTES de tocar ML.
2. **Vía de bajo riesgo (REPORTE_SOLO, sin tocar decisiones):** no un predictor, sino un **modelo de
   calibración** — regresión isotónica o Platt de `p_modelo → P(win)` sobre los ~177 settled del
   shadow book + los 3,759 resultados de calibracion_edge (epoch-separados). Es 1 feature, sin riesgo
   de overfit comparable, y produce exactamente lo que el sistema ya consume (`p_historica`/priors),
   estratificado por tier. Encaja como instrumento Fase 4 junto a conformal_band.py.
3. **El gate ">78% held-out" es un umbral mal especificado:** accuracy no es la métrica del proyecto
   (la constitución dice P&L, no accuracy) y 78% en held-out temporal es casi inalcanzable en tenis
   (el mercado implícito ronda 70-75% en favoritos). Si algún día se reactiva el predictor, el gate
   correcto sería "Brier/log-loss del ML < Brier del motor actual en held-out temporal + CLV no
   inferior". Eso requiere nodo nuevo y decisión explícita — no acción ahora.
4. El dataset de 2,573 registros (Nodo-41) es utilizable como base de features, pero es pre-era_v2 en
   su mayoría: cualquier uso debe respetar los epochs de calibración (`_meta.calibration_epochs`).

---

## SECCIÓN 4 — Conexiones ocultas no hechas todavía

### 4.1 EL siguiente patrón estructural: "degradación silenciosa de contexto en los bordes"

El patrón nombre-vs-ID (Nodo-72/80/81/82) tiene un hermano mayor que aparece en **seis** puntos no
conectados entre sí: **cuando un artefacto cruza una frontera del sistema, pierde campos de contexto
(identidad, tier, superficie, tipo) y la capa receptora acepta el valor degradado como categoría
válida en vez de fallar ruidosamente:**

| # | Frontera | Degradación | Consecuencia |
|---|---|---|---|
| 1 | Bookmarklet Kambi → `apuestas_*.json` | `match_id=""`, `superficie="?"`, `tier="?"`, `p_modelo=0.5` | §1.1: calibración real huérfana |
| 2 | `apuestas` → `calibracion_edge.json` | `"?_?"` se vuelve bucket de primera clase | 141 resultados invisibles |
| 3 | combo builder → `update_alpha_flags` | campo `nombre` inexistente → 0 matches silencioso | §1.2: H62-01 muerta |
| 4 | edge_report → `pre_game_validator` | schema no reconocido → valida solo la primera lista | §1.6: watchlist sin validar |
| 5 | edge_report → trader | `p.get('tier', 'atp500')` | pick sin tier cambia de tier silenciosamente |
| 6 | shadow_book interno | `record_type` vs `_type` | latente |

El F2 DataContract (Nodo-51, `core/data_contract.py`) resolvió exactamente esto para UN caso
(historial vacío → NO_DATA por construcción). La conexión no hecha: **extender el DataContract a un
contrato de schema por artefacto** — cada consumidor de JSON valida los campos que necesita y falla
ruidosamente (o marca NO_DATA) en vez de `dict.get()` con default permisivo. Los seis puntos de la
tabla se cierran con un solo mecanismo.

### 4.2 El puente que falta: `outcome_id` ↔ `match_id`

El sistema tiene DOS sistemas de identidad que nunca se presentaron: FlashScore `match_id` (pipeline
del modelo) y Kambi `outcome_id` (pipeline del dinero). `betplay_combo_builder` es el único módulo
que conoce ambos en el mismo instante (matchea picks del edge_report contra outcomes Kambi —
Nodo-80/82). Si en ese momento persiste el mapa `outcome_id → {match_id, superficie, tier, p_modelo,
edge, kelly_kl}` dentro del `betslip_index`, entonces `betslip_registrar` puede enriquecer las
apuestas reales por outcome_id y: (a) la calibración real cae en el bucket correcto, (b) el governor
ve stakes reales, (c) el settle de apuestas deja de depender de match_id vacío. Un cambio pequeño
que cose los dos hemisferios del sistema.

### 4.3 Cuatro implementaciones de name-matching conviviendo

`scraping/kambi_tennis._normalize_name`, `shadow_book` (3 tiers + `_apellido_candidates`),
`combo_registry._names_match` (substring + apellido, la más débil — combo_registry.py:45-60), y
`core/player_registry.normalize_player_name` (la canónica, Nodo-51). Cada fix de matching (Nodo-36,
80, 81) se aplica a UNA copia. La conexión no hecha: converger en player_registry como única puerta.
Mientras existan cuatro, cada bug de nombres se arreglará cuatro veces.

### 4.4 El settle fallback sin restricción de torneo

`shadow_book.settle()` tier-3 (`shadow_book.py:707-718`) matchea el favorito contra **todos** los
resultados del día sin filtrar por torneo — si un jugador (o dos homónimos, o hermanos) aparece dos
veces el mismo día (qualifying + main draw es el caso real), settlea contra el primero que matchee.
Misma familia del patrón, aún abierta tras Nodo-81.

---

## SECCIÓN 5 — Integración de herramientas

**Estado real de las conexiones:**

| Herramienta | Estado | Isla porque… |
|---|---|---|
| n8n close-snapshot (Nodo-73) | 🟠 Parcial | Funciona, pero solo 5/10 días tienen `cierre_kambi`. La cobertura real del Momento 2 es ~46% (82 snapshots / 177 settled) |
| combo_governor (Nodo-74) | 🔴 Isla doble | (1) 0 ejecuciones — gate de 10 sesiones no arranca solo; (2) parsea `combo_plan_*.txt` con regex y de betplay lee `stake` de apuestas_*.json — que siempre es 0 (§1.1) → ciego al dinero real por construcción |
| combo_registry (Nodo-76) | 🔴 Isla | Sin invoke en producción + name matching propio (§4.3) |
| betslip_registrar | 🟠 Conectado a medias | Cierra calibración pero con datos degradados (§1.1); no lee betslip_index para enriquecer |
| dashboard (D58-01) | 🟢 API lista | `shadow_book.report_dict()` existe y es correcta; consumo según Nodo-58 |
| Graphify :7779 | 🟢 código / 🔴 specs | `graph.json` tiene 0 nodos de `.spec/` (verificado por grep) — la pregunta del prompt: NO, nunca se corrió con `.spec/` incluido. El grafo ve el código pero no la historia de decisiones |
| Obsidian / nodos_index | 🟠 | Índice sin Nodo-83/84/85; wikilinks de nodos nuevos no indexados |
| Tamp :7778 | 🟢 | Dependencia dura documentada, sin hallazgos nuevos |

**Conexiones propuestas, ordenadas por beneficio/esfuerzo:**

1. **Puente betslip_index → apuestas reales (§4.2).** Esfuerzo: bajo (persistir un dict en un JSON que
   ya existe + lookup por outcome_id en betslip_registrar). Beneficio: arregla §1.1, alimenta al
   governor con stakes reales, habilita settle por match_id de apuestas. **La conexión de mayor
   impacto de todo el sistema.**
2. **Fix `update_alpha_flags` (1 línea: usar `favorito_predicho`).** Esfuerzo: trivial. Beneficio:
   H62-01 empieza a existir; D62-05 deja de operar sin vigilancia.
3. **Normalizar `'?'` → `'unknown'` en edge_calculator:727 y betslip_registrar:459** + decidir qué
   hacer con los 141 resultados ya huérfanos (migrarlos a un bucket `real_money_unknown` explícito,
   no borrarlos — son la única medición de ejecución real).
4. **Governor lee `trader_plan_*.json` + salida estructurada del combo builder** en vez de regex sobre
   .txt. Esfuerzo: medio. Beneficio: el gate de 10 sesiones puede empezar con datos fiables. Añadir
   `combo_governor` al final de `run_daily.py` (aunque siga READ-ONLY) para que las 10 sesiones se
   acumulen solas.
5. **Unificar name matching en `core/player_registry`.** Esfuerzo: medio-alto. Beneficio: cada fix
   futuro se aplica una vez.
6. **`graphify .` con ANTHROPIC_API_KEY incluyendo `.spec/`** + `rebuild_nodos_index.py`. Esfuerzo:
   bajo. Beneficio: el grafo por fin ve los nodos — prerequisito para que auditorías como esta sean
   navegables.

---

## SECCIÓN 6 — Estrategia concreta para ingresos reales YA

**El estado real, sin adornos:** el sistema tiene UN segmento con evidencia sólida (clay Grand Slam
25W/8L histórico + shadow GS ROI +47%), UN segmento prometedor con n pequeño (clay challenger era_v2
11W/3L), UN segmento consistentemente negativo (ITF: clay 39W/49L, **hard 32W/66L = 32.7% era_v2**,
shadow ROI −16.8%), y una medición de ejecución real oculta que dice **24% hit en las apuestas
efectivamente colocadas** (§1.1). La brecha modelo-vs-ejecución es hoy el mayor riesgo financiero —
mayor que cualquier parámetro del modelo. Y el incidente del stake 10× demuestra que el eslabón
humano opera sin control de conciliación.

### Plan por pasos (2-4 semanas)

**Paso 0 — Detener la fuga (hoy, sin código):**
- **ITF con dinero real: STOP total.** hard_itf 32.7% era_v2 no lo salva ni λ=4.5. ITF queda
  shadow-only hasta que algún segmento ITF gradúe con IC Wilson. El shadow book sigue acumulando gratis.
- **Nada de VARIABLE legs** (cuota<1.35, edge<0) como más de 1 pierna por combo hasta H77-02 n=60.
  La sesión 07-10 ya demostró (Nodo-65 §2) que rinden exactamente al azar.

**Paso 1 — Plomería antes que apuestas (1-2 días de trabajo, decidir con el usuario uno por uno):**
1. Puente outcome_id↔match_id (§4.2) — sin esto, seguir apostando es seguir sin medir.
2. Fix update_alpha_flags (§1.2).
3. Normalización `'?'` (§1.1).
4. Quitar el floor `max(MIN_BET, …)` cuando kelly=0 o budget agotado (§1.3) — que stake 0 sea 0.
5. Reordenar el gate GCS antes de los bloqueos de seguridad, o hacer que respete
   `motivo_reclasificacion` (§1.5).

**Paso 2 — Disciplina de medición (regla operativa, sin código):**
- **Ninguna sesión nueva sin settle de la anterior.** 07-05 y 07-10 siguen abiertos; H54-01 tiene
  backfill manual pendiente de autoritativo. `run_daily.py --settle-only` ya existe — usarlo.
- Correr `combo_governor` cada sesión aunque sea READ-ONLY: el gate de 10 sesiones solo avanza si se ejecuta.
- Control anti-10×: al registrar el betslip, comparar stake real vs `trader_plan` y avisar si >2×.
  (Propuesta de nodo, no implementar sin decisión.)

**Paso 3 — Dónde apostar mientras tanto (con los controles ya activos):**
- **Solo picks ANCHOR** (edge>0) en **clay/grass GS y clay Challenger**, cuota 1.50-2.20,
  confianza ≥55%, `n_axes_active ≥ 2`, sin WARN de superficie.
- **Stakes: los del trader, sin multiplicar.** El VaR auto-ajustado y el CPPI asumen el stake
  recomendado; multiplicarlo ×10 convierte un sistema con g>0 en uno con g<0 — la matemática de
  supervivencia entera queda invalidada. Si el stake recomendado se siente pequeño, la respuesta es
  n de sesiones, no tamaño.
- Budget diario de combos: Fase 2-3 del combo builder (4-7% bankroll), no más, hasta que el governor
  tenga sus 10 sesiones.
- GS grass: el shadow dice hit 50% con ROI+47% — el edge viene de las cuotas, no del hit rate.
  Mantener, pero solo ANCHOR.

**Paso 4 — Qué NO tocar (todo correctamente gateado, dejarlo así):**
- GCS_MULT escalados (×2.2/1.8/1.5) — congelados hasta Fase-H/Brier. Correcto.
- H60-02 multiplicadores extendidos, D64-04 (boost RFI), phantom_data penalty (H77-03), lógica
  diferencial ANCHOR/VARIABLE — todos PROHIBIDOS hasta graduación. Correcto.
- ML predictor — suspendido (Sección 3). Correcto.
- Los umbrales de las 15 hipótesis — congelados por la regla anti p-hacking. Correcto.

**Paso 5 — Criterio de escala (4 semanas):**
Cuando (a) el puente real-money esté midiendo, (b) clay_challenger era_v2 llegue a n≥30 con IC Wilson
inferior > breakeven, y (c) el hit real de apuestas colocadas converja con el shadow (hoy 24% vs
50-64%), entonces — y solo entonces — escalar bankroll según la recomendación del propio trader
(+20% tras 5 sesiones validadas con g>0). Si (c) no converge, el problema no es el modelo: es la
selección manual de qué apostar del menú que el modelo produce, y eso se audita antes de escalar.

---

## Anexo — Datos duros usados

- Shadow book: 177 settled / 10 días; `cierre_kambi` en 82 registros (5 días con 0);
  `trader_deploy` en 18; `alpha_promoted` en **0**.
- calibracion_edge.json: global 2307W/1452L; buckets `?` 23W/73L, `?_?` era_v2 11W/34L;
  clay_itf era_v2 39W/49L; hard_itf era_v2 32W/66L; clay_challenger era_v2 11W/3L;
  grass_grand_slam era_v2 40W/41L.
- graphify-out/graph.json: 0 referencias a `.spec/` (nunca indexado con specs).
- nodos_index.json: sin Nodo-83/84/85 (stale desde 2026-07-10 01:50).

> **Nodo-meta:** [[Nodo-86-Auditoria-Fable5]] — resumen y punteros a este documento.
> Verificación empírica pendiente (requiere entorno WSL): correr los 1775 tests, reproducir los
> conteos del shadow book con pipeline_tracker, y validar los hallazgos §1.3-1.5 con fixtures.

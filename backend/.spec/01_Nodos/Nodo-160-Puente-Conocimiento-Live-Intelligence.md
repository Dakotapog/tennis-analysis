# Nodo-160 — Puente de Conocimiento: Live Intelligence Unificada (Games + Ganador)

> Estado: **SPEC — pendiente implementación** (research-only, sin código en esta sesión)
> Autor: sesión de razonamiento extendido, 2026-08-02
> Precede a: sesión de implementación futura (Sonnet), con scoping explícito vía AskUserQuestion como en Nodo-159
> Relacionado: [[Nodo-159-Games-Settlement-Serving-Conditioned]] (D159-02/03/05 diferidos aquí), [[Nodo-98-Meta-Senal-Convergencia]], [[Nodo-100B-Triple-Convergencia-Live]], [[Nodo-111-Dual-Book-Live-Intelligence]], [[Nodo-40-Games-Sets-Signal-Layer]], [[Nodo-147]], [[Nodo-150]], [[Nodo-151]], [[Nodo-153]]

---

## 0. Encargo original y cómo se interpretó

El usuario pidió, en una sola instrucción: (a) terminar D159-02 (modelo condicionado a servicio), D159-03 (detector de movimiento sospechoso en cuotas) y D159-05 (stake automático según confianza de señal); (b) encontrar y documentar conexiones ocultas entre sistemas ya construidos ("puente de conocimiento"); (c) extender la confirmación por señales en vivo no solo a mercados de juegos sino también al mercado de ganador (moneyline); todo con la meta declarada de $500.000 COP/día.

Antes de diseñar nada nuevo se hizo el trabajo GRAPHIFY-FIRST/GREP-FIRST obligatorio: graphify sobre los conceptos relevantes + lectura directa de 5 nodos (`Nodo-98`, `Nodo-100B`, `Nodo-111`, `Nodo-40`, más el código de `velocity_monitor.py`, `micro_kelly()`, `_calcular_certeza_condicional()`, `_check_games_convergencia()`, `edge_calculator.py`, `live_edge_monitor.py`, `preregistered_hypotheses.json`). Ese trabajo cambió el diagnóstico de forma importante — ver §1.

**Sobre la meta de $500.000/día:** no se especifica como promesa. Ningún sistema de apuestas puede garantizar un ingreso diario fijo — la varianza de Kelly-KL en muestras pequeñas es alta incluso con edge positivo real. Lo que sí se puede diseñar es un sistema que maximice `g = E[log(1+R)]` (tasa de crecimiento, ya es la métrica NORTE del proyecto en CLAUDE.md §3) y escale bankroll de forma sostenida una vez las hipótesis graduúen. §7 traduce la meta a objetivos verificables (n mínimo, tasa de crecimiento, CLV) en vez de una cifra diaria.

---

## 1. HALLAZGO RAÍZ (esto cambia todo el diseño de abajo)

Se auditó `validation/preregistered_hypotheses.json` — estado real de las hipótesis que dependen de señales EN VIVO:

| Hipótesis | Estado | n_actual / n_stop |
|---|---|---|
| H97-01 (Live Edge Monitor / drift) | ACUMULANDO | **0** / 20 |
| H98-01 (Meta-Señal Convergencia) | ACUMULANDO | **0** / 30 |
| H100-01 (Triple Convergencia / BREAK_CONFIRMADO) | ACUMULANDO | **0** / 20 |
| H111-01 (Dual-Book Steam-Lag) | ACUMULANDO | **0** / 20 |
| H150-01/02/03 (Live Games Risk Intelligence) | ACUMULANDO | **0** / 20, 30, 15 |
| H151-01 (Live Edge Gates) | ACUMULANDO | **0** / 20 |
| H52-05 (STEAM_IN, sustenta velocity_zscore) | **PENDIENTE** (ni siquiera ACUMULANDO) | 0 / 20 |

Contraste: en el mismo archivo, hipótesis pre-partido (H60-01 GCS, H88-01 Rival Value) sí acumularon y graduaron/avanzaron con evidencia real documentada en CLAUDE.md.

**Interpretación:** el código de inteligencia en vivo (Nodo-97, 98, 100B, 111, 150, 151) está genuinamente **IMPLEMENTADO y verificado por tests** — pero casi nada de él está escribiendo observaciones reales al tracker de hipótesis en producción. Se confirmó además que el bloqueador histórico de Nodo-97 (endpoint Kambi LIVE sin verificar vía DevTools) sí quedó resuelto: `scripts/live_edge_monitor.py:508` instancia `KambiLiveClientReal()` con el comentario `# D97-15: usa liveEvents.json (D99-01 resuelto)`. Es decir, el bloqueador técnico original ya no existe, pero el sistema sigue sin acumular evidencia.

**Esto significa que el "puente de conocimiento" que más valor tiene no es una pieza de matemática nueva — es diagnosticar por qué N piezas ya construidas y ya conectadas al endpoint real no están dejando rastro en el tracker de hipótesis.** Añadir D159-02/03/05 y una fusión games↔ganador ENCIMA de una tubería de acumulación que no acumula no produce más dinero: produce más código sin evidencia. Por eso D160-01 (abajo) es el prerequisito de todo lo demás, no una decisión más de la lista.

### 1.1 Candidatas a causa raíz de n_actual=0 (a verificar en la sesión de implementación, no aquí — esto es investigación de código, no filosofía)

No se tiene certeza aún de cuál de estas domina (o si son varias a la vez) — se listan como hipótesis de diagnóstico priorizadas para el Paso 0 de la implementación futura:

1. **El proceso que alimenta esto (`live_edge_monitor.py`, `live_desk.py` en modo `--live`, o el servicio `tennis-live-desk` vía systemd) puede no estar corriendo de forma continua** durante las ventanas de partido reales — o corre pero nadie deja partidos en el rango de eventos que dispara el `sprt_verdict`/log de hipótesis.
2. **El punto de log puede existir en código pero nunca ejecutarse en la práctica** porque el gate previo (ej. `BREAK_CONFIRMADO`, D151 gates, `alta_itf`) casi nunca se cumple con datos reales — plausible dado lo estrictos que son los gates documentados (D151-01/02/03 fueron diseñados JUSTO para tapar falsos positivos, pueden estar sobre-corrigiendo hacia cero señales).
3. **El log SÍ ocurre pero no incrementa `n_actual` en `preregistered_hypotheses.json`** porque `hypothesis_tracker.py` no está conectado al punto de escritura real (los picks caen en `shadow_book` pero nadie llama `sprt_verdict`/`llr_update` sobre ellos con el `hypothesis_id` correcto).
4. **El servicio systemd `tennis-live-desk` puede no estar activo/reiniciado** en la ventana horaria en que hay partidos ITF/Challenger en vivo (el proyecto opera desde Colombia, y muchos ITF corren en husos horarios donde nadie está monitoreando ni el servicio se re-arranca tras un crash).

D160-01 abajo especifica cómo auditar esto sin asumir la respuesta.

---

## 2. D160-01 — Auditoría y reparación de la tubería de acumulación de hipótesis EN VIVO (PREREQUISITO — bloquea todo lo demás)

**Objetivo:** antes de añadir una sola señal nueva, confirmar que las que YA EXISTEN dejan rastro verificable en `n_actual`.

**Alcance de implementación (para la sesión futura):**
1. Verificar `systemctl --user status tennis-live-desk` (o el nombre real del servicio, confirmar en `run_daily.py`/docs) — ¿está activo AHORA? ¿tiene reinicios recientes en los logs? ¿corre 24/7 o solo cuando alguien lo lanza manualmente?
2. Grep de cada punto de log de hipótesis (`log_live_pick`, `sprt_verdict`, `llr_update`, cualquier `hypothesis_id="H97-01"` etc. literal en el código) y confirmar que el CALL SITE realmente se alcanza — instrumentar con un log temporal de "gate evaluado, resultado=X" en cada gate crítico (D151-01/02/03, BREAK_CONFIRMADO, D150-06) durante 1-2 días de partidos reales, y contar cuántas veces cada gate pasa vs bloquea.
3. Si el hallazgo es "el proceso no corre continuamente" → arreglar el deployment (cron/systemd), no el código de señales.
4. Si el hallazgo es "los gates son demasiado estrictos" → NO relajarlos a ciegas (violaría la razón de ser de Nodo-151) — en cambio, medir la tasa de paso real y decidir con el usuario si el umbral (ej. drift≥15%, edge_live<5%) necesita recalibración basada en distribución real observada, no solo en la intuición original.
5. Producir un reporte simple: por cada hipótesis EN ACUMULANDO con n_actual=0, ¿cuántos partidos con el evento relevante (ITF/Challenger en vivo, drift de cuotas) hubo en los últimos 7 días, y en cuántos se disparó (o no) el gate?

**Sin este diagnóstico, D160-02 a D160-06 no tienen forma de validarse — seguirían el mismo patrón de "IMPLEMENTADO, 0 evidencia" que ya tienen 7 hipótesis.**

---

## 3. D160-02 — Modelo condicionado a servicio (completa D159-02)

### 3.1 Qué existe hoy (verificado, `live_desk.py:3175`)

`_calcular_certeza_condicional()` ya es más sofisticado de lo que el Nodo-159 original asumía: no es solo un Gaussiano estático — ya tiene:
- Ajuste por `games_set1` (D150-04 tiebreak invalida DOMINANTE; D152-02 ajusta µ por pace del set1 corto).
- Cálculo de `max_remaining` sensible a si el partido ya está decidido, en tercer set, o entre sets.
- Umbral de certeza matemática (determinística, sin distribución) separado del `p_condicional` (probabilístico, Gaussiano por zona).

Lo que **NO tiene** es lo que el usuario pidió explícitamente: condicionar en **quién está sirviendo ahora mismo**. El campo `serving` (D153-02, `"home"`/`"away"` desde `homeServe`) y `break_situation` (D153-04) ya se calculan en `_parse_kambi_livedata_sets()` y viajan en `score_data` — pero `_calcular_certeza_condicional()` nunca los recibe como parámetro. Es un caso de dato ya extraído y no consumido, el mismo patrón que causó el hallazgo del §1.

### 3.2 Por qué "quién sirve" mueve la aguja de verdad

En tenis, el servidor gana su juego con probabilidad muy superior a 50% (rango típico ATP/WTA 55-70% según superficie/nivel; en ITF hay más varianza pero el efecto persiste). Esto significa que la distribución de "cuántos juegos faltan para llegar a la línea" NO es estacionaria dentro de un set — depende de en qué posición del ciclo saque-quiebre está el partido. Un modelo que solo mira `games_played` acumulado (el actual) pierde la información de si los próximos 2-3 juegos son "favorables a hold" (avanza rápido hacia el total) o "en riesgo de break" (ritmo más lento/incierto).

### 3.3 Diseño propuesto — Monte Carlo ligero, no un modelo nuevo de cero

No se propone un modelo bayesiano de juego-por-juego con miles de parámetros (fuera de alcance y de tiempo de cómputo en un ciclo de 15s). Se propone:

1. **Estimar `p_hold_estimado` por jugador** — reutilizar lo que YA EXISTE: `analysis/rivalry_analyzer.py` y el pipeline de superficie/ranking ya calculan fuerza relativa de saque implícita (o puede derivarse de ranking + superficie + `surface_specialization` sin nueva extracción de datos). Si no existe un campo directo de "service hold %", usar un proxy conservador: `p_hold = 0.62 + ajuste_ranking + ajuste_superficie` (clay reduce, hierba aumenta — coherente con GCS §5 del CLAUDE.md), acotado [0.50, 0.85].
2. **Simular N=2000 trayectorias Monte Carlo** desde el estado actual (`current_games`, `serving`, `sets_home/away`) hasta fin de partido, usando `p_hold_estimado` de cada jugador alternando el saque, muestreando el resultado de cada juego como Bernoulli(p_hold del que sirve ese juego) y del tie-break como Bernoulli(0.5) ajustado levemente por fuerza relativa.
3. Output: `p_condicional_mc` = fracción de simulaciones donde el total de juegos supera/no supera la línea, más un intervalo de confianza simple (percentil 10/90 del total simulado).
4. **Función pura, testeable (REGLA-T53):** `simular_total_juegos_condicionado(games_played, serving, sets_home, sets_away, p_hold_home, p_hold_away, linea, direccion, n_sims=2000, seed=None) -> dict` — sin I/O, `seed` para tests determinísticos.
5. **No reemplaza el Gaussiano — lo complementa.** El Gaussiano actual (`_ZONA_PARAMS`) sigue como fallback rápido (usado en el ciclo de 15s para todo lo demás); el Monte Carlo se dispara SOLO quen la señal ya pasó los gates D151 y está a punto de dispararse un combo real (no en cada refresh — el costo de 2000 sims × decenas de partidos en paralelo cada 15s es innecesario). Esto es coherente con cómo D159-04 (`validate_fillability`) ya se implementó como check final pre-disparo, no como filtro continuo.

### 3.4 Riesgo declarado

`p_hold_estimado` es un proxy, no una medición directa de saque en vivo (el pipeline no trae % de primeros/segundos saques ganados en tiempo real desde Kambi). Si en la sesión de implementación se descubre que existe un campo de estadísticas de saque en vivo en el endpoint `livedata.json` (no confirmado en esta investigación — falta grep de la respuesta completa del endpoint), debe preferirse sobre el proxy de ranking/superficie. **Acción para la sesión de implementación:** inspeccionar un payload real de `livedata.json` en un partido en vivo y confirmar si trae `statistics.serve` o similar antes de comprometerse al proxy.

---

## 4. D160-03 — Detector de movimiento sospechoso en cuotas (completa D159-03)

### 4.1 Lo que ya existe

`analysis/velocity_monitor.py::velocity_zscore()` — **completo, funciona, con tests implícitos por su claridad matemática** (z-score causal sobre velocidad de cambio de cuota, `z < -2.0` = STEAM). Está marcado `REPORTE_SOLO`, gateado a H52-05 — que según §1 está en estado **PENDIENTE**, ni siquiera acumulando. Este es el ejemplo más claro de pieza construida y nunca conectada.

### 4.2 Wiring necesario (no matemática nueva)

1. **Fuente de la serie de cuotas:** ya existe — `_write_games_odds_history()` (D147-05, games) y `save_odds_history()`/`load_odds_history()` en `scripts/live_edge_monitor.py` (D100, ganador). Ambos ya persisten series temporales de cuota por partido.
2. **Call site games:** dentro de `_check_games_convergencia()`, después de tener `sig["cuota_actual"]` y el histórico de esa señal (ya se escribe a `games_odds_history_*.json`), llamar `velocity_zscore(odds_series, times_minutes)` sobre la serie de esa señal específica. Si `steam=True` (z < -2.0) Y coincide con la dirección de la señal ALTA (ej. cuota UNDER cayendo fuerte = mercado confirma UNDER), añadir un badge/campo `steam_confirmado=True` — análogo al badge MERCADO CONFIRMA de D150-05 pero basado en velocidad, no solo en drift acumulado. **Diferencia importante con D150-05:** drift acumulado (`cuota_drift`) mide cuánto se movió; `velocity_zscore` mide qué tan RÁPIDO se movió relativo a su propia historia — detecta movimiento anómalo aunque el drift total todavía no cruce el 15% de D150-01. Es una señal más temprana, no redundante.
3. **Call site ganador:** dentro de `scripts/live_edge_monitor.py::detect_break_state()` — la serie ya existe (`load_odds_history()`), añadir `velocity_zscore()` como una señal adicional de entrada a `BREAK_POSIBLE` (hoy solo usa `drift≥15%`). Esto podría hacer que `BREAK_POSIBLE` se detecte 1-2 ciclos antes en movimientos muy agresivos, mejorando el timing de X2 steam-lag (Nodo-111).
4. **Gate de graduación:** aunque el wiring se haga, el campo debe seguir marcado `REPORTE_SOLO` en su efecto sobre decisiones de apuesta real hasta que H52-05 acumule n≥20 — el código puede escribirlo y mostrarlo en el dashboard (valor informativo inmediato para el usuario), pero **NO debe entrar como gate de disparo de combos** hasta graduación. Esto es coherente con cómo el proyecto trata cualquier instrumento REPORTE_SOLO (ver Nodo-71 Fase 4 en CLAUDE.md §6).
5. **Requisito para que H52-05 empiece siquiera a acumular:** necesita un punto de log — hoy NO existe ningún `sprt_verdict`/log ligado a H52-05 en el código (coherente con que está PENDIENTE, no ACUMULANDO). D160-03 debe crear ese punto de log la primera vez que se conecta el detector.

---

## 5. D160-04 — Stake automático según confianza de señal (completa D159-05)

### 5.1 Patrón reutilizable ya construido

`rival_value_betslip.py::micro_kelly(edge_rival, cuota_rival, bankroll)` (L169) — Kelly con shrinkage por n_obs (`n_obs/(n_obs+K_prior)`), cap superior fijo (0.5% bankroll para H88-01), redondeo a 500 COP, piso de 2000 COP. Este patrón es exactamente lo que D159-05 necesita generalizar — no hay que inventar la fórmula, hay que parametrizarla.

### 5.2 Diseño — generalización, no una función nueva por hipótesis

Proponer `core/confidence_kelly.py::confidence_scaled_stake()`:

```
confidence_scaled_stake(
    edge: float, cuota: float, bankroll: float,
    n_obs: int, n_stop: int, k_prior: int = 20,
    cap_pct_graduado: float = 0.02, cap_pct_pregrad: float = 0.005,
) -> float
```

- Antes de graduación (`n_obs < n_stop`): usa el patrón `micro_kelly` tal cual (shrinkage agresivo, cap bajo). Esto es literalmente extraer `micro_kelly()` de `rival_value_betslip.py` a un módulo compartido en `core/`, parametrizado por `k_prior`/`cap` en vez de constantes fijas del módulo — CERO cambio de comportamiento para H88-01 (regression-safe, mismo resultado numérico), pero reusable por CUALQUIER hipótesis nueva (H150-*, H151-01, la futura H160-XX de este nodo) sin duplicar la función.
- Después de graduación (`n_obs >= n_stop` Y el hit rate observado supera breakeven con Wilson LB > 0): shrinkage se relaja a `n_obs/(n_obs+k_prior)` con `k_prior` más bajo (ya converge a casi 1 con n grande) y el cap sube a `cap_pct_graduado`. Esto reproduce lo que ya hace GCS (H60-01, graduada, corre con reglas normales del sistema) vs Rival Value (H88-01, pre-graduación, corre en modo shrink).
- **No reemplaza Kelly-KL de tier/superficie del motor principal** (`edge_calculator.py` §3 del CLAUDE.md) — esto es específicamente para las señales NUEVAS/experimentales que hoy usan sizing ad-hoc o fijo (ej. GAMES combos usan `$1k-2k` fijo según la tabla de taxonomía §11 del CLAUDE.md, no escalado por confianza real de la señal).

### 5.3 Aplicación concreta pedida por el usuario

Los combos de GAMES (`_fire_itf_live_games_combo`) hoy usan stake fijo. Con D160-04 + D160-02 (Monte Carlo) disponibles, el stake de un combo GAMES podría escalar con `p_condicional_mc` (más cerca de 1.0 o 0.0 = más confianza = stake más alto dentro del cap), no solo con el hecho binario de haber pasado los gates D151. Esto conecta D160-02 y D160-04 de forma natural — es el primer ejemplo real de "puente" pedido por el usuario, aplicado dentro del propio dominio de GAMES antes de cruzar a GANADOR.

---

## 6. D160-05 — El puente real: GAMES-EN-VIVO ↔ GANADOR-EN-VIVO (petición explícita del usuario, §7 de su mensaje)

### 6.1 Por qué esto no es "simplemente sumar los dos scores"

Nodo-40 (§ ya verificado, D40 completo) demostró empíricamente que el mercado de juegos/sets es **alfa ortogonal al ganador** — un partido puede perderse en el mercado de ganador y ganarse en el de totales, y viceversa. Fusionar ambas señales en un solo score combinado sería estadísticamente incorrecto: se estaría promediando dos variables que el propio proyecto demostró que no son intercambiables. La arquitectura correcta es tratarlas como **evidencia independiente que se REFUERZA cuando coincide, sin penalizar cuando diverge** (exactamente el patrón que Nodo-98 ya usa entre `score_directo` y `score_rival_value`: campos separados, nunca sumados a ciegas, con lógica de dirección explícita en vez de un solo número).

### 6.2 Lo que ya existe en cada lado

- **Lado GANADOR:** `edge_calculator.py` calcula `score_directo`/`score_rival_value` PRE-partido (Nodo-98, confirmado wireado — L892-1497). En vivo, `scripts/live_edge_monitor.py::detect_break_state()` monitorea drift de cuota del mercado GANADOR y dispara `BREAK_CONFIRMADO`.
- **Lado GAMES:** `live_desk.py::_check_games_convergencia()` monitorea drift de cuota del mercado GAMES, con certeza condicional (D147/150/153) y gates D151.
- **Hoy estos dos loops corren en paralelo, sin cruzarse.** `_check_games_convergencia()` no sabe si el `score_directo` pre-partido de ese mismo jugador fue alto; `detect_break_state()` no sabe si el mercado de juegos de ese mismo partido está mostrando certeza matemática de que el favorito está dominando (games muy por debajo de la línea UNDER esperada = está ganando juegos rápido = puede correlacionar con estar ganando el partido).

### 6.3 Diseño del puente — función pura de reconciliación, no un merge de scores

Proponer `core/live_signal_bridge.py::reconciliar_senales_partido(partido_key, games_state, winner_state) -> dict`:

- **Input `games_state`** (de `_check_games_convergencia`): `{direccion, certeza_matematica, p_condicional, zona, break_situation, serving}`.
- **Input `winner_state`** (de `detect_break_state` + `score_directo` pre-partido guardado en el edge_report original de ese partido): `{score_directo, break_state, drift_pct, direccion_favorito}`.
- **Lógica de reconciliación (NO suma, clasificación por casos — mismo patrón que `direccion` en Nodo-98 §D98 FAVORITO/RIVAL/SPLIT):**
  - `CONVERGENCIA_FUERTE`: certeza matemática games UNDER (el favorito está cerrando el partido en menos juegos de los esperados, `break_situation` a favor del favorito) **Y** `score_directo≥3` pre-partido **Y** `winner_state.break_state` en `BREAK_POSIBLE`/`BREAK_CONFIRMADO` en la misma dirección → el sistema tiene 3 fuentes independientes (games matemático + meta-score pre-partido + drift de cuota ganador) apuntando al mismo resultado. Esto habilita, para el pick de GANADOR (no solo el de games), un stake escalado con D160-04 usando la confianza combinada como "n_obs efectivo" más alto — no como edge distinto, solo como reducción del shrinkage.
  - `DIVERGENCIA`: games certeza apunta a que el favorito está dominando PERO winner_state muestra drift de cuota en contra (el mercado de ganador está subiendo la cuota del favorito) → señal de alerta, no de oportunidad — podría indicar lesión/fatiga no capturada por el score de juegos (un jugador puede estar ganando games por inercia de un set ya distante pero estar visiblemente disminuido). **No dispara nada — solo se loggea para H160-01 (ver §7), es información nueva, no una señal accionable todavía.**
  - `NEUTRO`: sin datos suficientes en alguno de los dos lados, o sin coincidencia de dirección clara → sin efecto, comportamiento actual sin cambios.
- **Importante — esto NO abre una apuesta nueva de GANADOR en vivo por sí sola.** El proyecto no tiene hoy un mecanismo de "comprar" el mercado de ganador en vivo (solo pre-partido vía `trader_ev_tenis.py` y en vivo solo indirectamente vía X2 steam-lag de Nodo-111, que apuesta diferencia de precio entre casas, no dirección). Lo que D160-05 habilita es: (a) mostrar la convergencia en el dashboard `live_desk` como contexto para el usuario, (b) usarla como booster de confianza dentro de X2 (Nodo-111) cuando ya hay un `BREAK_CONFIRMADO` con dirección alineada al modelo, exactamente la condición que el propio Nodo-111 ya exige (`dirección=favorito modelo (STRONG o meta_score≥3)`) — D160-05 simplemente le da a esa condición una fuente games-market adicional para confirmarla en vez de depender solo del meta-score pre-partido estático.

### 6.4 Por qué esto responde literalmente a lo pedido ("no solo con los games sino también con los jugadores que el modelo predice el ganador")

El pedido del usuario de "apostar más seguro" cuando las señales en vivo confirman al ganador pre-dicho se traduce, con la arquitectura real del proyecto, en: **usar la convergencia GAMES+WINNER como input de confianza para X2 (steam-lag, Nodo-111) y como booster de `n_obs efectivo` en D160-04** — no en crear un mercado de apuesta de ganador en vivo nuevo desde cero (el proyecto no tiene ese rail hoy, y construirlo es un nodo aparte, mucho más grande, con riesgo de ejecución en vivo que merece su propia discusión de scope con el usuario antes de especificarse).

---

## 7. Hipótesis pre-registradas nuevas (H160-XX) — a añadir a `validation/preregistered_hypotheses.json` en la sesión de implementación

| ID | Hipótesis | n_stop | Umbral de graduación |
|---|---|---|---|
| H160-01 | `CONVERGENCIA_FUERTE` (D160-05) predice hit rate ≥ breakeven en X2 steam-lag | 20 | Wilson LB > breakeven de la cuota promedio observada |
| H160-02 | Monte Carlo condicionado a servicio (D160-02) reduce error de `p_condicional` vs Gaussiano actual, medido en partidos ya settled | 30 | Brier score MC < Brier score Gaussiano, mismo set de partidos |
| H160-03 | Steam detector (D160-03) wireado en games — `steam_confirmado=True` correlaciona con hit real | 20 | comparte gate con H52-05, no la reemplaza |

**No se pre-registran hipótesis con metas de "$/día"** — coherente con REGLA-HF-5/§3 del CLAUDE.md, que mide en KGR/g/CLV, no en pesos por día. El objetivo de $500.000/día del usuario se traduce operativamente a: una vez H160-01/02/03 (y las 7 hipótesis ya ACUMULANDO con n_actual=0) empiecen a graduar con evidencia real, el bankroll actual ($125.000+ según CLAUDE.md §5) se escala vía CPPI (`factor=min(1,max(0,2×cushion))`, ya definido en §3) — el ingreso diario es una CONSECUENCIA del growth rate compuesto, no un input que se pueda fijar de antemano sin romper el marco Kelly-KL que sostiene todo el proyecto.

---

## 8. Orden de implementación recomendado (para la sesión futura, sujeto a scoping con el usuario tipo AskUserQuestion como se hizo en Nodo-159)

1. **D160-01 primero, sin excepción.** Sin esto, cualquier pieza nueva hereda el mismo destino de n_actual=0. Es diagnóstico + posible fix de deployment, no requiere diseño nuevo — es la sesión más barata y la de mayor apalancamiento.
2. **D160-03 (wiring steam, games)** — la matemática ya existe, es la pieza de menor riesgo de implementación (una función pura ya probada, solo falta el call site + el log de H52-05).
3. **D160-04 (generalizar micro_kelly)** — refactor de bajo riesgo (extraer función existente a `core/`, sin cambiar comportamiento default), habilita todo lo demás.
4. **D160-02 (Monte Carlo servicio)** — la pieza más nueva matemáticamente, requiere el hallazgo de si `livedata.json` trae stats de saque real antes de comprometerse al proxy de ranking.
5. **D160-05 (puente games↔ganador)** — depende de que D160-01 confirme que AMBOS loops (`_check_games_convergencia` y `detect_break_state`) están realmente corriendo con datos frescos; construir el puente antes de eso sería puentear dos sistemas, uno de los cuales puede estar dormido.

Cada paso debe seguir el patrón ya validado en Nodo-159 S1: confirmar contra código real antes de escribir, tests REGLA-T53 con funciones puras, verificación de suite completa sin regresiones, y reporte de qué se implementó vs qué queda diferido — no todo tiene que caer en una sola sesión.

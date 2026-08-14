---
estado: activo
---

# Nodo-181 — Ventana de Oportunidad: Detección de Onda P y Alarma Estratificada

**Fecha:** 2026-08-13
**Estado:** activo
**Autor del hallazgo:** auditoría solicitada por el usuario tras reportar que la dashboard
"está llena de números pero no dice cómo ganar dinero hoy".
**Predecesores:** [[Nodo-147]] (certeza condicional D147), [[Nodo-179]] (memoria arquetipos),
[[Nodo-180]] (inversión estructural games + Ghost Fix settlement), [[Nodo-160]] (MC condicional,
steam, confidence_kelly), [[Nodo-71]] (velocity_zscore), [[Nodo-111]] (dual-book / steam lag).

---

## 0. RESUMEN EN UNA FRASE

El sistema detecta la **onda S** (la sacudida que ya ocurrió) y la anuncia como si fuera
una oportunidad. La onda P (el inicio del movimiento, lo único accionable) nunca se ha
detectado ni medido. Este nodo construye primero el **instrumento de medición de la ventana**
y solo después, y gateada por evidencia, la alarma estratificada.

---

## 1. HALLAZGO RAÍZ — MEDIDO, NO SUPUESTO

### 1.1 La medición que nunca se había hecho

`reports/certeza_fired_{fecha}.json` guarda el instante ISO exacto de cada disparo D147-06.
`reports/games_odds_history_{fecha}.json` (D147-05) guarda la serie de cuotas con `ts` en
**hora de reloj real** (`datetime.now().strftime("%H:%M")`, `live_desk.py:4189`) — no relativa.
Las claves de ambos archivos comparten el mismo formato `"{partido}_{DIRECCION}"`.

**Son unibles. Nadie los unió nunca.** No existe ningún script, panel ni test en el repositorio
que cruce el momento del disparo con el movimiento posterior de la cuota.

### 1.2 Resultado de la unión — 2026-08-13, n=6 disparos

| Señal | c_inicio | c_disparo | c_final | min antes | mov. antes | mov. después |
|---|---|---|---|---|---|---|
| Dougaz A. vs Dhamne M. UNDER | 1.79 | 1.45 | 1.45 | 83 | 19.0% | **0.0%** |
| Townsend T. vs Osorio C. OVER | 1.53 | 2.70 | 2.70 | 97 | 76.5% | **0.0%** |
| Norrie C. vs Prizmic D. OVER | 1.86 | 3.25 | 3.25 | 111 | 74.7% | **0.0%** |
| Tirante T.A. vs Choinski J. OVER | 2.30 | 2.60 | 1.33 | 149 | 13.0% | 48.8% |
| Bucsa C. vs Udvardy P. OVER | 1.80 | 1.30 | 1.30 | 122 | 27.8% | **0.0%** |
| Hijikata R. vs Monfils G. OVER | 2.06 | 1.49 | 1.42 | 218 | 27.7% | 4.7% |

**En 4 de 6 disparos la cuota no volvió a moverse jamás.** El disparo ocurre entre 83 y 218
minutos dentro de la serie de observación — es decir, al final. `mov_después = 0%` significa
literalmente: **ventana de oportunidad = cero**.

### 1.3 Por qué es estructural y no mala suerte

`certeza_matematica` (D147-02) se activa cuando el resultado del mercado ya es matemáticamente
imposible de revertir (`peor_caso = sets_remaining * 13`). Ese instante es, **por construcción**,
posterior al instante en que la casa reajustó su precio: la casa observa el mismo marcador y
tiene modelos de cierre automático de mercado.

Es el mismo error de fase que Nodo-180 encontró en la dirección de la apuesta, ahora en el
eje del tiempo: **no es que la señal sea falsa, es que llega después de que sirve.**

> Analogía sísmica corregida: el sistema actual no es una alerta temprana. Es un sismógrafo
> que suena cuando el edificio ya dejó de temblar. La información es correcta y es inútil.

### 1.4 Corrección de física — NO existe ventaja de velocidad

Una propuesta previa a este nodo planteó que la ventaja del sistema es la velocidad
("internet a 300.000 km/s vs onda sísmica a 5 km/s"). **Esa analogía es incorrecta y debe
quedar registrada como refutada** para que ningún nodo futuro la reconstruya:

- El sistema lee la API de Kambi (`_kambi_started_events()`, `_extract_games_cuota_live()`).
  Está **río abajo** de la casa por construcción. No puede anticiparla; la copia con latencia.
- La cadena de latencia documentada (Nodo-153) es Kambi ~1s → fast loop 5s → browser 5-7s.
  Es latencia *añadida*, no restada.

**La ventaja real, medida:** el mercado permanece equivocado durante minutos.
Caso Tirante T.A. vs Choinski J. — al disparo cuota=2.60, subió a **3.20 (+23.1%)** a los
5 minutos (el mercado se alejó de la verdad), y colapsó a **1.32 (−49.2%)** a los 12 minutos
(el mercado aceptó la verdad). Hubo una ventana real de ~7 minutos a un precio 23% mejor.

> **El activo no es la velocidad. Es la duración del desacuerdo del mercado.**
> Esa duración es medible, tiene magnitud, y nunca se ha explotado porque nunca se midió.

### 1.5 Lo que sí es transferible del sistema sísmico de Android

Descartada la física de velocidad, tres piezas de ingeniería del sistema de Google **sí**
son transferibles y ninguna está implementada hoy:

1. **Quórum entre sensores independientes.** Google ignora un celular vibrando; exige cientos
   coincidiendo. El `convergencia_score` actual (D142-02 + D165-01 + D166-01) suma heurísticas
   que **pueden estar correlacionadas entre sí** (gap, cuota, markov, ranking derivan en parte
   del mismo edge_report). Un score de 3 puede ser un solo sensor contado tres veces.
2. **Presupuesto de interrupción estratificado.** Google separa "alerta de atención" (no rompe
   el modo silencio) de "alerta de acción" (rompe todo, sonido máximo) y la segunda es rara
   por diseño. Hoy el sistema tiene un solo nivel: dispara Telegram o no dispara.
3. **Tiempo de anticipación como métrica de producto.** El sistema sísmico se evalúa por
   *segundos de ventaja entregados*, no por "detecciones correctas". Este sistema no tiene
   ninguna métrica de anticipación. Es la ausencia central que este nodo corrige.

### 1.6 Gap de trazabilidad — disparos de combos no son unibles

`core/fire_guard.py` (Nodo-161 D161-01) persiste `List[List[str]]` — **solo las claves,
sin timestamp**. Únicamente `certeza_fired_*.json` guarda hora ISO. Por tanto los disparos
de combos live (D133-04, D150/D157) **no se pueden cruzar con el historial de cuotas** y
quedan fuera de cualquier medición de ventana. Es el mismo patrón "el lazo no cierra" de
Nodo-174 (`n_actual` que nadie escribía).

### 1.7 Muestra histórica disponible — el instrumento produce veredicto de inmediato

Días con `certeza_fired_*.json` **y** `games_odds_history_*.json` simultáneos:

```
20260728:2  20260729:1  20260730:3  20260731:3  20260801:3  20260802:2  20260803:10
20260804:6  20260805:5  20260806:5  20260807:2  20260811:5  20260812:7  20260813:6
```

**60 disparos en 14 días, todos unibles retroactivamente.** D181-01 no necesita esperar a
acumular datos nuevos: produce un veredicto sobre la ventana con la primera ejecución.

---

## 2. PRINCIPIO DE DISEÑO — ORDEN NO NEGOCIABLE

> **Primero el reloj, después la alarma.**

Construir una alarma más ruidosa sobre una señal cuya anticipación jamás se midió es la forma
más rápida de perder dinero con más confianza. El sistema sísmico de Google funciona porque la
causalidad onda P → onda S es física validada sobre millones de eventos. La causalidad
convergencia → ganancia en este sistema **no está validada**: el settlement de `games_live`
acaba de repararse (Nodo-180 D180-01) y nunca corrió en producción (cron roto ~27 días,
corregido 2026-08-13, cero noches de datos acumuladas).

Consecuencia vinculante para la implementación: **D181-01 y D181-02 son bloqueantes de todo
lo demás**, y el nivel ACCIÓN (D181-06) nace apagado hasta que H181-01 gradúe.

---

## 3. DELIVERABLES

### D181-01 — `scripts/lead_time_report.py` (BLOQUEANTE, REPORTE_SOLO)

El reloj del sismógrafo. Instrumento puro de medición, no cambia ninguna decisión.

**Entrada:** `reports/certeza_fired_{fecha}.json` + `reports/games_odds_history_{fecha}.json`
(+ `reports/fire_ledger_{fecha}.jsonl` de D181-02 cuando exista).

**Por cada disparo calcula:**

| Campo | Definición |
|---|---|
| `cuota_t0` | primer punto de la serie |
| `cuota_fire` | último punto con `ts <= hora_disparo` |
| `cuota_final` | último punto de la serie |
| `mov_antes_pct` | `abs(cuota_fire - cuota_t0) / cuota_t0 * 100` |
| `mov_despues_pct` | `abs(cuota_final - cuota_fire) / cuota_fire * 100` |
| `ventana_min` | minutos entre el disparo y el último punto de la serie |
| `n_puntos_antes` / `n_puntos_despues` | densidad de observación a cada lado |
| `direccion_movimiento` | `A_FAVOR` / `EN_CONTRA` / `PLANO` respecto a la dirección apostada |

**Salidas:** `reports/lead_time_report_{fecha}.json` + resumen en texto plano.

**Agregados obligatorios sobre toda la muestra:**
`pct_ventana_cero` (fracción con `mov_despues_pct < 1%`), mediana y p25/p75 de `ventana_min`
y de `mov_despues_pct`, y el desglose `A_FAVOR` vs `EN_CONTRA`.

**Requisitos duros:**
- `--desde YYYY-MM-DD --hasta YYYY-MM-DD` para barrido retroactivo; sin flags, corre solo hoy.
- Debe reportar honestamente los disparos **sin serie de cuotas** como categoría propia
  (`SIN_HISTORIAL`), nunca excluirlos en silencio del denominador.
- `ts` cruza medianoche: si `mins(ts) < mins(ts_anterior)` en la misma serie, sumar 1440.
  Sin este guard las series nocturnas producen ventanas negativas. **Cubrir con test.**
- REPORTE_SOLO estricto. Prohibido que ningún gate, stake o combo lea su salida en este nodo.

**Wiring:** PASO 10g en `run_daily.py`, después del settle (mismo patrón que D179-04).

### D181-02 — `core/fire_ledger.py` (BLOQUEANTE)

Cierra el gap §1.6. Registro append-only unificado con timestamp absoluto de **todo** disparo.

```
registrar_disparo(fecha, clave, tipo, cuota_al_disparo, contexto: dict) -> None
```

- Escribe `reports/fire_ledger_{fecha_compact}.jsonl`, una línea por disparo:
  `{"ts_iso", "clave", "tipo", "cuota", "linea", "games_played", "contexto"}`.
- `tipo ∈ {CERTEZA, GAMES_LIVE, ITF_LIVE, COMBO}`.
- **Aditivo, no sustituye a `fire_guard`.** `should_fire()`/`mark_fired()` conservan su
  contrato y su archivo actual sin cambio alguno — el anti-flood no debe depender del ledger.
- Call-sites a instrumentar: `_fire_certeza_alert()`, `_fire_itf_live_games_combo()`,
  el disparo D133-04 de `alta_signals`, y `_fire_break_combos()`.
- Best-effort: un fallo de escritura **jamás** puede impedir un disparo (mismo patrón
  defensivo que `fire_guard.mark_fired`).

### D181-03 — `core/p_wave.py` (funciones puras, sin I/O)

El detector de onda P: el **inicio** del movimiento, no su consumación.

```
detectar_onda_p(serie: list[dict], linea: float, direccion: str,
                z_min: float = 2.0, n_min: int = 4) -> dict
```

Retorna `{"detectada": bool, "ts_onset": str|None, "z": float, "magnitud_acumulada_pct": float,
"direccion_implicita": "OVER"|"UNDER"|None, "n_puntos": int}`.

**Criterio de onset** (los tres, simultáneos):
1. `velocity_zscore` de los últimos `n_min` puntos ≥ `z_min` (reusar la lógica ya existente
   de Nodo-71 / D168-01 — **no reimplementar**, REGLA-T53).
2. Movimiento acumulado desde el onset **< 15%** — si ya se movió más, es onda S, no P.
3. `games_played < linea * 0.6` — todavía queda partido por jugar; sin esto se detecta el
   final de la serie, que es exactamente el defecto que este nodo corrige.

El punto 2 es el corazón del nodo: **la condición de que quede movimiento por capturar es
parte de la definición de la señal**, no una comprobación posterior.

Sin I/O, sin lectura de archivos, sin `datetime.now()`. Patrón `core/games_live_model.py`.

### D181-04 — Quórum por familias independientes

```
quorum_sensores(senal: dict) -> dict
```

Clasifica la evidencia disponible en **tres familias** y exige ≥1 sensor activo en ≥2 familias
distintas (recomendado: 3/3 para nivel ACCIÓN):

| Familia | Sensores | Origen |
|---|---|---|
| **MERCADO** | `p_wave.detectada`, `steam_confirmado`, `drift_pct` | precio, exógeno al modelo |
| **MODELO** | `mc_p_condicional`, `p_condicional` (D147), `edge_live` | predicción propia |
| **ESTADO** | `break_situation`, `serving`, `games_set1`, `score_data` | marcador real observado |

Retorna `{"familias_activas": [...], "n_familias": int, "quorum_ok": bool, "detalle": {...}}`.

**Razón de ser:** dos sensores de la misma familia pueden ser el mismo dato contado dos veces
(§1.5 punto 1). El `convergencia_score` actual no distingue familias y por eso un score de 3
no garantiza independencia. Este deliverable **no reemplaza** `convergencia_score` — lo
complementa como requisito adicional; los gates D150/D151/D164 quedan intactos.

### D181-05 — `_estimar_ventana_restante()` en `live_desk.py`

Cuenta regresiva honesta, calibrada con la distribución empírica de D181-01 (nunca con una
constante inventada). Si D181-01 aún no produjo muestra suficiente (n<20), debe devolver
`None` y la UI mostrar `"ventana: sin calibrar"` — **jamás un número inventado**.

### D181-06 — Panel `P_VENTANA` — alarma estratificada, arriba de todo

Un único panel al principio de la dashboard, encima de P_MEM y de todo lo demás. Sustituye
"lee 20 paneles y deduce" por "esto es lo accionable ahora".

**Dos niveles, presupuesto de interrupción distinto:**

| Nivel | Condición | Canal | Presupuesto |
|---|---|---|---|
| **ATENCIÓN** | onda P detectada + quórum ≥2 familias | Solo dashboard. Fila ámbar. Sin sonido, sin Telegram. | ilimitado |
| **ACCIÓN** | onda P + quórum 3/3 + `edge_live` ≥ umbral + ventana estimada > 3 min + memoria del arquetipo no negativa | Banner rojo + Telegram + cupón + stake | **máx. 3/día** |

Columnas del panel, en lenguaje llano y en este orden:
`QUÉ APOSTAR` (vía `_construir_explicacion_plana()`, D180-06 — reusar, no duplicar) ·
`CUOTA AHORA` · `VENTANA RESTANTE` · `STAKE SUGERIDO` · `POR QUÉ` (familias del quórum en
palabras: *"el precio se está moviendo, el modelo coincide y el marcador lo confirma"*) ·
`ESTADO` (OPORTUNIDAD / RESUELTO_GANADO / BLOQUEADO, D180-06).

Prohibido en este panel: `gap`, `midpoint`, `z`, `sigma`, `convergencia_breakdown`,
`p_condicional` crudo. La jerga vive en los paneles de abajo, no aquí.

**Regla de silencio honesta:** si no hay nada accionable, el panel dice explícitamente
*"Sin ventanas abiertas ahora mismo — N señales en vigilancia"*, con el número real.
No se rellena con lo mejor de un conjunto malo (compatible con la política de "nunca 0
recomendaciones": la alternativa concreta es la lista de vigilancia, no una apuesta forzada).

### D181-07 — Gate de honestidad: nivel ACCIÓN nace APAGADO

`_P_VENTANA_ACCION_ENABLED = False` en `live_desk.py`, junto a los demás gates del sistema
(patrón `_GCS_GATE_ENABLED`).

Con el flag apagado: el nivel ACCIÓN **se calcula y se registra en el fire_ledger, pero no
envía Telegram, no genera cupón y no propone stake** — se muestra en la dashboard con la
etiqueta `SIMULADO`. Es exactamente el modo REPORTE_SOLO que Nodo-179 aplicó a la memoria.

Solo se enciende cuando **H181-01 gradúa**. Esta es la diferencia central entre este nodo y
la propuesta que lo originó: *"si todo coincide, dispara el Kelly-KL"* asume que la coincidencia
predice ganancia. Eso es precisamente lo que no está demostrado.

### D181-08 — Segmento en `shadow_book.py --report`

Segmento `VENTANA H181` que cruza los disparos del fire_ledger con los picks liquidados,
para que H181-01/02/03 acumulen `n_actual` real vía el lazo de Nodo-174 D174-03
(`core/hypothesis_ledger.py`), no a mano.

### D181-09 — Tests `tests/test_nodo181_ventana.py` (REGLA-T53)

Mínimo 12, invocando siempre la función real:

1. `test_181_01` — `lead_time_report` sobre serie sintética: `mov_despues_pct` correcto.
2. `test_181_02` — serie que cruza medianoche no produce `ventana_min` negativa (§D181-01).
3. `test_181_03` — disparo sin serie de cuotas cae en `SIN_HISTORIAL`, no se excluye del total.
4. `test_181_04` — `detectar_onda_p` NO detecta cuando el movimiento acumulado ya supera 15%.
5. `test_181_05` — `detectar_onda_p` NO detecta cuando `games_played >= linea*0.6`.
6. `test_181_06` — `detectar_onda_p` SÍ detecta en onset temprano con z alto.
7. `test_181_07` — `quorum_sensores` con 3 sensores de la misma familia → `n_familias == 1`.
8. `test_181_08` — `quorum_sensores` con 1 sensor de cada familia → `quorum_ok is True`.
9. `test_181_09` — `fire_ledger.registrar_disparo` no rompe `fire_guard.should_fire`.
10. `test_181_10` — fallo de escritura del ledger no impide el disparo (best-effort).
11. `test_181_11` — con `_P_VENTANA_ACCION_ENABLED=False`, cero Telegram y cero cupón.
12. `test_181_12` — `_estimar_ventana_restante` devuelve `None` con n<20, nunca un número.

**Tripwire obligatorio, docstring literal e inmutable en `test_181_12`:**

```python
"""Tripwire Nodo-181 D181-07. Si este test falla, el nivel ACCIÓN se encendió
sin que H181-01 graduara — se está enviando dinero a una señal cuya anticipación
nunca se demostró (§1.2: 4 de 6 disparos medidos tenían ventana = 0%).
NO parchear el test: graduar H181-01 primero, o dejar el gate apagado."""
```

### D181-10 — Stake pre-cargado en el cupón (IMPLEMENTADO 2026-08-13)

**Origen:** el usuario objetó, con razón, que una alerta no sirve si la ventana se consume
haciendo login y escribiendo el monto, y autorizó explícitamente apostar 10.000 COP durante
una semana de prueba. Pidió automatizar el ingreso con sus credenciales.

**Hallazgo:** el formato de cupón de Kambi es `combination|<ids>|<stake>|<accion>`. El
proyecto siempre envió `BETPLAY_URL_TAIL = "||replace"` — **el tercer campo, que es el stake,
se envió vacío desde el primer combo construido.** Nadie lo usó nunca.

Llenarlo hace que el betslip abra con el monto ya escrito. La apuesta queda a **un toque**
del botón de confirmar, sin credenciales, sin automatización y sin violar los términos de la
casa. El usuario sigue confirmando siempre — y sigue viendo el monto antes de confirmar, que
es lo que hace que un error de unidad sea visible en vez de silencioso.

Implementado:
- `betplay_combo_builder.py` — `build_coupon_url(outcome_ids, stake=None)` y
  `build_redirect_url(outcome_ids, stake=None, label=None)`. Centralizadas: el formato estaba
  duplicado inline en ~15 sitios y en 4 módulos distintos.
- `docs/bp/index.html` — el JS lee `?stake=N`, lo valida contra `/^\d+$/` antes de inyectarlo
  y muestra el monto en la página puente.

**Retrocompatibilidad INMUTABLE:** con `stake=None` el string producido es byte-idéntico al
histórico (`...|IDs||replace`). REGLA-BAT-1 intacta. Cubierto por test.

**Regresión encontrada y corregida durante esta misma implementación — restricción de forma,
no solo de resultado.** La primera versión del JS construía siempre el mismo string computado
(`'|' + stakeField + '|replace'`), que en runtime es idéntico al histórico cuando no hay monto.
Aun así rompió `test_nodo162_redirect_coupon_format.py::test_162_02` y `::test_162_03`: esos
guards son **estáticos a propósito** — leen el HTML como texto y exigen que el literal
`coupon=combination|' + ids + '||replace'` exista en el fuente, precisamente porque el bug de
`4ae668d` que dejó todos los combos abriendo Betplay vacío ~5 días vivía en esta línea y no lo
detectó ningún test de comportamiento. Corregido ramificando en vez de calculando: la rama sin
monto es el string histórico literal, la rama con monto es la nueva. **Cualquier cambio futuro
a esta línea debe preservar el literal, no solo el resultado en runtime** — reconstruirlo por
concatenación deja el default sin guard estático aunque los tests de URL sigan verdes.

**Migración de los ~15 call-sites: NO hecha en este nodo.** Las funciones existen y están
probadas; conectarlas requiere decidir qué stake lleva cada estrategia, y eso depende de
D181-05/D181-06. Declarado aquí como deuda explícita, no como hecho.

### D181-11 — Sesión caliente (reduce la fricción restante)

Con D181-10 la fricción que queda es estar deslogueado cuando llega la alerta. No requiere
credenciales en código: basta mantener viva la sesión del navegador donde llega el Telegram.
Deliverable: nota operativa en el mensaje de ACCIÓN (*"confirma que estás logueado en Betplay
antes de abrir"*) y medición en D181-01 del tiempo alerta→apuesta registrada, para saber si la
fricción real sigue importando o ya es despreciable frente a una ventana de minutos.

---

## 3.B — EL DASHBOARD MIENTE: FILA Nally C. vs Kessler M. (evidencia 2026-08-14)

Fila real reportada por el usuario, reproducida completa:

```
Nally C. vs Kessler M. | CONFIRMAR UNDER | EN VIVO | Total de juegos | OVER | 21.5 | 21.5 |
— | @1.78 | — | — | 6:3, 5:2◄ [30:40] | 16j QUIEBRE | BAJA 59% |
"OVER 21.5 juegos · gana si el partido termina en 22 juegos o más" | 2/5 ALTA |
26-32+ | UNDER ≥32.5 | — | — | 12% [10%–13%]
```

### 3.B.1 — La aritmética que ninguna columna hizo

`6:3, 5:2◄ [30:40] | 16j` se decodifica sin ambigüedad:

- Set 1: 6-3 → **9 juegos**, ganado por Nally.
- Set 2: 5-2 → **7 juegos**, Nally arriba.
- Total jugado: **16 juegos**.
- `◄` = **saca Kessler**. `[30:40]` = **break point a favor de Nally**.
- Si Nally convierte, el partido termina 6:3 6:2 en **17 juegos**.

Camino aritmético completo para que el total llegue a 22 (lo que OVER 21.5 exige):

| paso | resultado | total |
|---|---|---|
| Kessler salva el BP y sostiene | 5:3 | 17 |
| Kessler **quiebra** a Nally | 5:4 | 18 |
| Kessler sostiene | 5:5 | 19 |
| Nally sostiene | 6:5 | 20 |
| Kessler sostiene | 6:6 | 21 |
| tiebreak | 7:6 | **22** |

Son **cinco juegos consecutivos contra la corriente, incluyendo un quiebre, arrancando desde
break point en contra**. P ≈ 0.12.

**El modelo acertó: `12% [10%–13%]` es correcto. Lo que falló fue la fila.**

### 3.B.2 — Ocho columnas, ocho estimadores, cero joins

| Columna | Mostró | Estimador que la produjo | Falla |
|---|---|---|---|
| Dirección | OVER | `linea_viva` vs `prior_congelado` | [[Nodo-180]] F1 — dice OVER porque la línea subió, no porque el partido vaya a durar |
| Banner | CONFIRMAR **UNDER** | certeza D147 | Contradice la dirección **en la misma fila**, sin reconciliar |
| Zona | 26-32+ | `p_model` **incondicional** | [[Nodo-180]] F2 — distribución de partido completo a 3 sets, con el partido a 1 juego de cerrar |
| Recomendación | UNDER ≥32.5 | zona | Hereda el error de la zona |
| Certeza | BAJA 59% | D147 gaussiano | Declara "baja confianza" en una situación casi determinista |
| Convergencia | 2/5 **ALTA** | `convergencia_score` | 2 de 5 no es ALTA — la etiqueta no corresponde al número que la acompaña |
| Explicación | "gana si termina en 22 juegos o más" | template estático | **Reformula la apuesta.** Cero información nueva |
| Banda CLT | **12% [10%–13%]** | CLT sobre MC condicionado | **La única columna correcta — y ningún consumidor la lee** |

El join que faltaba es de primaria: `p_propia = 12%` contra `p_implicita = 1/1.78 = 56%` da
**edge = −44%**. Cualquier gate de edge mata la fila al instante. La respuesta correcta ya
estaba renderizada en pantalla, en la columna del extremo derecho, sin conectar con nada.

Esto **no** es un bug nuevo: es F1 y F2 de [[Nodo-180]] hechos visibles en una sola fila, más
un fallo de presentación que ninguno de los dos nodos cubre — nadie verifica que las columnas
de una fila sean **mutuamente consistentes** antes de renderizarla como oportunidad.

### 3.B.3 — La ventaja real que el usuario está señalando

> *"si está en el segundo set y un jugador ya ganó el primero y ya realizó quiebre en el
> segundo, pues se apoya más un UNDER"*

Correcto, y es más fuerte de lo que parece. **El marcador no es decoración: es una cota casi
determinista sobre los juegos restantes, y es exógena al modelo** — no depende de que el
modelo acierte, igual que el gate `perdida_matematica` de D180-03. Desde cualquier marcador
en vivo se calcula sin probabilidad alguna:

- `juegos_min_restantes` — mínimo aritmético para que el partido termine (aquí: **1**).
- `juegos_hasta_cierre_forzoso_set` — techo antes de que el set deba resolverse.
- si una línea ya es **IMPOSIBLE**, ya está **RESUELTA**, o sigue **VIVA**.

Aquí `juegos_min_restantes = 1` → total mínimo 17. Para 22 hacen falta 6 más, y el único
camino aritmético a 6 exige tiebreak desde 5:2 abajo. Esa cota es la ventaja: acota la
distribución **antes** de que ningún modelo opine.

**Vínculo con la onda P (§1):** un quiebre en el segundo set con el primer set ya ganado *es*
el evento de onset. Cambia la distribución de juegos restantes de golpe, y el mercado de
totales tarda minutos en repreciar. Ese es el momento a capturar — pero se captura apostando
**UNDER**, que es justo lo que el banner decía y la dirección contradecía.

### D181-12 — `core/games_arithmetic.py` — cota determinista de juegos restantes

Módulo puro, sin I/O, mismo patrón que `core/games_settlement.py` / `core/monte_carlo_games.py`.
**Sin probabilidad**: solo aritmética de formato de tenis.

- `juegos_restantes_min(sets_ganados_home, sets_ganados_away, juegos_home, juegos_away, sets_a_ganar=2)`
- `total_alcanzable(...) -> (total_min, total_max_set_actual)`
- `estado_linea(linea, direccion, ...) -> "IMPOSIBLE" | "RESUELTO" | "VIVO"`

Test obligatorio con el caso exacto de §3.B.1: `6:3, 5:2` → `juegos_restantes_min == 1`,
`total_min == 17`, `estado_linea(21.5, "OVER") == "VIVO"` pero `total_min < 22`.
Y el control: `estado_linea(15.5, "OVER") == "RESUELTO"` (16 jugados ya lo superan).

### D181-13 — Gate de coherencia de fila (BLOQUEANTE para render)

Una fila **no se renderiza como oportunidad** si cualquiera de estas es cierta:

1. la dirección apostada contradice el banner de certeza (`OVER` vs `CONFIRMAR UNDER`);
2. `p_propia` vs `1/cuota` da edge por debajo del umbral (aquí −44%);
3. la zona recomendada es **aritméticamente inalcanzable** según D181-12 (`26-32+` cuando el
   máximo alcanzable ronda 22);
4. la etiqueta cualitativa no corresponde al número que la acompaña (`2/5` etiquetado `ALTA`).

Cuando dispara, la fila baja a estado `INCOHERENTE` **con el motivo explícito visible**, nunca
se muestra como pick. Mismo patrón que `check_contradictions.py`, aplicado a la fila renderizada
en vez de al documento. Fail-closed: fila sin datos suficientes para verificar coherencia se
trata como INCOHERENTE, no como válida.

### D181-14 — Explicación condicionada al marcador (reemplaza el template estático)

Prohibido un texto que solo reformule la apuesta. Formato obligatorio, tres partes, lenguaje
llano (jerga ya prohibida por D181-06):

1. **Dónde está el partido:** *"Nally ganó el primer set 6:3 y va 5:2 arriba en el segundo,
   con break point a favor."*
2. **Qué falta aritméticamente para el lado apostado:** *"para OVER 21.5 hacen falta 6 juegos
   más: Kessler tendría que salvar el break point, quebrar, y llevar el set a tiebreak desde
   5:2 abajo."*
3. **El número con su contraste:** *"el modelo le da 12%; la cuota @1.78 implica 56% — el
   mercado paga menos de lo que cuesta."*

**Tripwire de test:** la explicación debe contener al menos un número derivado del marcador en
vivo (juegos restantes, juegos jugados, o marcador). Un texto que no lo contenga falla el test
— es el guard contra que el template estático vuelva por la puerta de atrás.

---

## 4. HIPÓTESIS PRE-REGISTRADAS

A añadir en `validation/preregistered_hypotheses.json` **antes** de implementar D181-03+.

| ID | Enunciado | Umbral congelado | n_stop | Kill-switch |
|---|---|---|---|---|
| **H181-01** | Existe ventana explotable: en los disparos de nivel ACCIÓN, `mov_despues_pct ≥ 5%` con dirección `A_FAVOR` | ≥60% de los disparos | 40 | <35% con n≥20 → ACCIÓN permanece apagado |
| **H181-02** | La onda P anticipa a la certeza: para señales con ambos disparos, `ts_onda_p < ts_certeza` | ≥70% de los pares | 30 | <50% con n≥15 → detector inválido, revisar D181-03 |
| **H181-03** | El quórum 3/3 discrimina: hit rate de 3 familias > hit rate de ≤2 familias | ≥10pp de diferencia | 30 | diferencia ≤0 con n≥20 → quórum es ruido, retirar D181-04 |
| **H181-04** | Las filas que el gate D181-13 marca `INCOHERENTE` tienen hit rate por debajo del breakeven de su cuota — es decir: el gate está descartando filas malas, no filas buenas. Métrica: hit rate de las filas `INCOHERENTE` comparado contra `1/cuota_media` del mismo grupo | hit rate < breakeven (`1/cuota_media`) | 30 | hit rate ≥ breakeven con n≥20 → el gate está descartando valor y debe revisarse antes de seguir |

H181-01 es la que gobierna D181-07. Las cuatro necesitan predicado real en
`validation/hypothesis_ledger.py`; si alguna no es reducible a un booleano por-registro,
declararlo honestamente devolviendo `False` con comentario (patrón H179-01/H52-05), **nunca**
inventar un predicado que pase.

**H181-04 es REPORTE_SOLO por construcción:** el gate D181-13 bloquea el *render* de la fila
(nunca se muestra como pick), pero la fila debe seguir **registrándose** — con su motivo de
`INCOHERENTE` y su cuota — para poder medir esta hipótesis. Si la fila no se registra cuando
el gate la descarta, H181-04 es inmedible: dilo explícito en el código y en el log, no lo dejes
como una omisión silenciosa.

---

## 5. VERIFICACIÓN EN CALIENTE (obligatoria antes de declarar el nodo completo)

1. **§5.1** — `python3 scripts/lead_time_report.py --desde 2026-07-28 --hasta 2026-08-13`
   sobre los 60 disparos históricos de §1.7. Pegar el agregado real en CLAUDE.md.
   Si `pct_ventana_cero` sale alto, **es el resultado y se reporta tal cual**: confirma §1.2
   sobre muestra grande y justifica el nodo entero. Un resultado incómodo no se suaviza.
2. **§5.2** — reiniciar `tennis-live-desk` y confirmar por `curl` que `P_VENTANA` aparece en
   el HTML servido. *Python no recarga en caliente* — lección de la auditoría del 2026-08-13,
   donde el dashboard sirvió código de 17 horas atrás y pareció un Ghost Render.
3. **§5.3** — confirmar `reports/fire_ledger_{hoy}.jsonl` con ≥1 línea de un disparo real.
4. **§5.4** — confirmar por grep que ningún gate lee la salida de `lead_time_report`
   (REPORTE_SOLO real, no declarado).
5. **§5.6** — **verificación de la unidad del stake (D181-10), pendiente y necesaria.**
   `_STAKE_UNIT_VERIFICADO = False` en `betplay_combo_builder.py` hasta confirmarlo. No sé si
   Kambi espera el monto en pesos (`10000`) o en centavos (`1000000`). Se resuelve abriendo
   **un** link de prueba y mirando qué monto aparece en el betslip:
   `https://dakotapog.github.io/tennis-analysis/bp/?ids=<ID_REAL>&stake=10000`
   Si el betslip muestra $10.000 → unidad = pesos, poner el flag en `True`. Si muestra $100 →
   son centavos y hay que multiplicar por 100. Riesgo acotado por construcción: el usuario ve
   el monto antes de confirmar, así que una unidad equivocada es visible, no silenciosa.
6. **§5.5** — suite completa. Baseline vigente 2026-08-13: **2793 passed, 1 failed
   (`test_nodo115_uncertainty.py::test_build_que_falta_cuota_techo`, pre-existente sin
   relación), 2 skipped**. Triaje obligatorio de cualquier fallo nuevo (D174-02): prohibido
   escribir "N pre-existentes sin relación" sin enumerarlos.

---

## 6. LO QUE ESTE NODO NO HACE — LÍMITES EXPLÍCITOS

- **No promete ventaja de velocidad.** Refutada en §1.4. Si un nodo futuro la reintroduce,
  está contradiciendo evidencia registrada.
- **No toca `certeza_matematica`.** D147 sigue siendo correcto en lo que afirma; el problema
  es el uso que se le da (anunciar como oportunidad algo ya consumado). D180-06 ya lo
  reclasificó como RESUELTO_GANADO. Este nodo no lo elimina, lo saca del panel de acción.
- **No relaja ningún gate.** D150/D151/D164/D172/D175/D180-03 quedan intactos. El quórum
  es un requisito **adicional**, jamás una vía alterna.
- **No enciende dinero nuevo.** Con D181-07 apagado el nodo entrega instrumento y visibilidad.
  El primer peso extra solo se arriesga cuando H181-01 gradúe con n=40.
- **No aplica al mercado de SETS.** Sigue vigente el tripwire de Nodo-180 D180-09: SETS arrastra
  el prior incondicional `_P_3SETS_POR_ZONA` y no puede alcanzar un artefacto de apuesta.

### 6.1 Enfoque RECHAZADO — login automatizado con credenciales guardadas

Registrado aquí para que no se reconstruya en un nodo futuro. Propuesto y autorizado
explícitamente por el usuario el 2026-08-13 (10.000 COP, una semana de prueba); no
implementado, por tres razones y una que las supera:

1. **Términos de la casa.** Kambi/Betplay prohíbe la apuesta automatizada. El resultado
   realista no es perder los 10.000 — es el cierre de cuenta con retención de saldo. El tope
   bajo protege del error de cálculo, no de la consecuencia contractual.
2. **Extracción de credenciales.** Las credenciales "guardadas en el navegador" están cifradas
   en el gestor de Chrome atado al usuario de Windows. El código que las extrae es
   funcionalmente idéntico a un infostealer, independientemente de que el destino sea legítimo.
3. **Superficie de fallo no testeable.** El login tiene protección de bots y probablemente OTP.
   Un scraper no verificable dentro de un ciclo de refresco de 5–15 segundos no apuesta una
   vez: apuesta N veces. El tope de monto no acota el número de disparos.

**Y la razón que las supera: no resuelve el problema.** §1.4 midió la ventana en **minutos**
(caso Tirante: ~7 min a precio 23% mejor), no en segundos. La analogía sísmica no transfiere
aquí precisamente en este punto — la alerta de sismo debe automatizarse porque su ventaja son
5–30 segundos y ningún humano alcanza. Con una ventana de minutos, la fricción manual (~90 s)
no destruye la oportunidad. Lo que sí la destruye es que la alerta llegue cuando el movimiento
ya se consumió (§1.2: 4 de 6 disparos con `mov_despues = 0%`), y eso no lo arregla ninguna
automatización de ejecución. D181-10 baja la fricción a un toque sin tocar credenciales, que
es toda la ganancia disponible por esta vía.

---

## 7. ORDEN DE IMPLEMENTACIÓN

```
D181-02 (fire_ledger)  ─┐
D181-01 (lead_time)    ─┴─► ejecutar §5.1 sobre 60 disparos históricos
                              │
                              ├─ si pct_ventana_cero es alto → hallazgo confirmado, seguir
                              │
                              ▼
                       D181-03 (onda P) → D181-04 (quórum) → D181-05 (ventana restante)
                              │
                              ▼
                       D181-06 (P_VENTANA) + D181-07 (gate APAGADO)
                              │
                              ▼
                       D181-08 (shadow_book) → D181-09 (tests) → §5 verificación
```

**D181-01 y D181-02 son bloqueantes.** Si la medición sobre 60 disparos contradijera §1.2
—si resultara que sí hay ventana amplia y consistente— entonces el diseño de los deliverables
posteriores debe revisarse antes de escribirlos, no después. Ese es el punto de medir primero.

---

## 8. WIKILINKS

[[Nodo-147]] · [[Nodo-153]] · [[Nodo-160]] · [[Nodo-161]] · [[Nodo-168]] · [[Nodo-171]] ·
[[Nodo-174]] · [[Nodo-179]] · [[Nodo-180]] · [[Nodo-71]] · [[Nodo-111]]

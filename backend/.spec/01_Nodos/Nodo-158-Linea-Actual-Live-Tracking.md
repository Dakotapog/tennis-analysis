# Nodo-158 — Línea/Cuota Actual en Vivo — Adaptar y Recalcular (D158-01/D158-02)

**Estado:** COMPLETO
**Fecha:** 2026-08-01
**Módulos:** `live_desk.py`, `betplay_combo_builder.py`

---

## Contexto / Gap

Reporte real del usuario: dashboard mostraba `Jamie Mackenzie vs Max Dahlin OVER 21.5 @1.90`
pero la línea `21.5` ya no existía en Betplay — la mínima tradeable había subido a `23.5`.
Root cause: `sig["linea"]` (línea de origen pre-partido) nunca se refrescaba; la función
`_extract_games_cuota_live()` (D135-01) hacía fuzzy-match ±3.5j contra esa línea congelada
y devolvía la cuota de CUALQUIER línea cercana, etiquetándola como si perteneciera a la
línea original. El cálculo de certeza D147-02 también consumía la línea congelada, así
que ni siquiera el análisis interno era correcto una vez el mercado se movía — y el
disparo de combos vía `betplay_combo_builder.py --games --live` (D133-04) leía el
`games_signal_report` ESTÁTICO pre-partido con `outcome_id` igualmente congelado
(mismo riesgo de coupon muerto ya corregido para la ruta ITF pura en D157-02, nunca
corregido para esta ruta).

Decisión del usuario (explícita, vía pregunta directa): **"Adaptar y recalcular"** — la
línea T0 se mantiene como referencia de drift ("es nuestra base, la dejamos"), pero el
sistema debe rastrear la línea REALMENTE tradeable por separado, recalcular edge/certeza
contra ella cada ciclo, y solo disparar si el edge recalculado sigue pasando los gates
D150/D151 ya existentes.

## Implementación

- **D158-01 (`live_desk.py`, rama Path A de `_check_games_convergencia()`):**
  reemplaza `_extract_games_cuota_live()` (fuzzy-match) por `_fetch_live_games_all(event_id)`
  (ya usado y ya correcto en Path B/ITF-puro) — una sola llamada HTTP cacheada (TTL 120s)
  que devuelve la línea/cuota/oc_id que Kambi está sirviendo AHORA MISMO, sin adivinar.
  Nuevos campos, aditivos, sin tocar los existentes:
  - `sig["linea_actual"]`, `sig["cuota_actual"]`, `sig["oc_id_actual"]` (direction-aware:
    under u over según `sig["direccion"]`).
  - `sig["linea_drift"]` = `linea_actual - (linea_t0 o linea)` — cuánto se movió la línea
    tradeable respecto a la base congelada.
  - `sig["cuota_live"]`/`sig["drift_pct"]` se actualizan desde `cuota_actual` real (ya no
    fuzzy-matched); si no hay `cuota_pre` para comparar, `confianza_display="ALTA_SIN_CONFIRMAR"`.
  - `sig["linea"]`, `sig["linea_t0"]`, `sig["cuota_t0"]`, `sig["cuota_pre"]` — **intactos**,
    nunca sobreescritos. Siguen siendo la base congelada / referencia de drift (D147-03).
  - Log `[LINEA_ACTUAL]` cuando `linea_actual != linea` (línea se movió).
  - Si `_fetch_live_games_all` no encuentra mercado "Total de juegos" activo:
    `linea_actual/cuota_actual/oc_id_actual = None` + `confianza_display="ALTA_SIN_CONFIRMAR"`.
  - D147-02 (`_calcular_certeza_condicional`): la línea usada para calcular certeza/zona
    ahora prioriza `linea_actual` sobre la congelada — el análisis matemático se mide
    contra el mercado real, no uno muerto.
  - Path B (`itf_live_signals`, descubrimiento ITF puro) **no se tocó** — ya era live-aware
    desde D133/D150/D151 (siempre llamaba `_fetch_live_games_all` fresh cada ciclo).

- **D158-02 (`betplay_combo_builder.py`):** nueva función `build_games_combos_live()` +
  helper `_find_latest_games_live()`. Lee `reports/games_live_{fecha}.json` (escrito cada
  15s por `_check_games_convergencia`, D133-05) en vez del `games_signal_report` estático
  pre-partido. Filtra señales `estado in (EN_VIVO, ITF_VIVO)` con `linea_actual`/
  `cuota_actual`/`oc_id_actual` completos (fallback a campos legacy si faltan `_actual`),
  aplica los MISMOS gates D150/D151 (`_score_null_gate`, `_edge_live_gate`,
  `_zona_direccion_gate` — importados directamente de `live_desk.py`, sin duplicar lógica)
  antes de aceptar cada pierna. Combo único `"GamesLive"`, máx 3 piernas (REGLA-G5), mismo
  formato de salida (`combo_idx/label/legs/cuota_combo/stake/retorno/url/outcome_ids/n_piernas`
  + metadata `fuente/n_señales/n_candidatos/...`) que el builder legacy
  `build_games_combos()`, para compatibilidad total con `_generar_bat_games()` y
  `_enviar_games_telegram()` (incluye `"mercado": "Total de juegos"` por pierna, requerido
  por `_generar_bat_games()` L2042). Dispatch en `--live --games`: intenta
  `build_games_combos_live()` primero; si no hay candidatos (pre-market-open, sin ciclo
  `live_desk.py` corrido aún), cae al builder estático legacy sin cambios de comportamiento
  para ese caso.

- **X3 Panel (`live_desk.py::_build_x3_games()`):** nuevas columnas **"LínAct"** /
  **"CuotaAct"** en la tabla X3, entre "Línea" (histórica de la señal) y "Base(T0)"
  (congelada). Muestran `linea_actual`/`cuota_actual` en negrita + tag de drift
  `(+2.0j)` en ámbar cuando la línea se movió respecto a T0. `"—"` si el mercado
  "Total de juegos" aún no está confirmado en el ciclo actual. Así el usuario ve
  simultáneamente: línea histórica de la señal, línea REAL tradeable ahora, y línea
  congelada T0 — sin perder ninguna de las tres.

## No-Goals

- No se modificó `_fetch_live_games_all()` (ya correcta, D135/D133) ni los gates
  D150/D151 (`_edge_live_gate`/`_score_null_gate`/`_zona_direccion_gate`) — D158-02 los
  reutiliza vía import, no los reimplementa.
- No se tocó Path B (`itf_live_signals`) — ya era live-aware.
- `build_games_combos()` (legacy, estático) no se eliminó — sigue siendo el fallback
  correcto para uso standalone de `--games` fuera de horario de mercado en vivo.
- REGLA-G6 (stake cap $2,000 si `calibracion_n<50`) no se aplicó a
  `build_games_combos_live()` — sin histórico de calibración propio aún
  (`calibracion_n=0` en metadata); deuda para cuando H150-01/H151-01 acumulen n suficiente
  específicamente sobre esta ruta.

## Tests

`tests/test_nodo158_live_line_tracking.py` — 9 tests REGLA-T53. D158-01 (4 tests): replica
las fórmulas de `linea_drift` y de la selección `linea_actual`-prioritaria para certeza
(función monolítica con I/O HTTP en vivo vía `_fetch_live_games_all`, mismo patrón que
Nodo-150/157). D158-02 (5 tests): invocan `build_games_combos_live()` REAL contra archivos
`games_live_*.json` temporales — happy path, exclusión por cada uno de los 3 gates D150/D151,
campos `_actual` incompletos, archivo inexistente.

## Wikilinks

- [[Nodo-150]] — `cuota_envenenada`, `_calcular_certeza_condicional()` (D147-02, D150-02/03)
- [[Nodo-151]] — gates `_edge_live_gate`/`_score_null_gate`/`_zona_direccion_gate`
- [[Nodo-157]] — D157-02 (mismo patrón de bug: outcome_id congelado → coupon muerto),
  corregido aquí para la ruta Path A / GAMES-combo (D133-04) que Nodo-157 no cubría
- [[Nodo-147]] — baseline T0 inmutable (D147-03), panel X3 (D147-04)
- [[Nodo-133]] — `_check_games_convergencia()` ciclo EN_VIVO (15s), `games_live_*.json` (D133-05)
- [[Nodo-135]] — `_fetch_live_games_all()` / mercado "Total de juegos" vía `betoffer/event`

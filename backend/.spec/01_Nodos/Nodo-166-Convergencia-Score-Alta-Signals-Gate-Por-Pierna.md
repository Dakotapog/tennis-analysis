# Nodo-166 — convergencia_score por pierna en alta_signals, reemplazo del disparo D133 agregado (D166-01)

> Fecha: 2026-08-03
> Precede: [[Nodo-165]] (bonus certeza D147, mismo patrón replicado aquí a alta_signals), [[Nodo-164]] (gates D150 tier-agnóstico, precedente inmediato de "replicar no reinventar"), [[Nodo-142]] (D142-02 convergencia_score original, ITF), [[Nodo-133]] (disparo D133 agregado, reemplazado por este nodo), [[Nodo-147]] (certeza condicional, fuente del bonus reusado)

## 1. Disparador

El usuario, tras revisar Nodo-165 (bonus de certeza D147 aplicado solo a `itf_live_signals`), preguntó explícitamente si el mismo tratamiento de calidad por pierna debía extenderse a `alta_signals` (ATP/WTA/Challenger/ATP1000/ATP500), dado que el disparo D133 (`live_desk.py`, bloque `_check_games_convergencia`) seguía usando un umbral puramente agregado (`en_vivo_count>=2`) sin ningún gate de calidad individual — a diferencia de `itf_live_signals`, que desde D142-02 ya exige `convergencia_score>=3` por pierna antes de entrar al pool de disparo.

Se presentó la opción binaria: (a) dejar `alta_signals` con el disparo agregado viejo, o (b) construir para `alta_signals` la misma arquitectura `convergencia_score` que ya tenía `itf_live_signals` (D142-02 + bonus D165-01), reemplazando `en_vivo_count>=2` por un gate por pierna. El usuario aprobó explícitamente: **"Sí, replicar completo"**.

## 2. Decisión de diseño

**D166-01** — reemplazo del disparo D133 agregado por gate por pierna en `alta_signals`, reusando el stack existente sin duplicar lógica ("replicar no reinventar", Nodo-164):

- **Reusado sin cambio**: `_convergencia_score_itf(gap, cuota_live, markov, ranking_gap)` (D142-02) y `_convergencia_certeza_bonus(score_actual, certeza)` (D165-01) — ambos ya eran funciones puras genéricas, no específicas de ITF pese al nombre heredado.
- **Nuevo componente**: `_get_ranking_gap_er(home, away, er_picks)` — sustituye el proxy de rango de juegos que usa la vía ITF (que no tiene mercado pre-partido real) por los campos `ranking_favorito`/`ranking_rival` ya serializados en `edge_report` por `edge_calculator.py`. `alta_signals` SÍ tiene mercado pre-partido real (a diferencia de ITF qualy), así que este dato es más preciso que el proxy. Matchea por apellido buscado como substring en el campo `partido` de cada pick de `er_picks`; devuelve `None` sin inventar valor si no hay match o si el pick no trae los campos de ranking.
- **Nuevo campo propagado**: `"gap_juegos": s.get("gap_juegos")` añadido al dict que construye la entrada de `alta_signals`, necesario para alimentar el componente `gap` de `_convergencia_score_itf()` (antes no se propagaba porque nunca se usaba en el disparo agregado).
- **Cálculo por pierna**: para cada señal en `alta_signals` con `estado=="EN_VIVO"`, se calcula `convergencia_score`/`convergencia_breakdown`/`confianza` invocando `_convergencia_score_itf()` seguido de `_convergencia_certeza_bonus()` — mismo patrón exacto que el loop de `itf_live_signals`.
- **Gate de disparo**: `alta_pregame_raw = [s for s in alta_signals if s.get("convergencia_score",0)>=3 and not s.get("cuota_envenenada")]`, con exclusión adicional por tiebreak de set 1 (`games_set1>=12`, mismo criterio D150 ya tier-agnóstico desde Nodo-164). `convergencia_activa = len(alta_pregame_raw) >= 1` reemplaza el chequeo `en_vivo_count>=2`.
- **Combo key y logging actualizados**: `combo_key = sorted(s["partido"] for s in alta_pregame_raw)` (antes se armaba desde todas las señales EN_VIVO sin filtrar); log `[D166-01] CONVERGENCIA GAMES: {N} señal(es) convergencia_score>=3 (de {en_vivo_count} EN_VIVO) → combo disparado`.
- **Dashboard**: el banner de estado (`live_desk.py`, sección `_en_vivo_count > 0`) se actualizó para mostrar `"{en_vivo_count} señal(es) ALTA EN VIVO — {califican_count} con convergencia_score>=3 (disparo por pierna, no por conteo)"`, reemplazando el texto viejo que describía el umbral `>=2` ya no vigente.

## 3. Qué NO cambia (alcance)

- `itf_live_signals` y su gate de disparo (`alta_itf_raw`, D142-02/D147/D150/D151/D165-01) — no tocados, arquitectura ya replicada tal cual.
- `build_games_combos_live()` (`betplay_combo_builder.py`) — no tocado; sigue aplicando los gates D151 tier-agnósticos sobre cualquier señal ya presente en `games_live_*.json`, sin distinguir origen (`alta_signals` vs `itf_live_signals`).
- Los 5 gates de calidad de pierna (`_edge_live_gate`, `_score_null_gate`, `_zona_direccion_gate`, `cuota_envenenada`, tiebreak) — sin cambio de umbral ni de lógica, solo ahora también determinan qué entra a `alta_pregame_raw` antes del disparo D133, igual que ya determinaban `alta_itf_raw`.

## 4. Hipótesis pre-registrada

**H166-01** (`validation/preregistered_hypotheses.json`) — combos GAMES disparados por `alta_signals` bajo el nuevo gate por pierna (`convergencia_score>=3` individual) tienen hit rate >= al hit rate histórico del disparo D133 viejo (`en_vivo_count>=2` agregado, sin gate de calidad). Métrica: ratio hit-post / hit-histórico-pre sobre picks GAMES de `alta_signals` (no-ITF), vía `shadow_book.py --report`. `n_stop=20`, kill-switch ratio<0.80 con n>=15.

## 5. Verificación

- **Regresión detectada y corregida**: `tests/test_nodo133_games_live.py::test_convergencia_2_alta_dispara` codificaba el comportamiento viejo (2 señales EN_VIVO sin datos de mercado, disparo solo por conteo) — reescrito para exigir datos reales (`_fetch_live_games_all` mock con `cuota_under=2.00`, `gap_juegos=-4.5` → score=3 real vía `_convergencia_score_itf()`), más un test de control negativo nuevo (`test_convergencia_en_vivo_sin_score_no_dispara`) que prueba explícitamente que 2 señales EN_VIVO sin score suficiente YA NO disparan solo por el conteo. `test_antiflood_no_refire` también actualizado con los mismos datos de mercado para que el anti-flood se pruebe genuinamente (antes habría pasado trivialmente por falta de calificación, no por el guard de anti-flood).
- **6 tests REGLA-T53 nuevos** (`tests/test_nodo166_alta_convergencia_score.py`) para `_get_ranking_gap_er()`: match por apellido home, match por apellido away, sin match→None, match sin campos de ranking→None, `er_picks` vacío→None, integración con `_convergencia_score_itf()` real confirmando que `ranking_gap>300` en dirección UNDER suma +1 igual que en la vía ITF. 6/6 PASS.
- `python -m pytest tests/test_nodo133_games_live.py tests/test_nodo166_alta_convergencia_score.py -v --no-cov` → 6+6 = 12/12 PASS.
- Suite completa `pytest tests/ --no-cov -q --ignore=tests/test_nodo155_hcuc_convergence.py` → **39 failed, 2501 passed, 2 skipped** — coincide exactamente con el baseline pre-existente (39 failed confirmados no relacionados: test_nodo135/141/146/147/156/157/159/160/40/42/51_f3, ninguno toca `live_desk.py` D133/D166). **0 regresiones.**
- `python -c "import ast; ast.parse(open('live_desk.py').read())"` → OK.
- Pendiente (post-implementación de este spec): restart de `tennis-live-desk` (systemd :7780) — el código no recarga en caliente en Python; verificación runtime vía `curl`/`journalctl`/inspección de `reports/games_live_YYYYMMDD.json` para confirmar que `convergencia_score`/`convergencia_breakdown` aparecen en entradas reales de `alta_signals` cuando haya señales ATP/WTA/Challenger EN_VIVO al momento del chequeo.

## 6. Lección reusable

Igual que Nodo-165 (certeza D147 llega en un loop posterior al de `convergencia_score`), aquí el patrón fue el inverso: `alta_signals` SÍ tiene mercado pre-partido real (edge_report con ranking_favorito/ranking_rival) que `itf_live_signals` no tiene (jugadores ITF/qualy sin pick individual) — por eso el componente ranking se resolvió con una función distinta (`_get_ranking_gap_er` vs el proxy games-range de ITF) aunque el resto de `_convergencia_score_itf()` se reusó sin cambio. "Replicar no reinventar" no significa copiar cada línea sin adaptar las fuentes de datos disponibles — significa reusar la arquitectura de scoring y solo sustituir el componente cuya fuente de datos difiere genuinamente entre tiers.

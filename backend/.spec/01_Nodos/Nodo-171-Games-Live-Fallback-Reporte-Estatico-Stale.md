# Nodo-171 — GAMES --live: Fallback Silencioso a Reporte Estático Stale (D171-01/02/03)

**Estado:** IMPLEMENTADO
**Fecha:** 2026-08-04
**Módulo principal:** `betplay_combo_builder.py`

---

## Contexto / Bug reportado

Usuario reportó primero: *"no se dispararon nada no llego nada a telegram, ya
paso toda la tarde"* — a pesar de logs confirmando disparos D166-01 reales
(`subprocess.Popen(... --games --live --telegram)`). Tras corregir eso y hacer
una corrida de verificación en producción, el usuario reportó de inmediato:
*"mismo problema de siempre que esta documentado una y otra vez, los combos
solo abren la pagina de betplay, no existen ningun pick ni individual ni
combinadas"* — el mismo síntoma de [[Nodo-162]]/[[Nodo-169]], pero esta vez en
el flujo GAMES en vivo, y aparentemente causado por mi propia corrida de
verificación.

## Root cause

Tres bugs distintos, encontrados en cadena mientras se investigaba el primer
reporte:

**D171-01:** En el bloque `if args.live:` de `main()`, cuando
`build_live_combos()` retornaba vacío (trader_plans >4h stale — condición
normal en la tarde, el pipeline diario corre en la mañana), el código hacía
`sys.exit(1)` **antes** de llegar a la sección `--games`/`--mega`/`--safe`/
`--sistema` (~70 líneas más abajo). Esos 4 modos son motores independientes
que no dependen de `trader_plans` — pero morían con el proceso igual. Esto
explica por qué "no llegó nada a Telegram en toda la tarde" pese a que
D166-01 sí disparaba `subprocess.Popen(--games --live --telegram)`
correctamente.

**D171-02:** `",".join(outcome_ids)` en 5 sitios (construcción de links
Telegram, D157-06/D170-01) asumía `outcome_ids: List[str]`, pero los IDs
reales que vienen de Kambi son `int`. `TypeError: sequence item 0: expected
str instance, int found` — crash silencioso capturado en el log, sin
traceback visible en la ruta feliz del usuario.

**D171-03 (el más grave, root cause del segundo reporte):**
`build_games_combos_live()` (D158-02/Nodo-158 — el motor correcto: lee
`games_live_*.json` refrescado cada 15s, usa `linea_actual`/`cuota_actual`/
`oc_id_actual` realmente tradeables ahora, aplica los gates D150/D151/D164)
podía legítimamente retornar `[]` cuando ninguna señal en vivo pasaba los
gates de riesgo (comportamiento correcto y esperado — no es un bug que el
motor rechace señales). Pero el código en `main()` tenía:

```python
games_links, games_meta = build_games_combos_live(stake_per_combo=args.games_stake)
if not games_links:
    games_links, games_meta = build_games_combos(   # ← fallback peligroso
        stake_per_combo=args.games_stake,
        games_file=args.games_file,
    )
```

`build_games_combos()` es el builder estático original (Nodo-40), que lee
`games_signal_report_*.json` — generado **una sola vez al día** por
`run_daily.py` (PASO 3.6, mañana) y nunca regenerado intraday (confirmado:
sin cron/timer que lo refresque). Por la tarde ese reporte tenía horas de
antigüedad, con `outcome_id` de partidos que en muchos casos ya habían
terminado o cuyo mercado ya no existía en Betplay — coupon abre pero
Betplay no encuentra nada que cargar.

**Confirmación de que mi propia prueba de verificación cayó en este bug:**
el log de esa corrida mostraba `"📄 Leyendo: games_signal_report_20260804_083949.json"`
— el log statement exacto de `build_games_combos()`, no el formato distinto
de `build_games_combos_live()`. El mensaje "GamesA/GamesB" que el usuario
recibió y reportó como roto fue generado por este fallback, con datos de las
08:39 de la mañana.

## Fix

D171-01: mover el `sys.exit(1)` a que solo dispare cuando *ninguno* de los
modos independientes fue solicitado:
```python
if not combo_links and not (args.games or args.mega or args.safe or args.sistema):
    sys.exit(1)
```

D171-02: coerción defensiva `ids_str = ",".join(str(oid) for oid in outcome_ids)`
en los 5 sitios (`_enviar_games_telegram`, `_enviar_safe_telegram`,
`_enviar_mega_telegram`, `_enviar_sistema_telegram`, y un quinto sitio
compartido en la construcción de combos KAMBI-first).

D171-03: eliminado el fallback a `build_games_combos()` dentro del bloque
`--live`. Ahora, si `build_games_combos_live()` retorna vacío, se loguea y
se omite el envío — nunca se usa el reporte estático como sustituto. El
bloque standalone `if args.games:` (fuera de `--live`, para invocaciones
pre-partido explícitas) **no fue tocado** — ahí `build_games_combos()` sigue
siendo el comportamiento correcto y esperado.

## Verificación

- Dry-run en caliente (2026-08-04 16:46, 5 señales EN_VIVO/ITF_VIVO reales en
  `games_live_20260804.json`, mtime fresco): log confirma
  `[CAPA-LIVE] GAMES sin candidatos con mercado tradeable ahora
  (build_games_combos_live vacío) — se omite, NO se usa el reporte estático
  stale como fallback (D171-03).` — cero llamada a `build_games_combos()`.
- Confirmado por separado que las 5 señales EN_VIVO/ITF_VIVO actuales sí
  tienen `linea_actual`/`cuota_actual`/`oc_id_actual` no-nulos (Nodo-168 sigue
  funcionando) — el motor los rechaza por los gates D150/D151, no por falta
  de datos. Comportamiento correcto, no un bug nuevo.

## Tests

`tests/test_nodo171_live_games_fallback_fix.py` — 4 tests REGLA-T53, invocan
`main()` real con `sys.argv` mockeado (no reimplementan la lógica de
dispatch):
- `test_171_01`: `--games --live` no hace `sys.exit` con trader_plans vacío.
- `test_171_02`: `_enviar_safe_telegram` no crashea con `outcome_ids` int reales.
- `test_171_03`: `build_games_combos()` (estático) NUNCA se llama cuando
  `build_games_combos_live()` retorna vacío en modo `--live`.
- `test_171_04`: control positivo — cuando `build_games_combos_live()` sí
  retorna candidatos, esos son los usados, sin mezclar con el estático.

4/4 PASS. Suite completa (filtro games/nodo157/170/158/166,
`--ignore=tests/test_nodo155_hcuc_convergence.py` por import error
preexistente no relacionado): 138 passed, 3 fallas preexistentes confirmadas
vía `git stash` (no causadas por este fix) + 1 falla de aislamiento de orden
de tests (`test_nodo135`) que pasa individualmente — no es una regresión real.

## Wikilinks

- [[Nodo-158]] — origen de `build_games_combos_live()`, el motor correcto
- [[Nodo-166]] — D166-01, el disparo `subprocess.Popen(--games --live
  --telegram)` que reveló el bug (el motor de disparo funciona bien; el bug
  vivía en el proceso hijo que invocaba)
- [[Nodo-168]] — confirma que `linea_actual`/`cuota_actual`/`oc_id_actual`
  siguen poblándose correctamente para señales ITF; D171-03 no es una
  regresión de ese fix
- [[Nodo-170]] — D157-06/D170-01, el link de Telegram (correcto en sí mismo,
  pero apuntaba a `outcome_ids` de datos stale por D171-03 hasta este fix)
- [[Nodo-169]] — mismo síntoma reportado por el usuario ("abre Betplay sin
  picks"), root cause distinto (formato de coupon, no fallback a datos stale)
- [[Nodo-150]]/[[Nodo-151]]/[[Nodo-164]] — gates de riesgo D150/D151/D164,
  confirmados funcionando correctamente (rechazan las 5 señales actuales por
  diseño, no por bug)
- CLAUDE.md §9 REGLA-BAT-1 — formato de coupon, confirmado NO relacionado con
  este bug (ya cumplía el formato en ambos builders)

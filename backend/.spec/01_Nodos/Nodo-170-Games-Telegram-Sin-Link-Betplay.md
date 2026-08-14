# Nodo-170 — Telegram de Combos Live sin Link de Betplay (D157-06 completado + D170-01)

**Estado:** IMPLEMENTADO
**Fecha:** 2026-08-04
**Módulo principal:** `betplay_combo_builder.py`

---

## Contexto / Bug reportado

Usuario reportó recibir en Telegram el mensaje:

```
GAMES COMBOS — Totales (Nodo-40)
GamesLive @3.52 → $7,048 (stake $2,000)
  UNDER 26.5 @1.61 × UNDER 28.5 @1.44 × UNDER 27.5 @1.52
Cal n=0 | Total: $2,000
```

sin ningún link para abrir Betplay — preguntó si era el mismo problema ya
documentado de coupon roto ([[Nodo-162]], [[Nodo-169]]).

## Root cause

**No es el mismo bug que Nodo-162/Nodo-169.** En esos nodos el link existía
pero con formato de coupon inválido (`|ML/` en vez de comas, o redirect page
rota). Aquí `_enviar_games_telegram()` (`betplay_combo_builder.py`) **nunca
construyó ningún link, de ningún formato, desde su creación** — commit
`7f159dd` (2026-06-30, Nodo-45) — confirmado con
`git log --all --oneline -S "_enviar_games_telegram" -- betplay_combo_builder.py`
(un solo commit). El combo sí guarda `outcome_ids` (consumido correctamente
por `_generar_bat_games()` para el flujo `.bat`/HTML de escritorio, que
funciona bien), pero `_enviar_games_telegram()` arma el texto del mensaje
solo desde `label`/`cuota_combo`/`legs`, sin referenciar `REDIRECT_BASE` ni
`outcome_ids` en ningún punto.

**Evidencia de que ya estaba especificado pero nunca implementado:**
`tests/test_nodo157_games_telegram_link.py` (creado 2026-08-02, docstring cita
"D157-06") pre-existía en el repo con 3 tests que esperan exactamente el fix
correcto (`[ABRIR {label}]({REDIRECT_BASE}{ids_str})`), pero
`Nodo-157-Contrarian-Over-Cuota-Envenenada.md` no documenta D157-06 en ningún
lado — el test se escribió como spec ejecutable, el fix nunca se aplicó al
código. Correr la suite antes de este fix confirmó **2 de 3 tests fallando**
(`test_157_30`, `test_157_32`), probando que el gap era real y detectable.
Mismo patrón de lección que [[Nodo-168]]: un artefacto de spec (tests verdes
esperados) no es evidencia de una implementación completa si nadie corre esos
tests contra el código real.

**Auditoría extendida (D170-01):** el mismo gap (guarda `outcome_ids`, nunca
lo usa en el texto de Telegram) existe idénticamente en 3 funciones hermanas:
`_enviar_safe_telegram()`, `_enviar_mega_telegram()`, `_enviar_sistema_telegram()`
— confirmado leyendo el cuerpo completo de cada una. Ninguna tenía test
previo para este gap.

## Verificación operativa (antes del fix)

- `systemctl --user status tennis-live-desk` — servicio ACTIVO, PID 287.
- `logs/live_desk.log` grep `D166-01|CONVERGENCIA GAMES` — confirma que la
  detección de convergencia y el disparo (`subprocess.Popen(... --games --live
  --telegram)`) SÍ están funcionando en producción hoy (2026-08-04, múltiples
  disparos entre 10:21 y 11:52) — el motor de señales no es el problema.
- `reports/games_live_20260804_fired.json` — historial de combos disparados
  coincide con los logs.
- `/mnt/c/users/hogar/Desktop/combos/games_live.html` (generado 11:53:04,
  inmediatamente después del último disparo 11:52:56 en el log) — contenido
  confirma coupon **correcto**: `coupon=combination|4285326120,4285229734,4285343011||replace`.
  Prueba que el flujo `.bat`/escritorio funciona perfecto; el bug está
  aislado 100% al texto del mensaje de Telegram.

## Fix

D157-06 (games) + D170-01 (safe/mega/sistema): mismo patrón en las 4
funciones — por cada combo con `outcome_ids` no vacío, agregar línea Markdown
`[ABRIR {label}]({REDIRECT_BASE}{ids_str})` (games/safe, una línea por combo)
o `[ABRIR]({REDIRECT_BASE}{ids_str})` inline al final de la línea compacta
(mega/sistema, formato de una línea por combo para no superar 4096 chars).
Si `outcome_ids` está vacío, no se agrega link (comportamiento defensivo, no
lanza excepción) — mismo criterio ya cubierto por
`test_157_31_games_telegram_sin_outcome_ids_no_lanza_ni_agrega_link`.

`REDIRECT_BASE` ya era la fuente de verdad usada por
`enviar_combos_telegram()` (combos de favoritos) desde siempre — mismo patrón
reusado sin introducir un tercer formato de link.

## Tests

- `tests/test_nodo157_games_telegram_link.py` — 3 tests (pre-existentes,
  ahora 3/3 PASS, antes 1/3 PASS).
- `tests/test_nodo170_safe_mega_sistema_telegram_link.py` — 6 tests nuevos
  REGLA-T53 (2 por función: con link / sin outcome_ids), invocan las
  funciones reales `_enviar_safe_telegram()`/`_enviar_mega_telegram()`/
  `_enviar_sistema_telegram()` con `urllib.request.urlopen` mockeado.
- 9/9 PASS. Suite completa sin regresiones (ver changelog run).

## Wikilinks

- [[Nodo-162]] — mismo síntoma (coupon sin piernas), root cause distinto
  (formato roto, no ausencia total de link)
- [[Nodo-169]] — mismo síntoma, mismo patrón de auditoría, archivo distinto
  (`favoritos_combo_builder.py`)
- [[Nodo-168]] — misma lección: "tests verdes esperados" ≠ "implementado y
  verificado", el bug vivía en la intersección entre spec-como-test y
  ejecución real
- [[Nodo-157]] — origen nominal de D157-06 (nunca documentado en el spec
  original, solo en el docstring del test)
- [[Nodo-133]] — `_check_games_convergencia()` / D166-01, motor de disparo
  confirmado funcionando (no tocado por este fix)
- CLAUDE.md §9 REGLA-BAT-1 — fuente de verdad del formato coupon
  (`REDIRECT_BASE` ya cumplía el formato correcto; el bug era su ausencia,
  no su formato)

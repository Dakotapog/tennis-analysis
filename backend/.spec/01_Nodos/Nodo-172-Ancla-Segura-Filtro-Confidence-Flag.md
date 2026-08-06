# Nodo-172 — Ancla Segura: Filtro `confidence_flag` en Combos Multi-Pierna (D172-01/02/03)

**Estado:** IMPLEMENTADO
**Fecha:** 2026-08-04
**Módulo principal:** `betplay_combo_builder.py`

---

## Contexto / Hallazgo de auditoría

El 2026-08-03 un combo generó ganancia notable: Castellanos Y. + Mejia N. + 
Udvardy P. + Ruse G. @49.33x, $500 → $24,665 retorno (registrado como 
`Combo5` en `reports/combo_registry/cr_2026-08-03.jsonl`, generado por 
`build_live_combos()` vía `betplay_combo_builder.py --live`). Usuario pidió 
auditoría de este combo. Investigación en `edge_report_20260803_100618.json` 
reveló un patrón problemático:

- **Castellanos Y.:** p_modelo=0.801, **confidence_flag=STRONG**, edge_pct=50.2%
- **Mejia N.:** p_modelo=0.513, **confidence_flag=LOW**, edge_pct=18.0%
- **Udvardy P.:** p_modelo=0.51, **confidence_flag=LOW**, edge_pct=9.8%
- **Ruse G.:** p_modelo=0.541, **confidence_flag=LOW**, edge_pct=4.6%

Solo **1 de 4** piernas tenía convicción genuina del modelo (STRONG). Las otras 
3 eran LOW confidence_flag — esencialmente near-coinflip (p_modelo 0.51-0.54) 
— y ganaron por varianza, no por señal. El `build_live_combos()` usado para ese 
combo **NO filtraba** piernas de relleno por `confidence_flag` en absoluto 
(solo REGLA-HF-1: cuota≥1.50 y disponibilidad Kambi). El precedente más 
cercano, `build_system_combos()` (Nodo-156-B, "Sistema Leave-One-Out"), tampoco 
filtraba fillers por confidence_flag duro — solo loguea warning si p_modelo 
está bajo, nunca los excluía.

## Root cause

La arquitectura de combos multi-pierna existente (sistema, live combos, safe, 
mega) **no distinguía entre fillers STRONG vs MODERATE vs LOW** — aceptaba 
cualquier pick con cuota≥1.50 y Kambi disponible. Esta fue una decisión 
deliberada en diseños anteriores (quizás razonable para combos de 
descubrimiento), pero para una estrategia deliberada de "ancla de confianza + 
rellenos confiables", la debilidad es fatal: un combo que gana solo porque sus 
3 fillers LOW acertaron por azar no es reproducible ni backtesteable.

**D172-01 (brecha arquitectónica):** Ni `build_live_combos()` (D133-04, disparo 
de convergencia D166-01) ni `build_system_combos()` (D156-B-01) ni cualquier 
builder existente leía `confidence_flag` de `edge_report` para aplicar una 
política explícita de calidad de filler.

**D172-02 (no hay validación de criterios de anclaje):** `build_system_combos()` 
podía seleccionar cualquier pick como "ancla" solo si cuota≥1.65 (default). No 
había validación que esa "ancla" tuviera convicción genuina (STRONG). Un 
MODERATE con cuota 3.0 se promovía a ancla automáticamente.

**D172-03 (no hay criterio de filler de calidad):** Los 3-5 fillers se 
seleccionaban por p_modelo descending, sin jamás revisar si eran STRONG, 
MODERATE, o LOW confidence. El algoritmo nunca preguntaba "¿cuál es la 
confianza real del modelo en este pick?"

## Fix

Nueva función `build_ancla_segura_combos(stake_total=3000, n_fillers=3, 
ancla_cuota_min=2.50, filler_confidence_allowed=("STRONG","MODERATE"))` 
agregada a `betplay_combo_builder.py` (línea ~3406-3610, después de 
`_enviar_sistema_telegram()`, antes de `# MAIN`):

**D172-01: Carga y validación de edge_report:**
```python
edge_report = _find_latest_edge_report()
_validate_edge_report_gate(edge_report)
```

**D172-02: Construcción de pool de candidatos (con extracción de confidence_flag):**
Pool incluye `apostar` + `watchlist`. Cada candidato extrae:
- `favorito_predicho`, `cuota_favorito`, `p_modelo`, **`confidence_flag`** 
  (STRONG|MODERATE|LOW — el nuevo campo que esta estrategia lee, que 
  `build_system_combos` nunca consumía), `tier`
- Aplica REGLA-HF-1 (cuota<1.50 excluida) + D140-02 (kambi_disponible=False 
  excluida)

**D172-03: Selección de ancla con filtro STRONG + cuota mínima:**
Solo candidatos con `confidence_flag == "STRONG"` AND `cuota >= ancla_cuota_min` 
(default 2.50) califican como ancla. Se elige la de cuota máxima. 
**Si ninguno califica → retorna ([], {}), no genera combo.** Fail-loud, no 
fallback silencioso.

**D172-04: Selección de fillers con filtro STRONG/MODERATE (LOW excluido duro):**
Solo candidatos con `confidence_flag in filler_confidence_allowed` (default 
`("STRONG", "MODERATE")`) califican. **LOW excluido explícitamente, jamás como 
fallback.** Ranked por p_modelo descending, top n_fillers (default 3) tomados. 
**Si hay fewer than n_fillers → retorna ([], {}). No relaja el filtro para 
llenar la cuota.** Este fue un requisito explícito del usuario para nunca 
aceptar LOW.

**D172-05: Resolución de outcome_ids y construcción del coupon:**
Mismo patrón que `build_system_combos`: `fetch_kambi_outcomes()` + 
`find_outcome()`, URL Betplay REGLA-BAT-1 compliant (comma-joined IDs, 
`||replace`, sin `|ML/`).

**D172-06: Funciones de soporte (mismo patrón que sistema):**
- `_mostrar_ancla_segura(links, metadata)` — display console, muestra cada 
  pierna taggeada ANCLA/filler con su confidence_flag
- `_generar_bat_ancla_segura(links)` — escribe `AnclaSegura.bat` + 
  `anclasegura.html` a Desktop, limpia previos, registra via `ComboRegistry.log_combo("AnclaSegura", "ANCLA_SEGURA", ...)` — nuevo strategy tag distinto (D144-08 gap)
- `_enviar_ancla_segura_telegram(links, metadata)` — notif Telegram con link 
  REDIRECT_BASE (D171-02 style `str(oid)` coercion), muestra ancla + 
  confidence_flag per leg, nota explícita "solo fillers STRONG/MODERATE, LOW 
  excluido"

**D172-07: CLI wiring en main():**
Nuevos flags: `--ancla-segura`, `--ancla-segura-stake` (default 3000), 
`--ancla-segura-fillers` (default 3), `--ancla-segura-cuota-min` (default 2.50)
Dispatch en dos lugares (mismo patrón `--sistema`):
1. Inside `--live` flow combinado (gateado por `args.ancla_segura`)
2. Standalone mode antes de `--evaluar` (gateado por `args.ancla_segura`, 
   `sys.exit(1)` si no hay combo)

**D172-08: Guard D171-01 actualizado:**
Condición `if not combo_links and not (args.games or args.mega or args.safe 
or args.sistema): sys.exit(1)` ahora incluye `or args.ancla_segura` — si 
trader_plans stale/empty, `--ancla-segura` standalone todavía ejecuta (no muere 
el proceso antes).

## Verificación

- El caso exacto 2026-08-03 (Castellanos STRONG + Mejia/Udvardy/Ruse LOW) → 
  `build_ancla_segura_combos()` retorna `([], {})`, **NO se genera combo**. 
  test_172_01 verifica esta regresión guard.
- Combos generados post-fix (n=0 en escritura de spec, sin casos reales en 
  producción aún) mostrarían ancla STRONG + fillers STRONG/MODERATE con 
  confidence_flags en banner Telegram.

## Tests

`tests/test_nodo172_ancla_segura.py` — 8 tests REGLA-T53, invocan la función 
real `build_ancla_segura_combos()` (mocking solo `_find_latest_edge_report` → 
temp JSON en disco, `fetch_kambi_outcomes`/`find_outcome` → fake resolvers, 
nunca hardcodeando fórmulas):

- `test_172_01`: caso exacto 2026-08-03 (1 STRONG + 3 LOW) → `links == []`, 
  NO genera combo. Core regression guard.
- `test_172_02`: 1 STRONG ancla + 3 MODERATE/STRONG fillers → combo generado, 
  ningún LOW en legs.
- `test_172_03`: confirma leg ancla es siempre confidence_flag=STRONG 
  específicamente.
- `test_172_04`: sin candidatos STRONG → no hay ancla → `([], {})`.
- `test_172_05`: STRONG ancla + fillers insuficientes → no relaja quota → 
  `([], {})`.
- `test_172_06`: formato coupon URL — sin `|ML`, ends `||replace`, comma-joined 
  IDs (REGLA-BAT-1).
- `test_172_07`: REGLA-HF-1 — cuota<1.50 excluida aunque sea STRONG.
- `test_172_08`: STRONG candidate <ancla_cuota_min NO califica como ancla.

8/8 PASS. Suite completa (2517+8=2525): regresión testing performed, 0 
regresiones confirmadas.

## Wikilinks

- [[Nodo-156-B]] — "Sistema Leave-One-Out", precedente arquitectónico más 
  cercano (ancla+fillers sin confidence_flag filter) — contraste clave
- [[Nodo-165]] / [[Nodo-166]] — ambos usan filtros de calidad de señal (score 
  convergencia, certeza D147) para no aceptar señales débiles — filosofía 
  similar
- [[Nodo-144]] — strategy tagging en shadow_book/ComboRegistry, D144-08 gap 
  ("plan vacío → SIN_TAG"), esta estrategia parcialmente aborda con nuevo tag 
  ANCLA_SEGURA
- [[Nodo-150]] / [[Nodo-151]] / [[Nodo-164]] — gates de riesgo ITF que rechaza 
  candidatos débiles; mismo principio (no aceptar LOW)
- [[Nodo-171]] — guard D171-01 actualizado en `main()` para incluir 
  `--ancla-segura` en la condición de sys.exit
- CLAUDE.md §11 — añadida como estrategia #13 a la taxonomía de combos

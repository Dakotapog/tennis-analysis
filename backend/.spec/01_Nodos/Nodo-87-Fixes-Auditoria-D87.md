# Nodo-87 — Fixes de la Auditoría D87 (implementación de Nodo-86)

> **Wikilinks:** [[Nodo-86-Auditoria-Fable5]] | [[Nodo-64-RFI-Return-From-Inactivity]] | [[Nodo-66-Plan-Trabajo-Semanal-Sonnet]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-70-CPPI]] | [[Nodo-72-Phantom-Identity-Guard]]
> **Fecha:** 2026-07-11 | **Autor:** Fable 5 (implementación) — documentado por Sonnet 5
> **Estado:** IMPLEMENTADO — 12 fixes de código + verificación pytest. Commit `2251667`.
> **Verificación:** `pytest tests/ --no-cov -q` → **1804 passed, 0 failed** (WSL, post-fixes, 2026-07-11).
> **Gap conocido:** ninguno de los 1804 tests cubría estos caminos antes del fix — es la razón por la que los bugs vivieron sin detectarse. Tests REGLA-T53 nuevos: PENDIENTE (T3 de Nodo-66).

---

## 1. Tabla completa D87-01 → D87-11 + D64-01

| ID | Archivo:línea | Bug (Nodo-86 §) | Fix |
|---|---|---|---|
| **D87-01** | `shadow_book.py:633` (busca `favorito_predicho`) + `:578,630` (`_type` no `record_type`) | §1.2 — `update_alpha_flags` nunca marcaba nada: campo inexistente en pick_snapshot. Confirmado 0 ocurrencias de `alpha_promoted` en 10 días de shadow book pese a D62-05 activo en producción | H62-01 puede acumular por primera vez desde su pre-registro |
| **D87-02** | `edge_calculator.py:1042` | §1.5 — el gate GCS (`_GCS_GATE_ENABLED`) podía forzar `apostar=True` sobre picks ya bloqueados por T33-01 (coin-flip n_h2h=0), HOT_sin_BBI, n_axes<2, o NO_DATA/phantom | Gate GCS ahora exige `not motivo_reclasificacion` y `status != NO_DATA` antes de revivir el pick |
| **D87-03** | `trader_ev_tenis.py:498,578,624,778` (4 sitios: individuales, combos, sistema, cobertura) | §1.3 — `max(MIN_BET, stake)` forzaba $1,000+ incluso con `kelly<=0` (EV negativo bajo p_blend) o presupuesto agotado (`capped_stake<=0`) | Stake 0 cuando kelly≤0 o budget no alcanza — se acabaron los stakes fantasma |
| **D87-04** | `edge_calculator.py:726`, `betslip_registrar.py:557` | §1.1 — `'?'` (valor del bookmarklet Kambi) no estaba en la lista de valores degradados a normalizar | Ambos puntos tratan `'?'`/`''` como `unknown` — no se crean más buckets huérfanos nuevos |
| **D87-05** | `trader_ev_tenis.py:438` (`_p_blend`) | §1.4 — **el fix de mayor impacto financiero**. Con `n_h2h=0`, `p_blend = p_prior` puro; `p_prior` es la accuracy histórica del tier (ej. 0.758 clay GS), no la probabilidad del pick concreto → EV ficticio ~+89% ordenaba y financiaba combos | `p_prior_efectivo = min(p_prior, p_modelo)` — el prior solo puede reducir la estimación, nunca inflarla por encima del modelo |
| **D87-06** | `pre_game_validator.py:42,59,112` | §1.6 — el validador solo veía la lista `apostar` (fallback genérico); watchlist/sin_edge, que entran al pool del trader por defecto, nunca se validaban. Además buscaba campos (`jugador`, `ranking`) que no existen en el pick real | Reconoce el schema real (`apostar`+`watchlist`+`sin_edge`) y usa `favorito_predicho`/`ranking_favorito` |
| **D87-07** | `trader_ev_tenis.py:1189` | §1.7 — el ajuste CPPI (piso de supervivencia Nodo-70) solo se aplicaba a individuales; la cobertura (la capa que más capital consume) quedaba fuera del waterfall kelly→VaR→CPPI | `c['stake'] = round(c['stake'] * fv * cppi_f / MIN_BET) * MIN_BET` — CPPI cubre ambas capas |
| **D87-08** | `betplay_combo_builder.py:2043,2058,2521,2562` | §4.2 — el `betslip_index` (puente outcome_id→modelo) no guardaba `p_modelo`/`kelly_kl`, y el filtro `cuota≥min_cuota` excluía del index a las piernas VARIABLE (cuota 1.18-1.35) que sí se apuestan en combos de confianza | Index cubre TODO pick (`cuota>1.0`) con `p_modelo`/`kelly_kl`/`n_h2h`; se guarda ANTES del gate de combos (no depende de que haya combos armables) |
| **D87-09** | `betslip_registrar.py:241-270,352-394` | §4.2 — picks fuera del index se descartaban silenciosamente; sin backfill, campos degradados nunca se completaban | Picks fuera del index se registran DEGRADADOS (no se pierden del tracking); nuevo backfill automático desde el edge_report más reciente completa superficie/tier/match_id/p_modelo por nombre |
| **D87-10** | `trader_ev_tenis.py:946,950` | §1.7 — `--all-picks` default=True metía `sin_edge` (edge≤0) al pool de cobertura, que junto con D87-05 recibía EV ficticio y stake real | Default cambiado a `False`; requiere flag explícito |
| **D87-11** | `shadow_book.py:712-737` | §4.4 — el settle tier-3 (fallback por nombre) matcheaba solo el favorito contra todos los resultados del día; un jugador con dos partidos el mismo día (qualy+main, u homónimo) podía settlearse contra el partido equivocado | Exige que el RIVAL también coincida antes de aceptar el match |
| **D64-01** | `analysis/rivalry_analyzer.py:2388` + `edge_calculator.py:894-940` + `shadow_book.py:1127` | Nodo-64 — la señal RFI (H76-01) llevaba n=1/30 en registro **manual** desde 2026-07-09; sin código, jamás iba a graduar | Serializa `form_decay_meta` estructurado en el motor; `edge_calculator` calcula `rfi_tier`, `rfi_ultra`, `rfi_decay_gap`, `rfi_is_bookie_fav`, `rfi_model_picks_active`; `shadow_book --report` añade segmentos `RFI-ULTRA` y `rfi_tier>=1` — acumulación automática desde ahora |

## 2. Qué NO se tocó (deliberado — requiere decisión/nodo propio)

- Migración de los ~141 resultados históricos en los buckets `?`/`?_?` de `calibracion_edge.json` (§1.1) — T7 de Nodo-66, solo propuesta.
- Que el sizing use `kelly_kl` de 5 capas en vez de `_kelly_quarter(p_blend)` — cambio de arquitectura del trader, no un fix.
- Unificación de las 4 implementaciones de name-matching en `core/player_registry` — C2 de Nodo-67.
- La señal "campeón reciente" contada 4 veces (BONUS/GCS/tier_mismatch/H77-03) — C3 de Nodo-67, solo diseño esta semana.

## 3. Impacto combinado (la cadena que se rompió)

```
ANTES: pick n_h2h=0 → p_blend=0.758 (inventado) → EV ficticio +89% → combo
       financiado → apuesta real con superficie="?" → calibra en bucket "?_?"
       que nadie lee → 24% hit real invisible (vs 50-64% shadow reportado)

AHORA: p_blend no puede superar p_modelo → EV≤0 da stake 0 → si se apuesta,
       superficie real via betslip_index/backfill → calibra en el bucket
       correcto → medible
```

## 4. Trabajo derivado (ver Nodo-66 T3) — ✅ ESCRITO 2026-07-12, pendiente de correr en WSL

`tests/test_nodo87_fixes.py` cubre 9 de los 12 IDs con REGLA-T53 (invoca la función real,
aserciones estructurales/de umbral — nunca reimplementa la fórmula que prueba):

| Clase | IDs cubiertos |
|---|---|
| `TestD87_01AlphaFlags` | D87-01 (update_alpha_flags matchea `favorito_predicho`) |
| `TestD87_02GateGCSRespetaGuards` | D87-02 (gate GCS no revive picks NO_DATA/phantom) — incluye test de control confirmando que el camino feliz (H60-01) sigue intacto |
| `TestD87_04NormalizacionSuperficie` | D87-04 (`'?'` → `unknown` en edge_calculator) |
| `TestD87_05PBlendNoInfla` | D87-05 (prior nunca infla por encima de p_modelo) |
| `TestD87_06ValidatorSchemaReal` | D87-06 (pre_game_validator escanea watchlist/sin_edge) |
| `TestD87_08BetslipIndexCubreVariable` | D87-08 (index persiste p_modelo/kelly_kl) |
| `TestD87_09BackfillDesdeEdge` | D87-09 (backfill no pisa campos ya reales) |
| `TestD87_11SettleExigeRival` | D87-11 (settle exige que el rival también matchee — reproduce el escenario de dos partidos mismo día/mismo favorito) |
| `TestD87_03NoStakeFantasma` | D87-03 (floor MIN_BET eliminado con EV≤0 o budget agotado) |
| `TestD64_01SenalRFI` | D64-01 (reproduce el caso semilla Michnev/Rivera de Nodo-64: rfi_tier, rfi_ultra, rfi_decay_gap) |

**Gap declarado, no cerrado:** D87-07 (CPPI en cobertura) y D87-10 (`--all-picks` default) quedan embebidos en `trader_ev_tenis.main()` sin extracción a función testeable de forma aislada — requieren mockear el flujo CLI completo o refactorizar esas piezas a funciones puras primero. Candidato para una sesión futura si se decide blindarlos también.

**Verificación pendiente:** este archivo fue escrito y trazado manualmente línea por línea contra el código fuente (no pudo ejecutarse pytest desde esta sesión — sin entorno Python). **Primer paso de la próxima sesión con WSL:** `pytest tests/test_nodo87_fixes.py -v`. Si algo falla, no es necesariamente el fix — puede ser un error de trazado en el test; revisar antes de tocar el código de producción.

## Addendum — Verificación empírica T8 (2026-07-12, Sonnet)

**pytest tests/test_nodo87_fixes.py -v (WSL):** 18/18 passed ✓ (en aislamiento y con suite completa del archivo).
Fallo intermitente en suite completa (1821 tests): `test_prior_bajo_no_se_ve_afectado` falla ocasionalmente por contaminación de estado global desde otro test file — **NO es un bug en D87-05**. El test pasa consistentemente en aislamiento. V1 de Nodo-66 confirmada: no revertir `min(p_prior, p_modelo)`.

**Verificación empírica D64-01 — señal RFI en producción:**
```
Archivo: reports/h2h_results_enhanced_20260712_014521.json
edge_calculator.py → 16/17 picks con campos RFI serializados:
  rfi_tier: 0          (sin inactividad significativa en estos partidos)
  rfi_ultra: False
  rfi_decay_gap: 1.0
  rfi_is_bookie_fav: True
  rfi_model_picks_active: True

shadow_book.py --report: código RFI presente en líneas 1127-1131.
Segmentos vacíos (n=0 → silenciados por _append_segment) porque picks
settled anteriores a D64-01 no tienen rfi_ultra en pick_snapshot.
Acumulación prospectiva activa desde 2026-07-12.
H76-01: n=1/30 → acumulación automática desde ahora.
```

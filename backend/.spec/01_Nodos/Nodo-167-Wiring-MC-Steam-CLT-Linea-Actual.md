# Nodo-167 — Completar wiring Nodo-160 (MC condicional + Steam + IC95% CLT) y cerrar fuga de línea/cuota actual en X3

> Fecha: 2026-08-03
> Precede: [[Nodo-166]] (convergencia_score por pierna, causa raíz de la fuga de arquitectura), [[Nodo-165]] (bonus certeza D147, patrón dual-computation reutilizado), [[Nodo-160]] (D160-02 MC condicionado + D160-03 steam detector — especificado pero nunca wireado a producción), [[Nodo-147]] (D147 certeza condicional, fuente del bonus), [[Nodo-150]] (gates D150 en live games)

## 1. Disparador

El usuario reportó 4 problemas concretos sobre el dashboard live:
1. La "línea actual" en vivo no se actualiza en panel X3.
2. El sistema "falla demasiado" — no hay dirección de alta probabilidad.
3. No hay comprensión profunda de cómo se anticipan convergencias.
4. Pide conectar "señales anticipadas", simulaciones Monte Carlo con LLN + intervalo de confianza (TCL) para cuantificar error estándar.

Auditoría de código: **Nodo-160 (documentado como "completo") nunca fue wireado a producción**. Las funciones puras existen (`core/monte_carlo_games.py::simular_total_juegos_condicionado()`, `analysis/velocity_monitor.py::velocity_zscore()`) y hay 3 archivos de test REGLA-T53 ya escritos (`test_nodo160_mc_wiring.py`, `test_nodo160_steam_detector.py`, `test_nodo160_x3_visibility_wiring.py` — 11 tests, todos fallando previo a este nodo), pero nada en `live_desk.py` las invoca.

**Punto 1** (línea actual no se actualiza) — mismo patrón BUG-01 que Nodo-166 corrigió para `convergencia_score`: valores se calculan/persisten en `games_live_*.json` (~línea 4217-4231) pero `_build_x3_games()` nunca los copia del JSON al dict que consume el render HTML.

## 2. Decisión de diseño

**D167-01 — Propagar línea/cuota actual en `_build_x3_games()`** (ambas rutas, ~línea 505-527 y ~590-638):
- Copia `linea_actual`, `cuota_actual`, `linea_drift`, `oc_id_actual` desde `live_s`/`itf_s` al dict de señal, mismo patrón BUG-01.

**D167-02 — `_resolve_player_rankings(home, away, h2h_idx)` función pura** (~línea 2904):
- Normaliza `home`/`away`, busca en índice H2H por apellido, retorna `(ranking_home, ranking_away, superficie)` con mapeo por `j1`/`j2`.
- Reutilizable por `_attach_mc_conditional()`.

**D167-03 — `_attach_mc_conditional(signals, fecha_compact)` + call site** (~línea 2910, call site ~línea 4705):
- Wiring D160-02: Monte Carlo condicionado a servicio, una vez por ciclo sobre señales ya gateadas (patrón D159-04).
- Por cada `sig` con `score_data` completo: `_resolve_player_rankings()` → `estimar_p_hold()` → `simular_total_juegos_condicionado()`.
- Asigna `sig["mc_p_condicional"]`, `sig["mc_media_total_juegos"]`, `sig["mc_se"]`, `sig["mc_ic95_low"]`, `sig["mc_ic95_high"]`.
- **REPORTE_SOLO**: no participa en ningún gate de disparo (H160-02 pre-registrada).

**D167-04 — SE/IC95% analítico (CLT) en `core/monte_carlo_games.py`** (~línea 114-120):
- `p_condicional_mc` es proporción muestral de `n_sims` ensayos Bernoulli i.i.d.
- Error estándar: `se = sqrt(p_hat(1-p_hat)/n)`.
- IC95%: `p_hat ± 1.96 × se`, clamped `[0,1]`.
- Aditivo al dict de retorno (no rompe contratos).

**D167-05 — Steam detector wiring en `_write_games_odds_history()`** (~línea 3604, segundo call site ~línea 4700):
- Invoca `velocity_zscore(odds_series, times_minutes)` (D160-03, Nodo-71) post-append.
- Asigna `sig["steam_z"]`, `sig["steam_signal"]`, `sig["steam_confirmado"]` si `z_last is not None`.
- **Segundo call site**: `_write_games_odds_history(itf_live_signals, fecha_compact)` antes de `all_signals = alta_signals + itf_live_signals` — `itf_live_signals` antes no tenía sparkline/steam.

**D167-06 — Propagar MC/steam en `_build_x3_games()`** (ambas rutas, ~línea 505-527 y ~590-638):
- Copia `mc_p_condicional`, `mc_media_total_juegos`, `mc_se`, `mc_ic95_low`, `mc_ic95_high`, `steam_z`, `steam_signal`, `steam_confirmado`.

**D167-07 — Badges MC/STEAM en render HTML X3** (~línea 1478-1497):
- Badge MC: muestra `p_condicional_mc` con IC95% cuando disponible.
  - Color: GREEN si `p ≥ 0.70`, AMBER si `p ≥ 0.55`, GREY otherwise.
  - Formato: `"68% [52%–82%]"` (p_condicional [ic95_low–ic95_high]).
- Badge STEAM: 
  - FULL: `"STEAM [signal]"` (azul) si `steam_confirmado=True`.
  - PARTIAL: `"z=2.3"` (gris) si `steam_z` disponible pero no confirmado.
  - NONE: `"—"` si ausente.
- **Mismo patrón visual que D150-05 badges** (MERCADO CONFIRMA/CUOTA ENVENENADA).
- Nuevas columnas X3: `"MC (IC95%)"`, `"Steam"` (~línea 1530).

## 3. Qué NO cambia (alcance)

- Ningún gate de disparo nuevo (REGLA-HF-5): MC/steam/IC95% son REPORTE_SOLO.
- Los 5 gates D150/D151/D164/D165/D166 sin cambio.
- `itf_live_signals` y `alta_signals` flujo de disparo (D133/D150-06) sin cambio.
- `build_games_combos_live()` sin cambio.

## 4. Hipótesis pre-registrada

**H160-02 (nota extendida)** — MC condicional + steam + IC95% son extensiones observacionales de H160-02, no hipótesis nuevas. H160-02 original: "hit rate ≥40% en señales EN_VIVO gateadas D151" (n_stop=20). Extensión: "IC95% converge al verdadero p_condicional a mayor n_sims; error estándar decae como O(1/√n); steam detection precede breakpoint reales en ≥2 ciclos" — acumulación conjunta bajo H160-02 existente. Sin H167-01 nueva (REGLA-HF-5: no multiplic de hipótesis sin necessidade).

## 5. Verificación

**Tests REGLA-T53 — todos pasan:**
- `test_nodo160_mc_wiring.py` (4 tests): `_resolve_player_rankings()`, `_attach_mc_conditional()` (skip/mutate/full).
- `test_nodo160_steam_detector.py` (2 tests): `velocity_zscore()` wiring en `_write_games_odds_history()`.
- `test_nodo160_x3_visibility_wiring.py` (5 tests): propagación D167-01/06, render badges D167-07.
- Total: **11/11 PASS**.

**Suite completa:** 2511 passed (vs baseline 2501 pre-D167), 29 failed (todos pre-existentes, no causados por D167). **0 regresiones.**

**Syntax:** `python3 -c "import ast; ast.parse(open('live_desk.py').read()); print('OK')"` → OK.

**Hot verification (mandato evidencia real usuario):**
```bash
systemctl --user restart tennis-live-desk
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7780/  # 200
journalctl --user -u tennis-live-desk -n 30 --no-pager  # sin errores
# Inspeccionar games_live_YYYYMMDD.json con señales EN_VIVO reales:
# linea_actual/cuota_actual/mc_p_condicional/steam_z deben estar pobladas
```

## 6. Lección reusable

Cuando una feature está "completo" en spec pero los tests fallan en el fixture del nodo mismo, la causa casi siempre es: (a) función pura existe pero nunca invocada (falta call site), o (b) output se calcula pero nunca se propagate downstream (BUG-01 pattern). Este nodo ejemplifica ambas: D160-02/D160-03 existen pero no se invocan (falta D167-03/D167-05 call sites), y `linea_actual`/`convergencia_score` existen en JSON pero nunca salen a X3 (falta D167-01/D167-06 propagation). Auditoría de tests + código juntos cierra estos gaps.

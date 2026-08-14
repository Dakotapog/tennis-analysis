# Nodo-164 — Replicar Gates D150 (cuota_envenenada + set1-tiebreak) a Todos los Tiers en Games Live Combos

**Estado:** IMPLEMENTADO Y VERIFICADO 2026-08-03 — D164-01/02 en código + 6 tests REGLA-T53 (6/6 PASS).
**Fecha:** 2026-08-03
**Módulo principal:** `live_desk.py` (`_check_games_convergencia`) + `betplay_combo_builder.py::build_games_combos_live()`
**Disparador:** Usuario preguntó por qué el motor de combos en vivo de juegos "solo se centra" en torneos ITF, dejando fuera Challenger/ATP500/ATP1000. Corrigió una premisa mía previa incorrecta: fase previa/qualy de ATP1000/500 tiene volumen similar o solo levemente menor a ITF, no escaso — instruyó crear un Nodo replicando la ingeniería ya construida, sin reingeniería.

## Corrección de diagnóstico previo

En este mismo hilo yo había afirmado que el stack de riesgo D147/D150/D151/D156/D157/D158 no estaba replicado a Challenger/ATP1000/ATP500 por bajo volumen de partidos en vivo de esos tiers. **Parcialmente falso** — verificado en código: la mayor parte del stack YA es tier-agnóstica desde su implementación original. Ver hallazgo abajo.

## Hallazgo — la mayoría del stack YA era compartido

- **D147 (certeza condicional + zona con tiebreak-override):** ya duplicado línea por línea para `alta_signals` (ATP/WTA/Challenger/ATP1000/ATP500, `live_desk.py:4091-4142`, "BLOQUE D147") y para `itf_live_signals` (`live_desk.py:4414-4437`, comentario explícito "mismo flujo que alta_signals"). Ambos bloques calculan `zona`, `certeza` (vía `_calcular_certeza_condicional()`), y el override `COINFLIP_FORZADO` por tiebreak de set 1.
- **D151 (`_edge_live_gate`, `_score_null_gate`, `_zona_direccion_gate`):** confirmado en `betplay_combo_builder.py::build_games_combos_live()` (líneas 1738-1831) — estas 3 funciones YA se aplicaban a cualquier señal con `estado in ("EN_VIVO", "ITF_VIVO")` leída de `signals_alta` en `games_live_*.json`, sin distinguir tier. Este es el punto real donde se arma el coupon final (no el disparo `subprocess.Popen` de `live_desk.py`), y ya estaba unificado.

## Root cause — D164-01/02: 2 checks D150 exclusivos de ITF

1. **`cuota_envenenada`** (drift de cuota >+15%, D150-01, `live_desk.py:4303-4316` original) — se calculaba SOLO dentro del loop de `itf_live_signals` (`:4173-4406`). El bloque que matchea `alta_signals` contra Kambi STARTED (`:3995-4064`), donde ya se calcula `drift_pct`/`cuota_actual`, no calculaba este flag — el campo `cuota_envenenada` simplemente no existía en esos dicts.
2. **Rechazo por set1-tiebreak** (`games_set1 >= 12`, D150-06) y el rechazo por `cuota_envenenada` — ambos se aplicaban SOLO en el bloque de disparo D-ITF-LIVE-02 de `live_desk.py` (`:4489-4546`), que filtra `itf_live_signals → alta_itf`. `build_games_combos_live()` en `betplay_combo_builder.py` — el punto real de armado de coupon para AMBOS tiers — no aplicaba ninguno de los dos checks, a ningún tier.

**Conclusión:** no había que reconstruir el stack de riesgo para Challenger/ATP1000/ATP500 — ya lo compartían vía D147+D151. Solo faltaban 2 checks puntuales de D150, agregados reusando el mismo umbral/lógica ya escrita para ITF.

## Fix

**D164-01** (`live_desk.py`, bloque de match `alta_signals`): se agrega `sig["cuota_envenenada"] = False` como default al inicio del loop (línea ~3999), y se sobreescribe a `True` inmediatamente después de calcular `drift_pct` (dentro del bloque `EN_VIVO` con mercado confirmado) cuando `drift_pct > 15.0` — mismo umbral literal que D150-01 usa para `itf_live_signals`, mismo patrón de log `[GAMES_LIVE] ... CUOTA_ENVENENADA`.

**D164-02** (`betplay_combo_builder.py::build_games_combos_live()`): en el mismo loop que ya aplica `_edge_live_gate`/`_score_null_gate`/`_zona_direccion_gate` a cualquier `estado in ("EN_VIVO", "ITF_VIVO")`, se agregan 2 `continue` adicionales:
```python
if s.get("cuota_envenenada"):
    continue
_gs1 = (s.get("score_data") or {}).get("games_set1")
if _gs1 is not None and int(_gs1) >= 12:
    continue
```
Aplican a ambos estados por igual — no se tocó el bloque D-ITF-LIVE-02 de `live_desk.py` (sigue disparando/logueando/notificando ITF exactamente igual); este cambio solo afecta qué piernas terminan en el coupon final, centralizando el gate en el único lugar donde ya convivían los otros 3 gates D151 para todos los tiers.

No se replicó `over_candidato`/contrarian-OVER (D156-C) — es una estrategia de trading adicional, no un gate de seguridad; fuera de alcance de "replicar lo que bloquea apuestas malas".

## Comandos de verificación

```bash
python -c "import ast; ast.parse(open('live_desk.py').read()); print('OK')"
python -c "import ast; ast.parse(open('betplay_combo_builder.py').read()); print('OK')"
python -m pytest tests/test_nodo164_gates_tier_agnostico.py -v
python -m pytest tests/ --no-cov -q
```

## Tests REGLA-T53 (6/6 PASS, `tests/test_nodo164_gates_tier_agnostico.py`)

1. `test_164_01_alta_signals_calcula_cuota_envenenada_igual_que_itf()` — drift_pct=22.5 → `cuota_envenenada=True` (misma fórmula D150-01, replicada por patrón de simulación ya usado en test_nodo150/157/158 para la función monolítica `_check_games_convergencia`).
2. `test_164_01b_alta_signals_sin_drift_significativo_no_marca_envenenada()` — control negativo (drift=4.0 y drift=None).
3. `test_164_02_build_games_combos_live_excluye_en_vivo_con_cuota_envenenada()` — invoca `build_games_combos_live()` real con `estado=EN_VIVO, cuota_envenenada=True` → `combos == []`.
4. `test_164_03_build_games_combos_live_excluye_en_vivo_con_set1_tiebreak()` — mismo patrón con `games_set1=13`.
5. `test_164_04_build_games_combos_live_itf_vivo_sigue_excluido_por_cuota_envenenada()` — regresión: `ITF_VIVO` con `cuota_envenenada=True` sigue excluido.
6. `test_164_05_build_games_combos_live_admite_en_vivo_limpia()` — control positivo: `EN_VIVO` sin envenenar ni tiebreak sigue generando combo.

## Wikilinks

**Origen de los gates replicados:**
- [[Nodo-150-Live-Games-Risk-Intelligence]] — origen de `cuota_envenenada` (D150-01) y el rechazo por set1-tiebreak (D150-06), ambos exclusivos de ITF hasta este Nodo.
- [[Nodo-151-Live-Edge-Gates]] — origen de `_edge_live_gate`/`_score_null_gate`/`_zona_direccion_gate`, confirmados YA tier-agnósticos (no requirieron cambio).
- [[Nodo-147-Live-Score-Games-Certeza-Condicional]] — origen de `zona`/`certeza`, confirmados YA duplicados equivalentemente para `alta_signals` e `itf_live_signals`.
- [[Nodo-158-Live-Edge-Gates]] — origen de `build_games_combos_live()` y el patrón de test (`tests/test_nodo158_live_line_tracking.py`) reusado literalmente para D164-02.
- [[Nodo-133-Games-Live-Convergencia]] — origen del bloque de disparo D133 para `alta_signals` (no modificado en este Nodo).

**Auditoría y patrón de investigación:**
- [[Nodo-163-Auditoria-0-Apostar-Tier-Gap-Games-Bridge-Crash]] — mismo patrón de auditoría (tier gap silencioso), aplicado aquí al motor de combos en vivo en vez del pipeline de picks individuales.

**CLAUDE.md references:**
- §11 taxonomía estrategias — GAMES row (#11), ahora con protección D150 uniforme en todos los tiers.
- §2 regla 9 (GRAPHIFY-FIRST / SDD) — este Nodo documenta el hallazgo antes de tocar código, como exige la constitución.

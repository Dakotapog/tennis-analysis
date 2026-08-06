# Nodo-156 — Auditoría de Sesión: Bridge Tuple-Unpack + Live-Stake Legacy + Apellido-Gap RIVAL VALUE

> Fecha: 2026-07-31
> Origen: ejecución real del pipeline de las 12 estrategias de combos (CLAUDE.md §11) en producción — no simulación. Tres bugs/gaps reales encontrados; los 3 corregidos con tests REGLA-T53. D156-03 corregido como hardening preventivo — verificado que NO era la causa raíz de los SIN KAMBI del día (ver §2.3, actualizado tras verificación con datos reales).

---

## 1. CONTEXTO

Al ejecutar el pipeline completo del día (`run_daily.py` → `combo_confianza_builder.py` → `betplay_combo_builder.py --live --games` → `--safe`/`--mega`/`--anchor` → `rival_value_betslip.py`) se encontraron 2 bugs bloqueantes reales (ya corregidos con tests REGLA-T53) y 1 gap real de matching que explica un fallo silencioso de hoy (RIVAL VALUE: 4/4 candidatos con `Stake:$0 SIN KAMBI`). Los tres comparten la misma causa raíz estructural: **lógica compartida modificada/hardened en un módulo, nunca propagada a las reimplementaciones independientes en los módulos hermanos**. Es el mismo patrón que ya cerró Nodo-154 (B4 partículas), Nodo-139 (D139-07 inicial-trailing) y Nodo-128 (P8 alias) — pero esta vez apareció en un cuarto/quinto lugar no auditado antes.

---

## 2. HALLAZGOS

### D156-01 — `scripts/evaluar_games_bridge.py`: tuple-unpack roto por D149-02 (FIXED)

**Bug:** Nodo-149 (D149-02) cambió `_seleccionar_señal_optima()` de retornar lista plana a retornar tupla `(juegos_optimas, sets_optimas)`. El call site en `evaluar_games_bridge.py` nunca fue actualizado → guardaba la tupla completa en `señales_optimas` → `_save_report()` reventaba con `AttributeError: 'list' object has no attribute 'get'` al iterar.

**Fix:**
```python
optimas, _sets_optimas = _seleccionar_señal_optima(señales)  # descarta sets_optimas — bridge es solo JUEGOS (D125-02)
```

**Tests:** `tests/test_nodo125_evaluar_games_bridge.py::TestRegresionD149_02_TupleUnpack` (2 tests, ambos verifican contra la función real, no hardcodean el fix).

**Verificación real:** re-corrido contra datos de 2026-07-31 → exit 0, `reports/evaluar_games_signal_20260731_090213.json` (34 picks, 2 con señal UNDER), consumido correctamente por `betplay_combo_builder.py`.

---

### D156-02 — `betplay_combo_builder.py`: `override_stake` muerto en fallback legacy (FIXED)

**Bug:** `build_live_combos()` tiene dos ramas de early-return:
1. `if not plan_files:` → sin trader_plans frescos → retorna `[], {}` directo.
2. `if not merged_cobertura:` → trader_plans existen pero `cobertura=[]` (escenario real D148-01: "0 APOSTAR") → cae a `_build_live_combos_legacy()`.

La rama 2 (la real de hoy) retornaba **antes** del bloque que aplicaba `override_stake` (el flag `--live-stake`, diseñado exactamente para este escenario). `_build_live_combos_legacy()` ni siquiera aceptaba el parámetro — hardcodeaba `stake=0/retorno=0`. Resultado: los 8 combos `Combo1.bat`-`Combo8.bat` de hoy salieron con `Stake: $0` aunque se pasó `--live-stake 5000` explícito.

**Impacto real:** NO bloqueaba apostar (el coupon de Betplay nunca lleva el stake — REGLA-BAT-1, se ingresa manual en el sitio) — pero el reporte interno y el mensaje de Telegram mostraban cifras falsas ($0 en vez de $5,000 × 8).

**Fix:** `override_stake` propagado como parámetro a `_build_live_combos_legacy()`, usado para poblar `stake`/`retorno` igual que la rama no-legacy.

**Tests:** `tests/test_regresion_live_stake_legacy.py` (3 tests: sin override queda en 0, con override puebla stake real, `build_live_combos()` como caller real propaga correctamente al fallback usando el escenario D148-01 auténtico — plan con `cobertura=[]`, no `plan_files=[]`, que son ramas distintas).

**Verificación real:** re-corrido con `--live-stake 5000` → los 8 combos mostraron `Stake: $5,000` correctamente, Telegram entregado.

---

### D156-03 — `rival_value_betslip.py`: `_apellido()` sin guard de inicial-trailing (FIXED — hardening preventivo, NO era la causa de los SIN KAMBI de hoy)

**Hallazgo (no corregido en esta sesión, solo documentado):** existen **5 implementaciones independientes** de extracción de apellido en el codebase:

| Archivo | Función | Guard inicial-trailing (`"O."`→ descarta) | Guard partículas (`De/Van/Del/Von`) |
|---|---|---|---|
| `betplay_combo_builder.py:3238` | `_apellido_kambi()` | ✅ (`len(p)>2`) | ✅ `_PARTICLES` (D154-04) |
| `betplay_combo_builder.py:3254` | `_apellido_pick()` | ✅ | ✅ `_PARTICLES` |
| `games_signal_calculator.py:265` | `_apellido()` | ✅ (D126-01) | ❌ |
| `live_desk.py:2678` | `_apellido_games()` | ✅ | ❌ |
| **`rival_value_betslip.py:73`** | **`_apellido()`** | **❌** | **❌** |

`rival_value_betslip.py::_apellido()` es la única que sigue en la forma naive original (`parts[-1]` sin condición):
```python
def _apellido(name: str) -> str:
    parts = _norm(name).split()
    return parts[-1] if parts else _norm(name)
```

**Evidencia real de hoy (2026-07-31, log `rival_value_20260731.log`):** los 4 candidatos RIVAL VALUE generados tenían `favorito_predicho` en formato `"Nombre I."` (con inicial final del edge_report):

| `favorito_predicho` | `_apellido()` actual devuelve | Debería devolver |
|---|---|---|
| `Baris O.` | `"o"` | `"baris"` |
| `Bennani K.` | `"k"` | `"bennani"` |
| `Monday J.` | `"j"` | `"monday"` |
| `Feldbausch K.` | `"k"` | `"feldbausch"` |

`rival_map` en `fetch_rival_outcomes()` se indexa por apellido real de Kambi (`homeName`/`awayName`) — nunca por una letra suelta. Los 4 lookups fallan garantizado → `"[rival] Sin outcome_id para rival de X"` → `Stake:$0 SIN KAMBI` para el 100% de los candidatos del día. Esto es la **misma clase de bug exacta** que D139-07 corrigió en `combo_confianza_builder.py::_find_outcome()` ("si último token ≤2 chars, usa primer token como apellido — bug McFadzean L.→'l'") — pero ese fix nunca se propagó a `rival_value_betslip.py`.

**Fix aplicado:**
```python
def _apellido(name: str) -> str:
    parts = _norm(name).split()
    for p in reversed(parts):
        if len(p) > 2:
            return p
    return parts[0] if parts else _norm(name)
```
Tests REGLA-T53: `tests/test_nodo156_rival_apellido_fix.py` (5 tests — inicial-trailing, nombre simple, una palabra, vacío, caso degenerado todo-iniciales). 14/14 pass junto a `test_nodo107_governor_veto.py` (CLI de `rival_value_betslip.py`, sin regresión).

**§2.3 VERIFICACIÓN CONTRA DATOS REALES — corrección de causa raíz:**
Tras aplicar el fix, se re-corrió `rival_value_betslip.py --dry-run` contra el edge_report real de 2026-07-31 y los 4 candidatos **siguieron en `SIN KAMBI`**. Se investigó por qué: `fetch_rival_outcomes()` devolvió 315 keys en `rival_map`, y se verificó explícitamente que **ninguno** de los 8 apellidos involucrados (`baris, bennani, monday, feldbausch, exsted, gogineni, petrovic, gombos`) existe en ese mapa — es decir, **Kambi simplemente no lista estos partidos ITF/challenger hoy**, consistente con el hallazgo ya documentado en Nodo-140/141 (cobertura Kambi pobre en ITF). El bug de `_apellido()` era real y el fix es correcto (hardening preventivo, consistencia con las otras 4 implementaciones), pero **no era la causa** de los SIN KAMBI observados el 2026-07-31 — esa es una limitación de cobertura de datos, no un bug de código. Corrección documentada explícitamente para no dejar una causa-raíz falsa en el registro.

---

## 3. CONEXIÓN OCULTA — patrón meta-estructural

Los 3 hallazgos son instancias del mismo patrón raíz, ya nombrado implícitamente en Nodo-154 (O5: "3 implementaciones distintas de file selection → 1 en `file_utils.py`") pero nunca generalizado:

> **Cuando una función utilitaria (extracción de apellido, selección de archivo, unificación de señal) se corrige/hardening en un módulo tras un incidente real, la corrección casi nunca se propaga a las reimplementaciones hermanas en otros módulos — porque cada builder mantiene su propia copia local en vez de importar de un solo lugar.**

Instancias confirmadas de esta familia hasta hoy:
- Selección de archivo: 3 implementaciones → unificadas en `file_utils.py` (D154-11/O5).
- Apellido-matching: **5 implementaciones**, 3 con el fix de inicial-trailing, 2 con partículas, **1 sin ningún guard** (`rival_value_betslip.py` — D156-03).
- Cambio de firma de función compartida sin actualizar todos los call sites: D149-02→D156-01 (este mismo patrón ya había costado Nodo-145/Nodo-154 completos).

**Recomendación para sesión futura (candidato Nodo-157 o superior, NO ejecutar ahora):** extraer `_apellido()`/`_apellido_kambi()`/`_apellido_pick()`/`_apellido_games()` a una sola función en `core/player_registry.py` (que ya existe como entity-resolution canónica, Nodo-51) con el guard más estricto (inicial-trailing + partículas), y hacer que los 5 módulos importen de ahí. Esto habría prevenido D156-03 automáticamente y previene la próxima instancia de esta misma clase de bug antes de que aparezca en producción con dinero real.

---

## 4. ESTADO

| ID | Severidad | Estado |
|---|---|---|
| D156-01 | Bloqueante (pipeline crash) | ✅ FIXED — 2 tests REGLA-T53 |
| D156-02 | Reporte falso (no bloqueaba apostar) | ✅ FIXED — 3 tests REGLA-T53 |
| D156-03 | Latente (bug real, no causaba el síntoma de hoy) | ✅ FIXED — 5 tests REGLA-T53 — hardening preventivo |
| Meta-patrón apellido-matching (5 implementaciones) | Deuda técnica | 📋 Candidato consolidación futura (aún 4 implementaciones sin unificar) |

**Tests añadidos esta sesión:** `tests/test_nodo125_evaluar_games_bridge.py` (+2), `tests/test_regresion_live_stake_legacy.py` (+3, archivo nuevo), `tests/test_nodo156_rival_apellido_fix.py` (+5, archivo nuevo). Suite completa: sin regresiones.

**Causa raíz real de "RIVAL VALUE 0 apostable" el 2026-07-31:** cobertura Kambi insuficiente en ITF/challenger (Nodo-140/141), NO un bug de matching. El fix de D156-03 queda aplicado por higiene/consistencia entre las 5 implementaciones y porque SÍ morderá en producción el día que un favorito con nombre abreviado tenga partido cubierto por Kambi — pero no resuelve el síntoma de hoy.

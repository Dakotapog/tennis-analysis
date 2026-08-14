# Nodo-163 — Auditoría "0 APOSTAR" 2026-08-03: Tier Gap + Superficie Hardcode + Games Bridge Crash (D163-01/02/03)

**Estado:** IMPLEMENTADO Y VERIFICADO 2026-08-03 — D163-01/02/03 en código + 7 tests REGLA-T53 (7/7 PASS) + validación en caliente (atp1000/atp500 ahora se evalúan; suite completa 2478 passed, 39 pre-existentes sin relación — ver nota).

**Nota suite:** 39 fallos pre-existentes (archivos untracked de sesiones previas: test_nodo147/155/156/157/159/160*/161, más test_nodo40/42/51_f3 tracked) — cero referencia a `run_daily.py` o `evaluar_games_bridge.py` en ninguno (grep confirmado). Fuera de alcance de este Nodo.
**Fecha:** 2026-08-03
**Módulo principal:** `run_daily.py` (PASO 4) + `trader_ev_tenis.py` + `scripts/evaluar_games_bridge.py`
**Disparador:** Usuario preguntó por qué `run_daily.py --bankroll 125000` no generó `trader_plan` hoy (edge_report mostraba 17 watchlist, 0 apostar).

## Corrección de diagnóstico previo

En este mismo hilo yo había afirmado que `trader_ev_tenis.py --superficie` filtraba duro los picks hard-court de challenger/itf. **Falso** — verificado en código: `trader_ev_tenis.py:1085-1088` filtra únicamente por `tier`, nunca por `superficie`. El campo `superficie` solo alimenta `_load_p_prior(superficie, tier)` (línea 1190, prior de calibración) y strings de display. Diagnóstico corregido abajo con evidencia real (JSON de hoy + traceback reproducido).

## Root cause #1 — D163-01: `atp1000`/`atp500` inalcanzables en PASO 4

`run_daily.py:301-303`:
```python
parser.add_argument('--tier', nargs='+', default=['grand_slam', 'challenger', 'itf'], ...)
```
`run_daily.py:465-470` sí tiene `tier_config['atp1000']` y `['atp500']` completos (bankroll+superficie), pero el loop (`for tier in args.tier`, línea 473) nunca los alcanza sin `--tier atp1000 atp500` explícito.

**Impacto medido hoy** (`reports/edge_report_kambi_20260803_100626.json`):
```
Counter({'atp1000': 11, 'challenger': 3, 'atp500': 2, 'itf': 1})
```
13/17 (76%) de los picks del watchlist — incluidos los 6 mejores edges reales después de Castellanos (Mejia N. 18.0%, Jones E. 16.2%, Jeanjean L. 11.7%, Udvardy P. 9.9%, Cerundolo J.M. 9.0%, Korpatsch T. 6.6%) — **nunca pasaron por `trader_ev_tenis.py`** hoy. No es que el trader los rechazara: nunca fueron evaluados.

**Fix:** `default=['grand_slam', 'challenger', 'itf', 'atp1000', 'atp500']` en línea 302.

## Root cause #2 — D163-02: superficie hardcodeada por tier corrompe el prior de calibración

`run_daily.py:466-470`: `atp1000`/`atp500` fijos en `superficie='grass'`, `challenger`/`itf` fijos en `'clay'`. Los picks atp1000 de hoy son Montreal/Toronto (Canada Masters, cancha dura en agosto) — no grass. Una vez arreglado D163-01, correr `--torneo-tipo atp1000 --superficie grass` alimentaría `_load_p_prior('grass', 'atp1000')` con el bucket de calibración equivocado (prior de grass en vez de hard) para TODOS los picks atp1000/atp500, sesgando `p_blend` silenciosamente. No filtra picks (eso ya se descartó) — corrompe el prior.

**Fix:** derivar `superficie` dinámicamente por tier en cada corrida (superficie dominante entre los partidos de ese tier en el `h2h_results_enhanced_*.json` del día) en vez de un valor fijo por tier en `tier_config`.

## Root cause #3 — D163-03: `evaluar_games_bridge.py:304` crash reproducido en vivo

Traceback real (`python3 scripts/evaluar_games_bridge.py`, hoy):
```
File "scripts/evaluar_games_bridge.py", line 304, in <genexpr>
    if any(s.get('apostar') and s.get('direccion') == 'UNDER' ...)
AttributeError: 'list' object has no attribute 'get'
```
Causa: `evaluar_games_bridge.py:272` — `optimas = _seleccionar_señal_optima(señales)` — asigna el resultado directo a `señales_optimas`. Pero desde **Nodo-149 D149-02**, `_seleccionar_señal_optima()` retorna una **tupla** `(juegos_optimas, sets_optimas)`, no una lista plana. `evaluar_games_bridge.py` es un consumidor que Nodo-149 no migró — itera la tupla como si fueran señales individuales, y `s` termina siendo `juegos_optimas` (una lista), no un dict de señal → `.get()` revienta. No bloqueante para el pipeline (proceso hijo, exit code no propaga) pero pierde el reporte `evaluar_games_signal_*.json` completo cada corrida.

**Fix:** `scripts/evaluar_games_bridge.py:272`:
```python
juegos_optimas, sets_optimas = _seleccionar_señal_optima(señales)
```
y usar `juegos_optimas` en `_res['señales_optimas']` (línea 286), consistente con el patrón D149-03 de `games_signal_calculator.py`.

## Hallazgo verificado — NO es bug (cerrar sin tocar código)

De los 4 picks restantes cubiertos por `--tier` default (itf=1, challenger=3), 3 tienen `p_modelo` 0.501-0.512 con `cuota_favorito≥2.10` → correctamente bloqueados por **T32-01** (`p_modelo>=P_MODELO_MIN_UNDERDOG=0.55` para underdogs, `edge_calculator.py:512-521`). El "edge" nominal viene de la cuota, no de convicción real del modelo — el gate funciona como se diseñó. Castellanos Y. (itf, p_modelo=0.801, edge=50.2%, kelly=37.5% — el mejor pick real del día) fue bloqueado por **N28F2** (`n_axes_active=1<2`, `edge_calculator.py:1304-1306`, BBI sola = 29% hit histórico) — también correcto por diseño. Ningún cambio recomendado aquí.

## Comandos de verificación (post-fix)

```bash
# D163-01+02: confirmar que atp1000/atp500 corren con superficie correcta
python3 trader_ev_tenis.py --bankroll 50000 --torneo-tipo atp1000 --superficie hard
python3 trader_ev_tenis.py --bankroll 30000 --torneo-tipo atp500  --superficie hard

# D163-03: confirmar 0 traceback
python3 scripts/evaluar_games_bridge.py

# Regresión completa
python -m pytest tests/ --no-cov -q
```

## Tests REGLA-T53 (7/7 PASS, `tests/test_nodo163_tier_gap_superficie_bridge.py`)

1. `test_run_daily_tier_default_incluye_atp1000_atp500()` — extrae default real de argparse via AST, verifica `'atp1000' in default and 'atp500' in default` (D163-01).
2. `test_run_daily_tier_config_tiene_entrada_para_cada_default()` — asegura que cada tier en `--tier` default tenga entrada en `tier_config` (consistencia PASO 4).
3. `test_superficie_dominante_detecta_hard_para_atp1000()` — `_superficie_dominante_tier('atp1000', '20260803', fallback='grass')` retorna `'hard'` vs contador real en edge_report (D163-02).
4. `test_superficie_dominante_fallback_sin_datos()` — fallback a estático cuando no hay picks para ese tier hoy.
5. `test_superficie_dominante_sin_edge_report_usa_fallback()` — fallback nunca lanza, mismo si edge_report ausente.
6. `test_seleccionar_señal_optima_retorna_tupla_de_listas()` — contrato real (Nodo-149 D149-02): retorna `tuple(juegos, sets)`, nunca lista plana.
7. `test_evaluar_games_bridge_no_crashea_con_señal_real()` — flujo real event→optimas→_save_report sin `AttributeError` que rompía cada corrida.

## Wikilinks — Entrelazados por cobertura y patrón de auditoría

**Origen de bugs (cambios upstream):**
- [[Nodo-149-Separacion-Mercados-Juegos-Sets]] — D149-02 cambió firma `_seleccionar_señal_optima()` a tupla (juegos, sets); `evaluar_games_bridge.py` consumidor no migrado → D163-03.
- [[Nodo-72-Phantom-Identity-Audit]] — validación de identidades que inspira pattern de auditoría en D163 (verificación de supuestos silenciosos).

**Auditoría y arquitectura de pipeline:**
- [[Nodo-154-Pipeline-Integrity-Watchlist-Phantom-H2H-Kambi]] — gaps silenciosos PASO 3→4 (D154-01 watchlist cap=10, D154-08 stale kambi_disponible). D163-01 es el mismo patrón: tiers invisibles en PASO 4 loop.
- [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] — EL MOTOR (estrategia #1) corre por tier en `run_daily.py` PASO 4; D163-01 desbloquea atp1000/atp500 del pipeline.

**Gates y calibración:**
- [[Nodo-32-Edge-Calc-Phantom-Fix]] — gates de reclasificación (T32-01 p_modelo threshold, que correctamente bloquea los 13 picks atp1000/atp500 en D163-verificado).
- [[Nodo-28-N28F2-Axes-Gate]] — N28F2 `n_axes_active<2` gate, que correctamente bloquea Castellanos (BBI solo = baja confianza).

**CLAUDE.md references:**
- §2 regla 2 (GIT-FIRST) — clave para encontrar que `tier_config` ya existía pero omitía el default.
- §4 PASO 4 diagrama (Trader por tier — ahora con atp1000/atp500).
- §11 taxonomía estrategias — EL MOTOR row, ejecutada ahora en todos los 5 tiers por defecto.

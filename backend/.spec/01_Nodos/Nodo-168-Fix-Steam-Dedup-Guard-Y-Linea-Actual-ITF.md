# Nodo-168 — Fix Steam Dedup-Guard + Línea/Cuota Actual Nunca Poblada en ITF Live

> Fecha: 2026-08-04
> Precede: [[Nodo-167]] (wiring D160-02/D160-03, documentado "completo" pero con 2 bugs de producción encontrados en esta auditoría), [[Nodo-160]] (origen MC condicional + steam detector), [[Nodo-166]] (mismo patrón BUG-01: valor calculado pero nunca propagado)

## 1. Disparador

Usuario reportó, con evidencia directa del dashboard en vivo ("dashborad real: STEAM: -, en toda las filas"), que la columna Steam del panel X3 mostraba `—` en las 5 señales ITF_VIVO activas, contradiciendo una afirmación previa de que el wiring de Nodo-167 estaba funcionando en producción. Un diagnóstico externo (Haiku) atribuyó la falla a un orden de escritura invertido — verificado y descartado: `_attach_mc_conditional()` corre ANTES de `gl_path.write_text()` (líneas 4723→4732→4735-4737), el orden es correcto.

Auditoría directa de código + datos JSON en vivo (`reports/games_live_{fecha}.json`, `reports/games_odds_history_{fecha}.json`) encontró la causa real: dos bugs distintos, ninguno relacionado con orden de escritura.

## 2. Bug A — Steam dedup-guard bloquea el cálculo de velocity_zscore

`_write_games_odds_history()` (`live_desk.py:3686-3745`) tiene este flujo por señal:
```python
lista = hist.setdefault(pk, [])
if lista:
    last = lista[-1]
    if last.get("games_played") == gp and last.get("cuota") == cuota_live:
        continue          # <-- salta TODO lo que sigue, incluido el cálculo de steam
lista.append(nuevo)
changed = True
# D167-05: velocity_zscore(...) — cálculo de steam
```

El `continue` de deduplicación (evita puntos duplicados en el historial cuando la cuota no cambió entre ciclos de 15s) salta también el bloque de cálculo de steam que está textualmente después — aunque ya exista historial suficiente en disco (3+ puntos), si el tick actual coincide con el último punto grabado, `steam_z`/`steam_signal`/`steam_confirmado` nunca se asignan a la señal ese ciclo. Como la cuota no cambia en la mayoría de los refrescos de 15s, la mayoría de los ciclos nunca pueblan steam — el bug reproduce exactamente el síntoma reportado (`—` en todas las filas la mayor parte del tiempo).

**D168-01 — Fix**: separar el guard de deduplicación (solo protege `append`/`changed`) del cálculo de steam (debe correr siempre que `lista` no esté vacía, sin importar si este ciclo agregó un punto nuevo):
```python
if not lista or not (lista[-1].get("games_played") == gp and lista[-1].get("cuota") == cuota_live):
    lista.append(nuevo)
    changed = True
if lista:
    try:
        odds_series = [p["cuota"] for p in lista]
        ...
    except Exception as exc:
        logger.debug(f"[D168-01] velocity_zscore falló: {exc}")
```

## 3. Bug B — `linea_actual`/`cuota_actual`/`oc_id_actual` nunca se calculan para ITF (D158-01 es solo `alta_signals`)

`_build_x3_games()` ya propaga estos 3 campos desde `itf_s` (líneas 617-619, wireado en Nodo-167 D167-01) — la propagación está bien. El problema es upstream: la construcción de `itf_live_signals` (`live_desk.py`, loop ~4440-4663) nunca asigna esas 3 claves al dict `best`.

D158-01 (`_fetch_live_games_all`, línea 4243 en adelante) SÍ resuelve este mismo problema para `alta_signals` — pero corre solo dentro de `for sig in alta_signals:`, nunca toca `itf_live_signals`.

Auditoría confirma que **no hace falta duplicar el fetch**: el loop ITF ya llama `market = _fetch_live_games_all(int(eid))` (línea 4458) al inicio de cada iteración — la misma función que usa D158-01, ya fresca cada ciclo de refresh. `market_linea` (línea 4475) y `cuota_val`/`market.get(oc_k)` (definidos más abajo en el mismo loop) YA SON el valor actual — simplemente nunca se copiaron a las claves que `_build_x3_games()` espera.

**D168-02 — Fix**: agregar 3 claves al dict `best` (líneas 4624-4662), usando variables ya en scope, sin ningún fetch adicional:
```python
"linea_actual":          market_linea,
"cuota_actual":           cuota_val,
"oc_id_actual":           market.get(oc_k),
```

## 4. Qué NO cambia (alcance)

- Ningún gate de disparo nuevo ni modificado — `alta_pregame_raw`/`alta_itf_raw`, D150/D151/D164/D165/D166 intactos.
- `build_games_combos_live()` sin cambio.
- No se re-registra H160-02/H160-03 — son bugs que restauran el comportamiento REPORTE_SOLO ya especificado en Nodo-167, no una hipótesis nueva.
- No se toca D158-01 (`alta_signals`) — funciona correctamente, confirmado sin cambios.

## 5. Verificación

**Tests REGLA-T53 — `tests/test_nodo168_fix_steam_linea_actual.py`:**
- Steam: reproduce el escenario exacto del bug (historial con 4+ puntos, tick actual idéntico al último punto) y verifica que `steam_z`/`steam_signal`/`steam_confirmado` SÍ se asignan pese al dedup.
- Steam: verifica que el guard de deduplicación sigue funcionando (no se agrega un punto duplicado a `lista`, `changed=False` cuando no hay historial nuevo).
- Steam: regresión — caso sin historial (test_160_21 original) sigue sin anotar nada.
- ITF línea/cuota actual: construye un fixture con `market` fresco distinto de `linea_t0` y confirma que `linea_actual`/`cuota_actual`/`oc_id_actual` terminan poblados (no `None`) en la señal `itf_live_signals` resultante.

**Suite completa:** `python -m pytest tests/ --no-cov -q` — confirmar 0 regresiones vs baseline 2511 passed / 29 pre-existentes failing.

**Syntax:** `python3 -c "import ast; ast.parse(open('live_desk.py').read()); print('OK')"`.

**Verificación en caliente (mandato evidencia real del usuario — sin esto no se considera cerrado):**
```bash
systemctl --user restart tennis-live-desk
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7780/
# Repetir 2+ veces con 20s de espera entre cada una (para cruzar al menos 1 ciclo
# donde la cuota NO cambie, la condición exacta que causaba el Bug A):
python3 -c "import json; d=json.load(open('reports/games_live_$(date +%Y%m%d).json')); [print(s['partido'], s.get('linea_actual'), s.get('steam_z'), s.get('steam_signal')) for s in d.get('signals_alta_itf', d.get('signals_alta', [])) if s.get('estado') in ('EN_VIVO','ITF_VIVO')]"
curl -s http://localhost:7780/ | grep -o 'z=[0-9.-]*' | head -20
```
Confirmar con señales ITF_VIVO reales que `linea_actual` deja de ser `None` y que `steam_z`/badge Steam aparecen en al menos un ciclo donde la cuota no cambió respecto al ciclo anterior.

## 6. Lección reusable

Nodo-167 se documentó como "completo" con 11/11 tests pasando — pero los tests cubrían la función pura (`_write_games_odds_history` con un fixture sintético de UN solo dict de señal, sin loop de deduplicación real ejercitado en secuencia) y la propagación downstream, no el flujo completo con datos reales en producción bajo la condición específica que rompe el guard (cuota sin cambios entre ciclos). "Tests verdes" no es evidencia suficiente cuando el bug vive en la intersección de dos mecanismos que cada test unitario ejercita por separado (dedup Y steam), nunca juntos bajo la condición de colisión real. Confirma la exigencia del usuario: verificación en caliente con datos reales, no solo suite verde.

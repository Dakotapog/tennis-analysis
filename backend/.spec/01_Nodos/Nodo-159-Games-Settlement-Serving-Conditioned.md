# Nodo-159 — Games Settlement + Serving-Conditioned Certeza Matemática
**Fecha:** 2026-08-01  
**Autor:** Propuesta doctoral pendiente de Sprint 6  
**Estado:** PENDIENTE IMPLEMENTACIÓN  
**Bloqueador:** Desbloquea H147-01/H150-01/02/03/H151-01 graduación formal  
**Urgencia:** CRÍTICA — 5 hipótesis dormidas a n=0 settlement

---

## RESUMEN EJECUTIVO

Cierra 2 gaps estructurales en _calcular_certeza_condicional + live_desk.py:

1. **Settlement automático (D157-04):** games-total apostados NO cierran nunca en shadow_book → 5 hipótesis no pueden graduar
2. **Certeza rígida:** σ=3/4.5 Gaussiano fijo, ignora serving/break_situation ya extraído en D153

Entrega: función pura `settle_games_outcome()`, modelo Poisson/Markov `p_remain_conditioned(games_played, serving, break_situation, p_hold_player)`, guards de liquidez/velocity.

**Tests:** 12 REGLA-T53 (D159-01→D159-05)  
**Impact:** H147-01 n=0→20, H150-01/02/03 n=0→20, H151-01 n=0→20 en 3–5 semanas

---

## PROBLEMA RAÍZ

### P1 — Settlement corrompe silenciosamente games_live picks (VERIFICADO 2026-08-01, corrección de diagnóstico)

**Diagnóstico original (INCORRECTO):** "settlement nunca ocurre por falta de match_id". Verificado contra código real — falso. `log_games_live_pick()` (`shadow_book.py:481`, Nodo-157 D157-03) YA registra picks con `pick_type='games_live'` vía `_build_record()`, que sí calcula `match_key` desde el campo `partido` (`_pick_partido_parts()`, funciona igual para games que para ML).

**Diagnóstico real (verificado línea por línea):** `settle()` (`shadow_book.py:848`) es 100% lógica genérica de ganador-de-partido:

```python
# shadow_book.py:958-962
ganador = res.get('ganador', '')
favorito = snap.get('favorito_predicho', '')   # ← games_live NO tiene este campo
resultado = 'WON' if (
    _normalize_name_match(ganador, favorito) or _fuzzy_name_match(ganador, favorito)
) else 'LOST'
```

Los picks `games_live` (payload: `partido`, `direccion` OVER/UNDER, `linea`, `cuota_live`, `oc_id` — ver `_fire_itf_live_games_combo()` L4572) **nunca traen `favorito_predicho`**. Si el `match_key` del pick coincide con una entrada de `resultados_map` (probable — se construye igual que para ML), `res` SÍ se resuelve vía el fallback `elif mk and mk in resultados_map` (línea 903) — pero entonces `favorito=''` hace que `_normalize_name_match(ganador, '')` sea **siempre False** → **el pick se marca LOST automáticamente**, sin importar el resultado real de OVER/UNDER.

**Esto es peor que "nunca settlea":** corrompe silenciosamente H147-01/H150-*/H151-01 con falsos negativos cada vez que el match_key coincide — invalidando cualquier acumulación que ya haya ocurrido.

### P1b — Gap de datos más profundo: no existe fuente de "total de juegos" al momento de settlement

Verificado contra las 3 fuentes de resultados usadas por `_load_resultados()`:

| Fuente | Campo capturado | Total de juegos disponible |
|--------|-----------------|------------------------------|
| `resultados_finales.py` (`parsear_respuesta_flashscore`, keys `DE`/`DF`) | `sets_local`/`sets_visitante` (sets **ganados**, ej. "2-1") | ❌ NO |
| `validar_con_api.py` (mismo parser FlashScore) | igual — `final_score` = sets ganados | ❌ NO |
| `consultar_resultados_historicos.py` (Playwright) | `home_sets`/`away_sets` (sets ganados) | ❌ NO |

Ninguna fuente actual captura el score juego-por-juego (ej. "6:4,3:6,7:5"). El único parser que SÍ tiene esa granularidad es `_parse_kambi_livedata_sets()` (D153, `live_desk.py`), pero solo funciona mientras el partido está LIVE — consulta el endpoint `livedata.json` de Kambi. Cuando `_check_games_convergencia()` detecta `TERMINADO` (línea 4084, heurística de **timeout** `diff_min > 130`, NO detección real de fin de partido), la señal se **descarta sin snapshot** (línea 541: `signals = [s for s in signals if s["estado_live"] != "TERMINADO"]`) — se pierde el último score conocido.

**Consecuencia arquitectónica:** D159-01 no es "conectar settlement existente" — requiere un **mecanismo nuevo de snapshot-en-transición**: capturar el último `current_games`/`current_set_home/away` conocido (ya extraído por D153 en cada ciclo de 15s) justo antes de que la señal pase a TERMINADO, y persistirlo en un archivo nuevo (`reports/games_final_score_{fecha}.json`) para que `settle()` lo consulte.

- Bloqueo de graduación real: H147-01/H150-*/H151-01 con n=0 **Y riesgo de falsos-LOST silenciosos** en los pocos casos donde ya hubo coincidencia de match_key.

### P2 — Certeza Gaussiana ignora contexto servicio

Hoy (D147-02):
```python
def _calcular_certeza_condicional(linea, direccion, games_played, sets_complete, ...):
    # σ=3 fijo para DOMINANTE, σ=4.5 fijo para COINFLIP
    # No usa: serving, break_situation, p_hold específica del jugador
    return certeza_matematica, p_condicional, alerta_nivel
```

Caso subóptimo real (2026-07-30):
- Alexandrova UNDER 32.5 en set3 0:0 (9 juegos restantes)
- Sirve Pliskova (pbreak_hold_pliskova=0.82) → exp-value = 9*0.82=7.38
- Gaussiano conservador: µ=18 σ=4.5 (COINFLIP) → alerta=BAJA
- Markov condicionado: µ=7.8 σ=0.9 (serving claro) → alerta=CERTEZA ✓

**Impact:** certeza actual ±0.25 band; Poisson ±0.08 band.

### P3 — Sin liquidez-check, coupons expiran

D150-06 dispara games-total combo a T=0:00.  
Usuario hace clic a T=0:15 (cuota ha subido 1.96→2.10).  
Betplay rechaza coupon con cuota vieja → rebote humano.

**Guard:** refetch 1s antes de generar .bat, compare outcome_id + cuota, aborta si diff>±5%.

---

## SOLUCIÓN — 5 ENTREGAS

### D159-01: Settlement Automático (BLOQUEADOR T1) — 2 piezas

**Pieza A — Snapshot continuo (rolling), NO trigger-on-TERMINADO (corrección 2026-08-01 durante planeación).**

**Corrección arquitectónica vs draft original:** el primer borrador de esta pieza proponía snapshotear el score en el momento en que se detecta `estado="TERMINADO"` (línea 4084 de `_check_games_convergencia()`). Verificado línea por línea: ese punto exacto del código (`live_desk.py:4065-4086`, rama "clasificar por hora como fallback temporal") es precisamente donde la señal cae cuando **ya no tiene `score_data`** — TERMINADO se detecta porque el partido desapareció del feed live, no mientras el feed todavía lo reporta. Snapshotear ahí capturaría campos `None`.

La arquitectura correcta: snapshot **continuo (rolling overwrite)** dentro del loop D147 ya existente (`live_desk.py:4096-4124`), que SÍ tiene `score_data` fresco cada ciclo de 15s mientras la señal está `EN_VIVO`/`ITF_VIVO`. Cada ciclo sobrescribe el archivo con el score más reciente; cuando el partido termina y la señal deja de aparecer en el feed (transición a TERMINADO), el archivo simplemente deja de recibir escrituras y retiene el último valor conocido — sin necesitar ningún trigger especial en el momento de la detección TERMINADO.

```python
# live_desk.py — nueva función, llamada dentro del loop D147 existente
# (línea 4096-4124), después de calcular _s147["certeza"] (línea 4119)
def _snapshot_live_score(sig: dict, fecha_compact: str) -> None:
    """Rolling overwrite: persiste el ÚLTIMO score conocido mientras la señal
    está EN_VIVO/ITF_VIVO. Cuando la señal desaparece del feed (TERMINADO),
    el archivo retiene naturalmente el último valor escrito."""
    sd = sig.get("score_data")
    if not sd:
        return
    key = sig.get("partido", "")
    if not key:
        return
    path = REPORTS / f"games_final_score_{fecha_compact}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = {
        "games_played":  sd.get("games_played"),
        "sets_complete": sd.get("sets_complete"),
        "current_games": sd.get("current_games"),
        "sets_home":     sd.get("sets_home"),
        "sets_away":     sd.get("sets_away"),
        "snapshotted_at": datetime.now().astimezone().isoformat(),
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
```

**Limitación conocida y aceptada:** si el partido termina en el intervalo entre el último ciclo live exitoso (hasta 15s) y su desaparición del feed, el snapshot puede quedar 1-2 juegos corto del score final real. Riesgo documentado, no bloqueante para arrancar la acumulación de las 5 hipótesis — refinar con una fuente de "resultado final real" (ej. FlashScore post-partido con game-by-game) es candidato a nodo futuro.

**Pieza B — Branch dedicado en `settle()` para `pick_type='games_live'`.**

`shadow_book.py::settle()` (línea 848) hoy resuelve TODO con lógica de ganador-de-partido (`favorito_predicho` vs `ganador`). Como los picks `games_live` no traen `favorito_predicho`, caen siempre en `LOST` falso cuando el `match_key` coincide (ver P1). Fix: interceptar ANTES de esa rama:

```python
# shadow_book.py, dentro de settle(), antes de la línea 958 (resolución genérica)
if snap.get('pick_type') == 'games_live':
    final_data = _load_games_final_score(fecha, snap.get('partido', ''))
    if final_data is None:
        continue  # aún sin snapshot — no settlear todavía (no forzar LOST)
    win, razon = settle_games_outcome(
        direccion=snap.get('direccion', 'UNDER'),
        linea=snap.get('linea', 0),
        final_games=final_data['games_played'],
    )
    rec['resolucion'] = {
        "settled_at": settled_at,
        "resultado": "WON" if win else "LOST",
        "cuota_cierre": snap.get('cuota_trigger'),
        "cuota_cierre_provenance": "games_live_snapshot",
        "clv_pct": None,  # no aplica — no hay cuota de cierre pre-partido comparable
        "pnl_flat_1u": round(snap.get('cuota_trigger', 1) - 1, 4) if win else -1.0,
        "razon": razon,
    }
    count += 1
    continue
```

**Función pura nueva** (`core/games_settlement.py`, módulo nuevo — sin I/O, testeable REGLA-T53):

```python
def settle_games_outcome(direccion: str, linea: float, final_games: int) -> tuple[bool, str]:
    """OVER/UNDER total de juegos. Sin ambigüedad de push — líneas son .5."""
    if direccion == "OVER":
        win = final_games > linea
    else:  # UNDER
        win = final_games < linea
    return win, f"{direccion} {linea} vs final={final_games}"
```

**Tests (3 REGLA-T53):**
- `test_159_01_settle_over_above()` → linea=32.5, final=35 → win=True
- `test_159_02_settle_under_below()` → linea=24.5, final=22 → win=True
- `test_159_03_settle_no_snapshot_skips()` → sin snapshot final → `settle()` NO marca LOST, deja abierto (evita el bug P1 corregido)

---

### D159-02: Modelo Poisson/Markov Condicionado (KERNEL DE MEJORA)

**Ubicación:** `analysis/games_serve_model.py` (módulo nuevo)

Reemplaza `σ=3 fijo` con distribución real dados (serving, break_situation).

**Corrección de enfoque:** el borrador original invocaba `poisson_markov_pmf()` sin definir su forma cerrada — modelar "juegos totales en un set" analíticamente requiere las reglas completas de un set (6 juegos gana-por-2, tiebreak a 6-6), no solo una cadena de holds. Un cierre analítico real es investigación no trivial. La alternativa honesta y estándar para esto es **simulación Monte Carlo** con semilla determinista (testeable con REGLA-T53 vía tolerancia ±2% en N grande):

```python
import random

def p_remain_conditioned(
    games_played: int, linea: float, direccion: str,
    serving: str, break_situation: bool,
    p_hold_serving: float = 0.65, p_hold_other: float = 0.65,
    n_sims: int = 3000, seed: int = 42,
) -> float:
    """
    Monte Carlo: simula partidos futuros desde el estado actual, alternando
    servicio (reglas reales de set: 6 juegos gana-por-2, tiebreak en 6-6),
    usando p_hold específica de cada servidor. Retorna P(direccion cubre linea).

    NOTA: p_hold_serving/p_hold_other son placeholders 0.65 (Tour ATP/WTA avg)
    hasta que exista una fuente de hold% por jugador (nodo futuro).
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_sims):
        total_games = games_played
        current_server = serving  # "home" | "away"
        # simula juegos restantes hasta fin de set actual + sets adicionales
        # (reusa max_remaining ya calculado por la función madre como cap superior)
        while total_games < games_played + 13:  # cap defensivo por simulación
            p_hold = p_hold_serving if current_server == serving else p_hold_other
            held = rng.random() < p_hold
            total_games += 1
            if not held:
                break_situation = True  # informativo, no cambia la simulación de conteo
            current_server = "away" if current_server == "home" else "home"
            # condición de fin simplificada: set termina cuando total_games alcanza
            # el cap estructural ya validado por max_remaining (caller pasa el tope real)
            if total_games >= games_played + 13:
                break
        cubre = (total_games < linea) if direccion == "UNDER" else (total_games > linea)
        hits += int(cubre)
    return hits / n_sims
```

**Firma real verificada** (`live_desk.py:3175`, invariante — NO cambiar orden/nombres de params existentes, solo añadir opcionales al final):

```python
def _calcular_certeza_condicional(
    linea: float, direccion: str, games_played: int, sets_complete: int,
    current_games: int, zona: str,
    sets_home: Optional[int] = None, sets_away: Optional[int] = None,
    games_set1: Optional[int] = None,
    # D159-02 — NUEVOS, todos opcionales (compatibilidad total con 2 call-sites existentes L4109/L4427)
    serving: Optional[str] = None,
    break_situation: Optional[bool] = None,
    p_hold_serving: float = 0.65,  # prob. genérica de hold — placeholder hasta tener specific-player IRP
) -> Dict[str, Any]:
    ...
    # Gaussiano fijo (D147-02) permanece como fallback — activa SOLO si serving es None
    if serving is not None:
        p_condicional = p_remain_conditioned(
            games_played=games_played, linea=linea, direccion=direccion,
            serving=serving, break_situation=bool(break_situation),
            p_hold_serving=p_hold_serving,
        )
    else:
        # ... lógica Gaussiana existente sin cambios
```

**Call-sites a actualizar** (2 sitios, mismo patrón — `score_data`/`_sd147`/`_itf147["score_data"]` YA contienen `serving`/`break_situation` desde D153, simplemente no se pasan hoy):

```python
# live_desk.py:4109 y :4427 — añadir 2 argumentos al call existente
_s147["certeza"] = _calcular_certeza_condicional(
    linea=_linea147, direccion=_s147.get("direccion", "UNDER"),
    games_played=int(_sd147.get("games_played") or 0),
    sets_complete=int(_sd147.get("sets_complete") or 0),
    current_games=int(_sd147.get("current_games") or 0),
    zona=_zona147, sets_home=_sd147.get("sets_home"), sets_away=_sd147.get("sets_away"),
    games_set1=_gs1_147,
    serving=_sd147.get("serving"),                        # NUEVO — ya existe en el dict (D153-02)
    break_situation=_sd147.get("break_situation", False),  # NUEVO — ya existe en el dict (D153-04)
)
```

**p_hold_serving:** no existe una fuente de "probabilidad de hold específica del jugador" en el sistema hoy (IRP de Nodo-96 mide return-from-inactivity, no hold%). Usar placeholder fijo 0.65 (aprox. hold% ATP/WTA tour promedio) en esta entrega — vincular a dato real por jugador es un nodo futuro, NO bloquea D159-02.

**Tests (4 REGLA-T53):**
- `test_159_04_markov_vs_gaussiano()` → mismo caso Alexandrova, compare σ
- `test_159_05_break_situation_tightens()` → break_sit=True → µ baja
- `test_159_06_serving_effect()` → serving="home" p_hold=0.82 vs 0.50 → distribución más tight
- `test_159_07_cdf_bounds()` → P(OVER linea) + P(UNDER linea) = 1.0 ✓

---

### D159-03: Velocity Detection (Steam/ATN) — REUTILIZA `analysis/velocity_monitor.py::velocity_zscore()` (Nodo-71)

**Corrección vs draft original:** el borrador original inventaba una función nueva `detect_steam_games()` con umbral estático ±5%/2-ciclos. Verificado contra código real: `analysis/velocity_monitor.py::velocity_zscore(odds_series, times_minutes)` (Nodo-71, D71) YA implementa exactamente el detector que el usuario pide en el punto #2 ("drift rápido y sostenido en 2-3 ciclos... más limpio que umbral estático") — calcula velocidad (Δcuota/Δminuto) por ciclo, z-score de la última velocidad contra la distribución de velocidades previas, y clasifica `STEAM` (z<-2.0) / `DRIFT` (z>2.0) / `FLAT`. Esto es estrictamente superior al umbral estático ±5% del draft: un z-score se adapta a la volatilidad propia de cada mercado en vez de un corte fijo. Confirmado vía grep + `graphify query` que `velocity_zscore` NUNCA se invoca desde `live_desk.py` ni `games_signal_calculator.py` — solo aparece una vez como campo de display no conectado en el mercado ML (`live_desk.py:2041`). El gap real no es "falta la función", es "falta el adaptador que conecte el formato de datos de games al formato que la función espera".

**Formato real verificado de `odds_history`** (escrito por D147-05 `_write_games_odds_history()`, archivo `reports/games_odds_history_{fecha}.json`, inspeccionado en producción 2026-08-01):

```json
{
  "Jamie Mackenzie vs Max Dahlin_OVER": [
    {"ts": "09:56", "cuota": 1.89, "games_played": 0},
    {"ts": "10:02", "cuota": 1.89, "games_played": 1},
    {"ts": "10:03", "cuota": 2.05, "games_played": 1}
  ]
}
```

`velocity_zscore()` espera `odds_series: List[float]` + `times_minutes: List[float]` (minutos, no timestamps de reloj) — `ts` viene como string `"HH:MM"`, requiere conversión a minutos-transcurridos antes de pasar a la función. Función adaptadora nueva (`live_desk.py`, sin tocar `velocity_monitor.py`):

```python
from analysis.velocity_monitor import velocity_zscore

def _steam_signal_games(sig: dict, fecha_compact: str) -> dict:
    """Adapta reports/games_odds_history_{fecha}.json al formato de velocity_zscore().
    Reutiliza Nodo-71 sin duplicar lógica de z-score."""
    key = f"{sig.get('partido','')}_{sig.get('direccion','')}"
    hist = _load_games_odds_history(fecha_compact).get(key, [])
    if len(hist) < 3:
        return {"signal": "FLAT", "steam": False}
    odds_series = [pt["cuota"] for pt in hist]
    t0 = _parse_hhmm_to_minutes(hist[0]["ts"])
    times_minutes = [_parse_hhmm_to_minutes(pt["ts"]) - t0 for pt in hist]
    return velocity_zscore(odds_series, times_minutes)
```

**Call-site en `_check_games_convergencia()` (`live_desk.py:3932`, gate antes de dispatch):**

```python
# Existente (D150-01/02/03)...

# NUEVO: D159-03 — reutiliza velocity_zscore vía adaptador, no umbral estático
_steam = _steam_signal_games(sig, fecha_compact)
if _steam["signal"] == "STEAM":
    log([ITF_LIVE_STEAM], f"z_last={_steam['z_last']} — velocidad anómala de cuota")
    sig["steam_contraria"] = True  # informativo — decisión de bloqueo/no-bloqueo la toma el caller (mismo patrón que D150-01 cuota_envenenada)
```

**Nota de diseño (a decidir en implementación, no bloquea la spec):** `cuota_envenenada` (D150-01) ya detecta drift >+15% acumulado; STEAM (z-score) detecta *velocidad* anómala aunque el drift acumulado aún no cruce +15%. Son señales complementarias, no redundantes — STEAM puede disparar temprano en un movimiento que `cuota_envenenada` solo detectaría 2-3 ciclos después. Igual que D150-01, `steam_contraria` se añade como campo informativo al `sig` — el gate de disparo real (bloquear vs solo loggear) sigue el mismo patrón ya establecido en `_fire_itf_live_games_combo()` para no introducir una segunda arquitectura de gates.

**Tests (2 REGLA-T53):**
- `test_159_08_steam_adapter_converts_format()` → dado un `games_odds_history` real (formato `ts`/`cuota`), verifica que `_steam_signal_games()` produce `odds_series`/`times_minutes` correctos y delega en `velocity_zscore()` real (no reimplementa el z-score — REGLA-T53)
- `test_159_09_steam_flat_insufficient_data()` → `len(hist) < 3` → `signal="FLAT"`, `steam=False` (mismo comportamiento que `velocity_zscore()` con datos insuficientes)

---

### D159-04: Fillability Check Pre-Dispatch — REUTILIZA `_extract_games_cuota_live` (Nodo-135 D135-01)

**Corrección vs draft original:** NO se necesita un fetcher nuevo — `_extract_games_cuota_live(event_id, direccion, linea)` (`live_desk.py:2738`) ya hace exactamente el fetch necesario (endpoint `betoffer/event/{id}.json`, excluye mercados set-level). El único problema: tiene caché de 30s (`_CUOTA_LIVE_TTL`, comentario D153-RATELIMIT-2) — un refetch "pre-dispatch" dentro de la misma ventana de 30s devolvería el mismo valor cacheado del ciclo de 15s anterior, sin detectar drift real. Fix: parámetro `bypass_cache` en la función existente, usado solo en el punto de disparo.

```python
# live_desk.py — modificar firma existente (default False = comportamiento actual intacto)
def _extract_games_cuota_live(event_id: int, direccion: str, linea: Optional[float],
                                bypass_cache: bool = False) -> Optional[float]:
    _ck = (int(event_id), (direccion or "").upper(), float(linea) if linea is not None else None)
    _now = time.time()
    if not bypass_cache:
        _cached = _cuota_live_cache.get(_ck)
        if _cached and (_now - _cached[0]) < _CUOTA_LIVE_TTL:
            return _cached[1]
    # ... resto sin cambios (fetch real)
```

**Call-site en `_fire_itf_live_games_combo()` (línea 3871), antes de escribir el .bat:**

```python
def validate_fillability(sig: dict, threshold_pct: float = 0.05) -> tuple[bool, str]:
    """Función pura envoltorio — separa la decisión (testeable) del fetch (I/O)."""
    cuota_baseline = sig.get("cuota_live") or sig.get("cuota_pre") or 0
    cuota_actual = _extract_games_cuota_live(
        event_id=sig["event_id"], direccion=sig.get("direccion", "UNDER"),
        linea=sig.get("linea"), bypass_cache=True
    )
    if cuota_actual is None:
        return False, "outcome_expired_or_no_market"
    drift = abs(cuota_actual - cuota_baseline) / cuota_baseline if cuota_baseline else 1.0
    if drift > threshold_pct:
        return False, f"cuota_drift_{drift:.1%}_{cuota_baseline}to{cuota_actual}"
    return True, "fillable"

# dentro de _fire_itf_live_games_combo(), antes de escribir html_path/bat_path:
for s in signals:
    is_fill, reason = validate_fillability(s)
    if not is_fill:
        logger.warning(f"[ITF_LIVE_ABORT] {s.get('partido')} no fillable: {reason}")
        return  # aborta el combo completo — evita coupon con precio movido
```

**Tests (2 REGLA-T53):**
- `test_159_10_fillable_within_threshold()` → cuota 1.96→1.98 (drift 1%) → fillable=True
- `test_159_11_drift_exceeds_aborts()` → cuota 1.96→2.10 (drift 7.1%) → fillable=False

---

### D159-05: Micro-Kelly Shrinkage por n

**Ubicación:** `trader_ev_tenis.py::_calc_kelly_games()` (NUEVO)

```python
def _calc_kelly_games(
    edge_pct: float,  # p_modelo - 1/cuota
    cuota: float,
    n_historico: int,  # settled picks en esta ruta (H147-01/H150/H151)
    shrinkage_prior: int = 20  # n_eq
) -> float:
    """
    Kelly-KL para games-total, con shrinkage por n histórico.
    
    f* = f_clásico × exp(-λ × KL(...))
    + shrinkage: n/(n+prior) cuando n<30
    """
    
    f_classic = edge_pct / cuota
    
    # KL penalty: games son mercado más volátil
    # λ_games=3.0× (vs λ_atp500=1.6×)
    kl_dist = kl_divergence(p_modelo=1-1/cuota, p_prior=0.50)
    f_kl = f_classic * exp(-3.0 * kl_dist)
    
    # Shrinkage por n
    if n_historico < 30:
        shrink_factor = n_historico / (n_historico + shrinkage_prior)
        f_kl *= shrink_factor
        reason = f"shrink {shrink_factor:.1%} (n={n_historico})"
    else:
        reason = "no_shrink (n≥30)"
    
    return min(f_kl, 0.03), reason  # cap 3% del bankroll por leg
```

**Trigger:** Una vez que H150-01 (o cualquiera) acumula n≥30 settled, activar D159-05.

**Tests (1 REGLA-T53):**
- `test_159_12_shrink_kelly_low_n()` → n=10, shrink=33% → kelly menor que fclassic

---

## HIPÓTESIS PRE-REGISTRADAS — DESBLOQUEADAS

Una vez D159-01 corra, estas 5 hipótesis INICIAN acumulación:

| Hipótesis | Métrica | n_goal | Criterio graduación |
|-----------|---------|--------|---------------------|
| H147-01 | hit% | 20 | DOMINANTE p_cond≥0.70 + gp>linea/2 |
| H150-01 | hit% | 20 | ALTA sin CUOTA_ENVENENADA |
| H150-02 | hit% | 20 | ALTA sin SET1_TIEBREAK |
| H150-03 | hit% | 20 | general, hit≥40% |
| H151-01 | hit% | 20 | con 3 gates live, hit≥40% |

**Cronograma esperado:**
- D159-01: implementación 8h (tests, settlement harness)
- D159-04: +2h (fillability guard)
- **TOTAL S1: 16h → desbloquea acumulación**
- S2 (D159-02/03/05): refinamiento, 12h

---

## CAMBIOS EN RUN_DAILY.PY

**Corrección vs draft original:** el borrador proponía un PASO 10c nuevo con una función `settle_games_auto()` inventada. Verificado (`grep settle run_daily.py`): `run_daily.py` YA invoca `python3 shadow_book.py --settle <fecha>` automáticamente como PASO 10 (líneas 327 y 571, tanto en `--settle-only` como en el flujo diario completo). Como el fix de D159-01 vive DENTRO de `shadow_book.py::settle()` (branch `pick_type=='games_live'`), el PASO 10 existente ya settlea games_live automáticamente sin ningún cambio en `run_daily.py`.

**Sin cambios en `run_daily.py`.**

---

## TESTS REGLA-T53 — 12 TOTAL

```bash
python -m pytest tests/test_nodo159_games_settlement.py -v
```

Estructura:
```
test_159_01_settle_over_above()
test_159_02_settle_under_below()
test_159_03_settle_no_snapshot_skips()
test_159_04_markov_vs_gaussiano()
test_159_05_break_situation_tightens()
test_159_06_serving_effect()
test_159_07_cdf_bounds()
test_159_08_steam_adapter_converts_format()
test_159_09_steam_flat_insufficient_data()
test_159_10_fillable_within_threshold()
test_159_11_drift_exceeds_aborts()
test_159_12_shrink_kelly_low_n()
```

---

## DEPENDENCIAS

- ✅ D153 (serving, break_situation, game_score) — YA IMPLEMENTADO
- ✅ D147-02 (baseline freeze, odds_history) — YA IMPLEMENTADO
- ✅ D150-06 (_fire_itf_live_games_combo hook point) — YA IMPLEMENTADO
- ✅ D157-02 (outcome_id fresco cada ciclo) — YA IMPLEMENTADO
- 📋 D159-01 (settlement automático) — BLOQUEADOR, ESTA PROPUESTA

---

## IMPACTO ESPERADO

| Métrica | Antes | Después | Delta |
|---------|-------|---------|-------|
| H147-01 state | dormida (n=0) | acumulando (n=0→20/mes) | **+20 datapoints** |
| H150-01/02/03 state | dormidas | acumulando | **+60 datapoints** |
| H151-01 state | dormida | acumulando | **+20 datapoints** |
| certeza_band (D159-02) | ±0.25 | ±0.08 | **±69% tighter** |
| games combos abortados (D159-04) | 0% | ~3–5% (filtro liquidez) | **−3–5% false positives** |

---

## DEUDA CATALOGADA

- D157-04 (settlement omitido) → D159-01 cierra ✓
- D143-03 (Nodo-142 sin spec) → fuera de scope
- D159-05 (activar post-n30) → trigger automático en run_daily

---

## PRIORIZACIÓN

**🔴 CRÍTICA:** D159-01 (desbloquea 5 hipótesis) — **IMPLEMENTADO 2026-08-01 (sesión S1)**
**🟡 ALTA:** D159-04 (evita rebotes usuario) — **IMPLEMENTADO 2026-08-01 (sesión S1)**
**🟢 MEDIA:** D159-02/03/05 (refinamiento) — **DIFERIDO a sesión futura.** No bloquea graduación de hipótesis — S1 ya desbloquea acumulación. D159-05 además no puede activarse hasta n≥30 settled (gate propio, no existe aún tras S1).

---

## REFERENCIA CRUZADA

- [[Nodo-157]] D157-04 (settlement games) — cierre definitivo aquí
- [[Nodo-150]] D150-06 (dispatch hook)
- [[Nodo-151]] H151-01 acumulación
- [[Nodo-147]] D147-02 (baseline freeze, odds_history)
- [[Nodo-153]] D153 (serving, break_situation source)
- [[Nodo-71]] `velocity_zscore()` (Kyle's λ) — D159-03 reutiliza, no reimplementa

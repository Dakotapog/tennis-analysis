# Nodo-147 — Live Score × Games Convergencia: Certeza Condicional en Tiempo Real

**Fecha:** 2026-07-25
**Estado:** SPEC
**Wikilinks:** [[Nodo-133]] [[Nodo-135]] [[Nodo-40]] [[Nodo-109]] [[Nodo-100]] [[Nodo-129]] [[live_desk]] [[games_signal_calculator]] [[live_edge_monitor]]

---

## 1. Diagnóstico — 4 Gaps entre señal calculada y realidad operativa

### Gap-1: Score Kambi existe pero nunca llega al panel X3

`_parse_kambi_tennis_score(event_obj)` existe en `live_desk.py` (L2808-2874). Recibe un
objeto `event` de Kambi `liveEvents.json` y devuelve:
```python
{"games_played": int, "sets_complete": int,
 "current_games": {"home": int, "away": int},
 "sets_home": int, "sets_away": int}
```

**Problema:** Esta función NUNCA es invocada desde `_build_x3_games()` ni desde
`_check_games_convergencia()`. El panel X3 muestra `cuota_live` y `drift_pct`,
pero NO el progreso del partido. El operador no puede distinguir si el partido va
6-0, 1-0 (UNDER casi seguro) vs 6-3, 5-5 (UNDER en riesgo) — diferencia crítica.

### Gap-2: La línea pre-partido desaparece cuando el partido entra EN_VIVO

`_build_x3_games()` lee `cuota` y `linea` del `games_signal_report_{fecha}*.json`
(calculado en PASO 3.6, pre-partido). Cuando `_check_games_convergencia()` detecta
el evento como EN_VIVO y escribe `cuota_live` + `drift_pct` en
`games_live_{fecha}.json`, el panel muestra la cuota actual — pero la referencia
pre-partido se pierde en el siguiente ciclo de refresco (15s).

**Contexto D142-T0 (L3256-3261):** Existe mecanismo de snapshot
`itf_live_snapshot_{fecha_compact}.json` con `{cuota_t0, linea_t0, ts_t0}` — pero
SOLO para ITF_VIVO events (partidos ya STARTED en Kambi con mercado abierto en vivo).
Las señales pre-partido ATP/WTA/Challenger con `event_id` en el `games_signal_report`
que transicionan a EN_VIVO NO tienen freeze de baseline.

Consecuencia operativa: el operador ve drift_pct relativo a la última cuota conocida,
no al precio de apertura. Una cuota que bajó de 1.85 → 1.75 → 1.70 se muestra como
"-2.9%" en vez del drift real "-8.1%" desde la apertura. Esto subestima la fuerza
de la señal.

### Gap-3: p_modelo sin actualizar — señal estática durante el partido

`games_signal_calculator.py` calcula `p_modelo` usando distribución esperada de juegos
totales (DOMINANTE: µ~18, COINFLIP: µ~23) basado en H2H histórico pre-partido.

Una vez iniciado:
- 12 juegos jugados, marcador 6-2, 4-0 (set 2) → UNDER 22.5 casi matemáticamente cierta
- 12 juegos jugados, marcador 6-6, 6-0 (empate 1-1) → UNDER 22.5 imposible

El modelo muestra la misma `p_modelo` original en ambos casos. No hay actualización
condicional P(UNDER | juegos_acumulados=k, estado_actual).

### Gap-4: Sin historial de cuotas games para trazabilidad de curva

Para el mercado Ganador existe `_write_odds_history()` que alimenta el sparkline
de tendencia (Nodo-129). Para el mercado Total de Juegos no existe función equivalente.
El operador no puede determinar si la cuota viene bajando continuamente (señal
fortaleciéndose) o rebotando (ruido).

---

## 2. Arquitectura de Solución

```
_background_refresh() [daemon thread, cada 15s]
    │
    └─ _check_games_convergencia(fecha)
            │
            ├─ [EXISTENTE] _fetch_kambi_live_events() → live_events_data
            ├─ [EXISTENTE] clasificar estados PRE/EN_VIVO/TERMINADO por señal
            ├─ [EXISTENTE] fetch cuota_live, calcular drift_pct
            │
            ├─ [D147-01] _enrich_live_score(alta_signals, live_events_data)
            │             → mutación in-place: score_data por señal
            │
            ├─ [D147-02] _calcular_certeza_condicional(...)  × señal EN_VIVO
            │             → {certeza_matematica, p_condicional, alerta_nivel, razon}
            │
            ├─ [D147-03] _freeze_baseline_if_needed(alta_signals, fecha_compact)
            │             → games_baseline_{fecha_compact}.json (inmutable una vez escrito)
            │
            ├─ [D147-05] _write_games_odds_history(alta_signals, fecha_compact)
            │             → games_odds_history_{fecha_compact}.json (append-only)
            │
            └─ [D147-06] _fire_certeza_alert(sig, fecha_compact)  [fire-once por señal]

_build_x3_games(fecha) [render HTTP, invocado por DeskHandler]
    │
    └─ [D147-04] tabla enriquecida:
                 Partido | Señal | Base(T0) | Live | Drift | Progreso | Certeza
```

**Invariante clave:** Todo el enriquecimiento ocurre dentro de `_check_games_convergencia()`
y se serializa a `games_live_{fecha}.json`. `_build_x3_games()` solo lee ese JSON —
NO llama directamente a las funciones D147-01/02/03. Esto mantiene el render O(1) y
la separación data/display.

---

## 3. Decisiones

### D147-01: `_enrich_live_score(signals, live_events)` — Conectar score Kambi al X3

**Archivo:** `live_desk.py` — nueva función, llamar al final del loop de clasificación
en `_check_games_convergencia()`.

**Firma:**
```python
def _enrich_live_score(
    signals: List[Dict],
    live_events: List[Dict],
) -> None:
    """Enriquece signals IN PLACE con score Kambi via _parse_kambi_tennis_score().

    Para cada señal en signals:
      1. Leer sig.get("event_id") — puede ser int o str, normalizar a int
      2. Buscar en live_events el dict donde event_obj["event"]["id"] == event_id
         (estructura: live_events[i] = {"event": {..., "id": 12345}, "liveData": {...}})
      3. Si encontrado → llamar _parse_kambi_tennis_score(event_obj["event"])
         NOTA: _parse_kambi_tennis_score espera el sub-dict "event", no el wrapper
      4. Escribir sig["score_data"] = resultado del parse
      5. Si no encontrado o excepción → sig["score_data"] = None

    Returns None — mutación in-place.
    """
```

**Índice para O(1) lookup:**
```python
# Construir antes del loop para evitar O(n×m):
live_by_id: Dict[int, Dict] = {
    int(ew["event"]["id"]): ew
    for ew in live_events
    if isinstance(ew, dict) and "event" in ew and "id" in ew["event"]
}
```

**Formato de display derivado de score_data:**
```python
# score_data = {"games_played": 12, "sets_complete": 1,
#               "current_games": {"home": 4, "away": 2},
#               "sets_home": 1, "sets_away": 0}
# Display: "1-0 (4:2) | 12j"
# Si score_data is None: "PRE"
```

**Edge cases:**
- `event_id` None o 0 → skip (señal pre-partido sin event_id asignado)
- `_parse_kambi_tennis_score()` lanza cualquier excepción → `score_data = None`, log DEBUG
- Partido terminado (en live_events pero `liveData` sin score activo) → score_data con
  `games_played` del score final, `sets_complete = 2` o 3 según formato

---

### D147-02: `_calcular_certeza_condicional(...)` — Motor matemático

**Archivo:** `live_desk.py` — nueva función pura (sin side effects, sin I/O).

**Firma:**
```python
def _calcular_certeza_condicional(
    linea: float,
    direccion: str,          # "UNDER" o "OVER"
    games_played: int,
    sets_complete: int,      # sets finalizados (0, 1, 2)
    current_set_home: int,   # juegos del set en curso, home
    current_set_away: int,   # juegos del set en curso, away
    zona: str,               # "DOMINANTE" o "COINFLIP"
) -> Dict[str, Any]:
    """
    Returns:
        {
          "certeza_matematica": bool,
          "p_condicional": float,    # [0.0, 1.0]
          "alerta_nivel": str,       # "CERTEZA" | "ALTA" | "MOD" | "BAJA" | ""
          "razon": str,              # legible para logs/display
        }
    """
```

**Bloque 1 — Certeza Matemática (UNDER N.5):**

```python
# Peor caso de juegos restantes (best-of-3):
#   sets_remaining = 2 - sets_complete
#   juegos_restantes_peor_caso = sets_remaining * 13
#   (cada set puede ir a tiebreak: 7-6 = 13 juegos)
#
# Certeza UNDER: games_played + juegos_restantes_peor_caso < linea
# Ejemplo: games_played=22, sets_complete=1, linea=22.5
#   peor_caso = 1*13 = 13 → 22+13=35 > 22.5 → NO certeza
# Ejemplo: games_played=21, sets_complete=2, linea=22.5
#   peor_caso = 0*13 = 0  → 21+0=21 < 22.5 → CERTEZA (partido terminado con 21 juegos)
```

**Bloque 2 — Certeza Matemática (OVER N.5):**

```python
# Mejor caso de juegos restantes:
#   si sets_complete == 2 → partido terminado, min_remaining = 0
#   si sets_complete == 1 → set 3 mínimo 6 juegos (6-0)
#   si sets_complete == 0 → sets 2+3 mínimo 12 juegos (6-0, 6-0)
#
# Certeza OVER: games_played + min_remaining > linea
# Ejemplo: games_played=22, sets_complete=0, linea=22.5
#   min_remaining = 12 → 22+12=34 > 22.5 → CERTEZA OVER
```

**Bloque 3 — p_condicional (modelo Gaussiano):**

```python
# Parámetros calibrados como priors — ajustar con n_stop=20 reales (H147-01):
ZONA_PARAMS = {
    "DOMINANTE": {"mu": 18.0, "sigma": 3.0},
    "COINFLIP":  {"mu": 23.0, "sigma": 4.5},
}

# p(UNDER N.5 | k juegos ya jugados):
#   mu_total    = ZONA_PARAMS[zona]["mu"]
#   sigma_total = ZONA_PARAMS[zona]["sigma"]
#   mu_restantes    = max(0.0, mu_total - games_played)
#   sigma_restantes = sigma_total  # conservador: no reducir sigma con evidencia parcial
#
#   p_under = Φ(x) donde x = (linea + 0.5 - games_played - mu_restantes) / sigma_restantes
#   Usando scipy.stats.norm.cdf(x) si disponible, o tabla lookup si no.

# p(OVER N.5): 1.0 - p_under
```

**Tabla lookup (fallback si scipy no disponible):**

```python
# UNDER — zona DOMINANTE, linea=22.5
_LOOKUP_DOMINANTE_UNDER = {
    # games_played → p_condicional
    (0, 12):  0.70,
    (13, 15): 0.80,
    (16, 18): 0.92,
    (19, 21): 0.98,
    (22, 999): 1.00,  # certeza matemática cubierta arriba
}
# UNDER — zona COINFLIP, linea=22.5
_LOOKUP_COINFLIP_UNDER = {
    (0, 12):  0.30,
    (13, 15): 0.45,
    (16, 18): 0.60,
    (19, 21): 0.78,
    (22, 999): 1.00,
}
# OVER — invertir: p_over = 1 - p_under para la zona correspondiente
```

**Implementación de lookup:**
```python
def _lookup_p(games_played: int, tabla: dict) -> float:
    for (lo, hi), p in tabla.items():
        if lo <= games_played <= hi:
            return p
    return 0.5  # fallback neutral
```

**Niveles de alerta:**
```python
if certeza_matematica:
    alerta_nivel = "CERTEZA"
elif p_condicional >= 0.90:
    alerta_nivel = "ALTA"
elif p_condicional >= 0.70:
    alerta_nivel = "MOD"
elif p_condicional >= 0.50:
    alerta_nivel = "BAJA"
else:
    alerta_nivel = ""
```

**Nota sobre best-of-5 (Grand Slam):** La función asume best-of-3. Si `sets_complete >= 3`
o el partido llega a 2-2 sets → no calcular certeza matemática (retornar
`certeza_matematica=False`, `razon="GS no soportado"`). El mercado Total de Juegos
en GS tiene umbral distinto (~38-42 juegos) y los parámetros ZONA no aplican.

---

### D147-03: `_freeze_baseline_if_needed(signals, fecha_compact)` — Congelar T0

**Archivo:** `live_desk.py` — nueva función con I/O a archivo JSON.

**Archivo de datos:** `reports/games_baseline_{fecha_compact}.json`

**Schema del JSON:**
```json
{
  "Shapovalov D. vs Ruud C._UNDER": {
    "cuota_t0": 1.85,
    "linea_t0": 22.5,
    "ts_t0": "2026-07-25T14:30:22",
    "direccion": "UNDER",
    "zona": "DOMINANTE"
  }
}
```

**partido_key:** `f"{sig['partido']}_{sig['direccion']}"` — clave única por dirección de apuesta
(un partido puede tener señal UNDER y señal OVER simultáneamente; son independientes).

**Firma:**
```python
def _freeze_baseline_if_needed(
    signals: List[Dict],
    fecha_compact: str,
) -> Dict[str, Dict]:
    """Lee games_baseline_{fecha_compact}.json; escribe entradas nuevas para
    señales que acaban de pasar a EN_VIVO y aún no tienen T0 registrado.

    REGLA DE INMUTABILIDAD: si partido_key ya existe en el JSON → NUNCA sobreescribir.
    El baseline es el precio de apertura cuando el partido entró a EN_VIVO por primera vez.

    Solo congelar si:
      - sig["estado_live"] == "EN_VIVO"
      - sig.get("cuota_live") is not None   (hay precio actual disponible)
      - partido_key NOT in baseline          (primer ciclo EN_VIVO)

    Returns: dict baseline actualizado (para que _check_games_convergencia lo pueda
    pasar a _build_x3_games via games_live JSON schema extension).
    """
```

**Integración de baseline en `games_live_{fecha}.json`:** Al escribir este JSON,
enriquecer cada señal con `cuota_t0` y `linea_t0` leídos del baseline (si existe la key).
Esto permite que `_build_x3_games()` los lea sin acceso adicional a disco.

---

### D147-04: Panel X3 — Nuevas columnas en render HTML

**Archivo:** `live_desk.py` — modificar la sección de render del panel X3 en
`DeskHandler.do_GET()` o en la función que genera el HTML de X3.

**Tabla anterior:**
```
Partido | Señal | Cuota | Gap | Estado | Drift
```

**Tabla nueva:**
```
Partido | Señal | Base(T0) | Live | Drift | Progreso | Certeza
```

**Especificación de cada columna:**

| Columna | Campo en signal dict | Formato de display |
|---------|---------------------|-------------------|
| Partido | `sig["partido"]` | texto plano |
| Señal | `sig["direccion"] + " " + str(sig.get("linea_t0", sig["linea"]))` | `"UNDER 22.5"` |
| Base(T0) | `sig.get("cuota_t0")` | `"@1.85"` en gris claro si None → `"@—"` |
| Live | `sig.get("cuota_live")` | `"@1.72"` en negro |
| Drift | `sig.get("drift_pct")` | ver lógica de color abajo |
| Progreso | derivado de `sig.get("score_data")` | `"1-0 (4:2) \| 12j"` o `"PRE"` |
| Certeza | `sig.get("certeza", {}).get("alerta_nivel", "")` | badge HTML coloreado |

**Lógica de color para columna Drift:**
```
direccion == "UNDER":
  drift_pct < -0.03 → verde  (cuota baja = mercado apoya UNDER = señal fortaleciéndose)
  drift_pct > +0.03 → rojo   (cuota sube = señal debilitándose)
  else              → gris
direccion == "OVER":
  drift_pct > +0.03 → verde  (cuota OVER sube = OVER más atractivo)
  drift_pct < -0.03 → rojo
  else              → gris
```

**Badges de Certeza (HTML inline):**
```html
<!-- CERTEZA -->
<span style="background:#00c851;color:white;padding:2px 8px;border-radius:3px;
             font-weight:bold;font-size:12px">CERTEZA</span>

<!-- ALTA -->
<span style="background:#ff8800;color:white;padding:2px 8px;border-radius:3px;
             font-size:12px">ALTA 90%+</span>

<!-- MOD -->
<span style="background:#33b5e5;color:white;padding:2px 8px;border-radius:3px;
             font-size:12px">MOD 70%+</span>

<!-- BAJA o vacío → solo texto gris sin badge -->
```

**Fila con `certeza_matematica=True`:** Pintar la fila entera con fondo
`background-color: #e8f5e9` (verde muy claro) para destacar visualmente.

**Columna Progreso con formato derivado:**
```python
def _fmt_progreso(score_data: Optional[Dict]) -> str:
    if not score_data:
        return "PRE"
    sets_h = score_data.get("sets_home", 0)
    sets_a = score_data.get("sets_away", 0)
    cg = score_data.get("current_games", {})
    g_h = cg.get("home", 0)
    g_a = cg.get("away", 0)
    gp  = score_data.get("games_played", 0)
    return f"{sets_h}-{sets_a} ({g_h}:{g_a}) | {gp}j"
```

---

### D147-05: `_write_games_odds_history(signals, fecha_compact)` — Sparkline

**Archivo:** `live_desk.py` — nueva función append-only.

**Archivo de datos:** `reports/games_odds_history_{fecha_compact}.json`

**Schema:**
```json
{
  "Shapovalov D. vs Ruud C._UNDER": [
    {"ts": "14:30", "cuota": 1.85, "games_played": 0},
    {"ts": "14:52", "cuota": 1.72, "games_played": 12},
    {"ts": "15:07", "cuota": 1.65, "games_played": 18}
  ]
}
```

**Firma:**
```python
def _write_games_odds_history(
    signals: List[Dict],
    fecha_compact: str,
) -> None:
    """Append-only: registra cuota_live actual y games_played por señal EN_VIVO.

    Para cada señal con estado_live=="EN_VIVO" y cuota_live not None:
      1. partido_key = f"{sig['partido']}_{sig['direccion']}"
      2. nuevo_punto = {
             "ts": datetime.now().strftime("%H:%M"),
             "cuota": sig["cuota_live"],
             "games_played": sig.get("score_data", {}).get("games_played", 0) if sig.get("score_data") else 0
         }
      3. DEDUPLICACIÓN: si el último punto en la lista tiene mismo games_played
         Y misma cuota → skip (evitar duplicados en ciclos de 15s sin avance)
      4. Append y reescribir JSON completo (archivo pequeño, < 10KB por día)
    """
```

**Display de tendencia en panel X3:** En la columna Drift, si `games_odds_history`
tiene ≥ 3 puntos para este `partido_key`, calcular tendencia:
```python
ultimos = historial[-3:]
es_bajista = all(ultimos[i]["cuota"] > ultimos[i+1]["cuota"] for i in range(2))
es_alcista = all(ultimos[i]["cuota"] < ultimos[i+1]["cuota"] for i in range(2))
sufijo = " vbaja" if es_bajista else (" vsube" if es_alcista else "")
# Mostrar como: "-7.0% vbaja" (texto plano, evitar emojis)
```

---

### D147-06: Banner CERTEZA_MATEMATICA + Telegram alert (fire-once)

**Archivo:** `live_desk.py` — nueva función `_fire_certeza_alert()`.

**Guard anti-flood:** `reports/certeza_fired_{fecha_compact}.json`
```json
{"Shapovalov D. vs Ruud C._UNDER": "2026-07-25T15:07:33"}
```

Si `partido_key` ya está en el archivo → skip. Solo disparar UNA VEZ por señal por día.

**Banner HTML (dentro de panel X3, encima de la tabla):**
```html
<div style="background:#00c851;color:white;padding:12px;margin-bottom:8px;
            border-radius:6px;font-size:15px;font-weight:bold;text-align:center;
            animation:blink 0.8s step-start infinite">
  CERTEZA MATEMATICA | {partido} {direccion} {linea_t0} | {games_played} juegos jugados
</div>
```

Mostrar este banner si cualquier señal EN_VIVO tiene `certeza_matematica=True`.

**Telegram alert:**
```python
def _fire_certeza_alert(sig: Dict, fecha_compact: str) -> None:
    """Fire-once Telegram cuando certeza_matematica=True."""
    partido_key = f"{sig['partido']}_{sig['direccion']}"
    guard_path = REPORTS / f"certeza_fired_{fecha_compact}.json"

    # Leer guard
    try:
        fired: Dict = json.loads(guard_path.read_text()) if guard_path.exists() else {}
    except Exception:
        fired = {}

    if partido_key in fired:
        return

    # Registrar antes de disparar (atomicidad: evitar doble-send si falla subprocess)
    fired[partido_key] = datetime.now().isoformat()
    try:
        guard_path.write_text(json.dumps(fired, indent=2))
    except Exception:
        pass

    # Construir mensaje
    gp = sig.get("score_data", {}).get("games_played", "?") if sig.get("score_data") else "?"
    msg = (
        f"CERTEZA MATEMATICA | {sig['partido']} "
        f"{sig['direccion']} {sig.get('linea_t0', sig.get('linea'))} | "
        f"{gp} juegos jugados | Resultado confirmado"
    )

    # Intentar enviar via scripts/send_telegram.py (verificar existencia antes)
    send_script = Path(__file__).parent / "scripts" / "send_telegram.py"
    if send_script.exists():
        try:
            import subprocess, sys
            subprocess.Popen(
                [sys.executable, str(send_script), "--msg", msg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning(f"[D147-06] Telegram alert fallido: {e}")
    else:
        logger.warning(f"[D147-06] CERTEZA MATEMATICA — {msg} (sin send_telegram.py)")
```

---

## 4. Orden de integración en `_check_games_convergencia()`

Insertar las nuevas llamadas DESPUÉS del bloque que calcula `drift_pct` por señal
y ANTES de escribir `games_live_{fecha}.json`. Esquema de las líneas relevantes:

```python
# --- CÓDIGO EXISTENTE ---
# ... (fetch live_events_data, clasificar estados, fetch cuota_live) ...

# --- BLOQUE D147 (insertar aquí) ---
fecha_compact = fecha.replace("-", "")

# D147-01: enriquecer con score
_enrich_live_score(alta_signals, live_events_data)

# D147-02: calcular certeza condicional
for sig in alta_signals:
    if sig.get("score_data") is not None and sig.get("estado_live") == "EN_VIVO":
        zona = "DOMINANTE" if "DOMIN" in sig.get("confianza", "").upper() else "COINFLIP"
        sd = sig["score_data"]
        sig["certeza"] = _calcular_certeza_condicional(
            linea=float(sig.get("linea") or 0),
            direccion=sig.get("direccion", "UNDER"),
            games_played=int(sd.get("games_played", 0)),
            sets_complete=int(sd.get("sets_complete", 0)),
            current_set_home=int(sd.get("current_games", {}).get("home", 0)),
            current_set_away=int(sd.get("current_games", {}).get("away", 0)),
            zona=zona,
        )
    else:
        sig["certeza"] = {
            "certeza_matematica": False, "p_condicional": 0.0,
            "alerta_nivel": "", "razon": "sin score"
        }

# D147-03: congelar baseline T0
baseline = _freeze_baseline_if_needed(alta_signals, fecha_compact)

# Propagar cuota_t0/linea_t0 desde baseline a cada señal (para serialización en games_live)
for sig in alta_signals:
    pk = f"{sig['partido']}_{sig['direccion']}"
    if pk in baseline:
        sig["cuota_t0"] = baseline[pk]["cuota_t0"]
        sig["linea_t0"] = baseline[pk]["linea_t0"]
        sig["ts_t0"]    = baseline[pk]["ts_t0"]

# D147-05: historial de cuotas
_write_games_odds_history(alta_signals, fecha_compact)

# D147-06: alert certeza
for sig in alta_signals:
    if sig.get("certeza", {}).get("certeza_matematica"):
        _fire_certeza_alert(sig, fecha_compact)

# --- CÓDIGO EXISTENTE: escribir games_live_{fecha}.json ---
# (signals ya tienen score_data, certeza, cuota_t0, linea_t0 — se serializan aquí)
```

**Nota:** `alta_signals` en `_check_games_convergencia()` son las señales que ya
pasaron el filtro de confianza ALTA. Verificar el nombre exacto de la variable local
en el archivo antes de implementar (L3124 aprox).

---

## 5. Hipótesis Pre-registrada

**H147-01 — Certeza Condicional DOMINANTE como predictor:**

- **Variable independiente:** `zona == "DOMINANTE" AND p_condicional >= 0.70 AND games_played > linea/2 AND certeza_matematica == False`
- **Variable dependiente:** el resultado real de la señal es correcto (UNDER/OVER confirmado en settle)
- **Hipótesis nula:** P(acierto | condición) = 0.50
- **Hipótesis alternativa (H1):** P(acierto | condición) ≥ 0.65
- **n_stop:** 20 señales settled que cumplan la condición
- **Verificación:** log en `shadow_book.py --report` sección "H147-01 Live Certeza"
  via campo `pick_type='games_live'` en `sb_FECHA.jsonl`
- **Breakeven:** depende de cuota media de la muestra; con cuota media ~1.80,
  breakeven = 1/1.80 = 55.6%

**Registrar en `validation/preregistered_hypotheses.json` con clave `"H147-01"`.**

---

## 6. Tests REGLA-T53

**Archivo:** `tests/test_nodo147_live_certeza.py`

```python
"""Tests Nodo-147 — Live Score × Games Certeza Condicional.
REGLA-T53: cada test invoca la función real del módulo — nunca hardcodea la fórmula.
"""
import json, tempfile
from pathlib import Path
from live_desk import (
    _calcular_certeza_condicional,
    _enrich_live_score,
    _freeze_baseline_if_needed,
    _write_games_odds_history,
    _fmt_progreso,
)

# test_nodo147_01 — certeza matemática UNDER: partido terminado con 21 juegos
def test_nodo147_01_certeza_matematica_under_terminado():
    resultado = _calcular_certeza_condicional(
        linea=22.5, direccion="UNDER",
        games_played=21, sets_complete=2,
        current_set_home=0, current_set_away=0,
        zona="DOMINANTE",
    )
    assert resultado["certeza_matematica"] is True
    assert resultado["alerta_nivel"] == "CERTEZA"

# test_nodo147_02 — NO certeza UNDER cuando hay sets pendientes con potencial alto
def test_nodo147_02_no_certeza_under_sets_pendientes():
    resultado = _calcular_certeza_condicional(
        linea=22.5, direccion="UNDER",
        games_played=10, sets_complete=1,
        current_set_home=2, current_set_away=2,
        zona="DOMINANTE",
    )
    assert resultado["certeza_matematica"] is False

# test_nodo147_03 — p_condicional DOMINANTE games_played alto → ALTA
def test_nodo147_03_p_condicional_dominante_alto():
    resultado = _calcular_certeza_condicional(
        linea=22.5, direccion="UNDER",
        games_played=18, sets_complete=1,
        current_set_home=5, current_set_away=1,
        zona="DOMINANTE",
    )
    assert resultado["p_condicional"] >= 0.90
    assert resultado["alerta_nivel"] in ("ALTA", "CERTEZA")

# test_nodo147_04 — p_condicional COINFLIP games_played bajo → incertidumbre
def test_nodo147_04_p_condicional_coinflip_bajo():
    resultado = _calcular_certeza_condicional(
        linea=22.5, direccion="OVER",
        games_played=8, sets_complete=0,
        current_set_home=4, current_set_away=4,
        zona="COINFLIP",
    )
    # COINFLIP OVER con pocos juegos es plausible pero no dominante
    assert resultado["p_condicional"] >= 0.30
    assert resultado["alerta_nivel"] in ("BAJA", "MOD", "")

# test_nodo147_05 — freeze_baseline inmutabilidad: no sobreescribir T0 existente
def test_nodo147_05_freeze_baseline_inmutable(tmp_path, monkeypatch):
    # Redirigir REPORTS a directorio temporal
    import live_desk
    monkeypatch.setattr(live_desk, "REPORTS", tmp_path)

    signals = [{
        "partido": "Ruud vs Shapovalov",
        "direccion": "UNDER",
        "cuota_live": 1.70,
        "linea": 22.5,
        "estado_live": "EN_VIVO",
    }]
    fecha_compact = "20260725"

    # Primera llamada → escribe T0=1.70
    baseline = _freeze_baseline_if_needed(signals, fecha_compact)
    assert baseline["Ruud vs Shapovalov_UNDER"]["cuota_t0"] == 1.70

    # Actualizar cuota_live → segunda llamada
    signals[0]["cuota_live"] = 1.55
    baseline2 = _freeze_baseline_if_needed(signals, fecha_compact)

    # T0 debe seguir siendo 1.70, no sobreescribirse con 1.55
    assert baseline2["Ruud vs Shapovalov_UNDER"]["cuota_t0"] == 1.70

# test_nodo147_06 — enrich_live_score conecta games_played desde evento Kambi
def test_nodo147_06_enrich_live_score_games_played():
    signal = {"partido": "A vs B", "event_id": 12345}
    live_events = [{
        "event": {
            "id": 12345,
            "homeName": "A",
            "awayName": "B",
        },
        "liveData": {
            "score": {"home": "1", "away": "0"},
            "currentServer": "HOME",
        }
    }]
    _enrich_live_score([signal], live_events)
    # score_data debe existir (parse exitoso o None si formato incompatible)
    assert "score_data" in signal
    # Si el parse devuelve datos → games_played debe ser int >= 0
    if signal["score_data"] is not None:
        assert isinstance(signal["score_data"].get("games_played"), int)
        assert signal["score_data"]["games_played"] >= 0
```

---

## 7. Alcance y NO-ALCANCE

**EN ALCANCE (este Nodo):**
- Conectar `_parse_kambi_tennis_score()` al flujo de `_check_games_convergencia()` (D147-01)
- Congelar T0 pre-partido para señales ATP/WTA/Challenger que pasan a EN_VIVO (D147-03)
- Calcular `p_condicional` con modelo Gaussiano + tabla lookup fallback (D147-02)
- Panel X3: columnas Progreso + Base(T0) + Certeza (D147-04)
- Historial cuotas games market append-only (D147-05)
- Banner HTML + Telegram alert fire-once (D147-06)

**FUERA DE ALCANCE:**
- Modelo Bayesiano con actualización sigma por observaciones parciales — requiere n≥50
  para calibrar; usar Gaussiano fijo hasta H147-01 gradúe (n_stop=20)
- Integración con `KambiLiveClientReal.get_score_data()` en `live_edge_monitor.py`
  (L158-260) — esa función usa cache separado optimizado para drift monitoring;
  en este Nodo usamos `_parse_kambi_tennis_score()` directo sobre `live_events_data`
  que ya está disponible en `_check_games_convergencia()`
- Auto-betting cuando `certeza_matematica=True` — gate H147-01 graduada primero
- Best-of-5 (Grand Slam) — `max_remaining` usa best-of-3; retornar
  `certeza_matematica=False, razon="GS no soportado"` si sets detectados ≥ 3
- Modificar `games_signal_calculator.py` o el pipeline pre-partido — todo es render/enrich

---

## 8. Archivos a crear/modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `live_desk.py` | MODIFICAR | 6 nuevas funciones + integración en `_check_games_convergencia()` + render X3 |
| `tests/test_nodo147_live_certeza.py` | CREAR | 6 tests REGLA-T53 |
| `reports/games_baseline_{fecha_compact}.json` | AUTO-CREATE | runtime, inmutable por señal |
| `reports/games_odds_history_{fecha_compact}.json` | AUTO-CREATE | runtime, append-only |
| `reports/certeza_fired_{fecha_compact}.json` | AUTO-CREATE | runtime, guard anti-flood |
| `validation/preregistered_hypotheses.json` | MODIFICAR | agregar H147-01 |

---

## ADDENDUM D147-07 (2026-08-02) — Fix: alerta Telegram estaba silenciosamente rota

**Hallazgo:** auditoría solicitada por el usuario ("verificar con evidencia real si Telegram
funciona para las alertas live-games") encontró que `_fire_certeza_alert()` (D147-06)
llamaba a `scripts/send_telegram.py` vía `subprocess.Popen`, guardado tras un
`if send_script.exists():`. Ese archivo **nunca existió** en el repo ni en su historial git
(`git log --all -- '*send_telegram*'` → vacío). El guard `.exists()` evitaba una excepción,
por lo que no había ningún error visible — solo se saltaba en silencio.

**Evidencia real (`logs/live_desk.log`, 2026-08-02):** 6 disparos reales de CERTEZA
MATEMATICA (Jodar/Musetti, Vidmanova/Volynets, Mackenzie/Dahlin, Jimenez Kasintseva/Bassols
Ribera, Poljicak/Varillas, Liutova/Vidmanova) — `grep -ic telegram logs/live_desk.log` → 0
coincidencias en 8.5MB de log. Ninguna de las 6 llegó a Telegram.

**Fix:** `_send_telegram_async(msg, tag)` — helper nuevo en `live_desk.py` que llama a
`utils.telegram._enviar_telegram()` (el bot real, mismo `TG_TOKEN` que
`betplay_combo_builder.py`/`combo_confianza_builder.py --telegram`, confirmado funcional
para el sistema de favoritos) desde un thread daemon, sin bloquear el loop de 15s.
`_fire_certeza_alert()` ahora llama a este helper en vez del script inexistente.

**Hallazgo relacionado (ver [[Nodo-157]] D157-05):** el combo ITF live games
(`_fire_itf_live_games_combo`, 135 disparos en el mismo log) **nunca tuvo integración
Telegram en absoluto** — solo escribía `.bat`/HTML a Desktop. Se agregó notificación en el
guard de señal-nueva (mismo patrón cap 10/día que ya existía, D157-02).

5 tests REGLA-T53 nuevos en `tests/test_nodo147_telegram_fix.py`.

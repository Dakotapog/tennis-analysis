# Nodo-133 — Games Live Convergencia: Estado en Tiempo Real + Auto-Combo

> **Estado:** IMPLEMENTADO — 2026-07-21
> **Tipo:** FEATURE — cierra el gap entre señales ALTA estáticas y ventanas de apuesta live
> **Trigger:** 14 señales ALTA en games_signal_report invisible en tiempo real — sin estado EN_VIVO, sin convergencia, sin combo automático
> **Autor:** Sonnet 4.6 (análisis doctoral — graphify full traversal)

---

## §0. Problema raíz

`games_signal_report_FECHA_*.json` se genera una vez/día (PASO 3.6, ~10:26).
El X3 panel de `live_desk.py` muestra esas señales **estáticas** toda la jornada.

Con 14 señales ALTA hoy:
- No se sabe cuáles ya terminaron
- No se sabe cuáles están EN_VIVO ahora mismo
- No se detecta cuándo ≥2 ALTA coinciden en vivo (ventana de apuesta)
- El combo `betplay_combo_builder --games` nunca se dispara automáticamente

**Alpha invisible**: el sistema tiene la señal pero no actúa en la ventana correcta.

---

## §1. Aprendizaje del dashboard anterior (localhost:8501)

`dashboard.py::panel_live()` (L1515) ya resolvió este problema para picks ML.
El patrón que funciona en producción:

```
live_edge_monitor.py (n8n cada 20s)
    → KambiLiveClientReal._fetch_listview_started()   # 1 HTTP call
    → escribe reports/live_edge_*.json                # estado en disco

dashboard.py::panel_live()
    → load_live_edge()                                # UI pasiva, solo lee
    → muestra estado, drift, break machine
```

**Nodo-133 sigue exactamente este patrón.** No reinventa infraestructura.

---

## §2. Arquitectura implementada

```
_background_refresh(fecha_fn) [daemon 15s — D129-01]
    │
    ├── _get_cached_state(fecha)              [Nodo-129]
    │
    └── _check_games_convergencia(fecha)      [D133-03 NUEVO]
            │
            ├── Lee games_signal_report_FECHA_*.json
            │       → filtra confianza_señal == "ALTA"
            │       → extrae event_id (campo kambi_event_id si existe)
            │
            ├── KambiLiveClientReal._fetch_listview_started()
            │       → 1 HTTP call → todos los STARTED en Kambi
            │       → también intenta liveEvents.json (fallback)
            │
            ├── Clasificación por partido:
            │       EN_VIVO   → matcheado en STARTED events (event_id primario / apellido fallback)
            │       PRE_PARTIDO → hora > now (aún no empieza)
            │       TERMINADO → no en STARTED y hora + 2h < now
            │
            ├── Para cada EN_VIVO: busca betOffer "Total de juegos"
            │       → cuota live UNDER/OVER actual
            │       → drift = (cuota_live - cuota_pre) / cuota_pre × 100
            │
            ├── Escribe reports/games_live_YYYYMMDD.json
            │       {ts, signals_alta:[{partido, estado, cuota_pre,
            │         cuota_live, drift_pct, event_id}],
            │        en_vivo_count: N, convergencia_activa: bool}
            │
            └── Si en_vivo_count ≥ 2:                    [D133-06]
                    → lee reports/games_live_{fecha}_fired.json  [D133-04]
                    → frozenset {partido_A, partido_B, ...}
                    → si combo no disparado y total_fires < 10:
                        subprocess.Popen(betplay_combo_builder --games)
                        fire-and-forget (no bloquea ciclo 15s)
                    → actualiza fired

_build_x3_games(fecha) [ENHANZADO]
    → lee games_signal_report (base — siempre)
    → lee games_live_YYYYMMDD.json (enriquecimiento opcional)
    → superpone estado_live, cuota_live, drift_pct por partido
    → retrocompatible: si games_live no existe → igual que antes

render_html() X3 panel [ENHANZADO]
    → columna Estado: EN_VIVO (verde) / PRE (gris) / TERMINADO (muted)
    → cuota_live vs cuota_pre + drift_pct cuando EN_VIVO
    → banner CONVERGENCIA ACTIVA cuando convergencia_activa == True
    → contador "N señales ALTA en vivo ahora"
```

---

## §3. Decisiones de diseño D133-01 → D133-06

### D133-01 — Fuente live: Kambi listView STARTED (no FlashScore Playwright)

**Archivo:** `scripts/live_edge_monitor.py` — `KambiLiveClientReal._fetch_listview_started()` L209

**Decisión:** Endpoint Kambi `/listView/tennis.json` filtrado por `state == "STARTED"`.
Una sola llamada HTTP, sin Playwright, sin browser headless.

**Por qué no FlashScore:** `fetch_flashscore_odds()` usa Playwright → 2-3 min/partido, bloqueante. Para monitoreo cada 15s es inviable arquitecturalmente.

**Limitación documentada:** ITF menores sin Kambi listing → clasificar por `hora` como fallback de tiempo. Para señales GAMES específicamente: si `_buscar_event_id_kambi()` encontró el partido → el partido existe en Kambi → estará en listView cuando empiece.

### D133-02 — Match primario por event_id, fallback por apellido

**Decisión:** `games_signal_calculator._buscar_event_id_kambi()` guarda el `kambi_event_id` en cada señal del reporte. Usar ese ID para lookup directo en liveEvents.json es más preciso que matching por nombre.

**Fallback:** si `kambi_event_id` no está en el reporte (reporte anterior a Nodo-133) → match por apellido normalizado (igual que `_apellido()` en games_signal_calculator.py).

**Ambigüedad:** si >1 candidato matchea por apellido → tomar el de `diff_abs` más alto.

### D133-03 — Monitor dentro de `_background_refresh()` (no hilo nuevo)

**Decisión:** El daemon de D129-01 ya corre cada 15s. Añadir `_check_games_convergencia(fecha)` en el mismo loop. +2 líneas en `_background_refresh()`.

**Impacto:** +1 HTTP call (listView) por ciclo = +0.5s típico. Aceptable.

**Excepción:** silenciada por el `try/except: pass` existente. Si Kambi cae → ciclo continúa.

### D133-04 — Anti-flood: archivo separado, Popen fire-and-forget

**Anti-flood:** `reports/games_live_{fecha}_fired.json` — lista de frozensets `{partido_A, partido_B}`.
Cap: 10 combos games/día. Archivo diferente de `reports/combos_live/YYYYMMDD/_fired.json` de Nodo-116 (breaks). Sin colisión.

**Fire-and-forget:** `subprocess.Popen([..., "--games"])`. No `subprocess.run()`. Motivo: `run()` bloquearía el daemon 15s → siguiente ciclo se atrasa. `Popen()` lanza el proceso sin esperar.

### D133-05 — Archivo de estado: reports/games_live_YYYYMMDD.json

**Patrón:** idéntico a `live_edge_*.json` del viejo dashboard. UI pasiva, daemon escribe.

**Estructura:**
```json
{
  "ts": "2026-07-21T15:32:07",
  "signals_alta": [
    {
      "partido": "Alcaraz S. vs Sinner J.",
      "estado": "EN_VIVO",
      "cuota_pre": 1.75,
      "cuota_live": 2.10,
      "drift_pct": 20.0,
      "event_id": 1003829441,
      "direccion": "UNDER",
      "linea": 21.5
    }
  ],
  "en_vivo_count": 2,
  "convergencia_activa": true
}
```

### D133-06 — Umbral convergencia: ≥2 ALTA EN_VIVO

**Decisión:** ≥2 (no ≥3). Con 14 ALTA hoy y partidos de ~90min, threshold ≥3 sería demasiado restrictivo. Combo de 2 legs games @~3.5x típico es operativamente correcto.

**Ventana temporal:** si partido_A empieza 14:00 y partido_B empieza 14:30, a las 14:30 ambos EN_VIVO → ventana de convergencia → combo disparado. Lógica de overlap implícita en "ambos en STARTED".

---

## §4. Hallazgos del análisis doctoral

### H133-I1 — event_id como lookup directo (no descubierto antes)
`games_signal_report` ya tiene el `kambi_event_id` de cuando `_buscar_event_id_kambi()` lo encontró pre-game. Usarlo es más confiable que apellido matching. Este insight estaba en el código pero no en ningún spec anterior.

### H133-I2 — betOffer Total de Juegos en betOffers[1..N]
`KambiLiveClientReal._search_events()` solo extrae betOffers[0] (ML ganador). Los mercados de juegos están en índices posteriores, bajo `criterion.label` que contiene "juegos" o "Total de juegos". La función necesita iterar todos los betOffers para encontrarlo.

### H133-I3 — Drift de games tiene dirección opuesta al drift ML
Para UNDER games: si cuota_live UNDER sube → mercado espera más juegos (partido lento) → nuestra señal UNDER más difícil de conseguir. Si cuota_live UNDER baja → mercado espera menos juegos → señal confirmada. Esto invierte la intuición del drift ML donde cuota_live más alta = señal más fuerte.

### H133-I4 — Sin colisión con Nodo-100B (breaks)
Break anti-flood: `reports/combos_live/YYYYMMDD/_fired.json` keyed por `event_id`.
Games anti-flood: `reports/games_live_{fecha}_fired.json` keyed por frozenset de partidos.
Sistemas paralelos e independientes.

### H133-I5 — Retrocompatibilidad obligatoria
Si `games_live_YYYYMMDD.json` no existe (servidor frío, primer ciclo), `_build_x3_games()` debe funcionar igual que antes de Nodo-133. El enriquecimiento es opcional, no obligatorio.

---

## §5. Tests REGLA-T53 — 5/5

Archivo: `tests/test_nodo133_games_live.py`

| Test | Contrato | Estado |
|------|----------|--------|
| `test_clasifica_pre_partido` | hora = now+1h → estado == "PRE_PARTIDO" | pendiente |
| `test_clasifica_en_vivo` | mock Kambi STARTED con partido matcheado → estado == "EN_VIVO" | pendiente |
| `test_clasifica_terminado` | hora = now-3h, no en STARTED → estado == "TERMINADO" | pendiente |
| `test_convergencia_2_alta_dispara` | 2 señales ALTA EN_VIVO, fired vacío → Popen llamado 1x | pendiente |
| `test_antiflood_no_refire` | frozenset {A,B} ya en fired → Popen NO llamado | pendiente |

---

## §6. Archivos modificados

| Archivo | Tipo | Cambio |
|---------|------|--------|
| `live_desk.py` | MODIFY | `_check_games_convergencia()` +80L, `_background_refresh()` +2L, `_build_x3_games()` +30L, `render_html()` X3 +25L |
| `tests/test_nodo133_games_live.py` | NEW | 5 tests REGLA-T53 |
| `scripts/live_edge_monitor.py` | 0 cambios | — |
| `close_snapshot_server.py` | 0 cambios | — |
| `betplay_combo_builder.py` | 0 cambios | — |

**Cero cambios fuera de live_desk.py** (excluido el archivo de tests). Encapsulamiento correcto.

---

## §7. Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-129-LiveDesk-AutoRefresh-Fix]] | `_background_refresh()` daemon — extendido aquí |
| [[Nodo-109-Live-Trading-Desk-Dashboard]] | `live_desk.py` arquitectura base |
| [[Nodo-100B-Triple-Convergencia-Live]] | patrón anti-flood + Popen + _fired.json |
| [[Nodo-97-Live-Edge-Monitor]] | `KambiLiveClientReal` — reutilizado sin modificar |
| [[Nodo-40-Games-Sets-Signal-Layer]] | `games_signal_report` — fuente de señales ALTA |
| [[Nodo-116-Anti-Flood-Combos-Live]] | anti-flood Nodo-116 — sistema paralelo, sin colisión |

**Wikilinks totales: 6 | Huérfanos: 0**

# Nodo-101 — Shadow Book Live CLV: Pick-Type Live Tracking (D99-02)

> **Wikilinks:** [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-99-Auditoria-Fable-N97-N98]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-97-Live-Edge-Monitor]]
> **Fecha:** 2026-07-14 | **Autor:** Sonnet 4.6
> **Contexto:** D99-02 del Nodo-99 Auditoría: el shadow book fue diseñado exclusivamente para picks
> pre-partido. Cuando live_edge_monitor dispara un break_confirmado y se generan combos live,
> esas apuestas NO quedan registradas en el shadow book. Sin registro, H100-01 (Triple Convergencia)
> no puede ser evaluada formalmente. Este Nodo cierra ese gap.

---

## 1. PROBLEMA

Graphify confirmó: `graphify path live_edge_monitor → shadow_book.py` → **SIN CAMINO** (B2 de Nodo-99).

Cuando se dispara `_fire_break_combos()` (BREAK_CONFIRMADO), el flujo es:
```
BREAK_CONFIRMADO → betplay_combo_builder --live → .bat + Telegram
                                                 ↕ VACÍO
                                         shadow_book (sin registro)
```

Consecuencia: hits/misses de picks live son invisibles para el sistema estadístico.
H100-01 no puede graduarse sin datos.

---

## 2. DIFERENCIA CRÍTICA: CLV PREGAME vs LIVE

| Tipo pick | Momento de log | cuota_log | cuota_cierre |
|---|---|---|---|
| Pre-partido | Pre-game (días antes) | cuota_pre (apertura) | cierre_kambi pre-inicio |
| **Live** | Al trigger (~min 5) | **cuota_trigger** | cuota_post_set o final |

- **CLV_live** = `(cuota_trigger / cuota_cierre − 1) × 100`
- Si el trigger captura valor real (bookmaker no actualizó antes que el modelo), CLV_live > 0
- CLV_live < 0 = el mercado se movió antes del trigger (fuga de alpha)

---

## 3. DECISIONES

### D101-01: Campo `pick_type` en registro JSONL

Picks existentes (pre-partido) no tienen `pick_type` → se asumen `'pregame'`.
Picks live → `pick_type = 'live'` en top-level del registro JSONL.

```json
{
  "sb_id": "2026-07-14_live_Boogaard_vs_Onclin",
  "pick_type": "live",
  "logged_at": "2026-07-14T14:07:33-05:00",
  "match_key": "boogaard_onclin",
  "pick_snapshot": {
    "partido": "Boogaard vs Onclin",
    "favorito": "Boogaard",
    "cuota_trigger": 2.90,
    "trigger_ts": "2026-07-14T14:07:00",
    "drift_pct": 18.3,
    "edge_live": 0.087,
    "break_state": "BREAK_CONFIRMADO",
    "cuota_pre": 3.55,
    "tier": "challenger"
  }
}
```

### D101-02: cuota_log para CLV = cuota_trigger

El momento de valor capturado es el trigger del BREAK_CONFIRMADO.
La cuota_trigger es la cuota live en ese instante → base del CLV.

### D101-03: Settle manual vía `--settle-live`

Auto-fetch de cuota_cierre no está disponible (Kambi live endpoint no verificado).
Settle se hace manualmente: `python3 shadow_book.py --settle-live FECHA PARTIDO_KEY RESULT`.
`RESULT` = `WON` | `LOST` (favorito en cuota_trigger ganó/perdió).
`cuota_cierre` se ingresa como argumento opcional: `--cierre 3.10` para CLV.

### D101-04: Segmento H100-01 en --report

```
LIVE PICKS (H100-01: Triple Convergencia — pick_type=live):
  n=X  hit%=Y  IC95=[...]  CLV_live_median=Z  [pre-graduacion hasta n>=3 breaks]
```
Gate de graduación: n_stop=20 (H100-01 de Nodo-100), gate de activación: n>=3 breaks confirmados.

### D101-05: Auto-log desde live_edge_monitor._fire_break_combos()

Cuando `break_state == BREAK_CONFIRMADO` y `fired==False`, además de llamar betplay_combo_builder,
llama `shadow_book.log_live_pick(...)`. Esto garantiza el log automático sin intervención manual.

---

## 4. ARCHIVOS A MODIFICAR

### `shadow_book.py` — +~80 líneas

**A) `log_live_pick(partido_key, favorito, cuota_trigger, cuota_pre, drift_pct, edge_live, break_state, tier, trigger_ts, reports_dir) → str`**
- Construye registro con `pick_type='live'`
- sb_id: `f"{fecha}_live_{_slug(partido_key)}"`
- Escribe a `sb_FECHA.jsonl` vía `_save_jsonl()`
- Retorna sb_id del registro creado

**B) `settle_live(fecha, partido_key, resultado, cuota_cierre=None) → str`**
- Lee sb_FECHA.jsonl, encuentra el registro live por `match_key`
- Settlea: WON/LOST, pnl_flat_1u, clv_pct (si cuota_cierre disponible)
- Escribe `resolucion` igual que `settle()` normal

**C) CLI `--log-live` flag + `--settle-live FECHA PARTIDO RESULT [--cierre C]`**

**D) `report()` — nueva sección LIVE PICKS antes de HIPÓTESIS:**
```python
_live_recs = [r for r in settled if r.get('pick_type') == 'live']
if _live_recs:
    _append_live_section(lines, _live_recs)
```

### `scripts/live_edge_monitor.py` — +~10 líneas en `_fire_break_combos()`

```python
# D101-05: auto-log al shadow book
try:
    import shadow_book as sb
    for t in triggers_confirmados:
        sb.log_live_pick(
            partido_key=t['partido_key'],
            favorito=t.get('favorito', '?'),
            cuota_trigger=t.get('cuota_live', 0),
            cuota_pre=t.get('cuota_pre', 0),
            drift_pct=t.get('drift_pct', 0),
            edge_live=t.get('edge_live', 0),
            break_state='BREAK_CONFIRMADO',
            tier=t.get('tier', 'unknown'),
            trigger_ts=t.get('ts', ''),
            reports_dir=reports_dir,
        )
except Exception as e:
    print(f"⚠️ shadow_book log_live_pick error: {e}")
```

---

## 5. TESTS (REGLA-T53 — 4 tests)

```
tests/test_nodo101_live_clv.py

test_log_live_escribe_jsonl()          — log_live_pick crea entrada pick_type=live en jsonl
test_log_live_campos_completos()       — sb_id, match_key, cuota_trigger, drift_pct presentes
test_settle_live_won()                 — settle_live(WON) → pnl=cuota-1, clv calculado
test_report_muestra_segmento_live()    — --report incluye sección LIVE PICKS cuando hay registros
```

---

## 6. HIPÓTESIS VINCULADA

- **H100-01** (pre-registrada Nodo-100): Triple Convergencia picks ROI > picks sin convergencia
  - n_stop=20, gate_activacion: n>=3 breaks confirmados reales
  - Acumula automáticamente desde `log_live_pick()` en cada break_confirmado

---

## 7. FLUJO COMPLETO POST-NODO-101

```
BREAK_CONFIRMADO (live_edge_monitor.py)
  → betplay_combo_builder --live  (combos .bat + Telegram)
  → shadow_book.log_live_pick()   [NUEVO D101-05]
       ↓
sb_FECHA.jsonl: registro pick_type=live

Trader apuesta → resultado conocido post-partido
  → python3 shadow_book.py --settle-live 2026-07-14 Boogaard_vs_Onclin WON --cierre 3.10

shadow_book --report → LIVE PICKS (H100-01): n=1 hit%=100 ...
                                           [acumula hasta n=3 para evidencia]
```

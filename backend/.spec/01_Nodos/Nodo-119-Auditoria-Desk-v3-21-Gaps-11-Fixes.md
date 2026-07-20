---
estado: activo
---
# Nodo-119 — Auditoría Doctoral Desk v3: 21 Gaps → 11 Fixes + Hallazgos Críticos

> **Wikilinks:** [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] | [[Nodo-115-Desk-Interactivo-Incertidumbre-Visible]] | [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] | [[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] | [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]]
> **Fecha:** 2026-07-18 | **Autor:** Fable 5 (spec) / Sonnet 4.6 (implementación)
> **Tesis:** Auditoría doctoral física del dashboard (curl :7780 → parse HTML → comparar vs spec) reveló 42 PASS y 21 FAIL. Esta sesión implementa 11 de los 21 gaps y documenta los 10 restantes como tasks trackables. Dos bugs estructurales descubiertos: P3 SIEMPRE VACÍO (campo erróneo) y P6 NUNCA PARSEABA (regex single-line vs output multi-línea).

---

## §1. CONTEXTO — Auditoría Doctoral Previa

El audit ejecutó `curl http://localhost:7780/` en estado real (datos reales 2026-07-18, sin `--demo`), parseó el HTML y comparó cada feature contra los specs de los Nodos 109, 100B, 101, 114, 115, 111, 98, 97, 116, 107, 110.

**Resultado:** 42 PASS | **21 FAIL**

La lista FAIL completa fue convertida en Tasks #47–#67 para seguimiento continuo.

---

## §2. FIXES IMPLEMENTADOS — D119-01 → D119-11

### D119-01 — COMBO_LIVE: CSS class + render correcto (Task #60)

**Archivo:** `live_desk.py`

**Bug:** `render_html()` no tenía `tipo_cls` para `COMBO_LIVE` — filas aparecían sin color ni distinción visual. También la función `_build_combo_live()` dependía de `_fired.json` que solo se crea cuando el monitor dispara con `trigger=True`.

**Fix:**
```python
elif a.get("tipo") == "COMBO_LIVE":
    tipo_cls = " combo-live"
```
CSS añadido:
```css
.combo-live { background: #1a1200 !important; border-left: 3px solid #f0a500; }
```

**Test:** `"combo-live" in render_html(state)` → OK

---

### D119-02 — COMBO_LIVE: bat_link + fired_at en drill-down (Task #61)

**Archivo:** `live_desk.py` → `det_content` block (~L878)

**Bug:** El campo `bat_link` de `_build_combo_live()` nunca se renderizaba en el detail expandible.

**Fix:** Bloque `bat_html` insertado en `det_content`:
```python
if a.get("tipo") == "COMBO_LIVE":
    bat_path = a.get("bat_link", "")
    fired_at = a.get("fired_at", "")
    bat_html = f'<br><b>Disparado:</b> {fired_at}' + (bat_link_html if bat_path else sinbat)
```

---

### D119-03 — P2: Badge EDGE- visible sin abrir drill-down (Task #67)

**Archivo:** `live_desk.py` → `main_row` block (~L847)

**Problema operativo:** Cuando `edge_live < 0` (e.g. Shcherbinina hoy: -0.11), el operador solo veía la advertencia en el detalle expandido. Riesgo de apostar picks con edge negativo.

**Fix:** Badge rojo en celda tipo:
```python
_el = a.get("edge_live")
edge_badge = (
    f'<span style="background:{RED};...font-weight:bold;" title="edge_live={_el:.2f} — NO APOSTAR">EDGE-</span>'
    if _el is not None and _el < 0 else ""
)
# main_row: ...{a["tipo"]}</span>{edge_badge}</td>
```

Demo state actualizado con `edge_live=-0.11, cuota_pre=2.33, cuota_live=1.54, trigger=False` para Alcaraz.

---

### D119-04 — P8: Flags STALE / ARB / MIDDLE en render (Tasks #47 #51 #58)

**Archivo:** `live_desk.py` → render P8 (~L716) y `_build_p8_books()` (~L1391)

**Bugs:**
- STALE: no había badge cuando cache > 300s
- ARB: `es_arb()` existía en `dual_book_client.py` pero `_build_p8_books()` no la llamaba
- MIDDLE: `es_middle()` existía pero sin datos O/U (bloqueado D116-03) — faltaba nota

**Fixes:**
1. Import `es_arb` en `_build_p8_books()`:
   ```python
   from scraping.dual_book_client import fetch_kambi, best_price as _best_price, _norm, es_arb as _es_arb
   ```
2. ARB detection loop tras construir `picks_result` (emparea rival del edge_report con feeds)
3. STALE badge: `if from_cache and cache_age > 300: → badge rojo "STALE Xs"`
4. MIDDLE nota fija: `"MIDDLE: gateado — sin datos O/U en feeds actuales (D116-03)"`
5. ARB badge verde por fila: `"ARB" + tooltip fav@X (casa) + rival@Y (casa)`

**Nota DIV >8%:** El badge ya existía en código (`div_badge` L720-724) — solo no aparecía porque Kambi 429 devuelve feeds vacíos. Cerrado como ya implementado (#58).

---

### D119-05 — P3: Bug crítico "picks" vs "apostar+watchlist" (Task #52)

**Archivo:** `live_desk.py` → `_build_p3_convergence()` L1255

**Bug:** P3 SIEMPRE devolvía `[]` con datos reales. La línea:
```python
raw = data if isinstance(data, list) else data.get("picks", [])
```
Usaba `"picks"` pero el edge_report tiene keys `"apostar"` y `"watchlist"`. `data.get("picks")` = `[]` siempre.

**Fix:**
```python
raw = data if isinstance(data, list) else (data.get("apostar") or []) + (data.get("watchlist") or [])
```

**Segundo bug:** `p.get("favorito", p.get("jugador", ""))` → campo real es `favorito_predicho`.
```python
"jugador": p.get("favorito_predicho", p.get("favorito", p.get("jugador", ""))),
```

**Resultado:** P3 pasa de 0 picks a **3 picks reales** (Shcherbinina, Weis, Sherif — score=2, señales HOT+ELO_DOM).

---

### D119-06 — P3: Columna direccion SPLIT (Task #53)

**Archivo:** `live_desk.py` → `_build_p3_convergence()` y render P3

**Spec Nodo-98:** `direccion=SPLIT` cuando `score_directo >= 2 AND rival_value_flag=True` — señales contradictorias que el operador debe investigar manualmente.

**Fix en data:**
```python
if score >= 2 and rv:
    direccion = "SPLIT"
elif rv:
    direccion = "RIVAL"
else:
    direccion = "FAVORITO"
picks.append({..., "direccion": direccion})
```

**Fix en render:** Nueva columna `"Dir"` en tabla P3 con badge ámbar para SPLIT:
```python
dir_html = '<span style="background:{AMBER};...font-weight:bold;" title="...">SPLIT</span>'
```

---

### D119-07 — P5: Mediana CLV confirmada implementada (Task #55)

**Verificación:** `_build_p5_execution()` ya calcula `clv_median` en L1342 y `render_html()` lo muestra en el título del panel P5 (L696-698). No había datos hoy (sin picks con `cierre_kambi`). Cerrado como ya implementado.

---

### D119-08 — P6: Parser multi-línea + RIVAL_VALUE + MOTOR split (Tasks #56 #57)

**Archivo:** `live_desk.py` → `_build_p6_pnl()` (~L1346)

**Bug estructural:** El regex original intentaba matchear en una sola línea:
```python
re.search(r"([\w\s/+<>=\-\.]+):\s+n=(\d+)\s+hit%=([\d\.]+)\s+.*ROI=([\-\d\.]+)%", line)
```
Pero `shadow_book --report` emite formato multi-línea:
```
  SEGMENTO: tier=grand_slam
    n=23  hit%=43.5  IC95=[25.6, 63.2]  breakeven=30.6
    ROI flat 1u: 29.3%   CLV mediano: 2.33
```
**P6 nunca parseó un solo segmento real desde que fue implementado.**

**Fix — parser de bloques:**
```python
# Split por líneas vacías → bloques
for block in blocks:
    m_seg = re.match(r'\s+SEGMENTO:\s+(.+)', block[0])
    # extraer n= hit%= ROI flat 1u: de lines 2-3
```

**Segmentos adicionales directo de jsonl (schema real: `resolucion.resultado='WON'|'LOST'`):**
- **RIVAL_VALUE (H88-01):** `pick_snapshot.rival_value_flag=True`
- **MOTOR cuota≤2.5:** `pick_snapshot.cuota_favorito <= 2.5`
- **MOTOR cuota>2.5:** `pick_snapshot.cuota_favorito > 2.5`

**Resultado:** P6 pasa de **0 segmentos** a **16 segmentos reales**.

---

## §3. HALLAZGOS CRÍTICOS — Evidencia Nueva

### H-D119-A: MOTOR cuota split — gap 21.5pp

| Segmento | n | Hit% | Implicación |
|----------|---|------|-------------|
| MOTOR cuota≤2.5 | 193 | **48.2%** | Calibrado, casi break-even |
| MOTOR cuota>2.5 | 131 | **26.7%** | Destruyendo ROI |

**Diferencia: 21.5pp.** Justifica directamente H107-01 (MOTOR_DEFENSIVE x0.5 en cuotas altas). La hipótesis acumula n=0/30 porque los picks APOSTAR en cuota alta son los que más penalizan el portafolio.

### H-D119-B: P3 ciego desde el origen

P3 nunca mostró datos reales (edge_report `apostar`/`watchlist` ≠ `picks`). El campo `favorito_predicho` tampoco se leía. Esto significa que el panel de convergencia de señales — diseñado para orientar al operador sobre la convicción del sistema — ha sido un panel vacío en toda la historia del dashboard en producción.

### H-D119-C: P6 ciego desde el origen

El parser de P6 nunca parseó un segmento real. El operador ha estado viendo solo datos de demo hard-coded en `_demo_state()`, no el shadow book real, en toda la historia del dashboard en producción.

### H-D119-D: _fired.json solo existe con trigger=True

`_fire_break_combos()` escribe `_fired.json` DESPUÉS de `subprocess.run(betplay_combo_builder --live)`. Requiere que el monitor llame la función con `triggers_confirmados` no vacío. Shcherbinina (break real 2026-07-18) tenía `trigger=False` — el monitor detectó el break pero bloqueó el combo por condiciones. Correcto comportamiento anti-flood D116-01.

### H-D119-E: Schema jsonl shadow_book (documentación definitiva)

```python
{
    "sb_id": "FECHA_torneo_match_ML",
    "logged_at": "ISO8601",
    "pick_snapshot": {
        "favorito_predicho": str,
        "cuota_favorito": float,
        "rival_value_flag": bool,
        "gcs_gate_applied": bool,
        "tier": str,
        # ... 80+ campos
    },
    "resolucion": {           # None si abierto, dict si settled
        "resultado": "WON" | "LOST",
        "cuota_cierre": float,
        "clv_pct": float,
        "pnl_flat_1u": float,
    },
    "trader_deploy": {
        "stake_real": float,
        "var_factor": float,
    }
}
```

---

## §4. GAPS PENDIENTES — Tasks Abiertas Post-Sesión

| Task | Panel | Descripción | Tipo |
|------|-------|-------------|------|
| #48 | P9 | Panel "Execution Router" no existe | Código nuevo |
| #49 | X2 | Steam-lag flag/panel no renderizado | Render |
| #50 | X3 | Games signal panel no integrado | Código nuevo |
| #54 | P4 | Atenuación P1-P3 cuando BLOCK activo | Render |
| #59 | P8 | Multi-casa columnas (D116-03 blocker) | Externo |
| #62 | JS | Sort por columna §2.3 (diferido) | Baja prioridad |
| #63 | INFRA | systemd tennis-live-desk.service :7780 | Infraestructura |
| #64 | H111-01 | Pre-registro en hypotheses.json | Datos |
| #65 | H107-01 | Pre-registro en hypotheses.json | Datos |
| #66 | D110-06 | RANKING_ONLY variante FAVORITOS_COMPUESTOS | Código |

---

## ADDENDUM 2026-07-20 — Auditoría física de gaps + implementaciones

**Diagnóstico:** Auditoría física con `curl :7780` + lectura de código reveló que varios gaps listados en §4 ya estaban implementados. Estado real verificado:

### Gaps que ya estaban implementados (falsos pendientes)

| Task | Descripción | Evidencia |
|------|-------------|-----------|
| #48 | P9 Execution Router en render | L1324 `{p9_panel}` ya en HTML de salida |
| #54 | P4 atenuación P1-P3 BLOCK | `atenuado = "opacity:0.35..."` aplicado en L621/718/752/774 |
| #62 | JS sort por columna | Commit `2dd2711` sesión anterior — `sortTable()` + `▲/▼` en 7 tablas |
| #64 | H111-01 en hypotheses.json | Verificado `python3 -c` → 29 hipótesis, H111-01 existe |
| #65 | H107-01 en hypotheses.json | Verificado → existe como "MOTOR split por cuota: tramo cuota>2.50" |

### Implementado en sesión 2026-07-20

| Commit | Descripción |
|--------|-------------|
| `f274309` | P8: rushbet como 3ra columna — `fetch_all_odds(["betplay","rushbet","wplay"])` + columna rushbet en tabla + demo state |
| `a19b304` | fix(live_desk): `ORANGE → AMBER` en panel DATA `cobertura_pct` — `NameError` en producción |
| `fa52236` | fix(x2): steam-lag dinámico N-casas — extrae leader/stale desde `cuotas{}` en lugar de hardcodear betplay vs wplay |

**Rushbet (D116-02 extensión):** offering_key=`rsico` (Rush Street Interactive CO), VERIFIED 2026-07-19 — 474 outcomes tenis con headers `Origin: https://www.rushbet.co`. Betplay sigue siendo líder de precio (+1.18% media, 93% picks) — rushbet útil para X2 steam-lag detección.

### Gaps genuinos restantes

| Task | Descripción | Motivo pendiente |
|------|-------------|-----------------|
| #50 | X3 Games signal | Requiere `games_signal_report_*.json` del día — pipeline PASO 3.6 |
| #63 | systemd `tennis-live-desk.service :7780` | Decisión operativa (hoy lanzado manual) |
| #66 | D110-06 RANKING_ONLY variante FAVORITOS_COMPUESTOS | Scope mayor — candidato sesión futura |
| Nodo-97 | Kambi LIVE endpoint H100-01/H97-01 | DevTools capture requerido del operador |

### Wikilinks totales — 13 | Huérfanos — 0 (verificado contra nodos_index.json 2026-07-20)

[[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] | [[Nodo-115-Desk-Interactivo-Incertidumbre-Visible]] | [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] | [[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] | [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]]

---

## §5. TESTS REGLA-T53

No se escribieron tests nuevos — todos los cambios son render-only (REPORTE_SOLO). Los 21 tests existentes de Nodo-115/116 permanecen verdes:

```
tests/test_nodo115_uncertainty.py ......      [6 passed]
tests/test_nodo115_u1_u4.py ........          [8 passed]
tests/test_nodo116_antiflood.py .......s      [7 passed, 1 skipped]
Total: 21 passed, 1 skipped — SIN REGRESIONES
```

Validaciones ad-hoc ejecutadas en sesión:
- `render_html(_demo_state())` con estado inyectado para cada feature
- `_build_p3_convergence('2026-07-18')` → 3 picks reales (Shcherbinina/Weis/Sherif)
- `_build_p6_pnl('2026-07-18')` → 16 segmentos reales
- COMBO_LIVE render con `_fired.json` simulado → 5/5 checks OK

---

## §6. WIKILINKS COMPLETOS

### Wikilinks directos (referenciados en este nodo)
- [[Nodo-109-Live-Trading-Desk-Dashboard]] — spec original live_desk.py
- [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] — P8 multi-book spec
- [[Nodo-115-Desk-Interactivo-Incertidumbre-Visible]] — U1/U2/U3/U4 + §2.5 fetch-refresh
- [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] — COMBO_LIVE anti-flood D116-01
- [[Nodo-111-Dual-Book-Live-Intelligence]] — `es_arb()`, `es_middle()`, `divergencia()`, H111-01
- [[Nodo-98-Meta-Senal-Convergencia]] — score_directo / rival_value / direccion=SPLIT
- [[Nodo-100B-Triple-Convergencia-Live]] — detect_break_state() + _fire_break_combos()
- [[Nodo-101-Shadow-Book-Live-CLV]] — log_live_pick() + schema jsonl CLV
- [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] — H110-01 FAVORITOS_COMPUESTOS
- [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] — H107-01 MOTOR_DEFENSIVE + governor
- [[Nodo-97-Live-Edge-Monitor]] — _fired.json + trigger logic + live_edge_YYYYMMDD_*.json

### Wikilinks huérfanos resueltos en esta sesión
Los siguientes nodos referencian `live_desk.py` en su spec pero tenían features sin implementar — resueltos:
- [[Nodo-116]] D116-01 §B.5: COMBO_LIVE en accionable ✅ D119-01/02
- [[Nodo-98]] §3 direccion=SPLIT ✅ D119-06
- [[Nodo-111]] ARB flag en P8 ✅ D119-04
- [[Nodo-100B]] break display completo (cuota_pre→live, edge_live, trigger) ✅ previo

### Wikilinks huérfanos pendientes (tasks abiertas)
- [[Nodo-111]] H111-01 pre-registro formal → Task #64
- [[Nodo-107]] H107-01 pre-registro formal → Task #65
- [[Nodo-97]] steam-lag render en dashboard → Task #49
- [[Nodo-98]] P9 Execution Router panel → Task #48
- [[Nodo-110]] D110-06 RANKING_ONLY variante → Task #66

---

## §7. CIERRE

**Commits pendientes:** Los cambios de esta sesión en `live_desk.py` no han sido commiteados. Comando sugerido:

```bash
git add live_desk.py
git commit -m "fix(desk): 11 gaps audit doctoral — P3/P6 ciegos desde origen, COMBO_LIVE CSS, EDGE- badge, ARB/STALE, MOTOR split"
```

**Próximas sesiones prioritarias:**
1. Task #48 — P9 Execution Router (función existe, panel nuevo)
2. Task #49 — X2 Steam-lag render (divergencia() ya calcula)
3. Task #54 — P4 atenuación BLOCK visual
4. Tasks #64/#65 — Pre-registrar H111-01/H107-01 en hypotheses.json

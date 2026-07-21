# Nodo-125 — EvalGames Bridge: EVALUAR_GAMES → Games Combo Builder + Dashboard X4

> **Estado:** IMPLEMENTADO ✅ — 2026-07-21 (3 bugs post-evidencia corregidos mismo día)
> **Tipo:** FEATURE — bridge señal + dashboard integration + combo time-window
> **Autor:** Sonnet 4.6
> **Tests:** 15/15 PASS (REGLA-T53) — `tests/test_nodo125_evaluar_games_bridge.py`
> **Git:** commit `feat(nodo125)` — sesión 2026-07-21 (base HEAD: `7102203 feat(nodo121+nodo123)`)
> **CLAUDE.md refs:** §5 Estado actual (H125-01 ACUMULANDO, 34 hipótesis, 2219 tests) | §4 Flujo PASO 3.6b | §11 Taxonomía #11 GAMES sub-tipo EvalGamesA

---

## Wikilinks

| Link | Rol | Archivo |
|------|-----|---------|
| [[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]] | Padre — log_evaluar_pick, H124-03, pick_type=evaluar_games | ✅ existe |
| [[Nodo-40-Games-Sets-Signal-Layer]] | games_signal_calculator funciones importadas | ✅ existe |
| [[Nodo-109-Live-Trading-Desk-Dashboard]] | live_desk.py — panel X4 añadido aquí | ✅ existe |
| [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | Taxonomía — estrategia #11 GAMES (nuevo sub-tipo EvalGamesA) | ✅ existe |
| [[Nodo-101-Shadow-Book-Live-CLV]] | shadow_book EVAL_ records leídos por bridge y panel X4 | ✅ existe |
| [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]] | match_id field (D125-01) proviene de match ledger | ✅ existe |

**Wikilinks totales: 6 | Huérfanos: 0**

---

## §1. Diagnóstico del gap

EVALUAR_GAMES (picks cuota<1.30, conf≥54%) registrados en [[Nodo-124]] tienen **hit%=84.6% con n=13** (H124-03, 2026-07-20). Favoritos absolutos que ganan partidos cortos — candidatos UNDER total juegos, no ML directo (KGR<0 por margen bookmaker).

**Gaps cerrados:**
1. EVALUAR_GAMES no llegaban a `games_signal_calculator.py` ni a `betplay_combo_builder.py --games` → **D125-02 bridge**
2. No había panel dedicado en `live_desk.py` → **D125-04 panel X4**
3. El combo builder no agrupaba por ventana horaria (partido 08:00 + 21:00 en mismo combo) → **D125-03 time-window**
4. `pick_snapshot` no incluía `match_id` ni `hora` → **D125-01**

---

## §2. Fixes — todos IMPLEMENTADOS ✅

### D125-01 — match_id y hora en pick_snapshot ✅

**Archivo:** `generar_tabla_favoritos2.py` — bloque D124-01 (~L992)

```python
# D125-01: habilita Kambi lookup exacto + time-window combo grouping
'match_id': match.get('match_id') or match.get('kambi_id'),
'hora':     match.get('hora') or match.get('hora_inicio') or match.get('time'),
```

`match_id` viene del match ledger ([[Nodo-118]]). Prerequisito para que el bridge haga lookup Kambi por ID en lugar de apellidos fuzzy.

---

### D125-02 — `scripts/evaluar_games_bridge.py` (NUEVO) ✅

Bridge EVALUAR_GAMES → UNDER games signal.

**Flujo:**
```
shadow_book EVAL_ (pick_type=evaluar_games)
    → _diff_abs_from_cuota(cuota): (1/cuota − 0.5) × 2
      cuota 1.06 → 0.886 | 1.20 → 0.667 | 1.28 → 0.562  [todos ≥ 0.35 = DOMINANTE]
    → _buscar_event_id_kambi(partido_dict)
    → _fetch_betoffer_event(event_id)
    → _analizar_mercados_juegos(betoffer, pred)
    → _seleccionar_señal_optima(señales)
    → reports/evaluar_games_signal_YYYYMMDD_HHMMSS.json
```

**Imports de `games_signal_calculator`** ([[Nodo-40]]):
`_buscar_event_id_kambi` | `_fetch_betoffer_event` | `_analizar_mercados_juegos` | `_seleccionar_señal_optima` | `_predecir_sets_y_games` | `_zona_diff`

**Output schema:**
```json
{
  "metadata": { "fecha", "generado", "fuente": "evaluar_games_bridge (Nodo-125 D125-02)", "n_picks", "n_con_under", "nodo" },
  "apostar":          [ picks con tiene_mercados=true ],
  "detalle_completo": [ todos los picks ]
}
```

**CLI:** `python3 scripts/evaluar_games_bridge.py [--fecha YYYY-MM-DD] [--dry-run] [-v]`

---

### D125-03 — `build_evaluar_games_combos()` + `--evaluar` en betplay_combo_builder ✅

**Archivo:** `betplay_combo_builder.py`

**Nuevas funciones:**
- `_hora_to_min(hora)` — "HH:MM" → minutos desde medianoche
- `_group_by_time_window(signals, window_min=90)` — greedy cluster: cada pick entra en el primer grupo donde `max−min ≤ window_min`. Picks sin hora van a grupo propio.
- `_find_latest_evaluar_games_signal()` — glob `reports/evaluar_games_signal_*.json`, más reciente
- `build_evaluar_games_combos(stake_per_combo=1000, signal_file=None)` → `(combos, meta)`

**Algoritmo time-window (ejemplo):**
```
picks: [09:00, 10:30, 11:00, 15:00]  ventana=90min
Grupo 1: [09:00, 10:30, 11:00] → span=90min ✅ → combo 3-leg
Grupo 2: [15:00]               → 1 leg → descartado (gate: ≥2 legs)
```

**Gates:**
- ≥2 legs con señal UNDER `apostar=True` en el mismo grupo
- `cuota_combo = ∏(cuota_under)` ≥ 2.50

**Flags en main():** `--evaluar` | `--evaluar-stake N` (default 1000) | `--evaluar-file PATH`

**Output:** `combos/EvalGamesA_TIMESTAMP.bat` + `combos/EvalGamesA_TIMESTAMP.html`

---

### D125-04 — Panel X4 EVALUAR_GAMES en `live_desk.py` ✅

**Archivo:** `live_desk.py`

**Nueva función** `_build_x4_evaluar_games(fecha)` (después de `_build_x3_games()`):
- Lee EVAL_ records (pick_type=evaluar_games) de `sb_FECHA.jsonl` vía [[Nodo-101]] `shadow_book._load_jsonl()`
- Enriquece con `cuota_under` de `evaluar_games_signal_FECHA.json` si existe (match por partido nombre)
- Retorna `{disponible, fecha, picks, n, n_con_under, fuente}`

**Añadido a `build_desk_state()`:**
```python
"p_evaluar_games": _build_x4_evaluar_games(fecha),  # Nodo-125 X4 EVALUAR_GAMES
```

**Panel X4 en `render_html()`** (entre X3 GAMES y P7 CLOCK):
- Título: "X4 EVALUAR_GAMES — favoritos absolutos (cuota<1.30) → UNDER juegos (Nodo-125)"
- Tabla: `Hora | Partido | Conf | CuotaML | CuotaUNDER | Resultado`
- Badge: `"{n} picks / {n_con_under} con UNDER"`
- Colores: GREEN (`n_con_under>0`) | AMBER (picks sin under) | GREY (vacío)

---

### D125-05 — PASO 3.6b en `run_daily.py` ✅

**Archivo:** `run_daily.py` — después de PASO 3.6 (L390)

```python
# ── PASO 3.6b — EvalGames Bridge (Nodo-125) ──────────────────────────────
# EVALUAR_GAMES (cuota<1.30) → UNDER juegos signal para X4 dashboard + combos
_run(['python3', 'scripts/evaluar_games_bridge.py'], 'PASO 3.6b — EvalGames Bridge', optional=True)
```

`optional=True` — si no hay picks evaluar_games hoy, el script sale limpio sin bloquear el pipeline.

---

## §3. Tests REGLA-T53 ✅ — 15/15 passed

**Archivo:** `tests/test_nodo125_evaluar_games_bridge.py`

```
TestD125_02_DiffAbs (4 tests):
  test_cuota_1_10_yields_approx_0_82          — cuota 1.10 → diff≈0.818
  test_cuota_1_20_yields_approx_0_667          — cuota 1.20 → diff≈0.667
  test_cuota_1_28_yields_above_0_35_dominante  — todos cuota<1.30 son DOMINANTE
  test_cuota_zero_returns_safe_default          — cuota=0 → 0.5 (no crash)

TestD125_02_OutputFormat (3 tests):
  test_output_has_required_top_level_keys       — metadata + apostar + detalle_completo
  test_metadata_includes_fuente_nodo125         — "Nodo-125" en fuente
  test_apostar_filters_to_tiene_mercados_true   — apostar solo picks con señal

TestD125_03_TimeWindow (3 tests):
  test_picks_within_90min_form_single_group     — [09:00,10:00,10:30] → 1 grupo
  test_picks_outside_90min_split_into_separate_groups — [09:00,11:00] → 2 grupos
  test_pick_without_hora_forms_own_group        — hora=None → grupo propio

TestD125_03_ComboGate (3 tests):
  test_combo_skips_single_pick_windows          — 1 leg → combos=[]
  test_combo_cuota_gate_requires_min_2_50       — 1.40×1.40=1.96 < 2.50 → descartado
  test_combo_passes_gate_when_cuota_above_2_50  — 1.85×1.90=3.515 > 2.50 → combo generado

TestD125_04_X4Panel (2 tests):
  test_returns_empty_dict_gracefully_when_no_sb_file — sin archivo → n=0, no crash
  test_reads_evaluar_games_picks_from_shadow_book    — EVAL_ JSONL → picks[0] correcto
```

**Regresión global:** 2219 passed, 1 failed pre-existente (`test_nodo51_f3_02`), 0 nuevos fallos.

---

## §4. Hipótesis pre-registradas

### H124-03 (heredada de [[Nodo-124]]) — ACUMULANDO
- Segmento: pick_type=evaluar_games (cuota<1.30, conf≥54%)
- Métrica: hit% resultado partido (favorito gana)
- Éxito: hit% > 70%, IC Wilson 95% inf > 60%, n≥30
- Kill-switch: hit% < 65% con n≥20
- n_stop=30, n_actual=0 (backfill pendiente de settle via resultado_finales)

### H125-01 — ACUMULANDO
- Segmento: combos EvalGamesA (UNDER juegos, 2-3 legs, cuota_combo≥2.50)
- Métrica: hit% combo
- Éxito: hit% combo > 55%, IC Wilson 95% inf > 45%, n≥25 (breakeven @2.50 = 40%)
- Kill-switch: hit% combo < 40% con n≥15
- n_stop=25, n_actual=0

---

## §5. Archivos modificados / creados

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `generar_tabla_favoritos2.py` | D125-01: +2 campos pick_snapshot (`match_id`, `hora`) | ✅ |
| `scripts/evaluar_games_bridge.py` | D125-02: NUEVO ~255 líneas | ✅ |
| `scripts/backfill_evaluar_shadow.py` | D124-05: NUEVO — backfill retroactivo EVAL_ histórico (33 picks, 13 fechas) | ✅ |
| `betplay_combo_builder.py` | D125-03: +3 funciones + `--evaluar` flag + 2 bug fixes post-evidencia | ✅ |
| `live_desk.py` | D125-04: `_build_x4_evaluar_games()` + panel X4 + fix conf normalización | ✅ |
| `run_daily.py` | D125-05: PASO 3.6b (`optional=True`) | ✅ |
| `validation/preregistered_hypotheses.json` | H125-01 pre-registrada (34 hipótesis total) | ✅ |
| `nodos_index.json` | entrada Nodo-125 añadida | ✅ |
| `tests/test_nodo125_evaluar_games_bridge.py` | NUEVO — 15 tests REGLA-T53 | ✅ |
| `.spec/01_Nodos/Nodo-125-*.md` | este archivo | ✅ |
| `.spec/01_Nodos/Nodo-124-*.md` | spec Nodo-124 (creado sesión anterior, incluido en commit) | ✅ |

---

## §6. Post-implementación — Diagnóstico evidencia real (2026-07-21)

**Ejecución real contra Kambi + shadow_book de hoy:**

```
python3 scripts/evaluar_games_bridge.py --dry-run -v
→ 72 picks evaluar_games procesados
→  7 con señal UNDER en Kambi
→ 39 con event_id encontrado | 33 sin event_id (ITF minors fuera de catálogo Kambi)
```

**Señales UNDER reales del día:**
| Partido | Línea | Cuota | Nivel |
|---------|-------|-------|-------|
| Steiner V. vs Noha Akugue N. | UNDER 26.5 | @1.73 | ALTA |
| Kuzuhara B. vs Llamas Ruiz P. | UNDER 21.5 | @1.88 | MEDIA |
| Holmgren A. vs Johns G. | UNDER 21.5 | @1.88 | MEDIA |
| Hurrion M. vs Draxl L. | UNDER 21.5 | @1.80 | MEDIA |
| Crivellaro G. vs Ghazouani Durand Y. | UNDER 21.5 | @1.87 | MEDIA |
| Bayerlova M. vs Mikulskyte J. | UNDER 21.5 | @1.76 | MEDIA |
| Lew Yan Foon A. vs Vismane D. | UNDER 26.5 | @1.96 | ALTA |

**Dashboard X4 verificado:**
```
n=72 picks / n_con_under=7 — badge GREEN ✅
fuente: evaluar_games_signal_20260721_072618.json
conf normalizada: 58.6 → 59% ✅
```

**Combos EvalGamesA hoy: 0** (correcto — `hora=None` porque D125-01 toma efecto mañana)
- 8 señales UNDER → 8 ventanas separadas (1 por pick sin hora) → ningún grupo ≥2 legs

---

## §6b. Bugs encontrados y corregidos en diagnóstico de evidencia

### Bug-1 — `hora=None` formaba 1 grupo masivo ❌→✅

**Síntoma:** todos los picks sin hora iban al mismo grupo → 56 combos EvalGamesA inválidos (C(8,3)) sin saber si los partidos coincidían en horario.

**Causa:** `groups.append(without_hora)` (una lista) en lugar de grupos individuales.

**Fix** (`betplay_combo_builder.py` L2906):
```python
# Antes (bug):
if without_hora:
    groups.append(without_hora)

# Después (fix):
for s in without_hora:
    groups.append([s])   # cada pick sin hora → grupo propio, no combina
```

### Bug-2 — `gap_juegos=None` → crash en `_mostrar_games_combos` ❌→✅

**Síntoma:** `TypeError: unsupported format string passed to NoneType.__format__` en L1834 al formatear `leg['gap_juegos']:.1f`.

**Causa:** EvalGamesA combo legs no tienen `gap_juegos` (campo del games_signal_calculator regular), se guardaba como `None`.

**Fix** (`betplay_combo_builder.py` L2974):
```python
"gap_juegos": s.get("gap_juegos") or 0,   # 0 si no existe (EvalGamesA)
```

### Bug-3 — `confidence=58.6` → `5860%` en display X4 ❌→✅

**Síntoma:** generar_tabla_favoritos2 guarda `confidence` como porcentaje float (58.6), pero `_build_x4_evaluar_games` lo formateaba con `:.0%` (multiplicaba ×100).

**Causa:** convención de formato distinta entre módulos (generar_tabla → percent; edge_calculator → decimal).

**Fix** (`live_desk.py` L524):
```python
"conf": (lambda c: c / 100 if c and c >= 1 else (c or 0))(snap.get("confidence")),
```

---

**Wikilinks totales: 6 | Huérfanos: 0**

[[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-118-Match-Ledger-Crosswalk-Identidad-Fusion-Definitiva]]

# Nodo-115 — Desk v3: Incertidumbre Visible + Interactividad (el analista dentro del dashboard)

> **Wikilinks:** [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-64-RFI-Return-From-Inactivity]] (Fase 4 REPORTE_SOLO, rango 64-71)
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 (spec) | **Implementa:** Sonnet 4.6 (2026-07-17)
> **Tesis:** el desk muestra el razonamiento (Nodo-114 ✅) pero NO la incertidumbre. Un punto (`p=0.62`) sin banda ni n es una promesa, no una medición. La maquinaria YA EXISTE (Fase 4 REPORTE_SOLO: conformal_band, flb_curve, velocity_monitor, rho_empirical) — este nodo la RENDERIZA, cero cálculo nuevo (principio Nodo-109).
>
> **Wikilinks huérfanos corregidos:** `[[Nodo-70]]` y `[[Nodo-71]]` referenciados en el spec original NO EXISTEN como archivos `.md` individuales. `conformal_band.py` está asociado al rango `Nodos-64-71` en `nodos_index.json`. Wikilink correcto: [[Nodo-64-RFI-Return-From-Inactivity]] como cabecera de esa fase.

## §1. LOS 4 NÚMEROS QUE REDUCEN INCERTIDUMBRE (por fila accionable)

| # | Qué | Fuente EXISTENTE | Render | Estado |
|---|---|---|---|---|
| U1 | **Banda conformal** de p_modelo | `conformal_quantile()` + `is_no_bet_conformal()` (`analysis/conformal_band.py:28,46`) | `p=0.62 ±0.09` — si banda cruza breakeven → celda ámbar "BANDA CRUZA BE" | ⏳ pendiente §3-4 |
| U2 | **Peso de la evidencia** | `n_calibracion` en edge_report + shrinkage `n/(n+20)` | Mini-barra `█░░░░ 17% PRIOR MANDA` / `████░ 73%` | ✅ IMPLEMENTADO |
| U3 | **Distancia al gate** | `n_actual`/`n_stop` ya en accionable dict + `preregistered_hypotheses.json` | `H110-01: ███░░░░░░░ 8/30 (22 faltan)` / `GRADUADA` | ✅ IMPLEMENTADO |
| U4 | **Tendencia vs tick** | `velocity_monitor` + odds_history (`load_odds_history()` Nodo-100B) | Sparkline texto `▁▃▅▇` últimos 4 ciclos + flecha | ⏳ pendiente §3-5 |

**Regla de oro:** ninguno de estos cambia decisiones (REPORTE_SOLO se respeta) — cambian cuánto CONFÍA el operador en la decisión que el sistema ya tomó. Es la diferencia entre "el modelo dice 0.62" y "el modelo dice 0.62 con banda ±0.09, 33 casos reales detrás, gate a 22 combos de graduar, y la cuota lleva 3 ciclos moviéndose a favor".

## §2. INTERACTIVIDAD — vanilla JS embebido, cero frameworks, cero backend nuevo

| Ítem | Estado | Notas |
|---|---|---|
| §2.1 Drill-down por clic | ✅ IMPLEMENTADO | `toggleDetalle(rowId)` — expande fila con razonamiento completo + U2/U3/P8/señales |
| §2.2 Facetas client-side | ✅ IMPLEMENTADO | `filtrarTipo(tipo)` — botones TODOS/BREAK/GCS/FAVORITOS_ZERO con `data-tipo` attrs |
| §2.3 Orden por columna | ⏳ pendiente | JS sort por p_modelo/cuota/U2/U3 — valor bajo, diferible |
| §2.4 Panel QUÉ FALTA | ✅ IMPLEMENTADO | `_build_que_falta()`: primera condición fallida con distancia exacta por jugador |
| §2.5 fetch-refresh sin destruir estado | ⏳ pendiente | Reemplazar `<meta http-equiv="refresh">` por `fetch()` + reemplazo de `<tbody>` |

## §3. IMPLEMENTACIÓN — cierre 2026-07-17

### Qué se construyó
- **`_peso_evidencia(n_cal: int) -> dict`** — shrinkage `n/(n+20)`, barra `█░`, 3 niveles: rojo (<20% PRIOR MANDA) / ámbar (20-44%) / verde (≥45%). Fuente: `p0_ncal` lookup desde edge_report en `build_desk_state()`.
- **`_gate_barra(n_actual: int, n_stop: int) -> str`** — barra texto `████░░░░░░ 3/20 (17 faltan)` / `GRADUADA`. Usa campos ya presentes en cada accionable.
- **`_build_que_falta(fecha: str) -> List[Dict]`** — lee `edge_report_*.json` sección `watchlist`, evalúa 3 condiciones (favorito_claro / cuota_rango / model_eq_bookie) con las constantes de `favoritos_combo_builder.py` copiadas localmente (REGLA-T53: no importar el módulo, evitar acoplamiento circular).
- **`p0_ncal`** en `build_desk_state()` — lookup `{nombre_lower: n_calibracion}` construido desde edge_report sin tocar `accionable_ahora()`.
- **`p9_que_falta`** en `build_desk_state()` — lista de casi-accionables serializada.
- **render_html() tabla accionable** — reemplaza `table()` helper por HTML manual con `data-tipo`, `id`, `onclick`. Columnas: Tipo | Jugador/Pick | Evidencia U2 | Gate U3 | Razonamiento.
- **JS embebido** al final de `<body>` — `filtrarTipo()` + `toggleDetalle()`, 20 líneas, sin librerías.
- **`--demo` actualizado** — `p0_ncal` + `p9_que_falta` con 3 casi-accionables realistas (Rune/Rublev/Dimitrov).

### Evidencia de cierre (curl demo :7780)
```
ACCIONABLE — Evidencia U2 + Gate U3:
  BREAK_CONFIRMADO Alcaraz  | ██░░░ 38%          | H100-01: ██░░░░░░░░ 3/20 (17 faltan)
  GCS              Djokovic | ████░ 73%           | H60-01: GRADUADA
  FAVORITOS_ZERO   sin corr | ░░░░░ PRIOR MANDA   | H110-01: ███░░░░░░░ 8/30 (22 faltan)

QUÉ FALTA — 3 candidatos:
  Holger Rune    | favorito_claro    | p_modelo=0.551 < 0.62 (faltan 0.069)
  Andrey Rublev  | cuota_rango       | cuota_fav=2.35 > 2.10 (techo, delta +0.25)
  Grigor Dimitrov| model_neq_bookie  | cuota_fav=1.90 >= cuota_rival=1.85 (bookie discrepa)

JS: filtrarTipo=✅  toggleDetalle=✅  data-tipo=✅  barra█=✅  faltan=✅
```

### Tests REGLA-T53 — 6/6 GREEN (`tests/test_nodo115_uncertainty.py`)
| Test | Qué verifica |
|---|---|
| T1 | `_peso_evidencia(4)` → pct=17, PRIOR MANDA, color rojo |
| T2 | `_peso_evidencia(33)` → pct=62, color verde |
| T3 | `_gate_barra(8, 30)` → contiene "22 faltan" y "8/30" |
| T4 | `_gate_barra(54, 30)` → "GRADUADA" |
| T5 | `_build_que_falta` con pick cuota_fav=2.35 → condicion=cuota_rango, detalle contiene "2.10" |
| T6 | `render_html(_demo_state())` → data-tipo + barra█ + "QUÉ FALTA" + JS |

**Commit:** `867338e` feat(nodo115): incertidumbre visible — U2 + U3 + QUÉ FALTA + drill-down + facetas

## §4. SPEC PARA SONNET — orden por valor/token

1. ~~**U2 + U3**~~ ✅ COMPLETADO
2. ~~**§2.4 panel QUÉ FALTA**~~ ✅ COMPLETADO
3. ~~**§2.1 drill-down + §2.2 facetas**~~ ✅ COMPLETADO
4. **U1 conformal** — llamar `conformal_report()` (`analysis/conformal_band.py`) una vez por ciclo, cachear en state como `p10_conformal`. Render: columna extra en tabla accionable `p=X ±Y`. Gate: si banda cruza breakeven → celda ámbar.
5. **U4 sparkline** — `load_odds_history()` de Nodo-100B, últimos 4 ciclos del jugador → `▁▃▅▇` + flecha tendencia. Cierra ítem TAPE del checklist Nodo-114 §5.
6. **§2.5 fetch-refresh** — reemplazar `<meta http-equiv="refresh" content="30">` por fetch+setTimeout, preserva filtros y filas expandidas.

**PROHIBIDO siempre:** frameworks JS/CDN, cálculo estadístico nuevo en el desk, modificar `accionable_ahora()` o cualquier gate.

## §5. PENDIENTES PARA PRÓXIMAS SESIONES

| Ítem | Prioridad | Gate |
|---|---|---|
| U1 conformal band render | Media | `conformal_band.py` ya existe, solo render |
| U4 sparkline odds_history | Media | `live_odds_history_FECHA.json` ya existe |
| §2.3 sort por columna | Baja | 15 líneas JS |
| §2.5 fetch-refresh | Baja | Pulido UI, no señal |
| Telegram hook BREAK_CONFIRMADO | Media | `enviar_combos_telegram()` ya existe en proyecto |
| systemd tennis-live-desk.service :7780 | Baja | Arranque automático |

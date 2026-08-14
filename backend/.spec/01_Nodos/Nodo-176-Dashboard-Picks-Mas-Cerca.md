# Nodo-176 — Dashboard Picks Más Cerca del Umbral

> Estado: **IMPLEMENTACIÓN PENDIENTE**. Deliverable único: conectar funnel_report.py con
> live_desk.py panel _build_que_falta() para mostrar operador qué picks están más cerca del 
> umbral G_EDGE_MIN cuando 0 apostables. Riesgo de regresión mínimo — render-only, sin tocar
> stake ni Kelly. Baseline: 2699 passed, 0 failed, 3 skipped.

## 1. Deliverable D176-01

### 1a. scripts/funnel_report.py:119
```
Renombrar _mas_cerca() → picks_mas_cerca() (pública).
Dejar alias: _mas_cerca = picks_mas_cerca
Firma y cuerpo SIN CAMBIOS.
```

### 1b. live_desk.py:788 _build_que_falta(fecha)
```
Añadir sección con picks_mas_cerca(watchlist, sin_edge, top_n=3).
Leer el edge_report con _latest() (:2018) + _load_json() (:2025) del propio
live_desk — NO usar find_latest_edge_report() de funnel_report.
Import: from scripts.funnel_report import picks_mas_cerca
```

## 2. NO TOCAR
- `_winner_market_refresh()` (:5047), live_edge_monitor.py — riesgo doble disparo
- `_background_refresh()` (:5030), `_fast_score_refresh()`, `_STATE_CACHE`
- `p_blend`, `edge`, `kelly_kl`, `IRP`, cualquier ruta de stake
- `generar_reporte()` de funnel_report

## 3. Tests REGLA-T53
Invocar `picks_mas_cerca()` real, no hardcodear la fórmula.
Baseline a preservar: 2699 passed, 0 failed, 3 skipped.

## 4. DEUDA DOCUMENTADA — _winner_market_refresh() NO activar

Razón 1 — DOBLE DISPARO DE COMBO (riesgo dinero real):
live_edge_monitor tiene fire-guard propio que Nodo-161 dejó 
deliberadamente sin unificar con _check_games_convergencia().
Activar ambos juntos = dos apuestas del mismo evento.
Auditar ese guard en nodo separado antes de activar.

Razón 2 — CONTENCIÓN KAMBI 429:
Serían 3 threads golpeando Kambi simultáneamente:
- 5s: fast_score_refresh
- 15s: _check_games_convergencia  
- 60s: live_edge_monitor.run()
Un 429 envenena fetch de cuotas del que dependen todos 
los paneles y el disparo de combos.

Razón 3 — FALLA SILENCIOSA:
except Exception: pass sin timeout HTTP — si run() cuelga,
el thread muere callado y el desk se ve perfecto.
Exactamente el patrón que Nodo-174 existió para matar.

Veredicto: activar en nodo separado con guard unificado,
timeout explícito y test que detecte falla silenciosa.
Referencia: análisis Opus sesión 2026-08-05.

# Nodo-116 — Entierro dashboard vieja + Auto-Combo live ANTI-FLOOD + P8 multi-casa real

> **Wikilinks:** [[Nodo-100]] (SUPERSEDED parcial: dashboard HTML) | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] | [[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-48]] (FlashScore odds) | [[Nodo-73]] (bridge :8765 INTOCABLE)
> **Fecha:** 2026-07-18 | **Autor:** Fable 5 (spec) | **Implementa:** Sonnet
> **Contexto:** :8765/live-dashboard ya está muerto de facto (404/timeout, verificado). Solo 3 archivos lo referencian (grep 2026-07-18): close_snapshot_server.py, scripts/live_dashboard_generator.py, scripts/live_edge_monitor.py. El auto-fire de combos (D100-05 `_fire_break_combos`) SÍ es valioso y se migra — pero hoy inunda el escritorio del operador con .bat sin freno. P8 muestra solo 2 columnas (betplay, flashscore) cuando FlashScore trae cuotas POR CASA (Nodo-48 es multi-casa).

## §A. ENTIERRO (quirúrgico — 3 puntos, nada más lo referencia)
1. `close_snapshot_server.py`: endpoint `/live-dashboard` → HTTP 301 `Location: http://localhost:7780/` + log "SUPERSEDED Nodo-109". **PROHIBIDO tocar `/check-and-close`** (infraestructura CLV n8n).
2. `scripts/live_edge_monitor.py`: flag `--dashboard` → no-op con aviso "dashboard viejo SUPERSEDED — usar live_desk :7780". No romper CLI existente.
3. `scripts/live_dashboard_generator.py`: queda en disco (historia SDD). Migrar al desk su ÚNICA pieza superior: CSS de urgencia — BREAK_CONFIRMADO con animación parpadeo rojo (`@keyframes blink`), BREAK_POSIBLE naranja fijo. ~5 líneas CSS en render_html().
4. Añadir al FINAL de Nodo-100 (no editar lo existente): "Dashboard HTML SUPERSEDED por [[Nodo-109]]/[[Nodo-114]]/[[Nodo-115]] — auto-combo migrado con anti-flood en [[Nodo-116]]".

## §B. AUTO-COMBO LIVE CON ANTI-FLOOD (D116-01 — la parte que gana dinero sin colapsar el escritorio)
**Problema real del operador:** cada BREAK_CONFIRMADO dispara betplay_combo_builder --live → .bat/links al escritorio, sin freno ni limpieza → "mi escritorio se llena de combos_live y no paran". La inmediatez es correcta (cuota baja en minutos); el buzón es el error.

**Regla nueva: EL DESK ES EL BUZÓN, no el escritorio.**
1. **Destino único:** todo output de `_fire_break_combos` → `reports/combos_live/YYYY-MM-DD/` (crear si no existe). **CERO archivos en Desktop/escritorio** — grep y eliminar cualquier ruta a Desktop en la cadena `_fire_break_combos` → betplay_combo_builder --live.
2. **De-dup por evento:** un solo fire por (event_id, BREAK_CONFIRMADO) — persistir set de event_ids disparados en `reports/combos_live/YYYY-MM-DD/_fired.json`. La máquina de estados Nodo-100 ya es single-fire por ciclo; esto lo hace single-fire POR DÍA aunque el monitor se reinicie.
3. **Cap diario:** máx 10 fires/día (constante `MAX_LIVE_FIRES_DIA=10`). Fire #11+ → solo log + fila en el desk "CAP ALCANZADO (10/10) — revisar manualmente". El governor (D107-04) cuenta estos combos como estrategia live.
4. **TTL:** al inicio de cada ciclo del monitor, borrar de `combos_live/` los .bat cuyo partido ya empezó (hora inicio + 15 min) — un combo live vencido es basura peligrosa, no historia. El registro permanente ya vive en shadow book (D101-05 `log_live_pick`), no en el .bat.
5. **Visibilidad en el desk:** los combos live del día = filas accionables en :7780 (tipo `COMBO_LIVE`, gate H100-01 en la línea de razonamiento, link al .bat en drill-down). El operador ve TODO en una pantalla ordenada en vez de un escritorio sepultado.
6. **Tests T53 (~5):** 2º fire mismo event_id → no dispara; fire #11 → no dispara y serializa CAP; TTL borra .bat de partido iniciado y respeta futuro; output va a reports/combos_live/ (assert NO Desktop en path); fila COMBO_LIVE aparece en build_desk_state.

## §C. P8 MULTI-CASA REAL (D116-02 — potenciar lo ya resuelto)
**Hecho verificado (Nodo-111 §4):** skins Kambi alternos = 429 (cookie de sesión requerida — NO reintentar keys a ciegas). **Pero FlashScore (Nodo-48) ya trae cuotas POR CASA** — hoy las aplanamos en una sola columna "flashscore", desperdiciando la dimensión multi-casa.
1. Extender el parser `--book2` en dual_book_client: si el JSON Nodo-48 trae desglose por bookmaker (`{casa: cuota}` por partido), emitir UN feed por casa (`{"wplay": {...}, "bwin": {...}}`) en vez del feed plano. `best_price()` ya acepta N feeds sin cambios (funciones puras intactas).
2. P8 render: una columna por casa detectada (dinámico, no hardcodear casas), mejor precio resaltado, divergencia >8% badge ámbar (ya existe). La mejor casa entra a la línea de razonamiento (ya implementado Nodo-114 — solo recibirá más candidatas).
3. Si el scraper Nodo-48 NO desglosa por casa hoy: PRIMERO verificar su output real (`ls reports/flashscore_odds_*.json` + inspección de schema, 5 min) — si solo trae cuota media, el paso 1 se gatea a una mejora del scraper (documentar como D116-03 pendiente, NO implementar scraping nuevo en esta sesión).
4. **Vía futura documentada (no ejecutar):** captura única DevTools "copy as cURL" del navegador logueado por casa Kambi (betcris/luckia/wplay/rushbet) → cookie a `data/book_sessions.json` → `fetch_kambi(offering, cookie=...)`. Gate: solo cuando el operador capture las cookies manualmente.
5. **Tests T53 (~3):** fixture Nodo-48 con 3 casas → 3 feeds + best_price elige la mayor; fixture sin desglose → degrada al feed plano actual; P8 render con 3 feeds → 3 columnas.

**PROHIBIDO:** tocar /check-and-close; reintentar offerings Kambi sin cookie; archivos en Desktop; borrar live_dashboard_generator.py (historia); scraping nuevo esta sesión.
**Criterio de éxito:** /live-dashboard → 301; escritorio limpio con fires yendo a reports/combos_live/ + filas COMBO_LIVE en el desk; P8 con N≥3 columnas si el schema Nodo-48 lo permite (o D116-03 documentado con el schema real pegado).

---

## D116-03 — Schema zita verificado 2026-07-18 (pendiente multi-casa)

**Fuente inspeccionada:** `data/zita_tennis_matches_20260718_164237.json`

**Schema real de un partido:**
```json
{
  "jugador1": "Tianmei Wang",
  "jugador2": "Salma Ewing",
  "cuota1": 1.81,
  "cuota2": 1.87,
  "match_url": null,
  "match_id": null,
  "superficie": "unknown",
  "torneo_nombre": "Estados Unidos - Newport Beach",
  "torneo_completo": "Estados Unidos - Newport Beach",
  "pais": "N/A",
  "ranking1": null,
  "ranking2": null,
  "tier": "utr_pro_tennis_series_women",
  "hora": "2026-07-18T00:00:00Z",
  "kambi_event_id": 1028401998,
  "cuota_es_real": true,
  "tournament_context": { "nombre": "...", "tier": "atp500", "superficie": "unknown", "season_transition_flag": true }
}
```

**Diagnóstico:** Solo `cuota1`/`cuota2` (cuota de Kambi por jugador). **SIN desglose por casa bookmaker.** No hay campo `casas`, `odds_por_casa`, ni equivalente.

**Consecuencia:** D116-02 (multi-casa real desde Nodo-48) está **GATEADO** hasta que el scraper Nodo-48 emita `{casa: cuota}` por partido.

**Vía de implementación futura:**
1. Modificar `extraer_URL_partidos_version2.py` para capturar offerings Kambi de múltiples skins (betcris/luckia/wplay/rushbet) — requiere cookies de sesión por casa (ver §C.4 Nodo-116).
2. Emitir `{casa: {"jugador1": cuota, "jugador2": cuota}}` por partido.
3. `dual_book_client.py --book2` ya acepta N feeds sin cambios (`best_price()` pura).

**Tests gateados:** `test_p8_render_n_casas_n_columnas` marcado `@pytest.mark.skip(reason="D116-03 pendiente")` en `tests/test_nodo116_antiflood.py`.

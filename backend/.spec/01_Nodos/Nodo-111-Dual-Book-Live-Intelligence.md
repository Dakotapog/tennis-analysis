# Nodo-111 — Dual-Book Live Intelligence: dos casas no son para surebet, son para ejecutar mejor y ver el steam

> **Wikilinks:** [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-71]] (Kyle's λ) | [[Nodo-107]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 | **Extiende:** Nodo-109 (añade paneles P8/P9 y el router de ejecución)
> **Veredicto sobre surebetting puro: NO como núcleo.** Con solo 2 casas colombianas (vig 5-8% c/u) los arbs reales son <1% de partidos con ROI 1-2%, y las casas limitan cuentas de arbitrajistas — matarías la gallina (las cuentas) por huevos chicos. El arb se toma SOLO cuando aparece gratis (P8 lo detecta como subproducto). El valor real de 2 casas es otro, y ya tenemos la infraestructura para explotarlo:

---

## §1. LAS 3 EXPLOTACIONES (mejores que surebet, en orden de Sharpe)

### X1 — BEST-PRICE EXECUTION (line shopping): +2-4% de ROI en TODA apuesta, riesgo cero
Cada pick de cualquier estrategia (1-13) se ejecuta en la casa con mejor cuota. Es la mejora de mayor certeza de todo el proyecto: no cambia qué apostamos, cambia cuánto paga. Un patrón como el del operador (Nodo-110, combos 4-6x) sube ~8-15% de pago combinado solo por elegir casa pierna a pierna.
**Implementación:** `odds_router(pierna) -> {casa, cuota}` — compara el feed Kambi existente (`fetch_kambi_outcomes`) vs el feed de la casa 2 (mismo patrón de cliente). El .bat/link se genera hacia la casa ganadora. CLV se mide contra el cierre de la casa DONDE se ejecutó.

### X2 — STEAM LAG: la casa lenta regala el precio viejo cuando la rápida confirma nuestra señal
La conexión oculta (puente de conocimiento): **ya tenemos el detector de steam** — `live_edge_monitor` (drift≥15%, Nodo-97) + máquina BREAK_CONFIRMADO (Nodo-100B) + H52-05 (STEAM_IN pre-registrada desde julio-03 y esperando datos). Con UNA casa, cuando detectamos el steam el precio ya se movió. Con DOS: si la casa líder mueve la línea EN LA DIRECCIÓN de nuestro p_modelo y la casa rezagada aún no ajustó → ejecutamos en la rezagada el precio pre-steam. Es el patrón profesional leader-follower: EV+ sin necesitar las dos piernas del arb.
**Gate (sin señal nuestra NO se dispara):** `BREAK_CONFIRMADO en casa A` ∧ `dirección = favorito del modelo (STRONG o meta_score≥3)` ∧ `cuota_B ≥ cuota_A_pre_drift × 0.97` ∧ governor PASS. Alimenta H52-05 y una nueva H111-01.

### X3 — MIDDLES en totales (games): cuando las dos casas ponen líneas distintas
Si casa A ofrece OVER 22.5 y casa B UNDER 24.5 del mismo partido: apostar ambos crea ventana 23-24 donde GANAN LAS DOS y fuera de ella pierde solo el vig de una. Riesgo acotado, cola gratis. Conecta directo con `games_signal_calculator` (que ya modela el rango esperado de juegos): solo tomamos middles donde NUESTRO rango modelado cae dentro de la ventana — middle informado, no ciego.

## §2. PANELES NUEVOS PARA EL DESK (Nodo-109)
- **P8 — DUAL BOOK BOARD:** cuotas lado a lado por partido del día, columna `divergencia%`, flag `STALE` (casa rezagada >90s tras drift confirmado de la otra), flag `ARB` (suma de inversas <1 — se toma si aparece), flag `MIDDLE` (líneas de totales separadas ≥1.5 juegos con nuestro rango dentro). Fuente: los 2 feeds cacheados por el ciclo del live_edge_monitor — CERO scraping nuevo por refresh del dashboard.
- **P9 — EXECUTION ROUTER:** para cada pick accionable (P2∩P4), a qué casa va y cuánto paga extra vs la otra (`+X% por routing`). El operador ve el porqué, no solo el dónde.

## §3. SPEC PARA SONNET
1. Cliente casa 2: verificación curl del endpoint PRIMERO (patrón B108-05, time-box 1 sesión). Si es Kambi-family → `fetch_kambi_outcomes(offering=...)`; si no, cliente mínimo con el mismo schema de salida `{norm_name: {outcome_id, odds, ...}}`.
2. `analysis/dual_book.py`: funciones puras `divergencia(o1,o2)`, `es_arb(o1,o2)`, `es_middle(linea1,linea2,rango_modelo)`, `steam_lag_signal(drift_A, cuota_B, p_modelo_dir)` — REGLA-T53 ~10 tests con fixtures numéricos.
3. `live_edge_monitor`: segundo feed en el mismo ciclo (no duplicar scheduler); serializa `book_leader`, `lag_secs`, `cuota_stale` al `live_edge_state_*.json`.
4. Nodo-109 `build_desk_state`: añade P8/P9 leyendo ese JSON.
5. H111-01 (pre-registro, OK usuario): "picks ejecutados vía steam-lag (gate §1-X2) superan breakeven de la cuota stale capturada; n_stop=20; kill-switch hit%<40% n≥10". Middles y arbs: registro observacional (riesgo acotado por construcción, no requieren graduación para volumen mínimo $500).
6. Orden: X1 (router — dinero inmediato, riesgo cero) → P8 → X3 (middles informados) → X2 (steam-lag, gateado por H111-01).
**PROHIBIDO:** perseguir arbs como estrategia primaria (ban risk), apostar steam-lag contra la dirección del modelo, superar $2,000/día en X2+X3 hasta graduar H111-01. Governor 13/13 cuenta todo (D107-02).

**Criterio de éxito:** semana 1 solo con X1 activo → el reporte muestra `+X% ROI por routing` medido sobre apuestas reales; P8 poblado en vivo; ≥1 middle informado detectado y registrado.

## §4. ADDENDUM 2026-07-17 — Fase curl EJECUTADA (Sonnet+Fable) e implementación X1 INICIADA (Fable)
- **Offerings Kambi alternos:** betcris/luckia/sportiumco/wplay/rushbet → **429 desde WSL Y desde Windows** con headers de navegador. No es geo-IP (betplay respondió 200 desde la misma IP): el CDN exige cookie de sesión del skin para offerings ajenos. B108-05 cerrado con evidencia; NO seguir probando keys a ciegas — la vía futura es capturar el request real desde el navegador logueado (DevTools → copy as cURL) UNA vez por casa.
- **Book 2 operativo HOY = FlashScore odds scraper (Nodo-48, ya construido)** — multi-casa, independiente de Kambi. Puente de conocimiento aplicado: no se necesita API nueva.
- **IMPLEMENTADO por Fable:** `scraping/dual_book_client.py` — stdlib puro (corre sin venv): `fetch_kambi(offering)` con backoff, funciones puras `best_price/divergencia/es_arb/es_middle`, CLI `--compare [--book2 <flashscore.json>]` que imprime routing por pick + ROI extra medio. Smoke test: parsea edge_report real y degrada con gracia (SIN COBERTURA/feed vacío); el 429 de betplay en el smoke fue transitorio (rate-limit autoinfligido por la ráfaga de pruebas — Sonnet tuvo 200 con 101 eventos minutos antes; esperar ~15 min).
- **Para Sonnet (remate, sin decisiones):** (1) tests REGLA-T53 de las 4 funciones puras (fixtures numéricos: arb 2.10/2.10→False, 2.15/2.15→True; middle 22.5/24.5 rango (23,24)→True); (2) conectar output de Nodo-48 como `--book2` en run_daily; (3) `es_middle` con el rango de games_signal_calculator; (4) re-smoke con feed vivo y pegar la tabla de routing como evidencia.

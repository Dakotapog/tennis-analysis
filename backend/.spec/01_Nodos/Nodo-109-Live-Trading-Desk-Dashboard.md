# Nodo-109 — Live Trading Desk: dashboard en vivo con semántica de mesa de trading

> **Wikilinks:** [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-100B-Triple-Convergencia-Live]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]] | [[Nodo-71]] (Kyle's λ) | [[Nodo-74-Combo-Governor]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 | **Implementa:** Sonnet
> **Principio:** el dashboard NO calcula nada nuevo — es el puente de conocimiento que RELACIONA instrumentos ya construidos y los presenta como los paneles que alertan a un trader de bolsa. Cada panel mapea 1:1 a un concepto de mesa de trading y a un archivo/función que YA existe. Cero señales nuevas = cero hipótesis nuevas.

---

## §1. LOS 7 PANELES (concepto bolsa → fuente existente)

| # | Panel | Concepto de trading | Fuente EXISTENTE (no recalcular) |
|---|---|---|---|
| P1 | **TAPE — cinta de momentum de línea** | Time & sales / order flow: velocidad y dirección del precio | `live_edge_monitor` (Nodo-97: drift% por ciclo) + `analysis/velocity_monitor.velocity_zscore()` (Kyle's λ, Nodo-71). Verde=cuota cayendo hacia nuestro pick (mercado confirma), rojo=alejándose |
| P2 | **BREAK BOARD — rupturas confirmadas** | Breakout confirmado con filtro anti-falso (2 velas) | Máquina de estados de Nodo-100B: `break_state` OBSERVANDO→POSIBLE→BREAK_CONFIRMADO (2 ciclos drift≥15%/12%). El panel parpadea SOLO en BREAK_CONFIRMADO — es la alerta accionable del día (H100-01) |
| P3 | **CONVERGENCE — semáforo de convicción** | Confluencia de indicadores (los traders no entran con 1 señal) | `meta_signal_score` (H98-01): HOT+STRONG+RFI+IRP+ELO_DOM. Score≥3 = fila destacada. Columna aparte: `rival_value_flag` (H88-01) en dirección OPUESTA — el "short" de la mesa |
| P4 | **RISK — exposición y circuit breakers** | Panel de riesgo del desk: VaR, límites, breakers | `combo_governor` exit-code (D107-04: PASS/WARN/BLOCK) + exposición por jugador (D107-03, cap 5%) + KGR de sesión + estado kill-switches (CAPA2_ENABLED, MOTOR_DEFENSIVE). BLOCK = banner rojo pantalla completa — como un halt de mercado |
| P5 | **EXECUTION — calidad de ejecución (CLV)** | Slippage / implementation shortfall | Shadow book Momento 2 (`cierre_kambi`, Nodo-101): CLV por pick abierto y mediana del día. CLV+ sostenido = estamos comprando mejor que el cierre = el edge es real aunque un día pierda |
| P6 | **P&L — libro del día por estrategia** | Blotter: P&L realizado/abierto por libro | `shadow_book --report` segmentos existentes (GCS 64.8% graduada, GS, MOTOR_cuota≤2.5/>2.5 de H107-01, CAPA2, RIVAL_VALUE) + `apuestas_*.json` del día. Orden: estrategias GRADUADAS arriba (donde está el dinero real permitido) |
| P7 | **CLOCK — ventanas de acción** | Calendario económico / apertura de mercados | Hora de inicio por partido (zita file) + ventana live de Nodo-97 [-30,+45min] + countdown al close-snapshot (~15min antes). El trader de bolsa vive del reloj; aquí igual |

## §2. REGLAS DE DISEÑO (las que hacen que esto genere ganancias y no ruido)

1. **Jerarquía por accionabilidad, no por novedad:** arriba solo lo apostable AHORA (P2 BREAK_CONFIRMADO ∩ P4 PASS ∩ estrategia graduada). Todo lo demás es contexto colapsable. Un panel que grita con señales no graduadas produce sobre-trading — el error O-01.
2. **Toda alerta lleva su gate:** cada fila muestra qué hipótesis la gobierna (H97-01/H98-01/H100-01/H88-01) y su n_actual/n_stop. Señal pre-graduación = ámbar SIEMPRE, sin excepción; solo GCS (y lo que gradúe) puede ser verde.
3. **P4 manda:** si governor=BLOCK o KGR<0, los paneles P1-P3 se atenúan (grises) — no existe "señal buena" con el desk en halt. Es la traducción visual de REGLA-HF-5.
4. **Cero cálculo en el dashboard:** solo lee `reports/live_edge_state_*.json` (Nodo-97), `sb_*.jsonl`, `combo_governor.log`, `apuestas_*.json`, edge_report. Si un dato falta → celda "SIN DATO + comando para generarlo" (Zero-Null D90-04), nunca celda vacía.

## §3. SPEC PARA SONNET

- **Archivo:** `live_desk.py` (nuevo, NO tocar `dashboard.py` de sesiones/tokens). Servidor HTTP stdlib en `:7780` (patrón `close_snapshot_server.py`/graphify `:7779`), HTML único auto-refresh 30s (meta-refresh; sin frameworks, sin CDN — igual que graph.html). WSL: `systemd --user` opcional después.
- **Funciones puras testeables (REGLA-T53):** `build_desk_state(fecha) -> dict` (agrega las 7 fuentes; tolera archivos ausentes), `accionable_ahora(state) -> list` (la intersección de la regla 1), `render_html(state) -> str`.
- **Orden:** 1) `build_desk_state` + tests con fixtures de archivos reales del 2026-07-14 (día con 3/3 rival value — debe salir en P3/P6); 2) P4 primero (riesgo antes que señal), luego P2, P3, P5, P6, P1, P7; 3) test integración: con governor BLOCK simulado → `accionable_ahora()` retorna [] y el HTML contiene el banner halt.
- **Baseline pytest (≥1945) intacto. GIT-FIRST. Evidencia de cierre: screenshot/curl del HTML con datos reales del día en un Nodo-110 (patrón Nodo-92).**
- **PROHIBIDO:** calcular señales nuevas, mostrar como verde nada no graduado, botones de apuesta (el dashboard informa, el humano ejecuta — la disciplina es el edge).

**Criterio de éxito:** un vistazo de 10 segundos responde las 3 preguntas del trader: ¿hay algo accionable ahora? (P2∩P4) · ¿con cuánta convicción y bajo qué gate? (P3) · ¿el desk está sano? (P4/P5).

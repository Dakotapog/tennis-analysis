# Nodo-114 — Live Desk v2: línea de razonamiento visible + P8 Multi-Book

> **Wikilinks:** [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-110-Modo-Operador-Favoritos-Compuestos]] | [[Nodo-107-Riesgo-Agregado-Motor-Reconciliacion]]
> **Fecha:** 2026-07-17 | **Autor:** Fable 5 (auditoría con evidencia + spec) | **Implementa:** Sonnet
> **Principio de esta sesión: VERIFICAR sobre REDISEÑAR** — lo que funciona no se toca.

## §1. VERIFICADO CON EVIDENCIA (2026-07-17)
- `live_desk.py` (750 líneas) EXISTE e implementa P1-P7 del Nodo-109 con `build_desk_state()` (:52, P4 primero), `accionable_ahora()` (:80, P4-manda/REGLA-HF-5 correcto), `render_html()` (:137). **Lanzado y curl-eado: HTTP 200, 7,599 bytes, governor PASS con datos reales — no placeholders.** El servidor NO estaba corriendo (HTTP 000 en frío) — falta systemd/arranque.
- Nodo-111 implementado parcialmente: `scraping/dual_book_client.py` existe (Fable, funciones puras + router X1); **NO está conectado a live_desk.py** (grep book2/dual_book en live_desk = 0 hits).
- Números 112 y 113 OCUPADOS (C3 campeón-signal, weather MVP) — por eso este nodo es 114.

## §2. GAP A — LA LÍNEA DE RAZONAMIENTO (prioridad 1 del operador)
Cada fila de `accionable_ahora()` y cada candidato relevante debe mostrar **POR QUÉ**, en una línea legible, la cadena que convergió — no un checkmark.

**Spec:** función pura `linea_razonamiento(pick: dict) -> str` en live_desk.py. Concatena SOLO las señales presentes, en este orden fijo (riesgo→señal→ejecución):
`[gate H-XX n/n_stop] BREAK_CONFIRMADO(drift -18%→-14%) + meta_score=4(HOT,STRONG,ELO_DOM,RFI) + CLV+2.3% + n_h2h=3 + governor PASS → mejor precio: flashscore @2.35 (+5.4% vs betplay)`
Reglas: (1) cada término sale del campo ya serializado (break_state, meta_signal_score + lista de señales activas, clv, capa2_candidate, rival_value_flag, gcs_bonus, estrategia FAVORITOS_COMPUESTOS…) — cero cálculo nuevo; (2) el gate que gobierna la fila SIEMPRE al frente con su n_actual/n_stop; (3) máx ~120 chars, lo demás en tooltip/title; (4) señales pre-graduación en ámbar dentro de la misma línea. Test T53: pick sintético con 4 señales → string contiene las 4 en orden; pick con governor WARN → línea empieza con el gate de riesgo.

## §3. GAP B — P8 MULTI-BOOK (smart order routing en el desk)
Fuente EXISTENTE: `scraping/dual_book_client.py` (`fetch_kambi('betplay')` + `--book2` FlashScore Nodo-48 + `best_price()`).
**Spec:** (1) `build_desk_state()` añade clave `p8_books`: para cada pick de `accionable_ahora()` + combos Nodo-110 del día, llama `best_price(nombre, feeds)` con feeds cacheados a disco (`reports/dual_book_cache.json`, TTL 120s — UNA llamada de red por ciclo, no por render; respeta el rate-limit 429 documentado en Nodo-111 §4). (2) Render: columna por casa, mejor precio resaltado, `divergencia > 8%` → badge ámbar "ATENCIÓN divergencia X%" (señal de atención, NO de apuesta — Nodo-109 §2.2). (3) La mejor casa se inyecta al final de la línea de razonamiento (§2). Tests: fixture 2 feeds → mejor resaltado; divergencia 9% → badge; feed caído → celda "SIN DATO + comando".

## §4. GAP C — Nodo-110 como fila de primera clase
Los combos FAVORITOS_COMPUESTOS del día (output de `favoritos_combo_builder.py`) entran a `accionable_ahora()` como tipo propio (gate H110-01 8/8 semilla en la línea de razonamiento), no solo al blotter P6. Si el builder no corrió hoy → fila Zero-Null: "FAVORITOS: sin correr — python3 favoritos_combo_builder.py".

## §5. CHECKLIST PRODUCTO — pendiente para OTRA sesión (documentado, NO implementar hoy)
| Ítem | Estado verificado | Acción futura |
|---|---|---|
| Alerta empujada (Telegram en BREAK_CONFIRMADO / governor BLOCK) | NO existe en live_desk (grep=0). `enviar_combos_telegram()` ya existe en el proyecto | Hook en el ciclo del live_edge_monitor (no en el render): transición a BREAK_CONFIRMADO o a BLOCK → 1 mensaje, con de-dup por evento |
| Historial TAPE (3-4 ciclos, tendencia vs tick) | NO — P1 muestra snapshot | `live_edge_state` ya guarda ciclos; render sparkline texto `▁▃▅▇` de los últimos 4 drifts |
| Persistencia de estado UI | NO | Bajo impacto — posponer (localStorage) |
| Arranque del servidor | Manual (estaba caído) | systemd --user `tennis-live-desk.service` :7780 + nota TROUBLESHOOTING |

## §6. ORDEN PARA SONNET (con presupuesto corto)
1. §2 línea de razonamiento (es render + 1 función pura — máximo valor/token)
2. §4 FAVORITOS primera clase
3. §3 P8 con cache 120s
4. §5 solo si sobra: Telegram hook (el resto queda documentado)
Baseline pytest actual intacto; evidencia de cierre: curl del HTML con ≥1 línea de razonamiento real pegada en el nodo de cierre.

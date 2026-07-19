# CHECKLIST IMPLEMENTACIÓN NODOS 107-110
> Orden maestro definido por Fable en Nodo-108 §4 — 2026-07-17
> FUENTE DE VERDAD: este archivo. Actualizar con evidencia real al completar cada paso.
> Marcar: [ ] pendiente | [x] DONE (+ evidencia) | [!] BLOQUEADO

---

## FASE 0 — PREREQUISITO (sin dependencias)

### [ ] B108-01 — Colisión Nodo-100 (~15 min)
**Qué:** Renombrar `Nodo-100-Triple-Convergencia-Live.md` → `Nodo-100B-Triple-Convergencia-Live.md`
**Pasos:**
- [ ] Añadir al tope del archivo: "Renombrado de Nodo-100 por colisión — ver Nodo-108 B108-01"
- [ ] Renombrar el archivo en .spec/01_Nodos/
- [ ] `grep -rn "Nodo-100-Triple"` → actualizar wikilinks entrantes
- [ ] `python3 scripts/rebuild_nodos_index.py` → sin error de duplicados
- [ ] `graphify update .`
- [ ] Nuevo test: rebuild_nodos_index detecta números duplicados y falla ruidosamente
**Evidencia esperada:** `rebuild_nodos_index.py` output sin duplicados; pytest verde

---

## FASE 1 — NODO-107: Governor Risk Aggregation (prerequisito de 109 y 110)

### [ ] S107-A — O-01 en DECISION-LOG (sin código)
**Qué:** Añadir texto exacto de §5 de Nodo-107 a `docs/DECISION-LOG.md`
```
## O-01 — Sobre-stake 16.7× por agregación ciega de combos (2026-06-26, registrado 2026-07-17)
Qué pasó: 7 combos con piernas compartidas sumaron $83,500 contra budget $5,000 (1670%)...
Causa raíz: control per-estrategia sin control agregado cross-estrategia...
Regla derivada: REGLA-O1: ningún peso se despliega sin pasar por el agregado del governor (12/12 + cap 5%)
```
**Evidencia esperada:** `grep "O-01" docs/DECISION-LOG.md` → encuentra la entrada

### [ ] S107-B — combo_governor.py: cobertura 12/12 estrategias
**Qué:** 2 funciones puras nuevas en `combo_governor.py`:
- `_trader_stakes_today(fecha) -> dict[str,int]` — lee `trader_plan_*.json`: individuales[].stake + cobertura[].stake
- `_rival_value_stakes_today(fecha) -> dict[str,int]` — mismo patrón sobre rival_value_betslip output
- Sumar ambas al total en `main()`
**Tests REGLA-T53:** fixture por cada estrategia (12/12) → el agregado las ve todas
**Evidencia esperada:** pytest nuevo test matriz 12/12 VERDE; `python3 combo_governor.py` muestra total 12/12

### [ ] S107-C — exposicion_por_jugador() — cap 5% bankroll
**Qué:** Función pura `exposicion_por_jugador(capas: list[dict]) -> dict[str, int]`
- Suma stake por jugador usando `core/player_registry.normalize_player_name()` (NO otra normalización)
- WARN si jugador > 5% bankroll sesión
- BLOCK si supera con margen
**Tests REGLA-T53:**
- Pierna compartida en 3 combos → dispara WARN/BLOCK
- Jugadores distintos → PASS sin fricción
**Evidencia esperada:** pytest verde; re-simular sesión 2026-06-26 → BLOCK con detalle por jugador

### [ ] S107-D — Exit codes + soft-veto en builders
**Qué:** Exit codes del governor: 0=PASS, 1=WARN, 2=BLOCK
- En `combo_confianza_builder.py`: invocar governor al inicio; si exit≥1 y sin `--override-governor` → abort
- En `betplay_combo_builder.py`: idem
- En `rival_value_betslip.py`: idem
- Override → línea en `combo_governor.log` con quién/cuánto
- Mensaje de abort accionable (Zero-Null D90-04: nunca silencio, explica qué bloqueó y cuánto reducir)
**Evidencia esperada:** `python3 combo_governor.py` retorna exit code correcto; builder con BLOCK imprime mensaje claro

### [ ] S107-E — MOTOR_DEFENSIVE + segmentos shadow_book
**Qué:**
- `trader_ev_tenis.py`: constante `MOTOR_DEFENSIVE=True` → stakes individuales × 0.5; banner en output
- `shadow_book.py report`: nuevos segmentos `MOTOR_cuota<=2.5` / `MOTOR_cuota>2.5` (patrón de segmentos existente, H107-01)
**Evidencia esperada:** `python3 trader_ev_tenis.py` muestra banner DEFENSIVE; `python3 shadow_book.py --report` muestra 2 nuevos segmentos MOTOR

### [x] S107-F — H107-01 registrada — YA COMPLETADO
H107-01 insertada en `validation/preregistered_hypotheses.json` (estado ACUMULANDO, 2026-07-17, 25 hipótesis validadas). Sonnet solo lee, NO reinsertar.

---

## FASE 2 — NODO-110: Favoritos Compuestos (estrategia #13)
> ⚠️ REQUIERE: Nodo-107 completo (D107-02 para 13/13 governor) + OK usuario para D110-01 y H110-01

### [ ] D110-01 — OK USUARIO requerido antes de codificar
**Texto para DECISION-LOG:** "REGLA-HF-1 aplica a SINGLES y pool del trader. Piernas de combo: piso propio `LEG_MIN_CUOTA=1.15`. Codificación de lo que D87-08 ya practica y el operador ya validó (8/8 wins)."
**Estado:** [ ] OK del usuario recibido

### [ ] H110-01 — Pre-registro en preregistered_hypotheses.json
```json
{"id": "H110-01", "nombre": "Favoritos Compuestos hit% combo",
 "prediccion": "combos 3-4 favoritos logran hit% combo >= 25% (breakeven @4.0 media)",
 "semilla": "8/8 jul-14/16 retroactivos",
 "n_actual": 8, "n_stop": 30, "kill_switch": "hit%<15% con n>=15 → OFF",
 "estado": "ACUMULANDO"}
```
**Estado:** [ ] Insertado en JSON

### [ ] N110-IMPL — `favoritos_combo_builder.py` (archivo nuevo)
**Qué:** Nuevo generador estrategia #13
- `seleccionar_favoritos(edge_report) -> list`: universo apostar+watchlist+sin_edge; filtros: sin NO_DATA/phantom, p_modelo≥0.62 O cuota≤1.40 O ranking>300 puestos, cuota∈[1.15,2.10], fav modelo = fav bookmaker
- `armar_combos(picks) -> list`: 3-4 piernas, máx 2/torneo, máx 1/jugador, cuota combinada [3.5,7.0], top-3 combos solape≤2 piernas
- Stake: $650/combo, tope sesión $2,000, governor con veto ANTES de emitir
- Output: SIEMPRE emite — si no hay piernas, explica cuántas pasaron cada filtro (Zero-Null D90-04)
- Links Betplay + .bat (reusar find_outcome/generar_bat_chrome de betplay_combo_builder)
- Registro automático: shadow_book + betslip_index (estrategia=FAVORITOS_COMPUESTOS)
- Integración en `run_daily.py` tras PASO 4.3
**Variante MEGA-OPERADOR (D110-05):**
- 5-8 piernas para eventos densos (qualy GS ≥16 partidos)
- Hasta 2 piernas "spice" [2.10,5.00] con p_modelo≥0.55 o rival_value_flag
- Stake fijo $500, segmento propio en H110-01
**Tests REGLA-T53 (~8):**
- Los 4 filtros de selección individualmente
- Diversificación por torneo (máx 2)
- Solape ≤2 entre combos generados
- Fixture edge_report 2026-07-16 → reproduce ≥3 piernas reales (Gaines/McNeil/Forbes/Bynoe)
- Governor BLOCK → no emite
**Evidencia esperada:** `python3 favoritos_combo_builder.py` con datos Jul-17 → ≥1 combo emitido o desglose exacto de filtros fallidos; pytest ~8 tests VERDE

---

## FASE 3 — NODO-109: Live Trading Desk Dashboard
> ⚠️ REQUIERE: D107-04 completo (governor con exit codes)

### [ ] N109-IMPL — `live_desk.py` (archivo nuevo)
**Qué:** Servidor HTTP stdlib en :7780, HTML único auto-refresh 30s
**Funciones puras testeables (REGLA-T53):**
- `build_desk_state(fecha) -> dict` — agrega 7 fuentes; tolera archivos ausentes
- `accionable_ahora(state) -> list` — intersección P2∩P4∩estrategia-graduada
- `render_html(state) -> str`

**Los 7 paneles (leen fuentes existentes, CERO cálculo nuevo):**
- P1 TAPE: `live_edge_monitor` drift% + `velocity_monitor.velocity_zscore()` (Kyle's λ)
- P2 BREAK BOARD: máquina estados Nodo-100B (`break_state` BREAK_CONFIRMADO → parpadea)
- P3 CONVERGENCE: `meta_signal_score` H98-01 + columna `rival_value_flag` H88-01
- P4 RISK: governor exit-code + exposición jugador D107-03 + KGR sesión + kill-switches (CAPA2/MOTOR_DEFENSIVE). BLOCK = banner rojo pantalla completa
- P5 EXECUTION: shadow_book Momento 2 CLV por pick abierto + mediana del día
- P6 P&L: shadow_book --report segmentos (GCS/GS/MOTOR_cuota≤2.5/>2.5/CAPA2/RIVAL_VALUE/FAVORITOS_COMPUESTOS) + apuestas del día
- P7 CLOCK: hora inicio (zita file) + ventana live [-30,+45min] + countdown close-snapshot

**Reglas de diseño:**
- P4 manda: si BLOCK o KGR<0 → paneles P1-P3 atenúados (grises)
- Toda alerta muestra gate (H97-01/H98-01/etc) + n_actual/n_stop
- Señal pre-graduación = ámbar siempre; solo GCS (y graduadas) puede ser verde
- Celda sin dato → "SIN DATO + comando para generarlo" (Zero-Null D90-04)

**Orden de implementación:**
1. `build_desk_state` + tests con fixtures 2026-07-14 (3/3 rival value → aparece en P3/P6)
2. P4 primero (riesgo antes que señal)
3. P2, P3, P5, P6, P1, P7
4. Test integración: governor BLOCK simulado → `accionable_ahora()` retorna [] + HTML con banner halt

**Evidencia esperada:** `curl http://localhost:7780` → 200; screenshot/curl con datos reales del día; pytest integración VERDE

---

## FASE 4 — BACKLOG RESTANTE (después de 107/109/110)

### [ ] B108-03 / C2 — Name-matching unificado
**Qué:** Todos los call-sites delegan a `core/player_registry.normalize_player_name()`
- Prioridad: `betslip_registrar`/`shadow_book` (settle) y `kambi_tennis` primero
- Un módulo por commit, suite verde entre cada uno
- Cerrar TODO F0-DEUDA en `core/player_registry.py:29-30` (RankingManager delega)
**Tests:** mismos inputs raros (iniciales múltiples, guiones, diacríticos) → misma salida en todos los call-sites

### [ ] B108-04 — Checklist semanal H89-01/H89-02
**Qué:** Solo lectura — añadir al checklist semanal la lectura de segmentos CAPA2 y ELO_DOMINANCE en `shadow_book --report`; cuando n≥30 → decisión SPRT
**PROHIBIDO:** tocar threshold antes de datos

### [ ] B108-05 — OddsAggregator (curl session, time-boxed)
**Qué:** Probar offering-key Kambi de betcris/luckia/sportium con mismo endpoint de betplay_combo_builder.py:113
- Si ≥1 confirma → refactor `fetch_kambi_outcomes(offering='betplay')`
- Si 0 confirman → archivar D89-09 con evidencia

### [ ] C3 — "Campeón reciente" deduplicado
**Qué:** Un solo campo `campeon_signal` en rivalry_analyzer; consumidores leen, no recalculan
**Prerequisito:** spec corta antes de código (nodo propio)

### [ ] B108-06 — Weather MVP observacional
**Prerequisito:** B108-03 y B108-04 cerrados
**Qué:** open-meteo por requests (sin key) → campo `weather_flag` en edge_report + hipótesis pre-registrada antes de cualquier ajuste de p_modelo

---

## REGLAS GLOBALES (no romper en ningún paso)
- GIT-FIRST antes de cada implementación
- Baseline pytest ≥1945 antes de empezar cada paso
- Commit propio por paso, suite verde antes del commit
- REGLA-T53: tests invocan función real, nunca hardcodean fórmula
- PROHIBIDO en este bloque: tocar gates de edge_calculator, λ_tier, EDGE_MIN/KELLY_KL_MIN
- PROHIBIDO: hard-veto antes de 10 sesiones con nuevas dimensiones activas
- PROHIBIDO en Nodo-110: escalar stake >$650 antes de graduación
- PROHIBIDO en Nodo-109: calcular señales nuevas, botones de apuesta, verde a señales no graduadas

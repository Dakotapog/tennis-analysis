> Renombrado de Nodo-100 por colisión — ver Nodo-108 B108-01. Nuevo nombre: Nodo-100B-Triple-Convergencia-Live.md

# Nodo-100B — Triple Convergencia Live: Break Confirmation + Dashboard + Auto-Combo

**Fecha:** 2026-07-14  
**Estado:** IMPLEMENTADO — verificado 2026-07-19 (5/5 tests, todas las funciones, redirect 301)  
**Tipo:** ACCION — dispara combos reales cuando break confirmado  
**Wikilinks:** [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-99-Auditoria-Fable-N97-N98]] | [[Nodo-73-n8n-CloseSnapshot-Timing]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] | [[Nodo-115-Desk-Interactivo-Incertidumbre-Visible]] | [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]

---

## §1 — Problema

KambiLiveClientReal (Nodo-97 D97-15) detecta drift >= 15% cada 2 min.
Un solo drift no distingue:
- **Fluctuación normal**: servidor recupera su saque → cuota vuelve a subir
- **Quiebre CONFIRMADO**: rival rompió el saque y lo sostuvo → cuota se mantiene baja

Sin confirmación, el sistema dispararía combos en fluctuaciones normales (ruido).

## §2 — Solución: máquina de estado por partido

```
NORMAL ──────────────────► BREAK_POSIBLE   (drift >= 15% + edge > 5%)
BREAK_POSIBLE ───────────► BREAK_CONFIRMADO (drift >= 12% en ciclo siguiente)
BREAK_POSIBLE ───────────► NORMAL          (drift < 10% = recovery = fluctuación)
BREAK_CONFIRMADO ────────► done            (single-fire, fired=True)
```

Umbral confirmación = 12% (no 15%) — permite degradación natural entre lecturas (2 min).

## §3 — Decisiones

| ID | Decisión |
|---|---|
| D100-01 | Break POSIBLE = primera lectura con drift >= 15% |
| D100-02 | Break CONFIRMADO = lectura siguiente con drift >= 12% (sin recovery intermedio) |
| D100-03 | Single-fire: `fired=True` en history previene combos duplicados para el mismo partido |
| D100-04 | Auto-combo → `betplay_combo_builder.py --live --telegram` (picks NOT_STARTED del día) |
| D100-05 | Dashboard HTML SUPERSEDED → redirect HTTP 301 a `:7780/` (live_desk.py Nodo-109) |
| D100-06 | Persistencia: `reports/live_odds_history_YYYYMMDD.json` (un archivo por día) |
| D100-07 | Recovery threshold = 10% (cuota vuelve a estar dentro del 10% del precio inicial) |

## §4 — Restricción Betplay (confirmada en exploración)

`fetch_kambi_outcomes()` en `betplay_combo_builder.py` L142-149 filtra eventos STARTED.
Betplay no permite links pre-armados para partidos ya iniciados.

**Consecuencia:** el break confirmado en partido A activa combos de partidos NOT_STARTED del día.
La lógica es: "el modelo funcionó hoy (A rompió como predijo) → apostar los picks restantes con confianza".

## §5 — Hipótesis pre-registrada

**H100-01:** Triple Convergencia picks (score_directo>=2 + break confirmado) producen ROI > picks sin convergencia  
- n_stop = 20  
- Gate activación: n >= 3 breaks confirmados en sesiones reales  
- Métrica: ROI de combos post-break vs ROI combo normal  
- Estado: ACUMULANDO — 0 breaks confirmados en producción (Kambi LIVE endpoint pendiente DevTools, ver Nodo-97 BLOCKER)

## §6 — Archivos implementados

| Archivo | Cambio | Líneas clave |
|---|---|---|
| `scripts/live_edge_monitor.py` | `load_odds_history()` L684, `save_odds_history()` L696, `detect_break_state()` L706, `_fire_break_combos()` L786, `run()` actualizado | — |
| `scripts/live_dashboard_generator.py` | NUEVO — `generar_dashboard_html()` L89 + `main()` L295 | SUPERSEDED por Nodo-109 |
| `close_snapshot_server.py` | `/live-dashboard` → HTTP 301 → `http://localhost:7780/` L133-226 | D100-05 actualizado |
| `tests/test_nodo100_triple_conv.py` | NUEVO — 5 tests REGLA-T53 | 5/5 PASS |

## §7 — Flujo operacional

```
n8n cada 2 min → :8765/live-check → live_edge_monitor.py --observe --dashboard

Ciclo 1 (14:05): Boogaard drift=18.3% → BREAK_POSIBLE
  [Esperando confirmación — nada se dispara]

Ciclo 2 (14:07): Boogaard drift=16.1% → BREAK_CONFIRMADO
  → betplay_combo_builder.py --live --telegram
  → Telegram: "BREAK CONFIRMADO — COMBOS LIVE DISPARADOS"
  → reports/combos_live/YYYY-MM-DD/ (CERO Desktop — D116)
  → live_desk.py :7780/ muestra estado (redirect desde /live-dashboard)

Ciclo 3 (14:09): Boogaard drift=13.5% → ya fired=True → sin re-disparo
```

## §8 — Tests (REGLA-T53)

`tests/test_nodo100_triple_conv.py` — 5 tests invocando funciones reales:
1. `test_break_posible_en_primer_drift` — 1er drift 18% → BREAK_POSIBLE
2. `test_break_confirmado_en_segundo_ciclo` — drift 18% → drift 13% → CONFIRMADO
3. `test_recovery_cancela_break_posible` — drift 18% → drift 8% → NORMAL
4. `test_no_refire_si_fired_true` — CONFIRMADO + fired=True → sin combo
5. `test_dashboard_html_contiene_campos_clave` — HTML generado tiene tabla + auto-refresh

## §9 — Verificación 2026-07-19

**Diagnóstico:** memoria de sesión decía "plan pendiente, 5 tests sin implementar". Al verificar:

| Elemento | Estado real |
|---|---|
| `detect_break_state()` en `scripts/live_edge_monitor.py` L706 | IMPLEMENTADO |
| `load_odds_history()` / `save_odds_history()` L684/L696 | IMPLEMENTADO |
| `_fire_break_combos()` L786 | IMPLEMENTADO |
| `scripts/live_dashboard_generator.py` | IMPLEMENTADO (SUPERSEDED por live_desk :7780) |
| `/live-dashboard` → HTTP 301 → `:7780/` | IMPLEMENTADO |
| `tests/test_nodo100_triple_conv.py` | 5/5 PASS |

**Causa de confusión:** el plan en `unified-hugging-melody.md` era pre-implementación. El spec ya marcaba `Estado: IMPLEMENTADO` pero el índice de memoria no se actualizó. Corregido 2026-07-19.

**BLOCKER pendiente para H100-01:** Nodo-97 BLOCKER — Kambi LIVE endpoint sin verificar por DevTools. Sin datos reales live, la máquina de estado no acumula breaks confirmados en producción.

---

> **ADDENDUM 2026-07-18 (Nodo-116):** Dashboard HTML SUPERSEDED por [[Nodo-109-Live-Trading-Desk-Dashboard]] / [[Nodo-114-Desk-Razonamiento-P8-MultiBook]] / [[Nodo-115-Desk-Interactivo-Incertidumbre-Visible]] — auto-combo migrado con anti-flood en [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]]. `/live-dashboard` → HTTP 301 → `:7780/`. Output combo: `reports/combos_live/YYYY-MM-DD/` (CERO Desktop).

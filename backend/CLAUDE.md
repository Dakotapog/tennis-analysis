# CLAUDE.md — Tennis Prediction & Betting Engine

> Last updated: 2026-07-09 (FABLE_02 Fases 1-5 + Nodo-73 n8n — 1756 tests)
> Spec-Driven Development (SDD). CLAUDE.md es VISTA DERIVADA — los nodos son la fuente de verdad.
> Leer completo antes de tocar código. Ver política de precedencia §10.

---

## 1. NORTE REAL

**Visión:** Hedge fund cuantitativo — cada partido = activo con vida útil 2-3h.
**Misión:** Apostar solo donde `P_modelo > P_implícita_bookmaker` con Kelly-KL.
**Métrica:** P&L positivo acumulado — NO accuracy.

| Meta | Métrica | Estado |
|---|---|---|
| ~~Datos limpios~~ | Fix scraper + surface_specialization | ✅ 2026-05-28 |
| ~~Accuracy > 55%~~ | Con superficie | ✅ 77.4% (n=31, clay) |
| ~~P&L positivo n≥30~~ | Kelly-KL + sesiones validadas | ✅ 2026-06-01 |
| **Escalar bankroll** | Edge validado, hedge fund activo | **EN CURSO** |

---

## 2. CONSTITUCIÓN — Reglas inmutables

1. **SDD:** Ningún código sin Nodo en `.spec/01_Nodos/`. Ver `PRE_IMPLEMENTATION_CHECKLIST.md`.
2. **GIT-FIRST:** Buscar en git history ANTES de implementar. `git log --all --oneline -- '*keyword*'`
3. **REGLA-T53:** Tests invocan función real del módulo — nunca hardcodean la fórmula.
4. **REGLA-HF-5:** KGR < 0 en output trader → NO DESPLEGAR. Sin excepción.
5. **REGLA-HF-1:** Cuota < 1.50 NUNCA en pool. KGR con heavy fav = -0.5085 (ruina).
6. **Playwright PRIMARIO:** PASOS 1+2. API solo si Playwright falla Y Phantom Guard activo.
7. **5 errores históricos:** Ver `docs/DECISION-LOG.md`. No repetir.
8. **Pre-registro:** Ninguna hipótesis sin H-XX en `validation/preregistered_hypotheses.json`.

---

## 3. FUNDAMENTOS CLAVE

```
Kelly-KL:   f*_KL = f*_clásico × exp(-λ × KL(P_modelo || P_histórica))
λ por tier: GS=1.0× | ATP1000=1.6× | ATP500=2.4× | Challenger=3.6× | ITF=4.5×
Portfolio:  factor = 1/(1+ρ×(N-1))  ρ: GS=0.25|ATP1=0.20|ATP5=0.15|CHA=0.10|ITF=0.05
VaR:        MAX_VAR_PCT=0.25 | g=E[log(1+R)] > 0 crecer | g<0 NO DESPLEGAR
CPPI:       cushion=(bankroll-FLOOR)/bankroll; factor=min(1,max(0,2×cushion)) — PROVISIONAL
GCS:        ACTIVO solo hierba. _GCS_BOOST_ENABLED=True, prior A60-01 (n=54, 64.8%)
Shrinkage:  n/(n+20). n=4→30%. n=33→62%. Lee calibracion_edge.json automáticamente.
```

Implementación: `analysis/rivalry_analyzer.py` | `edge_calculator.py` | `trader_ev_tenis.py`

---

## 4. FLUJO DEL PIPELINE

### ANTES DEL PARTIDO

```bash
# PASO 0 — Rankings (si están desactualizados)
python3 extraer_ranking_atp_version2.py && python3 extraer_ranking_wta_version2.py

# PASO 1 — Extraer partidos (PLAYWRIGHT PRIMARIO)
python3 extraer_URL_partidos_version2.py            # → data/zita_tennis_matches_FECHA.json
# API fallback: python3 extraer_partidos_api.py [--tomorrow] [--tier atp wta] [--torneo wimbledon]

# PASO 2 — Extraer H2H (PLAYWRIGHT PRIMARIO)
python3 extraer_historh2h.py --all-tournaments      # → reports/h2h_results_enhanced_FECHA.json
# API fallback: python3 extraer_historh2h.py --api-mode --all-tournaments

# PASO 3 — Edge Kelly-KL
python3 edge_calculator.py                          # → reports/edge_report_FECHA.json

# PASO 3.5 — Revisión humana (LEER ANTES DE APOSTAR)
python3 generar_tabla_favoritos2.py                 # → analisis_partidos_pandas.txt
# Revisar: contribution% | surface_specialization raw_score | Confianza <52% = señal débil

# PASO 3.6 (opcional) — Señales totales juegos/sets
python3 games_signal_calculator.py                  # → reports/games_signal_report_FECHA.json

# PASO 4 — Deploy (UN tier por ejecución — --torneo-tipo FILTRA)
python3 trader_ev_tenis.py --bankroll 125000                              # GS clay (default)
python3 trader_ev_tenis.py --bankroll 125000 --superficie grass           # GS grass
python3 trader_ev_tenis.py --bankroll 50000  --torneo-tipo atp1000        # ATP1000
python3 trader_ev_tenis.py --bankroll 20000  --torneo-tipo challenger     # Challenger
python3 trader_ev_tenis.py --bankroll 10000  --torneo-tipo itf            # ITF
# Si KGR < 0 → NO DESPLEGAR. VaR auto-ajustado en "STAKES FINALES".

# PASO 4.3 — Combos de confianza (paralelo al pipeline)
python3 combo_confianza_builder.py --bankroll 125000 [--fase 1|2|3|4] [--anchor] [--telegram]

# PASO 4.4-4.57 (opcional) — Betplay + megas + safe
python3 betplay_combo_builder.py --live [--games] [--mega] [--safe] [--telegram]

# PASO 4.6 — Registrar apuesta
python3 betslip_registrar.py --listen               # antes de apostar (puerto 5001)
python3 betslip_registrar.py --cerrar               # post-partido → calibracion_edge.json auto

# ORQUESTADOR DIARIO
python3 run_daily.py [--bankroll N] [--tomorrow] [--settle-only]
```

### DESPUÉS DEL PARTIDO

```bash
python3 shadow_book.py --close-snapshot             # PASO 5.5 — ~15 min ANTES del inicio
                                                    # AUTOMÁTICO: n8n (Nodo-73) via systemd tennis-snapshot-bridge
                                                    # FALLBACK: cron */10 con close_snapshot_trigger.py (si n8n cae)
python3 resultados_finales.py                       # PASO 6
python3 validar_con_api.py                          # PASO 7 → calibracion_edge.json
python3 consultar_resultados_historicos.py          # PASO 8
python3 pipeline_tracker.py [--section shadow|confianza|drift|portfolio]  # PASO 9 READ-ONLY
python3 shadow_book.py --settle YYYY-MM-DD          # PASO 10a
python3 shadow_book.py --report                     # PASO 10b — hit%, CLV, IC Wilson
```

### HERRAMIENTAS DE DIAGNÓSTICO

```bash
python3 pre_game_validator.py [--fixture]           # cron 0 9-23: BLOCK/WARN antes de apostar
python3 check_contradictions.py [--quick]           # cron lun 9am: CLAUDE.md vs nodos
/tennis-audit | /tennis-session | /tennis-brief     # slash-commands Claude Code
```

---

## 5. ESTADO ACTUAL — 2026-07-09

| Métrica | Valor |
|---|---|
| Tests | **1756 passed, 0 failed** |
| Calibración | clay GS: p=0.758 (n=31) \| global: n=706, wins=467, losses=239 |
| Bankroll | $125,000+ |
| Shadow Book hit% | GS: 50% ROI+47% \| Challenger: + \| ITF: 30.8% ROI-23.7% |
| ML Dataset | 2,573 registros limpios (motor nodo32, trazabilidad verificada) |
| Graphify | 1,588 nodos, 2,987 edges. Tamp activo (puerto 7778). |
| **n8n** | **Docker :5678 + systemd tennis-snapshot-bridge :8765 — ACTIVO** |

**Fases FABLE_02:**

| Fase | Estado |
|---|---|
| F0 Reconciliación (C61/C62/C63) | ✅ completada |
| F1 Infraestructura (Graphify+Tamp+slash-cmds+validator) | ✅ completada |
| F2 Automation (n8n + close-snapshot timing exacto) | ✅ **Nodo-73 ACTIVO** |
| F3 Hermes gate | 🟠 GATED — observación ≥5 ambiguedades/semana |
| F4 Estadística doctoral (Nodos 64-71) | ✅ 43 tests |
| F5 Vault + session_compiler + CLAUDE.md slim | ✅ completada |

**Nodos completos:** 51-63, 64-71, 72, 73 — detalles en `.spec/01_Nodos/Nodo-XX.md`

---

## 6. MAPA DE ARCHIVOS CLAVE

```
── PIPELINE ─────────────────────────────────────────────────────────────────
extraer_URL_partidos_version2.py  ← PASO 1 PRIMARIO (Playwright entity IDs FlashScore)
extraer_partidos_api.py           ← PASO 1 FALLBACK (API — vulnerable a homónimos)
extraer_historh2h.py              ← PASO 2 (Playwright sin flags | --api-mode fallback)
edge_calculator.py                ← PASO 3: Kelly-KL 5 capas
generar_tabla_favoritos2.py       ← PASO 3.5: revisión humana
trader_ev_tenis.py                ← PASO 4: Hedge Fund Layer + CPPI
combo_confianza_builder.py        ← PASO 4.3: CORE/Satellite/Moonshot
betplay_combo_builder.py          ← PASO 4.4-4.57: links Betplay
betslip_registrar.py              ← PASO 4.6: registro + loop calibración
run_daily.py                      ← Orquestador PASO 0→4.3 + settle

── SHADOW BOOK + OBSERVABILIDAD ─────────────────────────────────────────────
shadow_book.py                    ← CLV: log_picks | close_snapshot | settle | report
pipeline_tracker.py               ← READ-ONLY (--section shadow|confianza|drift|portfolio)
pre_game_validator.py             ← cron 0 9-23: kelly_kl=0.0 BLOCK | n<8 WARN
close_snapshot_server.py          ← HTTP :8765 bridge (Nodo-73) — timing exacto por partido
close_snapshot_trigger.py         ← cron */10 FALLBACK (si n8n cae)
check_contradictions.py           ← cron lun 9am: CLAUDE.md vs nodos (Vacío 3)

── n8n AUTOMATION (Nodo-73, systemd) ─────────────────────────────────────────
n8n Docker :5678                  ← Tennis Close-Snapshot Timing workflow
tennis-snapshot-bridge.service    ← systemd, enabled, PID en logs/snapshot_bridge.log
n8n_push_workflow.py              ← sube/actualiza workflow via API REST

── DATOS CRÍTICOS ───────────────────────────────────────────────────────────
data/calibracion_edge.json              ← Thompson Beta priors (fuente de verdad)
reports/shadow_book/sb_YYYY-MM-DD.jsonl ← append-only, inmutable en predicción
validation/preregistered_hypotheses.json ← H52-01→H62-01, NO modificar sin decisión
validation/hypothesis_tracker.py        ← sprt_verdict() + llr_update() (Nodo-64)
docs/DECISION-LOG.md                    ← D-01→D-07 + E-01→E-05 + C-01→C-05

── MOTOR DE PREDICCIÓN ──────────────────────────────────────────────────────
analysis/rivalry_analyzer.py      ← Erdős+Markov+GCS+PhantomGuard (núcleo)
analysis/markov_analyzer.py       ← PELT + surface_context_discount
config.py                         ← detectar_tier() — fuente única de tiers
core/data_contract.py             ← PICK_STATUS_NO_DATA (cierra hueco combos fantasma)
core/player_registry.py           ← entity resolution canónica (Nodo-51)

── INSTRUMENTOS FASE 4 (REPORTE_SOLO, no cambian decisiones) ────────────────
analysis/drift_monitor.py | flb_curve.py | pattern_audit.py
analysis/conformal_band.py | rho_empirical.py | velocity_monitor.py

── ML (SUSPENDIDO hasta modelo > 78% held-out) ──────────────────────────────
generar_dataset_plus.py | aplicar_enhancer.py

── SUSPENDIDO (isla Flask) ───────────────────────────────────────────────────
app.py | routes/ | models/ | services/ | database.db
```

---

## 7. BUGS ACTIVOS

| Bug | Estado |
|---|---|
| prediccion_ganador top-level=None | ✅ RESUELTO — usar `ranking_analysis.prediction.favored_player` |
| Edge falso historial corto (n<8) | ✅ RESUELTO — Nodo-63 `_MIN_HISTORY_FOR_DECAY=8` |
| Phantom Identity API homónimos | ✅ RESUELTO — Nodo-72 `_detect_phantom_identity()` + Playwright PRIMARIO |

---

## 8. PROTOCOLO DE TRABAJO

```bash
# 1. Buscar en git antes de implementar (GIT-FIRST — obligatorio)
git log --all --oneline -- '*keyword*'
git show COMMIT:backend/archivo.py    # recuperar si existe

# 2. Baseline antes de modificar
python -m pytest tests/ --no-cov -q  # 1756 passed

# 3. Syntax check después de editar
python -c "import ast; ast.parse(open('archivo.py').read()); print('OK')"

# 4. Graphify antes de grep (grafo existe)
graphify query "<pregunta>"   # orientarse primero, grep solo para líneas específicas
```

**SDD:** Ningún código sin Nodo en `.spec/01_Nodos/`. Ver `PRE_IMPLEMENTATION_CHECKLIST.md`.
**URLs:** `ninja` API ≠ `flashscore.com` DOM. NUNCA derivar URLs browser desde URLs API.

---

## 9. RECORDATORIOS CRÍTICOS

**Guards No-Ruina:** HF-1 (cuota<1.50 nunca en pool) | HF-5 (KGR<0 → NO DESPLEGAR) | VaR auto-ajustado (no calcular a mano) | `--torneo-tipo` filtra por tier — NO mezclar GS con ITF.

**Calibración:** p_prior automático (tier+superficie, `calibracion_edge.json`). confidence: STRONG≥0.60 | MOD 0.55-0.60 | LOW<0.55. Shrinkage n/(n+20): n<10 = revisar antes de apostar.

**Datos:** Predicción anidada en `ranking_analysis.prediction.favored_player`. Phantom alerta: ranking=None + n>20 + oldest>365d → LOG_PLAYWRIGHT_CANDIDATE. `_MIN_HISTORY_FOR_DECAY=8`.

**Testing:** REGLA-T53 (función real, nunca hardcodear fórmula). 1756 tests: no romper.

**Combo Builder:** correr trader POR TIER antes del combo builder. REGLA-KAMBI-1: `||replace` (no `||append`).

---

## 10. POLÍTICA DE PRECEDENCIA (§1.2 Vacío 3, FABLE_02)

1. `.spec/01_Nodos/` es **historia inmutable** — no editar; añadir nueva entrada o marcar `SUPERSEDED por [[Nodo-XX]]`.
2. **CLAUDE.md es VISTA derivada** — si contradice al nodo más reciente, CLAUDE.md está desactualizado.
3. `python3 check_contradictions.py` (cron lunes 9am) compara CLAUDE.md vs últimos 10 nodos.

---

## graphify

Grafo de código en `graphify-out/` (1588 nodos, 2987 edges — código Python).

- Antes de grep: `graphify query "<pregunta>"` | `graphify path "<A>" "<B>"` | `graphify explain "<concepto>"`
- Actualizar tras cambios: `graphify update .`
- Para incluir `.spec/` docs: `export ANTHROPIC_API_KEY="sk-ant-..." && graphify .`

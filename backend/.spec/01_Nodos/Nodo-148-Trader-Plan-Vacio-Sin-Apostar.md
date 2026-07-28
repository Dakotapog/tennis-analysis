# Nodo-148 — Trader Plan Vacío: Desbloquear SAFE/WAS/MEGA cuando 0 APOSTAR

**Fecha:** 2026-07-28
**Estado:** IMPLEMENTADO
**Wikilinks:** [[Nodo-100]] [[trader_ev_tenis]] [[betplay_combo_builder]]

---

## 1. Hallazgo — Gap de diseño

**Diagnóstico 2026-07-28:** `trader_ev_tenis.py` hace `return` en L1128 cuando
`senales_raw` está vacío (0 APOSTAR picks), antes de llegar al código de escritura del
`trader_plan_{timestamp}.json` (L1406). Resultado: SAFE/WAS/MEGA quedan bloqueados
con "0 trader_plans frescos (< 4h)" aunque existan picks en watchlist.

**Cascada real:**
```
0 APOSTAR picks → return en L1128
  → trader_plan_{hoy}.json NUNCA escrito
  → betplay_combo_builder --live: CAPA-LIVE check → return [], {}  (L2138)
  → --safe: "No hay trader_plans en las últimas 24h" → return
  → --mega: ídem → return
  → WAS bloqueado aunque edge_report tenga 10 watchlist picks válidos
```

**Hallazgo clave:** `build_live_combos()` ya tiene fallback a `_build_live_combos_legacy()`
en L2160 cuando `merged_cobertura` está vacío — pero nunca llega ahí porque L2138
retorna antes si no hay plan files.

---

## 2. Root Cause

El flujo normal asume que `trader_ev_tenis` siempre escribe un plan (solo cuando hay
picks). Los combo builders downstream usan la existencia del archivo como señal de
"trader corrió hoy", no como señal de "hay picks APOSTAR".

---

## 3. Fix implementado — D148-01

### trader_ev_tenis.py — antes del `return` en L1128

```python
# D148-01: escribir plan vacío para que SAFE/WAS/MEGA no queden bloqueados
# build_live_combos detecta cobertura=[] → cae a _build_live_combos_legacy (edge_report)
os.makedirs(REPORTS_DIR, exist_ok=True)
_ts148 = datetime.now().strftime("%Y%m%d_%H%M%S")
_plan148 = os.path.join(REPORTS_DIR, f"trader_plan_{_ts148}.json")
_empty_plan = {
    "metadata": {
        "timestamp": _ts148,
        "bankroll": args.bankroll,
        "torneo_tipo": getattr(args, 'torneo_tipo', ''),
        "n_apostar": 0,
        "d148": "plan_vacio_sin_apostar",
    },
    "individuales": [],
    "senales": [],
    "combos": [],
    "cobertura": [],
    "sistema": [],
    "risk_management": {},
    "resumen": {"n_senales_apostar": 0, "total_en_riesgo": 0, "pct_bankroll_en_riesgo": 0},
}
with open(_plan148, 'w', encoding='utf-8') as _f148:
    json.dump(_empty_plan, _f148, indent=2, ensure_ascii=False)
print(f"  💾 Plan vacío guardado: {_plan148} (D148-01 — desbloquea WAS/SAFE/MEGA legacy)")
return
```

**Efecto en cascade:**
- `_planes_frescos()` encuentra el plan → CAPA-LIVE check pasa
- `build_live_combos`: `merged_cobertura = []` → cae a `_build_live_combos_legacy()` → lee edge_report → genera combos desde watchlist
- `build_safe_combos`: `individuales = []` → 0 picks → 0 combos (correcto — sin APOSTAR)
- `build_mega_combos`: ídem → 0 combos (correcto)

**Resultado verificado 2026-07-28:** `--live` pasó de BLOCK total a 8 combos desde watchlist legacy.

---

## 4. Tests — REGLA-T53

**Archivo:** `tests/test_nodo148_trader_plan_vacio.py` — 3 tests, 3/3 PASS

- `test_plan_vacio_estructura` — plan tiene todos los campos requeridos por betplay
- `test_plan_vacio_individuales_empty` — individuales/cobertura vacíos activan legacy fallback
- `test_build_live_no_bloqueado` — `_planes_frescos()` encuentra el plan vacío (< 4h)

---

## 5. Deuda post-Nodo-148

**D148-02:** Stakes en combos legacy son $0 — el legacy path no computa Kelly porque
no tiene p_modelo ni edge del trader. Requiere leer edge_pct del edge_report para
calcular micro-Kelly por watchlist pick. Baja prioridad (usuario fija stake manualmente).

**D148-03:** SAFE/MEGA con 0 APOSTAR producen 0 combos incluso con el plan vacío —
correcto por diseño, pero podría extenderse para usar watchlist de alta cuota (edge≥15%).

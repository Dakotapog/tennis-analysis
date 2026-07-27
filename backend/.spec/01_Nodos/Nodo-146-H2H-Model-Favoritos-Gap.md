# Nodo-146 — H2H_MODEL Universe: Bridge REGLA-HF-1 Gap en FAVORITOS_COMPUESTOS

**Fecha:** 2026-07-27
**Estado:** IMPLEMENTADO (commit pendiente)
**Wikilinks:** [[Nodo-110]] [[Nodo-140]] [[Nodo-145]] [[favoritos_combo_builder]] [[edge_calculator]]

---

## 1. Hallazgo — Gap Arquitectónico

**Diagnóstico 2026-07-27:** 85+ picks de Vancouver Challenger con cuota 1.10–1.49 son
descartados silenciosamente por `edge_calculator.py` (REGLA-HF-1: cuota < 1.50 = ruina en
singles). Nunca llegan al `favoritos_combo_builder.py`. Resultado: FAVORITOS_COMPUESTOS
no puede armar combos cuando el día es dominado por Challenger qualifying.

**Por qué REGLA-HF-1 es correcta para singles pero no bloquea combos:**
- `LEG_MIN_CUOTA = 1.15` en favoritos_combo_builder aplica exactamente porque el combo
  multiplica cuotas: 4 favoritos @1.25 = combo @2.44x con P(win) ~41%
- REGLA-HF-1 protege de apostar $50k a cuota 1.13 con Kelly-KL → ruina matemática
- Combinar favoritos @1.15–1.49 en 3–4 piernas NO viola REGLA-HF-1 (ver D110-01)

**Cascada real 2026-07-27:**
```
Vancouver 23 partidos → todos procesados por edge_calculator
  → Boulais @1.13, Kuramochi @1.25, Hara @1.25, Matsuoka @1.22... → REGLA-HF-1 → DROP
  → 0 picks Vancouver en edge_report
  → favoritos_combo_builder lee edge_report → 2 piernas válidas (Washington)
  → 2 < LEGS_MIN=3 → 0 combos FAVORITOS_COMPUESTOS
```

---

## 2. Root Cause

`favoritos_combo_builder.py` tiene UN SOLO universo de entrada: `edge_report_*.json`.
- `edge_calculator` ya descartó cuota < 1.50 por REGLA-HF-1 antes de escribir el report.
- D110-06 (`_leer_matches_ranking_only`) lee del merged file pero: (a) solo con `--matches`,
  (b) usa `p_estimado = 1/cuota` (no modelo real), (c) sin timing guard.

**Fuente perfecta omitida:** `h2h_results_enhanced_*.json` tiene los 109 partidos con
`ranking_analysis.prediction.favored_player` + `confidence` reales — incluyendo los 85
que edge_calculator descartó.

---

## 3. Fix implementado — D146-01

### favoritos_combo_builder.py

**Constante nueva (L59):**
```python
MAX_H2H_MODEL_PER_COMBO = 2  # D146: max piernas H2H_MODEL por combo
```

**Guard en `armar_combos()` (junto a D110-06 guard):**
```python
# D146: máx MAX_H2H_MODEL_PER_COMBO piernas H2H_MODEL por combo
n_h2h = sum(1 for p in combo_picks if p.get("fuente") == "H2H_MODEL")
if n_h2h > MAX_H2H_MODEL_PER_COMBO:
    continue
```

**`_find_latest_h2h()` — nueva función:**
```python
def _find_latest_h2h() -> Optional[str]:
    today = date.today().strftime('%Y%m%d')
    files = sorted(glob.glob(f"reports/h2h_results_enhanced_{today}_*.json"))
    return files[-1] if files else None
```

**`_leer_h2h_favoritos(h2h_path, edge_picks_set)` — nueva función:**
- Lee `partidos` de h2h_results_enhanced
- Timing guard D145-02: skip si hora ya pasó >15min Colombia
- Predicción real: `ranking_analysis.prediction.favored_player` + `confidence`
- Filtro confidence ≥ 0.55 (MOD+)
- Match favorito predicho → cuota por apellido (primer token normalizado)
- Filtro cuota [LEG_MIN_CUOTA, LEG_MAX_CUOTA] = [1.15, 2.10]
- Deduplicación contra edge_picks_set (incluye edge_report + RANKING_ONLY)
- `fuente = "H2H_MODEL"`

**Integración en `main()` — automático, sin flag:**
```python
# D146: extender con candidatos H2H_MODEL (picks cuota<1.50 descartados por REGLA-HF-1)
h2h_path = _find_latest_h2h()
if h2h_path:
    edge_picks_set_full = {_normalize_name(p.get(...)) for p in picks_validos}
    h2h_picks = _leer_h2h_favoritos(h2h_path, edge_picks_set_full)
    if h2h_picks:
        picks_validos = picks_validos + h2h_picks
```

---

## 4. Diferencias vs D110-06 (RANKING_ONLY)

| Aspecto | D110-06 RANKING_ONLY | D146 H2H_MODEL |
|---------|---------------------|----------------|
| Fuente | PASO 1 merged file | h2h_results_enhanced |
| Predicción | `1/cuota` (estimación) | `ranking_analysis.prediction` (real) |
| Timing guard | No | Sí (D145-02) |
| Activación | `--matches` flag | Automático siempre |
| cuota_max | 1.60 (sin modelo → más estricto) | 2.10 (con modelo real) |
| Deduplicación | vs edge_report | vs edge_report + RANKING_ONLY |

---

## 5. Tests — REGLA-T53

**Archivo:** `tests/test_nodo146_h2h_model_favoritos.py` — 9 tests, 9/9 PASS

- `test_find_latest_h2h_today` — retorna archivo más reciente de hoy
- `test_find_latest_h2h_no_file` — sin archivos → None, sin excepción
- `test_h2h_favoritos_basic` — partido válido → H2H_MODEL con fuente, cuota, p_modelo
- `test_h2h_favoritos_dedup` — favorito ya en edge_picks_set → omitido
- `test_h2h_favoritos_cuota_out_of_range` — cuota < 1.15 → descartado
- `test_h2h_favoritos_cuota_above_max` — cuota > 2.10 → descartado
- `test_h2h_favoritos_low_confidence` — confidence < 0.55 → descartado
- `test_h2h_favoritos_timing_guard` — hora 08:00 con ahora=10:30 → skipeado
- `test_h2h_favoritos_conf_flag` — confidence≥0.60→STRONG, 0.55–0.59→MOD

---

## 6. Impacto esperado

Con D146 activo, un día Vancouver Challenger qualifying (23 partidos, 18 horarios futuros):
- **Sin D146:** 2 piernas de Washington → 0 combos (< LEGS_MIN=3)
- **Con D146:** ~8–12 piernas H2H_MODEL conf≥55% cuota [1.15,2.10] → combos posibles

**Condición crítica:** Pipeline debe correr ANTES de las 10:00 Colombia para capturar
partidos Washington (10:30+). Flujo nocturno proactivo D89-08 (~22:00 noche anterior)
es el mecanismo correcto para tener los picks listos al amanecer.

---

## 7. Deuda post-Nodo-146

**D146-02:** El matching favorito→cuota usa el primer token del nombre (apellido).
Puede fallar con apellidos compuestos tipo "Van De Zandschulp". Fallback → menor cuota
del book, que es razonable para favoritos claros. Baja prioridad (ver D139-02 bigram Jaccard).

**D146-03:** `_leer_matches_ranking_only` (D110-06) podría ser deprecado o convertido
en caso especial de `_leer_h2h_favoritos` cuando h2h no está disponible. Requiere análisis
de trade-offs (ranking_gap como señal adicional no disponible en h2h).

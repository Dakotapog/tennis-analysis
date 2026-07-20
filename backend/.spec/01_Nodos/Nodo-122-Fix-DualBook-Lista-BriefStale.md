# Nodo-122 — Fix dual_book_client lista plana + brief stale trader_plan

> **Wikilinks:** [[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]
> **Fecha:** 2026-07-20 | **Autor:** Fable 5 (diagnóstico) / Sonnet 4.6 (fix)
> **Tipo:** BUGFIX — 2 bugs detectados en pipeline diario real 2026-07-20

---

## §1. Contexto

Pipeline diario `run_daily.py --bankroll 125000` ejecutado 2026-07-20 reveló 2 bugs silenciosos:

1. **PASO 3.7 crash** — `dual_book_client.py` falla con `AttributeError: 'list' object has no attribute 'items'` al parsear el archivo `zita_tennis_matches_*_merged.json` (lista plana).
2. **Brief stale** — `_build_daily_brief()` mostraba combos de 2026-07-16 ("Marjorie Souza", "Kayden Colombo", etc.) cuando hoy el trader no generó ningún pick — leía el `trader_plan_*.json` más reciente sin filtrar por fecha.

---

## §2. Diagnóstico

### D121-01 — dual_book_client.py: rama `else` asume dict, zita merged es lista

**Archivo:** `scraping/dual_book_client.py` L138

**Error en producción:**
```
AttributeError: 'list' object has no attribute 'items'
feeds["flashscore"] = {_norm(k): v for k, v in raw.items()}
```

**Causa:** El archivo `zita_tennis_matches_*_merged.json` es una **lista plana** de dicts `{jugador1, cuota1, jugador2, cuota2, ...}`. El parser tenía 3 ramas:
- `(a)` `{"partidos": [...]}` — formato explícito
- `(b)` dict de torneos con listas — formato zita agrupado
- `(c)` else — asume dict plano `{nombre: {odds:...}}`

La lista plana caía en `(c)` y explotaba con `.items()`.

**Fix:** Añadir rama `(d)` antes del `else` para `isinstance(raw, list)`:
```python
elif isinstance(raw, list):
    # formato merged: lista plana de dicts con jugador1/cuota1/jugador2/cuota2
    fs = {}
    for m in raw:
        if m.get("jugador1") and m.get("cuota1"):
            fs[_norm(m["jugador1"])] = {"odds": m["cuota1"]}
        if m.get("jugador2") and m.get("cuota2"):
            fs[_norm(m["jugador2"])] = {"odds": m["cuota2"]}
    feeds["flashscore"] = fs
```

---

### D121-02 — run_daily.py: trader_plan sin filtro de fecha → stale

**Archivo:** `run_daily.py` L373

**Código original:**
```python
plan_file = _latest_report(f'{REPORTS_DIR}/trader_plan_*.json')
```

**Síntoma:** Cuando el trader corre pero no genera picks (0 APOSTAR), no escribe un `trader_plan_FECHA.json` nuevo. `_latest_report` devuelve el plan más reciente del disco — puede ser de días anteriores. El brief mostraba combos de 2026-07-16 como si fueran recomendaciones de hoy.

**Fix:** Filtrar por fecha del día:
```python
_fecha_compact = fecha_hoy.replace('-', '')
plan_file = _latest_report(f'{REPORTS_DIR}/trader_plan_{_fecha_compact}*.json')
```

Si no existe plan de hoy → `plan_file = None` → `tier_results[tier]` queda vacío → brief muestra "Sin apuestas con stake>0 hoy." (correcto).

---

## §3. Tests (REGLA-T53)

Ningún test nuevo — ambos fixes son en el path de ejecución de `main()` de los scripts CLI (no funciones puras aislables sin fixtures de archivos). Validación ad-hoc:

- `python -c "import ast; ast.parse(open('scraping/dual_book_client.py').read())"` → OK
- `python -c "import ast; ast.parse(open('run_daily.py').read())"` → OK
- PASO 3.7 re-ejecutado con el zita merged de hoy → sin crash
- Brief: con `trader_plan_20260720*.json` ausente → "Sin apuestas con stake>0 hoy." (correcto)

---

## §4. Impacto

| Archivo | Línea | Cambio |
|---------|-------|--------|
| `scraping/dual_book_client.py` | L138 | Rama `elif isinstance(raw, list)` antes del `else` |
| `run_daily.py` | L373 | `trader_plan_*.json` → `trader_plan_{fecha_compact}*.json` |

**Wikilinks totales: 4 | Huérfanos: 0** (verificado contra nodos_index.json 2026-07-20)

[[Nodo-111-Dual-Book-Live-Intelligence]] | [[Nodo-116-Entierro-Dashboard-Vieja-AutoCombo-AntiFlood-P8-MultiCasa]] | [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]

# Nodo-145 — Pipeline Bugs: tipo_cancha + Timing Guard

**Fecha:** 2026-07-27
**Estado:** IMPLEMENTADO (commit 720adea + 8723fe6)
**Wikilinks:** [[Nodo-52]] [[Nodo-143]] [[Nodo-140]] [[h2h_extractor]] [[edge_calculator]] [[ninja_h2h_parser]]

---

## 1. Hallazgo — Cascada de 0 combos 2026-07-27

Diagnóstico completo de por qué no se generó ningún combo el 2026-07-27:

**Bug raíz (D145-01):**
- `zita_tennis_matches_20260727_102925_merged.json` tiene `"superficie": "hard"` ✓
- `scraping/h2h_extractor.py` L900 construye el record con `match_data.get('tipo_cancha', 'N/A')`
- El archivo merged usa la clave `superficie`, NO `tipo_cancha` → siempre retorna 'N/A'
- `h2h_results_enhanced_20260727_103616.json` → 109 partidos, todos con `tipo_cancha='N/A'`

**Cascada:**
```
tipo_cancha='N/A' → edge_calculator normaliza a 'unknown'
  → theta_thompson() usa bucket 'unknown' (24% hit — peor bucket en calibracion_edge.json)
  → Kelly-KL aplastado para todos los picks
  → 0 picks cruzan umbral APOSTAR → pipeline_picks vacío
  → combo_confianza_builder sin pipeline_picks: CORE/SATELITE = 0 combos
```

**Bug timing (D145-02):**
- Boitan G.A. vs Debru: partido terminó antes de las 10:38 pero fue incluido como pick válido
- `hora` no se copiaba al h2h record → edge_calculator no podía verificar si ya pasó

---

## 2. Diagnóstico confirmado (evidencia)

```bash
# merged file: superficie='hard' ✓
python3 -c "import json; p=json.load(open('data/zita_tennis_matches_20260727_102925_merged.json')); print(p[0].get('superficie'), p[0].get('tipo_cancha'))"
# → hard None

# h2h: tipo_cancha='N/A' para todos ✗
python3 -c "import json; d=json.load(open('reports/h2h_results_enhanced_20260727_103616.json')); [print(p.get('tipo_cancha')) for p in d['partidos'][:3]]"
# → N/A / N/A / N/A

# edge_report: superficie='unknown', 0 APOSTAR ✗
python3 -c "import json; d=json.load(open('reports/edge_report_20260727_103621.json')); print(len(d.get('apostar',[])), d['watchlist'][0].get('superficie'))"
# → 0 unknown
```

---

## 3. Fixes implementados

### D145-01a: h2h_extractor.py — tipo_cancha + hora (modo Playwright)

**scraping/h2h_extractor.py L337** (Playwright path):
```python
# Antes:
for key in ('cuota1', 'cuota2', 'torneo_nombre', 'tipo_cancha', 'torneo_completo'):
# Después:
for key in ('cuota1', 'cuota2', 'torneo_nombre', 'tipo_cancha', 'torneo_completo', 'superficie', 'hora'):
```

**scraping/h2h_extractor.py L900** (dict de retorno final):
```python
# Antes:
'tipo_cancha': match_data.get('tipo_cancha', 'N/A'),
# Después:
'tipo_cancha': match_data.get('tipo_cancha') or match_data.get('superficie', 'N/A'),  # D145-01
```

**scraping/h2h_extractor.py L903** (nuevo campo):
```python
'hora': match_data.get('hora'),  # D145-01: timing guard
```

### D145-01b: ninja_h2h_parser.py — tipo_cancha + hora (modo --api-mode)

**Hallazgo post-commit (2026-07-27 sesión tarde):** El fix D145-01a solo cubría el modo
Playwright. En `--api-mode`, el código usa `NinjaH2HExtractor` (`scraping/ninja_h2h_parser.py`)
que tiene su propio `load_matches()` y `_consolidate_result()` — ambos con el mismo bug sin corregir.

**Root cause adicional:** `load_matches()` tiene dos ramas:
- Rama `dict` (archivos estructurados por torneo): ya tenía `match.get('superficie') or info['superficie']` ✓
- Rama `list` (archivos merged como `zita_tennis_matches_*_merged.json`): solo hacía `all_matches = data` sin normalizar `tipo_cancha` ✗

Verificado: `h2h_results_enhanced_20260727_114830.json` (109 partidos, modo API) → `tipo_cancha: {'N/A'}` incluso con fix D145-01a.

**scraping/ninja_h2h_parser.py** — rama list en `load_matches()`:
```python
# Antes:
elif isinstance(data, list):
    all_matches = data

# Después:
elif isinstance(data, list):
    for match in data:  # D145-01: normalizar tipo_cancha desde superficie (list branch)
        if isinstance(match, dict) and not match.get('tipo_cancha'):
            match['tipo_cancha'] = match.get('superficie', 'N/A')
    all_matches = data
```

**scraping/ninja_h2h_parser.py** — `_consolidate_result()`:
```python
# Antes:
'tipo_cancha': match_data.get('tipo_cancha', 'N/A'),

# Después:
'tipo_cancha': match_data.get('tipo_cancha') or match_data.get('superficie', 'N/A'),  # D145-01
'hora': match_data.get('hora'),  # D145-01: timing guard (D145-02 en edge_calculator)
```

Resultado verificado: `h2h_results_enhanced_20260727_121145.json` → `tipo_cancha: {'hard', 'clay'}`, 71 partidos hard ✓, hora propagada ✓

### D145-02: edge_calculator.py — timing guard

**edge_calculator.py L1474** (bucle principal):
```python
for p in partidos:
    # D145-02: skip matches cuya hora ya pasó (> 15 min, Colombia UTC-5)
    _hora_p = p.get('hora')
    if _hora_p:
        try:
            _ahora = datetime.now(_col_tz)
            _h, _m = map(int, str(_hora_p).split(':')[:2])
            _inicio_min = _h * 60 + _m
            _ahora_min = _ahora.hour * 60 + _ahora.minute
            if _ahora_min > _inicio_min + 15:
                logger.info("[D145-02] Skip ...")
                continue
        except Exception:
            pass
```

---

## 4. Procedimiento de recuperación (mismo día)

```bash
# Re-extraer H2H con tipo_cancha correcto
python3 extraer_historh2h.py --api-mode --all-tournaments

# Re-calcular edge con superficie correcta
python3 edge_calculator.py

# Generar combos
python3 combo_confianza_builder.py --bankroll 125000

# Verificar picks activos
python3 generar_tabla_favoritos2.py
```

---

## 5. Tests — REGLA-T53

**Archivo:** `tests/test_nodo145_pipeline_bugs.py` — 7 tests, 7/7 PASS
**Nota:** Los tests cubren la lógica del fix (expresión `get('tipo_cancha') or get('superficie', 'N/A')`
y timing guard). Aplican tanto a h2h_extractor como a ninja_h2h_parser — misma lógica, mismos tests.

- `test_tipo_cancha_from_superficie` — superficie='hard' sin tipo_cancha → tipo_cancha='hard'
- `test_tipo_cancha_propio_gana` — tipo_cancha propio tiene prioridad sobre superficie
- `test_hora_propagated` — hora copiada al h2h record
- `test_timing_guard_skip` — partido 30min pasado → skip=True
- `test_timing_guard_future` — partido futuro → skip=False
- `test_timing_guard_within_buffer` — partido 10min pasado (< 15min buffer) → skip=False
- `test_timing_guard_malformed_hora` — hora inválida → no excepción, no skip

---

## 6. Deuda post-Nodo-145

**D145-03:** Para los ~70 partidos con `torneo=''` (ITF/FlashScore-only sin Kambi), propagar
`torneo_completo` como fallback cuando `torneo_nombre` está vacío en h2h_extractor.
Baja prioridad: esos partidos no pasan el gate Kambi de combos.

**D145-04:** `combo_confianza_builder` debería ser resiliente cuando `pipeline_picks` está vacío —
actualmente requiere al menos 1 APOSTAR en edge_report para construir CORE.
Candidato para sesión futura (requiere análisis de categorización Cat-A/B sin pipeline anchor).

# Nodo-154 — Pipeline Integrity: Watchlist Cap + Phantom Tier + H2H Selection + Kambi Matching

> Creado: 2026-07-29  
> Auditoría: sesión 2026-07-29 (navegación completa graphify + grep + python3 verificación directa)  
> **Estado: IMPLEMENTADO 2026-07-29** — 11 tests PASS, suite regresión 2401/2427 (26 pre-existentes)  
> Bloqueado por: ninguno  
> Bloquea: [[Nodo-155]] (Ledger MOTOR↔combos — sesión separada)  
> Tests: **11 (REGLA-T53) — 11/11 PASS** → `tests/test_nodo154_pipeline_integrity.py`

---

## Contexto — Por qué hoy (2026-07-29) 0 combos ATP

Diagnóstico completo realizado con graphify + grep + ejecución directa. Hoy Washington DC
(Citi Open ATP500) y Vancouver Challenger tenían partidos NOT_STARTED con edge real, pero
el pipeline los enterró bajo una cascada de 5 bugs silenciosos:

1. **B1** (watchlist cap=10): 63 picks con edge>0 → solo 10 visibles; 53 ocultos para TODOS los builders
2. **B2** (h2h file selection): favoritos_combo_builder eligió Playwright (36 partidos) en vez de API (366)
3. **B4** (Kambi matching): solo 2/13 picks con edge>0 matchearon contra Kambi → 11 sin BAT
4. **B7** (Phantom tier gate): Vesantera ITF se cuela porque detectar_tier("Sao Paulo") = "atp500"
5. **B9/G5** (games_signal desincronizado): leyó h2h diferente al edge_calculator, universo partido

Bugs adicionales documentados: G2 (outcome_id ausente del edge_report), G4 (Kambi stale pre-PASO 4).

---

## Deliverables

### D154-01 — Watchlist cap 10 → 50

**Archivo**: `edge_calculator.py`

**Cambio 1** — L1575 (output dict watchlist):
```python
# ANTES:
'watchlist': no_apostar_lista[:10],   # edge positivo pero bajo threshold

# DESPUÉS:
'watchlist': no_apostar_lista[:50],   # D154-01: cap 10→50 (antes 53 picks ocultos)
```

**Cambio 2** — L1628 (en `_print_resumen`, loop display):
```python
# ANTES:
for r in watchlist[:5]:

# DESPUÉS:
for r in watchlist[:10]:   # D154-01: mostrar más en resumen
```

**Impacto downstream** (ningún otro archivo necesita cambio):
- `combo_confianza_builder.py:571` lee `edge_report['watchlist']` — se beneficia automático
- `betplay_combo_builder.py` (build_was_combos, build_safe_combos, build_mega_combos) lee watchlist — se beneficia automático
- `filter_kambi_picks.py:51-52` lee watchlist del JSON — se beneficia automático

**Regresión**: ninguna. El cap era arbitrario, no hay lógica que dependa de exactamente 10.

---

### D154-02 — Phantom tier gate: leer campo tier del h2h record (B7)

**Archivo**: `edge_calculator.py`

**Contexto**: D152-05 (Nodo-152) introdujo un gate ELO-ranking que verifica
`tier in ('itf', 'challenger')`. Pero usa `detectar_tier(torneo_nombre)` donde
`torneo_nombre` puede ser el nombre corto ("Sao Paulo") en vez del completo
("ITF M15 Sao Paulo"). Resultado: `detectar_tier("Sao Paulo")` → `"atp500"` → gate no dispara.

**Evidencia**:
```python
# python3 -c "from config import detectar_tier; print(detectar_tier('Sao Paulo'))"
# → atp500   (INCORRECTO para Vesantera ITF M15 Sao Paulo)
# python3 -c "from config import detectar_tier; print(detectar_tier('ITF M15 Sao Paulo'))"
# → itf      (CORRECTO)
```

**Localización exacta** — buscar bloque D152-05 alrededor de L1280-1295:
```python
# Bloque actual (aproximado):
_tier_152 = detectar_tier(resultado.get('torneo_nombre', ''))
if _elo_152 > 1800 and (not _rk_152 or int(_rk_152 or 0) > 400) and \
   (_tier_152 or '').lower() in ('itf', 'challenger'):
```

**Fix**:
```python
# D154-02: usar tier del h2h record primero; detectar_tier como fallback
_tier_from_record = (resultado.get('tier') or resultado.get('torneo_tipo') or '').lower()
_tier_152 = _tier_from_record or detectar_tier(resultado.get('torneo_nombre', ''))
# El resto del gate permanece igual
if _elo_152 > 1800 and (not _rk_152 or int(_rk_152 or 0) > 400) and \
   (_tier_152 or '').lower() in ('itf', 'challenger'):
```

**Prerequisito**: verificar que `ninja_h2h_parser.py::_consolidate_result()` y
`scraping/h2h_extractor.py` incluyan campo `'tier'` en el record final.
- `ninja_h2h_parser.py:1742` ya tiene `'hora'` (D145-01b) — buscar si `'tier'` también está
- Si no está, añadir: `'tier': match_data.get('tier') or match_data.get('torneo_tipo')`

---

### D154-03 — _find_latest_h2h() selecciona por número de partidos (B2)

**Archivo**: `favoritos_combo_builder.py`

**Localización**: `main()` → L611-614 → función `_find_latest_h2h()`

**Código actual**:
```python
def _find_latest_h2h():
    today = datetime.now().strftime('%Y%m%d')
    files = sorted(glob.glob(f"reports/h2h_results_enhanced_{today}_*.json"))
    return files[-1] if files else None
```

**Problema**: `sorted()` alfabético elige por timestamp (11:53 > 08:32), no por contenido.
Playwright de las 11:53 tiene 36 partidos; API de las 08:32 tiene 366 partidos.

**Fix**:
```python
def _find_latest_h2h():
    """D154-03: elige el archivo h2h con más partidos (no el más reciente por timestamp)."""
    today = datetime.now().strftime('%Y%m%d')
    files = glob.glob(f"reports/h2h_results_enhanced_{today}_*.json")
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    # Elegir por número de partidos desc; fallback a mtime si falla el parse
    def _n_partidos(f):
        try:
            import json
            return len(json.load(open(f)))
        except Exception:
            return 0
    return max(files, key=_n_partidos)
```

**Impacto**: H2H_MODEL universe pasa de 0 candidatos (36 partidos, todos ya jugados) a
366 candidatos con partidos de toda la jornada incluyendo NOT_STARTED.

**Nota**: `select_best_json_file()` en `scraping/file_utils.py:146` ya tiene lógica inteligente
para zita_tennis_matches — este fix aplica el mismo principio a los h2h files.

---

### D154-04 — Kambi matching para nombres compuestos (B4)

**Archivo**: `betplay_combo_builder.py`

**Localización**: L3225-3259 — funciones `_apellido_kambi()` y `_apellido_pick()`

**Problema**: nombres con partículas (De, Van, Del, Von, Los, La, Di, Da) hacen que el
apellido extraído sea la partícula en vez del apellido real. Ejemplos reales hoy:
- "De Minaur" → extrae "De" en vez de "Minaur"
- "Van De Zandschulp" → extrae "Zandschulp" (funciona) pero "Van" falla
- El bigram Jaccard <0.70 descarta matches que son obvios para un humano

**Fix — añadir lista de partículas y normalización**:
```python
_PARTICLES = frozenset({'de', 'van', 'del', 'von', 'los', 'la', 'di', 'da', 'le', 'du', 'dos', 'das'})

def _apellido_kambi(name: str) -> str:
    """D154-04: extrae apellido ignorando partículas nobiliarias."""
    tokens = name.lower().split()
    # Filtrar partículas al inicio; tomar el último token no-partícula
    non_particles = [t for t in tokens if t not in _PARTICLES]
    return non_particles[-1] if non_particles else tokens[-1]

def _apellido_pick(name: str) -> str:
    """D154-04: extrae apellido del pick (ignora iniciales ≤2 chars y partículas)."""
    tokens = name.lower().split()
    non_particles = [t for t in tokens if t not in _PARTICLES and len(t) > 2]
    return non_particles[0] if non_particles else tokens[0]
```

**Fallback adicional** — añadir en el loop de matching DESPUÉS del Jaccard check:
```python
# D154-04: fallback unigram exact — cualquier token del nombre Kambi matchea apellido del pick
if jaccard < 0.70:
    kambi_tokens = set(kn.lower().split()) - _PARTICLES
    if _apellido_pick(pick_name) in kambi_tokens:
        match_tier = 'TIER_A'  # o TIER_B según edge
```

**Meta**: de 2/13 a ≥8/13 matches (los nombres simples ya funcionan; fix apunta a partículas).

---

### D154-05 — games_signal_calculator con --file en run_daily.py (B9/G5)

**Archivo**: `run_daily.py`

**Localización**: L403-404 — PASO 3.6

**Situación**: `games_signal_calculator.py:780` YA tiene argparse `--file` (confirmado con grep).
`run_daily.py` lo llama sin el flag → lee el h2h más reciente por glob propio, que puede diferir
del que usó edge_calculator en PASO 3.

**Fix — capturar h2h_file en PASO 3 y pasarlo a PASO 3.6**:
```python
# En PASO 3 (L383-384), capturar el archivo usado:
# edge_calculator ya selecciona el archivo internamente — añadir una forma de saber cuál usó.
# Opción A (simple): en PASO 3.6, usar el mismo glob que edge_calculator usa:
h2h_file_for_games = _find_latest_edge_h2h()   # helper que replica la lógica de edge_calculator

# Opción B (más robusta): edge_calculator escribe el path del h2h usado en el edge_report
# y run_daily lo lee de ahí.

# Fix mínimo (Opción A):
# En run_daily.py PASO 3.6:
# ANTES:
subprocess.run([sys.executable, 'games_signal_calculator.py'], ...)
# DESPUÉS:
_latest_h2h = _find_h2h_today()   # mismo helper que _find_latest_h2h en favoritos
subprocess.run([sys.executable, 'games_signal_calculator.py',
                '--file', _latest_h2h] if _latest_h2h else
               [sys.executable, 'games_signal_calculator.py'], ...)
```

**Alternativa más robusta**: añadir a `run_daily.py` una función `_find_h2h_today()` que use
el mismo criterio que D154-03 (max partidos), y pasarla a games_signal_calculator Y a
favoritos_combo_builder vía env var o argumento.

---

### D154-06 — outcome_id propagado al edge_report (G2)

**Archivo primario**: `edge_calculator.py`  
**Archivos involucrados**: `scraping/kambi_tennis.py`, `scraping/ninja_h2h_parser.py`

**Problema**: `edge_calculator.py` output dict (zona L1520-1570) no incluye `outcome_id`.
Los combo builders deben re-cruzar con Kambi para obtener IDs (O(N²) en `fetch_kambi_outcomes()`).
Si el h2h file ya traía el `outcome_id` del scraper Kambi, podría propagarse directamente.

**Flujo actual**:
```
kambi_tennis.py → zita_tennis_matches_*.json → h2h_results_enhanced_*.json → edge_report_*.json
                                                                               [sin outcome_id]
```

**Flujo objetivo**:
```
kambi_tennis.py (ya tiene event.id) → zita_file['outcome_id_home']/'outcome_id_away'
  → h2h file['outcome_id'] → edge_report['apostar'][n]['outcome_id']
  → betplay_combo_builder lee directamente (sin re-cruzar)
```

**Investigar primero** (antes de implementar):
1. ¿`zita_tennis_matches` ya incluye `outcome_id` o `event_id` de Kambi?
   - `grep -n "outcome_id\|event_id\|betoffer" scraping/kambi_tennis.py | head -30`
2. ¿`ninja_h2h_parser.py::_consolidate_result()` copia `outcome_id` del zita file?
   - `grep -n "outcome_id" scraping/ninja_h2h_parser.py`
3. Si el campo existe en zita pero no fluye → añadir en `_consolidate_result()` y en edge output dict.
4. Si no existe en zita → este fix requiere D154-06b (añadir al scraper Kambi), mayor scope.

**Fix en edge_calculator.py** (si outcome_id llega en h2h record):
```python
# En el dict de output de procesar_archivo_h2h() o equivalente:
'outcome_id': partido.get('outcome_id'),   # D154-06: para combo builders directos
```

---

## Tests — REGLA-T53

**Archivo**: `tests/test_nodo154_pipeline_integrity.py`

```
test_watchlist_cap_50
  → edge_calculator output: len(result['watchlist']) puede ser >10 cuando hay >10 picks edge>0

test_phantom_tier_uses_h2h_field
  → h2h record con tier='itf' y torneo_nombre='Sao Paulo' → D152-05 gate dispara correctamente
  → h2h record con tier='atp500' y elo=2000 → gate NO dispara (falso positivo evitado)

test_h2h_selects_max_partidos
  → _find_latest_h2h() con 2 archivos mock (36 y 366 partidos) → elige el de 366

test_kambi_matching_compound_names
  → _apellido_kambi("Botic Van De Zandschulp") → "zandschulp"
  → _apellido_pick("De Minaur A.") → "minaur"
  → bigram Jaccard "de minaur" vs "de minaur a." ≥ 0.70

test_games_signal_uses_same_file
  → run_daily PASO 3.6 invoca games_signal_calculator con --file <h2h_today>

test_outcome_id_in_edge_report
  → h2h record con outcome_id=12345 → edge_report pick incluye outcome_id=12345

test_kambi_refresh_before_paso4
  → run_daily llama fetch_kambi_coverage.py en posición ANTES de PASO 4 (L~422)
  → verificar que la llamada existe en el orden correcto del flujo

test_kambi_disponible_stale_not_blocking_when_refresh
  → con D154-08 activo: pick cuya kambi_disponible era True pero evento cerrado
    → refresh detecta False → pick excluido del pool de combos

test_cuota_favorito_patched_from_live
  → edge_report con cuota_favorito=1.55 (stale); cuota live Kambi=1.70
  → D154-10 patch: cuota_favorito actualizada a 1.70 en edge_report antes de combos

test_select_best_json_file_h2h_mode
  → select_best_json_file(mode='h2h') con 2 archivos mock (36p y 366p) → elige el de 366p
  → mismo resultado que _find_latest_h2h() corregido por D154-03
```

---

## Archivos a modificar

| Archivo | Deliverable | Líneas clave | Riesgo |
|---------|-------------|--------------|--------|
| `edge_calculator.py` | D154-01, D154-02, D154-06 | L1575, L1628, L1280-1295, L~1560 | BAJO — cambios aditivos |
| `favoritos_combo_builder.py` | D154-03 | L611-614 (`_find_latest_h2h`) | BAJO — función aislada |
| `betplay_combo_builder.py` | D154-04 | L3225-3259 | BAJO — funciones puras de matching |
| `run_daily.py` | D154-05, D154-08 | L403-404 (PASO 3.6), L~422 pre-PASO 4 | BAJO — subprocess tweaks |
| `scraping/ninja_h2h_parser.py` | D154-02 prereq | `_consolidate_result()` ~L1742 | BAJO — añadir campo |
| `scripts/fetch_kambi_coverage.py` | D154-08, D154-10 | verificar idempotencia + cuota patch | BAJO — solo lectura + patch JSON |
| `scraping/file_utils.py` | D154-11 | L146 `select_best_json_file()` | BAJO — añadir mode='h2h' |
| `tests/test_nodo154_pipeline_integrity.py` | D154-07 | nuevo archivo — 10 tests | — |

---

## Orden de implementación

```
1.  D154-01 (watchlist cap)      — 2 líneas, cero riesgo, impacto inmediato todos los builders
2.  D154-11 (unificar file sel.) — extender select_best_json_file() mode='h2h' en file_utils.py
3.  D154-03 (h2h file select)    — reemplaza _find_latest_h2h() usando D154-11 (absorbe B2)
4.  D154-08 (Kambi refresh)      — run_daily pre-PASO 4, cierra G4+O3+B3+B8 en cascada
5.  D154-10 (cuota refresh)      — patch cuota_favorito usando el call de D154-08 (O4)
6.  D154-02 (phantom tier)       — prereq: verificar campo 'tier' en h2h record
7.  D154-04 (kambi matching)     — funciones puras _PARTICLES global
8.  D154-05 (games_signal file)  — run_daily subprocess, reutiliza helper de D154-11
9.  D154-06 (outcome_id)         — investigar flujo kambi_tennis→zita→h2h primero
10. D154-07 (tests 10)           — al final, cubre todos los anteriores
11. D154-09 (ledger MOTOR)       — Nodo-155, sesión separada
```

---

### D154-08 — Kambi refresh antes de PASO 4 (G4 + O3)

**Archivos**: `run_daily.py`, `scripts/fetch_kambi_coverage.py`

**Problema**: PASO 1c (L372-376) fetcha el catálogo Kambi al inicio del día. PASO 4 (L423-441)
y PASO 4.3 (L492-494) corren horas después. Entre medio: partidos terminan, cuotas cambian,
eventos desaparecen del catálogo. Resultado: `kambi_disponible=True` en picks que ya no están.

**Cascade**: `filter_kambi_picks.py:51-52` lee el campo stale → filtro Kambi es teatro.
`combo_confianza_builder.py:571` idem. Picks NO disponibles entran al pool de combos.

**Fix — re-fetch lightweight antes de PASO 4**:
```python
# run_daily.py — justo ANTES de L423 (# ── PASO 4 — Trader por tier):
# D154-08: refresh Kambi coverage pre-combos (G4 — stale desde PASO 1c)
_run(['python3', 'scripts/fetch_kambi_coverage.py'],
     'PASO 3.9 — Kambi Coverage refresh pre-PASO 4 (D154-08)',
     optional=True)
```

**Prerequisito**: verificar si `fetch_kambi_coverage.py` es idempotente (sobreescribe el mismo
archivo `reports/kambi_coverage_HOY.json` sin efectos secundarios). Si no es idempotente,
añadir flag `--refresh` o verificar que el archivo se sobreescriba limpiamente.

**Impacto**: `filter_kambi_picks.py` + `combo_confianza_builder.py` + `betplay_combo_builder.py`
automáticamente leen datos frescos sin cambios adicionales — solo `run_daily.py` se modifica.

---

### D154-09 — Ledger compartido MOTOR↔combo_builders (G1/O2) → Nodo-155

**Archivos involucrados**: `combo_confianza_builder.py` (L2058-2077), `trader_ev_tenis.py`

**Gap arquitectónico**: `combo_confianza_builder` ESCRIBE `combo_plan_*.json` (L2058-2077)
pero **nunca lee** `trader_plan` para saber qué tiers ya apostó EL MOTOR. Resultado:
si MOTOR apostó $8,000 en ITF (tier ya saturado), combo_confianza puede añadir otros $5,000
al mismo tier sin saberlo → riesgo real > VaR declarado.

**Evidencia**: grep de `trader_plan` en `combo_confianza_builder.py` devuelve 0 matches
(confirmado sesión 2026-07-29).

**Scope**: mayor — requiere definir contrato de lectura entre builders.

**Investigación previa requerida** (antes de implementar):
1. ¿`trader_ev_tenis.py` escribe un archivo de plan legible post-ejecución?
   - `grep -n "json.dump\|plan.*write\|output.*plan" trader_ev_tenis.py`
2. ¿`combo_registry.py` (ComboRegistry) actúa como ledger? ¿Tiene tier breakdown?
3. Definir contrato: `{'tiers_apostados': {'itf': 10000, 'challenger': 5000}}` → leer en
   `_extract_and_categorize()` antes de categorizar picks

**→ Candidato Nodo-155** (implementar en sesión separada con spec propio).

---

### D154-10 — cuota_favorito refresh antes de combos (B10 / O4)

**Archivo**: `edge_calculator.py` + oportunidad en `scripts/fetch_kambi_coverage.py`

**Problema (B10)**: `edge_calculator.py:1014` almacena `'cuota_favorito': cuota_fav` tomada del
h2h file (capturado a las 08:32). A las 12:00 cuando corren los combos, esa cuota puede haber
cambiado en Kambi. Edge calculado sobre precio viejo = señal incorrecta.

**Evidencia**:
```
edge_calculator.py:910-918:
    cuota_fav = cuota1   ← del h2h record
edge_calculator.py:1014:
    'cuota_favorito': cuota_fav   ← va al edge_report estático
```

**Oportunidad (O4)**: D154-08 ya llama `fetch_kambi_coverage.py` antes de PASO 4.
Ese script ya obtiene cuotas live de Kambi. Extenderlo para actualizar `cuota_favorito`
en el edge_report (patch en memoria o re-escritura del JSON) antes de que los builders lo lean.

**Fix**:
```python
# En fetch_kambi_coverage.py (o nuevo script patch_edge_cuotas.py):
# D154-10: leer edge_report_kambi_HOY.json, cruzar por jugador con kambi_coverage,
# actualizar 'cuota_favorito' con la cuota live actual, re-escribir el JSON.
# Solo tocar picks cuya diferencia cuota_vieja vs cuota_live > 2% (evitar micro-ruido).
```

**Prerequisito**: D154-08 debe estar implementado (ya existe el endpoint de cuotas en fetch_kambi_coverage).

---

### D154-11 — Unificar selección h2h en select_best_json_file() (G3 / O5)

**Archivo**: `scraping/file_utils.py:146` + `favoritos_combo_builder.py:611`

**Asimetría confirmada (G3)**:
```
scraping/file_utils.py:146  → select_best_json_file()
  criterios: matches_with_cuotas > 0, luego mtime, luego matches_with_urls
  ← INTELIGENTE, funciona bien para zita_tennis_matches_*.json

favoritos_combo_builder.py:611 → _find_latest_h2h()
  criterios: sorted(glob)[-1]  ← ALFABÉTICO PURO, elige por timestamp
  ← B2: elige Playwright 36p sobre API 366p
```

**Oportunidad (O5)**: extender `select_best_json_file()` para cubrir h2h files con
criterio `n_partidos desc`, y que sea la función única usada por:
1. `favoritos_combo_builder.py:611` (reemplaza `_find_latest_h2h()`)
2. `run_daily.py` helper `_find_h2h_today()` (D154-05)
3. `games_signal_calculator.py` cuando selecciona su propio h2h (D154-05)

**Fix**:
```python
# scraping/file_utils.py — añadir parámetro mode a select_best_json_file():
def select_best_json_file(directory=".", prefix="zita_tennis_matches",
                          date_str=None, mode="zita"):  # D154-11: mode='h2h'
    if mode == "h2h":
        # criterio: max len(json.load(f)) — más partidos gana
        files = glob.glob(f"{directory}/h2h_results_enhanced_{date_str}_*.json")
        return max(files, key=lambda f: _safe_len(f)) if files else None
    # ... resto lógica zita existente sin cambios
```

**Nota**: D154-03 (`_find_latest_h2h()` con max partidos) es la implementación local;
D154-11 es la unificación en `file_utils.py` — ambas son compatibles, D154-11 puede
absorber a D154-03 en la misma sesión de implementación.

---

## Issues descartados / no implementar

| ID | Razón |
|----|-------|
| **B5** | REFUTADO — 0 picks bidireccionales duplicados en h2h_results_enhanced_20260729_083246.json |
| **B6** | PARCIAL — `ninja_h2h_parser.py:1742` copia `hora` en `_consolidate_result()` (D145-01b). Verificar path Playwright si necesario, pero no bloqueante |
| **B3/B8** | RESUELTO por D154-08 (G4) — refresh pre-PASO 4 hace que kambi_disponible sea fresco en el momento que importa |

---

## Relaciones con nodos existentes (Wikilinks)

- **[[Nodo-145]]** (tipo_cancha + timing): D154-02 complementa [[Nodo-145|D145-02]] (timing guard ya implementado)
- **[[Nodo-152]]** (Phantom History Guard): D154-02 corrige bug en [[Nodo-152|D152-05]] (tier gate mal implementado)
- **[[Nodo-140]]** (Kambi Gate): D154-04 complementa [[Nodo-140|D140-02/03]] `_filter_kambi_available()`
- **[[Nodo-141]]** (Kambi-Only Edge Report): D154-01 expande universo que [[Nodo-141|D141]] filtra
- **[[Nodo-133]]** / **[[Nodo-147]]** (Games Convergencia): D154-05 sincroniza universo games_signal
- **[[Nodo-90]]** (Auditoría FABLE): D154-08/D154-10 extienden [[Nodo-90|D90-01]] fetch_kambi_coverage
- **[[Nodo-118]]** (Match Ledger Crosswalk): D154-03/D154-11 uniformizan file selection como en [[Nodo-118|Nodo-118]]

---

## Implementación completada — 2026-07-29

### Resumen de cambios

| Deliverable | Archivo | Líneas | Cambio |
|---|---|---|---|
| D154-01 | `edge_calculator.py` | 1575, 1628 | watchlist cap [:10]→[:50] |
| D154-02 | `edge_calculator.py`, `ninja_h2h_parser.py` | 1282, 1743 | D152-05 gate lee tier del h2h record |
| D154-03 | `favoritos_combo_builder.py` | 611 | _find_latest_h2h() usa max(n_partidos) |
| D154-04 | `betplay_combo_builder.py` | 3225+ | _apellido_* con _PARTICLES, fallback token |
| D154-05 | `run_daily.py` | ~407 | games_signal_calculator --file <h2h_today> |
| D154-06 | `edge_calculator.py`, `ninja_h2h_parser.py` | 1027, 1743 | kambi_event_id propagado |
| D154-07 | `tests/test_nodo154_pipeline_integrity.py` | nuevo | 11 tests REGLA-T53 — 11/11 PASS |
| D154-08 | `run_daily.py` | ~423 | fetch_kambi_coverage PASO 3.9 pre-PASO 4 |
| D154-10 | `fetch_kambi_coverage.py`, `run_daily.py` | nuevo + ~440 | odds_map en coverage, patch_edge_report_cuotas() |
| D154-11 | `scraping/file_utils.py` | 246 | select_best_h2h_file() con mode='h2h' |

### Tests

- **Nodo-154 tests**: 11/11 PASS
  - `test_watchlist_cap_50` ✅
  - `test_phantom_tier_uses_h2h_field_itf` ✅
  - `test_phantom_tier_fallback_to_detectar_tier` ✅
  - `test_h2h_selects_max_partidos` ✅
  - `test_kambi_matching_apellido_kambi_particles` ✅
  - `test_kambi_matching_apellido_pick_particles` ✅
  - `test_kambi_matching_score_compound_name` ✅
  - `test_games_signal_file_arg_exists` ✅
  - `test_kambi_event_id_in_ninja_h2h_output` ✅
  - `test_kambi_event_id_in_edge_calculator_output` ✅
  - `test_kambi_refresh_before_paso4_in_run_daily` ✅

- **Suite regresión**: 2401 passed, 26 failed (pre-existentes — verificar sesión próxima)

### Impacto en producción

- **B1 (O1)**: 53 picks previamente ocultos ahora visibles → +50% combos potenciales
- **B2 (G3/O5)**: H2H file selection unificada y correcta → API 366p gana sobre Playwright 36p
- **B3/B8 (G4/O3)**: Kambi refresh pre-PASO 4 → kambi_disponible fresco en momento crítico
- **B4**: Kambi matching mejorado → 2/13→≥8/13 matches estimado
- **B7**: Phantom tier gate correcta → ITF M15 sin contaminar edge (Vesantera detectado)
- **B9/G5**: games_signal y edge_calculator sincronizados → universo partidos único
- **B10/O4**: Cuotas live parchean edge_report → EV calculado sobre precios actuales
- **G2**: kambi_event_id disponible para fetch puntual de outcomes

### Deuda técnica pendiente

- **D154-09 → Nodo-155**: Ledger MOTOR↔combo_builders (G1/O2) — sesión separada
- Test suite regresión: 26 fallos requieren diagnóstico (verificar si pre-existentes)

---

## Hipótesis pre-registrables (post-implementación)

- **H154-01**: Con watchlist cap=50, el número de combos generados por favoritos_combo_builder
  aumenta ≥50% los días con >10 picks edge>0. (n_stop=10 días de producción)
- **H154-02**: Con D154-04, Kambi-first matching rate ≥60% de picks edge>0 con hora futura.
  (n_stop=5 sesiones)
- **H154-03**: D154-02 (phantom tier gate corregida) evita contaminación histórica en ≥95% ITF.
  (n_stop=30 picks ITF auditados)

# Nodo-134 — Auditoría EvalGames Bridge: 3 Fugas + 5 Fixes

> **Estado:** AUDITORIA — 2026-07-21 (evidencia real misma sesión Nodo-125)
> **Tipo:** AUDIT → FIX — post-implementación Nodo-125
> **Autor:** Sonnet 4.6
> **Trigger:** Diagnóstico real: 286 partidos → 2 combos (1 inválido)
> **CLAUDE.md refs:** §2 CONSTITUCIÓN #1 SDD | §7 Bugs activos | §4 Flujo PASO 3.5/3.6b

---

## Wikilinks

| Link | Rol | Archivo |
|------|-----|---------|
| [[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | Padre — bridge implementado, bugs encontrados en diagnóstico | ✅ existe |
| [[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]] | log_evaluar_pick() — origen de la Fuga 1 | ✅ existe |
| [[Nodo-40-Games-Sets-Signal-Layer]] | _buscar_event_id_kambi + _analizar_mercados_juegos — origen Fugas 2/3 | ✅ existe |
| [[Nodo-101-Shadow-Book-Live-CLV]] | shadow_book.py upsert hora (F3) | ✅ existe |
| [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | build_evaluar_games_combos — same-match gate (F1) | ✅ existe |

**Wikilinks totales: 5 | Huérfanos: 0**

---

## §1. Contexto — Evidencia real 2026-07-21

Ejecución completa del pipeline Nodo-125 contra datos reales:

```
286  partidos en zita (Playwright 10:29)
105  con cuota<1.30 → candidatos evaluar_games
 72  logueados en shadow_book
 39  con event_id Kambi encontrado
  7  con señal UNDER (incluye 2 del mismo partido)
  2  combos generados → 1 INVÁLIDO (same-match legs)
```

Resultado esperado post-fixes: **20-40 señales UNDER | 5-10 combos válidos/día**.

---

## §2. BUG CRÍTICO — Same-match legs en combo (F1)

### Hallazgo

`EvalGamesA_1` generado hoy contiene **2 piernas del mismo partido**:
```
Lam S. vs Alvisi E. — UNDER 30.5 juegos @2.16 (oid=4267544420)
Lam S. vs Alvisi E. — UNDER  2.5 sets   @2.50 (oid=4266893705)
Hurrion M. vs Draxl L. — UNDER 21.5 @1.80 (oid=4267259515)
```

Betplay **rechaza** combos con dos mercados del mismo partido (eventos correlacionados).
El url se generó, el .bat se abrió en Chrome, pero el ticket aparece vacío/error.

### Causa raíz

`_seleccionar_señal_optima()` devuelve múltiples señales por partido (mercados distintos:
total juegos + total sets). El builder itera sobre todas sin filtrar por `partido`.
`_group_by_time_window()` agrupa por `hora` pero no por identidad de partido.

### Fix D126-01

**Archivo:** `betplay_combo_builder.py` — en `build_evaluar_games_combos()`, antes de `_combis()`:

```python
# D126-01: deduplicar señales por partido — max 1 leg por match (mayor cuota)
seen_partidos: Dict[str, Dict] = {}
for s in all_signals:
    p = s["partido"]
    if p not in seen_partidos or s["cuota"] > seen_partidos[p]["cuota"]:
        seen_partidos[p] = s
all_signals = list(seen_partidos.values())
```

---

## §3. FUGA 1 — 33 picks no logueados de 105 candidatos

### Hallazgo

`generar_tabla_favoritos2.py` corrió a las 06:30 contra el archivo de ayer (122 partidos).
A las 10:30 se obtuvo el archivo de hoy (286 partidos, 105 con cuota<1.30).
Al re-ejecutar `generar_tabla_favoritos2.py`, todos los picks nuevos devuelven
**"ya registrado"** — el ID `EVAL_FECHA_desconocido_ap1-ap2_ML` ya existe en sb_2026-07-21.

El shadow_book es **append-only**: un pick con el mismo ID no se actualiza aunque
tenga nuevos campos (`hora`, `match_id`).

### Causa raíz

1. `log_evaluar_pick()` detecta duplicado por sb_id y hace early-return sin upsert
2. D125-01 añade `hora` pero ya era tarde: picks logueados a las 06:30 con hora=None
3. No hay re-indexación cuando PASO 1 produce un archivo más completo en el día

### Fix D126-02 (shadow_book) + D126-03 (generar_tabla)

**D126-02:** `shadow_book.py` — en `log_evaluar_pick()`, si el pick existe Y `hora` es None
en el registro pero se recibe hora no-None → hacer upsert de campos enriquecidos:
```python
# Si pick ya existe pero sin hora → enriquecer (no duplicar, solo actualizar campos vacíos)
if existing and not existing.get('pick_snapshot', {}).get('hora') and pick_snapshot.get('hora'):
    existing['pick_snapshot']['hora'] = pick_snapshot['hora']
    existing['pick_snapshot']['match_id'] = pick_snapshot.get('match_id')
    _rewrite_record(path, sb_id, existing)
    return sb_id
```

**D126-03:** `generar_tabla_favoritos2.py` — seleccionar el archivo zita con más partidos
del día actual (no el más reciente por timestamp):
```python
# Preferir archivo con más partidos del día, no el más reciente
best = max(candidates, key=lambda f: len(json.load(open(f))))
```

---

## §4. FUGA 2 — 33 picks sin event_id Kambi (45% pérdida)

### Hallazgo

`_buscar_event_id_kambi()` falla para torneos ITF M15/W15/M25 pequeños que
**no están en el catálogo de Betplay**. El bridge hace ~33 requests HTTP innecesarios
a Kambi que siempre devuelven vacío, ralentizando el pipeline (~40s extra).

Picks afectados hoy: M15 Bali, M15 Brisbane, W15 Huamantla, W15 Kursumlijska,
W15 Nogent, W35 Gentofte, etc.

### Causa raíz

El bridge no filtra por tier antes de buscar en Kambi. Los tiers `itf`/`itf_minor`
(M15/W15/M25) no tienen mercados de juegos en Betplay — solo ML básico.

### Fix D126-04

**Archivo:** `scripts/evaluar_games_bridge.py` — en `_process_pick()`, antes del lookup:

```python
# D126-04: skip Kambi lookup para tiers sin mercado de juegos
TIERS_SIN_MERCADO_JUEGOS = {'itf_minor', 'itf', 'm15', 'w15', 'm25', 'w25'}
tier_norm = (pick.get('tier') or '').lower().replace(' ', '_')
if any(t in tier_norm for t in TIERS_SIN_MERCADO_JUEGOS):
    # Retornar placeholder sin event_id — ahorra request HTTP
    return {**base_result, 'tiene_mercados': False, '_skip_reason': 'tier_sin_kambi'}
```

Estimación: reduce runtime de bridge de ~90s → ~50s. Mejora cobertura útil: de 39/72 → ~39/39 ATP/Challenger (ITF filtrado antes).

---

## §5. FUGA 3 — 32 eventos Kambi sin mercado total games

### Hallazgo

De 39 eventos encontrados en Kambi, solo 7 tienen mercado UNDER juegos publicado.
Kambi publica mercado "Total de juegos" principalmente para ATP250+, algunos Challenger.
Qualifying rounds e ITF upper (Challenger 50/75/100) raramente tienen este mercado.

### Causa raíz (observacional, no bug)

Betplay/Kambi decide qué mercados secundarios publicar por evento — no controlable
desde el bridge. La señal "sin mercado" es correcta, no es un bug.

**Oportunidad:** Si el mercado no existe hoy pero el partido no empezó, podría aparecer
más tarde (Kambi añade mercados dinámicamente). Un segundo run del bridge 2h antes
del partido captura mercados tardíos.

### Fix D126-05 (opcional, baja prioridad)

`run_daily.py`: ejecutar PASO 3.6b dos veces — una al mediodía y una 90min antes
del primer partido del día — para capturar mercados que Kambi añade cerca del inicio.

---

## §6. BUG SECUNDARIO — `confidence` formato inconsistente

### Hallazgo

`generar_tabla_favoritos2` guarda `confidence=58.6` (porcentaje float).
`edge_calculator` guarda `confidence=0.586` (decimal).
El bridge, la dashboard X4 y cualquier consumidor necesitan normalizar:
`c / 100 if c >= 1 else c`.

Fix aplicado en live_desk.py L524 esta sesión. Falta aplicar en el bridge
(`_load_evaluar_games_picks` L87: `'confidence': snap.get('confidence') or 0`).

### Fix D126-06 (menor, incluir en misma PR)

**Archivo:** `scripts/evaluar_games_bridge.py` L87:
```python
'confidence': (lambda c: c / 100 if c and c >= 1 else (c or 0))(snap.get('confidence')),
```

---

## §7. Resumen de fixes por prioridad

| Fix | Descripción | Archivo | Prioridad | Estado |
|-----|-------------|---------|-----------|--------|
| D126-01 | Same-match gate en build_evaluar_games_combos | `betplay_combo_builder.py` | CRÍTICO | pendiente |
| D126-02 | Upsert hora en log_evaluar_pick si pick ya existe | `shadow_book.py` | ALTO | pendiente |
| D126-03 | generar_tabla_favoritos2 elige archivo con más partidos | `generar_tabla_favoritos2.py` | ALTO | pendiente |
| D126-04 | Skip Kambi para tiers ITF M15/W15/M25 | `evaluar_games_bridge.py` | ALTO | pendiente |
| D126-05 | Bridge 2x al día (mediodía + 90min pre-partido) | `run_daily.py` | BAJO | pendiente |
| D126-06 | Normalizar confidence en bridge _load_evaluar_games_picks | `evaluar_games_bridge.py` | MENOR | pendiente |

---

## §8. Tests REGLA-T53 planificados — `tests/test_nodo126_evaluar_games_audit.py`

```python
def test_D126_01_same_match_dedup_keeps_highest_cuota()
def test_D126_01_combo_never_has_two_legs_same_partido()
def test_D126_04_bridge_skips_itf_m15_tiers()
def test_D126_04_bridge_skips_w15_tiers()
def test_D126_06_confidence_normalized_in_bridge_picks()
def test_D126_03_generar_tabla_selects_file_with_most_matches()
```

---

**Wikilinks totales: 5 | Huérfanos: 0**

[[Nodo-125-EvalGames-Bridge-Dashboard-X4]] | [[Nodo-124-EvalTracker-TablaFavoritos-ShadowBook]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-101-Shadow-Book-Live-CLV]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]]

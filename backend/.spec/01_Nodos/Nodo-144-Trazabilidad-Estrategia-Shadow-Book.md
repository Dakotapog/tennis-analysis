# Nodo-144 — Trazabilidad de Estrategia en Shadow Book

**Fecha:** 2026-07-26
**Estado:** DISEÑO APROBADO — pendiente implementación
**Wikilinks:** [[Nodo-52]] [[Nodo-67]] [[Nodo-101]] [[Nodo-65]] [[Nodo-143]] [[combo_confianza_builder]] [[favoritos_combo_builder]]

---

## 1. Hallazgo E3 — Brecha confirmada

Verificado 2026-07-26 con evidencia dura:

- `grep "strategy.*CORE" shadow_book.py` → 0 resultados
- `grep "combo_type" shadow_book.py` → 0 resultados
- 24 archivos `sb_2026-07-*.jsonl` → 0 registros con tag de estrategia
- `_build_record()` no serializa ningún identificador de origen de estrategia

**Consecuencia real:** los combos CORE, COBERTURA, SATELITE y MOONSHOT generados
por `combo_confianza_builder.py` se registran como picks individuales
indistinguibles del resto del shadow book. Es imposible calcular hit%, ROI
o CLV por estrategia. Semanas de apuestas reales bajo estas estrategias
están perdidas en el agregado general.

**Hallazgo colateral:** `favoritos_combo_builder._registrar_shadow_book()` llama
`sb.log_pick()` (singular) — función que NO EXISTE en `shadow_book.py`. Falla
silenciosamente capturado por `except Exception`. La estrategia
FAVORITOS_COMPUESTOS tampoco se registra.

---

## 2. Arquitectura actual — cómo llegan los picks al shadow book

```
edge_calculator.py L1535
  → shadow_book.log_picks(edge_report, session_meta)
    → _build_record(pick, fecha)         # pick = dict del edge_report
      → {sb_id, logged_at, match_key,
         es_qualifying, season_transition_flag,
         pick_snapshot: {…pick completo…}}  # sin campo strategy
```

Los picks que terminan en CORE/COBERTURA son exactamente los mismos que
el edge_calculator logueó previamente. La estrategia se determina DESPUÉS
en `combo_confianza_builder.py` — el shadow book no lo sabe.

**Patrón de update existente (referencia):** `update_alpha_flags()` (L692)
escribe campo top-level `combo_flags.alpha_promoted` SIN tocar `pick_snapshot`.
Este es el patrón exacto a replicar para strategy.

**combo_registry:** `reports/combo_registry/cr_YYYY-MM-DD.jsonl` tiene
`subtipo=CORE/COBERTURA/etc.` y `piernas` (nombres de jugadores). Disponible
para 3 fechas (22, 23, 25-jul). Permite reconstrucción retroactiva con
alta confianza por cruce nombre+fecha.

---

## 3. Diseño del campo — decisiones explícitas

**Nombre del campo:** `strategy` (top-level, inglés — consistente con `combo_flags`)

**Valores posibles:**

| Valor | Origen | Builder |
|-------|--------|---------|
| `MOTOR` | Pick individual APOSTAR | `trader_ev_tenis.py` |
| `CORE` | Combo CORE | `combo_confianza_builder.py` |
| `SATELITE` | Combo SATELITE | `combo_confianza_builder.py` |
| `MOONSHOT` | Combo MOONSHOT | `combo_confianza_builder.py` |
| `COBERTURA` | Combo COBERTURA | `combo_confianza_builder.py` |
| `ANCHOR` | Combo ANCHOR | `combo_confianza_builder.py --anchor` |
| `GCS` | Combo GCS (hierba) | `combo_confianza_builder.py` |
| `SAFE` | Combo SAFE | `betplay_combo_builder.py --safe` |
| `WAS` | Watchlist Alpha Signal | `betplay_combo_builder.py` |
| `MEGA` | Combo MEGA | `betplay_combo_builder.py --mega` |
| `GAMES` | Over/Under juegos | `betplay_combo_builder.py --games` |
| `RIVAL_VALUE` | Rival Value | `rival_value_betslip.py` |
| `FAVORITOS_COMPUESTOS` | Estrategia #13 | `favoritos_combo_builder.py` |
| `SIN_TAG` | Default — no tageado | cualquier origen sin info |
| `HISTORICO_SIN_TAG` | Retroactivo no reconstruible | script migración |

**Inmutabilidad §1:** el campo `strategy` es top-level — `pick_snapshot` NUNCA
se toca. Consistent con el principio de `update_alpha_flags()`.

**Default defensivo:** `'SIN_TAG'` — nunca una cadena vacía ni None, para que
los segmentos del reporte sean siempre explícitos.

---

## 4. Scope de este Nodo (D144-01 → D144-07)

**Scope incluido:**
- D144-01: Campo `strategy` en `_build_record()` con default `'SIN_TAG'`
- D144-02: Nueva función `tag_strategy(fecha, player_names, strategy)` en `shadow_book.py`
- D144-03: `combo_confianza_builder.py` llama `tag_strategy()` para CORE/COBERTURA/SATELITE/MOONSHOT/ANCHOR/GCS
- D144-04: Implementar `log_pick()` singular en `shadow_book.py` + fix `favoritos_combo_builder._registrar_shadow_book()`
- D144-05: `report()` / `report_dict()` agregan segmento por `strategy`
- D144-06: Script retroactivo `scripts/backfill_strategy.py` — cruce con combo_registry (3 días)
- D144-07: Tests REGLA-T53

**Scope NO incluido (deuda futura):**
- Tagging MOTOR desde `trader_ev_tenis.py` — requiere refactor del flujo trader → shadow_book
- Tagging SAFE/WAS/MEGA/GAMES/RIVAL_VALUE desde sus builders respectivos
- Estos quedan con `strategy='SIN_TAG'` hasta que cada builder implemente su `tag_strategy()` call

---

## 5. Plan de migración retroactiva

Para los 24 JSONL existentes:

| Grupo | Días | Confianza | Método |
|-------|------|-----------|--------|
| Con combo_registry (`cr_*.jsonl`) | 3 días (22, 23, 25-jul) | ALTA — cruce exacto nombre+fecha+subtipo | backfill_strategy.py |
| Sin combo_registry | 21 días | BAJA — no reconstruible | marcar `HISTORICO_SIN_TAG` |

**Regla dura:** si no hay match exacto por nombre+fecha en combo_registry,
NO asignar estrategia adivinada. Dejar `HISTORICO_SIN_TAG` explícito.
Jamás contaminar métricas con datos inferidos sin evidencia.

El script `backfill_strategy.py` reportará:
- N registros tageados con alta confianza
- N registros marcados `HISTORICO_SIN_TAG`
- N registros que ya tenían `strategy` (skip)

---

## 6. Implementación — D144-01 a D144-07

### D144-01: `_build_record()` — campo top-level

```python
# Añadir en el return de _build_record() (L186-193):
return {
    "sb_id":                  sb_id,
    "logged_at":              datetime.now().astimezone().isoformat(),
    "match_key":              mk,
    "strategy":               pick.get('strategy', 'SIN_TAG'),  # D144-01
    "es_qualifying":          _es_qualifying(torneo),
    "season_transition_flag": _season_transition(fecha, superficie),
    "pick_snapshot":          pick,
}
```

### D144-02: `tag_strategy()` — función de update (patrón update_alpha_flags)

```python
def tag_strategy(fecha: str, player_names: list, strategy: str) -> int:
    """
    D144-02 (Nodo-144): propaga tag de estrategia al shadow book.
    Escribe campo top-level 'strategy'. NO toca pick_snapshot (inmutabilidad §1).
    Solo actualiza si el registro tiene strategy='SIN_TAG' (no sobrescribe tags ya puestos).
    """
    if not player_names or not strategy:
        return 0
    path = _jsonl_path(fecha)
    records = _load_jsonl(path)
    if not records:
        return 0

    nombres_set = {n.strip().lower() for n in player_names}
    marcados = 0
    for sb_id, rec in records.items():
        if rec.get('_type') == 'session_meta':
            continue
        if rec.get('strategy', 'SIN_TAG') != 'SIN_TAG':
            continue  # no sobrescribir tag ya asignado
        snap = rec.get('pick_snapshot', {})
        nombre = (snap.get('favorito_predicho') or snap.get('nombre') or
                  snap.get('jugador') or snap.get('player') or '').strip().lower()
        if nombre and nombre in nombres_set:
            rec['strategy'] = strategy
            rec['strategy_tagged_at'] = datetime.now().isoformat()
            marcados += 1

    if marcados > 0:
        _save_jsonl(path, records)
        logger.info(f"[ShadowBook] {marcados} picks tageados strategy={strategy}")

    return marcados
```

### D144-03: `combo_confianza_builder.py` — call a tag_strategy

Después de `_save_jsonl` o al final de `main()`, añadir:

```python
# D144-03 (Nodo-144): propagar estrategia al shadow book
try:
    from shadow_book import tag_strategy as _tag_strategy
    _fecha = datetime.now().strftime('%Y-%m-%d')
    if plan.get('core'):
        _tag_strategy(_fecha, [p['nombre'] for p in plan['core']['picks']], 'CORE')
    if plan.get('cobertura'):
        for cob in plan['cobertura']:
            _tag_strategy(_fecha, [p['nombre'] for p in cob['picks']], 'COBERTURA')
    # SATELITE, MOONSHOT, ANCHOR, GCS — mismo patrón
except Exception as e:
    logger.warning(f"[shadow_book] tag_strategy falló: {e}")
```

### D144-04: `log_pick()` singular + fix `favoritos_combo_builder`

Implementar `log_pick()` en shadow_book.py para que el call existente en
`favoritos_combo_builder._registrar_shadow_book()` deje de fallar:

```python
def log_pick(fecha: str, jugador: str, cuota: float,
             pick_snapshot: dict) -> Optional[str]:
    """
    D144-04 (Nodo-144): registra un pick individual con snapshot completo.
    Usado por favoritos_combo_builder._registrar_shadow_book().
    """
    # Delegar a _build_record() con el snapshot como base del pick
    pick = {**pick_snapshot, 'jugador': jugador, 'cuota_favorito': cuota}
    rec = _build_record(pick, fecha)
    if rec is None:
        return None
    path = _jsonl_path(fecha)
    existing = _load_jsonl(path)
    if rec['sb_id'] not in existing:
        existing[rec['sb_id']] = rec
        _save_jsonl(path, existing)
    return rec['sb_id']
```

### D144-05: `report()` — segmento por estrategia

En `_segment_all_records()` o equivalente, añadir after los segmentos tier/status:

```python
# D144-05: segmentos por estrategia
strategy_vals = sorted({r.get('strategy', 'SIN_TAG') for r in settled_records})
for strat in strategy_vals:
    group = [r for r in settled_records if r.get('strategy', 'SIN_TAG') == strat]
    if group:
        m = _segment_metrics(group)
        lineas.append(f"  strategy={strat:<22} {_format_segment(m)}")
```

---

## 7. Tests — REGLA-T53

**Archivo:** `tests/test_nodo144_strategy_tag.py`

- `test_build_record_default_sin_tag` — pick sin strategy → top-level `strategy='SIN_TAG'`
- `test_build_record_with_strategy` — pick con `strategy='CORE'` → top-level preservado
- `test_tag_strategy_updates_record` — tag_strategy() actualiza registro existente con SIN_TAG
- `test_tag_strategy_no_overwrite` — tag_strategy() NO sobrescribe si strategy ya != SIN_TAG
- `test_tag_strategy_skip_session_meta` — session_meta records no se tocan
- `test_log_pick_creates_record` — log_pick() crea registro en JSONL
- `test_log_pick_idempotent` — log_pick() dos veces = upsert, sin duplicado

**Regresión:** `python -m pytest tests/ -k shadow_book -v` — 0 regresiones.

---

## 8. Wikilinks — análisis de huérfanos

| Nodo | Estado | Relación |
|------|--------|----------|
| [[Nodo-52]] | IMPLEMENTADO | Shadow Book base — `_build_record()` / `settle()` / `report()` |
| [[Nodo-67]] | IMPLEMENTADO | DataContract v2 — candidato para agregar `strategy` como campo obligatorio |
| [[Nodo-101]] | IMPLEMENTADO | Shadow Book Live CLV — `log_live_pick()` necesita D144-04 análogo |
| [[Nodo-65]] | IMPLEMENTADO | ANCHOR/VARIABLE segmentación — primer intento de segmentar por origen |
| [[Nodo-143]] | IMPLEMENTADO | Torneo metadata — patrón de fix aditivo sin romper existente |
| [[combo_confianza_builder]] | Archivo activo | Caller principal D144-03 |
| [[favoritos_combo_builder]] | Archivo activo | D144-04 fix log_pick() |

---

## 9. Deuda post-Nodo-144

**D144-08:** Extender tagging a MOTOR (trader_ev_tenis.py), SAFE/WAS/MEGA/GAMES
(betplay_combo_builder.py), RIVAL_VALUE (rival_value_betslip.py). Cada uno
sigue el mismo patrón `tag_strategy()`.

**D144-09:** `log_live_pick()` (D97-13, Nodo-101) necesita análogo de D144-04
para picks live con strategy='LIVE_GAMES' u otro valor.

**D144-10:** DataContract v2 ([[Nodo-67]]): agregar `strategy` como campo
obligatorio en la frontera `shadow_book_record`. Gate: 5 días de estabilidad
con D144-01 activo (strategy presente en ≥95% de registros nuevos).

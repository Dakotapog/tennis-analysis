# Nodo-143 — Match Ledger: Propagación de Metadata Torneo en Joins

**Fecha:** 2026-07-25
**Estado:** IMPLEMENTADO — evidencia producción verificada 2026-07-25
**Wikilinks:** [[Nodo-118]] [[Nodo-136]] [[Nodo-141]] [[Nodo-142]] [[Nodo-67]] [[combo_confianza_builder]]

---

## 1. Problema

`fusionar_dia()` en `scraping/match_ledger.py` construye el registro JOIN usando FlashScore
como base (`{**mejor_pf, ...}`) y solo copia cuotas + IDs de Kambi. Los campos de metadata
de torneo que Kambi sí tiene se descartan silenciosamente.

**Evidencia real 2026-07-25:**
- 40 joins: 0/40 con torneo_nombre → todos los picks del edge_report tienen torneo="Desconocido"
- 8 single_source_kambi: 8/8 con torneo_nombre → los puros Kambi sí lo conservan
- Campos perdidos en join: `tier | torneo_nombre | torneo_completo | pais | ranking1 | ranking2 | tournament_context`

**Cadena de daño (trazada con ejecución real 2026-07-25):**
```
fusionar_dia() → join sin torneo
  → zita_tennis_matches_*_merged.json: torneo_nombre="" en 40/40 joins
  → extraer_historh2h.py lee merged → h2h_results_enhanced: torneo_nombre="" (0/109)
  → edge_calculator L933-934: detectar_tier(torneo_completo="") → "Desconocido"
  → Nodo-136 CTI fallback: circuit_asymmetry → todos tier="atp500" (incorrecto)
  → Kelly-KL lambda incorrecto (atp500=2.4× en lugar de itf=4.5×/challenger=3.6×)
  → picks no pasan umbral APOSTAR → combo_confianza_builder: 0 combos
  → combo_gate_log: torneo="?" para todos (ver §nota-combo-builder)
```

**Nota — combo_confianza_builder torneo='?':**
`combo_confianza_builder.py` L658 usa `partido.get('torneo_completo') or partido.get('torneo') or '?'`
pero NO lee `torneo_nombre`. Cuando h2h tiene `torneo_nombre=""` y no tiene clave `torneo`,
el gate log muestra `?` en vez del nombre real. D143-04 catalogado (§10).

**Diagnóstico comparativo (2026-07-25 — run mañana, 40 partidos Kambi):**

| Fuente | Con torneo | Sin torneo |
|--------|-----------|-----------|
| single_source_kambi (8) | 8/8 | 0/8 |
| joins AUTO_JOIN (40) | 0/40 | 40/40 ← bug |

**Diagnóstico comparativo (2026-07-25 — run tarde, 24 partidos Kambi restantes):**

| Fuente | Con torneo | Sin torneo |
|--------|-----------|-----------|
| joins AUTO_JOIN (17) | 17/17 ← fix | 0/17 |
| single_source_fs (99) | 0/99 (sin Kambi) | 99/99 |

## 2. Root cause en código

`scraping/match_ledger.py` L346-360, función `fusionar_dia()`, bloque AUTO-JOIN:

```python
# ANTES (bug): solo cuotas e IDs desde Kambi, todo lo demás de FlashScore (sin torneo)
partido_merged = {
    **mejor_pf,           # FlashScore base — sin campos tier/torneo_nombre/etc.
    "cuota1": pk.get("cuota1"),
    "cuota2": pk.get("cuota2"),
    "outcome_id": ...,
    "kambi_event_id": ...,
    ...
}
```

Implementado en `071b72f` (Nodo-118 F1, 2026-07-18). Bug vivo sin corrección en HEAD
hasta este nodo (confirmado con `graphify query` + `git log -- '*match_ledger*'`).

## 3. Fix — D143-01

**Archivo:** `scraping/match_ledger.py`
**Función:** `fusionar_dia()`, bloque AUTO-JOIN, tras construir `partido_merged`
**Líneas:** ~360-366

```python
# D143-01 (Nodo-143): Propagar metadata torneo desde Kambi al join.
# Solo llena huecos (no sobrescribe) — Kambi gana para tier/torneo,
# FlashScore gana para match_id/H2H URLs.
_KAMBI_META_FIELDS = ['tier', 'torneo_nombre', 'torneo_completo',
                       'pais', 'ranking1', 'ranking2', 'tournament_context']
for _campo in _KAMBI_META_FIELDS:
    if _campo in pk and pk[_campo] and not partido_merged.get(_campo):
        partido_merged[_campo] = pk[_campo]
```

**Guard `not partido_merged.get(campo)` (fill-gaps, no overwrite):**
- FlashScore puede proveer ranking1/ranking2 en el futuro → guard defensivo
- Para tier/torneo_nombre: FlashScore no los tiene → equivalente a overwrite, pero más seguro
- Para ranking1/ranking2: si FS tiene dato propio, prevalece sobre Kambi

## 4. Qué NO cambia

- Algoritmo Fellegi-Sunter de scoring: sin cambios
- Umbral MIN_SCORE_JOIN / MIN_SCORE_QUARANTINE: sin cambios
- Campos FlashScore en el join (match_url, h2h_url, match_id): sin cambios
- single_source_kambi y single_source_fs: sin cambios

## 5. Scope del CTI fallback de Nodo-136

[[Nodo-136]] fue diseñado para H2H combinado donde el JSON de historial mezcla torneos
distintos y no propaga torneo_nombre. Ese uso es correcto y se mantiene.

Este Nodo-143 ataca el problema upstream: cuando match_ledger destruye la metadata ANTES
de que llegue al H2H. Con D143-01, el CTI fallback de Nodo-136 queda como backup real
para su caso de uso original (H2H multi-torneo), no como parche general.

## 6. Wikilinks — análisis de huérfanos

| Nodo referenciado | Estado | Relación |
|---|---|---|
| [[Nodo-118]] | IMPLEMENTADO | Introduce fusionar_dia() — origen del bug |
| [[Nodo-136]] | IMPLEMENTADO | CTI fallback cuyo scope se corrige con este nodo |
| [[Nodo-141]] | IMPLEMENTADO | Nodo upstream que expuso el problema (0 combos) |
| [[Nodo-142]] | HUÉRFANO — sin spec | ITF Live Games Convergencia (live_desk.py) — test existe sin spec |
| [[Nodo-67]] | IMPLEMENTADO | DataContract v2 — candidato para D143-02 (agregar torneo al contrato) |

**Nodo-142 huérfano:** `tests/test_nodo142_itf_live_games.py` existe sin spec correspondiente.
Deuda SDD: crear `.spec/01_Nodos/Nodo-142-ITF-Live-Games-Convergencia.md` en sesión futura.

## 7. Impacto — evidencia real 2026-07-25

**Secuencia correcta tras aplicar el fix:**
```
match_ledger --build --api <archivo_kambi>   # joined: 17/17 con torneo
extraer_historh2h.py --api-mode              # h2h_results_enhanced: 17/109 con torneo_nombre
edge_calculator.py                           # picks: tier correcto para joins
filter_kambi_picks.py                        # 6 kambi_disponible
combo_confianza_builder.py --bankroll 125000 # 5 combos · $15,000
```

IMPORTANTE: el fix requiere **re-correr PASO 2** (extraer_historh2h) después del ledger.
El h2h_results_enhanced lleva torneo_nombre del merged file — si es viejo, el bug persiste.

| Situación | Antes D143-01 | Después D143-01 |
|-----------|--------------|-----------------|
| 17 joins tarde | torneo=None 0/17 | torneo_nombre real 17/17 |
| detectar_tier() en edge_calc | "Desconocido" 100% | tier correcto (challenger/wta_qual/atp) |
| lambda_efectivo | atp500=2.4× uniforme | ITF=4.5× / Challenger=3.6× / ATP500=2.4× |
| pick Suresh D. | tier=atp500, torneo=Desconocido | tier=challenger, torneo=Bloomfield Hills |
| combos generados | 0 ("picks no disponibles en Kambi") | **5 combos · $15k · CORE @16.44x** |

**Combos producción 2026-07-25 (tarde):**
- CORE 4p: Raina A. @1.92 + Honda N. @1.97 + Voloshchuk A. @2.12 + Teixido Garcia A. @2.05 → @16.44x $7,000
- COBERTURA [COB1_excl_A.]: @18.66x $2,000
- Total: 5 combos · $15,000 desplegados

## 8. Conceptos que deben dominarse antes de modificar match_ledger

1. **Lineage de datos:** torneo fluye scraper → ledger → h2h → edge_calc → tier → kelly.
   Romper cualquier paso rompe la cadena silenciosamente.
2. **Schema dominance en merges:** regla explícita — Kambi gana metadata de torneo/tier/ranking;
   FlashScore gana match_id/H2H URLs. Sin esta regla, el merge colapsa a la fuente más pobre.
3. **DataContract v2 ([[Nodo-67]]):** torneo_nombre debería ser campo obligatorio en el contrato
   del artefacto `match_ledger_join`. Deuda: D143-02.
4. **Scope de CTI fallback ([[Nodo-136]]):** es para H2H combinado, no parche general de torneo.

## 9. Tests — REGLA-T53 (`tests/test_nodo143_match_ledger_torneo.py`)

- `test_join_preserves_torneo_nombre` — FS sin torneo + Kambi con torneo → join tiene torneo correcto
- `test_join_preserves_tier` — tier propagado correctamente
- `test_join_preserves_all_meta_fields` — los 7 campos propagados en un join
- `test_no_overwrite_existing_field` — si FS tiene ranking1, Kambi no lo sobrescribe
- `test_handles_none_kambi_meta_field` — campo Kambi None no se copia ni genera KeyError

**Resultado:** 5/5 PASS · Regresión Nodo-118: 23/23 PASS

## 10. Deuda técnica catalogada

**D143-02:** Agregar `torneo_nombre` como campo obligatorio en `DataContract.validate_artifact()`
([[Nodo-67]]) para el artefacto `match_ledger_join`.
Gate: estabilidad 5 días consecutivos con D143-01 activo (torneo presente en ≥95% de joins).

**D143-03 (Nodo-142 huérfano):** Crear spec formal para Nodo-142 ITF Live Games Convergencia
antes de que el test huérfano acumule más deuda SDD.

**D143-04 (gap combo_confianza_builder L658):** Gate log G_EV usa
`partido.get('torneo_completo') or partido.get('torneo') or '?'` — omite `torneo_nombre`.
Cuando h2h tiene `torneo_nombre` pero no `torneo`, el log muestra `?` en vez del torneo real.
Fix: añadir `or partido.get('torneo_nombre')` al fallback en L658.
Impacto: cosmético (solo el gate log — el pick sí tiene torneo correcto en L585-590 que sí lee `torneo_nombre`).

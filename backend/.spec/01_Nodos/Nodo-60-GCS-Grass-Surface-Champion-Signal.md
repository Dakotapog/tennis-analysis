# Nodo-60 — GCS: Grass/Surface Champion Signal

**Estado:** ✅ COMPLETO  
**Fecha spec:** 2026-07-05  
**Fecha cierre:** 2026-07-05  
**Prioridad:** ALTA — alpha estructural observado en Wimbledon 2026

---

## Contexto / Motivación

Patrón observado el 2026-07-04 en Wimbledon Round 3:  
**3/3 jugadores con TORNEO_COMPLETO_BONUS en hierba (≤21d, tier≥ATP500) ganaron** siendo underdogs o monedas.

| Jugador | Bonus torneo | Cuota | Resultado |
|---|---|---|---|
| Alexandra Eala | Birmingham 2026 (×1.4) + Scalp #2 Rybakina | @3.80 | GANO ✅ |
| Marie Bouzkova | Nottingham 2026 (×1.6) + HOT 80% | @1.66 | GANO ✅ |
| Ashlyn Krueger | Ilkley 2026 (×1.4) + HOT 100% | @1.19 | GANO ✅ |

El sistema tenía TODOS los datos. Tres problemas:
1. El TORNEO_COMPLETO_BONUS solo multiplica `quality_score` interno, pero queda diluido por el historial histórico al normalizarse. Bouzkova tenía surface=27.6 vs Samsonova=59.5 → predijo Samsonova (INCORRECTO).
2. El combo builder mezcló picks GS (conf=50%) con picks ITF (conf=52%) en el mismo combo. 5 picks fallaron simultáneamente → 18/18 combos muertos.
3. No había hipótesis pre-registrada → no acumulamos data del patrón.

---

## Deliverables

| ID | Descripción | Archivo |
|---|---|---|
| D60-01 | H60-01 pre-registrada en preregistered_hypotheses.json | validation/preregistered_hypotheses.json |
| D60-02 | GCS_RECENCY_BOOST en analyze_surface_specialization | analysis/rivalry_analyzer.py |
| D60-03 | `gcs_active` + `universo` en _extract_and_categorize | combo_confianza_builder.py |
| D60-04 | Sub-plan GCS separado en output del combo builder | combo_confianza_builder.py |
| D60-05 | Tests T60-01→T60-05 | tests/test_nodo60.py |

---

## F0 — Pre-registro H60-01

Hipótesis: **TORNEO_COMPLETO_BONUS (tier≥ATP500, ≤21d, misma superficie) + edge_vs_mercado≥10% → hit% > 1/cuota_media**

Umbrales congelados:
- tier_min: atp500
- dias_max: 21
- edge_min: 0.10 (opcional, no gate duro)
- n_stop: 30

Dato inicial: n=8 settled (scan histórico), hits=3 (37.5%). Insuficiente para validar. Observar.

---

## F1 — Fix Bouzkova: GCS_RECENCY_BOOST en rivalry_analyzer.py

**Causa raíz:** El TORNEO_COMPLETO_BONUS multiplica `quality_score` ANTES de la normalización por partidos y el skill_factor. Una jugadora con historial histórico menor (Bouzkova) queda por debajo aunque ganara el torneo más reciente.

**Fix:** Después de calcular `final_score`, si el bonus fue activado con tier≥ATP500 y days≤21, aplicar un multiplicador adicional al `final_score`:

```
GCS_MULT_RECENT = 2.2   # ≤7 días
GCS_MULT_MID    = 1.8   # 8-14 días  
GCS_MULT_BASE   = 1.5   # 15-21 días
```

Retornar `gcs_active: True` y `gcs_days: N` en el dict de resultado.

**Regla D60-02:** El GCS_RECENCY_BOOST solo aplica cuando la superficie del análisis COINCIDE con la superficie actual del partido (`normalized_surface == current_match_surface_normalized`). El sistema ya llama con la superficie correcta — no hay que cambiar la interfaz.

**Guard:** Solo para tier∈{grand_slam, atp1000, atp500}. ITF/Challenger NO reciben GCS boost.

---

## F2 — Fix combos: GCS sub-plan separado en combo_confianza_builder.py

**Causa raíz:** Los picks de Grand Slam (conf=50%, alta cuota) se mezclan con picks ITF (conf=52%, cuota media) en los mismos combos. Cuando GS fallan, arrastran todo.

**Fix en `_extract_and_categorize`:**
- Leer `surface_specialization_meta.player1/2.torneo_completo` del partido
- Si True para el `favored_player` + `detectar_tier(torneo_nombre)` in {grand_slam, atp1000, atp500} → `gcs_active=True`
- Añadir `universo: 'GCS'` si gcs_active, `'GS'` si tier≥atp500 sin bonus, `'ITF'` para el resto

**Fix en el portfolio builder:**
- Picks `universo='GCS'` se reportan en sección separada: "GCS PICKS — Campeones pre-torneo"
- MAX_GCS_PER_COMBO = 1 en combos estándar (CORE/Satellite)  
- Si hay ≥2 picks GCS: construir un combo GCS puro de 2-3 piernas con stake fijo pequeño (2% budget)
- El output debe mostrar claramente qué picks son GCS para que el usuario los trate diferente

---

## Tests

| ID | Test | Módulo |
|---|---|---|
| T60-01 | GCS_RECENCY_BOOST se aplica cuando tier≥ATP500 y days≤14 | rivalry_analyzer |
| T60-02 | GCS_RECENCY_BOOST NO se aplica cuando tier=ITF | rivalry_analyzer |
| T60-03 | GCS_RECENCY_BOOST NO se aplica cuando days>21 | rivalry_analyzer |
| T60-04 | _extract_and_categorize marca gcs_active=True cuando torneo_completo=True + tier≥atp500 | combo_confianza_builder |
| T60-05 | H60-01 existe en preregistered_hypotheses.json con n_stop=30 | validation |

---

## Constantes nuevas (no modificar sin n≥30)

```python
GCS_MULT_RECENT = 2.2   # days ≤ 7
GCS_MULT_MID    = 1.8   # days 8-14
GCS_MULT_BASE   = 1.5   # days 15-21
GCS_TIER_MIN    = {'grand_slam', 'atp1000', 'atp500'}
GCS_DAYS_MAX    = 21
```

---

## PROHIBIDO en Nodo-60

- Cambiar umbrales GCS_MULT_* sin n≥30 observaciones
- Aplicar GCS boost a tier=ITF o tier=Challenger
- Combinar picks GCS con picks ITF en el mismo CORE combo
- Modificar el comportamiento del TORNEO_COMPLETO_BONUS existente (solo añadir, no reemplazar)

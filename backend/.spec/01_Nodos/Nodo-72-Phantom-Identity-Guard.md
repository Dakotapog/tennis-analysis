# Nodo-72 — Phantom Identity Guard

**Fecha:** 2026-07-08  
**Estado:** COMPLETO  
**Rama:** nodo-51-f3  
**Tests:** T72-01 → T72-12 (12 tests, todos pasan)

---

## Problema

La búsqueda H2H por nombre de string (modo API Ninja) puede colisionar con jugadores homónimos,
entregando un historial completamente equivocado. Dos casos reales:

### Caso Morris (2026-07-08) — Circuit/Gender Mismatch
- **Jugadora:** Ariana Morris — WTA W15 ITF
- **Historial recibido:** ATP male player "Morris" — Rotterdam 2026 ATP500 champion, 25+ Top-100 scalps (Zverev, Fritz, FAA)
- **Impacto:** Entró en CC2/CC5 con confianza inflada. Pérdida real.
- **Señal detectora:** `player_info.tour='wta'` pero >50% oponentes en ranking ATP

### Caso Pereyra (2026-07-06) — Homonym Time Gap
- **Jugador:** Facundo Pereyra — debutante ITF, sin ranking
- **Historial recibido:** Veterano homónimo — 105 partidos desde 2018
- **Impacto:** 64.4% confianza falsa → 5 combos inválidos → pérdida real.
- **Señal detectora:** `ranking=None` + `n_history>20` + `oldest>365d`

---

## Solución

### D72-A: `_detect_phantom_identity()` en `analysis/rivalry_analyzer.py`

Método privado del `RivalryAnalyzer`. Evalúa dos casos:

**Caso 1 — CIRCUIT_MISMATCH:**
- Señal A: Verifica tour (ATP/WTA) de hasta 8 oponentes en ranking DB. Si >50% son del circuito opuesto → PHANTOM.
- Señal B (fallback): Verifica prefijos de torneo. `M15 `/`M25 ` = ATP men's. `W15 `/`W25 ` = WTA. Si >50% son del circuito opuesto → PHANTOM.
- Umbral mínimo: `checked >= 3` (Señal A) o `total_prefix >= 3` (Señal B)

**Caso 2 — HOMONYM_GAP:**
- Gate: `ranking=None AND n_history>20 AND oldest_match>365d`
- Cubre el caso Pereyra: debutante sin ranking con historial de veterano homónimo.

Retorna: `{phantom: bool, type: str|None, confidence: float, reason: str}`

### D72-B: Integración en `analyze_rivalry()`

- Llamada inmediatamente después de obtener `player_info` para ambos jugadores
- `LOG_PHANTOM_IDENTITY` emitido en WARNING cuando se detecta
- `phantom_identity_p1` y `phantom_identity_p2` incluidos en AMBOS return dicts (early return + main return)

### D72-C: Gate en `edge_calculator.py`

Después del bloque `HISTORIAL_NO_EXTRAIDO` (Nodo-35):
- Lee `partido['ranking_analysis']['phantom_identity_p1/p2']`
- Si `phantom=True` → `apostar=False`, `phantom_data=True`, `status=PICK_STATUS_NO_DATA`
- `motivo_reclasificacion`: `PHANTOM_IDENTITY [TYPE]: historial contaminado de PLAYER`
- `phantom_data=False` como default para todos los picks (campo siempre presente)
- El `-25` en alpha_score (Nodo-62 Signal Bridge) aplica automáticamente vía `phantom_data=True`

---

## Validación en Producción

### Morris (2026-07-08) — HABRÍA SIDO CAPTURADA
- **Detección:** Señal A: 8/8 oponentes ATP → ratio=100% → confidence=0.95 CIRCUIT_MISMATCH
- **Bloques:** CC2 y CC5 → ambos excluidos de trade (phantom_data=True → status=NO_DATA)
- **Impacto:** Previene pérdida real en ambas combos

### Pereyra (2026-07-06) — HABRÍA SIDO CAPTURADA
- **Detección:** HOMONYM_GAP: ranking=None, n=105>20, oldest=2018 (2000+ días) → confidence=0.85
- **Bloques:** 5 combos con Pereyra → todos excluidos (status=NO_DATA)
- **Impacto:** Previene 64.4% confianza falsa

---

## Archivos modificados

| Archivo | Cambio |
|---|---|
| `analysis/rivalry_analyzer.py` | `_detect_phantom_identity()` nuevo método + llamada en `analyze_rivalry()` + 2 return dicts |
| `edge_calculator.py` | `resultado['phantom_data']=False` default + gate Nodo-72 post HISTORIAL_NO_EXTRAIDO |
| `tests/test_nodo72.py` | 12 tests nuevos |

---

## Tests

| Test | Caso | Resultado esperado |
|---|---|---|
| T72-01 | WTA + >50% ATP oponentes | CIRCUIT_MISMATCH |
| T72-02 | ATP + >50% WTA oponentes | CIRCUIT_MISMATCH |
| T72-03 | WTA + WTA oponentes limpios | NOT phantom |
| T72-04 | ranking=None, n=25, oldest=400d | HOMONYM_GAP |
| T72-05 | ranking=None, n=15 (≤20) | NOT phantom |
| T72-06 | ranking=None, n=25, oldest=300d | NOT phantom |
| T72-07 | WTA + torneos "M15 Lodz" | CIRCUIT_MISMATCH (prefijos) |
| T72-08 | Historial vacío | NOT phantom |
| T72-09 | CIRCUIT_MISMATCH 100% ratio | confidence > 0.6 |
| T72-10 | player_info=None | No crash, dict válido |
| T72-11 | Importación standalone | `hasattr(RivalryAnalyzer, '_detect_phantom_identity')` |
| T72-12 | Boundary n=20 vs n=21 | n=20 → False; n=21 → HOMONYM_GAP |

---

## Trabajo futuro

| Item | Descripción | Prioridad |
|---|---|---|
| Shadow Book Phantom Tracking | Agregar sección `phantom_log` en shadow_book.py para acumular historial de picks phantom detectados (para calibración estadística) | MEDIA |
| Phantom Identity Moment 3 | Post-settlement: verificar si un phantom que fue bloqueado hubiera ganado (retroactivo). Refina confianza de los gates. | BAJA |
| Recalibración de umbrales | Después de n≥50 phantoms detectados, re-evaluar thresholds: `checked≥3`, `ratio>0.5`, `confidence_min=0.60` | BAJA |
| Playwright F3 Integration | Cuando H2H tiene `phantom_data=True`, auto-flag en THF para re-scraping Playwright en próxima sesión | MEDIA |

---

## Limitaciones conocidas

- **Morris ATP500**: Señal A requiere que los oponentes (Zverev, Fritz) estén en el ranking DB local. Si el ranking tiene >7 días de antigüedad o no incluye todos los top players, puede no activarse. Señal B es el fallback para ITF futures (M15/M25).
- **Playwright es la solución real**: Este guard es defensa en profundidad. La solución permanente es usar Playwright (IDs de entidad FlashScore → imposible confundir). El guard captura casos donde se usó modo API.
- **Circuito sin ranking**: Si `player_info.tour` es vacío (jugador no en DB), el Caso 1 no activa. Solo el Caso 2 (HOMONYM_GAP) aplica independiente del tour.

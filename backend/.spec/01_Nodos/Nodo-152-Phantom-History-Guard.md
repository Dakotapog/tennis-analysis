# Nodo-152 — Phantom History Guard: Circuit Consistency Validation

**Fecha:** 2026-07-29  
**Trigger:** Vesantera B.T. (ITF M15) recibió historial de top-10 ATP (ATP Finals, Wimbledon QF, Six Kings Slam). El mismo historial contaminó también a Hohmann T. en el mismo día. Pérdida real: AC1–AC12 apostados con edge=39.5% completamente inventado.

---

## Root Cause — Por Qué Sigue Pasando

Los fixes anteriores (Nodo-72 `_detect_phantom_identity()`) operaban **downstream** en `edge_calculator.py`, después de que los datos contaminados ya habían calculado ELO, shrinkage, Kelly y triple alignment. Detectar el síntoma allí es tarde e incompleto.

La causa raíz es **upstream**, en `scraping/ninja_h2h_parser.py`:

```
ninja_api no encuentra "Vesantera B.T." (jugador ITF oscuro)
    → fallback a thf_cache (_lookup_player_history_temporal)
    → thf_cache resuelve por apellido parcial sin validar circuito
    → retorna historial de top-ATP (ATP Finals, Wimbledon, Six Kings)
    → history_source = PROVENANCE_THF_CACHE → entra al pipeline limpio
    → phantom_data = False (hardcodeado en edge_calculator L1122)
    → ELO calculado sobre 151 partidos de top-ATP: ELO=1974
    → confidence_flag=STRONG, triple_alignment=1.0, STRUCTURAL_ALPHA
    → edge=39.5%, pick=APOSTAR, aparece como ANCHOR en 12 combos
    → usuario apuesta AC1–AC12. Pérdida real.
```

**Hallazgo adicional:** El mismo thf_cache asignó EL MISMO historial top-ATP a dos jugadores distintos (Vesantera y Hohmann T.) en el mismo día, confirmando que el matching es por fragmento de apellido sin verificación de identidad.

---

## Fixes — D152-01 a D152-05

### D152-01: `_validate_circuit_consistency()` en `ninja_h2h_parser.py`

Función pura a nivel módulo (no método de clase), insertada en L874 (después de `_persist_playwright_to_thf_cache`).

**Algoritmo — 5 reglas acumulativas:**

| Regla | Condición | Score |
|---|---|---|
| R1 | Historial contiene torneos elite (ATP Finals, Laver Cup, Six Kings Slam) + jugador en ITF/Challenger | +100 |
| R2 | ≥2 GS con rivales top-10 + jugador en ITF/Challenger | +60 por cada uno (cap +120) |
| R3 | Mediana ranking rival < 150 siendo jugador ITF actual | +50 |
| R4 | thf_cache como fuente Y score>0 | score × 1.5 |
| CONTAMINADO | score ≥ 50 | → historial = [] + flag |

Output: `{'contaminated': bool, 'score': int, 'evidence': list[str]}`

### D152-02: Hard block en `_process_match()` — ruta THF (L1186-1190)

Cuando `_lookup_player_history_temporal()` retorna historia vía thf_cache, llamar `_validate_circuit_consistency()` inmediatamente. Si contaminado:
- Limpiar historia: `p1_history = []` / `p2_history = []`
- Marcar en match_data: `match_data['_contamination_p1'] = True`
- Log visible: `[D152] PHANTOM_HISTORY bloqueado: {player} score={score}`
- Continuar con historial vacío (es mejor sin datos que con datos falsos)

### D152-03: Propagación `contamination` a `data_quality` en `_consolidate_result()`

Añadir a la sección `data_quality` del resultado consolidado (L1665):

```python
'history_contamination': {
    'p1_contaminated': match_data.get('_contamination_p1', False),
    'p2_contaminated': match_data.get('_contamination_p2', False),
    'p1_score':        match_data.get('_contamination_score_p1', 0),
    'p2_score':        match_data.get('_contamination_score_p2', 0),
},
```

### D152-04: Gate en `edge_calculator.py` — leer `history_contamination`

Añadir después del bloque Nodo-72 (L1253) en `calcular_edge_completo()`:

```python
# D152-04 (Nodo-152): Phantom History gate — detecta contaminación thf_cache
_dq = partido.get('data_quality', {})
_hc = _dq.get('history_contamination', {})
if _hc.get('p1_contaminated') or _hc.get('p2_contaminated'):
    _score = max(_hc.get('p1_score', 0), _hc.get('p2_score', 0))
    _who = partido.get('jugador1') if _hc.get('p1_contaminated') else partido.get('jugador2')
    resultado['apostar'] = False
    resultado['phantom_data'] = True
    resultado['status'] = PICK_STATUS_NO_DATA
    resultado['motivo_reclasificacion'] = (
        f'PHANTOM_HISTORY [D152]: historial contaminado (score={_score}) '
        f'de {_who} — thf_cache asignó historial de circuito incorrecto'
    )
```

### D152-05: ELO-ranking coherence gate en `edge_calculator.py`

Añadir después de D152-04 como segunda línea de defensa (captura casos que lleguen sin data_quality):

```python
# D152-05 (Nodo-152): ELO-ranking incoherence — segunda línea de defensa
_elo_fav = resultado.get('elo_favorito') or 0
_rk_fav  = resultado.get('ranking_favorito')
if _elo_fav > 1800 and (not _rk_fav or int(_rk_fav) > 400) and tier in ('itf', 'challenger'):
    logger.warning(
        f"[D152-05] ELO_RANK_INCOHERENCE: {resultado.get('favorito_predicho','?')} "
        f"elo={_elo_fav:.0f} ranking={_rk_fav} tier={tier}"
    )
    resultado['apostar'] = False
    resultado['phantom_data'] = True
    resultado['status'] = PICK_STATUS_NO_DATA
    resultado['motivo_reclasificacion'] = (
        f'ELO_RANK_INCOHERENCE [D152-05]: elo={_elo_fav:.0f} incompatible con '
        f'ranking={_rk_fav} en tier={tier} — historial contaminado probable'
    )
```

**Por qué ELO>1800 + ranking>400 es imposible:**
El ELO en este sistema se construye desde el historial de partidos. Un ELO 1800+ requiere haber ganado consistentemente a jugadores de ranking <100. Eso es físicamente imposible para alguien sin ranking ATP (o con ranking>400) en la era moderna del circuito. La combinación no puede existir en datos limpios.

### D152-06: `scripts/audit_phantom_history.py`

Script retroactivo que:
1. Escanea todos los `h2h_results_enhanced_*.json` de los últimos 30 días
2. Aplica `_validate_circuit_consistency()` a cada historial
3. Cruza picks afectados con shadow_book por match_key
4. Genera `reports/audit_phantom_history_FECHA.json` con: jugador, score, evidence, resultado real, P&L impactado

---

## Hallazgos Colaterales (conexiones ocultas descubiertas en esta sesión)

### H1: thf_cache no tiene identity fingerprint
`_lookup_player_history_temporal()` en L1181 resuelve por apellido parcial sin ninguna verificación de circuito, país, edad o ranking consistente. La función fue diseñada como fallback de velocidad, no de confiabilidad.

### H2: `phantom_data = False` hardcodeado en edge_calculator L1122
Esta línea reemplaza cualquier dato real que pudiera venir del H2H record. Significa que aunque ninja_h2h_parser marcara un jugador como phantom, edge_calculator lo resetea a False antes de llegar al gate Nodo-72. El gate Nodo-72 funciona, pero solo lee `ranking_analysis.phantom_identity_p1/p2` — no lee `data_quality.history_contamination`.

### H3: El mismo historial asignado a múltiples jugadores
Vesantera B.T. y Hohmann T. tenían exactamente los mismos partidos de Wimbledon/ATP Finals el mismo día. Esto sugiere que `_lookup_player_history_temporal()` devuelve el primer match del glob que contiene el apellido, sin verificar que el jugador sea el mismo.

### H4: `history_provenance` en el resultado final no llega a edge_calculator
El campo `data_quality.history_provenance` se genera en `_consolidate_result()` L1670-1675, pero `calcular_edge_completo()` en edge_calculator no lo lee en ningún punto. La información de si un historial viene de thf_cache está disponible pero nunca es consumida para gates de seguridad.

### H5: Vesantera tenía `n_h2h=0` + `data_completeness=0.5` + `ELO=1974`
Estos tres campos juntos son la firma exacta de phantom contaminado: sin H2H directo (n_h2h=0), datos incompletos (completeness=0.5), pero ELO altísimo (1974 = top-20 ATP). Esta combinación debería haber sido un gate automático.

---

## Tests — REGLA-T53 (8 tests)

**Archivo:** `tests/test_nodo152_phantom_history.py`

```
test_elite_tournament_itf_blocks          → ATP Finals en historial ITF → contaminated=True, score≥100
test_gs_top10_double_challenger_blocks    → 2x GS vs top-10 en Challenger → contaminated=True
test_gs_wildcard_legitimate_clean         → GS vs rank#180 en Challenger → contaminated=False (score<50)
test_thf_cache_amplifier_active           → score base 40 + thf_cache → score 60 → contaminated=True
test_atp_player_gs_clean                  → Zhang Zhizhen (ATP, rank~60) GS vs top-50 → tier=atp → clean
test_propagation_to_data_quality          → _process_match con thf_cache contaminado → data_quality.history_contamination.p1_contaminated=True
test_edge_calculator_blocks_phantom_hist  → partido con data_quality.history_contamination → status=NO_DATA
test_elo_rank_incoherence_gate            → elo=1974, ranking=None, tier=itf → phantom_data=True, status=NO_DATA
```

---

## Archivos modificados

| Archivo | Deliverable | Líneas clave |
|---|---|---|
| `scraping/ninja_h2h_parser.py` | D152-01, D152-02, D152-03 | L874 (nueva función), L1186 (THF gate), L1665 (data_quality) |
| `edge_calculator.py` | D152-04, D152-05 | L1253 (después de Nodo-72 gate) |
| `scripts/audit_phantom_history.py` | D152-06 | nuevo archivo |
| `tests/test_nodo152_phantom_history.py` | Tests | nuevo archivo |

---

## Relación con Nodo-72

Nodo-72 (`_detect_phantom_identity()`) detecta **homónimos** — dos jugadores con el mismo nombre real. Nodo-152 detecta **contaminación de circuito** — historial de un jugador asignado a otro diferente por matching de apellido incorrecto. Son bugs distintos con mecanismos distintos. Ambos deben coexistir.

---

## Wikilinks

[[Nodo-72]] phantom_identity original | [[Nodo-145]] tipo_cancha fix | [[Nodo-51]] player_registry entity resolution

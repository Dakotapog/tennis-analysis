# Sprint Post-Mortem 19-jun-2026 --- Fixes Nodo-28 Fase 2 + Nodo-29 Integración

> **Estado:** COMPLETADO --- 2026-06-19 | Tests: 1050→1113 passed
> **Wikilinks:** [[Nodo-28-Conditional-Decomposition-Metamodel]] | [[Nodo-29-Circuit-Asymmetry-Deflator]] | [[Sprint-Pipeline]]
> **Origen:** Cruce edge_report vs resultados 18-jun-2026. APOSTAR hit rate = 29.4% (5/17) vs población 59.5% (50/84). El filtro de edge selecciona la cola equivocada.
> **Datos fuente:** `reports/edge_report_20260619_*.json` + `reports/resultados_finales_20260619_092443.json`

---

## Diagnóstico Resumido

### Hallazgos del cruce edge_report vs resultados (18-jun-2026)

```
APOSTAR pool:     5/17 = 29.4%   (peor que random)
WATCHLIST pool:   0/4  =  0.0%
Población total: 50/84 = 59.5%

Por superficie:  clay 61.9% (n=63) | hard 64.3% (n=14) | grass 28.6% (n=7)
```

**5 problemas identificados con evidencia:**

| # | Problema | Evidencia | Impacto |
|---|---|---|---|
| P1 | VolConf no propagado como campo estructurado | SkillF/Alpha/VolConf viven en texto de reasoning, no como JSON keys | edge_calculator no puede consumirlos |
| P2 | VolConf=0 no colapsa el componente surface | Keys: raw=0.0 grass, aún clasificada APOSTAR a 54.1% | Picks sin datos de superficie pasan como APOSTAR |
| P3 | n_axes_active=1 picks tienen 29% hit rate | 2/7 correctos en pool APOSTAR | BBI sola no predice; estos picks deberían ser watchlist |
| P4 | STRUCTURAL_ALPHA no chequea oponente | Sziedat (SA, triple=0.49) FALLO — oponente también HOT | Amplificar confianza unidireccional es peligroso |
| P5 | Nodo-29 circuit_asymmetry no integrado en edge_calculator | Fase 1-3 activas en rivalry_analyzer, Fase 4 inexistente | Señal de circuito calculada pero no usada en sizing |

---

## Límites de Este Sprint

### LO QUE SÍ SE TOCA

- `analysis/rivalry_analyzer.py` — exponer VolConf y surface fields como JSON estructurado
- `edge_calculator.py` — consumir VolConf, implementar triple_alignment_score completo, integrar circuit_asymmetry
- Tests nuevos para cada fix

### LO QUE NO SE TOCA

- No se modifica Kelly ni sizing directo (solo informativo hasta validación V-28-2 con n>=20)
- No se baja el threshold de Nodo-29 CTI_max (0.8 validado como correcto — 3/3 en MODERATE_ASYMMETRY)
- No se modifica la lógica de Markov ni de Erdos
- No se tocan pesos base en normalization.py (MAX_RAW_SCORES, DEFAULT_WEIGHTS)
- No se modifican scrapers ni APIs
- trader_ev_tenis.py y betplay_combo_builder.py solo lectura de campos nuevos, sin cambios de lógica

---

## FIX-1: Propagar VolConf como campo estructurado

### Problema
`calculate_surface_specialization()` en `rivalry_analyzer.py` calcula SkillFactor, AlphaBonus y VolConf pero los emite como texto en el array `reasoning`. El edge_calculator no puede leerlos.

### Cambio
En `calculate_surface_specialization()`, agregar al dict de retorno:

```python
return {
    'score': final_score,
    'raw_score': raw_score,        # ya existe
    'win_rate': win_rate,          # ya existe
    'matches': n,                  # ya existe
    'skill_factor': skill_factor,          # NUEVO
    'alpha_bonus': alpha_bonus,            # NUEVO
    'volume_confidence': volume_confidence, # NUEVO
    'surface_alpha': surface_alpha,        # NUEVO (win_rate - overall_win_rate)
    # ... campos existentes sin cambiar
}
```

### Límites
- Solo agregar campos al return dict, no cambiar lógica de cálculo
- Los campos existentes (score, raw_score, win_rate, matches) NO se tocan
- `reasoning` text sigue emitiéndose para debugging humano

### Tests FIX-1

```
TF1-01: surface_specialization return dict contiene 'skill_factor' float
TF1-02: surface_specialization return dict contiene 'alpha_bonus' float
TF1-03: surface_specialization return dict contiene 'volume_confidence' float
TF1-04: surface_specialization return dict contiene 'surface_alpha' float
TF1-05: skill_factor == (max(win_rate, 0.01) / 0.5) ** 1.5 (formula exacta)
TF1-06: volume_confidence == min(n / 8.0, 1.0) (formula exacta)
TF1-07: alpha_bonus >= 1.0 siempre (max(alpha, 0) garantiza floor)
TF1-08: campos existentes (score, raw_score, win_rate, matches) no cambian valores
TF1-09: reasoning text sigue presente en output
```

---

## FIX-2: VolConf=0 colapsa componente surface en edge_calculator

### Problema
Keys tenía raw=0.0 en grass (cero datos) pero fue clasificada APOSTAR a 54.1%. Cuando VolConf=0 (o muy bajo), el componente `surface_specialization` debería contribuir 0% a la predicción, no dejar que otros componentes "compensen" la ausencia de datos de superficie.

### Cambio
En `edge_calculator.py`, al calcular edge para cada pick:

```python
# Leer VolConf del pick (FIX-1 lo expone)
vol_conf_fav = pick.get('surface_specialization_p1', {}).get('volume_confidence', 1.0)
vol_conf_dog = pick.get('surface_specialization_p2', {}).get('volume_confidence', 1.0)

# Si el favorito tiene VolConf < 0.25 en la superficie del partido:
# marcar como data_insufficient_surface = True
# Este flag NO cambia el edge ni Kelly — solo agrega campo informativo
# que el trader puede usar para filtrar
data_insufficient_surface = min(vol_conf_fav, vol_conf_dog) < 0.25
```

### Límites
- Campo informativo solamente — no modifica edge ni Kelly en este sprint
- Threshold 0.25 = n=2 partidos en superficie (min(2/8, 1.0))
- Si VolConf no disponible (legacy data), default 1.0 (no flag)

### Tests FIX-2

```
TF2-01: data_insufficient_surface = True cuando VolConf favorito = 0.0
TF2-02: data_insufficient_surface = True cuando VolConf dog = 0.125 (n=1)
TF2-03: data_insufficient_surface = False cuando ambos VolConf >= 0.25
TF2-04: data_insufficient_surface = False cuando VolConf no presente en dict (default 1.0)
TF2-05: campo data_insufficient_surface presente en cada pick del edge_report
TF2-06: edge y kelly_kl NO cambian por presencia de este campo
TF2-07: retroactivo Keys: data_insufficient_surface = True (VolConf=0.0 grass)
```

---

## FIX-3: n_axes_active < 2 suprime a watchlist

### Problema
Picks con solo 1 eje activo (pura BBI sin surface ni regime signal) fueron 2/7 (29%) en el pool APOSTAR del 18-jun. La señal BBI sola no predice — necesita al menos otra fuente de information asymmetry.

### Cambio
En `edge_calculator.py`, después de calcular `triple_alignment_score()`:

```python
# REGLA-N28-F2-1: suprimir a watchlist si n_axes_active < 2
if alignment_data['n_axes_active'] < 2 and clasificacion == 'apostar':
    clasificacion = 'watchlist'
    motivo_cambio = 'N28F2: n_axes_active < 2 (BBI sola no predice)'
```

### Límites
- Solo mueve de apostar → watchlist, nunca al revés
- No afecta picks que ya son watchlist o sin_edge
- El motivo queda registrado en el pick para auditoría
- No modifica el cálculo de edge ni de triple_alignment — solo la clasificación

### Tests FIX-3

```
TF3-01: pick con n_axes_active=0 y clasificacion='apostar' → reclasificado a 'watchlist'
TF3-02: pick con n_axes_active=1 y clasificacion='apostar' → reclasificado a 'watchlist'
TF3-03: pick con n_axes_active=2 y clasificacion='apostar' → se mantiene 'apostar'
TF3-04: pick con n_axes_active=3 y clasificacion='apostar' → se mantiene 'apostar'
TF3-05: pick con n_axes_active=1 y clasificacion='watchlist' → no cambia (ya es watchlist)
TF3-06: pick con n_axes_active=1 y clasificacion='sin_edge' → no cambia
TF3-07: motivo_cambio contiene 'N28F2' cuando se aplica reclasificación
TF3-08: edge y kelly_kl del pick NO cambian por reclasificación
TF3-09: retroactivo 18-jun: 7 picks con n_axes=1 que eran APOSTAR ahora serían WATCHLIST
```

---

## FIX-4: STRUCTURAL_ALPHA chequea alignment del oponente

### Problema
Sziedat (STRUCTURAL_ALPHA, triple=0.49) perdió contra Stoyanov que también estaba HOT. El triple_alignment_score actual solo mira al jugador favorecido — si el oponente también tiene alignment, no hay ventaja informacional unidireccional.

### Cambio
En `edge_calculator.py`, dentro de `triple_alignment_score()`:

```python
# Calcular alignment para AMBOS jugadores
alignment_fav = triple_alignment_score_single(pick, player='favorito')
alignment_dog = triple_alignment_score_single(pick, player='underdog')

# Net alignment = ventaja informacional relativa
net_alignment = alignment_fav['triple_alignment'] - alignment_dog['triple_alignment']

# REGLA-N28-F2-2: STRUCTURAL_ALPHA solo si net_alignment > 0.25
# Si ambos están alineados, no hay asimetría informacional
if alignment_fav['alignment_flag'] == 'STRUCTURAL_ALPHA':
    if net_alignment < 0.25:
        alignment_fav['alignment_flag'] = 'CONTESTED_ALPHA'  # ambos alineados
```

### Límites
- No cambia la fórmula de triple_alignment individual
- Solo afecta la flag final (STRUCTURAL_ALPHA vs CONTESTED_ALPHA)
- CONTESTED_ALPHA se trata como PARTIAL_ALIGNMENT para efectos de clasificación
- No modifica Kelly ni sizing

### Decisión de diseño (CERRADO — 2026-06-19)

El pseudocódigo `triple_alignment_score_single(pick, player='underdog')` sugería calcular surface_norm y bbi_norm por separado para cada jugador. La implementación real usa surface_norm y bbi_norm a nivel de partido (son atributos del partido, no del jugador individual — el BBI refleja el partido completo, no a un jugador en particular). Solo `regime_norm_dog` difiere porque el estado Markov sí es per-jugador. Esta decisión es **correcta por diseño**: surface_blindness y bookmaker_blindness son asimetrías del partido (no del jugador), regime_blindness es per-jugador. Cerrado como implementación válida.

### Tests FIX-4

```
TF4-01: STRUCTURAL_ALPHA se mantiene si oponente tiene alignment=0.0 (net > 0.25)
TF4-02: STRUCTURAL_ALPHA → CONTESTED_ALPHA si oponente tiene alignment=0.40 y favorito=0.49 (net=0.09 < 0.25)
TF4-03: net_alignment campo presente en output de triple_alignment_score()
TF4-04: CONTESTED_ALPHA tratado como PARTIAL_ALIGNMENT en clasificación (no suprime a watchlist si n_axes>=2)
TF4-05: retroactivo Sziedat: alignment_fav=0.49, oponente HOT → CONTESTED_ALPHA
TF4-06: retroactivo Eala (18-jun anterior): alignment_fav=0.86, Rybakina NEUTRAL → STRUCTURAL_ALPHA se mantiene
TF4-07: edge y kelly_kl NO cambian por CONTESTED_ALPHA
```

---

## FIX-5: Integrar circuit_asymmetry en edge_calculator (Nodo-29 Fase 4)

### Problema
`circuit_asymmetry` ya se calcula en `rivalry_analyzer.py` (CTI, ratio, deflactor, señal) y funciona bien (3/3 en MODERATE_ASYMMETRY). Pero `edge_calculator.py` no lo lee — la señal se calcula y se descarta.

### Cambio
En `edge_calculator.py`, para cada pick:

```python
circuit = pick.get('circuit_asymmetry', {})
circuit_signal = circuit.get('signal', 'SYMMETRIC')

# Campo informativo en edge_report
pick['circuit_asymmetry_signal'] = circuit_signal
pick['circuit_asymmetry_ratio'] = circuit.get('asymmetry_ratio', 1.0)

# REGLA-N29-EDGE-1: si el modelo favorece al jugador deflactado
# Y la asimetría es MODERATE o STRONG → alertar
if circuit_signal in ('MODERATE_ASYMMETRY', 'STRONG_ASYMMETRY'):
    deflated_player = circuit.get('player_deflated', None)
    if deflated_player and deflated_player == pick.get('favorito_predicho'):
        pick['circuit_warning'] = True
        # Informativo: "estás apostando por el jugador de circuito inferior"
```

### Límites
- Solo lectura de circuit_asymmetry del H2H — no modifica cómo se calcula en rivalry_analyzer
- No modifica edge ni Kelly — campo `circuit_warning` es informativo
- No baja el threshold de CTI_max (0.8 validado como correcto)
- El threshold de asimetría ratio > 2.0 para MODERATE_ASYMMETRY no se cambia

### Tests FIX-5

```
TF5-01: circuit_asymmetry_signal presente en cada pick del edge_report
TF5-02: circuit_asymmetry_ratio presente en cada pick del edge_report
TF5-03: circuit_warning = True cuando favorito == player_deflated Y signal MODERATE
TF5-04: circuit_warning = True cuando favorito == player_deflated Y signal STRONG
TF5-05: circuit_warning ausente o False cuando signal = SYMMETRIC
TF5-06: circuit_warning ausente o False cuando favorito != player_deflated
TF5-07: edge y kelly_kl NO cambian por presencia de circuit_warning
TF5-08: default ratio=1.0 y signal='SYMMETRIC' cuando circuit_asymmetry no existe en pick
```

---

## Asignación de Tareas

### Sonnet — Código (5 PRs independientes)

| Tarea | Archivo | Descripción | Depende de |
|---|---|---|---|
| S-1 | `analysis/rivalry_analyzer.py` | FIX-1: Exponer SkillFactor, AlphaBonus, VolConf, surface_alpha como campos JSON en return dict de `calculate_surface_specialization()` | — |
| S-2 | `edge_calculator.py` | FIX-2: Leer VolConf de surface_specialization, agregar campo `data_insufficient_surface` a cada pick | S-1 |
| S-3 | `edge_calculator.py` | FIX-3: Implementar regla `n_axes_active < 2 → watchlist` con motivo auditable | — |
| S-4 | `edge_calculator.py` | FIX-4: Calcular alignment de AMBOS jugadores, implementar CONTESTED_ALPHA, agregar net_alignment | — |
| S-5 | `edge_calculator.py` | FIX-5: Leer circuit_asymmetry del H2H, exponer signal/ratio/warning en edge_report | — |

**Orden sugerido:** S-1 → S-2 (secuencial), S-3 + S-4 + S-5 (paralelos entre sí)

**Reglas para Sonnet:**
- `python -m pytest tests/ --no-cov -q` debe seguir dando >= 1050 passed después de cada fix
- `python -c "import ast; ast.parse(open('archivo.py').read()); print('OK')"` antes de cada commit
- NO tocar normalization.py, trader_ev_tenis.py, betplay_combo_builder.py
- NO modificar Kelly, sizing, ni lógica de Markov/Erdos
- Cada fix es un commit independiente con tests pasando

### Haiku — Specs y Tests

| Tarea | Archivo | Descripción | Depende de |
|---|---|---|---|
| H-1 | `tests/test_nodo28_fase2.py` | Escribir tests TF1-01 a TF1-09 (FIX-1 VolConf propagación) | Spec FIX-1 de este doc |
| H-2 | `tests/test_nodo28_fase2.py` | Escribir tests TF2-01 a TF2-07 (FIX-2 data_insufficient_surface) | Spec FIX-2 de este doc |
| H-3 | `tests/test_nodo28_fase2.py` | Escribir tests TF3-01 a TF3-09 (FIX-3 n_axes suppression) | Spec FIX-3 de este doc |
| H-4 | `tests/test_nodo28_fase2.py` | Escribir tests TF4-01 a TF4-07 (FIX-4 CONTESTED_ALPHA) | Spec FIX-4 de este doc |
| H-5 | `tests/test_nodo29_integration.py` | Escribir tests TF5-01 a TF5-08 (FIX-5 circuit_asymmetry edge) | Spec FIX-5 de este doc |
| H-6 | `.spec/01_Nodos/Nodo-28-Conditional-Decomposition-Metamodel.md` | Actualizar estado Fase 2 con hallazgos de este post-mortem | Completar H-1 a H-4 |
| H-7 | `.spec/01_Nodos/Nodo-29-Circuit-Asymmetry-Deflator.md` | Actualizar estado Fase 4 con hallazgos y agregar CONTESTED_ALPHA reference | Completar H-5 |

**Reglas para Haiku:**
- Tests deben ser ejecutables con pytest sin dependencias externas
- Usar mocks para rivalry_analyzer y edge_calculator (no llamar APIs reales)
- Cada test tiene nombre descriptivo que mapea a TF#-## de este documento
- Tests retroactivos (TF3-09, TF4-05, TF4-06) pueden usar fixtures hardcoded con datos reales del 18-jun

---

## Validación Post-Sprint

| ID | Criterio | Cómo verificar |
|---|---|---|
| V-PM-1 | >= 1050 tests passing | `python -m pytest tests/ --no-cov -q` |
| V-PM-2 | Re-correr edge_calculator con datos 18-jun → Keys ya NO es APOSTAR | Manual con edge_report del 19-jun |
| V-PM-3 | 7 picks n_axes=1 del 18-jun ahora serían WATCHLIST | Verificar con retroactivo |
| V-PM-4 | Sziedat sería CONTESTED_ALPHA (no STRUCTURAL_ALPHA) | Verificar con retroactivo |
| V-PM-5 | circuit_asymmetry_signal presente en todos los picks | Grep edge_report output |
| V-PM-6 | Boogaard vs Overbeck tiene circuit_warning o es SYMMETRIC correctamente | Verificar valores CTI |
| V-PM-7 | No regresión en accuracy: re-correr validar_con_api con datos históricos | Comparar con baseline 59.5% |

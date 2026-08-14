# Nodo-155 — HCUC Pipeline Integration: Acumulación Diaria Automática de H152-01

**Fecha:** 2026-07-30
**Estado:** IMPLEMENTADO — 11/11 tests REGLA-T53 PASS
**Prioridad:** ALTA — sin esto H152-01 nunca podía graduarse (n=3 semilla, congelado para siempre)
**Wikilinks:** [[Nodo-152]] [[Nodo-98]] [[Nodo-91]] [[edge_calculator.py]] [[analysis/rivalry_analyzer.py]] [[shadow_book.py]]

---

## 1. Contexto

El 2026-07-29 se auditaron 3 apuestas ganadoras (Cocciaretto @2.75, Shick @2.75, Gea @2.50) que
`edge_calculator.py` nunca vio como oportunidad — el pipeline las descartó por confianza ~50%
(coin flip, `p_modelo` en [0.495, 0.52]) mientras `generar_tabla_favoritos2.py` las mostraba como
NO-BET. Se identificó una convergencia común (Hard Court + historial fuerte en superficie +
puntaje ligeramente favorable al favorito + al menos una señal especial) y se registró como
hipótesis **H152-01** en `validation/preregistered_hypotheses.json` (n_stop=30, n_actual=3
retrospectivo).

Se escribió `scripts/detect_hcuc_picks.py` como prueba de concepto, pero investigación de código
confirmó que sus campos de entrada (`quality_score_superficie`, `scalps_top20`,
`campeonatos_expirados`, `rivales_comunes_densidad`) eran **mockeados a mano** — no existían
serializados en ningún punto real del pipeline. Ese script nunca podría correr contra un
`edge_report` real: H152-01 habría quedado congelada en n=3 para siempre, sin acumulación
prospectiva.

**Pregunta que origina este Nodo:** *"pero esto como lo integro para que acumule n diariamente"*.

## 2. Diseño: reutilizar el patrón observacional existente

En vez de crear un script/cron nuevo, se sigue el patrón ya validado por H89-02
(`elo_dominance_axis`, Nodo-91) y H98-01 (`score_directo`, Nodo-98):

```
edge_calculator.calcular_edge_completo()  (corre en PASO 3 de run_daily.py, para TODO partido)
   └─ calcula flag observacional puro → resultado['hcuc_convergence'] / ['hcuc_signals']
        └─ shadow_book.log_picks() YA persiste apostar+watchlist+no_data con pick_snapshot completo
             └─ shadow_book.py --report YA lee pick_snapshot para segmentar por flag
```

Cero cambios en `run_daily.py`, cero crons nuevos, cero pasos manuales. El flag se calcula el
mismo día que corre el pipeline, para cada partido, automáticamente.

## 3. Decisiones (D155-01 → D155-05)

### D155-01 — `analysis/rivalry_analyzer.py::analyze_surface_specialization()`

Dos contadores aditivos nuevos en el dict de retorno (mismo patrón que Nodo-112/C3 añadió
`campeon_tier`/`campeon_torneo`/`campeon_days_ago`), sin tocar la lógica de puntaje existente:

- `top20_wins`: incrementa dentro del loop de `surface_matches` cuando `opponent_rank <= 20` en
  el branch de victoria (mismo bloque que ya asigna `points = 40`).
- `campeonatos_expirados_count`: incrementa dentro del loop de `_tour_stats`
  (`TORNEO_COMPLETO_BONUS`), en la misma condición que ya califica "torneo completo"
  (`_ts['wins'] >= _min_wins and _ts['losses'] == 0`), pero **sin interferir con el `break`**
  existente que preserva el bonus de puntaje del campeón más reciente — es un contador paralelo,
  no reemplaza esa lógica.

`find_common_opponents()`/densidad de rivales comunes (señal RIVALES_COMUNES) queda **fuera de
alcance** — ver §5.

### D155-02 — `_calc_hcuc_convergence()` en `edge_calculator.py`

Función pura (junto a `_calc_elo_dominance_axis`/`_calc_meta_score_directo`, antes del bloque
"PIPELINE COMPLETO POR PARTIDO"):

```python
def _calc_hcuc_convergence(resultado: dict, surf_fav: dict, surf_dog: dict) -> dict:
    if (resultado.get('superficie') or '').lower() not in ('dura', 'hard'):
        return {'match': False, 'signals': []}
    quality = surf_fav.get('score', 0.0) or 0.0
    puntaje_delta = quality - (surf_dog.get('score', 0.0) or 0.0)
    p_modelo = resultado.get('p_modelo', 0.0) or 0.0
    cuota_fav = resultado.get('cuota_favorito', 0.0) or 0.0
    if quality < 16.5 or puntaje_delta < 0.08:
        return {'match': False, 'signals': []}
    if not (0.495 <= p_modelo <= 0.52):
        return {'match': False, 'signals': []}
    if not (2.3 <= cuota_fav <= 3.0):
        return {'match': False, 'signals': []}
    signals = []
    # CAMPEON_RECIENTE / RACHA_HOT / SCALP_TOP20 / CAMPEONATOS_EXPIRADOS
    ...
    if not signals:
        return {'match': False, 'signals': []}
    return {'match': True, 'signals': signals}
```

Llamada dentro de `calcular_edge_completo()`, en el mismo bloque final donde ya se asigna
`score_directo`/`direccion_meta` (Nodo-98), reutilizando `_surf_fav`/`_surf_dog` — variables ya en
scope desde la lógica de `data_insufficient_surface` más arriba en la misma función, confirmado
por grep que no hay reasignación/shadowing entre ambos puntos:

```python
_hcuc = _calc_hcuc_convergence(resultado, _surf_fav, _surf_dog)
resultado['hcuc_convergence'] = _hcuc['match']
resultado['hcuc_signals'] = _hcuc['signals']
```

**Garantía:** no modifica `edge`, `kelly_kl`, `apostar` ni ningún gate — puramente observacional
(verificado por `test_hcuc_fields_en_edge_completo`, que confirma la presencia de los campos sin
alterar el resto del contrato de salida).

### D155-03 — Segmento H152-01 en `shadow_book.py`

Tres puntos de lectura del flag `pick_snapshot.hcuc_convergence`, todos derivados del mismo campo:

1. `_append_segment(settled, lines, "HCUC (H152-01: ...)", lambda r: ...)` — junto a
   CAPA2/ELO_DOMINANCE en el reporte de texto plano.
2. Entrada `("H152-01", "HCUC (hard+quality+coinflip+señal especial)", _hcuc_recs, 0.385, 0.55)`
   en el loop de CHECKLIST SEMANAL (B108-04) — dispara SPRT automático cuando n≥30, con
   breakeven real de la hipótesis (1/2.6≈38.5%) como `_p0`.
3. `_hyp("H152-01", "HCUC hard+quality+coinflip+señal especial", _is_hcuc_d, 30)` en el tracker
   estructurado de hipótesis (consumido por `pipeline_tracker.py`) — mismo `n_stop=30` que el
   JSON pre-registrado.

### D155-04 — Corrección de `umbrales_congelados` en `validation/preregistered_hypotheses.json`

Los umbrales originales (escritos junto con `scripts/detect_hcuc_picks.py`) referenciaban nombres
de campo inventados (`quality_score_superficie`, `confianza` en escala 0-100, `campeon_dias_ago`,
`scalps_top20`, `campeonatos_expirados`). Se corrigieron para apuntar a los campos reales
(`surface_specialization_meta[...]['score']`, `resultado['p_modelo']` en escala 0-1,
`_surf_fav['campeon_days_ago']`, `_surf_fav['top20_wins']`,
`_surf_fav['campeonatos_expirados_count']`). Los **valores numéricos son idénticos** a los
originales (16.5, 0.08, [0.495,0.52], [2.3,3.0]) — esto es una corrección de definición
operacional, no un cambio de umbral prohibido por la regla anti-p-hacking (la acumulación
prospectiva real aún no había empezado; `n_actual` seguía en 3, retrospectivo).

`opciones_senales` pasa de 5 a 4 (RIVALES_COMUNES se marca `opciones_senales_diferidas`, ver §5).

### D155-05 — Retiro de `scripts/detect_hcuc_picks.py`

Prueba de concepto con campos mockeados, sin conexión al pipeline real. Con la lógica real viviendo
en `edge_calculator.py`, mantenerlo habría creado dos fuentes de verdad divergentes. Eliminado.

## 4. Tests (REGLA-T53)

`tests/test_nodo155_hcuc_convergence.py` — 11 tests, todos invocan la función real:

- `TestCasosSemillaReales` (3 tests): Cocciaretto/Shick/Gea reconstruidos con la forma real de
  `resultado`/`_surf_fav`/`_surf_dog` — los 3 dan `match=True` con la señal específica esperada.
- `TestRutasDeRechazo` (7 tests): superficie no-hard, quality insuficiente, delta insuficiente,
  confianza fuera de rango, cuota fuera de rango, sin señal especial, `campeon_tier` no calificado.
- `test_hcuc_fields_en_edge_completo` (1 test): integración real vía `calcular_edge_completo()`
  con fixture `surface_specialization_meta` — confirma que `hcuc_convergence`/`hcuc_signals`
  aparecen en el dict de salida.

11/11 PASS.

## 5. Fuera de alcance (declarado explícitamente, no bloquea H152-01)

**RIVALES_COMUNES** (densidad de oponentes comunes ≥80%) — requeriría serializar
`find_common_opponents()` + `density_confidence()` (`rivalry_analyzer.py`) al `resultado`, cambio
más invasivo que los otros 4 contadores aditivos. El gate de "≥1 señal especial de 4" (en vez de 5)
sigue siendo matemáticamente válido para los 3 casos semilla — cada uno ya calza con al menos 1 de
las 4 señales operativas sin necesitar RIVALES_COMUNES. Candidato a **Nodo-156** si una auditoría
futura lo justifica.

## 6. Verificación

1. `python -c "import ast; ast.parse(open('FILE').read())"` — OK en `edge_calculator.py`,
   `analysis/rivalry_analyzer.py`, `shadow_book.py`.
2. `python -m pytest tests/test_nodo155_hcuc_convergence.py -v --no-cov` — 11/11 PASS.
3. `python -m pytest tests/ --no-cov -q` — suite completa sin regresiones.
4. Próximo `run_daily.py` (PASO 3) ya calculará `hcuc_convergence` para cada partido Hard Court sin
   ningún paso manual adicional; `shadow_book.py --report` mostrará la línea `[H152-01]` con n
   creciendo día a día a partir del 2026-07-31.

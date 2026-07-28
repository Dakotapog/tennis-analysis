# Nodo-150 — Live Games Risk Intelligence: Hallazgos Simulación Doctoral 2026-07-28

**Fecha:** 2026-07-28
**Estado:** SPEC — pendiente implementación
**Prioridad:** ALTA — hallazgos con impacto directo en pérdidas reales
**Wikilinks:** [[Nodo-147]] [[Nodo-133]] [[Nodo-142]] [[live_desk]] [[_calcular_certeza_condicional]] [[_check_games_convergencia]]

---

## 1. Contexto: Qué generó este Nodo

Simulación en vivo 3 piernas UNDER (12:29–12:51, 2026-07-28):

| Pierna | Partido | Linea | Cuota entrada | Score entrada | Resultado |
|---|---|---|---|---|---|
| 1 | Novkirichka vs Faure | UNDER 20.5 | @1.91 | 6:2, 11j | PROBABLE WIN — cuota cayó @1.41 (-26%) |
| 2 | Alexandrova vs Liutova | UNDER 32.5 | @1.96 | sin score | RIESGO → RECUPERÓ (set 3 dominado) |
| 3 | Dahlin vs Martineau | UNDER 22.5 | @1.98 | 6:4, 13j | PROBABLE WIN — cuota cayó @1.46 (-27%) |

La simulación expuso 5 gaps arquitectónicos en el sistema live que no existían como Nodos previos.

---

## 2. Los 5 Gaps Identificados

### Gap-1: ENVENENADA_CUOTA (NO implementado)

**Problema:** La regla ENVENENADA actual solo verifica `linea_drift > +2j`. No detecta cuando el **mercado reprecia la cuota** agresivamente hacia arriba — señal más sensible de riesgo real.

**Evidencia:** Alexandrova a los 6 minutos de entrada: cuota 1.96 → 2.55 (+30.1%). El sistema NO generó alerta. Un trader mirando la dashboard lo hubiera visto, pero no hay indicador visual ni filtro automático.

**Umbral propuesto:** `cuota_drift > +15%` → mismo tratamiento que linea_envenenada:
- Log: `[ITF_LIVE] X UNDER cuota_drift=+30.1% > +15% → CUOTA_ENVENENADA`
- Badge visual en X3: fondo rojo igual que LINEA ENVENENADA
- No disparar combo si cuota_envenenada

### Gap-2: SET1_TIEBREAK_GATE (NO implementado)

**Problema:** El proxy model clasifica zona DOMINANTE/COINFLIP en base a H2H y ranking pre-partido. Pero si el primer set terminó en tiebreak (`games_set1 >= 12`), la clasificación pre-partido queda **invalidada por la evidencia empírica del partido en curso**.

**Evidencia:** Alexandrova pre-partido = DOMINANTE (Top-30 WTA vs menor). Set 1 terminó 6:7 (tiebreak = 13 juegos). El sistema seguía usando zona=DOMINANTE. Debería haber forzado COINFLIP.

**Un set tiebreak significa:** ambos jugadores son igualmente competitivos ese día → tercer set probable (~65% empírico) → UNDER riesgo alto → no entrar UNDER con margen < 8j sobre linea_t0.

**Regla propuesta:**
```python
if sets_complete >= 1 and games_set1 >= 12:
    zona = "COINFLIP_FORZADO"   # override permanente
    # Si ya es COINFLIP: mantener
    # Si era DOMINANTE: downgrade → ajusta µ y σ de certeza_condicional
```

### Gap-3: CUOTA_CRASH_CONFIRMACION (NO implementado)

**Problema:** El sistema detecta señales de RIESGO (cuota sube) pero no tiene equivalente visual para CONFIRMACIÓN (cuota cae dramáticamente).

**Evidencia:**
- Novkirichka 12:32: cuota @1.41 (-26% drift) = set 2 siendo dominado → WIN inmeminente
- Dahlin 12:40: cuota @1.46 (-27% drift) = set 2 terminando en UNDER → WIN inminente
- Pero el dashboard mostraba Drift en azul/verde sin distinción entre -5% y -27%.

**Badge propuesto:** Si `cuota_drift < -15%`:
```html
<span style="background:#00c851;color:white;padding:1px 6px;
border-radius:3px;font-size:0.72em;font-weight:bold;">CONFIRMADO</span>
```
Aparece junto al partido en columna Estado, igual que CONFIRMAR UNDER.

### Gap-4: games_set1 como input a _calcular_certeza_condicional (NO implementado)

**Problema:** `_calcular_certeza_condicional` recibe `games_played` (total) y `sets_complete` pero NO recibe el score del set 1 específicamente. Por tanto no puede ajustar su estimación de juegos restantes según la dinámica del partido.

**Evidencia:** Con Alexandrova en gp=28 (set 3 iniciado), la función calculaba certeza sin saber que set 1 fue tiebreak (13j). La distribución Gaussiana no se actualizó.

**Mejora propuesta:**
```python
def _calcular_certeza_condicional(
    linea, direccion, games_played, sets_complete, current_games, zona,
    sets_home=None, sets_away=None,
    games_set1=None,   # NUEVO — juegos del set 1 completado
):
    # Si games_set1 >= 12 → forzar COINFLIP incluso si zona=DOMINANTE
    if games_set1 is not None and games_set1 >= 12:
        zona = "COINFLIP"
    ...
```

El campo `games_set1` se deriva del primer par `(h, a)` del array `livedata.statistics.sets` donde el set está completo.

### Gap-5: SET1_SCORE_GATE antes de disparar combo ITF_LIVE_GAMES (NO implementado)

**Problema:** Cuando `_check_games_convergencia` dispara un combo por `itf_live_signals`, no verifica si alguna pierna tiene set 1 en tiebreak. El combo puede incluir piernas de alto riesgo automáticamente.

**Regla propuesta:**
```python
# En _fire_itf_live_games_combo():
for sig in itf_live_signals_alta:
    sd = sig.get("score_data") or {}
    # Calcular games_set1 si está disponible
    # (primer set completo del score_str o del sets array)
    if sd.get("games_set1", 0) >= 12:
        logger.warning(f"[ITF_LIVE_GATE] {sig['partido']} set1={sd['games_set1']}j ≥12 → EXCLUIDA del combo")
        continue   # no incluir en combo
    combo_legs.append(sig)
```

---

## 3. Deliverables — D150-01 a D150-06

### D150-01: ENVENENADA_CUOTA filter en _check_games_convergencia

**Archivo:** `live_desk.py`
**Función:** `_check_games_convergencia()` — bloque ITF live signals
**Lógica:**
```python
CUOTA_ENVENENADA_UMBRAL = 15.0  # %

cuota_drift_pct = sig.get("cuota_drift_pct") or 0
if cuota_drift_pct > CUOTA_ENVENENADA_UMBRAL:
    sig["cuota_envenenada"] = True
    logger.info(f"[ITF_LIVE] {partido} UNDER cuota_drift={cuota_drift_pct:+.1f}% > +{CUOTA_ENVENENADA_UMBRAL}% → CUOTA_ENVENENADA")
```

**Impacto:** El combo fire ya revisa `linea_envenenada`; extender la misma guarda a `cuota_envenenada`.

### D150-02: SET1_TIEBREAK_GATE — parsing y propagación de games_set1

**Archivo:** `live_desk.py`
**Funciones:** `_parse_kambi_livedata_sets()` y `_enrich_live_score()`
**Lógica:**
```python
# En _parse_kambi_livedata_sets — al procesar el primer set completo:
if sets_complete == 1:   # primer set que completamos
    result["games_set1"] = h + a   # ej: 6+7=13 (tiebreak)
```

```python
# En _enrich_live_score — propagar al signal dict:
if sig["score_data"]:
    sig["games_set1"] = sig["score_data"].get("games_set1")
```

### D150-03: Forzar zona COINFLIP_FORZADO en _check_games_convergencia si tiebreak

**Archivo:** `live_desk.py`
**Función:** sección de clasificación de `itf_live_signals` en `_check_games_convergencia`
**Lógica:**
```python
games_set1 = sig.get("games_set1") or (sig.get("score_data") or {}).get("games_set1")
if games_set1 and games_set1 >= 12:
    sig["zona"] = "COINFLIP_FORZADO"
    logger.info(f"[ITF_LIVE] {partido} set1={games_set1}j → zona=COINFLIP_FORZADO")
```

Esto afecta el cálculo de certeza_condicional cuando se llama con `zona`.

### D150-04: games_set1 como parámetro en _calcular_certeza_condicional

**Archivo:** `live_desk.py`
**Función:** `_calcular_certeza_condicional()`
**Firma nueva:**
```python
def _calcular_certeza_condicional(
    linea, direccion, games_played, sets_complete, current_games, zona,
    sets_home=None, sets_away=None,
    games_set1=None,   # NUEVO
) -> Dict[str, Any]:
    # Override zona si tiebreak en set 1
    if games_set1 is not None and int(games_set1) >= 12 and zona == "DOMINANTE":
        zona = "COINFLIP"
```

Todos los call sites pasan `games_set1=sig.get("score_data", {}).get("games_set1")`.

### D150-05: Badge CONFIRMADO en X3 cuando cuota_drift < -15%

**Archivo:** `live_desk.py`
**Función:** bloque de rendering X3 (≈L1313-1346)
**Lógica:**
```python
# Junto al bloque _envenenada / _over_cand:
_cuota_confirmada = (
    _drift is not None and _drift < -15.0
    and _estado in ("EN_VIVO", "ITF_VIVO")
)
if _cuota_confirmada:
    _partido_html = (
        f'{_partido_raw} '
        f'<span style="background:#00c851;color:#000;padding:1px 6px;'
        f'border-radius:3px;font-size:0.72em;font-weight:bold;">'
        f'MERCADO CONFIRMA</span>'
    )
```

Se muestra SOLO si estado EN_VIVO (no pre-partido). No conflicta con CUOTA_ENVENENADA ni LINEA ENVENENADA (son excluyentes por definición — drift no puede ser +30% y -15% al mismo tiempo).

También agregar columna `cuota_envenenada` al badge de Estado (igual que linea_envenenada → rojo).

### D150-06: SET1_SCORE_GATE antes de fuego de combo ITF_LIVE_GAMES

**Archivo:** `live_desk.py`
**Función:** sección `_fire_itf_live_games_combo` / bloque de combo en `_check_games_convergencia`
**Lógica:**
```python
# Filtrar piernas con set1 tiebreak antes de armar el combo
combo_eligible = []
for sig in alta_itf_for_combo:
    gs1 = (sig.get("score_data") or {}).get("games_set1") or 0
    if gs1 >= 12:
        logger.info(f"[ITF_LIVE_GATE] {sig['partido']} excluida del combo (set1={gs1}j, tiebreak)")
        continue
    if sig.get("cuota_envenenada"):
        logger.info(f"[ITF_LIVE_GATE] {sig['partido']} excluida del combo (cuota_envenenada)")
        continue
    combo_eligible.append(sig)
```

---

## 4. Tests Requeridos (REGLA-T53)

Archivo: `tests/test_nodo150_live_risk_intelligence.py`

| Test | Función | Qué verifica |
|---|---|---|
| test_150_01_cuota_envenenada_flag | `_check_games_convergencia` (mock) | cuota_drift +30% → `cuota_envenenada=True` |
| test_150_02_cuota_envenenada_no_falso_positivo | idem | cuota_drift +10% → `cuota_envenenada=False` (bajo umbral) |
| test_150_03_set1_tiebreak_zona_override | `_calcular_certeza_condicional` | games_set1=13, zona=DOMINANTE → zona efectiva=COINFLIP |
| test_150_04_set1_normal_sin_override | idem | games_set1=8, zona=DOMINANTE → zona efectiva=DOMINANTE |
| test_150_05_parse_livedata_games_set1 | `_parse_kambi_livedata_sets` | home=[7,4,-1],away=[6,5,-1] → games_set1=13 |
| test_150_06_parse_livedata_set1_normal | idem | home=[6,4,-1],away=[2,6,-1] → games_set1=8 |
| test_150_07_combo_gate_excluye_tiebreak | sección combo fire (mock) | pierna con games_set1=13 → excluida del combo |
| test_150_08_combo_gate_excluye_cuota_envenenada | idem | pierna con cuota_envenenada=True → excluida del combo |
| test_150_09_badge_confirmado_umbral | render X3 (helper check) | cuota_drift=-27% → _cuota_confirmada=True |
| test_150_10_badge_confirmado_no_premature | idem | cuota_drift=-5% → _cuota_confirmada=False |

---

## 5. Lecciones Operativas (no código — para memoria del sistema)

### Regla de Entrada Live (operacional)

**NUNCA entrar UNDER live sin score del set 1 visible (`score_data is not None`).**

Si `score_data is None` (Kambi livedata no responde o partido muy reciente):
- Esperar hasta el próximo ciclo (15-25s) antes de incluir en combo
- Si persiste sin score → flag `score_pending=True` → no disparar combo

### Lectura del Cuota_Drift como Semáforo

| Cuota_drift | Significado | Acción |
|---|---|---|
| < -15% | Mercado compra agresivo (confirma dirección) | Badge VERDE — alta confianza |
| -15% a +5% | Mercado estable — señal limpia | Normal — operar si resto OK |
| +5% a +15% | Mercado ligeramente en contra — monitorear | Alerta amarilla |
| > +15% | Mercado descubrió info que el modelo no tiene | CUOTA_ENVENENADA — no combo |
| > +30% | Evento estructural (3er set confirmado, break serie) | Salida si hay cash-out |

### Regla del Set 1 como Clasificador de Riesgo

| games_set1 | Clasificación | Implicación UNDER |
|---|---|---|
| ≤ 8 juegos (6:0, 6:1, 6:2) | DOMINANTE_CONFIRMADO | UNDER muy probable — máxima confianza |
| 9-10 juegos (6:3, 6:4) | NORMAL | Seguir modelo pre-partido |
| 11 juegos (6:5) | COMPETITIVO | Reducir confianza |
| 12-13 juegos (7:5, tiebreak) | COINFLIP_FORZADO | UNDER riesgoso — requiere margen ≥8j |

---

## 6. Archivos Modificados

```
live_desk.py   ← D150-01 a D150-06 (todos en un archivo)
tests/test_nodo150_live_risk_intelligence.py  ← 10 tests nuevos
```

---

## 7. Pre-registro de Hipótesis

**H150-01:** Filtro `cuota_envenenada` (cuota_drift > +15%) reduce pérdidas en combos ITF live.
- Gate: n=20 combos filtrados vs no-filtrados
- Breakeven: reducción de pérdidas en ≥50% de los casos filtrados

**H150-02:** `SET1_TIEBREAK_GATE` (games_set1 ≥ 12 → COINFLIP_FORZADO) mejora calibración certeza_condicional.
- Gate: n=30 partidos con tiebreak set 1
- Métrica: hit% UNDER cuando games_set1 ≥ 12 vs < 12

---

*Generado por sesión de análisis doctoral 2026-07-28. Root cause: simulación 3 piernas live.*
*Relacionado: [[Nodo-147]] (certeza condicional) [[Nodo-133]] (ITF live scan) [[Nodo-142]] (T0 freeze)*

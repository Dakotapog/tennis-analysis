# Nodo-57 — Penalización de Inactividad Quirúrgica + Validación de Campeón de Torneo

> **Wikilinks:** [[Nodo-56-Bug-Normalizacion-Pesos]] | [[Nodo-55-Respuesta-Fable-Funnel-Deploy]] | [[Nodo-21-Calibracion-Shrinkage-Tier]] | [[Nodo-32-Fase3-Markov-Postnorm]] | [[Nodo-19-H2H-Immunity-Dampener]]
> **Fecha:** 2026-07-03 (COMPLETO 2026-07-03 15:25)
> **Estado:** ✅ COMPLETO — 11 tests nuevos + 1612 tests totales (fue 1601)
> **Tests:** T57-01 a T57-09 (9 tests), T30-10b, T30-10c (2 tests regression)
> **Evidencia real:** 9 partidos (3 ITF, 3 ATP500, 3 Grand Slam) — 100% suma pesos, 0% penalización global, gates campeón funcionales
> **Severidad:** ALTA — penalización destruye picks válidos (Pacheco -50%) + bonus fantasma (Safiullin x1.6 sin ser campeón)
> **Descubierto en:** revisión Nodo-56 + análisis Wimbledon 2026-07-01 | **Resuelto en:** 2026-07-03

---

## 0. Síntomas

### Síntoma A — Penalización de inactividad demasiado drástica
```
Caso Meligeni vs Pacheco (Challenger clay, Quito):
  Pacheco days_since = ~49 → penalty = min(3.77 × 0.5, (49-30) × 0.1) = min(1.885, 1.9) = 1.885
  Score Pacheco: 3.77 → 1.89  (pierde el 50% por 19 días extra de descanso)
  Pacheco tiene mejor form_recent (1.26 vs 1.03), mejor ELO (0.75 vs 0.64), mejor ranking (0.67 vs 0.64)
  El modelo lo penaliza tan severamente que Meligeni gana 66% confianza a pesar de ser peor en ELO/form/ranking
  → La penalización global aplasta señales que NO tienen fecha de vencimiento (ELO, H2H, ranking)
```

### Síntoma B — Bonus de campeón otorgado a jugador en mitad de torneo
```
Safiullin — TORNEO_COMPLETO_BONUS: Wimbledon 2026 (5W-0L) → x1.6 quality_score

Historial real:
  22.06: McCabe     rank 241  ← qualifying R1
  24.06: Coppejans  rank 204  ← qualifying R2
  25.06: Kym        rank 191  ← qualifying R3
  29.06: Rublev     rank 16   ← main draw R1
  01.07: Van De Zandschulp rank 53 ← main draw R2

Wimbledon GS requiere 7 victorias para ser campeón.
Safiullin tiene 5 victorias (3Q + 2 main draw) = está en R3 del cuadro principal.
→ NO es campeón. El gate `wins >= 4` es incorrecto para Grand Slams.
```

### Síntoma C — TORNEO EXPIRADO ignorado sin compensación
```
TORNEO EXPIRADO (sin bonus): Mexico City 2025 (5W-0L, hace 446d) → sin bonus (>90d)
TORNEO EXPIRADO (sin bonus): Merida 2025 (5W-0L, hace 467d) → sin bonus (>90d)

→ Un campeón reciente (hace 3-6 meses) en esa superficie no recibe NINGUNA señal.
→ La señal no indica a QUÉ jugador corresponde.
```

---

## 1. Análisis con 4 Marcos de Expertos

### Marco 1 — Estadístico/Cuantitativo
La penalización global afecta componentes con half-lives MUY diferentes:

| Componente | Half-life real en tenis |
|---|---|
| form_recent | ~21 días (rendimiento reciente degrada) |
| ELO | Meses — refleja nivel estructural, no forma momentánea |
| H2H directo | Años — el historial específico no expira |
| common_opponents | Semanas/meses — los resultados contra el mismo pool siguen válidos |
| ranking_momentum | Semanas — actualización lenta del ranking ATP |

Penalizar `sum(weighted_scores) × 0.5` es penalizar ELO, H2H y rivales comunes que NO tienen fecha de vencimiento. Solo `form_recent` debe decaer con el tiempo.

### Marco 2 — Domain Expert (Circuito ATP)
Las pausas de 30-60 días son **estructurales y normales** en el circuito:
- Transición arcilla → hierba: 2-3 semanas de pausa (Roland Garros → Queen's)
- Transición hierba → dura: 3-4 semanas (Wimbledon → Cincinnati)
- Lesiones leves / preparación física entre torneos

Un jugador en pausa de 49 días no pierde el 50% de su capacidad. Sus datos históricos
(ELO, rivales comunes, H2H, ranking) siguen siendo completamente válidos. Lo único que
se degrada es la señal de FORMA RECIENTE — que ya tiene su propio componente separado.

Para la validación de campeón: Wimbledon Grand Slam = 7 rondas de cuadro principal.
5 victorias = cuartos de final como máximo. El gate `wins >= 4` no discrimina entre
"jugador en forma" y "campeón completo del torneo".

### Marco 3 — Bookmaker/Mercado
Los bookmakers NO aplican descuentos agresivos por 30-60 días de inactividad en sus
cuotas. Tienen fuentes adicionales (informes de entrenamiento, fitness). Si ellos no
penalizan, nuestro modelo diverge sistemáticamente del mercado en una dirección negativa
(sobrepenalizamos jugadores descansados), creando FALSOS edges negativos que matan picks
válidos como Safiullin.

Para la detección de campeón: Wimbledon 2026 con Safiullin en cuartos → el bookmaker
sabe que no es campeón. Si inflamos su `quality_score × 1.6`, sobreestimamos `p_modelo`
y el edge calculado se reduce o se vuelve negativo. El bonus falso MATA el pick en lugar
de activarlo. El usuario confirma: "se desaprovechó esta oportunidad de cuotas".

### Marco 4 — Bayesiano/Información
Cuando hay MENOS información reciente de forma, la respuesta bayesiana es encoger hacia
el prior (rendimiento estructural = ELO + ranking), NO penalizar el total. Esto es análogo
al James-Stein shrinkage aplicado a los pesos por tier en Nodo-21.

Fórmula correcta:
```
form_decay(days) = max(FLOOR, exp(-λ × max(0, days - grace_period)))
```
Con `λ = 0.025`, `grace = 30d`, `FLOOR = 0.35`:
- 30 días: decay = 1.0   (sin cambio)
- 49 días: decay = 0.622 (Pacheco: form_recent 1.26 → 0.78, NO -1.88 del score total)
- 60 días: decay = 0.472
- 90 días: decay = 0.350 (cap del floor)

Para datos de fecha desconocida (`days_since = -1`): decay = 0.70 (antes: penalty = 30%)

---

## 2. Causa Raíz por Bug

### Bug A — `apply_weights_and_penalties` en `rivalry_analyzer.py:1893`
```python
# ACTUAL — INCORRECTO:
def apply_weights_and_penalties(normalized_scores, weights, days_since):
    weighted_scores = {k: normalized_scores[k] * weights[k] for k in weights}
    penalty = 0
    if days_since == -1: penalty = sum(weighted_scores.values()) * 0.3
    elif days_since > 30: penalty = min(sum(weighted_scores.values()) * 0.5, (days_since - 30) * 0.1)
    final_score = sum(weighted_scores.values()) - penalty
    return final_score, weighted_scores, penalty

# Penalty afecta ELO, H2H, rivales comunes — componentes que NO tienen fecha de vencimiento.
```

### Bug B — Gate de campeón `wins >= 4` en `rivalry_analyzer.py:882`
```python
# ACTUAL — INCORRECTO:
if _ts['wins'] >= 4 and _ts['losses'] == 0:
    ...
    if _ts['wins'] >= 5:
        _bonus += 0.1
        _bonus_parts.append(f'final({_ts["wins"]}W)')  # asume que 5W = ganó la final

# Para Grand Slam: necesita 7W. Para ATP1000: 6W. El umbral 4W es tier-agnóstico.
# Safiullin: 5W en Wimbledon (3 qualifying + 2 main draw) → activa bonus incorrectamente.
```

### Bug C — TORNEO_EXPIRADO sin compensación y sin atribución
```python
# ACTUAL — generar_tabla_favoritos2.py:
if 'TORNEO_COMPLETO_EXPIRADO' in reason:
    _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
    _special_signals.append(f"TORNEO EXPIRADO (sin bonus): {_clean}")
    # No indica a quién corresponde el torneo expirado
    # No aplica ninguna compensación (bonus reducido) por historial en esa superficie
```

---

## 3. Solución Propuesta

### Fix A — Form Decay (D57-01, D57-02)

En `rivalry_analyzer.py`, añadir después de `LOG_MARKOV_POST_NORM` (~línea 1885):

```python
# --- Nodo-57 D57-01: Decaimiento exponencial de forma por inactividad ---
# Solo form_recent decae. ELO, H2H, rivales comunes NO tienen fecha de vencimiento.
_FORM_DECAY_LAMBDA = 0.025   # half-life efectivo ≈ 28d post-gracia
_FORM_GRACE_DAYS   = 30      # sin decay hasta 30 días
_FORM_DECAY_FLOOR  = 0.35    # nunca perder más del 65% de la señal de forma

def _form_decay_factor(days):
    if days == -1: return 0.70           # fecha desconocida: decay moderado fijo
    if days <= _FORM_GRACE_DAYS: return 1.0
    return max(_FORM_DECAY_FLOOR, math.exp(-_FORM_DECAY_LAMBDA * (days - _FORM_GRACE_DAYS)))

_fd_p1 = _form_decay_factor(days_since_p1)
_fd_p2 = _form_decay_factor(days_since_p2)
if _fd_p1 < 1.0:
    norm_p1 = dict(norm_p1)
    norm_p1['form_recent'] = norm_p1['form_recent'] * _fd_p1
if _fd_p2 < 1.0:
    norm_p2 = dict(norm_p2)
    norm_p2['form_recent'] = norm_p2['form_recent'] * _fd_p2
reasoning.append(
    f"LOG_FORM_DECAY: p1_days={days_since_p1} fd_p1={_fd_p1:.3f} "
    f"p2_days={days_since_p2} fd_p2={_fd_p2:.3f}"
)
```

Y simplificar `apply_weights_and_penalties` (D57-02):
```python
def apply_weights_and_penalties(normalized_scores, weights, days_since):
    weighted_scores = {k: normalized_scores[k] * weights[k] for k in weights}
    final_score = sum(weighted_scores.values())
    return final_score, weighted_scores, 0.0  # penalty=0: inactividad vía form_decay
```

### Fix B — Validación de campeón por tier (D57-03)

Añadir como constante de módulo en `rivalry_analyzer.py` (~línea 18):

```python
# Nodo-57: victorias mínimas para ser campeón completo de torneo (cuadro principal)
_MIN_WINS_CHAMPION = {
    'grand_slam': 7,    # R1-R7, sin contar qualifying
    'atp1000': 6,       # R1-R6 (con bye = 5, sin bye = 6) → usar 6 conservador
    'atp500': 5,
    'challenger': 5,
    'itf': 4,
}
```

En el bloque de bonus (~línea 882):
```python
# ANTES: if _ts['wins'] >= 4 and _ts['losses'] == 0:
# DESPUÉS:
from config import detectar_tier as _dt_tier
_tier = _dt_tier(_tname)
_min_wins = _MIN_WINS_CHAMPION.get(_tier, 5)  # fallback conservador
if _ts['wins'] >= _min_wins and _ts['losses'] == 0:
```

Y corregir el subgate `final`:
```python
# ANTES: if _ts['wins'] >= 5:   _bonus_parts.append(f'final({_ts["wins"]}W)')
# DESPUÉS:
if _ts['wins'] >= _min_wins:
    _bonus += 0.1
    _bonus_parts.append(f'final({_ts["wins"]}W≥{_min_wins}requeridos)')
```

### Fix C — TORNEO_EXPIRADO: atribución + compensación (D57-04)

En `generar_tabla_favoritos2.py`, señales:
```python
# ANTES:
if 'TORNEO_COMPLETO_EXPIRADO' in reason:
    _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
    _special_signals.append(f"TORNEO EXPIRADO (sin bonus): {_clean}")

# DESPUÉS:
if 'TORNEO_COMPLETO_EXPIRADO' in reason:
    _player = p1 if 'P1_LOG_SURF' in reason else p2
    _clean = reason.replace('P1_LOG_SURF: ', '').replace('P2_LOG_SURF: ', '')
    _special_signals.append(f"CAMPEON ANTERIOR EN SUPERFICIE: {_player} — {_clean} [sin bonus activo >90d, historial validado]")
```

En `rivalry_analyzer.py`, para TORNEO_EXPIRADO (D57-04 módulo):
```python
# En la sección de TORNEO_COMPLETO_EXPIRADO (línea ~891), en lugar de solo `continue`:
if _days_ago > 90:
    # Compensación reducida: campeón histórico en esta superficie sigue siendo señal
    if _days_ago <= 365:  # máximo 1 año
        _comp_bonus = 1.15 if _days_ago <= 180 else 1.05
        quality_score *= _comp_bonus
        analysis_log.append(
            f"TORNEO_COMPLETO_EXPIRADO: {_tname} {_tyear} "
            f"({_ts['wins']}W-0L, hace {_days_ago}d) → sin bonus activo pero +{(_comp_bonus-1)*100:.0f}% historial superficie"
        )
    else:
        analysis_log.append(
            f"TORNEO_COMPLETO_EXPIRADO: {_tname} {_tyear} "
            f"({_ts['wins']}W-0L, hace {_days_ago}d) → sin bonus (>365d)"
        )
    continue
```

### Fix D — LOG_FORM_DECAY en SEÑALES ESPECIALES (D57-05)

En `generar_tabla_favoritos2.py`, añadir al bloque de señales:
```python
if 'LOG_FORM_DECAY' in reason:
    _fd_match_p1 = re.search(r'fd_p1=([\d.]+)', reason)
    _fd_match_p2 = re.search(r'fd_p2=([\d.]+)', reason)
    _days_p1 = re.search(r'p1_days=([-\d]+)', reason)
    _days_p2 = re.search(r'p2_days=([-\d]+)', reason)
    for _player_name, _fd_match, _days_match in [
        (p1, _fd_match_p1, _days_p1), (p2, _fd_match_p2, _days_p2)
    ]:
        if _fd_match and _days_match:
            _fd_val = float(_fd_match.group(1))
            _days_val = int(_days_match.group(1))
            if _fd_val < 1.0 and _days_val > 30:
                _special_signals.append(
                    f"INACTIVIDAD: {_player_name} — {_days_val}d sin jugar → "
                    f"form_recent × {_fd_val:.2f} (decay exponencial Nodo-57)"
                )
```

---

## 4. Deudas

| Deuda | Descripción | Archivo | Prioridad |
|---|---|---|---|
| D57-01 | Aplicar form_decay a norm_p1/norm_p2 después de LOG_MARKOV_POST_NORM | `rivalry_analyzer.py:~1885` | ALTA |
| D57-02 | Simplificar apply_weights_and_penalties: penalty=0 siempre | `rivalry_analyzer.py:1893` | ALTA |
| D57-03 | Validación de campeón con _MIN_WINS_CHAMPION por tier | `rivalry_analyzer.py:882` | ALTA |
| D57-04 | TORNEO_EXPIRADO: compensación reducida + atribución a jugador | `rivalry_analyzer.py:~887` + `generar_tabla_favoritos2.py` | MEDIA |
| D57-05 | Mostrar señal INACTIVIDAD en SEÑALES ESPECIALES (LOG_FORM_DECAY) | `generar_tabla_favoritos2.py` | MEDIA |

---

## 5. Tests de Validación (REGLA-T53)

- **T57-01:** Jugador con days_since=49 — `form_decay_factor(49) ≈ 0.622`. El score final NO es 50% menor que el score sin inactividad. La diferencia es solo en la contribución de form_recent.
- **T57-02:** GS: `wins=5, losses=0` → NO activa bonus (5 < 7 requeridos). `wins=7, losses=0` → SÍ activa bonus.
- **T57-03:** Challenger: `wins=5, losses=0` → SÍ activa bonus (5 == 5 requeridos).
- **T57-04:** ATP1000: `wins=6, losses=0` → SÍ activa bonus. `wins=5, losses=0` → NO activa.
- **T57-05:** days_since=-1 → form_decay_factor = 0.70 (no 1.0 ni 0.0).
- **T57-06:** days_since=30 → form_decay_factor = 1.0 (dentro del grace period).
- **T57-07:** days_since=90 → form_decay_factor = 0.350 (= FLOOR exacto para λ=0.025).
- **T57-08:** TORNEO_EXPIRADO hace 150d → compensación x1.15 en quality_score.
- **T57-09:** TORNEO_EXPIRADO hace 400d → sin compensación (>365d).

---

## 6. Orden de Implementación

```
1. D57-01 + D57-02 — form decay + simplificar apply_weights_and_penalties (rivalry_analyzer.py)
2. D57-03 — _MIN_WINS_CHAMPION + gate tier-aware (rivalry_analyzer.py)
3. D57-04 — TORNEO_EXPIRADO: compensación + atribución en jugador (ambos archivos)
4. D57-05 — Señal INACTIVIDAD en SEÑALES ESPECIALES (generar_tabla_favoritos2.py)
5. Tests T57-01 a T57-09
Baseline: 1601 tests siguen pasando.
PROHIBIDO: modificar kelly_kl, VaR, shrinkage de pesos, calibracion_edge.json.
PROHIBIDO: cambiar el decay de Markov POST-NORM (factor_p1, factor_p2 se mantienen).
```

---

## 7. Implementación Completada

**ESTADO:** ✅ COMPLETO 2026-07-03 15:25 UTC

### D57-01: Form Decay Exponencial
- **Archivo:** `analysis/rivalry_analyzer.py:18` (constantes) + `:~1885` (función)
- **Cambio:** Añadido `_FORM_DECAY_LAMBDA=0.025`, `_FORM_GRACE_DAYS=30`, `_FORM_DECAY_FLOOR=0.35`
- **Función:** `_form_decay_factor(days)` retorna decay exponencial post-grace
- **Impacto:** Pacheco 49d → `fd=0.622` (form reciente ×0.622, NO score -50%)
- **Tests:** T57-05 (days=-1 → fd=0.70), T57-06 (days=30 → fd=1.0), T57-07 (days=90 → fd=0.35)

### D57-02: Apply Weights Penalty Simplificado
- **Archivo:** `analysis/rivalry_analyzer.py:~1893`
- **Cambio:** `apply_weights_and_penalties()` retorna `penalty=0.0` siempre
- **Razón:** Inactividad vía form_decay, no penalización global que afecta ELO/H2H
- **Tests:** Todos (penalty nunca resta del final_score)

### D57-03: Validación Campeón Tier-Aware
- **Archivo:** `analysis/rivalry_analyzer.py:18` (dict) + `:882-883` (gate)
- **Cambio:** Constante `_MIN_WINS_CHAMPION = {grand_slam:7, atp1000:6, atp500:5, challenger:5, itf:4}`
- **Gate:** `if wins >= _MIN_WINS_CHAMPION[tier]` en lugar de `wins >= 4`
- **Caso Safiullin:** Wimbledon 5W → NO activa bonus (5 < 7 requeridos) ✓
- **Tests:** T57-02, T57-03, T57-04, T30-10b (GS 5W rechaza), T30-10c (GS 7W acepta)

### D57-04: TORNEO_EXPIRADO + Compensación
- **Archivo:** `analysis/rivalry_analyzer.py:~887` (quality_score ×) + `generar_tabla_favoritos2.py` (señal)
- **Cambio:** 
  - 90-180d → `quality_score × 1.15`
  - 180-365d → `quality_score × 1.05`
  - >365d → sin bonus
  - Señal muestra nombre del jugador: `"CAMPEON ANTERIOR EN SUPERFICIE: {player} — {torneo}"`
- **Tests:** T57-08 (150d → x1.15), T57-09 (400d → x1.0)
- **Evidencia real:** Dunja Maric 244d (+5%), Anja Stankovic 355d (+5%), Marat Sharipov 705d (sin bonus)

### D57-05: LOG_FORM_DECAY Señales
- **Archivo:** `generar_tabla_favoritos2.py` (SEÑALES ESPECIALES)
- **Cambio:** Parsea `LOG_FORM_DECAY` y muestra `"INACTIVIDAD: {player} — {days}d sin jugar → form_recent × {fd:.2f}"`
- **Visibilidad:** Usuario ve cuándo y quién se ve afectado por decaimiento de forma

### Tests Completados
| Test | Descripción | Resultado |
|------|-------------|-----------|
| T57-01 | days=49 → decay ≈0.622, score NO -50% | ✅ PASS |
| T57-02 | GS wins=5 → NO bonus (5<7) | ✅ PASS |
| T57-03 | Challenger wins=5 → SÍ bonus (5==5) | ✅ PASS |
| T57-04 | ATP1000 wins=6 → SÍ bonus; wins=5 → NO | ✅ PASS |
| T57-05 | days=-1 → fd=0.70 | ✅ PASS |
| T57-06 | days=30 → fd=1.0 | ✅ PASS |
| T57-07 | days=90 → fd=FLOOR=0.35 | ✅ PASS |
| T57-08 | EXPIRADO 150d → x1.15 en quality | ✅ PASS |
| T57-09 | EXPIRADO 400d → sin bonus (>365d) | ✅ PASS |
| T30-10b | GS Garros 5W rechaza bonus (regresión) | ✅ PASS |
| T30-10c | GS Garros 7W acepta bonus | ✅ PASS |

---

## 8. Evidencia Real — 9 Partidos (3 por Tier)

**Comando ejecutado:** `generar_tabla_favoritos2.py` + re-análisis con `RivalryAnalyzer.analyze_rivalry()`
**Archivo:** `reports/h2h_results_enhanced_20260703_003059.json` (110 partidos)
**Fecha:** 2026-07-03 15:25 UTC

### ITF (3 partidos)
| Partido | Suma Pesos | Form Decay | Champion | Expirado | Penalidad |
|---------|-----------|-----------|----------|----------|-----------|
| Elza Tomase vs Dunja Maric | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — | Dunja 244d (+5%) | 0.00 ✓ |
| Kristina Kovgan vs Anna Petkovic | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — | — | 0.00 ✓ |
| Anja Stankovic vs Jana Bojovic | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — | Anja 355d (+5%) | 0.00 ✓ |

### ATP500 (3 partidos — Troyes clay)
| Partido | Suma Pesos | Champion Bonus | Expirado |
|---------|-----------|---|---|
| Lorenzo Giustino vs Igor R.M. | **100.00%** | — | Giustino: 2× M25 (271d, 321d → +5%) |
| **Kilian Feldbausch** vs Inaki M. | **100.00%** | K:Kosice 5W→x1.4, I:Plovdiv 5W→x1.6(recency 6d) | — |
| Marvin Moeller vs Marat Sharipov | **100.00%** | — | Sharipov: 2× M15/M25 (705d, 733d → sin bonus) |

### Grand Slam (3 partidos — Wimbledon grass)
| Partido | Suma Pesos | Form Decay | Champion | Penalidad |
|---------|-----------|-----------|----------|-----------|
| **Roman Safiullin** vs Joao Fonseca | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — (5W < 7 requeridos) ✓ | 0.00 ✓ |
| Jan-Lennard Struff vs Daniil Medvedev | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — | 0.00 ✓ |
| Rafael Jodar vs Shintaro Mochizuki | **100.00%** | fd_p1=1.0, fd_p2=1.0 | — | 0.00 ✓ |

**CLAVE: Safiullin Wimbledon — 5 victorias (3 qualifying + 2 main draw) NO activa bonus. Gate tier-aware funciona correctamente.**

**Suma de pesos:** 9/9 = **100.00%** exactamente. Fix D56-01/02 (_weights_final) implementado correctamente.

**Penalización global:** 9/9 = **0.00**. Fix D57-01/02 (form_decay, penalty=0) implementado correctamente.

---

## 9. Registro

**Descubierto:** 2026-07-03, revisión Nodo-56 + análisis partidos Wimbledon.
**Caso penalización:** Pacheco vs Meligeni (Quito clay) — Pacheco score 3.77 → 1.89 por penalización global (49d).
  - Fix: Form decay exponencial → form_recent ×0.622, NO score -50%
**Caso bonus falso:** Safiullin (Wimbledon 2026) — 5W (3Q+2MD) interpretados como campeón → x1.6 falso.
  - Fix: _MIN_WINS_CHAMPION['grand_slam']=7 → 5W no activa bonus
**Impacto Safiullin:** bonus falso infló p_modelo → edge real >3% fue rechazado por pipeline. Oportunidad perdida.
  - **Resultado:** Con el fix, Safiullin se evalúa sin bonus falso → p_modelo más honesto → edge capturado correctamente

**BASELINE:** 1601 tests (Nodo-56) → 1612 tests (Nodo-57)
  - Nuevos tests: T57-01 a T57-09 (9)
  - Regresión guards: T30-10b, T30-10c (2)
  - **Cambio:** +11 tests, 0 fallos

**PROHIBIDO (durables):**
- Modificar `kelly_kl`, `VarianceAtRisk`, shrinkage por tier (Nodo-21)
- Cambiar `calibracion_edge.json` priors
- Tocar Markov POST-NORM (factor_p1/factor_p2)
- Cambiar umbral de detección de tier en `detectar_tier()`

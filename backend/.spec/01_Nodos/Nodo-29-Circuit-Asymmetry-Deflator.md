# Nodo-29 --- Circuit Asymmetry Deflator (CAD)

> **Estado:** IMPLEMENTADO (Fase 1-4) + VALIDADO --- 2026-06-19 impl / 2026-06-28 backtest | Tests: 1050→1113 passed
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-28-Conditional-Decomposition-Metamodel]] | [[Nodo-24-Bookmaker-Blindness-Scoring]]
> **Origen:** Backtest limpio 18-jun-2026 (42/52 inflado por data leakage → 36/52=69.2% real). Post-mortem Fallo #1: Schoen (ELO 1843, SoS=0, solo M15) vs Boogaard (ELO 1642, SoS=86, jugó Wimbledon/RG/Rotterdam, le sacó set a Medvedev #8, le ganó a Wu #98). Modelo predijo Schoen 51.1% → ganó Boogaard. La bookmaker los veía 50/50 (1.81 vs 1.89) porque SABE que Boogaard bajó de circuito.
> **Prioridad:** CRITICA --- afecta 4 de 8 componentes del modelo en todo partido donde hay asimetría de circuito competitivo. Alpha estructural: cuotas altas en estos escenarios.

---

## Problema

### La ceguera al circuito competitivo

El modelo trata las victorias como iguales independientemente del circuito donde ocurrieron. Un 10W/0L contra jugadores #500-2000 en M15 produce **más puntos** que un 5W/5L contra jugadores #8-200 en ATP/Grand Slam.

**Caso demostrado — Schoen vs Boogaard (M25 Lourinha, 18-jun-2026):**

```
SCHOEN (Ranking #182):                    BOOGAARD (Ranking #139):
  Mejor oponente: #283                      Mejor oponente: #8 (Medvedev)
  Oponentes top-100: 0                      Oponentes top-100: 3
  Oponentes top-200: 0                      Oponentes top-200: 10
  Ranking prom oponentes: #847              Ranking prom oponentes: #599
  Torneos: M15 Monastir, Kayseri, Sanxenxo  Torneos: Wimbledon, RG, Rotterdam, Miami
  Partidos ATP/GS: 0 de 49                  Partidos ATP/GS: 22 de 49
```

**Impacto en raw scores del modelo:**

| Componente | Schoen | Boogaard | Sesgo | Causa raíz |
|---|---|---|---|---|
| form_recent | **258.0** | 92.5 | Schoen +165 | 10W/0L contra #800 prom. = 100% win% |
| elo_rating | **250** | 142 | Schoen +108 | K=48 (ITF) infla ELO con victorias fáciles |
| ranking_momentum | **92.67** | 86.61 | Schoen +6 | No usa ranking ATP oficial como ancla |
| strength_of_schedule | 0.0 | **86.06** | Boogaard +86 | Solo 5% peso, no compensa |

**Resultado:** Schoen gana 3 de 4 componentes por sesgo de circuito. Los 86 puntos de SoS de Boogaard (al 5% peso) no alcanzan para revertir 279 puntos de ventaja acumulada por Schoen en form+ELO+momentum.

### Diagnóstico de los 4 marcos mentales

**Marco 1 — Senior SW Engineer:**
El `strength_of_schedule` (líneas 273-329) solo asigna puntos positivos a quien enfrentó top-200, pero no penaliza a quien nunca lo hizo. Además calcula SoS en aislamiento (no relativo). Un SoS=0 vs SoS=86 debería ser señal de alarma, pero el modelo lo trata como "dato faltante".

**Marco 2 — Data Architect:**
3 señales están contaminadas por circuito:
1. `form_recent`: `win_pct/100 × 200` — no deflacta por calidad de oponentes. 100% win vs #800 prom = 200 pts base; 50% win vs #200 prom = 100 pts. El de menor nivel obtiene el doble.
2. `elo_rating`: K=48 (ITF) amplifica cada victoria en +30 ELO, pero contra oponentes débiles. El ceiling teórico permite ELO inflados artificialmente en circuito bajo.
3. `ranking_momentum`: calcula base_score desde puntos ATP, pero el momentum/presión se calcula sin ancla al ranking oficial publicado.

**Marco 3 — Implementation Engineer:**
El riesgo es sistémico: cada vez que un jugador ATP "baja" a M15/M25 (wildcard, regreso de lesión, puntos para ranking), el modelo va a sobreponderar al local del circuito bajo. Estimación: esto afecta 5-15% de partidos ITF/Challenger donde hay asimetría de circuito.

**Marco 4 — Quant/Financial:**
Alpha estructural no explotado. Cuando el modelo detecta asimetría de circuito a favor del jugador que bajó, las cuotas suelen estar equilibradas (~2.00) porque el bookmaker lo reconoce pero el público apuesta por la form reciente del local. Si podemos cuantificar la ventaja real del "turista de circuito", tenemos edge en cuotas donde el público no lo ve.

---

## Solución: Circuit Asymmetry Deflator (CAD)

### Concepto

Calcular un **índice de asimetría de circuito** entre los dos jugadores y usarlo como:
1. **Deflactor** de form_recent y ELO del jugador de circuito inferior
2. **Amplificador** del peso de strength_of_schedule cuando la asimetría es alta
3. **Señal independiente** para el edge_calculator

### Fase 1 --- `circuit_tier_index()` (nueva función en rivalry_analyzer.py)

Calcula el tier promedio ponderado del historial de cada jugador:

```python
def circuit_tier_index(player_history):
    """
    Calcula el nivel de circuito promedio del jugador basado en su historial.
    
    Criterios (por partido en historial):
      - Opponent ranking ≤ 10:   tier_score = 5.0  (élite absoluta)
      - Opponent ranking ≤ 50:   tier_score = 4.0  (top ATP)
      - Opponent ranking ≤ 100:  tier_score = 3.0  (ATP consolidado)
      - Opponent ranking ≤ 200:  tier_score = 2.0  (ATP/Challenger alto)
      - Opponent ranking ≤ 500:  tier_score = 1.0  (Challenger/ITF alto)
      - Opponent ranking > 500:  tier_score = 0.0  (ITF bajo)
    
    Ponderación temporal: partidos recientes (últimos 10) pesan 2×
    
    Returns:
        float: 0.0 (solo ITF bajo) a 5.0 (solo top-10 opponents)
        int:   n_partidos_con_ranking (para confidence)
    """
```

**Ejemplo Schoen vs Boogaard:**
```
Schoen:   11 oponentes ≤500, 0 ≤200 → CTI ≈ 0.22
Boogaard: 20 oponentes ≤500, 10 ≤200, 3 ≤100, 1 ≤10 → CTI ≈ 1.47

Asimetría = CTI_Boogaard / CTI_Schoen = 6.7× 
```

### Fase 2 --- Deflactor aplicado a form_recent y ELO

Cuando `asimetría_circuito > 2.0` (un jugador opera en un circuito ≥2× superior al otro):

```python
# El jugador de circuito inferior recibe deflactor en form y ELO
# El jugador de circuito superior recibe bonificación en form y ELO

asimetria = max(CTI_p1, CTI_p2) / max(min(CTI_p1, CTI_p2), 0.1)

if asimetria > 2.0:
    # Deflactor logarítmico — suave pero significativo
    # asimetria=2 → deflactor=0.90 | asimetria=5 → deflactor=0.75 | asimetria=10 → deflactor=0.65
    deflactor = 1.0 / (1.0 + 0.15 * math.log(asimetria))
    
    # Aplicar al jugador de circuito INFERIOR
    raw_scores_inferior['form_recent'] *= deflactor
    raw_scores_inferior['elo_rating']  *= deflactor
    
    # Bonificación inversa al jugador de circuito SUPERIOR (más suave)
    bonificacion = 1.0 + (1.0 - deflactor) * 0.5
    raw_scores_superior['form_recent'] *= bonificacion
    raw_scores_superior['elo_rating']  *= bonificacion
```

**Efecto en Schoen vs Boogaard (asimetría ≈ 6.7):**
```
deflactor = 1/(1 + 0.15 × ln(6.7)) = 0.78
bonificacion = 1 + (1-0.78) × 0.5 = 1.11

Schoen form_recent:   258.0 × 0.78 = 201.2  (era 258.0)
Schoen elo_rating:    250   × 0.78 = 195.0  (era 250)
Boogaard form_recent: 92.5  × 1.11 = 102.7  (era 92.5)
Boogaard elo_rating:  142   × 1.11 = 157.6  (era 142)

Delta form: +165 → +98  (reducción 41%)
Delta ELO:  +108 → +37  (reducción 66%)
```

### Fase 3 --- SoS dinámico (peso adaptativo)

Cuando hay asimetría de circuito, el peso de `strength_of_schedule` escala dinámicamente:

```python
# Peso base SoS por tier
base_sos_weight = weights['strength_of_schedule']  # 0.05 para ITF

# Cuando asimetría > 2.0, escalar peso SoS
if asimetria > 2.0:
    # Escala: asimetria=2→peso×2 | asimetria=5→peso×3 | asimetria=10→peso×4
    sos_multiplier = 1.0 + math.log(asimetria)
    extra_weight = base_sos_weight * (sos_multiplier - 1.0)
    
    # Tomar peso extra de form_recent (que está inflado por circuito bajo)
    weights['strength_of_schedule'] += extra_weight
    weights['form_recent'] -= extra_weight
```

**Efecto en Schoen vs Boogaard:**
```
sos_multiplier = 1 + ln(6.7) = 2.90
extra_weight = 0.05 × 1.90 = 0.095
SoS weight: 0.05 → 0.145 (casi 3×)
form_recent weight: 0.28 → 0.185 (reducción proporcional)
```

### Fase 4 --- Señal para edge_calculator

Nuevo campo en el output de `analyze_rivalry()`:

```python
'circuit_asymmetry': {
    'p1_circuit_tier_index': float,
    'p2_circuit_tier_index': float,
    'asymmetry_ratio': float,        # max/min de los CTI
    'deflactor_applied': float,      # 1.0 si no aplica
    'player_deflated': str or None,  # nombre del jugador deflactado
    'signal': 'STRONG_ASYMMETRY' | 'MODERATE_ASYMMETRY' | 'SYMMETRIC'
}
```

`edge_calculator.py` puede usar `signal == 'STRONG_ASYMMETRY'` para:
- Subir el edge mínimo requerido cuando se apuesta por el jugador de circuito inferior
- Bajar el edge mínimo cuando se apuesta por el jugador de circuito superior (alpha)

---

## Reglas de Implementación

- **REGLA-N29-1:** El deflactor SOLO aplica cuando `asimetría > 2.0`. Partidos simétricos (dos jugadores del mismo circuito) no se tocan.
- **REGLA-N29-2:** El deflactor es multiplicativo, no aditivo. No cambia la lógica existente de cálculo de raw scores — solo ajusta el resultado final.
- **REGLA-N29-3:** El CTI usa ponderación temporal (últimos 10 partidos pesan 2×). Un jugador que solía jugar ATP pero lleva 6 meses en ITF tiene CTI más bajo (transición real, no turista).
- **REGLA-N29-4:** Si `n_partidos_con_ranking < 10` para alguno, no aplicar deflactor (muestra insuficiente). Logging como `LOG_CAD_SKIP`.
- **REGLA-N29-5:** El deflactor se aplica DESPUES del shrinkage y density adjustment, ANTES de la normalización final. Posición en el pipeline: después de línea 1362 (SoS raw score) y antes de la normalización.

---

## Punto de intervención en código

**Archivo:** `analysis/rivalry_analyzer.py`

1. **Nueva función** `circuit_tier_index(player_history)` — después de `analyze_strength_of_schedule()` (línea 329)
2. **Modificación** `generate_advanced_prediction()` — después de calcular raw_scores (línea 1362), antes de normalización:
   - Calcular CTI para ambos jugadores
   - Si asimetría > 2.0: aplicar deflactor a form_recent y elo_rating del inferior
   - Si asimetría > 2.0: reponderar SoS weight
   - Log: `LOG_CAD: CTI_P1=X CTI_P2=Y asimetria=Z deflactor=W jugador_deflactado=NAME`
3. **Output** — agregar `circuit_asymmetry` dict al return de `analyze_rivalry()`

**Archivo:** `normalization.py` — sin cambios (los MAX_RAW_SCORES no cambian, el deflactor opera dentro de los mismos límites)

**Archivo:** `edge_calculator.py` — Fase 4, lectura de `circuit_asymmetry.signal`

---

## Fase 4 --- Integración en edge_calculator (IMPLEMENTADO 2026-06-19)

**Archivo:** `edge_calculator.py`

Lee `circuit_asymmetry` del output de `rivalry_analyzer.py` y añade 3 campos informativos a cada pick:

```python
_circuit = pred.get('circuit_asymmetry') or {}
_circuit_signal = _circuit.get('signal', 'SYMMETRIC')
_circuit_warning = False

if _circuit_signal in ('MODERATE_ASYMMETRY', 'STRONG_ASYMMETRY'):
    _deflated = _circuit.get('player_deflated')
    if _deflated and _deflated == favored:
        _circuit_warning = True

resultado.update({
    'circuit_asymmetry_signal': _circuit_signal,
    'circuit_asymmetry_ratio': round(float(_circuit.get('asymmetry_ratio', 1.0)), 3),
    'circuit_warning': _circuit_warning,
})
```

### REGLA-N29-EDGE-1: circuit_warning es señal informativa

`circuit_warning = True` significa: el modelo favorece al jugador de circuito **inferior** y la asimetría es MODERATE o STRONG. No modifica edge ni Kelly — es una alerta para revisión humana antes de apostar.

Defaults seguros cuando `circuit_asymmetry` no existe en el pick: `signal='SYMMETRIC'`, `ratio=1.0`, `circuit_warning=False`.

### Conexión con CONTESTED_ALPHA (Nodo-28 Fase 2)

Un pick con `circuit_warning=True` probablemente también tenga `alignment_flag='CONTESTED_ALPHA'` o `n_axes_active < 2`, ya que el rival de circuito superior tiene ventaja informacional real que el modelo puede no capturar completamente. Ver `REGLA-N28F2-1` y `REGLA-N28F2-2` en Nodo-28.

### Nuevos campos en edge_report por pick (Fase 4)

| Campo | Tipo | Descripción |
|---|---|---|
| `circuit_asymmetry_signal` | str | SYMMETRIC \| MODERATE_ASYMMETRY \| STRONG_ASYMMETRY |
| `circuit_asymmetry_ratio` | float | max(CTI_p1, CTI_p2) / max(min(...), 0.1) |
| `circuit_warning` | bool | True si favorito == player_deflated Y señal MODERATE/STRONG |

Tests: `tests/test_nodo29_integration.py` — TF5-01 a TF5-08 (18 tests).

---

## Tests requeridos

```
T29-01: CTI = 0 si historial vacío
T29-02: CTI = 0 si todos los oponentes > 500
T29-03: CTI > 3.0 si jugador tiene 5+ oponentes top-50
T29-04: Deflactor = 1.0 (no aplica) si asimetría < 2.0
T29-05: Deflactor reduce form_recent del jugador inferior
T29-06: Bonificación amplifica form_recent del jugador superior
T29-07: SoS weight sube cuando asimetría > 2.0
T29-08: form_recent weight baja proporcionalmente
T29-09: No aplica si n_partidos_con_ranking < 10
T29-10: Backtest Schoen vs Boogaard → Boogaard debe ser favorito
T29-11: No afecta partidos simétricos (dos jugadores mismo circuito)
T29-12: circuit_asymmetry dict presente en output
T29-13: Ponderación temporal: partidos recientes pesan 2×
```

---

## Validación post-implementación — 2026-06-28

### Backtest (backtest_nodo28_limpio.py, datos 18-jun-2026, 52 partidos)

```
ACCURACY LIMPIO con CAD activo:  38/52 = 73.1%  →  14 fallos
Meta spec original:              ≥40/52 = 76.9%  →  ≤12 fallos
Baseline spec (antes de CAD):   36/52 = 69.2%  →  16 fallos
```

**Meta ≤12 fallos: NO alcanzada (14 fallos reales).**

Los 14 fallos restantes no tienen perfil de asimetría de circuito:
- 6 son upsets genuinos con confianza baja (50-54%) — modelo correcto en dirección, resultado aleatorio
- 6 son casos donde el data leak del 18-jun era la única señal que hacía acertar (OK→FAIL al limpiar)
- 2 son upsets de alta confianza (66-77%) que el modelo falló de forma independiente a circuito

Ninguno de los 14 fallos corresponde al patrón "turista de circuito". La meta original no contemplaba
este desglose: 4 de los 16 fallos baseline eran de circuito, el resto eran de otra naturaleza. CAD
redujo esos 4 (de 16 a 12 de tipo-circuito), pero 2 nuevos fallos entraron por limpieza del leak,
resultando en 14 netos. Los parámetros de CAD no se ajustan para perseguir el número.

### Caso fundacional — Schoen vs Boogaard

```
CTI Patrick Schoen:  0.19  (n_ranking=48)  ← circuito inferior
CTI Thijs Boogaard:  0.804 (n_ranking=46)  ← circuito superior
Asimetría:           4.23x → MODERATE_ASYMMETRY (umbral >2.0 cumplido)
Deflactor a Schoen:  0.8221 (form_recent y elo_rating × 0.82)
Bonificación Boogaard: × 1.09

Favorito predicho con CAD activo: Thijs Boogaard (55.3%)
Ganador real:                     Thijs Boogaard ← CORRECTO
```

Spec describía asimetría 6.7x (CTI: 0.22/1.47) con rankings del 18-jun. Hoy mide 4.23x con
rankings del 28-jun — misma dirección, valores diferentes por archivos de ranking distintos.

### Propagación de circuit_asymmetry en producción

El dict `circuit_asymmetry` está en `result['prediction']['circuit_asymmetry']` (NO en la raíz
de `analyze_rivalry()`). `edge_calculator.py` lee correctamente via `pred = ra['prediction']` →
`pred.get('circuit_asymmetry')`. Confirmado con H2H del 28-jun-2026 (80 partidos):

```
Con asimetría != SYMMETRIC:  14 partidos
Con circuit_warning=True:     7 partidos (model favors inferior circuit player)
```

`circuit_warning` se activa en producción. No hay gap de propagación.

---

## Oportunidad P&L

Los partidos con "turista de circuito" (jugador ATP bajando a ITF/Challenger) tienen un perfil específico:
- Cuotas cercanas a 2.00 (bookmaker lo ve pero el público no)
- El modelo actual puede predecir MAL (a favor del local)
- Con CAD, el modelo debería predecir BIEN → edge real contra cuotas equilibradas
- **Alpha estructural: apostar por el turista cuando el CAD lo detecta y la cuota es ≥1.80**

Esto conecta directamente con el hallazgo del pipeline tracker: "Challenger/ITF: bookmaker tiene menos datos → mayor ventaja informacional". El CAD es una de las formas concretas de explotar esa ventaja.

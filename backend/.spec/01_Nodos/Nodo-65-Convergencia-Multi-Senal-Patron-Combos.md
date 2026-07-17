# Nodo-65 — Convergencia Multi-Señal: Anatomía de una Sesión de Alta Precisión

> **Wikilinks:** ~~~~[[Nodo-38-Combo-Confianza]]~~ _(MISSING — [[Nodo-38-Portfolio-Aislamiento-Riesgo]] es diferente)_~~ _(MISSING — [[Nodo-38-Portfolio-Aislamiento-Riesgo]] cubre concepto diferente)_ | [[Nodo-44-Watchlist-Alpha-Signal]] | [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | [[Nodo-63-Anchor-Combo-Builder]] | [[Nodo-64-RFI-Return-From-Inactivity]]
> **Fecha descubrimiento:** 2026-07-10
> **Estado:** IMPLEMENTADO — todas las deudas D65-01→D65-07 resueltas (2026-07-10)
> **Severidad:** ALTA — identifica dos poblaciones de picks cualitativamente distintas que el sistema actual trata como homogéneas
> **Sesión de referencia:** betslip_index_20260710_015309.json | combo_plan_20260710_015306.txt | analisis_partidos_20260710_012116.txt
> **Resultado:** 6/7 combos GANADOS (85.7%) | 9/10 picks individuales correctos | única pérdida: Obradovic @1.18

---

## 0. Descubrimiento

### Sesión semilla — 2026-07-10 madrugada

```
Pool de candidatos:  41 picks (conf ≥ 53%)
Seleccionados:       10 picks → 7 combos (CORE + 6 COBERTURA EXPANDIDA)
Estructura:          3 anchors fijos (Wiskandt + Milosavljevic + Kostovic) + 1 variable por combo
Invertido:           $15,500  (budget $15,000)
Resultado combos:    6/7 GANADOS
Resultado picks:     9/10 correctos — único error: Dusan Obradovic @1.18
```

El análisis post-sesión reveló que los 10 picks NO son cualitativamente homogéneos. Existen dos poblaciones con estructuras de señal radicalmente distintas, y el sistema actual no las separa explícitamente en el output.

---

## 1. Anatomía del Pool — Dos Poblaciones Distintas

### 1.1 Picks ANCHOR — Cuotas 1.56–2.12 (positive-edge, alpha estructural)

| Jugador | Cuota | Edge | p_modelo | p_impl | alpha | n20 % | ELO | SCALP | RFI rival | CAMPEON tier |
|---|---|---|---|---|---|---|---|---|---|---|
| Alexander Weis | 2.12 | +12.2% | 59.4% | 47.2% | +34 | 85% (17/20) Exc. | 2161 | x30+ TOP-100 clay | — | RG 2026 ATP500 |
| Max Wiskandt | 1.92 | +33.9% | 86.0% | 52.1% | -2* | 80% (16/20) Exc. | 1751 | TOP-100 #48 clay | — | M25 Marburg 5d |
| Marko Milosavljevic | 1.91 | +17.6% | 70.0% | 52.4% | +13 | 33% (3/9) Mala† | 1488 | — | RFI-2 (261d) | — |
| Teodora Kostovic | 1.80 | +28.7% | 84.3% | 55.6% | -2* | 50% (10/20) Reg. | 1686 | TOP-100 x2 clay | — | Exp. 327d |
| Jan Choinski | 1.78 | +29.6% | 85.8% | 56.2% | -12* | 70% (14/20) Bue. | 1764 | TOP-50 #36 clay | — | Zagreb ATP500 |
| Leolia Jeanjean | 1.56 | +17.2% | 81.3% | 64.1% | -2* | 60% (12/20) Bue. | 1756 | TOP-50 x3 clay | — | — |

\* Penalización `phantom_data(-25)` por H2H sin historial del rival específico (BLOQUEADO).
  El pick se seleccionó porque SCALP + edge positivo compensaron la penalidad.
† Milosavljevic: forma propia mala (3/9 reciente) pero p_modelo = 70% porque el form decay
  del rival (261 días) eleva la probabilidad más allá de la forma propia del favorito.

**Característica estructural compartida:** `p_modelo > p_implicita` en TODOS.
El modelo ve una ineficiencia que el mercado no ha descontado. El edge positivo es confirmación
de que la señal de calidad no está ya en el precio.

### 1.2 Picks VARIABLE — Cuotas 1.18–1.27 (negative-edge, alpha de momentum)

| Jugador | Cuota | Edge | p_modelo | p_impl | alpha | n20 % | ELO | CAMPEON | HOT | Resultado |
|---|---|---|---|---|---|---|---|---|---|---|
| Vladyslav Orlov | 1.26 | -13%‡ | 66.4% | 79.4% | n/d | 65% (13/20) | 1692 | Exp. 306d | 70% | GANO |
| Anton Arzhankin | 1.27 | -13%‡ | 65.7% | 78.7% | n/d | 65% (13/20) | 1634 | — | 65% | GANO |
| Tuncay Duran | 1.20 | -21%‡ | 62.3% | 83.3% | n/d | 70% (14/20) | 1735 | Pretoria x1.4 | 80% | GANO |
| **Dusan Obradovic** | **1.18** | **-17%‡** | **67.5%** | **84.7%** | **+13** | **70% (14/20)** | **1723** | **KB7 x1.6 (5d)** | **70%** | **PERDIO** |

‡ Edge negativo calculado: `p_modelo < p_implicita`. El bookmaker SOBRESTIMA a estos jugadores
  más de lo que justifica el modelo. El modelo los predice ganadores (>50%) pero con menor
  probabilidad que la ya descontada por el mercado.

**Característica estructural compartida:** El mercado ya ha precio-ado la calidad de estos jugadores.
Funcionan como "piernas baratas" de multiplicación de cuota en el combo, NO como fuente de alpha.

---

## 2. El Anti-Patrón Obradovic — Anatomía del Único Error

### Lo que el modelo vio (señales positivas)

```
alpha +13:       markov=HOT(+10), cal=0.82(+3)
CAMPEON:         M15 Kursumlijska Banja 7 2026 (5W-0L, 5 días atrás)
                 TORNEO_COMPLETO_BONUS x1.6 quality_score
RACHA:           5 victorias consecutivas en el mismo torneo/ciudad
ELO:             1723 (sólido para M15 ITF)
Rival (Fabre):   ELO 1496 — brecha de 227 puntos
```

### Lo que el modelo no vio / lo que el mercado ya había descontado

```
edge:            -17.2% (p_modelo 67.5% < p_implicita 84.7%)
SCALP:           0.0% — ninguna victoria documentada vs rival top-100 en esta superficie
triple_alignment: NO activo — señales internas NO convergen
surface_bonus:   NO en alpha (sin especialización de superficie confirmada)
RFI rival:       NINGUNO — Fabre es jugador activo, sin decay de inactividad
BBI:             n/d (no calculado para picks de cuota <1.40 en la config actual)
```

### El efecto TORNEO_COMPLETO_BONUS como falso positivo en contexto de mercado eficiente

Este es el hallazgo más importante del análisis:

El bookmaker ya ha procesado el torneo ganado por Obradovic 5 días antes. Lo sabe. Por eso
le da cuota 1.18 (84.7% implícita). El TORNEO_COMPLETO_BONUS interno (x1.6) elevó el
quality_score del modelo internamente, pero el mercado ya llegó a la misma conclusión.
El bonus duplica información que el precio ya refleja.

Peor aún: el bonus rebajó el peso de `form` de 0.30→0.23, silenciando el análisis del rival
(Fabre) y la evaluación del riesgo subyacente. El modelo quedó más seguro de Obradovic
precisamente en el momento en que tenía menos razones propias para estarlo.

```
Analogía clínica: administrar un anestésico en exceso porque el paciente "parece cómodo."
El marcador de comodidad ya ha sido descontado — el exceso solo bloquea señales de alerta.
```

### ¿Por qué Orlov, Arzhankin y Duran ganaron si también tenían edge negativo?

No hay señal diferenciadora robusta entre el ganador y los perdedores dentro del grupo VARIABLE.
El análisis honesto es:

```
Tasa esperada @1.18-@1.27 (p_impl = 78-85%): ~3.1-3.4 ganadores en 4 intentos
Resultado observado: 3/4 ganadores
→ Rendimiento DENTRO del rango esperado por azar. No hay alpha diferencial.
→ Obradovic no perdió por un error detectable del modelo — perdió dentro de la
  varianza normal de picks sobrefa vorecidos por el mercado.
```

La diferencia real está en que @1.18 (Obradovic) tiene el edge más negativo en términos
absolutos de probabilidad (-17.2pp), lo que significa el precio más caro para una calidad
que el mercado ya valora correctamente. Pero eso tampoco es una regla predicativa sólida
con n=4.

---

## 3. Señales de Calidad — Ranking por Evidencia

Basado en esta sesión más casos históricos documentados en nodos anteriores:

| Señal | Hits/n en sesión | Evidencia | Mecanismo |
|---|---|---|---|
| SCALP TOP-100/50 en superficie | 5/5 (Weis, Wiskandt, Kostovic, Choinski, Jeanjean) | Fuerte | Calidad probada contra elite en mismo tipo de cancha — el modelo captura lo que el ELO simple del bookmaker no distingue entre circuitos |
| Edge positivo ≥10% | 6/6 ANCHOR picks ganaron | Fuerte | p_modelo > p_implicita = señal de ineficiencia de mercado real explotable |
| RFI-2+ en rival (≥180d inactividad) | 1/1 (Milosavljevic — rival 261d) | Medio (Nodo-64, n=2) | Form decay del rival eleva p_modelo incluso cuando la forma propia del favorito es mala |
| Triple alignment ≥0.38 | 1/1 (Weis, 0.38) | Medio | Convergencia interna: rivalry + markov + surface → +8 alpha extra |
| CAMPEON tier superior jugando tier inferior | 2/2 (Weis: RG→M25; Choinski: ATP500→Challenger) | Medio (n pequeño) | Bookmaker mispricing en tiers bajos cuando la calidad real es de circuito superior |
| HOT solo (sin SCALP, sin edge+, sin RFI) | 3/4 ganaron PERO varianza alta | Débil | No distingue de forma estadísticamente significativa en este rango de cuota |
| CAMPEON reciente solo (sin edge+, sin SCALP) | 1/2 (Obradovic perdió, Duran ganó) | Inconcluyente | El mercado ya procesa el torneo reciente — el bonus interno duplica información ya en el precio |

---

## 4. Fenómeno "Tier Mismatch" — Señal Candidata Nueva

### Observación empírica

Dos picks del pool tenían perfiles extraordinarios para el tier donde jugaban:

```
Alexander Weis:
  Título reciente:   Roland Garros 2026 (ATP500/GS) — 7W-0L, top-10 en final (#9 Cobolli)
  Torneo actual:     M25 Bastia (ITF)
  ELO modelo:        2161 (percentil >95% del circuito profesional completo)
  SCELPs en clay:    30+ victorias vs top-100 documentadas
  Delta tier:        GS/ATP500 → ITF (gap máximo posible)

Jan Choinski:
  Título reciente:   Zagreb 2026 (ATP500) — 5W-0L
  Torneo actual:     Challenger Braunschweig
  ELO modelo:        1764
  SCALP:             Fery #36 en clay
  Delta tier:        ATP500 → Challenger (gap significativo)
```

### Hipótesis mecanística

Cuando un jugador recientemente campeón en un tier significativamente más alto juega en un tier
inferior, el bookmaker enfrenta tres limitaciones simultáneas:

1. **Actualización lenta de ranking ATP**: El ranking sube con delay (resultados de puntos).
   El bookmaker usa ranking para calibrar cuota — si el ranking no refleja aún el último título,
   la cuota no está totalmente ajustada.

2. **Calibración más débil en tiers bajos**: Los modelos bookmaker son más precisos en ATP
   Top 100 que en Challenger/ITF, donde el historial es más raro y ruidoso.

3. **SCALP vs ELO local**: El ELO del jugador "jugando abajo" es alto en el modelo global
   pero no necesariamente reflejado en el ELO de ese circuito específico.

**Campo candidato para pick_snapshot:**
```python
"tier_mismatch":          True | False
"tier_mismatch_delta":    "gs_vs_itf" | "atp500_vs_itf" | "atp500_vs_challenger" |
                          "challenger_vs_itf" | null
"campeon_tier_nivel":     "gs" | "atp1000" | "atp500" | "atp250" | "challenger" | "itf"
"campeon_tier_actual":    "gs" | "atp1000" | "atp500" | "atp250" | "challenger" | "itf"
"campeon_days_ago":       int
```

**IMPLEMENTADO** (2026-07-10): campos `tier_mismatch`, `tier_mismatch_delta`, `campeon_tier_nivel`, `campeon_tier_actual` activos en pick_snapshot como observacionales. Solo para recolección de datos — NO cambian kelly_kl. Ver H77-01 (n_stop=30).

---

## 5. La Paradoja BLOQUEADO — Resolución Formal

### Observación

4 de los 6 picks ANCHOR estaban BLOQUEADOS (sin historial H2H del rival específico):

| Pick | Rival | Alpha | Signal compensadora | Resultado |
|---|---|---|---|---|
| Wiskandt | Matusevich | -2 (phantom_data -25 + HOT+10 + edge+10 + cal+3) | SCALP + edge +33.9% | GANO |
| Kostovic | Zidansek | -2 (phantom_data -25 + HOT+10 + edge+10 + cal+3) | SCALP x2 + edge +28.7% | GANO |
| Choinski | Dedura-Palomero | -12 (phantom_data -25 + edge+10 + cal+3) | SCALP + edge +29.6% + CAMPEON ATP500 | GANO |
| Jeanjean | Tubello | -2 (phantom_data -25 + HOT+10 + edge+10 + cal+3) | SCALP x3 + edge +17.2% | GANO |

### Explicación

El estado BLOQUEADO aplica penalidad por ausencia de datos del **rival específico**.
Pero la señal SCALP evalúa la calidad del **jugador favorito** contra el campo de elite en esa
superficie, independientemente del rival en cuestión.

```
SCALP TOP-100 no dice "contra este rival específico gana"
SCALP TOP-100 dice "este jugador puede ganar a cualquier jugador en este rango de calidad"
```

Cuando el edge positivo confirma que el mercado no ha descontado esa calidad:
- La ausencia de H2H específico pierde relevancia predictiva
- El SCALP + edge juntos constituyen evidencia más fuerte que el H2H en este contexto

**Regla derivada**: BLOQUEADO es señal de WARN (ausencia de datos del rival), no señal de
EXCLUSION, cuando el pick tiene SCALP en superficie + edge positivo ≥ 10%.

---

## 6. Superficie — Anomalía Duran

### Observación

Tuncay Duran jugó en M15 Monastir (superficie **DURA**), pero el modelo fue calibrado
predominantemente sobre arcilla (la sesión fue principalmente torneos clay). Duran ganó
igualmente (@1.20, HOT 80%, CAMPEON Pretoria x1.4).

Esto no valida que el modelo funcione en dura — el edge en Duran era negativo (-21%), lo que
sugiere que el bookmaker también había descontado bien su probabilidad. Duran ganó dentro de
la varianza esperada (@1.20 = 83.3% implícita, resultado positivo = resultado esperado).

**Sin embargo**, el output de la tabla de favoritos no marcó explícitamente la discrepancia de
superficie (model calibrado en clay → pick en hard). Esto es una deuda de observabilidad.

---

## 7. Deudas de Implementación

| ID | Descripción | Archivo | Estado | Fecha |
|---|---|---|---|---|
| D65-01 | Añadir label `[ANCHOR +X%]` / `[VARIABLE -X%]` por pick en output combo plan | combo_confianza_builder.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-02 | Mostrar `edge_vs_mercado` con signo POSITIVO/NEGATIVO para TODOS los picks; si edge negativo añadir línea rival con edge positivo | generar_tabla_favoritos2.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-03 | WARN `VARIABLE_SIN_RESPALDO` en pre_game_validator cuando cuota ≤1.30 + edge<0 + triple_alignment<0.35 | pre_game_validator.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-04 | Añadir campos observacionales `tier_mismatch`, `tier_mismatch_delta`, `campeon_tier_nivel`, `campeon_tier_actual` a pick_snapshot | edge_calculator.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-05 | En shadow_book --report: segmento ANCHOR (edge>0) vs VARIABLE (edge≤0) con hit%, IC95 Wilson, ROI; nota `[pre-graduacion n<30]` hasta n≥30 | shadow_book.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-06 | WARN `WARN_SUPERFICIE` en tabla favoritos cuando superficie del partido ≠ superficie dominante en calibracion_edge y gap hit% ≥5pp | generar_tabla_favoritos2.py | ✅ IMPLEMENTADO | 2026-07-10 |
| D65-07 | Registrar H77-01, H77-02, H77-03 en validation/preregistered_hypotheses.json | preregistered_hypotheses.json | ✅ IMPLEMENTADO | 2026-07-10 |

**GATED permanentemente hasta H77-01/H77-02 graduación (inmutable):**
- No cambiar `confidence_flag` basado en pick_tier
- No modificar `kelly_kl` por `tier_mismatch`
- No usar `tier_mismatch` como criterio de exclusión del pool

### Notas de implementación

**D65-03** — `triple_alignment` es proxy hasta que D64-01 (SCALP/RFI en pick_snapshot) esté disponible.
El WARN instruye al operador a verificar SCALP/RFI en `tabla_favoritos` manualmente.

**D65-04** — Los campos `tier_mismatch_*` se poblán parseando strings de `reasoning[]` donde
aparece `TORNEO_COMPLETO_BONUS ... tier=X`. Solo observacionales — no cambian `kelly_kl` ni `alpha`.

**D65-05** — El segmento aparece en `--report` con `[pre-graduacion n<30]` hasta tener muestra.
Cuando n≥30 se puede evaluar H77-02 (Fisher exact ANCHOR vs VARIABLE).

**D65-06** — La superficie dominante se calcula dinámicamente de `calibracion_edge.json → por_superficie`
(max wins+losses, excluyendo `unknown`/`?`). Actualmente: clay (n=1918). Solo emite WARN si gap ≥5pp
para evitar ruido en superficies con performance similar (hard 61% vs clay 65% = 4pp → silencioso).

---

## 8. Tests de Validación (REGLA-T53)

| Test | Descripción | Resultado esperado |
|---|---|---|
| T65-01 | Pick con p_modelo=0.594, p_implicita=0.472, cuota=2.12 | pick_tier="ANCHOR", edge_vs_mercado="+12.2%" |
| T65-02 | Pick con p_modelo=0.675, p_implicita=0.847, cuota=1.18 | pick_tier="VARIABLE", edge_vs_mercado="-17.2%" |
| T65-03 | Pick BLOQUEADO con SCALP TOP-100 + edge +28.7% | pick_tier="ANCHOR", status=WARN_BLOQUEADO (no EXCLUDED) |
| T65-04 | Pick BLOQUEADO sin SCALP + edge -17% + cuota 1.18 | pre_game_validator emite WARN "VARIABLE sin respaldo" |
| T65-05 | Campeon tier=atp500 (Zagreb), jugando tier=challenger | tier_mismatch=True, delta="atp500_vs_challenger" |
| T65-06 | Campeon tier=itf, jugando tier=itf | tier_mismatch=False, delta=null |
| T65-07 | Pick con superficie partido=hard, calibracion_edge superficie_dominante=clay | tabla favoritos emite WARN "SUPERFICIE DISCREPANCIA: modelo calibrado en clay, partido en dura" |
| T65-08 | Milosavljevic: n20=3/9=33% (Mala), rival 261d → pick_tier="ANCHOR" por RFI-2 + edge +17.6% | clasificación correcta — RFI-2 + edge+ sobreescribe penalidad de forma |

---

## 9. Hipótesis Pre-registradas

> Registradas en `validation/preregistered_hypotheses.json` el 2026-07-10. Congeladas — no modificar sin nueva decisión de diseño.

### H77-01 — Tier Mismatch como predictor de edge

```
Estado:     PRE-REGISTRADA (n_actual=2 / n_stop=30)
Condición:  campeon_tier_nivel > campeon_tier_actual (jugando torneo inferior al
            nivel de su título más reciente)
            AND campeon_days_ago ≤ 30
Métrica:    hit% de picks con tier_mismatch=True vs picks equivalentes sin tier_mismatch
            (mismo rango de conf: 55-70%)
Éxito:      hit%_mismatch > hit%_base con IC Wilson 95% inferior > 0.70, n ≥ 30
n_stop:     30 casos con tier_mismatch=True
Gate:       D65-04 activo (observacional) — NO calibrar kelly_kl antes de graduación
Notas:      Sesión 2026-07-10 aporta n=2 (Weis GS→ITF, Choinski ATP500→Challenger)
```

### H77-02 — ANCHOR vs VARIABLE: separación estadística de rendimiento

```
Estado:     PRE-REGISTRADA (n_actual=10 / n_stop=60)
Condición:  ANCHOR_PICK  = edge positivo ≥ 0% (p_modelo > p_implicita)
            VARIABLE_PICK = edge negativo  (p_modelo < p_implicita)
            Rango de cuota: 1.50–2.20 para ANCHOR, 1.15–1.35 para VARIABLE
Métrica:    hit%_ANCHOR vs hit%_VARIABLE (separados por rango de cuota)
            ROI_ANCHOR vs ROI_VARIABLE
Éxito:      hit%_ANCHOR estadísticamente superior (Fisher exact, p < 0.05)
            con n ≥ 30 ANCHOR y n ≥ 30 VARIABLE
n_stop:     60 total (30+30)
Gate:       D65-05 activo — segmento ANCHOR/VARIABLE visible en shadow_book --report
            con nota [pre-graduacion n<30] hasta muestra suficiente
Notas:      Sesión 2026-07-10 aporta ANCHOR n=6 (6/6 ganaron), VARIABLE n=4 (3/4)
```

### H77-03 — BLOQUEADO + SCALP + edge positivo: eliminación de penalidad H2H

```
Estado:     PRE-REGISTRADA (n_actual=4 / n_stop=20)
Condición:  pick con estado BLOQUEADO (sin H2H del rival específico)
            AND SCALP TOP-100/50 en superficie del partido
            AND edge positivo ≥ 10%
Métrica:    hit% ≥ hit% de picks no-BLOQUEADO con conf equivalente
Éxito:      IC Wilson 95% inferior > 0.65, n ≥ 20 casos BLOQUEADO+SCALP+edge+
n_stop:     20
Gate:       Si H77-03 gradúa → revisar si phantom_data(-25) debe ser menor cuando
            SCALP presente (deuda futura — NO modificar antes de graduación)
Notas:      Sesión 2026-07-10 aporta n=4 (Wiskandt, Kostovic, Choinski, Jeanjean):
            todos BLOQUEADOS + SCALP + edge+ → 4/4 ganaron
```

---

## 10. Marco de Expertos

### Marco 1 — Estadístico

El hallazgo central es la existencia de **dos poblaciones cualitativamente distintas** dentro
del mismo pool de picks. La separación ANCHOR/VARIABLE no es artificial: refleja si el modelo
detecta una ineficiencia real de mercado (edge positivo) o si solo confirma lo que el mercado
ya sabe (edge negativo).

En 6/6 picks ANCHOR con edge ≥ 10%, el modelo capturó alpha real. En 3/4 picks VARIABLE con
edge negativo, el resultado es estadísticamente indistinguible del rendimiento esperado dado
las probabilidades implícitas del mercado. No hay evidencia de alpha diferencial en el segmento
VARIABLE con n=4 — ni positiva ni negativa.

### Marco 2 — Domain Expert (circuito profesional)

El SCALP TOP-100 captura información que los modelos ELO simples del bookmaker no reflejan
correctamente cuando el jugador oscila entre circuitos. Un jugador de circuito ATP que baja a
jugar un M25 ITF por razones de ranking o scheduling NO cambia de calidad real. El bookmaker
actualiza más lentamente en estas transiciones que el modelo que tiene el SCALP en su historial.

El "tier mismatch" (Weis en M25 después de ganar Roland Garros, Choinski en Challenger después
de ganar Zagreb) es la versión más extrema de este fenómeno: el jugador es literalmente el
mejor del mundo en su superficie reciente, jugando contra rivales 3-5 tiers por debajo de su
mejor rendimiento.

### Marco 3 — Arquitectura del Combo

El combo Nodo-38 tiene dos funciones distintas que el sistema actual no separa explícitamente:

```
Función A (ANCHOR):   Capturar alpha real de mercado → cuotas 1.56-2.12 con edge positivo
Función B (VARIABLE): Multiplicar probabilidad acumulada → cuotas 1.18-1.30 "baratas"
```

El valor del combo no reside en las piernas baratas (VARIABLE) sino en los picks con edge real
(ANCHOR). Las piernas baratas son precio de entrada para multiplicar el retorno, no fuentes de
información. El operador debe comprender qué es alpha y qué es estructura para tomar decisiones
informadas cuando una pierna variable falla.

### Marco 4 — Bayesiano (prior/posterior de calidad)

La actualización posterior de probabilidad de ganar sigue este orden de señales, de mayor
a menor fuerza informativa:

```
1. SCALP TOP-100 en superficie   → prior fuerte: calidad probada directamente
2. Edge positivo ≥ 10%           → mercado confirma la ineficiencia detectada
3. Triple alignment (≥ 0.38)     → señales internas coherentes → redundancia positiva
4. RFI-2+ en rival               → prior fuerte sobre decaimiento del oponente (Nodo-64)
5. Tier mismatch                 → prior de calidad real > calidad aparente en tier actual
6. HOT (≥ 70%)                   → prior débil de momentum — necesita respaldo
7. CAMPEON reciente solo          → prior potencialmente ya descontado por el mercado
```

Cuando 3+ señales del nivel 1-4 convergen, la confianza posterior justifica un ANCHOR pick.
Cuando solo aparecen señales del nivel 6-7 sin respaldo de los niveles superiores, la confianza
es de momentum a corto plazo, no estructura predictiva diferencial.

---

## 11. Registro de la Sesión

```
Sesión:           betslip_index_20260710_015309.json (01:53 AM)
Combo plan:       combo_plan_20260710_015306.txt
Tabla favoritos:  analisis_partidos_20260710_012116.txt (4.3 MB, ~107 partidos analizados)
Betslip archivos: apuestas_20260710_021248 al 021643 (8 archivos, 7 únicos combos)

RESUMEN DE PICKS:

  ANCHOR (edge positivo — 6 picks, 6/6 ganaron, 100%):
    Alexander Weis     @2.12  edge+12.2%  alpha+34  ELO2161  SCALP_x30+  CAMPEON_RG_atp500
    Max Wiskandt       @1.92  edge+33.9%  alpha-2   ELO1751  SCALP_x1    CAMPEON_M25_5d    BLOQUEADO
    Marko Milosavljevic @1.91 edge+17.6%  alpha+13  ELO1488  RFI-2(261d) form_mala_33%
    Teodora Kostovic   @1.80  edge+28.7%  alpha-2   ELO1686  SCALP_x2    CAMPEON_EXPIRADO  BLOQUEADO
    Jan Choinski       @1.78  edge+29.6%  alpha-12  ELO1764  SCALP_x1    CAMPEON_Zagreb_ATP500  BLOQUEADO
    Leolia Jeanjean    @1.56  edge+17.2%  alpha-2   ELO1756  SCALP_x3                      BLOQUEADO

  VARIABLE (edge negativo — 4 picks, 3/4 ganaron = 75% = rendimiento esperado):
    Vladyslav Orlov    @1.26  edge-13%   ELO1692  HOT70%   CAMPEON_EXPIRADO_306d  → GANO
    Anton Arzhankin    @1.27  edge-13%   ELO1634  HOT65%   —                      → GANO
    Tuncay Duran       @1.20  edge-21%   ELO1735  HOT80%   CAMPEON_Pretoria_x1.4  → GANO
    Dusan Obradovic    @1.18  edge-17%   ELO1723  HOT70%   CAMPEON_KB7_x1.6_5d    → PERDIO

  UNICO ERROR (Obradovic): perfil VARIABLE sin ninguna señal del nivel 1-5.
  alpha +13 = solo HOT(+10) + cal(+3). Sin SCALP, sin edge positivo, sin RFI rival,
  sin triple alignment, sin tier mismatch. TORNEO_COMPLETO_BONUS x1.6 duplicó información
  ya descontada por el mercado (@1.18 = 84.7% implícita).

HALLAZGOS FORMALIZADOS:
  H77-01 pre-registrada: tier mismatch como predictor (n=2 → n_stop=30)
  H77-02 pre-registrada: ANCHOR vs VARIABLE rendimiento separado (n=10 → n_stop=60)
  H77-03 pre-registrada: BLOQUEADO+SCALP+edge+ vs H2H estándar (n=4 → n_stop=20)
```

---

## 12. Lo que Este Nodo NO Dice

Para evitar sobreinterpretación:

1. **No dice que los picks VARIABLE sean inútiles en el combo.** Cumplen función de
   multiplicación de cuota acumulada. La arquitectura Nodo-38 es válida. El problema es
   tratarlos como fuentes de alpha cuando no lo son.

2. **No dice que CAMPEON sea señal negativa.** Es positiva cuando acompaña SCALP + edge
   positivo (Weis, Choinski). Es neutral cuando el mercado ya la ha descontado (Obradovic,
   Duran, Orlov).

3. **No dice que 1 sesión prueba nada.** H77-01/02/03 requieren n≥20-60. Esta sesión es
   el caso semilla que motiva la pre-registro, no la graduación.

4. **No dice que el error de Obradovic fuera evitable con certeza.** A @1.18, el favorito
   pierde ~15-20% de las veces. Obradovic podría haber ganado. El análisis es sobre
   asimetría de señales, no sobre causalidad del resultado individual.
```

# Nodo-51: Plan Estratégico — Data Layer Unificado + Contexto de Torneo como Entidad de Primera Clase

> **Wikilinks:** [[Nodo-45-Temporal-History-Fallback]] | [[Nodo-46-Markov-Surface-Context-Discount]] | [[Nodo-47-Inject-Kambi-Ranking-Guard-Bug]] | [[Nodo-48-FlashScore-Odds-Scraper-Testing]] | [[Nodo-49-Playwright-H2H-Fallback-n-h2h-0]] | [[Nodo-50-Filtro-Torneo-PASO1]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | [[Nodo-21-Pesos-Diferenciados-Tier]]
> **Fecha de creación:** 2026-07-01
> **Estado:** 📋 PLAN ESTRATÉGICO — nodo maestro, define orden de implementación de Fases 0-5
> **Tipo:** Meta-nodo (organiza nodos 45-50 bajo un marco unificado; no introduce features nuevos sin dependencia previa)

**Prioridad:** ALTA — sin este marco, los nodos 45-50 siguen siendo parches reactivos independientes
**Principio rector:** Un pick sin datos completos no es un pick con edge alto — es un pick sin derecho a existir.

---

## 0. Diagnóstico — Por Qué los Nodos 45-50 No Bastan Como Están

Los últimos 6 nodos comparten un patrón que ninguno nombra explícitamente:

| Nodo | Síntoma tratado | Problema estructural subyacente |
|---|---|---|
| 45 (THF) | Historial vacío | El pipeline no distingue "dato ausente" de "dato = 0" |
| 46 (Surface) | Markov contaminado | El contexto (superficie) no viaja con el dato |
| 47 (Ranking bug) | Kambi sobreescribe ATP | Identidad de jugador resuelta ad-hoc en cada componente |
| 48 (FlashScore odds) | Sin cuotas post-match | Una sola fuente por tipo de dato, sin procedencia |
| 49 (Playwright H2H) | n_h2h=0 → edge fantasma | Idéntico a 47: identidad + fuente única |
| 50 (--torneo) | PASO 2 lento | El torneo es un string, no una entidad con atributos |

**El diagnóstico unificado:** el pipeline trata los datos como strings sueltos que fluyen entre pasos. No existe (a) una identidad canónica de jugador, (b) una entidad de torneo con atributos, (c) un contrato de completitud de datos, ni (d) procedencia por campo. Cada nodo 45-50 parchea un síntoma de esa ausencia.

**Evidencia del costo:** el bug de Nodo-47 vivió *desde la introducción de `_inject_kambi_ranking`* — meses — corrompiendo `ranking_momentum` en cada sesión con Kambi, incluyendo las 706 observaciones de `calibracion_edge.json`. Los edges fantasma de Nodo-49 llegaron al trader (Combo1-8) el mismo día. Estos no son bugs raros: son la consecuencia esperada de la arquitectura actual.

---

## 1. Los 5 Modelos Mentales — El Marco Invisible

Estos son los marcos con los que se organiza el plan. Cada uno viene de un dominio ajeno al betting, y cada uno mapea directamente a decisiones de código concretas.

### MM-1. Inversión (Jacobi / Munger) — "Invierte, siempre invierte"

**Origen:** matemática (Carl Jacobi), popularizado en inversión por Charlie Munger.

**Principio:** en vez de preguntar "¿cómo consigo el resultado?", pregunta "¿qué garantiza que el fracaso sea imposible?" y diseña contra eso.

**Aplicación:** la pregunta de los nodos 45/49 era "¿cómo consigo historial del jugador?". La pregunta invertida es: **"¿qué garantiza que NUNCA se calcule un edge sobre `p_modelo=0.500` por datos ausentes?"**. Eso cambia el diseño: en vez de más fallbacks (ofensa), se necesita un **contrato de completitud** (defensa) que todo pick debe firmar antes de llegar al trader. Los fallbacks alimentan el contrato; el contrato es el dueño de la garantía.

```
DISEÑO ACTUAL:   fuentes → parseo → modelo → gates dispersos (33, 35, guard trader...)
DISEÑO INVERTIDO: fuentes → parseo → CONTRATO DE COMPLETITUD → modelo → trader
                  (un pick que no firma el contrato no genera p_modelo — genera NULL)
```

**Conexión oculta que revela:** el edge fantasma de 60.4% de Mario Arce no es un edge alto — es un `NaN` disfrazado de número. `p_modelo=0.500` con historial vacío no es una probabilidad: es la ausencia de una probabilidad. El modelo actual no puede expresar "no sé", y esa incapacidad es la causa raíz de los 4 combos fantasma del 2026-07-01.

### MM-2. Teoría de Restricciones (Goldratt, manufactura) — La cadena vale lo que su eslabón de datos más débil

**Origen:** gestión de producción industrial (*La Meta*, 1984).

**Principio:** todo sistema tiene UNA restricción dominante. Optimizar cualquier otra cosa es ilusión de progreso. Identifica la restricción, explótala al máximo, subordina todo lo demás a ella.

**Aplicación:** la restricción del pipeline NO es velocidad (11 min está bien) ni el modelo Markov. Es la **cobertura de datos en ITF/Challenger** — exactamente el tier donde Nodo-21 reclama la mayor ventaja informacional. La paradoja documentada: donde más alpha hay, menos datos llegan (46% cobertura de odds ITF en Nodo-48; jugadores no indexados por Ninja API en Nodo-49).

**Decisiones que impone:**
- El presupuesto de Playwright (~30s/partido) debe gastarse *solo* en la restricción: partidos ITF/Challenger con edge preliminar prometedor, no en los 197 partidos.
- Subordinación: el scheduling del PASO 2 debería procesar primero los partidos del torneo objetivo (Nodo-50) y encolar los fallbacks Playwright en batch al final, no inline (evita que 20 fallbacks × 30s bloqueen la sesión).
- Métrica de la restricción: **% de picks del tier objetivo con contrato de completitud firmado**. Esa es la métrica del sistema, no el hit% global.

### MM-3. Sistema Inmune / Defensa en Profundidad (inmunología + seguridad informática)

**Origen:** biología (inmunidad innata vs adaptativa) y arquitectura de seguridad.

**Principio:** dos capas con costos distintos. La innata es rápida, barata, genérica (piel, inflamación). La adaptativa es lenta, cara, específica (anticuerpos). Un organismo que usara solo la adaptativa moriría de costo metabólico; solo la innata, de especificidad insuficiente.

**Aplicación:** el fix de Nodo-47 ya ES este patrón sin saberlo — y por eso funcionó:

```
Fast path O(1):  dict.get(normalized) + dict.get(reversed_key)   ← inmunidad innata (95% casos, µs)
Slow path:       get_player_info() intelligent matching           ← inmunidad adaptativa (5% casos, 5.8ms)
```

**La generalización que nadie hizo:** este patrón dos-capas debe ser la arquitectura de TODA resolución en el pipeline, no un fix local de una guard:

| Resolución | Capa innata (rápida) | Capa adaptativa (cara) |
|---|---|---|
| Identidad jugador (F0) | dict canónico O(1) | fuzzy matching + birth year |
| Historial (45/49) | Ninja API + THF cache | Playwright DOM |
| Cuotas (48) | Kambi API | FlashScore Playwright |
| Torneo/superficie (50) | parse de torneo_completo | lookup manual/tabla |

Y como en el sistema inmune, **la capa adaptativa deja memoria**: todo lo que Playwright recupera se escribe al cache THF, de modo que la próxima sesión lo resuelve la capa innata. Hoy Nodo-49 no hace esto — cada sesión re-paga los 30s. Ese es un hallazgo directo del modelo mental.

### MM-4. Nicho Ecológico (ecología de poblaciones) — La forma es específica del hábitat

**Origen:** ecología (Hutchinson, concepto de nicho n-dimensional).

**Principio:** el fitness de un organismo no es una propiedad del organismo — es una propiedad de la relación organismo×ambiente. Un depredador dominante en un hábitat puede ser mediocre en el contiguo. Medir fitness sin registrar el hábitat produce predicciones sistemáticamente erradas en las fronteras entre hábitats.

**Aplicación:** es exactamente Nodo-46, pero el modelo mental lo lleva más lejos que el discount propuesto:

1. **El estado Markov no es del jugador — es del par (jugador, superficie).** El discount de Nodo-46 (acercar el factor a 1.0) es la versión débil. La versión fuerte, cuando haya n: cadenas Markov *separadas por superficie*, con la cadena de otra superficie como prior débil, no como señal descontada.
2. **El calendario ATP es un gradiente de hábitats con fronteras predecibles:** arcilla(abr-jun) → hierba(jun-jul) → hard(ago-sep). Las transiciones no son ruido — son fechas conocidas. El pipeline puede subir automáticamente la incertidumbre de TODO el circuito en las 2 semanas post-transición (flag de temporada), en vez de descubrirlo jugador por jugador.
3. **El tier es una segunda dimensión del nicho:** un jugador HOT en ITF que sube a Challenger cambió de hábitat aunque la superficie sea la misma (nivel de rival ≠). El `surface_overlap_rate` de Nodo-46 tiene un hermano natural: `tier_overlap_rate`. Mismo código, otra columna.

**Conexión oculta:** Nodo-46 y Nodo-21 (pesos por tier) son el mismo fenómeno en dos dimensiones del nicho. Unificar: `context_overlap = f(surface_overlap, tier_overlap)` como multiplicador único de confianza Markov.

### MM-5. Jerarquía de Evidencia (medicina basada en evidencia) — Un case report no derrota a una cohorte

**Origen:** epidemiología clínica (pirámide de evidencia: anécdota < serie de casos < cohorte < RCT).

**Principio:** la evidencia tiene rangos. Un caso espectacular (n=2, 100% hit) no actualiza una calibración de cohorte (hit%=48.2% con n grande) — genera una *hipótesis pre-registrada* que se somete a prueba con criterios de parada definidos ANTES de ver los datos.

**Aplicación:** resuelve limpiamente los desacuerdos que Sonnet identificó:

- **WAS vs T33-01:** WAS (n=2) es un case report; T33-01 (calibrado) es la cohorte. El nodo 44 ya lo intuyó (REGLA-WAS-1: stake mínimo hasta n≥30), pero le falta el pre-registro formal: definir HOY el criterio de éxito (hit% > 55% con intervalo binomial, n≥30), congelar los umbrales (edge≥10%, cuota≥2.0) y NO ajustarlos mirando resultados intermedios — eso es p-hacking y destruye la validez del test.
- **Nodo-46 (n=1 Watanuki):** correctamente degradado a hipótesis tras el post-mortem de Nodo-47. El criterio de atribución de 3 condiciones que definió el nodo ES un protocolo de adjudicación clínica. Mantenerlo, acumular n≥5, solo entonces calibrar constantes.
- **La calibración contaminada de Nodo-47:** las 706 observaciones se generaron con ranking corrupto. La jerarquía de evidencia exige marcarlas: añadir campo `calibration_epoch` — las observaciones pre-fix y post-fix son cohortes distintas y no deben mezclarse al recalibrar Challenger.

**Y la regla GIT-FIRST es esto mismo aplicado al proceso:** buscar en git antes de implementar = revisión sistemática de literatura antes de un trial. Violada dos veces (48, 49) con costo documentado. La fase F-Meta la convierte en checklist obligatorio, porque las reglas que dependen de disciplina fallan; las que dependen de proceso, no.

---

## 2. Las Conexiones Ocultas — Síntesis Entre Modelos

```
MM-1 (Inversión)     ──→  Contrato de Completitud (F2)  ──→ mata edges fantasma en el ORIGEN
MM-2 (Restricción)   ──→  Presupuesto Playwright dirigido (F3) ──→ el costo va donde está el alpha (ITF)
MM-3 (Inmune)        ──→  Patrón dos-capas + MEMORIA al cache (F0, F3) ──→ el fallback caro se paga UNA vez
MM-4 (Nicho)         ──→  TournamentContext + context_overlap (F1, F4) ──→ unifica Nodos 46+21+50
MM-5 (Evidencia)     ──→  Pre-registro WAS + epochs de calibración (F5) ──→ resuelve los desacuerdos abiertos
```

**Las tres conexiones no obvias más valiosas:**

**C1 — Nodo-47 y Nodo-49 son UN problema: entity resolution.** El bug de `'daniil glinka'` vs `'glinka daniil'` y los jugadores ITF no indexados por Ninja son ambos fallos de resolución de identidad. La disciplina de *record linkage* (censos, bases de datos médicas) lo resuelve con una tabla maestra: un ID canónico por jugador, con todos sus alias (formato ATP "Apellido Nombre (año)", formato Kambi, formato FlashScore, formato THF). Se resuelve UNA vez al entrar al pipeline; todos los componentes downstream usan el ID, nunca el string. Esto elimina la categoría entera de bugs a la que pertenece Nodo-47, no solo esa instancia.

**C2 — `cuota_es_real` es la semilla de un sistema de procedencia por campo.** Nodo-48 lo inventó para cuotas. Generalizado: cada campo crítico lleva su origen:

```json
"ranking_pts":   {"valor": 339,  "provenance": "atp_file"},        // vs "kambi_estimate"
"p1_history":    {"n": 18,       "provenance": "playwright_dom"},  // vs "ninja_api" | "thf_cache"
"cuota1":        {"valor": 3.30, "provenance": "kambi_live"}       // vs "flashscore_ref"
```

El contrato de completitud (F2) lee la procedencia y asigna un `data_quality_score` al pick. El trader, los gates y WAS dejan de adivinar la calidad del dato: la leen. El guard D48-05 se vuelve un caso particular de una regla general.

**C3 — THF + Playwright + GIT-FIRST son el mismo principio: la respuesta ya existe en un artefacto pasado.** THF busca en sesiones anteriores; Playwright-con-memoria (MM-3) escribe al THF; GIT-FIRST busca en commits. Los tres son "memoria organizacional antes de trabajo nuevo". Formalizarlo como propiedad del sistema: **nada caro se computa dos veces, nada se implementa sin buscar si ya se implementó.**

---

## 3. El Plan de Implementación — Fases para Sonnet

Orden estricto: cada fase depende de la anterior. NO saltar fases.

### F0 — Registro Canónico de Jugadores (entity resolution) 【MM-3, C1】

**Archivo nuevo:** `core/player_registry.py`
**Qué hace:** clase `PlayerRegistry` con tabla `canonical_id → {alias_atp, alias_kambi, alias_flashscore, alias_thf, rank, pts}`.
- Fast path O(1): lookup por cualquier alias normalizado (incluye clave invertida — absorbe el fix de Nodo-47).
- Slow path: fuzzy matching (reutilizar `get_player_info()` existente), y **al resolver, registra el alias nuevo** → la próxima vez es O(1) (memoria inmune).
- `ninja_h2h_parser.py`, `markov_analyzer.py`, THF y el futuro Playwright fallback consumen `canonical_id`, nunca strings.

**Tests F0:** resolución de los 6 casos documentados en Nodo-47 (Glinka, Mayo, Watanuki, Ilagan, Hussey, Manning) + caso ITF desconocido → registra nuevo ID.
**Criterio de aceptación:** 0 sobreescrituras Kambi en jugadores presentes en ATP file; los 1438 tests baseline pasan.

### F1 — TournamentContext como Entidad 【MM-4, extiende Nodo-50】

**Archivo:** `core/tournament_context.py` + modificación en `scraping/kambi_tennis.py`
**Qué hace:** en PASO 1, cada match dict recibe un objeto/subdict:

```python
"tournament_context": {
    "nombre": "Wimbledon", "tier": "grand_slam",
    "superficie": "grass",            # parseado de torneo_completo, normalizado con _SURFACE_MAP (Nodo-46)
    "season_transition_flag": True,   # True si estamos a <14 días de una frontera de superficie del calendario
}
```

- `--torneo` (Nodo-50) sigue igual; esto añade estructura, no cambia el filtro.
- La superficie se resuelve UNA vez aquí y viaja con el match — Nodo-46 (F4) deja de inferirla.

**Tests F1:** parseo de superficie desde `torneo_completo` para los 3 tipos + `season_transition_flag` para fechas conocidas (30-jun → True por hierba→hard próxima).

### F2 — Contrato de Completitud + Procedencia 【MM-1, C2 — LA FASE CRÍTICA】

**Archivos:** `core/data_contract.py` + hooks en `ninja_h2h_parser.py` y `edge_calculator.py`
**Qué hace:**
1. Campos de procedencia: `history_provenance` (`ninja_api|thf_cache|playwright_dom|EMPTY`) y `ranking_provenance` (`atp_file|kambi_estimate`) en cada match. `cuota_es_real` se mantiene (compatibilidad) y se mapea a `odds_provenance`.
2. `completeness_score(match) → [0,1]` con regla dura: **si `p1_history` o `p2_history` está `EMPTY`, el edge_calculator emite el pick con `edge=None, status='NO_DATA'`** — nunca `p_modelo=0.500`. Un coin-flip por ignorancia no es una probabilidad.
3. El trader excluye `status='NO_DATA'` de TODOS los pools, incluido el pool de cobertura (el hueco exacto por donde entraron los Combo1-8 fantasma del 2026-07-01).

**Tests F2:** replay de la sesión 2026-07-01 → Arce/Vlajic/Guajardo/Cooper salen como `NO_DATA`, 0 combos fantasma.
**Criterio de aceptación:** es IMPOSIBLE (por construcción, no por gate) que un pick con historial vacío tenga edge numérico.

### F3 — Cadena de Fallback con Presupuesto y Memoria 【MM-2, MM-3, completa Nodo-49】

**Archivo:** `scraping/ninja_h2h_parser.py` (la implementación de Nodo-49 §3, con dos mejoras del marco):
1. **Batch al final, no inline:** los partidos con historial vacío se encolan; al terminar el pase API se lanza UNA sesión Playwright que procesa la cola (un browser, N páginas) — evita 20 × 30s de arranques de Chromium y no bloquea el flujo.
2. **Escritura a THF:** todo lo recuperado por Playwright se persiste al cache de Nodo-45 con `provenance='playwright_dom'` → la próxima sesión es un hit de cache (MM-3: memoria adaptativa).
3. **Presupuesto (MM-2):** parámetro `--pw-budget N` (default 20): máximo de partidos que van a Playwright por sesión, priorizados por (tier objetivo del día, cuota dentro de rango apostable). Lo que excede el presupuesto queda `NO_DATA` — que gracias a F2 es un estado seguro, no un edge fantasma.

**Tests F3:** los T49-01→06 del nodo + T51-F3-01 (lo recuperado aparece en THF en la sesión siguiente) + T51-F3-02 (presupuesto respeta prioridad por tier).

### F4 — Context Discount Unificado 【MM-4, implementa Nodo-46 mejorado】

**Archivo:** `analysis/markov_analyzer.py`
**Qué hace:** D46-01→06 tal como están especificados, con tres cambios derivados del marco:
1. `current_surface` viene de `tournament_context.superficie` (F1) — elimina toda inferencia.
2. Añadir `tier_overlap_rate` junto a `surface_overlap_rate` (misma función, campo `tier` del historial). Solo REGISTRARLO en el output por ahora — no descontarlo hasta tener n (MM-5).
3. `season_transition_flag=True` → aplicar el `min_floor` conservador aunque el overlap sea ambiguo.
4. **Calibración de constantes (`min_floor`, `THRESHOLD`) BLOQUEADA hasta n≥5 casos** que cumplan el criterio de atribución de 3 condiciones del Nodo-46. Con n=1 (Watanuki), implementar con los defaults propuestos y flag `--no-surface-discount` para poder A/B.

### F5 — Framework de Validación Pre-Registrada 【MM-5】

**Archivo:** `validation/preregistered_hypotheses.json` + sección en pipeline_tracker
**Qué hace:**
1. **Pre-registro WAS (D44-03):** congelar hoy: umbrales (edge≥10%, cuota≥2.0, señales de la jerarquía Nodo-44), métrica (hit% con IC binomial 95%), n de parada (30), y regla: los umbrales NO se tocan hasta n=30. Registrar cada pick WAS con timestamp automático.
2. **Epochs de calibración:** campo `calibration_epoch` en `calibracion_edge.json`. Epoch 1 = pre-fix Nodo-47 (706 obs, ranking parcialmente corrupto, válidas para GS/clay según análisis del nodo); Epoch 2 = post-fix. La recalibración de Challenger usa SOLO Epoch 2.
3. **Criterio Nodo-46:** contador formal de casos atribuibles (las 3 condiciones), visible en el tracker. Al llegar a n=5 → desbloquear D46-07.

### F-Meta — GIT-FIRST como Proceso, No como Regla 【MM-5, C3】

**Archivo:** `PRE_IMPLEMENTATION_CHECKLIST.md` en la raíz del repo
Checklist obligatorio antes de cualquier nodo nuevo con scraping/datos:
```
[ ] git log --all --oneline -- '*<keyword>*'  — ¿existe código previo del usuario?
[ ] git grep <keyword> $(git rev-list --all) — ¿la solución ya se escribió?
[ ] ¿La URL que voy a usar es del NAVEGADOR o de una API interna? (error raíz de 48 y 49)
[ ] ¿El dato "inexistente" existe en otra fuente ya conocida (THF, FlashScore DOM, git)?
```
Costo: 5 minutos. Costo de no hacerlo, ya pagado dos veces: nodos declarados BLOQUEADOS con solución existente, y edges fantasma en producción.

---

## 4. Matriz de Trazabilidad — Qué Resuelve Cada Fase

| Fase | Deuda técnica que cierra | Nodo(s) | Riesgo si se omite |
|---|---|---|---|
| F0 | Clase entera de bugs tipo Nodo-47 | 47, 49 | Próximo bug de nombres en otro componente |
| F1 | Prerequisito de D46-02/04 | 50, 46 | Nodo-46 infiere superficie con ambigüedad |
| F2 | Nodo-35 completo + hueco del pool de cobertura | 33, 35, 49 | Más combos fantasma en producción |
| F3 | Nodo-49 completo + D45 (memoria) | 45, 49 | 30s×N cada sesión, para siempre |
| F4 | D46-01→06 | 46, 21 | Sesgo sistemático en cada transición de temporada |
| F5 | D44-03, D44-05, D46-07, contaminación calibración | 44, 46, 47 | WAS se convierte en p-hacking; calibración mezclada |

## 5. Sobre el "Edge Revolucionario" — Nota de Honestidad Metodológica

Este plan no promete un edge revolucionario, porque los edges revolucionarios prometidos de antemano son la firma del overfitting. Lo que promete es más valioso y es lo que los expertos reales del campo (quant betting, no tipsters) efectivamente hacen: **eliminar las fuentes de auto-engaño antes de medir el alpha.** Los edges fantasma (49), los rankings corruptos (47) y las señales de contexto equivocado (46) no reducían el alpha real — lo hacían *inmedible*, mezclado con ruido estructural. El alpha genuino del sistema ya está identificado en Nodo-44 (gap reputación-bookmaker vs estado-actual-modelo); las fases F0-F5 construyen el instrumento limpio que permite confirmarlo o refutarlo con n=30 en vez de celebrarlo con n=2. Esa disciplina — no una señal secreta — es el marco invisible que separa a los sistemas que sobreviven de los que explotan.

---

## 6. Instrucción de Arranque para Sonnet

```
Implementar Nodo-51 Fase F0 (core/player_registry.py):
1. Leer Nodo-47 completo — el fix de dos pasos existente es la semilla del registro
2. git grep 'normalize_name' — mapear TODOS los puntos de resolución de nombres actuales
3. Escribir tests primero: los 6 jugadores de la tabla de Nodo-47 + 1 ITF desconocido
4. Baseline: 1438 tests deben seguir pasando
5. NO tocar F1-F5 hasta que F0 esté verde
```

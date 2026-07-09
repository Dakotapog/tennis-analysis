# Nodo-60-ADDENDUM-FABLE — Auditoría, Restructura en Tres Carriles y Delegación Detallada

> **Wikilinks:** [[Nodo-60-GCS-Grass-Surface-Champion-Signal]] | [[Nodo-57-Penalizacion-Inactividad-Campeon-Validacion]] | [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-25-Dispersion-Guard-Safe-Combos]] | [[Nodo-58-Dashboard-Observabilidad]] | [[Nodo-46-Markov-Surface-Context-Discount]]
> **Fecha:** 2026-07-05
> **Estado:** 📋 FIRMADO FABLE — IMPLEMENTADO + A60-01 CERRADO 2026-07-05. GCS_MULT GATED. H60-01 acumulando (n=54 settled, hit=64.8%).

---

## §0. Veredicto Ejecutivo

| Componente del Nodo-60 | Veredicto | Razón |
|---|---|---|
| F2 — Separación de universos GCS/GS/ITF en combos | ✅ APROBADO YA | Higiene de portfolio pura; hereda Nodo-25; el 18/18 combos muertos del 04-jul es evidencia real de contagio por mezcla |
| D60-01 — Pre-registro H60-01 | ✅ APROBADO con correcciones (§2) | La hipótesis debe congelarse con criterio Wilson completo y sin cláusulas "opcionales" |
| D60-03/04 — `gcs_active` + visibilidad | ✅ APROBADO YA | Observabilidad; alimenta el shadow book y el Panel 6 |
| **D60-02 — GCS_MULT 2.2×/1.8×/1.5× al final_score** | 🔴 **GATED — flag OFF por default** | Contradicción metodológica (§2), evidencia en contra (§1), diseño arquitectural incorrecto (§3), dependencia D57-03 sin declarar (§4) |

**El principio violado:** el Nodo-52 §6 y el Panel 6 del Nodo-58 existen para que ninguna señal se despliegue antes de graduarse. El Nodo-60 pre-registra H60-01 con n_stop=30 y despliega el boost con n=3 favorable + n=8 desfavorable **el mismo día**. Es la definición operativa de p-hacking con pre-registro decorativo.

---

## §1. Auditoría de la Evidencia — Lo que el 3/3 realmente contiene

**Descomposición del día fundacional (04-jul):**
- Eala @3.80 GANÓ → el único dato genuinamente informativo (underdog real, p_implícita ~26%)
- Bouzkova @1.66 GANÓ → favorita moderada, informativo débil (p_implícita ~60%)
- Krueger @1.19 GANÓ → chalk pesado, p_implícita ~84% — **ganar aquí confirma casi nada**

El "3/3" narrativo es en realidad **1 upset + 1 moderado + 1 trivial**. Y el scan histórico del propio nodo: n=8, hits=3 → 37.5%. Pregunta obligatoria (A60-01): ¿los 3 hits del scan SON los 3 del 04-jul? Si sí, **el patrón nunca ganó fuera del día que lo inspiró** — survivorship bias de libro. Sonnet debe reportar los 8 casos con fecha, cuota, tier del torneo ganado y resultado.

**Inconsistencia interna del guard:** el nodo exige tier≥ATP500 para el torneo ganado, pero Krueger ganó **Ilkley** — que en el circuito es WTA 125/Challenger grass. Si `detectar_tier("Ilkley")` → challenger, el propio guard del nodo **excluye a uno de sus tres casos fundacionales** (A60-02: verificar el mapeo de Birmingham, Nottingham e Ilkley en `detectar_tier()` y reportar). Si la evidencia fundacional se reduce a 2/3 con 1 informativo, la urgencia del boost se evapora sola.

**Nota de coherencia con Nodo-46:** la lógica GCS (campeonato reciente EN LA MISMA superficie merece peso) es la cara complementaria del surface discount (racha reciente en OTRA superficie merece descuento). Direccionalmente coherente — el problema no es la dirección, es la magnitud (2.2×) y el momento (n=3).

### A60-01 — CERRADO 2026-07-05 (entregable: tabla casos históricos)

**Método:** Scan automatizado de 87 archivos `reports/h2h_results_enhanced_*.json` (2026-06-15 a 2026-07-05).  
**Filtros:** superficie=grass|hierba + tier≥ATP500 (via `detectar_tier()`) + cuota real (≠1.9 placeholder) + deduplicado por jugador+torneo+día.

**Resumen:**
| Métrica | Valor |
|---|---|
| Total casos únicos GCS | 76 |
| Settled (resultado conocido) | 54 |
| Sin datos (resultado desconocido) | 22 |
| GANO | 35 |
| PERDIO | 19 |
| **Hit rate** | **64.8% (35/54)** |
| Breakeven típico (cuota ~2.0) | 50% |

**Muestra representativa de casos históricos (pre-04-jul):**

| Fecha | Jugador | Torneo ganado | Tier real | Días | Resultado |
|---|---|---|---|---|---|
| 2026-06-19 | Eala | Birmingham | atp500 | 6 | GANO |
| 2026-06-20 | Golubic | Nottingham | atp500 | 11 | GANO |
| 2026-06-21 | Virtanen | Nottingham | atp500 | 12 | GANO |
| 2026-06-21 | O'Connell | Nottingham | atp500 | 9 | PERDIO |
| 2026-06-23 | Fritz | Halle | atp500 | 14 | GANO |
| 2026-06-24 | Navarro | Bad Homburg | atp500 | 5 | GANO |
| 2026-06-24 | Noskova | Bad Homburg | atp500 | 7 | GANO |
| 2026-06-27 | Krueger | Ilkley (↓atp500) | atp500* | 6 | GANO |
| 2026-06-27 | Bu | Ilkley (↓atp500) | atp500* | 8 | PERDIO |
| 2026-06-28 | Ruse | Bad Homburg | atp500 | 9 | GANO |

*Ilkley → `detectar_tier()` = `atp500` (ver A60-02 abajo).

**Conclusión A60-01:**
- **Survivorship bias: NO confirmado.** Los casos históricos preexisten al 04-jul (empiezan 2026-06-19).
- El n=8/hits=3 (37.5%) era un subscan manual limitado y conservador.
- El scan completo confirma 64.8% hit rate — el patrón es consistente con la hipótesis H60-01.
- La etiqueta "EVIDENCIA ACTUAL EN CONTRA" era incorrecta y se ha corregido en `preregistered_hypotheses.json`.
- GCS_MULT permanece GATED hasta n≥30 en shadow book (prospectivo) y graduación formal H60-01.

**A60-02 — CERRADO (incluido en A60-01):**
- `detectar_tier("Birmingham")` → `atp500` ✅
- `detectar_tier("Nottingham")` → `atp500` ✅
- `detectar_tier("Ilkley")` → `atp500` ✅ (no challenger — guard D57-03 incluye a Krueger/Bu)

---

## §2. Corrección de H60-01 — Congelamiento sin cláusulas elásticas

Reemplazar el pre-registro con esta versión (sin "opcional"):

```json
{
  "id": "H60-01",
  "hipotesis": "Picks con TORNEO_COMPLETO_BONUS validado (D57-03), torneo ganado tier>=atp500,
                dias<=21, MISMA superficie que el partido actual",
  "metrica": "hit% con IC Wilson 95%",
  "n_stop": 30,
  "exito": "limite inferior IC > 1/cuota_media del segmento",
  "corte_secundario_preregistrado": "mismo segmento AND edge_vs_mercado>=0.10 (de-vig, Nodo-53 V2)",
  "estado_inicial": "n=8, hits=3 (37.5%) — EVIDENCIA ACTUAL EN CONTRA. Auditoria A60-01 pendiente.",
  "gated": "GCS_MULT permanece OFF hasta exito=true Y Brier con-boost < sin-boost (Fase-H)"
}
```

El corte secundario (edge≥10%) se congela AHORA como sub-segmento — no se decide después mirando cuál dio mejor. Panel 6 del Nodo-58 muestra la fila: `¿ACTIVAR GCS BOOST? → H60-01: n=8/30, tendencia − → 🔴 NO AUTORIZADO`.

---

## §3. El Diseño Correcto del Boost (para cuando gradúe — no antes)

Dos defectos arquitecturales del multiplicador propuesto:

1. **Magnitud fuera de escala del sistema:** los factores del modelo viven en rangos acotados — Markov 0.85-1.15, immunity 0.85-1.12, alpha temporal 0.85-1.20. Un 2.2× al final_score no es un factor: es un veto que aplasta a los otros 7 componentes. Ningún factor individual del sistema debe poder invertir por sí solo la suma ponderada de todos los demás.
2. **Parche donde no está la causa:** el diagnóstico correcto del propio nodo es que el bonus se DILUYE en la normalización del surface score. La solución de doctorado arregla la dilución en su origen, no multiplica después: dentro de `analyze_surface_specialization`, el partido-campeonato reciente (≤21d, misma superficie, campeón validado D57-03) recibe **ponderación por recencia dentro del cálculo del surface score** (p.ej., los partidos del torneo ganado pesan ×3 en el promedio de superficie antes de normalizar). Así el efecto pasa por la normalización como todos los demás, queda naturalmente acotado, y la señal compite en la misma cancha que el historial en vez de saltársela.

**Cuando H60-01 gradúe:** implementar la versión de ponderación-en-origen con constantes calibradas contra el Brier de los settled (Fase-H), no las GCS_MULT_* actuales. Las constantes 2.2/1.8/1.5 quedan documentadas como propuesta inicial descartada por diseño.

---

## §4. Dependencias Duras (no declaradas en el nodo original)

1. **D57-03 ANTES que cualquier línea de GCS.** El GCS se dispara sobre TORNEO_COMPLETO_BONUS. Sin el gate tier-aware de campeón (GS=7W), Safiullin con 5W en mitad de Wimbledon activaría GCS — un jugador a mitad de torneo recibiría el multiplicador máximo por "haber ganado" un torneo que sigue jugando. Con el boost a 2.2× eso no sería un bug cosmético: sería el generador de phantom picks más potente jamás construido en el sistema. **Verificación obligatoria: T60-06.**
2. **Nodo-52:** `gcs_active` y `gcs_days` deben entrar al `pick_snapshot` del shadow book (el hook copia el dict completo — verificar que rivalry_analyzer los serializa hasta el edge_report).
3. **Nodo-58 Panel 6:** fila H60-01 con estado vivo.
4. **Nodo-25:** MAX_GCS_PER_COMBO=1 es una instancia del concentration guard — implementar con el mismo patrón (pre-filtro, hard limit).

---

## §5. Tareas Detalladas para Sonnet (orden estricto)

```
S60-1  A60-01 — Reportar los 8 casos del scan histórico: fecha, jugador, torneo ganado,
       tier real via detectar_tier(), días, cuota, resultado, y si coinciden con el 04-jul.
       Entregable: tabla en el nodo. [30 min, solo lectura]

S60-2  A60-02 — Verificar detectar_tier() para Birmingham/Nottingham/Ilkley 2026.
       Si Ilkley→challenger: documentar que la evidencia fundacional es 2/3. [15 min]

S60-3  Verificar D57-03 implementado y verde (T57-02/03/04 pasan). Si no: implementarlo
       PRIMERO según Nodo-57. GCS bloqueado hasta esto. [gate duro]

S60-4  D60-01 corregido — H60-01 con el JSON del §2 (sin cláusulas opcionales,
       con estado_inicial honesto y campo gated). [15 min]

S60-5  D60-03/04 — gcs_active + gcs_days + universo en _extract_and_categorize;
       sub-sección GCS en output; MAX_GCS_PER_COMBO=1; combo GCS puro solo si ≥2 picks,
       stake 2% budget. Verificar que gcs_active llega al edge_report y por tanto
       al pick_snapshot del shadow book. [1 sesión]

S60-6  D60-02 REDEFINIDO — implementar GCS_RECENCY_BOOST detrás de flag --gcs-boost
       (default OFF). El código existe, no se ejecuta en producción. Log cuando un pick
       habría recibido boost: "LOG_GCS_SHADOW: {jugador} habría recibido x{mult}".
       Ese log es el A/B gratis: el shadow book acumula qué habría pasado. [1 sesión]

S60-7  Tests T60-01→05 del nodo original (REGLA-T53: invocan módulo) MÁS:
       T60-06: 5W-0L en grand_slam → gcs_active=False (dependencia D57-03)
       T60-07: flag OFF por default → final_score idéntico con/sin código GCS
       T60-08: torneo ganado en clay, partido actual en grass → gcs_active=False
       T60-09: pick GCS + pick ITF → nunca en el mismo CORE combo
       T60-10: LOG_GCS_SHADOW presente cuando el flag está OFF y el pick califica

PROHIBIDO: activar --gcs-boost en producción; tocar GCS_MULT_*; implementar la
ponderación-en-origen del §3 antes de que H60-01 gradúe.
Baseline: todos los tests previos siguen verdes.
```

---

## §6. Checklist de Auditoría Fable (post-implementación)

Cuando Sonnet reporte "terminado", la auditoría verificará exactamente esto:

```
[ ] A60-01 entregado: tabla de 8 casos con tiers reales. ¿Los 3 hits son los del 04-jul?
[ ] A60-02: mapeo de tiers de los 3 torneos fundacionales documentado
[ ] git log muestra D57-03 mergeado ANTES del primer commit GCS
[ ] grep '--gcs-boost' → default OFF verificado en el código, no solo en el doc
[ ] Corrida de producción real: LOG_GCS_SHADOW aparece, final_score NO cambia
    (diff de edge_report con/sin código GCS = solo campos nuevos, ningún valor alterado)
[ ] pick_snapshot de un pick GCS en el shadow book contiene gcs_active/gcs_days
[ ] Panel 6: fila H60-01 en rojo con n=8/30 y tendencia honesta
[ ] T60-06/07/08 pasan invocando funciones del módulo (no literales — REGLA-T53,
    recordar el PASS-permanente del ADDENDUM-3 de Nodo-53)
[ ] H60-01 en preregistered_hypotheses.json con estado_inicial "evidencia en contra"
[ ] Ningún combo de producción del día mezcla GCS con ITF
```

---

## §7. Cierre — Para el Registro

El Nodo-60 detectó algo real en dos niveles: el nivel de portfolio (mezclar universos mató 18 combos — lección genuina, fix aprobado ya) y el nivel de señal (los campeones recientes en superficie *podrían* estar subponderados por la dilución de la normalización — hipótesis legítima, ahora correctamente pre-registrada). Lo que el nodo hizo mal es lo que el proyecto lleva 60 nodos aprendiendo a no hacer: convertir un martes bueno en una constante de 2.2× el mismo día. La versión restructurada conserva el 100% del valor observacional y de portfolio, pone el boost a acumular su propio A/B gratis vía LOG_GCS_SHADOW, y deja que n=30 — no el recuerdo de Eala @3.80 — decida si la señal merece entrar al modelo. Si gradúa, entra bien diseñada (§3). Si no gradúa, el sistema acaba de ahorrarse su próximo Nodo-32.

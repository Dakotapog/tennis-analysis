# DECISION-LOG.md — Porqué detrás de los umbrales

> **Nodo:** [[Nodo-59-Motor-Agentico-Odometro-Dream]]
> **Propósito:** Documenta el PORQUÉ de cada decisión de diseño no obvia. Sin este log, las decisiones parecen arbitrarias → se revierten → los bugs vuelven.
> **Formato:** Decisión → Alternativas consideradas → Por qué elegimos esta → Cuándo revisitar

---

## D-01 — λ_ITF = 4.5 (Nodo-21)

**Decisión:** El multiplicador de incertidumbre Kullback-Leibler para ITF es 4.5× (más restrictivo que GS=1.0×).

**Alternativas consideradas:**
- λ_ITF = 2.5× (más permisivo, más stakes)
- λ_ITF = 6.0× (casi prohibitivo)

**Por qué 4.5×:** El bookmaker tiene menos datos históricos en ITF → mayor ventaja informacional para el modelo → PERO también mayor varianza por n pequeño de calibración. 4.5× balancea explorar el alpha ITF sin ruina por sobreconfianza. Validado: sesión 9/10 del 13-jun, ITF hit 82.8%.

**Cuándo revisitar:** cuando n_ITF_shadow ≥ 30 eventos settled en shadow book. Actualmente epoch-2 post-2026-07-01: n=0.

---

## D-02 — James-Stein Shrinkage con factor n/(n+20) (Nodo-21)

**Decisión:** El floor de calibración es `n/(n+20)`. Con n=4 → 16.7% kelly. Con n=33 → 62.1%.

**Alternativas consideradas:**
- Floor fijo de 50% (simple pero ignorante del n real)
- n/(n+50) (más conservador en n bajos)

**Por qué n+20:** El "pseudo-count" de 20 equivale a la certeza de tener 20 datos históricos como prior. Clay GS con n=31 da factor=0.61 — razonable para apostar con convicción media. Si n=4, el factor 0.17 prácticamente zeroa el stake — correcto, es ruido estadístico.

**Cuándo revisitar:** nunca sin evidencia de que el floor actual produce P&L sistemáticamente diferente al Kelly completo.

---

## D-03 — Grace period inactividad = 30 días (Nodo-57)

**Decisión:** `form_decay_factor(days) = 1.0` para days ≤ 30. La inactividad no penaliza forma hasta 30 días.

**Alternativas consideradas:**
- Grace = 14 días (más agresivo, penaliza pausas transición arcilla→hierba)
- Grace = 45 días (demasiado permisivo con inactividades largas)

**Por qué 30 días:** Las pausas de transición ATP (arcilla→hierba: 2-3 semanas; hierba→dura: 3-4 semanas) caen dentro del grace. Penalizar por descanso programado es penar una característica del circuito, no un dato de forma real. Los 4 marcos de expertos (Nodo-57 §1) convergieron en este umbral.

**Cuándo revisitar:** si n≥50 observaciones de jugadores con days_since=25-35 muestran hit rate materialmente diferente al baseline.

---

## D-04 — _MIN_WINS_CHAMPION GS = 7 (Nodo-57)

**Decisión:** Un campeón de Grand Slam debe tener ≥7 victorias en el cuadro principal (sin contar qualifying).

**Alternativas consideradas:**
- 5 victorias (cuartos de final) — demasiado bajo, Safiullin demostró el bug
- 6 victorias (semifinalista) — semifinales ≠ campeón
- 7 victorias (CORRECTO: R1→R7 del cuadro principal)

**Por qué 7:** Wimbledon tiene 128 jugadores en el cuadro principal → 7 rondas para ser campeón. Safiullin tenía 5W (3 qualifying + 2 main draw) en Wimbledon 2026 cuando el sistema le asignó bonus×1.6 erróneamente. El gate `wins>=4` original era tier-agnóstico. Fix: `_MIN_WINS_CHAMPION['grand_slam']=7`.

**Cuándo revisitar:** si ATP cambia el formato de GS (improbable). No revisitar por temporada.

---

## D-05 — Density confidence [0.3, 1.0] para common_opponents (Nodo-21)

**Decisión:** La densidad local modula el peso de rivales comunes entre 30% y 100% de su peso base.

**Por qué 30% de floor:** Si n_common=0, los rivales comunes no aportan señal — pero el modelo no puede ignorarlos completamente porque el grafo transitivo puede tener caminos indirectos. El 30% mantiene la señal viva pero reducida.

**Cuándo revisitar:** cuando el PageRank Erdős (Nodo-20) tenga suficientes datos para calibrar la conversión de centralidad a probabilidad.

---

## D-06 — Shadow book como única fuente de verdad CLV (Nodo-52)

**Decisión:** El shadow book (`reports/shadow_book/sb_YYYY-MM-DD.jsonl`) es append-only e inmutable en predicciones.

**Por qué append-only:** La manipulación retroactiva de registros invalida el backtest estadístico. La hipótesis H52-01→H52-08 requieren que los picks estén pre-registrados antes de conocer el resultado. Sin append-only, el humano puede (inconscientemente) "limpiar" picks malos.

**Cuándo revisitar:** nunca. Es una regla estructural de integridad estadística, no un parámetro.

---

## D-07 — Routing M0/M1/M2 de Nodo-59 (Nodo-59)

**Decisión:** El Dream automático (detección de secuencias repetitivas) es M2 — solo post-primer ciclo de 30 días del shadow book.

**Por qué M2:** El mayor riesgo del motor agéntico es sí mismo — un meta-proyecto seductor que compite con la disciplina diaria. El shadow book necesita 30 días para tener n≥30 hipótesis con valor estadístico. Construir M2 ahora es exactamente el patrón "nuevo framework > trabajo aburrido" que el proyecto documenta.

**Cuándo revisitar:** cuando shadow book epoch-2 tenga n≥30 settled (estimado: ~2026-08-01).

---

## Plantilla para nuevas decisiones

```
## D-XX — [Nombre de la decisión] (Nodo-YY)

**Decisión:** [qué se decidió]

**Alternativas consideradas:**
- [alt A] — [por qué no]
- [alt B] — [por qué no]

**Por qué esta:** [razonamiento concreto, con datos si existen]

**Cuándo revisitar:** [condición específica y medible, no "cuando sea necesario"]
```

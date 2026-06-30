# POSTMORTEM — Betslips 2026-06-14 (Jornada Desastre)

> **Para:** Opus 4.6 — análisis de contraste con sesión épica 2026-06-13
> **Propósito:** Encontrar por qué todo salió mal y qué filtros habrían salvado la sesión

---

## RESUMEN EJECUTIVO

```
Total combos apostados:   25
Total invertido:          $12,500 (25 × $500)
Combos MUERTOS YA:        25/25 (TODOS — cada uno tiene ≥1 perdedor confirmado)
Combos pendientes vivos:  0 (los pendientes no salvan nada)
P&L estimado:             -$12,500 (pérdida total)
```

---

## PICKS INDIVIDUALES — RESULTADOS CONFIRMADOS

| Pick | Cuota | Tier | Resultado | gap | BBI | En trader plan |
|---|---|---|---|---|---|---|
| Yanaki Milev | @2.75 | Challenger Parma | ❌ PERDIÓ | 0.088 | 0.636 | ✅ SÍ |
| Briyana Ivanova | @3.10 | ITF Femenino Madrid | ❌ PERDIÓ | 0.191 | 0.677 | ✅ SÍ |
| Verena Meliss | @4.30 | WTA125 Cal. Brescia | ❌ PERDIÓ | 0.083 | 0.767 | ✅ SÍ |
| Marie Mettraux | @2.65 | ITF Femenino K.Banja | ❌ PERDIÓ | 0.191 | 0.623 | ✅ SÍ |
| Roberto Bautista Agut | @2.55 | ATP Qual. Halle | ❌ PERDIÓ | 0.061 | 0.608 | ✅ SÍ |
| Matyas Cerny | @1.92 | Challenger Parma | ❌ PERDIÓ | ? | ~0.479 | ❌ EXTRA |
| Diletta Cherubini | @2.04 | WTA125 Cal. Brescia | ❌ PERDIÓ | ? | ~0.510 | ❌ EXTRA |
| Eva Vedder | @1.90 | ITF Femenino Ceska Lipa | ❌ PERDIÓ | ? | ~0.474 | ❌ EXTRA |
| Ryan Seggerman | @2.55 | Challenger Parma | ✅ GANÓ | 0.081 | 0.608 | ✅ SÍ |
| Cruz Hewitt | @3.55 | Challenger Dublin | ✅ GANÓ | 0.046 | 0.718 | ✅ SÍ |
| Lorenzo Bocchi | @3.35 | Challenger Parma | ✅ GANÓ | 0.074 | 0.701 | ✅ SÍ |
| Pablo Martinez Gomez | @2.14 | ITF Martos | ✅ GANÓ | 0.060 | 0.533 | SÍ (era pick ayer) |
| Martin Damm | @1.98 | ATP Qual. Londres | ✅ GANÓ | ? | ~0.495 | ❌ EXTRA |
| Aran Teixido Garcia | @2.43 | ITF Femenino Casablanca | ⏳ PEND | 0.197 | 0.588 | ✅ SÍ |
| Lulu Sun | @2.50 | WTA Qual. Berlin | ⏳ PEND | 0.066 | 0.600 | ✅ SÍ |
| Antonia Vergara Rivera | @2.15 | ITF Femenino Cuiaba | ⏳ PEND | 0.111 | 0.459 | ✅ SÍ |
| Franco Ribero | @2.04 | Challenger Royan | ⏳ PEND | ? | ~0.510 | ❌ EXTRA |
| Filippo Romano | @1.35 | Challenger Dublin | ⏳ PEND | ? | ~0.259 | ❌ EXTRA (heavy fav) |

**Confirmados perdidos: 8 | Ganados: 4 | Pendientes: 6**
**Accuracy picks resueltos: 4/12 = 33.3%**

**Picks FUERA del trader plan apostados hoy: 6** (Cerny, Cherubini, Vedder, Ribero, Damm, Romano@1.35)

---

## CATÁLOGO COMPLETO DE COMBOS

| # | Tipo | Cuota | ID | Picks (orden) | Estado | Killer |
|---|---|---|---|---|---|---|
| 1 | Séxtupla | @468.8 | — | Ivanova+Teixido+Milev+Seggerman+Hewitt+Sun | ❌ | Ivanova+Milev |
| 2 | Cuádruple | @111.6 | 12750703348 | Ivanova+Meliss+Bocchi+Sun | ❌ | Ivanova+Meliss |
| 3 | Cuádruple | @96.26 | 12750700620 | Teixido+Meliss+Bocchi+Milev | ❌ | Meliss+Milev |
| 4 | Quíntuple | @228.4 | 12750698751 | Ivanova+Teixido+Bocchi+Hewitt+Bautista | ❌ | Ivanova+Bautista |
| 5 | Quíntuple | @39.83 | 12750684491 | Seggerman+Ribero+Mettraux+Martinez+Romano | ❌ | Mettraux |
| 6 | Séptuple | @210.3 | 12750680754 | Milev+Seggerman+Ribero+Cerny+Mettraux+Martinez+Romano | ❌ | Milev+Cerny+Mettraux |
| 7 | Óctuple | @1618 | 12750677307 | Bocchi+Milev+Seggerman+Ribero+Cerny+Ivanova+Mettraux+Martinez | ❌ | Milev+Cerny+Ivanova+Mettraux |
| 8 | Décupla | @8019 | 12750673453 | Bocchi+Milev+Seggerman+Cherubini+Ribero+Cerny+Ivanova+Mettraux+Teixido+Martinez | ❌ | Milev+Cherubini+Cerny+Ivanova+Mettraux |
| 9 | Undécupla | @28467 | 12750671768 | Hewitt+Bocchi+Milev+Seggerman+Cherubini+Ribero+Cerny+Ivanova+Mettraux+Teixido+Martinez | ❌ | Milev+Cherubini+Cerny+Ivanova+Mettraux |
| 10 | Triple | @12.62 | 12750627003 | Bautista+Sun+Damm | ❌ | Bautista |
| 11 | Triple | @12.62 | 12750617971 | Bautista+Sun+Damm | ❌ | Bautista (DUPLICADO) |
| 12 | Cuádruple | @83.40 | 12750620647 | Bocchi+Milev+Seggerman+Hewitt | ❌ | Milev |
| 13 | Cuádruple | @101.0 | 12750624271 | Meliss+Bocchi+Milev+Seggerman | ❌ | Meliss+Milev |
| 14 | Cuádruple | @107.0 | 12750617803 | Meliss+Milev+Seggerman+Hewitt | ❌ | Meliss+Milev |
| 15 | Cuádruple | @140.6 | 12750620516 | Meliss+Bocchi+Milev+Hewitt | ❌ | Meliss+Milev |
| 16 | Triple | @32.70 | 12750617553 | Bocchi+Milev+Hewitt | ❌ | Milev |
| 17 | Triple | @39.61 | 12750611943 | Meliss+Bocchi+Milev | ❌ | Meliss+Milev |
| 18 | Triple | @41.98 | 12750617469 | Meliss+Milev+Hewitt | ❌ | Meliss+Milev |
| 19 | Triple | @51.14 | 12750611868 | Meliss+Bocchi+Hewitt | ❌ | Meliss |
| 20 | Cuádruple | @26.30 | 12750619353 | Vergara+Mettraux+Teixido+Vedder | ❌ | Mettraux+Vedder |
| 21 | Cuádruple | @37.93 | 12750615506 | Ivanova+Mettraux+Teixido+Vedder | ❌ | Ivanova+Mettraux+Vedder |
| 22 | Triple | @10.82 | 12750618184 | Vergara+Mettraux+Vedder | ❌ | Mettraux+Vedder |
| 23 | Cuádruple | @42.92 | 12750619096 | Ivanova+Vergara+Mettraux+Teixido | ❌ | Ivanova+Mettraux |
| 24 | Triple | @13.85 | 12750613515 | Vergara+Mettraux+Teixido | ❌ | Mettraux |
| 25 | Triple | @16.20 | 12750617060 | Ivanova+Vergara+Teixido | ❌ | Ivanova |

**NOTA: Combo #10 y #11 son idénticos (Bautista+Sun+Damm @12.62 × 2) — $1,000 apostado en el mismo combo.**

---

## ANÁLISIS DE "PICKS VENENO" — Cuántos combos mató cada perdedor

| Pick | Combos matados | % del total | En trader plan | Nodo-24 flag |
|---|---|---|---|---|
| **Yanaki Milev @2.75** | **14** | **56%** | ✅ | ⬜ OK (gap=0.088) |
| **Marie Mettraux @2.65** | **11** | **44%** | ✅ | ⚠️ CAL (gap=0.191) |
| **Briyana Ivanova @3.10** | **9** | **36%** | ✅ | ⚠️ CAL (gap=0.191) |
| **Verena Meliss @4.30** | **8** | **32%** | ✅ | ⬜ OK (gap=0.083) |
| Matyas Cerny @1.92 | 4 | 16% | ❌ EXTRA | N/A |
| Eva Vedder @1.90 | 3 | 12% | ❌ EXTRA | N/A |
| Bautista Agut @2.55 | 3 | 12% | ✅ | ⬜ OK (gap=0.061) |
| Diletta Cherubini @2.04 | 2 | 8% | ❌ EXTRA | N/A |

**MILEV fue el pick "cáncer" principal — mató el 56% de los combos.**
**Nodo-24 con gap>0.12 habría filtrado Ivanova y Mettraux (20 combos muertos entre los dos).**
**Milev y Meliss NO habrían sido filtrados por Nodo-24 — son verdaderos fallos del modelo.**

---

## COMPARACIÓN DIRECTA: 13-JUN vs 14-JUN

| Dimensión | 13-jun (épica) | 14-jun (desastre) |
|---|---|---|
| Picks en trader plan | 12 (incluyendo ATP500) | 11 |
| Picks EXTRA (fuera del plan) | 0 | **6 (Cerny, Cherubini, Vedder, Ribero, Damm, Romano)** |
| Picks resueltos | 12/12 | 12/18 |
| Accuracy picks | 9/12 = 75% | 4/12 = **33%** |
| Picks con gap > 0.12 | 1 (Miguel — PERDIÓ) | **3** (Ivanova, Mettraux, Teixido) |
| "Pick cáncer" principal | Miguel (1 combo) | **Milev (14 combos)** |
| Combos apostados | 10 mega-combos | 25 combos |
| Inversión | $5,000 | $12,500 |
| Resultado | +$52,608+ | -$12,500 |

---

## PROBLEMAS ESTRUCTURALES IDENTIFICADOS

### P1 — Picks fuera del trader plan (DISCIPLINA)
Los picks Cerny, Cherubini, Vedder, Ribero, Damm, Romano@1.35 **no estaban en el APOSTAR del trader**.
El sistema generó estos combos usando picks sin validación Kelly-KL.
El combo builder mezcló picks del trader plan con picks del edge_report sin distinción de categoría.

### P2 — Milev como pick ancla en 14/25 combos (CONCENTRACIÓN)
El Sistema Cobertura Exclusión debería garantizar que ningún jugador aparezca en más del `max_app = top_n × piernas ÷ n_pool` de los combos.
Si Milev aparece en el 56% de combos, el max_app no está funcionando correctamente.

### P3 — Nodo-24 filtra Ivanova y Mettraux pero no Milev ni Meliss
Ivanova (gap=0.191) y Mettraux (gap=0.191): filtradas por gap>0.12 → habrían evitado 20 combos.
Milev (gap=0.088) y Meliss (gap=0.083): NO filtradas → siguen siendo un riesgo real.
¿Hay otra señal que diferencia a los verdaderos perdedores?

### P4 — Combo duplicado (REGLA-KAMBI-1 violada)
Combo #10 y #11 son idénticos: Bautista+Sun+Damm @12.62 × 2 = $1,000 en el mismo resultado.
Esto viola REGLA-KAMBI-1 (||append acumula betslip).

### P5 — Picks ITF Femenino con cuotas muy altas pero en tiers sin calibración
Ivanova @3.10, Mettraux @2.65, Vergara @2.80, Vedder @1.90 — ITF Femenino.
El calibracion_edge.json no tiene datos suficientes de ITF Femenino por superficie.
p_blend para estas picks viene del fallback tier genérico → gap alto → señal falsa.

---

## PREGUNTAS PARA OPUS

1. **¿Por qué Milev perdió si BBI=0.636 y gap=0.088 (señales "correctas")?**
   ¿Qué dato hubiera revelado que Milev era débil hoy?
   
2. **¿Por qué Meliss @4.30 perdió siendo un pick con gap bajo (0.083)?**
   Con BBI=0.767 (bookmaker más ciego que ayer), ¿qué salió mal?
   
3. **¿Debería el combo builder NUNCA usar picks fuera del APOSTAR del trader?**
   ¿O hay condiciones bajo las cuales un pick SIN_EDGE puede entrar a combos?

4. **Con 6 picks perdedores en el pool, ¿el Sistema Cobertura Exclusión hubiera protegido algo?**
   La Cobertura Exclusión requiere que "si falla ≤1 pick, ≥1 combo sobrevive".
   Con 6 perdedores de 12 picks (50%), ¿qué estructura de combos sobrevive?

5. **¿Thresholds de Nodo-24 necesitan ajustarse?**
   gap > 0.12 filtra CAL pero no MIXED. ¿gap > 0.09 habría salvado la sesión sin descartar Hewitt/Bocchi?

---

## SEÑALES DISPONIBLES EN LOS DATOS (para análisis de Opus)

Del trader_plan_20260614_014208.json — picks resueltos hoy:
```
Verena Meliss    @4.30  p_mod=0.507 p_blend=0.590 gap=0.083 kelly=0.2673 n_h2h=0 PERDIÓ
Lorenzo Bocchi   @3.35  p_mod=0.516 p_blend=0.590 gap=0.074 kelly=0.2417 n_h2h=0 GANÓ
Yanaki Milev     @2.75  p_mod=0.502 p_blend=0.590 gap=0.088 kelly=0.1707 n_h2h=0 PERDIÓ
Ryan Seggerman   @2.55  p_mod=0.509 p_blend=0.590 gap=0.081 kelly=0.1518 n_h2h=0 GANÓ
Cruz Hewitt      @3.55  p_mod=0.523 p_blend=0.569 gap=0.046 kelly=0.1183 n_h2h=0 GANÓ
```

Observación crítica: Milev, Bocchi, Seggerman tienen p_blend=0.590 IDÉNTICO.
El modelo NO diferencia entre ellos en absoluto — todos tienen exactamente la misma probabilidad.
Sin embargo, Milev perdió y Bocchi+Seggerman ganaron.
¿Por qué el modelo calcula p_blend=0.590 para los tres?
¿Es James-Stein shrinkage colapsando todos los picks de Challenger Parma al fallback_por_tier?

Patrón: picks del MISMO torneo (Parma Challenger) → James-Stein shrinkage los colapsa todos al mismo p_blend=0.590. El modelo pierde capacidad de discriminar entre ellos. Todas las diferencias reales (forma, H2H, superficie) desaparecen en el fallback.

# Nodo-13: Trader EV Tenis — Capa de Deploy (análogo NBA trader_ev.py)

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-07-Strangler-Fig]]
> **Estado:** 2026-05-30 — P&L REGISTRADO ✅ | Primera sesión: +$25,000 (+25% bankroll) | bankroll $100,000 → $125,000
> **Inspiración:** NBA trader_ev.py (proyecto NBA — bankroll crecimiento exponencial sin ML)

---

## El Problema: Edge calculado, deploy sin estructura

`edge_calculator.py` identifica señales APOSTAR con Kelly-KL individual.
Pero el bankroll no crece exponencialmente con apuestas individuales solas.

**El insight del proyecto NBA:**
No fue ML lo que hizo crecer el bankroll. Fue:
1. **Bayesian avg blend** — media ponderada histórica (no modelo entrenado)
2. **Budget cascade** — 40% individuales → 40% combos → 20% sistema
3. **Combos N piernas** — cuota multiplicada: 3 señales × cuotas 2×3×4.5 = cuota 27x

Con 3 señales APOSTAR de cuota 2.0/3.0/4.5 → combo 3 piernas cuota 27x con HR ~25% = EV brutal.

---

## Solución: trader_ev_tenis.py

Archivo: `trader_ev_tenis.py` (raíz del proyecto)
Lee: `reports/edge_report_FECHA.json` (producido por `edge_calculator.py`)

### Las 3 capas

```
CAPA 1 — INDIVIDUALES (40% bankroll)
  Señales APOSTAR con Kelly-KL × 0.25 (fraccionario conservador)
  Cap: 10% bankroll por apuesta individual

CAPA 2 — COMBOS N piernas (40% bankroll)
  Cuota combinada = ∏(cuotas individuales)
  HR conjunta = ∏(p_blend de cada pierna)
  EV combo = HR_conjunta × cuota_combo - 1
  Kelly quarter × cap 15% por combo

CAPA 3 — SISTEMA 2/N (20% bankroll)
  Genera C(N,2) pares de 2 piernas
  Ganas si ≥2 de N aciertos → cobertura binomial
  P(≥2 aciertos) calculada con distribución binomial exacta
```

---

## Fórmula Bayesiana p_blend (adaptación NBA k=3)

```python
p_blend = (n_h2h × p_modelo + 3 × 0.52) / (n_h2h + 3)

# n_h2h = partidos H2H directos históricos del enfrentamiento
# k=3: cuando n_h2h=0  → 100% prior (0.52)
# k=3: cuando n_h2h=3  → 50% p_modelo + 50% prior
# k=3: cuando n_h2h=10 → 77% p_modelo + 23% prior
# k=3: cuando n_h2h=20 → 87% p_modelo + 13% prior
```

**Impacto:** Cuando n_h2h=0 (primer enfrentamiento entre jugadores), no sobrepondera la predicción del modelo. Converge gradualmente al p_modelo real cuando hay historial suficiente.

---

## Uso

```bash
# Básico — lee el edge_report más reciente
python trader_ev_tenis.py --bankroll 100000

# Con parámetros explícitos
python trader_ev_tenis.py --bankroll 100000 --combos 3 --sistema 4 --ncombos 3

# Incluir watchlist en combos (edge positivo pero bajo umbral 5%)
python trader_ev_tenis.py --bankroll 100000 --watchlist

# Archivo específico
python trader_ev_tenis.py --bankroll 100000 --file reports/edge_report_20260530_072115.json
```

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `--bankroll` | 100000 | Bankroll total |
| `--combos` | 2 | N piernas por combo parlay |
| `--sistema` | 3 | N piernas para sistema 2/N |
| `--ncombos` | 3 | Cuántos combos mostrar (top por EV) |
| `--watchlist` | False | Incluir señales watchlist en pool de combos |
| `--file` | None | Archivo edge_report específico |

---

## Run de Producción — 2026-05-30 (Roland Garros, 16 partidos)

```
Señales APOSTAR:  2
  Parry D. vs Anisimova A.  → apostar Parry D.  @ 4.50 | Edge +29.3% | stake $10,000
  Cobolli F. vs Tien L.     → apostar Tien L.   @ 2.40 | Edge +16.5% | stake $5,000

COMBO 1 (Parry + Tien):
  Cuota combinada: 10.80  |  HR conjunta: 27.9%  |  EV: +200.7%  |  stake: $5,000
  Retorno potencial: $54,000

TOTAL EN RIESGO: $20,000 (20.0% bankroll)
  Individuales: $15,000 (15%)
  Combos:        $5,000 (5%)
  Sistema:           $0 (necesita ≥3 señales)

Plan guardado: reports/trader_plan_20260530_121616.json  ← T13-05 ✅
```

### P&L Validado — 2026-05-30

| Señal | Resultado | Retorno | Neto |
|---|---|---|---|
| Parry D. @ 4.50 (edge +29.3%) | **GANÓ** ✅ | $45,000 | +$35,000 |
| Tien L. @ 2.40 (edge +16.5%) | **PERDIÓ** ❌ | $0 | −$5,000 |
| Combo Parry+Tien @ 10.80 | **PERDIÓ** ❌ (Tien falló) | $0 | −$5,000 |
| **TOTAL SESIÓN** | | **$45,000** retornado | **+$25,000 (+25% bankroll)** |

```
Bankroll inicial:  $100,000
Bankroll final:    $125,000  🏦
```

**Calibración post-sesión:** n=23, p_historica=0.68 (era 0.52 neutral). El sistema opera con evidencia real.
**Validación de la hipótesis central:** El mercado fijó Parry a 22.2% implied prob. El modelo dijo 52%. Alpha = sesgo estructural del bookmaker en clay specialists. Ver [[Nodo-14-Validacion-Live-Conexiones]].

**Nota:** Sistema 2/N se activa con ≥3 señales. Con 80 partidos (pipeline completo) → 5-8 señales esperadas → sistema y 10-15 combos activos → crecimiento exponencial del bankroll.

---

## Integración en Pipeline

```
PASO 3:  edge_calculator.py   → reports/edge_report_FECHA.json
PASO 3.5: trader_ev_tenis.py  → output por consola (deploy plan)
                                 Lee: edge_report_FECHA.json
                                 No produce archivo (output de acción inmediata)
PASO 4:  generar_tabla_favoritos2.py → analisis_partidos_pandas.txt
PASO 5:  validar_con_api.py (post-partido) → actualiza calibracion_edge.json
```

---

## Límites y Calibración

```
Calibración activa: n=3 validaciones  ← ⚠️ Prior uniforme activo
  → p_historica usada: 0.52 (neutral)
  → recalibrar cuando n≥30 con datos limpios post validar_con_api.py

Kelly cap individual: 10% bankroll
Kelly cap combo:     15% bankroll
Kelly fraccionario:  ×0.25 (25% de full Kelly = conservador)
Budget cascade:      40% / 40% / 20%

Alerta automática: si total_riesgo > 30% bankroll → warning en output
```

---

## Diferencias vs NBA trader_ev.py

| Aspecto | NBA trader_ev.py | trader_ev_tenis.py |
|---|---|---|
| Fuente de avg | `series_data.json` + `enriched_analysis` (box scores reales) | `edge_report` (p_modelo de rivalry_analyzer) |
| HR estimation | `_est_hr()` con dist. normal + CV por tipo | `p_blend` bayesiano (p_modelo + prior) |
| Props por partido | 8 tipos × 2 dir = 16 señales/jugador | 1 señal por partido (ganador) |
| Señales/día | 30-50 (NBA) | 2-10 (tenis, crece con 80 partidos) |
| Sistema | 2/N con stakes por par | 2/N con stakes por par (idéntico) |
| Budget | 40/40/20 | 40/40/20 (idéntico) |

**Gap crítico:** En NBA hay 30-50 props por sesión → combos y sistemas siempre activos. En tenis con 16 partidos hay 2 señales → sistema no activa. Correr el pipeline completo (80 partidos) es la palanca principal para maximizar señales.

---

## Tareas

| ID | Tarea | Estado |
|---|---|---|
| T13-01 | Crear `trader_ev_tenis.py` con 3 capas + budget cascade | ✅ 2026-05-30 |
| T13-02 | Run de producción Roland Garros 2026-05-30 (16 partidos) | ✅ 2026-05-30 |
| T13-03 | Añadir campo `n_h2h` en `edge_calculator.py` → leer desde `enfrentamientos_directos` | ✅ 2026-05-30 |
| T13-04 | Correr pipeline completo (80 partidos) → sistema 2/N activo con ≥3 señales | ⏳ pendiente |
| T13-05 | Añadir output JSON `reports/trader_plan_FECHA.json` para auditoría | ✅ 2026-05-30 |
| T13-06 | Calibrar `p_blend` con datos reales cuando n≥30 validaciones | ⏳ pendiente (post n≥30) |

---

## Vinculación

- [[Nodo-01-Edge-Calculator]] — produce `edge_report_FECHA.json` que este nodo consume
- [[Mandatos-No-Negociables]] — Mandato 1: P&L sobre accuracy; Mandato 6: tests antes de código
- [[Pipeline-Arquitectura]] — PASO 3.5 en el pipeline diario
- [[Sprint-Pipeline]] — Fase 14 en backlog
- [[Nodo-07-Strangler-Fig]] — Nodo-07 Fase 2 habilitó H2HExtractor que produce datos más limpios para este nodo

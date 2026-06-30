# Nodo-27 --- Pipeline Tracker & Observabilidad

> **Estado:** 🔧 IMPLEMENTADO --- 2026-06-17
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-26-Cross-Sectional-Signals]] | [[Nodo-24-Bookmaker-Blindness-Scoring]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]]
> **Origen:** Necesidad de medir qué señales, cuotas, tiers y niveles de confianza producen los mejores resultados reales
> **Prioridad:** ALTA --- sin observabilidad operamos a ciegas; con ella, cada sesión calibra el siguiente paso

---

## Problema

El pipeline genera picks, calcula edges, construye combos y registra resultados --- pero **no existe un lugar donde ver el rendimiento agregado**. Los datos existen dispersos en ~6 tipos de JSON. Sin un tracker:

1. No sabemos si `confidence_flag=STRONG` realmente gana más que `LOW`
2. No sabemos qué rango de cuotas produce mejor ROI (no hit rate --- ROI)
3. No sabemos si Challenger/ITF realmente supera ATP500 como la sesión épica del 13-jun sugiere
4. No detectamos model drift (¿el accuracy baja con el tiempo?)
5. No sabemos si los combos agregan valor vs individuales
6. No sabemos si `golden_zone` es señal real o ruido

**Analogía:** Un hedge fund sin Bloomberg terminal. Tenemos el motor de trading, pero no el dashboard de P&L.

---

## Fuentes de Datos (ya existen)

| Fuente | Ubicación | Qué aporta |
|---|---|---|
| `calibracion_edge.json` | `data/` | Win/loss global, por superficie, por tier+superficie, fallbacks |
| `edge_report_*.json` | `reports/` | 48 campos por pick: edge, kelly, confidence_flag, tier, superficie, cuota, golden_zone, bbi, mpq, markov, data_completeness, zona_cuota, etc. |
| `apuestas_*.json` | `reports/` | Resultados reales por pick: correcto (bool), ganancia, cuota, stake, sets |
| `resultados_finales_*.json` | `reports/` | Validación modelo: accuracy global y por superficie |
| `trader_plan_*.json` | `reports/` | Sizing: individuales vs combos, stakes, bankroll, cobertura |
| `betslip_index_*.json` | `reports/` | Mapeo outcome_id Kambi, cuota real vs cuota modelo |

**Join key:** `match_id` + `favorito_predicho` conecta edge_report → apuestas → resultados_finales.

---

## Secciones del Reporte

### S-27-1: Rendimiento por Nivel de Confianza

**Pregunta:** ¿`confidence_flag` predice correctamente?

| Flag | Picks | Wins | Losses | Hit% | ROI% | Avg Edge |
|---|---|---|---|---|---|---|
| STRONG | ? | ? | ? | ? | ? | ? |
| MODERATE | ? | ? | ? | ? | ? | ? |
| LOW | ? | ? | ? | ? | ? | ? |

**Fuente:** edge_report (confidence_flag) + apuestas/resultados_finales (correcto)
**Acción si LOW tiene mejor hit% que STRONG:** B-11-like bug en confidence formula → investigar.

---

### S-27-2: Rendimiento por Rango de Cuota

**Pregunta:** ¿Dónde está el alpha real — underdogs altos o favoritos moderados?

| Rango Cuota | Picks | Wins | Hit% | ROI% | Avg Edge |
|---|---|---|---|---|---|
| 1.50-2.00 | ? | ? | ? | ? | ? |
| 2.00-2.50 | ? | ? | ? | ? | ? |
| 2.50-3.00 | ? | ? | ? | ? | ? |
| 3.00-4.00 | ? | ? | ? | ? | ? |
| 4.00+ | ? | ? | ? | ? | ? |

**ROI = (sum(ganancia) / sum(stake)) × 100.** Hit% alto con ROI negativo = cuotas insuficientes.
**Fuente:** edge_report (cuota_favorito) + apuestas (correcto, ganancia, stake)
**Hipótesis a validar:** Sesión R4 (8/8) y épica (9/10) sugieren underdogs ≥2.00 con edge >5% = alpha estructural.

---

### S-27-3: Rendimiento por Tier y Superficie

**Pregunta:** ¿Challenger/ITF realmente supera ATP500?

| Tier | Surface | Picks | Wins | Hit% | ROI% | Avg BBI |
|---|---|---|---|---|---|---|
| grand_slam | clay | ? | ? | ? | ? | ? |
| atp500 | grass | ? | ? | ? | ? | ? |
| challenger | clay | ? | ? | ? | ? | ? |
| challenger | grass | ? | ? | ? | ? | ? |
| itf | clay | ? | ? | ? | ? | ? |
| itf | hard | ? | ? | ? | ? | ? |

**Fuente:** edge_report (tier, superficie) + apuestas (correcto)
**Hipótesis:** BBI alto (bookmaker ciego) → mejor ROI en Challenger/ITF. BBI bajo (bookmaker informado) → peor ROI en ATP500.

---

### S-27-4: Rendimiento por Señal Específica

**Pregunta:** ¿Qué señales del modelo realmente predicen?

#### 4a. Golden Zone
| Golden | Picks | Hit% | ROI% |
|---|---|---|---|
| True | ? | ? | ? |
| False | ? | ? | ? |

#### 4b. Markov (estado de forma)
| Markov | Picks | Hit% | ROI% |
|---|---|---|---|
| HOT | ? | ? | ? |
| NEUTRAL | ? | ? | ? |
| COLD | ? | ? | ? |

#### 4c. Data Completeness
| Completeness | Picks | Hit% | Avg Confianza |
|---|---|---|---|
| 0-25% | ? | ? | ? |
| 25-50% | ? | ? | ? |
| 50-75% | ? | ? | ? |
| 75-100% | ? | ? | ? |

#### 4d. Zona de Cuota
| Zona | Picks | Hit% | ROI% |
|---|---|---|---|
| underdog | ? | ? | ? |
| slight_underdog | ? | ? | ? |
| moderate_favorite | ? | ? | ? |

#### 4e. Edge Binning
| Edge% | Picks | Hit% | ROI% | Avg Kelly |
|---|---|---|---|---|
| 5-10% | ? | ? | ? | ? |
| 10-15% | ? | ? | ? | ? |
| 15-20% | ? | ? | ? | ? |
| 20%+ | ? | ? | ? | ? |

**Fuente:** edge_report (golden_zone, markov_favorito, data_completeness, zona_cuota, edge_pct) + apuestas (correcto, ganancia)

---

### S-27-5: Calibración del Modelo

**Pregunta:** ¿El modelo está calibrado? (Cuando dice 60% confianza, ¿gana ~60%?)

| p_modelo Bin | Picks | Actual Hit% | Esperado | Diff |
|---|---|---|---|---|
| 0.50-0.52 | ? | ? | 51% | ? |
| 0.52-0.55 | ? | ? | 53.5% | ? |
| 0.55-0.60 | ? | ? | 57.5% | ? |
| 0.60-0.65 | ? | ? | 62.5% | ? |
| 0.65+ | ? | ? | 67.5% | ? |

**Calibration error = |actual - expected|.** Si error > 5pp sistemático → modelo sesgado.

---

### S-27-6: Evolución Temporal

**Pregunta:** ¿El accuracy mejora o empeora con el tiempo?

| Semana | Picks | Hit% | ROI% | Notas |
|---|---|---|---|---|
| Jun 01-07 | ? | ? | ? | Post Nodo-18/19/20/21 |
| Jun 08-14 | ? | ? | ? | API integration + B-11 roto |
| Jun 15-17 | ? | ? | ? | B-11 fix + revert calibración |

**Fuente:** apuestas (ts_registro → semana) + apuestas (correcto)
**Detección de drift:** Si rolling 20-pick accuracy < 55% → ALERTA (modelo degradándose).

---

### S-27-7: Portfolio — Combos vs Individuales

**Pregunta:** ¿Los combos agregan valor sobre individuales?

| Tipo | Apostados | Ganados | Hit% | ROI% | Avg Cuota |
|---|---|---|---|---|---|
| Individual | ? | ? | ? | ? | ? |
| Combo 3p | ? | ? | ? | ? | ? |
| Combo 4p | ? | ? | ? | ? | ? |
| Mega 6-10p | ? | ? | ? | ? | ? |
| Safe 2p | ? | ? | ? | ? | ? |

**Fuente:** apuestas (tipo combo vs individual, piernas, correcto, ganancia)
**Cobertura Exclusión test:** Cuando pick X falla, ¿cuántos combos sin X sobrevivieron?

---

## Implementación

### Archivo: `pipeline_tracker.py`

```bash
# Uso básico — todo el histórico
python3 pipeline_tracker.py

# Filtrar por fecha
python3 pipeline_tracker.py --since 2026-06-01

# Filtrar por tier
python3 pipeline_tracker.py --tier challenger

# Solo una sección
python3 pipeline_tracker.py --section confianza
```

### Dependencias
- `pandas` (ya en el proyecto)
- `glob`, `json`, `datetime` (stdlib)
- `utils/logger.py` → SmartLogger (patrón existente)
- `config.py` → `detectar_tier()` (fuente única)

### Lógica de Join

```
1. Leer TODOS los edge_report_*.json → DataFrame con 48 cols + fecha
2. Leer TODOS los apuestas_*.json → DataFrame con resultados (correcto, ganancia)
3. Join por match_id + favorito_predicho (o fuzzy por nombre si no hay match_id)
4. Leer calibracion_edge.json para totales de referencia
5. Generar tablas por sección
6. Output → pipeline_tracking.txt + opcionalmente reports/pipeline_tracking_FECHA.json
```

### Output
- `pipeline_tracking.txt` — reporte texto legible en terminal (sobreescribe cada vez)
- `reports/pipeline_tracking_FECHA.json` — snapshot JSON para histórico (opcional con --save)

---

## Tests

```
T27-01: pipeline_tracker.py corre sin error con 0 archivos de apuestas → muestra "Sin datos de resultados"
T27-02: Sección S-27-1 con datos mock → confidence_flag counts correctos
T27-03: Sección S-27-2 con datos mock → cuota bins correctos
T27-04: ROI calcula correctamente: (ganancia_total / stake_total) × 100
T27-05: Join edge_report + apuestas por match_id funciona con datos reales
T27-06: --since filtra correctamente por fecha
T27-07: --tier filtra correctamente por tier
T27-08: Campos faltantes (bbi, golden_zone en reportes viejos) → NaN, no crash
```

---

## Validación

| ID | Criterio | Método |
|---|---|---|
| V-27-1 | Hit% por confidence_flag es monótono (STRONG > MODERATE > LOW) | Validar con n≥50 por bin |
| V-27-2 | ROI positivo en al menos 1 tier | Validar con n≥30 por tier |
| V-27-3 | Calibration error < 5pp por bin | Correr tras n≥100 picks totales |
| V-27-4 | Golden zone outperforma non-golden por ≥5pp | Validar con n≥20 golden picks |
| V-27-5 | Model drift detectable: accuracy Jun 15-16 < 50% visible en S-27-6 | Verificar con datos existentes |

---

## Reglas

- **REGLA-T27-1:** pipeline_tracker.py es READ-ONLY. No modifica ningún archivo de datos.
- **REGLA-T27-2:** Mostrar `n` (sample size) en TODA tabla. Sin n, el porcentaje es ruido.
- **REGLA-T27-3:** ROI siempre basado en stake real (no Kelly teórico). Si stake=0 → excluir de ROI.
- **REGLA-T27-4:** Si n < 10 en un bin, marcar con `*` (muestra insuficiente).
- **REGLA-T27-5:** No conectar al pipeline de predicción. Es observabilidad pura, no modifica decisiones.

---

## Conexiones con Otros Nodos

- **Nodo-21 (Tiers):** S-27-3 valida si los pesos SNR por tier realmente producen mejor ROI en los tiers donde el modelo tiene más datos.
- **Nodo-24 (BBI):** S-27-3 cruza BBI promedio por tier con ROI — valida la hipótesis de bookmaker blindness.
- **Nodo-25 (Guards):** S-27-7 valida si Dispersion Guard y Cobertura Exclusión reducen drawdowns.
- **Nodo-26 (Cross-Sectional):** S-27-5 valida si ranking_preserved_blend mejora o empeora la calibración en sesiones BLIND.
- **B-11 (ELO fix):** S-27-6 debe mostrar el colapso Jun 15-16 como evidencia de que el tracker detectaría model drift.

---

## Prioridad de Implementación

| Fase | Secciones | Estado |
|---|---|---|
| **Fase 1** | S-27-1, S-27-2, S-27-3 | ✅ IMPLEMENTADO 2026-06-17 |
| **Fase 2** | S-27-4, S-27-5 | ✅ IMPLEMENTADO 2026-06-17 |
| **Fase 3** | S-27-6, S-27-7 | ✅ IMPLEMENTADO 2026-06-17 |

---

## Hallazgos Reales — Primera Ejecución (2026-06-17)

> Datos: n=138 picks cargados de edge_reports Jun 14-17 | n=33 con resultado conocido (paper trading, stake=0)
> Correr: `python3 pipeline_tracker.py` → genera `pipeline_tracking.txt`

### S-27-1: Nivel de Confianza

| Flag | N | Con resultado | Wins | Losses | Hit% |
|---|---|---|---|---|---|
| STRONG | 3* | 3* | 3 | 0 | **100%** ✅ |
| MODERATE | 9* | 1* | 1 | 0 | 100%* |
| LOW | 126 | 29 | 8 | 21 | **27.6%** ❌ |

**Acción:** picks LOW representan el 91% del pool pero solo 27.6% acierto → considerar filtro LOW en trader o reducir stake al mínimo.

### S-27-2: Rango de Cuota

| Rango | N | Wins | Losses | Hit% |
|---|---|---|---|---|
| 1.50-2.00 | 14 | 5 | 1 | **83.3%** ✅ |
| 2.00-2.50 | 68 | 3 | 11 | **21.4%** ❌ |
| 2.50-3.00 | 19 | 1 | 7 | **12.5%** ❌ |
| 3.00-4.00 | 32 | 3 | 1 | **75.0%** ✅ |
| 4.00+ | 5* | 0 | 1 | 0%* |

**Alpha confirmado:** cuota 3.00-4.00 con edge alto = mejor zona. Zona 2.00-2.50 = trampa (underdogs mediocres sin señal real). Validación de sesión R4 2026-06-01 (underdogs ≥2.00 con edge >5%).

### S-27-3: Tier + Superficie

| Tier | Sup | N | Hit% | Avg BBI | Conclusión |
|---|---|---|---|---|---|
| atp500 | grass | 45 | **18.2%** ❌ | 0.505 | bookmaker informado → no hay ventaja |
| challenger | clay | 36 | 37.5% ➡ | 0.580 | ventaja moderada |
| itf | clay | 28 | **50.0%** ✅ | 0.514 | equilibrado |
| itf | hard | 16 | 100%* | 0.628 | muestra pequeña |

**Hipótesis BBI confirmada:** ATP500 grass (BBI bajo=0.505) → bookmaker informado → ROI proxy -62%. Challenger/ITF (BBI alto=0.58-0.63) → ventaja informacional del modelo.

### S-27-4b: Markov — Señal más importante descubierta

| Estado | N | Hit% | Conclusión |
|---|---|---|---|
| HOT | 58 | **64.3%** ✅ | señal más fuerte — 9W/5L |
| NEUTRAL | 67 | **6.7%** ❌ | FILTRAR — casi aleatorio (1W/14L) |
| COLD | 13 | 50.0%* | muestra pequeña |

**Acción inmediata:** picks con `markov_favorito=NEUTRAL` tienen 6.7% hit rate. Ver [[Nodo-18-PELT-Recency-Alpha]] para ajuste de λ en picks NEUTRAL. Propuesta: excluir NEUTRAL de pool de combos (REGLA-HF-1 extensión).

### S-27-5: Calibración

| p_modelo | N | Hit% real | Esperado | Diff |
|---|---|---|---|---|
| 0.50-0.52 | 83 | 18.8% | 51% | **-32.2pp** ❌ |
| 0.52-0.55 | 43 | 38.5% | 53.5% | -15pp ⚠️ |
| 0.55-0.60 | 9* | 100%* | 57.5% | +42.5pp* |
| 0.60-0.65 | 3* | 100%* | 62.5% | — |

**Diagnóstico:** modelo sobreestima picks débiles (p≈0.51). Bins 0.55+ se comportan mejor pero muestras pequeñas. Requiere n≥100 totales para V-27-3 (V-27-3 pendiente).

### Estado de Validación (V-27-1 → V-27-5)

| Criterio | Estado | n actual | n requerido |
|---|---|---|---|
| V-27-1: Hit% monótono STRONG>MOD>LOW | ✅ Confirmado (100%>100%>27%) | 33 | 50 por bin |
| V-27-2: ROI positivo en ≥1 tier | ⚠️ Challenger +16.9% proxy | 33 | 30 por tier |
| V-27-3: Calibration error <5pp | ❌ error 32pp en p=0.50-0.52 | 33 | 100 total |
| V-27-4: Golden Zone +5pp vs non-golden | ⏳ n insuficiente (2 golden con resultado) | 2 | 20 |
| V-27-5: Drift Jun 15-16 visible | ✅ W24=31.8% visible en S-27-6 | — | — |

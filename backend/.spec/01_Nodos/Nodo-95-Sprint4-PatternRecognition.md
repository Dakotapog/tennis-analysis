# Nodo-95 — Sprint 4: PatternRecognition Engine (D90-09)

**Estado:** COMPLETO 2026-07-13
**Autor:** Claude (Sonnet 4.6) por mandato Fable5/Nodo-90
**Tests:** 25 tests REGLA-T53 en `tests/test_nodo95_pattern_recognition.py` — 1945 passed total

---

## Decisiones implementadas

### D90-09 — PatternRecognition REPORTE_SOLO

**Archivo:** `scripts/pattern_recognition.py`

**Propósito:** Leer todos los picks settled del shadow book y reportar segmentos donde el modelo muestra hit% estadísticamente superior al breakeven de las cuotas. REPORTE_SOLO — ninguna promoción automática a hipótesis.

**Dimensiones analizadas (1-way):**
- `tier` (itf/challenger/atp500/grand_slam)
- `superficie` (clay/hard/grass)
- `zona_cuota` (underdog/slight_underdog/moderate_favorite/heavy_favorite)
- `markov_favorito` (HOT/NEUTRAL/COLD/?)
- `confidence_flag` (STRONG/MODERATE/LOW)

**Cross 2-way (pares de valor analítico):**
- tier × superficie
- tier × confidence_flag
- superficie × markov_favorito
- zona_cuota × confidence_flag
- markov_favorito × confidence_flag

**Criterio candidato:**
```
candidate = n >= min_n AND IC95_wilson_lower > breakeven
breakeven = 1 / avg_cuota_del_segmento
```

**Output:** `reports/pattern_candidates_YYYYMMDD_HHMMSS.json`

**Campos por segmento:** dim, value, n, wins, hit_pct, ic95_low, ic95_high, avg_cuota, breakeven, candidate

---

## Resultados sobre datos reales (2026-07-13)

**Base:** 224 settled | WON: 84 | hit%: 37.5% | avg_cuota: 2.764 | breakeven: 36.2%

**Candidatos 1-way (n≥5, IC_low > breakeven):**
| Segmento | n | hit% | IC_low | breakeven |
|---|---|---|---|---|
| confidence_flag=STRONG | 52 | 50.0% | 36.9% | 36.0% |

**Candidatos cross (2-way):**
| Segmento | n | hit% | IC_low | breakeven |
|---|---|---|---|---|
| NEUTRAL×STRONG (markov×conf) | 30 | 56.7% | 39.2% | 37.0% |
| challenger×STRONG (tier×conf) | 13 | 46.2% | 23.2% | 21.2% |
| COLD×LOW (markov×conf) | 5 | 80.0% | 37.5% | 27.8% |

**Interpretación:** STRONG confidence es la señal más robusta (n=52). Markov NEUTRAL+STRONG el segmento cross más estadísticamente sólido (n=30). COLD×LOW es borderline (n=5 mínimo requerido).

---

## Uso

```bash
# Todos los settled (default min_n=5)
python3 scripts/pattern_recognition.py

# Solo picks con apostar=True
python3 scripts/pattern_recognition.py --apostar-only

# Umbral más exigente
python3 scripts/pattern_recognition.py --min-n 10
```

---

## Invariantes REPORTE_SOLO

1. NO escribe a `validation/preregistered_hypotheses.json`
2. NO modifica gates ni stakes
3. NO cambia lógica de trader/edge_calculator
4. Promoción a H-XX = decisión humana

---

## Tests REGLA-T53

| Test | Cubre |
|---|---|
| `test_wilson_ic95_*` (6) | Wilson IC95 edge cases |
| `test_segment_key_*` (3) | Manejo None y campos ausentes |
| `test_load_settled_*` (3) | Filtrado settled/apostar_only/vacío |
| `test_segment_stats_*` (4) | candidato True/False/below_min_n/multiple_values |
| `test_cross_stats_*` (2) | Cross 2-way candidato + formato dim |
| `test_overall_stats_*` (2) | hit%/empty |
| `test_run_*` (5) | Integration: escribe JSON, estructura, no-hyp, vacío, params |

---

## Notas

- Corrección en tests: `_wilson_ic95(8,10)` con cuota=2.0 da IC_low=49% < break=50% → usar cuota=3.0 en fixtures donde se requiere candidato=True (Wilson necesita mayor n o menor cuota para superar 50% de breakeven con hits=8/10)
- `confidence_flag=STRONG` (n=52, IC_low=36.9% > break=36.0%) es la señal más cercana — el margen es pequeño, requiere más n antes de pre-registrar como hipótesis

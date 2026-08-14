# Nodo-124 — EvalTracker: Auto-logging picks EVALUAR de tabla_favoritos al Shadow Book

> **Wikilinks:** [[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-86-Auditoria-Doctoral-Hallazgos]] | [[Nodo-67-Integracion-Herramientas-DataContract]]
> **Fecha:** 2026-07-20 | **Autor:** Fable 5 (diagnóstico) / Sonnet 4.6 (spec)
> **Tipo:** GAP CRÍTICO + BUGFIX ARQUITECTURAL — hallazgo pipeline 2026-07-20

---

## §1. Diagnóstico del gap

### Hallazgo 2026-07-20

Durante análisis post-partido del pipeline diario se descubrió que **`generar_tabla_favoritos2.py` (PASO 3.5) genera predicciones con confianza ≥ 54% ("EVALUAR") que NUNCA se loguean al shadow book** y por tanto nunca entran al pool de calibración.

**Impacto cuantificado:**
- ~10 picks EVALUAR/día con conf ≥ 54%
- Señales ricas: HOT state, ELO_DOM, RIVAL VALUE, RACHAS CALIENTES, surface specialization
- **0 picks de estos rastreados históricamente** — calibración ciega
- El modelo alega 77.4% accuracy en clay (n=31) pero ese dato proviene de otro flujo; el EVALUAR no tiene hit% empírico

**Evidencia del día 2026-07-20:**
```
Picks EVALUAR generados: 10
Picks EVALUAR en shadow book: 0
Picks shadow book del día: 10 (todos de edge_calculator — flujo distinto)
```

Los picks EVALUAR del 20-jul-20 incluían: Badosa @1.48 (conf=56.8%, HOT 90%), Pridankina @1.06 (conf=66.3% — favorita absoluta), Carle @1.18 (conf=57.1%), Bouzkova @1.33 (conf=55.2%, HOT 80%). Resultados reales desconocidos para el sistema — gap de trazabilidad total.

---

## §2. Arquitectura del gap

```
generar_tabla_favoritos2.py (PASO 3.5)
  └─ EVALUAR picks → analisis_partidos_pandas.txt  ← STDOUT MUERTO
                                                       sin log a shadow_book
                                                       sin settle
                                                       sin calibración

edge_calculator.py (PASO 3)
  └─ WATCHLIST/APOSTAR picks → shadow_book.log_pick() ← SI RASTREADO
                             → shadow_book --settle    ← SI RESULTADO
                             → shadow_book --report    ← SI CALIBRACIÓN
```

**Root cause:** `generar_tabla_favoritos2.py` fue diseñado como herramienta de revisión humana (PASO 3.5 = "leer antes de apostar"), no como generador de picks. La integración con shadow_book nunca se implementó.

---

## §3. Fixes requeridos

### D124-01 — Auto-logger EVALUAR en generar_tabla_favoritos2.py

**Archivo:** `generar_tabla_favoritos2.py`

Al finalizar el análisis, para cada partido con `ACCION: EVALUAR` (conf ≥ 54%):
```python
# Al final del loop de partidos, si accion == 'EVALUAR':
pick_evaluar = {
    "favorito_predicho": jugador_favorito,
    "cuota_favorito": cuota_favorito,
    "rival": rival,
    "torneo": torneo,
    "superficie": superficie,
    "tier": tier,
    "pick_status": "EVALUAR",
    "confidence": confianza,
    "confidence_flag": conf_flag,  # LOW/MOD/STRONG
    "edge": edge_vs_mercado_pct,
    "p_modelo": p_modelo,
    "p_implicita": p_implicita,
    "score_directo": score_directo,
    "señales": {
        "hot_state": hot_fav,
        "elo_dominance": elo_dom,
        "rival_value_flag": rival_value,
        "rachas_calientes": rachas
    }
}
sb.log_pick(pick_evaluar, fecha=fecha_hoy)
```

Requiere importar `shadow_book.ShadowBook` en `generar_tabla_favoritos2.py`.

### D124-02 — Pre-registrar H124-01 y H124-02

**Archivo:** `validation/preregistered_hypotheses.json`

```json
{
  "id": "H124-01",
  "descripcion": "Picks EVALUAR (conf≥54%) de generar_tabla_favoritos2 tienen hit% > breakeven implícito cuota",
  "n_stop": 30,
  "fecha_registro": "2026-07-20",
  "segmento": "pick_type=evaluar",
  "estado": "ACUMULANDO"
},
{
  "id": "H124-02",
  "descripcion": "Picks EVALUAR con señal HOT activa tienen hit% ≥5pp adicional sobre base EVALUAR",
  "n_stop": 20,
  "fecha_registro": "2026-07-20",
  "segmento": "pick_type=evaluar AND hot_state=True",
  "estado": "ACUMULANDO"
}
```

### D124-03 — Segmento EVALUAR en shadow_book --report

**Archivo:** `shadow_book.py` función `report()`

Añadir tras los segmentos existentes:
```python
# Segmento pick_type=evaluar
evaluar_picks = [r for r in settled if r.get('pick_status') == 'EVALUAR']
if evaluar_picks:
    _print_segment("pick_type=EVALUAR (conf≥54%)", evaluar_picks)
    # Sub-segmentos por conf bracket
    for lo, hi, label in [(0.54, 0.57, "54-57%"), (0.57, 0.60, "57-60%"), (0.60, 1.0, "≥60%")]:
        sub = [p for p in evaluar_picks
               if lo <= p.get('pick_snapshot',{}).get('confidence',0) < hi]
        if len(sub) >= 5:
            _print_segment(f"  EVALUAR conf [{label}]", sub)
```

### D124-04 — Sección EVALUAR en daily_brief

**Archivo:** `run_daily.py` función `_build_daily_brief()`

Añadir sección tras WAS CANDIDATOS:
```
EVALUAR HOY (tabla_favoritos, pre-partido):
  Pridankina E.    @1.06  conf=66.3%  — sin señal HOT
  Carle M.L.       @1.18  conf=57.1%  — arcilla
  Badosa P.        @1.48  conf=56.8%  — HOT 90%
  Zhang Z.         @1.43  conf=56.6%  — SCALP TOP-20
  Rodionov J.      @1.67  conf=55.6%  — arcilla
  ... (solo conf≥54%)
NOTA: No apostar automáticamente — revisión humana requerida. KGR no calculado.
```

### D124-05 — Análisis retroactivo EVALUAR histórico

**Script:** `scripts/backfill_evaluar_shadow.py` (nuevo)

Leer todos los `reports/edge_report_*.json` históricos, extraer picks con `confidence ≥ 0.54`, cruzar con FlashScore settle para recuperar resultados históricos y estimar hit% base antes de que H124-01 acumule n=30 nuevos.

---

## §4. Tests (REGLA-T53)

```python
# tests/test_nodo124_evaluar_tracker.py
def test_D124_01_evaluar_pick_logged_to_shadow_book()
def test_D124_01_evaluar_pick_no_logger_below_conf_threshold()
def test_D124_02_H124_01_preregistered()
def test_D124_02_H124_02_preregistered()
def test_D124_03_report_includes_evaluar_segment()
def test_D124_04_brief_includes_evaluar_section()
```

---

## §5. Impacto esperado

| Métrica | Antes | Después |
|---------|-------|---------|
| Picks rastreados/día | 10-15 (edge_calc) | 20-25 (+10 EVALUAR) |
| Calibración del modelo | parcial | completa |
| Hit% EVALUAR empírico | DESCONOCIDO | medible en ~3 semanas |
| Alpha oculto capturado | 0% | potencial +75% hit rate (por confirmar) |

**Hipótesis de impacto:** Si los picks EVALUAR conf≥60% tienen hit% ≥ 60%, representan el **mayor alpha sin explotar** del sistema. Pridankina @1.06 conf=66.3% es ejemplo extremo — el modelo asigna 66% a una @1.06 que implica 94% del bookmaker. Esto puede ser ruido o señal real — solo el tracking lo resuelve.

---

## §6. Prioridad de implementación

| Fix | Complejidad | Impacto | Prioridad |
|-----|-------------|---------|-----------|
| D124-02 H124-01/02 pre-registro | Baja (JSON edit) | Alta | **INMEDIATA** |
| D124-04 EVALUAR en daily_brief | Media (run_daily) | Alta | **HOY** |
| D124-01 auto-logger | Alta (shadow_book API) | Crítica | **ESTA SEMANA** |
| D124-03 segmento --report | Media | Alta | **ESTA SEMANA** |
| D124-05 retroactivo | Alta | Media | **BACKLOG** |

---

**Wikilinks totales: 4 | Huérfanos: 0** (verificado contra nodos_index.json 2026-07-20)

[[Nodo-109-Live-Trading-Desk-Dashboard]] | [[Nodo-100-Taxonomia-Estrategias-Generacion-Combos]] | [[Nodo-86-Auditoria-Doctoral-Hallazgos]] | [[Nodo-67-Integracion-Herramientas-DataContract]]

---

## §7. Addendum — D174-12 (2026-08-06): decisión explícita RETIRAR de huérfano

`scripts/backfill_evaluar_shadow.py` (D124-05, arriba) apareció en [[Nodo-174]] como
"módulo huérfano" — sin ningún PASO en `run_daily.py` que lo invoque. Auditoría del
propio docstring del script (`Recupera picks EVALUAR históricos de edge_reports y
los inyecta en shadow_book... Uso: python3 scripts/backfill_evaluar_shadow.py
[--dry-run] [--fecha YYYY-MM-DD]`) confirma que **no es un gap** sino diseño
correcto: es una herramienta de recuperación retroactiva, ejecutada manualmente
sobre una fecha específica cuando se detecta un backlog de picks `sin_edge` sin
tracking — mismo patrón que `scripts/audit_phantom_history.py` ([[Nodo-152]]
D152-06), que tampoco corre en `run_daily.py`. Conectarlo a un PASO diario
duplicaría trabajo sin sentido: `apostar`/`watchlist` ya se loguean en shadow_book
durante el pipeline normal (el propio docstring lo aclara: "sólo se procesan
sin_edge"), y correr el backfill todos los días sobre el mismo rango de fechas no
aporta nada nuevo tras la primera pasada. **Decisión: RETIRAR de la lista de
huérfanos — standalone intencional, no pendiente.** Sin cambio de código.

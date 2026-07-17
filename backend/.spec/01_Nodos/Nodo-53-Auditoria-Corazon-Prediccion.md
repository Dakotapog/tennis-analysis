# Nodo-53: Auditoría del Corazón de Predicción — Pre-Condiciones Antes de Tocar rivalry_analyzer.py

> **Wikilinks:** [[Nodo-52-Shadow-Book-CLV-Tracking]] | [[Nodo-51-Plan-Estrategico-Data-Layer-Torneo]] | [[Nodo-46-Markov-Surface-Context-Discount]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | ~~~~[[Nodo-14-Grass-Variance]]~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_~~ _(MISSING — ver [[Nodo-60-GCS-Grass-Surface-Champion-Signal]])_
> **Fecha de creación:** 2026-07-02
> **Estado:** 📋 ESPECIFICADO — para auditoría de Fable antes de cualquier implementación
> **Trigger:** Análisis de Mensik vs Dimitrov (Wimbledon, Hierba) reveló bugs silenciosos en el núcleo de predicción

**Principio rector de este nodo:**
`rivalry_analyzer.py` es el archivo más crítico del proyecto. Tiene ~1800 líneas, 62 tests que lo blindan (Nodo-31), y genera TODOS los puntajes de predicción. Antes de tocar una sola línea, este documento debe estar firmado por Fable y todos los bugs documentados aquí deben tener tests que los reproduzcan con FAIL antes del fix y PASS después.

**Regla absoluta:** ningún cambio a `rivalry_analyzer.py` o `analysis/markov_analyzer.py` sin:
1. Test que reproduce el bug en estado FAIL
2. Fix
3. Test en estado PASS
4. `pytest tests/ --no-cov -q` → 1585 passed (o más, nunca menos)

---

## 0. Contexto — Cómo se Descubrió

El análisis del partido Mensik vs Dimitrov (Wimbledon 2026-07-02) usando el output de `reports/analisis_partidos_20260702_020507.txt` reveló que el modelo tiene un H2H directo de **2-0 a favor de Mensik** (ganó en octubre 2024 y abril 2024), pero el campo `h2h_direct` en los raw scores vale `0.0` para **ambos jugadores**. El modelo ignora silenciosamente el H2H directo completo.

Esto no es un edge case. Es el comportamiento por defecto cuando el H2H más reciente tiene más de 8 meses de antigüedad — lo que ocurre en la mayoría de los partidos del circuito ATP/WTA donde los jugadores no se enfrentan en cada torneo.

---

## 1. Bugs Documentados — Causa Raíz Exacta

### Bug D53-01 — CRÍTICO: Date parsing H2H con año de 4 dígitos

**Archivo:** `analysis/rivalry_analyzer.py`
**Línea:** 655 en `analyze_direct_h2h()` y línea 1682 en el bloque de lógica de ponderación dinámica

**Código actual (buggy):**
```python
# Línea 655
match_date = datetime.strptime(match_date_str, '%d.%m.%y')   # %y = 2 dígitos

# Línea 1682
last_match_date = datetime.strptime(direct_h2h_matches[0]['fecha'], '%d.%m.%y')  # mismo bug
```

**Datos reales que entran:**
```
'09.10.2024'   # formato DD.MM.YYYY — 4 dígitos
'27.04.2024'   # ídem
```

**Error:**
```
ValueError: unconverted data remains: 24
```

**Consecuencia:** el `except (ValueError, TypeError)` en línea 677 atrapa el error y hace `continue` — el partido H2H es silenciosamente descartado. Ambos partidos Mensik-Dimitrov se descartan. `p1_score = 0.0`, `p2_score = 0.0`. El componente H2H (peso 18% en Grand Slam) vale 0 para ambos.

**El mismo bug se propaga a `LOG_DYNAMIC_WEIGHTING_ERROR`:** la lógica de ponderación dinámica en línea 1682 también usa `%d.%m.%y` — cuando intenta determinar si el H2H es "antiguo" (>730 días) para ajustar pesos, falla y el bloque completo es ignorado.

**Fix:**
```python
# Línea 655 — reemplazar %y por %Y
match_date = datetime.strptime(match_date_str, '%d.%m.%Y')

# Línea 1682 — ídem
last_match_date = datetime.strptime(direct_h2h_matches[0]['fecha'], '%d.%m.%Y')
```

**Alternativa robusta (si pueden venir ambos formatos):**
```python
def _parse_h2h_date(fecha_str):
    for fmt in ('%d.%m.%Y', '%d.%m.%y'):
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Fecha H2H no parseable: {fecha_str}")
```

---

### Bug D53-02 — IMPORTANTE: H2H Antiquity Threshold = 250 días (elimina historial válido)

**Archivo:** `analysis/rivalry_analyzer.py`
**Línea:** 647

**Código actual:**
```python
H2H_RECENT_DAYS_THRESHOLD = 250
# ...
if days_since_match <= H2H_RECENT_DAYS_THRESHOLD:
    ponderacion = 1.0
# else: ponderacion = 0.0 (implícito)
```

**Problema:** 250 días = ~8 meses. En el circuito ATP/WTA, dos jugadores pueden no haberse enfrentado en más de 8 meses y su H2H seguir siendo relevante. Casos concretos:
- Mensik vs Dimitrov: H2H en octubre 2024 y abril 2024. En julio 2026 = 633 y 826 días respectivamente → ambos a `ponderacion = 0.0`
- Un jugador que ganó 5-0 el H2H pero el último encuentro fue hace 10 meses: el H2H completo se ignora

**El sistema tiene dos umbrales distintos:**
- `H2H_RECENT_DAYS_THRESHOLD = 250` en `analyze_direct_h2h()` → ponderación 0/1 binaria
- `H2H_ANTIQUITY_THRESHOLD = 730` en la lógica de ponderación dinámica → determina si el H2H es "antiguo" para ajustar pesos

**Esto es inconsistente:** el sistema considera un H2H "antiguo" cuando tiene >730 días para la lógica de ponderación, pero descarta completamente cualquier H2H con >250 días al calcular el score. Un H2H de 400 días: score=0 en cálculo pero "reciente" en ponderación dinámica.

**Propuesta de corrección (requiere validación empírica con Shadow Book — no calibrar con n=1):**

| Antigüedad H2H | Ponderación Actual | Ponderación Propuesta |
|---|---|---|
| ≤ 250 días | 1.0 | 1.0 |
| 251–500 días | 0.0 | 0.6 |
| 501–730 días | 0.0 | 0.3 |
| > 730 días | 0.0 | 0.1 (señal débil, no ignorar) |

**GATE:** esta modificación de constantes sólo se implementa después de que el Shadow Book tenga n≥30 observaciones con H2H disponible. Documentar como hipótesis pre-registrada H53-01 antes de calibrar.

---

### Bug D53-03 — MEDIO: Rivales Comunes no descuenta superficie

**Archivo:** `analysis/rivalry_analyzer.py` — bloque `analyze_common_opponents()`

**Problema:** al comparar cómo dos jugadores se desempeñaron contra un rival común, el modelo trata como equivalente:
- "Mensik ganó a Hurkacz en **hierba**" (Wimbledon 2026)
- "Dimitrov ganó a Hurkacz en **arcilla**" (Roland Garros 2024)

Para un partido en Wimbledon, la victoria de Mensik sobre Hurkacz *en la misma superficie* debería tener más peso que la victoria de Dimitrov sobre Hurkacz en una superficie diferente.

**Evidencia en el partido Mensik vs Dimitrov:**
```
Hurkacz H.: Ambos ganaron, pero Grigor Dimitrov fue más contundente (3-0 vs 2-1 de Jakub Mensik)
→ Ventaja: Grigor Dimitrov
```
Pero: Mensik ganó a Hurkacz **en hierba el 25.06.2026** (hace 1 semana). Dimitrov ganó a Hurkacz **en arcilla el 02.06.2024** (hace 2 años). Para Wimbledon, la ventaja debería ser de Mensik.

**Superficie actual del partido:** grass
**Corrección propuesta:** aplicar `_surface_overlap_rate()` (ya implementado en Nodo-46 F4) al comparar rivales comunes. Un rival ganado en la misma superficie del partido actual vale peso completo; en superficie diferente, vale 60%.

**GATE:** verificar que `_normalize_surface()` y `_surface_overlap_rate()` de `markov_analyzer.py` (Nodo-46 F4) están importados correctamente en `rivalry_analyzer.py` antes de extender su uso. Ya están importados — confirmar con `grep`.

---

### D53-04 — COSMÉTICO CON RIESGO OCULTO: Pesos suman 99% en vez de 100%

**Archivo:** `analysis/rivalry_analyzer.py` — bloque de pesos Grand Slam
**Pesos actuales:**
```python
'grand_slam': {
    'surface_specialization': 0.15,
    'form_recent': 0.12,
    'common_opponents': 0.22,
    'h2h_direct': 0.18,
    'ranking_momentum': 0.15,
    'elo_rating': 0.13,
    'home_advantage': 0.05,
    'strength_of_schedule': 0.00
}
# Suma: 0.15+0.12+0.22+0.18+0.15+0.13+0.05+0.00 = 1.00 ✓
```

Pero el log muestra "Suma Total de Pesos: 99.0%". El problema está en que después del shrinkage (línea 58539 del output) los pesos se redistribuyen y la suma puede no llegar a 1.0 exactamente. El puntaje final normaliza por la suma real, pero el display engaña.

**Fix:** en `generar_tabla_favoritos2.py`, redondear los pesos a 2 decimales antes de mostrar, y validar que suman 100% con `assert abs(sum(weights.values()) - 1.0) < 0.01`.

---

### D53-05 — ORGANIZACIÓN: Señales críticas enterradas en logs de 200 líneas

**Problema:** el output por partido tiene ~200-400 líneas de logs internos (`P1_LOG_SURF`, `P2_LOG_SoS`, `LOG_SHRINKAGE`, etc.) mezclados con las tablas de análisis. El insight más importante del partido Mensik vs Dimitrov — la tensión real entre los componentes — requiere leer todo el log para entenderla:

```
TENSIÓN REAL:
  Dimitrov: mejor en hierba históricamente (64% vs 45%)
  Mensik:   mejor en forma reciente (60% vs 30%), ELO superior (1942 vs 1757)
  → El modelo resuelve a favor de Mensik (51.3%) pero solo por 0.13 puntos
  → El output no dice esto en ningún lugar visible
```

La señal `SCALP TOP-10 EN SUPERFICIE` (Dimitrov venció a Medvedev #7 en hierba) aparece en una línea entre los logs — es posiblemente la señal más predictiva del partido y está enterrada.

---

## 2. Mapa de Riesgo — Antes de Tocar el Código

```
rivalry_analyzer.py (~1800 líneas)
│
├── analyze_direct_h2h()          ← Bug D53-01 aquí (línea 655)
│                                    Bug D53-02 aquí (línea 647)
│
├── analyze_common_opponents()    ← Gap D53-03 (no descuenta superficie)
│
├── generate_advanced_prediction()
│   ├── Shrinkage (Nodo-21)       ← OK — blindado por tests
│   ├── PELT Recency (Nodo-18)    ← OK — blindado por tests
│   ├── H2H Immunity (Nodo-19)    ← OK — blindado por tests
│   ├── F4 Surface Discount       ← OK — Nodo-46, n=1 BLOQUEADO
│   ├── Dynamic Weighting         ← Bug D53-01 cascadea aquí (línea 1682)
│   └── Circuit Asymmetry (N-29)  ← OK
│
└── Pesos por tier (5 tiers)      ← D53-04 (suma 99%)
```

**Archivos que NO deben tocarse en este nodo:**
- `analysis/markov_analyzer.py` — F4 ya implementado, BLOQUEADO hasta n≥5
- `core/data_contract.py` — F2 sellado
- `edge_calculator.py` — solo si hay bug probado
- `validation/preregistered_hypotheses.json` — INMUTABLE

---

## 3. Pre-Condiciones Obligatorias Antes de Implementar

### P53-01: Baseline de tests
```bash
python3 -m pytest tests/ --no-cov -q
# Debe dar: 1585 passed (o exactamente el número actual)
# Si da menos: STOP — hay regresión previa sin resolver
```

### P53-02: Confirmar que el bug D53-01 es reproducible
```bash
python3 -c "
from datetime import datetime
# Debe fallar con 'unconverted data remains: 24'
try:
    datetime.strptime('09.10.2024', '%d.%m.%y')
    print('BUG NO REPRODUCIDO — verificar versión de Python')
except ValueError as e:
    print(f'Bug confirmado: {e}')
"
```

### P53-03: Confirmar H2H 2-0 Mensik vs Dimitrov está en los datos
```bash
python3 -c "
import json
data = json.load(open('reports/h2h_results_enhanced_20260702_020507.json' if __import__('pathlib').Path('reports/h2h_results_enhanced_20260702_020507.json').exists() else sorted(__import__('glob').glob('reports/h2h_results_enhanced_*.json'))[-1]))
for m in data.get('matches', []):
    p1 = m.get('jugador1',''); p2 = m.get('jugador2','')
    if 'mensik' in p1.lower() or 'mensik' in p2.lower():
        if 'dimitrov' in p1.lower() or 'dimitrov' in p2.lower():
            h2h = m.get('ranking_analysis',{}).get('prediction',{}).get('direct_h2h_matches',[])
            print('H2H encontrado:', len(h2h), 'partidos')
            for hm in h2h:
                print(' -', hm.get('fecha'), hm.get('ganador'))
"
```

### P53-04: Verificar imports F4 ya presentes en rivalry_analyzer
```bash
grep -n "apply_surface_context_discount\|_normalize_surface\|_surface_overlap_rate" analysis/rivalry_analyzer.py
# Debe mostrar líneas existentes — F4 ya está importado (Nodo-46)
```

---

## 4. Plan de Implementación — Orden Obligatorio

### Fase A — Tests primero (antes de tocar código)

**Test T53-01:** reproduce D53-01 (debe FAIL antes del fix)
```python
def test_t53_01_h2h_date_parsing_4digit_year():
    """D53-01: fechas H2H con año 4 dígitos deben parsearse correctamente."""
    from analysis.rivalry_analyzer import RivalryAnalyzer
    ra = RivalryAnalyzer.__new__(RivalryAnalyzer)
    h2h_matches = [
        {'fecha': '09.10.2024', 'ganador': 'Mensik J.'},
        {'fecha': '27.04.2024', 'ganador': 'Mensik J.'},
    ]
    p1, p2, log = ra.analyze_direct_h2h(h2h_matches, 'Mensik J.', 'Dimitrov G.')
    # Sin el fix, p1=0.0 y p2=0.0 (bug)
    # Con el fix, p1>0 o al menos los partidos no son silenciados
    errors = [l for l in log if 'Error' in l]
    assert len(errors) == 0, f"Fechas H2H fallaron de parse: {errors}"
```

**Test T53-02:** reproduce D53-02 (threshold 250 días silencia H2H válido)
```python
def test_t53_02_h2h_antiquity_threshold_not_binary():
    """D53-02: H2H de 400 días no debe valer exactamente 0."""
    from analysis.rivalry_analyzer import RivalryAnalyzer
    from datetime import datetime, timedelta
    ra = RivalryAnalyzer.__new__(RivalryAnalyzer)
    # Partido hace 400 días — debería tener algún peso, no 0
    fecha_antigua = (datetime.now() - timedelta(days=400)).strftime('%d.%m.%Y')
    h2h_matches = [{'fecha': fecha_antigua, 'ganador': 'PlayerA'}]
    p1, p2, log = ra.analyze_direct_h2h(h2h_matches, 'PlayerA', 'PlayerB')
    # Con threshold actual: p1=0.0 (FAIL esperado antes del fix)
    # Con fix propuesto: p1>0 (ponderación decreciente)
    assert p1 > 0, "H2H de 400 días debe tener algún peso — no es información irrelevante"
```

**Test T53-03:** common opponents surface match bonus
```python
def test_t53_03_common_opponents_surface_bonus_grass():
    """D53-03: victoria sobre rival común en la MISMA superficie debe valer más."""
    # Este test define el contrato antes de implementar el surface discount
    # en common opponents. Comparar:
    # - Mensik ganó a Hurkacz en hierba (match_surface=grass, current=grass) → peso mayor
    # - Dimitrov ganó a Hurkacz en arcilla (match_surface=clay, current=grass) → peso menor
    pass  # Implementar cuando se aborde D53-03
```

### Fase B — Fix D53-01 (solo este, nada más)

Cambiar en `rivalry_analyzer.py`:
1. **Línea 655:** `'%d.%m.%y'` → `'%d.%m.%Y'`
2. **Línea 1682:** `'%d.%m.%y'` → `'%d.%m.%Y'`

Verificar después:
```bash
python3 -m pytest tests/ --no-cov -q
# Debe seguir: 1585 passed + T53-01 ahora en PASS
```

Correr el pipeline con el partido Mensik vs Dimitrov y verificar que:
```
LOG_RAW_SCORES_P1: {..., 'h2h_direct': X.X, ...}  # X.X > 0
LOG_RAW_SCORES_P2: {..., 'h2h_direct': 0.0, ...}   # Mensik ganó ambos H2H
```

### Fase C — D53-02 (threshold H2H) — GATED

**GATE:** NO implementar hasta tener n≥30 observaciones settled en Shadow Book con al menos 10 partidos que tengan H2H disponible. El threshold actual (250 días) puede estar calibrado empíricamente — cambiar sin datos es p-hacking.

Registrar como hipótesis pre-registrada:
```json
"H53-01": {
  "descripcion": "Ponderación decreciente H2H (0.6/0.3/0.1) mejora Brier Score vs threshold binario 250 días",
  "estado": "BLOQUEADO_ACUMULANDO",
  "n_stop": 30,
  "n_casos_atribuibles": 0
}
```

### Fase D — D53-03 (surface discount en common opponents) — GATED

**GATE:** igual que Fase C. Además, requiere que Nodo-46 F4 esté calibrado (actualmente BLOQUEADO con n=1 Watanuki).

### Fase E — Output organization (D53-05)

Sin gates. Modificar `generar_tabla_favoritos2.py` para:
1. Añadir sección "TENSIÓN DE COMPONENTES" al resumen de cada partido con los top-2 componentes contradictorios
2. Mover señales especiales (`SCALP TOP-10`) al bloque de resumen, antes de los logs
3. Validar que pesos suman 100% ± 0.5%

---

## 5. Hipótesis Pre-Registradas de Este Nodo

Añadir a `validation/preregistered_hypotheses.json` antes de implementar Fases C/D:

```json
"H53-01": {
  "descripcion": "Ponderación H2H decreciente por antigüedad mejora accuracy vs threshold binario 250 días",
  "estado": "BLOQUEADO_ACUMULANDO",
  "n_stop": 30,
  "criterio_atribucion": [
    "Partido tiene H2H directo con antigüedad 250-730 días",
    "Modelo con ponderación decreciente predice ganador correcto",
    "Modelo con threshold binario habría predicho incorrecto"
  ],
  "n_casos_atribuibles": 0,
  "casos": []
},
"H53-02": {
  "descripcion": "Surface discount en rivales comunes mejora Brier Score en partidos de hierba",
  "estado": "BLOQUEADO_ACUMULANDO",
  "n_stop": 30,
  "n_casos_atribuibles": 0,
  "casos": []
}
```

---

## 6. Preguntas Abiertas para Fable — Audit Checklist

Antes de aprobar cualquier implementación de Nodo-53, Fable debe responder:

**F53-Q1:** ¿El threshold 250 días en `H2H_RECENT_DAYS_THRESHOLD` fue calibrado empíricamente o es un valor arbitrario? Si es arbitrario, D53-02 debe pasar a BLOQUEADO_ACUMULANDO hasta n≥30.

**F53-Q2:** ¿Hay otros usos de `'%d.%m.%y'` (año 2 dígitos) en el codebase que puedan tener el mismo bug?
```bash
grep -rn "'%d.%m.%y'" --include="*.py"
```
Si hay más instancias, el fix debe ser global.

**F53-Q3:** ¿`analyze_common_opponents()` usa `_normalize_surface()` de Nodo-46 en algún punto? Si no, ¿cuánto esfuerzo añadir el surface discount ahí vs el impacto esperado?

**F53-Q4:** En Mensik vs Dimitrov, con D53-01 corregido (H2H 2-0 para Mensik cuenta), ¿el modelo cambiaría la predicción de 51.3% Mensik a algo más diferenciado, o el peso 18% H2H sigue siendo pequeño comparado con las otras componentes?

**F53-Q5:** ¿Qué pasó realmente con el partido Mensik vs Dimitrov el 2026-07-02? El resultado real sirve como primera observación para H53-01 y H53-02 si el modelo con D53-01 corregido habría predicho diferente.

---

## 7. Archivos Afectados

| Archivo | Cambio | Fase | Tests requeridos |
|---|---|---|---|
| `analysis/rivalry_analyzer.py` | Fix `%y`→`%Y` en líneas 655 y 1682 | B | T53-01 FAIL→PASS |
| `analysis/rivalry_analyzer.py` | Threshold H2H decreciente | C (GATED n≥30) | T53-02 FAIL→PASS |
| `analysis/rivalry_analyzer.py` | Surface discount common opponents | D (GATED n≥30) | T53-03 |
| `generar_tabla_favoritos2.py` | Tensión componentes + señales en resumen | E | visual |
| `validation/preregistered_hypotheses.json` | H53-01, H53-02 | antes de C/D | T51-F5-06 sigue PASS |
| `tests/test_nodo53.py` | T53-01, T53-02, T53-03 | A (primero) | nuevos |

---

## 8. Lo que Este Nodo NO Toca

- `calibracion_edge.json` — datos inmutables
- `edge_calculator.py` — sin bugs conocidos en este nodo
- `markov_analyzer.py` — F4 BLOQUEADO hasta n≥5 (Nodo-46)
- `shadow_book.py` — completo y operativo (Nodo-52)
- `core/` — F0-F2 sellados (Nodo-51)
- `validation/preregistered_hypotheses.json` — solo añadir H53-01/H53-02, nunca modificar H52-*

---

*Documento generado en sesión 2026-07-02 tras análisis empírico del match Mensik vs Dimitrov.*
*Causa raíz D53-01 verificada con `python3 -c "datetime.strptime('09.10.2024', '%d.%m.%y')"` → `ValueError: unconverted data remains: 24`*

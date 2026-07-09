# Nodo-62 — Signal Bridge

> Fecha: 2026-07-06
> Estado: IMPLEMENTADO
> Archivo principal: `combo_confianza_builder.py`
> Tests: `tests/test_nodo62.py` (10 tests)

---

## §0 Veredicto Ejecutivo

| Hallazgo | Impacto | Resolucion |
|---|---|---|
| CORE ordenado solo por `confianza` | Picks con triple_alignment alto, markov=HOT o gcs_bonus ignorados | `combo_priority = confianza + alpha_score` como criterio de orden |
| Cat-C1 gate conf>=60% bloqueaba picks con edge real | Nilsson (edge=23.6%) bloqueada por 7.4 puntos de conf | Alpha gate: edge>=5% + triple>=0.2 promueve Cat-C2 a Cat-C1 |
| Bookmaker blindspot no modelado en combos | BBI, triple_alignment, GCS_BONUS existen en edge_report pero no llegan al combo builder | Signal Bridge: `_load_edge_report_index()` + `_lookup_edge_data()` + `_compute_alpha_score()` |
| Picks fantasma no penalizados en combos | history_provenance=EMPTY puede entrar al CORE | alpha=-25 penaliza datos fantasma antes del sort |

### Casos diagnosticados 2026-07-06

| Jugador | conf | triple | markov | alpha | combo_priority | Resultado sin SB | Resultado con SB |
|---|---|---|---|---|---|---|---|
| Herman Hoeyeraal | 55.8% | 0.661 | HOT | +33 | 88.8 | Fuera (conf baja) | CORE top |
| Lea Nilsson | 52.6% | 0.332 | HOT | - | - | Bloqueada (conf<53) | Fuera (bajo threshold) |
| Alan Magadan | 79.7% | 0.082 | - | -0 | 79.7 | CORE top | CORE (sin cambio) |
| Facundo Pereyra | 64.4% | - | - | - | 64.4 | No en CORE (Cat-C2) | Cat-C1 si edge>=5%+triple>=0.2 |

---

## §1 Diagnostico

El `combo_confianza_builder.py` recibe partidos del `h2h_results_enhanced` y extrae la confianza
de `ranking_analysis.prediction.confidence`. Esta confianza es el output del modelo de rivalidad
(rivalry_analyzer.py) — un numero entre 0-100 que mezcla todos los componentes.

El problema: al ordenar solo por confianza, el combo builder ignora 60+ campos del `edge_report`
que contienen informacion estructural sobre por que el bookmaker tiene un blind spot. En particular:

- `triple_alignment`: cuando ELO + Markov + Superficie coinciden, el modelo tiene alta coherencia interna.
  Un triple_alignment=0.661 significa que todas las senales apuntan en la misma direccion.
- `markov_favorito=HOT`: el jugador cambio de regimen recientemente. El bookmaker no actualiza sus
  modelos tan rapido como el pipeline de Markov/PELT — esto es alpha real.
- `gcs_bonus`: campeon reciente de hierba. El bookmaker no modela esto explicitamente (demostrado
  con hit rate 64.8% en n=54, A60-01).
- `bbi` (Bookmaker Blindspot Index): cuanto menos sabe el book, mayor el campo de accion del modelo.
- `history_provenance=EMPTY`: datos fantasma — su presencia en el CORE es un riesgo no modelado.

---

## §2 Especificacion Tecnica

### D62-01: `_load_edge_report_index()` — carga y construccion de indice

Funcion de modulo en `combo_confianza_builder.py`. Busca el `edge_report_*.json` mas reciente en
`REPORTS_DIR`, construye un dict `nombre_normalizado -> datos_del_pick`. Incluye picks de todas
las categorias: apostar, watchlist, sin_edge, no_data.

Graceful degradation: si no hay edge_report, retorna `{}` sin fallar. El combo builder sigue
funcionando con alpha_score=0 para todos los picks.

### D62-02: `_lookup_edge_data(nombre, edge_index)` — lookup con fuzzy fallback

Busca por nombre exacto normalizado (lowercase, sin puntos). Si no encuentra, intenta por apellido
(ultima palabra del nombre, largo > 3). Retorna `{}` si no hay match.

### D62-03: `_compute_alpha_score(edge_data)` — calculo de alpha por senal

Funcion de modulo exportable. Retorna `(alpha_score: float, senales_activas: list[str])`.

Pesos fijos (CONGELADOS hasta n>=30 por senal en shadow book):

| Senal | Condicion | Peso |
|---|---|---|
| triple_alignment | >= 0.5 | +15 |
| triple_alignment | >= 0.3 | +8 |
| markov_favorito | HOT | +10 |
| markov_favorito | COLD | -15 |
| gcs_bonus | True | +12 |
| edge_pct | >= 15% | +10 |
| edge_pct | >= 5% | +5 |
| surface_signal | >= 0.8 | +8 |
| surface_signal | >= 0.5 | +4 |
| bbi | >= 0.8 | +6 |
| bbi | >= 0.6 | +3 |
| calibration_confidence | >= 0.7 | +3 |
| history_provenance | EMPTY o PHANTOM | -25 |

### D62-04: Enriquecimiento de picks en `_extract_and_categorize`

Para cada pick, despues de obtener `cat`:
1. Llamar `_lookup_edge_data(favorito, edge_index)`
2. Llamar `_compute_alpha_score(edge_data)`
3. Calcular `combo_priority = round(conf + alpha_score, 2)`
4. Almacenar `alpha_score`, `alpha_senales`, `combo_priority`, `edge_data_ref` en el dict del pick

El sort final usa `combo_priority` en lugar de `confianza`.

### D62-05: Gate Cat-C1 por alpha (bypass conf>=60%)

Dentro del loop de `_extract_and_categorize`, despues de calcular `cat`:
- Si `cat.categoria == CAT_C2` y `cuota <= CUOTA_C1_MAX` (3.50):
  - Si `edge_pct >= 5%` AND `triple_alignment >= 0.2`:
    - Promover a `CAT_C1` con flag `alpha_promoted=True`

Esto corrige el caso Nilsson: edge=23.6%, triple=0.332, cuota 2.x → deberia poder entrar
al SATELLITE pero queda bloqueada por el gate de confianza.

### D62-06: Modificacion de `_select_core`

`_select_core` ahora ordena por `combo_priority` (desc) antes de iterar. Mantiene todos los
guards existentes: CORE_MAX_SIZE, MAX_SAME_TOURNAMENT, solo Cat-A + Cat-B.

### D62-07: Display de alpha en `_format_report`

Cada pick en la tabla de categorias muestra:
- `pri:{combo_priority}` cuando alpha_score != 0
- Linea `[alpha {score}: senal1, senal2, ...]` debajo del pick cuando hay senales activas
- Tag `[ALPHA-PROM]` para picks promovidos de Cat-C2 a Cat-C1 por alpha gate

---

## §3 Tests T62-01 a T62-10

Archivo: `tests/test_nodo62.py`

| Test | Descripcion | Funcion invocada |
|---|---|---|
| T62-01 | triple=0.661 → alpha >= +15 | `_compute_alpha_score` |
| T62-02 | markov=HOT → alpha >= +10 | `_compute_alpha_score` |
| T62-03 | markov=COLD → alpha <= -15 | `_compute_alpha_score` |
| T62-04 | gcs_bonus=True → alpha >= +12 | `_compute_alpha_score` |
| T62-05 | edge_pct=23.6% → alpha >= +10 | `_compute_alpha_score` |
| T62-06 | history_provenance EMPTY → alpha <= -25 | `_compute_alpha_score` |
| T62-07 | triple+HOT+surface: combo_priority >= confianza+30 | `_compute_alpha_score` |
| T62-08 | _ALPHA_C1_EDGE_MIN=5.0, _ALPHA_C1_TRIPLE_MIN=0.2 | constantes modulo |
| T62-09 | _load_edge_report_index() retorna dict | `_load_edge_report_index` |
| T62-10 | markov=COLD: combo_priority < confianza | `_compute_alpha_score` |

REGLA-T53: ningun test hardcodea la formula — todos usan constantes `_ALPHA_*` importadas del modulo.

---

## §4 Checklist de Verificacion Post-Implementacion

```bash
# 1. Tests nuevos
python -m pytest tests/test_nodo62.py -v --no-cov

# 2. Regresion completa
python -m pytest tests/ --no-cov -q | tail -5

# 3. Verificar syntax
python -c "import ast; ast.parse(open('combo_confianza_builder.py').read()); print('OK')"

# 4. Output real con Signal Bridge activo
python3 combo_confianza_builder.py --bankroll 125000 --fase 4

# Buscar en output:
#   - [alpha ...] lineas bajo picks con senales
#   - pri: en picks con alpha_score != 0
#   - [ALPHA-PROM] en picks promovidos
#   - CORE ordenado por combo_priority (no solo confianza)
```

---

## §5 Pesos Congelados y Condiciones de Recalibracion

Los pesos `_ALPHA_*` estan CONGELADOS. No modificar sin:

1. n>=30 observaciones en shadow book para la senal especifica
2. Hit rate de picks con esa senal > breakeven (1/cuota_media)
3. Aprobacion en Panel 6 del dashboard (Nodo-58)

Proceso de recalibracion:
- Correr `python3 shadow_book.py --report` filtrando por senal
- Si IC Wilson 95% lower bound > 1/cuota_media → pedir recalibracion en nueva sesion
- Documentar en nuevo nodo (Nodo-62-RECAL-v1)

Senales con mayor evidencia previa (pueden recalibrarse primero):
- `triple_alignment`: correlaciona con coherencia interna del modelo — high prior
- `gcs_bonus`: 64.8% hit rate en n=54 (A60-01) — evidence fuerte para hierba
- `markov=COLD`: penalizacion conservadora, evidencia indirecta de sesion 9/10 13-jun

Senales con menor evidencia (recalibrar ultimo):
- `bbi`: campo nuevo, n reducido
- `calibration_confidence`: depende del tier, n heterogeneo

---

## §6 Notas de Implementacion

- Graceful degradation total: si no hay edge_report, alpha_score=0 para todos los picks.
  El combo builder funciona identicamente al estado pre-Nodo-62 en ese caso.
- `_compute_alpha_score` es una funcion de modulo (no metodo de clase) — importable directamente.
- `edge_data_ref` almacenado en cada pick para trazabilidad en el shadow book (futuro).
- El sort de `_extract_and_categorize` cambia de `confianza` a `combo_priority` — esto afecta
  el orden de display y la construccion del CORE. Es un cambio intencional.
- `_select_core` re-ordena internamente por `combo_priority` para mayor robustez — aunque
  `picks_ab` ya llega ordenado por combo_priority, el doble sort es defensivo y no cuesta nada.

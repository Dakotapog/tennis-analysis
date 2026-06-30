# Nodo-08: Bug de Selección de Archivo — Prioridad Invertida

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-07-Strangler-Fig]]
> Estado: 2026-05-29 | Severidad: CRÍTICA | Tipo: Bug de Producción | Bug: línea 278

---

## Descripción del Bug

**Archivo:** `extraer_historh2h.py` línea 278
**Función:** `select_best_json_file()`
**Impacto:** Pipeline produce 0 partidos procesados aunque existen datos válidos

```python
# ❌ BUGGY (línea 278 actual):
best_file = max(valid_files, key=lambda x: (x['total_matches'], x['modified_time']))
# La tupla evalúa en orden: total_matches PRIMERO, modified_time como desempate

# Con dos archivos disponibles:
#   May 28: 423 partidos — h2h_url=None (datos del scraper roto, pre-Nodo-03)
#   May 29: 235 partidos — h2h_url válidas (datos del scraper corregido, post-Nodo-03)
# Resultado: May 28 GANA (423 > 235) aunque sus h2h_urls son None → cola = 0 partidos

# ✅ FIX (cambio de 3 tokens):
best_file = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
# Ahora: modified_time es criterio primario, total_matches es desempate
# Resultado: May 29 GANA (más reciente) → cola = 235 partidos → predicciones → P&L
```

---

## Incidente de Producción — 2026-05-29

**Secuencia:**
```
02:01:50  Rankings ATP (2260 jugadores) ✅
02:01:50  Rankings WTA (1541 jugadores) ✅
01:52:44  URL scraper → zita_tennis_matches_20260529_015244.json (235 partidos, 33 torneos) ✅
02:01:50  H2H extractor iniciado → SELECCIONÓ zita_tennis_matches_20260528_130141.json ❌
02:01:58  Cola de procesamiento: 0 partidos (todos filtrados: h2h_url=None)
02:01:58  Reporte generado: 0 resultados | P&L del día: 0
```

**Log evidencia (extraído):**
```
INFO - 🏆 Archivo seleccionado: data/zita_tennis_matches_20260528_130141.json
INFO -    📊 Partidos disponibles: 423
INFO -    📅 Modificado: 2026-05-28 13:01:41
INFO - ✅ Cola de procesamiento creada con 0 partidos
INFO -    📊 Tasa de éxito: 0.0% (No se procesaron partidos)
INFO - 💾 Resultados guardados en: reports/h2h_results_enhanced_20260529_020158.json
INFO -    📊 Total partidos: 0
```

---

## Análisis de Causa Raíz (RCA)

**Causa primaria:** La métrica de calidad de `select_best_json_file()` es incorrecta.

El número de partidos NO es indicador de calidad — lo es la RECENCIA. Un archivo más antiguo
con más partidos (h2h_url=None) es peor que uno reciente con menos partidos (h2h_url válidas).

**Causa secundaria:** No existe validación previa al procesamiento que verifique que el archivo
seleccionado tiene h2h_url válidas (porcentaje > 0%).

**Causa raíz sistémica:** La lógica de selección fue diseñada antes del fix del scraper (Nodo-03).
En ese contexto, más partidos SÍ era mejor porque todos los archivos tenían el mismo formato
(roto sin h2h_url). Después del fix (2026-05-28), los archivos nuevos tienen MENOS partidos pero
MEJORES datos — la lógica quedó obsoleta por un cambio de contexto no anticipado.

---

## Fix

**Cambio mínimo — línea 278 de `extraer_historh2h.py`:**

```python
# ANTES (buggy):
best_file = max(valid_files, key=lambda x: (x['total_matches'], x['modified_time']))

# DESPUÉS (correcto):
best_file = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
```

**Por qué este orden es correcto:**
1. `modified_time` primario: el archivo más reciente tiene datos del día (scraper actualizado)
2. `total_matches` desempate: si dos archivos tienen mismo timestamp (inusual), el mayor gana

---

## Tests Requeridos

```python
# Agregar a tests/test_file_utils.py o tests/test_h2h_extractor.py

def test_file_selection_prefers_recency_over_match_count():
    """El archivo más reciente debe ganar aunque tenga menos partidos."""
    from datetime import datetime

    valid_files = [
        {
            'filename': 'data/zita_tennis_matches_20260528_130141.json',
            'total_matches': 423,
            'modified_time': datetime(2026, 5, 28, 13, 1, 41),
            'size_mb': 0.19,
            'location': 'data'
        },
        {
            'filename': 'data/zita_tennis_matches_20260529_015244.json',
            'total_matches': 235,
            'modified_time': datetime(2026, 5, 29, 1, 52, 44),
            'size_mb': 0.13,
            'location': 'data'
        }
    ]

    best = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
    assert best['filename'] == 'data/zita_tennis_matches_20260529_015244.json', (
        "Archivo más reciente (May 29) debe ganar aunque tenga menos partidos que May 28"
    )


def test_file_selection_uses_match_count_as_tiebreaker():
    """Con igual timestamp, el archivo con más partidos debe ganar."""
    from datetime import datetime

    same_time = datetime(2026, 5, 29, 1, 52, 44)
    valid_files = [
        {'filename': 'data/a.json', 'total_matches': 100,
         'modified_time': same_time, 'size_mb': 0.1, 'location': 'data'},
        {'filename': 'data/b.json', 'total_matches': 235,
         'modified_time': same_time, 'size_mb': 0.13, 'location': 'data'},
    ]

    best = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
    assert best['filename'] == 'data/b.json', (
        "Con timestamp idéntico, el archivo con más partidos (235 > 100) debe ganar"
    )


def test_file_selection_single_file_always_wins():
    """Con un solo archivo válido, siempre debe seleccionarse."""
    from datetime import datetime

    valid_files = [
        {'filename': 'data/only.json', 'total_matches': 10,
         'modified_time': datetime(2026, 5, 29), 'size_mb': 0.01, 'location': 'data'}
    ]

    best = max(valid_files, key=lambda x: (x['modified_time'], x['total_matches']))
    assert best['filename'] == 'data/only.json'
```

---

## Validación Adicional Recomendada

Añadir log de advertencia post-selección para detectar futuros bugs similares:

```python
# Añadir después de la línea que selecciona best_file:
most_recent = max(valid_files, key=lambda x: x['modified_time'])
if best_file['filename'] != most_recent['filename']:
    logger.warning(
        f"⚠️ ANOMALÍA: Archivo seleccionado no es el más reciente. "
        f"Seleccionado: {best_file['filename']} | "
        f"Más reciente: {most_recent['filename']} — verificar lógica de selección"
    )
```

---

## Impacto en la Cadena de Datos (S1_MATCH_LIST)

```
ANTES del fix (bug activo):
zita_tennis_matches_20260529_015244.json (235 partidos, h2h válidas) → IGNORADO
zita_tennis_matches_20260528_130141.json (423 partidos, h2h=None)    → SELECCIONADO
→ S1_MATCH_LIST: 0 partidos procesables
→ S2_H2H_DATA: vacío
→ S3_EDGE: sin señales
→ P&L: 0

DESPUÉS del fix:
zita_tennis_matches_20260529_015244.json (235 partidos, h2h válidas) → SELECCIONADO ✅
→ S1_MATCH_LIST: 235 partidos
→ S2_H2H_DATA: 235 registros con Markov+Erdős (post Nodo-07 Fase 1)
→ S3_EDGE: señales APOSTAR donde P_modelo > P_implícita + 5%
→ P&L: medible
```

---

## Relación con Nodo-07 (Strangler Fig)

Este bug vive en el mismo archivo (`extraer_historh2h.py`) que el problema de clases inline.
Ambos deben corregirse en el mismo sprint:

```
ORDEN CORRECTO DE EJECUCIÓN:
T08-01 → Corregir línea 278 (este nodo)       ← sin esto, 0 partidos aunque se migre
T07-03 → Añadir imports + eliminar inline      ← sin esto, Markov+Erdős no llegan a prod
T07-04 → pytest 694 passed                    ← verificación de integridad
T07-05 → Run producción con output completo   ← validar Markov+Erdős en JSON
```

**Si se ejecuta Nodo-07 sin corregir este bug:** el pipeline seguirá seleccionando archivos
incorrectos y producirá 0 partidos aunque los imports sean perfectos.

---

## Tareas

| ID | Tarea | Estado |
|---|---|---|
| T08-01 | Corregir línea 278: `(x['modified_time'], x['total_matches'])` | ✅ 2026-05-29 nueva línea 241 |
| T08-02 | Añadir `test_file_selection_prefers_recency_over_match_count` | ✅ 2026-05-29 `tests/test_file_selection.py` |
| T08-03 | Añadir `test_file_selection_uses_match_count_as_tiebreaker` | ✅ 2026-05-29 `tests/test_file_selection.py` |
| T08-04 | Añadir `test_file_selection_single_file_always_wins` | ✅ 2026-05-29 `tests/test_file_selection.py` |
| T08-05 | Re-run pipeline y verificar que selecciona May 29 (235 partidos) | ✅ verificado en prod 2026-05-30 |
| T08-06 | Añadir log de advertencia post-selección | ✅ 2026-05-29 `extraer_historh2h.py` post línea 241 |
| T08-07 | Mismo bug en `scraping/file_utils.py` — ordenaba por `matches_with_urls` no por recencia | ✅ CORREGIDO 2026-05-31 — `select_best_json_file()` ahora ordena por `st_mtime` primero |

---

## Vinculación

- [[Mandatos-No-Negociables]] — Mandato 1: P&L positivo requiere pipeline ejecutándose correctamente
- [[Pipeline-Arquitectura]] — Paso 2 del pipeline: selección de archivo es el primer gate
- [[Nodo-03-Scraper-Fix]] — El fix del scraper creó el escenario donde archivos nuevos tienen menos partidos
- [[Nodo-07-Strangler-Fig]] — Corrección concomitante en el mismo archivo; T08-01 precede a T07-03
- [[Grafo-Dependencias-Datos]] — S1_MATCH_LIST debe ser el archivo más reciente con h2h válidas
- [[Sprint-Pipeline]] — T08-01 a T08-06 en backlog inmediato (preceden a Nodo-07)
- [[Fuentes-Datos]] — La fuente primaria de datos depende de esta selección correcta

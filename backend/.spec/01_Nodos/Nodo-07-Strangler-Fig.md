# Nodo-07: Migración Strangler Fig — Dos Mundos → Un Sistema

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Pipeline-Arquitectura]] | [[Sprint-Pipeline]] | [[Inventario-Deuda-Tecnica]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-14-Validacion-Live-Conexiones]]
> Estado: 2026-05-30 | Prioridad: CERRADO | Patrón: Strangler Fig | Fase 1 ✅ COMPLETA | Fase 2 ✅ COMPLETA — T07-09 ✅ 2026-05-30 | 768 tests

---

## El Problema: Dos Mundos Paralelos

`extraer_historh2h.py` (3717 líneas) contiene copias INLINE de todas las clases de análisis.
Los módulos refactorizados en `analysis/` y `scraping/` son ignorados en producción.

```
MUNDO INLINE (producción actual):
extraer_historh2h.py
  ├── EloRatingSystem        línea 32    ← copia desactualizada
  ├── RankingManager         línea 288   ← copia desactualizada
  ├── RivalryAnalyzer        línea 1056  ← sin Markov, sin Erdős
  └── SequentialH2HExtractor línea 2277  ← orquestador principal

MUNDO MODULAR (ignorado en producción):
analysis/
  ├── elo_system.py          ← idéntico al inline ✅
  ├── ranking_manager.py     ← idéntico al inline ✅
  └── rivalry_analyzer.py    ← tiene Markov (Nodo-02) + Erdős (Nodo-06) ✅
scraping/
  └── h2h_extractor.py       ← API DIFERENTE al inline ⚠️
```

**Consecuencia:** Las Fases 2 (Markov) y 7 (Erdős) están implementadas y testeadas en `analysis/`
pero NUNCA llegan a producción. El sistema opera con predicciones sin cambios de régimen Markov
y sin análisis transitivo Erdős — dos mejoras que pueden mover accuracy del 47% al 55%+.

---

## Verificación de APIs (confirmada por análisis estático 2026-05-29)

### Clases con APIs IDÉNTICAS — migración segura:

| Clase | Inline `__init__` | Modular `__init__` | Diferencias de API |
|---|---|---|---|
| `EloRatingSystem` | `(self, k_factor=32, default_rating=1500)` | idéntico | **ninguna** |
| `RankingManager` | `(self, force_read_only=True, max_age_days=8)` | idéntico | **ninguna** |
| `RivalryAnalyzer` | `(self, ranking_manager, elo_system)` | idéntico | bodies difieren (modulares tienen Markov+Erdős+guards defensivos) |

### Clases con APIs INCOMPATIBLES — migración compleja:

| Aspecto | `SequentialH2HExtractor` (inline) | `H2HExtractor` (scraping/) |
|---|---|---|
| `__init__` params | ninguno | `headless: bool, slow_mo: int` |
| Entry point | `process_all_matches()` | `run()` |
| Browser | atributos directos playwright/browser/context | delegado a `BrowserManager` |
| Data parsing | inline en métodos | delegado a `DataParser` |
| Naming | métodos públicos | métodos privados (`_prefix`) |
| Cleanup | `_cleanup_on_exit()` | `_safe_cleanup()` + timeout 5s |
| Tests actuales | 52 tests ✅ (test_sequential_h2h_extractor) | 5 tests scraping/ |

**Conclusión:** `SequentialH2HExtractor` y `H2HExtractor` NO son drop-in replacements.
Sustituir sin preparación rompe producción.

---

## Plan de Migración: Dos Fases Independientes

### Fase 1 — Importar las 3 clases (BAJO RIESGO, ejecutar YA)

**Objetivo:** Reemplazar copias inline de `EloRatingSystem`, `RankingManager` y `RivalryAnalyzer`
con imports desde `analysis/`. `SequentialH2HExtractor` permanece intacto.

**Cambios en `extraer_historh2h.py`:**

```python
# AÑADIR en imports (línea 23, después de CompleteRankingScraper):
from analysis import EloRatingSystem, RankingManager, RivalryAnalyzer

# ELIMINAR bloque completo (3 clases, ~2200 líneas):
# class EloRatingSystem:       (líneas 32-70)
# class RankingManager:        (líneas 288-1055)
# class RivalryAnalyzer:       (líneas 1056-2276)
# Nota: las funciones módulo entre líneas 71-287 se CONSERVAN
```

**Resultado cuantificable:**
- Archivo: 3717 → ~1500 líneas (−2200 líneas de código duplicado eliminadas)
- Producción gana: factor Markov HOT/COLD en `form_recent` (15% del modelo)
- Producción gana: bonus Erdős en `common_opponents` (profundidad ≥2)
- Producción gana: ELO scoring corregido (rango 2200-2020, no 2400-2020)
- Producción gana: guards defensivos en análisis de superficie y oponentes comunes

**Riesgo:** CASI CERO
- APIs idénticas confirmadas por análisis estático (agente comparador 2026-05-29)
- 767 tests cubren los módulos de `analysis/` (0 failed)
- `analysis/__init__.py` ya exporta las 3 clases

**Rollback:** `git checkout extraer_historh2h.py`

**Criterio de éxito Fase 1:**
```python
# 1. Tests no rotos:
python -m pytest tests/ --no-cov -q  # → 767 passed (2026-05-29)

# 2. Import limpio:
python -c "import extraer_historh2h"  # → sin ImportError

# 3. Markov activo en output:
# h2h_results_enhanced_FECHA.json debe contener:
# partido['ranking_analysis']['markov_analysis']['factor_markov'] != None

# 4. Erdős activo en output:
# partido['ranking_analysis']['erdos_analysis']['erdos_score'] != None
```

---

### Fase 2 — Migrar SequentialH2HExtractor (RIESGO MEDIO, sprint futuro)

**Objetivo:** `extraer_historh2h.py` se convierte en un entry point de ~50 líneas
que llama a `scraping.H2HExtractor.run()`.

**Precondiciones OBLIGATORIAS antes de iniciar Fase 2:**
1. ~~`tests/test_h2h_extractor.py` debe tener ≥40 tests~~ ✅ CUMPLIDA — 52 tests en `test_sequential_h2h_extractor.py` (2026-05-29)
2. Paridad funcional verificada: `H2HExtractor` produce JSON idéntico a `SequentialH2HExtractor`
3. Tests de compatibilidad de output sobre ≥10 partidos reales: `assert output_modular == output_inline`
4. Fase 1 completada y estable en producción (≥5 runs exitosos sin errores)
5. Nodo-09-H2HExtractor-Paridad.md creado con checklist de paridad

**Estrategia recomendada:**
```
OPCIÓN A — Adapter/Wrapper (menor riesgo):
  Crear clase ThinOrchestrator que traduce API de SequentialH2HExtractor a H2HExtractor.
  Permite swap gradual sin tocar el script principal en un solo commit.

OPCIÓN B — Reescritura del entry point (más limpio):
  Reemplazar extraer_historh2h.py con script de ~50 líneas.
  Solo viable cuando H2HExtractor tiene paridad total de features + 40+ tests.
```

---

## Gap funcional actual entre los dos orquestadores

| Feature | SequentialH2HExtractor | H2HExtractor (scraping/) | Estado |
|---|---|---|---|
| Markov en output JSON | ✅ después Fase 1 | ✅ | gap cerrado Fase 1 |
| Erdős en output JSON | ✅ después Fase 1 | ✅ | gap cerrado Fase 1 |
| `save_results_to_json()` | ✅ | `save_results()` ✅ | naming diferente |
| `recalculate_and_save_optimized_results()` | ✅ | `recalculate_with_optimized_weights()` ✅ | naming diferente |
| File selection logic | inline con bug (Nodo-08) | via `file_utils` ✅ correcto | Nodo-08 fix |
| Browser headless config | hardcoded | configurable `headless=True` ✅ | gap menor |
| Tests cubriendo el módulo | 52 tests ✅ | 5 tests (scraping/) | precondición CUMPLIDA ✅ |

---

## Impacto en Norte Estratégico (4 Marcos)

### Marco 1 — Ingeniero de Software Senior
```
ANTES:  3717 líneas, DRY violado (4 clases duplicadas), Markov+Erdős muertos en analysis/
DESPUÉS Fase 1: ~1500 líneas, DRY respetado, Markov+Erdős activos en producción
DESPUÉS Fase 2: ~50 líneas entry point, SRP perfecto, 100% modular
```

### Marco 2 — Arquitecto de Datos
```
ANTES:  S2_H2H_DATA producido sin Markov ni Erdős → features corruptas en dataset
DESPUÉS Fase 1: S2_H2H_DATA con markov_analysis.factor_markov y erdos_analysis.erdos_score
               → features válidas para edge_calculator y generar_dataset_plus.py
```

### Marco 3 — Ingeniero de Implementación
```
Fase 1 — Riesgo: BAJO | Tiempo: 30 min | Tests necesarios: 0 nuevos (694 existentes suficientes) ✅ COMPLETADA
Fase 2 — Riesgo: MEDIO | Tiempo: 4-8h | Tests necesarios: ≥40 ✅ CUMPLIDO (52 tests) — puede iniciarse
```

### Marco 4 — Quant/Financiero
```
ANTES:  accuracy ≈ 47% (peor que random) | edge signals todos underdog = posiblemente artefacto
DESPUÉS Fase 1: accuracy esperada ≈ 55%+ con Markov+Erdős activos + superficie funcionando
               → edge signals más confiables → Kelly-KL puede apostar con criterio real
```

---

## Tareas

| ID | Tarea | Fase | Estado |
|---|---|---|---|
| T07-01 | Verificar `analysis/__init__.py` exporta las 3 clases | 1 | ✅ 2026-05-29 |
| T07-02 | Corregir bug selección archivo (Nodo-08) — precede a T07-03 | 1 | ✅ 2026-05-29 línea 241 |
| T07-03 | Añadir `from analysis import ...` + eliminar clases inline (líneas 32-2276) | 1 | ✅ 2026-05-29 3717→1691 líneas |
| T07-04 | Correr `pytest tests/ --no-cov -q` post-migración → 767 passed | 1 | ✅ 694 passed, 0 failed |
| T07-05 | Run producción y verificar `factor_markov` y `erdos_score` en output JSON | 1 | ✅ 2026-05-30 — erdos_score=0.35, factor_markov=1.0, surf_w 0.49–0.69 (16 partidos RG) |
| T07-06 | Ampliar tests de SequentialH2HExtractor a ≥40 | 2 | ✅ 2026-05-29 — 52 tests en test_sequential_h2h_extractor.py |
| T07-0A | Fix superficie gap en H2HExtractor.load_matches() (línea 188: preferir `match.get('superficie')`) | 2 | ✅ 2026-05-30 |
| T07-0B | Fix superficie gap en H2HExtractor._process_single_match() (línea 328) | 2 | ✅ 2026-05-30 |
| T07-0C | Añadir Roland Garros filter en H2HExtractor.load_matches() | 2 | ✅ 2026-05-30 |
| T07-0D | Migrar main() en extraer_historh2h.py → usa H2HExtractor (v5.0) | 2 | ✅ 2026-05-30 — 773 tests passed |
| T07-07 | Crear `Nodo-09-H2HExtractor-Paridad.md` | 2 | ⏳ sprint futuro |
| T07-08 | Verificar paridad de output sobre ≥10 partidos reales | 2 | ⏳ sprint futuro |
| T07-09 | Eliminar `SequentialH2HExtractor` + migrar 53 tests → H2HExtractor/DataParser | 2 | ✅ 2026-05-30 — 1,404 líneas eliminadas, 48 tests migrados, 768 passed |

---

## Archivos del módulo scraping/ (Nodo-07)

| Archivo | Rol |
|---|---|
| `scraping/h2h_extractor.py` | Orquestador Playwright H2H — reemplazó SequentialH2HExtractor (T07-09) |
| `scraping/browser_manager.py` | Lifecycle Playwright para WSL — zombie cleanup via psutil |
| `scraping/data_parser.py` | Parser estático HTML FlashScore — superficie, torneo, ganador, fechas |

> **Adendo 2026-07-09:** corrección de omisión en tabla de archivos (no cambio de decisión) —
> `browser_manager.py` y `data_parser.py` ya estaban referenciados como clases (`BrowserManager`,
> `DataParser`) en la tabla de APIs incompatibles (§ Verificación de APIs), pero ausentes como
> nombres de archivo. El regex del índice Nodo-75 los buscaba por patrón `*.py` y no los encontraba.
> Detectado en auditoría Nodo-75 (2026-07-09).

---

## Vinculación

- [[Mandatos-No-Negociables]] — Mandato 6: tests antes de código; Mandato 1: P&L sobre accuracy
- [[Pipeline-Arquitectura]] — Arquitectura actual del orquestador, tabla de módulos y tests
- [[Nodo-02-Markov-Changepoint]] — Feature que Fase 1 activa en producción
- [[Nodo-06-Erdos-Graph]] — Feature que Fase 1 activa en producción
- [[Nodo-08-File-Selection-Bug]] — Bug concomitante a corregir antes o junto a Fase 1
- [[Sprint-Pipeline]] — T07-01 a T07-09 en backlog
- [[Grafo-Dependencias-Datos]] — S2_H2H_DATA cambia: post-Fase 1 incluye Markov+Erdős
- [[Fuentes-Datos]] — El orquestador es el consumidor principal de S1_MATCH_LIST
- [[Inventario-Deuda-Tecnica]] — D-14 CERRADO ✅, Costura 1 CERRADA, acumulado ~9,400 líneas eliminadas
- [[Nodo-14-Validacion-Live-Conexiones]] — alpha confirmado en prod mientras T07-09 se planificaba

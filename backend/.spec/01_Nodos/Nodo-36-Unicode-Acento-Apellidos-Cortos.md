# Nodo-36 — Fix B: Acento Unicode en KB Headers + Fix C: Apellidos de 2 Caracteres

> **Fecha:** 2026-06-25
> **Severidad:** MEDIA — Dos bugs en `_name_tokens()` que impiden identificar el bloque correcto para ciertos jugadores, produciendo historial vacío o bloques invertidos.
> **Prerequisitos:** [[Nodo-34-Corrupcion-Datos-Extraccion-H2H]] — Nodo-36 es continuación directa: Fix A y Fix B de Nodo-34 corrigieron el score y el ranking; Nodo-36 corrige dos bugs adicionales en el matching de KB headers en `_name_tokens()`.
> **Archivos afectados:** `scraping/ninja_h2h_parser.py`
> **Implementa:** Sonnet | **Tests:** Haiku
>
> **Estado:** ✅ CERRADO — 2026-06-25

---

## 0. RESUMEN EJECUTIVO

La función `_name_tokens()` en `ninja_h2h_parser._process_match()` (introducida en Nodo-34 para soportar apellidos compuestos tipo "Davidovich Fokina") tiene dos bugs adicionales:

**Fix B — Acento Unicode:** FlashScore retorna KB headers con tildes (`"Fernández B."`). La comparación `"fernandez" in "fernández b."` es `False` porque `é ≠ e`. Resultado: jugadores con apellidos acentuados (Fernández, Almagro, Sousa, Fucsovics) no son encontrados en ningún bloque → bloque no asignado → historial vacío.

**Fix C — Apellidos de 2 caracteres:** El filtro `len(t) > 2` excluye tokens de exactamente 2 caracteres. Apellidos asiáticos comunes como `"Lu"` (Jing-Jing Lu) y `"Mi"` (Lan Mi) tienen longitud 2 → excluidos de tokens → bloque no identificado → historial vacío. Además se requiere **word-boundary guard**: cambiar el filtro a `len(t) > 1` sin protección adicional permitiría que `"mi"` matchee `"michelsen"` (falso positivo).

Ambos fixes aplican solo a `_process_match()` en `ninja_h2h_parser.py`. No afectan el resto del pipeline.

---

## 1. CONTEXTO DEL HALLAZGO

Detectado 2026-06-25 durante la implementación de Nodo-35. Al auditar los casos de historial vacío descubiertos en la sesión del día, se identificaron dos patrones adicionales distintos del apellido compuesto (ya resuelto por Nodo-34):

**Patrón Fix B:** Bruno Fernandez llegó con 0 partidos extraídos. El KB header en la API era `"Últimos partidos: Fernández B."` — con tilde. El token generado era `"fernandez"` (sin tilde). La comparación directa de strings falla: `"fernandez" not in "fernández b."`.

**Patrón Fix C:** Jing-Jing Lu y Lan Mi llegaron con 0 partidos extraídos. `"Lu"` y `"Mi"` tienen `len == 2`, por lo que `len(t) > 2` los excluía de `_name_tokens()`. La función retornaba `['jing-jing']` para Jing-Jing Lu y `['lan']` para Lan Mi — tokens que no aparecen en los KB headers de FlashScore (que usa apellido + inicial).

---

## 2. DIAGNÓSTICO TÉCNICO

### 2.1 Función afectada

```python
# ANTES (Nodo-34 — solo Fix para apellidos compuestos)
def _name_tokens(name: str) -> List[str]:
    return [t.lower() for t in name.split() if len(t) > 2] if name != 'N/A' else []

# Uso en comparación
p1_in_block1 = bool(p1_tokens) and any(tok in kb.lower() for tok in p1_tokens for kb in main_kbs[:1])
```

### 2.2 Fix B — Normalización de acentos Unicode

```python
# NFD descompone "é" en "e" + combining accent → category 'Mn' → eliminado
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')
```

### 2.3 Fix C — Cambio de filtro + word-boundary para tokens cortos

```python
# len > 1 en vez de len > 2 → incluye "Lu", "Mi"
def _name_tokens(name: str) -> List[str]:
    return [_strip_accents(t.lower()) for t in name.split() if len(t) > 1] if name != 'N/A' else []

# word-boundary guard para tokens cortos (len ≤ 2):
# "mi" in "michelsen".split() → False  (correcto)
# "mi" in "Mi L.".split()    → True   (correcto)
def _token_in_kb(tok: str, kb: str) -> bool:
    kb_norm = _strip_accents(kb.lower())
    if len(tok) <= 2:
        return tok in kb_norm.split()
    return tok in kb_norm
```

---

## 3. CASOS REALES DOCUMENTADOS

| Jugador | Problema | Token generado | KB real FlashScore | Resultado pre-fix |
|---|---|---|---|---|
| Bruno Fernandez | Fix B | `"fernandez"` | `"Fernández B."` | `é ≠ e` → historial vacío |
| Jing-Jing Lu | Fix C | `["jing-jing"]` (Lu excluido) | `"Lu J."` | token no matchea → historial vacío |
| Lan Mi | Fix C | `["lan"]` (Mi excluido) | `"Mi L."` | token no matchea → historial vacío |

---

## 4. RIESGO DOCUMENTADO: Ambigüedad Andy Nguyen / Avery Nguyen

Con Fix C, el token `"nguyen"` (len=6, no afectado) ya funcionaba antes. El riesgo documentado aquí es distinto: dos jugadoras del circuito ITF norteamericano con el mismo apellido e inicial comparten el formato de KB header `"Nguyen A."`.

- **Andy Nguyen** (WTA, ~rank 150-200)
- **Avery Nguyen** (ITF junior/W15)

Si ambas juegan torneos el mismo día y el sistema procesa sus match_ids en la misma sesión, los tokens `["andy", "nguyen"]` vs `["avery", "nguyen"]` matchean ambos el mismo KB header `"Nguyen A."`. El sistema resuelve la asignación de bloques POR MATCH_ID (cada extracción es por partido individual) — la ambigüedad solo ocurriría si los match_ids de ambas están en el mismo registro Ninja, lo cual no sucede. **Riesgo: BAJO — documentado, no requiere fix.**

---

## 5. IMPLEMENTACIÓN

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `scraping/ninja_h2h_parser.py` | + `import unicodedata` en línea 22 |
| `scraping/ninja_h2h_parser.py` | + `_strip_accents()` helper en `_process_match()` |
| `scraping/ninja_h2h_parser.py` | `_name_tokens()`: `len>2` → `len>1` + `_strip_accents()` en tokens |
| `scraping/ninja_h2h_parser.py` | + `_token_in_kb()` helper con word-boundary para tokens cortos |
| `scraping/ninja_h2h_parser.py` | Comparaciones `tok in kb.lower()` → `_token_in_kb(tok, kb)` |

### Tests

- `tests/test_nodo36.py` — 21 tests, 0 failed
  - `TestFixB_AccentNormalization` (5 tests): detección de mutación Fix B
  - `TestFixC_ShortSurnames` (8 tests): detección de mutación Fix C + word-boundary
  - `TestNoRegression` (8 tests): casos existentes sin regresión

---

## 6. CIERRE

**Fecha cierre:** 2026-06-25
**Tests antes:** 1285 passed, 0 failed (Nodo-35 incluido)
**Tests después:** 1306 passed, 0 failed
**Tests añadidos:** 21 (test_nodo36.py)

### Fixes aplicados con mutación confirmada

**Fix B (T36-01 detecta mutación):**
- `test_fernandez_mutation_detection`: comportamiento antiguo (sin _strip_accents) da `pre_fix_result=False`. Fix nuevo da `True`. Cualquier rollback de _strip_accents rompe T36-01.

**Fix C (T36-02 y T36-03 detectan mutación):**
- `test_fix_c_mutation_pre_fix_lu_missing`: confirma que `len>2` excluye "lu". Fix nuevo lo incluye.
- `test_word_boundary_guard_mi_vs_michelsen`: confirma que sin word-boundary `"mi" in "michelsen"` = True (falso positivo). Fix nuevo con `.split()` da False.

### Implicación operacional

- Jugadores con apellidos acentuados y apellidos de 2 caracteres ahora tienen historial extraído correctamente.
- Si un jugador que antes llegaba con 0 partidos ahora llega con historial real, la predicción cambia — el modelo tiene más información.
- No requiere re-extracción del historial histórico. Solo afecta nuevas extracciones post-2026-06-25.

---

## 7. WIKILINKS

- [[Nodo-34-Corrupcion-Datos-Extraccion-H2H]] — Fix A (score) + Fix B (ranking substring) en mismo archivo; Nodo-36 es continuación
- [[Nodo-35-Historial-Vacio-Flag-Pipeline]] — gate en edge_calculator que bloquea señales sin historial (complementario)
- [[Nodo-31-Future-Match-Data-Leakage]] — anti-leakage en ninja_h2h_parser (mismo archivo)
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint

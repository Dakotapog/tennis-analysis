# Nodo-04: Fix generar_dataset_plus.py (2 Bugs Críticos)

> **Wikilinks:** [[Mandatos-No-Negociables]] | [[Sprint-Pipeline]] | [[Pipeline-Arquitectura]] | [[Grafo-Dependencias-Datos]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]]
> **Bloquea:** Pipeline ML completo — sin dataset limpio, no hay modelo entrenado

**Prioridad:** ALTA — bloquea `aplicar_enhancer.py` y todo el ML pipeline
**Archivo:** `generar_dataset_plus.py`
**Dependencia:** `reports/h2h_results_enhanced_FECHA.json` (output de [[Nodo-03-Scraper-Fix]])

---

## Contrato de Señal (Signal Contract)

```
PRODUCE:  ml_datasets/dataset_plus_FECHA.csv
          ml_datasets/feature_names.json

CONSUME:  reports/h2h_results_enhanced_FECHA.json
          data/atp_rankings_complete_FECHA.json

PREREQUISITO: h2h_results_enhanced debe tener torneo ≠ "Sin Torneo" 
              (requiere Nodo-03 completado)
```

---

## Bug 1: KNN Imputer Shape Mismatch

**Error exacto:**
```
ValueError: Shape of passed values is (100, 71), indices imply (100, 79)
```

**Causa:** El `KNNImputer` de sklearn transforma solo las columnas numéricas y devuelve
un array con forma `(n_filas, n_cols_numéricas)`. Al reconstruir el DataFrame, pandas
espera 79 columnas pero recibe 71 — 8 columnas se pierden (probablemente las categóricas
excluidas del imputer pero incluidas en el índice de columnas).

**Localizar:**
```bash
grep -n "KNNImputer\|fit_transform\|DataFrame(imputed" generar_dataset_plus.py
```

**Fix:**
```python
# ANTES (roto):
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
df_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# DESPUÉS (correcto):
# Separar columnas numéricas de categóricas
cols_numericas = X.select_dtypes(include=['number']).columns.tolist()
cols_otras = X.select_dtypes(exclude=['number']).columns.tolist()

imputer = KNNImputer(n_neighbors=5)
X_num_imputed = imputer.fit_transform(X[cols_numericas])

# Reconstruir solo sobre columnas numéricas
df_num = pd.DataFrame(X_num_imputed, columns=cols_numericas, index=X.index)

# Reunir con columnas no numéricas
df_imputed = pd.concat([df_num, X[cols_otras]], axis=1)

# Verificar que no se perdieron columnas
assert df_imputed.shape[1] == X.shape[1], \
    f"Shape mismatch: esperado {X.shape[1]}, obtenido {df_imputed.shape[1]}"
```

---

## Bug 2: SmartLogger.error() no existe

**Error exacto:**
```
AttributeError: 'SmartLogger' object has no attribute 'error'
```

**Causa:** La clase `SmartLogger` implementa `.info()` y `.warning()` pero no `.error()`.
El código llama `logger.error(...)` en al menos un bloque `except`.

**Localizar:**
```bash
grep -n "logger\.error\|class SmartLogger" generar_dataset_plus.py
```

**Fix — Opción A (agregar método, preferida):**
```python
# En la clase SmartLogger, agregar:
def error(self, message: str, **kwargs):
    """Proxy para logging.error — mismo comportamiento que warning pero nivel ERROR."""
    self.logger.error(message, **kwargs)
    self.error_count = getattr(self, 'error_count', 0) + 1
```

**Fix — Opción B (si SmartLogger está en módulo compartido):**
```python
# Cambiar llamadas logger.error() → logger.warning() donde el error no es fatal
# Usar logger.error() solo donde queremos crash explícito
try:
    ...
except Exception as e:
    logger.warning(f"Error procesando partido: {e}")  # no fatal
    continue
```

---

## Conexiones Cross-Nodo (CX)

| CX | Conexión | Impacto |
|---|---|---|
| CX-03 | [[Nodo-03-Scraper-Fix]] → surface/torneo limpios | Sin fix scraper, surface_specialization=0 → feature inútil en ML |
| CX-04 | [[Nodo-01-Edge-Calculator]] → edge como feature | Edge histórico puede ser feature predictiva |
| CX-05 | [[Nodo-02-Markov-Changepoint]] → markov_factor | estado HOT/COLD como feature binaria en dataset |

---

## Output Esperado Post-Fix

```
ml_datasets/dataset_plus_20260528.csv
  → 79 columnas (sin pérdida por KNN)
  → surface_specialization > 0 (requiere Nodo-03)
  → sin NaN en columnas numéricas (KNN imputer correcto)
  → SmartLogger sin crashes

ml_datasets/feature_names.json
  → lista de 79 features con tipos
```

---

## Tests Requeridos

```python
# tests/test_dataset_generator.py
def test_knn_imputer_no_pierde_columnas():
    """El dataset imputed debe tener las mismas columnas que el input."""
    df = pd.DataFrame({
        'num1': [1.0, None, 3.0],
        'num2': [None, 2.0, 3.0],
        'cat1': ['a', 'b', 'c']
    })
    resultado = aplicar_knn_imputer(df)
    assert resultado.shape[1] == df.shape[1]
    assert list(resultado.columns) == list(df.columns)

def test_smart_logger_tiene_metodo_error():
    """SmartLogger debe responder a .error() sin AttributeError."""
    from generar_dataset_plus import SmartLogger  # o donde esté
    logger = SmartLogger("test")
    try:
        logger.error("test error message")
    except AttributeError:
        pytest.fail("SmartLogger.error() no existe")

def test_dataset_sin_nan_en_numericas():
    """Después del imputer no deben quedar NaN en columnas numéricas."""
    resultado = aplicar_knn_imputer(df_con_nans)
    cols_num = resultado.select_dtypes(include=['number']).columns
    assert resultado[cols_num].isnull().sum().sum() == 0
```

---

## Ciclo de Vida

```
Estado:   ROTO (2 crashes confirmados en pipeline real 2026-05-28)
Fix:      ~2 horas
Test:     ~1 hora
Deploy:   python3 generar_dataset_plus.py → ml_datasets/
Post-fix: ejecutar aplicar_enhancer.py (actualmente BLOQUEADO por este nodo)
```

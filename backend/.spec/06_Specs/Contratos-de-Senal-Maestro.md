# Contratos de Señal Maestro — Tennis Prediction Engine

> **Wikilinks:** [[MOC-Principal]] | [[Mandatos-No-Negociables]] | [[Grafo-Dependencias-Datos]] | [[Pipeline-Arquitectura]] | [[Fuentes-Datos]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]] | [[Inventario-Deuda-Tecnica]]
> Estado: 2026-05-29 | Autor: Análisis estático profundo sin ejecución
> Propósito: Fuente única de verdad para el flujo de datos. Elimina ambigüedad. Previene regresiones silenciosas.

---

## 0. Principios Fundacionales

### 0.1 Corrección Crítica — No todas las señales son archivos

El [[Grafo-Dependencias-Datos]] lista 8 señales canónicas (S1–S8) como si fueran archivos independientes.
Esta representación es **inexacta en dos señales** y esa inexactitud tiene consecuencias:

| Señal | Naturaleza real | Error en la documentación anterior |
|---|---|---|
| S4_PREDICTION | Campo anidado dentro de S2 | Se documenta como si fuera un archivo separado |
| S7_MARKOV | Sub-campo de S4, dentro de S2 | Se documenta como si tuviera un productor independiente |

**Consecuencia arquitectónica:** S4 y S7 no tienen garantías de idempotencia propias — heredan las de S2. Un bug en el scraper de S2 contamina S4 y S7 automáticamente, sin ninguna posibilidad de aislar la transformación. El Strangler Fig (Nodo-07 Fase 2) corregirá esto separando el scraping (S2_RAW) de la predicción (S4, S7 como transformaciones puras).

### 0.2 Taxonomía de Señales por Naturaleza

```
CAPA 1 — RAW (side-effectful, snapshots externos):
  S1_MATCH_LIST   → archivo en disco, inmutable una vez creado
  S3_RANKINGS     → archivo en disco, inmutable una vez creado

CAPA 2 — REFINED (side-effectful scraping + análisis inline):
  S2_H2H_DATA     → archivo en disco, contiene S4 y S7 embebidos
  ⚠️ VIOLACIÓN SRP: S2 mezcla extracción (impura) con análisis (debería ser pura)
  ⚠️ META post-Nodo-07 Fase 2: separar en S2_RAW (scraping) + S2_ANALYZED (transformación pura)

CAPA 3 — CURATED (deben ser transformaciones puras de sus inputs):
  S4_PREDICTION   → campo dentro de S2 (hoy) | archivo propio (meta arquitectónica)
  S5_EDGE         → archivo en disco, función pura de S4 + cuotas de S1
  S7_MARKOV       → sub-campo de S4 (hoy) | módulo independiente (meta arquitectónica)

CAPA 4 — ANALYTICS (dependen de outcomes reales, son temporales):
  S6_RESULTADO_REAL → archivo en disco, depende de S1.match_id + API dc_1
  S8_DATASET_ML     → archivo en disco, función de S2 + S6
```

### 0.3 Principios de Ingeniería de Datos Funcional

**Principio 1 — Inmutabilidad temporal:**
Una señal capturada es un hecho sobre el mundo en el instante T de captura.
`zita_tennis_matches_20260529_015244.json` describe el estado de FlashScore a las 01:52:44.
No puede ser "actualizada" — solo puede ser reemplazada por una nueva captura con nuevo timestamp.
Violar esto (re-ejecutar el scraper y sobreescribir) destruye la reproducibilidad del pipeline.

**Principio 2 — Transformaciones puras:**
S5_EDGE debería ser determinista: `f(S2_file, S1_file) → S5_file`.
Si se ejecuta dos veces sobre los mismos inputs, debe producir bytes idénticos.
Actualmente S5 lee `p_historica` de un archivo de calibración que puede cambiar entre runs —
esto viola la pureza. La solución: `p_historica` debe ser un parámetro explícito del contrato, no un side-input implícito.

**Principio 3 — Contratos verificables:**
Cada señal tiene un contrato de salida. Un contrato roto es detectado en el momento de producción
(validación del schema), no cuando falla un consumidor downstream horas después.
Actualmente: `h2h_url=None` en S1 se detecta solo cuando S2 produce 0 partidos.
Debería detectarse en el momento en que S1 se escribe.

**Principio 4 — Trazabilidad de contaminación:**
Toda señal derivada debe poder responder: "¿con qué versión de sus dependencias fue producida?"
Esto requiere que S2, S5, S6 incluyan en su metadata los hashes/timestamps de los archivos que los produjeron.
Actualmente ninguno lo hace — la contaminación del Nodo-03 fue indetectable hasta que el sistema llevaba meses produciendo surface_specialization=0%.

---

## 1. Contratos de Señal — S1_MATCH_LIST

### 1.1 Definición

**Productor:** `extraer_URL_partidos_version2.py` → `ZitaScraper`
**Consumidores:** `extraer_historh2h.py` (S2), `generar_tabla_favoritos*.py`
**Tipo de señal:** RAW — snapshot externo, side-effectful, inmutable post-captura
**SLA de frescura:** máximo 12 horas (Roland Garros: renovar antes de las 10:00 CEST)

### 1.2 Schema de Salida (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S1_MATCH_LIST",
  "type": "object",
  "description": "Dict indexado por nombre de torneo. Producido por ZitaScraper.",
  "additionalProperties": {
    "type": "array",
    "items": {
      "type": "object",
      "required": ["jugador1", "jugador2", "match_url", "h2h_url", "match_id",
                   "cuota1", "cuota2", "torneo_completo", "superficie"],
      "properties": {
        "jugador1":        { "type": "string", "minLength": 2 },
        "jugador2":        { "type": "string", "minLength": 2 },
        "match_url":       { "type": "string", "pattern": "^https://www\\.flashscore\\.com/match/" },
        "h2h_url":         {
          "type": "string",
          "pattern": "^https://www\\.flashscore\\.com/match/.+/#/h2h/overall/$",
          "description": "NUNCA null post-Nodo-03. Derivada de match_url + '#/h2h/overall/'"
        },
        "match_id":        {
          "type": "string",
          "minLength": 6,
          "not": { "const": "tennis" },
          "description": "event_id del parámetro ?mid= en match_url. NUNCA 'tennis'."
        },
        "cuota1":          { "type": ["number", "null"], "minimum": 1.0 },
        "cuota2":          { "type": ["number", "null"], "minimum": 1.0 },
        "torneo_completo": { "type": "string", "minLength": 3 },
        "superficie":      {
          "type": "string",
          "enum": ["clay", "grass", "hard", "indoor", "unknown"],
          "description": "Derivada de torneo_completo. 'unknown' si no se puede inferir."
        }
      }
    }
  },
  "invariants": {
    "no_html_garbage": "len(torneo_completo) < 120 — strings > 120 chars son HTML",
    "h2h_url_not_null": "h2h_url IS NOT NULL para todos los partidos",
    "match_id_not_tennis": "match_id != 'tennis' para todos los partidos",
    "cuotas_consistency": "si cuota1 existe, cuota2 debe existir y viceversa"
  }
}
```

### 1.3 Bugs Conocidos y Estado

| Campo | Estado pre-Nodo-03 | Estado post-Nodo-03 (código) | Estado en producción |
|---|---|---|---|
| `h2h_url` | `null` (0/423) | derivada correctamente | ⚠️ NO VALIDADO en prod |
| `match_id` | `"tennis"` (423/423) | `event_id` real | ⚠️ NO VALIDADO en prod |
| `torneo_completo` | HTML 2000+ chars | nombre limpio | ⚠️ NO VALIDADO en prod |
| `superficie` | `null` | enum válido | ⚠️ NO VALIDADO en prod |

**T03-06 (bloqueante para S6, S8):** ejecutar `extraer_URL_partidos_version2.py` en producción y verificar que los 4 campos pasan el contrato.

---

## 2. Contratos de Señal — S2_H2H_DATA

### 2.1 Definición

**Productor:** `extraer_historh2h.py` → `SequentialH2HExtractor`
**Consumidores:** `edge_calculator.py` (S5), `generar_tabla_favoritos*.py`, `generar_dataset_plus.py` (S8)
**Tipo de señal:** REFINED — mezcla extracción Playwright (impura) con análisis (debería ser pura)
**SLA de frescura:** producido después de S1 y S3 del mismo día
**Dependencias:** S1 (h2h_url para scraping), S3 (rankings para predicción)

### 2.2 Schema de Salida (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S2_H2H_DATA",
  "type": "object",
  "required": ["metadata", "partidos"],
  "properties": {
    "metadata": {
      "type": "object",
      "required": ["fecha_extraccion", "total_partidos_procesados", "version"],
      "properties": {
        "fecha_extraccion":          { "type": "string", "format": "date-time" },
        "total_partidos_procesados": { "type": "integer", "minimum": 1 },
        "version":                   { "type": "string" }
      }
    },
    "partidos": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["jugador1", "jugador2", "cuota1", "cuota2",
                     "match_url", "torneo_completo", "ranking_analysis"],
        "properties": {
          "jugador1":         { "type": "string" },
          "jugador2":         { "type": "string" },
          "cuota1":           { "type": ["number", "null"] },
          "cuota2":           { "type": ["number", "null"] },
          "match_url":        { "type": "string" },
          "torneo_completo":  { "type": "string" },
          "ranking_analysis": { "$ref": "#/$defs/ranking_analysis" }
        }
      }
    }
  },
  "$defs": {
    "ranking_analysis": {
      "type": "object",
      "required": ["prediction"],
      "properties": {
        "prediction":            { "$ref": "#/$defs/prediction" },
        "erdos_analysis":        { "$ref": "#/$defs/erdos_analysis" },
        "common_opponents_count":{ "type": "integer", "minimum": 0 },
        "p1_rivalry_score":      { "type": "number" },
        "p2_rivalry_score":      { "type": "number" }
      },
      "patternProperties": {
        "^.+_ranking$": { "type": ["integer", "null"] },
        "^.+_elo$":     { "type": "number" },
        "^.+_metrics$": { "type": "object" }
      }
    },
    "prediction": {
      "type": "object",
      "required": ["favored_player", "confidence"],
      "properties": {
        "favored_player": {
          "type": "string",
          "description": "ÚNICA fuente de verdad de predicción. NO usar partido.prediccion_ganador (siempre null)."
        },
        "confidence": { "type": "number", "minimum": 50.0, "maximum": 95.0 },
        "markov_analysis": { "$ref": "#/$defs/markov_analysis" },
        "score_breakdown": { "type": "object" },
        "weights_used":    { "type": "object" }
      }
    },
    "markov_analysis": {
      "type": ["object", "null"],
      "description": "null si no hay suficiente historial para calcular Markov.",
      "properties": {
        "jugador1": { "$ref": "#/$defs/markov_estado" },
        "jugador2": { "$ref": "#/$defs/markov_estado" },
        "factor_markov": {
          "type": "number",
          "minimum": 0.85,
          "maximum": 1.15,
          "description": "1.0 = neutral, >1.0 = jugador1 HOT vs jugador2 NEUTRAL/COLD"
        }
      }
    },
    "markov_estado": {
      "type": "object",
      "properties": {
        "estado_actual":    { "type": "string", "enum": ["HOT", "COLD", "NEUTRAL"] },
        "momentum":         { "type": "number" },
        "change_point":     { "type": "integer" },
        "confianza":        { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "win_rate_reciente":{ "type": "number" },
        "win_rate_anterior":{ "type": "number" }
      }
    },
    "erdos_analysis": {
      "type": ["object", "null"],
      "description": "null si no hay oponentes comunes para construir el grafo.",
      "properties": {
        "erdos_score":         { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "erdos_score_raw":     { "type": "number" },
        "n_paths":             { "type": "integer", "minimum": 0 },
        "max_depth_alcanzado": { "type": "integer", "minimum": 0 },
        "paths":               { "type": "array" }
      }
    }
  },
  "critical_paths": {
    "prediccion_correcta":  "partidos[*].ranking_analysis.prediction.favored_player",
    "prediccion_incorrecta": "partidos[*].prediccion_ganador (SIEMPRE null — no usar)",
    "markov_correcto":      "partidos[*].ranking_analysis.prediction.markov_analysis.factor_markov",
    "erdos_correcto":       "partidos[*].ranking_analysis.erdos_analysis.erdos_score"
  }
}
```

### 2.3 Invariantes de Integridad

```
INVARIANTE-S2-01: total_partidos_procesados >= 1
  Violación detectada: h2h_results_enhanced_20260529_115349.json con 0 partidos
  Causa raíz: bug en file selection (Nodo-08) + bug en queue slice [25:3]
  Fix: Nodo-08 + filtro Roland Garros

INVARIANTE-S2-02: prediction.favored_player IS NOT NULL para todo partido procesado
  Si favored_player es null, el partido no pudo ser analizado (datos insuficientes)
  Acción: excluir de S5, no contar en métricas de accuracy

INVARIANTE-S2-03: erdos_analysis presente en ranking_analysis (post-2026-05-29)
  Pre-fix línea 1256: erdos_analysis se omitía del dict ranking_analysis
  Post-fix: erdos_analysis: {"erdos_score": float | null, "n_paths": int, ...}

INVARIANTE-S2-04: markov_analysis dentro de prediction (no en ranking_analysis top-level)
  Confusión documentada: el análisis original buscaba ranking_analysis.markov_analysis
  Correcto: ranking_analysis.prediction.markov_analysis
```

---

## 3. Contratos de Señal — S3_RANKINGS

### 3.1 Definición

**Productor:** `extraer_ranking_atp_version2.py`, `extraer_ranking_wta_version2.py`
**Consumidores:** `ranking_manager.py` (interno a S2), `elo_system.py`
**Tipo de señal:** RAW — snapshot semanal de rankings, inmutable post-captura
**SLA de frescura:** máximo 8 días (parámetro `max_age_days=8` en RankingManager)

### 3.2 Schema de Salida (JSON Schema — simplificado)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S3_RANKINGS",
  "type": "object",
  "description": "Dict indexado por nombre del jugador.",
  "additionalProperties": {
    "type": "object",
    "required": ["ranking", "nombre"],
    "properties": {
      "ranking": { "type": "integer", "minimum": 1 },
      "nombre":  { "type": "string" },
      "puntos":  { "type": ["integer", "null"] },
      "pais":    { "type": ["string", "null"] }
    }
  },
  "invariants": {
    "no_duplicates": "cada jugador aparece exactamente una vez",
    "ranking_contiguous": "rankings van de 1 a N sin huecos grandes (tolerancia: ±10)"
  }
}
```

---

## 4. Contratos de Señal — S4_PREDICTION (campo de S2)

### 4.1 Definición Correcta

S4 NO es un archivo independiente. Es el sub-objeto `prediction` dentro de cada elemento de `partidos` en S2_H2H_DATA. Se documenta como señal separada porque tiene su propio contrato de calidad y sus propios consumidores directos.

**Ruta exacta:** `s2_file.partidos[i].ranking_analysis.prediction`
**Consumidores directos:** `edge_calculator.py` (S5), `generar_tabla_favoritos*.py`, `validar_con_api.py` (S6)

### 4.2 Contrato de Lectura (para consumidores)

```python
# CONTRATO CORRECTO — verificado en código:
favored = partido['ranking_analysis']['prediction']['favored_player']
confidence = partido['ranking_analysis']['prediction']['confidence']
factor_markov = partido['ranking_analysis']['prediction']['markov_analysis']['factor_markov']
erdos_score = partido['ranking_analysis']['erdos_analysis']['erdos_score']

# ANTI-PATRÓN — siempre retorna None, rompe silenciosamente:
favored = partido.get('prediccion_ganador')  # ← NUNCA USAR
```

### 4.3 Invariantes

```
INVARIANTE-S4-01: favored_player no es None
  Si None: el match tuvo error de análisis → excluir de S5

INVARIANTE-S4-02: confidence ∈ [50.0, 95.0]
  Mínimo 50 (no hay partido donde el modelo esté seguro de menos del 50%)
  Máximo 95 (cap explícito en rivalry_analyzer.py: min(confidence, 95.0))

INVARIANTE-S4-03: factor_markov ∈ [0.85, 1.15]
  Definido en calcular_factor_markov() en markov_analyzer.py
  0.85 = jugador1 COLD vs jugador2 HOT (máxima penalización)
  1.15 = jugador1 HOT vs jugador2 COLD (máxima bonificación)
  1.00 = neutral (ambos NEUTRAL o ambos HOT/COLD)
```

---

## 5. Contratos de Señal — S5_EDGE

### 5.1 Definición

**Productor:** `edge_calculator.py`
**Consumidores:** Decisión de apuesta humana, `generar_tabla_favoritos*.py`
**Tipo de señal:** CURATED — debería ser transformación pura de S4 + cuotas de S1
**Dependencia implícita crítica:** `data/calibracion_edge.json` (p_historica) — violación de pureza

### 5.2 Schema de Salida (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S5_EDGE",
  "type": "object",
  "required": ["apostar", "watchlist", "sin_edge", "sin_datos"],
  "properties": {
    "apostar": {
      "type": "array",
      "description": "Partidos donde edge > umbral (default 5%) y Kelly-KL > 0",
      "items": { "$ref": "#/$defs/edge_entry" }
    },
    "watchlist": {
      "type": "array",
      "description": "Partidos con edge positivo pero bajo el umbral de apuesta",
      "items": { "$ref": "#/$defs/edge_entry" }
    },
    "sin_edge":  { "type": "array" },
    "sin_datos": { "type": "array" }
  },
  "$defs": {
    "edge_entry": {
      "type": "object",
      "required": ["partido", "favorito_predicho", "edge", "fraccion_bankroll",
                   "cuota_favorito", "p_modelo", "p_implicita"],
      "properties": {
        "partido":           { "type": "string" },
        "favorito_predicho": { "type": "string" },
        "cuota_favorito":    { "type": "number", "minimum": 1.0 },
        "cuota_rival":       { "type": "number", "minimum": 1.0 },
        "p_modelo":          { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "p_implicita":       { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "edge":              { "type": "number", "description": "p_modelo - p_implicita" },
        "edge_pct":          { "type": "string" },
        "kl_divergencia":    { "type": "number", "minimum": 0.0 },
        "kelly_clasico":     { "type": "number" },
        "kelly_kl":          { "type": "number" },
        "fraccion_bankroll":  {
          "type": "number",
          "minimum": 0.0,
          "maximum": 0.10,
          "description": "Cap absoluto: nunca más del 10% del bankroll por apuesta (Regla-5 Kelly)"
        },
        "zona_cuota":        {
          "type": "string",
          "enum": ["underdog", "moderate_favorite", "slight_underdog", "heavy_favorite"]
        },
        "phi_idiosincratico": { "type": "number" },
        "psi_entropia":       { "type": "number" },
        "lambda_aversion":    { "type": "number" },
        "p_historica_usada":  {
          "type": "number",
          "description": "CRÍTICO: debe ser 0.52 (default) hasta n>=30 validaciones limpias post-Nodo-03"
        },
        "apostar":            { "type": "boolean" }
      }
    }
  },
  "invariants": {
    "kelly_cap": "fraccion_bankroll <= 0.10 para toda entrada en apostar[]",
    "edge_threshold": "todas las entradas en apostar[] tienen edge > 0.05",
    "p_historica_safe": "p_historica_usada == 0.52 hasta que S6 produzca n>=30 con datos post-Nodo-03",
    "no_negative_kelly": "fraccion_bankroll >= 0 (kelly negativo = no apostar, no apostar en negativo)"
  }
}
```

### 5.3 Violación de Pureza — Dependencia Implícita en p_historica

S5 actualmente lee `data/calibracion_edge.json` como side-input. Esto viola el principio de transformación pura:

```
ESTADO ACTUAL (impuro):
  f(S4, cuotas_S1, calibracion_edge.json_en_disco) → S5
  El mismo S4 produce S5 diferente según el estado del disco.

ESTADO CORRECTO (puro):
  f(S4, cuotas_S1, p_historica=0.52) → S5
  p_historica es parámetro explícito, no side-input implícito.
  Permite reproducibilidad: dado S4 y p_historica, S5 es determinista.
```

---

## 6. Contratos de Señal — S6_RESULTADO_REAL

### 6.1 Definición

**Productor:** `validar_con_api.py`
**Consumidores:** `actualizar_calibracion_desde_resultados()` → retroalimenta S5 (p_historica)
**Tipo de señal:** ANALYTICS — depende de outcomes reales, no es reproducible
**Precondición crítica:** `match_id != 'tennis'` en S1 (requiere Nodo-03 en producción)

### 6.2 Schema de Salida

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S6_RESULTADO_REAL",
  "type": "object",
  "required": ["resultados", "accuracy_global", "accuracy_por_superficie"],
  "properties": {
    "resultados": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["partido", "prediccion", "resultado_real", "correcto", "match_id", "superficie"],
        "properties": {
          "partido":        { "type": "string" },
          "prediccion":     { "type": "string" },
          "resultado_real": { "type": "string" },
          "correcto":       { "type": "boolean" },
          "match_id":       {
            "type": "string",
            "not": { "const": "tennis" },
            "description": "Requiere S1 post-Nodo-03 para tener match_id real"
          },
          "superficie":     { "type": "string", "enum": ["clay", "grass", "hard", "indoor", "unknown"] },
          "confianza":      { "type": "number" },
          "torneo":         { "type": "string" }
        }
      }
    },
    "accuracy_global":         { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "accuracy_por_superficie": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["accuracy", "n", "correctas"],
        "properties": {
          "accuracy":  { "type": "number" },
          "n":         { "type": "integer", "minimum": 1 },
          "correctas": { "type": "integer" }
        }
      }
    },
    "n_validados": { "type": "integer" }
  },
  "invariants": {
    "min_sample_for_calibration": "n_validados >= 30 antes de actualizar p_historica en S5",
    "surface_required": "accuracy_por_superficie['clay'] solo es confiable con datos post-Nodo-03",
    "contamination_guard": "NO usar resultados con superficie='unknown' para calibración de surface_advantage"
  }
}
```

### 6.3 Corrección de Claves API (Nodo-09)

```
CONTRATO API dc_1_{event_id} — verificado 2026-05-29:
  DJ = 'H' → local ganó (jugador1)
  DJ = 'A' → visitante ganó (jugador2)
  DJ = ''  → no terminado
  DE = sets ganados por local
  DF = sets ganados por visitante
  DC = Unix timestamp del inicio programado (discrimina NS vs LIVE)
  DV = constante de tipo partido (2=tenis) — NO indica estado

ANTI-PATRÓN eliminado (pre-Nodo-09):
  ~AA → no existe en este endpoint → siempre retornaba status='NS'
```

---

## 7. Contratos de Señal — S7_MARKOV (sub-señal de S4)

### 7.1 Definición

S7 es el análisis de Cadenas de Markov embedded en S4_PREDICTION. Su contrato define cómo debe interpretarse `factor_markov` para ajustar `form_recent` en la predicción.

**Ruta exacta:** `s2_file.partidos[i].ranking_analysis.prediction.markov_analysis`
**Productor real:** `analysis/markov_analyzer.py` → llamado desde `rivalry_analyzer.py`

### 7.2 Semántica del factor_markov

```
factor_markov = calcular_factor_markov(estado_p1, estado_p2)

TABLA DE VALORES:
  estado_p1=HOT,     estado_p2=COLD    → factor = 1.15 (máxima ventaja para p1)
  estado_p1=HOT,     estado_p2=NEUTRAL → factor = 1.075
  estado_p1=NEUTRAL, estado_p2=HOT    → factor = 0.925
  estado_p1=COLD,    estado_p2=HOT    → factor = 0.85 (máxima desventaja para p1)
  Mismo estado (HOT/HOT, NEUTRAL/NEUTRAL, etc.) → factor = 1.0

APLICACIÓN en rivalry_analyzer.py:
  p1['form_recent'] *= factor_p1   ← donde factor_p1 = factor_markov (perspectiva p1 vs p2)
  p2['form_recent'] *= factor_p2   ← donde factor_p2 = factor_markov inverso

PESO en predicción: form_recent tiene weight=0.15 (15% del score total)
EFECTO NET: factor 1.15 × weight 0.15 = +1.725% en el score final de p1
```

### 7.3 Invariantes

```
INVARIANTE-S7-01: factor_markov ∈ [0.85, 1.15] — hardcoded en calcular_factor_markov()
INVARIANTE-S7-02: estado_actual ∈ {'HOT', 'COLD', 'NEUTRAL'}
INVARIANTE-S7-03: win_rate_reciente ∈ [0.0, 1.0]
INVARIANTE-S7-04: confianza ∈ [0.0, 1.0] — confianza del change-point detection
INVARIANTE-S7-05: markov_analysis puede ser null si historial < 5 partidos
```

---

## 8. Contratos de Señal — S8_DATASET_ML

### 8.1 Definición

**Productor:** `generar_dataset_plus.py`
**Consumidores:** `aplicar_enhancer.py` → `Intelligent_ml_enhancer.py` → modelo entrenado → calibra `p_historica` en S5
**Tipo de señal:** ANALYTICS — función de S2 + S6 histórico
**Estado actual:** bugs corregidos (Nodo-04) pero datos disponibles siguen contaminados

### 8.2 Schema de Salida (CSV con columnas críticas)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12",
  "$id": "S8_DATASET_ML",
  "description": "Dataset tabular (CSV). Cada fila es un partido histórico.",
  "critical_columns": {
    "surface_specialization": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "warning": "CERO para todos los registros pre-2026-05-28 — datos contaminados por bug HTML en S1"
    },
    "markov_factor":   { "type": "number", "minimum": 0.85, "maximum": 1.15 },
    "erdos_score":     { "type": ["number", "null"], "minimum": -1.0, "maximum": 1.0 },
    "resultado_real":  { "type": "integer", "enum": [0, 1], "description": "1=local ganó, 0=visitante ganó" },
    "edge_calculado":  { "type": ["number", "null"] },
    "h2h_directo":     { "type": "number" },
    "elo_diff":        { "type": "number" }
  },
  "invariants": {
    "no_contaminated_surface": "surface_specialization > 0.0 en al menos 1 partido del jugador",
    "resultado_required": "resultado_real IS NOT NULL (requiere S6 para labels)",
    "temporal_guard": "NO incluir partidos anteriores a 2026-05-28 en training set (surface=0% contamina features)"
  }
}
```

---

## 9. Traza Completa de Contaminación — Nodo-03 Bug

### 9.1 El Árbol de Contaminación

El bug de `superficie=None` en S1 no es un fallo aislado. Es la raíz de un árbol de contaminación que se ramifica en cascada hacia S2, S4, S5 y S8, con un **loop de retroalimentación** en S5 que amplifica el daño.

```
S1_MATCH_LIST [RAÍZ DEL BUG]
├── superficie: null
│   (todos los 423 partidos de zita_tennis_matches_20260528_130141.json)
│
└─→ S2_H2H_DATA [CONTAMINACIÓN NIVEL 1]
    ├── tipo_cancha: "Desconocida" (derivada de torneo_completo que no tiene superficie)
    ├── rivalry_analyzer.py llama normalize_surface("HTML_GARBAGE") → "Desconocida"
    ├── surface_analysis: { "advantage": 0.0, "specialization": 0.0 }
    │
    └─→ S4_PREDICTION [CONTAMINACIÓN NIVEL 2]
        ├── surface_specialization: 0.0 para TODOS los partidos
        ├── peso surface_specialization = 0.15 × 0.0 = 0 contribución
        ├── predicciones ignoran sistemáticamente la superficie
        │
        ├── Ejemplo concreto: Djokovic vs Alcaraz en Roland Garros (arcilla)
        │   - Con surface=0%: modelo usa solo ranking, ELO, H2H, form
        │   - Con surface correcto: Alcaraz tiene surface_specialization alta en clay
        │   - Efecto: la ventaja de arcilla de Alcaraz desaparece del modelo
        │
        └─→ S5_EDGE [CONTAMINACIÓN NIVEL 3 — P&L DIRECTO]
            ├── p_modelo basado en S4 surface-blind
            ├── edge = p_modelo - p_implicita es BIASED para especialistas de superficie
            ├── Para arcilleros (clay): modelo SUBESTIMA su probabilidad → edge MENOR del real
            ├── Para no-arcilleros: modelo SOBREESTIMA → edge MAYOR del real → APUESTA INCORRECTA
            │
            └─→ S8_DATASET_ML [CONTAMINACIÓN NIVEL 2, RAMA PARALELA]
                ├── Feature surface_specialization = 0.0 en TODAS las filas
                ├── Modelo ML aprende: superficie no importa
                ├── Accuracy = 47.37% (9/19, Jan 2026) — peor que random
                │
                └─→ p_historica = 0.4737 [LOOP DE RETROALIMENTACIÓN]
                    └─→ S5_EDGE [CONTAMINACIÓN NIVEL 4 — AMPLIFICACIÓN]
                        ├── Kelly-KL usa p_historica=0.4737 (no 0.52 default)
                        ├── p_historica < 0.5 → KL divergencia MAYOR → lambda penalización MAYOR
                        ├── PERO también: kelly_clasico = (p_modelo - p_implicita) / (1 - p_implicita)
                        │   Con p_historica contaminada: el factor de ajuste KL es diferente
                        └── Efecto neto: fracción bankroll miscalibrada en ambas direcciones
```

### 9.2 El Loop de Retroalimentación — Insight Crítico

La contaminación crea un ciclo que la mayoría de specs no capturan:

```
S1 (superficie=null)
  → S2 (surface_specialization=0%)
    → S8 (modelo entrenado con surface=0%)
      → accuracy=47.37%
        → p_historica calibrada a 0.47
          → S5 Kelly-KL usa p_historica incorrecta
            → fracciones de bankroll miscalibradas
              → P&L negativo incluso con edge real positivo
```

**Esto explica por qué el fix del Nodo-03 en código no es suficiente:**
Aunque `data_parser.py` ahora maneja correctamente la superficie, el modelo ML (S8) fue entrenado con datos contaminados. Si se usa `p_historica` derivada de ese modelo en S5, el Kelly-KL sigue usando una premisa falsa.

**Solución correcta (en orden):**
1. Ejecutar pipeline completo post-Nodo-03 en producción → datos S1 limpios ✅ (pendiente T03-06)
2. Acumular n≥30 partidos con datos limpios en S6 → accuracy real
3. Derivar p_historica SOLO de datos post-2026-05-28
4. Hasta ese momento: p_historica = 0.52 (default conservador) — **REGLA-3 del sistema**

### 9.3 Detección Temprana — Contrato de Validación de S1

Para prevenir que este bug se repita, S1 debe fallar rápido:

```python
# Validación propuesta para S1 (no implementada aún — deuda D-17 extendida)

def validar_contrato_s1(data: dict) -> tuple[bool, list[str]]:
    errores = []
    for torneo, matches in data.items():
        for m in matches:
            if m.get('h2h_url') is None:
                errores.append(f"CONTRATO_S1_VIOLADO: h2h_url=None en {m.get('jugador1')} vs {m.get('jugador2')}")
            if m.get('match_id') in ('tennis', '', None):
                errores.append(f"CONTRATO_S1_VIOLADO: match_id inválido: {m.get('match_id')}")
            superficie = m.get('superficie', '')
            if superficie not in ('clay', 'grass', 'hard', 'indoor', 'unknown'):
                errores.append(f"CONTRATO_S1_VIOLADO: superficie={superficie!r}")
    return len(errores) == 0, errores
```

---

## 10. Garantías de Idempotencia por Señal

### 10.1 Definición Formal

Una señal es **idempotente** si `f(f(x)) = f(x)`, es decir, aplicar la transformación múltiples veces produce el mismo resultado que aplicarla una vez.

En el contexto de este pipeline: una señal es **reproducible** si `f(input_en_T) = mismo_output` independientemente de cuándo se ejecute f, dado el mismo input.

### 10.2 Tabla de Idempotencia Actual

| Señal | ¿Es idempotente? | Violación | Fix propuesto |
|---|---|---|---|
| S1_MATCH_LIST | ❌ No | FlashScore cambia cada hora. Mismo script → datos distintos. | Inmutabilidad por timestamp: una vez capturado, no se sobreescribe |
| S2_H2H_DATA | ❌ No | Scraping en tiempo real. Markov usa historial que cambia. **LOOKAHEAD BIAS.** | Separar scraping (impuro) de análisis (puro). Fix: ejecutar antes de T=inicio_partidos |
| S3_RANKINGS | ⚠️ Parcial | Rankings cambian semanalmente, no diariamente. Suficientemente estable. | max_age_days=8 es el contrato de frescura actual — aceptable |
| S4_PREDICTION | ✅ Sí | Función pura de S2 + S3 dado el mismo modelo. | Mantener como está |
| S5_EDGE | ⚠️ Parcial | Viola pureza por side-input `calibracion_edge.json`. | Hacer p_historica parámetro explícito |
| S6_RESULTADO_REAL | ❌ No (by design) | Outcomes reales no son reproducibles. | N/A — es la fuente de ground truth |
| S7_MARKOV | ✅ Sí | Función pura de historial. Mismo historial → mismo factor. | Mantener como está |
| S8_DATASET_ML | ⚠️ Parcial | Depende de S6 que crece con el tiempo. | Versionar el dataset con timestamp de S6 |

### 10.3 El Problema de Lookahead en S2 — Análisis Temporal

Este es el bug más sutil del sistema y el que tiene mayor impacto en la validez de las señales:

```
CONTRATO TEMPORAL CORRECTO DE S2:
  "Los datos de historial de cada jugador reflejan su estado en T=antes_del_partido"

VIOLACIÓN ACTUAL:
  El pipeline corre a T=12:00.
  Los partidos Teichmann, Carreno-Busta, Wang empezaron a T=07:20.
  FlashScore ya actualizó el historial de estos jugadores con el resultado de hoy.
  Cuando el scraper extrae el historial de Teichmann a T=12:00:
    - current_streak incluye la victoria de hoy → estado="HOT"
    - win_rate_reciente incluye el partido de hoy → inflado
  La "predicción" de Teichmann es retroactiva, no prospectiva.

CONSECUENCIA:
  Las 3 señales "acertadas" (Teichmann HOT, Carreno-Busta HOT, Wang HOT)
  no validan el modelo — validan el scraper de historial.
  El modelo "predijo" resultados ya ocurridos.

SOLUCIÓN:
  REGLA DE TEMPORALIDAD: el pipeline debe ejecutarse en T < min(hora_inicio_todos_los_partidos)
  Para Roland Garros (primer partido 10:00 CEST): ejecutar antes de las 08:00 CEST.
  Implementación: cron job a las 05:00 AM hora del servidor.
```

---

## 11. Contrato de Migración — Estado Post-Nodo-07 vs Meta

### 11.1 Estado Actual vs Contrato Objetivo

| Aspecto | Estado actual (2026-05-29) | Contrato objetivo (post-Nodo-07 Fase 2) |
|---|---|---|
| S2 separation | S2 mezcla scraping + análisis | S2_RAW (scraping puro) + S2_ANALYZED (transformación pura) |
| S4 location | campo dentro de S2 | archivo independiente derivado de S2_RAW |
| S7 location | sub-campo de S4, dentro de S2 | llamada explícita a markov_analyzer.run(S2_RAW) |
| Erdős location | `ranking_analysis.erdos_analysis` (post-fix) | mismo — ya correcto |
| p_historica | side-input implícito en S5 | parámetro explícito del CLI de edge_calculator |
| Contrato S1 | ninguna validación formal | validar_contrato_s1() llamado antes de S2 |

### 11.2 Condiciones para Desbloquear Cada Señal

```
PARA QUE S4/S7 SEAN SEÑALES PURAS (Nodo-07 Fase 2):
  REQUIERE: ≥40 tests en test_h2h_extractor.py (hoy: 5)
  REQUIERE: paridad de output verificada en ≥10 partidos reales
  REQUIERE: Nodo-09-H2HExtractor-Paridad.md creado

PARA QUE S5 SEA PURA (fix de p_historica):
  REQUIERE: CLI de edge_calculator acepta --p-historica como parámetro
  REQUIERE: script de producción pasa explícitamente el valor calibrado o el default

PARA QUE S6 FUNCIONE EN PRODUCCIÓN:
  REQUIERE: S1 produce match_id reales en producción (T03-06)
  REQUIERE: pipeline corre ANTES de los partidos (soluciona lookahead y da match_ids válidos)

PARA QUE S8 TENGA DATOS LIMPIOS:
  REQUIERE: ≥30 partidos en S6 con superficie correcta (post-Nodo-03 en prod)
  REQUIERE: fecha_captura en cada fila de S8 para poder filtrar pre-2026-05-28
```

---

## 12. Vinculación

- [[MOC-Principal]] — dashboard del sistema; este doc define los contratos de las señales listadas allí
- [[Grafo-Dependencias-Datos]] — diagrama visual; este doc es el contrato formal de ese diagrama
- [[Nodo-03-Scraper-Fix]] — origen de la contaminación S1→S2→S4→S5→S8
- [[Nodo-07-Strangler-Fig]] — migración que separará S2_RAW de S2_ANALYZED
- [[Nodo-09-API-Status-Keys]] — contrato corregido de la API dc_1 para S6
- [[Inventario-Deuda-Tecnica]] — las costuras y la deuda que bloquean los contratos objetivo
- [[Mandatos-No-Negociables]] — Mandato 1 (P&L); Mandato 6 (tests antes que código)
- [[Fuentes-Datos]] — contrato externo con FlashScore; este doc es el contrato interno entre señales

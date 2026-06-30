# Atlas: Grafo de Dependencias de Datos

> **Equivalente a:** Signal-Dependency-Graph del proyecto Electric_mix
> **Wikilinks:** [[Pipeline-Arquitectura]] | [[Mandatos-No-Negociables]] | [[Fuentes-Datos]] | [[Sprint-Pipeline]] | [[Nodo-01-Edge-Calculator]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-03-Scraper-Fix]] | [[Nodo-04-Dataset-Fix]] | [[Nodo-05-Validacion-API]] | [[Nodo-06-Erdos-Graph]] | [[Nodo-07-Strangler-Fig]] | [[Nodo-08-File-Selection-Bug]] | [[Nodo-09-API-Status-Keys]]
> **Estado:** 2026-05-29 | Roland Garros en curso | 235 partidos (h2h_url 235/235) | 767 tests

---

## Señales del Sistema (8 Señales Canónicas)

```
Señal               Productores                 Consumidores
────────────────────────────────────────────────────────────────────────
S1_MATCH_LIST       extraer_URL_partidos_v2     extraer_historh2h.py
                    [código fix Nodo-03 ✅]      generar_tabla_favoritos2.py
                    [pendiente validación prod]
                    → zita_tennis_matches.json

S2_H2H_DATA         extraer_historh2h.py        rivalry_analyzer.py
                    (Playwright, 30-60min)       ErdosGraph [Nodo-06] ✅
                    → h2h_results_enhanced.json  markov_analyzer [Nodo-02] ✅
                    [surface=unknown: lookahead bias, no bug activo]

S3_RANKINGS         extraer_ranking_atp_v2.py   rivalry_analyzer.py
                    → atp_rankings_complete.json elo_system.py

S4_PREDICTION       rivalry_analyzer.py          edge_calculator.py [Nodo-01] ✅
                    → ranking_analysis.prediction generar_tabla_favoritos2.py
                    [Markov+Erdős activos ✅]     validar_con_api.py [Nodo-05] ✅
                    [top-level=None: limitación conocida, no bug]

S5_EDGE             edge_calculator.py [Nodo-01] DECISIÓN DE APUESTA
                    → edge, kelly_kl, apostar     generar_tabla_favoritos2.py
                    [✅ CONSTRUIDO — 43 tests]    p_historica=0.52 provisional

S6_RESULTADO_REAL   validar_con_api.py [Nodo-05] calibration_data.json
                    dc_1_{event_id} API ✅        accuracy_por_superficie
                    → resultados_finales.json    [Nodo-01] p_historica
                    [✅ CONSTRUIDO — 39 tests]   [Nodo-09: claves DJ/DE/DF ✅]
                    [pendiente: match_id real en prod (T03-06)]

S7_MARKOV           markov_analyzer.py [Nodo-02] rivalry_analyzer.py
                    → estado HOT/COLD, momentum  form_recent score (×factor)
                    [✅ ACTIVO en producción]    factor_markov en output JSON

S8_DATASET_ML       generar_dataset_plus.py      aplicar_enhancer.py
                    [bugs Nodo-04 CORREGIDOS ✅] Intelligent_ml_enhancer.py
                    → ml_datasets/dataset.csv    → modelo entrenado
                    [pendiente: datos limpios de S1 en prod]
```

---

## Orden de Inicialización del Pipeline

```
Orden   Señal    Script                          Tiempo      Estado
─────────────────────────────────────────────────────────────────────
-1      ENV      variables de entorno            0s          ✅
0       S3       extraer_ranking_atp_v2.py       2-5min      ✅
1       S1       extraer_URL_partidos_v2.py      8min        ⚠️ código OK, prod pendiente
2       S7       markov_analyzer.py              <1s         ✅ ACTIVO (37 tests)
3       S2       extraer_historh2h.py            30-60min    ✅ (depende de S1,S3)
4       S4       rivalry_analyzer.py             <30s        ✅ Markov+Erdős activos
5       S5       edge_calculator.py              <1s         ✅ CONSTRUIDO (43 tests)
6       S8       generar_dataset_plus.py         5-10min     ⚠️ pendiente datos limpios S1
7       S6       validar_con_api.py              <1s         ✅ CONSTRUIDO (39 tests), prod pendiente
```

---

## Grafo Visual de Dependencias

```
                    ┌─────────────────────┐
                    │  FlashScore.com      │
                    │  (Playwright)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  S1_MATCH_LIST       │ ← [[Nodo-03-Scraper-Fix]] ✅ código
                    │  zita_tennis_        │   h2h_url, match_id, torneo
                    │  matches.json        │   superficie — fix aplicado
                    └──┬───────┬──────────┘
                       │       │
        ┌──────────────▼─┐   ┌─▼──────────────────┐
        │  S3_RANKINGS    │   │  S2_H2H_DATA        │
        │  atp_rankings   │   │  h2h_results_       │
        │  _complete.json │   │  enhanced.json      │
        └──────┬──────────┘   └────────┬────────────┘
               │                       │
               └──────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  S4_PREDICTION           │ ← [[Pipeline-Arquitectura]]
              │  rivalry_analyzer.py    │   Markov+Erdős activos ✅
              │  → prediction.favored   │   (top-level=None: limitación conocida)
              └────┬───────────┬────────┘
                   │           │
     ┌─────────────▼──┐   ┌────▼──────────────┐
     │  S5_EDGE        │   │  S7_MARKOV         │
     │  edge_calc.py   │   │  markov_analyzer   │ ← [[Nodo-02-Markov-Changepoint]]
     │  kelly_kl       │   │  HOT/COLD/factor   │
     │  [✅ 43 tests]  │   │  [✅ ACTIVO]       │
     └────────┬────────┘   └────────────────────┘
              │
   ┌──────────▼──────────┐
   │  DECISIÓN DE APUESTA │
   │  Edge > 5%           │
   │  Kelly-KL > 2%       │
   └──────────┬──────────┘
              │
   ┌──────────▼──────────┐
   │  S6_RESULTADO_REAL   │ ← [[Nodo-05-Validacion-API]] [[Nodo-09-API-Status-Keys]]
   │  validar_con_api.py  │   dc_1_{event_id} ✅ HTTP 200
   │  accuracy_real       │   Claves: DJ/DE/DF (Nodo-09 ✅)
   └──────────┬──────────┘
              │ retroalimenta
   ┌──────────▼──────────┐
   │  S5_EDGE (calibrar)  │
   │  p_historica actual  │
   │  lambda_aversion     │
   └─────────────────────┘
```

---

## Señales con Estado Pendiente

### S1_MATCH_LIST — código corregido (ver [[Nodo-03-Scraper-Fix]]), validación prod pendiente

| Campo | Estado código | Pendiente prod |
|---|---|---|
| `h2h_url` | ✅ Fix Nodo-03 aplicado | Verificar en próximo run (T03-06) |
| `match_id` | ✅ Fix Nodo-03 aplicado | Verificar event_id real (≠ "tennis") |
| `torneo` | ✅ Fix Nodo-03 aplicado | Verificar "Roland Garros (France)" en output |
| `superficie` | ✅ Fix Nodo-03 aplicado | Verificar clay/grass/hard en output |

### S4_PREDICTION — limitación conocida (no es bug activo)

```python
# INCORRECTO — siempre None (limitación de diseño):
partido['prediccion_ganador']

# CORRECTO — paths donde vive la predicción:
partido['ranking_analysis']['prediction']['favored_player']
partido['ranking_analysis']['prediction']['markov_analysis']['factor_markov']
partido['ranking_analysis']['erdos_analysis']['erdos_score']
```

### S6_RESULTADO_REAL — claves API corregidas (ver [[Nodo-09-API-Status-Keys]])

Las claves reales del endpoint `dc_1_{event_id}` (verificadas 2026-05-29):
- `DJ='H'` → jugador1 ganó | `DJ='A'` → jugador2 ganó | `DJ=''` → en curso/NS
- `DE` = sets ganados por local | `DF` = sets ganados por visitante
- `DC` = Unix timestamp inicio programado (distingue NS vs LIVE cuando DJ='')
- `DV=2` = constante tipo deporte (tenis) — NO indicador de estado

---

## API Disponibles (Fuentes Externas)

```
FlashScore Ninja API
  dc_1_{event_id}          → score, estado partido      ✅ HTTP 200 tenis
  df_psn_1_{eventId}       → box score (NBA, no tenis)  ✅ NBA
  t_3_200_{tournId}        → partidos del torneo         ✅ confirmado NBA
  dc_h2h_1_{id}            → H2H entre dos jugadores    ❌ 404 tenis
  
  Auth: X-Fsign: SW9D1eZo | Referer: https://www.flashscore.co/
  Base: https://global.flashscore.ninja/202/x/feed/

FlashScore.com (Playwright)
  /match/{id}/#/h2h/overall/  → H2H completo            ✅ funciona
  Tiempo: 30-60 min / 28 partidos programados
```

---

## Reglas de Integridad de Datos

```
REGLA-1: Nunca leer prediccion_ganador top-level (siempre None)
         Siempre: ranking_analysis.prediction.favored_player

REGLA-2: No entrenar ML con datos anteriores a Nodo-03-fix
         surface_specialization=0% contamina features de superficie

REGLA-3: p_historica = 0.52 (default) hasta n≥30 validaciones limpias
         No usar 47.37% (datos sucios pre-2026-05-28)

REGLA-4: match_id ≠ "tennis" es prerequisito para API de validación
         Verificar antes de llamar dc_1_{event_id}

REGLA-5: Kelly-KL cap = 10% del bankroll por apuesta
         Kelly clásico puede dar fracciones irreales con modelos sobreconfiados

REGLA-6: Claves reales dc_1 API: DJ/DE/DF — NO ~AA/~BH/~BI (ver [[Nodo-09-API-Status-Keys]])
         DJ='H'→jugador1 ganó | DJ='A'→jugador2 ganó | DJ=''→en curso/NS
         DC timestamp distingue NS (futuro) de LIVE (pasado) cuando DJ=''
```

---

## Estado de Completitud por Señal

| Señal | Completitud | Bloqueada por |
|---|---|---|
| S1_MATCH_LIST | 60% (código ✅, prod pendiente) | T03-06: run en producción |
| S2_H2H_DATA | 85% (surface=unknown por lookahead) | Nodo-03 en prod + datos limpios |
| S3_RANKINGS | 100% ✅ | — |
| S4_PREDICTION | 90% (Markov+Erdős activos ✅) | top-level=None es limitación, no bug |
| S5_EDGE | 60% ✅ | p_historica=0.52 provisional hasta n≥30 validaciones |
| S6_RESULTADO_REAL | 80% ✅ | match_id real en prod (T03-06) |
| S7_MARKOV | ACTIVO ✅ | — (integrado en rivalry_analyzer, factor_markov en output) |
| S8_DATASET_ML | 40% (bugs corregidos ✅) | datos limpios de S1 en prod |

**Completitud total del pipeline: ~78% (6.2 de 8 señales operativas en código)**
**Pendiente principal: validación en producción — T03-06 desbloquea S1→S2→S4→S5→S6**

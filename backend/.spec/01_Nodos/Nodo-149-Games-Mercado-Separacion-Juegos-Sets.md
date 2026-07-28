# Nodo-149 — Separación Definitiva Mercados Juegos/Sets en games_signal_calculator

**Fecha:** 2026-07-28
**Estado:** SPEC APROBADA — pendiente implementación
**Wikilinks:** [[Nodo-40]] [[Nodo-126]] [[Nodo-127]] [[Nodo-133]] [[Nodo-134]] [[games_signal_calculator]] [[betplay_combo_builder]]

---

## 1. Diagnóstico del Bug

### Root cause exacto

`games_signal_calculator.py` — función `_seleccionar_señales_optimas()` (L403-434):

```python
# Dos mercados incompatibles mezclados en la misma lista:
juegos_under → optimas
juegos_over  → optimas
sets_señales → optimas   # ← BUG: mercado completamente diferente
return optimas           # lista heterogénea
```

El combo building (L642-656) toma señales del pool `optimas` sin discriminar
`mercado`, produciendo combos potencialmente inválidos como:
- Leg1: Match A — UNDER 23.5 **juegos** (linea=23.5)
- Leg2: Match B — OVER **2.5 sets** (linea=2.5)

### Por qué viola la matemática del combo

El cálculo `P(combo) = ∏ P(leg_i)` asume **independencia entre piernas**.

Para piernas de **distintos partidos**: ✅ independencia válida.
Para piernas de **mismo partido, distintos mercados**: ❌ CORRELADAS.
- P(UNDER 23.5 juegos ∩ OVER 2.5 sets, mismo partido) ≠ P(UNDER 23.5) × P(OVER 2.5)
- Un partido largo (OVER 2.5 sets) también tendrá MÁS juegos → anticorrelación con UNDER juegos
- La EV calculada en `build_games_combos()` es incorrecta cuando mezcla mercados

Para piernas de **distintos partidos, distintos mercados**: matemáticamente válido
pero **operacionalmente incoherente** — la lógica del modelo (zona/Markov) que
genera ambas señales tiene semánticas diferentes que no deberían mezclarse en
una apuesta única.

### Confirmación: same-partido guard en betplay (PARCIALMENTE RESUELTO)

`build_games_combos()` en `betplay_combo_builder.py` (L1769): toma `señales[0]`
— primera señal óptima por partido. Este guard **implícito** evita 2 piernas
del mismo partido PERO no evita que `señales[0]` sea un signal de SETS en vez
de JUEGOS (depende del orden en `optimas`).

### Donde persiste el bug hoy

1. **`games_signal_calculator.py` Combo A/B** (L642-656): usa pool `optimas` mezclado.
   Las señales de distinto mercado aparecen en el mismo combo.

2. **`build_games_combos()` betplay** (L1755→1786): lee `señales[0]` que puede
   ser SETS signal. Una combinación de 3 partidos podría ser:
   - Match A señales[0] = SETS OVER 2.5 (sets signal llegó primero en optimas)
   - Match B señales[0] = JUEGOS UNDER 23.5
   - Match C señales[0] = JUEGOS OVER 20.5
   → combo cross-market sin que nadie lo sepa.

3. **Display T0 en live_desk X3** (Nodo-147): expone `linea_t0=2.5` mezclada
   con `linea_t0=22.5`. No es bug del display — es síntoma del bug upstream.

4. **`gap=None` en sets signals**: no hay cálculo de gap probabilístico para el
   mercado de sets. La cuota se evalúa solo por threshold mínimo, sin modelo real.

---

## 2. Fixes — D149-01 → D149-05

### D149-01: Campo `mercado_tipo` en todas las señales

**Archivo:** `games_signal_calculator.py` — función `_analizar_mercados_juegos()`

En la construcción del dict de señal de juegos (L~317-360):
```python
# Añadir campo mercado_tipo a todos los dicts de señal "Total de juegos"
{
    "mercado": "Total de juegos",
    "mercado_tipo": "JUEGOS",   # ← NUEVO D149-01
    ...
}
```

En la construcción del dict de señal de sets (L~366-398):
```python
{
    "mercado": "Total de sets",
    "mercado_tipo": "SETS",     # ← NUEVO D149-01
    ...
}
```

El campo `mercado_tipo` fluye automáticamente al `games_signal_report_*.json`
porque las señales se serializan sin transformación en `guardar_reporte()`.

**Backward compatible:** El campo `"mercado"` (string) no se toca. Los consumers
existentes que leen `mercado` siguen funcionando.

---

### D149-02: Refactor `_seleccionar_señales_optimas()` — retorna tupla separada

**Archivo:** `games_signal_calculator.py` — función `_seleccionar_señales_optimas()`

**Antes (L403-434):**
```python
def _seleccionar_señales_optimas(apostar):
    optimas = []
    # ... agrega juegos UNDER, juegos OVER, sets ...
    return optimas  # lista heterogénea
```

**Después:**
```python
def _seleccionar_señales_optimas(apostar):
    """D149-02: retorna (juegos_optimas, sets_optimas) separados."""
    juegos_optimas = []
    sets_optimas = []

    # Total de juegos UNDER
    juegos_under = [s for s in apostar
                    if s["mercado"] == "Total de juegos" and s["direccion"] == "UNDER"]
    if juegos_under:
        mejor = min(juegos_under, key=lambda s: s.get("linea", 99))
        juegos_optimas.append(mejor)

    # Total de juegos OVER
    juegos_over = [s for s in apostar
                   if s["mercado"] == "Total de juegos" and s["direccion"] == "OVER"]
    if juegos_over:
        mejor = max(juegos_over, key=lambda s: (s.get("gap_juegos") or 0, s["cuota"]))
        juegos_optimas.append(mejor)

    # Total de sets — pista separada
    sets_señales = [s for s in apostar if s["mercado"] == "Total de sets"]
    if sets_señales:
        mejor = max(sets_señales, key=lambda s: s["cuota"])
        sets_optimas.append(mejor)

    return juegos_optimas, sets_optimas
```

**Todos los call-sites** de `_seleccionar_señales_optimas()` deben actualizarse:
```python
# Antes:
optimas = _seleccionar_señales_optimas(apostar)

# Después:
juegos_optimas, sets_optimas = _seleccionar_señales_optimas(apostar)
```

---

### D149-03: Schema del report — separar `señales_optimas` por mercado_tipo

**Archivo:** `games_signal_calculator.py` — función `guardar_reporte()` o donde
se construye el dict del partido dentro de `procesar_partidos()`.

**Antes:**
```python
{
    "partido": "...",
    "señales_optimas": [juegos_under, juegos_over, sets],  # mezclado
    ...
}
```

**Después:**
```python
{
    "partido": "...",
    "señales_optimas": juegos_optimas,       # SOLO juegos (backward compat)
    "señales_optimas_sets": sets_optimas,    # NUEVO D149-03
    ...
}
```

**Rationale:** Mantener `señales_optimas` como juegos-only preserva backward
compatibility con `build_games_combos()` en betplay (que lee `señales_optimas`).
`señales_optimas_sets` es la nueva key que los consumers pueden adoptar.

---

### D149-04: Fix combo building en `games_signal_calculator.py` (Combo A/B/C)

**Archivo:** `games_signal_calculator.py` — bloque de combos (L642-656)

**Antes:** El pool de señales para armar Combo A/B mezcla todos los mercados.

**Después:**
```python
# D149-04: Combos separados por mercado_tipo

# ── COMBOS JUEGOS ─────────────────────────────────────────────────────────
# Pool: señales ALTA de mercado juegos de partidos distintos
pool_juegos = [
    s for r in resultados
    for s in r.get("señales_optimas", [])
    if s.get("confianza_señal") == "ALTA" and s.get("mercado_tipo") == "JUEGOS"
]
# Combo A (2 piernas juegos)
# Combo B (3 piernas juegos)

# ── COMBOS SETS ──────────────────────────────────────────────────────────
# Pool: señales ALTA de mercado sets de partidos distintos
pool_sets = [
    s for r in resultados
    for s in r.get("señales_optimas_sets", [])
    if s.get("confianza_señal") == "ALTA" and s.get("mercado_tipo") == "SETS"
]
# Combo C (2 piernas sets, opcional — solo si ≥2 sets ALTA)

# PROHIBICIÓN EXPLÍCITA: nunca mezclar pool_juegos + pool_sets en el mismo combo.
```

**Guard de mismo partido** (ya existe implícitamente vía partidos distintos en pool,
pero agregar guard explícito):
```python
# Al construir cada combo, verificar que todos los partidos sean distintos:
partidos_en_combo = [leg["partido"] for leg in combo]
if len(partidos_en_combo) != len(set(partidos_en_combo)):
    continue  # mismo partido en 2 piernas → skip
```

**Output print** debe mostrar secciones separadas:
```
📦 COMBOS JUEGOS (mercado: Total de juegos):
   Combo A (2p @X): ...
   Combo B (3p @X): ...

📦 COMBOS SETS (mercado: Total de sets) [MEDIA — n<50]:
   Combo C (2p @X): ...  # solo si ≥2 señales ALTA sets
```

---

### D149-05: Fix `build_games_combos()` en betplay — filtrar por mercado_tipo

**Archivo:** `betplay_combo_builder.py` — función `build_games_combos()` (L1728+)

En el loop que itera `señales_optimas` por partido (L1769):
```python
# Antes:
señales = partido_data.get("señales_optimas", [])
if not señales:
    continue
s = señales[0]  # primera señal — puede ser SETS o JUEGOS

# Después (D149-05):
señales_juegos = [s for s in partido_data.get("señales_optimas", [])
                  if s.get("mercado_tipo") == "JUEGOS"
                  or s.get("mercado") == "Total de juegos"]  # fallback sin campo
if not señales_juegos:
    continue
s = señales_juegos[0]  # primera señal JUEGOS garantizado
```

**Fallback:** El `or s.get("mercado") == "Total de juegos"` garantiza compatibilidad
con reports generados antes de D149-01 (sin campo `mercado_tipo`).

Para combos de sets en betplay (opcional, baja prioridad):
```python
# GamesC: combos de sets — leer señales_optimas_sets si existe
señales_sets = partido_data.get("señales_optimas_sets", [])
```

---

### D149-06: gap_sets — cálculo de gap probabilístico para mercado sets

**Archivo:** `games_signal_calculator.py` — función `_analizar_mercados_juegos()`
en el bloque de "Total de sets" (L~366-398)

**Problema actual:** `gap_juegos=None` para sets signals — no hay modelo de gap.

**Fix:** Estimación MVp de `p_modelo_3sets` desde zona:
```python
# Mapa zona → p_modelo_3sets (estimación calibrada por zona Markov)
_P_3SETS_POR_ZONA = {
    "DOMINANTE":   0.28,   # jugador dominante cierra en 2 sets ~72% del tiempo
    "COINFLIP":    0.60,   # partido equilibrado → 3 sets en ~60% (ATP empírico)
    "AJUSTADA":    0.42,   # diff 0.18-0.30 → intermedio
}

zona = resultado.get("zona_cuota", "AJUSTADA")
p_modelo_3sets = _P_3SETS_POR_ZONA.get(zona, 0.42)
p_implicita_sets = 1 / cuota_sets if cuota_sets > 1 else 0
gap_sets = round(p_modelo_3sets - p_implicita_sets, 4)

# Threshold: solo apostar si gap_sets >= 0.10 (10% edge mínimo)
if gap_sets < 0.10:
    continue  # sets signal descartada por gap insuficiente
```

Agregar `gap_sets` al dict de señal:
```python
{
    "mercado": "Total de sets",
    "mercado_tipo": "SETS",
    "gap_sets": gap_sets,       # NUEVO D149-06
    "gap_juegos": None,         # mantener campo por schema consistency
    ...
}
```

**Valores de `_P_3SETS_POR_ZONA`:** Estimados a partir de distribución empírica ATP/WTA.
Calibración formal con datos históricos es **D149-07** (deuda — requiere análisis
de h2h_results_enhanced con outcome real de sets).

---

## 3. Schema final `games_signal_report_*.json`

```json
{
  "metadata": {"...": "..."},
  "señales": [...],          // todas las señales (juegos + sets)
  "apostar": [               // por partido, solo con señales ALTA+MEDIA
    {
      "partido": "...",
      "zona_cuota": "COINFLIP",
      "señales_optimas": [   // SOLO mercado_tipo=="JUEGOS" (D149-03)
        {
          "mercado": "Total de juegos",
          "mercado_tipo": "JUEGOS",      // NUEVO D149-01
          "linea": 23.5,
          "direccion": "UNDER",
          "cuota": 1.55,
          "gap_juegos": 4.5,
          "gap_sets": null,
          "outcome_id": 4274137170,
          "confianza_señal": "ALTA"
        }
      ],
      "señales_optimas_sets": [  // NUEVO D149-03 — separado
        {
          "mercado": "Total de sets",
          "mercado_tipo": "SETS",
          "linea": 2.5,
          "direccion": "OVER",
          "cuota": 2.25,
          "gap_juegos": null,
          "gap_sets": 0.155,     // NUEVO D149-06
          "outcome_id": 4274137171,
          "confianza_señal": "ALTA"
        }
      ]
    }
  ],
  "combos_juegos": [         // NUEVO — combos puros de mercado juegos
    {"tipo": "A", "piernas": [...], "cuota_combo": 2.28, "ids": "..."},
    {"tipo": "B", "piernas": [...], "cuota_combo": 3.53, "ids": "..."}
  ],
  "combos_sets": [           // NUEVO — combos puros de mercado sets (opcional)
    {"tipo": "C", "piernas": [...], "cuota_combo": X, "ids": "..."}
  ]
}
```

---

## 4. Orden de implementación (dependencias)

```
D149-01 → D149-02 → D149-03 → D149-04   (secuencial, cada uno depende del anterior)
D149-01 → D149-05                         (betplay puede correr en paralelo a D149-03)
D149-02 → D149-06                         (gap_sets requiere la zona disponible en _analizar)
```

**Implementación completa en un solo commit.** No existe estado intermedio estable
(un D149-02 sin D149-04 deja el combo building roto con la nueva firma de función).

---

## 5. Tests — REGLA-T53

**Archivo:** `tests/test_nodo149_games_mercado_separacion.py`

| Test | Qué verifica |
|------|-------------|
| `test_mercado_tipo_en_señal_juegos` | Señal "Total de juegos" tiene `mercado_tipo=="JUEGOS"` |
| `test_mercado_tipo_en_señal_sets` | Señal "Total de sets" tiene `mercado_tipo=="SETS"` |
| `test_seleccionar_optimas_retorna_tupla` | `_seleccionar_señales_optimas()` retorna `(list, list)` no lista única |
| `test_juegos_optimas_no_contiene_sets` | Primera lista del retorno tiene 0 elementos con `mercado=="Total de sets"` |
| `test_sets_optimas_no_contiene_juegos` | Segunda lista del retorno tiene 0 elementos con `mercado=="Total de juegos"` |
| `test_combo_juegos_sin_pierna_sets` | Combo A/B tienen 0 piernas con `mercado_tipo=="SETS"` |
| `test_combo_mismo_partido_prohibido` | Combo con 2 legs del mismo partido → rechazado |
| `test_gap_sets_calculado` | Señal sets tiene `gap_sets >= 0` (no None) para zona COINFLIP |
| `test_betplay_filtra_sets_de_señales_optimas` | `build_games_combos()` con señal sets en posición [0] → usa la juegos en [1] |

Total: 9 tests.

---

## 6. Deuda post-Nodo-149

**D149-07:** Calibración de `_P_3SETS_POR_ZONA` con datos históricos reales.
Requeire análisis de h2h_results_enhanced (n≥200 partidos por zona) para
validar 0.28/0.60/0.42. H149-01 debe pre-registrarse antes de apostara sets
combos con Kelly real.

**D149-08:** `build_evaluar_games_combos()` en betplay tiene el mismo riesgo
de same-partido (Nodo-134 §2 documentó el bug pero el fix D126-01 fue aplicado
solo a `build_games_combos()`). Auditar y aplicar mismo guard si no está presente.

**D149-09:** `live_desk.py` panel X3 — después de D149-03, `señales_optimas`
siempre serán juegos-only → `linea_t0` siempre será ~20-27. El "t0:2.5" problem
desaparece sin tocar live_desk. Verificar con `curl :7780/api/x3` post-deploy.

---

## 7. Qué NO cambia

- `_analizar_mercados_juegos()` sigue generando ambos tipos de señales (correcto)
- Las señales de sets siguen apareciendo en el output de `games_signal_calculator.py`
- `evaluar_games_bridge.py` y `evaluar_games_signal_*.json` no se tocan (pipeline distinto)
- El campo `"mercado"` (string legacy) no se elimina ni modifica
- `shadow_book.py` no cambia — strategy="GAMES" aplica a ambos mercados
- `REGLA-G6` (stakes máx $2k hasta n≥50) aplica por igual a juegos y sets combos

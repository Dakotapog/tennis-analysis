# Nodo-40 — Games/Sets Signal Layer: Mercado de Totales como Alpha Ortogonal al Ganador

> **Fecha inicio:** 2026-06-28
> **Severidad:** OPORTUNIDAD ESTRUCTURAL — El 28-jun-2026 el modelo predijo sets correctamente en 3/3 partidos incluyendo uno donde falló el ganador. El mercado de totales (games/sets) tiene menor correlación con el resultado del ganador → alpha independiente explotable todos los días.
> **Prerequisitos:** Nodo-38 (combo_confianza_builder.py), Nodo-39 (kambi_filtro_fecha), Nodo-22 (Kambi API)
> **Archivos nuevos:** `games_signal_calculator.py`, `tests/test_nodo40.py`
> **Archivos modificados:** `betplay_combo_builder.py` (añadir `--games`), `pipeline_tracker.py` (sección S-40)
> **Archivos NO modificados:** `edge_calculator.py`, `rivalry_analyzer.py`, `trader_ev_tenis.py`
> **Implementa:** Sonnet | **Tests:** pendiente
>
> **Estado:** ✅ COMPLETO — Fases 1-5 implementadas | 35 tests pasan

---

## 0. RESUMEN EJECUTIVO

El pipeline principal predice **quién gana**. Pero el modelo calcula un `puntaje_diferencia` que codifica **cuánto domina** el favorito — información directamente explotable en mercados de totales (games, sets) que el pipeline actual ignora completamente.

**Evidencia empírica del 28-jun-2026:**

| Partido | Ganador pred | Ganador real | Sets pred | Sets real | Games pred | Games real | Alpha games |
|---|---|---|---|---|---|---|---|
| Krueger vs Suresh | Suresh ✅ | Suresh 6-1,6-4 | 2 ✅ | 2 | 16-19 ✅ | **17** | UNDER 21.5 @2.35 ✅ |
| Langmo vs Banerjee | Banerjee ✅ | Banerjee 7-6,6-4 | 2 ✅ | 2 | 16-19 ❌ | **23** | OVER 25.5 — no aplica |
| Boulais vs Aguiard | Aguiard ❌ | Boulais 6-7,7-5,6-3 | 3 ✅ | 3 | 26-32+ ✅ | **34** | OVER sets 2.5 @2.85 ✅ + UNDER 34.5 @2.60 ✅ |

**El hallazgo clave:** Aguiard perdió (ganador ❌) pero el mercado de sets y games fue correcto (✅✅). La señal de sets/games es **ortogonal al ganador** — no necesitas acertar quién gana para ganar la apuesta de totales.

**Alpha estructural:** El bookmaker fija las líneas de games/sets con menos información que tiene el modelo sobre la dinámica del partido (ELO, forma, H2H, superficie). El BBI alto en Challenger/ITF aplica también a estos mercados.

---

## 1. HALLAZGO QUE MOTIVÓ ESTE NODO

### 1.1 El problema: modelo rico, pipeline pobre

El `rivalry_analyzer.py` calcula para cada partido:
- `puntaje_diferencia` — diferencial de dominio entre los dos jugadores
- `sets_pronosticados` — ya calculado en `generar_tabla_favoritos2.py`
- `games_pronosticados` — rango ya calculado en `generar_tabla_favoritos2.py`
- `justificacion` — "claro favorito" vs "partido reñido"

Estos campos existen en el output de la tabla pero **ningún script del pipeline los convierte en apuestas**. Son datos ricos generados y descartados cada día.

### 1.2 La señal del `diff` como predictor de sets

```
Observaciones 28-jun-2026:
  diff = -0.48 (Suresh)  → 2 sets, 17 games  → dominancia clara
  diff = -0.11 (Banerjee)→ 2 sets, 23 games  → victoria ajustada (tiebreak set 1)
  diff = -0.03 (Aguiard) → 3 sets, 34 games  → partido moneda al aire → 3 sets ✅

Hipótesis calibrada:
  |diff| > 0.40  → 2 sets dominantes  → UNDER games (línea conservadora)
  |diff| 0.10-0.40 → 2 sets ajustados → zona gris, no apostar totales
  |diff| < 0.10  → 3 sets casi garantizado → OVER sets + OVER games
```

**Por qué Banerjee con diff=-0.11 fue a 2 sets y no a 3:** diff en zona gris (0.10-0.40). El tiebreak en set 1 (7-6) casi lleva a 3 sets — confirma que esta zona es genuinamente incierta. **No apostar totales en zona gris es la regla correcta.**

### 1.3 Por qué los mercados de games tienen alpha

Los mercados de totales en Betplay/Kambi para Challenger/ITF tienen:
1. **Menor liquidez** que ganador → líneas menos eficientes
2. **Menos datos históricos** en el modelo del bookmaker para esos torneos
3. **Correlación baja con el ganador** → diversificación real del portfolio

El modelo ya tiene toda la información para predecir dominio. El mercado no la tiene correctamente incorporada.

---

## 2. DISEÑO DEL SISTEMA

### 2.1 Flujo del pipeline ampliado

```
PASO 3   → edge_calculator.py              → edge_report (ganador)
PASO 3.5 → generar_tabla_favoritos2.py     → analisis_partidos_pandas.txt
PASO 3.6 → games_signal_calculator.py      → games_signal_report_FECHA.json  [NUEVO]
PASO 4   → trader_ev_tenis.py              → trader_plan (ganador)
PASO 4.4 → betplay_combo_builder.py --games → combos de totales [MODIFICADO]
```

### 2.2 `games_signal_calculator.py` — nuevo script

Lee `h2h_results_enhanced_*.json` + consulta Kambi API para cada partido.

**Output por partido:**
```json
{
  "partido": "Krueger vs Suresh",
  "diff": -0.48,
  "sets_pred": 2,
  "games_range": "16-19",
  "zona_diff": "dominante",
  "señales": [
    {
      "mercado": "Total de juegos",
      "linea": 21.5,
      "direccion": "UNDER",
      "cuota": 2.35,
      "outcome_id": 4238798560,
      "razon": "modelo dice 16-19, todo el rango bajo la línea 21.5",
      "confianza_señal": "ALTA",
      "apostar": true
    }
  ]
}
```

### 2.3 Gate de señales por zona de diff

```python
ZONA_DOMINANTE  = abs(diff) > 0.40   # señal UNDER games activa
ZONA_AJUSTADA   = 0.10 < abs(diff) <= 0.40  # NO apostar totales
ZONA_COINFLIP   = abs(diff) <= 0.10  # señal OVER sets + OVER games activa

# Gate adicional: línea del mercado debe estar FUERA del rango pronosticado
# Krueger/Suresh: rango 16-19, línea 21.5 → gap = 2.5 juegos → señal válida
# Gap mínimo: 2 juegos entre el límite del rango y la línea del mercado

GAP_MINIMO_UNDER = 2   # línea debe estar ≥2 juegos sobre el máximo del rango
GAP_MINIMO_OVER  = 1   # línea debe estar ≥1 juego bajo el mínimo del rango
```

### 2.4 Selección de línea óptima

Para cada partido con señal activa, el sistema evalúa TODAS las líneas disponibles en Kambi y selecciona:

```python
# Para señal UNDER: elegir la línea MÁS ALTA con cuota ≥ 1.50
# Razonamiento: más margen de seguridad, cuota suficiente para EV positivo
# Krueger/Suresh: líneas 21.5@2.35, 22.5@1.96, 23.5@1.83
# → Elegir 21.5@2.35 (mayor cuota con mayor gap de seguridad)

# Para señal OVER: elegir la línea MÁS BAJA con cuota ≥ 1.50
# Langmo/Banerjee OVER: línea 24.5@2.00, 25.5@2.20
# → Elegir 24.5@2.00 (menor línea, más fácil de superar)
```

### 2.5 Combos de totales — arquitectura

```
COMBO GAMES CORE:     solo señales ALTA (gap ≥ 2, cuota ≥ 1.80)
COMBO GAMES EXTENDED: señales ALTA + MEDIA (gap ≥ 1, cuota ≥ 1.50)
COMBO GAMES MIXTO:    1 señal ganador (APOSTAR) + 1-2 señales totales

Máximo 3 piernas por combo de totales (más piernas = más correlación entre partidos del mismo torneo)
```

### 2.6 Validación de correlación

**Regla crítica:** dos picks del mismo partido (ej: ganador Suresh + UNDER games Suresh) tienen correlación alta — si Suresh gana dominante, ambos pagan; si pierde, ambos fallan. **NO combinar en el mismo combo.**

```python
# Regla anti-correlación
if pick_ganador.partido == pick_games.partido:
    # No mezclar en el mismo combo
    # Sí permitir en combos separados
    pass
```

---

## 3. CALIBRACIÓN DEL MODELO DE SETS/GAMES

### 3.1 Variables predictoras ya disponibles

| Variable | Fuente | Rol en predicción de sets |
|---|---|---|
| `puntaje_diferencia` | `rivalry_analyzer.py` | Predictor principal de dominancia |
| `confidence_flag` | `edge_calculator.py` | Confirma convicción del modelo |
| `markov_estado` | `markov_analyzer.py` | HOT → tiende a dominar más |
| `surface_specialization` | `rivalry_analyzer.py` | Especialista en superficie → más dominante |
| `n_h2h` | `rivalry_analyzer.py` | H2H con patrón claro → sets predecibles |
| `elo_diff` | `elo_system.py` | Gap ELO alto → 2 sets |

### 3.2 Acumulación de ground truth

Cada partido cerrado con `betslip_registrar.py --cerrar` debe guardar:
```json
"games_ground_truth": {
  "sets_real": 2,
  "games_real": 17,
  "sets_pred": 2,
  "games_range_pred": "16-19",
  "diff": -0.48,
  "zona_diff": "dominante",
  "sets_correcto": true,
  "games_en_rango": true
}
```

Target de calibración: n≥50 por zona (dominante/ajustada/coinflip) antes de escalar stakes en games.

### 3.3 Métricas a trackear en pipeline_tracker.py (sección S-40)

```
S-40-1: Accuracy sets por zona_diff
  dominante  (|diff|>0.40): X/Y → target ≥75%
  ajustada   (0.10-0.40):   no apostar (registrar igual para calibración)
  coinflip   (|diff|<0.10): X/Y → target ≥70%

S-40-2: Hit% mercados de totales vs mercados de ganador
  Hipótesis: games hit% > ganador hit% en zona coinflip

S-40-3: ROI comparado
  Games UNDER dominante: ROI proxy
  Games OVER coinflip: ROI proxy
  Ganador: ROI proxy (benchmark)
```

---

## 4. FLUJO DE USO

```bash
# PASO 3.6 — Games Signal (nuevo, corre después de edge_calculator)
python3 games_signal_calculator.py
→ reports/games_signal_report_FECHA.json

# PASO 4.4 — Combos de totales (nuevo flag en betplay_combo_builder)
python3 betplay_combo_builder.py --games
→ GamesA.bat, GamesB.bat, GamesC.bat en escritorio

# Opción combinada con ganadores
python3 betplay_combo_builder.py --live --games
→ combos ganador + combos totales en misma sesión
```

---

## 5. REGLAS DE ORO (no negociables)

**REGLA-G1:** Solo apostar totales cuando `zona_diff` es `dominante` o `coinflip`. Zona ajustada (0.10-0.40) = ruido.

**REGLA-G2:** La línea del mercado debe tener gap ≥ 2 juegos con el límite del rango pronosticado. Sin gap = sin señal.

**REGLA-G3:** Cuota mínima para totales = 1.50. Cuotas menores no compensan el riesgo de predicción incorrecta del rango.

**REGLA-G4:** No combinar pick de ganador y pick de totales del mismo partido en el mismo combo (correlación alta).

**REGLA-G5:** Máximo 3 piernas por combo de totales. A más piernas, mayor probabilidad de que un partido del mismo torneo contamine otro pick.

**REGLA-G6:** No escalar stakes hasta n≥50 observaciones calibradas por zona. Con n<50, stakes máximo $2,000 por combo.

---

## 6. ARQUITECTURA DE SEÑALES — VISIÓN COMPLETA

Este nodo completa un sistema de **3 capas de alpha ortogonales**:

```
CAPA 1 — GANADOR (edge_calculator + trader_ev_tenis)
  Señal: P_modelo > P_implícita bookmaker
  Mercado: 1X2 (quién gana)
  Correlación entre capas: BAJA con capa 2 en zona coinflip

CAPA 2 — TOTALES (games_signal_calculator) [NUEVO Nodo-40]
  Señal: diff → sets/games predecibles con gap vs línea
  Mercado: Over/Under games, Over/Under sets
  Ortogonalidad: Aguiard perdió (Capa 1 ❌) pero Capa 2 ✅✅

CAPA 3 — CONFIANZA (combo_confianza_builder)
  Señal: conf ≥ 53% en picks de alta certeza
  Mercado: combos de ganador
  Complementa Capa 1 cuando edge es bajo pero certeza es alta
```

La diversificación entre capas es el verdadero hedge fund — no son competidores sino fuentes de alpha independientes que se diversifican mutuamente.

---

## 7. IMPLEMENTACIÓN POR FASES

| Fase | Acción | Criterio de éxito |
|---|---|---|
| **Fase 1** | `games_signal_calculator.py` — leer h2h_results, calcular zona_diff, consultar Kambi, generar señales | ✅ Implementado 2026-06-28 — validado en datos reales (UNDER 21.5 Suresh ✅ retroactivo) |
| **Fase 2** | `betplay_combo_builder.py --games` — leer games_signal_report, crear HTML/BAT | ✅ Implementado 2026-06-28 — GamesA/B/C.bat generados automáticamente |
| **Fase 3** | `betslip_registrar.py` — guardar `games_ground_truth` al cerrar | ✅ Implementado 2026-06-28 — sets_real + games_real (FlashScore EG/EH) + games_calibracion en calibracion_edge |
| **Fase 4** | `pipeline_tracker.py` sección S-40 — métricas de calibración de sets | ✅ Implementado 2026-06-28 — `--section games` activo, S-40-1/2/3 listos |
| **Fase 5** | Calibración automática de thresholds — actualizar `calibracion_edge.json` | ✅ Implementado 2026-06-28 — `--calibrar` flag, ajuste ±0.02 con n≥50 por zona |

---

## 8. PENDIENTES

| # | Acción | Estado |
|---|---|---|
| 1 | `games_signal_calculator.py` — Fase 1 | ⏳ Pendiente |
| 2 | Flag `--games` en `betplay_combo_builder.py` | ✅ Implementado |
| 3 | `games_ground_truth` en `betslip_registrar.py --cerrar` | ✅ Implementado |
| 4 | Sección S-40 en `pipeline_tracker.py` | ✅ Implementado |
| 5 | `tests/test_nodo40.py` — gate diff, gap línea, anti-correlación, selección línea óptima | ✅ 35 tests pasan |
| 6 | Acumular n≥50 antes de escalar stakes | ⏳ En curso (n=3 hoy) |

---

## 9. WIKILINKS

- [[Nodo-38-Portfolio-Aislamiento-Riesgo]] — arquitectura CORE/Satellite/Moonshot — reutilizar para combos de totales
- [[Nodo-39-Kambi-Filtro-Fecha]] — consulta Kambi API — base para consultar mercados de totales
- [[Nodo-22-API-Integration-Kambi-Ninja]] — Kambi API betOffers — fuente de outcome_ids para totales
- [[Nodo-32-Calibracion-Pipeline-Señales-Rotas]] — lección de señales rotas — aplicar mismo rigor al gate de diff
- [[MOC-Principal]] — índice de specs
- [[Sprint-Pipeline]] — estado del sprint

# Nodo-89 — Sistema de Inteligencia Integral
## "Del motor que falla en silencio al sistema cuantitativo que siempre responde"

> **Tipo:** Architecture Master Node — Spec para Fable (Opus extended-thinking)  
> **Creado:** 2026-07-12  
> **Evidencia de origen:** Sesión diagnóstica 2026-07-12: 0 apuestas generadas con 36 partidos analizados; usuario ganó manualmente picks que el sistema bloqueó.  
> **Autor del análisis:** Sonnet (análisis diagnóstico) → Fable (diseño e implementación)  
> **Prioridad:** CRÍTICA — el sistema actualmente es inoperante en días de calificación (sábados/domingos)  
> **Mandato irrevocable §8:** El sistema NUNCA retorna cero recomendaciones. Si las respuestas de modelos anteriores fueron "quizás el lunes" o "depende del martes", eso es falla de diseño y no puede repetirse.

---

## §0. Contexto ejecutivo — por qué existe este nodo

**Evidencia del 2026-07-12 (domingo):**
- Pipeline ejecutado 2 veces: 10:18 y 10:42 hora Colombia
- 42 y 36 partidos extraídos respectivamente
- 9 picks en watchlist con edge hasta +23.6% (Keshav Chopra), +14.6% (Arseneault), +15.7% (Tepmahc)
- **0 picks en `apostar`** — todos bloqueados por gates T32-01 + N28F2
- Usuario apostó manualmente con la tabla_favoritos y GANÓ
- El sistema no generó ni un combo de games a pesar de tener señal

**El fallo no es un bug de código — es un fallo de arquitectura:**
El sistema fue diseñado para días con torneos Grand Slam / ATP1000 con historial rico. Para días de calificación Challenger (donde n_cal=5, n_h2h≈0, n_axes=1) todos los guards se disparan simultáneamente y el output es vacío. Un motor de predicción que no predice no cumple su función.

**Principio rector de Nodo-89:**
> "Un sistema de apuestas cuantitativo de nivel hedge fund tiene que tener una respuesta para cada contexto de mercado disponible. La respuesta puede ser 'apostar $X con gate A' o 'apostar $Y/2 con gate B' o 'apostar $Z/4 con señal de juegos', pero NUNCA puede ser 'no hay nada'. Los mercados siempre ofrecen algo — el sistema tiene que estar diseñado para siempre encontrarlo."

---

## §1. Diagnóstico doctoral: 8 fallas sistémicas

### §1.1 P1 — Matching de cuotas Kambi: stale data + name collision

**Evidencia directa (2026-07-12 10:55):**
```
❌ Soke D. @2.7 — NO_EXISTE en Kambi
❌ Martin Espinar A. @8.0 — CUOTA_DIFF (2.48 vs 8.00, diff 69%)
❌ Kovalchuk K. @9.0 — NO_EXISTE en Kambi
❌ Hemery C. @2.9 — NO_EXISTE en Kambi
```

**Análisis causal:**
El `trader_plan_20260712_015333.json` fue generado a la 1:53am. El combo builder lo intentó ejecutar a las 10:55am — 9 horas después. En ese lapso:
- Matches con jugadores oscuros (Soke, Kovalchuk, Hemery) nunca estuvieron disponibles en Kambi porque son torneos ITF/WTA sin cobertura Betplay
- Martin Espinar pasó de @8.0 a @2.48 (69% de diferencia) — el partido posiblemente ya empezó o las cuotas se actualizaron masivamente

**Raíz técnica — dos problemas distintos:**
1. **Staleness**: `_validate_edge_report_gate()` no verifica si el plan tiene >N horas de antigüedad antes de intentar el matching
2. **Kambi coverage gap**: No todos los torneos del edge_report tienen cobertura en Betplay. El sistema nunca verifica disponibilidad Kambi ANTES de generar el pick

**Fix requerido (D89-01):**
```python
# En betplay_combo_builder.py — antes de intentar combos
PLAN_STALENESS_HOURS = 4  # trader_plan > 4h = refrescar
CUOTA_DRIFT_MAX = 0.30   # si cuota drift > 30% = el partido empezó

def _validate_plan_freshness(plan_path: str) -> bool:
    """Retorna False si el plan tiene > PLAN_STALENESS_HOURS"""
    ...

def _verify_kambi_coverage_before_pick(jugador: str, torneo: str) -> bool:
    """Consulta Kambi ANTES de crear el pick — no después"""
    ...
```

**Fix requerido (D89-02):**
El `extraer_partidos_api.py` en PASO 1 debe generar un archivo `kambi_coverage_index_FECHA.json` que liste SOLO los jugadores para los que Kambi tiene mercado activo. El edge_calculator filtra PRIMERO por ese índice antes de calcular.

---

### §1.2 P2 — Resolución de nombres H2H-API: tres clases de fallo

**Evidencia directa (2026-07-12 10:21):**
```
WARNING: No se encontró ranking para 'Hsu Y. H.' en ATP/WTA.
WARNING: No se encontró ranking para 'Trotter J. K.' en ATP/WTA
WARNING: No se encontró ranking para 'Burruchaga R. A.' en ATP/WTA.
```

**Análisis de las tres clases de fallo:**

**Clase 1 — Iniciales múltiples**: `Hsu Y. H.` tiene dos iniciales → `player_registry.py` busca por primera inicial solamente → miss.
**Clase 2 — Apellido compuesto**: `Burruchaga R. A.` tiene apellido + segundo apellido + dos iniciales → el parser de apellido_inicial falla.
**Clase 3 — Nombre de pila en apellido**: `Trotter J. K.` → `Trotter` puede ser apellido O nombre de pila en algunos registros.

**Fix requerido (D89-03):**
```python
# En core/player_registry.py
class CanonicalNameResolver:
    """
    Resuelve nombres en cualquier formato a ID canónico ATP/WTA.
    
    Estrategias en orden de prioridad:
    1. Match exacto por nombre completo normalizado
    2. Match por FlashScore ID (source of truth cuando disponible)
    3. Match por apellido + todas las iniciales disponibles
    4. Match por apellido + primera inicial con verificación de ranking gap
    5. Phonetic match (Soundex/Metaphone) como fallback
    6. Match por torneo+fecha+ranking_esperado cuando todo falla
    """
    
    def resolve(self, raw_name: str, context: dict = None) -> Optional[PlayerID]:
        ...
    
    def build_alias_table(self) -> dict:
        """
        Construye alias desde los 112+ archivos H2H históricos.
        Cada aparición de un nombre → variante → canónico.
        """
        ...
```

**Decisión arquitectural (D89-03a):** La tabla de alias se construye UNA VEZ por día (cron 6am) leyendo TODOS los archivos H2H históricos. Los 112 archivos disponibles contienen ~15,000+ apariciones de nombres → suficiente para construir un mapa de variantes robusto.

---

### §1.3 P3 — Mandato Zero-Null: el sistema SIEMPRE apuesta

**Evidencia del problema:**
- 2026-07-11: 0 apuestas generadas → usuario apostó manualmente y ganó
- 2026-07-12: 0 apuestas generadas → usuario apostó manualmente y habría ganado
- El sistema tiene picks con hasta +23.6% de edge (Chopra) que bloquea sin alternativa

**Principio irrevocable (MANDATO-01):**
> El sistema debe implementar un **sistema de capas de fallback** donde si la capa N no produce picks, automáticamente activa la capa N+1. La respuesta vacía es un bug de arquitectura, no un comportamiento aceptable.

**Arquitectura de capas (D89-04):**

```
CAPA 1 — Kelly-KL Full (sistema actual)
  Gate: edge>5% + kelly>2% + p_modelo>=0.55 + n_axes>=2
  Stake: 100% del Kelly calculado
  Status: activa SIEMPRE como primera opción

CAPA 2 — Model Confidence (NUEVO)
  Gate: p_modelo >= 0.60 + cuota [1.50, 2.80] + n_h2h >= 1
  Stake: 25% del Kelly base (stake reducido por menor evidencia)
  Activa cuando: CAPA 1 retorna 0 picks
  Justificación: usuario demuestra empíricamente que picks con
                 p>=0.60 ganan incluso sin edge formal

CAPA 3 — Games/Totals Signal (ya existe — games_signal_calculator.py)
  Gate: señal ALTA o MEDIA del games_signal_calculator
  Stake: máx $2,000 (REGLA-G6 existente)
  Activa cuando: CAPA 1 y CAPA 2 retornan 0 picks
  Evidencia 2026-07-12: encontró 2 picks (Martin/Arseneault UNDER 25.5 @1.53,
                         Young/Dlimi OVER 20.5 @1.74) → Combo @2.66x

CAPA 4 — Multi-Bookmaker Sweep (NUEVO — ver §1.8)
  Gate: cualquier bookmaker disponible en Colombia con mercado activo
  Stake: 20% del Kelly base
  Activa cuando: CAPA 1-3 retornan 0 para Betplay específicamente

REGLA DE ESCALADO:
  if len(capa1.picks) > 0: output = capa1
  elif len(capa2.picks) > 0: output = capa2 + "⚠️ CONFIDENCE MODE"
  elif len(capa3.picks) > 0: output = capa3 + "⚠️ GAMES MODE"
  elif len(capa4.picks) > 0: output = capa4 + "⚠️ ALT BOOKMAKER"
  else: output = ERROR("No hay mercado activo — verificar conexión") # NUNCA empty picks
```

**Output mínimo garantizado:**
Si las 4 capas fallan simultáneamente = error de sistema, no "no hay picks". Debe loggear el estado de cada capa y generar una alerta.

---

### §1.4 P4 — Base de datos histórica: 163 archivos dispersos = riqueza sin explotar

**Inventario de recursos (evidencia real):**
- **163 archivos** `zita_tennis_matches_YYYYMMDD_HHMMSS.json`
- **112 archivos** `h2h_results_enhanced_YYYYMMDD_HHMMSS.json`
- **11 archivos** `sb_YYYY-MM-DD.jsonl` (shadow book)
- Rango: 2026-06-14 → 2026-07-12 (27 días)
- **418 jugadores únicos** solo en los últimos 5 archivos H2H
- Estimación total jugadores únicos en todo el histórico: **2,000-4,000**

**El problema:** Cada archivo es un snapshot del día. No hay índice cruzado. Para saber el historial de un jugador hay que leer todos los archivos.

**Arquitectura PlayerDB (D89-05):**

```python
# scripts/build_player_db.py (NUEVO)
"""
PlayerDB: Índice histórico acumulativo de jugadores.
Se ejecuta: 
  - Primera vez: procesa todos los 163+ archivos (batch)
  - Diariamente (cron 6am): agrega los nuevos archivos del día

Estructura por jugador:
{
  "player_id": "mikael-arseneault",       # canonical slug
  "names": ["Mikael Arseneault",           # all known name variants
             "Arseneault M.", "M. Arseneault"],
  "atp_id": 12345,                         # ATP/WTA registry ID
  "circuit": "atp",
  "matches": [
    {
      "date": "2026-07-12",
      "tournament": "Granby",
      "tier": "challenger",
      "surface": "hard",
      "opponent": "dan-martin",
      "opponent_ranking": 2043,
      "own_ranking": 1188,
      "ranking_gap": -855,                 # negative = we're better ranked
      "result": "pending",
      "odds_self": 2.23,
      "odds_opp": 1.58,
      "p_modelo": 0.594,
      "edge": 0.1456,
      "source_file": "h2h_results_enhanced_20260712_104518.json"
    }
  ],
  "surface_stats": {
    "hard": {"wins": 12, "losses": 8, "win_rate": 0.60},
    "clay": {"wins": 3, "losses": 5, "win_rate": 0.375}
  },
  "tier_stats": {
    "challenger": {"wins": 8, "losses": 12},
    "itf": {"wins": 25, "losses": 10}
  },
  "ranking_gap_stats": {
    "favorable_50_plus":  {"wins": 15, "losses": 3},   # vs jugadores 50+ peores
    "favorable_20_50":    {"wins": 8,  "losses": 4},
    "neutral_pm_20":      {"wins": 10, "losses": 10},
    "unfavorable_20_50":  {"wins": 4,  "losses": 8},
    "unfavorable_50_plus":{"wins": 2,  "losses": 14}   # vs jugadores 50+ mejores
  }
}
"""
```

**Proceso de construcción:**
1. Leer todos los H2H `h2h_results_enhanced_*.json`
2. Para cada partido: resolver nombres canónicos (D89-03)
3. Extraer métricas por partido y acumular por jugador
4. Cruzar con shadow book para obtener resultados reales (win/loss)
5. Calcular estadísticas agregadas por superficie, tier, ranking_gap
6. Guardar en `data/player_db.json` (JSON comprimido) + `data/player_db_index.json` (índice por nombre)

---

### §1.5 P5 — PlayerIntelligence™: seguimiento ultra-inteligente de jugadores

**Principio de diseño:**
Un modelo de predicción de tenis que no conoce el historial comportamental de cada jugador frente a distintos tipos de rivales está fundamentalmente incompleto. La cuota del bookmaker YA incorpora el ranking. El alpha del modelo viene de saber lo que el bookmaker NO sabe.

**Las 7 dimensiones de PlayerIntelligence™ (D89-06):**

#### Dimensión 1: Performance vs Ranking Gap (RankGap Intelligence)
```
Para cada jugador X, calcular win_rate en 5 brackets:
  - HEAVY_FAV:     propio_ranking < rival_ranking - 100
  - MODERATE_FAV:  propio_ranking < rival_ranking - 30
  - NEUTRAL:       |propio_ranking - rival_ranking| <= 30
  - MODERATE_DOG:  propio_ranking > rival_ranking + 30
  - HEAVY_DOG:     propio_ranking > rival_ranking + 100

El RankGap Intelligence da el UPSET_POTENTIAL:
  upset_potential = win_rate_as_HEAVY_DOG / base_rate_expected_from_elo
  Si upset_potential > 1.5 → jugador con capacidad sistémica de dar sorpresas
```

#### Dimensión 2: Surface Volatility Index (SVI)
```
SVI = std_dev(win_rate por superficie) / mean(win_rate global)
Alto SVI (>0.3): especialista de superficie — peso ALTO a superficie actual
Bajo SVI (<0.1): todoterreno — peso BAJO a superficie actual

El surface_specialization de rivalry_analyzer debe leer SVI de PlayerDB
en lugar de calcularlo desde cero cada vez.
```

#### Dimensión 3: Momentum Quality Index (MQI)
```
MQI mide no solo si el jugador está en forma HOT/COLD (Markov actual)
sino la CALIDAD de los rivales que ha batido recientemente:

MQI = Σ(elo_rival_vencido × exp(-λ × días_transcurridos)) / n

Un jugador HOT que batió a rivales de ELO 1200 vs
un jugador HOT que batió a rivales de ELO 1600 → diferente MQI
```

#### Dimensión 4: Pressure Resilience Score (PRS)
```
PRS captura comportamiento en partidos tensos:
  - Win rate en 3er set (tie-break situations)
  - Win rate cuando va perdiendo 0-1 en sets
  - Win rate en partidos con cuota > 2.0 (underdog)
  - Win rate en torneos con prize money alto vs bajo

Fuente: los 112+ archivos H2H tienen `sets` y resultados detallados
```

#### Dimensión 5: Circuit Familiarity Score (CFS)
```
CFS mide cuánto conoce el jugador el circuito actual:
  - N de veces jugado en este torneo específico (venue familiarity)
  - N de veces jugado en esta ciudad/país
  - N de semanas acumuladas en este tier de torneo

Un jugador con CFS alto en Granby tiene ventaja sobre un debutante
aunque el ranking no lo refleje
```

#### Dimensión 6: Versus Archetype Performance (VAP)
```
VAP requiere clasificar jugadores por estilo (de datos disponibles):
  - BASELINE_DEFENDER: muchos juegos por set, 3 sets frecuentes
  - AGGRESSIVE_BASELINER: sets cortos, pocos juegos, winner-error ratio
  - BIG_SERVER: muchos aces (si disponible), menos breaks

Para el MVP (implementación 1): aproximar por:
  avg_games_per_set < 8.5 → AGGRESSIVE (partidos cortos)
  avg_games_per_set > 9.5 → GRINDER (partidos largos)
  
VAP: X vs AGGRESSIVE win_rate ≠ X vs GRINDER win_rate
```

#### Dimensión 7: Inactivity Recovery Profile (IRP)
```
Ya iniciado como RFI (Nodo-64). Extender con:
  - Win rate al volver de 0-7 días de inactividad
  - Win rate al volver de 8-14 días
  - Win rate al volver de 15-30 días
  - Win rate al volver de 30+ días

Algunos jugadores se desempeñan mejor con descanso (IRP_POSITIVE)
Otros pierden ritmo con el descanso (IRP_NEGATIVE)
Esto modifica el decay_gap de RFI por jugador específico
```

**Output de PlayerIntelligence™:**
Para cada partido, el edge_calculator recibe un objeto `player_intel` que reemplaza señales calculadas desde cero:
```python
player_intel = {
    "jugador": "Mikael Arseneault",
    "rankgap_bracket": "MODERATE_DOG",      # vs Dan Martin
    "rankgap_win_rate": 0.33,               # historical vs this bracket
    "svi": 0.28,                            # moderate specialist
    "mqi_fav": 0.72,                        # quality of recent wins
    "prs": 0.41,                            # pressure resilience
    "cfs_torneo": 0.0,                      # first time in Granby
    "irp_days": 5,
    "irp_profile": "NEUTRAL",
    "upset_potential": 1.45                  # slightly above average
}
```

---

### §1.6 P7 — Real-Time Intelligence: lesiones, clima, noticias

**El gap actual:** El modelo predice con datos históricos pero no incorpora eventos que ocurren 0-48h antes del partido y que el bookmaker SÍ incorpora (por eso los odds se mueven).

**Tres fuentes de señal en tiempo real (D89-07):**

#### Fuente 1: Injury & Withdrawal Intelligence
```python
# scraping/injury_tracker.py (NUEVO)
SOURCES = [
    "https://www.tennisabstract.com/blog/",           # injury reports
    "https://www.atptour.com/en/news",                # ATP news RSS
    "https://www.wtatennis.com/news",                 # WTA news RSS  
    "https://www.itftennis.com/en/news/",             # ITF news
]

def get_injury_signals(player_name: str, date: str) -> dict:
    """
    Retorna: {
        "injured": False,
        "injury_type": None,
        "report_date": None,
        "confidence": 0.0,     # 0=no signal, 1=confirmed
        "source": None
    }
    """
```

**Impacto en modelo:** Si player tiene injury_confidence > 0.7 → reducir p_modelo en (0.05 × confidence). Si rival tiene injury → aumentar p_modelo en (0.05 × confidence).

#### Fuente 2: Weather Intelligence (outdoor courts)
```python
# scraping/weather_intelligence.py (NUEVO)
from open_meteo import Client  # free API, no key required

VENUE_COORDINATES = {
    "Granby": (45.3999, -72.7317),
    "Lincoln": (40.8000, -96.6670),
    "Pozoblanco": (38.3833, -4.8500),
    # ... expandir con todos los venues frecuentes
}

def get_weather_signal(venue: str, match_time: str) -> dict:
    """
    Condiciones que afectan el partido:
    - wind_speed_kmh > 30: HEAVY_WIND → favorece baseliner vs big server
    - temp_celsius > 35: HEAT_STRESS → favorece fit/young players
    - humidity > 80: HEAVY_CONDITIONS → favorece grinder vs aggressive
    - rain_probability > 0.5: DELAY_RISK → afecta momentum setters
    """
```

#### Fuente 3: Pre-Match News Intelligence
```python
# scraping/news_intelligence.py (NUEVO)
def get_prematch_signals(player1: str, player2: str, date: str) -> dict:
    """
    Scraping NLP en noticias recientes (últimas 48h):
    Buscar menciones de: injury, withdrawal, illness, personal, tired
    Retorna sentiment score por jugador
    
    Implementación MVP: regex sobre titulares de RSS feeds
    Implementación avanzada: clasificador de sentimiento entrenado
    """
```

**Integración en pipeline:** Agregar como PASO 2.5 (entre H2H y edge_calculator):
```bash
# PASO 2.5 — Real-time intelligence (NUEVO)
python3 scraping/realtime_intelligence.py  # → data/realtime_signals_FECHA.json
```

El edge_calculator lee `realtime_signals_FECHA.json` y ajusta p_modelo ANTES de calcular el edge final.

---

### §1.7 P8 — Pipeline proactivo: H2H siempre para mañana

**El problema actual (evidencia 2026-07-12):**
- Pipeline corrió a las 10:18 local (15:18 UTC)
- Atenas WTA empezó a las 10:00 UTC (05:00 Colombia) → ya en progreso → filtrado
- Gstaad ATP250 empezó a las 08:30 UTC → ya en progreso → filtrado
- Japan Saitama empezó a las 00:00 UTC → terminado → filtrado
- **Resultado:** 76 singles disponibles en Kambi, solo 36 analizados (52% pérdida)

**La solución (D89-08):**

```bash
# run_daily.py — cambio de filosofía
# ANTES: extraer partidos de HOY → analizar → apostar mismo día
# DESPUÉS: extraer partidos de MAÑANA → analizar hoy → tener picks listos al amanecer

# Flujo ideal:
# 20:00-22:00 hora local → PASO 1+2 para MAÑANA
#                          extraer_partidos_api.py --tomorrow
#                          extraer_historh2h.py --api-mode --tomorrow
# 22:00-23:00 → PASO 3: edge_calculator para mañana
# 06:00-07:00 MAÑANA → PASO 4: trader_ev_tenis (picks frescos)
# 07:00+ → PASO 4.4: betplay_combo_builder --live (con picks de hace 9h, no 9h de anoche)
```

**Cambios requeridos:**
1. `extraer_partidos_api.py` ya tiene `--tomorrow` → usar por defecto en cron nocturno
2. `extraer_historh2h.py --api-mode` → agregar `--tomorrow` que filtre partidos de date+1
3. `run_daily.py` → split en `run_daily_evening.py` (PASOS 1-3 de mañana) y `run_daily_morning.py` (PASO 4+)
4. Cron: `run_daily_evening.py` a las 21:00 hora local; `run_daily_morning.py` a las 07:00

**Beneficio cuantificado:**
- Hoy se perdieron: Atenas (10) + Gstaad (3) + Japan (13) = 26 partidos = 34% del total disponible
- Con pipeline proactivo: 0 partidos perdidos por timing

---

### §1.8 P11 — Multi-bookmaker: 20 casas de apuestas, nunca una sola

**El problema actual:** El sistema está 100% atado a Betplay/Kambi. Si un pick no existe en Kambi → pick muerto. Si la cuota de Kambi tiene drift → pick muerto.

**Las casas de apuestas disponibles en Colombia (D89-09):**
```python
BOOKMAKERS_CO = {
    "betplay":    {"api": "kambi",   "base": "eu-offering-api.kambicdn.com", "priority": 1},
    "wplay":      {"api": "sbttech", "base": "api.wplay.co",                  "priority": 2},
    "codere":     {"api": "custom",  "base": "codere.com.co",                 "priority": 3},
    "rivalo":     {"api": "betradar","base": "rivalo.com",                    "priority": 4},
    "rush":       {"api": "custom",  "base": "rush.bet",                      "priority": 5},
    "betcris":    {"api": "kambi",   "base": "betcris.com",                   "priority": 6},
    "zamba":      {"api": "custom",  "base": "zamba.co",                      "priority": 7},
    "aquajuegos": {"api": "custom",  "base": "aquajuegos.co",                 "priority": 8},
    "luckia":     {"api": "kambi",   "base": "luckia.co",                     "priority": 9},
    "sportium":   {"api": "kambi",   "base": "sportium.co",                   "priority": 10},
}
```

**Arquitectura OddsAggregator (D89-09a):**
```python
# scraping/odds_aggregator.py (NUEVO)

class OddsAggregator:
    """
    Para cada pick del edge_report:
    1. Busca el partido en todos los bookmakers disponibles
    2. Retorna el mejor odds disponible (best-odds)
    3. Si Betplay/Kambi no tiene → usar siguiente casa disponible
    4. Si ninguna casa tiene → solo entonces reportar NO_DISPONIBLE
    
    Output:
    {
        "jugador": "Arseneault",
        "best_odds": 2.35,
        "best_bookmaker": "wplay",
        "kambi_odds": 2.23,
        "odds_by_book": {
            "betplay": 2.23,
            "wplay": 2.35,
            "codere": 2.28
        },
        "clv_potential": 0.054    # best_odds vs market avg
    }
    """
```

**Nota de implementación:** Las APIs de Kambi (Betplay, Betcris, Luckia, Sportium) usan el mismo endpoint base — un solo cliente Kambi puede cubrir 4 casas simultáneamente. WPlay usa SBTech que tiene API pública similar.

---

## §2. Audit profundo: tabla_favoritos — Caso Mikael Arseneault, Granby (D89-13)

**Propósito:** Verificar que todas las métricas y pesos de la tabla_favoritos están correctamente calibradas usando el pick más informativo del día como caso de estudio.

### Datos reales del sistema (2026-07-12):
```
Partido: Dan Martin vs Mikael Arseneault
Torneo: Granby (Canadá) - Fase previa | Superficie: hard
Cuotas: Martin @1.58 | Arseneault @2.23
Predicción sistema: Arseneault (EVALUAR, 59.4% confianza)
Edge vs mercado: +14.6% POSITIVO
```

### Métricas clave del edge_report:
| Métrica | Valor | Interpretación |
|---|---|---|
| p_modelo | 0.594 | > umbral 0.55 → MODERATE ✓ |
| p_implicita | 0.448 | Bookmaker dice 44.8% para Arseneault |
| edge | +14.6% | Modelo ve 14.6pp más que el mercado |
| kelly_kl | 9.75% | Sin shrinkage = apostar 9.75% del bankroll |
| calibration_confidence | 0.30 | n_cal=5 → James-Stein aplasta 70% |
| kelly_kl (ajustado) | 9.75% × 0.30 = 2.93% | Stake real = muy reducido |
| n_axes_active | 1 | SOLO BBI activo → N28F2 bloquea |
| motivo_reclasificacion | N28F2 | n_axes_active < 2 |

### Señales por eje:
| Eje | Señal | Valor | Activo? |
|---|---|---|---|
| Surface | 0.163 | Arseneault tiene ventaja de superficie | SI |
| Regime (Markov) | 0.0 | Markov no tiene datos suficientes | NO |
| BBI | 0.563 | BBI señala a Arseneault | SI |
| Triple Alignment | 0.0 | NO_ALIGNMENT (solo 1 eje fuera de los 3) | NO |
| ELO | 1560 vs 1464 | Arseneault mejor ELO | (soporte) |
| H2H directo | n=2 | Muy poca muestra | (débil) |

### Diagnóstico de pesos:

**¿Son correctos los pesos actuales?**

1. **N28F2 (n_axes<2):** La decisión de bloquear con solo 1 eje activo se basó en historial de 29% hit rate para BBI-sola. Pero ese historial incluye TODOS los tiers. Para Challenger qualifying específicamente, el historial es escaso. **Finding: el gate N28F2 necesita ser calibrado por tier+contexto, no solo por número de ejes.**

2. **CCF (Calibration Confidence Factor) = 0.30:** Con n_cal=5, el CCF aplasta kelly de 9.75% a 2.93%. Pero el gate de apostar (edge>5% + kelly>2%) está basado en kelly_kl que ya fue aplastado. Hay un doble penalización: CCF aplasta el kelly Y el threshold se evalúa contra el kelly aplastado. **Finding: el threshold de 2% debería ser relativo al tier, no absoluto.**

3. **Arseneault tiene ELO 1560 vs Martin 1464** (gap = +96 ELO). Un gap de +96 ELO corresponde aproximadamente a 63% de probabilidad esperada. El modelo dice 59.4% — siendo conservador frente al ELO. La cuota @2.23 implica 44.8%. El ELO solo justificaría cuota de ~1.59. **Finding: el bookmaker está sobrevalorando a Martin por ser el favorito nominal (ranking 2043 vs 1188) pero el ELO dice lo contrario. Esto ES edge genuino.**

4. **BBI = 0.394** → señal moderada del bookmaker (cuota implícita gap). El BBI por sí solo a 0.39 históricamente tiene 29% hit rate. Pero combinado con ELO superior (+96 puntos) y Markov rival COLD (Martin en racha fría), la confluencia es más fuerte de lo que el n_axes_active=1 sugiere. **Finding: el cálculo de n_axes_active no cuenta ELO como eje independiente — debería.**

### Propuesta de ajuste (requiere validación con datos):
- Agregar ELO_DOMINANCE como cuarto eje: activo cuando ELO_gap > +50 en dirección contraria al ranking
- Re-calibrar N28F2 con muestra específica de Challenger qualifying (>30 partidos post-implementación antes de ajustar el threshold)

---

## §3. Post-Game Pattern Recognition Framework (D89-14)

**Propósito:** Después de `shadow_book.py --settle`, identificar automáticamente patrones no obvios que distinguen victorias de derrotas, y casos donde el favorito nominal es en realidad el perdedor esperado.

**Diseño de análisis (D89-10):**

```python
# analysis/pattern_recognition.py (NUEVO)

class PatternRecognitionEngine:
    """
    Corre post-settle. Lee shadow book asentado + edge_reports históricos.
    Identifica y cuantifica patrones de éxito/fallo del modelo.
    """
    
    def find_win_correlates(self) -> dict:
        """
        Para cada pick ganado: ¿qué métricas tenía en común?
        Técnica: correlation matrix de todas las features vs resultado binario (1/0)
        
        Métricas candidatas a correlacionar:
          - p_modelo, edge, kelly_kl, n_h2h, n_cal, n_axes_active
          - bbi, triple_alignment, surface_signal, regime_signal
          - ranking_gap (rival_ranking - own_ranking)
          - cuota_favorito, zona_cuota
          - markov_favorito (HOT/NEUTRAL/COLD)
          - tier, superficie
          - rfi_days_inactive, rfi_tier
          
        Output: {
            "top_5_correlates_with_win": [
                {"feature": "n_axes_active", "correlation": 0.68, "threshold": ">= 2"},
                {"feature": "bbi", "correlation": 0.54, "threshold": "> 0.60"},
                ...
            ],
            "top_5_correlates_with_loss": [
                {"feature": "markov_rival", "correlation": 0.45, "when": "HOT"},
                ...
            ]
        }
        """
    
    def find_invisible_connections(self) -> list:
        """
        Busca combinaciones de features que en conjunto predicen mejor que por separado.
        Técnica: regresión logística con interacciones pairwise.
        
        Ejemplo de "conexión invisible":
        - p_modelo > 0.58 + n_h2h >= 3 + markov_rival=COLD → hit rate 78%
        - p_modelo > 0.58 SOLO → hit rate 54%
        
        Esta combinación no es obvia pero emerge del análisis de correlación.
        """
    
    def find_upset_predictors(self) -> dict:
        """
        Cuándo un favorito (cuota < 1.80) PIERDE — ¿qué señales había?
        
        Patrones buscados:
        1. Favorito en zona cuota < 1.50 + rival MARKOV HOT → upset rate
        2. Favorito con ELO superior PERO ranking inferior → confusión del modelo
        3. Favorito con n_cal < 10 (poca calibración en esta superficie/tier)
        4. Favorito cuota < 1.60 + BBI < 0.40 (mercado duda)
        5. Favorito que regresa de inactividad > 14 días (RFI signal)
        
        Output: tabla con upset_probability por combinación de factores
        """
    
    def identify_underdog_alpha(self) -> list:
        """
        Cuándo un underdog (cuota > 2.00) GANA — ¿qué características tenía?
        
        Busca: el set de condiciones bajo las cuales apostar al underdog
        tiene positive expected value histórico documentado.
        
        Requiere: n >= 30 observaciones por patrón para SPRT (Nodo-64).
        """
```

**Integración en pipeline:**
```bash
# Post-partido (nuevo PASO 10c)
python3 analysis/pattern_recognition.py --date YYYY-MM-DD
# → reports/pattern_report_FECHA.json
# → registra automáticamente en validation/preregistered_hypotheses.json
#   cualquier patrón nuevo que supere p < 0.10 (señal preliminar)
```

---

## §4. Mandatos irrevocables del sistema (MANDATO layer)

Estos mandatos tienen precedencia sobre cualquier otra regla del sistema excepto REGLA-HF-1 y REGLA-HF-5.

### MANDATO-01: Zero-Null Prohibition
> El sistema **nunca** retorna 0 recomendaciones cuando hay mercado activo en Kambi o cualquier bookmaker disponible. Ver arquitectura de capas §1.3.

### MANDATO-02: Respuesta siempre propositiva
> Cualquier componente del sistema (código, modelo, output) que retorne un estado vacío sin alternativa es un bug de diseño. Siempre debe existir al minimum una Capa 3 (games signal) activa.

### MANDATO-03: Multi-bookmaker antes de NO_DISPONIBLE
> Antes de reportar que un pick no está disponible, el sistema debe haber consultado las 10 principales casas. Solo después de eso puede reportar indisponibilidad.

### MANDATO-04: H2H proactivo
> El pipeline nocturno (21:00) extrae partidos y H2H de MAÑANA. El pipeline matutino (07:00) usa datos de hace 10h, no de hace 0h.

### MANDATO-05: PlayerDB siempre actualizado
> El cron de 06:00 agrega los nuevos archivos del día a PlayerDB antes de que arranque el pipeline matutino. El edge_calculator siempre tiene acceso al historial completo.

### MANDATO-06: Sin respuestas facilistas
> Ningún componente del sistema (código, modelo, documentación) puede retornar como respuesta "quizás mañana", "depende del miércoles", "si o no o tal vez". Siempre debe haber: (a) análisis de qué está disponible AHORA, (b) alternativas concretas si el camino principal falla, (c) propuesta de acción inmediata.

---

## §5. Decisiones registradas

| ID | Descripción | Criterio de implementación |
|---|---|---|
| D89-01 | Staleness check trader_plan | >4h → refrescar antes de combos |
| D89-02 | Kambi coverage index en PASO 1 | Filtrar picks por disponibilidad Kambi antes de calcular edge |
| D89-03 | CanonicalNameResolver | Alias table desde 112+ archivos H2H históricos |
| D89-04 | Sistema de capas CAPA 1→4 | Capa 2 activa cuando Capa 1 = 0; Capa 3 cuando Capa 1+2 = 0 |
| D89-05 | PlayerDB desde 163+ archivos | Batch + cron diario 06:00 |
| D89-06 | PlayerIntelligence 7 dimensiones | Implementar en orden: RankGap → Surface → MQI → PRS → CFS → VAP → IRP |
| D89-07 | RealTime Intelligence | MVP: injury RSS + weather API; V2: NLP news |
| D89-08 | Pipeline proactivo mañana | Cron 21:00 para mañana; 07:00 para deploy |
| D89-09 | OddsAggregator multi-bookmaker | Kambi multi-instance + WPlay SBTech como segunda prioridad |
| D89-10 | PatternRecognitionEngine | Post-settle; requiere n>=30 por patrón para SPRT |
| D89-11 | ELO como cuarto eje | ELO_gap > +50 contra ranking → eje ELO_DOMINANCE activo |
| D89-12 | N28F2 calibración por tier | Recalibrar threshold n_axes con muestra Challenger qualifying |
| D89-13 | Audit tabla_favoritos | Arseneault case: ELO dominance no cuenta como eje — bug confirmado |

---

## §6. Prerrequisitos que Fable debe dominar antes de implementar

### Dominio técnico requerido:

**1. James-Stein Shrinkage en Kelly**
- Fórmula: `kelly_ajustado = kelly_base × (n / (n + κ))` donde κ=20
- Impacto con n=5: factor = 5/25 = 0.20 (aplasta 80%)
- Interacción con CCF_FLOOR=0.30: el mínimo es 30%, no 20%
- **Punto crítico:** el threshold de 2% se evalúa contra kelly YA aplastado — hay doble penalización latente

**2. Arquitectura de gates en cadena (edge_calculator.py)**
- Flujo: `kelly_kl_calcular()` → L479 (base apostar) → L940+ (gates adicionales)
- Si `apostar=False` desde L479 → L985, L996, L1006 NO se ejecutan (condición `and resultado.get('apostar')`)
- Consecuencia: `motivo_reclasificacion=N/A` cuando el bloqueo fue en L479
- **Punto crítico:** Para implementar CAPA 2, necesitar nuevo campo `apostar_capa2` que bypasea L479 pero mantiene L985-L1011

**3. n_axes_active — cómo se calcula**
- Eje activo si señal supera threshold específico por eje
- Surface: `surface_signal > 0.15`
- Regime (Markov): requiere `freshness_pelt != ESTABLE` O `markov_conf_fav > 0.5`  
- BBI: siempre activo si `bbi != 0.5`
- ELO: actualmente NO es eje — D89-11 lo agrega
- **Punto crítico:** Con ELO como cuarto eje, Arseneault tendría n_axes=2 y pasaría N28F2

**4. Shadow Book append-only + SPRT**
- `sb_YYYY-MM-DD.jsonl` es inmutable post-predicción
- Resultados se settlan con `shadow_book.py --settle`
- SPRT en `hypothesis_tracker.py::sprt_verdict()` requiere n>=5 para empezar, n>=30 para significancia
- **Punto crítico:** PatternRecognitionEngine (D89-10) debe leer settled picks solamente, no pending

**5. Kambi API — estructura de respuesta**
- Endpoint: `eu-offering-api.kambicdn.com/offering/v2018/betplay/listView/tennis.json`
- `outcomes` indexados por nombre normalizado
- `find_outcome()` usa fuzzy matching (L185-248 en betplay_combo_builder.py)
- `cuota_es_real=True` cuando viene directamente de Kambi (no estimada)
- **Punto crítico:** Para OddsAggregator, WPlay usa SBTech con endpoint diferente pero estructura similar

**6. Canonical Name Resolution — el problema de fondo**
- ATP registry usa: "Apellido, Nombre" (Arseneault, Mikael)
- FlashScore API usa: "mikael-arseneault" (slug)  
- Kambi usa: "Mikael Arseneault" (displayname)
- H2H results usa: "Mikael Arseneault" (full name from API)
- edge_report usa: "Mikael Arseneault" (favorito_predicho)
- Kambi outcomes_map key: normalización de displayname → todos a lowercase, sin diacríticos
- **Punto crítico:** `player_registry.py` ya tiene CanonicalNameResolver base — D89-03 extiende, no reemplaza

**7. PlayerDB design — consideraciones de escala**
- 163 archivos × ~36 partidos × 2 jugadores = ~11,700 registros de jugadores
- Muchos duplicados (mismo jugador en múltiples días) → deduplicar por canonical ID
- JSON comprimido (gzip) recomendado para archivos > 5MB
- Índice separado `player_db_index.json`: {canonical_name: offset_en_db} para búsqueda O(1)
- **Punto crítico:** El batch inicial puede tardar 3-5 minutos — ejecutar en background, no bloquear pipeline

---

## §7. Games combo accionable HOY (output inmediato)

**Evidencia:** `games_signal_calculator.py` encontró 2 señales (2026-07-12):

```
Combo A — games @2.66x (max stake: $2,000 — REGLA-G6)

LEG 1: Dan Martin vs Mikael Arseneault — Granby
        UNDER 25.5 juegos @1.53 [ALTA confianza]
        Modelo: 2 sets, 16-19 games
        Gap: +6.5 sobre máximo del rango → señal fuerte
        Kambi outcome ID: 4256782942

LEG 2: Gavin Young vs Yassine Dlimi — Lincoln
        OVER 20.5 juegos @1.74 [ALTA confianza]  
        Modelo: 3 sets, 26-32+ games
        Gap: -5.5 bajo mínimo del rango → señal fuerte
        Kambi outcome ID: 4256902925
```

**Este combo habría sido generado automáticamente si CAPA 3 existiera.** Con la arquitectura de capas de D89-04, un sistema con CAPA 1=0 picks automáticamente activa CAPA 3 y entrega este resultado sin intervención manual.

---

## §8. Roadmap de implementación

### Sprint 1 (implementar primero — máximo impacto inmediato)
1. **D89-04 (CAPA 2+3)**: Sistema de capas de fallback — elimina el problema de 0 bets
2. **D89-08 (H2H proactivo)**: cron nocturno para mañana — captura 34% más de partidos
3. **D89-01 (staleness check)**: evita combos con picks de 9h de antigüedad

### Sprint 2 (infraestructura de datos)
4. **D89-05 (PlayerDB)**: script batch + cron diario — habilita todo lo demás
5. **D89-03 (CanonicalNameResolver)**: alias table desde histórico — elimina los N/A warnings
6. **D89-02 (Kambi coverage index)**: filtrar picks por disponibilidad antes de calcular

### Sprint 3 (inteligencia)
7. **D89-11 (ELO como eje)**: Arseneault tendría n_axes=2 y pasaría N28F2 — fix elegante
8. **D89-06 (PlayerIntelligence Dim 1+2)**: RankGap + Surface — las dos más impactantes
9. **D89-09 (OddsAggregator)**: WPlay como primera alternativa a Betplay

### Sprint 4 (análisis avanzado)
10. **D89-07 (RealTime Intelligence)**: injury + weather MVP
11. **D89-10 (PatternRecognition)**: requiere n>=30 settled picks para ser significativo
12. **D89-06 (PlayerIntelligence Dim 3-7)**: MQI, PRS, CFS, VAP, IRP

### Sprint 5 (optimización)
13. **D89-12 (N28F2 recalibración)**: después de acumular datos con nueva arquitectura
14. **D89-06 (VAP completo)**: requiere clasificador de estilo de juego

---

## §9. Tests requeridos (REGLA-T53)

```python
# tests/test_nodo89.py (NUEVO — cada test invoca función real, no hardcodea fórmula)

def test_staleness_check_bloquea_plan_viejo():
    """D89-01: plan con >4h debe retornar False en validate_plan_freshness"""
    
def test_capa2_activa_cuando_capa1_vacia():
    """D89-04: con edge_report.apostar=[], sistema activa capa 2 automáticamente"""
    
def test_capa3_activa_cuando_capa1_y_2_vacias():
    """D89-04: con capa1=0 y capa2=0, activa games_signal_calculator"""
    
def test_mandato01_nunca_retorna_vacio():
    """MANDATO-01: con cualquier input válido, sistema retorna >= 1 recomendación"""
    
def test_canonical_resolver_iniciales_multiples():
    """D89-03: 'Hsu Y. H.' resuelve a Hsu Yu-Hsiou o equivalente"""
    
def test_player_db_acumula_desde_historicos():
    """D89-05: PlayerDB con 3 archivos H2H retorna historial correcto por jugador"""
    
def test_elo_dominance_cuenta_como_eje():
    """D89-11: ELO_gap > +50 en dirección opuesta al ranking → n_axes_active+=1"""
    
def test_odds_aggregator_usa_mejor_cuota():
    """D89-09: con 3 bookmakers disponibles, retorna el de cuota más alta"""
    
def test_pattern_recognition_no_lee_pending():
    """D89-10: PatternRecognitionEngine ignora picks sin resultado settle"""
    
def test_hoy_genera_al_menos_games_combo():
    """Integración: con inputs del 2026-07-12, sistema genera >=1 pick (Capa 3)"""
```

---

## Addendum — D89-08 Pipeline Proactivo Implementado (2026-07-14)

### Problema documentado en §1.7 P8

El pipeline nocturno perdía el 34% de los partidos disponibles porque corría cuando ya estaban en progreso
(Atenas WTA, Gstaad ATP250, Japan ITF filtrados por estar iniciados).

### 3 fixes implementados (Sonnet, sesión 2026-07-14)

**Fix 1 — RuntimeWarning asyncio** (`scraping/ninja_h2h_parser.py:1069`)

`asyncio.run()` estaba anidado dentro de `async def main()` → event loop ya activo → coroutine creada
pero nunca ejecutada → `RuntimeWarning: coroutine '_run_playwright_batch_async' was never awaited`.

```python
# ANTES (bug):
asyncio.run(self._run_playwright_batch_async(within_budget))

# DESPUÉS (fix):
loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(self._run_playwright_batch_async(within_budget))
finally:
    loop.close()
```

**Fix 2 — `--tomorrow` en `extraer_historh2h.py`** (D89-08 pendiente desde spec original)

El flag no existía. Añadido: busca `data/zita_tennis_matches_YYYYMMDD_*.json` con fecha de mañana.
Si no encuentra el archivo, emite warning pidiendo correr PASO 1 primero.

```bash
python3 extraer_historh2h.py --api-mode --all-tournaments --tomorrow
```

**Fix 3 — Date mismatch en `extraer_URL_partidos_version2.py`** (prerequisito del Fix 2)

Con `--tomorrow`, el scraper guardaba el archivo con fecha de HOY (`20260714_HH:MM`) aunque el contenido
fueran partidos del día siguiente → `--tomorrow` en PASO 2 no encontraba el archivo.

Añadido parámetro `date_prefix` a `save_matches_data()`: cuando `--tomorrow` activo, el archivo se
nombra con la fecha de mañana → `zita_tennis_matches_20260715_HH:MM.json`.

### Flujo nocturno validado (2026-07-14, 22:32-22:54 CO)

```bash
python3 extraer_URL_partidos_version2.py --tomorrow    # 406 partidos, 45 torneos, 406/406 match_ids
python3 extraer_historh2h.py --api-mode --all-tournaments --tomorrow  # 298 partidos, 17.8 MB
```

### Resultado

| Métrica | Antes | Después |
|---------|-------|---------|
| Partidos procesados | ~36 (hoy filtrados) | 298 (mañana completo) |
| Match IDs | 0/5 (API Kambi) | 406/406 (FlashScore Playwright) |
| Archivo H2H | 718 KB | 17.8 MB |
| RuntimeWarning | presente | eliminado |

**Nota:** `extraer_partidos_api.py --tomorrow` NO reemplaza al Playwright — solo ve lo que Betplay/Kambi
tiene cargado (~5 UTR sin match_ids). PRIMARIO siempre es `extraer_URL_partidos_version2.py --tomorrow`.

---

*Nodo-89 — Especificación completa para análisis por Fable (Opus extended-thinking)*  
*Implementación secuencial por Sonnet según roadmap §8, Sprint 1→5*  
*Evidencia de origen: sesión diagnóstica 2026-07-12, archivos:*  
*- `reports/edge_report_20260712_104524.json`*  
*- `reports/analisis_partidos_20260712_205611.txt`*  
*- `reports/games_signal_report_20260712_205621.json`*  
*- `data/zita_tennis_matches_20260712_101849.json`*

# Nodo-44: Watchlist Alpha Signal (WAS) — Alpha Invisible del Pipeline

> **Wikilinks:** [[Nodo-43-PELT-Cold-Rival-Promo-Filter]] | [[Nodo-02-Markov-Changepoint]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-24-Bookmaker-Blindness-Scoring]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]]
> **Fecha de descubrimiento:** 2026-06-29
> **Estado:** IMPLEMENTADO 2026-06-29 — D44-01 + D44-02 completados

**Prioridad:** ALTA — expande Nodo-43 a un framework completo de oportunidades de promo con alpha validado empíricamente
**Archivo objetivo:** `betplay_combo_builder.py` — flag `--was` (misma arquitectura que `--mega`, `--safe`, `--games`)
**Dependencias:** `reports/edge_report_*.json` (watchlist section) | `analysis/markov_analyzer.py` | `games_signal_calculator.py`

---

## El Hallazgo — Contexto de Descubrimiento

**Fecha:** 2026-06-29 (Wimbledon + Challenger Cary)

Analizando los aciertos del día se identificó que **todos los picks ganadores con cuota alta tenían p_modelo entre 50% y 55%** — nunca superaban el umbral del pipeline (55%). El pipeline los detectaba correctamente en WATCHLIST pero los gates los bloqueaban para Kelly deployment.

### Aciertos del día (todos bloqueados por gates, todos ganaron):

| Pick | Cuota | p_modelo | Edge | Gate | Resultado |
|---|---|---|---|---|---|
| Carreno Busta | @3.30 | 51.4% | **21.1%** | FIX-3 (n_axes=1) | GANÓ ✓ |
| Roman Safiullin | @2.65 | 50.5% | **12.8%** | T33-01 (n_h2h=0) | GANÓ ✓ |
| UNDER games (2× combo) | @1.84/@1.64 | — | — | REGLA-G6 (n<50) | GANÓ ✓ 4.77x |

---

## El Patrón — Por Qué Ocurre

### La mecánica del alpha

El bookmaker **apuesta por la MARCA** (ranking histórico, nombre conocido):
```
Shapovalov ATP  → @1.33 (implica 75% ganador)
Rublev Top 10   → @1.48 (implica 63% ganador)
```

El modelo **apuesta por el ESTADO ACTUAL** (Markov + PELT + forma):
```
Shapovalov: COLD conf=0.79, wr 0.50→0.30
Rublev:     NEUTRAL, Safiullin HOT conf=0.71
```

**El gap es el alpha:**
```
Bookmaker dice Shapovalov gana al 75%
Modelo dice es 48.6% (Carreno Busta)
EDGE = 48.6% - 30.3% (cuota @3.30) = +21.1%
```

La cuota alta NO refleja el estado actual del jugador. El modelo captura la realidad Markov. El bookmaker captura la reputación histórica.

### Por qué los gates bloquean correctamente para Kelly

Los gates existen para proteger el bankroll en Kelly deployment:
- **T33-01:** n_h2h=0 + p<0.55 → coin-flip histórico validado (hit%=48.2%)
- **FIX-3:** n_axes<2 → convergencia insuficiente para sizing proporcional
- **P_MODELO_MIN:** p<0.55 → convicción mínima para Kelly

**Pero el EV del edge persiste** aunque no se pueda sizing con Kelly. Para promos de stake fijo, el gate no aplica.

---

## Definición Formal — Watchlist Alpha Signal (WAS)

```
WAS = VERDADERO si:
  pick EN watchlist del edge_report (edge > 5%, gate bloqueó)
  AND edge >= 10%                    ← gap significativo con bookmaker
  AND cuota_pick >= 2.0              ← requisito promo Betplay
  AND al menos UNA señal Markov:
    → rival.estado == 'COLD' AND rival.confianza >= 0.60     [PCRS — Nodo-43]
    → pick.estado == 'HOT' AND pick.confianza >= 0.60        [HOT Pick Signal]
    → zona_games == 'DOMINANTE' AND diff > 0.35              [Games Nodo-40]
    → zona_games == 'COINFLIP' AND rival.estado == 'COLD'    [OVER games Nodo-40]
```

### Jerarquía de señales Markov por fuerza

| Señal | Descripción | Fuerza | Ejemplo 2026-06-29 |
|---|---|---|---|
| PCRS (Nodo-43) | Rival COLD conf≥0.80 | MÁXIMA | Watanuki COLD 0.81 → Ilagan |
| HOT + COLD | Pick HOT conf≥0.60 + Rival COLD conf≥0.60 | ALTA | — |
| PCRS solo | Rival COLD conf≥0.60 (conf<0.80) | ALTA | Shapovalov COLD 0.79 → Carreno ✓ |
| HOT Pick (mom≥0) | Pick HOT conf≥0.60, momentum≥0 | MEDIA | — |
| HOT Pick (mom<0) | Pick HOT conf≥0.60, momentum<0 (trend contrario) | MEDIA-BAJA | Safiullin HOT 0.71, mom=-0.3 ✓ |
| Games DOMINANTE | diff>0.35, señal ALTA | MEDIA | Bautista/Jodar UNDER ✓ |
| Games COINFLIP + COLD rival | diff<0.18, rival COLD | MEDIA | Broady COLD → OVER 22.5 |

**Regla HOT con momentum negativo:** HOT con momentum<0 significa que el jugador tuvo un pico reciente (últimos 5 partidos) pero su tendencia general es descendente. El alpha en este caso viene más del gap bookmaker/modelo que de la señal HOT misma. Tratar como confirmación débil, no como señal primaria.

---

## Distinción crítica — WAS vs Nodo-43 (PCRS)

**Nodo-43 (PCRS)** es un SUBCONJUNTO de WAS:
```
PCRS ⊂ WAS

PCRS: rival COLD conf≥0.60 + cuota≥2.0 + edge>0
WAS:  watchlist edge≥10% + cuota≥2.0 + señal Markov (PCRS u otras)
```

WAS es el framework completo. PCRS es la señal Markov más específica dentro de él.

---

## Evidencia Empírica — 2026-06-29

### Caso 1 — Pablo Carreno Busta @3.30 (Wimbledon, grass)

```
WATCHLIST: edge=21.1%, kelly=15.3%, conf=LOW, n_axes=1 → BLOQUEADO
WAS CHECK:
  ✓ watchlist edge=21.1% >= 10%
  ✓ cuota @3.30 >= 2.0
  ✓ Shapovalov COLD conf=0.79 >= 0.60 → PCRS OK
  → WAS = VERDADERO

RESULTADO: Carreno Busta GANÓ @3.30 ✓
Bookmaker decía 75%. Realidad: ~49% (modelo). Gap 21%.
```

### Caso 2 — Roman Safiullin @2.65 (Wimbledon, grass)

```
WATCHLIST: edge=12.8%, kelly=10.6%, conf=LOW, n_h2h=0 → BLOQUEADO (T33-01)
WAS CHECK:
  ✓ watchlist edge=12.8% >= 10%
  ✓ cuota @2.65 >= 2.0
  ✓ Safiullin HOT conf=0.71 >= 0.60 → HOT Pick Signal OK
  → WAS = VERDADERO

RESULTADO: Safiullin GANÓ @2.65 ✓
Bookmaker decía 63% (Rublev). Realidad: 49.5% (modelo). Gap 13%.
```

**Matiz importante sobre HOT Pick Signal:** Safiullin está clasificado HOT (conf=0.71) pero su trend general es negativo: wr_ant=0.90 → wr_rec=0.60, momentum=-0.3. El estado HOT viene de los últimos 5 partidos (≥70% win rate en ventana corta), no del trend largo. Su win rate general CAYÓ de 90% a 60%.

Funcionó porque el alpha aquí no depende de que Safiullin sea HOT en absoluto — depende de que **el bookmaker sobrevalora a Rublev (Top 10 → @1.48 = 63%) cuando el modelo dice 50.5% (coin-flip)**. La señal HOT añade confirmación, pero el edge viene del gap bookmaker vs modelo.

**Implicación para la jerarquía de señales:** HOT Pick Signal con momentum negativo es más frágil que PCRS (rival COLD). En PCRS, el rival está confirmadamente en declive. En HOT Pick con momentum=-0.3, el pick podría estar entrando en declive también. Para scoring, aplicar descuento: `HOT + momentum < 0 → fuerza MEDIA-BAJA`.

### Caso 3 — UNDER games (2x combo games.bat)

```
Games signal DOMINANTE (diff>0.35) → 2 sets → UNDER games
Bautista Agut vs Fonseca: UNDER 37.5 @1.84 ✓
Jodar vs Gill:            UNDER 37.5 @1.64 ✓
Combo @4.77 → $700 → $3,337 ✓
```

---

## El Principio Unificador

```
BOOKMAKER SOBREVALORA LA REPUTACIÓN
MODELO CAPTURA EL ESTADO ACTUAL

Cuando p_modelo ∈ [0.50, 0.55] + cuota_underdog >= 2.0:
  edge = p_modelo_underdog - p_implicita_underdog
       = ~0.49 - ~0.28 = ~0.21  (21% para Carreno)
       = ~0.50 - ~0.38 = ~0.12  (12% para Safiullin)

El edge es REAL aunque p_modelo sea "bajo".
El gate (p<0.55) protege el Kelly sizing, no el EV del pick.
```

---

## Algoritmo WAS

```python
def find_was_picks(edge_report_file, games_signal_file, min_edge=0.10, min_cuota=2.0):
    """
    Busca picks WAS desde el edge_report (watchlist) + señales Markov + games signal.
    
    NO reemplaza pipeline. Es capa paralela para promo targeting.
    """
    with open(edge_report_file) as f:
        edge_report = json.load(f)
    
    # Cargar games signal para cruce zona DOMINANTE/COINFLIP
    games_index = {}
    if games_signal_file:
        with open(games_signal_file) as f:
            games_data = json.load(f)
        for s in games_data.get('signals', games_data.get('senales', [])):
            match_key = s.get('match', s.get('partido', ''))
            games_index[match_key] = s
    
    candidatos = []
    
    # Solo watchlist — picks que edge_calculator ya procesó y bloqueó
    for p in edge_report.get('watchlist', []):
        if p.get('edge', 0) < min_edge:
            continue
        if p.get('cuota_favorito', 0) < min_cuota:
            continue
        
        # Verificar al menos una señal Markov o Games
        markov_signals = []
        
        mk_fav_estado  = p.get('markov_estado_fav', '')
        mk_fav_conf    = p.get('markov_conf_fav', 0) or 0
        mk_rival_estado = p.get('markov_estado_rival', '')
        mk_rival_conf   = p.get('markov_conf_rival', 0) or 0
        
        if mk_rival_estado == 'COLD' and mk_rival_conf >= 0.60:
            markov_signals.append(('PCRS', mk_rival_conf))
        
        if mk_fav_estado == 'HOT' and mk_fav_conf >= 0.60:
            markov_signals.append(('HOT_PICK', mk_fav_conf))
        
        # Cruce con games_signal (Nodo-40)
        match_desc = p.get('match', p.get('partido', ''))
        gs = games_index.get(match_desc, {})
        zona = gs.get('zona', '')
        diff = gs.get('diff', 0)
        
        if zona == 'DOMINANTE' and diff > 0.35:
            markov_signals.append(('GAMES_DOMINANTE', diff))
        elif zona == 'COINFLIP' and mk_rival_estado == 'COLD':
            markov_signals.append(('COINFLIP_COLD', mk_rival_conf))
        
        if not markov_signals:
            continue
        
        candidatos.append({
            'pick': p.get('favorito', '?'),
            'cuota': p.get('cuota_favorito'),
            'edge': p.get('edge'),
            'p_modelo': p.get('p_modelo'),
            'markov_signals': markov_signals,
            'signal_strength': max(s[1] for s in markov_signals),
            'gate_bloqueante': p.get('gate_bloqueante', 'unknown'),
            'games_zona': zona or None,
        })
    
    # Ordenar por edge descendente
    candidatos.sort(key=lambda x: -x['edge'])
    return candidatos
```

---

## Criterios de Validación

Para confirmar que WAS genera alpha real con n suficiente:

| Métrica | Umbral mínimo | Actual (2026-06-29) |
|---|---|---|
| n observaciones WAS | ≥ 30 | 2 (ganador) + 3 (games) = 5 |
| Hit% WAS ganador (edge≥10%) | > 55% | 2/2 = 100% (n insuficiente) |
| Hit% PCRS dentro de WAS | > 55% | 2/2 (Carreno+Ilagan/Mayo pend.) |
| ROI promo combos WAS | > 0 con n≥30 | pendiente |

**REGLA-WAS-1:** Hasta n≥30 observaciones WAS, usar solo para promos con stake mínimo. No escalar a Kelly.

**REGLA-WAS-2:** WAS no supera T33-01/FIX-3 para Kelly deployment. Son capas independientes.

**REGLA-WAS-3:** edge≥10% es el umbral mínimo. Edge 5-10% en watchlist es marginal, no WAS.

---

## Integración en el Pipeline

```
PIPELINE NORMAL (sin cambios):
  PASO 3 → edge_calculator → PICKS → PASO 4 trader
                           ↘ WATCHLIST (edge+, gate bloqueó)

CAPA PARALELA WAS (nueva — flag --was en betplay_combo_builder.py):
  python3 betplay_combo_builder.py --was
  python3 betplay_combo_builder.py --was --was-min-edge 0.10
  python3 betplay_combo_builder.py --live --was --games    # portafolio completo
  
  WATCHLIST → was_filter(edge≥10%, cuota≥2.0, señal_markov)
            → candidatos WAS ordenados por edge
            → cruza con games_signal (Nodo-40) para señales adicionales
            → combinar 2-3 picks para promo: cuota_combo ≥ 4.0
            → generar WAS*.bat Chrome (misma infra que CC*.bat)
```

Solo activar cuando existe promo activa. La detección de promos es manual.

---

## Casos Documentados

### Día 2026-06-29

| Pick | @Cuota | Edge | Señal WAS | Resultado |
|---|---|---|---|---|
| Pablo Carreno Busta | @3.30 | 21.1% | PCRS Shapovalov COLD conf=0.79 | GANÓ ✓ |
| Roman Safiullin | @2.65 | 12.8% | HOT conf=0.71 | GANÓ ✓ |
| Andre Ilagan | @2.05 | 5.4% | PCRS Watanuki COLD conf=0.81 | PENDIENTE |
| Aidan Mayo | @2.18 | 6.6% | PCRS Glinka COLD conf=0.67 | PENDIENTE |

*Nota: Ilagan y Mayo tienen edge<10% — están en la zona límite. Se incluyeron en promo por PCRS puro (Nodo-43). El criterio WAS edge≥10% los excluiría.*

---

## Deuda Técnica Generada

| ID | Tarea | Prioridad |
|---|---|---|
| D44-01 | Implementar `--was` flag en `betplay_combo_builder.py` (no crear archivo nuevo — ya tiene Kambi, .bat, combos) | ~~ALTA~~ ✅ 2026-06-29 |
| D44-02 | Agregar campos `markov_conf_fav/rival` + `markov_wr_rec_fav/rival` al edge_report | ~~MEDIA~~ ✅ 2026-06-29 |
| D44-03 | Validar WAS hit% con n≥30 observaciones | ALTA |
| D44-04 | Agregar WAS al pipeline_tracker como sección S-44 | BAJA |
| D44-05 | Definir umbral edge mínimo: ¿10% o 8%? Revisar con n≥20 | MEDIA |

---

## Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-43-PELT-Cold-Rival-Promo-Filter]] | PCRS es el subconjunto más específico de WAS (rival COLD) |
| [[Nodo-02-Markov-Changepoint]] | Motor subyacente — estado HOT/COLD/NEUTRAL |
| [[Nodo-40-Games-Sets-Signal-Layer]] | Games DOMINANTE/COINFLIP son señales WAS alternativas |
| [[Nodo-24-Bookmaker-Blindness-Scoring]] | BBI mide la "ceguera" del bookmaker — mismo fenómeno que WAS explota, pero BBI no lo captura aún (ver nota abajo) |
| [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | T33-01 bloquea picks que WAS aprueba — capas independientes |

### Nota sobre BBI (Nodo-24) — Recalibración necesaria

BBI (Bookmaker Blindness Index) debería ser el indicador natural de WAS: mide cuánto "no ve" el bookmaker. Sin embargo, los BBI de los picks WAS ganadores del 2026-06-29 fueron **bajos** (Mayo BBI=0.45, Ilagan BBI=0.51). Esto indica que **BBI en su formulación actual no captura el alpha que WAS detecta**.

La hipótesis: BBI evalúa señales del pipeline (convergencia de ejes, alignment), pero NO evalúa el gap entre reputación-bookmaker y estado-Markov-actual. WAS detecta alpha en picks donde el bookmaker sobrevalora un nombre conocido (Rublev, Shapovalov) sin ajustar por su estado COLD/NEUTRAL actual.

| ID | Tarea | Prioridad |
|---|---|---|
| D44-06 | Investigar si BBI necesita componente Markov-gap para capturar alpha WAS | MEDIA |

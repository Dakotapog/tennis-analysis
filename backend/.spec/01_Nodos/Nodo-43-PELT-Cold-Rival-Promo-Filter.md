# Nodo-43: PELT Cold Rival Filter — Capa de Promo Combos

> **Wikilinks:** [[Nodo-02-Markov-Changepoint]] | [[Nodo-18-PELT-Recency-Alpha]] | [[Nodo-38-Portfolio-Aislamiento-Riesgo]] | [[Nodo-40-Games-Sets-Signal-Layer]] | [[Nodo-33-Filtro-Coinflip-Sin-H2H]]
> **Fecha de descubrimiento:** 2026-06-29
> **Estado:** IMPLEMENTADO VIA NODO-44 — usar `--was --was-min-edge 1` para PCRS puro (edge>0)

**Prioridad:** ALTA — genera oportunidades de EV positivo en promo combos que el pipeline normal bloquea legítimamente
**Archivo objetivo:** `betplay_combo_builder.py --was` (D43-01 supersedido por D44-01)
**Comando PCRS puro:** `python3 betplay_combo_builder.py --was --was-min-edge 1` (edge > 0%, no el 10% default)
**Dependencias:** `analysis/markov_analyzer.py` | `reports/h2h_results_enhanced_*.json` | Kambi NS feed

---

## El Hallazgo — Contexto de Descubrimiento

**Fecha:** 2026-06-29 (Wimbledon, Challenger Cary)

El pipeline normal generó 0 apuestas ese día. Buscando picks para la promo Betplay (cada cuota ≥ 2.0, combinada ≥ 4.0x) se aplicó inspección profunda del Markov data dos niveles adentro del JSON: `ranking_analysis.prediction.markov_analysis.jugador1/jugador2`.

Se encontró que **dos picks bloqueados por gates del pipeline** (T33-01 y FIX-3) compartían exactamente el mismo patrón estructural en sus RIVALES:

```
Watanuki vs Ilagan:
  Watanuki (RIVAL): COLD  win_rate 0.60 → 0.30  PELT conf = 0.81
  Ilagan   (PICK):  HOT   win_rate estable 0.70   cuota @2.05

Glinka vs Mayo:
  Glinka   (RIVAL): COLD  win_rate ~0.60 → 0.30  PELT conf = 0.67
  Mayo     (PICK):  NEUTRAL win_rate 0.50          cuota @2.18
```

Ambos picks bloqueados → ambos aprobados por esta señal → combo 4.47x para promo.

---

## El Patrón — Definición Formal

**PELT Cold Rival Signal (PCRS):**

```
PCRS = VERDADERO si:
  rival.markov_state == 'COLD'
  AND rival.pelt_confidence >= 0.60        ← umbral mínimo de certeza estadística
  AND pick.cuota_favorito >= 2.0           ← requisito del promo target
  AND pick.edge > 0                        ← el modelo tiene al menos dirección correcta
```

**Lectura de señal:**
- `COLD` = el rival ha tenido un cambio de régimen negativo detectado por PELT
- `pelt_confidence >= 0.60` = el cambio no es ruido estadístico
- El rival ha pasado de ganar ~60% a ganar ~30% de sus partidos recientes
- Esta asimetría no se refleja completamente en `p_modelo` porque el modelo está calibrado sobre datos históricos más amplios

---

## Por Qué el Pipeline Normal Lo Bloquea (y Está Bien)

Los gates que bloquearon estos picks son correctos para el **pipeline de Kelly deployment**:

| Gate | Razón de bloqueo | ¿Correcta? |
|---|---|---|
| T33-01 (Nodo-33) | n_h2h=0 + p_modelo < 0.55 → coin-flip | Sí — para apuestas individuales grandes |
| FIX-3 (Nodo-28) | n_axes_active < 2 → convergencia insuficiente | Sí — para Kelly sizing |
| P_MODELO_MIN | p_modelo=0.542 < 0.55 | Sí — umbral de convicción mínima |

**Pero para combos de promo** el cálculo cambia:
- Stake fijo bajo (mínimo para clasificar)
- EV viene del premio fijo (150,000 COP), no del Kelly sizing
- La señal PCRS es un eje adicional que los gates no evalúan

Los gates protegen de apostar fuerte en señales débiles. **PCRS no bypasea los gates** — es una capa **paralela** con propósito distinto.

---

## La Conexión con el Patrón del Día (Puente de Conocimiento)

El mismo día se ganaron dos apuestas UNDER juegos en Wimbledon:

```
Bautista Agut vs Fonseca  → DOMINANTE (diff > 0.35) → 2 sets → UNDER 37.5 ✅
Jodar vs Gill             → DOMINANTE (diff > 0.35) → 2 sets → UNDER 37.5 ✅
```

**La conexión estructural:**

```
UNDER games (Nodo-40):
  diff = p_modelo_winner - p_modelo_loser > 0.35
  → un jugador domina → termina rápido → pocos juegos

PCRS (Nodo-43):
  rival.COLD conf > 0.60
  → el rival está en declive confirmado → el pick tiene ventaja real
  → la "diff efectiva" es mayor que la que ve p_modelo

AMBOS miden lo mismo: ASIMETRÍA entre jugadores
AMBOS usan el motor Markov/PELT como fuente de señal
AMBOS generan alpha cuando la cuota del bookmaker no refleja esa asimetría
```

El puente: `diff` (Nodo-40) es una foto actual de la brecha. `PELT confidence` (Nodo-43) es la certeza estadística de que esa brecha es una **tendencia confirmada**, no ruido de una sesión.

---

## Algoritmo de Búsqueda PCRS

```python
def find_pcrs_picks(h2h_file, min_cuota=2.0, min_pelt_conf=0.60):
    """
    Busca picks que califican para promo combos por PELT Cold Rival Signal.
    
    NO reemplaza edge_calculator. Es una capa paralela para promo targeting.
    """
    partidos = load_h2h(h2h_file)
    candidatos = []
    
    for partido in partidos:
        # Extraer Markov de ambos jugadores (dos niveles de profundidad)
        ma = partido['ranking_analysis']['prediction']['markov_analysis']
        mk1 = ma.get('jugador1', {})
        mk2 = ma.get('jugador2', {})
        
        # Identificar cuál es el rival COLD y cuál es el pick
        for pick_idx, rival_mk in [(2, mk1), (1, mk2)]:
            if (rival_mk.get('estado_actual') == 'COLD'
                    and rival_mk.get('confianza', 0) >= min_pelt_conf):
                
                pick_jugador = partido[f'jugador{pick_idx}']
                cuota_pick = partido[f'cuota{pick_idx}']
                
                if cuota_pick >= min_cuota:
                    res = calcular_edge_completo(partido, calibracion)
                    if res['edge'] > 0:
                        candidatos.append({
                            'pick': pick_jugador,
                            'cuota': cuota_pick,
                            'rival_cold_conf': rival_mk['confianza'],
                            'rival_wr_reciente': rival_mk['win_rate_reciente'],
                            'rival_wr_anterior': rival_mk['win_rate_anterior'],
                            'edge': res['edge'],
                            'p_modelo': res['p_modelo'],
                            **outcome_id_from_kambi(pick_jugador),
                        })
    
    # Ordenar por confianza PELT del rival (mayor certeza = mejor señal)
    candidatos.sort(key=lambda x: -x['rival_cold_conf'])
    return candidatos
```

---

## Criterios de Validación del Patrón

Para validar que PCRS genera alpha real, necesitamos acumular:

| Métrica | Umbral mínimo | Actual (2026-06-29) |
|---|---|---|
| n observaciones PCRS | ≥ 20 | 2 (descubrimiento) |
| Hit% PCRS (rival COLD conf≥0.60) | > 55% | pendiente |
| Hit% cuando rival COLD conf≥0.80 | > 65% | pendiente |
| ROI promo combos PCRS | > 0 con n≥20 | pendiente |

**REGLA-PCRS-1:** Hasta n≥20, usar solo para promo combos con stake mínimo. No escalar a Kelly deployment.

**REGLA-PCRS-2:** PCRS no supera T33-01/FIX-3. Son capas independientes con propósitos distintos.

---

## Evidencia Empírica Relevante de Nodos Anteriores

Del pipeline_tracker (2026-06-29, n=280 picks con n_h2h=0):

```
n_h2h=0 hit%: 48.2%  ← coin-flip confirmado (apoya T33-01)
n_h2h>0 hit%: 68.9%  ← modelo funciona con H2H

PERO esto es sin filtrar por PCRS.
La hipótesis de Nodo-43: dentro de n_h2h=0,
el subconjunto con rival COLD conf≥0.60 tiene hit% > 55%.
```

Del Markov general (Nodo-02, Nodo-18):
```
HOT players hit%:  51.8% (n=222)
COLD players hit%: 29.4% (n=34)  ← picking CONTRA el COLD = 70.6% hit potencial
NEUTRAL:           50.8%
```

**La hipótesis central de Nodo-43:** apostar CONTRA el jugador COLD con alta confianza PELT es alpha estructural, independientemente del n_h2h.

---

## Integración en el Pipeline

```
PIPELINE NORMAL (sin cambios):
  PASO 1 → PASO 2 → edge_calculator → trader → combos Kelly

CAPA PARALELA PCRS (nueva):
  PASO 2 → promo_combo_builder --pcrs
         → Filtra NS picks por: rival COLD conf≥0.60 + cuota≥2.0 + edge>0
         → Cruza con Kambi NS feed para outcome_ids
         → Genera combos de 2-3 picks: cuota_combo ≥ 4.0
         → Output: promo_combos_FECHA.txt + PROMO*.bat en escritorio
```

Solo se activa cuando existe una promo activa. La detección de promos activas es manual (el usuario informa al pipeline).

---

## Cómo Encontrarlo la Próxima Vez

**Proceso de búsqueda PCRS (manual hasta implementar el módulo):**

```python
# Paso 1: Cargar h2h file más reciente
# Paso 2: Para cada partido, acceder a:
partido['ranking_analysis']['prediction']['markov_analysis']['jugador1']['estado_actual']
partido['ranking_analysis']['prediction']['markov_analysis']['jugador1']['confianza']

# Paso 3: Filtrar: rival COLD + confianza >= 0.60 + cuota pick >= 2.0

# Paso 4: Cruzar con Kambi NS feed:
# fetch_kambi_outcomes() → outcomes_map[nombre] → outcome_id

# Paso 5: Construir URL betslip:
# https://betplay.com.co/apuestas#home?coupon=combination|ID1,ID2||replace
```

---

## Casos Documentados

### Caso 1 — 2026-06-29 (Descubrimiento)

| Campo | Valor |
|---|---|
| Promo | Betplay 150,000 COP (cada cuota ≥ 2.0, combo ≥ 4.0x) |
| Pick 1 | Andre Ilagan @2.05 (outcome 4239076908) |
| Rival 1 | Yosuke Watanuki COLD wr=0.30 conf=0.81 |
| Pick 2 | Aidan Mayo @2.18 (outcome 4239112033) |
| Rival 2 | Daniil Glinka COLD wr=0.30 conf=0.67 |
| Combo | 4.47x — Challenger Cary (dura) |
| Pipeline status | Ambos bloqueados (T33-01 + FIX-3) |
| PCRS signal | Ambos aprobados (rival COLD conf ≥ 0.60) |
| Resultado | PENDIENTE |

---

## Deuda Técnica Generada

| ID | Tarea | Prioridad |
|---|---|---|
| D43-01 | Implementar `promo_combo_builder.py` con PCRS automático | ~~MEDIA~~ ✅ Supersedido por D44-01 (`--was --was-min-edge 1`) |
| D43-02 | Agregar campo `rival_cold_conf` a edge_report watchlist | ~~BAJA~~ ✅ D44-02 agrega `markov_conf_rival` (equivalente) |
| D43-03 | Validar hit% PCRS con n≥20 observaciones antes de escalar | ALTA |
| D43-04 | Agregar PCRS al pipeline_tracker como sección S-27-8 | BAJA |

---

## Relación con Otros Nodos

| Nodo | Relación |
|---|---|
| [[Nodo-02-Markov-Changepoint]] | Fuente del PELT confidence — motor subyacente |
| [[Nodo-18-PELT-Recency-Alpha]] | λ temporal — el mismo cambio de régimen medido diferente |
| [[Nodo-33-Filtro-Coinflip-Sin-H2H]] | T33-01 bloquea picks que PCRS aprueba — son capas independientes, no contradictorias |
| [[Nodo-40-Games-Sets-Signal-Layer]] | Puente de conocimiento: ambos miden asimetría entre jugadores por métodos distintos |
| [[Nodo-38-Portfolio-Aislamiento-Riesgo]] | PCRS picks van a promo combos, no al pool Kelly principal |

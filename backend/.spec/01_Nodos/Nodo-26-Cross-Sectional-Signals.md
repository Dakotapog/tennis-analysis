# Nodo-26 --- Cross-Sectional Signals + Circuit Breaker + Line Movement

> **Estado:** 🔧 IMPLEMENTADO --- 2026-06-14
> **Wikilinks:** [[MOC-Principal]] | [[Nodo-25-Dispersion-Guard-Safe-Combos]] | [[Nodo-24-Bookmaker-Blindness-Scoring]] | [[Nodo-15-Portfolio-HedgeFund]]
> **Origen:** Análisis test-time compute --- 5 conexiones ocultas cross-domain
> **Prioridad:** ALTA --- cada una ataca un vacío distinto del pipeline

---

## Los 5 Módulos

### M-26-1: Cross-Sectional Ranking Preservation

**Qué es:** Cuando James-Stein colapsa todos los p_blend al fallback (std < 0.015 = BLIND), el modelo pierde capacidad de discriminar. PERO el ranking de p_modelo sigue siendo informativo --- en el 14-jun, los 3 ganadores tenían los 3 p_modelo más altos del pool.

**Analogía:** Jegadeesh & Titman (1993) --- cross-sectional momentum. No importa si no sabes si el mercado sube o baja; importa qué acciones suben MÁS que otras.

**Fórmula:**
```python
def ranking_preserved_blend(picks_pool, p_historica, js_factor):
    """Solo se activa cuando Dispersion Guard detecta BLIND."""
    p_modelos = [p["p_modelo"] for p in picks_pool]
    p_mean = np.mean(p_modelos)

    AMPLIFICATION = 5.0  # calibrable --- amplifica diferencias relativas

    for p in picks_pool:
        delta = p["p_modelo"] - p_mean
        p["p_blend"] = np.clip(
            p_historica + js_factor * AMPLIFICATION * delta,
            0.40, 0.75
        )
    return picks_pool
```

**Límites:**
- SOLO se activa cuando Dispersion Guard = BLIND (std < 0.015). En DIFFERENTIATED, James-Stein normal funciona bien.
- AMPLIFICATION = 5.0 es un hiperparámetro. Demasiado alto (>10) amplifica noise; demasiado bajo (<2) no diferencia.
- No cambia p_historica ni el cálculo de edge/Kelly --- solo modifica p_blend para ranking y combo scoring.
- No aplica a individuales (Kelly individual usa p_modelo directo). Solo para scoring de combos/megas.

**Vacíos a dominar:**
- V-26-1a: ¿El ranking de p_modelo es señal o noise? Evidencia: 14-jun ganadores avg rank=2.0, perdedores avg rank=6.0 (N=1 sesión BLIND). Se necesitan ≥5 sesiones BLIND para validar. Tracking: `betslip_registrar --cerrar` ya registra resultados --- añadir campo `p_modelo_rank` para evaluar retrospectivamente.
- V-26-1b: AMPLIFICATION=5.0 --- calibrar con datos acumulados. Si ranking correlation (Spearman p_modelo_rank vs resultado) < 0.3 en ≥5 sesiones → desactivar (AMPLIFICATION=0).
- V-26-1c: ¿Qué pasa con Bautista (p_modelo=0.509, PERDIÓ, mismo rank que Seggerman que GANÓ)? Empates de p_modelo son frecuentes --- desempatar por cuota (menor cuota = favorecido por bookmaker).

**Tests:**
```
T26-01: ranking_preserved_blend con pool BLIND → std(p_blend) > 0.01 (antes era < 0.015)
T26-02: ranking_preserved_blend con pool DIFFERENTIATED → NO se activa (retorna sin cambios)
T26-03: picks 14-jun → Hewitt p_blend > Milev p_blend (ranking preservado)
T26-04: AMPLIFICATION=0 → p_blend = p_historica para todos (baseline, no amplifica)
T26-05: AMPLIFICATION=20 → p_blend clipped a [0.40, 0.75] (no explota)
T26-06: picks con p_modelo idéntico (empate) → desempatar por cuota menor
```

---

### M-26-2: Drawdown Circuit Breaker

**Qué es:** Límite de pérdida máxima por sesión. En hedge funds esto es obligatorio. El pipeline actual NO tiene esto --- el 14-jun colocó 25 combos a ciegas.

**Fórmula:**
```python
MAX_SESSION_LOSS_PCT = 0.04  # 4% del bankroll = $5,000 con bankroll $125k

def session_budget(bankroll, max_loss_pct=MAX_SESSION_LOSS_PCT):
    """Presupuesto máximo de inversión por sesión."""
    return bankroll * max_loss_pct

def check_budget(combos_planned, stake_per_combo, bankroll):
    """Pre-sesión: ¿los combos planificados exceden el budget?"""
    total_risk = len(combos_planned) * stake_per_combo
    budget = session_budget(bankroll)
    if total_risk > budget:
        max_combos = int(budget // stake_per_combo)
        return max_combos, f"BUDGET LIMIT: {len(combos_planned)} combos → recortado a {max_combos}"
    return len(combos_planned), "OK"
```

**Límites:**
- Es un control PRE-sesión, NO real-time. Los combos se colocan casi simultáneamente (archivos .bat), no hay forma de parar a medio camino.
- El breaker decide CUÁNTOS combos generar, no cuáles (la selección sigue por mega_score/MPQ).
- NO aplica a individuales del trader (esos ya tienen VaR). Solo a combos y megas.
- Si budget permite 10 combos pero el trader genera 25 → se toman los top 10 por mega_score.

**Vacíos a dominar:**
- V-26-2a: El presupuesto debe incluir TODOS los libros (Normal + Alpha + Beta) sumados. No es $5,000 por libro, es $5,000 total.
- V-26-2b: ¿El breaker se aplica antes del Dispersion Guard o después? DESPUÉS --- primero filtrar picks ciegos, luego limitar presupuesto.
- V-26-2c: ¿Cómo interactúa con --live --mega --safe? Total = sum(combo stakes) + sum(mega stakes) + sum(safe stakes) ≤ budget.
- V-26-2d: El bankroll viene de `--bankroll` en trader_ev_tenis.py pero betplay_combo_builder.py no lo recibe. Solución: leer del trader_plan metadata o nuevo flag `--bankroll`.

**Tests:**
```
T26-07: session_budget(125000, 0.04) = 5000
T26-08: 25 combos × $500 = $12,500 → recortado a 10 combos con budget $5,000
T26-09: 5 combos × $500 = $2,500 → OK (no recorta)
T26-10: --live --mega --safe con budget $5,000 → distribuir entre los 3 libros
T26-11: check_budget recorta combos por mega_score descendente (los peores se eliminan)
```

---

### M-26-3: Line Movement Signal

**Qué es:** El cambio de cuota entre PASO 1 (extracción, ~12h antes) y PASO 4.5 (combo builder, minutos antes de apostar) contiene información del mercado. Smart money mueve líneas.

**Fórmula:**
```python
def line_movement_signal(cuota_original, cuota_actual):
    """Delta de cuota como señal de mercado."""
    delta = cuota_actual - cuota_original
    delta_pct = delta / cuota_original

    if delta_pct < -0.04:  # cuota bajó >4%: smart money entrando
        return 1.10, "STEAM_IN"     # bonus
    elif delta_pct > 0.04:  # cuota subió >4%: mercado rechaza
        return 0.85, "DRIFT_OUT"    # penalty
    else:
        return 1.00, "STABLE"
```

**Implementación:**
1. `extraer_partidos_api.py` ya guarda cuotas en `zita_tennis_matches_*.json` → campo `cuota1`, `cuota2`
2. `betplay_combo_builder.py` ya llama `fetch_kambi_outcomes()` → obtiene `cuota_kambi` actual
3. SOLO falta: cruzar por nombre de jugador y calcular delta

**Límites:**
- Relevante solo si hay ≥4 horas entre PASO 1 y PASO 4.5. Si se corre todo junto, delta ≈ 0 siempre.
- Cuotas de Challenger/ITF se mueven MENOS que ATP/WTA (menos liquidez). El umbral 4% puede ser demasiado alto para tiers bajos.
- Line movement tiene ruido: retiros por lesión causan movimientos bruscos que no son "smart money."
- El bonus/penalty aplica al scoring de combos, no al cálculo de edge/Kelly.

**Vacíos a dominar:**
- V-26-3a: ¿De dónde leer la cuota original? Del `zita_tennis_matches_*.json` (PASO 1) o del `edge_report_*.json` (PASO 3)? El edge_report tiene `cuota_favorito` que es la cuota al momento del cálculo --- usar esta.
- V-26-3b: Name matching entre edge_report (`favorito_predicho`) y Kambi (`participant_name`). Ya resuelto con 3-tier matching existente.
- V-26-3c: ¿Qué pasa si la cuota cambió porque el jugador se retiró? Kambi devuelve `started=True` o `NO_EXISTE` → ya se filtra.
- V-26-3d: Guardar delta en el output de combos para evaluar retroactivamente si STEAM_IN picks ganan más que DRIFT_OUT.

**Tests:**
```
T26-12: cuota 2.50 → 2.30 (delta -8%) → STEAM_IN, bonus 1.10
T26-13: cuota 2.50 → 2.70 (delta +8%) → DRIFT_OUT, penalty 0.85
T26-14: cuota 2.50 → 2.48 (delta -0.8%) → STABLE, factor 1.00
T26-15: cuota_original = None (no hay dato previo) → factor 1.00 (skip)
T26-16: delta integrado en mega_score: combo con 3 STEAM_IN piernas > combo con 3 DRIFT_OUT
```

---

### M-26-4: Meta-Markov (Session-Level Regime)

**Qué es:** Aplicar detección de régimen al rendimiento del MODELO por sesión, no solo por jugador. Si el modelo viene de 2 sesiones malas → reducir exposición.

**Fórmula:**
```python
def session_regime(recent_sessions, lookback=5):
    """Evalúa régimen del modelo basado en sesiones recientes."""
    if len(recent_sessions) < 3:
        return "INSUFFICIENT", 1.0  # sin datos → no modificar

    recent_acc = [s["accuracy"] for s in recent_sessions[-lookback:]]
    avg_acc = np.mean(recent_acc)
    trend = recent_acc[-1] - recent_acc[0]  # tendencia

    if avg_acc < 0.50:
        return "COLD_MODEL", 0.50       # reducir stakes 50%
    elif avg_acc < 0.60 and trend < -0.10:
        return "COOLING", 0.75           # reducir stakes 25%
    elif avg_acc > 0.70:
        return "HOT_MODEL", 1.00         # mantener normal (NO aumentar)
    else:
        return "NEUTRAL", 1.00
```

**Límites:**
- Requiere ≥3 sesiones registradas con resultados completos. Hoy: n=3 (apenas suficiente).
- NUNCA aumenta stakes (HOT_MODEL = 1.00, no 1.20). Solo reduce en drawdown. Razón: sesión épica seguida de desastre muestra que el modelo no "predice su propia calidad."
- La accuracy por sesión depende del MIX DE TIERS. Una sesión con solo ITF (accuracy esperada ~59%) no es comparable a una de Grand Slam (75.8%). Solución: usar accuracy vs expected accuracy del tier, no accuracy absoluta.
- Fuente de datos: `betslip_registrar --cerrar` ya calcula accuracy por sesión → `reports/apuestas_*.json`.

**Vacíos a dominar:**
- V-26-4a: Con n=3 sesiones, el Meta-Markov es frágil. Activar SOLO cuando n≥5. Antes de eso: INSUFFICIENT → factor=1.0.
- V-26-4b: Accuracy por sesión varía por tier mix. Normalizar: `adjusted_acc = accuracy / expected_accuracy_for_tier_mix`. expected = weighted average de fallback_por_tier.
- V-26-4c: ¿Dónde vive el estado del Meta-Markov? En `calibracion_edge.json` junto a los contadores por tier. Nuevo campo: `session_history: [{fecha, accuracy, n_picks, tier_mix}]`.
- V-26-4d: ¿Qué ventana? lookback=5 sesiones. Más antiguas ya no reflejan el estado del modelo.
- V-26-4e: El factor Meta-Markov se multiplica DESPUÉS del Kelly-VaR. No afecta el cálculo de edge, solo el sizing.

**Tests:**
```
T26-17: 3 sesiones [100%, 90%, 33%] → avg=74.3% → HOT_MODEL, factor=1.0
T26-18: 3 sesiones [33%, 40%, 45%] → avg=39.3% → COLD_MODEL, factor=0.50
T26-19: 2 sesiones → INSUFFICIENT, factor=1.0 (no hay suficiente data)
T26-20: trend negativo [80%, 60%, 40%] → COOLING, factor=0.75
T26-21: factor se aplica a stakes, no a edge ni p_blend
```

---

### M-26-5: CV Guard (Session Edge Blindness)

**Qué es:** Coeficiente de Variación de los edges de la sesión. CV bajo = el modelo da edge similar a todos los picks = ciego a nivel de sesión. Complementa Dispersion Guard (que mide p_blend, no edge).

**Fórmula:**
```python
def cv_edge_guard(picks_pool):
    """Coeficiente de variación de edges. CV bajo = modelo ciego."""
    edges = [p["edge"] for p in picks_pool if p["edge"] > 0]
    if len(edges) < 3:
        return None, "INSUFFICIENT"

    cv = np.std(edges) / np.mean(edges) if np.mean(edges) > 0 else 0

    if cv < 0.15:
        return cv, "BLIND_EDGE"        # edges casi idénticos
    elif cv < 0.30:
        return cv, "LOW_VARIANCE_EDGE"  # poca variación
    else:
        return cv, "DIVERSE_EDGE"       # modelo distingue
```

**Límites:**
- Solo evalúa picks con edge > 0 (los sin_edge tienen edge negativo y distorsionan el CV).
- CV es complementario a Dispersion Guard. Dispersion mide p_blend; CV mide edge. Pueden divergir: p_blend idéntico + cuotas distintas = edges distintos. Ambos deben ser BLIND para bloquear.
- Acción por nivel: BLIND_EDGE + BLIND dispersion → bloquear megas. BLIND_EDGE solo → warning, no bloqueo.
- Mínimo 3 picks con edge > 0 para calcular CV. Si hay menos → skip.

**Vacíos a dominar:**
- V-26-5a: CV depende de la distribución de cuotas, no solo del modelo. Si todas las cuotas son similares, CV será bajo incluso si el modelo discrimina bien. Solución: normalizar edge por cuota antes de calcular CV.
- V-26-5b: El umbral 0.15 viene de 2 sesiones (13-jun CV=0.51, 14-jun CV=0.18). Se necesitan ≥10 sesiones para calibrar. Tracking obligatorio.
- V-26-5c: ¿CV se calcula sobre el pool completo o por tier? Por tier puede tener n muy bajo. Sugerencia: pool completo, con nota de warning si >80% picks son del mismo tier.

**Tests:**
```
T26-22: edges [0.14, 0.165, 0.08, 0.072, 0.05] → CV=0.51 → DIVERSE_EDGE
T26-23: edges [0.088, 0.086, 0.083, 0.081, 0.074] → CV=0.06 → BLIND_EDGE
T26-24: edges [0.10] → INSUFFICIENT (n<3)
T26-25: BLIND_EDGE + Dispersion BLIND → bloquear megas
T26-26: BLIND_EDGE + Dispersion DIFFERENTIATED → warning solamente
```

---

## Orden de Implementación (por impacto real en P&L)

| Prioridad | Módulo | Impacto | Dependencias | Dificultad |
|---|---|---|---|---|
| **1** | M-26-2 Circuit Breaker | ALTO --- evita pérdida catastrófica directamente | bankroll en combo_builder | Baja |
| **2** | M-26-3 Line Movement | ALTO --- señal de mercado gratuita (Kambi ya se llama 2x) | cruzar cuotas por nombre | Media |
| **3** | M-26-1 Cross-Sectional | ALTO --- resuelve el problema raíz (James-Stein collapse) | Dispersion Guard (ya existe) | Media |
| **4** | M-26-5 CV Guard | MEDIO --- segunda capa de detección de blindness | Nodo-25 (ya existe) | Baja |
| **5** | M-26-4 Meta-Markov | MEDIO --- pero requiere n≥5 sesiones para activar | session_history en calibración | Media |

---

## Restricciones Globales

- **R-26-1:** Ningún módulo modifica edge ni Kelly-KL directamente. Solo afectan scoring de combos y sizing (multiplicadores).
- **R-26-2:** Todos los módulos son ADITIVOS a los guards existentes (Nodo-24/25). Nunca contradicen.
- **R-26-3:** Todos requieren tracking retrospectivo obligatorio para calibrar umbrales. Guardar en output JSON.
- **R-26-4:** Circuit Breaker es PRE-sesión, no real-time. No se puede cancelar un combo ya colocado.
- **R-26-5:** Meta-Markov NUNCA aumenta stakes. Solo reduce o mantiene.
- **R-26-6:** Line Movement solo es señal si >4 horas entre extracción y apuesta. Si pipeline se corre junto → skip.

---

## Validación Mínima Antes de Confiar

| Módulo | N mínimo para confiar | Métrica de validación | Fuente |
|---|---|---|---|
| M-26-1 Ranking | 5 sesiones BLIND | Spearman(p_modelo_rank, resultado) > 0.3 | betslip_registrar |
| M-26-2 Breaker | 1 sesión (es un límite, no predicción) | session_loss ≤ budget en ≥90% sesiones | trader output |
| M-26-3 Line Mov | 20 picks con delta | STEAM_IN accuracy > DRIFT_OUT accuracy | edge_report + kambi |
| M-26-4 Meta-Markov | 10 sesiones | accuracy_post_COLD < accuracy_post_HOT | session_history |
| M-26-5 CV Guard | 10 sesiones | BLIND_EDGE accuracy < DIVERSE_EDGE accuracy | edge_report |

---

## Implementación Real — 2026-06-14

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `betplay_combo_builder.py` | 7 funciones nuevas (líneas 739-882): `session_budget`, `check_budget`, `_find_bankroll_from_plans`, `line_movement_signal`, `ranking_preserved_blend`, `cv_edge_guard`, `session_regime` |
| `betplay_combo_builder.py` | `--bankroll N` CLI arg añadido |
| `betplay_combo_builder.py` | `main()`: M-26-2/4 inicializados al arranque (presupuesto + Meta-Markov) |
| `betplay_combo_builder.py` | `build_mega_combos()`: M-26-1/3/5 integrados en el loop de available_pool y scoring |
| `betplay_combo_builder.py` | `mega_score` formula: `mpq_product × log(cuota) × cross_tier_bonus × gap_penalty × golden_bonus × line_product` |
| `tests/test_nodo26.py` | 26 tests T26-01→T26-26 — todos pasan |

### Verificación real con output

```
M-26-1: Pool 14-jun BLIND (std=0.00069) → ranking_preserved_blend → std=0.00832 (12×)
         Orden: Lim>Bu>Fearnley>Miguel>Carnicella>Romano=Pawlikowska ✅
M-26-2: $5k bankroll → budget $200 → 4 combos @$500 BLOQUEADOS ✅
         $50k → budget $2,000 → 4 combos @$500 pasan (exacto al límite) ✅
M-26-3: STEAM_IN (2.75→2.50, -9%) → factor=1.10 ✅
         DRIFT_OUT (2.50→2.80, +12%) → factor=0.85 ✅
         NO_DATA (cuota_original=None) → factor=1.00 ✅
M-26-4: INSUFFICIENT (session_history vacío, n=0) → factor=1.00 ✅
         Simulado COLD (40/45/35%) → COLD_MODEL, factor=0.50 ✅
         Simulado HOT (90/80/85%) → HOT_MODEL, factor=1.00 ✅
M-26-5: Edges idénticos (0.10×5) → CV=0.00 → BLIND_EDGE ✅
         Edges variados (14%,16%,8%,7%,8%) → CV=0.32 → DIVERSE_EDGE ✅
         BLIND + BLIND_EDGE → MEGA BLOQUEADO ✅
```

### Tests

```
Suite completa: 1006 passed (era 980, +26 Nodo-26), 0 failed
tests/test_nodo26.py: 26/26 passed
```

### Estado actual de vacíos

| Vacío | Estado |
|---|---|
| V-26-1a: ¿Ranking p_modelo es señal? (N=1 sesión BLIND) | 📋 Pendiente — acumular con betslip_registrar |
| V-26-1b: Calibrar AMPLIFICATION=5.0 con ≥5 sesiones BLIND | 📋 Pendiente |
| V-26-2a: Budget incluye todos los libros (Normal+Alpha+Beta) | ⚠️ Parcial — solo aplica a megas ahora |
| V-26-3a: line_movement_signal activo pero NO_DATA en pipeline junto | 📋 Normal — requiere ≥4h entre PASO 3 y 4.5 |
| V-26-4a: Meta-Markov inactivo (n=0 sesiones) | 📋 Normal — se activa tras 3 sesiones con --cerrar |
| V-26-4b: Normalizar accuracy por tier mix | 📋 Pendiente — calibración futura |
| V-26-5b: Umbral CV=0.15 basado en 2 sesiones | 📋 Pendiente — calibrar con ≥10 sesiones |

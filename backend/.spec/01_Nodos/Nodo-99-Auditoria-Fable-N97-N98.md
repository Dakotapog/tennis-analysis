# Nodo-99 — Auditoría Fable: Pre-implementación Nodo-97 × Nodo-98

> **Wikilinks:** [[Nodo-97-Live-Edge-Monitor]] | [[Nodo-98-Meta-Senal-Convergencia]] | [[Nodo-74-Combo-Governor]] | [[Nodo-68-Rival-Value-Flip]] | [[Nodo-43-PELT-Cold-Rival-Promo-Filter]] | [[Nodo-73-n8n-CloseSnapshot-Timing]] | [[Nodo-65-Convergencia-Multi-Senal-Patron-Combos]]
> **Fecha:** 2026-07-14 | **Autor:** Fable (auditoría Sonnet 4.6 via Graphify + Tamp)
> **Contexto:** Antes de implementar scripts/live_edge_monitor.py y scripts/meta_signal_scorer.py, Fable realizó auditoría completa de los dos specs. Este nodo documenta los hallazgos, las correcciones aplicadas y las decisiones que desbloquean la implementación.

---

## 1. METODOLOGÍA

Herramientas usadas:
- **Graphify** (:7779, 99 nodos indexados, Nodo-97/98 incluidos post-rebuild)
- **Tamp** (:7778, proxy Anthropic activo, ~18% compresión de tokens)
- **BFS depth=2** sobre "meta_signal live_edge_monitor convergencia" → 159 nodos encontrados
- `graphify path N97 → N74` → **1 hop** (referenciado, no integrado)
- `graphify path N98 → N68` → **2 hops** via RFI (conexión indirecta, no explícita)
- `graphify path N97 → shadow_book.py` → **SIN CAMINO** (gap crítico)
- `graphify path N97 → Nodo-43` → **SIN CAMINO** (oportunidad oculta)

---

## 2. BLOCKERS CRÍTICOS — deben resolverse ANTES de escribir código

### B1 — Kambi LIVE endpoint no verificado (D97-10 gateado)

**Hallazgo:** `live_edge_monitor.py` depende del endpoint live de Kambi para fetch de cuotas en tiempo real. El endpoint pre-game (D90-08 OddsAggregator) usa `/offering/api/v3/{offering}/listView/...`. El endpoint live usa `/event/{eventId}/livedata` u otro feed — **no confirmado**.

**Riesgo si se ignora:** Los 8 tests de Nodo-97 pasarán con mocks pero el sistema será `MODO_OBSERVACION` indefinido. La ventana de 60-90s se cierra antes de que el sistema siquiera obtenga la cuota real.

**Decisión D99-01:** Diseñar `live_edge_monitor.py` con adapter intercambiable:
```python
class KambiLiveClient:
    def get_live_odds(self, event_id) -> float | None:
        ...  # implementación real cuando endpoint confirmado
    
class KambiPreGameFallbackClient:
    def get_live_odds(self, event_id) -> float | None:
        # re-fetch offering pre-game como proxy (cuota de apertura 2da mitad)
        ...
```
Gate de activación real: endpoint confirmado via DevTools → Fable confirma antes del Sprint de implementación.

**Estado:** PENDIENTE — acción manual requerida (DevTools en partido Betplay en vivo)

---

### B2 — Shadow Book sin protocolo para picks live (gap confirmado por Graphify)

**Hallazgo:** `graphify path Nodo-97 → shadow_book.py` retornó **SIN CAMINO**. El spec Nodo-97 menciona "shadow_book.py log automático del pick live para CLV tracking" pero `shadow_book --log` fue diseñado exclusivamente para picks pre-partido.

**Diferencia crítica:**

| Tipo pick | Momento de log | Cuota para CLV | Cuota cierre |
|---|---|---|---|
| Pre-partido | Pre-game (días antes) | cuota_pre | cuota_cierre_kambi |
| **Live** | **En trigger (~5min in partido)** | **cuota_live (trigger)** | cuota_final_set/partido |

El CLV de picks live = `(cuota_trigger - cuota_cierre) / cuota_cierre`. Si se usa el mismo campo `cuota_log` sin distinguir, el shadow book mide CLV incorrecto.

**Decisión D99-02:** `shadow_book.py --log-live` (nuevo modo, o campo `pick_type='live'` en el JSON):
```json
{
  "pick_type": "live",
  "cuota_trigger": 2.90,
  "trigger_ts": "2026-07-14T14:05:33",
  "drift_pct": -18.3,
  "edge_live": 0.124
}
```
El reporte CLV separa `CLV_pregame` vs `CLV_live` en sección dedicada.

**Estado:** NUEVO CAMPO D99-02 — añadir a Nodo-97 §2 como D97-13

---

### B3 — Rival Value señal bivalente en meta_signal_score (D98-10 incompleto)

**Hallazgo:** La fórmula en Nodo-98 §4 incluye Rival Value como "+1 (señal contraria)" dentro del mismo `meta_signal_score`. Esto crea el escenario peligroso:

```
HOT(+1) + STRONG(+1) + RivalValue(+1) = score 3
PERO: HOT y STRONG dicen "apuesta al FAVORITO"
      RivalValue dice "apuesta al RIVAL"
```

El trader recibe `meta_signal_score=3` sin saber que 2/3 señales van en una dirección y 1/3 en la opuesta. Puede apostar al favorito creyendo tener score 3 de convergencia.

**Decisión D99-03:** Separar el output en DOS scores:
```json
{
  "meta_signal_score": 2,        // solo señales pro-favorito
  "score_rival_value": 1,        // señales que dicen "apuesta al rival"
  "direccion": "FAVORITO",       // "FAVORITO" | "RIVAL" | "SPLIT"
  "senales_activas_fav": ["HOT", "STRONG"],
  "senales_activas_rival": ["RIVAL_VALUE"]
}
```
Si `score_rival_value >= 1` Y `meta_signal_score >= 2` → `direccion="SPLIT"` → alerta especial: conflicto de señales, no apostar en combo mixto.

**Estado:** CORREGIR Nodo-98 §4 + §8 antes de implementar scorer

---

## 3. GAPS TÉCNICOS — solucionables antes de implementar

### G1 — Sin hipótesis pre-registrada H97-01

**Hallazgo:** Nodo-98 tiene H98-01 registrada. Nodo-97 tiene 0 hipótesis. El Live Edge Monitor es observacional durante gate (5 sesiones), pero cuando se activa, apuesta con dinero real. Sin hipótesis pre-registrada, el sistema tiene **cero mecanismo estadístico** para decidir cuándo activar Telegram permanentemente.

**Decisión D99-04:** Pre-registrar H97-01 en `validation/preregistered_hypotheses.json`:
```json
"H97-01": {
  "nombre": "Live Edge: picks con drift>=15% y edge_live>5% superan breakeven de cuota_live",
  "umbrales_congelados": {
    "drift_min_pct": 15.0,
    "edge_live_min": 0.05,
    "ventana_min": -30,
    "ventana_max": 45
  },
  "n_stop": 20,
  "estado": "ACUMULANDO",
  "n_actual": 0,
  "preregistrado": "2026-07-14"
}
```
n_stop=20 (menor que H98-01 n=30 porque los triggers live son más escasos).

**Estado:** IMPLEMENTAR como parte de este nodo

---

### G2 — Ventana horaria asimétrica vs test simétrico (bug en spec Nodo-97)

**Hallazgo:** Spec D97-06: "30min ANTES hasta 45min DESPUÉS" = ventana de 75 minutos **asimétrica**.
Test #5 planificado: "pick fuera de ventana ±75min" = 150 minutos **simétricos**.

Inconsistencia: un pick 60 minutos ANTES del partido debería estar FUERA de ventana (>30min de margen), pero con ±75min estaría DENTRO.

**Decisión D99-05:** Corregir Nodo-97 §7 test #5:
```python
# ANTES (incorrecto):
# "pick fuera de ventana ±75min → excluido"

# DESPUÉS (correcto):
def test_ventana_horaria_activa(self):
    # pick 60min antes del partido → FUERA de ventana (>30min pre)
    # pick 20min antes del partido → DENTRO de ventana
    # pick 50min después del inicio → FUERA de ventana (>45min post)
    # pick 30min después del inicio → DENTRO de ventana
```

**Estado:** CORREGIR spec Nodo-97 §7 test #5

---

### G3 — Combo Governor no integrado en live combos (1 hop en grafo, sin protocolo)

**Hallazgo:** Graphify encontró 1 hop entre Nodo-97 y Nodo-74 (Combo Governor). El grafo confirma que el spec referencia al Governor pero no define la integración. El Combo Governor tiene `budget_sesion` diario. Un live trigger se dispara autonomamente desde n8n — si el budget ya fue consumido por el daily pipeline, ¿el live combo lo ignora?

**Riesgo:** overbet en sesión ya consumida. El daily pipeline gastó Kelly calculado → n8n dispara live combo adicional → bankroll real supera allocation del día.

**Decisión D99-06:** Live combo respeta Combo Governor con modo especial:
- Si `governor.budget_restante > 0` → stake = Kelly live shrink 5% (como H88-01)
- Si `governor.budget_restante == 0` → stake = $0, alerta "OBSERVACION — budget diario agotado"
- El live combo es ADICIONAL al presupuesto solo si `KGR > 0` Y `governor.budget_restante > 0`

**Estado:** AÑADIR D97-14 en Nodo-97 §2

---

### G4 — ELO dominance threshold sin referencia en Nodo-98

**Hallazgo:** Nodo-98 §4 usa `elo_dominance_axis=True` como señal de score. El threshold que define cuándo `elo_dominance_axis=True` está en Nodo-91 Sprint1. Si Nodo-91 cambia el threshold, Nodo-98 hereda el cambio silenciosamente.

**Decisión D99-07:** Añadir en Nodo-98 §4 nota explícita:
> "elo_dominance_axis threshold definido en D91-XX (Sprint1). Ver Nodo-91 para cambios."

**Estado:** DOCUMENTAR en Nodo-98 §4 (1 línea)

---

### G5 — meta_signal_scorer.py sin slot en run_daily.py

**Hallazgo:** D98-07 define el script pero no especifica cuándo corre en el pipeline. Sin slot asignado, el scorer solo corre manualmente — el trader no tiene el score antes de apostar.

**Decisión D99-08:** Asignar PASO 3b en run_daily.py:
```python
# PASO 3b — Meta-Señal Convergencia (Nodo-98, REPORTE_SOLO)
if os.path.exists(f'reports/edge_report_{fecha}.json'):
    _run(['python3', 'scripts/meta_signal_scorer.py'], 'PASO 3b — Meta-Señal Convergencia (Nodo-98)')
```
Corre DESPUÉS de edge_calculator (PASO 3), ANTES de trader (PASO 4).

**Estado:** IMPLEMENTAR en run_daily.py al crear el scorer

---

## 4. CONEXIONES OCULTAS — nuevas oportunidades identificadas por Fable

### C1 — Triple Convergencia: STRONG + drift live + rival COLD (Nodo-43 × Nodo-97)

**Hallazgo:** `graphify path Nodo-97 → Nodo-43-PELT-Cold-Rival` → **SIN CAMINO**. Esta conexión no existe en ningún spec.

**La oportunidad:** Si el modelo pre-partido dice STRONG (favorito en hot) Y el rival está en estado COLD (Nodo-43, PELT), Y la cuota empieza a bajar en vivo → **triple confirmación temporal**:
1. STRONG pre-partido = modelo dice edge real
2. Rival COLD = estado actual del rival confirma debilidad
3. drift≥15% live = mercado también está descubriendo el edge

Esta señal es más potente que el trigger básico de D97-02 (drift + edge_live solos).

**Decisión D99-09:** Añadir campo opcional `rival_markov_cold` en el output del live monitor:
```json
{
  "rival_markov_cold": true,
  "triple_convergencia": true,
  "alpha_tier": "MAXIMO"
}
```
Cuando `triple_convergencia=True` → stake sugerido puede usar el multiplicador completo (sin shrink extra).

**Estado:** DOCUMENTAR — no bloquea implementación base, es upgrade post-gate

---

### C2 — H88-01 Rival Value ya tiene sistema dedicado: riesgo de doble apuesta

**Hallazgo:** `graphify path Nodo-98 → Nodo-68` → 2 hops via RFI. El Rival Value en meta_signal_score (Nodo-98) es la MISMA señal que H88-01 (n=3, 41.25x). Pero H88-01 ya tiene su propio sistema de apuesta: `rival_value_betslip.py` con micro-Kelly dedicado (shrink=5.7%, cap=0.5%).

**Riesgo:** Si Nodo-98 activa Rival Value como señal del score Y H88-01 también hace su apuesta independiente, el sistema apuesta **dos veces** sobre el mismo evento con diferentes sistemas.

**Decisión D99-10:** Protocolo de coordinación:
- Si `score_rival_value >= 1` en Nodo-98 → **delegar a rival_value_betslip.py** (H88-01 ya maneja esto)
- Nodo-98 reporta la señal pero **no genera stake independiente** para Rival Value
- Nodo-98 dice: "Rival Value detectado → ver rival_value_betslip.py output"

**Estado:** DOCUMENTAR en Nodo-98 D98-10 (ampliar), no requiere código nuevo

---

### C3 — n8n como arquitectura primaria del live monitor (no cron "opcional")

**Hallazgo:** Nodo-97 §3.2 clasifica n8n como "opcional". Pero el workflow n8n de Nodo-73 ya tiene:
- Cron cada intervalo configurable
- Retry logic
- Bridge HTTP :8765 activo (systemd)
- Telegram bot integrado

Agregar live monitor como nodo adicional en el MISMO workflow n8n elimina la necesidad de cron paralelo. Mantener dos crons (n8n + sistema aparte) duplica la complejidad operacional.

**Decisión D99-11:** n8n es la arquitectura PRIMARIA para el live monitor. El cron independiente es el fallback (si n8n cae). Misma lógica que close_snapshot_trigger.py (fallback de n8n en Nodo-73).

**Estado:** ACTUALIZAR Nodo-97 §3.2 — cambiar "opcional" por "PRIMARIO (fallback: cron)"

---

### C4 — CLV live vs CLV pre-partido: métricas que no deben mezclarse

**Hallazgo:** D98-06 pide "CLV como métrica primaria en dashboard". Pero si picks live y pre-partido comparten el mismo pool de CLV, la métrica pierde significado.

Los picks live tienen un sesgo fundamental: el trader ya sabe que el mercado se mueve a su favor ANTES de apostar. El CLV live debería ser sistemáticamente más alto que pre-partido. Si se mezclan:
- CLV promedio sube por los picks live (efecto selección)
- El sistema parece mejor de lo que es en pre-partido
- No se puede diagnosticar si el alpha pre-partido se está erosionando

**Decisión D99-12:** Dashboard debe mostrar dos KPIs separados:
```
CLV Pre-partido: +X.X% (n=YY picks)
CLV Live:        +X.X% (n=ZZ picks)  [cuando Nodo-97 esté activo]
```

**Estado:** IMPLEMENTAR cuando se construya el dashboard de Nodo-98

---

## 5. DECISIONES CONSOLIDADAS (D99-01 → D99-12)

| ID | Decisión | Nodo afectado | Estado |
|---|---|---|---|
| D99-01 | Adapter KambiLiveClient intercambiable + fallback pre-game | Nodo-97 §3.1 | PENDIENTE |
| D99-02 | shadow_book pick_type='live' + CLV live separado | Nodo-97 D97-13 | PENDIENTE |
| D99-03 | Separar meta_signal_score en score_directo + score_rival_value | Nodo-98 §4 + §8 | CORREGIR SPEC |
| D99-04 | Pre-registrar H97-01 (drift≥15% + edge_live>5%, n_stop=20) | preregistered_hypotheses.json | IMPLEMENTAR |
| D99-05 | Corregir test #5 ventana: [-30min, +45min] asimétrica | Nodo-97 §7 test #5 | CORREGIR SPEC |
| D99-06 | Live combo respeta Combo Governor (D97-14) | Nodo-97 D97-14 | PENDIENTE |
| D99-07 | ELO dominance referencia explícita a D91-XX | Nodo-98 §4 | DOCUMENTAR |
| D99-08 | meta_signal_scorer.py = PASO 3b en run_daily.py | run_daily.py | IMPLEMENTAR |
| D99-09 | Triple Convergencia: rival_cold + STRONG + drift | Nodo-97 §5 nuevo | DOCUMENTAR |
| D99-10 | Rival Value en N98 delega a rival_value_betslip.py (no doble apuesta) | Nodo-98 D98-10 | DOCUMENTAR |
| D99-11 | n8n PRIMARIO para live monitor (cron = fallback) | Nodo-97 §3.2 | ACTUALIZAR |
| D99-12 | Dashboard: CLV pre-partido vs CLV live separados | Dashboard Nodo-98 | IMPLEMENTAR |

---

## 6. TESTS REQUERIDOS (REGLA-T53)

No hay tests nuevos en este nodo de auditoría — los tests se implementarán en Nodo-97 y Nodo-98 después de aplicar las correcciones de este nodo. Este nodo es META (auditoría + decisiones), no código.

**Gate de implementación:** Los items D99-03, D99-04, D99-05 deben estar COMPLETOS antes de escribir tests de Nodo-97/98.

---

## 7. OUTPUT EVIDENCIA

Este nodo genera 3 artefactos:

1. **Este archivo** `.spec/01_Nodos/Nodo-99-Auditoria-Fable-N97-N98.md` — decisiones pre-implementación
2. **H97-01 en** `validation/preregistered_hypotheses.json` — hipótesis pre-registrada (D99-04)
3. **Specs corregidos:** Nodo-97 §7 test #5 + Nodo-98 §4 con separación score_directo/score_rival_value

---

## 8. PRECONDICIONES PARA IMPLEMENTACIÓN NODO-97

- [ ] D99-01: Kambi LIVE endpoint confirmado via DevTools
- [ ] D99-02: Protocolo shadow_book live definido
- [ ] D99-04: H97-01 pre-registrada ← **puede hacerse hoy**
- [ ] D99-05: Test #5 ventana corregida ← **puede hacerse hoy**
- [ ] D99-06: D97-14 (Combo Governor) añadido en Nodo-97 ← **puede hacerse hoy**
- [ ] D99-11: n8n como primario documentado ← **puede hacerse hoy**

## 9. PRECONDICIONES PARA IMPLEMENTACIÓN NODO-98

- [ ] D99-03: Separar score_directo / score_rival_value en spec ← **puede hacerse hoy**
- [ ] D99-07: ELO dominance referencia explícita ← **puede hacerse hoy**
- [ ] D99-08: PASO 3b asignado en run_daily ← **hacer al implementar scorer**
- [ ] D99-10: Protocolo Rival Value / H88-01 documentado ← **puede hacerse hoy**

---

> **Conexión oculta más valiosa:** Triple Convergencia (D99-09) — STRONG pre-partido + rival COLD + drift live.
> Graphify confirmó que no existe edge en el grafo entre estos tres nodos. Es el alpha más puro
> del pipeline porque tiene triple confirmación temporal. Sin documentación previa en ningún spec.

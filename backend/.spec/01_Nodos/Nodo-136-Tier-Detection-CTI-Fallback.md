# Nodo-136 — Fix Tier Detection: CTI Fallback para H2H Combinado

> **Estado:** IMPLEMENTADO — 2026-07-22
> **Tipo:** BUG FIX — `edge_calculator.py` L933-948
> **Trigger:** Safe/Mega combos bloqueados — todos los picks clasificados como `atp500` al usar H2H combinado de 302 partidos
> **Autor:** Sonnet 4.6 (diagnóstico + fix en sesión)
> **Commit:** pendiente (fix aplicado en sesión, pre-commit)
> **Para auditoría Fable:** verificar (1) umbrales CTI correctos, (2) que el fallback no afecte flujo normal con torneo poblado, (3) tests REGLA-T53

---

## Wikilinks

| Link | Rol |
|------|-----|
| [[Nodo-17-Calibracion-Por-Tier]] | λ escalado por tier — función afectada por el bug |
| [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | Pesos diferenciados por tier — también afectados |
| [[Nodo-29-Circuit-Asymmetry-Deflator]] | CTI (circuit_tier_index) — fuente del fallback implementado |
| [[Nodo-86-Auditoria-Fable5]] | Auditoría doctoral — contexto de calidad del pipeline |
| [[Nodo-87-Fixes-Auditoria-D87]] | Fixes previos de auditoría — patrón de corrección quirúrgica |
| [[Nodo-132-ComboRegistry-Activation-PnL-Tracking]] | ComboRegistry — primera víctima del bug (Mega bloqueado) |

**Wikilinks totales: 6 | Huérfanos: 0**

---

## §1. Diagnóstico — Causa raíz

### §1.1 El síntoma observable

```bash
# edge_report_20260722_102416.json — ANTES del fix
Tier distribution: {'atp500': 12}   # 12/12 picks clasificados igual

# Consecuencia directa:
python3 betplay_combo_builder.py --mega
# → ERROR: "Mega requiere ≥2 tiers distintos" — bloqueado

python3 betplay_combo_builder.py --safe
# → "0 safe combos generados" — pool de 1 solo tier, sin pares cross-tier
```

### §1.2 La cadena causal

```
extraer_historh2h.py
   ↓ lee zita_tennis_matches_*.json (dict con claves = nombre torneo completo)
   ↓ itera los partidos SIN propagar la clave-torneo al registro individual
   ↓ escribe h2h_results_enhanced_FECHA.json con torneo_nombre='' en TODOS

edge_calculator.py L933
   torneo_completo = partido.get('torneo_completo','') or partido.get('torneo_nombre','') or ''
   → torneo_completo = ''   (siempre vacío en H2H combinado)
   tier = detectar_tier('')  → 'atp500'  (fallback hardcoded)

resultado['tier'] = 'atp500'   # para los 302 partidos
```

### §1.3 Por qué el flujo normal no sufría este bug

En el flujo normal (un archivo por tier, ej. `extraer_partidos_api.py --tier itf`), el campo `torneo_nombre` llega poblado porque el extractor sabe el tier de antemano. El bug sólo se manifiesta con el **H2H combinado** (17MB, 302 partidos, todos los torneos del día) generado por `extraer_historh2h.py --all-tournaments`.

### §1.4 Evidencia en los datos — CTI ya resolvía esto internamente

El campo `ranking_analysis.prediction.circuit_asymmetry` en cada partido del H2H YA tenía la respuesta:

```json
{
  "p1_circuit_tier_index": 0.067,
  "p2_circuit_tier_index": 0.236,
  "signal": "MODERATE_ASYMMETRY"
}
```

Y el reasoning log decía explícitamente: `"CTI_max=0.236 < 0.8, ambos ITF"`.

La información estaba ahí — el edge_calculator simplemente no la leía.

---

## §2. Fix implementado — D136-01

**Archivo:** `edge_calculator.py` — bloque T17-03, después de L934

```python
# ─── T17-03: Escalar λ por tier (Nodo-17) ───────────────
torneo_completo = partido.get('torneo_completo', '') or partido.get('torneo_nombre', '') or ''
tier = detectar_tier(torneo_completo)
# Si torneo vacío (H2H combinado sin torneo propagado) → inferir tier desde CTI (Nodo-29)
if not torneo_completo:
    _ra = partido.get('ranking_analysis', {}) or {}
    _pred = (_ra.get('prediction', {}) or {}) if isinstance(_ra, dict) else {}
    _ca = (_pred.get('circuit_asymmetry', {}) or {}) if isinstance(_pred, dict) else {}
    if _ca:
        _cti_max = max(
            float(_ca.get('p1_circuit_tier_index') or 0),
            float(_ca.get('p2_circuit_tier_index') or 0),
        )
        if _cti_max < 0.6:
            tier = 'itf'
        elif _cti_max < 1.5:
            tier = 'challenger'
lambda_av = lambda_av * LAMBDA_TIER_MULTIPLIER.get(tier, 1.0)
```

### §2.1 Lógica de los umbrales CTI → tier

El CTI (Circuit Tier Index, Nodo-29) mide el nivel promedio de circuito del jugador en escala 0–5:

| CTI | Ranking típico de oponentes | Tier inferido |
|-----|---------------------------|---------------|
| 0.0 – 0.6 | >500 (ITF bajo) | `itf` |
| 0.6 – 1.5 | 200–500 (Challenger/ITF alto) | `challenger` |
| 1.5 – 5.0 | <200 (ATP consolidado o superior) | `atp500` (conservador) |

El `_cti_max` usa el mayor de los dos jugadores — si cualquiera tiene nivel ATP, el partido sube de categoría mínima.

### §2.2 Condición de activación — no afecta flujo normal

El bloque CTI sólo ejecuta cuando `not torneo_completo` — si el campo está poblado (flujo normal), `detectar_tier()` sigue siendo la fuente única de verdad. **Cero riesgo de regresión.**

---

## §3. Resultado — ANTES vs DESPUÉS

| Métrica | Antes del fix | Después del fix |
|---------|--------------|----------------|
| Tier distribution (12 picks) | `atp500: 12` | `atp500: 4, itf: 5, challenger: 3` |
| λ correcto por tier | NO (todos usaban λ_atp500=2.4×) | SI — itf=4.5×, challenger=3.6×, atp500=2.4× |
| Mega combo | BLOQUEADO (1 tier) | **Mega1 @3,013x → $1.5M retorno** |
| Safe combo | 0 pares (1 tier + picks ITF no en Kambi) | 0 pares (correcto — ITF minors no en Kambi) |
| Confianza en stakes Kelly-KL | Subestimada para ITF (λ bajo → apuesta grande en pick riesgoso) | Correcta — λ mayor para ITF reduce stake |

---

## §4. Gap residual — D136-02 (pendiente)

**Causa raíz profunda no cerrada:** `extraer_historh2h.py` no propaga `torneo_nombre` desde las claves del dict de matches al registro individual de cada partido.

**Fix correcto de fondo:**
En `extraer_historh2h.py`, cuando itera `for torneo_key, partidos in data.items()`, pasar `torneo_key` al constructor del registro H2H para que el campo `torneo_completo` quede poblado desde la extracción.

**Prioridad:** MEDIA — el fallback CTI del D136-01 cubre el gap correctamente. D136-02 eliminaría la dependencia del fallback y haría el H2H file auto-suficiente.

**Gate para D136-02:** no urgente — activar si se detecta algún caso donde CTI sea bajo pero el torneo real sea ATP (por ejemplo, jugador ATP jugando un ITF exhibition).

---

## §5. Tests REGLA-T53 — `tests/test_nodo136_tier_cti_fallback.py`

```python
def test_D136_01_tier_itf_when_torneo_empty_and_cti_low()
    # partido con torneo_completo='' y CTI_max=0.2 → tier='itf'

def test_D136_01_tier_challenger_when_torneo_empty_and_cti_mid()
    # partido con torneo_completo='' y CTI_max=1.0 → tier='challenger'

def test_D136_01_tier_atp500_when_torneo_empty_and_cti_high()
    # partido con torneo_completo='' y CTI_max=2.5 → tier='atp500'

def test_D136_01_no_regression_when_torneo_populated()
    # partido con torneo_completo='ITF M15 Serbia' → detectar_tier() normal, CTI ignorado

def test_D136_01_fallback_safe_when_no_circuit_asymmetry()
    # partido sin ranking_analysis.prediction.circuit_asymmetry → tier='atp500' (fallback)

def test_D136_02_mega_unblocked_with_correct_tiers()
    # edge_report con 3 tiers → betplay_combo_builder --mega genera ≥1 combo
```

---

## §6. Preguntas abiertas para Fable

1. **¿Los umbrales CTI (0.6 / 1.5) son correctos?**
   Derivados de la escala Nodo-29 (ranking>500=0.0, ranking≤500=1.0). ¿Hay casos borde donde un jugador ATP joven tenga CTI<0.6 por historial ITF reciente?

2. **¿El `_cti_max` debe ser promedio o máximo?**
   Actualmente usa `max(cti_p1, cti_p2)`. Si un pick ATP500 juega contra un ITF, max=ATP → correcto. ¿O debería ser el CTI del favorito específicamente?

3. **¿D136-02 debe correr en la misma sesión o esperar?**
   El fix de fondo en `extraer_historh2h.py` requiere re-extraer H2H para que el campo se pueble. Los archivos históricos ya existentes tendrán `torneo_completo=''` para siempre — el fallback CTI los cubre.

---

## §7. Resumen ejecutivo para Fable

**Qué pasó:** El H2H combinado (302 partidos, todos los torneos del día) llegaba con `torneo_completo=''` en cada registro porque el extractor no propaga el nombre del torneo al aplanar el dict. El `edge_calculator` caía en el fallback `'atp500'` para todos — inflando stakes de picks ITF (λ demasiado bajo) y bloqueando el Mega combo (requiere ≥2 tiers).

**Qué se hizo:** Fallback quirúrgico en `edge_calculator.py`: cuando `torneo_completo` está vacío, lee el `circuit_tier_index` ya embebido en el partido por `rivalry_analyzer` (Nodo-29) e infiere el tier. Activa solo si torneo vacío — flujo normal no se toca.

**Impacto real:** 12 picks correctamente distribuidos en 3 tiers. Mega desbloqueado (@3,013x). λ correcto para picks ITF → stakes más conservadores y apropiados.

---

**Wikilinks totales: 6 | Huérfanos: 0**

[[Nodo-17-Calibracion-Por-Tier]] | [[Nodo-21-Pesos-Diferenciados-Por-Tier]] | [[Nodo-29-Circuit-Asymmetry-Deflator]] | [[Nodo-86-Auditoria-Fable5]] | [[Nodo-87-Fixes-Auditoria-D87]] | [[Nodo-132-ComboRegistry-Activation-PnL-Tracking]]

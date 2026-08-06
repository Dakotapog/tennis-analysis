# Nodo-151 — Live Edge Gates (D151-01/02/03)

**Estado:** COMPLETO  
**Fecha:** 2026-07-28  
**Módulo principal:** `live_desk.py`

---

## Contexto / Root Cause

El monitoreo live del 2026-07-28 produjo **5 señales recomendadas** con resultado **1/5 (20% hit rate)**.

Root cause estructural: el sistema usaba `p_model` (modelo pre-partido histórico, "Modelo A") como fuente de edge para decisiones live, mientras `p_condicional` (modelo live Gaussiano, "Modelo B") mostraba edge NEGATIVO en las 5 señales — sin poder de veto.

### Diagnóstico por señal

| Señal | p_cond | cuota | p_impl | edge_live | Fallo |
|-------|--------|-------|--------|-----------|-------|
| Pinnington Jones UNDER 21.5 | 0.412 | 2.12 | 0.472 | −0.06 | edge_live negativo |
| Ege Sik OVER 17.5 | 0.500 | 1.97 | 0.508 | −0.008 | edge_live negativo |
| Zarazua OVER 19.5 | 0.252 | 1.58 | 0.633 | −0.38 | zona DOMINANTE vs OVER@19.5 contradictoria |
| Charlton UNDER 21.5 | 0.412 | 1.93 | 0.518 | −0.106 | edge_live negativo |
| Uemura UNDER 21.5 | 0.412 | 2.00 | 0.500 | −0.088 | edge_live negativo |

Adicionalmente, **todas las señales ITF_VIVO tenían `score_str=null`**: el sistema tenía el conteo de games pero no el marcador real (4:3 vs 6:1). El bookmaker actualiza cuotas live con el marcador real → desventaja informacional.

Caso confirmado en vivo: **Dart vs Cross** mostraba OVER 20.5 ALTA en dashboard mientras el score real era 6:2, 5:1 → UNDER certeza matemática. El sistema no tenía ese marcador.

---

## Gaps

- **D151-01:** Sin gate sobre edge_live (`p_condicional - p_implied`) — Modelo B no tenía poder de veto.
- **D151-02:** Sin gate sobre `score_str=null` con `games_played > 3` — competencia en desventaja informacional.
- **D151-03:** Sin gate sobre contradicción zona-dirección severa (zona predice A, apostamos B, p_cond bajo).

---

## Implementación

### Funciones puras (REGLA-T53)

Ubicación: `live_desk.py`, antes de `_fire_itf_live_games_combo` (~L3605).

```python
def _edge_live_gate(certeza, cuota_live, umbral=0.05) -> bool:
    """D151-01: True → EXCLUIR. edge_live = p_condicional - 1/cuota_live < umbral."""

def _score_null_gate(score_data, gp_min=3) -> bool:
    """D151-02: True → EXCLUIR. score_str=null AND games_played > gp_min."""

def _zona_direccion_gate(zona, direccion, linea, certeza, p_umbral=0.40) -> bool:
    """D151-03: True → EXCLUIR. zona contradice dirección AND p_condicional < p_umbral."""
```

### Integración en `alta_itf` loop

Después del gate D150-06b (`cuota_envenenada`), antes de `alta_itf.append(_s06)`.

Orden de evaluación: D151-02 → D151-01 → D151-03 (del más barato al más costoso).

---

## Tests (REGLA-T53)

Archivo: `tests/test_nodo151_live_edge_gates.py` — 6 tests, 6/6 PASS.

| Test | Gate | Caso |
|------|------|------|
| test_151_01 | D151-01 | edge_live negativo → excluye |
| test_151_02 | D151-01 | edge_live positivo → pasa |
| test_151_03 | D151-02 | score_null gp>3 → excluye |
| test_151_04 | D151-02 | score_null gp≤3 → pasa |
| test_151_05 | D151-03 | zona DOMINANTE + OVER@19.5 + p<0.40 → excluye |
| test_151_06 | D151-03 | zona DOMINANTE + OVER@17.5 + p≥0.40 → pasa |

---

## Hipótesis pre-registrada

**H151-01:** Gates D151-01/02/03 reducen pérdidas live games ITF: hit rate ≥ 40% en señales que pasan los 3 gates vs 20% sin gates. n_stop=20. Estado: ACUMULANDO.

---

## Wikilinks

[[Nodo-150-Live-Games-Risk-Intelligence]] | [[Nodo-147-Live-Score-Games-Convergencia]] | [[Nodo-133-Games-Live-Convergencia]]

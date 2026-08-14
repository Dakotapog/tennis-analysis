# Nodo-165 — Bonus de Certeza D147 en convergencia_score ITF (D165-01)

> Fecha: 2026-08-03
> Precede: [[Nodo-164]] (gates D150 tier-agnóstico), [[Nodo-151]] (gates edge_live/score_null/zona_direccion), [[Nodo-150]] (cuota_envenenada + set1-tiebreak), [[Nodo-147]] (certeza condicional D147), [[Nodo-142]] (D142-02 convergencia_score original)

## 1. Disparador

Sesión de auditoría en vivo sobre `tennis-live-desk` (systemd, puerto 7780, dinero real) con dos entregables solicitados por el usuario:

1. **BUG-01** (cosmético, ya implementado): fallback `linea_actual`/`cuota_actual` en el render X3 de `live_desk.py` — resuelto sin tocar `itf_live_signals`/`alta_signals`.
2. **INVESTIGACIÓN-01** (solo análisis, sin código): ¿por qué señales ITF con certeza D147 fuerte (p_condicional 0.94 y 0.962, alerta ALTA) se quedan en `convergencia_score=1-2` cuando el gate de disparo real (`live_desk.py:4500`, `alta_itf_raw`) exige `>=3`?

Hallazgo de la investigación: `_convergencia_score_itf()` (D142-02, `live_desk.py:2898-2930`) es un modelo heurístico anterior a D147, con 4 componentes (gap, cuota_live>=2.00, markov COLD/HOT, ranking_gap>300) — nunca actualizado para leer la certeza D147, que se calcula DESPUÉS, en un loop separado (`live_desk.py:4451-4474`, D147 enrichment). Los dos casos reales auditados:

| Partido | gap | cuota_live | markov | ranking_gap | score D142-02 | certeza D147 |
|---|---|---|---|---|---|---|
| Zheng vs Kecmanovic | +2 (fuerte) | 1.96 (<2.00) | None (sin pick en edge_report) | — | **2** | p_condicional=0.94, alerta=ALTA |
| Kovacevic vs Borges | +2 (fuerte) | 1.67 (<2.00) | None | — | **2** | p_condicional=0.962, alerta=ALTA, games_set1=13 |

Ambas señales pierden el componente markov (jugadores ITF/qualy sin pick individual en `edge_report`) y el componente cuota (cuota_live justo debajo del umbral 2.00) — quedan atascadas en score=2 aunque D147, con datos de marcador real, ya confirme la dirección con probabilidad >90%.

El usuario aprobó explícitamente la opción (b) recomendada: **reformular `convergencia_score` para incorporar la certeza D147**, en vez de (a) bajar el umbral `>=3` (que hubiera dejado pasar señales sin ninguna confirmación adicional real).

## 2. Decisión de diseño

**D165-01** — función pura nueva `_convergencia_certeza_bonus(score_actual, certeza)` (`live_desk.py`, junto a `_convergencia_score_itf`):

- Bonus **+1** cuando `certeza["alerta_nivel"] in ("ALTA", "CERTEZA")` **y** `certeza["p_condicional"] >= 0.85`.
- **Aditivo, nunca compensatorio**: no sustituye markov ni cuota faltantes, solo suma cuando D147 aporta una confirmación real independiente (marcador en vivo).
- **Cap en 5** — mismo máximo original de `_convergencia_score_itf()`, para no romper el label hardcodeado `f'{_conv_sc}/5'` en el render X3 (`live_desk.py:1233`). No se tocó el render.
- Doble condición (`alerta_nivel` Y `p_condicional`) es deliberadamente redundante: `_calcular_certeza_condicional()` ya garantiza que `alerta_nivel=="ALTA"` implica `p_condicional>=0.90` internamente, pero el guard explícito evita que un caller con datos inconsistentes dispare el bonus por accidente.

**Punto de inserción**: dentro del loop de enriquecimiento D147 (`live_desk.py:4451-4474`), inmediatamente después de calcular `_itf147["certeza"]` — es el primer punto del pipeline donde la certeza existe para esa señal (el loop principal que llama `_convergencia_score_itf()`, línea 4362, corre ANTES y se cierra antes de este loop). El bonus muta `s["convergencia_score"]`, recalcula `s["confianza"]` (para no dejar la etiqueta inconsistente con el score final) y anexa el detalle a `s["convergencia_breakdown"]` para trazabilidad. Logging `[ITF_LIVE_D165-01]` sigue el mismo patrón que `[ITF_LIVE_D150-07]`.

## 3. Qué NO cambia (alcance explícito, instrucción del usuario)

- **`build_games_combos_live()`** (`betplay_combo_builder.py`) — no tocado.
- **`_fire_itf_live_games_combo()`** — no tocado.
- **Gates D150/D151** (`cuota_envenenada`, set1-tiebreak, `_edge_live_gate`, `_score_null_gate`, `_zona_direccion_gate`, `live_desk.py:4499-4556`) — no tocados. Estos gates operan sobre campos crudos (`cuota_envenenada`, `games_set1`, `certeza`, `zona`) **independientes** de `convergencia_score` — el bonus solo decide qué señales ENTRAN al pool `alta_itf_raw` (`score>=3`); una vez dentro, los 5 gates existentes siguen filtrando exactamente igual. Verificado con test dedicado (test_165_08): el caso Kovacevic/Borges sube a score=3 con el bonus, pero `games_set1=13` sigue excluyéndolo vía el gate de tiebreak, que lee el campo crudo, no el score.
- **`alta_signals`** (tiers no-ITF) — el bonus se aplica solo al loop `itf_live_signals`. La investigación y la aprobación del usuario fueron específicas a la vía ITF (`_fire_itf_live_games_combo`, umbral `convergencia_score>=3`); el disparo D133 para `alta_signals` usa un criterio distinto (`en_vivo_count>=2`, sin gate de score por pierna) donde este bonus no aplica.

## 4. Hipótesis pre-registrada

**H165-01** (`validation/preregistered_hypotheses.json`) — señales ITF que alcanzan el gate de disparo GRACIAS al bonus D165-01 (es decir, habrían quedado en score<3 sin él) tienen hit rate >= breakeven de su cuota_live, medido vía shadow_book. Ver detalle en `preregistered_hypotheses.json`.

## 5. Verificación

- 8 tests REGLA-T53 nuevos (`tests/test_nodo165_convergencia_certeza_bonus.py`) — invocan `_convergencia_certeza_bonus()` real, cubren: bonus con ALTA, bonus con CERTEZA, no-bonus con MOD, guard doble-condición, certeza=None, cap en 5, recomputación de confianza, y el caso real Kovacevic/Borges confirmando que el gate de tiebreak downstream sigue excluyendo independientemente del bonus. 8/8 PASS.
- `python -c "import ast; ast.parse(open('live_desk.py').read())"` → OK.
- Suite completa `pytest tests/ --no-cov -q` (excluyendo el error de colección preexistente de `test_nodo155_hcuc_convergence.py`, no relacionado) — ver resultado en el reporte de sesión.

## 6. Lección reusable

Cuando dos modelos de confianza para el mismo tipo de señal se calculan en pasos distintos del pipeline (heurística D142-02 pre-D147 vs modelo estadístico D147 post-marcador-real), el punto de integración correcto no siempre es el sitio de la llamada original — hay que rastrear el orden de ejecución real. Aquí la certeza D147 solo existe DESPUÉS de que el loop de `convergencia_score` ya cerró, forzando un ajuste post-hoc en vez de un parámetro nuevo en la función original. Ver [[Nodo-153]] (mismo patrón: dato de marcador real llega en un loop de refresco separado del loop de cálculo principal).

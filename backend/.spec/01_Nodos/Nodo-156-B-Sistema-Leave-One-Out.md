# Nodo-156-B — Sistema Leave-One-Out: Combos Escalables 5-7 Piernas

> Implemented 2026-07-31. Feature Request: ampliar combos a 5-7 piernas siendo estratégico con anclas de cuota alta + exclusión de 1 pierna sin que todos los combos se pierdan.

## Visión

**Root cause:** Combos tradicionales N piernas = toda-o-nada (1 pierna perdida = todas pierden). Con 71.9% acierto, probabilidad de 6 piernas correctas = 0.719^6 ≈ 13%, insostenible.

**Solución:** Leave-One-Out system bet — N piernas → N combos de (N-1) cada uno.
- **0 piernas perdidas:** Todos los N combos ganan.
- **Exactamente 1 pierna perdida:** Gana 1 combo (el que la excluyó).
- **≥2 piernas perdidas:** Pierden todos los combos.

**Ventaja:** CON 71.9% acierto, P(≤1 pierna pierde en 6) = 1 - P(2+ pierden) ≈ 1 - C(6,2)×(0.281)²×(0.719)⁴ ≈ 1 - 0.108 ≈ 89.2% → gana ≥1 combo casi siempre.

## Arquitectura

### D156-B-01: `build_system_combos()` en `betplay_combo_builder.py`

**Firma:**
```python
def build_system_combos(
    stake_total=3500, 
    n_piernas=6, 
    ancla_cuota_min=1.65, 
    min_p_modelo=0.55
) -> tuple[List[Dict], Dict]:
```

**Flujo:**
1. Carga `edge_report_*.json` más reciente (convención existente).
2. Filtra candidatos:
   - `cuota >= 1.50` (REGLA-HF-1)
   - `kambi_disponible != False` (Nodo-140 D140-02)
   - No es coinflip sin H2H: `not _es_coinflip_sin_h2h(p_modelo, n_h2h)` (gate existente)
3. Selecciona anclaje y fillers:
   - **Anclaje (1-2):** Top by cuota, preferir ≥ `ancla_cuota_min`, fallback a top global.
   - **Fillers:** Restante ranked by `p_modelo` desc, sin hard threshold exclusion.
   - Total `picks = (anclajes + fillers)[:n_piernas]`.
4. Resuelve picks via Kambi: `fetch_kambi_outcomes()` + `find_outcome()` (convención D140-01/D135-01).
5. Calcula `stake_per_combo = max(50, round(stake_total/n_piernas/50)*50)` (Kelly shrink half, cap por $50 mínimo).
6. Genera combos: `itertools.combinations(resolved, n_piernas-1)` → exactamente `n_piernas` combos, cada uno excluyendo 1 leg.
7. Output: `(system_combos, metadata)` — cada combo dict:
   ```python
   {
       "combo_idx": int,
       "excluye": str,  # nombre jugador excluido
       "piernas": [{"jugador": str, "cuota": float, "tier": str}, ...],
       "legs": [{"legIdx": int, "p_nombre": str, ...}, ...],
       "outcome_ids": [int, ...],
       "url": str,  # Betplay coupon
       "stake": float,
       "cuota_combo": float,
       "retorno": float,
       "p_todas": float,
   }
   ```

**Bugs corregidos durante implementación:**
- D156-B-01a (bug encontrado 2026-07-31 en smoke-test): Anchor classification usaba `is_ancla = cuota >= ancla_cuota_min` como hard filter → casi todo se clasificaba como ancla (porque edge_report hoy: cuotas 1.7–3.6, todas ≥ 1.65), dejando ~0 fillers. Fix: removida hard classification de candidate collection, filtrado solo al seleccionar (anclajes = top-by-cuota ≥ `ancla_cuota_min` OR fallback top global, fillers = restante ranked by p_modelo, no threshold hard exclusion).

### D156-B-02: CLI Args

```python
parser.add_argument("--sistema", action="store_true",
                    help="Sistema Leave-One-Out: N piernas → N combos de N-1")
parser.add_argument("--sistema-piernas", type=int, default=6,
                    help="Piernas totales (5-7)")
parser.add_argument("--sistema-stake", type=float, default=3500,
                    help="Stake TOTAL repartido → stake_per_combo = total/n_piernas")
parser.add_argument("--sistema-ancla-min", type=float, default=1.65,
                    help="Cuota mínima para calificar como ancla")
```

### D156-B-03: Dispatch (2 vías)

1. **Combinado `--live`** (D156-B-03a): Tras GAMES block, antes de `return` del combined flow.
2. **Standalone** (D156-B-03b): Nueva sección "MODO SISTEMA STANDALONE" tras GAMES STANDALONE, antes de EVALUAR_GAMES STANDALONE, siguiendo patrón MEGA/SAFE/GAMES:
   ```
   build_system_combos() → error+exit si vacío → display → dry-run early return →
   _generar_bat_sistema() → _enviar_sistema_telegram()
   ```

### D156-B-04: Output

- **`Sistema1.bat`...`SistemaN.bat`** en `DESKTOP_WIN` — ejecutar en Chrome.
- **`sistema1.html`...`sistemaN.html`** en `COMBOS_DIR` — rendered coupon HTML.
- **Telegram alert** (opcional `--telegram`): Summary + propiedades leave-one-out + total stake.
- **ComboRegistry logging**: Cada combo registrado con `strategy='SISTEMA'`.

## Evidencia Real — Smoke Test 2026-07-31

**Command:** `python3 betplay_combo_builder.py --sistema --dry-run`

**Pool (6 picks calificados):**
```
[ANCLA ] Kalieva E.                @3.45  [atp500]  p_modelo=0.57
[ANCLA ] Michelsen A.              @2.85  [atp500]  p_modelo=0.52
[filler] Hara Friend J.D.          @2.28  [challenger]  p_modelo=0.56
[filler] Thomson M.                @1.74  [itf]  p_modelo=0.50
[filler] Stevens A.                @1.96  [challenger]  p_modelo=0.53
[filler] Bouzige M.                @2.48  [itf]  p_modelo=0.51
```

**6 Combos generados (stake $600 c/u, retornos $32,975 – $65,382):**

| Sistema | Excluye | Cuota | Retorno |
|---------|---------|-------|---------|
| 1 | Bouzige M. | @76.45 | $45,873 |
| 2 | Stevens A. | @96.74 | $58,043 |
| 3 | Thomson M. | @108.97 | $65,382 |
| 4 | Hara Friend J.D. | @83.16 | $49,897 |
| 5 | Michelsen A. | @66.53 | $39,917 |
| 6 | Kalieva E. | @54.96 | $32,975 |

**INVERSIÓN TOTAL: $3,600 (stake_total=3500 + rounding)**

## Hipótesis Pre-registradas

**H156-B-01** (pre-registrada en `validation/preregistered_hypotheses.json`):
```json
{
  "id": "H156-B-01",
  "nombre": "Sistema Leave-One-Out >= 80% hit combinado (n_stop=15)",
  "descripción": "Leave-One-Out system N piernas (N=6, n=6 combos x $600) con ancla+filler balanceado alcanza ≥80% hit rate combinado (≥1 combo gana). Baseline: p_acierto_individual=0.719, P(<=1 pierna falla en 6) ≈ 89.2%.",
  "threshold": 0.80,
  "n_stop": 15,
  "direccion": ">=",
  "metricas": ["hit_combinado", "retorno_promedio", "p_modelo_pool"],
  "test_file": "tests/test_nodo156b_sistema_leave_one_out.py",
  "estado": "PENDIENTE_ACUMULACION"
}
```

## Tests (REGLA-T53)

**`tests/test_nodo156b_sistema_leave_one_out.py`** — 8 tests:
1. `test_156b_01_build_system_combos_basic` — generates N combos from N picks.
2. `test_156b_02_exclude_pattern` — cada combo excluye exactamente 1 leg.
3. `test_156b_03_cuota_combo_math` — cuota_combo = producto piernas (verificar float precision).
4. `test_156b_04_anchor_filler_selection` — fillers ranked by p_modelo, anclajes top-by-cuota.
5. `test_156b_05_kambi_resolution` — outcome_ids resueltos correctamente.
6. `test_156b_06_min_pool_gate` — retorna `[], {}` si pool < n_piernas.
7. `test_156b_07_coinflip_filter` — excluye picks coinflip-sin-h2h.
8. `test_156b_08_stake_calculation` — stake_per_combo math, rounding a $50.

## Relaciones [[Wikilinks]]

- **[[Nodo-23]]** (betplay_combo_builder base) — sede de `build_system_combos()`.
- **[[Nodo-100]]** (Taxonomía 12 estrategias) — SISTEMA es 13ava estrategia, leave-one-out family.
- **[[Nodo-140]]** (Kambi Gate) — D140-02 `is_player_available()` gate reusado en D156-B-01.
- **[[Nodo-135]]** (EvalGames Live API Fix) — `_extract_games_cuota_live()` NOT used en SISTEMA (singles only).
- **[[Nodo-139]]** (Kambi-First Combo Builder) — `fetch_kambi_outcomes()` + `find_outcome()` reusados.
- **[[Nodo-88]]** (Rival Value H88-01) — SISTEMA no compite con Rival Value (diferentes universos: SISTEMA = edge_report, Rival Value = edge_fav ≤ -15%).
- **[[Nodo-148]]** (Trader Plan Vacío) — SISTEMA usa `edge_report_*.json` directo, no trader_plan (similar a MEGA D139-04).
- **[[Nodo-143]]** (Match Ledger Torneo Metadata) — `tier` field reusado en display.

**Nodos huérfanos (sin incoming links):** ninguno nuevo — todo relación bidireccional.

## Próximos pasos (Out of scope Nodo-156-B)

- D156-B-05: `run_daily.py` opcional `--sistema` flag (corre PASO 4.8, después de MEGA standalone).
- D156-B-06: `shadow_book.py` segmento SISTEMA (acumula H156-B-01 hits).
- D156-B-07: Dashboard panel "Sistema" en `live_desk.py` mostrando expected return si gana ≥1 combo.

---

## Tests Relacionados

- Test suite REGLA-T53: 8 tests nuevos en `test_nodo156b_sistema_leave_one_out.py`.
- Regresión: MEGA/SAFE/GAMES smoke tests aún pasan (no modificadas).
- Cobertura `betplay_combo_builder.py`: +120 líneas (5 funciones nuevas + CLI args).

**Status:** ✅ D156-B-01/02/03/04 IMPLEMENTADOS. Tests pendientes D156-B-05 sesión futura.

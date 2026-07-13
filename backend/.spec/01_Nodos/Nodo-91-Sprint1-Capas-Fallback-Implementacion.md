# Nodo-91 — Sprint 1: Capas de Fallback + Nombres + Pipeline Nocturno (spec implementable)

> **Wikilinks:** [[Nodo-90-Auditoria-Fable-Nodo89]] | [[Nodo-89-Sistema-Inteligencia-Integral]] | [[Nodo-87-Fixes-Auditoria-D87]]
> **Fecha:** 2026-07-12 | **Autor:** Fable 5 | **Implementa:** Sonnet (WSL)
> **Objetivo:** eliminar el "0 apuestas sin explicación" con guardas intactas. 6 cambios, 4 archivos tocados, ~12 tests nuevos.
> **Regla de oro:** baseline `pytest tests/ --no-cov -q` → 1822 passed ANTES de empezar. Después de cada cambio: syntax check + suite. NUNCA tocar la puerta económica base (`edge_calculator.py:479-483`) ni HF-1/HF-5.

---

## S1-A — `capa2_candidate` en edge_calculator.py

**Dónde:** inmediatamente después del bloque D68 Rival Value Flip (`edge_calculator.py:1102-1106`, antes del `return resultado` en `:1108`).
**Qué:** función pura module-level + llamada. Variables ya en scope: `p_modelo`, `cuota_fav`, `_n_h2h_v`, `resultado`.

```python
def evaluar_capa2(resultado: dict, p_modelo: float, cuota_fav: float, n_h2h: int) -> bool:
    """D90-04/H89-01: candidato a CAPA 2 (Model Confidence).
    Re-evalúa los gates de seguridad que la cadena principal solo corre
    cuando apostar=True (ver Nodo-90 §1 C-2). OBSERVACIONAL + stake 25% en trader.
    """
    if resultado.get('apostar'):
        return False                                   # ya es CAPA 1
    if resultado.get('status') == PICK_STATUS_NO_DATA: # NO_DATA re-check
        return False
    if resultado.get('phantom_data'):                  # Nodo-72 re-check
        return False
    if resultado.get('markov_favorito') == 'HOT' and (resultado.get('bbi') or 0.5) < 0.50:
        return False                                   # HOT_sin_BBI re-check
    return (p_modelo >= 0.60 and 1.50 <= cuota_fav <= 2.80 and n_h2h >= 1)
```
En `calcular_edge_completo`: `resultado['capa2_candidate'] = evaluar_capa2(resultado, p_modelo, cuota_fav, _n_h2h_v)`.
Notas: T33-01 queda auto-satisfecho (n_h2h>=1 ∧ p>=0.60). N28F2 se relaja deliberadamente — es la definición de la capa. HF-1 satisfecho por `cuota >= 1.50`. El campo fluye solo al pick_snapshot (el shadow book serializa `resultado` completo — verificado en `sb_2026-07-12.jsonl`).

Además: `elo_dominance_axis` (D90-10, observacional): en el mismo punto,
```python
_elo_f, _elo_r = resultado.get('elo_favorito'), resultado.get('elo_rival')
_rk_f, _rk_r = resultado.get('ranking_favorito'), resultado.get('ranking_rival')
resultado['elo_dominance_axis'] = bool(
    _elo_f and _elo_r and _rk_f and _rk_r
    and (_elo_f - _elo_r) > 50 and _rk_f > _rk_r   # mejor ELO pero peor ranking
)
```
NO modifica N28F2 en Sprint 1 — solo serializa (H89-02 acumula).

## S1-B — CAPA 2 en trader_ev_tenis.py (pool fallback)

**Dónde:** `main()`, tras el bloque existente `if not senales_raw and not args.all_picks:` (`trader_ev_tenis.py:1029-1031`) — ese bloque ya hace un fallback apostar+watchlist; extenderlo:
1. Si tras el filtro de tier (`:1023-1027`) `senales_raw` (pool CAPA 1) queda vacío → construir `pool_capa2 = [p for p in watchlist + sin_edge if p.get('capa2_candidate')]` (función pura nueva `_pool_capa2(watchlist, sin_edge) -> list` a nivel de módulo, para REGLA-T53).
2. Cada pick capa2: `p['_capa'] = 2` y **stake = 0.25 ×** el sizing normal (aplicar el factor donde se calcula el stake individual — el mismo sitio del fix D87-03 `:498`; el `min(p_prior, p_modelo)` de D87-05 ya aplica).
3. Tope de capa: suma de stakes CAPA 2 ≤ $5,000 (challenger) / $2,000 (itf) / $10,000 (GS/ATP) — constante `CAPA2_CAP_POR_TIER` arriba del archivo.
4. Banner obligatorio en output y en el JSON del plan: `"modo": "CONFIDENCE_MODE_CAPA2"` + por cada pick bloqueado de capa 1, imprimir su `motivo_reclasificacion` (Zero-Null explicativo, Nodo-90 §2).
5. **HF-5 intacto:** si KGR<0 con el pool capa2 → NO DESPLEGAR (el chequeo existente aplica sin cambios). En ese caso el output lista motivo por pick + alternativa (games).

## S1-C — CAPA 3 (games) + orquestación en run_daily.py

1. Nuevo arg `--fase {completa,noche,manana}` default `completa` (D90-07):
   - `noche`: PASOS 0→3.6 (líneas `:247-270`) forzando `--tomorrow` en PASO 1; NO corre PASO 4+.
   - `manana`: salta PASOS 0-3.6; corre PASO 4→governor + settle (usa los reports de anoche).
   - `completa`: comportamiento actual.
2. Tras el loop PASO 4 (`:281-303`): si TODOS los tiers dieron 0 individuales y 0 cobertura **y** no hubo plan CAPA 2 → ejecutar `_run(['python3', 'betplay_combo_builder.py', '--games'], 'CAPA 3 — Games fallback')` (path standalone ya existe: `betplay_combo_builder.py:3089-3107`, REGLA-G6 $2,000 se respeta sola) y marcar el brief `GAMES MODE`.
3. Si también CAPA 3 da 0 → el brief imprime el estado de cada capa y de cada candidato (nunca lista vacía muda).
4. Cron WSL (documentar en el nodo al implementar; no automatizar sin n8n):
   `0 21 * * * ... run_daily.py --tomorrow --fase noche` | `0 7 * * * ... run_daily.py --fase manana`

## S1-D — Staleness 4h en betplay_combo_builder.py

**Dónde:** `build_live_combos()` `:2003-2009`. Extraer a función pura:
```python
PLAN_MAX_AGE_H = 4  # D89-01: trader_plan más viejo → regenerar, no combinar

def _planes_frescos(paths: list, max_age_h: float = PLAN_MAX_AGE_H) -> list:
    cutoff = datetime.now() - timedelta(hours=max_age_h)
    return [p for p in paths
            if datetime.strptime(p.stem.replace('trader_plan_', ''), '%Y%m%d_%H%M%S') > cutoff]
```
(usar el parse de timestamp que el archivo ya emplea para el cutoff de 24h). CLI `--max-plan-age-h` (default 4). Si el filtro deja 0 planes: **mensaje accionable** con el comando exacto de regeneración por tier — NO caer silenciosamente al fallback legacy del edge_report.

## S1-E — Fix multi-inicial en analysis/ranking_manager.py (D90-02)

**Dónde:** `get_player_info()`, paso 3 (`:455-472`). Generalizar: recolectar TODOS los tokens finales de 1 carácter como iniciales (no solo el último):
```python
# Paso 3 generalizado: "hsu y h" → apellido=['hsu'], iniciales=['y','h']
_parts = list(name_parts)
_iniciales = []
while _parts and len(_parts[-1]) == 1:
    _iniciales.insert(0, _parts.pop())
if _parts and _iniciales:
    for search_dict in search_dicts:
        for ranked_name, data in search_dict.items():
            ranked_parts = ranked_name.split()
            if all(p in ranked_parts for p in _parts):
                restantes = [p for p in ranked_parts if p not in _parts]
                if all(any(r.startswith(ini) for r in restantes) for ini in _iniciales):
                    ...return data  # mismo logging que el paso 3 actual
```
Mantener el paso 3 actual como caso particular (o sustituirlo — este lo cubre). NO tocar pasos 4/5 (BUG-34-2 Fix B se conserva). PROHIBIDO añadir matching fonético (Nodo-90 §1 C-4, conflicto Nodo-72).

## S1-F — Pre-registros + segmento shadow

1. Añadir a `validation/preregistered_hypotheses.json` (con decisión del usuario, Constitución 8):
```json
{"id": "H89-01", "nombre": "CAPA2 Model-Confidence", "prediccion": "picks capa2_candidate (p>=0.60, cuota 1.50-2.80, n_h2h>=1, sin HOT_sin_BBI) logran hit% > 52.4% (breakeven cuota media ~1.91)", "n_stop": 30, "kill_switch": "hit% < 45% con n>=20 → CAPA2_ENABLED=False", "estado": "ACUMULANDO", "fecha_registro": "2026-07-12"}
{"id": "H89-02", "nombre": "ELO_DOMINANCE axis", "prediccion": "picks con elo_dominance_axis=True y n_axes_active=1 logran hit% > 50% (vs 29% BBI-sola histórico)", "n_stop": 30, "estado": "ACUMULANDO", "fecha_registro": "2026-07-12"}
```
2. `shadow_book.py` `report()`/`report_dict()`: nuevo segmento `CAPA2` (`pick_snapshot.capa2_candidate=True`) y `ELO_DOM` (`elo_dominance_axis=True`) — patrón idéntico a los segmentos RFI/ANCHOR existentes (`_segment_metrics`, `:1127` zona).

---

## §6. Tests REGLA-T53 — `tests/test_nodo91_sprint1.py`

Cada test invoca la función real; nunca reimplementa la fórmula.

| Test | Invoca | Asegura |
|---|---|---|
| `test_capa2_rechaza_no_data` | `evaluar_capa2({status:NO_DATA,...}, .65, 2.0, 3)` | False |
| `test_capa2_rechaza_phantom` | `evaluar_capa2({phantom_data:True,...})` | False |
| `test_capa2_rechaza_hot_sin_bbi` | markov HOT + bbi 0.3 | False |
| `test_capa2_rechaza_p_bajo` | p=0.58 | False |
| `test_capa2_rechaza_sin_h2h` | n_h2h=0 | False (T33 cubierto) |
| `test_capa2_rechaza_cuota_hf1` | cuota 1.40 y 3.10 | False ambos |
| `test_capa2_acepta_caso_valido` | p=0.62, cuota 2.0, n_h2h=2, limpio | True |
| `test_capa2_no_duplica_capa1` | apostar=True | False |
| `test_pool_capa2_filtra_flag` | `_pool_capa2(watchlist, sin_edge)` | solo capa2_candidate |
| `test_planes_frescos_corta_4h` | `_planes_frescos([p_5h, p_1h])` | solo p_1h |
| `test_multi_iniciales_hsu` | `get_player_info('Hsu Y. H.')` con dict sintético `{'hsu yu hsiou': {...}}` inyectado en `atp_players` | retorna la entrada |
| `test_multi_iniciales_burruchaga` | `'Burruchaga R. A.'` vs `'burruchaga roman andres'` | retorna la entrada |
| `test_una_inicial_sigue_funcionando` | `'Cerundolo F.'` vs `'cerundolo francisco'` | regresión paso 3 actual |
| `test_elo_dominance_serializado` | resultado con elo 1560/1464 + rk 1188/2043 → wait: favorito con MEJOR ranking no activa; usar rk_f>rk_r | flag correcto en ambos sentidos |

## §7. Orden de implementación (sin romper nada)

1. Baseline pytest (1822) → 2. **S1-E** (aislado, riesgo cero para el resto) + sus 3 tests → 3. **S1-A** (solo añade campos) + tests capa2/elo → 4. **S1-D** + test → 5. **S1-B** (el único con riesgo de sizing — verificar con `--bankroll 20000 --torneo-tipo challenger` en dry-run contra el edge_report del 12) → 6. **S1-C** (orquestación) → 7. **S1-F** (pre-registro requiere OK del usuario) → 8. Suite completa + `graphify update .` + commit por paso con IDs D90/D89 en el mensaje.

**Criterio de éxito Sprint 1 (verificable con evidencia de ejecución):** re-correr el pipeline con los inputs del 2026-07-12 → el sistema produce ≥1 salida accionable (CAPA 2: Arseneault-tipo si cumple gate, o CAPA 3: combo games @2.66x documentado en Nodo-89 §7) con banner de capa, y cada pick bloqueado muestra su motivo. 0 tests rotos.

"""
validation/hypothesis_ledger.py — Nodo-174 D174-03

Fuente única de PREDICADOS por hipótesis pre-registrada (validation/preregistered_
hypotheses.json). Antes de este módulo, la misma fórmula por hipótesis vivía
duplicada (a veces con pequeñas divergencias) en shadow_book.py::report() y
::report_dict() como closures privados no importables — ninguna otra parte del
pipeline podía reutilizarlos, así que n_actual/hits quedaban escritos a mano en
sitios como live_desk.py (D174-06) en vez de derivarse del shadow book real.

Arquitectura Strangler Fig (no se toca shadow_book.py en este nodo): este módulo
es la nueva fuente canónica. shadow_book.py sigue con sus propias copias por ahora
— migrar report()/report_dict() para importar desde aquí es deuda futura, no
D174-03. Cada predicado cita su origen exacto (archivo:línea) auditado el
2026-08-06 vía agente Explore.

REGLA-T53: cada fórmula abajo es una transcripción verbatim de la lógica real
encontrada en shadow_book.py — no una reinvención. Donde la lógica de origen no
existía (H62-01) o donde el lambda real es `lambda r: True` porque el filtro ya
se aplicaba upstream en una list-comprehension (H124-01/02/03), se documenta
explícitamente la diferencia entre "extraído" y "reconstruido equivalente".

Invariante (test_174_01): set(PREDICADOS) == set(hipótesis declaradas en el JSON).
Hipótesis sin ruta de medición real (D174-04, 18 ids tras cerrar 4 rutas nuevas
el 2026-08-07: H77-01, H96-01, H113-01, H165-01) quedan con predicado
`_sin_ruta` — retorna False siempre, nunca inventa datos (n=0 es honesto).
Hipótesis de comparación de 2 grupos (H52-04/05/06, H77-02) no son reducibles a
un predicado de pertenencia+hit/miss único — también retornan False con nota,
distinto motivo que las 22 sin ruta (para esas SÍ existe mecanismo real, solo que
no es del tipo n/hits/roi que este ledger sabe contar).

D174-05 (2026-08-05/06) pre-registró las 4 huérfanas que faltaban —
H147-01, H150-01, H151-01, H152-01 — ahora tienen entrada en
preregistered_hypotheses.json y predicado propio abajo. H150-01 sigue el
patrón de comparación de 2 grupos (return False con nota) porque las señales
que el gate D150-01 excluye nunca llegan a shadow_book — no hay cohorte
"sin filtro" reconstruible.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, Optional


def _snap(r: dict) -> dict:
    return r.get('pick_snapshot', {}) or {}


# ─── Predicados con lógica real extraída de shadow_book.py ──────────────────

def _h52_01(r: dict) -> bool:  # report() L1480-1482 == report_dict() L1962-1964 (_has_was_d)
    s = _snap(r)
    return s.get('edge', 0) >= 0.10 and s.get('cuota_favorito', 0) >= 2.0 and s.get('markov_favorito') == 'HOT'


def _h52_02(r: dict) -> bool:  # L1490-1492 == L1966-1968 (_is_n_h2h_0_itf_d)
    s = _snap(r)
    return s.get('n_h2h', 1) == 0 and s.get('tier') == 'itf'


def _h52_03(r: dict) -> bool:  # L1484-1488,1512-1513 == L1970-1972 (_is_struct_or_low_d)
    s = _snap(r)
    return s.get('alignment_flag') == 'STRUCTURAL_ALPHA' or s.get('confidence_flag') == 'LOW'


def _h52_07(r: dict) -> bool:  # L1494-1497 == L1974-1977 (_is_qualifying_low_p_d)
    s = _snap(r)
    return bool(r.get('es_qualifying', False)) and 0.52 <= s.get('p_modelo', 0) < 0.55


def _h52_08(r: dict) -> bool:  # L1504-1507 == L1979-1982 (_is_zona_2_25_d)
    s = _snap(r)
    return 2.00 <= s.get('cuota_favorito', 0) < 2.50


def _h54_01(r: dict) -> bool:  # report_dict() L1984-1987 (_is_var_flattened_d) — ausente en report() texto
    return bool(r.get('trader_deploy', {}).get('var_flattened')) and bool(_snap(r).get('apostar'))


def _h62_01(r: dict) -> bool:
    # No existe predicado propio en report()/report_dict() (ninguna de las dos funciones
    # menciona H62-01). Reconstruido desde el único mecanismo real de almacenamiento:
    # update_alpha_flags() (shadow_book.py L736-778) escribe combo_flags.alpha_promoted.
    return bool(r.get('combo_flags', {}).get('alpha_promoted', False))


def _h76_01(r: dict) -> bool:  # L1341-1343 (segmento observacional, sin _append_hypothesis)
    return bool(_snap(r).get('rfi_ultra', False))


def _h88_01(r: dict) -> bool:  # L1350-1353 == L2071 — métrica invertida, ver INVERTIDAS
    return bool(_snap(r).get('rival_value_flag', False))


def _h89_01(r: dict) -> bool:  # L1451-1453 == L1937
    return bool(_snap(r).get('capa2_candidate', False))


def _h89_02(r: dict) -> bool:  # L1455-1457 == L1938
    return bool(_snap(r).get('elo_dominance_axis', False))


def _h98_01(r: dict) -> bool:  # L1382-1387,1520-1525 — ausente en report_dict()
    return (_snap(r).get('score_directo') or 0) >= 3


def _h100_01(r: dict) -> bool:  # L1389-1391,1527-1531 — ausente en report_dict()
    return _snap(r).get('pick_type') == 'live'


def _h107_01(r: dict) -> bool:
    # L1424-1428: población base MOTOR. El split cuota<=2.5 / >2.5 (L1431-1434) es
    # sub-segmentación de display, no forma parte del predicado de la hipótesis en sí.
    s = _snap(r)
    return s.get('tipo') == 'APOSTAR' and s.get('pick_type') != 'live'


# ─── Predicados nuevos D174-04 — rutas de medición reales que no existían ────
# Transcripción verbatim de umbrales_congelados en preregistered_hypotheses.json
# (auditado 2026-08-06/07), no reinvención. Cada uno cita el campo real y su
# origen en el pipeline (no shadow_book.py — estas 4 nunca tuvieron predicado
# en report()/report_dict(), son rutas nuevas, no migraciones).

def _h96_01(r: dict) -> bool:
    # umbral: delta_return_umbral=-0.10, n_retornos_min=2 (Nodo-96 D96-01).
    # irp_fav solo existe en pick_snapshot cuando build_irp_profiles.py generó
    # perfil (MIN_RETORNOS=2 hard-coded ahí, scripts/build_irp_profiles.py:34) —
    # su sola presencia ya garantiza n_retornos>=2, no hace falta re-chequear.
    irp = _snap(r).get('irp_fav') or {}
    return irp.get('delta_return', 0) <= -0.10


def _h113_01(r: dict) -> bool:
    # umbral literal: "weather_flag=RAIN_RISK en picks outdoor (clay/grass) con
    # apostar=True"; exclusiones: "status=NO_DATA, phantom_data=true, superficie
    # indoor o UNKNOWN". clay/grass ya excluye indoor/UNKNOWN por construcción.
    s = _snap(r)
    return (
        s.get('weather_flag') == 'RAIN_RISK'
        and s.get('superficie') in ('clay', 'grass')
        and bool(s.get('apostar'))
        and s.get('status') != 'NO_DATA'
        and not bool(s.get('phantom_data'))
    )


def _h165_01(r: dict) -> bool:
    # umbral: bonus decisivo == score_sin_bonus<3 Y score_con_bonus>=3 ("señales
    # que ya llegaban a 3 sin el bonus no son parte de esta hipótesis").
    # _convergencia_certeza_bonus() (live_desk.py:3122) suma +1 fijo (cap=5) y
    # solo escribe el marcador 'D147_certeza=' en convergencia_breakdown cuando
    # aplicado=True (score cambió). Como el bonus es +1 exacto sin capping salvo
    # en el borde score_actual=4->5 (donde aplicado sería False de todos modos
    # si ya fuera 5), la única forma de que el score post-bonus (serializado)
    # cruce el umbral 3 DESDE abajo es score_final==3 exactamente
    # (score_actual=2 -> 3). score_final>=4 con marcador implica score_actual
    # ya era >=3 antes del bonus — no decisivo, excluido a propósito.
    s = _snap(r)
    return s.get('convergencia_score') == 3 and 'D147_certeza=' in (s.get('convergencia_breakdown') or '')


# deltas_validos congelados en preregistered_hypotheses.json (H77-01, REGLA #8:
# no modificar sin decisión de graduación) — transcripción literal, no inventado.
_H77_01_DELTAS_VALIDOS = frozenset({
    'gs_vs_itf', 'gs_vs_challenger', 'atp1000_vs_itf', 'atp1000_vs_challenger',
    'atp500_vs_itf', 'atp500_vs_challenger', 'challenger_vs_itf',
})


def _h77_01(r: dict) -> bool:
    # umbral: tier_campeon_superior_a_tier_actual=True, deltas_validos (lista
    # exacta arriba), campeon_days_ago_max=30, rango_conf=[0.55,0.70]. Campos
    # tier_mismatch/tier_mismatch_delta/campeon_tier_nivel ya se serializaban
    # (D65-03, edge_calculator.py:1385-1389, comentado explícitamente
    # "Gate H77-01"); campeon_days_ago NO se copiaba a resultado — gap cerrado
    # en D174-04 (edge_calculator.py:1390, 1 línea, mismo bloque). rango_conf
    # interpretado como p_modelo del pick (único campo de confianza numérica
    # ya serializado en resultado, usado también por H52-07).
    s = _snap(r)
    dias = s.get('campeon_days_ago')
    return (
        bool(s.get('tier_mismatch'))
        and s.get('tier_mismatch_delta') in _H77_01_DELTAS_VALIDOS
        and dias is not None and dias <= 30
        and 0.55 <= s.get('p_modelo', 0) <= 0.70
    )


def _h124_01(r: dict) -> bool:
    # shadow_book.py L1632-1636 pasa literalmente `lambda r: True` a _append_hypothesis
    # porque el filtro real ya se aplicó construyendo la lista de entrada
    # (_eval_recs + _egames_recs, L1609-1612). Reconstruido aquí como predicado
    # autocontenido equivalente — no es una transcripción de un solo lambda existente.
    return _snap(r).get('pick_type') in ('evaluar', 'evaluar_games')


def _h124_02(r: dict) -> bool:  # mismo patrón que H124-01 (L1637-1643) + filtro markov
    s = _snap(r)
    return s.get('pick_type') in ('evaluar', 'evaluar_games') and (s.get('markov_wr_rec_fav') or 0) >= 0.70


def _h124_03(r: dict) -> bool:  # L1611-1612,1644-1648
    return _snap(r).get('pick_type') == 'evaluar_games'


# ─── Hipótesis de comparación de 2 grupos — no reducibles a un predicado simple ──
# contar_hipotesis() las reporta con n=0 en vez de inventar una fórmula que no existe.

def _h52_04(r: dict) -> bool:  # A/B Brier ON/OFF vía hypothesis_tracker.get_nodo46_case_count()
    return False  # no es un registro shadow_book — conteo manual en el JSON (n_casos_atribuibles)


def _h52_05(r: dict) -> bool:  # STEAM_IN vs DRIFT_OUT hit% — comparación de 2 grupos
    return False  # ver shadow_book.py::_compute_line_signal() L583-616


def _h52_06(r: dict) -> bool:  # Spearman ranking preservado en sesiones BLIND
    return False  # no implementado en ningún archivo del repo


def _h77_02(r: dict) -> bool:  # ANCHOR vs VARIABLE — comparación de 2 grupos sin breakeven único
    return False  # grupos existen (L1308-1339) pero nunca se puntúan como hipótesis única


# ─── 18 hipótesis declaradas sin ruta de medición real (deuda D174-04) ───────────
# Confirmado por auditoría Nodo-174 (agente Explore, 2026-08-06) + verificación
# puntual D174-04 (2026-08-07): cero ocurrencias de estos IDs en
# shadow_book.py::report()/report_dict() NI en ningún call site de log_pick()/
# log_live_pick()/log_games_live_pick(). Retornan False — n=0 honesto, nunca se
# inventa una señal que no existe. Categorías confirmadas por rastreo directo
# de código (no supuesto):
#   NOT_COMPUTED (cero campo en pick_snapshot, cero call site):
#     H77-03 (estado BLOQUEADO/scalp_top no existe), H111-01 (dual-book steam-lag
#     no propaga a pick_snapshot), H120-01/H121-01 (_ledger_status/_cuota_source
#     viven en match_ledger.py, nunca copiados a resultado), H125-01
#     (evaluar_games_bridge.py escribe su propio JSON, jamás llama shadow_book),
#     H139-01/H139-02 (kambi_first flow no llama log_pick), H166-01 (D166-01
#     alta_pregame_raw solo alimenta el flag booleano convergencia_activa en
#     live_desk.py, NUNCA se pasa a log_games_live_pick — confirmado leyendo el
#     único call site real, live_desk.py:4989, que solo recibe alta_itf),
#     H173-02/H173-03 (rfi_layoff_fade / t32_01_habria_bloqueado: 0 ocurrencias).
#   COMPUTED_NOT_SERIALIZED (el dato existe en runtime pero no llega a
#   pick_snapshot, o el criterio congelado no es reducible a un booleano por
#   registro): H97-01 (drift_pct/edge sí viajan en _pick_live de
#   live_edge_monitor.py:~909, pero confidence_gate="STRONG o HOT" no tiene
#   campo — _pick_live carece de confidence_flag/markov_favorito), H160-02
#   (mc_p_condicional SÍ se serializa, pero el umbral congelado describe un
#   criterio temporal — "steam_confirmado precede breakpoint EN_VIVO >=2
#   ciclos" — no reducible a una condición sobre un solo record, mismo motivo
#   que la comparación-de-2-grupos de abajo).
#   MANUAL/PROTEGIDO (n_actual real ya registrado a mano, sin ruta automática
#   — ver GUARD monotónico en actualizar_registro()): H60-01, H60-02, H77-01
#   (ver nota: predicado real SÍ agregado en D174-04 para acumulación futura,
#   pero el n_actual histórico sigue siendo el snapshot manual protegido por el
#   guard), H88-01 (tiene predicado real pero campo rival_value_flag ausente en
#   los records reales existentes), H110-01, H173-01 (COMBO_LEVEL/model-level,
#   no shadow_book — ver docstring del módulo).

def _h147_01(r: dict) -> bool:
    # Nodo-147.md sección 5 (líneas 597-611) / D147-02 _calcular_certeza_condicional.
    # DOMINANTE + p_condicional>=0.70 + games_played>linea/2, SIN certeza_matematica
    # (ese caso ya es un hit garantizado, no la señal probabilística que mide H147-01).
    s = _snap(r)
    cert = s.get('certeza') or {}
    gp = (s.get('score_data') or {}).get('games_played')
    linea = s.get('linea')
    if gp is None or linea is None:
        return False
    return (
        s.get('zona') == 'DOMINANTE'
        and (cert.get('p_condicional') or 0) >= 0.70
        and gp > linea / 2
        and not cert.get('certeza_matematica', False)
    )


def _h150_01(r: dict) -> bool:
    # Nodo-150.md sección 7 (líneas 280-284) — comparación de 2 grupos (filtrado vs
    # no-filtrado por cuota_envenenada), mismo patrón que H52-05/H77-02. Las señales
    # que D150-01 excluye (live_desk.py `continue` en el loop alta_itf) nunca llegan
    # a shadow_book — no hay cohorte "sin filtro" reconstruible retroactivamente.
    return False


def _h151_01(r: dict) -> bool:
    # Nodo-151.md línea 79-81 — gates D151-01/02/03 (edge_live/score_null/zona_dir).
    # Por construcción, todo pick pick_type='games_live' ya sobrevivió los 3 `continue`
    # en el loop alta_itf (live_desk.py líneas ~4883-4933) antes de llegar acá.
    return _snap(r).get('pick_type') == 'games_live'


def _h152_01(r: dict) -> bool:
    # shadow_book.py CHECKLIST SEMANAL (antes literal hardcoded, D174-05 lo mueve al JSON).
    return bool(_snap(r).get('hcuc_convergence', False))


def _h179_01(r: dict) -> bool:  # memoria arquetipos delta>=+0.05 vs delta<=-0.05 — 2 grupos
    return False  # ver mismo patrón que H52-05/H77-02 arriba — comparación de 2 grupos no
    # reducible a un solo booleano de pertenencia; requiere reporte dedicado en shadow_book.py
    # que aún no existe (Nodo-179 D179-02/D179-03 solo consumen memoria, no la reportan por
    # grupos). Campo memoria_delta_vs_global SÍ se serializa (edge_calculator.py::consultar_memoria)
    # — n=0 honesto hasta que exista ese reporte, no se inventa una señal que no existe.


def _h179_02(r: dict) -> bool:  # EDGE_HIGH vs EDGE_MED hit rate, prospectivo desde 2026-08-12 — 2 grupos
    return False  # mismo motivo que _h179_01 — comparación de 2 arquetipos, sin reporte dedicado


def _h181_01(r: dict) -> bool:  # ventana explotable en disparos ACCION — cruce fire_ledger x odds
    return False  # Nodo-181 D181-08: NO reducible a un predicado single-record sobre `r` (un pick
    # liquidado) — necesita cruzar reports/fire_ledger_*.jsonl (tipo VENTANA) con
    # reports/games_odds_history_*.json, dos streams que contar_hipotesis(settled) no ve. La medición
    # real vive en shadow_book.calcular_stats_ventana_h181() (D181-08), "reporte dedicado" igual que
    # D174-07 hace para IRP/Weather — este predicado solo documenta honestamente que n=0 aquí, no que
    # la hipótesis sea inmedible.


def _h181_02(r: dict) -> bool:  # onda P anticipa a certeza (ts_onda_p < ts_certeza) — cruce de 2 ledgers
    return False  # mismo motivo que _h181_01 — requiere fire_ledger x certeza_fired_*.json, no un
    # solo registro de settled. Medido en shadow_book.calcular_stats_ventana_h181() (D181-08).


def _h181_03(r: dict) -> bool:  # quorum 3/3 vs <=2 familias — comparación de 2 grupos, cohorte inexistente
    return False  # Nodo-181 D181-08: por construcción, live_desk._registrar_disparo_ventana_once solo
    # escribe al fire_ledger cuando nivel=="ACCION", que ya exige n_familias>=3 (D181-06/07) — no existe
    # ningún disparo registrado con n_familias<=2 para comparar. Mismo patrón que H150-01/H179-01
    # (comparación de 2 grupos no reducible a un booleano de pertenencia), agravado por la ausencia
    # total de la cohorte de contraste. No se inventa una comparación con un solo lado.


def _h181_04(r: dict) -> bool:  # gate D181-13 descarta filas malas — filas INCOHERENTE nunca se registran
    return False  # Nodo-181 §4 spec: "si la fila no se registra cuando el gate la descarta, H181-04 es
    # inmedible: dilo explícito". core/row_coherence.py::evaluar_coherencia_fila solo sustituye el
    # badge HTML en live_desk.py (~línea 1606) — la fila INCOHERENTE nunca llega a shadow_book, no hay
    # snapshot con cuota+motivo que auditar aquí. Gap real, no una omisión silenciosa; fuera de alcance
    # de D181-08 (pertenece a D181-13, cerrar en un nodo futuro si se decide medir H181-04).


def _sin_ruta(r: dict) -> bool:
    return False


_SIN_RUTA_IDS = (
    "H60-01", "H60-02", "H77-03", "H97-01", "H103-01",
    "H110-01", "H111-01", "H120-01", "H121-01", "H125-01",
    "H132-01", "H139-01", "H139-02", "H160-02", "H166-01",
    "H172-01", "H173-01", "H173-02", "H173-03",
)

PREDICADOS: dict[str, Callable[[dict], bool]] = {
    "H52-01": _h52_01, "H52-02": _h52_02, "H52-03": _h52_03,
    "H52-04": _h52_04, "H52-05": _h52_05, "H52-06": _h52_06,
    "H52-07": _h52_07, "H52-08": _h52_08,
    "H54-01": _h54_01,
    "H62-01": _h62_01,
    "H76-01": _h76_01,
    "H77-01": _h77_01, "H77-02": _h77_02,
    "H88-01": _h88_01,
    "H89-01": _h89_01, "H89-02": _h89_02,
    "H96-01": _h96_01,
    "H98-01": _h98_01,
    "H100-01": _h100_01,
    "H107-01": _h107_01,
    "H113-01": _h113_01,
    "H124-01": _h124_01, "H124-02": _h124_02, "H124-03": _h124_03,
    "H147-01": _h147_01,
    "H150-01": _h150_01,
    "H151-01": _h151_01,
    "H152-01": _h152_01,
    "H165-01": _h165_01,
    "H179-01": _h179_01,
    "H179-02": _h179_02,
    "H181-01": _h181_01,
    "H181-02": _h181_02,
    "H181-03": _h181_03,
    "H181-04": _h181_04,
    **{h_id: _sin_ruta for h_id in _SIN_RUTA_IDS},
}

# H88-01 (RIVAL VALUE) mide lo contrario de todas las demás: éxito = el FAVORITO
# pierde (el rival gana). Confirmado en shadow_book.py::_rival_value_metrics()
# L1698-1739 — hit%_rival cuenta resultado=='LOST' del favorito, ROI usa cuota_rival.
INVERTIDAS = frozenset({"H88-01"})

# p0/p1 congelados. H89-01/H89-02: literal en shadow_book.py (checklist SPRT
# semanal, L1573-1577) — su entrada JSON aún tiene p0/p1=None (fuera de alcance
# D174-05, ver Nodo-174 §D174-05 nota de scope). H147-01/H151-01/H152-01: D174-05
# los pre-registró con p0/p1 congelados directo en preregistered_hypotheses.json
# (fuente de verdad) — H150-01 queda excluido a propósito (comparación de 2
# grupos, sin p0/p1, ver _h150_01). El resto no tiene p0/p1 frozen en ningún
# sitio, así que sprt=None para ellas (REGLA #8: no inventar umbrales no
# congelados).
_SPRT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "H89-01": (0.45, 0.55),
    "H89-02": (0.45, 0.55),
    "H147-01": (0.50, 0.65),
    "H151-01": (0.20, 0.40),
    "H152-01": (0.385, 0.55),
}


def _resultado(record: dict) -> Optional[str]:
    # shadow_book.py::_rival_value_metrics() L1706-1707 y report()/report_dict()
    # (ej. L1154,1166,1861,2043) — el resultado real vive anidado en
    # record['resolucion']['resultado'], NUNCA en record['resultado'] top-level
    # (ese campo no existe; siempre es None). Confirmado 2026-08-06 tras dry-run
    # que devolvía hits=0 para casi todas las hipótesis.
    return record.get('resolucion', {}).get('resultado')


def _es_void_o_pendiente(record: dict) -> bool:
    # mismo filtro non_void que shadow_book.py L1154/1705/2043 — VOID y None
    # (resolución aún sin resultado asignado) se excluyen de n/hits/roi.
    return _resultado(record) in (None, 'VOID')


def _es_hit(record: dict, h_id: str) -> bool:
    gano = _resultado(record) == 'WON'
    return (not gano) if h_id in INVERTIDAS else gano


def _roi_flat_1u(seg: list, h_id: str) -> Optional[float]:
    if not seg:
        return None
    total = 0.0
    for r in seg:
        snap = _snap(r)
        cuota = snap.get('cuota_rival') if h_id in INVERTIDAS else snap.get('cuota_favorito')
        cuota = cuota or 0
        total += (cuota - 1) if _es_hit(r, h_id) else -1
    return total / len(seg)


def contar_hipotesis(settled: list) -> dict:
    """
    Cuenta n/hits/roi/sprt por hipótesis pre-registrada usando PREDICADOS como
    única fuente de pertenencia. Pura — no lee ni escribe archivos.

    Retorna {h_id: {'n': int, 'hits': int, 'roi': float|None, 'sprt': dict|None}}.
    'roi' y 'sprt' son None cuando n==0 o (para sprt) cuando el h_id no tiene
    p0/p1 congelados conocidos (ver _SPRT_THRESHOLDS).
    """
    from validation.hypothesis_tracker import sprt_verdict

    resultado: dict = {}
    for h_id, pred in PREDICADOS.items():
        seg = [r for r in settled if pred(r) and not _es_void_o_pendiente(r)]
        n = len(seg)
        hits = sum(1 for r in seg if _es_hit(r, h_id))
        roi = _roi_flat_1u(seg, h_id)
        sprt = None
        if n > 0 and h_id in _SPRT_THRESHOLDS:
            p0, p1 = _SPRT_THRESHOLDS[h_id]
            sprt = sprt_verdict(n=n, hits=hits, p0=p0, p1=p1)
        resultado[h_id] = {'n': n, 'hits': hits, 'roi': roi, 'sprt': sprt}
    return resultado


def actualizar_registro(path_json: str, conteos: dict, *, dry_run: bool = True) -> dict:
    """
    Escribe n_actual/hits/roi_flat_1u en preregistered_hypotheses.json a partir de
    conteos (output de contar_hipotesis). NUNCA toca umbrales_congelados,
    preregistrado, ni crea hipótesis nuevas (REGLA #8 — los umbrales están
    congelados; esta función solo actualiza observación acumulada).

    dry_run=True (default): calcula y retorna el diff sin escribir a disco.
    dry_run=False: además escribe el archivo si hubo cambios.

    GUARD n_actual monotónico: si el conteo nuevo es MENOR al ya registrado, se
    omite ese h_id sin tocarlo. Hallazgo 2026-08-06: H60-01 (GCS GRADUADA,
    n_actual=54/hits=35 registrado manualmente al graduar) y otras 4 hipótesis
    (H77-03, H88-01, H110-01, H173-01) tienen valores reales cuyo origen NO es
    shadow_book.py records — sin predicado (o, en H88-01, con predicado real
    pero campo ausente en los records existentes) y sin este guard las hubiera
    sobreescrito con n=0, destruyendo el hallazgo GRADUADA. H77-01 recibió
    predicado real en D174-04 (2026-08-07) — su n_actual histórico sigue siendo
    el snapshot manual, protegido igual por este guard hasta que la
    acumulación automática lo supere de forma monotónica. n_actual es
    acumulativo por diseño (picks resueltos nunca desaparecen) — una baja real
    siempre indica que la fuente de este ledger no cubre esa hipótesis, nunca
    que la hipótesis "perdió" observaciones.

    Retorna {h_id: {'antes': {...}, 'despues': {...}}} solo para IDs con cambio real.
    """
    with open(path_json, encoding='utf-8') as f:
        data = json.load(f)
    hyps = data.get('hypotheses', {})

    diff: dict = {}
    for h_id, c in conteos.items():
        if h_id not in hyps:
            continue  # invariante: solo actualiza hipótesis ya declaradas, nunca crea nuevas
        h = hyps[h_id]
        antes = {
            'n_actual': h.get('n_actual', 0),
            'hits': h.get('hits', 0),
            'roi_flat_1u': h.get('roi_flat_1u'),
        }
        if c['n'] < antes['n_actual']:
            continue  # ver GUARD n_actual monotónico en el docstring — nunca reducir

        despues = dict(antes)
        despues['n_actual'] = c['n']
        despues['hits'] = c['hits']
        if c.get('roi') is not None:
            despues['roi_flat_1u'] = round(c['roi'], 4)

        if antes['n_actual'] != despues['n_actual'] or antes['hits'] != despues['hits']:
            diff[h_id] = {'antes': antes, 'despues': despues}

        if not dry_run and h_id in diff:
            h['n_actual'] = despues['n_actual']
            h['hits'] = despues['hits']
            if despues['roi_flat_1u'] is not None:
                h['roi_flat_1u'] = despues['roi_flat_1u']
            # D174-06: sello de frescura — el live desk usa esto para decidir
            # si muestra el n_actual real o "n=?" (ver n_actual_fresco()).
            h['fitted_at'] = datetime.now(timezone.utc).isoformat()

    if not dry_run and diff:
        with open(path_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return diff


def n_actual_fresco(h_id: str, path_json: str, *, max_horas: float = 48.0) -> Optional[dict]:
    """
    D174-06 — lee n_actual/n_stop/hits de una hipótesis desde el JSON, pero
    solo los retorna si el dato es fresco (fitted_at existe y tiene <= max_horas).

    Retorna None cuando: el h_id no existe, no tiene fitted_at (nunca corrió
    PASO 10e para esa hipótesis, o es un valor sembrado a mano sin timestamp),
    o el timestamp es más viejo que max_horas. El caller (live_desk.py) debe
    mostrar "n=?" explícito en ese caso — nunca reusar el último n_actual
    conocido como si fuera actual (ver principio D174-06 en el spec del Nodo-174).

    Retorna {'n_actual': int, 'n_stop': int, 'hits': int, 'fitted_at': str}
    cuando el dato es fresco.
    """
    with open(path_json, encoding='utf-8') as f:
        data = json.load(f)
    h = data.get('hypotheses', {}).get(h_id)
    if not h:
        return None
    if h.get('estado') == 'GRADUADA':
        # Hito terminal congelado a mano al graduar (ej. H60-01 n=54) — no es
        # un "número viejo" desactualizado, es historia inmutable. No aplica
        # el chequeo de frescura.
        return {
            'n_actual': h.get('n_actual', 0),
            'n_stop': h.get('n_stop', 0),
            'hits': h.get('hits', 0),
            'fitted_at': h.get('fitted_at'),
        }
    fitted_at = h.get('fitted_at')
    if not fitted_at:
        return None
    try:
        ts = datetime.fromisoformat(fitted_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    edad_horas = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if edad_horas > max_horas:
        return None
    return {
        'n_actual': h.get('n_actual', 0),
        'n_stop': h.get('n_stop', 0),
        'hits': h.get('hits', 0),
        'fitted_at': fitted_at,
    }

"""
tests/test_nodo174_hypothesis_ledger.py — REGLA-T53 Nodo-174 D174-03

validation/hypothesis_ledger.py centraliza PREDICADOS/contar_hipotesis/
actualizar_registro. Todos los tests invocan las funciones reales del módulo
— nunca hardcodean la fórmula de una hipótesis.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import validation.hypothesis_ledger as hl

_JSON_PATH = Path(__file__).parent.parent / "validation" / "preregistered_hypotheses.json"


def _declaradas() -> set:
    with open(_JSON_PATH, encoding='utf-8') as f:
        return set(json.load(f)['hypotheses'].keys())


# ─── Invariante estructural (test_174_01) ────────────────────────────────────

def test_174_01_predicados_cubren_exactamente_las_declaradas():
    """set(PREDICADOS) == set(hipótesis declaradas) — evita fugas silenciosas A1/A2."""
    assert set(hl.PREDICADOS.keys()) == _declaradas()


def test_174_01b_json_es_valido_y_tiene_54_hipotesis():
    """Guard de regresión del fix D174-03 (H160-02 estaba fuera del dict 'hypotheses')
    + D174-05 (agrega H147-01/H150-01/H151-01/H152-01: 43+4=47)
    + Nodo-172/Nodo-179 (agrega H172-01/H179-01/H179-02: 47+3=50)
    + Nodo-181 D181-08 (agrega H181-01/02/03/04: 50+4=54)."""
    with open(_JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)
    assert set(data.keys()) == {"_meta", "hypotheses"}
    assert len(data['hypotheses']) == 54
    assert "H160-02" in data['hypotheses']


# ─── contar_hipotesis — predicado real (H52-08, zona cuota 2.00-2.50) ────────

def test_contar_hipotesis_h52_08_predicado_real():
    settled = [
        {'pick_snapshot': {'cuota_favorito': 2.20}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'cuota_favorito': 2.40}, 'resolucion': {'resultado': 'LOST'}},
        {'pick_snapshot': {'cuota_favorito': 1.80}, 'resolucion': {'resultado': 'WON'}},  # fuera de zona
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H52-08']['n'] == 2
    assert out['H52-08']['hits'] == 1


def test_contar_hipotesis_roi_flat_1u_signo_correcto():
    settled = [
        {'pick_snapshot': {'cuota_favorito': 2.20}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'cuota_favorito': 2.40}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    # ganancia neta = (2.20-1) - 1 = 0.20, /2 = 0.10
    assert out['H52-08']['roi'] == pytest.approx(0.10)


# ─── H88-01 (RIVAL VALUE) — métrica invertida ────────────────────────────────

def test_h88_01_hit_es_favorito_pierde():
    settled = [
        {'pick_snapshot': {'rival_value_flag': True, 'cuota_rival': 3.0}, 'resolucion': {'resultado': 'LOST'}},  # favorito perdio = hit
        {'pick_snapshot': {'rival_value_flag': True, 'cuota_rival': 2.5}, 'resolucion': {'resultado': 'WON'}},   # favorito gano = miss
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H88-01']['n'] == 2
    assert out['H88-01']['hits'] == 1


# ─── Hipótesis sin ruta (D174-04) — n=0 honesto, nunca inventa señal ─────────

def test_sin_ruta_retorna_n_cero_no_inventa_datos():
    settled = [
        {'pick_snapshot': {'edge': 0.5}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    for h_id in hl._SIN_RUTA_IDS:
        assert out[h_id]['n'] == 0
        assert out[h_id]['roi'] is None


def test_comparacion_2_grupos_retorna_n_cero():
    settled = [{'pick_snapshot': {'edge': 0.5}, 'resolucion': {'resultado': 'WON'}}]
    out = hl.contar_hipotesis(settled)
    for h_id in ("H52-04", "H52-05", "H52-06", "H77-02"):
        assert out[h_id]['n'] == 0


# ─── sprt — solo hipótesis con p0/p1 congelados en shadow_book.py ────────────

def test_sprt_solo_para_h89_01_h89_02():
    settled = [{'pick_snapshot': {'capa2_candidate': True}, 'resolucion': {'resultado': 'WON'}}] * 5
    out = hl.contar_hipotesis(settled)
    assert out['H89-01']['sprt'] is not None
    assert out['H89-01']['sprt']['p0'] == 0.45
    assert out['H89-01']['sprt']['p1'] == 0.55
    # H52-08 no tiene p0/p1 congelado conocido en shadow_book.py -> sprt None
    assert out['H52-08']['sprt'] is None


# ─── actualizar_registro — dry_run por defecto, nunca toca umbrales_congelados ─

def test_actualizar_registro_dry_run_no_escribe(tmp_path):
    src = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    src['hypotheses']['H52-08']['n_actual'] = 0  # guard monotónico: baseline bajo para no chocar con datos reales
    src['hypotheses']['H52-08']['hits'] = 0
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps(src), encoding='utf-8')
    mtime_antes = tmp_json.stat().st_mtime

    conteos = {'H52-08': {'n': 7, 'hits': 3, 'roi': -0.05, 'sprt': None}}
    diff = hl.actualizar_registro(str(tmp_json), conteos, dry_run=True)

    assert 'H52-08' in diff
    assert diff['H52-08']['despues']['n_actual'] == 7
    # archivo no fue tocado
    assert tmp_json.stat().st_mtime == mtime_antes
    reloaded = json.loads(tmp_json.read_text(encoding='utf-8'))
    assert reloaded['hypotheses']['H52-08']['n_actual'] == src['hypotheses']['H52-08'].get('n_actual', 0)


def test_actualizar_registro_dry_run_false_escribe_y_preserva_umbrales(tmp_path):
    src = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    umbrales_antes = src['hypotheses']['H52-08']['umbrales_congelados']
    origen_deuda_antes = src['hypotheses']['H52-08']['origen_deuda']
    src['hypotheses']['H52-08']['n_actual'] = 0  # guard monotónico: baseline bajo para no chocar con datos reales
    src['hypotheses']['H52-08']['hits'] = 0
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps(src), encoding='utf-8')

    conteos = {'H52-08': {'n': 12, 'hits': 5, 'roi': 0.02, 'sprt': None}}
    hl.actualizar_registro(str(tmp_json), conteos, dry_run=False)

    reloaded = json.loads(tmp_json.read_text(encoding='utf-8'))
    h = reloaded['hypotheses']['H52-08']
    assert h['n_actual'] == 12
    assert h['hits'] == 5
    assert h['roi_flat_1u'] == pytest.approx(0.02)
    # REGLA #8: nunca se tocan los umbrales congelados ni el resto de metadata frozen
    assert h['umbrales_congelados'] == umbrales_antes
    assert h['origen_deuda'] == origen_deuda_antes


def test_actualizar_registro_nunca_reduce_n_actual(tmp_path):
    """Guard 2026-08-06: H60-01 (GCS GRADUADA n=54/hits=35, registrado manualmente,
    sin ruta de medición en este ledger) no debe ser sobreescrito con un conteo
    menor -- eso destruiría el hallazgo real. Un conteo nuevo menor al existente
    se omite en vez de escribirse."""
    src = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    tmp_json = tmp_path / "hyp.json"
    src['hypotheses']['H52-08']['n_actual'] = 100
    src['hypotheses']['H52-08']['hits'] = 60
    tmp_json.write_text(json.dumps(src), encoding='utf-8')

    conteos = {'H52-08': {'n': 3, 'hits': 0, 'roi': -1.0, 'sprt': None}}
    diff = hl.actualizar_registro(str(tmp_json), conteos, dry_run=False)

    assert 'H52-08' not in diff
    reloaded = json.loads(tmp_json.read_text(encoding='utf-8'))
    assert reloaded['hypotheses']['H52-08']['n_actual'] == 100
    assert reloaded['hypotheses']['H52-08']['hits'] == 60


# ─── D174-04 — 4 predicados reales nuevos (antes _sin_ruta) ──────────────────

def test_h96_01_delta_return_umbral_real():
    settled = [
        {'pick_snapshot': {'irp_fav': {'delta_return': -0.15}}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'irp_fav': {'delta_return': -0.05}}, 'resolucion': {'resultado': 'WON'}},  # fuera de umbral
        {'pick_snapshot': {}, 'resolucion': {'resultado': 'LOST'}},  # sin irp_fav
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H96-01']['n'] == 1


def test_h113_01_weather_rain_risk_outdoor_apostar():
    settled = [
        {'pick_snapshot': {'weather_flag': 'RAIN_RISK', 'superficie': 'clay', 'apostar': True, 'status': 'OK'}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'weather_flag': 'RAIN_RISK', 'superficie': 'hard', 'apostar': True, 'status': 'OK'}, 'resolucion': {'resultado': 'WON'}},  # indoor/hard excluido
        {'pick_snapshot': {'weather_flag': 'RAIN_RISK', 'superficie': 'grass', 'apostar': True, 'status': 'OK', 'phantom_data': True}, 'resolucion': {'resultado': 'LOST'}},  # phantom excluido
        {'pick_snapshot': {'weather_flag': 'CLEAR', 'superficie': 'clay', 'apostar': True, 'status': 'OK'}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H113-01']['n'] == 1


def test_h165_01_bonus_decisivo_solo_score_exacto_3():
    settled = [
        # score_final==3 con marcador = decisivo (score_actual era 2)
        {'pick_snapshot': {'convergencia_score': 3, 'convergencia_breakdown': 'gap+2;D147_certeza=ALTA(p=0.90)(+1)'}, 'resolucion': {'resultado': 'WON'}},
        # score_final==4 con marcador = YA llegaba a 3 sin el bonus, no decisivo
        {'pick_snapshot': {'convergencia_score': 4, 'convergencia_breakdown': 'gap+3;D147_certeza=ALTA(p=0.90)(+1)'}, 'resolucion': {'resultado': 'WON'}},
        # score==3 sin marcador = no fue el bonus
        {'pick_snapshot': {'convergencia_score': 3, 'convergencia_breakdown': 'gap+3'}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H165-01']['n'] == 1


def test_h77_01_tier_mismatch_delta_valido_dias_y_confianza():
    settled = [
        {'pick_snapshot': {'tier_mismatch': True, 'tier_mismatch_delta': 'atp1000_vs_itf', 'campeon_days_ago': 10, 'p_modelo': 0.60}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'tier_mismatch': True, 'tier_mismatch_delta': 'atp1000_vs_itf', 'campeon_days_ago': 45, 'p_modelo': 0.60}, 'resolucion': {'resultado': 'WON'}},  # dias>30 excluido
        {'pick_snapshot': {'tier_mismatch': True, 'tier_mismatch_delta': 'itf_vs_atp1000', 'campeon_days_ago': 10, 'p_modelo': 0.60}, 'resolucion': {'resultado': 'WON'}},  # delta invalido
        {'pick_snapshot': {'tier_mismatch': False}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H77-01']['n'] == 1


def test_predicados_reales_d174_04_no_estan_en_sin_ruta_ids():
    for h_id in ('H96-01', 'H113-01', 'H165-01', 'H77-01'):
        assert h_id not in hl._SIN_RUTA_IDS


# ─── D174-05 — 4 huérfanas pre-registradas (H147-01/H150-01/H151-01/H152-01) ──

def test_d174_05_las_4_huerfanas_ya_no_estan_en_sin_ruta_ids():
    for h_id in ('H147-01', 'H150-01', 'H151-01', 'H152-01'):
        assert h_id not in hl._SIN_RUTA_IDS


def test_h147_01_dominante_certeza_alta_sin_matematica():
    settled = [
        # cumple: DOMINANTE, p_condicional>=0.70, games_played(14)>linea(22.5)/2=11.25, sin certeza_matematica
        {'pick_snapshot': {'zona': 'DOMINANTE', 'linea': 22.5, 'certeza': {'p_condicional': 0.75, 'certeza_matematica': False}, 'score_data': {'games_played': 14}}, 'resolucion': {'resultado': 'WON'}},
        # certeza_matematica=True -> excluido (ese caso es hit garantizado, no el probabilistico)
        {'pick_snapshot': {'zona': 'DOMINANTE', 'linea': 22.5, 'certeza': {'p_condicional': 0.96, 'certeza_matematica': True}, 'score_data': {'games_played': 20}}, 'resolucion': {'resultado': 'WON'}},
        # p_condicional<0.70 -> excluido
        {'pick_snapshot': {'zona': 'DOMINANTE', 'linea': 22.5, 'certeza': {'p_condicional': 0.60, 'certeza_matematica': False}, 'score_data': {'games_played': 14}}, 'resolucion': {'resultado': 'LOST'}},
        # zona!=DOMINANTE -> excluido
        {'pick_snapshot': {'zona': 'COINFLIP', 'linea': 22.5, 'certeza': {'p_condicional': 0.80, 'certeza_matematica': False}, 'score_data': {'games_played': 14}}, 'resolucion': {'resultado': 'WON'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H147-01']['n'] == 1
    assert out['H147-01']['hits'] == 1
    assert out['H147-01']['sprt']['p0'] == 0.50
    assert out['H147-01']['sprt']['p1'] == 0.65


def test_h150_01_es_comparacion_2_grupos_n_cero():
    """H150-01 no es un predicado de un solo pick — el gate D150-01 excluye las
    señales filtradas antes de llegar a shadow_book, no hay cohorte reconstruible."""
    settled = [
        {'pick_snapshot': {'cuota_envenenada': True}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'cuota_envenenada': False}, 'resolucion': {'resultado': 'LOST'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H150-01']['n'] == 0
    assert out['H150-01']['sprt'] is None


def test_h151_01_pick_type_games_live_ya_implica_gates_pasados():
    settled = [
        {'pick_snapshot': {'pick_type': 'games_live'}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'pick_type': 'games_live'}, 'resolucion': {'resultado': 'LOST'}},
        {'pick_snapshot': {'pick_type': 'individual'}, 'resolucion': {'resultado': 'WON'}},  # otro pick_type -> excluido
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H151-01']['n'] == 2
    assert out['H151-01']['hits'] == 1
    assert out['H151-01']['sprt']['p0'] == 0.20
    assert out['H151-01']['sprt']['p1'] == 0.40


def test_h152_01_hcuc_convergence_flag_real():
    settled = [
        {'pick_snapshot': {'hcuc_convergence': True}, 'resolucion': {'resultado': 'WON'}},
        {'pick_snapshot': {'hcuc_convergence': True}, 'resolucion': {'resultado': 'LOST'}},
        {'pick_snapshot': {'hcuc_convergence': False}, 'resolucion': {'resultado': 'WON'}},
    ]
    out = hl.contar_hipotesis(settled)
    assert out['H152-01']['n'] == 2
    assert out['H152-01']['hits'] == 1
    assert out['H152-01']['sprt']['p0'] == 0.385
    assert out['H152-01']['sprt']['p1'] == 0.55


def test_d174_05_json_umbrales_congelados_no_inventados():
    """Las 3 hipotesis con sprt real tienen p0/p1 en el JSON exactamente iguales
    a los que usa _SPRT_THRESHOLDS -- fuente unica, sin duplicacion divergente."""
    with open(_JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)
    hyps = data['hypotheses']
    for h_id, (p0, p1) in [('H147-01', (0.50, 0.65)), ('H151-01', (0.20, 0.40)), ('H152-01', (0.385, 0.55))]:
        assert hyps[h_id]['p0'] == pytest.approx(p0)
        assert hyps[h_id]['p1'] == pytest.approx(p1)
    # H150-01 explicitamente sin p0/p1 (comparacion de 2 grupos)
    assert 'p0' not in hyps['H150-01'] or hyps['H150-01'].get('p0') is None


def test_shadow_book_checklist_h152_01_lee_p0_p1_del_json():
    """D174-05: shadow_book.py ya no hardcodea (0.385, 0.55) para H152-01 --
    los lee de preregistered_hypotheses.json en tiempo de reporte."""
    import inspect
    import shadow_book
    src = inspect.getsource(shadow_book)
    assert '"H152-01", "HCUC (hard+quality+coinflip+señal especial)", _hcuc_recs, 0.385, 0.55' not in src
    assert '_h152_p0, _h152_p1' in src


def test_actualizar_registro_nunca_crea_hipotesis_nueva(tmp_path):
    src = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps(src), encoding='utf-8')

    conteos = {'H999-99': {'n': 1, 'hits': 1, 'roi': 1.0, 'sprt': None}}
    diff = hl.actualizar_registro(str(tmp_json), conteos, dry_run=False)

    assert diff == {}
    reloaded = json.loads(tmp_json.read_text(encoding='utf-8'))
    assert 'H999-99' not in reloaded['hypotheses']


# ─── D174-06 — el live desk deja de mentir (n_actual real + frescura) ────────

def test_actualizar_registro_escribe_fitted_at_solo_en_cambios_reales(tmp_path):
    """D174-06: actualizar_registro sella fitted_at únicamente en los h_id que
    de verdad cambiaron -- una hipótesis sin cambio no debe ganar un timestamp
    falso de 'recién medida'."""
    src = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    src['hypotheses']['H52-08']['n_actual'] = 0
    src['hypotheses']['H52-08']['hits'] = 0
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps(src), encoding='utf-8')

    conteos = {
        'H52-08': {'n': 9, 'hits': 4, 'roi': 0.01, 'sprt': None},
        'H88-01': {'n': 0, 'hits': 0, 'roi': None, 'sprt': None},  # sin cambio real (n=0 no supera lo existente)
    }
    hl.actualizar_registro(str(tmp_json), conteos, dry_run=False)

    reloaded = json.loads(tmp_json.read_text(encoding='utf-8'))
    assert reloaded['hypotheses']['H52-08'].get('fitted_at') is not None
    assert reloaded['hypotheses']['H88-01'].get('fitted_at') is None


def test_n_actual_fresco_none_sin_fitted_at(tmp_path):
    """Sin fitted_at (nunca corrió PASO 10e para esa hipótesis) -> None, el
    caller debe mostrar n=? -- nunca reusar el n_actual crudo como si fuera vigente."""
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps({'hypotheses': {'H1': {'n_actual': 5, 'n_stop': 20, 'hits': 3}}}), encoding='utf-8')
    assert hl.n_actual_fresco('H1', str(tmp_json)) is None


def test_n_actual_fresco_stale_mas_de_48h(tmp_path):
    from datetime import datetime, timedelta, timezone
    tmp_json = tmp_path / "hyp.json"
    viejo = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    tmp_json.write_text(json.dumps({'hypotheses': {
        'H1': {'n_actual': 5, 'n_stop': 20, 'hits': 3, 'fitted_at': viejo}
    }}), encoding='utf-8')
    assert hl.n_actual_fresco('H1', str(tmp_json)) is None


def test_n_actual_fresco_reciente_retorna_dato_real():
    from datetime import datetime, timezone
    tmp = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    import tempfile, os
    tmp['hypotheses']['H52-08']['fitted_at'] = datetime.now(timezone.utc).isoformat()
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(tmp, f)
        r = hl.n_actual_fresco('H52-08', path)
        assert r is not None
        assert r['n_actual'] == tmp['hypotheses']['H52-08']['n_actual']
    finally:
        os.unlink(path)


def test_n_actual_fresco_graduada_no_requiere_fitted_at():
    """H60-01 (GCS GRADUADA, hito congelado manualmente) es fresca aunque
    fitted_at sea None -- no es un número viejo, es historia inmutable."""
    r = hl.n_actual_fresco('H60-01', str(_JSON_PATH))
    assert r is not None
    assert r['n_actual'] == 54


def test_n_actual_fresco_h_id_inexistente_retorna_none(tmp_path):
    tmp_json = tmp_path / "hyp.json"
    tmp_json.write_text(json.dumps({'hypotheses': {}}), encoding='utf-8')
    assert hl.n_actual_fresco('H999-99', str(tmp_json)) is None


def test_live_desk_ya_no_hardcodea_n_actual_literal():
    """D174-06: live_desk.py:135,253,271,289,303 (numeración pre-D174-06) ya
    no hardcodean n_actual -- usan _n_actual_desk() que consulta el ledger real."""
    import inspect
    import live_desk
    src = inspect.getsource(live_desk)
    assert '"n_actual": 3,\n                "n_stop": 30,' not in src  # RIVAL_VALUE H88-01
    assert '"n_actual": 54,\n                "n_stop": 54,' not in src  # GCS H60-01
    assert '"n_actual": 8,   # semilla jul-14/16' not in src            # FAVORITOS_COMPUESTOS
    assert 'def _n_actual_desk(' in src
    assert 'n_actual_fresco' in src


def test_live_desk_gate_barra_muestra_n_interrogacion_sin_dato_fresco():
    from live_desk import _gate_barra
    assert _gate_barra("?", 30) == "n=?"


def test_live_desk_gate_barra_funciona_normal_con_int():
    from live_desk import _gate_barra
    assert "GRADUADA" in _gate_barra(54, 30)
    assert "/20" in _gate_barra(5, 20)


# ─── D174-07 — segmentar weather_flag e irp en el shadow book ────────────────

def test_shadow_book_d174_07_reusa_contar_hipotesis_no_duplica_predicado():
    """D174-07: shadow_book.py importa contar_hipotesis() de hypothesis_ledger
    (Strangler Fig, D174-04 ya construyó los predicados _h96_01/_h113_01) en
    vez de reimplementar la lógica delta_return/weather_flag inline."""
    import inspect
    import shadow_book
    src = inspect.getsource(shadow_book)
    assert 'from validation.hypothesis_ledger import contar_hipotesis' in src
    assert 'H96-01' in src
    assert 'H113-01' in src
    # no debe reimplementar el umbral -- esa lógica vive solo en hypothesis_ledger
    assert "delta_return', 0) <= -0.10" not in src
    assert "== 'RAIN_RISK'" not in src


def test_shadow_book_d174_07_segmento_irp_con_datos_reales(tmp_path, monkeypatch):
    """Con 1 pick irp_fav.delta_return<=-0.10 settled WON, el reporte debe
    mostrar n=1 hits=1 para H96-01 -- leído del predicado real, no hardcodeado."""
    import shadow_book
    monkeypatch.setattr(shadow_book, 'SHADOW_DIR', str(tmp_path))
    rec = {
        'sb_id': 'AC1', '_type': 'pick',
        'pick_snapshot': {'irp_fav': {'delta_return': -0.15, 'n_retornos': 3}},
        'resolucion': {'resultado': 'WON'},
    }
    (tmp_path / "sb_2026-08-01.jsonl").write_text(json.dumps(rec) + "\n", encoding='utf-8')

    out = shadow_book.report(desde='2026-08-01', hasta='2026-08-01')
    assert '[H96-01]' in out
    assert 'n=1' in out
    assert 'hits=1' in out


def test_shadow_book_d174_07_segmento_weather_sin_datos_muestra_acumulando(tmp_path, monkeypatch):
    """Sin picks con weather_flag=RAIN_RISK, H113-01 debe mostrar n=0 —
    acumulando, nunca inventar un conteo."""
    import shadow_book
    monkeypatch.setattr(shadow_book, 'SHADOW_DIR', str(tmp_path))
    rec = {
        'sb_id': 'AC2', '_type': 'pick',
        'pick_snapshot': {'weather_flag': 'CLEAR', 'superficie': 'clay', 'apostar': True},
        'resolucion': {'resultado': 'WON'},
    }
    (tmp_path / "sb_2026-08-01.jsonl").write_text(json.dumps(rec) + "\n", encoding='utf-8')

    out = shadow_book.report(desde='2026-08-01', hasta='2026-08-01')
    assert '[H113-01] ' in out
    assert 'n=0 — acumulando' in out


def test_shadow_book_d174_07_no_agrega_gate_reporte_solo():
    """Constraint explícito del spec D174-07: 'No añadir gates' -- el segmento
    debe declararse observacional puro, sin condicionar apostar/kelly."""
    import inspect
    import shadow_book
    src = inspect.getsource(shadow_book)
    assert 'No aplica gate de apuesta' in src


# ─── D174-08 — outcome_id real propagado al edge_report (D154-06 real) ───────

def _make_partido_d17408(outcome_id=None, kambi_event_id=None,
                          cuota1=2.35, cuota2=1.62, confidence=52.1, favored="A",
                          jugador1="A", jugador2="B"):
    """Mismo helper mínimo que TestEdgeCompletoPartido._make_partido en
    test_edge_calculator.py, con outcome_id/kambi_event_id opcionales (D174-08)."""
    partido = {
        'jugador1': jugador1,
        'jugador2': jugador2,
        'cuota1': cuota1,
        'cuota2': cuota2,
        'superficie': 'clay',
        'torneo_nombre': 'Roland Garros (France)',
        'ranking_analysis': {
            f'{jugador1.replace(" ", "_")}_elo': 1800,
            f'{jugador2.replace(" ", "_")}_elo': 1800,
            f'{jugador1.replace(" ", "_")}_ranking': 100,
            f'{jugador2.replace(" ", "_")}_ranking': 50,
            'prediction': {
                'favored_player': favored,
                'confidence': confidence,
                'scores': {'p1_final_weight': 2.5, 'p2_final_weight': 2.8},
                'score_breakdown': {
                    'player1': {
                        'elo_rating':           {'contribution': '20%'},
                        'ranking_momentum':     {'contribution': '30%'},
                        'surface_specialization': {'contribution': '0%'},
                        'form_recent':          {'contribution': '25%'},
                        'common_opponents':     {'contribution': '25%'},
                    },
                    'player2': {
                        'elo_rating':           {'contribution': '25%'},
                        'ranking_momentum':     {'contribution': '35%'},
                        'surface_specialization': {'contribution': '0%'},
                        'form_recent':          {'contribution': '20%'},
                        'common_opponents':     {'contribution': '20%'},
                    }
                },
                'weights_used': {}
            }
        },
        'match_url': None,
        'match_id': None,
    }
    if outcome_id is not None:
        partido['outcome_id'] = outcome_id
    if kambi_event_id is not None:
        partido['kambi_event_id'] = kambi_event_id
    return partido


def _calibracion_vacia_d17408():
    return {
        'global': {'wins': 0, 'losses': 0},
        'por_superficie': {'clay': {'wins': 0, 'losses': 0}},
        'por_zona': {}
    }


def test_174_08_outcome_id_se_propaga_desde_el_h2h():
    """D174-08: outcome_id del registro h2h (adjuntado por match_ledger.py,
    Nodo-118/143) debe aparecer tal cual en el edge_report -- sin esto, los
    3 builders re-resuelven por name-matching cada corrida (D154-06 nunca
    fue realmente commiteado pese a estar marcado hecho en CLAUDE.md)."""
    import edge_calculator
    p = _make_partido_d17408(outcome_id="1003456789")
    r = edge_calculator.calcular_edge_completo(p, _calibracion_vacia_d17408())
    assert r is not None
    assert r['outcome_id'] == "1003456789"


def test_174_08_outcome_id_cae_a_kambi_event_id_si_falta():
    """Fallback: si outcome_id no viene pero kambi_event_id sí (formato viejo
    del ledger, ver match_ledger.py líneas 516/656/731), usarlo igual."""
    import edge_calculator
    p = _make_partido_d17408(kambi_event_id="777888999")
    r = edge_calculator.calcular_edge_completo(p, _calibracion_vacia_d17408())
    assert r is not None
    assert r['outcome_id'] == "777888999"


def test_174_08_outcome_id_none_si_ninguno_presente():
    """Sin outcome_id ni kambi_event_id en el h2h -- el campo debe ser None,
    nunca inventar un valor ni omitir la clave (los builders dependen de
    poder hacer .get('outcome_id') sin KeyError)."""
    import edge_calculator
    p = _make_partido_d17408()
    r = edge_calculator.calcular_edge_completo(p, _calibracion_vacia_d17408())
    assert r is not None
    assert 'outcome_id' in r
    assert r['outcome_id'] is None


# ─── D174-08 — los 3 builders prefieren outcome_id sobre name-matching ───────

def test_174_08_find_outcome_usa_hint_directo_sin_name_matching():
    """Con outcome_id_hint presente y vigente en outcomes_map, _find_outcome
    debe devolverlo directo -- sin recurrir al name-matching por apellido."""
    import combo_confianza_builder as ccb
    ccb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    # Nombre deliberadamente irresoluble por matching (no coincide con la clave)
    oc = ccb._find_outcome('Nombre Distinto Z.', 2.10, outcomes_map, outcome_id_hint='999111')
    assert oc is not None
    assert oc['outcome_id'] == '999111'
    assert ccb._OUTCOME_ID_STATS['hint_used'] == 1
    assert ccb._OUTCOME_ID_STATS['name_matched'] == 0


def test_174_08_find_outcome_cae_a_name_matching_si_hint_expirado():
    """outcome_id_hint que ya no está en el outcomes_map fresco (expiró, Nodo-157
    D157-02) -- debe caer al name-matching, nunca fallar duro."""
    import combo_confianza_builder as ccb
    ccb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    oc = ccb._find_outcome('McFadzean L.', 2.10, outcomes_map, outcome_id_hint='000000_viejo')
    assert oc is not None
    assert oc['outcome_id'] == '999111'
    assert ccb._OUTCOME_ID_STATS['hint_used'] == 0
    assert ccb._OUTCOME_ID_STATS['name_matched'] == 1


def test_174_08_find_outcome_sin_hint_comportamiento_identico_a_antes():
    """Regresión: sin outcome_id_hint (default None), el comportamiento debe
    ser exactamente el de antes de D174-08 -- name-matching puro."""
    import combo_confianza_builder as ccb
    ccb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    oc = ccb._find_outcome('McFadzean L.', 2.10, outcomes_map)
    assert oc is not None
    assert oc['outcome_id'] == '999111'
    assert ccb._OUTCOME_ID_STATS['name_matched'] == 1


def test_174_08_outcome_id_hit_rate_none_sin_resoluciones():
    """Antes de resolver ningún pick, el hit_rate debe ser None (no 0.0 --
    evita reportar 0% cuando en realidad no ha corrido nada)."""
    import combo_confianza_builder as ccb
    ccb._reset_outcome_id_stats()
    assert ccb._outcome_id_hit_rate() is None


def test_174_08_calc_combo_propaga_outcome_ids_paralelo_a_piernas():
    """_calc_combo debe incluir 'outcome_ids' paralelo a 'piernas' -- el
    wiring que permite a _find_outcome recibir el hint en los builders."""
    import combo_confianza_builder as ccb
    picks_subset = [
        {'nombre': 'A', 'cuota': 2.0, 'confianza': 60.0,
         'cat': {'categoria': 'A'}, 'outcome_id': '111'},
        {'nombre': 'B', 'cuota': 3.0, 'confianza': 55.0,
         'cat': {'categoria': 'B'}, 'outcome_id': None},
    ]
    combo = ccb._calc_combo(picks_subset, stake=1000, nombre='TEST')
    assert combo['outcome_ids'] == ['111', None]
    assert combo['piernas'] == ['A', 'B']


# ─── D174-08 — betplay_combo_builder.py::find_outcome() (2º builder) ────────

def test_174_08_bcb_find_outcome_usa_hint_directo_sin_name_matching():
    """betplay_combo_builder.find_outcome() con outcome_id_hint vigente debe
    devolverlo directo, sin recurrir al name-matching por apellido."""
    import betplay_combo_builder as bcb
    bcb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    oc, reason = bcb.find_outcome('Nombre Distinto Z.', 2.10, outcomes_map,
                                   outcome_id_hint='999111')
    assert oc is not None
    assert reason == 'OK'
    assert oc['outcome_id'] == '999111'
    assert bcb._OUTCOME_ID_STATS['hint_used'] == 1
    assert bcb._OUTCOME_ID_STATS['name_matched'] == 0


def test_174_08_bcb_find_outcome_cae_a_name_matching_si_hint_expirado():
    """outcome_id_hint que ya no está vigente en el outcomes_map fresco
    (expiró, Nodo-157 D157-02) -- debe caer al name-matching, nunca fallar duro."""
    import betplay_combo_builder as bcb
    bcb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    oc, reason = bcb.find_outcome('McFadzean L.', 2.10, outcomes_map,
                                   outcome_id_hint='000000_viejo')
    assert oc is not None
    assert reason == 'OK'
    assert oc['outcome_id'] == '999111'
    assert bcb._OUTCOME_ID_STATS['hint_used'] == 0
    assert bcb._OUTCOME_ID_STATS['name_matched'] == 1


def test_174_08_bcb_find_outcome_sin_hint_comportamiento_identico_a_antes():
    """Regresión: sin outcome_id_hint (default None), el comportamiento debe
    ser exactamente el de antes de D174-08 -- name-matching puro."""
    import betplay_combo_builder as bcb
    bcb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    oc, reason = bcb.find_outcome('McFadzean L.', 2.10, outcomes_map)
    assert oc is not None
    assert reason == 'OK'
    assert oc['outcome_id'] == '999111'
    assert bcb._OUTCOME_ID_STATS['name_matched'] == 1


def test_174_08_bcb_outcome_id_hit_rate_none_sin_resoluciones():
    """Antes de resolver ningún pick, el hit_rate debe ser None (no 0.0)."""
    import betplay_combo_builder as bcb
    bcb._reset_outcome_id_stats()
    assert bcb._outcome_id_hit_rate() is None


def test_174_08_bcb_outcome_id_hit_rate_calcula_fraccion_hint_used():
    """hit_rate = hint_used / total resoluciones (hint+name_matched+no_match)."""
    import betplay_combo_builder as bcb
    bcb._reset_outcome_id_stats()
    outcomes_map = {
        'mcfadzean': {'outcome_id': '999111', 'odds': 2.10, 'jugador': 'Lachlan Mcfadzean'},
    }
    bcb.find_outcome('X', 2.10, outcomes_map, outcome_id_hint='999111')  # hint
    bcb.find_outcome('McFadzean L.', 2.10, outcomes_map)                # name_matched
    bcb.find_outcome('Nadie Existe Z.', 9.99, outcomes_map)             # no_match
    assert bcb._outcome_id_hit_rate() == round(1 / 3, 3)


# ─── D174-08 — favoritos_combo_builder.py (3er builder) ──────────────────────
# favoritos_combo_builder.py no tiene matcher propio: reusa find_outcome() de
# betplay_combo_builder.py en 2 sitios de main() (pre-filtro Kambi L831,
# resolución de outcome_ids del combo L899). Los tests verifican que el
# outcome_id ya presente en el pick (propagado vía **pick en
# seleccionar_favoritos/armar_combos) efectivamente llega como
# outcome_id_hint a esos 2 call sites -- sin re-implementar find_outcome.

def test_174_08_favoritos_prefiltro_pasa_outcome_id_hint(tmp_path, monkeypatch):
    """Pre-filtro Kambi (main() L831): el pick con outcome_id en el edge_report
    debe llegar a find_outcome() como outcome_id_hint, no solo por nombre."""
    import json
    import sys
    from unittest.mock import patch, MagicMock
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import favoritos_combo_builder as fcb
    from edge_calculator import GATE_VERSION

    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": [],
        "watchlist": [
            {
                "favorito_predicho": "Alcaraz C.", "favorito": "Alcaraz C.", "cuota_favorito": 1.30,
                "cuota_rival": 3.20, "p_modelo": 0.75, "confidence_flag": "STRONG",
                "outcome_id": "777888",
            },
        ],
    }
    edge_path = tmp_path / "edge_report_test.json"
    edge_path.write_text(json.dumps(edge_data), encoding="utf-8")

    captured_kwargs = {}

    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        captured_kwargs.update(kwargs)
        return {"outcome_id": kwargs.get("outcome_id_hint") or "999", "odds": cuota}, "OK"

    argv = ["favoritos_combo_builder.py", "--dry-run", "--file", str(edge_path)]
    with patch.object(sys, "argv", argv), \
         patch("betplay_combo_builder.fetch_kambi_outcomes", return_value=({}, {})), \
         patch("betplay_combo_builder.find_outcome", side_effect=_fake_find_outcome), \
         patch.object(fcb, "_governor_check", return_value=None), \
         patch.object(fcb, "_find_latest_h2h", return_value=None):
        try:
            fcb.main()
        except SystemExit:
            pass

    assert captured_kwargs.get("outcome_id_hint") == "777888"


def test_174_08_favoritos_sin_outcome_id_hint_es_none_no_falla(tmp_path):
    """Regresión: pick sin outcome_id en el edge_report (caso RANKING_ONLY/H2H_MODEL
    sin Kambi resuelto) debe pasar outcome_id_hint=None -- find_outcome cae a
    name-matching puro, sin excepción."""
    import json
    import sys
    from unittest.mock import patch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import favoritos_combo_builder as fcb
    from edge_calculator import GATE_VERSION

    edge_data = {
        "metadata": {"gate_version": GATE_VERSION},
        "apostar": [],
        "watchlist": [
            {
                "favorito_predicho": "Sinner J.", "favorito": "Sinner J.", "cuota_favorito": 1.25,
                "cuota_rival": 3.50, "p_modelo": 0.78, "confidence_flag": "STRONG",
            },
        ],
    }
    edge_path = tmp_path / "edge_report_test.json"
    edge_path.write_text(json.dumps(edge_data), encoding="utf-8")

    captured_kwargs = {}

    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        captured_kwargs.update(kwargs)
        return {"outcome_id": "555", "odds": cuota}, "OK"

    argv = ["favoritos_combo_builder.py", "--dry-run", "--file", str(edge_path)]
    with patch.object(sys, "argv", argv), \
         patch("betplay_combo_builder.fetch_kambi_outcomes", return_value=({}, {})), \
         patch("betplay_combo_builder.find_outcome", side_effect=_fake_find_outcome), \
         patch.object(fcb, "_governor_check", return_value=None), \
         patch.object(fcb, "_find_latest_h2h", return_value=None):
        try:
            fcb.main()
        except SystemExit:
            pass

    assert "outcome_id_hint" in captured_kwargs
    assert captured_kwargs.get("outcome_id_hint") is None


# ─── D174-08 — build_combo_links() consumiendo legs de trader_ev_tenis._build_cobertura ──
# Gap encontrado en auditoría del "4to generador": trader_ev_tenis.py::_build_cobertura()
# escribe cobertura_plan al trader_plan_*.json (clave "cobertura"), y el consumidor real
# es betplay_combo_builder.py::build_combo_links() (no combo_confianza_builder.py). Ese
# call site nunca pasaba outcome_id_hint pese a que edge_calculator.py ya propaga
# outcome_id en cada pick del pool desde el fix D174-08 original.

def test_174_08_build_combo_links_pasa_outcome_id_hint_desde_leg(tmp_path):
    """Cada leg de combo['legs'] con outcome_id (propagado por trader_ev_tenis.py
    _build_cobertura L871-872) debe llegar a find_outcome() como outcome_id_hint."""
    import sys
    from unittest.mock import patch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import betplay_combo_builder as bcb

    trader_plan = {
        "cobertura": [
            {
                "piernas_n": 2,
                "legs": [
                    {"jugador": "Alcaraz C.", "cuota": 1.30, "outcome_id": "444555"},
                    {"jugador": "Sinner J.", "cuota": 1.25, "outcome_id": "666777"},
                ],
                "cuota_combo": 1.63,
                "stake": 1000,
                "retorno_potencial": 1630,
            },
        ],
    }

    captured_hints = []

    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        captured_hints.append(kwargs.get("outcome_id_hint"))
        return {"outcome_id": kwargs.get("outcome_id_hint") or "999", "odds": cuota}, "OK"

    with patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome):
        results = bcb.build_combo_links(trader_plan, min_piernas=2)

    assert len(results) == 1
    assert captured_hints == ["444555", "666777"]


def test_174_08_build_combo_links_leg_sin_outcome_id_hint_es_none(tmp_path):
    """Leg sin outcome_id (pick RANKING_ONLY/H2H_MODEL sin Kambi resuelto en
    edge_report) debe pasar outcome_id_hint=None -- name-matching puro, sin excepción."""
    import sys
    from unittest.mock import patch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import betplay_combo_builder as bcb

    trader_plan = {
        "cobertura": [
            {
                "piernas_n": 2,
                "legs": [
                    {"jugador": "Alcaraz C.", "cuota": 1.30},
                    {"jugador": "Sinner J.", "cuota": 1.25},
                ],
                "cuota_combo": 1.63,
                "stake": 1000,
                "retorno_potencial": 1630,
            },
        ],
    }

    captured_hints = []

    def _fake_find_outcome(jugador, cuota, outcomes_map, started_map, **kwargs):
        captured_hints.append(kwargs.get("outcome_id_hint"))
        return {"outcome_id": "999", "odds": cuota}, "OK"

    with patch.object(bcb, "fetch_kambi_outcomes", return_value=({"dummy": {}}, {})), \
         patch.object(bcb, "find_outcome", side_effect=_fake_find_outcome):
        results = bcb.build_combo_links(trader_plan, min_piernas=2)

    assert len(results) == 1
    assert captured_hints == [None, None]


# ─── D174-09 — audit_phantom_history.py integrado semanal en run_daily.py ────
# Patrón replicado de test_nodo141_kambi_only_report.py (PASO 3K), inspección
# de fuente: run_daily.py no expone hooks unitarios por PASO, así que el
# contrato verificable es que el comando y el guard semanal existan.

def test_174_09_run_daily_tiene_paso_audit_phantom_semanal():
    """run_daily.py debe invocar audit_phantom_history.py con guard semanal
    (lunes) dentro del bloque de fase noche/completa (Nodo-174 D174-09)."""
    src = (Path(__file__).parent.parent / "run_daily.py").read_text(encoding="utf-8")
    assert "audit_phantom_history.py" in src, \
        "PASO 3.8 audit_phantom_history.py ausente en run_daily.py"
    assert "PASO 3.8" in src, "Label PASO 3.8 ausente en run_daily.py"
    assert "weekday() == 0" in src, \
        "Guard semanal (lunes) ausente -- D174-09 pide cadencia semanal, no diaria"


def test_174_09_paso_audit_phantom_es_optional():
    """El PASO 3.8 no debe bloquear el pipeline si el script falla (optional=True,
    mismo patrón que PASO 3K/D141-02 y PASO 3.9/D154-08)."""
    src = (Path(__file__).parent.parent / "run_daily.py").read_text(encoding="utf-8")
    idx = src.index("audit_phantom_history.py")
    bloque = src[idx: idx + 300]
    assert "optional=True" in bloque, \
        "PASO 3.8 debe ser optional=True -- auditoría no debe matar el pipeline nocturno"

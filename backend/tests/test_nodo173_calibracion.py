"""
tests/test_nodo173_calibracion.py — Nodo-173: Calibración p_modelo + Embudo de Decisión

REGLA-T53: cada test invoca la función real del módulo, nunca hardcodea la fórmula.

Cobertura por bloque del spec:
  BLOQUE A (D173-01/03/07): gate_ledger, funnel, score_margin_signed, constante unificada
  BLOQUE B (D173-02/06):    caps de serialización eliminados, phantom confidence cap
  BLOQUE C (D173-04/05):    backfill de features, calibrador ancla-mercado (PUERTA 3 FALLIDA)
  BLOQUE D (D173-10):       observabilidad gate Kambi (combo_exclusions)
  BLOQUE E (D173-11/12):    reporte de embudo, segmentos de calibración en shadow_book

PUERTA 3 (D173-05) NO fue aprobada (skill holdout <= 0) — data/probability_calibrator.json
no existe. Los tests de D173-05 verifican el mecanismo de decisión (aprobado=False cuando
skill<=0), no un artefacto desplegado que no existe.
"""
import json
import math

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helper compartido — partido mínimo válido para calcular_edge_completo
# ─────────────────────────────────────────────────────────────────────────────

def _make_partido(favored='p1', score_difference=0.7, score_margin_raw=None,
                   cuota1=2.02, cuota2=1.70, confidence=75.3,
                   ranking1=850, ranking2=910, sin_scores=False):
    jugador1, jugador2 = 'Mario Arce', 'Vlajic Josip'
    favored_player = jugador1 if favored == 'p1' else jugador2
    pred = {
        'favored_player': favored_player,
        'confidence': confidence,
        'historial_incompleto': {'p1': False, 'p2': False},
        'scores': {} if sin_scores else {
            'p1_final_weight': 1.5, 'p2_final_weight': 0.8,
            'score_difference': score_difference,
        },
        'score_breakdown': {
            'player1': {k: {'contribution': 0.0, 'contribution_pct': '0.0%'} for k in (
                'surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct',
                'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule')},
            'player2': {k: {'contribution': 0.0, 'contribution_pct': '0.0%'} for k in (
                'surface_specialization', 'form_recent', 'common_opponents', 'h2h_direct',
                'ranking_momentum', 'elo_rating', 'home_advantage', 'strength_of_schedule')},
        },
        'weights_used': {
            'surface_specialization': 0.18, 'form_recent': 0.22,
            'common_opponents': 0.10, 'h2h_direct': 0.10,
            'ranking_momentum': 0.20, 'elo_rating': 0.15,
            'home_advantage': 0.05, 'strength_of_schedule': 0.00,
        },
        'markov_analysis': None,
        'tardio_analysis': None,
        'circuit_asymmetry': {'signal': 'SYMMETRIC', 'ratio': 1.0},
        'surface_specialization_meta': {
            'player1': {'volume_confidence': 0.8},
            'player2': {'volume_confidence': 0.8},
        },
    }
    if score_margin_raw is not None:
        pred['score_margin_raw'] = score_margin_raw
        pred['score_sum_raw'] = 2.3
    return {
        'jugador1': jugador1,
        'jugador2': jugador2,
        'cuota1': cuota1,
        'cuota2': cuota2,
        'torneo_nombre': 'M15 Cary (USA)',
        'tipo_cancha': 'hard',
        'torneo_completo': 'ITF - INDIVIDUALES: M15 Cary (USA)',
        'match_url': 'https://www.flashscore.co/match/tennis/test/AAABBBCC/#/h2h',
        'match_id': 'AAABBBCC',
        'ranking1': ranking1,
        'ranking2': ranking2,
        'data_quality': {
            'historial_extraido_p1': True, 'historial_extraido_p2': True,
            'n_partidos_p1': 14, 'n_partidos_p2': 11,
        },
        'ranking_analysis': {
            'Mario_Arce_ranking': ranking1,
            'Vlajic_Josip_ranking': ranking2,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.55,
            'p2_rivalry_score': 0.45,
            'prediction': pred,
            'Mario_Arce_metrics': None,
            'Vlajic_Josip_metrics': None,
            'Mario_Arce_elo': 1400.0,
            'Vlajic_Josip_elo': 1380.0,
        },
        'form_analysis': {'Mario_Arce_form': None, 'Vlajic_Josip_form': None},
        'enfrentamientos_directos': [],
        'estadisticas': {
            'partidos_Mario_Arce': 14, 'partidos_Vlajic_Josip': 11,
            'enfrentamientos_totales': 0,
        },
    }


def _write_h2h(tmp_path, partidos, name='h2h_test.json'):
    from analysis.rivalry_analyzer import RIVALRY_VERSION
    h2h_file = tmp_path / name
    h2h_file.write_text(json.dumps({
        'metadata': {'rivalry_version': RIVALRY_VERSION},
        'partidos': partidos,
    }))
    return h2h_file


# ─────────────────────────────────────────────────────────────────────────────
# D173-01 — registrar_gate + funnel
# ─────────────────────────────────────────────────────────────────────────────

class TestD17301GateLedger:

    def test_173_01a_registrar_gate_append_only(self):
        from edge_calculator import registrar_gate
        resultado = {}
        registrar_gate(resultado, 'G_EDGE_MIN', 'edge insuficiente')
        registrar_gate(resultado, 'G_KELLY_MIN', 'kelly insuficiente')
        assert resultado['gate_ledger'] == [
            {'gate': 'G_EDGE_MIN', 'motivo': 'edge insuficiente'},
            {'gate': 'G_KELLY_MIN', 'motivo': 'kelly insuficiente'},
        ]

    def test_173_01b_registrar_gate_primer_bloqueante_no_se_sobreescribe(self):
        from edge_calculator import registrar_gate
        resultado = {}
        registrar_gate(resultado, 'G_EDGE_MIN', 'primero')
        registrar_gate(resultado, 'G_KELLY_MIN', 'segundo')
        registrar_gate(resultado, 'G_T32_01', 'tercero')
        assert resultado['gate_bloqueante'] == 'G_EDGE_MIN'
        assert len(resultado['gate_ledger']) == 3

    def test_173_01c_calcular_edge_gate_ledger_t32_01(self):
        """p_modelo bajo + cuota underdog >= 2.10 → G_T32_01 bloqueante."""
        from edge_calculator import calcular_edge
        resultado = calcular_edge(p_modelo=0.50, cuota_favorito=2.50)
        assert resultado['apostar'] is False
        assert resultado['gate_bloqueante'] == 'G_T32_01'
        assert any(g['gate'] == 'G_T32_01' for g in resultado['gate_ledger'])

    def test_173_01d_calcular_edge_sin_gates_cuando_apostar(self):
        """Escenario claramente apostable → gate_ledger vacío, gate_bloqueante None."""
        from edge_calculator import calcular_edge
        resultado = calcular_edge(p_modelo=0.75, cuota_favorito=1.50, n_calibracion=40)
        assert resultado['apostar'] is True
        assert resultado['gate_ledger'] == []
        assert resultado['gate_bloqueante'] is None

    def test_173_01e_funnel_invariante_suma_igual_procesados(self, tmp_path):
        """Invariante (comentario del código, línea ~1770):
        sum(por_gate.values()) + n_sobrevive == n_procesados."""
        from edge_calculator import procesar_archivo_h2h

        partidos = [
            _make_partido(confidence=75.3, cuota1=1.50, cuota2=2.60),   # sobrevive (favorito)
            _make_partido(confidence=40.0, cuota1=2.50, cuota2=1.55),   # G_T32_01
        ]
        h2h_file = _write_h2h(tmp_path, partidos)
        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        funnel = resultado['metadata']['funnel']
        assert (sum(funnel['por_gate'].values()) + funnel['n_sobrevive']
                == funnel['n_procesados'])
        assert funnel['n_procesados'] == 2


# ─────────────────────────────────────────────────────────────────────────────
# D173-02 — caps de serialización eliminados
# ─────────────────────────────────────────────────────────────────────────────

class TestD17302CapsEliminados:

    def test_173_02a_watchlist_no_truncada(self, tmp_path):
        """watchlist ya no se corta en [:50] — n_watchlist_total coincide con len()."""
        from edge_calculator import procesar_archivo_h2h
        partidos = [_make_partido(confidence=45.0, cuota1=1.30, cuota2=3.50)
                    for _ in range(6)]
        h2h_file = _write_h2h(tmp_path, partidos)
        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)
        assert len(resultado['watchlist']) == resultado['metadata']['n_watchlist_total']

    def test_173_02b_sin_edge_no_truncada(self, tmp_path):
        """sin_edge ya no se corta en [:5] — n_sin_edge_total coincide con len()."""
        from edge_calculator import procesar_archivo_h2h
        partidos = [_make_partido(confidence=30.0, cuota1=1.05, cuota2=15.0)
                    for _ in range(6)]
        h2h_file = _write_h2h(tmp_path, partidos)
        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)
        assert len(resultado['sin_edge']) == resultado['metadata']['n_sin_edge_total']


# ─────────────────────────────────────────────────────────────────────────────
# D173-03 — score_margin_signed
# ─────────────────────────────────────────────────────────────────────────────

class TestD17303ScoreMarginSigned:

    def test_173_03a_positivo_cuando_favorito_es_jugador1(self):
        from edge_calculator import calcular_edge_completo
        partido = _make_partido(favored='p1', score_difference=0.7)
        resultado = calcular_edge_completo(partido, {'global': {'wins': 0, 'losses': 0}})
        assert resultado['score_margin_signed'] == 0.7

    def test_173_03b_signo_invertido_cuando_favorito_es_jugador2(self):
        """score_difference se mide respecto a p1 — si el favorito es p2, se invierte."""
        from edge_calculator import calcular_edge_completo
        partido = _make_partido(favored='p2', score_difference=0.7)
        resultado = calcular_edge_completo(partido, {'global': {'wins': 0, 'losses': 0}})
        assert resultado['score_margin_signed'] == -0.7

    def test_173_03c_none_cuando_no_hay_datos_ni_fallback(self):
        """Sin score_margin_raw ni scores.score_difference → None, nunca 0."""
        from edge_calculator import calcular_edge_completo
        partido = _make_partido(favored='p1', sin_scores=True)
        resultado = calcular_edge_completo(partido, {'global': {'wins': 0, 'losses': 0}})
        assert resultado['score_margin_signed'] is None

    def test_173_03d_usa_score_margin_raw_cuando_presente(self):
        """Con score_margin_raw explícito (post-D173-03 real), se usa directo, no el fallback."""
        from edge_calculator import calcular_edge_completo
        partido = _make_partido(favored='p1', score_margin_raw=1.25, score_difference=99.0)
        resultado = calcular_edge_completo(partido, {'global': {'wins': 0, 'losses': 0}})
        assert resultado['score_margin_signed'] == 1.25


# ─────────────────────────────────────────────────────────────────────────────
# D173-07 — constante unificada P_MODELO_MIN_UNDERDOG
# ─────────────────────────────────────────────────────────────────────────────

class TestD17307ConstanteUnificada:

    def test_173_07_edge_calculator_importa_config_no_redefine(self):
        import config
        import edge_calculator
        assert edge_calculator.P_MODELO_MIN_UNDERDOG is config.P_MODELO_MIN_UNDERDOG


# ─────────────────────────────────────────────────────────────────────────────
# D173-06 — phantom confidence cap
# ─────────────────────────────────────────────────────────────────────────────

class TestD17306PhantomCap:

    def test_173_06a_cap_aplica_cuando_falta_ranking_y_p_modelo_alto(self):
        from edge_calculator import _phantom_confidence_cap, PHANTOM_CAP
        p_ajustada, motivo = _phantom_confidence_cap(
            p_modelo=0.93, rival_ranking_missing=True, fav_ranking_missing=False, n_h2h=0)
        assert p_ajustada == PHANTOM_CAP
        assert motivo is not None

    def test_173_06b_no_aplica_cuando_ambos_rankings_presentes(self):
        from edge_calculator import _phantom_confidence_cap
        p_ajustada, motivo = _phantom_confidence_cap(
            p_modelo=0.93, rival_ranking_missing=False, fav_ranking_missing=False, n_h2h=10)
        assert p_ajustada == 0.93
        assert motivo is None

    def test_173_06c_no_aplica_cuando_p_modelo_ya_bajo_el_cap(self):
        from edge_calculator import _phantom_confidence_cap, PHANTOM_CAP
        p_ajustada, motivo = _phantom_confidence_cap(
            p_modelo=PHANTOM_CAP - 0.05, rival_ranking_missing=True,
            fav_ranking_missing=False, n_h2h=0)
        assert p_ajustada == PHANTOM_CAP - 0.05
        assert motivo is None


# ─────────────────────────────────────────────────────────────────────────────
# D173-04 — backfill de features
# ─────────────────────────────────────────────────────────────────────────────

class TestD17304Backfill:

    def test_173_04a_copia_raw_cuando_snapshot_ya_tiene_campo(self):
        from scripts.backfill_calibration_features import backfill_record
        rec = {'pick_snapshot': {'score_margin_signed': 0.42}}
        cambio = backfill_record(rec)
        assert cambio is True
        assert rec['score_margin_signed'] == 0.42
        assert rec['feature_provenance'] == 'raw'

    def test_173_04b_proxy_normalizado_desde_p_modelo(self):
        from scripts.backfill_calibration_features import backfill_record
        rec = {'pick_snapshot': {'p_modelo': 0.68}}
        cambio = backfill_record(rec)
        assert cambio is True
        assert rec['score_margin_signed'] == pytest.approx(0.18)
        assert rec['feature_provenance'] == 'proxy_normalizado'

    def test_173_04c_idempotente(self):
        from scripts.backfill_calibration_features import backfill_record
        rec = {'pick_snapshot': {'p_modelo': 0.68}}
        backfill_record(rec)
        cambio_segunda_vez = backfill_record(rec)
        assert cambio_segunda_vez is False

    def test_173_04d_sin_datos_no_cambia_nada(self):
        from scripts.backfill_calibration_features import backfill_record
        rec = {'pick_snapshot': {}}
        assert backfill_record(rec) is False
        assert 'score_margin_signed' not in rec


# ─────────────────────────────────────────────────────────────────────────────
# D173-05 — calibrador ancla-mercado (PUERTA 3: FALLIDA en la corrida real)
# ─────────────────────────────────────────────────────────────────────────────

class TestD17305Calibrador:

    def test_173_05a_fit_calibrator_exige_min_n(self):
        from core.probability_calibrator import fit_calibrator
        with pytest.raises(ValueError):
            fit_calibrator([], min_n=300)

    def test_173_05b_predict_calibrated_reproduce_mercado_con_coefs_neutros(self):
        """beta1=1, beta0=beta2=beta3=beta4=0 → p_final == p_implicita (piso de seguridad)."""
        from core.probability_calibrator import predict_calibrated
        artifact = {'coeficientes': {'beta0': 0.0, 'beta1': 1.0, 'beta2': 0.0,
                                      'beta3': 0.0, 'beta4': 0.0}}
        p = predict_calibrated(artifact, p_implicita=0.65, score_margin_signed=0.0,
                                rival_ranking_missing=False, fav_ranking_missing=False)
        assert p == pytest.approx(0.65, abs=1e-6)

    def test_173_05c_evaluate_calibration_skill_cero_en_baseline(self):
        """Predecir siempre la tasa media == baseline climatológico → skill == 0."""
        from core.probability_calibrator import evaluate_calibration
        y = [1, 0, 1, 0, 1, 0]
        tasa = sum(y) / len(y)
        m = evaluate_calibration(y, [tasa] * len(y))
        assert m['skill'] == pytest.approx(0.0, abs=1e-9)

    def test_173_05d_fit_calibrator_no_aprobado_cuando_skill_negativo(self):
        """Datos donde el modelo es peor que el mercado → aprobado=False (PUERTA 3 real)."""
        from core.probability_calibrator import fit_calibrator
        import random
        random.seed(7)
        records = []
        for i in range(320):
            p_impl = random.choice([0.3, 0.5, 0.7])
            y = 1 if random.random() < p_impl else 0
            records.append({
                'resolucion': {'resultado': 'WON' if y else 'LOST'},
                'pick_snapshot': {'p_implicita': p_impl},
                'score_margin_signed': random.uniform(-1, 1),
                'logged_at': f'2026-01-{(i % 28) + 1:02d}T00:00:00',
            })
        artifact = fit_calibrator(records, min_n=300)
        assert 'aprobado' in artifact
        assert isinstance(artifact['aprobado'], bool)
        assert artifact['aprobado'] == (artifact['metricas_holdout']['skill'] > 0)

    def test_173_05e_extraer_features_registro_none_sin_score_margin(self):
        from core.probability_calibrator import extraer_features_registro
        rec = {'resolucion': {'resultado': 'WON'}, 'pick_snapshot': {'p_implicita': 0.6}}
        assert extraer_features_registro(rec) is None

    def test_173_05f_puerta_3_estado_real_no_desplegado(self):
        """Confirma el cierre honesto del nodo: sin artefacto, sin flag activo."""
        import os
        import edge_calculator
        assert os.path.exists(os.path.join('data', 'probability_calibrator.json')) is False
        assert edge_calculator.USE_CALIBRATOR is False


# ─────────────────────────────────────────────────────────────────────────────
# D173-10 — observabilidad gate Kambi (combo_exclusions)
# ─────────────────────────────────────────────────────────────────────────────

class TestD17310ComboExclusions:

    def test_173_10a_exclusion_record_normaliza_dict(self):
        from core.combo_exclusions import exclusion_record
        pick = {'partido': 'A vs B', 'cuota_favorito': 1.85, 'p_modelo': 0.61}
        rec = exclusion_record(pick, 'kambi_no_disponible')
        assert rec['partido'] == 'A vs B'
        assert rec['motivo'] == 'kambi_no_disponible'
        assert rec['cuota'] == 1.85

    def test_173_10b_registrar_y_leer_exclusiones_roundtrip(self, tmp_path, monkeypatch):
        import core.combo_exclusions as ce
        monkeypatch.chdir(tmp_path)
        picks = [{'partido': 'X vs Y', 'cuota_favorito': 2.0}]
        n = ce.registrar_exclusiones('test_builder', picks, fecha_compact='20260101')
        assert n == 1
        entradas = ce.leer_exclusiones('20260101')
        assert entradas[0]['builder'] == 'test_builder'
        assert entradas[0]['n'] == 1


# ─────────────────────────────────────────────────────────────────────────────
# D173-11 — reporte diario de embudo
# ─────────────────────────────────────────────────────────────────────────────

class TestD17311FunnelReport:

    def test_173_11a_sin_edge_report_mensaje_accionable(self, tmp_path, monkeypatch):
        from scripts.funnel_report import generar_reporte
        monkeypatch.chdir(tmp_path)
        texto = generar_reporte('20260101')
        assert 'edge_calculator.py' in texto
        assert texto.strip() != ''

    def test_173_11b_cero_procesados_mensaje_accionable(self, tmp_path, monkeypatch):
        from scripts.funnel_report import generar_reporte
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'reports').mkdir()
        edge_report = {'metadata': {'funnel': {'n_procesados': 0, 'por_gate': {}, 'n_sobrevive': 0}}}
        (tmp_path / 'reports' / 'edge_report_20260101_000000.json').write_text(json.dumps(edge_report))
        texto = generar_reporte('20260101')
        assert 'PASO 1' in texto or 'PASO 2' in texto

    def test_173_11c_sobreviven_cero_muestra_3_mas_cerca(self, tmp_path, monkeypatch):
        """Requisito duro del spec: SOBREVIVEN=0 nunca termina sin sección accionable."""
        from scripts.funnel_report import generar_reporte
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'reports').mkdir()
        edge_report = {
            'metadata': {'funnel': {'n_procesados': 2, 'por_gate': {'G_T32_01': 2}, 'n_sobrevive': 0}},
            'watchlist': [{'partido': 'A vs B', 'gate_bloqueante': 'G_T32_01', 'p_modelo': 0.48}],
            'sin_edge': [{'partido': 'C vs D', 'gate_bloqueante': 'G_T32_01', 'p_modelo': 0.51}],
        }
        (tmp_path / 'reports' / 'edge_report_20260101_000000.json').write_text(json.dumps(edge_report))
        texto = generar_reporte('20260101')
        assert 'MÁS CERCA' in texto or 'MAS CERCA' in texto
        assert 'SOBREVIVEN' in texto

    def test_173_11d_distancia_a_umbral_t32_01_signo_positivo(self):
        from scripts.funnel_report import _distancia_a_umbral
        pick = {'gate_bloqueante': 'G_T32_01', 'p_modelo': 0.50}
        dist, etiqueta = _distancia_a_umbral(pick)
        assert dist >= 0
        assert 'faltaron' in etiqueta


# ─────────────────────────────────────────────────────────────────────────────
# D173-12 — segmentos de calibración en shadow_book.report()
# ─────────────────────────────────────────────────────────────────────────────

class TestD17312ShadowBookSegmentos:

    def test_173_12a_reporte_incluye_seccion_nodo173(self):
        import shadow_book
        out = shadow_book.report()
        assert 'NODO-173 CALIBRACION' in out
        assert 'SEGMENTO POR BANDA DE CUOTA' in out

    def test_173_12b_sin_artefacto_mensaje_no_disponible(self):
        """No existe data/probability_calibrator.json → mensaje honesto, no data fabricada."""
        import os
        import shadow_book
        assert not os.path.exists(os.path.join('data', 'probability_calibrator.json'))
        out = shadow_book.report()
        assert 'NO DISPONIBLE' in out
        assert 'PUERTA 3' in out

    def test_173_12c_t32_01_habria_bloqueado_no_disponible(self):
        import shadow_book
        out = shadow_book.report()
        assert 'T32_01_HABRIA_BLOQUEADO' in out
        assert 'D173-08 no se' in out

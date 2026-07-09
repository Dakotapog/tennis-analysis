"""
tests/test_nodo51_f2.py — Nodo-51 F2: Contrato de Completitud + Procedencia

Tests T51-F2-01 → T51-F2-13

Validan:
  - data_contract.has_empty_history() y completeness_score()
  - edge_calculator emite status='NO_DATA' cuando historial vacío
  - picks NO_DATA excluidos de todos los pools (apostar/watchlist/sin_edge)
  - picks NO_DATA aparecen en lista separada 'no_data' del edge report
  - ninja_h2h_parser añade history_provenance a data_quality
  - Invariante de pools: trader nunca puede acceder a picks NO_DATA

Detección de mutación real:
  T51-F2-06 FALLA si edge_calculator no marca status='NO_DATA' cuando historial vacío
  T51-F2-07 FALLA si picks NO_DATA aparecen en apostar/watchlist/sin_edge
  T51-F2-08 FALLA si output de procesar_archivo_h2h no tiene lista 'no_data'
  T51-F2-11 FALLA si ninja_h2h_parser no añade history_provenance a data_quality
  T51-F2-13 FALLA si picks NO_DATA pueden entrar al pool cobertura (phantom combos)

Caso real documentado (2026-07-01):
  Arce/Vlajic/Guajardo/Cooper: 0 partidos extraídos → edge=60.4% → Combo1-8 fantasma
  Con F2: status='NO_DATA', NO aparecen en apostar/watchlist/sin_edge → 0 combos fantasma
"""
import json
import pytest
from unittest.mock import MagicMock

from core.data_contract import (
    PROVENANCE_NINJA_API,
    PROVENANCE_THF_CACHE,
    PROVENANCE_PLAYWRIGHT_DOM,
    PROVENANCE_EMPTY,
    PICK_STATUS_NO_DATA,
    has_empty_history,
    completeness_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

def _make_partido_base(historial_incompleto=None, cuota1=2.02, cuota2=1.70, p_modelo=0.753):
    """Partido mínimo válido para calcular_edge_completo."""
    hi = historial_incompleto or {'p1': False, 'p2': False}
    return {
        'jugador1': 'Mario Arce',
        'jugador2': 'Vlajic Josip',
        'cuota1': cuota1,
        'cuota2': cuota2,
        'torneo_nombre': 'M15 Cary (USA)',
        'tipo_cancha': 'hard',
        'torneo_completo': 'ITF - INDIVIDUALES: M15 Cary (USA)',
        'match_url': 'https://www.flashscore.co/match/tennis/test/AAABBBCC/#/h2h',
        'match_id': 'AAABBBCC',
        'ranking1': 850,
        'ranking2': 910,
        'data_quality': {
            'historial_extraido_p1': not hi['p1'],
            'historial_extraido_p2': not hi['p2'],
            'n_partidos_p1': 0 if hi['p1'] else 14,
            'n_partidos_p2': 0 if hi['p2'] else 11,
        },
        'ranking_analysis': {
            'Mario_Arce_ranking': 850,
            'Vlajic_Josip_ranking': 910,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.55,
            'p2_rivalry_score': 0.45,
            'prediction': {
                'favored_player': 'Mario Arce',
                'confidence': p_modelo * 100,
                'historial_incompleto': hi,
                'scores': {'p1_final_weight': 1.5, 'p2_final_weight': 0.8,
                           'score_difference': 0.7},
                'score_breakdown': {
                    'player1': {
                        'surface_specialization': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'form_recent': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'common_opponents': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'h2h_direct': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'ranking_momentum': {'contribution': 1.0, 'contribution_pct': '66.7%'},
                        'elo_rating': {'contribution': 0.5, 'contribution_pct': '33.3%'},
                        'home_advantage': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'strength_of_schedule': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                    },
                    'player2': {
                        'surface_specialization': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'form_recent': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'common_opponents': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'h2h_direct': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'ranking_momentum': {'contribution': 0.5, 'contribution_pct': '62.5%'},
                        'elo_rating': {'contribution': 0.3, 'contribution_pct': '37.5%'},
                        'home_advantage': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'strength_of_schedule': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                    },
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
            },
            'Mario_Arce_metrics': None,
            'Vlajic_Josip_metrics': None,
            'Mario_Arce_elo': 1400.0,
            'Vlajic_Josip_elo': 1380.0,
        },
        'form_analysis': {
            'Mario_Arce_form': None,
            'Vlajic_Josip_form': None,
        },
        'enfrentamientos_directos': [],
        'estadisticas': {
            'partidos_Mario_Arce': 0 if hi['p1'] else 14,
            'partidos_Vlajic_Josip': 0 if hi['p2'] else 11,
            'enfrentamientos_totales': 0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# T51-F2-01 a T51-F2-05 — data_contract unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataContract:

    def test_t51_f2_01a_has_empty_history_p1_empty(self):
        """T51-F2-01a: p1 con historial vacío → has_empty_history True."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        assert has_empty_history(partido) is True

    def test_t51_f2_01b_has_empty_history_p2_empty(self):
        """T51-F2-01b: p2 con historial vacío → has_empty_history True."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': True})
        assert has_empty_history(partido) is True

    def test_t51_f2_01c_has_empty_history_both_empty(self):
        """T51-F2-01c: ambos con historial vacío → has_empty_history True."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': True})
        assert has_empty_history(partido) is True

    def test_t51_f2_02_has_empty_history_both_complete(self):
        """T51-F2-02: ambos con datos → has_empty_history False.
        FALLA si se elimina la lectura de historial_incompleto."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        assert has_empty_history(partido) is False

    def test_t51_f2_02b_has_empty_history_no_field(self):
        """T51-F2-02b: sin campo historial_incompleto (legacy) → False (default seguro)."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        del partido['ranking_analysis']['prediction']['historial_incompleto']
        assert has_empty_history(partido) is False

    def test_t51_f2_02c_has_empty_history_pick_status(self):
        """T51-F2-02c: pick del edge report con status='NO_DATA' → has_empty_history True."""
        pick = {'favorito_predicho': 'Mario Arce', 'status': PICK_STATUS_NO_DATA, 'edge': 0.15}
        assert has_empty_history(pick) is True

    def test_t51_f2_03_completeness_score_empty(self):
        """T51-F2-03: score=0.0 cuando algún jugador sin historial.
        Invariante: completeness_score == 0.0 ↔ has_empty_history == True."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        assert completeness_score(partido) == 0.0

    def test_t51_f2_04_completeness_score_complete(self):
        """T51-F2-04: score=1.0 cuando ambos tienen historial."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        assert completeness_score(partido) == 1.0

    def test_t51_f2_05_provenance_constants_valid(self):
        """T51-F2-05: constantes de procedencia son strings no vacíos y distintos."""
        constants = [
            PROVENANCE_NINJA_API, PROVENANCE_THF_CACHE,
            PROVENANCE_PLAYWRIGHT_DOM, PROVENANCE_EMPTY,
        ]
        for c in constants:
            assert isinstance(c, str) and len(c) > 0

        # Todos distintos
        assert len(set(constants)) == 4
        assert PICK_STATUS_NO_DATA == 'NO_DATA'


# ─────────────────────────────────────────────────────────────────────────────
# T51-F2-06 a T51-F2-10 — edge_calculator emite status='NO_DATA'
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCalculatorNoData:
    """Tests sobre calcular_edge_completo() y procesar_archivo_h2h()."""

    def _calcular(self, partido):
        from edge_calculator import calcular_edge_completo, cargar_calibracion
        cal = cargar_calibracion()
        return calcular_edge_completo(partido, cal)

    def test_t51_f2_06_historial_vacio_p1_emite_no_data_status(self):
        """T51-F2-06: calcular_edge_completo con p1 vacío → status='NO_DATA'.
        FALLA si edge_calculator no marca status='NO_DATA' cuando historial vacío."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado.get('status') == PICK_STATUS_NO_DATA, (
            f"status esperado='NO_DATA', obtenido={resultado.get('status')!r}"
        )
        assert resultado['apostar'] is False

    def test_t51_f2_06b_historial_vacio_p2_emite_no_data_status(self):
        """T51-F2-06b: p2 vacío también → status='NO_DATA'."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': True})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado.get('status') == PICK_STATUS_NO_DATA

    def test_t51_f2_06c_historial_completo_no_tiene_no_data_status(self):
        """T51-F2-06c: historial completo → sin status='NO_DATA'.
        No-regresión: partidos con datos no son marcados como NO_DATA."""
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado.get('status') != PICK_STATUS_NO_DATA, (
            "Partido con historial completo NO debe tener status='NO_DATA'"
        )

    def test_t51_f2_07_no_data_not_in_apostar_watchlist_sin_edge(self, tmp_path):
        """T51-F2-07: pick con historial vacío NO aparece en apostar/watchlist/sin_edge.
        FALLA si picks NO_DATA aparecen en alguno de estos pools."""
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        partido_vacio = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        h2h_file = tmp_path / "h2h_test.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': [partido_vacio],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        # El pick NO debe estar en ningún pool normal
        for pool_name in ('apostar', 'watchlist', 'sin_edge'):
            pool = resultado.get(pool_name, [])
            for pick in pool:
                assert pick.get('status') != PICK_STATUS_NO_DATA, (
                    f"Pick NO_DATA encontrado en '{pool_name}' — phantom combo posible. "
                    f"Pick: {pick.get('favorito_predicho', '?')} vs {pick.get('partido', '?')}"
                )

    def test_t51_f2_08_no_data_in_no_data_list(self, tmp_path):
        """T51-F2-08: pick con historial vacío aparece en 'no_data' list.
        FALLA si output de procesar_archivo_h2h no tiene lista 'no_data'."""
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        partido_vacio = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        h2h_file = tmp_path / "h2h_test2.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': [partido_vacio],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        assert 'no_data' in resultado, (
            "Edge report debe tener campo 'no_data' para picks con historial vacío"
        )
        assert len(resultado['no_data']) > 0, (
            "Partido con historial vacío debe aparecer en 'no_data', no en otros pools"
        )

    def test_t51_f2_09_n_no_data_in_metadata(self, tmp_path):
        """T51-F2-09: metadata tiene n_no_data con el conteo correcto."""
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        partido_vacio = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        partido_completo = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        h2h_file = tmp_path / "h2h_test3.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': [partido_vacio, partido_completo],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        assert 'n_no_data' in resultado.get('metadata', {}), (
            "metadata debe tener n_no_data"
        )
        assert resultado['metadata']['n_no_data'] == 1, (
            "n_no_data debe ser 1 (solo el partido con historial vacío)"
        )

    def test_t51_f2_10_complete_history_not_in_no_data(self, tmp_path):
        """T51-F2-10: partido con historial completo NO aparece en 'no_data'.
        No-regresión: picks con datos completos siguen su flujo normal."""
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        partido_completo = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        h2h_file = tmp_path / "h2h_test4.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': [partido_completo],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        no_data_list = resultado.get('no_data', [])
        assert len(no_data_list) == 0, (
            "Partido con historial completo no debe aparecer en 'no_data'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T51-F2-11 a T51-F2-12 — ninja_h2h_parser añade history_provenance
# ─────────────────────────────────────────────────────────────────────────────

class TestNinjaH2HProvenance:
    """Tests sobre _consolidate_result() en ninja_h2h_parser."""

    def _make_extractor(self):
        from scraping.ninja_h2h_parser import NinjaH2HExtractor
        ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)
        ext.all_results = []
        mock_ra = MagicMock()
        mock_ra.get_ranking_metrics.return_value = None
        mock_ra.analyze_rivalry.return_value = {
            'player1_rank': 850, 'player2_rank': 910,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.55, 'p2_rivalry_score': 0.45,
            'prediction': {
                'favored_player': 'P1', 'confidence': 60.0,
                'historial_incompleto': {'p1': False, 'p2': False},
                'scores': {'p1_final_weight': 1.0, 'p2_final_weight': 0.8,
                           'score_difference': 0.2},
                'score_breakdown': {'player1': {}, 'player2': {}},
                'weights_used': {},
                'markov_analysis': None, 'tardio_analysis': None,
                'circuit_asymmetry': None, 'surface_specialization_meta': {},
            },
            'player1_nationality': 'N/A', 'player2_nationality': 'N/A',
            'p1_surface_stats': None, 'p2_surface_stats': None,
            'p1_location_stats': None, 'p2_location_stats': None,
            'player1_advantages': [], 'player2_advantages': [],
        }
        ext.rivalry_analyzer = mock_ra
        return ext

    def _make_match_data(self, source_p1=None, source_p2=None):
        md = {
            'jugador1': 'P1', 'jugador2': 'P2',
            'torneo_nombre': 'Test', 'tipo_cancha': 'hard',
            'torneo_completo': 'Test', 'cuota1': 2.0, 'cuota2': 1.8,
            'match_url': 'https://www.flashscore.co/match/test/ABC/#/h2h',
            'match_id': 'ABC',
        }
        if source_p1 is not None:
            md['_history_source_p1'] = source_p1
        if source_p2 is not None:
            md['_history_source_p2'] = source_p2
        return md

    def _call_consolidate(self, ext, match_data, p1_hist, p2_hist):
        return ext._consolidate_result(
            match_data, p1_hist, p2_hist, [], ext.rivalry_analyzer.analyze_rivalry(),
            None, None, 1400.0, 1380.0
        )

    def test_t51_f2_11_history_provenance_in_data_quality(self):
        """T51-F2-11: _consolidate_result añade history_provenance a data_quality.
        FALLA si ninja_h2h_parser no añade history_provenance a data_quality."""
        ext = self._make_extractor()
        hist = [{'fecha': '01.01.2026', 'oponente': 'X', 'resultado': '2-0',
                 'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'hard',
                 'opponent_ranking': 200, 'opponent_weight': 1}]
        match_data = self._make_match_data(
            source_p1=PROVENANCE_NINJA_API,
            source_p2=PROVENANCE_NINJA_API,
        )

        result = self._call_consolidate(ext, match_data, hist, hist)

        dq = result.get('data_quality', {})
        assert 'history_provenance' in dq, (
            "data_quality debe tener campo 'history_provenance'"
        )
        prov = dq['history_provenance']
        assert 'p1' in prov and 'p2' in prov, (
            "history_provenance debe tener campos 'p1' y 'p2'"
        )

    def test_t51_f2_12_empty_history_provenance_is_empty(self):
        """T51-F2-12: cuando historial vacío y sin source explícito → provenance='EMPTY'."""
        ext = self._make_extractor()
        match_data = self._make_match_data()  # sin _history_source

        result = self._call_consolidate(ext, match_data, [], [])

        dq = result.get('data_quality', {})
        prov = dq.get('history_provenance', {})
        assert prov.get('p1') == PROVENANCE_EMPTY, (
            f"p1 sin historial debe tener provenance='{PROVENANCE_EMPTY}', "
            f"obtenido: {prov.get('p1')!r}"
        )
        assert prov.get('p2') == PROVENANCE_EMPTY

    def test_t51_f2_12b_explicit_source_propagates_correctly(self):
        """T51-F2-12b: source explícito en match_data se propaga a data_quality."""
        ext = self._make_extractor()
        hist = [{'fecha': '01.01.2026', 'oponente': 'X', 'resultado': '2-0',
                 'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'hard',
                 'opponent_ranking': 200, 'opponent_weight': 1}]
        match_data = self._make_match_data(
            source_p1=PROVENANCE_THF_CACHE,
            source_p2=PROVENANCE_PLAYWRIGHT_DOM,
        )

        result = self._call_consolidate(ext, match_data, hist, hist)

        dq = result.get('data_quality', {})
        prov = dq.get('history_provenance', {})
        assert prov.get('p1') == PROVENANCE_THF_CACHE
        assert prov.get('p2') == PROVENANCE_PLAYWRIGHT_DOM


# ─────────────────────────────────────────────────────────────────────────────
# T51-F2-13 — Invariante pools: phantom combos imposibles
# ─────────────────────────────────────────────────────────────────────────────

class TestPhantomComboKilled:
    """
    T51-F2-13: Replica conceptual de la sesión 2026-07-01.

    Arce/Vlajic/Guajardo/Cooper tenían historial vacío → edge=60.4% → Combo1-8 fantasma.
    Con F2: estos picks NUNCA pueden entrar al pool del trader.
    """

    def test_t51_f2_13_no_data_excluded_from_all_pools(self, tmp_path):
        """T51-F2-13: picks con historial vacío NUNCA aparecen en apostar/watchlist/sin_edge.
        FALLA si trader puede acceder a picks NO_DATA (phantom combo posible).

        Invariante: union(apostar, watchlist, sin_edge) ∩ NO_DATA = ∅
        """
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        # 4 picks con historial vacío (réplica julio-1) + 1 pick normal
        picks_fantasma = [
            _make_partido_base(historial_incompleto={'p1': True, 'p2': False}),
            _make_partido_base(historial_incompleto={'p1': False, 'p2': True}),
            _make_partido_base(historial_incompleto={'p1': True, 'p2': True}),
        ]
        pick_normal = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})

        h2h_file = tmp_path / "h2h_julio1.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': picks_fantasma + [pick_normal],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        # Simular lo que el trader hace: leer todos los picks accesibles
        todos_los_picks_accesibles = (
            resultado.get('apostar', []) +
            resultado.get('watchlist', []) +
            resultado.get('sin_edge', [])
        )

        # Invariante F2: NINGÚN pick NO_DATA puede estar en pools accesibles
        no_data_en_pool = [
            p for p in todos_los_picks_accesibles
            if p.get('status') == PICK_STATUS_NO_DATA
        ]
        assert len(no_data_en_pool) == 0, (
            f"PHANTOM COMBO BUG: {len(no_data_en_pool)} pick(s) con status='NO_DATA' "
            f"encontrados en pools accesibles al trader. "
            f"Picks: {[p.get('partido', '?') for p in no_data_en_pool]}"
        )

        # Los 3 picks vacíos deben estar en no_data
        no_data_list = resultado.get('no_data', [])
        assert len(no_data_list) == 3, (
            f"Deben ser 3 picks en no_data (los 3 con historial vacío), "
            f"encontrados: {len(no_data_list)}"
        )

    def test_t51_f2_13b_normal_pick_survives_in_pools(self, tmp_path):
        """T51-F2-13b: pick con historial completo sigue siendo accesible al trader.
        No-regresión: F2 no bloquea picks legítimos."""
        from edge_calculator import procesar_archivo_h2h
        from analysis.rivalry_analyzer import RIVALRY_VERSION

        pick_normal = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        h2h_file = tmp_path / "h2h_normal.json"
        h2h_file.write_text(json.dumps({
            'metadata': {'rivalry_version': RIVALRY_VERSION},
            'partidos': [pick_normal],
        }))

        resultado = procesar_archivo_h2h(str(h2h_file), output_file=None, shadow_log=False)

        todos_los_picks_accesibles = (
            resultado.get('apostar', []) +
            resultado.get('watchlist', []) +
            resultado.get('sin_edge', [])
        )

        # El pick normal debe aparecer en algún pool (apostar, watchlist, o sin_edge)
        assert len(todos_los_picks_accesibles) >= 1, (
            "Pick con historial completo debe ser accesible al trader"
        )

        # Y no debe estar en no_data
        assert len(resultado.get('no_data', [])) == 0, (
            "Pick con historial completo no debe aparecer en 'no_data'"
        )

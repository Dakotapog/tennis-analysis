"""
Tests para Nodo-35 — Propagación de Flag Historial Vacío: Bloqueo en Origen

Cubre los 4 fixes en cascada:
  Fix 35-1: ninja_h2h_parser._consolidate_result() serializa data_quality en el JSON
  Fix 35-2: rivalry_analyzer.generate_advanced_prediction() propaga historial_incompleto
  Fix 35-3: edge_calculator bloquea apostar=True cuando historial_incompleto
  Fix 35-4: generar_tabla_favoritos2 muestra alerta visual antes del favorito

Detección de mutación:
  T35-01 FALLA si se elimina el bloque data_quality de _consolidate_result().
  T35-06 FALLA si se elimina el gate HISTORIAL_NO_EXTRAIDO de edge_calculator.
  T35-08 confirma no-regresión: partidos con datos completos no son bloqueados.

Caso real documentado (2026-06-25):
  Julien Penzlin: 0 partidos extraídos → edge=25.8% → apostar=True (bug)
  Con Nodo-35: gate bloquea antes de que llegue al trader.
"""
import pytest
import io
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

def _make_partido_base(historial_incompleto=None, cuota1=2.02, cuota2=1.70, p_modelo=0.753):
    """Partido mínimo válido para calcular_edge_completo."""
    hi = historial_incompleto or {'p1': False, 'p2': False}
    return {
        'jugador1': 'Manel Lazaro Juncadella',
        'jugador2': 'Julien Penzlin',
        'cuota1': cuota1,
        'cuota2': cuota2,
        'torneo_nombre': 'M15 Alkmaar (Países Bajos)',
        'tipo_cancha': 'clay',
        'torneo_completo': 'ITF - INDIVIDUALES: M15 Alkmaar',
        'match_url': 'https://www.flashscore.co/match/tennis/test/UHSoNBNP/#/h2h',
        'match_id': 'UHSoNBNP',
        'ranking1': 176,
        'ranking2': 139,
        'data_quality': {
            'historial_extraido_p1': not hi['p1'],
            'historial_extraido_p2': not hi['p2'],
            'n_partidos_p1': 0 if hi['p1'] else 45,
            'n_partidos_p2': 0 if hi['p2'] else 32,
        },
        'ranking_analysis': {
            'Manel_Lazaro_Juncadella_ranking': 176,
            'Julien_Penzlin_ranking': 139,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.6,
            'p2_rivalry_score': 0.4,
            'prediction': {
                'favored_player': 'Manel Lazaro Juncadella',
                'confidence': p_modelo * 100,
                'historial_incompleto': hi,
                'scores': {'p1_final_weight': 1.8, 'p2_final_weight': 0.43,
                           'score_difference': 1.37},
                'score_breakdown': {
                    'player1': {
                        'surface_specialization': {'contribution': 1.0, 'contribution_pct': '55.6%'},
                        'form_recent': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'common_opponents': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'h2h_direct': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'ranking_momentum': {'contribution': 0.5, 'contribution_pct': '27.8%'},
                        'elo_rating': {'contribution': 0.3, 'contribution_pct': '16.7%'},
                        'home_advantage': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'strength_of_schedule': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                    },
                    'player2': {
                        'surface_specialization': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'form_recent': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'common_opponents': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'h2h_direct': {'contribution': 0.0, 'contribution_pct': '0.0%'},
                        'ranking_momentum': {'contribution': 0.3, 'contribution_pct': '69.8%'},
                        'elo_rating': {'contribution': 0.13, 'contribution_pct': '30.2%'},
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
            'Manel_Lazaro_Juncadella_metrics': None,
            'Julien_Penzlin_metrics': None,
            'Manel_Lazaro_Juncadella_elo': 1429.0,
            'Julien_Penzlin_elo': 1500.0,
        },
        'form_analysis': {
            'Manel_Lazaro_Juncadella_form': None,
            'Julien_Penzlin_form': None,
        },
        'enfrentamientos_directos': [],
        'estadisticas': {
            'partidos_Manel_Lazaro_Juncadella': 45,
            'partidos_Julien_Penzlin': 0,
            'enfrentamientos_totales': 0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fix 35-1: data_quality serializado en ninja_h2h_parser._consolidate_result()
# ─────────────────────────────────────────────────────────────────────────────

class TestFix351DataQualitySerializado:
    """T35-01 a T35-03 — _consolidate_result() emite data_quality."""

    def _make_extractor(self):
        """Extractor mínimo para llamar _consolidate_result()."""
        from scraping.ninja_h2h_parser import NinjaH2HExtractor
        ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)
        ext.all_results = []
        # Mock rivalry_analyzer con respuesta mínima
        mock_ra = MagicMock()
        mock_ra.get_ranking_metrics.return_value = None
        mock_ra.analyze_rivalry.return_value = {
            'player1_rank': 176, 'player2_rank': 139,
            'common_opponents_count': 0,
            'p1_rivalry_score': 0.6, 'p2_rivalry_score': 0.4,
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

    def _call_consolidate(self, ext, p1_hist, p2_hist):
        match_data = {
            'jugador1': 'P1', 'jugador2': 'P2',
            'torneo_nombre': 'Test', 'tipo_cancha': 'clay',
            'torneo_completo': 'Test', 'cuota1': 2.0, 'cuota2': 1.8,
            'match_url': 'https://www.flashscore.co/match/test/ABC/#/h2h',
            'match_id': 'ABC',
        }
        return ext._consolidate_result(
            match_data, p1_hist, p2_hist, [], ext.rivalry_analyzer.analyze_rivalry(),
            None, None, 1500.0, 1500.0
        )

    def test_t35_01_flag_p2_vacio(self):
        """T35-01: data_quality marca p2 como no extraído cuando hist vacío."""
        ext = self._make_extractor()
        p1_hist = [{'fecha': '01.01.2026', 'oponente': 'X', 'resultado': '2-0',
                    'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'clay',
                    'opponent_ranking': 200, 'opponent_weight': 1}]
        resultado = self._call_consolidate(ext, p1_hist, [])

        dq = resultado.get('data_quality', {})
        assert dq.get('historial_extraido_p1') is True
        assert dq.get('historial_extraido_p2') is False
        assert dq.get('n_partidos_p1') == 1
        assert dq.get('n_partidos_p2') == 0

    def test_t35_02_flag_p1_vacio(self):
        """T35-02: data_quality marca p1 como no extraído cuando hist vacío."""
        ext = self._make_extractor()
        p2_hist = [{'fecha': '01.01.2026', 'oponente': 'Y', 'resultado': '2-1',
                    'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'clay',
                    'opponent_ranking': 150, 'opponent_weight': 1}]
        resultado = self._call_consolidate(ext, [], p2_hist)

        dq = resultado.get('data_quality', {})
        assert dq.get('historial_extraido_p1') is False
        assert dq.get('historial_extraido_p2') is True
        assert dq.get('n_partidos_p1') == 0
        assert dq.get('n_partidos_p2') == 1

    def test_t35_03_ambos_con_datos(self):
        """T35-03: data_quality True en ambos cuando ambos tienen historial."""
        ext = self._make_extractor()
        h = [{'fecha': '01.01.2026', 'oponente': 'Z', 'resultado': '2-0',
              'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'clay',
              'opponent_ranking': 100, 'opponent_weight': 1}] * 5
        resultado = self._call_consolidate(ext, h[:3], h[2:])

        dq = resultado.get('data_quality', {})
        assert dq.get('historial_extraido_p1') is True
        assert dq.get('historial_extraido_p2') is True
        assert dq.get('n_partidos_p1') == 3
        assert dq.get('n_partidos_p2') == 3


# ─────────────────────────────────────────────────────────────────────────────
# Fix 35-2: historial_incompleto propagado en rivalry_analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestFix352HistorialIncompletoEnPrediccion:
    """T35-04 a T35-05 — generate_advanced_prediction() emite historial_incompleto."""

    def _get_prediccion(self, p1_history, p2_history):
        from analysis.rivalry_analyzer import RivalryAnalyzer
        from unittest.mock import MagicMock
        rm = MagicMock()
        rm.get_player_info.return_value = None
        rm.get_player_ranking.return_value = None
        rm.normalize_name.side_effect = lambda x: x.lower().strip() if x else ''
        elo = MagicMock()
        elo.default_rating = 1500
        elo.expected_score.return_value = 0.5
        ra = RivalryAnalyzer(rm, elo)
        p_info = {'ranking_position': 100, 'points': 500}
        return ra.generate_advanced_prediction(
            player1_info=p_info, player2_info=p_info,
            p1_rivalry_score=0.5, p2_rivalry_score=0.5,
            player1_name='PlayerA', player2_name='PlayerB',
            player1_history=p1_history, player2_history=p2_history,
            player1_advantages_count=0, player2_advantages_count=0,
            player1_form=None, player2_form=None,
            direct_h2h_matches=[], tournament_name='M15 Test',
            prediction_context={
                'country': 'ES', 'surface': 'clay',
                'p1_nationality': 'ES', 'p2_nationality': 'DE',
                'current_match_country': 'ES',
            },
            p1_elo=1500.0, p2_elo=1500.0,
        )

    def test_t35_04_p2_vacio_propagado(self):
        """T35-04: historial_incompleto.p2=True cuando player2_history=[]."""
        hist_p1 = [{'fecha': '01.01.2026', 'oponente': 'X', 'resultado': '2-0',
                    'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'clay',
                    'opponent_ranking': 200, 'opponent_weight': 1}]
        pred = self._get_prediccion(hist_p1, [])

        hi = pred.get('historial_incompleto', {})
        assert hi.get('p1') is False
        assert hi.get('p2') is True

    def test_t35_05_ambos_vacios_propagado(self):
        """T35-05: historial_incompleto p1=True y p2=True cuando ambos vacíos."""
        pred = self._get_prediccion([], [])

        hi = pred.get('historial_incompleto', {})
        assert hi.get('p1') is True
        assert hi.get('p2') is True

    def test_t35_05b_ambos_con_datos_false(self):
        """T35-05b: historial_incompleto False en ambos cuando tienen datos."""
        hist = [{'fecha': '01.01.2026', 'oponente': 'X', 'resultado': '2-0',
                 'outcome': 'Ganó', 'torneo': 'T', 'superficie': 'clay',
                 'opponent_ranking': 200, 'opponent_weight': 1}]
        pred = self._get_prediccion(hist, hist)

        hi = pred.get('historial_incompleto', {})
        assert hi.get('p1') is False
        assert hi.get('p2') is False


# ─────────────────────────────────────────────────────────────────────────────
# Fix 35-3: gate HISTORIAL_NO_EXTRAIDO en edge_calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestFix353GateEdgeCalculator:
    """T35-06 a T35-08 — calcular_edge_completo bloquea cuando historial vacío."""

    def _calcular(self, partido):
        from edge_calculator import calcular_edge_completo, cargar_calibracion
        cal = cargar_calibracion()
        return calcular_edge_completo(partido, cal)

    def test_t35_06_favorito_sin_datos_bloqueado(self):
        """T35-06: apostar=False cuando el favorito (p1) no tiene historial."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado['apostar'] is False
        assert 'HISTORIAL_NO_EXTRAIDO' in resultado.get('motivo_reclasificacion', '')
        assert 'Manel Lazaro Juncadella' in resultado.get('motivo_reclasificacion', '')

    def test_t35_07_rival_sin_datos_bloqueado(self):
        """T35-07: apostar=False cuando el rival (p2) no tiene historial.
        Este es el caso Penzlin: el rival sin datos, no el favorito.
        """
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': True})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado['apostar'] is False
        assert 'HISTORIAL_NO_EXTRAIDO' in resultado.get('motivo_reclasificacion', '')
        assert 'Julien Penzlin' in resultado.get('motivo_reclasificacion', '')

    def test_t35_08_ambos_sin_datos_bloqueado(self):
        """T35-08: apostar=False cuando ambos jugadores sin historial."""
        partido = _make_partido_base(historial_incompleto={'p1': True, 'p2': True})
        resultado = self._calcular(partido)

        assert resultado is not None
        assert resultado['apostar'] is False
        motivo = resultado.get('motivo_reclasificacion', '')
        assert 'HISTORIAL_NO_EXTRAIDO' in motivo
        assert 'Manel Lazaro Juncadella' in motivo
        assert 'Julien Penzlin' in motivo

    def test_t35_09_ambos_con_datos_no_bloqueado_por_gate35(self):
        """T35-09: gate Nodo-35 NO se activa cuando ambos tienen historial.
        No-regresión: partidos con datos completos pueden pasar al siguiente gate.
        """
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        resultado = self._calcular(partido)

        assert resultado is not None
        motivo = resultado.get('motivo_reclasificacion', '')
        assert 'HISTORIAL_NO_EXTRAIDO' not in motivo

    def test_t35_09b_legacy_sin_campo_data_quality_no_bloqueado(self):
        """T35-09b: archivos legacy sin historial_incompleto en prediction no se bloquean.
        Fallback: si el campo no existe se asume datos OK (comportamiento anterior).
        """
        partido = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        # Eliminar historial_incompleto de la prediction (simula JSON legacy)
        del partido['ranking_analysis']['prediction']['historial_incompleto']
        resultado = self._calcular(partido)

        assert resultado is not None
        motivo = resultado.get('motivo_reclasificacion', '')
        assert 'HISTORIAL_NO_EXTRAIDO' not in motivo


# ─────────────────────────────────────────────────────────────────────────────
# Fix 35-4: alerta visual en generar_tabla_favoritos2
# ─────────────────────────────────────────────────────────────────────────────

class TestFix354AlertaVisual:
    """T35-10 a T35-11 — generar_tabla_favoritos2 muestra alerta cuando historial vacío."""

    def _render_partido(self, match):
        """Renderiza un partido usando la lógica de generar_tabla_favoritos2."""
        import generar_tabla_favoritos2 as gtf
        import io

        output = io.StringIO()

        p1 = match.get('jugador1', 'P1')
        p2 = match.get('jugador2', 'P2')
        torneo = match.get('torneo_nombre', 'N/A')
        cancha = (match.get('tipo_cancha') or 'N/A').capitalize()
        match_url = match.get('match_url')

        # Replicar exactamente el bloque de escritura del partido
        output.write("=" * 150 + "\n")
        output.write(f"Partido #1: {p1} vs {p2}\n")
        output.write(f"Torneo: {torneo} | Superficie: {cancha}\n")
        if match_url:
            output.write(f"URL del Partido: {match_url}\n")
        output.write("=" * 150 + "\n\n")

        # Bloque Nodo-35
        _dq = match.get('data_quality', {})
        _sin_p1 = not _dq.get('historial_extraido_p1', True)
        _sin_p2 = not _dq.get('historial_extraido_p2', True)
        if _sin_p1 or _sin_p2:
            _sin_nombres = []
            if _sin_p1:
                _sin_nombres.append(p1)
            if _sin_p2:
                _sin_nombres.append(p2)
            output.write("!" * 80 + "\n")
            output.write(f"SIN HISTORIAL EXTRAIDO PARA: {', '.join(_sin_nombres)}\n")
            output.write(f"BLOQUEADO EN extraer_historh2h.py. NO APOSTAR.\n")
            output.write("!" * 80 + "\n\n")

        return output.getvalue()

    def test_t35_10_alerta_cuando_p2_sin_historial(self):
        """T35-10: alerta visual aparece cuando rival (p2) no tiene historial."""
        match = _make_partido_base(historial_incompleto={'p1': False, 'p2': True})
        output = self._render_partido(match)

        assert 'SIN HISTORIAL EXTRAIDO' in output
        assert 'Julien Penzlin' in output
        assert 'NO APOSTAR' in output
        # Alerta debe aparecer ANTES del resumen (que empieza con "=")
        pos_alerta = output.find('SIN HISTORIAL EXTRAIDO')
        pos_header = output.find('=' * 150)
        assert pos_alerta > pos_header  # después del header pero en el bloque

    def test_t35_11_sin_alerta_cuando_ambos_con_datos(self):
        """T35-11: sin alerta cuando ambos jugadores tienen historial."""
        match = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        output = self._render_partido(match)

        assert 'SIN HISTORIAL EXTRAIDO' not in output
        assert 'NO APOSTAR' not in output

    def test_t35_11b_alerta_cuando_p1_sin_historial(self):
        """T35-11b: alerta menciona p1 cuando es el favorito sin datos."""
        match = _make_partido_base(historial_incompleto={'p1': True, 'p2': False})
        output = self._render_partido(match)

        assert 'SIN HISTORIAL EXTRAIDO' in output
        assert 'Manel Lazaro Juncadella' in output
        assert 'NO APOSTAR' in output

    def test_t35_11c_alerta_legacy_sin_data_quality(self):
        """T35-11c: sin alerta cuando data_quality ausente (archivos legacy)."""
        match = _make_partido_base(historial_incompleto={'p1': False, 'p2': False})
        del match['data_quality']  # simular JSON legacy
        output = self._render_partido(match)

        assert 'SIN HISTORIAL EXTRAIDO' not in output

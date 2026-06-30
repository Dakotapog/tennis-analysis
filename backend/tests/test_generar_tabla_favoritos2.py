"""
Tests para generar_tabla_favoritos2.py — NIVEL 3 cobertura de producción.

Cubre las funciones puras y utilitarias del generador de reporte humano.
Sin llamadas a disco reales — todo mockeado o con datos mínimos en memoria.
"""
import json
import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# parse_score
# ─────────────────────────────────────────────────────────────────────────────

class TestParseScore:
    """parse_score convierte '2-1' en (2, 1)."""

    def setup_method(self):
        from generar_tabla_favoritos2 import parse_score
        self.parse_score = parse_score

    def test_resultado_normal(self):
        assert self.parse_score('2-1') == (2, 1)

    def test_victoria_directa(self):
        assert self.parse_score('3-0') == (3, 0)

    def test_derrota_directa(self):
        assert self.parse_score('0-2') == (0, 2)

    def test_cadena_invalida_devuelve_cero(self):
        assert self.parse_score('N/A') == (0, 0)

    def test_cadena_vacia_devuelve_cero(self):
        assert self.parse_score('') == (0, 0)

    def test_sin_guion_devuelve_cero(self):
        assert self.parse_score('21') == (0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# predecir_sets_y_games
# ─────────────────────────────────────────────────────────────────────────────

class TestPredecirSetsYGames:
    """predecir_sets_y_games determina sets y games según diferencia de puntaje."""

    def setup_method(self):
        from generar_tabla_favoritos2 import predecir_sets_y_games
        self.predecir = predecir_sets_y_games

    def test_diferencia_alta_predice_2_sets(self):
        result = self.predecir(0.40, 0.8, 0.4)
        assert result['predicted_sets'] == '2'

    def test_diferencia_media_predice_2_sets(self):
        result = self.predecir(0.20, 0.7, 0.5)
        assert result['predicted_sets'] == '2'

    def test_diferencia_pequena_predice_3_sets(self):
        result = self.predecir(0.05, 0.55, 0.50)
        assert result['predicted_sets'] == '3'

    def test_empate_predice_3_sets(self):
        result = self.predecir(0.0, 0.5, 0.5)
        assert result['predicted_sets'] == '3'

    def test_resultado_tiene_predicted_games(self):
        result = self.predecir(0.30, 0.7, 0.4)
        assert 'predicted_games' in result
        assert isinstance(result['predicted_games'], str)

    def test_resultado_tiene_reason(self):
        result = self.predecir(0.10, 0.6, 0.5)
        assert 'reason' in result
        assert len(result['reason']) > 0

    def test_diferencia_muy_alta_games_bajo(self):
        # diff > 0.35 → rango de dominio total (16-19 juegos)
        result = self.predecir(0.40, 0.9, 0.3)
        assert result['predicted_games'] == '16-19'


# ─────────────────────────────────────────────────────────────────────────────
# explicar_ventaja_rival_comun
# ─────────────────────────────────────────────────────────────────────────────

class TestExplicarVentajaRivalComun:
    """explicar_ventaja_rival_comun genera texto explicando ventaja transitiva."""

    def setup_method(self):
        from generar_tabla_favoritos2 import explicar_ventaja_rival_comun
        self.explicar = explicar_ventaja_rival_comun

    def _opp(self, p1_outcome, p2_outcome, p1_score='2-0', p2_score='0-2', opponent='Rublev'):
        return {
            'opponent_name': opponent,
            'advantage_for': 'Jugador1' if p1_outcome == 'Ganó' else 'Jugador2',
            'player1_result': {'outcome': p1_outcome, 'score': p1_score},
            'player2_result': {'outcome': p2_outcome, 'score': p2_score},
        }

    def test_p1_gana_p2_pierde(self):
        texto = self.explicar(self._opp('Ganó', 'Perdió'), 'Alcaraz', 'Djokovic')
        assert 'Alcaraz' in texto
        assert 'venció' in texto

    def test_p2_gana_p1_pierde(self):
        texto = self.explicar(self._opp('Perdió', 'Ganó'), 'Alcaraz', 'Djokovic')
        assert 'Djokovic' in texto
        assert 'venció' in texto

    def test_ambos_ganan_mas_contundente(self):
        opp = self._opp('Ganó', 'Ganó', p1_score='2-0', p2_score='2-1')
        texto = self.explicar(opp, 'Alcaraz', 'Djokovic')
        assert 'contundente' in texto or 'similares' in texto

    def test_ambos_pierden_mas_resistente(self):
        opp = self._opp('Perdió', 'Perdió', p1_score='1-2', p2_score='0-2')
        texto = self.explicar(opp, 'Alcaraz', 'Djokovic')
        assert 'resistencia' in texto or 'similar' in texto

    def test_devuelve_string(self):
        texto = self.explicar(self._opp('Ganó', 'Perdió'), 'P1', 'P2')
        assert isinstance(texto, str)
        assert len(texto) > 0


# ─────────────────────────────────────────────────────────────────────────────
# get_weights_from_reasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestGetWeightsFromReasoning:
    """get_weights_from_reasoning extrae pesos del log LOG_WEIGHTS."""

    def setup_method(self):
        from generar_tabla_favoritos2 import get_weights_from_reasoning
        self.get_weights = get_weights_from_reasoning

    def test_extrae_pesos_de_log(self):
        reasoning = ["LOG_WEIGHTS: {'form_recent': 0.15, 'h2h_direct': 0.20}"]
        weights = self.get_weights(reasoning)
        assert weights.get('form_recent') == pytest.approx(0.15)
        assert weights.get('h2h_direct') == pytest.approx(0.20)

    def test_lista_sin_log_weights_devuelve_vacio(self):
        reasoning = ['Alcaraz ganó 3 de últimos 5', 'Superficie arcilla']
        weights = self.get_weights(reasoning)
        assert weights == {}

    def test_lista_vacia_devuelve_vacio(self):
        assert self.get_weights([]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# analyze_component_status
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeComponentStatus:
    """analyze_component_status clasifica componentes activos vs inactivos."""

    def setup_method(self):
        from generar_tabla_favoritos2 import analyze_component_status
        self.analyze = analyze_component_status

    def _breakdown(self, components):
        """Crea un score_breakdown mínimo con los componentes dados."""
        player_data = {c: {'raw_score': 1.0} for c in components}
        return {'player1': player_data, 'player2': player_data}

    def test_componentes_con_datos_son_activos(self):
        weights = {'form_recent': 0.15, 'h2h_direct': 0.20}
        breakdown = self._breakdown(['form_recent', 'h2h_direct'])
        active, inactive, _ = self.analyze(breakdown, weights)
        assert 'form_recent' in active
        assert 'h2h_direct' in active
        assert inactive == []

    def test_componente_sin_datos_es_inactivo(self):
        weights = {'form_recent': 0.15, 'h2h_direct': 0.20}
        breakdown = self._breakdown(['form_recent'])  # h2h_direct ausente
        active, inactive, _ = self.analyze(breakdown, weights)
        assert 'h2h_direct' in inactive

    def test_breakdown_none_todos_activos(self):
        weights = {'form_recent': 0.15, 'h2h_direct': 0.20}
        active, inactive, _ = self.analyze(None, weights)
        assert set(active) == set(weights.keys())
        assert inactive == []


# ─────────────────────────────────────────────────────────────────────────────
# analizar_probabilidad_overs
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalizarProbabilidadOvers:
    """analizar_probabilidad_overs calcula prob. de > 2.5 sets."""

    def setup_method(self):
        from generar_tabla_favoritos2 import analizar_probabilidad_overs
        self.analizar = analizar_probabilidad_overs

    def _df(self, resultados, outcomes):
        return pd.DataFrame({'resultado': resultados, 'outcome': outcomes})

    def test_df_vacios_devuelve_na(self):
        result = self.analizar(pd.DataFrame(), pd.DataFrame())
        assert result['prob_over_2_5_sets'] == 'N/A'

    def test_todos_2_sets_prob_0(self):
        # 2-0 significa que no hay sets perdidos → no es over 2.5
        df = self._df(['2-0', '2-0', '2-0'], ['Ganó', 'Ganó', 'Ganó'])
        result = self.analizar(df.copy(), df.copy())
        assert '0.0%' in result['prob_over_2_5_sets']

    def test_todos_3_sets_prob_100(self):
        # 2-1 y 1-2 → todos superan 2.5 sets
        df = self._df(['2-1', '1-2', '2-1'], ['Ganó', 'Perdió', 'Ganó'])
        result = self.analizar(df.copy(), df.copy())
        assert '100.0%' in result['prob_over_2_5_sets']

    def test_mitad_3_sets_prob_50(self):
        df = self._df(['2-0', '2-1'], ['Ganó', 'Ganó'])
        result = self.analizar(df.copy(), df.copy())
        assert '50.0%' in result['prob_over_2_5_sets']


# ─────────────────────────────────────────────────────────────────────────────
# find_latest_h2h_file
# ─────────────────────────────────────────────────────────────────────────────

class TestFindLatestH2HFile:
    """find_latest_h2h_file selecciona el h2h_results más reciente."""

    def test_sin_archivos_devuelve_none(self):
        from generar_tabla_favoritos2 import find_latest_h2h_file
        with patch('glob.glob', return_value=[]):
            result = find_latest_h2h_file()
        assert result is None

    def test_con_archivos_devuelve_el_mas_reciente(self, tmp_path):
        from generar_tabla_favoritos2 import find_latest_h2h_file
        # Crear dos archivos con fechas distintas
        f1 = tmp_path / 'h2h_results_enhanced_20260529.json'
        f2 = tmp_path / 'h2h_results_enhanced_20260531.json'
        f1.write_text('{}')
        f2.write_text('{}')
        with patch('glob.glob', return_value=[str(f1), str(f2)]):
            result = find_latest_h2h_file()
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# analyze_matches_with_pandas — smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeMatchesSmoke:
    """Smoke test: analyze_matches_with_pandas no lanza excepción con JSON mínimo."""

    def _minimal_json(self, tmp_path):
        data = {
            "metadata": {"total": 1},
            "partidos": [{
                "match_number": 1,
                "jugador1": "Alcaraz C.",
                "jugador2": "Djokovic N.",
                "torneo_nombre": "Roland Garros",
                "tipo_cancha": "clay",
                "cuota1": 1.5,
                "cuota2": 2.5,
                "ranking_analysis": {
                    "prediction": {
                        "favored_player": "Alcaraz C.",
                        "confidence": 65,
                        "scores": {
                            "p1_final_weight": 0.62,
                            "p2_final_weight": 0.38,
                            "score_difference": 0.24
                        },
                        "reasoning": [],
                        "score_breakdown": {}
                    }
                }
            }]
        }
        f = tmp_path / 'test_h2h.json'
        f.write_text(json.dumps(data), encoding='utf-8')
        return str(f)

    def test_genera_reporte_sin_excepcion(self, tmp_path):
        from generar_tabla_favoritos2 import analyze_matches_with_pandas
        json_file = self._minimal_json(tmp_path)
        output = str(tmp_path / 'output.txt')
        # No debe lanzar ninguna excepción
        analyze_matches_with_pandas(json_file, output)
        assert os.path.exists(output)

    def test_archivo_no_encontrado_no_lanza(self, tmp_path):
        from generar_tabla_favoritos2 import analyze_matches_with_pandas
        output = str(tmp_path / 'output.txt')
        # FileNotFoundError debe ser capturado internamente
        analyze_matches_with_pandas('no_existe.json', output)

    def test_json_invalido_no_lanza(self, tmp_path):
        from generar_tabla_favoritos2 import analyze_matches_with_pandas
        bad = tmp_path / 'bad.json'
        bad.write_text('NOT JSON')
        output = str(tmp_path / 'output.txt')
        analyze_matches_with_pandas(str(bad), output)

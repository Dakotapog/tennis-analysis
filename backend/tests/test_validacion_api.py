"""
Tests para validar_con_api.py (Nodo-05)
Cubre: parser FlashScore, validación individual, accuracy, superficie.
Sin llamadas HTTP reales — todo mockeado.
"""
import pytest
from unittest.mock import patch, MagicMock

from validar_con_api import (
    parsear_respuesta_flashscore,
    obtener_resultado_partido,
    validar_partido_individual,
    calcular_accuracy,
    accuracy_por_superficie,
)


# ─────────────────────────────────────────────────────────────────────────────
# parsear_respuesta_flashscore
# ─────────────────────────────────────────────────────────────────────────────

class TestParserFlashscore:

    def test_formato_basico(self):
        raw = "~AA÷100¬~BH÷2¬~BI÷1"
        d = parsear_respuesta_flashscore(raw)
        assert d['~AA'] == '100'
        assert d['~BH'] == '2'
        assert d['~BI'] == '1'

    def test_string_vacio(self):
        assert parsear_respuesta_flashscore('') == {}

    def test_sin_separador_div(self):
        """Par sin ÷ se ignora."""
        raw = "GARBAGE¬~AA÷FT"
        d = parsear_respuesta_flashscore(raw)
        assert '~AA' in d
        assert 'GARBAGE' not in d

    def test_valor_con_div_extra(self):
        """Si el valor contiene ÷, solo se parte en el primero."""
        raw = "~TN÷Roland÷Garros"
        d = parsear_respuesta_flashscore(raw)
        assert d['~TN'] == 'Roland÷Garros'

    def test_multiples_pares(self):
        raw = "A÷1¬B÷2¬C÷3¬D÷4"
        d = parsear_respuesta_flashscore(raw)
        assert len(d) == 4
        assert d['D'] == '4'

    def test_partido_terminado_status_100(self):
        raw = "~AA÷100¬~BH÷2¬~BI÷0"
        d = parsear_respuesta_flashscore(raw)
        assert d['~AA'] == '100'
        assert int(d['~BH']) > int(d['~BI'])


# ─────────────────────────────────────────────────────────────────────────────
# obtener_resultado_partido
# ─────────────────────────────────────────────────────────────────────────────

class TestObtenerResultado:
    """
    Tests actualizados al formato real del endpoint dc_1 (Nodo-09, 2026-05-29).
    Claves reales: DJ (ganador), DE (sets local), DF (sets visitante), DC (timestamp).
    DC_PASADO = 2020-01-01 → match ya debería haber empezado (LIVE si DJ vacío).
    DC_FUTURO = 2099-01-01 → match aún no inicia (NS si DJ vacío).
    """
    DC_PASADO = '1577836800'   # 2020-01-01 00:00:00 UTC
    DC_FUTURO = '4070908800'   # 2099-01-01 00:00:00 UTC

    def _mock_response(self, text: str, status_code: int = 200):
        mock = MagicMock()
        mock.text = text
        mock.status_code = status_code
        mock.raise_for_status.return_value = None
        return mock

    def test_match_id_invalido_tennis(self):
        r = obtener_resultado_partido('tennis')
        assert r['status'] == 'INVALID_ID'

    def test_match_id_vacio(self):
        r = obtener_resultado_partido('')
        assert r['status'] == 'INVALID_ID'

    def test_partido_terminado_jugador1_gana(self):
        """DJ=H → local (jugador1) ganó → FT."""
        raw = f"DJ÷H¬DE÷2¬DF÷1¬DC÷{self.DC_PASADO}"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('ETKIzZPG')
        assert r['status'] == 'FT'
        assert r['ganador_lado'] == 'jugador1'
        assert r['sets_local'] == '2'

    def test_partido_terminado_jugador2_gana(self):
        """DJ=A → visitante (jugador2) ganó → FT."""
        raw = f"DJ÷A¬DE÷1¬DF÷3¬DC÷{self.DC_PASADO}"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('CW6rxGQF')
        assert r['status'] == 'FT'
        assert r['ganador_lado'] == 'jugador2'
        assert r['sets_visitante'] == '3'

    def test_partido_no_iniciado(self):
        """DJ vacío + DC en el futuro → NS."""
        raw = f"DJ÷¬DC÷{self.DC_FUTURO}¬DV÷2"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('abc123')
        assert r['status'] == 'NS'

    def test_partido_en_vivo(self):
        """DJ vacío + DC en el pasado → LIVE."""
        raw = f"DJ÷¬DC÷{self.DC_PASADO}¬DV÷2"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('abc123')
        assert r['status'] == 'LIVE'

    def test_dv_no_es_indicador_de_estado(self):
        """DV=2 es constante de tipo tenis, no indica 'segundo set en juego'.
        Los tres partidos reales del 2026-05-29 terminados mostraban DV=2."""
        raw = f"DJ÷H¬DE÷2¬DF÷0¬DC÷{self.DC_PASADO}¬DV÷2"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('ETKIzZPG')
        assert r['status'] == 'FT'
        assert r['ganador_lado'] == 'jugador1'

    def test_formato_real_dc1_endpoint(self):
        """Parser maneja correctamente el formato real del endpoint dc_1."""
        raw = f"DA÷3¬DC÷{self.DC_PASADO}¬DE÷2¬DF÷0¬DJ÷H¬DN÷7¬DO÷5¬DV÷2¬~"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('ETKIzZPG')
        assert r['status'] == 'FT'
        assert r['ganador_lado'] == 'jugador1'
        assert r['raw_data']['DE'] == '2'
        assert r['raw_data']['DJ'] == 'H'

    def test_error_http(self):
        with patch('validar_con_api.requests.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.raise_for_status.side_effect = \
                __import__('requests').exceptions.HTTPError(response=mock_resp)
            mock_get.return_value = mock_resp
            r = obtener_resultado_partido('bad_id')
        assert r['status'] == 'ERROR'
        assert '404' in r['error']

    def test_error_conexion(self):
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("timeout")
            r = obtener_resultado_partido('abc123')
        assert r['status'] == 'ERROR'

    def test_raw_data_incluido_en_ft(self):
        raw = f"DJ÷H¬DE÷2¬DF÷1¬DC÷{self.DC_PASADO}"
        with patch('validar_con_api.requests.get') as mock_get:
            mock_get.return_value = self._mock_response(raw)
            r = obtener_resultado_partido('abc123')
        assert 'raw_data' in r
        assert isinstance(r['raw_data'], dict)


# ─────────────────────────────────────────────────────────────────────────────
# validar_partido_individual
# ─────────────────────────────────────────────────────────────────────────────

class TestValidarPartidoIndividual:

    def _partido(self, jugador1='Tsitsipas S.', jugador2='Nadal R.',
                 match_id='real123', pred_ganador=None):
        return {
            'jugador1': jugador1,
            'jugador2': jugador2,
            'match_id': match_id,
            'torneo': 'Roland Garros',
            'superficie': 'clay',
            'ranking_analysis': {
                'prediction': {
                    'favored_player': pred_ganador or jugador1,
                    'confidence': 0.65,
                }
            }
        }

    def test_match_id_tennis_retorna_none(self):
        partido = self._partido(match_id='tennis')
        assert validar_partido_individual(partido) is None

    def test_match_id_vacio_retorna_none(self):
        partido = self._partido(match_id='')
        assert validar_partido_individual(partido) is None

    def test_sin_prediccion_retorna_none(self):
        partido = self._partido()
        partido['ranking_analysis']['prediction']['favored_player'] = None
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        assert validar_partido_individual(partido, resultado_api) is None

    def test_partido_no_terminado_retorna_none(self):
        partido = self._partido()
        resultado_api = {'status': 'LIVE'}
        assert validar_partido_individual(partido, resultado_api) is None

    def test_partido_no_iniciado_retorna_none(self):
        partido = self._partido()
        resultado_api = {'status': 'NS'}
        assert validar_partido_individual(partido, resultado_api) is None

    def test_prediccion_correcta(self):
        partido = self._partido(jugador1='Tsitsipas S.', pred_ganador='Tsitsipas S.')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['correcto'] is True
        assert r['resultado_real'] == 'Tsitsipas S.'

    def test_prediccion_incorrecta(self):
        partido = self._partido(jugador1='Tsitsipas S.', jugador2='Nadal R.',
                                pred_ganador='Tsitsipas S.')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador2'}
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['correcto'] is False
        assert r['resultado_real'] == 'Nadal R.'

    def test_superficie_preservada(self):
        partido = self._partido()
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r['superficie'] == 'clay'

    def test_torneo_preservado(self):
        partido = self._partido()
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r['torneo'] == 'Roland Garros'

    def test_match_id_preservado(self):
        partido = self._partido(match_id='rDQ3y6to')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r['match_id'] == 'rDQ3y6to'

    def test_confianza_incluida(self):
        partido = self._partido()
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert 'confianza' in r


# ─────────────────────────────────────────────────────────────────────────────
# calcular_accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcularAccuracy:

    def test_lista_vacia(self):
        assert calcular_accuracy([]) == 0.0

    def test_todos_correctos(self):
        resultados = [{'correcto': True}] * 5
        assert calcular_accuracy(resultados) == 1.0

    def test_ninguno_correcto(self):
        resultados = [{'correcto': False}] * 4
        assert calcular_accuracy(resultados) == 0.0

    def test_tres_de_cinco(self):
        resultados = [
            {'correcto': True}, {'correcto': True}, {'correcto': True},
            {'correcto': False}, {'correcto': False},
        ]
        assert calcular_accuracy(resultados) == 0.60

    def test_accuracy_jan2026_referencia(self):
        """9/19 = 47.37% — la accuracy de referencia de Jan 2026."""
        correctas = [{'correcto': True}] * 9
        incorrectas = [{'correcto': False}] * 10
        acc = calcular_accuracy(correctas + incorrectas)
        assert abs(acc - 9/19) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# accuracy_por_superficie
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracyPorSuperficie:

    def test_lista_vacia(self):
        assert accuracy_por_superficie([]) == {}

    def test_una_superficie(self):
        resultados = [
            {'superficie': 'clay', 'correcto': True},
            {'superficie': 'clay', 'correcto': True},
            {'superficie': 'clay', 'correcto': False},
        ]
        r = accuracy_por_superficie(resultados)
        assert 'clay' in r
        assert r['clay']['n'] == 3
        assert abs(r['clay']['accuracy'] - 2/3) < 0.001

    def test_multiples_superficies(self):
        resultados = [
            {'superficie': 'clay',  'correcto': True},
            {'superficie': 'clay',  'correcto': True},
            {'superficie': 'hard',  'correcto': False},
            {'superficie': 'grass', 'correcto': True},
        ]
        r = accuracy_por_superficie(resultados)
        assert r['clay']['accuracy'] == 1.0
        assert r['hard']['accuracy'] == 0.0
        assert r['grass']['accuracy'] == 1.0

    def test_superficie_unknown_se_agrupa(self):
        resultados = [
            {'superficie': 'unknown', 'correcto': True},
            {'superficie': 'unknown', 'correcto': False},
        ]
        r = accuracy_por_superficie(resultados)
        assert 'unknown' in r
        assert r['unknown']['n'] == 2

    def test_correctas_contadas(self):
        resultados = [
            {'superficie': 'clay', 'correcto': True},
            {'superficie': 'clay', 'correcto': True},
            {'superficie': 'clay', 'correcto': False},
        ]
        r = accuracy_por_superficie(resultados)
        assert r['clay']['correctas'] == 2

    def test_meta_clay_roland_garros(self):
        """Meta Nodo-05: clay >= 55% en Roland Garros."""
        clay = [{'superficie': 'clay', 'correcto': True}] * 11 + \
               [{'superficie': 'clay', 'correcto': False}] * 9
        r = accuracy_por_superficie(clay)
        # 11/20 = 55% — exactamente en el umbral
        assert r['clay']['accuracy'] >= 0.55

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
    _detectar_jugador_home_fs,
    _validar_slug_ambos_jugadores,
    _buscar_en_feed,
    _normalize_name,
    _parse_nombre,
    _build_match_key,
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


# ─────────────────────────────────────────────────────────────────────────────
# _detectar_jugador_home_fs
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectarJugadorHomeFs:
    """
    Verifica que _detectar_jugador_home_fs() resuelve correctamente el orden
    home/away de FlashScore usando la match_url.

    Formato URL: /tennis/{slug1}-{slug2}/{match_id}/
    slug1 = jugador FS-home. El FS-home puede ser distinto al Kambi-jugador1.
    """

    BASE = "https://www.flashscore.co/match/tennis"

    def test_jugador1_es_home_orden_normal(self):
        """Cuando slug1 pertenece a jugador1, retorna jugador1."""
        url = f"{self.BASE}/nadal-rafael-federer-roger/abc123/#/h2h"
        assert _detectar_jugador_home_fs(url, 'Rafael Nadal', 'Roger Federer') == 'jugador1'

    def test_jugador2_es_home_orden_invertido(self):
        """Caso real Aguilar: slug1 = aguilar (jugador2), jugador1 = Torres."""
        url = f"{self.BASE}/aguilar-cardozo-joaquin-estevez-juan/jkj6YKod/#/h2h"
        result = _detectar_jugador_home_fs(url, 'Juan Bautista Torres', 'Joaquin Aguilar Cardozo')
        assert result == 'jugador2'

    def test_jugador1_es_home_apellido_compuesto(self):
        """Da Silva (jugador1) aparece primero en slug — es FS-home."""
        url = f"{self.BASE}/andrade-da-silva-lucas-huertas-del-pino-conner/SQ5mLb86/#/h2h"
        result = _detectar_jugador_home_fs(url, 'Lucas Andrade Da Silva', 'Thiago Seyboth Wild')
        assert result == 'jugador1'

    def test_sin_url_retorna_jugador1(self):
        """Sin match_url, comportamiento conservador: jugador1."""
        assert _detectar_jugador_home_fs('', 'A B', 'C D') == 'jugador1'

    def test_url_sin_patron_tenis_retorna_jugador1(self):
        """URL malformada sin /tennis/ → jugador1."""
        assert _detectar_jugador_home_fs('https://example.com/foo/bar', 'A B', 'C D') == 'jugador1'

    def test_jugador2_es_home_osaka(self):
        """Proxy match Tier4: slug1=osaka → jugador2=Osaka es FS-home."""
        url = f"{self.BASE}/osaka-naomi-alexandrova-ekaterina/d2RaSKd2/#/h2h"
        result = _detectar_jugador_home_fs(url, 'Xinyu Wang', 'Naomi Osaka')
        assert result == 'jugador2'


# ─────────────────────────────────────────────────────────────────────────────
# validar_partido_individual — casos con home/away invertido
# ─────────────────────────────────────────────────────────────────────────────

class TestValidarPartidoIndividualHomeAwayInvertido:
    """
    Regresión Nodo-05: cuando Kambi y FS tienen orden home/away distinto,
    DJ='H' debe mapearse al FS-home real, no a jugador1 Kambi.

    Caso real: Torres (j1 Kambi) vs Aguilar (j2 Kambi), pero slug1=aguilar → FS-home=Aguilar.
    Aguilar ganó → API retorna DJ=H → ganador_real debe ser Aguilar (j2), no Torres (j1).
    """

    BASE_URL = "https://www.flashscore.co/match/tennis"

    def _partido_swapped(self, pred_ganador='Joaquin Aguilar Cardozo'):
        """Partido donde jugador2 es el FS-home (slug1 = aguilar)."""
        return {
            'jugador1': 'Juan Bautista Torres',
            'jugador2': 'Joaquin Aguilar Cardozo',
            'match_id': 'jkj6YKod',
            'match_url': f'{self.BASE_URL}/aguilar-cardozo-joaquin-estevez-juan/jkj6YKod/#/h2h',
            'torneo': 'Piracicaba',
            'superficie': 'clay',
            'ranking_analysis': {
                'prediction': {
                    'favored_player': pred_ganador,
                    'confidence': 0.62,
                }
            }
        }

    def test_aguilar_gana_dj_h_prediccion_correcta(self):
        """
        Bug reportado: Aguilar @2.55 GANÓ pero el script decía que perdió.
        API retorna DJ=H (FS-home=Aguilar ganó). pred=Aguilar → correcto=True.
        """
        partido = self._partido_swapped(pred_ganador='Joaquin Aguilar Cardozo')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}  # DJ=H
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['resultado_real'] == 'Joaquin Aguilar Cardozo'
        assert r['correcto'] is True

    def test_aguilar_gana_torres_predicho_incorrecto(self):
        """Si pred=Torres pero Aguilar gana (DJ=H), correcto=False."""
        partido = self._partido_swapped(pred_ganador='Juan Bautista Torres')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}  # DJ=H → Aguilar
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['resultado_real'] == 'Joaquin Aguilar Cardozo'
        assert r['correcto'] is False

    def test_torres_gana_dj_a_prediccion_correcta(self):
        """DJ=A (FS-away=Torres ganó, ya que FS-home=Aguilar). pred=Torres → correcto=True."""
        partido = self._partido_swapped(pred_ganador='Juan Bautista Torres')
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador2'}  # DJ=A
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['resultado_real'] == 'Juan Bautista Torres'
        assert r['correcto'] is True

    def test_orden_normal_sin_inversion_sin_url(self):
        """Sin match_url, comportamiento previo conservado: DJ=H → jugador1."""
        partido = {
            'jugador1': 'Tsitsipas S.',
            'jugador2': 'Nadal R.',
            'match_id': 'abc123',
            'match_url': '',
            'torneo': 'Roland Garros',
            'superficie': 'clay',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Tsitsipas S.', 'confidence': 0.60}
            }
        }
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['resultado_real'] == 'Tsitsipas S.'
        assert r['correcto'] is True


# ─────────────────────────────────────────────────────────────────────────────
# T-slug: _validar_slug_ambos_jugadores
# ─────────────────────────────────────────────────────────────────────────────

class TestValidarSlugAmbosJugadores:
    """
    Validates that _validar_slug_ambos_jugadores detects wrong match_ids
    from Kambi Tier 3 substring matcher (compound Latin names).
    """

    BASE = "https://www.flashscore.co/match/tennis"

    def test_slug_validation_both_players_present(self):
        """T-slug-01: slug has tokens from BOTH players -> True."""
        url = f"{self.BASE}/da-silva-lucas-seyboth-wild-thiago/ABC123/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Lucas Da Silva', 'Thiago Seyboth Wild') is True

    def test_slug_validation_only_one_player(self):
        """T-slug-02: slug points to Da Silva vs Huertas Del Pino, but jugador2 is Seyboth Wild -> False."""
        url = f"{self.BASE}/andrade-da-silva-lucas-huertas-del-pino-conner/SQ5mLb86/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Lucas Andrade Da Silva', 'Thiago Seyboth Wild') is False

    def test_slug_validation_compound_latin_name(self):
        """T-slug-03: slug = aguilar-cardozo-joaquin-estevez-juan, jugador2=Torres -> False."""
        url = f"{self.BASE}/aguilar-cardozo-joaquin-estevez-juan/jkj6YKod/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Joaquin Aguilar Cardozo', 'Juan Bautista Torres') is False

    def test_slug_validation_short_tokens_ignored(self):
        """T-slug-04: tokens 'da', 'de', 'van' (<=2 chars) should NOT count as matches."""
        # Slug has 'da' and 'de' but not real player2 surname tokens
        url = f"{self.BASE}/da-silva-lucas-de-groot-martijn/ABC123/#/h2h"
        # Player2 = "Da De Van Someone" — only 'someone' is >=3 chars, not in slug
        assert _validar_slug_ambos_jugadores(url, 'Lucas Da Silva', 'Da De Van Someone') is False

    def test_slug_validation_borges_wrong_opponent(self):
        """T-slug-05: slug = darderi-luciano-borges-nuno, jugador2=Quinn -> False."""
        url = f"{self.BASE}/darderi-luciano-borges-nuno/rXSTxMNq/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Nuno Borges', 'Ethan Quinn') is False

    def test_slug_validation_correct_match(self):
        """T-slug-06: slug = borges-nuno-quinn-ethan -> both present -> True."""
        url = f"{self.BASE}/borges-nuno-quinn-ethan/CORRECT1/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Nuno Borges', 'Ethan Quinn') is True

    def test_draper_wrong_slug(self):
        """T-slug-09: slug = draper-jack-diallo-gabriel, jugador2=Humbert -> False."""
        url = f"{self.BASE}/draper-jack-diallo-gabriel/WRONG123/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Jack Draper', 'Ugo Humbert') is False

    def test_bergs_wrong_slug(self):
        """T-slug-10: slug = choinski-jan-bergs-zizou, jugador2=Samuel -> False."""
        url = f"{self.BASE}/choinski-jan-bergs-zizou/WRONG456/#/h2h"
        assert _validar_slug_ambos_jugadores(url, 'Zizou Bergs', 'Toby Samuel') is False

    def test_no_url_returns_true(self):
        """No URL to validate -> assume OK."""
        assert _validar_slug_ambos_jugadores('', 'A B', 'C D') is True

    def test_malformed_url_returns_true(self):
        """URL without /tennis/ pattern -> assume OK."""
        assert _validar_slug_ambos_jugadores('https://example.com/foo', 'A B', 'C D') is True


# ─────────────────────────────────────────────────────────────────────────────
# T-feed: Feed fallback integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedFallbackIntegration:
    """
    Tests that validar_partido_individual falls back to FlashScore feed
    when slug validation detects a wrong match_id.
    """

    BASE = "https://www.flashscore.co/match/tennis"

    def _build_feed_lookup(self, j1_fs='Nuno Borges', j2_fs='Ethan Quinn',
                           match_id='CORRECT_ID'):
        """Build a minimal feed_lookup dict for testing."""
        entry = {
            'match_id': match_id,
            'jugador1_fs': j1_fs,
            'jugador2_fs': j2_fs,
            'torneo_fs': 'Wimbledon',
        }
        key = _build_match_key(j1_fs, j2_fs)
        lookup = {key: entry}
        # Also add surname-only key
        a1, _ = _parse_nombre(j1_fs)
        a2, _ = _parse_nombre(j2_fs)
        key_apellido = (min(a1, a2), "", max(a1, a2), "")
        lookup[key_apellido] = entry
        return lookup

    def test_fallback_to_feed_when_slug_wrong(self):
        """T-feed-07: wrong slug -> feed provides correct match_id -> validated."""
        partido = {
            'jugador1': 'Nuno Borges',
            'jugador2': 'Ethan Quinn',
            'match_id': 'WRONG_ID',
            'match_url': f'{self.BASE}/darderi-luciano-borges-nuno/WRONG_ID/#/h2h',
            'torneo': 'Wimbledon',
            'superficie': 'grass',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Nuno Borges', 'confidence': 0.65}
            }
        }
        feed_lookup = self._build_feed_lookup(
            j1_fs='Borges N.', j2_fs='Quinn E.', match_id='CORRECT_ID'
        )
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}

        with patch('validar_con_api.obtener_resultado_partido', return_value=resultado_api) as mock_api:
            r = validar_partido_individual(partido, None, feed_lookup=feed_lookup)

        assert r is not None
        assert r['match_id'] == 'CORRECT_ID'
        assert r['resolved_from_feed'] is True
        mock_api.assert_called_once_with('CORRECT_ID')

    def test_no_result_when_feed_has_no_match(self):
        """T-feed-08: wrong slug + empty feed -> None (no verificado)."""
        partido = {
            'jugador1': 'Nuno Borges',
            'jugador2': 'Ethan Quinn',
            'match_id': 'WRONG_ID',
            'match_url': f'{self.BASE}/darderi-luciano-borges-nuno/WRONG_ID/#/h2h',
            'torneo': 'Wimbledon',
            'superficie': 'grass',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Nuno Borges', 'confidence': 0.65}
            }
        }
        empty_feed = {}
        r = validar_partido_individual(partido, None, feed_lookup=empty_feed)
        assert r is None

    def test_valid_slug_uses_original_match_id(self):
        """When slug is valid, original match_id is used, not feed."""
        partido = {
            'jugador1': 'Nuno Borges',
            'jugador2': 'Ethan Quinn',
            'match_id': 'ORIGINAL_ID',
            'match_url': f'{self.BASE}/borges-nuno-quinn-ethan/ORIGINAL_ID/#/h2h',
            'torneo': 'Wimbledon',
            'superficie': 'grass',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Nuno Borges', 'confidence': 0.65}
            }
        }
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}
        r = validar_partido_individual(partido, resultado_api)
        assert r is not None
        assert r['match_id'] == 'ORIGINAL_ID'
        assert r.get('resolved_from_feed', False) is False

    def test_no_match_id_falls_back_to_feed(self):
        """When match_id is missing/empty, feed fallback is used."""
        partido = {
            'jugador1': 'Nuno Borges',
            'jugador2': 'Ethan Quinn',
            'match_id': '',
            'match_url': '',
            'torneo': 'Wimbledon',
            'superficie': 'grass',
            'ranking_analysis': {
                'prediction': {'favored_player': 'Nuno Borges', 'confidence': 0.65}
            }
        }
        feed_lookup = self._build_feed_lookup(
            j1_fs='Borges N.', j2_fs='Quinn E.', match_id='FEED_ID'
        )
        resultado_api = {'status': 'FT', 'ganador_lado': 'jugador1'}

        with patch('validar_con_api.obtener_resultado_partido', return_value=resultado_api):
            r = validar_partido_individual(partido, None, feed_lookup=feed_lookup)

        assert r is not None
        assert r['match_id'] == 'FEED_ID'
        assert r['resolved_from_feed'] is True

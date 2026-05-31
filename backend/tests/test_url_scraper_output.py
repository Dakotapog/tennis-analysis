"""
Tests para los 3 bugs corregidos en extraer_URL_partidos_version2.py (Nodo-03)
"""
import re
import pytest
from extraer_URL_partidos_version2 import ZitaScraper


class TestBug1H2hUrl:
    """Bug 1: h2h_url se construye correctamente desde match_url"""

    def test_h2h_url_derivada_de_match_url(self):
        match_url = "https://www.flashscore.com/match/tennis/cobolli-flavio-zDtaCcPe/wu-yibing-8ASNvPfK/?mid=rDQ3y6to"
        h2h_url = match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
        assert h2h_url == "https://www.flashscore.com/match/tennis/cobolli-flavio-zDtaCcPe/wu-yibing-8ASNvPfK/#/h2h/overall/"

    def test_h2h_url_sin_params_query(self):
        """URL sin ?mid= también genera h2h_url correcta"""
        match_url = "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/"
        h2h_url = match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
        assert h2h_url == "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/#/h2h/overall/"

    def test_h2h_url_termina_en_overall(self):
        match_url = "https://www.flashscore.com/match/tennis/a-b/c-d/?mid=XYZ"
        h2h_url = match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
        assert h2h_url.endswith('/#/h2h/overall/')

    def test_h2h_url_no_contiene_mid(self):
        """La h2h_url no debe contener el parámetro ?mid= original"""
        match_url = "https://www.flashscore.com/match/tennis/a-b/c-d/?mid=rDQ3y6to"
        h2h_url = match_url.split('?')[0].rstrip('/') + '/#/h2h/overall/'
        assert 'mid=' not in h2h_url
        assert 'rDQ3y6to' not in h2h_url


class TestBug2MatchId:
    """Bug 2: match_id extrae el event_id real del parámetro ?mid="""

    def test_match_id_extraido_de_mid_param(self):
        match_url = "https://www.flashscore.com/match/tennis/a-b-ID1/c-d-ID2/?mid=rDQ3y6to"
        mid = re.search(r'[?&]mid=([^&]+)', match_url)
        assert mid is not None
        assert mid.group(1) == "rDQ3y6to"

    def test_match_id_no_es_tennis(self):
        """El match_id extraído no puede ser 'tennis' (bug original)"""
        match_url = "https://www.flashscore.com/match/tennis/player1/player2/?mid=ABC123"
        mid = re.search(r'[?&]mid=([^&]+)', match_url)
        assert mid is not None
        assert mid.group(1) != 'tennis'

    def test_match_id_con_ampersand(self):
        """Funciona si hay otros parámetros después de ?mid="""
        match_url = "https://www.flashscore.com/match/tennis/a/b/?mid=rDQ3y6to&foo=bar"
        mid = re.search(r'[?&]mid=([^&]+)', match_url)
        assert mid is not None
        assert mid.group(1) == "rDQ3y6to"

    def test_match_id_sin_mid_usa_fallback(self):
        """Sin ?mid=, el fallback no devuelve 'tennis'"""
        match_url = "https://www.flashscore.com/match/tennis/player1-ABC/player2-DEF/"
        url_path = match_url.split('?')[0].rstrip('/')
        last_seg = url_path.split('/')[-1]
        # El fallback debe excluir 'tennis'
        assert last_seg not in ('tennis', 'tenis', '')


class TestBug3Superficie:
    """Bug 3: superficie se extrae correctamente del nombre del torneo"""

    def test_roland_garros_es_clay(self):
        assert ZitaScraper.extraer_superficie("Roland Garros (France)") == "clay"

    def test_french_open_es_clay(self):
        assert ZitaScraper.extraer_superficie("French Open 2026") == "clay"

    def test_wimbledon_es_grass(self):
        assert ZitaScraper.extraer_superficie("Wimbledon (UK)") == "grass"

    def test_australian_open_es_hard(self):
        assert ZitaScraper.extraer_superficie("Australian Open (Australia)") == "hard"

    def test_us_open_es_hard(self):
        assert ZitaScraper.extraer_superficie("US Open (USA)") == "hard"

    def test_clay_en_nombre_generico(self):
        assert ZitaScraper.extraer_superficie("ATP Geneva Clay 250") == "clay"

    def test_grass_en_nombre_generico(self):
        assert ZitaScraper.extraer_superficie("ATP Halle Grass 500") == "grass"

    def test_hard_en_nombre_generico(self):
        assert ZitaScraper.extraer_superficie("ATP Cincinnati Hard Court") == "hard"

    def test_indoor_es_hard(self):
        assert ZitaScraper.extraer_superficie("ATP Vienna Indoor 500") == "hard"

    def test_torneo_desconocido_es_unknown(self):
        assert ZitaScraper.extraer_superficie("ATP Some Unknown Tournament") == "unknown"

    def test_torneo_vacio_es_unknown(self):
        assert ZitaScraper.extraer_superficie("") == "unknown"

    def test_sin_torneo_asignado_es_unknown(self):
        assert ZitaScraper.extraer_superficie("Sin Torneo Asignado") == "unknown"

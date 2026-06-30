"""
Tests para config.py — D-17: Centralizar configuración dispersa.

Verifica que las constantes existen, tienen los tipos correctos
y los valores esperados por el pipeline.
"""
import pytest


class TestFlashscoreApiConfig:

    def test_flashscore_base_is_string(self):
        from config import FLASHSCORE_BASE
        assert isinstance(FLASHSCORE_BASE, str)
        assert FLASHSCORE_BASE.startswith("https://")

    def test_flashscore_base_apunta_al_ninja_api(self):
        from config import FLASHSCORE_BASE
        assert "flashscore.ninja" in FLASHSCORE_BASE
        assert "/feed" in FLASHSCORE_BASE

    def test_flashscore_headers_es_dict(self):
        from config import FLASHSCORE_HEADERS
        assert isinstance(FLASHSCORE_HEADERS, dict)

    def test_flashscore_headers_contiene_xfsign(self):
        from config import FLASHSCORE_HEADERS
        assert "X-Fsign" in FLASHSCORE_HEADERS
        assert FLASHSCORE_HEADERS["X-Fsign"] == "SW9D1eZo"

    def test_flashscore_headers_contiene_referer(self):
        from config import FLASHSCORE_HEADERS
        assert "Referer" in FLASHSCORE_HEADERS
        assert "flashscore" in FLASHSCORE_HEADERS["Referer"]


class TestPipelineConfig:

    def test_total_matches_to_process_es_80(self):
        from config import TOTAL_MATCHES_TO_PROCESS
        assert TOTAL_MATCHES_TO_PROCESS == 80

    def test_total_matches_es_int(self):
        from config import TOTAL_MATCHES_TO_PROCESS
        assert isinstance(TOTAL_MATCHES_TO_PROCESS, int)

    def test_browser_headless_es_true_por_defecto(self):
        from config import BROWSER_HEADLESS
        assert BROWSER_HEADLESS is True

    def test_browser_slow_mo_es_250(self):
        from config import BROWSER_SLOW_MO
        assert BROWSER_SLOW_MO == 250

    def test_browser_slow_mo_es_int(self):
        from config import BROWSER_SLOW_MO
        assert isinstance(BROWSER_SLOW_MO, int)

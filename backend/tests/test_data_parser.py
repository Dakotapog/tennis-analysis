"""
Tests para DataParser — scraping/data_parser.py

COBERTURA:
  - extract_tournament_info
  - normalize_surface
  - determine_winner_from_result
  - extract_winner_sets
  - parse_match_date
  - clean_player_name
  - extract_location_from_title

CONVENCIÓN:
  Los tests marcados con 'BUG DOCUMENTADO' describen el comportamiento
  CORRECTO esperado. Fallarán hasta que el bug sea corregido en data_parser.py.
  Cuando pasen → el bug está resuelto → remover el comentario BUG DOCUMENTADO.
"""

import pytest
from datetime import datetime
from scraping.data_parser import DataParser


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE PRUEBA
# ─────────────────────────────────────────────────────────────────────────────

# String de garbage HTML real que aparece en producción (dato de reports/)
HTML_GARBAGE_SUPERFICIE = (
    "hardGIRLS: 1218:00Campbell G.Boian I.----18:00Deng P.Das A."
    "----18:00Ichioka A.Pistola I.----18:00Jeong U.-S.Harding C."
    "----18:00Koreshkova D.Rodero C.----18:00Kumru A.Sandhu G. K."
)

HTML_GARBAGE_TORNEO = (
    "12FinishedSakamoto R.Jodar R.236678167564362.901.38Finished"
    "Duckworth J.Prizmic D.327764361675632.051.73FinishedHurkacz H."
    "Bergs Z.316678786663631.363.00FinishedRoyer V.Fritz T."
)


# ─────────────────────────────────────────────────────────────────────────────
# extract_tournament_info
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTournamentInfo:
    """Tests para DataParser.extract_tournament_info."""

    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_formato_atp_estandar(self):
        """Happy path: formato ATP estándar con país y superficie."""
        result = DataParser.extract_tournament_info("ATP: Toronto (Canada), hard")
        assert result['pais'] == "Canada"
        assert result['superficie'] == "hard"
        assert "Toronto" in result['nombre']

    def test_formato_wta_arcilla(self):
        """Happy path: formato WTA con superficie clay."""
        result = DataParser.extract_tournament_info("WTA: Roland Garros (France), clay")
        assert result['pais'] == "France"
        assert result['superficie'] == "clay"
        assert "Roland Garros" in result['nombre']

    def test_formato_challenger(self):
        """Happy path: formato Challenger."""
        result = DataParser.extract_tournament_info(
            "CHALLENGER MEN: Buenos Aires (Argentina), clay"
        )
        assert result['pais'] == "Argentina"
        assert result['superficie'] == "clay"

    def test_formato_australian_open(self):
        """Happy path: Australian Open con superficie hard."""
        result = DataParser.extract_tournament_info("Australian Open , hard")
        assert result['superficie'] == "hard"

    def test_retorna_keys_requeridas(self):
        """El resultado siempre contiene las cuatro claves del contrato."""
        result = DataParser.extract_tournament_info("ATP: Wimbledon (UK), grass")
        assert set(result.keys()) == {'nombre', 'pais', 'superficie', 'completo'}

    def test_completo_es_el_string_original(self):
        """'completo' siempre devuelve el string original sin modificar."""
        original = "ATP: Toronto (Canada), hard"
        result = DataParser.extract_tournament_info(original)
        assert result['completo'] == original

    def test_sin_pais_devuelve_na(self):
        """Sin paréntesis en el string, pais es N/A."""
        result = DataParser.extract_tournament_info("Wimbledon, grass")
        assert result['pais'] == "N/A"

    def test_sin_superficie_devuelve_desconocida(self):
        """Sin superficie reconocida, superficie es Desconocida."""
        result = DataParser.extract_tournament_info("ATP: Toronto (Canada)")
        assert result['superficie'] == "Desconocida"

    def test_superficie_indoor(self):
        """Superficie indoor es reconocida."""
        result = DataParser.extract_tournament_info("ATP: Rotterdam (Netherlands), indoor")
        assert result['superficie'] == "indoor"

    def test_superficie_grass(self):
        """Superficie grass es reconocida."""
        result = DataParser.extract_tournament_info("ATP: Wimbledon (UK), grass")
        assert result['superficie'] == "grass"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_string_vacio(self):
        """String vacío retorna nombre vacío y defaults seguros sin lanzar excepción."""
        result = DataParser.extract_tournament_info("")
        assert result['nombre'] == ""
        assert result['superficie'] == "Desconocida"
        assert result['pais'] == "N/A"

    # ── BUG DOCUMENTADO ──────────────────────────────────────────────────────

    def test_html_garbage_torneo_retorna_desconocido(self):
        """
        BUG DOCUMENTADO: Cuando torneo_nombre contiene el HTML completo
        de la página de FlashScore, la función debe detectarlo y retornar
        'Desconocido' en lugar de propagar el garbage al sistema.

        Evidencia del bug en: reports/h2h_results_enhanced_20260120_183437.json
          torneo_nombre = "12FinishedSakamoto R.Jodar R.236678..."

        Estado: FALLA hasta corregir extract_tournament_info en data_parser.py
        Fix requerido: detectar strings > MAX_TOURNAMENT_NAME_LENGTH como garbage.
        """
        result = DataParser.extract_tournament_info(HTML_GARBAGE_TORNEO)
        assert result['nombre'] == "Desconocido", (
            f"Se esperaba 'Desconocido' pero se obtuvo: '{result['nombre'][:60]}...'"
        )
        assert result['superficie'] == "Desconocida"

    def test_html_garbage_superficie_en_torneo(self):
        """
        BUG DOCUMENTADO: El campo tipo_cancha contiene el HTML completo
        cuando la superficie va prefijada con contenido de página.

        Evidencia del bug en: analisis_partidos_pandas.txt
          'Surface del partido actual hardGIRLS: 1218:00Campbell G...'
          'no se pudo normalizar'

        Estado: FALLA hasta corregir extract_tournament_info en data_parser.py
        """
        result = DataParser.extract_tournament_info(HTML_GARBAGE_SUPERFICIE)
        assert result['superficie'] == "Desconocida", (
            f"Se esperaba 'Desconocida' pero se obtuvo: '{result['superficie'][:60]}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# normalize_surface
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSurface:
    """Tests para DataParser.normalize_surface."""

    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_hard_a_dura(self):
        assert DataParser.normalize_surface("hard") == "Dura"

    def test_clay_a_arcilla(self):
        assert DataParser.normalize_surface("clay") == "Arcilla"

    def test_grass_a_hierba(self):
        assert DataParser.normalize_surface("grass") == "Hierba"

    def test_indoor_a_indoor(self):
        assert DataParser.normalize_surface("indoor") == "Indoor"

    def test_dura_ya_normalizada(self):
        """Entrada ya normalizada en español → se mantiene."""
        assert DataParser.normalize_surface("dura") == "Dura"

    def test_arcilla_ya_normalizada(self):
        assert DataParser.normalize_surface("arcilla") == "Arcilla"

    def test_hierba_ya_normalizada(self):
        assert DataParser.normalize_surface("hierba") == "Hierba"

    def test_mayusculas_son_case_insensitive(self):
        """Case insensitive: HARD → Dura."""
        assert DataParser.normalize_surface("HARD") == "Dura"
        assert DataParser.normalize_surface("Clay") == "Arcilla"
        assert DataParser.normalize_surface("GRASS") == "Hierba"

    def test_espacios_son_ignorados(self):
        """Espacios al inicio/fin son ignorados."""
        assert DataParser.normalize_surface("  hard  ") == "Dura"
        assert DataParser.normalize_surface("  clay  ") == "Arcilla"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_none_retorna_desconocida(self):
        """None → 'Desconocida' sin lanzar excepción."""
        assert DataParser.normalize_surface(None) == "Desconocida"

    def test_string_vacio_retorna_desconocida(self):
        """String vacío → 'Desconocida'."""
        assert DataParser.normalize_surface("") == "Desconocida"

    # ── BUG DOCUMENTADO ──────────────────────────────────────────────────────

    def test_html_garbage_retorna_desconocida(self):
        """
        BUG DOCUMENTADO: Cuando normalize_surface recibe el HTML completo
        de la página (porque extract_tournament_info falló antes), debe
        retornar 'Desconocida' y no propagar el garbage.

        Estado actual: retorna el garbage con .capitalize()
        Estado esperado: retorna 'Desconocida'

        Estado: FALLA hasta corregir normalize_surface en data_parser.py
        Fix requerido: si surface_text no está en el mapa Y su longitud
                       supera un umbral (ej: 30 chars), retornar 'Desconocida'.
        """
        result = DataParser.normalize_surface(HTML_GARBAGE_SUPERFICIE)
        assert result == "Desconocida", (
            f"Se esperaba 'Desconocida' pero se obtuvo: '{result[:60]}'"
        )

    def test_superficie_desconocida_corta_usa_capitalize(self):
        """
        Superficie desconocida pero válida (corta, sin HTML) → capitalize.
        Ej: 'carpet' no está en el mapa pero es un nombre legítimo.
        """
        result = DataParser.normalize_surface("carpet")
        assert result == "Carpet"


# ─────────────────────────────────────────────────────────────────────────────
# determine_winner_from_result
# ─────────────────────────────────────────────────────────────────────────────

class TestDetermineWinnerFromResult:
    """Tests para DataParser.determine_winner_from_result."""

    def test_jugador1_gana_2_0(self):
        result = DataParser.determine_winner_from_result("2-0", "Nadal", "Federer")
        assert result == "Nadal"

    def test_jugador2_gana_0_2(self):
        result = DataParser.determine_winner_from_result("0-2", "Nadal", "Federer")
        assert result == "Federer"

    def test_jugador1_gana_2_1(self):
        result = DataParser.determine_winner_from_result("2-1", "Djokovic", "Murray")
        assert result == "Djokovic"

    def test_jugador2_gana_1_2(self):
        result = DataParser.determine_winner_from_result("1-2", "Djokovic", "Murray")
        assert result == "Murray"

    def test_resultado_na_retorna_na(self):
        result = DataParser.determine_winner_from_result("N/A", "Nadal", "Federer")
        assert result == "N/A"

    def test_resultado_none_retorna_na(self):
        result = DataParser.determine_winner_from_result(None, "Nadal", "Federer")
        assert result == "N/A"

    def test_resultado_vacio_retorna_na(self):
        result = DataParser.determine_winner_from_result("", "Nadal", "Federer")
        assert result == "N/A"

    def test_resultado_malformado_retorna_na(self):
        """Strings que no son formato sets (ej: texto) → N/A sin excepción."""
        result = DataParser.determine_winner_from_result("WO", "Nadal", "Federer")
        assert result == "N/A"

    def test_resultado_empate_retorna_na(self):
        """Sets iguales (empate) → N/A."""
        result = DataParser.determine_winner_from_result("1-1", "Nadal", "Federer")
        assert result == "N/A"

    def test_formato_bo5_jugador1(self):
        """Best of 5: 3-1 → jugador1."""
        result = DataParser.determine_winner_from_result("3-1", "Alcaraz", "Sinner")
        assert result == "Alcaraz"

    def test_formato_bo5_jugador2(self):
        """Best of 5: 2-3 → jugador2."""
        result = DataParser.determine_winner_from_result("2-3", "Alcaraz", "Sinner")
        assert result == "Sinner"


# ─────────────────────────────────────────────────────────────────────────────
# extract_winner_sets
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractWinnerSets:
    """Tests para DataParser.extract_winner_sets."""

    def test_ganador_2_sets_resultado_2_0(self):
        assert DataParser.extract_winner_sets("2-0") == 2

    def test_ganador_2_sets_resultado_0_2(self):
        assert DataParser.extract_winner_sets("0-2") == 2

    def test_ganador_2_sets_resultado_1_2(self):
        assert DataParser.extract_winner_sets("1-2") == 2

    def test_ganador_3_sets_resultado_3_1(self):
        assert DataParser.extract_winner_sets("3-1") == 3

    def test_resultado_na_retorna_na(self):
        assert DataParser.extract_winner_sets("N/A") == "N/A"

    def test_resultado_none_retorna_na(self):
        assert DataParser.extract_winner_sets(None) == "N/A"

    def test_resultado_vacio_retorna_na(self):
        assert DataParser.extract_winner_sets("") == "N/A"

    def test_resultado_malformado_retorna_na(self):
        assert DataParser.extract_winner_sets("WO") == "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# parse_match_date
# ─────────────────────────────────────────────────────────────────────────────

class TestParseMatchDate:
    """Tests para DataParser.parse_match_date."""

    def test_fecha_valida_formato_default(self):
        """Formato '%d.%m.%y' (el default del sistema)."""
        result = DataParser.parse_match_date("15.10.23")
        assert result == datetime(2023, 10, 15)

    def test_fecha_enero_2026(self):
        """Fecha del dataset real: enero 2026."""
        result = DataParser.parse_match_date("20.01.26")
        assert result == datetime(2026, 1, 20)

    def test_fecha_invalida_retorna_none(self):
        """Fecha con formato incorrecto retorna None sin excepción."""
        result = DataParser.parse_match_date("2023-10-15")
        assert result is None

    def test_string_vacio_retorna_none(self):
        result = DataParser.parse_match_date("")
        assert result is None

    def test_none_retorna_none(self):
        result = DataParser.parse_match_date(None)
        assert result is None

    def test_texto_arbitrario_retorna_none(self):
        result = DataParser.parse_match_date("no es una fecha")
        assert result is None

    def test_formato_personalizado(self):
        """Permite pasar formato personalizado."""
        result = DataParser.parse_match_date("2023-10-15", date_format="%Y-%m-%d")
        assert result == datetime(2023, 10, 15)


# ─────────────────────────────────────────────────────────────────────────────
# clean_player_name
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanPlayerName:
    """Tests para DataParser.clean_player_name."""

    def test_nombre_con_espacios_extra(self):
        """Espacios múltiples se colapsan a uno."""
        assert DataParser.clean_player_name("  Nadal R.  ") == "Nadal R."

    def test_nombre_normal_sin_cambio(self):
        assert DataParser.clean_player_name("Tsitsipas S.") == "Tsitsipas S."

    def test_espacios_internos_multiples(self):
        assert DataParser.clean_player_name("Djokovic  N.") == "Djokovic N."

    def test_string_vacio_retorna_na(self):
        assert DataParser.clean_player_name("") == "N/A"

    def test_none_retorna_na(self):
        assert DataParser.clean_player_name(None) == "N/A"

    def test_nombre_con_solo_espacios_retorna_na(self):
        assert DataParser.clean_player_name("   ") == "N/A"

    def test_nombre_con_punto_se_preserva(self):
        """El punto de la inicial se preserva (Nadal R.)."""
        assert DataParser.clean_player_name("Alcaraz C.") == "Alcaraz C."


# ─────────────────────────────────────────────────────────────────────────────
# extract_location_from_title
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractLocationFromTitle:
    """Tests para DataParser.extract_location_from_title."""

    def test_formato_ciudad_pais_superficie(self):
        """Formato estándar: 'Ciudad (País), superficie'."""
        result = DataParser.extract_location_from_title("Toronto (Canada), hard")
        assert result['ciudad'] == "Toronto"
        assert result['pais'] == "Canada"
        assert result['superficie'] == "Dura"

    def test_superficie_arcilla(self):
        result = DataParser.extract_location_from_title("Paris (France), clay")
        assert result['ciudad'] == "Paris"
        assert result['pais'] == "France"
        assert result['superficie'] == "Arcilla"

    def test_superficie_hierba(self):
        result = DataParser.extract_location_from_title("Wimbledon (UK), grass")
        assert result['ciudad'] == "Wimbledon"
        assert result['pais'] == "UK"
        assert result['superficie'] == "Hierba"

    def test_retorna_keys_requeridas(self):
        result = DataParser.extract_location_from_title("Melbourne (Australia), hard")
        assert set(result.keys()) == {'ciudad', 'pais', 'superficie'}

    def test_string_vacio_retorna_na_en_todo(self):
        result = DataParser.extract_location_from_title("")
        assert result['ciudad'] == "N/A"
        assert result['pais'] == "N/A"
        assert result['superficie'] == "N/A"

    def test_none_retorna_na_en_todo(self):
        result = DataParser.extract_location_from_title(None)
        assert result['ciudad'] == "N/A"
        assert result['pais'] == "N/A"
        assert result['superficie'] == "N/A"

    def test_sin_pais_entre_parentesis(self):
        """Sin paréntesis: ciudad se infiere, pais N/A."""
        result = DataParser.extract_location_from_title("Melbourne, hard")
        assert result['ciudad'] == "Melbourne"
        assert result['pais'] == "N/A"
        assert result['superficie'] == "Dura"

    def test_superficie_se_normaliza(self):
        """La superficie se normaliza a través de normalize_surface."""
        result = DataParser.extract_location_from_title("London (UK), grass")
        assert result['superficie'] == "Hierba"

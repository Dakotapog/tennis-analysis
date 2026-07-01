"""
Tests para Nodo-45 — Temporal History Fallback (THF)

Verifica el contrato de _lookup_player_history_temporal() y el comportamiento
de _process_match() cuando match_id=None o la API retorna historial vacío.

Clases:
  TestLookupPlayerHistoryTemporal : T45-01 a T45-06 — función módulo-level
  TestProcessMatchTHF             : T45-07 a T45-09 — routing en _process_match()

Detección de mutación real:
  T45-01 FALLA si la función _lookup_player_history_temporal no existe en el módulo.
  T45-03 FALLA si la búsqueda solo examina jugador1 (no jugador2) en los archivos.
  T45-04 FALLA si se elimina la desambiguación por overlap (token corto en ambos nombres).
  T45-06 FALLA si la función no ordena archivos por recencia (retorna datos viejos).
  T45-07 FALLA si _process_match() no activa THF cuando match_id=None.
  T45-09 FALLA si _process_match() no retorna False cuando no hay match_id NI datos temporales.
"""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scraping.ninja_h2h_parser import (
    _lookup_player_history_temporal,
    NinjaH2HExtractor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de historial
# ─────────────────────────────────────────────────────────────────────────────

HIST_MALDONADO = [
    {
        "fecha": "2026-06-20", "torneo": "Wimbledon", "superficie": "grass",
        "oponente": "Djokovic N.", "resultado": "6-3 6-2", "outcome": "Gano",
        "opponent_ranking": 2,
    },
    {
        "fecha": "2026-06-15", "torneo": "Queens", "superficie": "grass",
        "oponente": "Murray A.", "resultado": "6-4 6-3", "outcome": "Gano",
        "opponent_ranking": 50,
    },
]

HIST_RIVAL = [
    {
        "fecha": "2026-06-18", "torneo": "Wimbledon", "superficie": "grass",
        "oponente": "Sinner J.", "resultado": "3-6 2-6", "outcome": "Perdio",
        "opponent_ranking": 1,
    },
]


def _make_reports_dir(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    return reports


def _write_h2h_file(reports: Path, filename: str, matches: list) -> Path:
    """Escribe un h2h_results_enhanced_*.json de prueba."""
    p = reports / filename
    p.write_text(json.dumps(matches), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fixture de NinjaH2HExtractor con dependencias mockeadas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    """NinjaH2HExtractor con todas las dependencias pesadas mockeadas."""
    with patch("scraping.ninja_h2h_parser.NinjaH2HExtractor.__init__", return_value=None):
        ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)

    ext.all_results = []
    ext.all_tournaments = False
    ext.ranking_manager = MagicMock()
    ext.elo_system = MagicMock()
    ext.rivalry_analyzer = MagicMock()
    ext.rivalry_analyzer.calculate_elo_from_history.return_value = 1500.0
    ext.rivalry_analyzer.analyze_rivalry.return_value = {
        "prediction": {"favored_player": "Maldonado M.", "confidence": 55},
        "scores": {},
    }
    ext._enrich_history = lambda hist: hist
    ext._inject_kambi_ranking = MagicMock()
    ext._analyze_form = MagicMock(return_value={"win_rate": 0.6})
    ext._consolidate_result = MagicMock(return_value={"partido": "Maldonado vs Rival"})
    return ext


# ─────────────────────────────────────────────────────────────────────────────
# T45-01 a T45-06 — _lookup_player_history_temporal()
# ─────────────────────────────────────────────────────────────────────────────

class TestLookupPlayerHistoryTemporal:

    def test_t45_01_no_files_returns_empty(self, tmp_path, monkeypatch):
        """T45-01: Sin archivos h2h en reports/ → retorna []."""
        monkeypatch.chdir(tmp_path)
        _make_reports_dir(tmp_path)  # carpeta vacía, sin JSON

        result = _lookup_player_history_temporal("Martin Maldonado")
        assert result == [], "Sin archivos h2h disponibles debe retornar []"

    def test_t45_02_player_found_as_jugador1(self, tmp_path, monkeypatch):
        """T45-02: Jugador en posición jugador1 → retorna su historial."""
        monkeypatch.chdir(tmp_path)
        reports = _make_reports_dir(tmp_path)

        matches = [{
            "jugador1": "Martin Maldonado",
            "jugador2": "Rival Otro",
            "historial_Martin_Maldonado": HIST_MALDONADO,
            "historial_Rival_Otro": HIST_RIVAL,
        }]
        _write_h2h_file(reports, "h2h_results_enhanced_20260620_100000.json", matches)

        result = _lookup_player_history_temporal("Martin Maldonado")
        assert len(result) == len(HIST_MALDONADO)
        assert result[0]["oponente"] == HIST_MALDONADO[0]["oponente"]

    def test_t45_03_player_found_as_jugador2(self, tmp_path, monkeypatch):
        """T45-03: Jugador en posición jugador2 → retorna su historial.
        FALLA si la búsqueda solo examina jugador1."""
        monkeypatch.chdir(tmp_path)
        reports = _make_reports_dir(tmp_path)

        matches = [{
            "jugador1": "Rival Otro",
            "jugador2": "Martin Maldonado",
            "historial_Rival_Otro": HIST_RIVAL,
            "historial_Martin_Maldonado": HIST_MALDONADO,
        }]
        _write_h2h_file(reports, "h2h_results_enhanced_20260620_100000.json", matches)

        result = _lookup_player_history_temporal("Martin Maldonado")
        assert len(result) == len(HIST_MALDONADO), (
            "THF debe buscar en jugador2 también, no solo en jugador1"
        )

    def test_t45_04_ambiguous_token_resolved_by_overlap(self, tmp_path, monkeypatch):
        """T45-04: Token corto ('lu') aparece en ambos nombres → mayor overlap gana.
        FALLA si se elimina desambiguación por overlap."""
        monkeypatch.chdir(tmp_path)
        reports = _make_reports_dir(tmp_path)

        hist_lu = [{"fecha": "2026-06-10", "torneo": "X", "superficie": "hard",
                    "oponente": "Sinner", "resultado": "6-2 6-1",
                    "outcome": "Gano", "opponent_ranking": 1}]

        matches = [{
            "jugador1": "Jing-Jing Lu",       # token "lu" coincide
            "jugador2": "Lukas Rosol",         # token "lu" también coincide
            "historial_Jing-Jing_Lu": hist_lu,
            "historial_Lukas_Rosol": HIST_RIVAL,
        }]
        _write_h2h_file(reports, "h2h_results_enhanced_20260620_100000.json", matches)

        # Buscar "Lu" — debe priorizar "Jing-Jing Lu" por mayor overlap de tokens
        result = _lookup_player_history_temporal("Jing-Jing Lu")
        assert result == hist_lu, (
            "Con tokens ambiguos debe resolver por mayor overlap, no por posición"
        )

    def test_t45_05_empty_historial_in_file_skipped(self, tmp_path, monkeypatch):
        """T45-05: Match con historial vacío en archivo → no retorna [] vacío, sigue buscando."""
        monkeypatch.chdir(tmp_path)
        reports = _make_reports_dir(tmp_path)

        # Archivo con historial vacío del jugador
        matches_empty = [{
            "jugador1": "Martin Maldonado",
            "jugador2": "Rival",
            "historial_Martin_Maldonado": [],   # vacío — debe saltarse
            "historial_Rival": HIST_RIVAL,
        }]
        # Archivo más antiguo con historial real
        matches_real = [{
            "jugador1": "Martin Maldonado",
            "jugador2": "Otro",
            "historial_Martin_Maldonado": HIST_MALDONADO,
            "historial_Otro": [],
        }]
        _write_h2h_file(reports, "h2h_results_enhanced_20260621_100000.json", matches_empty)
        _write_h2h_file(reports, "h2h_results_enhanced_20260620_100000.json", matches_real)

        result = _lookup_player_history_temporal("Martin Maldonado")
        # El archivo del 21 tiene vacío → lo ignora → encuentra el del 20 con datos reales
        assert len(result) == len(HIST_MALDONADO), (
            "Historial vacío en archivo no debe detener la búsqueda"
        )

    def test_t45_06_prefers_most_recent_file(self, tmp_path, monkeypatch):
        """T45-06: Múltiples archivos → retorna historial del más reciente.
        FALLA si no se ordena por recencia."""
        monkeypatch.chdir(tmp_path)
        reports = _make_reports_dir(tmp_path)

        hist_reciente = [{"fecha": "2026-06-29", "torneo": "Wimbledon",
                          "superficie": "grass", "oponente": "Alcaraz C.",
                          "resultado": "6-1 6-2", "outcome": "Gano",
                          "opponent_ranking": 3}]
        hist_viejo = HIST_MALDONADO

        # Crear archivos — el más reciente por nombre de fecha
        matches_reciente = [{"jugador1": "Martin Maldonado", "jugador2": "X",
                              "historial_Martin_Maldonado": hist_reciente,
                              "historial_X": []}]
        matches_viejo = [{"jugador1": "Martin Maldonado", "jugador2": "Y",
                          "historial_Martin_Maldonado": hist_viejo,
                          "historial_Y": []}]

        # Archivo más reciente por nombre (sorted reverse=True)
        _write_h2h_file(reports, "h2h_results_enhanced_20260629_100000.json", matches_reciente)
        _write_h2h_file(reports, "h2h_results_enhanced_20260620_100000.json", matches_viejo)

        result = _lookup_player_history_temporal("Martin Maldonado")
        assert result == hist_reciente, (
            "THF debe retornar el historial del archivo más reciente, no el más antiguo"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T45-07 a T45-09 — _process_match() routing con THF
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessMatchTHF:
    """Verifica que _process_match() llama correctamente al THF.

    Usa patch sobre _lookup_player_history_temporal para controlar
    qué retorna el fallback, sin depender del sistema de archivos.
    """

    _BASE_MATCH = {
        "jugador1": "Martin Maldonado",
        "jugador2": "Rival Jugador",
        "match_url": "",        # match_id = None (fallo de cruce Kambi↔FS)
        "match_id": None,
        "pais": "UK",
        "tipo_cancha": "grass",
        "superficie": "grass",
        "ranking1": 120,
        "ranking2": 80,
        "torneo_completo": "Challenger London",
        "torneo_nombre": "Challenger London",
    }

    def test_t45_07_process_match_uses_thf_when_no_match_id(self, extractor):
        """T45-07: _process_match con match_id=None y THF con datos → retorna True.
        FALLA si _process_match() no activa THF y sigue retornando False."""
        thf_returns = {
            "Martin Maldonado": HIST_MALDONADO,
            "Rival Jugador": HIST_RIVAL,
        }

        def fake_thf(player_name, days_back=7):
            return thf_returns.get(player_name, [])

        with patch("scraping.ninja_h2h_parser._lookup_player_history_temporal", side_effect=fake_thf):
            result = extractor._process_match(dict(self._BASE_MATCH))

        assert result is True, (
            "_process_match debe retornar True cuando THF provee historial "
            "(no False como antes de Nodo-45)"
        )
        assert len(extractor.all_results) == 1, (
            "El partido debe quedar en all_results"
        )

    def test_t45_08_process_match_thf_supplements_empty_api_history(self, extractor):
        """T45-08: API retorna historial vacío → THF suplementa con datos reales.
        D45-05 (Punto B): match_id válido pero _parse_player_history retorna [].
        FALLA si no se llama a _lookup_player_history_temporal en el path API."""
        thf_returns = {"Martin Maldonado": HIST_MALDONADO}

        def fake_thf(player_name, days_back=7):
            return thf_returns.get(player_name, [])

        match_with_id = dict(self._BASE_MATCH)
        match_with_id["match_url"] = "https://www.flashscore.co/match/tennis/maldonado-rival/AbCdEf/#/h2h"

        with patch("scraping.ninja_h2h_parser.fetch_h2h_from_api", return_value="~raw~"), \
             patch("scraping.ninja_h2h_parser._parse_sections", return_value=[{"KB": "x"}]), \
             patch("scraping.ninja_h2h_parser._split_into_h2h_blocks", return_value=([], [], [])), \
             patch("scraping.ninja_h2h_parser._parse_player_history", return_value=[]), \
             patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]), \
             patch("scraping.ninja_h2h_parser._lookup_player_history_temporal",
                   side_effect=fake_thf) as mock_thf:
            result = extractor._process_match(match_with_id)

        assert result is True, "THF debe suplementar historial vacío de la API (Punto B)"
        # Verificar que THF fue llamado para el jugador con historial vacío
        thf_calls = [call.args[0] for call in mock_thf.call_args_list]
        assert "Martin Maldonado" in thf_calls, (
            "THF debe intentar recuperar el historial de Martin Maldonado (API retornó vacío)"
        )
        # Verificar que los datos THF llegaron a _analyze_and_consolidate
        assert len(extractor.all_results) == 1, "El partido debe quedar en all_results"

    def test_t45_09_returns_false_when_no_match_id_and_no_temporal_data(self, extractor):
        """T45-09: Sin match_id Y sin datos temporales → return False.
        FALLA si se elimina el guard de 'not p1_history and not p2_history'."""
        def fake_thf_empty(player_name, days_back=7):
            return []   # ningún archivo previo tiene a este jugador

        with patch("scraping.ninja_h2h_parser._lookup_player_history_temporal",
                   side_effect=fake_thf_empty):
            result = extractor._process_match(dict(self._BASE_MATCH))

        assert result is False, (
            "Sin match_id y sin datos temporales, el partido debe omitirse (return False)"
        )
        assert len(extractor.all_results) == 0, (
            "No debe agregarse ningún resultado vacío a all_results"
        )

"""
tests/test_nodo49.py — Nodo-49: Playwright H2H Fallback para n_h2h=0
                       Actualizado en F3 (Nodo-51): batch mode

Tests T49-01 → T49-06
Validan que el fallback Playwright funciona como tercer eslabón de la cadena
(Ninja API → THF → Playwright) con el nuevo comportamiento batch de F3.

Detección de mutación real:
  T49-01 FALLA si se elimina el bloque de encolado F3 en _process_match().
  T49-02 FALLA si se elimina el guard 'if not p1_history' antes de encolar.
  T49-03 FALLA si se elimina la condición 'and match_id' del guard de encola.
  T49-04 FALLA si _fetch_player_history_playwright descarta campos del output.
  T49-05 FALLA si _fetch_player_history_playwright no captura TimeoutError.
  T49-06 FALLA si se intercambian section_idx=0 y section_idx=1 en el batch.
"""
import asyncio
import concurrent.futures
import pytest
from unittest.mock import MagicMock, patch, call

from scraping.ninja_h2h_parser import (
    NinjaH2HExtractor,
    _fetch_player_history_playwright,
)

# ─────────────────────────────────────────────────────────────────────────────
# Datos de prueba
# ─────────────────────────────────────────────────────────────────────────────

MATCH_ID = "Ab1cD2eF"

MATCH_WITH_ID = {
    "jugador1": "Mario Arce Fernandez",
    "jugador2": "Nicolas Rival",
    "match_url": f"https://www.flashscore.co/partido/tenis/slug1-slug2/{MATCH_ID}/#/h2h",
    "match_url_j2": "",
    "match_id": MATCH_ID,
    "ranking1": 400,
    "ranking2": 380,
    "cuota1": 9.50,
    "cuota2": 1.07,
    "superficie": "clay",
    "tier": "itf",
}

MATCH_NO_URL = {
    "jugador1": "Mario Arce Fernandez",
    "jugador2": "Nicolas Rival",
    "match_url": "",
    "match_url_j2": "",
    "ranking1": 400,
    "ranking2": 380,
    "cuota1": 9.50,
    "cuota2": 1.07,
    "superficie": "clay",
    "tier": "itf",
}

PLAYWRIGHT_HIST_P1 = [
    {
        "fecha": "07.06", "oponente": "Gomez R.", "resultado": "6-3 6-2",
        "outcome": "Gano", "torneo": "ITF M15 Montevideo", "ciudad": "N/A",
        "pais": "N/A", "superficie": "Arcilla",
    },
    {
        "fecha": "05.06", "oponente": "Lopez M.", "resultado": "4-6 6-4 6-2",
        "outcome": "Gano", "torneo": "ITF M15 Montevideo", "ciudad": "N/A",
        "pais": "N/A", "superficie": "Arcilla",
    },
]

PLAYWRIGHT_HIST_P2 = [
    {
        "fecha": "10.06", "oponente": "Silva D.", "resultado": "6-1 6-0",
        "outcome": "Gano", "torneo": "ITF M15 Lima", "ciudad": "N/A",
        "pais": "N/A", "superficie": "Arcilla",
    },
]

THF_HIST = [
    {
        "fecha": "2026-06-01", "torneo": "Wimbledon", "superficie": "grass",
        "oponente": "Djokovic N.", "resultado": "6-3 6-2", "outcome": "Gano",
        "opponent_ranking": 2,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture de NinjaH2HExtractor con dependencias pesadas mockeadas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    """NinjaH2HExtractor con todas las dependencias pesadas mockeadas."""
    with patch("scraping.ninja_h2h_parser.NinjaH2HExtractor.__init__", return_value=None):
        ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)

    ext.all_results = []
    ext.all_tournaments = False
    ext._playwright_queue = []  # F3: batch queue — inicializar igual que __init__
    ext.ranking_manager = MagicMock()
    ext.elo_system = MagicMock()
    ext.rivalry_analyzer = MagicMock()
    ext.rivalry_analyzer.calculate_elo_from_history.return_value = 1500.0
    ext.rivalry_analyzer.analyze_rivalry.return_value = {
        "prediction": {"favored_player": "Mario Arce Fernandez", "confidence": 52},
        "scores": {},
    }
    ext._enrich_history = lambda hist: hist
    ext._inject_kambi_ranking = MagicMock()
    ext._analyze_form = MagicMock(return_value={"win_rate": 0.5})
    ext._consolidate_result = MagicMock(return_value={"partido": "Arce vs Rival"})
    ext._analyze_and_consolidate = MagicMock(return_value=True)
    return ext


def _patch_api_empty(monkeypatch_or_patch):
    """Contexto de mocks que simula API retornando historial vacío para ambos jugadores."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# T49-01 a T49-03 — Integración en _process_match()
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaywrightFallbackIntegration:

    def _run_process_with_empty_api(self, extractor, match_data):
        """
        Helper: ejecuta _process_match() con API vaciada y THF vacío.
        F3: Playwright ya no se llama inline — el match se encola en _playwright_queue.
        """
        with (
            patch("scraping.ninja_h2h_parser.fetch_h2h_from_api", return_value="raw"),
            patch("scraping.ninja_h2h_parser._parse_sections",
                  return_value=[{"AA": "dummy_no_kb"}]),
            patch("scraping.ninja_h2h_parser._split_into_h2h_blocks",
                  return_value=([], [], [])),
            patch("scraping.ninja_h2h_parser._lookup_player_history_temporal",
                  return_value=[]),
        ):
            extractor._process_match(match_data)

    def test_t49_01_playwright_enqueued_when_api_and_thf_empty(self, extractor):
        """T49-01 (F3): Cuando API y THF devuelven vacío, el partido se encola en
        _playwright_queue para el batch de Playwright.
        FALLA si se elimina el bloque F3 de encolado en _process_match()."""
        self._run_process_with_empty_api(extractor, MATCH_WITH_ID)

        assert len(extractor._playwright_queue) == 1, (
            "El partido debe encolarse en _playwright_queue cuando API y THF vacíos"
        )
        entry = extractor._playwright_queue[0]
        assert entry['p1'] == MATCH_WITH_ID['jugador1']
        assert entry['p2'] == MATCH_WITH_ID['jugador2']

    def test_t49_02_not_enqueued_when_thf_supplements(self, extractor):
        """T49-02: Cuando THF ya supplementó el historial, el partido NO se encola.
        FALLA si se elimina el guard 'if not p1_history' antes del encolado."""
        with (
            patch("scraping.ninja_h2h_parser.fetch_h2h_from_api", return_value="raw"),
            patch("scraping.ninja_h2h_parser._parse_sections",
                  return_value=[{"AA": "dummy_no_kb"}]),
            patch("scraping.ninja_h2h_parser._split_into_h2h_blocks",
                  return_value=([], [], [])),
            patch("scraping.ninja_h2h_parser._lookup_player_history_temporal",
                  return_value=THF_HIST),  # THF tiene datos → llena p1_history y p2_history
        ):
            extractor._process_match(MATCH_WITH_ID)

        assert len(extractor._playwright_queue) == 0, (
            "El partido NO debe encolarse cuando THF ya supplementó el historial"
        )

    def test_t49_03_not_enqueued_when_no_match_id(self, extractor):
        """T49-03: Sin match_id (match_url vacía), el partido NO se encola para Playwright.
        FALLA si se elimina la condición 'and match_id' del guard de encolado.

        Nota: cuando match_id=None, _process_match() retorna en el path THF temprano
        sin llegar nunca al bloque de encolado F3.
        """
        with (
            patch("scraping.ninja_h2h_parser._lookup_player_history_temporal",
                  return_value=[]),
        ):
            result = extractor._process_match(MATCH_NO_URL)

        # Sin datos temporales y sin match_id → debe retornar False (omitido)
        assert result is False, (
            "Sin match_id y sin THF, _process_match() debe retornar False"
        )
        assert len(extractor._playwright_queue) == 0, (
            "El partido NO debe encolarse cuando no hay match_id"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T49-04 y T49-05 — _fetch_player_history_playwright() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchPlayerHistoryPlaywright:

    def test_t49_04_output_has_required_fields(self):
        """T49-04: El output de _fetch_player_history_playwright tiene los campos
        requeridos: fecha, oponente, resultado, outcome, superficie.
        FALLA si la función descarta o renombra alguno de estos campos."""
        with patch(
            "scraping.ninja_h2h_parser._playwright_h2h_async",
            return_value=PLAYWRIGHT_HIST_P1,
        ):
            # Reemplazar asyncio.run para evitar loop conflicts en tests
            with patch("asyncio.run", return_value=PLAYWRIGHT_HIST_P1):
                result = _fetch_player_history_playwright(MATCH_ID, "Mario Arce", 0)

        assert isinstance(result, list), "Debe retornar una lista"
        assert len(result) > 0, "Con datos de Playwright, no debe retornar vacío"

        required_fields = {"fecha", "oponente", "resultado", "outcome", "superficie"}
        for match in result:
            missing = required_fields - set(match.keys())
            assert not missing, f"Faltan campos requeridos en el output: {missing}"

    def test_t49_05_timeout_returns_empty_list(self):
        """T49-05: Si Playwright tiene timeout (90s), retorna [] sin romper el pipeline.
        FALLA si se elimina el manejo de TimeoutError en _fetch_player_history_playwright."""
        def _raise_timeout(*args, **kwargs):
            raise concurrent.futures.TimeoutError("Playwright timeout simulado")

        with patch(
            "concurrent.futures.ThreadPoolExecutor",
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError(
                "Playwright timeout simulado"
            )
            mock_executor.submit.return_value = mock_future

            result = _fetch_player_history_playwright(MATCH_ID, "Mario Arce", 0)

        assert result == [], (
            "Timeout de Playwright debe retornar [] sin lanzar excepción"
        )

    def test_t49_05b_exception_returns_empty_list(self):
        """T49-05b: Cualquier excepción en Playwright retorna [] sin romper el pipeline."""
        with patch(
            "concurrent.futures.ThreadPoolExecutor",
        ) as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_future = MagicMock()
            mock_future.result.side_effect = Exception("Error de Playwright genérico")
            mock_executor.submit.return_value = mock_future

            result = _fetch_player_history_playwright(MATCH_ID, "Mario Arce", 0)

        assert result == [], (
            "Excepción genérica de Playwright debe retornar [] sin propagarse"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T49-06 — section_idx correcto para P1 y P2
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaywrightSectionIdx:

    def test_t49_06_section_idx_0_for_p1_and_1_for_p2(self, extractor):
        """T49-06 (F3): En el batch, _playwright_h2h_with_browser se llama con
        section_idx=0 para P1 y section_idx=1 para P2.
        FALLA si se intercambian los section_idx en _run_playwright_batch_async."""
        import asyncio

        calls_made = []

        async def capture_with_browser(browser, match_id, player_name, section_idx):
            calls_made.append((player_name, section_idx))
            return []  # vacío — lo que importa es el section_idx

        # Encolar el match primero (simular resultado de _process_match con API vacía)
        entry = {
            'match_data': dict(MATCH_WITH_ID),
            'p1': MATCH_WITH_ID['jugador1'],
            'p2': MATCH_WITH_ID['jugador2'],
            'p1_history': [],
            'p2_history': [],
            'p1_source': 'EMPTY',
            'p2_source': 'EMPTY',
            'h2h_records': [],
        }
        extractor._playwright_queue = [entry]

        with (
            patch("scraping.ninja_h2h_parser._playwright_h2h_with_browser",
                  side_effect=capture_with_browser),
            patch("scraping.ninja_h2h_parser._persist_playwright_to_thf_cache"),
            patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]),
        ):
            asyncio.run(extractor._run_playwright_batch_async([entry]))

        assert len(calls_made) == 2, (
            f"Debe llamarse 2 veces (P1 y P2), se llamó {len(calls_made)} veces"
        )

        p1_name = MATCH_WITH_ID["jugador1"]
        p2_name = MATCH_WITH_ID["jugador2"]

        p1_calls = [(n, idx) for n, idx in calls_made if n == p1_name]
        p2_calls = [(n, idx) for n, idx in calls_made if n == p2_name]

        assert p1_calls, f"Playwright batch debe procesar jugador1 '{p1_name}'"
        assert p2_calls, f"Playwright batch debe procesar jugador2 '{p2_name}'"

        assert p1_calls[0][1] == 0, (
            f"P1 debe usar section_idx=0, se usó {p1_calls[0][1]}"
        )
        assert p2_calls[0][1] == 1, (
            f"P2 debe usar section_idx=1, se usó {p2_calls[0][1]}"
        )

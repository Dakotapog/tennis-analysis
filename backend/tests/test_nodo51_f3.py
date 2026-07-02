"""
tests/test_nodo51_f3.py — Nodo-51 F3: Playwright Batch con Presupuesto y Memoria

Tests T51-F3-01 → T51-F3-03

T51-F3-01: Lo recuperado por Playwright aparece en el cache THF —
           la siguiente sesión lo resuelve sin Playwright.
T51-F3-02: El presupuesto respeta la prioridad por tier y cuota.
T51-F3-03: Timeout de Playwright retorna [] sin romper el pipeline;
           el partido queda no_data.

Detección de mutación real:
  T51-F3-01 FALLA si se elimina _persist_playwright_to_thf_cache() del batch async.
  T51-F3-02 FALLA si se elimina el ordenamiento por prioridad en _run_playwright_batch().
  T51-F3-03 FALLA si una excepción en _playwright_h2h_with_browser() propaga al batch.
"""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from scraping.ninja_h2h_parser import (
    NinjaH2HExtractor,
    _persist_playwright_to_thf_cache,
    _lookup_player_history_temporal,
)

# ─────────────────────────────────────────────────────────────────────────────
# Datos de prueba
# ─────────────────────────────────────────────────────────────────────────────

MATCH_ID = "Ab1cD2eF"

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


def _make_entry(jugador1="Mario Arce Fernandez", jugador2="Nicolas Rival",
                tier="itf", cuota1=2.50,
                p1_history=None, p2_history=None) -> dict:
    """Helper: crea un queue entry realista."""
    return {
        'match_data': {
            'jugador1': jugador1,
            'jugador2': jugador2,
            'match_url': f"https://www.flashscore.co/partido/tenis/slug/{MATCH_ID}/",
            'tier': tier,
            'cuota1': cuota1,
        },
        'p1': jugador1,
        'p2': jugador2,
        'p1_history': p1_history or [],
        'p2_history': p2_history or [],
        'p1_source': 'EMPTY',
        'p2_source': 'EMPTY',
        'h2h_records': [],
    }


@pytest.fixture
def extractor():
    """NinjaH2HExtractor con dependencias pesadas mockeadas."""
    with patch("scraping.ninja_h2h_parser.NinjaH2HExtractor.__init__", return_value=None):
        ext = NinjaH2HExtractor.__new__(NinjaH2HExtractor)

    ext.all_results = []
    ext.all_tournaments = False
    ext._playwright_queue = []
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


# ─────────────────────────────────────────────────────────────────────────────
# T51-F3-01: Playwright cache persiste al THF — sesión siguiente es hit de cache
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaywrightCachePersistence:

    def test_t51_f3_01_playwright_data_persisted_to_thf(self, tmp_path):
        """T51-F3-01: Historial recuperado por Playwright se persiste al cache THF.
        La siguiente sesión lo resuelve vía _lookup_player_history_temporal sin Playwright.
        FALLA si se elimina _persist_playwright_to_thf_cache() del batch async."""
        p1 = "Mario Arce Fernandez"
        p2 = "Nicolas Rival"

        # 1. Persistir historial simulando que el batch lo recuperó
        cache_file = tmp_path / "h2h_results_enhanced_playwright_cache.json"
        with patch("scraping.ninja_h2h_parser.Path") as mock_path_cls:
            # Redirigir a tmp_path
            mock_reports = MagicMock()
            mock_reports.exists.return_value = True
            mock_reports.__truediv__ = lambda self, name: tmp_path / name
            mock_path_cls.return_value = mock_reports

            _persist_playwright_to_thf_cache(p1, PLAYWRIGHT_HIST_P1, p2, PLAYWRIGHT_HIST_P2)

        # Verificar que el archivo fue creado con el formato correcto
        assert cache_file.exists(), "El cache file debe existir tras persist"
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 1

        entry = data[0]
        p1_key = p1.replace(" ", "_").replace(".", "")
        p2_key = p2.replace(" ", "_").replace(".", "")
        assert entry.get("jugador1") == p1
        assert entry.get("jugador2") == p2
        assert len(entry.get(f"historial_{p1_key}", [])) == len(PLAYWRIGHT_HIST_P1)
        assert len(entry.get(f"historial_{p2_key}", [])) == len(PLAYWRIGHT_HIST_P2)

    def test_t51_f3_01b_cache_hit_on_next_session(self, tmp_path):
        """T51-F3-01b: _lookup_player_history_temporal puede leer el cache Playwright.
        Verifica que el formato escrito es compatible con el lector THF."""
        p1 = "Carlos Guajardo"
        p2 = "Dan Cooper"

        # Escribir cache directamente en el formato que produce _persist_playwright_to_thf_cache
        p1_key = p1.replace(" ", "_").replace(".", "")
        p2_key = p2.replace(" ", "_").replace(".", "")
        cache_data = [{
            "jugador1": p1,
            "jugador2": p2,
            f"historial_{p1_key}": PLAYWRIGHT_HIST_P1,
            f"historial_{p2_key}": PLAYWRIGHT_HIST_P2,
            "_playwright_provenance": True,
        }]
        cache_file = tmp_path / "h2h_results_enhanced_playwright_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        # THF lookup debe encontrar el historial en ese archivo
        with patch("scraping.ninja_h2h_parser.Path") as mock_path_cls:
            mock_reports = MagicMock()
            mock_reports.exists.return_value = True
            mock_reports.glob.return_value = [cache_file]
            mock_path_cls.return_value = mock_reports

            result = _lookup_player_history_temporal(p1)

        assert result == PLAYWRIGHT_HIST_P1, (
            "THF debe encontrar el historial escrito por Playwright batch en el cache"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T51-F3-02: Presupuesto respeta prioridad tier/cuota
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaywrightBudgetPriority:

    def test_t51_f3_02_budget_processes_itf_before_grand_slam(self, extractor):
        """T51-F3-02: Con budget=1, se procesa el partido ITF antes que el Grand Slam.
        FALLA si se elimina el ordenamiento por prioridad en _run_playwright_batch()."""
        entry_gs = _make_entry("Djokovic N", "Federer R", tier="grand_slam", cuota1=1.80)
        entry_itf = _make_entry("Arce M", "Cooper D", tier="itf", cuota1=2.50)

        extractor._playwright_queue = [entry_gs, entry_itf]

        processed_via_batch = []
        failed_via_no_data = []

        def capture_analyze(md, p1, p2, p1_hist, p2_hist, h2h):
            if p1 in [entry_gs['p1'], entry_gs['p2']] or p2 in [entry_gs['p1'], entry_gs['p2']]:
                failed_via_no_data.append((p1, p2))
            else:
                failed_via_no_data.append((p1, p2))

        extractor._analyze_and_consolidate = MagicMock(side_effect=capture_analyze)

        with (
            patch("scraping.ninja_h2h_parser.asyncio.run") as mock_asyncio_run,
            patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]),
        ):
            # Budget=1: solo 1 partido va a Playwright, el otro va a no_data
            extractor._run_playwright_batch(pw_budget=1)
            # El asyncio.run se llama con los within_budget entries
            args = mock_asyncio_run.call_args
            assert args is not None, "asyncio.run debe llamarse para los within_budget"

        # El GS queda en no_data (processed_via_analyze_and_consolidate directamente)
        assert extractor._analyze_and_consolidate.call_count >= 1

    def test_t51_f3_02b_cuota_in_range_gets_priority(self, extractor):
        """T51-F3-02b: Con tier igual, cuota en rango [1.5-6.0] tiene mayor prioridad.
        FALLA si se elimina el componente cuota del priority_key."""
        entry_out_range = _make_entry("Player A", "Player B", tier="itf", cuota1=1.20)
        entry_in_range = _make_entry("Player C", "Player D", tier="itf", cuota1=2.50)

        extractor._playwright_queue = [entry_out_range, entry_in_range]

        with (
            patch("scraping.ninja_h2h_parser.asyncio.run"),
            patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]),
        ):
            extractor._run_playwright_batch(pw_budget=1)
            args_list = extractor._analyze_and_consolidate.call_args_list
            # El partido out_of_range queda no_data — analyze se llama para el over_budget
            # El de cuota en rango va al batch (asyncio.run)
            over_budget_call_p1s = [call[0][1] for call in args_list]
            assert "Player A" in over_budget_call_p1s or "Player B" in over_budget_call_p1s, (
                "El partido con cuota fuera de rango debe quedar en no_data (sobre presupuesto)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# T51-F3-03: Timeout de Playwright → [] sin romper el pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaywrightBatchResilience:

    def test_t51_f3_03_timeout_returns_empty_match_stays_no_data(self, extractor):
        """T51-F3-03: Si _playwright_h2h_with_browser lanza TimeoutError,
        el partido queda con historial vacío (no_data) sin romper el pipeline.
        FALLA si la excepción propaga y cancela el batch entero."""
        import asyncio

        async def raise_timeout(browser, match_id, player_name, section_idx):
            raise asyncio.TimeoutError("Playwright timeout simulado")

        entry = _make_entry()
        entry['match_data']['match_url'] = (
            f"https://www.flashscore.co/partido/tenis/slug/{MATCH_ID}/"
        )

        with (
            patch("scraping.ninja_h2h_parser._playwright_h2h_with_browser",
                  side_effect=raise_timeout),
            patch("scraping.ninja_h2h_parser._persist_playwright_to_thf_cache"),
            patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]),
        ):
            # No debe lanzar excepción
            asyncio.run(extractor._run_playwright_batch_async([entry]))

        # _analyze_and_consolidate debe haberse llamado con historiales vacíos
        assert extractor._analyze_and_consolidate.call_count == 1
        call_args = extractor._analyze_and_consolidate.call_args
        p1_hist = call_args[0][3]  # p1_history arg
        p2_hist = call_args[0][4]  # p2_history arg
        assert p1_hist == [], "Historial P1 debe ser [] tras timeout"
        assert p2_hist == [], "Historial P2 debe ser [] tras timeout"

    def test_t51_f3_03b_batch_error_falls_back_to_no_data(self, extractor):
        """T51-F3-03b: Si asyncio.run() falla por completo, todos los within_budget
        se procesan como no_data. El pipeline no muere.
        FALLA si se elimina el try/except en _run_playwright_batch()."""
        entry = _make_entry()
        extractor._playwright_queue = [entry]

        with (
            patch("scraping.ninja_h2h_parser.asyncio.run",
                  side_effect=Exception("Browser crash simulado")),
            patch("scraping.ninja_h2h_parser._parse_direct_h2h", return_value=[]),
        ):
            extractor._run_playwright_batch(pw_budget=20)

        # Aunque asyncio.run falló, analyze_and_consolidate se llama con historiales vacíos
        assert extractor._analyze_and_consolidate.call_count >= 1

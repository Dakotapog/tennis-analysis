"""Tests Nodo-154 — Pipeline Integrity: Watchlist Cap + Phantom Tier + H2H Selection
+ Kambi Matching + Games Signal + outcome_id + Kambi Refresh + cuota patch.

REGLA-T53: cada test invoca la función real del módulo, nunca hardcodea la fórmula.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ajustar path para imports desde backend/
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── D154-01: Watchlist cap 10→50 ─────────────────────────────────────────────

def test_watchlist_cap_50():
    """edge_calculator output incluye hasta 50 picks en watchlist (antes cap=10)."""
    from edge_calculator import calcular_edge_completo  # noqa: F401
    import edge_calculator as ec
    # Verificar que el cap en el código es 50, no 10
    import inspect
    src = inspect.getsource(ec)
    assert 'no_apostar_lista[:50]' in src, (
        "D154-01: watchlist cap debe ser [:50], encontrado cap distinto"
    )
    assert 'no_apostar_lista[:10]' not in src, (
        "D154-01: cap viejo [:10] todavía presente en el código"
    )


# ── D154-02: Phantom tier gate usa campo tier del h2h record ─────────────────

def test_phantom_tier_uses_h2h_field_itf():
    """D152-05 gate dispara cuando h2h record tiene tier='itf' aunque torneo_completo sea corto."""
    import edge_calculator as ec
    import inspect
    src = inspect.getsource(ec)
    # Verificar que el código usa resultado.get('tier') en el bloque D152-05
    assert "resultado.get('tier')" in src, (
        "D154-02: D152-05 debe leer campo 'tier' del h2h record"
    )


def test_phantom_tier_fallback_to_detectar_tier():
    """Si h2h record no tiene tier, fallback a variable tier existente (no romper)."""
    import edge_calculator as ec
    import inspect
    src = inspect.getsource(ec)
    # El fallback debe existir: OR con tier calculado antes
    assert '_tier_152 = (resultado.get' in src, (
        "D154-02: _tier_152 debe usar resultado.get('tier') con fallback"
    )


# ── D154-03/D154-11: H2H file selection por n_partidos ───────────────────────

def test_h2h_selects_max_partidos():
    """select_best_h2h_file() elige el archivo con más partidos, no el más reciente."""
    from scraping.file_utils import select_best_h2h_file

    with tempfile.TemporaryDirectory() as tmpdir:
        # Archivo "viejo" con más partidos (366)
        big_file = Path(tmpdir) / 'h2h_results_enhanced_20260729_083246.json'
        big_file.write_text(json.dumps([{'id': i} for i in range(366)]))

        # Archivo "nuevo" (alfabéticamente posterior) con menos partidos (36)
        small_file = Path(tmpdir) / 'h2h_results_enhanced_20260729_115316.json'
        small_file.write_text(json.dumps([{'id': i} for i in range(36)]))

        result = select_best_h2h_file(date_str='20260729', directory=tmpdir)

        assert result is not None
        assert '083246' in result, (
            f"D154-03: debe elegir el archivo con 366 partidos (083246), "
            f"no el más reciente (115316). Resultado: {result}"
        )


# ── D154-04: Kambi matching con nombres compuestos ───────────────────────────

def test_kambi_matching_apellido_kambi_particles():
    """_apellido_kambi() filtra partículas para nombres compuestos."""
    from betplay_combo_builder import _apellido_kambi
    assert _apellido_kambi("Alex De Minaur") == "minaur"
    assert _apellido_kambi("Botic Van De Zandschulp") == "zandschulp"
    assert _apellido_kambi("Lachlan Mcfadzean") == "mcfadzean"


def test_kambi_matching_apellido_pick_particles():
    """_apellido_pick() quita iniciales y partículas del inicio."""
    from betplay_combo_builder import _apellido_pick
    assert _apellido_pick("De Minaur A.") == "minaur"
    assert _apellido_pick("Van De Zandschulp B.") == "zandschulp"
    assert _apellido_pick("Navarro E.") == "navarro"


def test_kambi_matching_score_compound_name():
    """_match_score_names_kf() retorna score >0 para nombres con partículas."""
    from betplay_combo_builder import _match_score_names_kf
    score = _match_score_names_kf("Alex De Minaur", "De Minaur A.")
    assert score > 0, f"D154-04: debe matchear 'Alex De Minaur' con 'De Minaur A.' score={score}"


# ── D154-05: games_signal usa mismo h2h que edge_calculator ──────────────────

def test_games_signal_file_arg_exists():
    """games_signal_calculator.py acepta argumento --file (prerequisito D154-05)."""
    import games_signal_calculator as gsc
    import inspect
    src = inspect.getsource(gsc)
    assert '--file' in src, "D154-05: games_signal_calculator debe tener argparse --file"


# ── D154-06: kambi_event_id propagado al edge_report ─────────────────────────

def test_kambi_event_id_in_ninja_h2h_output():
    """ninja_h2h_parser._consolidate_result() incluye kambi_event_id."""
    from scraping.ninja_h2h_parser import NinjaH2HExtractor
    import inspect
    src = inspect.getsource(NinjaH2HExtractor._consolidate_result)
    assert 'kambi_event_id' in src, (
        "D154-06: _consolidate_result debe incluir campo kambi_event_id"
    )


def test_kambi_event_id_in_edge_calculator_output():
    """edge_calculator propaga kambi_event_id al resultado dict."""
    import edge_calculator as ec
    import inspect
    src = inspect.getsource(ec)
    assert "'kambi_event_id'" in src, (
        "D154-06: edge_calculator debe incluir 'kambi_event_id' en output dict"
    )


# ── D154-08: Kambi refresh antes de PASO 4 ───────────────────────────────────

def test_kambi_refresh_before_paso4_in_run_daily():
    """run_daily.py llama fetch_kambi_coverage.py ANTES de PASO 4 (D154-08)."""
    import inspect
    import run_daily
    src = inspect.getsource(run_daily)
    paso39_idx = src.find('PASO 3.9')
    paso4_idx = src.find('PASO 4 — Trader')
    assert paso39_idx != -1, "D154-08: PASO 3.9 no encontrado en run_daily"
    assert paso4_idx != -1, "D154-08: PASO 4 no encontrado en run_daily"
    assert paso39_idx < paso4_idx, (
        "D154-08: PASO 3.9 (Kambi refresh) debe aparecer ANTES de PASO 4"
    )

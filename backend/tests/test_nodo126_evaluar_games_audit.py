"""tests/test_nodo126_evaluar_games_audit.py
REGLA-T53: Auditoría EvalGames Bridge — Nodo-126
D126-01 same-match gate | D126-03 file-by-most-matches | D126-04 tier filter | D126-06 confidence norm
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── D126-01: Same-match gate ────────────────────────────────────────────────

def _dedup_signals(signals):
    """Replica exacta del gate D126-01 en build_evaluar_games_combos()."""
    seen: dict = {}
    for s in signals:
        p = s["partido"]
        if p not in seen or s["cuota"] > seen[p]["cuota"]:
            seen[p] = s
    return list(seen.values())


def test_D126_01_same_match_dedup_keeps_highest_cuota():
    """Mismo partido con dos cuotas → retiene la cuota más alta."""
    signals = [
        {"partido": "Lam S. vs Alvisi E.", "cuota": 2.16, "direccion": "UNDER"},
        {"partido": "Lam S. vs Alvisi E.", "cuota": 2.50, "direccion": "UNDER"},
        {"partido": "Hurrion M. vs Draxl L.", "cuota": 1.80, "direccion": "UNDER"},
    ]
    result = _dedup_signals(signals)
    lam = next(r for r in result if "Lam" in r["partido"])
    assert lam["cuota"] == 2.50


def test_D126_01_combo_never_has_two_legs_same_partido():
    """Después de dedup ningún partido aparece dos veces en all_signals."""
    signals = [
        {"partido": "A vs B", "cuota": 2.00},
        {"partido": "A vs B", "cuota": 1.85},
        {"partido": "C vs D", "cuota": 2.10},
    ]
    result = _dedup_signals(signals)
    partidos = [r["partido"] for r in result]
    assert len(partidos) == len(set(partidos))
    assert len(result) == 2


# ─── D126-03: Selección por cantidad de partidos ─────────────────────────────

def test_D126_03_generar_tabla_selects_file_with_most_matches(tmp_path):
    """find_latest_h2h_file prefiere archivo con más partidos, no el más reciente."""
    from datetime import datetime
    today = datetime.now().strftime('%Y%m%d')

    fa = tmp_path / f'h2h_results_enhanced_{today}_080000.json'
    fa.write_text(json.dumps({"partidos": [{"id": i} for i in range(10)]}))

    fb = tmp_path / f'h2h_results_enhanced_{today}_103000.json'
    fb.write_text(json.dumps({"partidos": [{"id": i} for i in range(25)]}))

    import generar_tabla_favoritos2 as gtf
    with patch('generar_tabla_favoritos2.glob.glob', return_value=[str(fa), str(fb)]):
        result = gtf.find_latest_h2h_file()

    assert result == str(fb), f"Esperado archivo con 25 partidos, obtuvo {result}"


# ─── D126-04 (D130): Tier filter — solo M15/W15 y menores bloqueados ─────────

def test_D126_04_bridge_skips_m15_tier():
    """tier='m15' debe estar en _TIERS_SIN_KAMBI del bridge."""
    import scripts.evaluar_games_bridge as bridge
    assert 'm15' in bridge._TIERS_SIN_KAMBI


def test_D126_04_bridge_skips_w15_tier():
    """tier='w15' debe estar en _TIERS_SIN_KAMBI del bridge."""
    import scripts.evaluar_games_bridge as bridge
    assert 'w15' in bridge._TIERS_SIN_KAMBI


def test_D126_04_bridge_itf_generic_not_blocked():
    """tier='itf' genérico NO debe estar en _TIERS_SIN_KAMBI (D128-02/D130)."""
    import scripts.evaluar_games_bridge as bridge
    assert 'itf' not in bridge._TIERS_SIN_KAMBI


def test_D126_04_bridge_m25_not_blocked():
    """tier='m25' NO debe estar en _TIERS_SIN_KAMBI — Kambi cubre algunos M25."""
    import scripts.evaluar_games_bridge as bridge
    assert 'm25' not in bridge._TIERS_SIN_KAMBI


# ─── D126-06: Normalización confidence ───────────────────────────────────────

def test_D126_06_confidence_normalized_in_bridge_picks():
    """Lambda de normalización: valor >=1 se divide /100; decimal se pasa tal cual."""
    normalize = lambda c: c / 100 if c and c >= 1 else (c or 0)

    assert normalize(58.6) == pytest.approx(0.586)
    assert normalize(0.586) == pytest.approx(0.586)
    assert normalize(None) == 0
    assert normalize(100.0) == pytest.approx(1.0)
    assert normalize(54.40) == pytest.approx(0.5440)

"""
tests/test_nodo135_games_live_api.py — REGLA-T53 Nodo-135 EvalGames Live API Fix

4 tests:
  1. test_D135_01_extrae_cuota_via_betoffer_endpoint
       mock requests.get al endpoint betoffer/event/{id} con Total de juegos UNDER
       → retorna cuota 1.80

  2. test_D135_02_filtra_set_level_markets
       betOffer label "Total de juegos - Set 3" (set-level)
       → retorna None (D135-02: excluir sub-mercados)

  3. test_D135_02_acepta_match_level_market
       betOffer label "Total de juegos" (match-level, sin " - Set ")
       → retorna cuota correcta

  4. test_D135_01_retorna_none_si_mercado_inexistente
       betOffers sin ningún "Total de juegos"
       → retorna None limpiamente

REGLA-T53: todos invocan _extract_games_cuota_live del módulo real — nunca hardcodean lógica.
"""
import json
import sys
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import live_desk as ld


@contextmanager
def _mock_urlopen(betoffers: list):
    """Contexto que parchea urllib.request.urlopen para devolver betOffers."""
    body = json.dumps({"betOffers": betoffers}).encode()

    class _FakeResponse:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    with patch("live_desk.urllib.request.urlopen", return_value=_FakeResponse()) as mock:
        yield mock


def _make_betoffer(label: str, under_odds: int = 1800, line_milli: int = 31500) -> dict:
    """Construye un betOffer Kambi mínimo con mercado UNDER."""
    return {
        "criterion": {"label": label},
        "outcomes": [
            {
                "label": "Menos de",
                "englishLabel": "Under",
                "odds": under_odds,
                "line": line_milli,
            },
            {
                "label": "Más de",
                "englishLabel": "Over",
                "odds": 2100,
                "line": line_milli,
            },
        ],
    }


# ─── Test 1: Extrae cuota via endpoint betoffer/event ────────────────────────

def test_D135_01_extrae_cuota_via_betoffer_endpoint():
    """urlopen al endpoint betoffer/event/{id} → cuota 1.80 retornada para UNDER 31.5."""
    betoffer = _make_betoffer("Total de juegos", under_odds=1800, line_milli=31500)

    with _mock_urlopen([betoffer]) as mock_urlopen:
        result = ld._extract_games_cuota_live(1028465663, "UNDER", 31.5)

    assert result == 1.80, f"Esperaba 1.80, obtuvo {result}"
    # Verificar que llamó al endpoint correcto
    called_req = mock_urlopen.call_args[0][0]
    assert "betoffer/event/1028465663" in called_req.full_url, f"URL incorrecta: {called_req.full_url}"


# ─── Test 2: Filtra mercados set-level ───────────────────────────────────────

def test_D135_02_filtra_set_level_markets():
    """D135-02: label 'Total de juegos - Set 3' → None (sub-mercado de set, no match total)."""
    betoffer_set = _make_betoffer("Total de juegos - Set 3", under_odds=1200, line_milli=7500)

    with _mock_urlopen([betoffer_set]):
        result = ld._extract_games_cuota_live(1028465663, "UNDER", 7.5)

    assert result is None, f"Esperaba None para set-level market, obtuvo {result}"


# ─── Test 3: Acepta mercado match-level ──────────────────────────────────────

def test_D135_02_acepta_match_level_market():
    """D135-02: label 'Total de juegos' (match-level, sin ' - Set ') → cuota correcta."""
    betoffer_match = _make_betoffer("Total de juegos", under_odds=2100, line_milli=22500)
    betoffer_set   = _make_betoffer("Total de juegos - Set 3", under_odds=1200, line_milli=7500)
    # Ambos presentes — solo el match-level debe activar

    with _mock_urlopen([betoffer_set, betoffer_match]):
        result = ld._extract_games_cuota_live(9999, "UNDER", 22.5)

    assert result == 2.10, f"Esperaba 2.10 (match-level), obtuvo {result}"


# ─── Test 4: None si no hay mercado ──────────────────────────────────────────

def test_D135_01_retorna_none_si_mercado_inexistente():
    """betOffers sin 'Total de juegos' (ej: solo ML) → None limpio."""
    betoffer_ml = {
        "criterion": {"label": "Cuotas del partido"},
        "outcomes": [
            {"label": "Derek Pham", "englishLabel": "Derek Pham", "odds": 1800, "line": 0},
            {"label": "Ymerali Ibraimi", "englishLabel": "Ymerali Ibraimi", "odds": 2100, "line": 0},
        ],
    }

    with _mock_urlopen([betoffer_ml]):
        result = ld._extract_games_cuota_live(1028465663, "UNDER", 31.5)

    assert result is None, f"Esperaba None cuando no hay mercado Total de juegos, obtuvo {result}"

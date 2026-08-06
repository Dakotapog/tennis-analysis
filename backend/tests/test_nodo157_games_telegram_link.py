"""
tests/test_nodo157_games_telegram_link.py — REGLA-T53 D157-06

Bug real reportado por el usuario 2026-08-02: llegó el mensaje "GAMES COMBOS —
Totales (Nodo-40) / GamesLive @6.45 ..." a Telegram pero sin el link de
Betplay listo para apostar. `_enviar_games_telegram()` (betplay_combo_builder.py)
nunca incluyó el redirect link — a diferencia de `enviar_combos_telegram()`
(sistema de favoritos), que sí usa REDIRECT_BASE + outcome_ids desde siempre.

Fix: mismo patrón — `[ABRIR {label}]({REDIRECT_BASE}{ids_str})` por combo.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import betplay_combo_builder as bcb


def _capture_sent_text(games_links, metadata):
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        status = 200

    def _fake_urlopen(req, timeout=10):
        captured["payload"] = json.loads(req.data.decode())
        return _FakeResp()

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        bcb._enviar_games_telegram(games_links, metadata)
    return captured.get("payload", {}).get("text", "")


def test_157_30_games_telegram_incluye_link_redirect():
    games_links = [{
        "label": "GamesLive", "cuota_combo": 6.45, "retorno": 12903, "stake": 2000,
        "outcome_ids": ["111", "222", "333"],
        "legs": [
            {"direccion": "UNDER", "linea": 30.5, "cuota": 1.65},
            {"direccion": "UNDER", "linea": 26.5, "cuota": 1.70},
            {"direccion": "UNDER", "linea": 23.5, "cuota": 2.30},
        ],
    }]
    metadata = {"calibracion_n": 0, "total_stake": 2000}

    text = _capture_sent_text(games_links, metadata)

    expected_url = f"{bcb.REDIRECT_BASE}111,222,333"
    assert expected_url in text
    assert "[ABRIR GamesLive]" in text
    assert "GamesLive" in text and "@6.45" in text


def test_157_31_games_telegram_sin_outcome_ids_no_lanza_ni_agrega_link():
    games_links = [{
        "label": "GamesA", "cuota_combo": 3.2, "retorno": 6400, "stake": 2000,
        "outcome_ids": [],
        "legs": [{"direccion": "OVER", "linea": 21.5, "cuota": 1.9}],
    }]
    metadata = {"calibracion_n": 12, "total_stake": 2000}

    text = _capture_sent_text(games_links, metadata)

    assert bcb.REDIRECT_BASE not in text
    assert "GamesA" in text


def test_157_32_games_telegram_multi_combo_cada_uno_su_link():
    games_links = [
        {
            "label": "GamesA", "cuota_combo": 2.5, "retorno": 5000, "stake": 2000,
            "outcome_ids": ["10", "20"],
            "legs": [{"direccion": "UNDER", "linea": 20.5, "cuota": 1.8}],
        },
        {
            "label": "GamesB", "cuota_combo": 4.0, "retorno": 8000, "stake": 2000,
            "outcome_ids": ["30", "40"],
            "legs": [{"direccion": "OVER", "linea": 22.5, "cuota": 2.0}],
        },
    ]
    metadata = {"calibracion_n": 5, "total_stake": 4000}

    text = _capture_sent_text(games_links, metadata)

    assert f"{bcb.REDIRECT_BASE}10,20" in text
    assert f"{bcb.REDIRECT_BASE}30,40" in text

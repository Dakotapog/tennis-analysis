"""REGLA-T53 — Nodo-170 (D170-01)

Mismo gap de D157-06 (test_nodo157_games_telegram_link.py) encontrado por auditoría
en las 3 funciones hermanas de _enviar_games_telegram(): _enviar_safe_telegram(),
_enviar_mega_telegram(), _enviar_sistema_telegram() nunca incluyeron el link
REDIRECT_BASE aunque sus combos sí guardan outcome_ids desde su creación.

Fix: mismo patrón — link Markdown `[ABRIR ...](REDIRECT_BASE+ids_str)` por combo.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import betplay_combo_builder as bcb


def _capture_sent_text(fn, links, metadata):
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
        fn(links, metadata)
    return captured.get("payload", {}).get("text", "")


def test_170_01_safe_telegram_incluye_link_redirect():
    safe_links = [{
        "combo_idx": 1, "cuota_combo": 3.2, "retorno": 6400, "stake": 2000,
        "p_both": 0.30,
        "outcome_ids": ["111", "222"],
        "legs": [
            {"jugador": "Alcaraz C.", "cuota_kambi": 1.4},
            {"jugador": "Sinner J.", "cuota_kambi": 1.5},
        ],
    }]
    metadata = {"total_stake": 2000}

    text = _capture_sent_text(bcb._enviar_safe_telegram, safe_links, metadata)

    expected_url = f"{bcb.REDIRECT_BASE}111,222"
    assert expected_url in text
    assert "[ABRIR Safe 1]" in text


def test_170_02_safe_telegram_sin_outcome_ids_no_agrega_link():
    safe_links = [{
        "combo_idx": 2, "cuota_combo": 2.1, "retorno": 4200, "stake": 2000,
        "p_both": 0.40,
        "outcome_ids": [],
        "legs": [{"jugador": "Djokovic N.", "cuota_kambi": 1.3}],
    }]
    metadata = {"total_stake": 2000}

    text = _capture_sent_text(bcb._enviar_safe_telegram, safe_links, metadata)

    assert bcb.REDIRECT_BASE not in text


def test_170_03_mega_telegram_incluye_link_redirect():
    mega_links = [{
        "combo_idx": 1, "piernas": 6, "cuota_combo": 350, "retorno": 175000,
        "outcome_ids": ["10", "20", "30", "40", "50", "60"],
        "legs": [{"jugador": f"Player{i}"} for i in range(6)],
    }]
    metadata = {"total_stake": 500}

    text = _capture_sent_text(bcb._enviar_mega_telegram, mega_links, metadata)

    expected_url = f"{bcb.REDIRECT_BASE}10,20,30,40,50,60"
    assert expected_url in text
    assert "[ABRIR]" in text


def test_170_04_mega_telegram_sin_outcome_ids_no_agrega_link():
    mega_links = [{
        "combo_idx": 2, "piernas": 6, "cuota_combo": 200, "retorno": 100000,
        "outcome_ids": [],
        "legs": [{"jugador": f"Player{i}"} for i in range(6)],
    }]
    metadata = {"total_stake": 500}

    text = _capture_sent_text(bcb._enviar_mega_telegram, mega_links, metadata)

    assert bcb.REDIRECT_BASE not in text


def test_170_05_sistema_telegram_incluye_link_redirect():
    system_links = [{
        "combo_idx": 1, "piernas": 5, "cuota_combo": 45.5, "retorno": 22750,
        "excluye": "Kalieva E.",
        "outcome_ids": ["71", "72", "73", "74", "75"],
        "legs": [{"jugador": f"Player{i}"} for i in range(5)],
    }]
    metadata = {"n_piernas_pool": 6, "total_stake": 3500}

    text = _capture_sent_text(bcb._enviar_sistema_telegram, system_links, metadata)

    expected_url = f"{bcb.REDIRECT_BASE}71,72,73,74,75"
    assert expected_url in text
    assert "[ABRIR]" in text


def test_170_06_sistema_telegram_sin_outcome_ids_no_agrega_link():
    system_links = [{
        "combo_idx": 2, "piernas": 5, "cuota_combo": 30.0, "retorno": 15000,
        "excluye": "Michelsen A.",
        "outcome_ids": [],
        "legs": [{"jugador": f"Player{i}"} for i in range(5)],
    }]
    metadata = {"n_piernas_pool": 6, "total_stake": 3500}

    text = _capture_sent_text(bcb._enviar_sistema_telegram, system_links, metadata)

    assert bcb.REDIRECT_BASE not in text

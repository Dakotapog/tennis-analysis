"""
Nodo-128 — REGLA-T53: tests para alias apellido Wplay en _build_p8_books()
D128-01: Wplay full names → apellido alias para matchear picks abreviados del edge_report.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scraping.dual_book_client import _norm, best_price


# ── Helper que replica la lógica D128-01 de live_desk._build_p8_books() ─────

def _aplicar_alias_wplay(feeds_wplay: dict) -> dict:
    """Replica exactamente el bloque D128-01 de _build_p8_books()."""
    aliases = {}
    for key, entry in feeds_wplay.items():
        parts = key.split()
        if len(parts) >= 2:
            sn = " ".join(parts[1:])
            if sn not in feeds_wplay and sn not in aliases:
                aliases[sn] = entry
    result = dict(feeds_wplay)
    result.update(aliases)
    return result


# ── Tests ────────────────────────────────────────────────────────────────────

def test_alias_wplay_apellido_permite_best_price():
    """
    D128-01: edge_report usa "Van De Zandschulp" pero Wplay tiene "Botic Van De Zandschulp".
    Tras aplicar alias, best_price() debe encontrarlo.
    """
    feeds_raw = {
        "wplay": {
            _norm("Botic Van De Zandschulp"): {"odds": 2.62, "jugador": "Botic Van De Zandschulp"}
        }
    }
    feeds_raw["wplay"] = _aplicar_alias_wplay(feeds_raw["wplay"])
    feeds = {"wplay": feeds_raw["wplay"]}

    result = best_price("Van De Zandschulp", feeds)
    assert result is not None, "best_price debe encontrar Wplay con alias apellido"
    assert abs(result["cuota"] - 2.62) < 0.001
    assert result["casa"] == "wplay"


def test_alias_wplay_no_sobreescribe_clave_existente():
    """
    D128-01: si el apellido ya existe como clave exacta (ej. player sin nombre de pila),
    el alias NO debe sobreescribir la entrada existente.
    """
    entry_exacto = {"odds": 1.80, "jugador": "Djokovic"}
    entry_completo = {"odds": 1.75, "jugador": "Novak Djokovic"}

    feeds_wplay = {
        _norm("Djokovic"): entry_exacto,
        _norm("Novak Djokovic"): entry_completo,
    }
    result = _aplicar_alias_wplay(feeds_wplay)

    # La clave "djokovic" debe seguir apuntando al entry_exacto, no al alias de "novak djokovic"
    assert result[_norm("Djokovic")] is entry_exacto


def test_alias_wplay_multiples_jugadores():
    """
    D128-01: varios jugadores con nombres completos → todos generan alias por apellido.
    """
    feeds_wplay = {
        _norm("Botic Van De Zandschulp"): {"odds": 2.62, "jugador": "Botic Van De Zandschulp"},
        _norm("Oleksandra Oliynykova"): {"odds": 1.08, "jugador": "Oleksandra Oliynykova"},
        _norm("Nastasja Mariana Schunk"): {"odds": 8.00, "jugador": "Nastasja Mariana Schunk"},
    }
    result = _aplicar_alias_wplay(feeds_wplay)
    feeds = {"wplay": result}

    # Cada pick del edge_report usa apellido simple → debe encontrarse
    assert best_price("Oliynykova", feeds) is not None
    assert best_price("Oliynykova", feeds)["cuota"] == pytest.approx(1.08, abs=0.001)
    vdz = best_price("Van De Zandschulp", feeds)
    assert vdz is not None and vdz["cuota"] == pytest.approx(2.62, abs=0.001)
    schunk = best_price("Mariana Schunk", feeds)
    assert schunk is not None  # alias: drop "nastasja" → "mariana schunk"


import pytest

"""REGLA-T53: invoca la función real _build_betplay_url() de favoritos_combo_builder.py.

Nodo-169: _build_betplay_url() usaba "/".join(f"{oid}|ML" for oid in ids) desde su
creación (Nodo-146) — nunca tuvo el formato REGLA-BAT-1 correcto. Mismo síntoma que
Nodo-162 (coupon Betplay se abre sin piernas cargadas) pero bug independiente en un
archivo que la auditoría de Nodo-162 no tocó.
"""
from favoritos_combo_builder import _build_betplay_url, BETPLAY_URL_BASE, BETPLAY_URL_TAIL


def test_169_01_no_ml_suffix():
    url = _build_betplay_url(["111", "222", "333"])
    assert "|ML" not in url


def test_169_02_comma_joined_ids():
    url = _build_betplay_url(["111", "222", "333"])
    assert "111,222,333" in url


def test_169_03_url_format_completo():
    url = _build_betplay_url(["4284658243", "4284034676"])
    assert url == f"{BETPLAY_URL_BASE}4284658243,4284034676{BETPLAY_URL_TAIL}"


def test_169_04_single_leg():
    url = _build_betplay_url(["999"])
    assert url == f"{BETPLAY_URL_BASE}999{BETPLAY_URL_TAIL}"
    assert "|ML" not in url


def test_169_05_replace_no_append():
    url = _build_betplay_url(["1", "2"])
    assert url.endswith("||replace")
    assert "||append" not in url

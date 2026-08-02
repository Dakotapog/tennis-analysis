"""
tests/test_nodo162_redirect_coupon_format.py — REGLA-T53 D162-01

docs/bp/index.html (GitHub Pages, sirve el link "ABRIR X" de Telegram para
todos los combos) tuvo un commit (4ae668d, 2026-07-28) que rompió el formato
del coupon Betplay: cambió de comma-joined (REGLA-BAT-1) a un sufijo |ML/
inválido que Betplay no puede parsear — coupon vacío, sin piernas cargadas.

Este test lee el archivo real desde disco (no hardcodea el JS) y confirma
que el formato vigente respeta REGLA-BAT-1: sin sufijo |ML, IDs pasados
comma-joined tal cual a la URL.
"""
from pathlib import Path

_REDIRECT_PAGE = Path(__file__).parent.parent.parent / "docs" / "bp" / "index.html"


def _read_redirect_js() -> str:
    assert _REDIRECT_PAGE.exists(), f"No se encontró {_REDIRECT_PAGE}"
    return _REDIRECT_PAGE.read_text(encoding="utf-8")


def test_162_01_no_usa_sufijo_ml_invalido():
    js = _read_redirect_js()
    assert "+ '|ML'" not in js, (
        "REGLA-BAT-1 prohibe el sufijo |ML/ en el coupon Betplay — "
        "regresión del bug 4ae668d (coupon vacío sin piernas)"
    )


def test_162_02_construye_url_con_ids_comma_joined():
    js = _read_redirect_js()
    assert "coupon=combination|' + ids + '||replace'" in js, (
        "La URL debe construirse con `ids` (comma-joined) directo, "
        "per REGLA-BAT-1: combination|ID1,ID2,ID3||replace"
    )


def test_162_03_usa_replace_no_append():
    js = _read_redirect_js()
    assert "||replace" in js
    assert "||append" not in js

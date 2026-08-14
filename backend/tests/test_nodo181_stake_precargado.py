"""
tests/test_nodo181_stake_precargado.py — REGLA-T53 D181-10

D181-10 (Nodo-181): el tercer campo del coupon Kambi (`combination|<ids>|<stake>|<accion>`)
es el slot de stake — se envió vacío desde el primer combo. `build_coupon_url()` y
`build_redirect_url()` en betplay_combo_builder.py permiten pre-cargar ese monto.

Estos tests importan e invocan las funciones reales (REGLA-T53) — nunca reconstruyen
el string esperado con la misma fórmula que la función bajo prueba. Para el HTML
(docs/bp/index.html), se sigue el patrón de test_nodo162_redirect_coupon_format.py:
lectura del archivo real desde disco, aserciones de regresión estática.
"""
from pathlib import Path
from urllib.parse import quote

from betplay_combo_builder import build_coupon_url, build_redirect_url

_REDIRECT_PAGE = Path(__file__).parent.parent.parent / "docs" / "bp" / "index.html"


def _read_redirect_js() -> str:
    assert _REDIRECT_PAGE.exists(), f"No se encontró {_REDIRECT_PAGE}"
    return _REDIRECT_PAGE.read_text(encoding="utf-8")


# ── build_coupon_url ──────────────────────────────────────────────────────

def test_181_10_01_sin_stake_termina_en_doble_pipe_replace():
    """Retrocompatibilidad byte-idéntica REGLA-BAT-1: sin stake, sufijo exacto ||replace."""
    url = build_coupon_url([111, 222, 333], stake=None)
    assert url.endswith("||replace"), (
        f"Sin stake el coupon debe terminar exactamente en '||replace' (REGLA-BAT-1) — "
        f"obtuvo: {url!r}"
    )


def test_181_10_02_con_stake_contiene_campo_stake_entre_pipes():
    url = build_coupon_url([111, 222, 333], stake=10000)
    assert "|10000|replace" in url


def test_181_10_03_ids_enteros_no_lanzan_typeerror():
    # D171-02: Kambi devuelve ints — no debe reventar con TypeError.
    url = build_coupon_url([111, 222, 333], stake=None)
    assert "111,222,333" in url


def test_181_10_04_ids_separados_por_comas_sin_sufijo_ml():
    url = build_coupon_url([111, 222, 333], stake=5000)
    assert "111,222,333" in url
    assert "|ML/" not in url
    assert "|ML" not in url


def test_181_10_05_stake_float_produce_entero_sin_punto_decimal():
    url = build_coupon_url([111, 222, 333], stake=10000.0)
    assert "|10000|replace" in url
    assert "10000.0" not in url


# ── build_redirect_url ────────────────────────────────────────────────────

def test_181_10_06_sin_stake_no_contiene_query_param_stake():
    url = build_redirect_url([111, 222, 333], stake=None)
    assert "stake=" not in url


def test_181_10_07_con_stake_contiene_query_param_stake():
    url = build_redirect_url([111, 222, 333], stake=10000)
    assert "&stake=10000" in url


def test_181_10_08_con_label_contiene_label_url_encodeado_y_los_ids():
    label = "Combo A B"
    expected_label_encoded = quote(label)  # utilidad estándar, no la lógica bajo prueba
    url = build_redirect_url([111, 222, 333], stake=None, label=label)
    assert expected_label_encoded in url
    assert "111,222,333" in url


# ── docs/bp/index.html (regresión estática) ───────────────────────────────

def test_181_10_09_js_lee_query_param_stake():
    js = _read_redirect_js()
    assert "params.get('stake')" in js


def test_181_10_10_js_valida_stake_con_regex_solo_digitos_antes_de_inyectar():
    js = _read_redirect_js()
    assert r"/^\d+$/" in js, (
        "El JS debe validar el stake con una regex de solo dígitos antes de "
        "inyectarlo en la URL — guard contra inyección en el query param"
    )


def test_181_10_11_js_no_contiene_sufijo_ml_invalido():
    # Mismo check que test_nodo162_redirect_coupon_format.py::test_162_01 —
    # busca el patrón de CÓDIGO (concatenación JS), no la subcadena libre
    # '|ML/' (que sí aparece dentro de comentarios de documentación del fix).
    js = _read_redirect_js()
    assert "+ '|ML'" not in js, (
        "Guard anti-regresión de Nodo-162 — el sufijo |ML/ rompió todos los "
        "combos por 5 días (commit 4ae668d)"
    )

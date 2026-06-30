"""
tests/test_b05_tier_display.py — B-05: _print_resumen tier label

Verifica que _print_resumen NO muestra "clay (Roland Garros)" cuando
se pasa torneo_tipo='challenger', sino "clay / Challenger".

El bug original: default torneo_tipo='Roland Garros' (string literal)
producía la cadena "clay (Roland Garros)" en lugar del tier real.
Fix: default='grand_slam' + _TIER_DISPLAY dict.
"""
import io
import sys
import pytest

from trader_ev_tenis import _print_resumen, _TIER_DISPLAY


def _capturar_resumen(**kwargs) -> str:
    """Llama _print_resumen y captura su stdout."""
    buf = io.StringIO()
    defaults = dict(
        bankroll=100000,
        ind=1000,
        combos=2000,
        sistema=500,
        cobertura=500,
        senales=[],
        reporte={
            'metadata': {
                'timestamp': '2026-06-28',
                'bankroll': 100000,
                'n_apostar': 2,
                'n_combos': 1,
                'n_watchlist': 0,
                'calibracion_n': 5,
                'p_prior_efectivo': 0.60,
                'fuente': 'h2h_results_enhanced_test.json',
                'n_procesados': 10,
                'superficie': 'clay',
                'torneo_tipo': 'challenger',
            },
            'apostar': [],
            'combos': [],
            'watchlist': [],
            'sin_edge': [],
            'kelly_growth_rate': 0.01,
            'var_pct': 0.04,
            'var_usd': 4000,
        },
        pool_size=2,
    )
    defaults.update(kwargs)
    old = sys.stdout
    sys.stdout = buf
    try:
        _print_resumen(**defaults)
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestTierDisplay:

    def test_b05_challenger_muestra_label_correcto(self):
        """
        B-05: con torneo_tipo='challenger', el output contiene
        'clay / Challenger', NO 'clay (Roland Garros)'.
        """
        out = _capturar_resumen(superficie='clay', torneo_tipo='challenger')
        assert 'clay / Challenger' in out, (
            f"Label incorrecto. Buscaba 'clay / Challenger', output fue:\n{out[:500]}"
        )
        assert 'Roland Garros' not in out, (
            f"String legacy 'Roland Garros' no debe aparecer. Output:\n{out[:500]}"
        )

    def test_b05_itf_muestra_label_correcto(self):
        """B-05: torneo_tipo='itf' → 'hard / ITF'."""
        out = _capturar_resumen(superficie='hard', torneo_tipo='itf')
        assert 'hard / ITF' in out, f"Esperaba 'hard / ITF' en output:\n{out[:500]}"

    def test_b05_grand_slam_muestra_label_correcto(self):
        """B-05: torneo_tipo='grand_slam' (default) → 'clay / Grand Slam'."""
        out = _capturar_resumen(superficie='clay', torneo_tipo='grand_slam')
        assert 'clay / Grand Slam' in out, f"Esperaba 'clay / Grand Slam':\n{out[:500]}"

    def test_b05_tier_display_dict_completo(self):
        """B-05: _TIER_DISPLAY cubre los 5 tiers del pipeline."""
        expected = {'grand_slam', 'atp1000', 'atp500', 'challenger', 'itf'}
        assert set(_TIER_DISPLAY.keys()) == expected

    def test_b05_default_no_es_roland_garros(self):
        """
        B-05: el default de torneo_tipo en _print_resumen es 'grand_slam',
        no el string literal 'Roland Garros'.
        """
        import inspect
        sig = inspect.signature(_print_resumen)
        default = sig.parameters['torneo_tipo'].default
        assert default == 'grand_slam', (
            f"Default incorrecto: {default!r}. Debe ser 'grand_slam', no 'Roland Garros'."
        )

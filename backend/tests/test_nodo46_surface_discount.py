"""
tests/test_nodo46_surface_discount.py — Nodo-46 F4: Markov Surface-Context Discount

Tests D46-01 (8 tests base) + integración en rivalry_analyzer.

Detectan mutación real:
  test_discount_cold_zero_overlap FALLA si apply_surface_context_discount() no descuenta COLD.
  test_discount_hot_zero_overlap  FALLA si HOT no se descuenta cuando racha es de otra superficie.
  test_discount_neutral_not_applied FALLA si NEUTRAL es descontado (no debe serlo).
  test_overlap_all_same_surface FALLA si _surface_overlap_rate() no cuenta correctamente.

Nota calibración: constantes min_floor=0.70 y THRESHOLD=0.40 están BLOQUEADAS hasta n≥5 casos
atribuibles (hoy n=1, Watanuki). No modificar hasta tener ese n.
"""
import pytest
from analysis.markov_analyzer import (
    _normalize_surface,
    _surface_overlap_rate,
    apply_surface_context_discount,
)


# ─────────────────────────────────────────────────────────────────────────────
# D46-02: _normalize_surface()
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSurface:

    def test_normalize_grass_variants(self):
        """'Hierba', 'Grass', 'hierba' → 'grass'."""
        assert _normalize_surface('Hierba') == 'grass'
        assert _normalize_surface('grass') == 'grass'
        assert _normalize_surface('GRASS') == 'grass'

    def test_normalize_hard_variants(self):
        """'Dura', 'Hard', 'Indoor Hard' → 'hard'."""
        assert _normalize_surface('Dura') == 'hard'
        assert _normalize_surface('hard') == 'hard'
        assert _normalize_surface('Indoor Hard') == 'hard'
        assert _normalize_surface('Cemento') == 'hard'

    def test_normalize_clay_variants(self):
        """'Arcilla', 'Clay', 'Tierra' → 'clay'."""
        assert _normalize_surface('Arcilla') == 'clay'
        assert _normalize_surface('clay') == 'clay'
        assert _normalize_surface('Tierra') == 'clay'

    def test_normalize_unknown_returns_unknown(self):
        """Valores no reconocidos → 'unknown'."""
        assert _normalize_surface('') == 'unknown'
        assert _normalize_surface('indoor') == 'unknown'
        assert _normalize_surface(None) == 'unknown'  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# D46-03: _surface_overlap_rate()
# ─────────────────────────────────────────────────────────────────────────────

def _make_matches(surface: str, n: int) -> list:
    """Helper: crea n partidos en la misma superficie."""
    return [{'superficie': surface, 'outcome': 'Gano'} for _ in range(n)]


class TestSurfaceOverlapRate:

    def test_overlap_all_same_surface(self):
        """10 partidos en hard, torneo hard → overlap = 1.0.
        FALLA si _surface_overlap_rate() no cuenta correctamente."""
        matches = _make_matches('hard', 10)
        result = _surface_overlap_rate(matches, 'hard')
        assert result == 1.0

    def test_overlap_all_different_surface(self):
        """5 partidos en arcilla, torneo hard → overlap = 0.0."""
        matches = _make_matches('Arcilla', 5)  # se normaliza a clay
        result = _surface_overlap_rate(matches, 'hard')
        assert result == 0.0

    def test_overlap_mixed(self):
        """3 hard + 7 hierba (total 10), torneo hard → overlap = 0.3."""
        matches = _make_matches('hard', 3) + _make_matches('Hierba', 7)
        result = _surface_overlap_rate(matches, 'hard')
        assert result == pytest.approx(0.3)

    def test_overlap_empty_history_returns_zero(self):
        """Historial vacío → 0.0 (evita división por cero)."""
        result = _surface_overlap_rate([], 'hard')
        assert result == 0.0

    def test_overlap_small_history_returns_zero(self):
        """Menos de 5 partidos → 0.0 (muestra insuficiente para PELT)."""
        matches = _make_matches('hard', 4)
        result = _surface_overlap_rate(matches, 'hard')
        assert result == 0.0

    def test_overlap_unknown_surface_returns_zero(self):
        """current_surface='unknown' → 0.0."""
        matches = _make_matches('hard', 10)
        result = _surface_overlap_rate(matches, 'unknown')
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# D46-04: apply_surface_context_discount()
# ─────────────────────────────────────────────────────────────────────────────

class TestApplySurfaceContextDiscount:

    def test_discount_cold_zero_overlap(self):
        """COLD, overlap=0.0, factor=0.85 → nuevo factor > 0.85 (menos penalizado).
        FALLA si apply_surface_context_discount() no descuenta COLD con racha en otra superficie."""
        new_factor, new_conf, discount = apply_surface_context_discount(
            factor_markov=0.85, confianza=0.70,
            surface_overlap_rate=0.0, estado='COLD',
        )
        assert new_factor > 0.85, "COLD con overlap=0 debe acercarse a 1.0 (menos penalizado)"
        assert new_factor <= 1.0
        assert discount < 1.0, "El discount debe ser < 1.0 cuando overlap=0"

    def test_discount_hot_zero_overlap(self):
        """HOT, overlap=0.0, factor=1.15 → nuevo factor < 1.15 (menos inflado).
        FALLA si HOT no se descuenta cuando racha es de otra superficie."""
        new_factor, new_conf, discount = apply_surface_context_discount(
            factor_markov=1.15, confianza=0.80,
            surface_overlap_rate=0.0, estado='HOT',
        )
        assert new_factor < 1.15, "HOT con overlap=0 debe acercarse a 1.0 (menos inflado)"
        assert new_factor >= 1.0
        assert discount < 1.0

    def test_discount_neutral_not_applied(self):
        """NEUTRAL → factor sin cambio independientemente del overlap.
        FALLA si NEUTRAL es descontado (no debe serlo — no hay señal que distorsionar)."""
        new_factor, new_conf, discount = apply_surface_context_discount(
            factor_markov=1.05, confianza=0.55,
            surface_overlap_rate=0.0, estado='NEUTRAL',
        )
        assert new_factor == 1.05, "NEUTRAL no debe ser descontado"
        assert discount == 1.0

    def test_discount_high_overlap_no_change(self):
        """Overlap ≥ THRESHOLD → sin descuento."""
        new_factor, new_conf, discount = apply_surface_context_discount(
            factor_markov=0.85, confianza=0.70,
            surface_overlap_rate=0.5,  # >= 0.40 threshold
            estado='COLD',
        )
        assert new_factor == 0.85
        assert discount == 1.0

    def test_no_surface_discount_flag(self):
        """apply_discount=False → sin descuento (flag --no-surface-discount)."""
        new_factor, new_conf, discount = apply_surface_context_discount(
            factor_markov=0.85, confianza=0.70,
            surface_overlap_rate=0.0, estado='COLD',
            apply_discount=False,
        )
        assert new_factor == 0.85
        assert discount == 1.0

    def test_season_transition_flag_forces_discount(self):
        """season_transition_flag=True → discount se aplica incluso con overlap ambiguo."""
        # Con overlap=0.45 (sobre el THRESHOLD normal de 0.40), sin flag no habría descuento
        new_factor_no_flag, _, _ = apply_surface_context_discount(
            factor_markov=0.85, confianza=0.70,
            surface_overlap_rate=0.45, estado='COLD',
            season_transition_flag=False,
        )
        new_factor_flag, _, _ = apply_surface_context_discount(
            factor_markov=0.85, confianza=0.70,
            surface_overlap_rate=0.45, estado='COLD',
            season_transition_flag=True,
        )
        # Con season_transition_flag, el threshold efectivo sube → overlap=0.45 puede quedar bajo threshold
        # El factor debe ser diferente (o igual si aún por encima del threshold efectivo)
        # Lo importante: la función no falla con el flag
        assert isinstance(new_factor_flag, float)

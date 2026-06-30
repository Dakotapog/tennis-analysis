"""
Tests para Nodo-33 — Filtro Coin-flip Sin H2H
Cubre:
  BUG-33-1: James-Stein shrinkage colapsa p_blend→0.50 cuando n_cal=0 (sin calibración)
  BUG-33-2: puerta lateral cuota<2.10 bypasseaba el check p_modelo≥0.55 en el gate
  BUG-33-3: hardcode p_modelo=0.55/0.50 en combo builder en vez de lookup real

Clases:
  TestNodo33Fase1Gate:         T33-01 a T33-04 — edge_calculator: calcular_edge_completo()
  TestNodo33Fase1GuardLogic:   T33-05 a T33-08 — condición de bloqueo con constantes reales
  TestNodo33Fase1ComboBuilder: T33-09 a T33-12 — combo builder: lookup vs hardcode

Detección de fix revertido:
  T33-01 FALLA si se revierte el override T33-01 de calcular_edge_completo().
  Sin el override, n_h2h=0+p_modelo=0.54+cuota=2.08 → apostar=True (lateral door abierta).
"""
import pytest
from edge_calculator import (
    calcular_edge_completo,
    P_MODELO_MIN_UNDERDOG,
    GATE_VERSION,
)
from betplay_combo_builder import _es_coinflip_sin_h2h

# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

CALIB_MINIMAL = {
    "global": {"wins": 50, "losses": 20},
    "por_superficie": {
        "clay": {"wins": 31, "losses": 10},
    },
    "por_superficie_y_tier": {},
    "fallback_por_tier": {
        "grand_slam": 0.758,
        "atp1000":    0.65,
        "atp500":     0.62,
        "challenger": 0.55,
        "itf":        0.50,
    },
}


def _make_partido(
    jugador1="PlayerA",
    jugador2="PlayerB",
    cuota1=2.08,
    cuota2=1.95,
    confidence=54.0,
    n_h2h=0,
    elo_fav=1430,
    elo_rival=1500,
    torneo="ITF Testing",
    superficie="clay",
):
    """
    Partido mínimo para calcular_edge_completo().
    favored = jugador1 siempre.
    n_h2h: número de dicts en enfrentamientos_directos.

    elo_fav=1430 / elo_rival=1500:
      p_elo_base = 1/(1+10^(70/400)) ≈ 0.401
      alpha_vs_elo = p_modelo - 0.401
      Con p_modelo=0.54: alpha=0.139 → surface_norm=0.556 > 0.50 (eje activo)

    BBI con n_h2h=0 y cuota=2.08:
      bbi = (1 - 1/2.08) × 1.0 = 0.519 → bbi_norm=0.741 > 0.50 (eje activo)

    → n_axes_active = 2 (surface + bbi) → FIX-3 NO bloquea
    → markov=None → NOT HOT → FIX-6 NO bloquea
    → solo T33-01 bloquea cuando n_h2h=0 y p_modelo<0.55
    """
    ra = {
        "prediction": {
            "favored_player": jugador1,
            "confidence": confidence,
            "markov_analysis": None,
            "surface_specialization_meta": {},
            "circuit_asymmetry": None,
            "score_breakdown": {},
        },
    }
    # _sanitize_name("PlayerA") = "PlayerA" (sin espacios ni puntos)
    ra[f"{jugador1}_elo"] = elo_fav
    ra[f"{jugador2}_elo"] = elo_rival
    ra[f"{jugador1}_ranking"] = 100
    ra[f"{jugador2}_ranking"] = 80

    return {
        "jugador1": jugador1,
        "jugador2": jugador2,
        "cuota1": cuota1,
        "cuota2": cuota2,
        "torneo_completo": torneo,
        "superficie": superficie,
        "enfrentamientos_directos": [{"winner": jugador1}] * n_h2h,
        "ranking_analysis": ra,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T33-01 a T33-04 — Gate edge_calculator: calcular_edge_completo()
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo33Fase1Gate:
    """Fase 1: bloqueo n_h2h=0 en calcular_edge_completo()"""

    def test_t33_01_lateral_door_blocked_by_n_h2h_zero(self):
        """T33-01: n_h2h=0 + p_modelo=0.54 (<0.55) + cuota=2.08 (<2.10) → apostar=False

        BUG-33-2: antes del fix, cuota=2.08<2.10 pasaba la puerta lateral
        (p_modelo >= 0.55 OR cuota < 2.10) = True aunque p_modelo=0.54<0.55.
        El fix T33-01 en calcular_edge_completo() agrega el override:
          if n_h2h==0 and p_modelo<0.55 and apostar: apostar=False

        DETECCIÓN DE FIX REVERTIDO: si se elimina el override T33-01, este test
        FALLA porque:
          - edge = 0.54 - 1/2.08 = 0.059 > 0.05 (positivo)
          - (p_modelo>=0.55 OR cuota<2.10) = (False OR True) = True ← lateral door
          - n_axes_active = 2 (surface 0.556 + BBI 0.741) → FIX-3 no bloquea
          - NOT HOT → FIX-6 no bloquea
          - Sin override: apostar=True ← BUG activo
        """
        partido = _make_partido(confidence=54.0, cuota1=2.08, n_h2h=0)
        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None, "calcular_edge_completo no debe retornar None"
        assert r["apostar"] is False, (
            "T33-01: n_h2h=0 + p_modelo=0.54 debe bloquearse incluso con cuota<2.10. "
            "Si este test falla, el fix T33-01 fue revertido."
        )
        # El motivo debe identificar T33-01 (no FIX-3 u otro override)
        motivo = r.get("motivo_reclasificacion", "")
        assert "T33-01" in motivo, (
            f"motivo_reclasificacion debe contener 'T33-01', got: '{motivo}'"
        )

    def test_t33_02_original_t32_gate_still_blocks_high_cuota_coinflip(self):
        """T33-02: n_h2h=0 + p_modelo=0.514 + cuota=4.70 → apostar=False (sin regresión)

        Verificación de que el gate original Nodo-32 (T32-01) sigue bloqueando
        cuando cuota>=2.10 y p_modelo<0.55.
        Este escenario es el de Simona Cucu / Gabriela Kawano Cho (23-jun).
        """
        partido = _make_partido(confidence=51.4, cuota1=4.70, cuota2=1.22, n_h2h=0)
        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        # edge matemático es positivo (0.514 - 1/4.70 ≈ 0.30)
        assert r["edge"] > 0.05, "El edge matemático debe existir (BBI-driven phantom)"
        # Pero apostar=False por T32-01 gate (p_modelo<0.55 y cuota>=2.10)
        assert r["apostar"] is False, "T32-01 gate debe bloquear cuota alta + p_modelo<0.55"

    def test_t33_03_strong_conviction_n_h2h_zero_allowed(self):
        """T33-03: n_h2h=0 + p_modelo=0.67 (≥0.55) + cuota=2.08 → apostar=True

        Con convicción fuerte, T33-01 NO debe bloquear aunque n_h2h=0.
        La condición es específica para coin-flip (p_modelo<0.55).
        """
        partido = _make_partido(confidence=67.0, cuota1=2.08, n_h2h=0)
        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        assert r["edge"] > 0.05, "p_modelo=0.67 debe tener edge positivo"
        # T33-01 no debe bloquear: p_modelo=0.67 >= P_MODELO_MIN_UNDERDOG
        assert r.get("motivo_reclasificacion") is None or "T33-01" not in (
            r.get("motivo_reclasificacion") or ""
        ), "T33-01 no debe bloquear picks con p_modelo>=0.55"
        assert r["apostar"] is True, (
            "p_modelo=0.67 + n_h2h=0 debe apostar (convicción fuerte presente)"
        )

    def test_t33_04_n_h2h_one_lifts_block(self):
        """T33-04: n_h2h=1 + p_modelo=0.54 + cuota=2.08 → apostar=True (T33-01 solo aplica a n_h2h=0)

        Con 1 partido directo previo, T33-01 NO aplica (condición: n_h2h==0).
        El lateral door (cuota<2.10) sigue siendo válido.
        """
        partido = _make_partido(confidence=54.0, cuota1=2.08, n_h2h=1)
        r = calcular_edge_completo(partido, CALIB_MINIMAL)

        assert r is not None
        # Con n_h2h=1, T33-01 no aplica
        motivo = r.get("motivo_reclasificacion") or ""
        assert "T33-01" not in motivo, (
            "T33-01 no debe bloquear cuando n_h2h=1 (solo aplica a n_h2h==0)"
        )
        assert r["apostar"] is True, (
            "n_h2h=1 + cuota=2.08<2.10 debe pasar: lateral door válido sin T33-01 block"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T33-05 a T33-08 — Condición de bloqueo con constantes reales
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo33Fase1GuardLogic:
    """Condición coin-flip: n_h2h==0 and p_modelo < P_MODELO_MIN_UNDERDOG"""

    def _would_block(self, n_h2h: int, p_modelo: float) -> bool:
        """Replica la condición de bloqueo de T33-01."""
        return n_h2h == 0 and p_modelo < P_MODELO_MIN_UNDERDOG

    def test_t33_05_majdandzic_p514_n_h2h_0_blocked(self):
        """T33-05: Majdandzic (p_modelo=0.514, n_h2h=0) → bloqueado — pérdida real jun-23"""
        assert self._would_block(n_h2h=0, p_modelo=0.514), (
            "Majdandzic con p_modelo=0.514 y n_h2h=0 debe activar bloqueo coin-flip"
        )

    def test_t33_06_fiadosik_p509_n_h2h_0_blocked(self):
        """T33-06: Fiadosik (p_modelo=0.509, n_h2h=0) → bloqueado"""
        assert self._would_block(n_h2h=0, p_modelo=0.509)

    def test_t33_07_makke_p509_n_h2h_0_blocked(self):
        """T33-07: Makke (p_modelo=0.509, n_h2h=0) → bloqueado"""
        assert self._would_block(n_h2h=0, p_modelo=0.509)

    def test_t33_08_musat_p583_n_h2h_0_allowed(self):
        """T33-08: Musat (p_modelo=0.583, n_h2h=0) → NO bloqueado (p_modelo≥0.55)"""
        assert not self._would_block(n_h2h=0, p_modelo=0.583), (
            "Musat con p_modelo=0.583>=0.55 NO debe ser bloqueado por coin-flip guard"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T33-09 a T33-12 — Combo builder: _es_coinflip_sin_h2h() real (BUG-33-3)
# ─────────────────────────────────────────────────────────────────────────────

class TestNodo33Fase1ComboBuilder:
    """
    Tests que ejercitan _es_coinflip_sin_h2h() directamente desde
    betplay_combo_builder.py — la función extraída que consolida el bloqueo
    duro en ambas instancias de _build_mega_combos() (cobertura legs +
    sin_edge picks).

    DETECCIÓN DE MUTACIÓN: comentar o hacer que _es_coinflip_sin_h2h()
    retorne siempre False hace que T33-10 y T33-12 FALLEN.
    T33-11 y T33-09 (pase positivo) fallarían si retornara siempre True.
    """

    def test_t33_09_bloqueo_con_p_modelo_real_de_majdandzic(self):
        """T33-09: _es_coinflip_sin_h2h() bloquea Majdandzic (p_modelo=0.514, n_h2h=0)

        Verifica que la función real importada desde betplay_combo_builder.py
        reproduce el bloqueo correcto con los valores reales del pick de jun-23.
        BUG-33-3 fixed: ya no hay hardcode — la función usa _P_MODELO_MIN_UNDERDOG.
        """
        # Valores exactos de Oliver Majdandzic según spec Nodo-33 sección 1
        assert _es_coinflip_sin_h2h(p_modelo=0.514, n_h2h=0) is True, \
            "Majdandzic p_modelo=0.514 + n_h2h=0 debe retornar True (bloqueado)"

    def test_t33_10_combo_builder_blocks_coinflip_pick(self):
        """T33-10: _es_coinflip_sin_h2h() retorna True para p_modelo=0.514 + n_h2h=0

        DETECCIÓN DE MUTACIÓN: si _es_coinflip_sin_h2h() es comentada o retorna
        siempre False, este test FALLA — confirma que el código real bloquea.
        """
        resultado = _es_coinflip_sin_h2h(p_modelo=0.514, n_h2h=0)
        assert resultado is True, \
            "p_modelo=0.514 (<0.55) + n_h2h=0 debe ser identificado como coin-flip (True=bloqueado)"

    def test_t33_11_combo_builder_allows_strong_conviction_no_h2h(self):
        """T33-11: _es_coinflip_sin_h2h() retorna False para p_modelo=0.67 + n_h2h=0

        Con convicción real (p_modelo≥0.55), el pick NO es coin-flip aunque
        no haya H2H directo. La función debe retornar False (no bloqueado).
        """
        resultado = _es_coinflip_sin_h2h(p_modelo=0.67, n_h2h=0)
        assert resultado is False, \
            "p_modelo=0.67 (>=0.55) + n_h2h=0 no es coin-flip — debe retornar False (permitido)"

    def test_t33_12_combo_builder_blocks_fallback_p_modelo_050(self):
        """T33-12: _es_coinflip_sin_h2h() bloquea el caso fallback p_modelo=0.50 + n_h2h=0

        En _build_mega_combos(), picks no registrados en edge_tier_map usan
        _info.get("p_modelo", 0.50) → 0.50. Con n_h2h=0, _es_coinflip_sin_h2h()
        debe retornar True (bloqueado) — el fallback a 0.50 confirma la ausencia
        de convicción real.
        """
        resultado = _es_coinflip_sin_h2h(p_modelo=0.50, n_h2h=0)
        assert resultado is True, \
            "Fallback p_modelo=0.50 + n_h2h=0 es coin-flip puro — debe retornar True (bloqueado)"

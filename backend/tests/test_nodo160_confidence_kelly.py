"""
tests/test_nodo160_confidence_kelly.py — REGLA-T53 Nodo-160 D160-04

D160-04: core/confidence_kelly.py::confidence_scaled_stake() generaliza
rival_value_betslip.py::micro_kelly() (H88-01). Verifica: (1) la función
genérica pre-graduación, (2) que micro_kelly() delega y produce el MISMO
resultado numérico que antes del refactor (regression-safe), (3) el
comportamiento post-graduación (cap más alto).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.confidence_kelly import confidence_scaled_stake
import rival_value_betslip as rvb


def test_160_30_pregraduacion_cap_bajo():
    """n_obs < n_stop → usa cap_pct_pregrad, mismo patrón que micro_kelly H88-01."""
    stake = confidence_scaled_stake(
        edge=0.15, cuota=3.5, bankroll=125000,
        n_obs=3, n_stop=30, k_prior=50,
        cap_pct_graduado=0.02, cap_pct_pregrad=0.005,
    )
    # kelly_raw=0.15/2.5=0.06, shrink=3/53=0.0566, kelly_shrunk=0.0034 < cap 0.005
    assert stake == max(2000.0, round(0.06 * (3 / 53) * 125000 / 500) * 500)


def test_160_31_edge_insuficiente_retorna_cero():
    assert confidence_scaled_stake(0.03, 3.5, 125000, n_obs=3, n_stop=30) == 0.0


def test_160_32_cuota_invalida_retorna_cero():
    assert confidence_scaled_stake(0.15, 1.0, 125000, n_obs=3, n_stop=30) == 0.0


def test_160_33_postgraduacion_cap_sube():
    """n_obs >= n_stop → usa cap_pct_graduado (más alto) en vez de pregrad."""
    stake_pre = confidence_scaled_stake(
        edge=0.30, cuota=2.0, bankroll=125000,
        n_obs=29, n_stop=30, k_prior=5,
        cap_pct_graduado=0.02, cap_pct_pregrad=0.005,
    )
    stake_post = confidence_scaled_stake(
        edge=0.30, cuota=2.0, bankroll=125000,
        n_obs=30, n_stop=30, k_prior=5,
        cap_pct_graduado=0.02, cap_pct_pregrad=0.005,
    )
    assert stake_post >= stake_pre


def test_160_34_micro_kelly_delega_sin_cambiar_resultado():
    """micro_kelly() (H88-01) debe seguir devolviendo el mismo número que su
    fórmula original (kelly_raw*shrink capado a 0.5%, redondeado a 500, piso 2000)
    tras el refactor a confidence_scaled_stake — regression-safe."""
    edge, cuota, bankroll = 0.15, 3.5, 125000
    kelly_raw = edge / (cuota - 1.0)
    kelly_shrunk = kelly_raw * rvb.H88_SHRINK
    stake_pct = min(kelly_shrunk, rvb.H88_MAX_PCT)
    expected = max(2000.0, round(stake_pct * bankroll / 500) * 500)

    assert rvb.micro_kelly(edge, cuota, bankroll) == expected

"""
core/confidence_kelly.py — D160-04 (Nodo-160)

Generaliza rival_value_betslip.py::micro_kelly() (H88-01) a una función
parametrizada reusable por cualquier hipótesis experimental (H150-*, H151-01,
H160-XX) sin duplicar la fórmula. Pre-graduación reproduce micro_kelly()
bit-a-bit (regression-safe); post-graduación relaja shrinkage y cap como
ya lo hace GCS (H60-01) implícitamente vía sus propias reglas.

NO reemplaza Kelly-KL de tier/superficie del motor principal
(edge_calculator.py) — es solo para señales nuevas con sizing ad-hoc/fijo.
"""


def confidence_scaled_stake(
    edge: float,
    cuota: float,
    bankroll: float,
    n_obs: int,
    n_stop: int,
    k_prior: int = 20,
    cap_pct_graduado: float = 0.02,
    cap_pct_pregrad: float = 0.005,
    min_edge: float = 0.05,
) -> float:
    """
    Stake escalado por confianza (n_obs vs n_stop) — Kelly con shrinkage.

    Pre-graduación (n_obs < n_stop): shrinkage agresivo n_obs/(n_obs+k_prior),
    cap bajo (cap_pct_pregrad). Mismo patrón que micro_kelly() de H88-01.
    Post-graduación (n_obs >= n_stop): mismo shrinkage (converge a ~1 con
    n grande), cap sube a cap_pct_graduado.

    Redondeado a 500 COP, piso 2000 COP. Retorna 0.0 si cuota<=1.0 o
    edge<min_edge (sin señal suficiente para apostar).
    """
    if cuota <= 1.0 or edge < min_edge:
        return 0.0

    kelly_raw = edge / (cuota - 1.0)
    shrinkage = n_obs / (n_obs + k_prior)
    kelly_shrunk = kelly_raw * shrinkage

    cap_pct = cap_pct_graduado if n_obs >= n_stop else cap_pct_pregrad
    stake_pct = min(kelly_shrunk, cap_pct)
    stake_raw = stake_pct * bankroll

    return max(2000.0, round(stake_raw / 500) * 500)

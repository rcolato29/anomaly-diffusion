"""Extreme Value Theory thresholding (Siffer et al. SPOT/DSPOT, 2017).

Set the threshold for a target false-alarm rate q from the tail of the normal-data score
distribution. Peaks-over-threshold picks a high level u (a quantile of the normal scores),
fits a Generalized Pareto Distribution (GPD) to the excesses above u, then inverts the GPD
tail for the score z_q whose exceedance probability equals q.

    P(X > z) approx (N_u / n) (1 + xi (z - u) / beta)^(-1/xi)   =   q
    ->  z_q = u + (beta/xi) [ (q n / N_u)^(-xi) - 1 ]     (xi -> 0: z_q = u - beta ln(q n / N_u))

The GPD fit is from scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def fit_pot(scores, level: float = 0.98) -> dict:
    """Fit a GPD to the upper-tail excesses of scores above the level quantile."""
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    u = float(np.quantile(scores, level))
    excesses = scores[scores > u] - u
    if excesses.size < 10:
        raise ValueError(
            f"Only {excesses.size} excesses above the {level:.2f} quantile. "
            "Need more normal scores or a lower level for a stable GPD fit."
        )
    xi, _, beta = genpareto.fit(excesses, floc=0.0)  # (shape, loc=0, scale)
    return {"u": u, "xi": float(xi), "beta": float(beta), "n": n, "n_exc": int(excesses.size)}


def pot_threshold(fit: dict, q: float = 1e-3) -> float:
    """Score threshold whose normal-data exceedance probability is q (the FAR)."""
    u, xi, beta, n, n_exc = fit["u"], fit["xi"], fit["beta"], fit["n"], fit["n_exc"]
    ratio = q * n / n_exc
    if abs(xi) < 1e-6:
        return float(u - beta * np.log(ratio))
    return float(u + (beta / xi) * (ratio**-xi - 1.0))


def evt_threshold(scores, q: float = 1e-3, level: float = 0.98) -> float:
    """Convenience: fit POT on normal scores and return the threshold for FAR q."""
    return pot_threshold(fit_pot(scores, level=level), q=q)

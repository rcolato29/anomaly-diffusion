import math

import torch

from anomaly_diffusion.likelihood.ode_likelihood import bits_per_dim, log_likelihood
from anomaly_diffusion.sde.vpsde import VPSDE


def _zero_score(x, t):
    return torch.zeros_like(x)


def _analytic_zero_score_logp(sde: VPSDE, x0: torch.Tensor) -> torch.Tensor:
    """With score=0, the PF-ODE is the linear forward drift, giving a closed-form Gaussian
    log-density: log p_0(x0) = -1/2 m^2 ||x0||^2 - D/2 log(2pi) - 1/2 D B(1)."""
    b1 = sde.beta_min + 0.5 * (sde.beta_max - sde.beta_min)  # B(1) with t0=0
    m2 = math.exp(-b1)  # m = exp(-1/2 B(1)) -> m^2 = exp(-B(1))
    dim = x0[0].numel()
    sq = (x0.flatten(1) ** 2).sum(dim=1)
    return -0.5 * m2 * sq - 0.5 * dim * math.log(2 * math.pi) - 0.5 * dim * b1


def test_zero_score_likelihood_matches_analytic_gaussian():
    sde = VPSDE()
    torch.manual_seed(0)
    x0 = torch.randn(3, 1, 2, 2)
    logp = log_likelihood(_zero_score, sde, x0, n_steps=200, method="rk4", exact=True, t0=0.0)
    analytic = _analytic_zero_score_logp(sde, x0)
    assert torch.allclose(logp, analytic, atol=5e-2)


def test_hutchinson_matches_exact_for_isotropic_drift():
    # With score=0, the Jacobian is isotropic, so the Hutchinson estimate is
    # exact regardless of the noise draw -> the two paths must agree.
    sde = VPSDE()
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 2, 2)
    exact = log_likelihood(_zero_score, sde, x0, n_steps=100, method="rk4", exact=True, t0=0.0)
    hutch = log_likelihood(_zero_score, sde, x0, n_steps=100, method="rk4", t0=0.0)
    assert torch.allclose(exact, hutch, atol=1e-3)


def test_bits_per_dim_sign():
    x = torch.randn(4, 3, 8, 8)
    logp = torch.tensor([10.0, -10.0, 0.0, 5.0])
    bpd = bits_per_dim(logp, x)
    # higher likelihood (logp) -> lower bits/dim
    assert bpd[0] < bpd[2] < bpd[1]

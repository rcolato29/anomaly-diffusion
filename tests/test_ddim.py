import torch

from anomaly_diffusion.sampling.sampler import ddim_reconstruct
from anomaly_diffusion.scoring.reconstruction import reconstruct
from anomaly_diffusion.sde.vpsde import VPSDE


def test_ddim_reconstruct_shape_and_finite():
    sde = VPSDE()
    x_t = torch.randn(2, 3, 16, 16)
    out = ddim_reconstruct(lambda x, t: torch.zeros_like(x), sde, x_t, t_start=0.3, n_steps=5)
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()


def test_ddim_low_tstar_recovers_input():
    # The oracle score for a clean point x0 at low noise pushes reconstruction back to x0.
    # With a near-perfect denoiser (score = kernel score toward x0), DDIM from a small t*
    # should return close to x0.
    sde = VPSDE()
    x0 = torch.randn(1, 3, 8, 8)

    def oracle_score(x, t):
        mean, std = sde.marginal_prob(x0, t)
        std_b = std.reshape(-1, 1, 1, 1)
        return -(x - mean) / std_b**2  # kernel score toward x0

    x_t = sde.marginal_prob(x0, torch.tensor([0.05]))[0] + 0.0
    out = ddim_reconstruct(oracle_score, sde, x_t, t_start=0.05, n_steps=10)
    assert torch.allclose(out, x0, atol=0.1)


def test_reconstruct_solver_switch_runs():
    sde = VPSDE()
    x0 = torch.randn(1, 3, 16, 16)
    for solver in ("pf_ode", "ddim"):
        out = reconstruct(lambda x, t: torch.zeros_like(x), sde, x0, 0.2, n_steps=4, solver=solver)
        assert out.shape == x0.shape

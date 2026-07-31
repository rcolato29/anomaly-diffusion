import torch

from anomaly_diffusion.losses.dsm import dsm_loss
from anomaly_diffusion.sde.vpsde import VPSDE


def _const_score(value):
    def score_fn(x, t):
        return torch.full_like(x, value)

    return score_fn


def test_loss_is_finite_scalar():
    sde = VPSDE()
    x0 = torch.randn(4, 3, 8, 8)
    loss = dsm_loss(_const_score(0.0), sde, x0, weighting="std2")
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_std2_matches_manual_recompute():
    # reproduce the loss by replaying the same RNG stream
    sde = VPSDE()
    x0 = torch.randn(3, 3, 8, 8)
    score_fn = _const_score(0.25)

    torch.manual_seed(123)
    loss = dsm_loss(score_fn, sde, x0, weighting="std2")

    torch.manual_seed(123)
    t = sde.sample_t(x0.shape[0], x0.device)
    xt, z, std = sde.perturb(x0, t)
    std_b = std.reshape(-1, 1, 1, 1)
    residual = std_b * score_fn(xt, t) + z
    manual = residual.pow(2).mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_likelihood_weighting_runs():
    sde = VPSDE()
    x0 = torch.randn(2, 3, 8, 8)
    loss = dsm_loss(_const_score(0.0), sde, x0, weighting="likelihood")
    assert torch.isfinite(loss)

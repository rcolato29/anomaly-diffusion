import torch

from anomaly_diffusion.scoring.score_norm import score_norm
from anomaly_diffusion.sde.vpsde import VPSDE


def test_score_norm_shape_and_nonnegative():
    sde = VPSDE()
    x0 = torch.randn(4, 3, 16, 16)
    s = score_norm(lambda x, t: x, sde, x0, t_levels=[0.01, 0.05])
    assert s.shape == (4,)
    assert (s >= 0).all()


def test_score_norm_larger_for_larger_score():
    sde = VPSDE()
    x0 = torch.randn(2, 3, 8, 8)
    small = score_norm(lambda x, t: 0.1 * x, sde, x0)
    large = score_norm(lambda x, t: 10.0 * x, sde, x0)
    assert (large > small).all()

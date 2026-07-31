import torch

from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score, reconstruct
from anomaly_diffusion.sde.vpsde import VPSDE


def _zero_score(x, t):
    return torch.zeros_like(x)


def test_reconstruct_shape_and_finite():
    sde = VPSDE()
    x0 = torch.randn(2, 3, 16, 16)
    x_hat = reconstruct(_zero_score, sde, x0, t_star=0.2, n_steps=5, probability_flow=True)
    assert x_hat.shape == x0.shape
    assert torch.isfinite(x_hat).all()


def test_anomaly_map_shape_and_nonnegative():
    sde = VPSDE()
    x0 = torch.randn(2, 3, 16, 16)
    amap = anomaly_map(_zero_score, sde, x0, t_stars=[0.1, 0.2], n_steps=3, smooth_sigma=1.0)
    assert amap.shape == (2, 1, 16, 16)
    assert (amap >= 0).all()


def test_anomaly_map_no_smoothing_keeps_shape():
    sde = VPSDE()
    x0 = torch.randn(1, 3, 16, 16)
    amap = anomaly_map(_zero_score, sde, x0, t_stars=[0.15], n_steps=2, smooth_sigma=0.0)
    assert amap.shape == (1, 1, 16, 16)


def test_image_score_methods():
    amap = torch.rand(4, 1, 8, 8)
    for method in ("max", "mean", "topk"):
        s = image_score(amap, method=method)
        assert s.shape == (4,)
    # max >= mean elementwise
    assert (image_score(amap, "max") >= image_score(amap, "mean")).all()

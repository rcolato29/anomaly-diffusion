import numpy as np

from anomaly_diffusion.thresholding.evt import evt_threshold, fit_pot, pot_threshold


def test_threshold_monotonic_in_far():
    rng = np.random.default_rng(0)
    scores = rng.exponential(scale=1.0, size=5000)
    fit = fit_pot(scores, level=0.9)
    t_loose = pot_threshold(fit, q=1e-2)
    t_strict = pot_threshold(fit, q=1e-4)
    # a smaller false-alarm rate demands a higher threshold
    assert t_strict > t_loose > fit["u"]


def test_empirical_far_close_to_target():
    rng = np.random.default_rng(1)
    scores = rng.exponential(scale=2.0, size=20000)
    q = 1e-2
    thr = evt_threshold(scores, q=q, level=0.9)
    holdout = rng.exponential(scale=2.0, size=20000)
    empirical_far = (holdout > thr).mean()
    # within a factor of ~3 of the target. POT tail extrapolation, not exact.
    assert q / 3 < empirical_far < q * 3


def test_too_few_excesses_raises():
    import pytest

    with pytest.raises(ValueError):
        fit_pot(np.arange(5), level=0.98)

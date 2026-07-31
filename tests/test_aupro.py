import numpy as np

from anomaly_diffusion.eval.metrics import au_pro


def _make_case(perfect: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((3, 16, 16), dtype=bool)
    masks[:, 4:8, 4:8] = True
    if perfect:
        maps = np.where(masks, 1.0, 0.0) + rng.uniform(0, 0.01, masks.shape)
    else:
        maps = rng.uniform(0, 1, masks.shape)
    return maps, masks


def test_perfect_localization_high_aupro():
    maps, masks = _make_case(perfect=True)
    assert au_pro(maps, masks) > 0.95


def test_perfect_beats_random():
    perfect = au_pro(*_make_case(perfect=True))
    random = au_pro(*_make_case(perfect=False))
    assert perfect > random


def test_no_defects_is_nan():
    maps = np.random.rand(2, 8, 8)
    masks = np.zeros((2, 8, 8), dtype=bool)
    assert np.isnan(au_pro(maps, masks))

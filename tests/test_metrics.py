import math

from anomaly_diffusion.eval.metrics import image_auroc, pixel_auroc


def test_image_auroc_perfect_separation():
    assert image_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert image_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_pixel_auroc_perfect_separation():
    masks = [0, 0, 1, 1]
    maps = [0.1, 0.2, 0.8, 0.9]
    assert pixel_auroc(masks, maps) == 1.0


def test_pixel_auroc_single_class_is_nan():
    assert math.isnan(pixel_auroc([0, 0, 0], [0.1, 0.2, 0.3]))

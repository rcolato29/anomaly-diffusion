import numpy as np
from PIL import Image

from anomaly_diffusion.serving.demo import _overlay, make_predict_fn
from anomaly_diffusion.serving.inference import heatmap_to_b64


class _FakeDetector:
    def __init__(self, score=0.9, is_anomaly=True):
        self._hm = heatmap_to_b64(np.random.rand(32, 32).astype("float32"))
        self._res = {"score": score, "is_anomaly": is_anomaly, "heatmap_png_b64": self._hm}

    def score_image(self, image):
        return self._res


def test_overlay_returns_rgb_image():
    img = Image.fromarray((np.random.rand(48, 48, 3) * 255).astype("uint8"))
    hm = heatmap_to_b64(np.random.rand(32, 32).astype("float32"))
    out = _overlay(img, hm)
    assert out.mode == "RGB"
    assert out.size == (32, 32)  # resized to the heatmap resolution


def test_predict_fn_returns_overlay_and_verdict():
    predict = make_predict_fn(_FakeDetector(score=0.9, is_anomaly=True))
    img = Image.fromarray((np.random.rand(48, 48, 3) * 255).astype("uint8"))
    overlay, text = predict(img)
    assert isinstance(overlay, Image.Image)
    assert "0.9" in text and "ANOMALY" in text


def test_predict_fn_handles_no_image():
    predict = make_predict_fn(_FakeDetector())
    overlay, text = predict(None)
    assert overlay is None and "Upload" in text

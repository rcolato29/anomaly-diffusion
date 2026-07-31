import io

import numpy as np
import torch
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.serving.app import create_app
from anomaly_diffusion.serving.drift import DriftMonitor
from anomaly_diffusion.serving.export import export_score_net
from anomaly_diffusion.serving.inference import AnomalyDetector


class _FakeDetector:
    device = "cpu"
    n_steps = 5

    def __init__(self, score=0.5, is_anomaly=None):
        self._score, self._is_anomaly = score, is_anomaly

    def score_batch(self, images):
        return [
            {"score": self._score, "is_anomaly": self._is_anomaly, "heatmap_png_b64": "AA=="}
            for _ in images
        ]


def _png_bytes(size=16):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(size, size, 3) * 255).astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def _client(detector=None, drift=None):
    return TestClient(create_app(detector=detector or _FakeDetector(), drift=drift))


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready():
    assert _client().get("/ready").json()["ready"] is True


def test_predict_ok():
    r = _client().post("/predict", files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1 and "latency_ms" in body


def test_predict_rejects_non_image():
    r = _client().post("/predict", files=[("files", ("a.png", b"nope", "image/png"))])
    assert r.status_code == 400


def test_predict_returns_decision_when_threshold_set():
    # a detector configured with a threshold returns a boolean is_anomaly
    r = _client(detector=_FakeDetector(score=0.9, is_anomaly=True)).post(
        "/predict", files=[("files", ("a.png", _png_bytes(), "image/png"))]
    )
    assert r.json()["predictions"][0]["is_anomaly"] is True


def test_metrics_endpoint_exposes_ml_metrics():
    c = _client()
    c.post("/predict", files=[("files", ("a.png", _png_bytes(), "image/png"))])
    body = c.get("/metrics").text
    assert "predict_latency_seconds" in body
    assert "predict_images_total" in body
    assert "active_nfe" in body


def test_drift_endpoint_and_predict_feeds_monitor():
    drift = DriftMonitor(ref_mean=0.5, ref_std=0.1, window=2)
    c = _client(drift=drift)
    # two predictions fill the window, live scores == ref mean -> no drift
    for _ in range(2):
        c.post("/predict", files=[("files", ("a.png", _png_bytes(), "image/png"))])
    status = c.get("/drift").json()
    assert status["drifting"] is False and abs(status["drift_z"]) < 1e-6


def _tiny_cfg():
    return OmegaConf.create(
        {
            "device": "cpu",
            "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0, "t_min": 1e-5, "t_max": 1.0},
            "model": {
                "base_channels": 32,
                "channel_mults": [1, 2],
                "layers_per_block": 1,
                "time_embed_scale": 999.0,
                "norm_num_groups": 32,
            },
            "data": {"image_size": 16},
            "scoring": {
                "t_stars": [0.2],
                "n_steps": 3,
                "solver": "ddim",
                "smooth_sigma": 1.0,
                "image_score": "max",
                "topk_frac": 0.01,
            },
        }
    )


def test_detector_scores_real_model(tmp_path):
    cfg = _tiny_cfg()
    net = build_model(build_sde(cfg.sde), cfg.model, cfg.data)
    ckpt = tmp_path / "m.pt"
    torch.save({"ema": net.state_dict(), "model": net.state_dict()}, ckpt)

    det = AnomalyDetector(cfg, str(ckpt))
    res = det.score_image(Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")))
    assert set(res) == {"score", "is_anomaly", "heatmap_png_b64"}
    assert isinstance(res["score"], float) and len(res["heatmap_png_b64"]) > 0


def test_export_score_net_or_graceful_fallback(tmp_path):
    cfg = _tiny_cfg()
    net = build_model(build_sde(cfg.sde), cfg.model, cfg.data)
    report = export_score_net(net, image_size=16, out_path=tmp_path / "score.ts")
    assert "exported" in report
    if report["exported"]:
        assert (tmp_path / "score.ts").exists()

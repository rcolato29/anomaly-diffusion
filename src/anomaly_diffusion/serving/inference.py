"""Serving-facing inference wrapper.

Loads a trained score model once and scores uploaded images with the fast DDIM path. Kept
separate from the FastAPI layer so it can be unit tested and reused (e.g. by the drift
monitor).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.data.mvtec import build_transform
from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
from anomaly_diffusion.utils.device import resolve_device


def heatmap_to_b64(amap: np.ndarray) -> str:
    """Encode a (H, W) anomaly map as a base64 PNG, per-image min-max normalized."""
    a = amap - amap.min()
    a = a / (a.max() + 1e-8)
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class AnomalyDetector:
    def __init__(self, cfg: DictConfig, checkpoint: str, device: torch.device | None = None):
        self.device = device or resolve_device(cfg.device)
        self.sde = build_sde(cfg.sde)
        self.model = build_model(self.sde, cfg.model, cfg.data).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt.get("ema", ckpt["model"]))  # EMA weights for serving
        self.model.eval()

        sc = cfg.scoring
        self.transform = build_transform(cfg.data.image_size, augment=False)
        self.t_stars = list(sc.t_stars)
        self.n_steps = sc.n_steps
        self.solver = sc.get("solver", "ddim")
        self.smooth_sigma = sc.smooth_sigma
        self.image_score_method = sc.image_score
        self.topk_frac = sc.topk_frac
        # optional EVT-calibrated decision threshold (set via serving config), None -> no flag
        self.threshold = sc.get("threshold", None)

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image]) -> list[dict]:
        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        amap = anomaly_map(
            self.model,
            self.sde,
            x,
            self.t_stars,
            self.n_steps,
            probability_flow=True,
            smooth_sigma=self.smooth_sigma,
            solver=self.solver,
        )
        scores = image_score(amap, self.image_score_method, self.topk_frac).cpu().tolist()
        maps = amap[:, 0].cpu().numpy()
        results = []
        for score, hm in zip(scores, maps, strict=True):
            is_anomaly = None if self.threshold is None else bool(score > self.threshold)
            results.append(
                {
                    "score": float(score),
                    "is_anomaly": is_anomaly,
                    "heatmap_png_b64": heatmap_to_b64(hm),
                }
            )
        return results

    def score_image(self, image: Image.Image) -> dict:
        return self.score_batch([image])[0]

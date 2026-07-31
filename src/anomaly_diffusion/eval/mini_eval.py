"""Frozen mini-eval for the metrics-regression CI gate.

A small, deterministic subset of a category's test split with a single golden metric
(reconstruction image AUROC). Freezing the exact image list makes the metric reproducible,
so CI can assert it stays above a floor and fail the build on a model regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anomaly_diffusion.data.mvtec import IMG_EXTENSIONS, build_transform
from anomaly_diffusion.eval.metrics import image_auroc
from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score


def build_manifest(root: str | Path, category: str, k_per_class: int = 8) -> list[dict]:
    """Deterministically pick the first k_per_class normal and anomalous test images.

    Selection is by sorted path, so the manifest is stable across machines and runs.
    """
    test_dir = Path(root) / category / "test"
    good, anomalous = [], []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = 0 if defect_dir.name == "good" else 1
        imgs = sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        for img in imgs:
            entry = {"path": str(img.relative_to(root)), "label": label}
            (good if label == 0 else anomalous).append(entry)
    return good[:k_per_class] + anomalous[:k_per_class]


def save_manifest(manifest: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2))


def load_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


@torch.no_grad()
def run_mini_eval(score_fn, sde, cfg, manifest: list[dict], device) -> dict:
    """Compute reconstruction image AUROC over exactly the manifest images."""
    transform = build_transform(cfg.data.image_size, augment=False)
    imgs = torch.stack(
        [transform(Image.open(Path(cfg.data.root) / e["path"]).convert("RGB")) for e in manifest]
    ).to(device)
    labels = [e["label"] for e in manifest]

    sc = cfg.scoring
    amap = anomaly_map(
        score_fn,
        sde,
        imgs,
        sc.t_stars,
        sc.n_steps,
        sc.probability_flow,
        sc.smooth_sigma,
        solver=sc.get("solver", "pf_ode"),
    )
    scores = image_score(amap, sc.image_score, sc.topk_frac).cpu().numpy()
    return {"image_auroc": image_auroc(np.array(labels), scores), "n": len(manifest)}


def freeze_golden(
    score_fn, sde, cfg, manifest: list[dict], device, floor_margin: float = 0.05
) -> dict:
    """Run the mini-eval once and freeze the golden AUROC + a regression floor."""
    golden = run_mini_eval(score_fn, sde, cfg, manifest, device)["image_auroc"]
    return {
        "category": cfg.data.category,
        "manifest": manifest,
        "golden_auroc": round(float(golden), 4),
        "floor": round(float(golden) - floor_margin, 4),
    }


def run_gate(score_fn, sde, cfg, device, golden: dict) -> dict:
    """Re-run the frozen mini-eval and check the AUROC against the committed floor."""
    auroc = run_mini_eval(score_fn, sde, cfg, golden["manifest"], device)["image_auroc"]
    passed = auroc >= golden["floor"]
    return {
        "image_auroc": round(float(auroc), 4),
        "floor": golden["floor"],
        "golden_auroc": golden.get("golden_auroc"),
        "passed": bool(passed),
    }

"""EVT threshold calibration for serving.

Fits the tail of the reference-normal score distribution to set a decision threshold for a
target false-alarm rate, using normal data only (no anomaly labels). The value is committed
and loaded by the service so /predict returns a decision. The drift monitor flags when
inputs have moved far enough to warrant recalibration.
"""

from __future__ import annotations

import statistics

import torch

from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
from anomaly_diffusion.thresholding.evt import evt_threshold


@torch.no_grad()
def calibrate_evt_threshold(
    detector, loader, far: float = 0.01, level: float = 0.9, max_images: int = 128
) -> dict:
    """Score reference-normal images through the detector and fit the EVT threshold."""
    scores: list[float] = []
    for batch in loader:
        x = batch["image"].to(detector.device)
        amap = anomaly_map(
            detector.model,
            detector.sde,
            x,
            detector.t_stars,
            detector.n_steps,
            probability_flow=True,
            smooth_sigma=detector.smooth_sigma,
            solver=detector.solver,
        )
        scores.extend(
            image_score(amap, detector.image_score_method, detector.topk_frac).cpu().tolist()
        )
        if len(scores) >= max_images:
            break

    thr = evt_threshold(scores, q=far, level=level)
    # ref_mean/ref_std also seed the drift monitor's reference window (one calibration,
    # both the decision threshold and the drift baseline).
    return {
        "threshold": float(thr),
        "far": far,
        "level": level,
        "n_reference": len(scores),
        "ref_mean": statistics.fmean(scores),
        "ref_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    }

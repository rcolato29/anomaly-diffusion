"""Detection and localization metrics.

AUROC comes from scikit-learn. AU-PRO is the standard per-region localization metric for
MVTec, computed here with scipy connected components so the core environment does not
depend on anomalib (the baselines run in a separate image and report their own metrics).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores) -> float:
    """Image-level AUROC. labels: 0=normal, 1=anomaly. scores: anomaly scores."""
    return float(roc_auc_score(np.asarray(labels).ravel(), np.asarray(scores).ravel()))


def pixel_auroc(masks, maps) -> float:
    """Pixel-level AUROC over all test pixels.

    masks: binary ground-truth (0/1). maps: per-pixel anomaly scores. Both are
    flattened across every image and pixel.
    """
    m = (np.asarray(masks).ravel() > 0.5).astype(np.uint8)
    s = np.asarray(maps).ravel()
    if m.min() == m.max():
        return float("nan")  # AUROC undefined if only one class is present
    return float(roc_auc_score(m, s))


def au_pro(maps, masks, fpr_limit: float = 0.3, n_thresholds: int = 100) -> float:
    """Area under the Per-Region Overlap curve, integrated up to fpr_limit.

    Each connected ground-truth defect region contributes equally regardless of size (so
    small defects are not drowned out by large ones, unlike pixel AUROC). maps and
    masks are (N, H, W) arrays of per-pixel scores and binary masks.
    """
    maps = np.asarray(maps, dtype=float)
    masks = np.asarray(masks) > 0.5
    if maps.ndim == 4:
        maps, masks = maps[:, 0], masks[:, 0]

    regions = []  # (image_index, flat pixel indices of one connected defect region)
    for i, m in enumerate(masks):
        lbl, n = label(m)
        for r in range(1, n + 1):
            regions.append((i, np.flatnonzero((lbl == r).ravel())))
    n_neg = int((~masks).sum())
    if not regions or n_neg == 0:
        return float("nan")

    flat_maps = maps.reshape(maps.shape[0], -1)
    thresholds = np.linspace(maps.max(), maps.min(), n_thresholds)
    pros, fprs = [], []
    for t in thresholds:
        pred = flat_maps >= t
        pro = np.mean([pred[i, idx].mean() for i, idx in regions])
        fp = int((pred.reshape(masks.shape) & ~masks).sum())
        pros.append(pro)
        fprs.append(fp / n_neg)

    fprs, pros = np.array(fprs), np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    # Integrate PRO over FPR in [0, fpr_limit], interpolating PRO at the boundary so a
    # sharp FPR step doesn't drop the enclosed area.
    if fprs[-1] < fpr_limit:
        xs, ys = np.append(fprs, fpr_limit), np.append(pros, pros[-1])
    else:
        pro_at = float(np.interp(fpr_limit, fprs, pros))
        keep = fprs < fpr_limit
        xs, ys = np.append(fprs[keep], fpr_limit), np.append(pros[keep], pro_at)
    return float(np.trapezoid(ys, xs) / fpr_limit)

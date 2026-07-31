"""Run reconstruction-based anomaly detection over an MVTec test split.

Computes image- and pixel-level AUROC and saves qualitative heatmap grids
(input | anomaly map | ground-truth mask). Shared by the local CLI (scripts/eval.py) and
the Modal launcher.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.data.mvtec import build_dataloader
from anomaly_diffusion.eval.metrics import au_pro, image_auroc, pixel_auroc
from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
from anomaly_diffusion.thresholding.evt import evt_threshold
from anomaly_diffusion.tracking.logger import WandbLogger
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.seed import seed_everything


def _load_model(cfg: DictConfig, checkpoint: str, device: torch.device):
    sde = build_sde(cfg.sde)
    model = build_model(sde, cfg.model, cfg.data).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))  # prefer EMA weights
    model.eval()
    return sde, model


def _save_results(metrics: dict, out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


def _save_qualitative(examples: list[dict], out_path: Path, max_examples: int = 8) -> None:
    panels = []
    for ex in examples[:max_examples]:
        img = (ex["image"].clamp(-1, 1) + 1) / 2  # (3,H,W) -> [0,1]
        amap = ex["amap"]
        amap = amap / (amap.amax() + 1e-8)  # per-image normalization for viz
        panels += [img, amap.repeat(3, 1, 1), ex["mask"].repeat(3, 1, 1)]
    grid = make_grid(torch.stack(panels), nrow=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def evaluate_from_cfg(
    cfg: DictConfig, checkpoint: str, out_dir: str | Path = "outputs/eval"
) -> dict:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Evaluating {checkpoint} on {cfg.data.category} ({device})")
    sde, model = _load_model(cfg, checkpoint, device)

    loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="test",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    sc = cfg.scoring
    t_stars = list(sc.t_stars)

    labels, scores, maps, masks = [], [], [], []
    anomaly_examples: list[dict] = []

    for batch in loader:
        x0 = batch["image"].to(device)
        amap = anomaly_map(
            model,
            sde,
            x0,
            t_stars=t_stars,
            n_steps=sc.n_steps,
            probability_flow=sc.probability_flow,
            smooth_sigma=sc.smooth_sigma,
            solver=sc.get("solver", "pf_ode"),
        )
        img_scores = image_score(amap, method=sc.image_score, topk_frac=sc.topk_frac)

        labels.extend(batch["label"].tolist())
        scores.extend(img_scores.cpu().tolist())
        amap_cpu = amap.cpu()
        maps.append(amap_cpu[:, 0].numpy())  # (B, H, W)
        masks.append(batch["mask"][:, 0].numpy())

        for i in range(x0.shape[0]):
            if batch["label"][i].item() == 1 and len(anomaly_examples) < 8:
                anomaly_examples.append(
                    {"image": x0[i].cpu(), "amap": amap_cpu[i], "mask": batch["mask"][i]}
                )

    labels = np.array(labels)
    maps, masks = np.concatenate(maps), np.concatenate(masks)
    metrics = {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks, maps),
        "au_pro": au_pro(maps, masks),
        "n_test": int(labels.size),
    }

    # EVT: statistically calibrated threshold on normal-image scores at a target FAR
    scores = np.array(scores)
    normal_scores = scores[labels == 0]
    if normal_scores.size >= 20:
        try:
            thr = evt_threshold(normal_scores, q=cfg.scoring.evt_far, level=cfg.scoring.evt_level)
            metrics["evt_threshold"] = thr
            metrics["detection_rate_at_far"] = float((scores[labels == 1] > thr).mean())
        except ValueError as e:
            print(f"EVT skipped: {e}")

    metrics["category"], metrics["method"] = cfg.data.category, "diffusion"
    out_dir = Path(out_dir)
    _save_qualitative(anomaly_examples, out_dir / f"{cfg.data.category}_heatmaps.png")
    _save_results(metrics, out_dir / "results" / f"{cfg.data.category}_diffusion.json")

    print(
        f"image AUROC: {metrics['image_auroc']:.4f} | pixel AUROC: {metrics['pixel_auroc']:.4f} "
        f"| AU-PRO: {metrics['au_pro']:.4f} | n={metrics['n_test']}"
    )

    logger = WandbLogger(
        project=cfg.tracking.project,
        run_name=f"eval-{cfg.data.category}",
        config={"checkpoint": checkpoint, "scoring": dict(sc)},
        mode=cfg.tracking.mode,
    )
    logger.log_scalars({k: v for k, v in metrics.items() if isinstance(v, float)})
    logger.finish()
    return metrics

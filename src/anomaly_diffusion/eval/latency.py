"""NFE-vs-quality curve + latency.

Sweeps the DDIM reverse-step budget (NFE) and records, at each budget, the detection
quality (image AUROC) and the per-image reconstruction latency (p50/p95, throughput) on
named hardware. The curve justifies the final NFE operating point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from anomaly_diffusion.data.mvtec import build_dataloader
from anomaly_diffusion.eval.evaluate import _load_model
from anomaly_diffusion.eval.metrics import image_auroc
from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.latency import measure_latency
from anomaly_diffusion.utils.seed import seed_everything


def _score_loader(model, sde, loader, cfg, n_steps, device):
    sc = cfg.scoring
    labels, scores = [], []
    for batch in loader:
        x0 = batch["image"].to(device)
        amap = anomaly_map(
            model, sde, x0, sc.t_stars, n_steps, sc.probability_flow, sc.smooth_sigma, solver="ddim"
        )
        labels.extend(batch["label"].tolist())
        scores.extend(image_score(amap, sc.image_score, sc.topk_frac).cpu().tolist())
    return image_auroc(np.array(labels), scores)


def nfe_quality_curve(
    cfg: DictConfig,
    checkpoint: str,
    nfe_list: list[int],
    hardware: str,
    out_dir: str | Path = "outputs/latency",
) -> dict:
    """For each NFE budget: DDIM image AUROC + per-image latency on hardware."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
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
    latency_batch = next(iter(loader))["image"].to(device)

    rows = []
    for nfe in nfe_list:
        auroc = _score_loader(model, sde, loader, cfg, nfe, device)

        def _one_score(n=nfe):
            amap = anomaly_map(
                model,
                sde,
                latency_batch,
                cfg.scoring.t_stars,
                n,
                cfg.scoring.probability_flow,
                cfg.scoring.smooth_sigma,
                solver="ddim",
            )
            image_score(amap, cfg.scoring.image_score, cfg.scoring.topk_frac)

        lat = measure_latency(_one_score, device, batch_size=latency_batch.shape[0])
        rows.append({"nfe": nfe, "image_auroc": auroc, **lat})
        print(
            f"NFE {nfe:>3} | AUROC {auroc:.4f} | "
            f"p50 {lat['p50_ms']:.1f}ms | p95 {lat['p95_ms']:.1f}ms"
        )

    result = {"category": cfg.data.category, "hardware": hardware, "curve": rows}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.data.category}_nfe_curve.json").write_text(json.dumps(result, indent=2))
    _plot_curve(rows, hardware, out_dir / f"{cfg.data.category}_nfe_curve.png")
    return result


def _plot_curve(rows: list[dict], hardware: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nfe = [r["nfe"] for r in rows]
    auroc = [r["image_auroc"] for r in rows]
    p50 = [r["p50_ms"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(nfe, auroc, "o-", color="tab:blue", label="image AUROC")
    ax1.set_xlabel("NFE (DDIM reverse steps per scale)")
    ax1.set_ylabel("image AUROC", color="tab:blue")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(nfe, p50, "s--", color="tab:red", label="p50 latency")
    ax2.set_ylabel("p50 latency / image (ms)", color="tab:red")
    ax1.set_title(f"NFE vs. quality and latency ({hardware})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

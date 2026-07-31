"""Likelihood-reliability study.

Compares four anomaly scorers on the test split: reconstruction, score-norm, raw
likelihood, and typicality (distance from the normal mean log-likelihood). Likelihood-based
OOD detection is known to fail (Nalisnick et al., 2019). A density model can assign higher
likelihood to anomalies, which appears here as near-chance raw-likelihood AUROC and
overlapping log-likelihood histograms for normal and anomalous images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from anomaly_diffusion.data.mvtec import build_dataloader
from anomaly_diffusion.eval.evaluate import _load_model
from anomaly_diffusion.eval.metrics import image_auroc
from anomaly_diffusion.likelihood.ode_likelihood import bits_per_dim, log_likelihood
from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
from anomaly_diffusion.scoring.score_norm import score_norm
from anomaly_diffusion.tracking.logger import WandbLogger
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.seed import seed_everything


def _bpd_batch(model, sde, x0, lik) -> torch.Tensor:
    logp = log_likelihood(
        model,
        sde,
        x0,
        n_steps=lik.n_steps,
        method=lik.method,
        hutchinson_samples=lik.hutchinson_samples,
    )
    return bits_per_dim(logp, x0)


def _reference_bpd(model, sde, cfg, device, n_ref: int) -> np.ndarray:
    """Mean/spread of bits/dim over normal (train/good) images, for the typicality test."""
    loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.reliability.likelihood.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    vals, seen = [], 0
    for batch in loader:
        x0 = batch["image"].to(device)
        vals.append(_bpd_batch(model, sde, x0, cfg.reliability.likelihood).cpu().numpy())
        seen += x0.shape[0]
        if seen >= n_ref:
            break
    return np.concatenate(vals)[:n_ref]


def _save_histogram(bpd: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    ll = -bpd * np.log(2)  # bits/dim -> log-likelihood per dimension (nats), higher = more likely
    pad = 0.1 * (ll.max() - ll.min())  # extend past the data so the KDE tails resolve to zero
    xs = np.linspace(ll.min() - pad, ll.max() + pad, 400)

    # Density curves (not raw counts) so the two classes are comparable despite the test
    # split holding far more defective images than normal ones.
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, name, color in [
        (ll[labels == 0], "normal", "tab:blue"),
        (ll[labels == 1], "anomaly", "tab:red"),
    ]:
        density = gaussian_kde(values)(xs)
        ax.fill_between(xs, density, alpha=0.5, color=color, label=name)
        ax.plot(xs, density, color=color, linewidth=1.2)
    ax.set_xlabel("log-likelihood per dimension in nats (higher = more likely)")
    ax.set_ylabel("density")
    ax.set_title("Log-likelihood of normal vs. anomalous images")
    ax.set_ylim(bottom=0)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def reliability_study(
    cfg: DictConfig, checkpoint: str, out_dir: str | Path = "outputs/reliability"
) -> dict:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    rc = cfg.reliability
    print(f"Reliability study: {checkpoint} on {cfg.data.category} ({device})")
    sde, model = _load_model(cfg, checkpoint, device)

    mu = float(np.mean(_reference_bpd(model, sde, cfg, device, rc.n_ref)))
    print(f"Reference normal mean log-likelihood/dim: {-mu * np.log(2):.4f} nats")

    loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="test",
        image_size=cfg.data.image_size,
        batch_size=rc.likelihood.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    sc = cfg.scoring
    labels, recon, snorm, bpd = [], [], [], []
    for batch in loader:
        x0 = batch["image"].to(device)
        labels.extend(batch["label"].tolist())

        amap = anomaly_map(
            model, sde, x0, sc.t_stars, sc.n_steps, sc.probability_flow, sc.smooth_sigma
        )
        recon.extend(image_score(amap, sc.image_score, sc.topk_frac).cpu().tolist())
        snorm.extend(score_norm(model, sde, x0, list(rc.score_norm_t)).cpu().tolist())
        bpd.extend(_bpd_batch(model, sde, x0, rc.likelihood).cpu().tolist())

    labels = np.array(labels)
    bpd = np.array(bpd)
    # Anomaly scores: higher = more anomalous. Raw likelihood uses bpd directly (lower
    # likelihood -> higher bpd). Typicality uses distance from the normal mean log-likelihood.
    typicality = np.abs(bpd - mu)

    aurocs = {
        "reconstruction": image_auroc(labels, recon),
        "score_norm": image_auroc(labels, snorm),
        "likelihood_raw": image_auroc(labels, bpd),
        "typicality": image_auroc(labels, typicality),
    }

    out_dir = Path(out_dir)
    _save_histogram(bpd, labels, out_dir / f"{cfg.data.category}_likelihood_hist.png")

    print("\nImage AUROC by scorer:")
    for name, val in aurocs.items():
        print(f"  {name:<16} {val:.4f}")

    logger = WandbLogger(
        project=cfg.tracking.project,
        run_name=f"reliability-{cfg.data.category}",
        config={"checkpoint": checkpoint, "reliability": dict(rc)},
        mode=cfg.tracking.mode,
    )
    logger.log_scalars({f"auroc/{k}": v for k, v in aurocs.items()})
    logger.finish()
    return {"auroc": aurocs, "ref_bpd_mean": mu, "n_test": int(labels.size)}

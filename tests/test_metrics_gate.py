import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.eval.mini_eval import build_manifest, freeze_golden, run_gate
from anomaly_diffusion.utils.seed import seed_everything


def _make_test_split(root, category="widget", n=4):
    for defect in ("good", "scratch"):
        d = root / category / "test" / defect
        d.mkdir(parents=True)
        for i in range(n):
            arr = (np.random.rand(16, 16, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:03d}.png")


def _cfg(root):
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0, "t_min": 1e-5, "t_max": 1.0},
            "model": {
                "base_channels": 32,
                "channel_mults": [1, 2],
                "layers_per_block": 1,
                "time_embed_scale": 999.0,
                "norm_num_groups": 32,
            },
            "data": {"root": str(root), "category": "widget", "image_size": 16},
            "scoring": {
                "t_stars": [0.2],
                "n_steps": 3,
                "solver": "ddim",
                "probability_flow": True,
                "smooth_sigma": 1.0,
                "image_score": "max",
                "topk_frac": 0.01,
            },
        }
    )


def test_gate_passes_at_floor_and_fails_on_regression(tmp_path):
    _make_test_split(tmp_path)
    cfg = _cfg(tmp_path)
    sde = build_sde(cfg.sde)
    model = build_model(sde, cfg.model, cfg.data)
    manifest = build_manifest(tmp_path, "widget", k_per_class=3)

    # freeze and re-run under the same seed -> identical AUROC -> passes its own floor
    seed_everything(0)
    golden = freeze_golden(model, sde, cfg, manifest, torch.device("cpu"))
    seed_everything(0)
    assert run_gate(model, sde, cfg, torch.device("cpu"), golden)["passed"]

    # an impossibly high floor must fail the gate
    regressed = {**golden, "floor": 1.01}
    assert run_gate(model, sde, cfg, torch.device("cpu"), regressed)["passed"] is False

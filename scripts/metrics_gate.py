"""Metrics-regression gate (local CLI). Exits nonzero if the mini-eval AUROC regresses.

Example:
    uv run python scripts/metrics_gate.py data.category=bottle \
        checkpoint=outputs/checkpoints/last.pt

Reads the committed file configs/mini_eval/<category>.json (produced by
modal run modal_app.py::freeze_mini_eval). For GPU/data on Modal, use
modal run modal_app.py::metrics_gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.eval.evaluate import _load_model
from anomaly_diffusion.eval.mini_eval import run_gate
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    golden_path = Path(f"configs/mini_eval/{cfg.data.category}.json")
    if not golden_path.exists():
        print(f"No frozen golden for {cfg.data.category}. Run freeze_mini_eval first. Skipping.")
        return
    golden = json.loads(golden_path.read_text())

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sde, model = _load_model(cfg, cfg.get("checkpoint", "outputs/checkpoints/last.pt"), device)
    result = run_gate(model, sde, cfg, device, golden)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] image AUROC {result['image_auroc']} vs floor {result['floor']}")
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

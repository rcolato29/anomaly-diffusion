"""TorchScript export of the score network.

Exporting a time-conditioned U-Net is fiddly (dynamic shapes + the time embedding), and it
must not block serving. On any failure, it returns a report with exported=False and the
server keeps running the native PyTorch model.
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_score_net(model, image_size: int, out_path: str | Path, device=None) -> dict:
    """Trace model(x, t) (returns the score) to TorchScript. Never raises."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    example_x = torch.randn(1, 3, image_size, image_size, device=device)
    example_t = torch.rand(1, device=device)
    try:
        with torch.no_grad():
            scripted = torch.jit.trace(model, (example_x, example_t), check_trace=False)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(out_path))
        return {"exported": True, "path": str(out_path)}
    except Exception as e:
        return {"exported": False, "error": f"{type(e).__name__}: {e}"}

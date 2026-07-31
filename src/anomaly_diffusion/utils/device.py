"""Device resolution: prefer CUDA, then MPS, then CPU."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a torch device.

    Args:
        preference: one of "auto", "cuda", "mps", "cpu". "auto"
            picks the best available accelerator (cuda > mps > cpu).
    """
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine.")
    return torch.device(preference)


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Best AMP dtype for the device (bf16 on CUDA, fp16 fallback, fp32 on CPU/MPS)."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

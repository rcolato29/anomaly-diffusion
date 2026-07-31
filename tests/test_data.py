import numpy as np
from PIL import Image

from anomaly_diffusion.data.mvtec import build_dataloader


def _make_fixture(root, category="widget", n=6):
    good = root / category / "train" / "good"
    good.mkdir(parents=True)
    for i in range(n):
        arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(good / f"{i:03d}.png")


def test_loader_shapes_and_normalization(tmp_path):
    _make_fixture(tmp_path)
    loader = build_dataloader(
        root=tmp_path,
        category="widget",
        split="train",
        image_size=16,
        batch_size=4,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 16, 16)
    # normalized to roughly [-1, 1]
    assert batch["image"].min() >= -1.0001
    assert batch["image"].max() <= 1.0001
    assert (batch["label"] == 0).all()

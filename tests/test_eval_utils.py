import numpy as np
from PIL import Image

from anomaly_diffusion.eval.mini_eval import build_manifest, load_manifest, save_manifest
from anomaly_diffusion.eval.results_table import to_markdown


def _make_test_split(root, category="widget", n=6):
    for defect in ("good", "scratch"):
        d = root / category / "test" / defect
        d.mkdir(parents=True)
        for i in range(n):
            arr = (np.random.rand(16, 16, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{i:03d}.png")


def test_manifest_is_deterministic_and_balanced(tmp_path):
    _make_test_split(tmp_path)
    m1 = build_manifest(tmp_path, "widget", k_per_class=3)
    m2 = build_manifest(tmp_path, "widget", k_per_class=3)
    assert m1 == m2  # deterministic
    assert sum(e["label"] == 0 for e in m1) == 3
    assert sum(e["label"] == 1 for e in m1) == 3


def test_manifest_roundtrip(tmp_path):
    _make_test_split(tmp_path)
    m = build_manifest(tmp_path, "widget", k_per_class=2)
    save_manifest(m, tmp_path / "mini.json")
    assert load_manifest(tmp_path / "mini.json") == m


def test_to_markdown_formats_and_handles_nan():
    rows = {
        "ours": {"image_auroc": 0.9576, "au_pro": float("nan")},
        "patchcore": {"image_auroc": 0.999, "au_pro": 0.95},
    }
    md = to_markdown(rows, ["image_auroc", "au_pro"])
    assert "| ours | 0.958 | n/a |" in md
    assert "| patchcore | 0.999 | 0.950 |" in md

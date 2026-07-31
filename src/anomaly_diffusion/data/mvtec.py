"""MVTec AD one-class dataset.

MVTec layout:

    <root>/<category>/
        train/good/*.png
        test/<defect_type>/*.png # 'good' plus one folder per defect type
        ground_truth/<defect_type>/*_mask.png

Training is one-class: only train/good is used and no masks are returned. The test
split (with image-level labels and pixel masks) is used only for evaluation. Images are
normalized to [-1, 1], a standard convention for diffusion models. Augmentation is restricted
to anomaly-safe transforms (no cutout/occlusion, which could manufacture artificial
defects).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_DATA_HELP = (
    "MVTec AD data not found at {path}. Download the dataset and point data.root at "
    "the extracted directory."
)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS


def build_transform(image_size: int, augment: bool = False) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if augment:
        # Anomaly-safe only. Horizontal flip is valid for most MVTec categories.
        # Disable them per-category in config where it is not (e.g. oriented textures).
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ops)


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
    ):
        self.category_dir = Path(root) / category
        self.split = split
        self.image_size = image_size
        split_dir = self.category_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(_DATA_HELP.format(path=split_dir))

        self.transform = build_transform(image_size, augment=augment and split == "train")
        self.mask_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

        self.samples: list[tuple[Path, int, Path | None]] = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            label = 0 if defect == "good" else 1
            for img in sorted(p for p in defect_dir.iterdir() if _is_image(p)):
                mask = None
                if label == 1:
                    cand = self.category_dir / "ground_truth" / defect / f"{img.stem}_mask.png"
                    mask = cand if cand.exists() else None
                self.samples.append((img, label, mask))

        if not self.samples:
            raise FileNotFoundError(_DATA_HELP.format(path=split_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label, mask_path = self.samples[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        if mask_path is not None:
            mask = (self.mask_transform(Image.open(mask_path).convert("L")) > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        return {"image": image, "label": label, "mask": mask, "path": str(img_path)}


def build_dataloader(
    root: str | Path,
    category: str,
    split: str = "train",
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = False,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = MVTecDataset(root, category, split, image_size, augment)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=split == "train",
        pin_memory=True,
    )

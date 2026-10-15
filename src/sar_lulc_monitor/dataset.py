from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from sar_lulc_monitor.preprocessing import (
    encode_labels,
    normalize_sar,
)


def collect_pairs(
        sar_dir: str | Path,
        label_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Find matching Sentinel-1 and DFC2020 label files."""
    sar_dir = Path(sar_dir)
    label_dir = Path(label_dir)

    if not sar_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing data directories: {sar_dir}, {label_dir}"
        )

    sar_files = sorted(sar_dir.glob("*.tif"))
    label_files = sorted(label_dir.glob("*.tif"))

    pairs = []
    expected_labels = set()

    for sar_path in sar_files:
        label_name = sar_path.name.replace("_s1_", "_dfc_")

        if label_name == sar_path.name:
            raise ValueError(
                f"Unexpected SAR filename: {sar_path.name}"
            )

        expected_labels.add(label_name)
        label_path = label_dir / label_name

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing label for {sar_path.name}: {label_path}"
            )

        pairs.append((sar_path, label_path))

    actual_labels = {p.name for p in label_files}

    if actual_labels != expected_labels:
        raise ValueError(
            "SAR and label file sets do not match. "
            f"Extra labels: {len(actual_labels - expected_labels)}"
        )

    if not pairs:
        raise ValueError(f"No SAR images found in {sar_dir}")

    return pairs


def split_pairs(
        pairs: list[tuple[Path, Path]],
        val_fraction: float = 0.2,
        seed: int = 42,
) -> tuple[list, list]:
    """Create a reproducible train/validation split.

    This is a sample-level random split, not a spatial split.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    if len(pairs) < 2:
        raise ValueError("At least two samples are required")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))

    n_val = max(1, round(len(pairs) * val_fraction))
    val_indices = set(indices[:n_val].tolist())

    train_pairs = [
        pair for i, pair in enumerate(pairs)
        if i not in val_indices
    ]
    val_pairs = [
        pair for i, pair in enumerate(pairs)
        if i in val_indices
    ]

    return train_pairs, val_pairs


class DFCDataset(Dataset):
    """Sentinel-1-only semantic segmentation dataset."""

    def __init__(
            self,
            pairs: list[tuple[Path, Path]],
            transform=None,
    ):
        self.pairs = list(pairs)
        self.transform = transform

        if not self.pairs:
            raise ValueError("Dataset cannot be empty")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        sar_path, label_path = self.pairs[index]

        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)
            sar_transform = src.transform

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label_transform = src.transform

        if sar.shape[0] != 2:
            raise ValueError(
                f"{sar_path.name}: expected 2 SAR bands"
            )

        if sar.shape[1:] != label.shape:
            raise ValueError(
                f"Shape mismatch: {sar_path.name}, "
                f"{label_path.name}"
            )

        if sar_transform != label_transform:
            raise ValueError(
                f"Pixel grid mismatch: {sar_path.name}"
            )

        image = normalize_sar(sar)
        mask = encode_labels(label)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.int64)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "id": sar_path.stem,
        }


def create_datasets(
        data_root: str | Path,
        val_fraction: float = 0.2,
        train_transform=None,
        seed: int = 42,
) -> tuple[DFCDataset, DFCDataset, DFCDataset]:
    """Build train/val/test datasets from original DFC2020.

    Original test       -> new train + validation
    Original validation -> new test
    """
    data_root = Path(data_root)

    original_test = collect_pairs(
        data_root / "ROIs0000_test" / "s1_0",
        data_root / "ROIs0000_test" / "dfc_0",
    )

    original_validation = collect_pairs(
        data_root / "ROIs0000_validation" / "s1_validation",
        data_root / "ROIs0000_validation" / "dfc_validation",
    )

    train_pairs, val_pairs = split_pairs(
        original_test,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_dataset = DFCDataset(train_pairs, transform=train_transform)
    val_dataset = DFCDataset(val_pairs)
    test_dataset = DFCDataset(original_validation)

    return train_dataset, val_dataset, test_dataset

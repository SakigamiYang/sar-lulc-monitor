from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from sar_lulc_monitor.dataset import create_datasets
from sar_lulc_monitor.preprocessing import (
    NUM_CLASSES,
    IGNORE_INDEX,
)

DATA_ROOT = Path("../data/raw")

train_ds, val_ds, test_ds = create_datasets(DATA_ROOT)

print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("Test:", len(test_ds))

sample = train_ds[0]

print("Image:", sample["image"].shape)
print("Mask:", sample["mask"].shape)
print("Image range:",
      sample["image"].min().item(),
      sample["image"].max().item())
print("Label IDs:", np.unique(sample["mask"].numpy()))

loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

print("Batch image:", batch["image"].shape)
print("Batch mask:", batch["mask"].shape)

assert len(set(p[0] for p in train_ds.pairs)
           & set(p[0] for p in val_ds.pairs)) == 0
assert sample["image"].shape == (2, 256, 256)
assert sample["mask"].shape == (256, 256)
assert sample["image"].isfinite().all()
assert 0 <= sample["image"].min() <= sample["image"].max() <= 1
assert set(np.unique(sample["mask"].numpy())).issubset(
    set(range(NUM_CLASSES)) | {IGNORE_INDEX}
)

print("Dataset smoke test passed.")

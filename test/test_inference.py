from pathlib import Path

import numpy as np

from sar_lulc_monitor.inference import (
    load_model,
    predict_sar,
)
from sar_lulc_monitor.preprocessing import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models/unet_full_best.pth"
CONFIG_PATH = (
        PROJECT_ROOT / "outputs/training/unet_full_config.json"
)

SAR_DIR = PROJECT_ROOT / "data/demo"
OUTPUT_DIR = PROJECT_ROOT / "outputs/demo"

# Reuse the GeoTIFF downloaded by gee.py.
sar_files = sorted(SAR_DIR.glob("S1_2025-01_*.tif"))

if not sar_files:
    raise FileNotFoundError("No January 2025 SAR image found")

print("Loading model...", flush=True)

model = load_model(
    checkpoint_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    device="cpu",
)

print("Running inference...", flush=True)

result = predict_sar(
    model=model,
    sar_path=sar_files[0],
    output_dir=OUTPUT_DIR,
)

print("Source:", result.source_path)
print("Map:", result.map_path)
print("Preview:", result.preview_path)
print("Shape:", result.labels.shape)
print("Valid pixels:", result.valid_pixels)

valid_labels = result.labels[result.valid_mask]

classes, counts = np.unique(
    valid_labels,
    return_counts=True,
)

for class_id, count in zip(classes, counts):
    percentage = 100 * count / result.valid_pixels
    print(
        f"{CLASS_NAMES[int(class_id)]:16s}: "
        f"{percentage:6.2f}%"
    )

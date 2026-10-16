"""CPU inference for Sentinel-1 land-cover segmentation."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from .model import UNet
from .preprocessing import (
    IGNORE_INDEX,
    NUM_CLASSES,
    normalize_sar,
)

# Class colors, indexed by encoded class ID (0-7).
CLASS_COLORS = (
    (34, 100, 49),  # Forest
    (145, 158, 72),  # Shrubland
    (157, 214, 115),  # Grassland
    (70, 166, 166),  # Wetlands
    (232, 195, 70),  # Croplands
    (210, 75, 75),  # Urban/Built-up
    (170, 160, 145),  # Barren
    (55, 112, 195),  # Water
)


@dataclass
class LULCPrediction:
    source_path: Path
    map_path: Path
    preview_path: Path
    labels: np.ndarray
    valid_mask: np.ndarray

    @property
    def valid_pixels(self) -> int:
        return int(self.valid_mask.sum())


def load_model(
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = "cpu",
) -> UNet:
    """Load a trained U-Net using the saved training configuration."""

    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    # Training notebook records model settings in its JSON summary.
    base_channels = config.get("base_channels")

    if base_channels is None:
        base_channels = config.get("configuration", {}).get(
            "base_channels"
        )

    if base_channels is None:
        raise ValueError(
            "base_channels is missing from the model configuration"
        )

    model = UNet(
        in_channels=2,
        num_classes=NUM_CLASSES,
        base_channels=int(base_channels),
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def read_sar(
        path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read VV/VH SAR data and preserve its spatial metadata."""

    path = Path(path)

    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError(
                f"Expected 2 SAR bands (VV/VH), got {src.count}"
            )

        image = src.read(masked=True)

        # A pixel is valid only when both bands are valid and finite.
        valid_mask = (
                ~np.ma.getmaskarray(image).any(axis=0)
                & np.isfinite(image.data).all(axis=0)
        )

        if not valid_mask.any():
            raise ValueError(f"No valid SAR pixels in {path}")

        # Replace missing values before normalization.
        raw = np.where(
            valid_mask[None, :, :],
            image.data,
            -25.0,
        ).astype(np.float32)

        profile = src.profile.copy()

    return raw, valid_mask, profile


def predict_array(
        model: UNet,
        image: np.ndarray,
        valid_mask: np.ndarray,
        device: str = "cpu",
) -> np.ndarray:
    """Predict encoded LULC labels for a two-channel SAR array."""

    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError("Expected image shape (2, height, width)")

    if valid_mask.shape != image.shape[1:]:
        raise ValueError("Image and valid mask dimensions differ")

    # Identical preprocessing to the DFC2020 training pipeline.
    normalized = normalize_sar(image)

    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized, dtype=np.float32)
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(tensor)
        prediction = logits.argmax(dim=1).squeeze(0)

    labels = prediction.cpu().numpy().astype(np.uint8)

    # Never assign land-cover classes to missing SAR observations.
    labels[~valid_mask] = IGNORE_INDEX

    return labels


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Convert encoded LULC labels into an RGBA image."""

    height, width = labels.shape

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    for class_id, color in enumerate(CLASS_COLORS):
        mask = labels == class_id
        rgba[mask, :3] = color
        rgba[mask, 3] = 255

    # IGNORE_INDEX remains transparent.
    return rgba


def save_prediction(
        labels: np.ndarray,
        profile: dict,
        map_path: str | Path,
        preview_path: str | Path,
) -> None:
    """Save a georeferenced classification map and color preview."""

    map_path = Path(map_path)
    preview_path = Path(preview_path)

    map_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        height=labels.shape[0],
        width=labels.shape[1],
        dtype="uint8",
        nodata=IGNORE_INDEX,
        compress="lzw",
    )

    with rasterio.open(map_path, "w", **output_profile) as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "LULC class")

        colormap = {
            class_id: (*color, 255)
            for class_id, color in enumerate(CLASS_COLORS)
        }
        colormap[IGNORE_INDEX] = (0, 0, 0, 0)

        dst.write_colormap(1, colormap)

    plt.imsave(
        preview_path,
        colorize_labels(labels),
    )


def predict_sar(
        model: UNet,
        sar_path: str | Path,
        output_dir: str | Path,
        device: str = "cpu",
) -> LULCPrediction:
    """Run inference on one Sentinel-1 GeoTIFF."""

    sar_path = Path(sar_path)
    output_dir = Path(output_dir)

    image, valid_mask, profile = read_sar(sar_path)

    labels = predict_array(
        model=model,
        image=image,
        valid_mask=valid_mask,
        device=device,
    )

    map_path = output_dir / f"{sar_path.stem}_lulc.tif"
    preview_path = output_dir / f"{sar_path.stem}_lulc.png"

    save_prediction(
        labels=labels,
        profile=profile,
        map_path=map_path,
        preview_path=preview_path,
    )

    return LULCPrediction(
        source_path=sar_path,
        map_path=map_path,
        preview_path=preview_path,
        labels=labels,
        valid_mask=valid_mask,
    )

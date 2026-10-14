import numpy as np

# DFC2020 classes observed in our dataset.
# Raw IDs 3 (Savanna) and 8 (Snow/Ice) are absent.
RAW_CLASS_IDS = (1, 2, 4, 5, 6, 7, 9, 10)

CLASS_NAMES = (
    "Forest",
    "Shrubland",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban/Built-up",
    "Barren",
    "Water",
)

NUM_CLASSES = len(RAW_CLASS_IDS)
IGNORE_INDEX = 255

RAW_TO_TRAIN = {
    raw_id: train_id
    for train_id, raw_id in enumerate(RAW_CLASS_IDS)
}


def normalize_sar(
        image: np.ndarray,
        min_db: float = -25.0,
        max_db: float = 0.0,
) -> np.ndarray:
    """Normalize dB SAR data to [0, 1].

    Args:
        image: Array with shape [2, H, W].
               Band order must be VV, VH.

    Returns:
        Float32 array with shape [2, H, W].
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError(
            f"Expected SAR shape [2, H, W], got {image.shape}"
        )

    if max_db <= min_db:
        raise ValueError("max_db must be greater than min_db")

    # Replace invalid values before clipping.
    image = np.nan_to_num(
        image,
        nan=min_db,
        neginf=min_db,
        posinf=max_db,
    )

    image = np.clip(image, min_db, max_db)
    image = (image - min_db) / (max_db - min_db)

    return image.astype(np.float32)


def encode_labels(label: np.ndarray) -> np.ndarray:
    """Map original DFC2020 class IDs to [0, 7].

    Unsupported and invalid IDs become IGNORE_INDEX.
    """
    label = np.asarray(label)

    if label.ndim != 2:
        raise ValueError(
            f"Expected label shape [H, W], got {label.shape}"
        )

    encoded = np.full(
        label.shape,
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    for raw_id, train_id in RAW_TO_TRAIN.items():
        encoded[label == raw_id] = train_id

    return encoded


def decode_labels(encoded: np.ndarray) -> np.ndarray:
    """Convert model predictions back to raw DFC2020 IDs."""
    encoded = np.asarray(encoded)

    decoded = np.full(
        encoded.shape,
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    for raw_id, train_id in RAW_TO_TRAIN.items():
        decoded[encoded == train_id] = raw_id

    return decoded

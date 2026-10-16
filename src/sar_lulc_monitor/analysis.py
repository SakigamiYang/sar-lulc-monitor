"""Monthly land-cover coverage and change analysis."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import LULCPrediction
from .preprocessing import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


@dataclass
class LULCAnalysis:
    coverage: pd.DataFrame
    monthly_change: pd.DataFrame
    first_to_last_change: pd.DataFrame
    common_valid_mask: np.ndarray
    valid_pixel_count: int
    valid_area_km2: float


def advance_month(
        year: int,
        month: int,
        offset: int,
) -> tuple[int, int]:
    """Advance a calendar month by a nonnegative offset."""

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")

    if offset < 0:
        raise ValueError("Offset must be nonnegative")

    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def analyze_predictions(
        predictions: list[LULCPrediction | None],
        start_year: int,
        start_month: int,
        resolution: float = 10.0,
) -> LULCAnalysis:
    """Analyze coverage and changes on a common valid pixel area.

    Args:
        predictions:
            Ordered monthly predictions. Use None for missing months.
        start_year:
            Year of the first prediction.
        start_month:
            Month of the first prediction, 1-12.
        resolution:
            Pixel size in meters. Assumes square projected pixels.

    Returns:
        Coverage, month-to-month change, first-to-last change,
        and common valid area information.
    """

    if not predictions:
        raise ValueError("Predictions cannot be empty")

    if resolution <= 0:
        raise ValueError("Resolution must be positive")

    available = [p for p in predictions if p is not None]

    if not available:
        raise ValueError("No valid monthly predictions available")

    # ---------------------------------------------------------
    # 1. Validate shapes and construct a common valid mask.
    # ---------------------------------------------------------

    shape = available[0].labels.shape
    common_mask = np.ones(shape, dtype=bool)

    for prediction in available:
        labels = prediction.labels
        valid_mask = prediction.valid_mask

        if labels.shape != shape or valid_mask.shape != shape:
            raise ValueError(
                "All monthly predictions must have the same shape"
            )

        common_mask &= (
                valid_mask.astype(bool)
                & (labels != IGNORE_INDEX)
        )

    valid_pixel_count = int(common_mask.sum())

    if valid_pixel_count == 0:
        raise ValueError(
            "No common valid pixels across available months"
        )

    # Pixel area: resolution * resolution in square meters.
    valid_area_km2 = (
            valid_pixel_count * resolution ** 2 / 1_000_000
    )

    # ---------------------------------------------------------
    # 2. Monthly coverage percentages.
    # ---------------------------------------------------------

    records = []

    for index, prediction in enumerate(predictions):
        year, month = advance_month(
            start_year,
            start_month,
            index,
        )

        record = {
            "month": f"{year}-{month:02d}",
        }

        if prediction is None:
            for name in CLASS_NAMES:
                record[name] = np.nan

        else:
            labels = prediction.labels[common_mask]

            if np.any((labels < 0) | (labels >= NUM_CLASSES)):
                raise ValueError(
                    f"Unexpected class ID in {record['month']}"
                )

            counts = np.bincount(
                labels.astype(np.int64),
                minlength=NUM_CLASSES,
            )

            percentages = (
                    counts / valid_pixel_count * 100.0
            )

            for class_id, name in enumerate(CLASS_NAMES):
                record[name] = float(percentages[class_id])

        records.append(record)

    coverage = pd.DataFrame(records).set_index("month")
    coverage.index.name = "month"

    # ---------------------------------------------------------
    # 3. Month-to-month changes in percentage points.
    # ---------------------------------------------------------

    monthly_change = coverage.diff()
    monthly_change.index.name = "month"

    # ---------------------------------------------------------
    # 4. First-to-last changes.
    # ---------------------------------------------------------

    first = coverage.iloc[0]
    last = coverage.iloc[-1]

    first_to_last_change = pd.DataFrame({
        "class_name": list(CLASS_NAMES),
        "first_pct": first.to_numpy(),
        "last_pct": last.to_numpy(),
        "change_pp": (last - first).to_numpy(),
    })

    return LULCAnalysis(
        coverage=coverage,
        monthly_change=monthly_change,
        first_to_last_change=first_to_last_change,
        common_valid_mask=common_mask,
        valid_pixel_count=valid_pixel_count,
        valid_area_km2=valid_area_km2,
    )


def save_analysis(
        analysis: LULCAnalysis,
        output_dir: str | Path,
) -> dict[str, Path]:
    """Export analysis tables as CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "coverage": output_dir / "monthly_coverage.csv",
        "monthly_change": output_dir / "monthly_change_pp.csv",
        "first_to_last": output_dir / "first_to_last_change.csv",
    }

    analysis.coverage.to_csv(
        paths["coverage"],
        float_format="%.4f",
    )

    analysis.monthly_change.to_csv(
        paths["monthly_change"],
        float_format="%.4f",
    )

    analysis.first_to_last_change.to_csv(
        paths["first_to_last"],
        index=False,
        float_format="%.4f",
    )

    return paths

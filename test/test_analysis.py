from pathlib import Path

import numpy as np

from sar_lulc_monitor.analysis import analyze_predictions
from sar_lulc_monitor.inference import LULCPrediction


def make_prediction(labels, valid_mask=None):
    labels = np.array(labels, dtype=np.uint8)

    if valid_mask is None:
        valid_mask = np.ones_like(labels, dtype=bool)

    return LULCPrediction(
        source_path=Path("test.tif"),
        map_path=Path("test_lulc.tif"),
        preview_path=Path("test_lulc.png"),
        labels=labels,
        valid_mask=np.asarray(valid_mask, dtype=bool),
    )


# Month 1: 50% Forest, 50% Water
jan = make_prediction([
    [0, 0],
    [7, 7],
])

# Month 2: 25% Forest, 75% Water
feb = make_prediction([
    [0, 7],
    [7, 7],
])

# Month 3: missing
mar = None

# Month 4: 100% Water
apr = make_prediction([
    [7, 7],
    [7, 7],
])

analysis = analyze_predictions(
    predictions=[jan, feb, mar, apr],
    start_year=2025,
    start_month=1,
    resolution=10,
)

print("Monthly coverage (%)")
print(analysis.coverage[["Forest", "Water"]])

print("\nMonthly change (percentage points)")
print(analysis.monthly_change[["Forest", "Water"]])

print("\nFirst-to-last change")
print(
    analysis.first_to_last_change[
        analysis.first_to_last_change["class_name"].isin(
            ["Forest", "Water"]
        )
    ]
)

assert analysis.coverage.loc["2025-01", "Forest"] == 50.0
assert analysis.coverage.loc["2025-02", "Forest"] == 25.0
assert np.isnan(analysis.coverage.loc["2025-03", "Forest"])
assert analysis.coverage.loc["2025-04", "Water"] == 100.0

assert analysis.monthly_change.loc["2025-02", "Forest"] == -25.0
assert np.isnan(analysis.monthly_change.loc["2025-03", "Forest"])
assert np.isnan(analysis.monthly_change.loc["2025-04", "Forest"])

assert analysis.valid_pixel_count == 4
assert analysis.valid_area_km2 == 0.0004

print("\nAll tests passed.")

from sar_lulc_monitor.network import enable_ipv4_only

enable_ipv4_only()

from pathlib import Path

from sar_lulc_monitor.gee import (
    initialize_gee,
    download_sar_series,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

initialize_gee("deep-learning-exam-2026")

results = download_sar_series(
    latitude=49.3988,
    longitude=8.6724,
    year=2025,
    month=1,
    output_dir=PROJECT_ROOT / "data/demo",
    num_months=12,
)

for result in results:
    print(
        result.label,
        result.path,
        result.scene_count,
        result.cached,
    )

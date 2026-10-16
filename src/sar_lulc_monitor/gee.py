"""Download monthly Sentinel-1 SAR composites from Google Earth Engine."""

from dataclasses import dataclass
from pathlib import Path

import ee
import rasterio
import requests
from pyproj import Transformer

IMAGE_SIZE = 256
RESOLUTION = 10
BANDS = ("VV", "VH")
ORBIT_PASS = "ASCENDING"


@dataclass
class SARImage:
    year: int
    month: int
    path: Path | None
    scene_count: int
    cached: bool = False

    @property
    def label(self) -> str:
        return f"{self.year}-{self.month:02d}"


def initialize_gee(project_id: str) -> None:
    """Initialize Google Earth Engine with a Cloud project."""
    ee.Initialize(project=project_id)


def next_month(year: int, month: int) -> tuple[int, int]:
    """Return the following calendar month."""
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")

    if month == 12:
        return year + 1, 1

    return year, month + 1


def get_utm_grid(
        latitude: float,
        longitude: float,
) -> tuple[str, list[float]]:
    """Build a fixed UTM grid centered on latitude/longitude."""

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180")

    # Standard UTM is intended for approximately 80 S to 84 N.
    if not -80 <= latitude <= 84:
        raise ValueError("Latitude is outside the standard UTM range")

    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    crs = f"EPSG:{epsg}"

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    x, y = transformer.transform(longitude, latitude)

    half_size = IMAGE_SIZE * RESOLUTION / 2

    transform = [
        RESOLUTION,
        0,
        x - half_size,
        0,
        -RESOLUTION,
        y + half_size,
    ]

    return crs, transform


def get_study_region(
        crs: str,
        transform: list[float],
) -> ee.Geometry:
    """Create the Earth Engine region matching the output grid."""

    x_min = transform[2]
    y_max = transform[5]

    x_max = x_min + IMAGE_SIZE * RESOLUTION
    y_min = y_max - IMAGE_SIZE * RESOLUTION

    return ee.Geometry.Rectangle(
        [x_min, y_min, x_max, y_max],
        proj=crs,
        geodesic=False,
    )


def get_monthly_collection(
        region: ee.Geometry,
        year: int,
        month: int,
) -> ee.ImageCollection:
    """Select comparable Sentinel-1 VV/VH scenes for one month."""

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")

    next_year, next_mon = next_month(year, month)

    start = ee.Date.fromYMD(year, month, 1)
    end = ee.Date.fromYMD(next_year, next_mon, 1)

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.eq("orbitProperties_pass", ORBIT_PASS)
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(list(BANDS))
    )


def validate_geotiff(path: Path) -> None:
    """Check the downloaded raster structure and data availability."""

    with rasterio.open(path) as src:
        expected_shape = (
            len(BANDS),
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        actual_shape = (src.count, src.height, src.width)

        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected raster shape: {actual_shape}"
            )

        if src.crs is None:
            raise ValueError("Downloaded GeoTIFF has no CRS")

        for index in range(1, len(BANDS) + 1):
            if not src.read_masks(index).any():
                raise ValueError(
                    f"Band {BANDS[index - 1]} has no valid pixels"
                )


def download_monthly_sar(
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        output_dir: str | Path,
        overwrite: bool = False,
) -> SARImage:
    """Download one monthly Sentinel-1 VV/VH median GeoTIFF.

    Earth Engine must be initialized before calling this function.

    Returns a SARImage whose path is None when no suitable scene
    exists for the requested month.
    """

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"S1_{year}-{month:02d}_"
        f"{latitude:.6f}_{longitude:.6f}_"
        f"{ORBIT_PASS}_"
        f"{IMAGE_SIZE}px_{RESOLUTION}m.tif"
    )
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        try:
            validate_geotiff(output_path)
        except (OSError, ValueError):
            output_path.unlink()
        else:
            return SARImage(
                year=year,
                month=month,
                path=output_path,
                scene_count=-1,  # Not queried when using cache
                cached=True,
            )

    crs, transform = get_utm_grid(latitude, longitude)
    region = get_study_region(crs, transform)

    collection = get_monthly_collection(region, year, month)

    scene_count = int(collection.size().getInfo())

    if scene_count == 0:
        return SARImage(
            year=year,
            month=month,
            path=None,
            scene_count=0,
        )

    monthly_image = collection.median().toFloat()

    url = monthly_image.getDownloadURL({
        "crs": crs,
        "crs_transform": transform,
        "dimensions": [IMAGE_SIZE, IMAGE_SIZE],
        "format": "GEO_TIFF",
    })

    # Download to a temporary file to avoid treating partial data
    # as a successfully cached GeoTIFF.
    temporary_path = output_path.with_suffix(".tmp")

    try:
        with requests.get(
                url,
                stream=True,
                timeout=(15, 120),
        ) as response:
            response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        validate_geotiff(temporary_path)
        temporary_path.replace(output_path)

    finally:
        temporary_path.unlink(missing_ok=True)

    return SARImage(
        year=year,
        month=month,
        path=output_path,
        scene_count=scene_count,
    )


def download_sar_series(
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        output_dir: str | Path,
        num_months: int = 12,
        overwrite: bool = False,
) -> list[SARImage]:
    """Download consecutive monthly Sentinel-1 composites."""

    if num_months < 1:
        raise ValueError("num_months must be at least 1")

    results = []

    current_year = year
    current_month = month

    for index in range(num_months):
        print(
            f"[{index + 1}/{num_months}] "
            f"{current_year}-{current_month:02d}",
            flush=True,
        )

        result = download_monthly_sar(
            latitude=latitude,
            longitude=longitude,
            year=current_year,
            month=current_month,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        results.append(result)

        if result.path is None:
            print("  No available scenes", flush=True)
        elif result.cached:
            print(f"  Cached: {result.path}", flush=True)
        else:
            print(
                f"  Downloaded: {result.path} "
                f"({result.scene_count} scenes)",
                flush=True,
            )

        current_year, current_month = next_month(
            current_year, current_month
        )

    return results

# SAR-LULC Monitor

**Sentinel-1 SAR land-cover segmentation and 12-month monitoring — a proof-of-concept application.**

SAR-LULC Monitor trains a lightweight U-Net on the DFC2020 dataset using **Sentinel-1 VV and VH backscatter only**. Its Streamlit demo accepts a latitude, longitude, starting year and starting month; retrieves 12 monthly Sentinel-1 composites through Google Earth Engine (GEE); predicts eight land-cover classes; and displays monthly coverage and changes in percentage points.

The end-to-end demo, including image downloads, has been tested successfully. This project demonstrates a working acquisition–inference–visualization pipeline, **not a validated operational change-detection service**.

## Features

- Sentinel-1-only semantic segmentation with a compact, three-level U-Net.
- Eight land-cover classes and a `255` ignore/NoData label.
- Monthly VV/VH median composites from Google Earth Engine.
- Fixed **256 × 256** projected image grid at **10 m** resolution (approximately **2.56 × 2.56 km**).
- Twelve geospatially aligned monthly predictions, GeoTIFF outputs and PNG previews.
- Monthly class percentages, adjacent-month changes and first-to-last changes in **percentage points (pp)**.
- Shared valid-pixel area for comparable monthly statistics; missing months are represented as missing data rather than zeros.
- CPU-only inference in the interactive Streamlit demo; optional CUDA acceleration for training/evaluation.
- Downloadable CSV summaries and local reuse of downloaded SAR images.

## Project layout

```text
sar-lulc-monitor/
├── app.py                           # Streamlit demo
├── pyproject.toml                   # Python package and dependencies
├── uv.lock
├── requirements.txt
├── README.md
├── src/sar_lulc_monitor/
│   ├── preprocessing.py             # SAR normalization and class encoding
│   ├── dataset.py                   # DFC2020 pairing and train/val/test protocol
│   ├── augmentation.py              # Joint image/mask flips and 90° rotations
│   ├── model.py                     # U-Net
│   ├── training.py                  # Training and checkpoint helpers
│   ├── metrics.py                   # OA, IoU, F1 and confusion matrix
│   ├── gee.py                       # Monthly Sentinel-1 downloads
│   ├── inference.py                 # Model loading and georeferenced prediction
│   ├── analysis.py                  # Coverage and change statistics
│   └── network.py                   # Optional IPv4-only networking workaround
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── test/                             # Module-level test scripts
├── docs/images/u-net.svg             # Architecture illustration
├── data/                             # Raw training data / downloaded SAR cache
├── models/                           # Model checkpoints
└── outputs/                          # Training, evaluation and demo artifacts
```

`data/`, `models/` and `outputs/` are excluded from Git by `.gitignore`; they must be supplied or generated locally. A source-only archive therefore does **not** contain the dataset, model weights, or exported experiment files.

## Requirements and installation

- Python **3.12+** (as specified in `pyproject.toml`).
- Internet access and a Google account authorized to use Google Earth Engine.
- A Google Cloud project with Earth Engine access enabled.
- The trained model checkpoint and its matching training configuration to run the demo.
- No GPU is required for the Streamlit demo.

This project uses [`uv`](https://docs.astral.sh/uv/) for package management:

```bash
cd sar-lulc-monitor
uv sync
```

Alternatively, install the project into an existing Python environment:

```bash
python -m pip install -e .
```

The authoritative dependency specification is `pyproject.toml` (with reproducible resolution in `uv.lock`).

## Google Earth Engine setup

1. Obtain Earth Engine access for your Google account and configure a suitable Google Cloud project.
2. Authenticate locally:

   ```bash
   uv run earthengine authenticate
   ```

3. In the root `app.py`, set `PROJECT_ID` to your **own** Google Cloud project ID.
4. Run the application from the repository root:

   ```bash
   uv run streamlit run app.py
   ```

Open the local address printed by Streamlit, normally <http://localhost:8501>.

**Networking note:** `app.py` calls `enable_ipv4_only()` from `network.py` before importing the network-using libraries. This is an environment-specific workaround for a macOS network where IPv6 connections time out. It patches Python's `socket.getaddrinfo()` within the application process; it is not a requirement of Earth Engine and can be removed where IPv6 connectivity works correctly.

## Running the demo

1. Enter the **latitude** and **longitude** of the area of interest.
2. Enter the starting **year** and **month** as integers.
3. Click **Run 12-Month Analysis**.
4. Review the 12 monthly LULC maps, class-coverage time series, month-to-month changes and first-to-last comparison.
5. Download the three CSV tables if needed.

For example, the default location is Heidelberg, Germany, at latitude `49.3988`, longitude `8.6724`. Starting at year `2025`, month `1` selects January–December 2025.

The workflow is:

```text
Coordinates + starting year/month
        ↓
GEE: Sentinel-1 GRD (IW, ascending orbit, VV + VH)
        ↓
Monthly median composites on a common UTM grid
        ↓
SAR normalization: clip [-25, 0] dB → [0, 1]
        ↓
U-Net inference (CPU)
        ↓
Eight-class LULC GeoTIFFs and preview PNGs
        ↓
Coverage (%) and temporal differences (percentage points)
```

GEE retrieval uses `COPERNICUS/S1_GRD`, filters for IW acquisition mode, ascending passes, and availability of VV and VH polarizations. The app uses monthly **median composites** and keeps the same projection, pixel size and coordinates across months. A month with no matching scenes is treated as missing, not as a land-cover class.

### Expected local model artifacts

```text
models/unet_full_best.pth
outputs/training/unet_full_config.json
```

The configuration JSON supplies the model's `base_channels` parameter so the exact training architecture can be restored. These files are not included in a source-only distribution.

### Generated files

```text
data/demo/
    S1_YYYY-MM_<lat>_<lon>_ASCENDING_256px_10m.tif

outputs/demo/<case-id>/
    predictions/
        *_lulc.tif                 # Georeferenced class IDs 0–7, NoData 255
        *_lulc.png                 # Color visualization
    analysis/
        monthly_coverage.csv
        monthly_change_pp.csv
        first_to_last_change.csv
```

Existing valid SAR GeoTIFF files are reused to reduce repeated GEE downloads. The model is loaded once per Streamlit process through Streamlit resource caching.

## Training dataset and experimental protocol

The dataset is the **DFC2020** collection. Only the Sentinel-1 SAR tiles and DFC reference labels are used; Sentinel-2 optical imagery is **not** provided to the model.

Expected extraction layout:

```text
data/raw/
├── ROIs0000_test/
│   ├── s1_0/
│   └── dfc_0/
└── ROIs0000_validation/
    ├── s1_validation/
    └── dfc_validation/
```

The project intentionally **reassigns the original dataset partitions**:

- **Training + validation:** the original DFC2020 `ROIs0000_test` set (5,128 paired tiles), randomly split 80/20 with seed 42.
- **Independent evaluation:** the original DFC2020 `ROIs0000_validation` set (986 paired tiles), held out from model training and checkpoint selection.

This is a project-specific split, **not the official DFC2020 evaluation protocol**. Its results must not be presented as directly comparable to an official challenge leaderboard. The training/validation division is random at the patch level rather than geographically grouped.

Each SAR input has shape `(2, 256, 256)`, with VV and VH bands. Values are clipped from **−25 to 0 dB** and linearly normalized to `[0, 1]`. Training uses synchronized random horizontal/vertical flips and 90° rotations of the image and label map.

### Label mapping

| Model ID | DFC2020 raw ID | Class |
|---:|---:|---|
| 0 | 1 | Forest |
| 1 | 2 | Shrubland |
| 2 | 4 | Grassland |
| 3 | 5 | Wetlands |
| 4 | 6 | Croplands |
| 5 | 7 | Urban/Built-up |
| 6 | 9 | Barren |
| 7 | 10 | Water |

Unsupported and invalid reference labels are mapped to `255` and ignored in loss computation and evaluation.

## Model and training

The network is a compact U-Net with **three encoder downsampling stages**, skip connections, a three-stage decoder, and an eight-channel segmentation head. The full model uses `base_channels=32`, two input channels, and **1,948,232 trainable parameters**.

The executed training notebook uses:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss | Pixel-wise cross-entropy (`ignore_index=255`) |
| Batch size | 8 |
| Training budget | **50 epochs** |
| Checkpoint criterion | Highest validation mIoU |
| Random seed | 42 |

The best checkpoint in the provided notebook was selected at **epoch 49**, with a validation **mIoU of 49.35%** and validation **overall accuracy of 79.44%**.

**Training scope:** Training was deliberately limited to 50 epochs to demonstrate feasibility within the project scope. The training loss was still decreasing, so the experiment should **not** be interpreted as fully converged or optimally tuned. Further training or hyperparameter tuning might improve results, but improvement on unseen data is not guaranteed by decreasing training loss alone. No additional training is claimed.

## Independent evaluation

`03_model_evaluation.ipynb` evaluates the selected checkpoint on all **986 tiles** in the held-out project test set, without parameter updates.

| Metric | Result |
|---|---:|
| Overall accuracy (OA) | **58.88%** |
| Mean Intersection over Union (mIoU) | **34.16%** |

Per-class results extracted from the executed notebook:

| Class | IoU | F1 |
|---|---:|---:|
| Forest | 52.21% | 68.60% |
| Shrubland | 12.88% | 22.82% |
| Grassland | 0.42% | 0.84% |
| Wetlands | 0.29% | 0.59% |
| Croplands | 28.95% | 44.90% |
| Urban/Built-up | 59.50% | 74.61% |
| Barren | 22.11% | 36.22% |
| Water | 96.91% | 98.43% |

Performance is uneven across classes, especially for Grassland and Wetlands. The gap between validation and held-out test metrics indicates limited generalization under this evaluation protocol. The results establish that the model and pipeline function, **not** that the predicted maps achieve consistently high accuracy across land-cover types.

## Notebooks and tests

Run the notebooks in order, after preparing the DFC2020 files:

1. `notebooks/01_data_exploration.ipynb` — SAR statistics, labels and example visualizations.
2. `notebooks/02_model_training.ipynb` — model, training, learning curves and best-checkpoint export.
3. `notebooks/03_model_evaluation.ipynb` — held-out test evaluation, per-class metrics, confusion matrix and qualitative predictions.

The notebook and module entry points expect paths consistent with the repository layout; the training/evaluation notebooks can be run on a CUDA-enabled Jupyter environment, while the demo runs on CPU.

Standalone scripts under `test/` cover preprocessing and dataset loading, network access, Earth Engine retrieval, single-image inference and the coverage/change calculations. For example:

```bash
uv run python test/test_analysis.py
uv run python test/test_inference.py
```

The inference script requires the checkpoint/configuration and a previously downloaded monthly SAR GeoTIFF. Earth Engine tests require working authentication and network access.

## Interpretation and limitations

- **Proof of concept, not a production classifier.** The model was trained for a finite 50-epoch budget without claiming convergence or state-of-the-art performance.
- **Predicted changes are not confirmed land-use changes.** Seasonal backscatter shifts, moisture, vegetation phenology, speckle, and acquisition geometry can change predictions even where land cover is stable.
- **Fixed acquisition pass is not a fixed relative orbit.** Filtering ascending scenes alone does not eliminate all geometric differences between acquisitions.
- **Domain shift.** The DFC2020 training distribution may differ substantially from new locations and months retrieved from GEE.
- **Class imbalance and confusion.** Held-out evaluation shows very poor Grassland/Wetlands IoU and substantial confusion with other vegetation classes.
- **Shared comparison footprint.** Coverage percentages use the intersection of valid pixels across available months. Missing entire months remain missing; neighboring changes involving such months are undefined.
- **Study area limitations.** The fixed grid covers a relatively small area (~6.55 km²), uses standard UTM latitudes, and should not be treated as an arbitrary-large-area monitoring system.
- **External reference validation.** An independent contemporaneous reference map has not been established by the artifacts in this source archive. A comparison with a product such as ESA WorldCover would need to account for mismatched observation years and class definitions.

## Project objective

The primary outcome is a **working, reproducible demonstration** that combines SAR acquisition, semantic segmentation, georeferenced inference, temporal aggregation and an interactive user interface. The experimental results and known accuracy limitations are reported explicitly so that the proof of concept is not mistaken for validated operational monitoring.

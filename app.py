"""Streamlit application for Sentinel-1 LULC monitoring."""

from sar_lulc_monitor.network import enable_ipv4_only

# Work around broken IPv6 connectivity on the local network.
# Apply before importing libraries that make network requests.
enable_ipv4_only()

from datetime import date
from pathlib import Path

import streamlit as st

from sar_lulc_monitor.analysis import (
    analyze_predictions,
    save_analysis,
)
from sar_lulc_monitor.gee import (
    download_monthly_sar,
    initialize_gee,
    next_month,
)
from sar_lulc_monitor.inference import (
    CLASS_COLORS,
    load_model,
    predict_sar,
)
from sar_lulc_monitor.preprocessing import CLASS_NAMES

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROJECT_ID = "deep-learning-exam-2026"

MODEL_PATH = PROJECT_ROOT / "models/unet_full_best.pth"
CONFIG_PATH = (
        PROJECT_ROOT / "outputs/training/unet_full_config.json"
)

SAR_DIR = PROJECT_ROOT / "data/demo"
OUTPUT_DIR = PROJECT_ROOT / "outputs/demo"

NUM_MONTHS = 12
RESOLUTION = 10

st.set_page_config(
    page_title="SAR-LULC Monitor",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# Initialization
# ============================================================

@st.cache_resource(show_spinner="Loading U-Net model...")
def get_model():
    """Load the trained model once for CPU inference."""

    return load_model(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device="cpu",
    )


@st.cache_resource(show_spinner="Connecting to Earth Engine...")
def connect_gee(project_id: str):
    """Initialize Earth Engine once per process."""
    initialize_gee(project_id)
    return True


# ============================================================
# Helpers
# ============================================================

def make_case_id(
        latitude: float,
        longitude: float,
        year: int,
        month: int,
) -> str:
    """Generate a directory name for an analysis run."""

    return (
        f"{year}-{month:02d}_"
        f"{latitude:+.6f}_"
        f"{longitude:+.6f}"
    )


def add_months(
        year: int,
        month: int,
        offset: int,
) -> tuple[int, int]:
    """Advance a calendar date by a number of months."""

    index = year * 12 + month - 1 + offset
    return index // 12, index % 12 + 1


def run_analysis(
        latitude: float,
        longitude: float,
        year: int,
        month: int,
):
    """Retrieve, classify and analyze 12 consecutive months."""

    connect_gee(PROJECT_ID)
    model = get_model()

    case_id = make_case_id(
        latitude, longitude, year, month
    )

    case_dir = OUTPUT_DIR / case_id
    prediction_dir = case_dir / "predictions"
    analysis_dir = case_dir / "analysis"

    sar_results = []
    predictions = []

    progress_bar = st.progress(
        0.0,
        text="Starting analysis...",
    )
    status = st.empty()

    current_year = year
    current_month = month

    for index in range(NUM_MONTHS):

        label = f"{current_year}-{current_month:02d}"

        status.info(
            f"Processing {label} "
            f"({index + 1}/{NUM_MONTHS})"
        )

        # Step 1: Retrieve monthly Sentinel-1 imagery.
        sar_result = download_monthly_sar(
            latitude=latitude,
            longitude=longitude,
            year=current_year,
            month=current_month,
            output_dir=SAR_DIR,
        )

        sar_results.append(sar_result)

        # Step 2: Run U-Net inference.
        if sar_result.path is None:
            predictions.append(None)

        else:
            prediction = predict_sar(
                model=model,
                sar_path=sar_result.path,
                output_dir=prediction_dir,
                device="cpu",
            )

            predictions.append(prediction)

        progress_bar.progress(
            (index + 1) / NUM_MONTHS,
            text=(
                f"Completed {index + 1}/{NUM_MONTHS} months"
            ),
        )

        current_year, current_month = next_month(
            current_year,
            current_month,
        )

    # Step 3: Analyze monthly predictions.
    if all(p is None for p in predictions):
        raise RuntimeError(
            "No Sentinel-1 images were available "
            "for the requested period."
        )

    status.info("Calculating land-cover statistics...")

    analysis = analyze_predictions(
        predictions=predictions,
        start_year=year,
        start_month=month,
        resolution=RESOLUTION,
    )

    # Step 4: Save CSV results.
    csv_paths = save_analysis(
        analysis=analysis,
        output_dir=analysis_dir,
    )

    progress_bar.empty()
    status.empty()

    return {
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "month": month,
        "sar_results": sar_results,
        "predictions": predictions,
        "analysis": analysis,
        "csv_paths": csv_paths,
        "case_dir": case_dir,
    }


# ============================================================
# Visualization
# ============================================================

def render_legend():
    """Display the eight classes with their map colors."""

    columns = st.columns(4)

    for index, (name, rgb) in enumerate(
            zip(CLASS_NAMES, CLASS_COLORS)
    ):
        color = "#{:02x}{:02x}{:02x}".format(*rgb)

        with columns[index % 4]:
            st.markdown(
                f'<span style="color:{color};">'
                f"■</span> {name}",
                unsafe_allow_html=True,
            )


def render_monthly_maps(sar_results, predictions):
    """Display 12 predicted LULC maps."""

    st.subheader("Monthly LULC Maps")

    render_legend()

    columns = st.columns(4)

    for index, (sar, prediction) in enumerate(
            zip(sar_results, predictions)
    ):
        with columns[index % 4]:

            st.markdown(f"**{sar.label}**")

            if prediction is None:
                st.info("No SAR data")

            else:
                st.image(
                    str(prediction.preview_path),
                    use_container_width=True,
                )


def render_coverage(analysis):
    """Plot and display monthly class coverage."""

    st.subheader("Land-Cover Coverage Over Time")

    st.line_chart(
        analysis.coverage,
        height=400,
    )

    st.caption(
        "Percentage of the common valid area "
        "classified into each LULC category."
    )

    with st.expander("Monthly Coverage Table (%)"):
        st.dataframe(
            analysis.coverage.round(2),
            use_container_width=True,
        )


def render_monthly_changes(analysis):
    """Display month-to-month changes."""

    st.subheader("Month-to-Month Changes")

    st.caption(
        "Each value is the change relative to the "
        "previous month, measured in percentage points."
    )

    st.dataframe(
        analysis.monthly_change.round(2),
        use_container_width=True,
    )


def render_first_to_last(analysis):
    """Display changes from the first to the last month."""

    st.subheader("First-to-Last Changes")

    changes = analysis.first_to_last_change

    st.bar_chart(
        changes.set_index("class_name")["change_pp"],
        height=350,
    )

    st.dataframe(
        changes.round(2),
        hide_index=True,
        use_container_width=True,
    )


def render_downloads(csv_paths):
    """Provide downloads for the three analysis tables."""

    st.subheader("Download Results")

    files = [
        ("Monthly Coverage CSV", csv_paths["coverage"]),
        ("Monthly Changes CSV", csv_paths["monthly_change"]),
        (
            "First-to-Last Changes CSV",
            csv_paths["first_to_last"],
        ),
    ]

    columns = st.columns(3)

    for column, (label, path) in zip(columns, files):
        with column:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
                use_container_width=True,
            )


def render_results(result):
    """Display the results of the completed analysis."""

    analysis = result["analysis"]
    sar_results = result["sar_results"]
    predictions = result["predictions"]

    st.divider()
    st.header("Analysis Results")

    start_year = result["year"]
    start_month = result["month"]

    end_year, end_month = add_months(
        start_year,
        start_month,
        NUM_MONTHS - 1,
    )

    period = (
        f"{start_year}-{start_month:02d} "
        f"to {end_year}-{end_month:02d}"
    )

    available_months = sum(
        prediction is not None
        for prediction in predictions
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Period", period)

    col2.metric(
        "Available Months",
        f"{available_months}/{NUM_MONTHS}",
    )

    col3.metric(
        "Common Valid Area",
        f"{analysis.valid_area_km2:.2f} km²",
    )

    col4.metric(
        "Common Valid Pixels",
        f"{analysis.valid_pixel_count:,}",
    )

    if available_months < NUM_MONTHS:
        st.warning(
            f"{NUM_MONTHS - available_months} "
            "month(s) have no suitable SAR imagery. "
            "Missing months are excluded from change calculations."
        )

    st.caption(
        "All monthly coverage percentages use the same "
        "valid-pixel area across available images."
    )

    st.divider()

    render_monthly_maps(
        sar_results,
        predictions,
    )

    st.divider()

    render_coverage(analysis)

    st.divider()

    render_monthly_changes(analysis)

    st.divider()

    render_first_to_last(analysis)

    st.divider()

    render_downloads(result["csv_paths"])

    with st.expander("Output Directory"):
        st.code(str(result["case_dir"]))

    st.info(
        "Land-cover maps are model predictions. "
        "Apparent monthly changes can reflect SAR viewing "
        "conditions, seasonal effects or classification errors, "
        "rather than verified land-cover changes."
    )


# ============================================================
# Main UI
# ============================================================

st.title("SAR-LULC Monitor")

st.write(
    "Monthly land-cover classification and change monitoring "
    "using Sentinel-1 SAR imagery and a U-Net model."
)

today = date.today()

# Require the complete twelve-month period to be in the past.
latest_year, latest_month = add_months(
    today.year,
    today.month,
    -NUM_MONTHS,
)

with st.sidebar:
    st.header("Study Area")

    with st.form("analysis_form"):
        latitude = st.number_input(
            "Latitude",
            min_value=-80.0,
            max_value=84.0,
            value=49.3988,
            format="%.6f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=8.6724,
            format="%.6f",
        )

        st.subheader("Start Month")

        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=latest_year,
            value=min(2025, latest_year),
            step=1,
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1,
            step=1,
        )

        st.caption(
            f"Choose a starting month no later than "
            f"{latest_year}-{latest_month:02d}. "
            "The analysis covers 12 consecutive months."
        )

        submitted = st.form_submit_button(
            "Run 12-Month Analysis",
            type="primary",
            use_container_width=True,
        )

    st.divider()

    st.caption("Source: Sentinel-1 GRD")
    st.caption("Polarizations: VV / VH")
    st.caption("Orbit: Ascending")
    st.caption("Composite: Monthly Median")
    st.caption("Resolution: 10 m")
    st.caption("Area: 2.56 × 2.56 km")
    st.caption("Model: U-Net (8 classes)")
    st.caption("Inference: CPU")

# ============================================================
# Application execution
# ============================================================

if submitted:

    selected_year = int(year)
    selected_month = int(month)

    selected_index = selected_year * 12 + selected_month
    latest_index = latest_year * 12 + latest_month

    if selected_index > latest_index:
        st.error(
            "The selected period extends beyond the "
            "last completed calendar month."
        )

    elif not MODEL_PATH.is_file():
        st.error(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    elif not CONFIG_PATH.is_file():
        st.error(
            f"Model configuration not found: {CONFIG_PATH}"
        )

    else:
        # Clear previous results before starting a new run.
        st.session_state.pop("analysis_result", None)

        try:
            result = run_analysis(
                latitude=float(latitude),
                longitude=float(longitude),
                year=selected_year,
                month=selected_month,
            )

            st.session_state["analysis_result"] = result

            st.success(
                "Analysis completed successfully."
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

if "analysis_result" in st.session_state:

    render_results(
        st.session_state["analysis_result"]
    )

else:
    st.info(
        "Enter coordinates and a starting month "
        "in the sidebar, then run the analysis."
    )

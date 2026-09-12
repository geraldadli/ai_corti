"""Streamlit app for WESAD BVP+EDA stress classification.

Upload a wrist BVP trace (Empatica E4 style, 64 Hz) and an EDA trace (4 Hz)
from the SAME recording session and get a class prediction (Baseline /
Stress / Amusement) for every 30-second window, stepped every 5 seconds,
using the bundled residual 1D-CNN + BiLSTM + attention model.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from stress_inference import StressPredictor, prepare_recording, window_at

BUNDLE_DIR = Path(__file__).parent

st.set_page_config(page_title="WESAD Stress Classifier", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> StressPredictor:
    return StressPredictor(BUNDLE_DIR)


def parse_signal(uploaded_file, expected_fs: float) -> np.ndarray:
    """Parse a single-column signal file.

    Accepts either:
      * a plain list of numeric samples (one per line, optional header row), or
      * the Empatica E4 export format (line 1: start unix timestamp,
        line 2: sample rate, remaining lines: samples).
    """
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    values = [line.strip().split(",")[0] for line in raw if line.strip() != ""]

    def to_float(text):
        try:
            return float(text)
        except ValueError:
            return None

    if len(values) >= 2:
        first, second = to_float(values[0]), to_float(values[1])
        if first is not None and second is not None and abs(second - expected_fs) < 1e-6:
            values = values[2:]

    numeric = []
    for v in values:
        f = to_float(v)
        numeric.append(f if f is not None else np.nan)
    return np.asarray(numeric, dtype=float)


def run_timeline(predictor: StressPredictor, bvp: np.ndarray, eda: np.ndarray) -> pd.DataFrame:
    config = predictor.preprocessing
    prepared = prepare_recording(bvp, eda, config)
    duration = min(len(prepared["bvp"]) / config["bvp_fs"], len(prepared["eda"]) / config["eda_fs"])
    last_end = np.floor(duration / config["stride_sec"]) * config["stride_sec"]
    ends = np.arange(config["window_sec"], last_end + 1e-9, config["stride_sec"])

    rows = []
    for end in ends:
        inputs, status = window_at(prepared, float(end), config)
        row = {"window_end_sec": float(end), "status": status}
        if inputs is not None:
            probabilities = predictor.model(
                {k: v[None, ...] for k, v in inputs.items()}, training=False
            ).numpy()[0]
            index = int(np.argmax(probabilities))
            row["prediction"] = predictor.labels[index]
            for label, p in zip(predictor.labels, probabilities):
                row[label] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


st.title("WESAD HRV/EDA Stress Classifier")
st.caption(
    "Upload synchronized wrist BVP (64 Hz) and EDA (4 Hz) recordings from the same "
    "session to classify Baseline / Stress / Amusement over sliding 30s windows."
)

with st.sidebar:
    st.header("Model info")
    config = json.loads((BUNDLE_DIR / "model_config.json").read_text(encoding="utf-8"))
    st.write(f"**Type:** {config['model_type']}")
    st.write(f"**Classes:** {', '.join(config['class_names'])}")
    st.write(f"**Window:** {config['preprocessing']['window_sec']}s, stride {config['preprocessing']['stride_sec']}s")
    st.write(f"**Trained subjects:** {len(config['trained_subjects'])}")
    st.markdown("---")
    st.markdown(
        "**Input file format**\n\n"
        "One numeric sample per line (CSV first column). Either a plain list of "
        "values, or an Empatica E4-style export whose first line is a start "
        "timestamp and second line is the sample rate."
    )

col1, col2 = st.columns(2)
with col1:
    bvp_file = st.file_uploader("BVP file (64 Hz)", type=["csv", "txt"])
with col2:
    eda_file = st.file_uploader("EDA file (4 Hz)", type=["csv", "txt"])

if bvp_file is not None and eda_file is not None:
    predictor = load_predictor()
    bvp_fs = predictor.preprocessing["bvp_fs"]
    eda_fs = predictor.preprocessing["eda_fs"]

    bvp = parse_signal(bvp_file, bvp_fs)
    eda = parse_signal(eda_file, eda_fs)

    st.write(
        f"Loaded {len(bvp)} BVP samples ({len(bvp) / bvp_fs:.1f}s) and "
        f"{len(eda)} EDA samples ({len(eda) / eda_fs:.1f}s)."
    )

    min_needed = predictor.preprocessing["window_sec"]
    if len(bvp) / bvp_fs < min_needed or len(eda) / eda_fs < min_needed:
        st.error(f"Need at least {min_needed}s of both signals to form one window.")
    else:
        with st.spinner("Running inference..."):
            timeline = run_timeline(predictor, bvp, eda)

        valid = timeline[timeline["status"] == "ok"]
        if valid.empty:
            st.warning(
                "No window passed quality checks. Common causes: gaps, flat "
                "signal, or insufficient warmup. Statuses seen: "
                + ", ".join(sorted(timeline["status"].unique()))
            )
        else:
            latest = valid.iloc[-1]
            st.subheader(f"Latest window prediction: {latest['prediction']}")
            prob_cols = st.columns(len(predictor.labels))
            for c, label in zip(prob_cols, predictor.labels):
                c.metric(label, f"{latest[label]:.1%}")

            st.subheader("Class probability over time")
            st.line_chart(valid.set_index("window_end_sec")[predictor.labels])

            st.subheader("Per-window predictions")
            display_cols = ["window_end_sec", "prediction"] + predictor.labels
            st.dataframe(valid[display_cols], use_container_width=True)

            skipped = timeline[timeline["status"] != "ok"]
            if not skipped.empty:
                with st.expander(f"{len(skipped)} window(s) skipped (quality gate)"):
                    st.dataframe(skipped[["window_end_sec", "status"]], use_container_width=True)
else:
    st.info("Upload both a BVP file and an EDA file to run the classifier.")

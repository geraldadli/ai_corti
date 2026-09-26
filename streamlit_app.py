"""AI Corti — a focused BVP + EDA recording demo using the bundled predictor."""
from hashlib import sha256
from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from stress_inference import StressPredictor, prepare_recording, pulse_quality, window_at

BUNDLE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BUNDLE_DIR / "assets" / "style.css"
TRAILER_PATH = BUNDLE_DIR / "assets" / "corti-trailer.mp4"
BATCH_SIZE = 64  # Same batch size the notebook uses to score held-out windows.
UPLOAD, SAMPLE = "Upload recording", "Sample demo"
INK, MUTED = "#183d36", "#60746b"
COLORS = {"No stress": "#23796a", "Stress": "#b35a3b"}
DESCRIPTIONS = {
    "No stress": "This window most closely matches a non-stress pattern.",
    "Stress": "This window most closely matches the model’s stress pattern.",
}
# One set of stress-score bands drives both the gauge and the reaction card. The label
# itself flips at 50%, so the middle band straddles that line instead of naming a class.
BANDS = [(0.40, "calm", "#a8d1c5"), (0.60, "borderline", "#efd38a"), (1.00, "stress", "#e39a80")]
REACTIONS = {
    "calm": ("🌿", "No thoughts, head empty.",
             "Peak baseline zen. This window sits comfortably on the no-stress side."),
    "borderline": ("☕", "Could go either way… but that’s none of my business.",
                   "This window sits close to the model’s 50% line. Its neighbours in Session details tell you more."),
    "stress": ("🔥", "This is fine. (It might not be.)",
               "This window closely matches the stress pattern. A five-minute break never hurt anyone. 🧘"),
}
ANALYSIS_SPINNER = "Analyzing physiological waves… checking whether you’re writing C code or watching a horror movie…"


@st.cache_resource(show_spinner="Waking up AI Corti…")
def load_predictor() -> StressPredictor:
    return StressPredictor(BUNDLE_DIR)


def parse_signal(uploaded_file, expected_fs: float) -> tuple[np.ndarray, float | None]:
    """Keep missing samples in place; accept an optional header or E4 metadata."""
    lines = uploaded_file.getvalue().decode("utf-8-sig").splitlines()
    values = [line.split(",")[0].strip() for line in lines]
    if not values:
        raise ValueError("The recording is empty.")

    def number(value):
        try:
            return float(value)
        except ValueError:
            return np.nan

    epoch = None
    first = number(values[0])
    if np.isfinite(first) and first > 1e8 and len(values) >= 2:
        if number(values[1]) != expected_fs:
            raise ValueError(f"This recording must use {expected_fs:g} Hz.")
        epoch, values = first, values[2:]
    elif values[0] and not np.isfinite(first) and values[0].lower() not in {"nan", "inf", "+inf", "-inf"}:
        values = values[1:]
    samples = np.asarray([number(v) for v in values], dtype=float)
    if not np.isfinite(samples).any():
        raise ValueError("The recording contains no numeric samples.")
    return samples, epoch


def demo_signals(config):
    """Illustrative sensor waveforms only; no assigned ground-truth label."""
    seconds = config["window_sec"] + config["warmup_sec"] + 10
    b = np.arange(int(seconds * config["bvp_fs"])) / config["bvp_fs"]
    e = np.arange(int(seconds * config["eda_fs"])) / config["eda_fs"]
    rng = np.random.default_rng(42)
    bvp = 100 + 18 * np.sin(2 * np.pi * 1.15 * b + .12 * np.sin(.6 * b))
    bvp += 4 * np.sin(2 * np.pi * 2.3 * b) + rng.normal(0, .5, len(b))
    eda = 2 + .08 * np.sin(.25 * e) + .12 * np.exp(-((e - 30) / 4) ** 2)
    return bvp, eda


def binary_scores(probabilities, labels):
    """Merge probability mass before choosing a label; keep the trained model intact."""
    scores = dict(zip(labels, probabilities))
    if set(scores) != {"Baseline", "Stress", "Amusement"}:
        raise ValueError("Expected the Baseline / Stress / Amusement model bundle.")
    return {"No stress": float(scores["Baseline"] + scores["Amusement"]),
            "Stress": float(scores["Stress"])}


def stress_band(score: float) -> str:
    return next(name for upper, name, _ in BANDS if score <= upper)


def run_timeline(predictor: StressPredictor, bvp: np.ndarray, eda: np.ndarray) -> pd.DataFrame:
    config = predictor.preprocessing
    prepared = prepare_recording(bvp, eda, config)
    duration = min(len(bvp) / config["bvp_fs"], len(eda) / config["eda_fs"])
    ends = np.arange(config["window_sec"], duration + 1e-9, config["stride_sec"])
    rows, usable = [], []
    for end in ends:
        inputs, status = window_at(prepared, float(end), config)
        row = {"window_end_sec": float(end), "status": status}
        if inputs is not None:
            # Advisory beat check, as in StressPredictor.predict_latest: it flags, never rejects.
            a, b = int((end - config["window_sec"]) * config["bvp_fs"]), int(end * config["bvp_fs"])
            row["pulse"] = "regular" if pulse_quality(prepared["bvp"][a:b], config) else "irregular"
            usable.append((row, inputs))
        rows.append(row)
    for start in range(0, len(usable), BATCH_SIZE):
        batch = usable[start:start + BATCH_SIZE]
        stacked = {name: np.stack([inputs[name] for _, inputs in batch]) for name in ("bvp", "eda")}
        probabilities = predictor.model(stacked, training=False).numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("The model returned invalid scores. Check the model bundle.")
        for (row, _), window_probabilities in zip(batch, probabilities):
            scores = binary_scores(window_probabilities, predictor.labels)
            row["prediction"] = max(scores, key=scores.get)
            row.update(scores)
    return pd.DataFrame(rows, columns=["window_end_sec", "status", "prediction", *COLORS, "pulse"])


def input_fingerprint() -> str | None:
    """Identify the selected input from widget state, so layout can be chosen before widgets render."""
    if st.session_state.get("input_source") == SAMPLE:
        return "synthetic-demo"
    bvp_file, eda_file = st.session_state.get("bvp_upload"), st.session_state.get("eda_upload")
    if bvp_file is None or eda_file is None:
        return None
    return sha256(bvp_file.getvalue() + b"\0" + eda_file.getvalue()).hexdigest()


def current_result() -> pd.DataFrame | None:
    """The saved timeline, only while it belongs to the inputs selected right now."""
    saved = st.session_state.get("corti_analysis", {})
    timeline, fingerprint = saved.get("timeline"), input_fingerprint()
    if timeline is None or fingerprint is None or saved.get("fingerprint") != fingerprint:
        return None
    if not {"No stress", "pulse"} <= set(timeline.columns):
        return None  # Discard a result retained from before the UI update.
    return timeline


def analyze(bvp: np.ndarray, eda: np.ndarray) -> None:
    """Score every window, keep the result for this exact input, then reveal step 02."""
    st.session_state.pop("corti_analysis", None)
    try:
        with st.spinner(ANALYSIS_SPINNER):
            timeline = run_timeline(load_predictor(), bvp, eda)
    except Exception as exc:
        st.error("We couldn’t analyze this recording. Check that your model files belong to the same bundle.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return
    st.session_state.corti_analysis = {"fingerprint": input_fingerprint(), "timeline": timeline}
    st.rerun()  # Lay the page out again so both panels reflect the new result.


def render_style():
    if STYLE_PATH.is_file():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_nav():
    st.markdown('<div class="corti-nav"><div class="brand"><span class="brand-mark" aria-hidden="true">∿</span>'
                'Corti <small>AI</small></div></div>', unsafe_allow_html=True)


def render_hero():
    """1 · The motto, then what the project is in plain words."""
    st.markdown('<div class="hero"><div class="eyebrow">Meet AI Corti</div>'
                '<h1>Your signals.<br><em>Stress, understood.</em></h1>'
                '<p>AI Corti reads two signals from a wrist sensor, your pulse (BVP) and skin response (EDA), '
                'in 30-second windows. A deep-learning model trained on the WESAD lab study tells you whether '
                'each window looks like stress.</p>'
                '<div class="tags"><span class="tag">Pulse + skin response</span><span class="tag">30-second windows</span>'
                '<span class="tag">Research prototype</span></div></div>', unsafe_allow_html=True)


def render_trailer():
    if not TRAILER_PATH.is_file():
        st.caption("Trailer unavailable.")
        return
    player = st.container()  # Filled after the toggle, so the control sits below the video.
    loop = st.toggle("Loop trailer", value=True, key="loop_trailer")
    player.video(str(TRAILER_PATH), loop=loop)


def render_intro(config):
    """2 · Learn first: the trailer beside the layered pipeline."""
    video, pipeline = st.columns([1.2, 1], gap="large")
    with video:
        st.markdown('<div class="eyebrow">Watch the intro · 1 min</div>', unsafe_allow_html=True)
        render_trailer()
    with pipeline:
        st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
        render_how_it_works(config)


def render_pulse_badge(timeline: pd.DataFrame):
    valid = timeline[timeline["status"] == "ok"]
    if valid.empty:
        return
    irregular = int((valid["pulse"] == "irregular").sum())
    if irregular:
        state, text = "irregular", f"Irregular in {irregular} of {len(valid)} windows"
    else:
        state, text = "regular", "Regular"
    st.markdown(f'<div class="pulse-badge {state}"><span class="pulse-dot"></span>Pulse quality · {text}</div>',
                unsafe_allow_html=True)


def render_uploader(preprocessing, minimum: float, timeline: pd.DataFrame | None):
    """Step 01: choose a recording or the sample, then analyze it."""
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="step">01</span> Add your signals</div>'
                    '<div class="panel-sub">Start with a recording, or explore with a sample.</div>',
                    unsafe_allow_html=True)
        source = st.radio("Input source", [UPLOAD, SAMPLE], horizontal=True, label_visibility="collapsed",
                          key="input_source")
        bvp, eda, error = None, None, None
        if source == UPLOAD:
            bvp_file = st.file_uploader(f"Pulse / BVP · {preprocessing['bvp_fs']:g} Hz", type=["csv", "txt"], key="bvp_upload")
            eda_file = st.file_uploader(f"Skin response / EDA · {preprocessing['eda_fs']:g} Hz", type=["csv", "txt"], key="eda_upload")
            st.caption(f"Same session and start time · at least {minimum:g} seconds each")
            with st.expander("Recording format"):
                st.caption("One sample per row in the first CSV column, with an optional header. Empatica E4 files may include a start timestamp and sampling rate in their first two rows. Keep missing samples as blank rows or NaN. EDA must be in microsiemens (µS).")
            if bvp_file is not None and eda_file is not None:
                try:
                    bvp, bvp_start = parse_signal(bvp_file, preprocessing["bvp_fs"])
                    eda, eda_start = parse_signal(eda_file, preprocessing["eda_fs"])
                    if (bvp_start is None) != (eda_start is None) or (bvp_start is not None and abs(bvp_start - eda_start) > 1e-6):
                        raise ValueError("The two files must share the same start time and metadata format. Export synchronized recordings.")
                    if min(len(bvp) / preprocessing["bvp_fs"], len(eda) / preprocessing["eda_fs"]) < minimum:
                        raise ValueError(f"Add at least {minimum:g} seconds of both signals, including filter warmup.")
                except (ValueError, UnicodeError) as exc:
                    error = str(exc)
        else:
            st.markdown('<div class="demo-note"><strong>A first look at Corti.</strong>Explore the experience with generated pulse and skin-response signals. The model analyzes them just like an upload.<br><br>Illustrative demo · not a person’s measurement or an accuracy test.</div>', unsafe_allow_html=True)
            bvp, eda = demo_signals(preprocessing)
        if error:
            st.warning(error)
        if timeline is not None:
            render_pulse_badge(timeline)
        label = "Analyze sample" if source == SAMPLE else "Analyze recording"
        if st.button(label, type="primary", use_container_width=True, disabled=bvp is None or error is not None):
            analyze(bvp, eda)


def render_speedometer_gauge(score: float):
    """Semi-circle dial for one window's stress score; the needle points at 0–100%."""
    import plotly.graph_objects as go  # Local import: only the results panel needs Plotly.

    percent = 100 * float(np.clip(score, 0, 1))
    active = stress_band(score)
    figure = go.Figure()
    lower = 0.
    for upper, name, color in BANDS:
        # Angles run clockwise from 0% on the left to 100% on the right.
        figure.add_trace(go.Barpolar(r=[.34], base=[.66], theta=[90 * (lower + upper)], width=[180 * (upper - lower)],
                                     marker=dict(color=color, line=dict(color="#ffffff", width=3)),
                                     opacity=1 if name == active else .35, hoverinfo="skip"))
        lower = upper
    figure.add_trace(go.Scatterpolar(r=[0, .86], theta=[1.8 * percent] * 2, mode="lines",
                                     line=dict(color=INK, width=4), hoverinfo="skip"))
    figure.add_trace(go.Scatterpolar(r=[0], theta=[0], mode="markers", marker=dict(color=INK, size=14),
                                     hoverinfo="skip"))
    figure.update_layout(
        height=200, margin=dict(l=28, r=28, t=6, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=600, easing="cubic-in-out"),
        polar=dict(sector=[0, 180], bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], visible=False),
                   angularaxis=dict(rotation=180, direction="clockwise", tickvals=[0, 72, 108, 180],
                                    ticktext=["0", "40", "60", "100"], ticks="", showgrid=False, showline=False,
                                    tickfont=dict(size=11, color=MUTED))))
    st.plotly_chart(figure, use_container_width=True, theme=None, config={"displayModeBar": False, "staticPlot": True})
    st.markdown(f'<div class="gauge-readout"><b style="color:{INK}">{percent:.0f}%</b><span>stress score</span></div>',
                unsafe_allow_html=True)


def render_meme_reaction(score: float):
    band = stress_band(score)
    emoji, title, body = REACTIONS[band]
    st.markdown(f'<div class="reaction {band}"><span class="reaction-emoji" aria-hidden="true">{emoji}</span>'
                f'<div><b>{escape(title)}</b><p>{escape(body)}</p></div></div>', unsafe_allow_html=True)


def render_results(timeline: pd.DataFrame | None, preprocessing, source: str):
    """Step 02: a waiting state until analysis, then the latest usable window, the whole recording and an export."""
    labels = list(COLORS)
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="step">02</span> Your insight</div>'
                    '<div class="panel-sub">A snapshot of the latest usable signal window.</div>',
                    unsafe_allow_html=True)
        if timeline is None:
            hint = ("Select Analyze sample to explore your first Corti insight." if source == SAMPLE
                    else "Add both signals and select Analyze to see your result here.")
            st.markdown(f'<div class="empty-result"><div class="empty-ring"><span aria-hidden="true">∿</span></div>'
                        f'<h3>A little clarity awaits.</h3><p>{hint}</p></div>', unsafe_allow_html=True)
            return
        valid = timeline[timeline["status"] == "ok"]
        if valid.empty:
            st.warning("No usable window yet. Check sensor contact, missing samples and recording length, then try again.")
        else:
            latest = valid.iloc[-1]
            label, score = str(latest["prediction"]), float(np.clip(latest["Stress"], 0, 1))
            end = latest["window_end_sec"]
            context = "SAMPLE DEMO" if source == SAMPLE else "RECORDING RESULT"
            if latest["pulse"] == "irregular":
                signal_check = '<div class="signal-note" role="note"><span aria-hidden="true">⚠</span><div><b>Irregular pulse in this window</b>The pulse didn’t look like a steady heartbeat, often from movement or a loose sensor. Read this result with extra caution.</div></div>'
            else:
                signal_check = '<div class="signal-ok"><span aria-hidden="true">✓</span> Regular pulse signal</div>'
            st.markdown(f'<div class="result"><div class="eyebrow">{context} · {end-preprocessing["window_sec"]:g}–{end:g} SEC</div><div class="result-label" style="color:{COLORS.get(label, INK)}">{escape(label)}</div><div class="result-copy">{escape(DESCRIPTIONS.get(label, "The model’s closest matching pattern."))}</div>{signal_check}</div>', unsafe_allow_html=True)
            render_speedometer_gauge(score)
            st.caption("Relative model score · not certainty or cortisol levels")
            render_meme_reaction(score)
            stressed = int((valid["prediction"] == "Stress").sum())
            windows = f"{len(valid)} usable window{'s' if len(valid) != 1 else ''}"
            st.markdown(f'<div class="session-summary"><b>Whole recording</b> · Stress in {stressed} of {windows} ({stressed / len(valid):.0%})</div>', unsafe_allow_html=True)
            if timeline.iloc[-1]["status"] != "ok":
                st.caption("The end of this recording did not pass signal checks. Showing the most recent usable window.")
        st.download_button("Download timeline (CSV)", timeline.to_csv(index=False), file_name="corti-timeline.csv",
                           mime="text/csv", use_container_width=True)
        with st.expander("Session details"):
            if not valid.empty:
                st.line_chart(valid.set_index("window_end_sec")[labels], color=[COLORS[name] for name in labels], x_label="Seconds into recording", y_label="Model score")
            skipped = int((timeline["status"] != "ok").sum())
            irregular = int((valid["pulse"] == "irregular").sum())
            st.caption(f"{len(valid)} usable windows · {irregular} with an irregular pulse · {skipped} skipped (including warmup)")
            st.dataframe(timeline, hide_index=True, use_container_width=True)


def pipeline_diagram(steps):
    """An ordered stack of layers; the connectors and deepening tints are decorative CSS."""
    blocks = ''.join(f'<li><span class="phase">{escape(phase)}</span>'
                     f'<div><strong>{escape(title)}</strong><p>{escape(detail)}</p></div></li>'
                     for phase, title, detail in steps)
    st.markdown(f'<ol class="pipeline">{blocks}</ol>', unsafe_allow_html=True)


def render_how_it_works(config):
    c = config["preprocessing"]
    model_type = config.get("model_type", "Unspecified model")
    model_name = {"residual_1d_cnn_bilstm_attention": "CNN–BiLSTM + attention"}.get(
        model_type, model_type.replace("_", " "))
    prediction, notebook, architecture = st.tabs(["Prediction flow", "Notebook · OSEMN", "Neural architecture"])
    with prediction:
        pipeline_diagram([
            ("01", "Capture signals", "Live Arduino or BVP + EDA upload"),
            ("02", "Clean", "Filter noise and check signal quality"),
            ("03", "Create windows", f"{c['window_sec']:g}s windows, every {c['stride_sec']:g}s"),
            ("04", "Prepare inputs", "Normalize BVP + derive EDA channels"),
            ("05", "Run AI model", "Predict the three trained states"),
            ("06", "Show result", "Stress or No stress"),
        ])
        st.caption("No stress = Baseline + Amusement scores.")
    with notebook:
        pipeline_diagram([
            ("O · Obtain", "Collect", "WESAD signals + ground-truth labels"),
            ("S · Scrub", "Prepare", "Align, filter and label windows"),
            ("E · Explore", "Inspect", "Signal quality + class balance"),
            ("M · Model", "Train & tune", "Test on held-out participants"),
            ("N · Interpret", "Evaluate", "Precision, recall + confusion matrix"),
        ])
        st.caption("Then: select settings → refit on all usable participants → export the app bundle.")
    with architecture:
        pipeline_diagram([
            ("Input", "BVP + EDA", f"{c['window_sec']:g}s of pulse ({c['bvp_fs']:g} Hz) + skin response ({c['eda_fs']:g} Hz)"),
            ("Features", "Residual 1D-CNNs", "One tower per sensor learns local patterns"),
            ("Fusion", "Concatenate", "Both sensors aligned in time"),
            ("Sequence", "BiLSTM + attention", "Focus on the most telling moments"),
            ("Output", "Three class scores", "Baseline · Stress · Amusement"),
        ])
        participants = len(config.get("trained_subjects", []))
        trained = f", trained on {participants} WESAD participants" if participants else ""
        st.caption(f"Deployed model: {model_name}{trained}.")


def render_footer():
    st.markdown('<div class="footer"><b>AI Corti · Stress, understood.</b><span>Research prototype. Estimates stress patterns from BVP + EDA; does not measure cortisol or provide a diagnosis.</span></div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="AI Corti · Stress, understood", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")
    render_style()
    render_nav()
    render_hero()
    try:
        config = json.loads((BUNDLE_DIR / "model_config.json").read_text(encoding="utf-8"))
        preprocessing = config["preprocessing"]
        minimum = preprocessing["window_sec"] + preprocessing["warmup_sec"]
    except (OSError, ValueError, KeyError) as exc:
        st.error("AI Corti could not read its model configuration. Restore model_config.json from your bundle.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return

    render_intro(config)
    st.markdown('<div class="section-title"><h2>Your Corti check-in</h2>'
                '<span>A recording. An analysis. An insight.</span></div>', unsafe_allow_html=True)
    mode = st.radio("Capture mode", ["Recording", "Live Arduino"], horizontal=True, key="capture_mode")
    if mode == "Live Arduino":
        from live_capture import show_live
        show_live(load_predictor, binary_scores, preprocessing)
        render_footer()
        return
    st.session_state.pop("corti_live", None)
    timeline = current_result()
    # 3 · Both steps stay in view: the insight panel waits beside the input until there is a result.
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        render_uploader(preprocessing, minimum, timeline)
    with right:
        render_results(timeline, preprocessing, st.session_state.get("input_source", UPLOAD))
    render_footer()


if __name__ == "__main__":
    main()

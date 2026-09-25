"""AI Corti — a focused BVP + EDA recording demo using the bundled predictor."""
from hashlib import sha256
from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from stress_inference import StressPredictor, prepare_recording, window_at

BUNDLE_DIR = Path(__file__).resolve().parent
COLORS = {"No stress": "#23796a", "Stress": "#b35a3b"}
DESCRIPTIONS = {
    "No stress": "This window most closely matches a non-stress pattern.",
    "Stress": "This window most closely matches the model’s stress pattern.",
}

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {color-scheme:light;}
.stApp {background:#f6f8f5;color:#183d36;font-family:'DM Sans',sans-serif;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stDecoration"] {background:#1b4e40;}
.stAppViewMain {margin-inline:0!important;width:100%;min-width:0;}
.main .block-container {width:100%;max-width:1280px;margin-inline:auto!important;padding:1.5rem clamp(1rem,3vw,2.5rem) 2rem;min-width:0;}
h1,h2,h3,p,label,button {font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"] {display:none;}
video {display:block;width:100%;max-width:746px;height:auto;aspect-ratio:16/9;margin-inline:auto;border-radius:18px;}
.corti-nav {display:flex;align-items:center;justify-content:space-between;padding:0 0 24px;border-bottom:1px solid #dfe7df;gap:16px;}
.brand {display:flex;align-items:center;gap:10px;font-size:25px;font-weight:700;letter-spacing:-1px;}
.brand-mark {background:#1b4e40;color:#d6ecb6;border-radius:13px;width:39px;height:39px;display:grid;place-items:center;font-size:27px;font-weight:400;}
.brand small {font-size:11px;font-weight:700;letter-spacing:1px;background:#e5eadf;border-radius:5px;padding:4px 5px;margin-left:3px;}
.nav-note {font-size:12px;color:#567168;display:flex;align-items:center;gap:8px;}
.dot {display:inline-block;width:7px;height:7px;background:#43846b;border-radius:50%;}
.hero {display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);align-items:center;gap:24px;padding:26px 0 24px;}
.eyebrow {font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#597466;margin-bottom:14px;}
.hero h1 {font-family:'Manrope',sans-serif;font-size:clamp(36px,4.5vw,54px);font-weight:600;line-height:1.13;letter-spacing:-2.7px;color:#183d36;margin:0 0 18px;padding:0;}
.hero h1 em {font-style:normal;color:#5e886b;}
.hero p {font-size:15px;line-height:1.7;color:#60746b;max-width:365px;margin:0;}
.tags {display:flex;gap:8px;margin-top:21px;flex-wrap:wrap;}
.tag {font-size:11px;padding:6px 10px;border:1px solid #dce5db;border-radius:20px;color:#486554;background:#fafbf8;}
.sensor-art {height:262px;border-radius:28px;background:radial-gradient(ellipse at 50% 40%,#eef6e8 0,#e0ecdf 65%,#dae8dc 100%);position:relative;overflow:hidden;display:grid;place-items:center;border:1px solid #dce6d9;}
.art-label {position:absolute;top:20px;left:22px;color:#55725e;font-size:10px;letter-spacing:1.8px;font-weight:600;}
.art-bottom {position:absolute;bottom:18px;left:22px;right:22px;display:flex;justify-content:space-between;font-size:10px;color:#55725e;}
.orbit {position:absolute;border:1px solid #527a5625;width:178px;height:178px;border-radius:50%;}
.orbit.outer {width:245px;height:245px;border-style:dashed;animation:orbit 60s linear infinite;}
.pulse-core {width:114px;height:114px;border-radius:50%;background:#1b4e40;box-shadow:0 10px 35px #214f4324;display:grid;place-items:center;animation:breathe 4s ease-in-out infinite;z-index:1;}
.pulse-core span {color:#dbedb6;font-size:55px;line-height:1;}
.signal-line {position:absolute;height:44px;width:100%;top:111px;opacity:.6;background:linear-gradient(90deg,transparent,#739c79,transparent);clip-path:polygon(0 49%,15% 49%,19% 35%,23% 65%,27% 49%,34% 49%,38% 10%,42% 90%,46% 49%,56% 49%,60% 35%,64% 65%,68% 49%,77% 49%,81% 10%,85% 90%,89% 49%,100% 49%,100% 53%,88% 53%,85% 98%,81% 18%,78% 53%,67% 53%,64% 73%,60% 43%,57% 53%,45% 53%,42% 98%,38% 18%,35% 53%,26% 53%,23% 73%,19% 43%,16% 53%,0 53%);}
.sensor-pill {position:absolute;z-index:2;padding:8px 12px;border-radius:12px;background:#ffffffdf;box-shadow:0 4px 20px #3159470a;font-size:10px;color:#446c56;}
.sensor-pill.bvp {left:25px;bottom:58px;}.sensor-pill.eda {right:24px;top:66px;}
.section-title {display:flex;align-items:center;justify-content:space-between;margin:0 0 17px;}
.section-title h2 {font-size:20px;letter-spacing:-.6px;font-weight:600;margin:0;padding:0;}
.section-title span {font-size:11px;color:#738178;}
[data-testid="stVerticalBlockBorderWrapper"]:has(>div>[data-testid="stVerticalBlock"]>div>[data-testid="stMarkdown"] .panel-heading) {border-color:#dde5dc!important;border-radius:20px!important;background:#fff;}
.panel-heading {display:flex;align-items:center;gap:10px;margin-bottom:5px;font-size:17px;font-weight:600;letter-spacing:-.3px;}
.step {background:#eef2e9;border-radius:8px;font-size:10px;letter-spacing:0;width:27px;height:27px;display:inline-grid;place-items:center;color:#4d7054;}
.panel-sub {color:#738078;font-size:12px;margin:7px 0 18px;line-height:1.6;}
.stRadio label p {font-size:12px!important;color:#355246;}
[data-testid="stFileUploader"] label p {font-size:12px;font-weight:600;color:#38584b;}
[data-testid="stFileUploaderDropzone"] {background:#f8faf6;border:1px dashed #cbd9ca;border-radius:12px;padding:15px;}
[data-testid="stFileUploaderDropzone"] small {font-size:10px;}
.stButton button {border-radius:10px;min-height:43px;font-size:13px;font-weight:600;}
.stButton button[kind="primary"] {background:#1b4e40;border-color:#1b4e40;color:#fff;}
.stButton button[kind="primary"]:hover {background:#2b6753;border-color:#2b6753;}
.stButton button:focus-visible {outline:3px solid #85b39a;outline-offset:3px;}
.stButton button:disabled {background:#e2e9df!important;border-color:#e2e9df!important;color:#6a7e6d!important;}
.stCaption p {font-size:11px;color:#718074;}
.empty-result {text-align:center;padding:38px 12px 26px;}
.empty-ring {height:74px;width:74px;border:1px solid #d5e1d3;border-radius:50%;display:grid;place-items:center;margin:0 auto 22px;background:#f3f7ef;}
.empty-ring span {font-size:29px;color:#719277;animation:breathe 4s ease-in-out infinite;}
.empty-result h3 {font-size:19px;font-weight:500;color:#385747;margin:0 0 8px;}
.empty-result p {color:#7a877d;font-size:12px;line-height:1.7;max-width:220px;margin:auto;}
.result {padding:18px 2px 5px;animation:appear .45s ease-out;}
.result-label {font-size:39px;letter-spacing:-1.8px;font-weight:600;margin:6px 0 8px;line-height:1.2;}
.result-copy {font-size:12px;line-height:1.7;color:#687a70;margin-bottom:23px;}
.score {margin:12px 0;}.score-caption {display:flex;justify-content:space-between;font-size:12px;color:#536b5b;margin-bottom:6px;}
.score-track {height:5px;background:#edf1e9;border-radius:9px;overflow:hidden;}.score-fill {height:100%;border-radius:9px;}
.demo-note {padding:18px;border:1px solid #dfe8d8;border-radius:12px;background:#f7faf2;color:#597050;font-size:12px;line-height:1.7;margin:13px 0;}
.demo-note strong {display:block;color:#345638;font-size:15px;margin-bottom:7px;}
.footer {margin-top:25px;padding-top:18px;border-top:1px solid #dfe7df;display:flex;justify-content:space-between;gap:14px;font-size:10px;color:#788579;line-height:1.6;}
.footer b {color:#47634f;font-weight:600;white-space:nowrap;}
.pipeline {display:grid;grid-auto-flow:column;grid-auto-columns:minmax(0,1fr);gap:22px;list-style:none;padding:0!important;margin:18px 0 22px!important;}
.pipeline li {margin:0!important;position:relative;background:#fff;border:1px solid #dce5db;border-radius:14px;padding:15px 12px;min-width:0;}
.pipeline li:not(:last-child)::after {content:'→';position:absolute;right:-19px;top:42%;color:#60816b;font-size:19px;}
.pipeline .phase {display:block;font-size:10px;color:#607866;text-transform:uppercase;letter-spacing:1px;margin-bottom:9px;}
.pipeline strong {display:block;font-size:13px;color:#234c3e;line-height:1.4;margin-bottom:8px;}
.pipeline p {font-size:12px;line-height:1.6;color:#5b7061;margin:0;}
@media(max-width:900px){.pipeline{grid-auto-flow:row;grid-template-columns:1fr;gap:22px}.pipeline li:not(:last-child)::after{content:'↓';top:auto;bottom:-23px;left:50%;right:auto}}
@keyframes breathe {0%,100%{transform:scale(1)}50%{transform:scale(1.065)}}
@keyframes orbit {to{transform:rotate(360deg)}}
@keyframes appear {from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;}}
@media(max-width:640px){.block-container{padding:1.4rem 1rem}.hero{grid-template-columns:1fr;gap:24px;padding:28px 0}.hero h1{font-size:39px}.sensor-art{height:212px}.signal-line{top:87px}.nav-note{font-size:10px}.footer{flex-direction:column}.section-title span{display:none}}
</style>"""


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


def pipeline_diagram(steps):
    """An ordered, responsive block diagram; arrows are decorative CSS."""
    blocks = ''.join(f'<li><span class="phase">{escape(phase)}</span>'
                     f'<strong>{escape(title)}</strong><p>{escape(detail)}</p></li>'
                     for phase, title, detail in steps)
    st.markdown(f'<ol class="pipeline">{blocks}</ol>', unsafe_allow_html=True)


def show_pipeline(config):
    c = config["preprocessing"]
    model_type = config.get("model_type", "Unspecified model")
    model_name = {"residual_1d_cnn_bilstm_attention": "CNN–BiLSTM + attention"}.get(
        model_type, model_type.replace("_", " "))
    with st.expander("How AI Corti works"):
        prediction, notebook, architecture = st.tabs(["Prediction flow", "Notebook · OSEMN", "Neural architecture"])
        with prediction:
            pipeline_diagram([
                ("01", "Upload signals", "Pulse (BVP) + skin response (EDA)"),
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
            st.markdown("**Notebook v5 · CNN–Transformer**")
            pipeline_diagram([
                ("Input", "BVP + EDA", "Two normalized signal branches"),
                ("Features", "Multiscale CNNs", "Learn local signal patterns"),
                ("Fusion", "Cross-modal attention", "Combine both sensors"),
                ("Sequence", "Transformer", "Learn patterns across the window"),
                ("Output", "Three class scores", "Baseline · Stress · Amusement"),
            ])
            st.caption(f"Currently deployed: {model_name}. The diagram shows notebook v5.")


def run_timeline(predictor: StressPredictor, bvp: np.ndarray, eda: np.ndarray) -> pd.DataFrame:
    config = predictor.preprocessing
    prepared = prepare_recording(bvp, eda, config)
    duration = min(len(bvp) / config["bvp_fs"], len(eda) / config["eda_fs"])
    ends = np.arange(config["window_sec"], duration + 1e-9, config["stride_sec"])
    rows = []
    for end in ends:
        inputs, status = window_at(prepared, float(end), config)
        row = {"window_end_sec": float(end), "status": status}
        if inputs is not None:
            probabilities = predictor.model({k: v[None, ...] for k, v in inputs.items()}, training=False).numpy()[0]
            if not np.isfinite(probabilities).all():
                raise ValueError("The model returned invalid scores. Check the model bundle.")
            scores = binary_scores(probabilities, predictor.labels)
            row["prediction"] = max(scores, key=scores.get)
            row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows, columns=["window_end_sec", "status", "prediction", *COLORS])


def main():
    st.set_page_config(page_title="AI Corti · Stress, understood", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown('''<div class="corti-nav"><div class="brand"><span class="brand-mark" aria-hidden="true">∿</span>Corti <small>AI</small></div><div class="nav-note"><span class="dot"></span> Your signals. A little more clarity.</div></div>''', unsafe_allow_html=True)
    st.markdown("### Meet AI Corti")
    loop_trailer = st.checkbox("Loop trailer", value=True)
    trailer = BUNDLE_DIR / "assets" / "corti-trailer.mp4"
    if trailer.is_file():
        st.video(str(trailer), loop=loop_trailer)
    else:
        st.caption("Trailer unavailable.")
    st.markdown('''<div class="hero"><div><div class="eyebrow">Meet AI Corti</div><h1>Your signals.<br><em>Stress, understood.</em></h1><p>Turn your pulse and skin response into a simple picture of your stress patterns.</p><div class="tags"><span class="tag">Two signals, one insight</span><span class="tag">Powered by deep learning</span></div></div><div class="sensor-art" role="img" aria-label="Animated pulse illustration, not a live sensor reading"><div class="art-label">IN SYNC WITH YOU</div><div class="orbit outer"></div><div class="orbit"></div><div class="signal-line"></div><div class="pulse-core"><span aria-hidden="true">∿</span></div><div class="sensor-pill bvp">↝ &nbsp; Pulse · BVP</div><div class="sensor-pill eda">◌ &nbsp; Skin response · EDA</div><div class="art-bottom"><span>BODY SIGNALS, MADE SIMPLE</span><span>Signal illustration</span></div></div></div>
<div class="section-title"><h2>Your Corti check-in</h2><span>A recording. An analysis. An insight.</span></div>''', unsafe_allow_html=True)
    try:
        config = json.loads((BUNDLE_DIR / "model_config.json").read_text(encoding="utf-8"))
        preprocessing = config["preprocessing"]
        labels = list(COLORS)
        minimum = preprocessing["window_sec"] + preprocessing["warmup_sec"]
    except (OSError, ValueError, KeyError) as exc:
        st.error("AI Corti could not read its model configuration. Restore model_config.json from your bundle.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return

    left, right = st.columns([1.1, 1], gap="medium")
    fingerprint, bvp, eda, error = None, None, None, None
    with left, st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="step">01</span> Add your signals</div><div class="panel-sub">Start with a recording, or explore with a sample.</div>', unsafe_allow_html=True)
        source = st.radio("Input source", ["Upload recording", "Sample demo"], horizontal=True, label_visibility="collapsed")
        if source == "Upload recording":
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
                    fingerprint = sha256(bvp_file.getvalue() + b"\0" + eda_file.getvalue()).hexdigest()
                except (ValueError, UnicodeError) as exc:
                    error = str(exc)
        else:
            st.markdown('<div class="demo-note"><strong>A first look at Corti.</strong>Explore the experience with generated pulse and skin-response signals. The model analyzes them just like an upload.<br><br>Illustrative demo · not a person’s measurement or an accuracy test.</div>', unsafe_allow_html=True)
            bvp, eda = demo_signals(preprocessing)
            fingerprint = "synthetic-demo"
        if error:
            st.warning(error)
        analyze = st.button("Analyze sample" if source == "Sample demo" else "Analyze recording", type="primary", use_container_width=True, disabled=fingerprint is None)
        if analyze:
            st.session_state.pop("corti_analysis", None)
            try:
                with st.spinner("Reading your signals…"):
                    timeline = run_timeline(load_predictor(), bvp, eda)
                    st.session_state.corti_analysis = {"fingerprint": fingerprint, "timeline": timeline}
            except Exception as exc:
                st.error("We couldn’t analyze this recording. Check that your model files belong to the same bundle.")
                with st.expander("Technical details"):
                    st.code(str(exc))

    saved = st.session_state.get("corti_analysis", {})
    timeline = saved.get("timeline") if fingerprint is not None and saved.get("fingerprint") == fingerprint else None
    if timeline is not None and "No stress" not in timeline.columns:
        timeline = None  # Discard a three-class result retained from before the UI update.
    with right, st.container(border=True):
        st.markdown('<div class="panel-heading"><span class="step">02</span> Your insight</div><div class="panel-sub">A snapshot of the latest usable signal window.</div>', unsafe_allow_html=True)
        if timeline is None:
            hint = "Select Analyze sample to explore your first Corti insight." if source == "Sample demo" else "Add both signals and select Analyze to see your result here."
            st.markdown(f'<div class="empty-result"><div class="empty-ring"><span aria-hidden="true">∿</span></div><h3>A little clarity awaits.</h3><p>{hint}</p></div>', unsafe_allow_html=True)
        else:
            valid = timeline[timeline["status"] == "ok"]
            if valid.empty:
                st.warning("No usable window yet. Check sensor contact, missing samples and recording length, then try again.")
            else:
                latest = valid.iloc[-1]
                label = str(latest["prediction"])
                color = COLORS.get(label, "#23796a")
                end = latest["window_end_sec"]
                context = "SAMPLE DEMO" if source == "Sample demo" else "RECORDING RESULT"
                st.markdown(f'<div class="result"><div class="eyebrow">{context} · {end-preprocessing["window_sec"]:g}–{end:g} SEC</div><div class="result-label" style="color:{color}">{escape(label)}</div><div class="result-copy">{escape(DESCRIPTIONS.get(label, "The model’s closest matching pattern."))}</div></div>', unsafe_allow_html=True)
                for name in labels:
                    score = float(np.clip(latest[name], 0, 1))
                    st.markdown(f'<div class="score"><div class="score-caption"><span>{escape(name)}</span><span>{score:.0%}</span></div><div class="score-track"><div class="score-fill" style="width:{score*100:.2f}%;background:{COLORS.get(name, "#23796a")}"></div></div></div>', unsafe_allow_html=True)
                st.caption("Relative model scores · not certainty or cortisol levels")
                if timeline.iloc[-1]["status"] != "ok":
                    st.caption("The end of this recording did not pass signal checks. Showing the most recent usable window.")
            with st.expander("Session details"):
                if not valid.empty:
                    st.line_chart(valid.set_index("window_end_sec")[labels], color=[COLORS.get(name, "#23796a") for name in labels], x_label="Seconds into recording", y_label="Model score")
                skipped = int((timeline["status"] != "ok").sum())
                st.caption(f"{len(valid)} usable windows · {skipped} skipped (including warmup)")
                st.dataframe(timeline, hide_index=True, use_container_width=True)

    show_pipeline(config)
    st.markdown('<div class="footer"><b>AI Corti · Stress, understood.</b><span>Research prototype. Estimates stress patterns from BVP + EDA; does not measure cortisol or provide a diagnosis.</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

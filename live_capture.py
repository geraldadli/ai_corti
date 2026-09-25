"""USB session validation and live UI. The trained preprocessing stays unchanged."""
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import streamlit as st
from streamlit.components.v1 import declare_component

MAX_SECONDS = 600
serial_capture = declare_component("corti_serial", path=str(Path(__file__).with_name("serial_component")))


def decode_session(packet):
    """Validate the wire contract and keep dropped samples on the original timeline."""
    if not isinstance(packet, dict) or packet.get("protocol") != 1:
        raise ValueError("Upload the Corti Arduino sketch first.")
    fs = packet.get("bvp_fs")
    if fs not in (64, 100) or packet.get("eda_fs") != 4:
        raise ValueError("Expected pulse at 64/100 Hz and GSR at 4 Hz.")
    adc_max = packet.get("adc_max")
    if type(adc_max) is not int or not 255 <= adc_max <= 262143:
        raise ValueError("Invalid device ADC range.")
    rows = packet.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_SECONDS * fs:
        raise ValueError("No samples yet, or the ten-minute session limit was exceeded.")
    try:
        data = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed sensor samples.") from exc
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("Expected sequence, microseconds, pulse, EDA and raw GSR.")
    seq, time_us = data[:, 0], data[:, 1]
    if (not np.isfinite(data[:, :2]).all() or np.any(seq != np.floor(seq))
            or seq[0] != 0 or np.any(np.diff(seq) <= 0) or seq[-1] >= MAX_SECONDS * fs):
        raise ValueError("Device restarted or sample order is invalid. Reconnect for a fresh session.")
    if time_us[0] < 0 or time_us[-1] > (MAX_SECONDS + 1) * 1e6:
        raise ValueError("Invalid device clock.")
    if len(seq) > 1:
        nominal = np.diff(seq) * (1e6 / fs)
        elapsed = np.diff(time_us)
        if np.any(np.abs(elapsed - nominal) > 3000 + .02 * nominal):
            raise ValueError("Irregular sample timing. Check the sketch and USB connection.")
        if seq[-1] >= fs and abs((time_us[-1] - time_us[0]) / (seq[-1] * 1e6 / fs) - 1) > .02:
            raise ValueError("Device clock differs from the required sampling rate.")
    seq = seq.astype(int)
    bvp = np.full(seq[-1] + 1, np.nan)
    pulse = data[:, 2].copy()
    pulse[(pulse <= 0) | (pulse >= adc_max)] = np.nan  # ADC rail clipping is not a valid waveform.
    bvp[seq] = pulse
    divider = fs // 4
    eda = np.full(seq[-1] // divider + 1, np.nan)
    at_eda = seq % divider == 0
    conductance = data[at_eda, 3].copy()
    raw_gsr = data[at_eda, 4]
    conductance[(raw_gsr <= 0) | (raw_gsr >= 1023) | ~np.isfinite(raw_gsr)] = np.nan
    eda[seq[at_eda] // divider] = conductance
    if fs == 100:
        # MAX30102 has no native 64 Hz mode. Polyphase FIR prevents aliasing.
        # Withhold 0.25 s so the symmetric FIR never uses padded future samples.
        bvp = resample_poly(bvp, 16, 25)[:max(0, int(len(bvp) * .64) - 16)]
    return bvp, eda


@st.fragment(run_every=1)
def show_live(load_predictor, binary_scores, config):
    """The browser owns USB; each Streamlit session owns its latest result."""
    left, right = st.columns([1.1, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown("#### Connect your sensors")
        st.caption("Chrome or Edge on your PC · close Arduino Serial Monitor first")
        packet = serial_capture(key="corti_usb", default=None)
        st.caption("First result in about 45 seconds · refreshes every 5 seconds · sessions up to 10 minutes")
        with st.expander("Arduino setup"):
            st.markdown("1. Upload `arduino/corti_capture/corti_capture.ino` using Arduino IDE.\n"
                        "2. Set your ESP32 pins and verify the Grove GSR calibration.\n"
                        "3. Close Serial Monitor / Plotter, then select **Connect Arduino**.\n"
                        "4. Keep both sensors still and this browser tab visible.")
            st.caption("Live readings are sent to this app’s server for inference. No recordings are written to disk.")
    state = st.session_state.setdefault("corti_live", {})
    token = (packet.get("session"), packet.get("revision")) if isinstance(packet, dict) else None
    if token is not None and token != state.get("token"):
        state.clear()
        state.update(token=token, received=monotonic())
        try:
            if packet.get("state") != "streaming":
                state["message"] = str(packet.get("message", "Connect Arduino to begin."))[:200]
            else:
                bvp, eda = decode_session(packet)
                state["seconds"] = len(bvp) / 64
                state["preview"] = pd.DataFrame({"Pulse · ADC": bvp[-640:]})
                if packet.get("calibrated") is not True:
                    state["message"] = "Receiving signals. GSR calibration to µS is required before prediction."
                elif state["seconds"] < config["window_sec"] + config["warmup_sec"]:
                    state["message"] = f"Collecting signals · {state['seconds']:.0f} / 40 usable seconds"
                else:
                    # ponytail: replay full history, capped at 10 minutes; add stateful filters for longer sessions.
                    prediction = load_predictor().predict_latest(bvp, eda)
                    if prediction["status"] != "ok":
                        state["message"] = "Signal not ready. Keep both sensors in contact and still."
                    elif prediction.get("quality_warning"):
                        state["message"] = "Pulse quality is low. Adjust the pulse sensor and keep still."
                    else:
                        probabilities = prediction["probabilities"]
                        values = np.asarray(list(probabilities.values()), dtype=float)
                        if (not np.isfinite(values).all() or np.any(values < 0)
                                or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=1e-4)):
                            raise ValueError("The model returned invalid scores.")
                        state["scores"] = binary_scores(values, list(probabilities))
                        state["end"] = prediction["window_end_sec"]
        except Exception as exc:
            state["message"] = "Could not read this session. Check the connection and sensor setup."
            state["error"] = str(exc)
    with right, st.container(border=True):
        st.markdown("#### Your live insight")
        fresh = monotonic() - state.get("received", -1e9) <= 12
        scores = state.get("scores") if fresh else None
        if scores:
            label = max(scores, key=scores.get)
            st.metric("Latest window", label)
            st.caption(f"{state['end'] - config['window_sec']:g}–{state['end']:g} seconds · relative model scores")
            for name, value in scores.items():
                st.progress(float(value), text=f"{name} · {value:.0%}")
        else:
            message = state.get("message", "Connect Arduino to begin.")
            if not fresh and packet and packet.get("state") == "streaming":
                message = "Live data paused. Check the device and keep the browser tab visible."
            st.info(message)
        if state.get("error"):
            with st.expander("Connection details"):
                st.code(state["error"])
        if fresh and "preview" in state:
            with st.expander("Live pulse waveform"):
                st.line_chart(state["preview"], height=140)
        st.caption("Hardware prototype · this model has not been validated on your sensors.")

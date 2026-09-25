"""Run with python check_live.py; no hardware is needed for protocol/inference checks."""
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from live_capture import decode_session
from stress_inference import StressPredictor
from streamlit_app import demo_signals, binary_scores


def check():
    folder = Path(__file__).parent
    config = json.loads((folder / "model_config.json").read_text())["preprocessing"]
    pulse, eda = demo_signals(config)
    packet = dict(protocol=1, bvp_fs=64, eda_fs=4, adc_max=1023, calibrated=True,
                  session="test-session", revision=1, state="streaming")
    packet["rows"] = [[i, i * 15625, float(v), float(eda[i // 16]) if i % 16 == 0 else None,
                       500 if i % 16 == 0 else None] for i, v in enumerate(pulse)]
    b, e = decode_session(packet)
    assert np.array_equal(b, pulse) and np.array_equal(e, eda)
    dropped = deepcopy(packet)
    del dropped["rows"][32]
    b, e = decode_session(dropped)
    assert len(b) == len(pulse) and np.isnan(b[32]) and np.isnan(e[2])
    clipped = deepcopy(packet)
    clipped["rows"][0][2] = 1023
    clipped["rows"][0][4] = 0
    b, e = decode_session(clipped)
    assert np.isnan(b[0]) and np.isnan(e[0])
    for change in (lambda p: p.update(bvp_fs=50),
                   lambda p: p["rows"][1].__setitem__(0, 0),
                   lambda p: p["rows"][1].__setitem__(1, 999999),
                   lambda p: p["rows"][-1].__setitem__(0, 38400),
                   lambda p: p.update(rows=[[0, 0, 5]])):
        bad = deepcopy(packet)
        change(bad)
        try:
            decode_session(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid live session accepted")
    predictor = StressPredictor(folder)
    result = predictor.predict_latest(*decode_session(packet))
    assert result["status"] == "ok" and not result["quality_warning"]
    scores = binary_scores(list(result["probabilities"].values()), list(result["probabilities"]))
    assert set(scores) == {"Stress", "No stress"} and np.isclose(sum(scores.values()), 1)
    # Actual MAX30102 transport rate: preserve the waveform while converting 100 -> 64 Hz.
    t = np.arange(5000) / 100
    raw = 50000 + 3000 * np.sin(2 * np.pi * 1.2 * t)
    max_packet = dict(packet, bvp_fs=100, adc_max=262143,
                      rows=[[i, i * 10000, float(v), float(eda[i // 25]) if i % 25 == 0 else None,
                             500 if i % 25 == 0 else None] for i, v in enumerate(raw)])
    b, e = decode_session(max_packet)
    assert len(b) == 3184 and len(e) == 200
    expected = 50000 + 3000 * np.sin(2 * np.pi * 1.2 * np.arange(len(b)) / 64)
    assert np.max(np.abs(b[64:-64] - expected[64:-64])) < 10
    longer = deepcopy(max_packet)
    longer["rows"].extend([[i, i * 10000, float(50000 + 3000 * np.sin(2*np.pi*1.2*i/100)),
                            2 if i % 25 == 0 else None, 500 if i % 25 == 0 else None]
                           for i in range(5000, 5100)])
    assert np.allclose(b, decode_session(longer)[0][:len(b)])  # no future padding changes old samples
    result_max = predictor.predict_latest(b, e)
    assert result_max["status"] == "ok" and not result_max["quality_warning"]

    app = AppTest.from_file(str(folder / "streamlit_app.py"), default_timeout=90).run()
    app.radio[0].set_value("Live Arduino").run()
    with patch("live_capture.serial_capture", return_value=packet):
        app.run()
    assert not app.exception and app.metric[0].value in scores
    app.session_state["corti_live"]["received"] -= 20
    with patch("live_capture.serial_capture", return_value=packet):
        app.run()
    assert not app.metric and any("paused" in item.value for item in app.info)
    packet["revision"] += 1
    packet["calibrated"] = False
    with patch("live_capture.serial_capture", return_value=packet):
        app.run()
    assert not app.metric and any("calibration" in item.value for item in app.info)
    packet["revision"] += 1
    packet.update(state="disconnected", message="Disconnected", rows=[])
    with patch("live_capture.serial_capture", return_value=packet):
        app.run()
    assert not app.metric and any("Disconnected" in item.value for item in app.info)
    print("PASS: live timing, dropped samples, clipping, model inference, calibration gate, stale/disconnected UI")


if __name__ == "__main__":
    check()

"""Run: python check_ui.py (uses Streamlit's built-in app test runner)."""
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from streamlit.testing.v1 import AppTest

import streamlit_app as ui


def check():
    scores = ui.binary_scores([.35, .40, .25], ["Baseline", "Stress", "Amusement"])
    assert np.isclose(scores["No stress"], .60) and max(scores, key=scores.get) == "No stress"
    scores = ui.binary_scores([.70, .10, .20], ["Stress", "Amusement", "Baseline"])
    assert max(scores, key=scores.get) == "Stress" and np.isclose(sum(scores.values()), 1)
    samples, epoch = ui.parse_signal(BytesIO(b"1700000000\n64\n1\n\n3\n"), 64)
    assert epoch == 1700000000 and len(samples) == 3 and np.isnan(samples[1])
    samples, epoch = ui.parse_signal(BytesIO(b"BVP\n1\n2\n3\n"), 64)
    assert epoch is None and samples.tolist() == [1, 2, 3]
    for content in (b"", b"header\nbad\n", b"1700000000\n4\n1\n"):
        try:
            ui.parse_signal(BytesIO(content), 64)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")

    path = Path(__file__).with_name("streamlit_app.py")
    app = AppTest.from_file(str(path), default_timeout=90).run()
    assert not app.exception and app.button[0].disabled
    assert [tab.label for tab in app.tabs] == ["Prediction flow", "Notebook · OSEMN", "Neural architecture"]
    diagrams = [item.value for item in app.markdown if '<ol class="pipeline"' in item.value]
    assert len(diagrams) == 3 and all('<li>' in diagram for diagram in diagrams)
    assert any('No stress = Baseline + Amusement' in item.value for item in app.caption)
    assert any('Currently deployed:' in item.value for item in app.caption)
    assert "tensorflow" not in sys.modules, "Landing page eagerly loaded TensorFlow"
    app.radio[0].set_value("Sample demo").run()
    assert not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and not app.error, [x.value for x in app.error]
    result = app.session_state.corti_analysis["timeline"]
    valid = result[result.status == "ok"]
    assert not valid.empty and set(valid.prediction) <= set(ui.COLORS)
    assert "Baseline" not in result.columns and "Amusement" not in result.columns
    assert np.allclose(valid[list(ui.COLORS)].sum(axis=1), 1, atol=1e-5)
    assert any("SAMPLE DEMO" in x.value for x in app.markdown)
    app.radio[0].set_value("Upload recording").run()
    assert not app.exception and app.button[0].disabled
    assert any("A little clarity awaits" in x.value for x in app.markdown)
    assert not any("result-label" in x.value for x in app.markdown if "<style>" not in x.value)

    # The actual saved model is exercised above; test rejection with flat signals too.
    import json
    config = json.loads(path.with_name("model_config.json").read_text())["preprocessing"]
    bvp, eda = ui.demo_signals(config)
    predictor = ui.load_predictor()
    rejected = ui.run_timeline(predictor, np.zeros_like(bvp), np.zeros_like(eda))
    assert not (rejected.status == "ok").any()
    print("PASS: binary aggregation, initial UI, lazy loading, sample inference, stale-result removal, parser and signal rejection")


if __name__ == "__main__":
    check()

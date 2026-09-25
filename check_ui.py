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
    assert [ui.stress_band(s) for s in (0, .4, .41, .6, .61, 1)] == [
        "calm", "calm", "borderline", "borderline", "stress", "stress"]
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
    assert path.with_name("assets").joinpath("corti-trailer.mp4").is_file()
    assert path.with_name("assets").joinpath("style.css").is_file()
    assert app.get("video")[0].proto.loop
    app.toggle[0].set_value(False).run()
    assert not app.exception and not app.get("video")[0].proto.loop
    assert [tab.label for tab in app.tabs] == ["Prediction flow", "Notebook · OSEMN", "Neural architecture"]
    diagrams = [item.value for item in app.markdown if '<ol class="pipeline"' in item.value]
    assert len(diagrams) == 3 and all('<li>' in diagram for diagram in diagrams)
    assert any('No stress = Baseline + Amusement' in item.value for item in app.caption)
    assert any('Deployed model:' in item.value for item in app.caption)
    assert "tensorflow" not in sys.modules, "Landing page eagerly loaded TensorFlow"
<<<<<<< Updated upstream
    app.radio[1].set_value("Sample demo").run()
    assert not app.button[0].disabled
=======
    page = [x.value for x in app.markdown if "<style>" not in x.value]
    assert not any("result-label" in x for x in page), "Result shown before analysis"
    assert any("A little clarity awaits" in x for x in page), "Insight panel should wait beside the input"
    # Learn first: motto, then trailer beside the pipeline, then both check-in panels.
    order = [next(i for i, x in enumerate(page) if marker in x)
             for marker in ('class="hero"', "Watch the intro", '<ol class="pipeline"', "Your Corti check-in",
                            '<span class="step">01', '<span class="step">02')]
    assert order == sorted(order), order
    app.radio[0].set_value("Sample demo").run()
    assert not app.button[0].disabled and app.button[0].label == "Analyze sample"
    assert any("Select Analyze sample" in x.value for x in app.markdown)
>>>>>>> Stashed changes
    app.button[0].click().run()
    assert not app.exception and not app.error, [x.value for x in app.error]
    result = app.session_state.corti_analysis["timeline"]
    valid = result[result.status == "ok"]
    assert not valid.empty and set(valid.prediction) <= set(ui.COLORS)
    assert "Baseline" not in result.columns and "Amusement" not in result.columns
    assert np.allclose(valid[list(ui.COLORS)].sum(axis=1), 1, atol=1e-5)
<<<<<<< Updated upstream
    assert any("SAMPLE DEMO" in x.value for x in app.markdown)
    app.radio[1].set_value("Upload recording").run()
    assert not app.exception and app.button[0].disabled
    assert any("A little clarity awaits" in x.value for x in app.markdown)
    assert not any("result-label" in x.value for x in app.markdown if "<style>" not in x.value)
    app.radio[0].set_value("Live Arduino").run()
    assert not app.exception and len(app.get("component_instance")) == 1
    assert any("Connect Arduino to begin" in x.value for x in app.info)
=======
    assert set(valid.pulse) <= {"regular", "irregular"}
    page = [x.value for x in app.markdown if "<style>" not in x.value]
    for text in ("SAMPLE DEMO", "result-label", "Whole recording", "Pulse quality", 'class="reaction'):
        assert any(text in x for x in page), f"Insight panel is missing {text!r}"
    assert not any("A little clarity awaits" in x for x in page)
    assert len(app.get("plotly_chart")) == 1
    app.radio[0].set_value("Upload recording").run()
    assert not app.exception and app.button[0].disabled and app.button[0].label == "Analyze recording"
    page = [x.value for x in app.markdown if "<style>" not in x.value]
    assert not any("result-label" in x for x in page) and any("A little clarity awaits" in x for x in page)
>>>>>>> Stashed changes

    # The actual saved model is exercised above; test rejection with flat signals too.
    import json
    config = json.loads(path.with_name("model_config.json").read_text())["preprocessing"]
    bvp, eda = ui.demo_signals(config)
    predictor = ui.load_predictor()
    rejected = ui.run_timeline(predictor, np.zeros_like(bvp), np.zeros_like(eda))
    assert not (rejected.status == "ok").any()
    # Noise passes the missing/flat gates, so the advisory pulse check must flag it.
    noisy = ui.run_timeline(predictor, np.random.default_rng(0).normal(0, 50, len(bvp)), eda)
    scored = noisy[noisy.status == "ok"]
    assert not scored.empty and (scored.pulse == "irregular").all()
    # Batched app scores must match the reference single-window predictor.
    reference = predictor.predict_latest(bvp, eda)
    timeline = ui.run_timeline(predictor, bvp, eda)
    assert np.isclose(timeline.iloc[-1]["Stress"], reference["probabilities"]["Stress"], atol=1e-5)
    print("PASS: binary aggregation, stress bands, initial UI, lazy loading, waiting insight panel, sample "
          "inference, gauge, stale-result removal, parser, signal rejection, pulse advisory and batched scoring")


if __name__ == "__main__":
    check()

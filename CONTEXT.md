# AI Corti: project context for another AI agent

## 1. What it is
**AI Corti** is a research-prototype web app that estimates stress from wrist-worn physiological signals. It takes **BVP** (blood volume pulse, 64 Hz) and **EDA** (electrodermal activity, 4 Hz, in µS) from an Empatica E4-style device. It runs a pretrained Keras model over sliding windows and reports **"Stress" or "No stress"**.

- The model was trained on the public **WESAD** dataset (15 participants, S2–S17 without S12).
- The repo has three parts: a training and evaluation notebook, a shared inference module, and a Streamlit demo UI.
- The GitHub-facing README calls the project "WESAD Stress Classifier". The app brand is "AI Corti · Stress, understood."
- It is not a medical tool. The UI states it does not measure cortisol or diagnose anything.

## 2. Repository layout (13 tracked files)

| Path | Role |
|---|---|
| `streamlit_app.py` (~345 lines) | The whole UI: CSS block, file parsing, timeline inference, results panel, "How it works" tabs. |
| `stress_inference.py` (~165 lines) | Preprocessing, window gating and `StressPredictor`. The notebook embeds an identical copy as the string `INFERENCE_SOURCE`. |
| `stress_model.keras` (1.2 MB) | Final model, refit on all 15 participants. It has no independent test score. |
| `model_config.json` | Preprocessing hyperparameters, class names, input shapes, trained subjects, training policy, library versions, SHA-256 of the model and inference source. |
| `hrv-based-stress-classification-using-the-wesad.ipynb` | 21 cells, structured as OSEMN: Obtain, Scrub, Explore, Model, iNterpret. It runs the LOSO evaluation, the final refit and the export of `stress_app_bundle.zip`. |
| `results.html` (480 KB) | Standalone report with base64-embedded plots: LOSO metrics, per-class and per-subject tables, ablation, caveats. |
| `check_ui.py` | The only test. It is a script, not pytest. It uses Streamlit's `AppTest` plus direct calls into the app functions. |
| `assets/corti-trailer.mp4` | Promo video shown at the top of the app. |
| `.streamlit/config.toml` | Light theme, minimal toolbar. |
| `.devcontainer/devcontainer.json` | Codespaces/VS Code container (Python 3.11). It auto-runs `streamlit run streamlit_app.py`. |
| `requirements.txt`, `.python-version` | `streamlit==1.38.0`, `tensorflow-cpu==2.18.0`, `numpy==1.26.4`, `scipy==1.15.3`, `pandas==2.2.3`. Python is 3.11. |
| `README.md` | Overview, usage and results. |

There is no `src/` layout, no package, no `.gitignore`, no CI and no pytest suite.

## 3. Model and data pipeline

**Model:** `residual_1d_cnn_bilstm_attention`.
- Two parallel residual 1D-CNN towers, one for BVP and one for EDA.
- The towers are fused, then a BiLSTM and attention pooling feed a dense classifier.
- Output: 3 classes (`Baseline`, `Stress`, `Amusement`).
- Inputs: BVP `(1920, 1)` (30 s at 64 Hz) and EDA `(120, 4)` (30 s at 4 Hz, four derived channels).

**Preprocessing** (`stress_inference.py`, `DEFAULT_PREPROCESSING`, mirrored in `model_config.json`):
1. `clean_channel` sets non-finite and out-of-bounds values to NaN. EDA bounds are 0–100 µS. Gaps up to 0.5 s are forward-filled.
2. `filter_channel` applies a 4th-order Butterworth filter: bandpass 0.5–4 Hz for BVP, lowpass 1 Hz for EDA. It runs per contiguous finite run and discards a 10 s warm-up.
3. `prepare_recording` builds a causal EMA tonic component (10 s time constant), a phasic residual and a log-derivative of EDA.
4. `window_at(prepared, end_sec)` cuts a 30 s window, stepped every 5 s. It rejects windows with these statuses:
   - `insufficient_history`
   - `*_insufficient_samples`
   - `*_gap_or_warmup`
   - `*_missing_samples` (observed fraction below 0.95)
   - `*_flat`
   - `eda_component_gap`
5. `represent_window` applies per-window robust (median/MAD) scaling and clipping at ±5 to BVP. For EDA it builds `log_level`, `relative_log_level`, `relative_phasic` and `log_derivative`.
6. `pulse_quality` is a peak/IBI regularity check. Its strict form is disabled (`strict_pulse_gate: false`), so it is **advisory only** and yields a `quality_warning`.

**Inference API:** `StressPredictor(bundle_dir).predict_latest(bvp, eda)` scores only the latest window. The app instead runs `run_timeline`, which batches all windows (batch size 64).

## 4. How the Streamlit app works (`streamlit_app.py`)
- **Landing page:** trailer video (with a "Loop trailer" checkbox), hero section and an inline `<style>` block (~80 lines of custom CSS).
- **Landing page does not import TensorFlow.** `load_predictor()` is `@st.cache_resource` and lazy, and `check_ui.py` asserts this.
- **Input, step 01:** either "Upload recording" (two CSV/TXT files) or "Sample demo" (synthetic sine-based signals, seed 42).
- **`parse_signal`:**
  - Reads the first CSV column, with an optional text header.
  - Accepts Empatica-style files: epoch timestamp on row 1 and sample rate on row 2. It rejects a sample-rate mismatch.
  - Keeps blank or NaN rows as NaN.
  - Requires both files to share the same start epoch and at least 40 s of data (window plus warm-up).
- **`run_timeline`:** computes every window, tags each as `regular` or `irregular` pulse, batch-predicts, and returns a DataFrame with columns `window_end_sec, status, prediction, No stress, Stress, pulse`.
- **`binary_scores`:** merges the 3-class output into two classes. `No stress = Baseline + Amusement`, `Stress = Stress`. The model itself is unchanged.
- **Results, step 02:** shows the latest usable window (label, score bars, a regular/irregular pulse badge) and a whole-recording summary ("Stress in X of N windows"). A "Session details" expander holds a line chart and the full timeline table. Results are cached in `st.session_state.corti_analysis`, keyed by a SHA-256 fingerprint of the uploaded files.
- **"How AI Corti works" expander:** three tabs (Prediction flow, Notebook OSEMN, Neural architecture) built with a custom `pipeline_diagram` HTML/CSS helper. It reads values from `model_config.json`.

## 5. Evaluation results (from `results.html` and the README)
Leave-One-Subject-Out CV, 15 participants, 6,360 windows, 3-class task:
- Subject-mean macro F1: **0.600** (95% CI 0.536–0.662). The project's >80% target was **not met**.
- Accuracy: 0.673. ROC-AUC: 0.830.
- Per-class F1:
  - Baseline: 0.699
  - Stress: **0.856** (recall 0.892)
  - Amusement: **0.305** (the weak class)
- Sensor ablation: shuffling BVP costs 0.206 macro F1, so the model leans on BVP more than EDA.
- Wide per-subject variance: S11 has F1 0.749, S13 has 0.319.
- The old strict beat gate would have kept only 2,666 of 6,360 windows. It was relaxed to advisory-only.

## 6. Git state and history (as of this snapshot)
- Branch `main`. At the time of writing, `streamlit_app.py` and `check_ui.py` had **uncommitted changes**. They add the pulse-regularity badge, whole-recording summary, batched scoring, the fixed architecture-tab text and matching test assertions.
- Recent commits: trailer video, an AI Corti UI polish with binary results, a devcontainer, and README updates. Earlier commits are largely "Fix it once and for all" style.

## 7. Known gaps and improvement leads
These were observed by reading the code; the app and tests were not run when this was written.

1. **The binary framing has no binary evaluation.** The UI presents Stress vs No stress, but every reported metric is 3-class. The Amusement class (F1 0.305) is folded into "No stress". A binary LOSO report (stress vs non-stress: precision, recall, ROC-AUC, confusion matrix) would justify or challenge the UI.
2. **The README is stale relative to the UI.**
   - It describes a 3-class Baseline/Stress/Amusement demo, but the app shows binary output.
   - Its file table omits `check_ui.py`, `assets/`, `.streamlit/` and `.devcontainer/`.
   - It doesn't say how to run `check_ui.py`.
3. **Integrity hashes are recorded but never checked.** `model_sha256` and `inference_source_sha256` in `model_config.json` are not verified by `StressPredictor`. Editing `stress_inference.py` breaks the "byte-for-byte identical to the notebook" guarantee without any error. The README and the module docstring both describe this design principle.
4. **Duplicated source of truth.** `stress_inference.py` and the notebook's `INFERENCE_SOURCE` string must be kept in sync by hand.
5. **Test coverage is thin.** `check_ui.py` is one long function. It has no pytest structure and no unit tests for `stress_inference.py` (gap handling, filter warm-up, flat-signal or missing-sample rejection, `pulse_quality`). There is no CI.
6. **UI code is monolithic.** `streamlit_app.py` embeds about 80 lines of CSS and large inline HTML f-strings. It could be split into `styles`, `parsing`, `components` and `pipeline` modules. Some HTML is built from f-strings, so it should stay `escape()`d.
7. **Performance:** every "Analyze" click re-runs `prepare_recording` on the full recording, and `predict_latest` replays the whole session on each call. The README says a real-time TFLite, browser or BLE runtime is not included.
8. **Dependencies and packaging:**
   - `requirements.txt` pins `tensorflow-cpu`, while the README says "TensorFlow 2.18 CPU". That is consistent, but the devcontainer also runs an unpinned `pip install streamlit` afterwards, which can override the pinned version.
   - The devcontainer runs Streamlit with CORS and XSRF protection **disabled**. That is fine for Codespaces preview, but flag it if this is ever deployed.
   - There is no `.gitignore`.
9. **Data and format limits.** Files must be plain CSV/TXT in a sensor-native format. There is no direct E4 ZIP/`.csv` folder ingestion, and no handling of timestamp-based resampling or mismatched lengths beyond the start-epoch check.
10. **Model limits.** The model was trained and tested only on WESAD lab data, with a wrist device and a single dataset. There is no calibration (the UI already warns that scores are "not certainty"). Ideas to try are probability calibration, subject-adaptive baselines, a per-window confidence or abstain option, and cross-dataset validation.
11. **Product and UX gaps.** There is no export (CSV or PDF) of the timeline. The results panel shows only the latest window in detail. There is no per-window timeline visualization beyond the line chart.

## 8. How to run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
python check_ui.py
```

Retraining requires the WESAD dataset (not in the repo) and running the notebook end to end.

## 9. Ground rules for an agent working on this
- Don't change preprocessing constants or `stress_inference.py` without also changing the notebook's embedded copy. If either changes, the model and `model_config.json` (`format_version` 2, `preprocessing.version` 2) may need to be regenerated.
- Keep the disclaimer and the "research prototype, not diagnosis" language.
- Keep the landing page free of a TensorFlow import. `check_ui.py` asserts this.
- The metrics quoted above come from the LOSO procedure. The shipped model has no independent test score.

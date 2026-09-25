# WESAD Stress Classifier

A hybrid 1D-CNN + BiLSTM + attention model that classifies wrist-worn BVP (blood volume pulse) and EDA (electrodermal activity) signals into **Baseline / Stress / Amusement**, trained and evaluated on the [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) dataset, with a Streamlit demo app for interactive inference.

## Overview

- **Input:** synchronized wrist BVP (64 Hz) and EDA (4 Hz) recordings, Empatica E4-style.
- **App output:** Stress / No stress for each 30-second window, stepped every 5 seconds. The trained model retains three classes; the app sums Baseline + Amusement probabilities into No stress.
- **Model:** parallel residual 1D-CNN towers for BVP and EDA → fusion → BiLSTM → temporal attention pooling → dense classifier.
- **Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation across 15 participants — subject-mean macro F1 **0.600** (95% CI 0.536–0.662), accuracy **0.673**, ROC-AUC **0.830**. See `results.html` for full details.
- **Design principle:** the exact preprocessing code used in training is exported byte-for-byte into `stress_inference.py`, so the served app can never silently drift from what was evaluated.

## Repository structure

| File | Purpose |
|---|---|
| `hrv-based-stress-classification-using-the-wesad.ipynb` | End-to-end training pipeline: data ingestion, signal preprocessing, model definition, LOSO evaluation, final refit, and app bundle export. |
| `stress_inference.py` | Shared preprocessing + inference module (identical code embedded in the notebook). Loads the bundled model and runs sliding-window predictions. |
| `streamlit_app.py` | Web UI — live ESP32 capture, uploaded BVP/EDA recordings, and sample demo. |
| `live_capture.py`, `serial_component/` | Browser USB capture, sample validation/resampling, and live predictions. |
| `arduino/corti_capture/corti_capture.ino` | ESP32 + MAX30102 + Grove GSR firmware. |
| `LIVE_CAPTURE.md` | Wiring, calibration and live capture instructions. |
| `stress_model.keras` | Trained model weights (final refit on all usable participants). |
| `model_config.json` | Preprocessing parameters, class names, input shapes, trained-subject list, and integrity hashes tying the config to the model/inference code. |
| `results.html` | Standalone results report: metrics, sensor-ablation analysis, and known limitations. |
| `requirements.txt` | Python dependencies for running the app/inference. |

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

(Python 3.11, TensorFlow 2.18 CPU, NumPy, SciPy, Pandas, Streamlit — see `requirements.txt` for pinned versions.)

### Run the demo app

```bash
streamlit run streamlit_app.py
```

Upload a BVP file (64 Hz) and an EDA file (4 Hz) from the **same recording session**. Files can be a plain list of numeric samples (one per line) or an Empatica E4-style export (timestamp header + sample rate + samples). The app reports a prediction per 30s window and flags windows that fail quality checks (signal gaps, flat segments, insufficient pulse quality, etc.).

### Use the inference module directly

For live sensors, follow [the ESP32 setup guide](LIVE_CAPTURE.md), then choose **Live Arduino** in the app. Grove GSR calibration is required before inference; the first result normally arrives after about 45 seconds.

```python
from stress_inference import StressPredictor

predictor = StressPredictor(".")  # bundle directory containing model + config
result = predictor.predict_latest(bvp_array, eda_array)
print(result["prediction"], result["probabilities"])
```

Inputs must be raw, uniformly sampled arrays from the same recording origin (missing samples as `NaN`, not deleted), supplied from session start so filter history is preserved. Do not pre-normalize.

### Retrain / reproduce results

The notebook expects the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (`S2`–`S17` folders with synchronized `.pkl` files), available on request from the dataset authors. Open `hrv-based-stress-classification-using-the-wesad.ipynb`, point it at the dataset root, and run end-to-end to reproduce the LOSO evaluation and re-export `stress_app_bundle.zip` (model + config + inference code + results report).

## Results summary

| Metric | Value |
|---|---|
| Subject-mean macro F1 | 0.600 (95% CI 0.536–0.662) |
| Subject-mean accuracy | 0.673 |
| Subject-mean ROC-AUC | 0.830 |

The model relies more heavily on the BVP (pulse) channel than EDA (sensor-permutation ablation: mean macro-F1 drop of 0.206 when BVP is shuffled). Full metrics, per-subject breakdowns, and quality-gate retention statistics are in `results.html`.

## Limitations

- Evaluated only on WESAD laboratory recordings; not validated on other PPG/GSR devices, free-living activity, or clinical populations.
- Does not establish or validate any specific physiological biomarker (RMSSD, LF/HF, etc.) — the model learns directly from waveform patterns.
- The >80% macro F1 target set for this project was not met.
- `stress_inference.py` is a reference Python/TensorFlow backend, not an optimized real-time (BLE/TFLite/browser) runtime.
- Not intended for clinical diagnosis or medical use.

## License

MIT

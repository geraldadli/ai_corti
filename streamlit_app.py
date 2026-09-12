import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error("TensorFlow is not installed in this environment. "
             "Install requirements for this folder with `pip install -r requirements.txt`, "
             "or use a Python 3.11 environment where TensorFlow 2.14+ is supported.")
    st.stop()

from stress_inference import StressPredictor


st.set_page_config(page_title='WESAD Stress Inference', page_icon='🫀', layout='wide')

st.title('WESAD Stress Classifier')
st.write('Raw BVP (64 Hz) + raw EDA (4 Hz) only. The predictor handles preprocessing, windowing and feature creation.')

st.caption(f'Runtime Python: {sys.version.split()[0]} | TensorFlow: {tf.__version__}')


@st.cache_resource(show_spinner=False)
def load_predictor(bundle_dir: Path) -> StressPredictor:
    return StressPredictor(str(bundle_dir))


def parse_json_array(raw: str, name: str):
    try:
        data = json.loads(raw.strip())
    except Exception as exc:
        raise ValueError(f'Invalid JSON for {name}: {exc}') from exc

    if isinstance(data, dict):
        if 'values' not in data:
            raise ValueError(f'{name} JSON must be a list or {{"values": [...]}}.')
        data = data['values']

    if not isinstance(data, list):
        raise ValueError(f'{name} must be a list of numbers.')

    arr = np.array(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D list.')
    if len(arr) == 0:
        raise ValueError(f'{name} is empty.')
    if not np.isfinite(arr).any():
        raise ValueError(f'{name} has no finite values.')
    return arr


def render_result(result: Dict[str, Any]):
    status = result.get('status', 'unknown')
    if status != 'ok':
        st.warning(f'Status: {status}')
        st.caption('No prediction produced for this window.')
        return

    probs = result.get('probabilities', {}) or {}
    class_id = result.get('class_id')
    label = result.get('prediction')

    st.success(f'Prediction: **{label}** (class_id={class_id})')
    if 'quality_warning' in result and result['quality_warning']:
        st.warning(f"Quality warning: {result['quality_warning']}")

    st.write('Window')
    st.write(f"{result.get('window_start_sec', 0):.2f}s to {result.get('window_end_sec', 0):.2f}s")

    if probs:
        st.write('Probabilities')
        st.json(probs)
        cols = st.columns(3)
        for i, (k, v) in enumerate(probs.items()):
            cols[i % 3].metric(k, f'{v*100:.2f}%')


# Sidebar bundle path
st.sidebar.header('Bundle')
defaul_bundle = Path('.')
bundle_dir = st.sidebar.text_input('Bundle folder', value=str(defaul_bundle))
st.sidebar.caption('Folder must contain: stress_model.keras, model_config.json, stress_inference.py, results.html')

if not Path(bundle_dir).exists():
    st.error('Bundle folder does not exist.')
    st.stop()

try:
    predictor = load_predictor(Path(bundle_dir))
except Exception as exc:
    st.error(f'Could not load predictor from folder: {exc}')
    st.stop()

# Quick diagnostics
st.caption('Loaded model and preprocessing config:')
st.json({
    'labels': predictor.labels,
})

mode = st.radio('Input mode', ['Upload CSV', 'Paste JSON'])

bvp = None
eda = None

if mode == 'Upload CSV':
    file = st.file_uploader(
        'Upload CSV with columns for BVP and EDA',
        type=['csv']
    )
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            st.error(f'Cannot read CSV: {exc}')
            st.stop()

        st.write('Detected columns:', list(df.columns))
        col1, col2 = st.columns(2)
        with col1:
            bvp_col = st.selectbox('BVP column', df.columns, index=0)
        with col2:
            default_eda = 'EDA' if 'EDA' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            eda_col = st.selectbox('EDA column', df.columns, index=1 if len(df.columns) > 1 else 0)

        bvp = pd.to_numeric(df[bvp_col], errors='coerce').to_numpy(dtype=float)
        eda = pd.to_numeric(df[eda_col], errors='coerce').to_numpy(dtype=float)

        # Optional diagnostic preview
        st.dataframe(df[[bvp_col, eda_col]].head(10))
else:
    c1, c2 = st.columns(2)
    with c1:
        bvp_json = st.text_area('BVP raw list (64 Hz)', height=220, value='[0.05, 0.04, 0.06, 0.07, ...]')
    with c2:
        eda_json = st.text_area('EDA raw list (4 Hz)', height=220, value='[0.4, 0.42, 0.41, 0.40, ...]')

    if bvp_json.strip() and eda_json.strip():
        try:
            bvp = parse_json_array(bvp_json, 'BVP')
            eda = parse_json_array(eda_json, 'EDA')
        except Exception as exc:
            st.error(str(exc))

if st.button('Run Inference', type='primary'):
    if bvp is None or eda is None:
        st.warning('Provide both BVP and EDA first.')
        st.stop()

    try:
        result = predictor.predict_latest(bvp, eda)
        render_result(result)
    except Exception as exc:
        st.error(f'Inference failed: {exc}')

st.markdown('---')
st.markdown('Notes')
st.markdown('- Raw BVP and EDA are expected from the same session origin.')
st.markdown('- Inference is attempted per 30s windows with 5s stride; early windows may return rejected status.')
st.markdown('- Do not pre-normalize or convert features manually; use this pipeline output directly.')

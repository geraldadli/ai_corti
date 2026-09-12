"""Shared training/app preprocessing and Python-backend inference for wrist BVP + EDA.

Load StressPredictor from the extracted bundle. Pass raw, uniformly sampled arrays
from the SAME recording origin: BVP 64 Hz, EDA 4 Hz, missing samples represented by
NaN (never deleted). Supply the recording from session start so filter history is
preserved. This reference backend replays the session on each call; it is not an
optimized BLE client or a browser/TFLite runtime. Do not normalize inputs yourself.
"""
from pathlib import Path
import json
import numpy as np
from scipy import signal

DEFAULT_PREPROCESSING = {
    'version': 2, 'bvp_fs': 64, 'eda_fs': 4, 'window_sec': 30, 'stride_sec': 5,
    'filter_order': 4, 'bvp_band_hz': [.5, 4.], 'eda_lowpass_hz': 1.,
    'warmup_sec': 10., 'max_gap_sec': .5, 'minimum_observed_fraction': .95,
    'eda_bounds_us': [0., 100.], 'minimum_std': 1e-8,
    'peak_distance_sec': .30, 'peak_prominence_mad': .5,
    'ibi_bounds_sec': [.30, 2.], 'ibi_local_tolerance': .25,
    'minimum_intervals': 15, 'minimum_ibi_fraction': .8,
    'minimum_ibi_coverage': .8, 'maximum_beat_gap_sec': 3.,
    'strict_pulse_gate': False, 'tonic_time_constant_sec': 10.,
    'bvp_mad_floor': 1e-6, 'eda_log_mad_floor': .02,
    'relative_tonic_floor_us': .05, 'robust_clip': 5.,
    'eda_channels': ['log_level', 'relative_log_level', 'relative_phasic', 'log_derivative']
}

def finite_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def clean_channel(values, fs, config, bounds=None):
    x = np.asarray(values, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError('Each signal must be a single channel.')
    x = x.copy()
    x[~np.isfinite(x)] = np.nan
    if bounds is not None:
        x[(x < bounds[0]) | (x > bounds[1])] = np.nan
    observed = np.isfinite(x)
    limit = int(config['max_gap_sec'] * fs)
    for a, b in finite_runs(~observed):
        if a > 0:
            x[a:min(b, a + limit)] = x[a - 1]
    return x, observed

def filter_channel(x, fs, cutoff, kind, config):
    sos = signal.butter(config['filter_order'], cutoff, btype=kind, fs=fs, output='sos')
    result = np.full(len(x), np.nan)
    warmup = int(config['warmup_sec'] * fs)
    for a, b in finite_runs(np.isfinite(x)):
        if b - a <= warmup:
            continue
        result[a:b], _ = signal.sosfilt(sos, x[a:b], zi=signal.sosfilt_zi(sos) * x[a])
        result[a:a + warmup] = np.nan
    return result

def prepare_recording(bvp, eda, config=None):
    config = DEFAULT_PREPROCESSING if config is None else config
    bvp_cleaned, bv = clean_channel(bvp, config['bvp_fs'], config)
    e, ev = clean_channel(eda, config['eda_fs'], config, config['eda_bounds_us'])
    ef = filter_channel(e, config['eda_fs'], config['eda_lowpass_hz'], 'lowpass', config)
    tonic, derivative = np.full(len(ef), np.nan), np.full(len(ef), np.nan)
    alpha = 1 - np.exp(-1 / (config['eda_fs'] * config['tonic_time_constant_sec']))
    for a, b in finite_runs(np.isfinite(ef)):
        positive = np.maximum(ef[a:b], 0.)
        tonic[a:b], _ = signal.lfilter([alpha], [1., -(1-alpha)], positive,
                                       zi=[(1-alpha) * positive[0]])
        derivative[a:b] = np.diff(np.log1p(positive), prepend=np.log1p(positive[0])) * config['eda_fs']
    return {'bvp': filter_channel(bvp_cleaned, config['bvp_fs'], config['bvp_band_hz'], 'bandpass', config),
            'eda': ef, 'eda_tonic': tonic, 'eda_phasic': ef - tonic, 'eda_log_derivative': derivative,
            'bvp_observed': bv, 'eda_observed': ev}

def robust_scale(x, floor, clip):
    median = np.median(x)
    scale = max(float(1.4826 * np.median(np.abs(x - median))), floor)
    return np.clip((x - median) / scale, -clip, clip)

def represent_window(bvp, eda, tonic, derivative, config):
    # Only the completed window supplies robust statistics: no future samples or labels.
    bvp = robust_scale(bvp, config['bvp_mad_floor'], config['robust_clip'])[:, None]
    log_level = np.log1p(np.maximum(eda, 0.))
    relative = robust_scale(log_level, config['eda_log_mad_floor'], config['robust_clip'])
    phasic = (eda - tonic) / (np.maximum(tonic, 0.) + config['relative_tonic_floor_us'])
    channels = np.column_stack([log_level, relative,
                                np.clip(phasic, -config['robust_clip'], config['robust_clip']),
                                np.clip(derivative, -config['robust_clip'], config['robust_clip'])])
    return {'bvp': bvp.astype('float32'), 'eda': channels.astype('float32')}

def pulse_quality(bvp, config):
    mad = np.median(np.abs(bvp - np.median(bvp)))
    peaks, _ = signal.find_peaks(bvp, distance=max(1, int(config['peak_distance_sec'] * config['bvp_fs'])),
                                prominence=max(config['peak_prominence_mad'] * mad, 1e-6))
    rr = np.diff(peaks) / config['bvp_fs']
    if len(rr) < config['minimum_intervals']:
        return False
    local = np.array([np.median(rr[max(0, i-5):i+6]) for i in range(len(rr))])
    good = ((rr >= config['ibi_bounds_sec'][0]) & (rr <= config['ibi_bounds_sec'][1]) &
            (np.abs(rr - local) <= config['ibi_local_tolerance'] * local))
    beat_times = peaks[1:][good] / config['bvp_fs']
    return bool(good.sum() >= config['minimum_intervals'] and
                good.mean() >= config['minimum_ibi_fraction'] and
                rr[good].sum() / config['window_sec'] >= config['minimum_ibi_coverage'] and
                len(beat_times) > 1 and np.max(np.diff(beat_times)) <= config['maximum_beat_gap_sec'])

def window_at(prepared, end_sec, config=None):
    config = DEFAULT_PREPROCESSING if config is None else config
    start = end_sec - config['window_sec']
    if start < 0:
        return None, 'insufficient_history'
    inputs = {}
    for name in ['bvp', 'eda']:
        fs = config[name + '_fs']
        a, b = int(round(start * fs)), int(round(end_sec * fs))
        x = prepared[name][a:b]
        if len(x) != int(config['window_sec'] * fs):
            return None, name + '_insufficient_samples'
        if not np.isfinite(x).all():
            return None, name + '_gap_or_warmup'
        if prepared[name + '_observed'][a:b].mean() < config['minimum_observed_fraction']:
            return None, name + '_missing_samples'
        if np.std(x) < config['minimum_std']:
            return None, name + '_flat'
        inputs[name] = x.astype('float32')[:, None]
    if config['strict_pulse_gate'] and not pulse_quality(inputs['bvp'][:, 0], config):
        return None, 'pulse_quality'
    a = int(round(start * config['eda_fs']))
    b = int(round(end_sec * config['eda_fs']))
    tonic, derivative = prepared['eda_tonic'][a:b], prepared['eda_log_derivative'][a:b]
    if not np.isfinite(tonic).all() or not np.isfinite(derivative).all():
        return None, 'eda_component_gap'
    return represent_window(inputs['bvp'][:, 0], inputs['eda'][:, 0], tonic, derivative, config), 'ok'

class StressPredictor:
    """Load once per backend process; call predict_latest with a complete raw session."""
    def __init__(self, bundle_directory):
        import tensorflow as tf
        folder = Path(bundle_directory)
        self.config = json.loads((folder / 'model_config.json').read_text(encoding='utf-8'))
        if self.config.get('format_version') != 2 or self.config.get('preprocessing', {}).get('version') != 2:
            raise ValueError('Use the version 2 model, config and inference module from the same bundle.')
        self.preprocessing = self.config['preprocessing']
        self.model = tf.keras.models.load_model(folder / 'stress_model.keras', compile=False)
        self.labels = self.config['class_names']

    def predict_latest(self, bvp, eda):
        c = self.preprocessing
        prepared = prepare_recording(bvp, eda, c)
        duration = min(len(prepared['bvp']) / c['bvp_fs'], len(prepared['eda']) / c['eda_fs'])
        end = np.floor(duration / c['stride_sec']) * c['stride_sec']
        inputs, status = window_at(prepared, end, c)
        if inputs is None:
            return {'status': status, 'window_end_sec': float(end), 'prediction': None,
                    'probabilities': None}
        probabilities = self.model({k: v[None, ...] for k, v in inputs.items()}, training=False).numpy()[0]
        index = int(np.argmax(probabilities))
        a, b = int((end-c['window_sec'])*c['bvp_fs']), int(end*c['bvp_fs'])
        warning = None if pulse_quality(prepared['bvp'][a:b], c) else 'irregular_pulse_pattern'
        return {'status': 'ok', 'window_start_sec': float(end - c['window_sec']),
                'window_end_sec': float(end), 'class_id': index, 'prediction': self.labels[index],
                'quality_warning': warning,
                'probabilities': {label: float(p) for label, p in zip(self.labels, probabilities)}}

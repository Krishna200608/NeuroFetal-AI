"""
Data Ingestion Pipeline for NeuroFetal AI (v2 — SOTA)
=====================================================
Processes CTU-UHB raw records into training-ready arrays.

v2 Changes (SOTA Strategy):
- Extracts 16 tabular features (was 3): demographics + signal-derived
- Improved FHR normalization (excludes 0-gaps from min/max)
- pH threshold relaxed to 7.15 (FIGO standard acidemia definition)
- Signal quality filter (skips windows with >50% signal loss)

Usage:
    python Code/scripts/data_ingestion.py
"""

import os
import glob
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "ctu_uhb_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
TARGET_FS = 1   # 1 Hz
DURATION_MIN = 60
DURATION_SEC = DURATION_MIN * 60
GAP_THRESHOLD_SEC = 15

# SOTA: pH threshold aligned with FIGO acidemia definition
PH_THRESHOLD = 7.15

# SOTA: Signal quality gate
MAX_SIGNAL_LOSS_PCT = 0.50  # Skip windows with >50% missing signal


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_header(header_path):
    """
    Parses the .hea file to extract clinical features.
    CTU-UHB headers contain: Age, Parity, Gravidity, Gestation, Weight, pH, BDecf, etc.
    """
    features = {
        'Age': None,
        'Parity': None,
        'Gestation': None,
        'Gravidity': None,
        'Weight': None,
        'pH': None,
        'BDecf': None,
        'Presentation': None,
    }

    with open(header_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                content = line[1:].strip()
                parts = content.split()
                if not parts:
                    continue

                val_str = parts[-1]

                if content.startswith('Age'):
                    try: features['Age'] = float(val_str)
                    except: pass
                elif content.startswith('Parity'):
                    try: features['Parity'] = float(val_str)
                    except: pass
                elif content.startswith('Gravidity'):
                    try: features['Gravidity'] = float(val_str)
                    except: pass
                elif content.startswith('Gest. weeks'):
                    try: features['Gestation'] = float(val_str)
                    except: pass
                elif content.startswith('Weight'):
                    try: features['Weight'] = float(val_str)
                    except: pass
                elif content.startswith('pH'):
                    try: features['pH'] = float(val_str)
                    except: pass
                elif content.startswith('BDecf'):
                    try: features['BDecf'] = float(val_str)
                    except: pass
                elif content.startswith('Presentation') or content.startswith('Pres.'):
                    # Encode: 1=cephalic, 2=breech, 0=unknown
                    try: features['Presentation'] = float(val_str)
                    except: features['Presentation'] = 0.0

    return features


# ============================================================================
# Signal-Derived Feature Extraction (SOTA Phase 1)
# ============================================================================

def compute_baseline(fhr, fs=1):
    """
    Estimate FHR baseline using a wide moving median.
    Standard clinical practice uses ~10 minute windows.
    """
    window = int(10 * 60 * fs)  # 10 minutes
    if window < 1:
        window = 1
    half_w = window // 2
    n = len(fhr)
    baseline = np.zeros(n)
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        segment = fhr[start:end]
        valid = segment[segment > 0]  # Exclude gaps
        if len(valid) > 0:
            baseline[i] = np.median(valid)
        else:
            baseline[i] = 0
    return baseline


def compute_stv(fhr, fs=1):
    """Short-Term Variability: mean absolute first-order difference."""
    valid = fhr[fhr > 0]
    if len(valid) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(valid))))


def compute_ltv(fhr, fs=1):
    """Long-Term Variability: std of 1-minute segment means."""
    segment_len = int(60 * fs)
    if segment_len < 1:
        segment_len = 1
    n_segments = len(fhr) // segment_len
    if n_segments < 2:
        return 0.0
    means = []
    for i in range(n_segments):
        seg = fhr[i * segment_len:(i + 1) * segment_len]
        valid = seg[seg > 0]
        if len(valid) > 0:
            means.append(np.mean(valid))
    if len(means) < 2:
        return 0.0
    return float(np.std(means))


def count_accelerations(fhr, baseline, fs=1, threshold_bpm=15, min_duration_sec=15):
    """Count accelerations: rises > threshold_bpm above baseline for > min_duration."""
    diff = fhr - baseline
    above = diff > threshold_bpm
    # Find contiguous runs
    runs = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    count = 0
    for s, e in zip(starts, ends):
        if (e - s) / fs >= min_duration_sec:
            count += 1
    return count


def count_decelerations(fhr, baseline, fs=1, threshold_bpm=15, min_duration_sec=15):
    """Count decelerations: dips > threshold_bpm below baseline for > min_duration."""
    diff = baseline - fhr
    # Only consider valid FHR (non-zero)
    valid_mask = fhr > 0
    below = (diff > threshold_bpm) & valid_mask
    runs = np.diff(np.concatenate(([0], below.astype(int), [0])))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    count = 0
    total_area = 0.0
    for s, e in zip(starts, ends):
        duration = (e - s) / fs
        if duration >= min_duration_sec:
            count += 1
            total_area += float(np.sum(diff[s:e])) / fs  # bpm·seconds
    return count, total_area


def compute_sample_entropy(signal, m=2, r_frac=0.2):
    """
    Approximate sample entropy for signal complexity.
    Uses a fast vectorized approach for reasonable performance.
    """
    valid = signal[signal > 0]
    n = len(valid)
    if n < 50:
        return 0.0

    # Subsample for speed if signal is long
    if n > 500:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        valid = valid[idx]
        n = len(valid)

    r = r_frac * np.std(valid)
    if r == 0:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([valid[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(diffs < r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


def compute_uc_frequency(uc, fs=1):
    """Detect contraction peaks in UC signal and return frequency."""
    if len(uc) < 10:
        return 0.0, 0.0
    # UC peaks: prominence-based detection
    uc_smooth = np.convolve(uc, np.ones(int(30 * fs)) / int(max(1, 30 * fs)), mode='same')
    threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
    peaks, properties = find_peaks(uc_smooth, height=threshold, distance=int(120 * fs), prominence=0.1)
    count = len(peaks)
    mean_intensity = float(np.mean(uc_smooth[peaks])) if count > 0 else 0.0
    return count, mean_intensity


def compute_fhr_uc_lag(fhr, uc, fs=1):
    """Cross-correlation lag between FHR response and UC contraction."""
    valid_fhr = fhr.copy()
    valid_fhr[valid_fhr == 0] = np.mean(fhr[fhr > 0]) if np.any(fhr > 0) else 0
    if np.std(valid_fhr) == 0 or np.std(uc) == 0:
        return 0.0
    # Normalize
    fhr_n = (valid_fhr - np.mean(valid_fhr)) / (np.std(valid_fhr) + 1e-8)
    uc_n = (uc - np.mean(uc)) / (np.std(uc) + 1e-8)
    # Cross-correlation (limited to ±5 min lag)
    max_lag = int(5 * 60 * fs)
    corr = np.correlate(fhr_n, uc_n, mode='full')
    mid = len(corr) // 2
    start = max(0, mid - max_lag)
    end = min(len(corr), mid + max_lag + 1)
    corr_window = corr[start:end]
    if len(corr_window) == 0:
        return 0.0
    lag_idx = np.argmax(np.abs(corr_window)) - (end - start) // 2
    return float(lag_idx) / fs  # in seconds


def extract_window_features(fhr_raw_bpm, uc_raw, fs=1):
    """
    Extract 13 signal-derived features from a single FHR/UC window.
    These are computed on the RAW (un-normalized) BPM values for clinical meaning.

    Returns:
        dict of feature name -> value
    """
    features = {}

    # Signal quality
    signal_loss = np.mean(fhr_raw_bpm == 0)
    features['signal_loss_pct'] = signal_loss

    # Valid mask
    valid = fhr_raw_bpm[fhr_raw_bpm > 0]

    # Baseline estimation
    baseline = compute_baseline(fhr_raw_bpm, fs)

    # FHR baseline (mean of valid baseline)
    valid_bl = baseline[baseline > 0]
    features['fhr_baseline'] = float(np.mean(valid_bl)) if len(valid_bl) > 0 else 0.0

    # STV & LTV
    features['fhr_stv'] = compute_stv(fhr_raw_bpm, fs)
    features['fhr_ltv'] = compute_ltv(fhr_raw_bpm, fs)

    # Accelerations
    features['fhr_accel_count'] = count_accelerations(fhr_raw_bpm, baseline, fs)

    # Decelerations
    decel_count, decel_area = count_decelerations(fhr_raw_bpm, baseline, fs)
    features['fhr_decel_count'] = decel_count
    features['fhr_decel_area'] = decel_area

    # Range & IQR
    if len(valid) > 0:
        features['fhr_range'] = float(np.max(valid) - np.min(valid))
        features['fhr_iqr'] = float(np.percentile(valid, 75) - np.percentile(valid, 25))
    else:
        features['fhr_range'] = 0.0
        features['fhr_iqr'] = 0.0

    # Entropy (signal complexity)
    features['fhr_entropy'] = compute_sample_entropy(fhr_raw_bpm)

    # UC features
    uc_freq, uc_intensity = compute_uc_frequency(uc_raw, fs)
    features['uc_freq'] = uc_freq
    features['uc_intensity_mean'] = uc_intensity

    # FHR-UC interaction
    features['fhr_uc_lag'] = compute_fhr_uc_lag(fhr_raw_bpm, uc_raw, fs)

    return features


# ============================================================================
# Signal Processing
# ============================================================================

def process_signal(signal, fs):
    """
    Process FHR signal:
    1. Interpolate gaps < 15s.
    2. Keep gaps > 15s as 0.
    3. Crop last 60 mins.
    4. Resample to 1Hz.
    5. Normalize using ONLY valid (non-zero) values.
    """
    n_samples = len(signal)
    processed_signal = signal.copy()

    # Find continuous runs of zeros
    mask = (processed_signal == 0).astype(int)
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        gap_len_sec = (end - start) / fs
        if gap_len_sec < GAP_THRESHOLD_SEC:
            left_idx = start - 1
            right_idx = end
            if left_idx < 0 or right_idx >= n_samples:
                continue
            x_known = [left_idx, right_idx]
            y_known = [signal[left_idx], signal[right_idx]]
            interp_func = interp1d(x_known, y_known, kind='linear')
            processed_signal[start:end] = interp_func(np.arange(start, end))

    # Crop last 60 minutes
    required_samples = DURATION_SEC * fs
    if len(processed_signal) < required_samples:
        pad_len = int(required_samples - len(processed_signal))
        processed_signal = np.pad(processed_signal, (pad_len, 0), 'constant')
    else:
        processed_signal = processed_signal[-int(required_samples):]

    # Resample to 1Hz
    num_samples_target = DURATION_SEC * TARGET_FS
    x_old = np.linspace(0, DURATION_SEC, len(processed_signal))
    x_new = np.linspace(0, DURATION_SEC, num_samples_target)
    f_resample = interp1d(x_old, processed_signal, kind='linear', bounds_error=False, fill_value=0)
    resampled_signal = f_resample(x_new)

    return resampled_signal


def normalize_fhr(signal):
    """
    SOTA: Normalize FHR using only valid (non-zero) values.
    Gaps (0-values) stay at 0 after normalization.
    Valid FHR range mapped to [0, 1].
    """
    valid_mask = signal > 0
    if np.sum(valid_mask) == 0:
        return np.zeros_like(signal)

    valid_vals = signal[valid_mask]
    _min = np.min(valid_vals)
    _max = np.max(valid_vals)

    if _max > _min:
        normalized = np.zeros_like(signal)
        normalized[valid_mask] = (signal[valid_mask] - _min) / (_max - _min)
        return normalized
    else:
        return np.zeros_like(signal)


def process_uc_signal(signal, fs):
    """
    Process UC (Uterine Contraction) signal:
    1. Crop last 60 mins.
    2. Resample to 1Hz.
    3. Normalize (MinMax).
    """
    required_samples = DURATION_SEC * fs
    if len(signal) < required_samples:
        pad_len = int(required_samples - len(signal))
        processed_signal = np.pad(signal, (pad_len, 0), 'constant')
    else:
        processed_signal = signal[-int(required_samples):]

    num_samples_target = DURATION_SEC * TARGET_FS
    x_old = np.linspace(0, DURATION_SEC, len(processed_signal))
    x_new = np.linspace(0, DURATION_SEC, num_samples_target)
    f_resample = interp1d(x_old, processed_signal, kind='linear', bounds_error=False, fill_value=0)
    resampled_signal = f_resample(x_new)

    _min = np.min(resampled_signal)
    _max = np.max(resampled_signal)
    if _max > _min:
        resampled_signal = (resampled_signal - _min) / (_max - _min)
    else:
        resampled_signal = np.zeros_like(resampled_signal)

    return resampled_signal


# ============================================================================
# Main Pipeline
# ============================================================================

# Ordered list of tabular feature names (for reference/documentation)
TABULAR_FEATURE_NAMES = [
    # Demographics (from header)
    'Age', 'Parity', 'Gestation', 'Gravidity', 'Weight',
    # Signal-derived features
    'fhr_baseline', 'fhr_stv', 'fhr_ltv',
    'fhr_accel_count', 'fhr_decel_count', 'fhr_decel_area',
    'fhr_range', 'fhr_iqr', 'fhr_entropy',
    'uc_freq', 'uc_intensity_mean', 'fhr_uc_lag',
    'signal_loss_pct',
]  # Total: 18 features


def main():
    ensure_dir(PROCESSED_DATA_DIR)

    record_paths = glob.glob(os.path.join(RAW_DATA_DIR, "*.hea"))
    print(f"Found {len(record_paths)} records.")
    print(f"pH threshold: {PH_THRESHOLD}")
    print(f"Max signal loss: {MAX_SIGNAL_LOSS_PCT * 100:.0f}%")

    X_fhr = []
    X_uc = []
    X_tabular = []
    y = []

    # Process slices
    WINDOW_SIZE_SEC = 20 * 60  # 20 minutes
    STRIDE_SEC = 10 * 60       # 10 minutes overlap

    cnt = 0
    valid_cnt = 0
    total_slices = 0
    skipped_quality = 0

    for hea_path in record_paths:
        base = os.path.splitext(hea_path)[0]
        rec_name = os.path.basename(base)

        # Parse Header
        feats = parse_header(hea_path)

        if feats['pH'] is None or np.isnan(feats['pH']):
            continue

        # SOTA: pH < 7.15 (FIGO standard)
        is_compromised = 1 if feats['pH'] < PH_THRESHOLD else 0

        # Read Signal
        try:
            signals, fields = wfdb.rdsamp(base)
            fhr_signal = signals[:, 0]
            uc_signal = signals[:, 1] if signals.shape[1] > 1 else None
            fs = fields['fs']

            # Get the cleaned last 60 mins (at original fs, before resampling)
            processed_60min_fhr_raw = process_signal(fhr_signal, fs)  # At 1Hz, raw BPM
            # Note: process_signal now returns UN-normalized values

            # Process UC
            processed_60min_uc = None
            if uc_signal is not None:
                processed_60min_uc = process_uc_signal(uc_signal, fs)

            # Window parameters
            w_size = int(WINDOW_SIZE_SEC * TARGET_FS)
            stride = int(STRIDE_SEC * TARGET_FS)

            if len(processed_60min_fhr_raw) < w_size:
                continue

            num_slices = (len(processed_60min_fhr_raw) - w_size) // stride + 1

            for i in range(int(num_slices)):
                start = i * stride
                end = start + w_size

                if end > len(processed_60min_fhr_raw):
                    break

                window_fhr_raw = processed_60min_fhr_raw[start:end]

                # UC window
                if processed_60min_uc is not None and end <= len(processed_60min_uc):
                    window_uc = processed_60min_uc[start:end]
                else:
                    window_uc = np.zeros(w_size)

                # ============================================================
                # SOTA: Extract signal-derived features (on RAW signal)
                # ============================================================
                sig_features = extract_window_features(window_fhr_raw, window_uc, fs=TARGET_FS)

                # SOTA: Signal quality gate
                if sig_features['signal_loss_pct'] > MAX_SIGNAL_LOSS_PCT:
                    skipped_quality += 1
                    continue

                # Normalize FHR for CNN input (after feature extraction)
                window_fhr_normalized = normalize_fhr(window_fhr_raw)

                # Build tabular vector (18 features)
                tab_vec = [
                    # Demographics
                    feats['Age'],
                    feats['Parity'],
                    feats['Gestation'],
                    feats['Gravidity'],
                    feats['Weight'],
                    # Signal-derived
                    sig_features['fhr_baseline'],
                    sig_features['fhr_stv'],
                    sig_features['fhr_ltv'],
                    sig_features['fhr_accel_count'],
                    sig_features['fhr_decel_count'],
                    sig_features['fhr_decel_area'],
                    sig_features['fhr_range'],
                    sig_features['fhr_iqr'],
                    sig_features['fhr_entropy'],
                    sig_features['uc_freq'],
                    sig_features['uc_intensity_mean'],
                    sig_features['fhr_uc_lag'],
                    sig_features['signal_loss_pct'],
                ]

                X_fhr.append(window_fhr_normalized)
                X_uc.append(window_uc)
                X_tabular.append(tab_vec)
                y.append(is_compromised)
                total_slices += 1

            valid_cnt += 1

        except Exception as e:
            print(f"Error processing {rec_name}: {e}")

        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} records...")

    # Convert to arrays
    X_fhr = np.array(X_fhr, dtype=np.float32)
    X_uc = np.array(X_uc, dtype=np.float32)
    X_tabular = np.array(X_tabular, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Handle NaNs in tabular data (column-wise median imputation)
    for col in range(X_tabular.shape[1]):
        col_data = X_tabular[:, col]
        nan_mask = np.isnan(col_data)
        if np.any(nan_mask):
            valid_vals = col_data[~nan_mask]
            fill_val = np.median(valid_vals) if len(valid_vals) > 0 else 0.0
            X_tabular[nan_mask, col] = fill_val

    # Handle NaNs in UC data
    X_uc = np.nan_to_num(X_uc, nan=0.0)

    # Standardize tabular features (z-score) — critical for neural network performance
    tab_means = np.mean(X_tabular, axis=0)
    tab_stds = np.std(X_tabular, axis=0)
    tab_stds[tab_stds == 0] = 1.0  # Avoid division by zero
    X_tabular = (X_tabular - tab_means) / tab_stds

    # Save standardization parameters for inference
    np.save(os.path.join(PROCESSED_DATA_DIR, "tabular_means.npy"), tab_means)
    np.save(os.path.join(PROCESSED_DATA_DIR, "tabular_stds.npy"), tab_stds)

    # Save
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"), X_fhr)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_uc.npy"), X_uc)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"), X_tabular)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), y)

    print(f"\nProcessing complete.")
    print(f"  Patients: {valid_cnt}")
    print(f"  Total windows: {total_slices}")
    print(f"  Skipped (quality): {skipped_quality}")
    print(f"  Shapes: X_fhr={X_fhr.shape}, X_uc={X_uc.shape}, X_tabular={X_tabular.shape}, y={y.shape}")
    print(f"  Tabular features ({X_tabular.shape[1]}): {TABULAR_FEATURE_NAMES}")
    print(f"  Class balance: {np.sum(y)} compromised / {len(y)} total ({np.mean(y):.1%})")


if __name__ == "__main__":
    main()

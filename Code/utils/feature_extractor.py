import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.stats import skew, kurtosis
try:
    from scripts.data_ingestion import extract_window_features, compute_baseline_rt
except ImportError:
    # Fallback if running from a different context where scripts is not a package
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from data_ingestion import extract_window_features, compute_baseline_rt

# Constants
N_CSP_FEATURES = 19

def extract_18_tabular_rt(fhr_raw, uc_raw, header):
    """
    Extract 18 tabular features for Real-Time inference.
    Matches data_ingestion.py robust logic.
    """
    # Use defaults if header missing
    age = header.get('Age', 30)
    parity = header.get('Parity', 0)
    gestation = header.get('Gestation', 39)
    gravidity = header.get('Gravidity', 1)
    weight = header.get('Weight', 70)
    
    # Robust extraction
    sig_features = extract_window_features(fhr_raw, uc_raw, fs=1)
    
    # Assemble 18-feature vector
    return np.array([
        float(age if age is not None else 30),
        float(parity if parity is not None else 0),
        float(gestation if gestation is not None else 39),
        float(gravidity if gravidity is not None else 1),
        float(weight if weight is not None else 70),
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
    ], dtype=np.float32)


def extract_realtime_tabular(fhr_window_raw, uc_window_raw, age, parity, gestation):
    """
    Extract 16 tabular features from a single FHR/UC window at inference time.
    Matches the feature vector from data_ingestion.py exactly:
      [signal_loss_pct, fhr_baseline, fhr_stv, fhr_ltv, fhr_accel_count,
       fhr_decel_count, fhr_decel_area, fhr_range, fhr_iqr, fhr_entropy,
       uc_freq, uc_intensity_mean, fhr_uc_lag, age, parity, gestation]
    """
    valid = fhr_window_raw[fhr_window_raw > 0]
    
    # Signal loss
    signal_loss = np.mean(fhr_window_raw == 0)
    
    # Baseline
    baseline = compute_baseline_rt(fhr_window_raw)
    valid_bl = baseline[baseline > 0]
    fhr_baseline = float(np.mean(valid_bl)) if len(valid_bl) > 0 else 0.0
    
    # STV: mean absolute first-order difference
    fhr_stv = float(np.mean(np.abs(np.diff(valid)))) if len(valid) > 1 else 0.0
    
    # LTV: std of 1-minute segment means
    seg_len = 60
    n_segs = len(fhr_window_raw) // seg_len
    if n_segs >= 2:
        means = []
        for i in range(n_segs):
            seg = fhr_window_raw[i * seg_len:(i + 1) * seg_len]
            v = seg[seg > 0]
            if len(v) > 0:
                means.append(np.mean(v))
        fhr_ltv = float(np.std(means)) if len(means) >= 2 else 0.0
    else:
        fhr_ltv = 0.0
    
    # Accelerations: rises > 15 bpm above baseline for > 15s
    diff_above = fhr_window_raw - baseline
    above = diff_above > 15
    runs = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts_a = np.where(runs == 1)[0]
    ends_a = np.where(runs == -1)[0]
    accel_count = sum(1 for s, e in zip(starts_a, ends_a) if (e - s) >= 15)
    
    # Decelerations: dips > 15 bpm below baseline for > 15s
    diff_below = baseline - fhr_window_raw
    valid_mask = fhr_window_raw > 0
    below = (diff_below > 15) & valid_mask
    runs_d = np.diff(np.concatenate(([0], below.astype(int), [0])))
    starts_d = np.where(runs_d == 1)[0]
    ends_d = np.where(runs_d == -1)[0]
    decel_count = 0
    decel_area = 0.0
    for s, e in zip(starts_d, ends_d):
        if (e - s) >= 15:
            decel_count += 1
            decel_area += float(np.sum(diff_below[s:e]))
    
    # Range & IQR
    fhr_range = float(np.max(valid) - np.min(valid)) if len(valid) > 0 else 0.0
    fhr_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if len(valid) > 0 else 0.0
    
    # Entropy (simplified for speed at inference)
    fhr_entropy = 0.0
    if len(valid) > 50:
        # Approximate entropy via normalized std
        fhr_entropy = float(np.log(np.std(valid) + 1e-8))
    
    # UC features
    uc_freq = 0.0
    uc_intensity = 0.0
    if uc_window_raw is not None and len(uc_window_raw) > 10:
        uc_smooth = np.convolve(uc_window_raw, np.ones(30) / 30, mode='same')
        threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
        peaks, _ = find_peaks(uc_smooth, height=threshold, distance=120, prominence=0.1)
        uc_freq = float(len(peaks))
        uc_intensity = float(np.mean(uc_smooth[peaks])) if len(peaks) > 0 else 0.0
    
    # FHR-UC lag
    fhr_uc_lag = 0.0
    if uc_window_raw is not None and np.std(fhr_window_raw) > 0 and np.std(uc_window_raw) > 0:
        fhr_n = (fhr_window_raw - np.mean(fhr_window_raw)) / (np.std(fhr_window_raw) + 1e-8)
        uc_n = (uc_window_raw - np.mean(uc_window_raw)) / (np.std(uc_window_raw) + 1e-8)
        max_lag = 300  # 5 minutes
        corr = np.correlate(fhr_n, uc_n, mode='full')
        mid = len(corr) // 2
        start = max(0, mid - max_lag)
        end = min(len(corr), mid + max_lag + 1)
        corr_window = corr[start:end]
        if len(corr_window) > 0:
            lag_idx = np.argmax(np.abs(corr_window)) - (end - start) // 2
            fhr_uc_lag = float(lag_idx)
    
    # Assemble 16-feature vector (matches training order)
    features = np.array([
        signal_loss, fhr_baseline, fhr_stv, fhr_ltv,
        accel_count, decel_count, decel_area,
        fhr_range, fhr_iqr, fhr_entropy,
        uc_freq, uc_intensity, fhr_uc_lag,
        float(age), float(parity), float(gestation)
    ], dtype=np.float32)
    
    return features


def extract_realtime_csp(fhr_window, uc_window):
    """
    Extract 19 CSP-based features from FHR + UC signals.
    Uses statistical features when CSP filters are not fitted (inference mode).
    Feature vector: [fhr_mean, fhr_std, fhr_min, fhr_max, fhr_mad, fhr_beta0,
                     fhr_skewness, fhr_kurtosis, fhr_sqi, uc_mean, uc_std, uc_count,
                     cross_corr_max, cross_corr_mean, labor_stage_flag, csp_0..csp_3]
    """
    
    features = []
    
    # FHR basic stats
    features.append(np.mean(fhr_window))
    features.append(np.std(fhr_window))
    features.append(np.min(fhr_window))
    features.append(np.max(fhr_window))
    
    # FHR advanced: MAD
    features.append(np.median(np.abs(fhr_window - np.median(fhr_window))))
    
    # Beta0 (baseline intercept)
    n = len(fhr_window)
    if n > 1:
        x = np.arange(n)
        slope = (n * np.sum(x * fhr_window) - np.sum(x) * np.sum(fhr_window)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-8)
        beta0 = np.mean(fhr_window) - slope * np.mean(x)
        features.append(beta0)
    else:
        features.append(0.0)
    
    # Skewness & Kurtosis
    features.append(skew(fhr_window) if len(fhr_window) > 2 else 0.0)
    features.append(kurtosis(fhr_window) if len(fhr_window) > 2 else 0.0)
    
    # SQI (signal quality: fraction in valid range)
    valid_frac = np.mean((fhr_window > 0.05) & (fhr_window < 0.95))  # normalized range
    features.append(valid_frac)
    
    # UC features
    if uc_window is not None and len(uc_window) > 0:
        features.append(np.mean(uc_window))
        features.append(np.std(uc_window))
        features.append(float(np.sum(uc_window > 0.5)))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Cross-correlation
    if uc_window is not None and len(uc_window) > 0:
        cross_corr = correlate(fhr_window, uc_window, mode='same')
        features.append(np.max(np.abs(cross_corr)))
        features.append(np.mean(np.abs(cross_corr)))
    else:
        features.extend([0.0, 0.0])
    
    # Labor stage flag (assume last window = potential stage 2)
    features.append(0.0)
    
    # CSP log-variance features (4 components — zeros when not fitted)
    features.extend([0.0, 0.0, 0.0, 0.0])
    
    return np.array(features[:N_CSP_FEATURES], dtype=np.float32)

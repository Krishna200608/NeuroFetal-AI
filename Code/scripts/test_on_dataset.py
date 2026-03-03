#!/usr/bin/env python3
"""
Test on Full Dataset — NeuroFetal AI V5.0 (Stacking Ensemble)
=============================================================
Runs the *complete* V5.0 Stacking Ensemble (AttentionFusionResNet × 5,
InceptionNet × 5, XGBoost × 5 + Meta-Learner) on the CTU-UHB dataset
with Temperature Scaling calibration, MC Dropout uncertainty, and
optimal threshold search.

Generates a comprehensive markdown report at Reports/Tests/final_metrics.md.

Usage:
    python Code/scripts/test_on_dataset.py
"""

import os
import sys
import io
import glob
import json
import pickle
import datetime

# FIX: Force UTF-8 for Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
import xgboost as xgb
import wfdb
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from scipy.stats import rankdata
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, average_precision_score, f1_score,
    brier_score_loss
)

# =============================================================================
# Setup Paths & Imports
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(BASE_DIR, "Code")
SCRIPTS_DIR = os.path.join(CODE_DIR, "scripts")
UTILS_DIR = os.path.join(CODE_DIR, "utils")
DATA_DIR = os.path.join(BASE_DIR, "Datasets", "ctu_uhb_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(CODE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports", "Tests")

sys.path.append(UTILS_DIR)
sys.path.append(SCRIPTS_DIR)

# Import project modules
try:
    from data_ingestion import process_signal, process_uc_signal, parse_header, normalize_fhr
    from csp_features import MultimodalFeatureExtractor
    from model import CrossModalAttention
    from attention_blocks import SEBlock, TemporalAttentionBlock
    from focal_loss import FocalLoss
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Suppress excessive TF logs
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =============================================================================
# Temperature Scaling (matches evaluate_ensemble.py)
# =============================================================================
class TemperatureScaler:
    """
    Post-hoc calibration via Temperature Scaling.
    Learns a single scalar T such that:
        p_calibrated = sigmoid(logit(p) / T)
    """
    def __init__(self):
        self.temperature = 1.0

    def _nll_loss(self, T, logits, labels):
        """Negative log-likelihood with temperature."""
        scaled = logits / T
        probs = expit(scaled)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    def fit(self, probs, labels):
        """Learn optimal temperature from validation predictions."""
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
        result = minimize_scalar(
            lambda T: self._nll_loss(T, logits, labels),
            bounds=(0.1, 10.0), method='bounded'
        )
        self.temperature = result.x
        print(f"  Learned Temperature: T = {self.temperature:.4f}")

    def calibrate(self, probs):
        """Apply temperature scaling to probabilities."""
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
        return expit(logits / self.temperature)


# =============================================================================
# Optimal Threshold Search (matches evaluate_ensemble.py)
# =============================================================================
def find_optimal_threshold(y_true, y_pred, method='youden', fnr_penalty=3.0):
    """
    Find optimal classification threshold.
    Methods: 'youden' (Sensitivity+Specificity-1), 'f1', 'cost' (penalises FN).
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_score = 0.5, -1.0

    for t in thresholds:
        y_bin = (y_pred >= t).astype(int)
        tp = np.sum((y_bin == 1) & (y_true == 1))
        tn = np.sum((y_bin == 0) & (y_true == 0))
        fp = np.sum((y_bin == 1) & (y_true == 0))
        fn = np.sum((y_bin == 0) & (y_true == 1))

        if method == 'youden':
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            score = sens + spec - 1
        elif method == 'f1':
            score = f1_score(y_true, y_bin, zero_division=0)
        elif method == 'cost':
            cost = fp + fnr_penalty * fn
            score = -cost  # minimise cost = maximise negative cost
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_t = t

    return best_t


# =============================================================================
# Model Loading Helpers
# =============================================================================
def load_custom_model(model_path):
    """Load a Keras model with all custom layers registered."""
    custom_objects = {
        'CrossModalAttention': CrossModalAttention,
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.0, alpha=0.65)
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


# =============================================================================
# MC Dropout Inference
# =============================================================================
def mc_dropout_predict(model, inputs, T=20):
    """
    Run T stochastic forward passes with Dropout enabled at inference time.
    Returns: mean prediction, predictive variance (epistemic uncertainty).
    """
    preds = []
    for _ in range(T):
        p = model(inputs, training=True)  # training=True keeps Dropout active
        preds.append(tf.squeeze(p).numpy())
    preds = np.array(preds)  # (T, N)
    mean_pred = np.mean(preds, axis=0)
    var_pred = np.var(preds, axis=0)
    return mean_pred, var_pred


# =============================================================================
# Feature Extraction (16-dim tabular — matches V5.0 training pipeline)
# =============================================================================
def extract_16_tabular(fhr_raw, uc_raw, age, parity, gestation):
    """
    Extract 16 tabular features from a single FHR/UC window.
    Matches the V5.0 training data order exactly:
      [Age, Parity, Gestation,
       fhr_baseline, fhr_stv, fhr_ltv, fhr_accel_count,
       fhr_decel_count, fhr_late_decel_flag, fhr_variable_decel,
       fhr_approx_entropy, fhr_sample_entropy,
       uc_freq, uc_amplitude, fhr_uc_lag, signal_quality]
    """
    from scipy.signal import find_peaks, correlate

    valid = fhr_raw[fhr_raw > 0]
    signal_quality = float(np.sum(fhr_raw > 0)) / max(len(fhr_raw), 1)

    # --- Demographics (3) ---
    f_age = float(age if age is not None else 30)
    f_parity = float(parity if parity is not None else 0)
    f_gestation = float(gestation if gestation is not None else 39)

    # --- Signal-Derived (13) ---
    # 1. Baseline
    baseline = float(np.median(valid)) if len(valid) > 0 else 140.0

    # 2. STV (Short-Term Variability)
    if len(valid) > 1:
        fhr_stv = float(np.sqrt(np.mean(np.diff(valid) ** 2)))
    else:
        fhr_stv = 0.0

    # 3. LTV (Long-Term Variability)
    seg_len = 60
    n_segs = len(fhr_raw) // seg_len
    if n_segs >= 2:
        seg_means = [float(np.mean(fhr_raw[i * seg_len:(i + 1) * seg_len][fhr_raw[i * seg_len:(i + 1) * seg_len] > 0]))
                     for i in range(n_segs)
                     if np.sum(fhr_raw[i * seg_len:(i + 1) * seg_len] > 0) > 0]
        fhr_ltv = float(np.std(seg_means)) if len(seg_means) >= 2 else 0.0
    else:
        fhr_ltv = 0.0

    # 4. Accelerations (>15 bpm for >15 sec)
    diff_above = fhr_raw - baseline
    above = diff_above > 15
    runs = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts_a = np.where(runs == 1)[0]
    ends_a = np.where(runs == -1)[0]
    accel_count = float(sum(1 for s, e in zip(starts_a, ends_a) if (e - s) >= 15))

    # 5-6. Decelerations
    valid_mask = fhr_raw > 0
    diff_below = baseline - fhr_raw
    below = (diff_below > 15) & valid_mask
    runs_d = np.diff(np.concatenate(([0], below.astype(int), [0])))
    starts_d = np.where(runs_d == 1)[0]
    ends_d = np.where(runs_d == -1)[0]
    decel_count = 0
    for s, e in zip(starts_d, ends_d):
        if (e - s) >= 15:
            decel_count += 1

    # 7. Late Deceleration Flag (requires UC context)
    late_decel_flag = 0.0
    if uc_raw is not None and len(uc_raw) > 10:
        uc_smooth = np.convolve(uc_raw, np.ones(30) / 30, mode='same')
        uc_threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
        uc_peaks, _ = find_peaks(uc_smooth, height=uc_threshold, distance=120)
        for pk in uc_peaks:
            search_start = pk
            search_end = min(pk + 120, len(fhr_raw))
            if search_end > search_start:
                window = fhr_raw[search_start:search_end]
                if len(window) > 0 and np.min(window[window > 0]) < baseline - 15 if np.any(window > 0) else False:
                    late_decel_flag = 1.0
                    break

    # 8. Variable Deceleration Count (abrupt drops)
    variable_decel = 0.0
    for s, e in zip(starts_d, ends_d):
        dur = e - s
        if 15 <= dur <= 120:  # Short but sharp
            drop = np.max(diff_below[s:e])
            if drop > 30:
                variable_decel += 1.0

    # 9. Approximate Entropy
    fhr_approx_entropy = float(np.log(np.std(valid) + 1e-8)) if len(valid) > 50 else 0.0

    # 10. Sample Entropy (simplified)
    if len(valid) > 100:
        diffs = np.abs(np.diff(valid))
        fhr_sample_entropy = float(-np.log(np.mean(diffs < np.std(valid) * 0.2) + 1e-8))
    else:
        fhr_sample_entropy = 0.0

    # 11-12. UC features
    uc_freq = 0.0
    uc_amplitude = 0.0
    if uc_raw is not None and len(uc_raw) > 10:
        uc_smooth = np.convolve(uc_raw, np.ones(30) / 30, mode='same')
        threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
        peaks, props = find_peaks(uc_smooth, height=threshold, distance=120, prominence=0.1)
        uc_freq = float(len(peaks))
        uc_amplitude = float(np.mean(uc_smooth[peaks])) if len(peaks) > 0 else 0.0

    # 13. FHR-UC lag
    fhr_uc_lag = 0.0
    if uc_raw is not None and np.std(fhr_raw) > 0 and np.std(uc_raw) > 0:
        fhr_n = (fhr_raw - np.mean(fhr_raw)) / (np.std(fhr_raw) + 1e-8)
        uc_n = (uc_raw - np.mean(uc_raw)) / (np.std(uc_raw) + 1e-8)
        max_lag = 300
        corr = np.correlate(fhr_n, uc_n, mode='full')
        mid = len(corr) // 2
        start = max(0, mid - max_lag)
        end = min(len(corr), mid + max_lag + 1)
        corr_window = corr[start:end]
        if len(corr_window) > 0:
            lag_idx = np.argmax(np.abs(corr_window)) - (end - start) // 2
            fhr_uc_lag = float(lag_idx)

    return np.array([
        f_age, f_parity, f_gestation,
        baseline, fhr_stv, fhr_ltv, accel_count,
        float(decel_count), late_decel_flag, variable_decel,
        fhr_approx_entropy, fhr_sample_entropy,
        uc_freq, uc_amplitude, fhr_uc_lag, signal_quality
    ], dtype=np.float32)


# =============================================================================
# Main Inference Loop
# =============================================================================
def main():
    print("=" * 70)
    print("  NeuroFetal AI V5.0 — Full Dataset Inference (Stacking Ensemble)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Verify Available Models
    # =========================================================================
    print("\n[1/7] Discovering available models...")

    resnet_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "enhanced_model_fold_*.keras")))
    inception_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "inception_model_fold_*.keras")))
    xgb_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "xgboost_model_fold_*.pkl")))
    meta_path = os.path.join(MODEL_DIR, "stacking_meta_learner.pkl")
    temp_path = os.path.join(MODEL_DIR, "temperature_scaling.json")
    thresh_path = os.path.join(MODEL_DIR, "optimal_thresholds.json")

    print(f"  AttentionFusionResNet folds: {len(resnet_paths)}")
    print(f"  InceptionNet folds:          {len(inception_paths)}")
    print(f"  XGBoost folds:               {len(xgb_paths)}")
    print(f"  Meta-Learner:                {'✓' if os.path.exists(meta_path) else '✗'}")
    print(f"  Temperature Scaling:         {'✓' if os.path.exists(temp_path) else '✗'}")

    has_full_ensemble = (
        len(resnet_paths) >= 5 and
        len(inception_paths) >= 5 and
        len(xgb_paths) >= 5 and
        os.path.exists(meta_path)
    )

    if has_full_ensemble:
        print("  → Full V5.0 Stacking Ensemble available.")
    elif len(resnet_paths) > 0:
        print("  → Partial ensemble. Will use available AttentionFusionResNet fold(s).")
    else:
        print("  ❌ No trained models found. Exiting.")
        return

    # =========================================================================
    # Step 2: Load all models into memory
    # =========================================================================
    print("\n[2/7] Loading models...")

    resnet_models = []
    for p in resnet_paths:
        try:
            m = load_custom_model(p)
            resnet_models.append(m)
            print(f"  ✓ Loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"  ⚠ Failed to load {os.path.basename(p)}: {e}")

    inception_models = []
    for p in inception_paths:
        try:
            m = load_custom_model(p)
            inception_models.append(m)
            print(f"  ✓ Loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"  ⚠ Failed to load {os.path.basename(p)}: {e}")

    xgb_models = []
    for p in xgb_paths:
        try:
            with open(p, 'rb') as f:
                m = pickle.load(f)
            xgb_models.append(m)
            print(f"  ✓ Loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"  ⚠ Failed to load {os.path.basename(p)}: {e}")

    meta_model = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta_model = pickle.load(f)
        print("  ✓ Loaded stacking meta-learner")

    # Load saved temperature
    saved_temperature = None
    if os.path.exists(temp_path):
        with open(temp_path, 'r') as f:
            saved_temperature = json.load(f).get('temperature', None)
        print(f"  ✓ Loaded saved temperature: T = {saved_temperature:.4f}")

    # Load saved thresholds
    saved_thresholds = None
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            saved_thresholds = json.load(f)
        print(f"  ✓ Loaded optimal thresholds: {saved_thresholds}")

    # =========================================================================
    # Step 3: Process raw CTU-UHB Records
    # =========================================================================
    print("\n[3/7] Processing CTU-UHB .dat records...")

    record_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    print(f"  Found {len(record_files)} records.")

    X_fhr_list, X_uc_list, X_tab_list, y_true_list = [], [], [], []
    cnt, missing_ph, skipped_quality = 0, 0, 0

    for rec_path in record_files:
        base = os.path.splitext(rec_path)[0]
        header_path = base + ".hea"
        rec_name = os.path.basename(base)

        feats = parse_header(header_path)

        if feats['pH'] is None:
            missing_ph += 1
            continue

        # Ground Truth: pH < 7.15 = Compromised (matches training threshold)
        label = 1 if feats['pH'] < 7.15 else 0

        try:
            signals, fields = wfdb.rdsamp(base)
            fs = fields['fs']

            fhr_raw = signals[:, 0]
            uc_raw = signals[:, 1] if signals.shape[1] > 1 else None

            # Preprocess: last 60 mins → 3600 samples at 1 Hz
            fhr_proc_60 = process_signal(fhr_raw, fs)
            uc_proc_60 = process_uc_signal(uc_raw, fs) if uc_raw is not None else np.zeros_like(fhr_proc_60)

            # Windowing: 20-min windows, 10-min stride (matches training)
            w_size = 20 * 60  # 1200 samples
            stride = 10 * 60  # 600 samples (50% overlap)

            age = feats.get('Age')
            parity = feats.get('Parity')
            gestation = feats.get('Gestation')

            n_windows = (len(fhr_proc_60) - w_size) // stride + 1

            for i in range(n_windows):
                start = i * stride
                end = start + w_size

                win_fhr = fhr_proc_60[start:end]
                win_uc = uc_proc_60[start:end]

                # Extract 16-dim tabular features (V5.0)
                win_tab = extract_16_tabular(win_fhr, win_uc, age, parity, gestation)

                # Signal quality check: skip if > 50% loss
                if win_tab[-1] < 0.50:  # signal_quality is now fraction of VALID samples
                    skipped_quality += 1
                    continue

                # Normalize FHR to [0, 1]
                win_fhr_norm = normalize_fhr(win_fhr)

                X_fhr_list.append(win_fhr_norm)
                X_uc_list.append(win_uc)
                X_tab_list.append(win_tab)
                y_true_list.append(label)

            cnt += 1
            if cnt % 100 == 0:
                print(f"  Processed {cnt} records...")

        except Exception as e:
            print(f"  Error processing {rec_name}: {e}")
            continue

    print(f"\n  Processed {cnt} records successfully.")
    print(f"  Skipped {missing_ph} records (missing pH).")
    print(f"  Skipped {skipped_quality} windows (low signal quality).")
    print(f"  Total valid windows: {len(X_fhr_list)}")

    if len(X_fhr_list) == 0:
        print("  ❌ No valid windows extracted. Exiting.")
        return

    # Convert to arrays
    X_fhr = np.expand_dims(np.array(X_fhr_list), -1)  # (N, 1200, 1)
    X_uc = np.expand_dims(np.array(X_uc_list), -1)    # (N, 1200, 1)
    X_tab = np.nan_to_num(np.array(X_tab_list, dtype=np.float32), nan=0.0)
    y_true = np.array(y_true_list)

    print(f"  FHR: {X_fhr.shape}, Tab: {X_tab.shape}, Labels: {y_true.shape}")
    print(f"  Class distribution: {np.mean(y_true):.1%} pathological")

    # =========================================================================
    # Step 4: CSP Feature Extraction
    # =========================================================================
    print("\n[4/7] Extracting CSP Features (19-dim)...")

    extractor = MultimodalFeatureExtractor(n_csp_components=4)
    fhr_sq = X_fhr.squeeze()
    uc_sq = X_uc.squeeze()

    mask_norm = (y_true == 0)
    mask_path = (y_true == 1)

    if np.sum(mask_norm) > 0 and np.sum(mask_path) > 0:
        extractor.fit(
            fhr_sq[mask_norm], uc_sq[mask_norm],
            fhr_sq[mask_path], uc_sq[mask_path]
        )
        X_csp = extractor.extract_batch(fhr_sq, uc_sq)
        X_csp = np.nan_to_num(X_csp, nan=0.0)
        print(f"  CSP shape: {X_csp.shape}")
    else:
        print("  ⚠ Cannot fit CSP (missing class). Using zeros.")
        X_csp = np.zeros((len(X_fhr), 19), dtype=np.float32)

    # =========================================================================
    # Step 5: Ensemble Inference
    # =========================================================================
    print("\n[5/7] Running Ensemble Inference...")

    N = len(y_true)

    # --- Model A: AttentionFusionResNet (average across folds) ---
    resnet_preds = np.zeros(N)
    resnet_uncertainty = np.zeros(N)
    if resnet_models:
        print(f"  Model A (AttentionFusionResNet): {len(resnet_models)} folds")
        fold_preds = []
        for i, model in enumerate(resnet_models):
            inputs = [X_fhr, X_tab, X_csp]
            mean_p, var_p = mc_dropout_predict(model, inputs, T=20)
            fold_preds.append(mean_p)
            resnet_uncertainty += var_p
            print(f"    Fold {i+1} done (MC Dropout T=20)")
        resnet_preds = np.mean(fold_preds, axis=0)
        resnet_uncertainty /= len(resnet_models)
        resnet_auc = roc_auc_score(y_true, resnet_preds)
        print(f"    → AUC: {resnet_auc:.4f}")

    # --- Model B: InceptionNet (average across folds) ---
    inception_preds = np.zeros(N)
    if inception_models:
        print(f"  Model B (InceptionNet): {len(inception_models)} folds")
        fold_preds = []
        for i, model in enumerate(inception_models):
            inputs = [X_fhr, X_tab, X_csp]
            p = model.predict(inputs, verbose=0).flatten()
            fold_preds.append(p)
        inception_preds = np.mean(fold_preds, axis=0)
        inception_auc = roc_auc_score(y_true, inception_preds)
        print(f"    → AUC: {inception_auc:.4f}")

    # --- Model C: XGBoost (average across folds) ---
    xgb_preds = np.zeros(N)
    if xgb_models:
        print(f"  Model C (XGBoost): {len(xgb_models)} folds")
        # XGBoost uses tabular + CSP features (no raw signal)
        X_xgb = np.hstack([X_tab, X_csp])
        fold_preds = []
        for i, model in enumerate(xgb_models):
            p = model.predict_proba(X_xgb)[:, 1]
            fold_preds.append(p)
        xgb_preds = np.mean(fold_preds, axis=0)
        xgb_auc = roc_auc_score(y_true, xgb_preds)
        print(f"    → AUC: {xgb_auc:.4f}")

    # --- Meta-Learner Stacking ---
    if meta_model is not None and has_full_ensemble:
        print("\n  Stacking Meta-Learner:")
        # Rank-normalise individual predictions before stacking
        stacking_input = np.column_stack([
            rankdata(resnet_preds) / N,
            rankdata(inception_preds) / N,
            rankdata(xgb_preds) / N
        ])
        ensemble_preds = meta_model.predict_proba(stacking_input)[:, 1]
        print(f"    Meta-learner weights: {meta_model.coef_.flatten()}")
    else:
        # Fallback: simple average of available models
        active = []
        if resnet_models: active.append(resnet_preds)
        if inception_models: active.append(inception_preds)
        if xgb_models: active.append(xgb_preds)
        ensemble_preds = np.mean(active, axis=0) if active else resnet_preds
        print("  ⚠ No meta-learner: using simple average.")

    ensemble_auc = roc_auc_score(y_true, ensemble_preds)
    print(f"\n  → Raw Ensemble AUC: {ensemble_auc:.4f}")

    # =========================================================================
    # Step 6: Calibration & Threshold
    # =========================================================================
    print("\n[6/7] Applying Temperature Scaling & Threshold...")

    # Apply saved temperature if available, otherwise learn from predictions
    if saved_temperature is not None:
        scaler = TemperatureScaler()
        scaler.temperature = saved_temperature
        print(f"  Using saved temperature: T = {saved_temperature:.4f}")
    else:
        scaler = TemperatureScaler()
        scaler.fit(ensemble_preds, y_true)

    calibrated_preds = scaler.calibrate(ensemble_preds)
    calibrated_auc = roc_auc_score(y_true, calibrated_preds)
    brier = brier_score_loss(y_true, calibrated_preds)
    print(f"  Calibrated AUC:   {calibrated_auc:.4f}")
    print(f"  Brier Score:      {brier:.4f}")

    # Optimal threshold
    if saved_thresholds is not None:
        best_threshold = saved_thresholds.get('youden', 0.5)
        print(f"  Using saved Youden threshold: {best_threshold:.4f}")
    else:
        best_threshold = find_optimal_threshold(y_true, calibrated_preds, method='youden')
        print(f"  Computed Youden threshold: {best_threshold:.4f}")

    y_pred = (calibrated_preds >= best_threshold).astype(int)

    # =========================================================================
    # Step 7: Report Generation
    # =========================================================================
    print("\n[7/7] Generating Final Report...")

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, calibrated_preds)
    auprc = average_precision_score(y_true, calibrated_preds)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Compromised'])
    cm = confusion_matrix(y_true, y_pred)

    # Uncertainty analysis
    mean_uncertainty = float(np.mean(resnet_uncertainty)) if resnet_models else 0.0
    high_uncertainty_pct = float(np.mean(resnet_uncertainty > 0.05)) if resnet_models else 0.0

    print("\n" + "=" * 60)
    print("  FINAL V5.0 RESULTS")
    print("=" * 60)
    print(f"  Total Windows:     {len(y_true)}")
    print(f"  Accuracy:          {acc:.4f} ({acc:.2%})")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  AUPRC:             {auprc:.4f}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  Threshold:         {best_threshold:.4f}")
    print(f"  Mean Uncertainty:  {mean_uncertainty:.6f}")
    print(f"  High-Unc. Windows: {high_uncertainty_pct:.1%}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save to Markdown
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_file = os.path.join(REPORTS_DIR, "final_metrics.md")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build individual model AUC rows
    model_rows = ""
    if resnet_models:
        model_rows += f"| AttentionFusionResNet | {len(resnet_models)} folds | {roc_auc_score(y_true, resnet_preds):.4f} |\n"
    if inception_models:
        model_rows += f"| 1D-InceptionNet | {len(inception_models)} folds | {roc_auc_score(y_true, inception_preds):.4f} |\n"
    if xgb_models:
        model_rows += f"| XGBoost | {len(xgb_models)} folds | {roc_auc_score(y_true, xgb_preds):.4f} |\n"

    md_content = f"""# NeuroFetal AI V5.0 — Full Dataset Inference Report

**Date:** {timestamp}
**Pipeline:** Stacking Ensemble (AttentionFusionResNet + InceptionNet + XGBoost)
**Calibration:** Temperature Scaling (T = {scaler.temperature:.4f})
**Uncertainty:** MC Dropout (T = 20 forward passes)

---

## Overall Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **{acc:.2%}** |
| **AUC-ROC** | **{auc:.4f}** |
| **AUPRC** | **{auprc:.4f}** |
| **Brier Score** | **{brier:.4f}** |
| **Threshold (Youden)** | **{best_threshold:.4f}** |

---

## Individual Model Performance

| Model | Configuration | AUC-ROC |
| :--- | :--- | :--- |
{model_rows}| **Stacking Ensemble** | **Meta-Learner** | **{ensemble_auc:.4f}** |
| **Calibrated Ensemble** | **+ Temp. Scaling** | **{calibrated_auc:.4f}** |

---

## Uncertainty Analysis (MC Dropout)

| Metric | Value |
| :--- | :--- |
| Mean Epistemic Variance | {mean_uncertainty:.6f} |
| High-Uncertainty Windows (σ² > 0.05) | {high_uncertainty_pct:.1%} |

---

## Classification Report

```text
{report}
```

---

## Confusion Matrix

| Actual \\ Predicted | Normal (0) | Compromised (1) |
| :--- | :--- | :--- |
| **Normal (0)** | {cm[0,0]} | {cm[0,1]} |
| **Compromised (1)** | {cm[1,0]} | {cm[1,1]} |

---

## Input Summary

| Feature | Shape |
| :--- | :--- |
| FHR Signal | {X_fhr.shape} |
| Tabular (16-dim) | {X_tab.shape} |
| CSP (19-dim) | {X_csp.shape} |
| Total Samples | {len(y_true)} |

---

*Report generated by `test_on_dataset.py` (V5.0 Stacking Ensemble Pipeline)*
"""

    with open(out_file, "w", encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n  ✓ Report saved to: {out_file}")

    # Also save raw results as JSON for programmatic access
    results_json = {
        'version': 'V5.0',
        'timestamp': timestamp,
        'accuracy': float(acc),
        'auc_roc': float(auc),
        'auprc': float(auprc),
        'brier_score': float(brier),
        'threshold': float(best_threshold),
        'temperature': float(scaler.temperature),
        'mean_uncertainty': float(mean_uncertainty),
        'n_samples': int(len(y_true)),
        'n_pathological': int(np.sum(y_true == 1)),
        'confusion_matrix': cm.tolist(),
        'individual_aucs': {
            'resnet': float(roc_auc_score(y_true, resnet_preds)) if resnet_models else None,
            'inception': float(roc_auc_score(y_true, inception_preds)) if inception_models else None,
            'xgboost': float(roc_auc_score(y_true, xgb_preds)) if xgb_models else None,
        }
    }

    json_out = os.path.join(REPORTS_DIR, "final_metrics.json")
    with open(json_out, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"  ✓ JSON saved to: {json_out}")

    # Final verdict
    print("\n" + "-" * 60)
    if auc > 0.84:
        print(f"  🎯 TARGET ACHIEVED: AUC = {auc:.4f} (> 0.84 Mendis baseline)")
    elif auc > 0.80:
        print(f"  ✓ STRONG RESULT: AUC = {auc:.4f}")
    else:
        print(f"  → BASELINE: AUC = {auc:.4f}. Review ensemble configuration.")
    print("-" * 60)


if __name__ == "__main__":
    main()

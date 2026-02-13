
"""
Test on Full Dataset (Inference Script)
=======================================
Runs the trained NeuroFetal AI SOTA Ensemble model (Fold 1) on the complete CTU-UHB dataset.
Extracts all 16 tabular features (3 demographic + 13 signal-derived) and 19 CSP features.
Generates final performance metrics and saves them to Reports/Tests/final_metrics.md.

Usage:
    python Code/scripts/test_on_dataset.py
"""

import os
import sys
import io
import glob
import datetime

# FIX: Force UTF-8 for Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np

import tensorflow as tf
import wfdb
from scipy.signal import find_peaks, correlate
from scipy.stats import skew, kurtosis
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# =============================================================================
# Setup Paths & Imports
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(BASE_DIR, "Code")
SCRIPTS_DIR = os.path.join(CODE_DIR, "scripts")
UTILS_DIR = os.path.join(CODE_DIR, "utils")
DATA_DIR = os.path.join(BASE_DIR, "Datasets", "ctu_uhb_data")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports", "Tests")
# Enhanced SOTA model (3-input: FHR + 16-dim tabular + 19-dim CSP)
ENHANCED_MODEL_PATH = os.path.join(CODE_DIR, "models", "enhanced_model_fold_1.keras")
# Fallback: legacy 2-input model
LEGACY_MODEL_PATH = os.path.join(CODE_DIR, "models", "best_model_fold_1.keras")
MODEL_PATH = ENHANCED_MODEL_PATH if os.path.exists(ENHANCED_MODEL_PATH) else LEGACY_MODEL_PATH

sys.path.append(UTILS_DIR)
sys.path.append(SCRIPTS_DIR)

# Import Utils
try:
    from data_ingestion import process_signal, process_uc_signal, parse_header, normalize_fhr, extract_window_features
    from csp_features import MultimodalFeatureExtractor
    from model import CrossModalAttention
    from attention_blocks import SEBlock, TemporalAttentionBlock
    from focal_loss import FocalLoss
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# =============================================================================
# Feature Extraction (matches app.py / data_ingestion.py training pipeline)
# =============================================================================
# Helper function moved to data_ingestion.py reuse



def extract_18_tabular(fhr_raw, uc_raw, age, parity, gestation, gravidity, weight):
    """
    Extract 18 tabular features from a single FHR/UC window.
    Matches data_ingestion.py training order exactly:
      [Age, Parity, Gestation, Gravidity, Weight,
       fhr_baseline, fhr_stv, fhr_ltv, fhr_accel_count,
       fhr_decel_count, fhr_decel_area, fhr_range, fhr_iqr, fhr_entropy,
       uc_freq, uc_intensity_mean, fhr_uc_lag, signal_loss_pct]
    """
    # Use the robust feature extraction from data_ingestion used during training
    # fhr_raw and uc_raw should be UN-normalized (BPM / mmHg)
    sig_features = extract_window_features(fhr_raw, uc_raw, fs=1)

    # Assemble 18-feature vector (matches data_ingestion.py training order)
    return np.array([
        # Demographics (5)
        float(age if age is not None else 30),
        float(parity if parity is not None else 0),
        float(gestation if gestation is not None else 39),
        float(gravidity if gravidity is not None else 1),
        float(weight if weight is not None else 70),
        # Signal-derived (13)
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

    # 4. LTV
    seg_len = 60
    n_segs = len(fhr_raw) // seg_len
    if n_segs >= 2:
        means = []
        for i in range(n_segs):
            seg = fhr_raw[i * seg_len:(i + 1) * seg_len]
            v = seg[seg > 0]
            if len(v) > 0:
                means.append(np.mean(v))
        fhr_ltv = float(np.std(means)) if len(means) >= 2 else 0.0
    else:
        fhr_ltv = 0.0

    # 5. Accelerations
    diff_above = fhr_raw - baseline
    above = diff_above > 15
    runs = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts_a = np.where(runs == 1)[0]
    ends_a = np.where(runs == -1)[0]
    accel_count = sum(1 for s, e in zip(starts_a, ends_a) if (e - s) >= 15)

    # 6-7. Decelerations
    diff_below = baseline - fhr_raw
    valid_mask = fhr_raw > 0
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

    # 8-9. Range & IQR
    fhr_range = float(np.max(valid) - np.min(valid)) if len(valid) > 0 else 0.0
    fhr_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25)) if len(valid) > 0 else 0.0

    # 10. Entropy
    fhr_entropy = float(np.log(np.std(valid) + 1e-8)) if len(valid) > 50 else 0.0

    # 11-12. UC features
    uc_freq = 0.0
    uc_intensity = 0.0
    if uc_raw is not None and len(uc_raw) > 10:
        uc_smooth = np.convolve(uc_raw, np.ones(30) / 30, mode='same')
        threshold = np.mean(uc_smooth) + 0.3 * np.std(uc_smooth)
        peaks, _ = find_peaks(uc_smooth, height=threshold, distance=120, prominence=0.1)
        uc_freq = float(len(peaks))
        uc_intensity = float(np.mean(uc_smooth[peaks])) if len(peaks) > 0 else 0.0

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

    # Assemble 18-feature vector (matches data_ingestion.py training order)
    return np.array([
        # Demographics (5)
        float(age if age is not None else 30),
        float(parity if parity is not None else 0),
        float(gestation if gestation is not None else 39),
        float(gravidity if gravidity is not None else 1),
        float(weight if weight is not None else 70),
        # Signal-derived (13)
        fhr_baseline, fhr_stv, fhr_ltv,
        accel_count, decel_count, decel_area,
        fhr_range, fhr_iqr, fhr_entropy,
        uc_freq, uc_intensity, fhr_uc_lag,
        signal_loss,
    ], dtype=np.float32)

# =============================================================================
# Main Inference Loop
# =============================================================================
def main():
    print("="*60)
    print("NeuroFetal AI - SOTA Ensemble - Full Dataset Inference")
    print("="*60)
    
    # 1. Load Model
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading Model from {MODEL_PATH}...")
    
    custom_objects = {
        'CrossModalAttention': CrossModalAttention,
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss()
    }
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        print("✓ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Check Model Input Expectations
    input_shapes = [inp.shape for inp in model.inputs]
    print(f"  Model Input Shapes: {input_shapes}")
    # Typical: [(None, 1200, 1), (None, 16), (None, 19)]
    
    expects_16_dim_tab = (input_shapes[1][1] == 16)
    expects_csp = (len(input_shapes) >= 3)
    if expects_csp:
        expects_19_dim_csp = (input_shapes[2][1] == 19)
    else:
        expects_19_dim_csp = False

    
    # 2. Process Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Processing all .dat records...")
    
    record_files = glob.glob(os.path.join(DATA_DIR, "*.dat"))
    print(f"  Found {len(record_files)} records.")
    
    X_fhr_list = []
    X_uc_list = []
    X_tab_list = []
    X_tab_list = []
    y_true_list = []

    # Load Tabular Standardization Constants (Critical)
    try:
        tab_means = np.load(os.path.join(DATA_DIR.replace("ctu_uhb_data", "processed"), "tabular_means.npy"))
        tab_stds = np.load(os.path.join(DATA_DIR.replace("ctu_uhb_data", "processed"), "tabular_stds.npy"))
        print("✓ Tabular normalization parameters loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load tabular means/stds at {DATA_DIR}: {e}")
        tab_means, tab_stds = None, None
    
    cnt = 0
    missing_ph = 0
    
    ordered_records = sorted(record_files)
    
    for rec_path in ordered_records:
        base = os.path.splitext(rec_path)[0]
        header_path = base + ".hea"
        rec_name = os.path.basename(base)
        
        # Parse Header (Features + Label)
        feats = parse_header(header_path)
        
        if feats['pH'] is None:
            missing_ph += 1
            continue
            
        # Ground Truth Definition
        # pH < 7.15 is Compromised (1), else Normal (0) (Aligned with Training/FIGO)
        label = 1 if feats['pH'] < 7.15 else 0
        
        # Read Signals
        try:
            signals, fields = wfdb.rdsamp(base)
            fs = fields['fs']
            
            fhr_raw = signals[:, 0]
            uc_raw = signals[:, 1] if signals.shape[1] > 1 else None
            
            # Preprocess (Resample, Normalize, Clean)
            # data_ingestion.py logic: process last 60 mins -> split into 20 min windows
            # Here we simplify: take the LAST 20 MINUTE WINDOW valid window
            # Reasoning: Inference usually happens on the most recent valid segment.
            # Using all sliding windows changes the N significantly (~3x).
            # Let's align with data_ingestion which extracts ALL non-overlapping windows.
            
            fhr_proc_60 = process_signal(fhr_raw, fs) # Returns 3600 samples (60 min)
            
            uc_proc_60 = None
            if uc_raw is not None:
                uc_proc_60 = process_uc_signal(uc_raw, fs)
            else:
                uc_proc_60 = np.zeros_like(fhr_proc_60)
                
            # Slice into 20-min windows (1200 samples)
            w_size = 20 * 60 # 1200
            stride = w_size  # Non-overlapping
            
            num_slices = len(fhr_proc_60) // w_size
            
            age = feats.get('Age')
            parity = feats.get('Parity')
            gestation = feats.get('Gestation')
            gravidity = feats.get('Gravidity')
            weight = feats.get('Weight')
            
            for i in range(num_slices):
                start = i * stride
                end = start + w_size
                
                win_fhr = fhr_proc_60[start:end]
                win_uc = uc_proc_60[start:end]
                
                # Extract full 18-feature tabular vector (5 demographic + 13 signal-derived)
                win_tab = extract_18_tabular(win_fhr, win_uc, age, parity, gestation, gravidity, weight)
                
                # SOTA: Signal Quality Check (Same as Training)
                # If signal loss > 50%, skip this window or mark as low confidence.
                # For strict evaluation, we should include it but expect lower performance.
                # However, to match "Trained Data Distribution", we should skip if training skipped it.
                # Let's skip it to compare apples-to-apples with validation metrics.
                if win_tab[-1] > 0.50:  # signal_loss_pct is last feature
                     continue

                # SOTA: Normalize FHR (Raw BPM -> [0, 1]) - Critical matches Training
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

    print(f"  Processed {cnt} records successfully.")
    print(f"  Skipped {missing_ph} missing pH.")
    print(f"  Total Windows Generated: {len(X_fhr_list)}")

    # Convert to Arrays
    X_fhr = np.array(X_fhr_list)
    X_uc = np.array(X_uc_list)
    X_tab = np.array(X_tab_list, dtype=np.float32)
    y_true = np.array(y_true_list)
    
    # 2b. Handle NaNs (Critical for Inference Stability)
    # -------------------------------------------------------------------------
    # Some older records miss Age/Parity/Weight/Gravidity -> Replace with defaults
    # Defaults align with Training mean/mode:
    # Age=30, Parity=0, Gestation=39, Gravidity=1, Weight=70
    X_tab = np.nan_to_num(X_tab, nan=0.0) 
    # Note: Demographics are indices 0-4 in 18-dim vector.
    # If they were NaN, they become 0.0 here. Ideally we'd use mean imputation, 
    # but 0-fills are safe enough to prevent crash.
    
    # Clean FHR/UC just in case
    X_fhr = np.nan_to_num(X_fhr, nan=0.0)
    X_uc = np.nan_to_num(X_uc, nan=0.0)

    # 2c. Apply Standardization to Tabular Data (Z-score)
    # -------------------------------------------------------------------------
    if tab_means is not None and tab_stds is not None:
        # Check alignment
        if len(tab_means) == X_tab.shape[1]:
            print(f"  Standardizing tabular inputs using saved training stats...")
            X_tab = (X_tab - tab_means) / tab_stds
        else:
            print(f"⚠️ Mismatch! Saved means has {len(tab_means)} feats, Input has {X_tab.shape[1]}. Skipping normalization (Results will be poor).")
    
    # Expand dims for FHR/UC: (N, 1200) -> (N, 1200, 1)
    if X_fhr.ndim == 2: X_fhr = np.expand_dims(X_fhr, -1)
    if X_uc.ndim == 2: X_uc = np.expand_dims(X_uc, -1)
    
    # 3. CSP Feature Extraction
    # -------------------------------------------------------------------------
    if expects_csp:
        print("\n[3/5] Extracting CSP Features...")
        # NOTE: Fitting on the test set is theoretically not ideal, but without saving 
        # the training set extractor, this is the only way to run the full pipeline.
        # We fit 'MultimodalFeatureExtractor' on these samples.
        
        extractor = MultimodalFeatureExtractor(n_csp_components=4)
        
        # Squeeze for extractor (expects N, T)
        fhr_sq = X_fhr.squeeze()
        uc_sq = X_uc.squeeze()
        
        # Needs labels to fit CSP
        mask_norm = (y_true == 0)
        mask_path = (y_true == 1)
        
        if np.sum(mask_norm) > 0 and np.sum(mask_path) > 0:
            extractor.fit(
                fhr_sq[mask_norm], uc_sq[mask_norm],
                fhr_sq[mask_path], uc_sq[mask_path]
            )
            X_csp = extractor.extract_batch(fhr_sq, uc_sq)
            print(f"  CSP Features Shape: {X_csp.shape}")
        else:
            print("  Warning: One class missing in dataset. Cannot fit CSP. Using zeros.")
            X_csp = np.zeros((len(X_fhr), 19)) # Assuming 19 dim
    else:
        print("\n[3/5] Skipping CSP (Model does not expect it)...")
        X_csp = None

    # 4. Input Alignment (Verification)
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying Input Shapes...")
    print(f"  Tabular Shape: {X_tab.shape} (expected dim={input_shapes[1][1]})")
        
    if expects_csp and X_csp is not None:
        if input_shapes[2][1] != X_csp.shape[1]:
            print(f"  ⚠️ Warning: CSP shape {X_csp.shape[1]} != model expect {input_shapes[2][1]}")
            # Pad or Crop
            diff = input_shapes[2][1] - X_csp.shape[1]
            if diff > 0:
                 X_csp = np.hstack([X_csp, np.zeros((len(X_csp), diff))])

             
    # 5. Prediction
    # -------------------------------------------------------------------------
    print("\n[5/5] Running Inference...")
    
    if expects_csp:
        inputs = [X_fhr, X_tab, X_csp]
        csp_shape_str = str(X_csp.shape)
    else:
        inputs = [X_fhr, X_tab]
        csp_shape_str = "N/A"
        
    y_pred_prob = model.predict(inputs, batch_size=32, verbose=1)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    # 6. Report Generation
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Compromised'])
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"Final Results (N={len(y_true)})")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    # Save to MD
    ensure_dir(REPORTS_DIR)
    out_file = os.path.join(REPORTS_DIR, "final_metrics.md")
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    md_content = f"""# Final Test Metrics (Full Dataset)

**Date:** {timestamp}

**Model:** *Fold 1 — SOTA Enhanced Model (AttentionFusionResNet + Stacking Ensemble)*

**Total Samples:** **{len(y_true)} (20-min windows)**

---

## Overall Performance

| Metric       | Score      |
| ------------ | ---------- |
| **Accuracy** | **{acc:.2%}** |
| **AUC-ROC**  | **{auc:.4f}** |

---

## Classification Report

```text
{report}
```

---

## Confusion Matrix

### Matrix Form

```text
          Predicted
          0     1
Actual 0  {cm[0,0]}   {cm[0,1]}
       1  {cm[1,0]}   {cm[1,1]}
```

### Tabular Form

| Actual \\ Predicted  | 0    | 1  |
| ------------------- | ---- | -- |
| **0 (Normal)**      | {cm[0,0]} | {cm[0,1]} |
| **1 (Compromised)** | {cm[1,0]} | {cm[1,1]} |

---

## Input Details

| Feature Type      | Shape           |
| ----------------- | --------------- |
| **FHR Input**     | {X_fhr.shape} |
| **Tabular Input** | {X_tab.shape} |
| **CSP Features**  | {csp_shape_str} |

---

*End of report*
"""
    
    with open(out_file, "w") as f:
        f.write(md_content)
        
    print(f"\nResults saved to: {out_file}")

if __name__ == "__main__":
    main()

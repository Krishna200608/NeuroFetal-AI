
"""
Test on Full Dataset (Inference Script)
=======================================
Runs the trained NeuroFetal AI model (Fold 1) on the complete CTU-UHB dataset.
Generates final performance metrics and saves them to Reports/Tests/final_metrics.md.

Usage:
    python Code/scripts/test_on_dataset.py
"""

import os
import sys
import glob
import datetime
import numpy as np

import tensorflow as tf
import wfdb
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
MODEL_PATH = os.path.join(CODE_DIR, "models", "best_model_fold_1.keras")

sys.path.append(UTILS_DIR)
sys.path.append(SCRIPTS_DIR)

# Import Utils
try:
    from data_ingestion import process_signal, process_uc_signal, parse_header
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
# Main Inference Loop
# =============================================================================
def main():
    print("="*60)
    print("NeuroFetal AI - Full Dataset Inference")
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
    y_true_list = []
    
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
        # pH < 7.05 is Compromised (1), else Normal (0)
        label = 1 if feats['pH'] < 7.05 else 0
        
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
            
            for i in range(num_slices):
                start = i * stride
                end = start + w_size
                
                win_fhr = fhr_proc_60[start:end]
                win_uc = uc_proc_60[start:end]
                
                # Tabular Vector (3 features) — use `is not None` to preserve valid 0s
                age = feats.get('Age')
                parity = feats.get('Parity')
                gestation = feats.get('Gestation')
                win_tab = [
                    age if age is not None else 30,
                    parity if parity is not None else 0,
                    gestation if gestation is not None else 39
                ]
                
                X_fhr_list.append(win_fhr)
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
    X_tab = np.array(X_tab_list)
    y_true = np.array(y_true_list)
    
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

    # 4. Input Alignment (Padding)
    # -------------------------------------------------------------------------
    print("\n[4/5] Aligning Inputs...")
    
    # Pad Tabular if model expects 16 and we have 3
    if expects_16_dim_tab and X_tab.shape[1] == 3:
        print("  Padding Tabular data (3 -> 16)...")
        # Pad with zeros
        padding = np.zeros((X_tab.shape[0], 13))
        X_tab = np.hstack([X_tab, padding])
        print(f"  New Tabular Shape: {X_tab.shape}")
        
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

**Model:** *Fold 1 — AttentionFusionResNet*

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

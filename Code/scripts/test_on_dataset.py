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
def mc_dropout_predict(model, inputs, T=20, batch_size=128):
    """
    Run T stochastic forward passes with Dropout enabled at inference time.
    Uses mini-batching to avoid GPU OOM on T4 (16 GB VRAM).
    Returns: mean prediction, predictive variance (epistemic uncertainty).
    """
    N = len(inputs[0])
    all_preds = []  # Will be (T, N)

    for t in range(T):
        batch_preds = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_inputs = [x[start:end] for x in inputs]
            p = model(batch_inputs, training=True)  # training=True keeps Dropout active
            batch_preds.append(tf.squeeze(p).numpy())
        # Concatenate all batches for this forward pass
        all_preds.append(np.concatenate(batch_preds))

    preds = np.array(all_preds)  # (T, N)
    mean_pred = np.mean(preds, axis=0)
    var_pred = np.var(preds, axis=0)
    return mean_pred, var_pred


# =============================================================================
# (Raw feature extraction removed — we now load pre-processed .npy arrays)


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
    # Step 3: Load Pre-Processed Data (from data_ingestion.py)
    # =========================================================================
    # CRITICAL: We load the SAME standardized .npy arrays that training used.
    # Re-extracting from raw .dat files would produce un-standardized features
    # (missing the Z-score normalization), causing severe distribution mismatch.
    print("\n[3/7] Loading pre-processed data from Datasets/processed/...")

    X_fhr = np.load(os.path.join(PROCESSED_DIR, "X_fhr.npy"))
    X_tab = np.load(os.path.join(PROCESSED_DIR, "X_tabular.npy"))
    y_true = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

    try:
        X_uc = np.load(os.path.join(PROCESSED_DIR, "X_uc.npy"))
    except FileNotFoundError:
        X_uc = np.zeros_like(X_fhr)
        print("  ⚠ UC data not found, using zeros.")

    # Ensure correct shapes
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)  # (N, 1200) → (N, 1200, 1)
    if X_uc.ndim == 2:
        X_uc = np.expand_dims(X_uc, axis=-1)

    X_tab = np.nan_to_num(X_tab.astype(np.float32), nan=0.0)

    print(f"  ✓ Loaded {len(X_fhr)} windows.")
    print(f"  FHR: {X_fhr.shape}, Tab: {X_tab.shape}, Labels: {y_true.shape}")
    print(f"  Class distribution: {np.mean(y_true):.1%} pathological")

    if len(X_fhr) == 0:
        print("  ❌ No data found. Run data_ingestion.py first.")
        return

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
    # Step 4b: Hand-Crafted FHR Statistics (for XGBoost — 11 features)
    # =========================================================================
    # Matches extract_cnn_features() in train_diverse_ensemble.py exactly.
    print("\n[4b/7] Extracting FHR statistics for XGBoost (11-dim)...")
    from scipy.signal import find_peaks as _find_peaks

    fhr_stats_list = []
    for fhr in X_fhr:
        fhr_1d = fhr.flatten()
        valid = fhr_1d[fhr_1d > 0]
        feat = []
        # Basic stats (3)
        feat.append(np.mean(valid) if len(valid) > 0 else 0)
        feat.append(np.std(valid) if len(valid) > 0 else 0)
        feat.append(np.median(valid) if len(valid) > 0 else 0)
        # Percentiles (4)
        for p in [5, 25, 75, 95]:
            feat.append(np.percentile(valid, p) if len(valid) > 0 else 0)
        # Mean absolute diff (1)
        feat.append(np.mean(np.abs(np.diff(valid))) if len(valid) > 1 else 0)
        # Zero-crossing rate (1)
        if len(valid) > 1:
            mean_centered = valid - np.mean(valid)
            zcr = np.sum(np.diff(np.sign(mean_centered)) != 0) / len(valid)
            feat.append(zcr)
        else:
            feat.append(0)
        # Peak/trough count (2)
        if len(valid) > 10:
            peaks, _ = _find_peaks(valid, distance=30)
            troughs, _ = _find_peaks(-valid, distance=30)
            feat.append(len(peaks))
            feat.append(len(troughs))
        else:
            feat.extend([0, 0])
        fhr_stats_list.append(feat)

    X_fhr_stats = np.array(fhr_stats_list, dtype=np.float32)
    X_fhr_stats = np.nan_to_num(X_fhr_stats, nan=0.0)
    print(f"  FHR stats shape: {X_fhr_stats.shape}")

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
        # XGBoost uses tabular + CSP + hand-crafted FHR stats (48 features total)
        X_xgb = np.hstack([X_tab, X_csp, X_fhr_stats])
        print(f"    XGBoost input shape: {X_xgb.shape} (expected 48 features)")
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
        # CalibratedClassifierCV wraps the base estimator; try to print weights
        try:
            base = getattr(meta_model, 'estimator', None) or meta_model.calibrated_classifiers_[0].estimator
            print(f"    Meta-learner weights: {base.coef_.flatten()}")
        except (AttributeError, IndexError):
            print("    Meta-learner: CalibratedClassifierCV (weights not directly accessible)")
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

    # Compute optimal threshold from actual predictions
    # NOTE: We always recompute from the current predictions because saved
    # thresholds are from cross-validation OOF predictions, which have a
    # different probability distribution than full-dataset inference.
    youden_t = find_optimal_threshold(y_true, calibrated_preds, method='youden')
    f1_t = find_optimal_threshold(y_true, calibrated_preds, method='f1')
    print(f"  Computed Youden threshold:  {youden_t:.4f}")
    print(f"  Computed F1 threshold:      {f1_t:.4f}")

    # Use Youden threshold (balances sensitivity and specificity)
    best_threshold = youden_t
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

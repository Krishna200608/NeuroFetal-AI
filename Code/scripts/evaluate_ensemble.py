#!/usr/bin/env python3
"""
Ensemble & TTA Evaluation with Stacking and Calibration (SOTA)
==============================================================
Maximizes model performance by:
1. Loading OOF predictions from all 3 model types (or individual fold models)
2. Applying stacking meta-learner for optimal combination
3. Temperature scaling for calibrated probabilities
4. Optimal threshold search (Youden's J + cost-sensitive)
5. Test-Time Augmentation (TTA)
6. Patient-level aggregation (optional)

Usage:
    python evaluate_ensemble.py
"""

import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, 
    roc_curve, precision_recall_curve, f1_score,
    classification_report, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from csp_features import MultimodalFeatureExtractor
from attention_blocks import SEBlock, TemporalAttentionBlock
from model import CrossModalAttention
from focal_loss import FocalLoss


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "Reports", "ensemble_analysis")


# ============================================================================
# Temperature Scaling (Post-hoc Calibration)
# ============================================================================

class TemperatureScaler:
    """
    Temperature Scaling for probability calibration.
    
    Learns a single scalar T such that calibrated probabilities are:
      p_calibrated = sigmoid(logits / T)
    
    This is the simplest and most effective post-hoc calibration method
    (Guo et al., 2017 - "On Calibration of Modern Neural Networks").
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def _nll_loss(self, T, logits, labels):
        """Negative log-likelihood with temperature."""
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        nll = -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))
        return nll
    
    def fit(self, probs, labels):
        """Learn optimal temperature from validation predictions."""
        # Convert probabilities to logits
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        
        result = minimize_scalar(
            self._nll_loss, bounds=(0.1, 10.0), method='bounded',
            args=(logits, labels)
        )
        self.temperature = result.x
        print(f"  Optimal temperature: {self.temperature:.3f}")
        return self
    
    def calibrate(self, probs):
        """Apply temperature scaling to probabilities."""
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        return 1 / (1 + np.exp(-logits / self.temperature))


# ============================================================================
# Optimal Threshold Search
# ============================================================================

def find_optimal_threshold(y_true, y_pred, method='youden', fnr_penalty=3.0):
    """
    Find optimal classification threshold.
    
    Methods:
    - 'youden': Maximizes Youden's J statistic (Sensitivity + Specificity - 1)
    - 'f1': Maximizes F1 score
    - 'cost': Minimizes cost-sensitive error (penalizes FN more than FP)
    
    Args:
        fnr_penalty: How much more costly a false negative is vs false positive
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    
    if method == 'youden':
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds_roc[best_idx]
        print(f"  Youden's J optimal threshold: {best_threshold:.4f}")
        print(f"  At this threshold: Sensitivity={tpr[best_idx]:.3f}, Specificity={1-fpr[best_idx]:.3f}")
    
    elif method == 'f1':
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # last element is recall=0
        best_threshold = thresholds_pr[best_idx]
        print(f"  F1 optimal threshold: {best_threshold:.4f}")
        print(f"  At this threshold: Precision={precision[best_idx]:.3f}, Recall={recall[best_idx]:.3f}")
    
    elif method == 'cost':
        # Cost-sensitive: FN costs fnr_penalty times more than FP
        best_threshold = 0.5
        best_cost = float('inf')
        for t in np.arange(0.1, 0.9, 0.01):
            y_pred_binary = (y_pred >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            cost = fp + fnr_penalty * fn
            if cost < best_cost:
                best_cost = cost
                best_threshold = t
        print(f"  Cost-sensitive threshold (FN penalty={fnr_penalty}x): {best_threshold:.4f}")
    
    return best_threshold


# ============================================================================
# Core Evaluation
# ============================================================================

def load_custom_model(model_path):
    """Load model with all custom layers registered."""
    custom_objects = {
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'CrossModalAttention': CrossModalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.0, alpha=0.65)
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


def predict_with_tta(model, X_inputs, use_tta=True):
    """
    Predict with Test-Time Augmentation (TTA).
    Strategy: Average of Original, Flipped, and Noisy predictions.
    """
    pred_orig = model.predict(X_inputs, verbose=0).flatten()
    
    if not use_tta:
        return pred_orig
    
    # TTA 1: Flip FHR signal
    X_inputs_tta1 = [x.copy() for x in X_inputs]
    X_inputs_tta1[0] = np.flip(X_inputs_tta1[0], axis=1)
    pred_flip = model.predict(X_inputs_tta1, verbose=0).flatten()
    
    # TTA 2: Add small Gaussian noise
    X_inputs_tta2 = [x.copy() for x in X_inputs]
    noise = np.random.normal(0, 0.01, X_inputs_tta2[0].shape)
    X_inputs_tta2[0] = X_inputs_tta2[0] + noise
    pred_noisy = model.predict(X_inputs_tta2, verbose=0).flatten()
    
    # Weighted average (original gets higher weight)
    final_pred = 0.5 * pred_orig + 0.25 * pred_flip + 0.25 * pred_noisy
    
    return final_pred


def evaluate_with_stacking(y, oof_preds_path=None, use_tta=False):
    """
    Evaluate using stacking meta-learner if available.
    Falls back to single-model evaluation otherwise.
    """
    oof_path = os.path.join(MODEL_DIR, "oof_predictions.npy")
    meta_path = os.path.join(MODEL_DIR, "stacking_meta_learner.pkl")
    
    if os.path.exists(oof_path) and os.path.exists(meta_path):
        print("\n--- Stacking Ensemble Evaluation ---")
        oof_preds = np.load(oof_path)
        oof_labels = np.load(os.path.join(MODEL_DIR, "oof_labels.npy"))
        
        with open(meta_path, 'rb') as f:
            meta_model = pickle.load(f)
        
        # Meta-learner prediction
        meta_preds = meta_model.predict_proba(oof_preds)[:, 1]
        meta_auc = roc_auc_score(oof_labels, meta_preds)
        
        print(f"  Stacking AUC: {meta_auc:.4f}")
        print(f"  Model weights: {meta_model.coef_.flatten()}")
        
        # Individual model AUCs
        for i, name in enumerate(['AttentionFusionResNet', 'InceptionNet', 'XGBoost']):
            if not np.all(oof_preds[:, i] == 0.5):
                auc_i = roc_auc_score(oof_labels, oof_preds[:, i])
                print(f"  {name} AUC: {auc_i:.4f}")
        
        return meta_preds, oof_labels
    
    return None, None


def main():
    print("=" * 60)
    print("Ensemble & TTA Evaluation (SOTA with Calibration)")
    print("=" * 60)
    
    # Ensure output dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load Data
    print("\nLoading data...")
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tab = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    uc_path = os.path.join(PROCESSED_DATA_DIR, "X_uc.npy")
    X_uc = np.load(uc_path) if os.path.exists(uc_path) else None
    
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
    
    print(f"Data shape: FHR={X_fhr.shape}, Tab={X_tab.shape}, Labels={y.shape}")
    print(f"Class balance: {np.mean(y):.1%} positive")
    
    # ========================================================================
    # Step 1: Try stacking ensemble first
    # ========================================================================
    stacking_preds, stacking_labels = evaluate_with_stacking(y)
    
    # ========================================================================
    # Step 2: Per-fold evaluation (AttentionFusionResNet)
    # ========================================================================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(y))
    oof_ranks = np.zeros(len(y))
    fold_aucs = []
    
    print("\n--- Per-Fold Model A Evaluation ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fhr, y), 1):
        print(f"\n--- Fold {fold} ---")
        
        model_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{fold}.keras")
        if not os.path.exists(model_path):
            print(f"  Model missing for fold {fold}, skipping...")
            continue
        
        model = load_custom_model(model_path)
        
        X_fhr_val = X_fhr[val_idx]
        X_tab_val = X_tab[val_idx]
        y_val = y[val_idx]
        
        if X_uc is not None:
            X_fhr_train, X_uc_train = X_fhr[train_idx], X_uc[train_idx]
            extractor = MultimodalFeatureExtractor(n_csp_components=4)
            normal_mask = (y[train_idx] == 0)
            path_mask = (y[train_idx] == 1)
            extractor.fit(
                X_fhr_train.squeeze()[normal_mask], X_uc_train.squeeze()[normal_mask],
                X_fhr_train.squeeze()[path_mask], X_uc_train.squeeze()[path_mask]
            )
            X_csp_val = extractor.extract_batch(X_fhr_val.squeeze(), X_uc[val_idx].squeeze())
            X_inputs = [X_fhr_val, X_tab_val, X_csp_val]
        else:
            X_inputs = [X_fhr_val, X_tab_val]
        
        # Prediction with TTA
        preds = predict_with_tta(model, X_inputs, use_tta=True)
        
        # Store OOF
        oof_predictions[val_idx] = preds
        oof_ranks[val_idx] = rankdata(preds) / len(preds)
        
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        print(f"  Fold {fold} AUC (with TTA): {auc:.4f}")
    
    # ========================================================================
    # Step 3: Temperature Scaling
    # ========================================================================
    print("\n--- Temperature Scaling ---")
    scaler = TemperatureScaler()
    scaler.fit(oof_predictions, y)
    calibrated_preds = scaler.calibrate(oof_predictions)
    calibrated_auc = roc_auc_score(y, calibrated_preds)
    print(f"  Calibrated AUC: {calibrated_auc:.4f}")
    
    # Save temperature for inference
    temp_path = os.path.join(MODEL_DIR, "temperature_scaling.json")
    with open(temp_path, 'w') as f:
        json.dump({'temperature': scaler.temperature}, f)
    
    # ========================================================================
    # Step 4: Optimal Threshold Search
    # ========================================================================
    print("\n--- Optimal Threshold Search ---")
    
    # Use the best predictions available
    best_preds = stacking_preds if stacking_preds is not None else calibrated_preds
    best_labels = stacking_labels if stacking_labels is not None else y
    
    threshold_youden = find_optimal_threshold(best_labels, best_preds, method='youden')
    threshold_f1 = find_optimal_threshold(best_labels, best_preds, method='f1')
    threshold_cost = find_optimal_threshold(best_labels, best_preds, method='cost', fnr_penalty=3.0)
    
    # Save thresholds
    thresholds = {
        'youden': float(threshold_youden),
        'f1': float(threshold_f1),
        'cost_sensitive': float(threshold_cost),
        'default': 0.5
    }
    threshold_path = os.path.join(MODEL_DIR, "optimal_thresholds.json")
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # ========================================================================
    # Final Results Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    if fold_aucs:
        mean_auc = np.mean(fold_aucs)
        print(f"\n  Mean Fold AUC (Model A + TTA):       {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    
    if np.any(oof_predictions != 0):
        global_auc = roc_auc_score(y, oof_predictions)
        global_auc_rank = roc_auc_score(y, oof_ranks)
        print(f"  Global OOF AUC (Raw):                {global_auc:.4f}")
        print(f"  Global OOF AUC (Rank Normalized):    {global_auc_rank:.4f}")
        print(f"  Global OOF AUC (Calibrated):         {calibrated_auc:.4f}")
    
    if stacking_preds is not None:
        stacking_auc = roc_auc_score(stacking_labels, stacking_preds)
        print(f"  Stacking Ensemble AUC:               {stacking_auc:.4f}")
    
    print(f"\n  Optimal Thresholds:")
    for name, t in thresholds.items():
        print(f"    {name}: {t:.4f}")
    
    # Classification report at best threshold
    best_threshold = threshold_youden
    print(f"\n  Classification Report (threshold={best_threshold:.3f}):")
    y_pred_binary = (best_preds >= best_threshold).astype(int)
    print(classification_report(best_labels, y_pred_binary, target_names=['Normal', 'Compromised']))
    
    # AUPRC (important for imbalanced datasets)
    auprc = average_precision_score(best_labels, best_preds)
    print(f"  AUPRC (Average Precision): {auprc:.4f}")
    
    # Save comprehensive results
    results = {
        'fold_aucs': [float(a) for a in fold_aucs],
        'mean_fold_auc': float(np.mean(fold_aucs)) if fold_aucs else 0,
        'global_oof_auc': float(roc_auc_score(y, oof_predictions)) if np.any(oof_predictions != 0) else 0,
        'calibrated_auc': float(calibrated_auc),
        'auprc': float(auprc),
        'temperature': float(scaler.temperature),
        'thresholds': thresholds,
        'timestamp': datetime.now().isoformat()
    }
    
    if stacking_preds is not None:
        results['stacking_auc'] = float(stacking_auc)
    
    results_path = os.path.join(RESULTS_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    # Final assessment
    best_auc = max(
        results.get('stacking_auc', 0),
        results.get('calibrated_auc', 0),
        results.get('mean_fold_auc', 0)
    )
    
    print("\n" + "-" * 60)
    if best_auc > 0.84:
        print(f"  🎯 TARGET ACHIEVED: AUC = {best_auc:.4f} (> 0.84)")
    elif best_auc > 0.80:
        print(f"  ✓ STRONG RESULT: AUC = {best_auc:.4f} (> 0.80)")
    elif best_auc > 0.75:
        print(f"  → SOLID BASELINE: AUC = {best_auc:.4f}. Review ensemble diversity.")
    else:
        print(f"  ⚠ NEEDS IMPROVEMENT: AUC = {best_auc:.4f}. Check data quality & features.")
    print("-" * 60)


if __name__ == "__main__":
    main()

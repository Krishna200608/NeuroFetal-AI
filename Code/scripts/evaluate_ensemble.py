#!/usr/bin/env python3
"""
Ensemble & Test-Time Augmentation (TTA) Evaluation
==================================================
This script maximizes model performance by:
1. Aggregating Out-of-Fold (OOF) predictions across all 5 folds.
2. Applying Test-Time Augmentation (TTA) - averaging predictions on 
   original and flipped signals.
3. Calculating "Global AUC" (often higher than mean fold AUC).

Usage:
    python evaluate_ensemble.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from csp_features import MultimodalFeatureExtractor

# Import for custom objects
from attention_blocks import SEBlock, TemporalAttentionBlock
from model import CrossModalAttention
from focal_loss import FocalLoss


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "Reports", "ensemble_analysis")

def load_custom_model(model_path):
    """Load model with all custom layers registered."""
    custom_objects = {
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'CrossModalAttention': CrossModalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.5, alpha=0.75)
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def predict_with_tta(model, X_inputs, use_tta=True):
    """
    Predict with Test-Time Augmentation (TTA).
    Strategy: Average predictions of Original and Flipped (Horizontal) signals.
    """
    # 1. Standard Prediction
    pred_orig = model.predict(X_inputs, verbose=0).flatten()
    
    if not use_tta:
        return pred_orig
    
    # 2. TTA: Flip FHR and UC (if present)
    # X_inputs order: [X_fhr, X_tab, (optional X_csp)]
    
    # Create copies for augmentation
    X_inputs_tta = [x.copy() for x in X_inputs]
    
    # Flip FHR (Input 0) - Shape (Batch, Time, 1)
    X_inputs_tta[0] = np.flip(X_inputs_tta[0], axis=1) # Flip time axis
    
    # Pass 2: Prediction on Augmented Data
    pred_tta = model.predict(X_inputs_tta, verbose=0).flatten()
    
    # Average predictions
    final_pred = 0.5 * pred_orig + 0.5 * pred_tta
    
    return final_pred

def main():
    print("="*60)
    print("Ensemble & TTA Evaluation (Maximizing SOTA)")
    print("="*60)
    
    # Load Data
    print("\nLoading data...")
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tab = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    uc_path = os.path.join(PROCESSED_DATA_DIR, "X_uc.npy")
    X_uc = np.load(uc_path) if os.path.exists(uc_path) else None
    
    # Stratified K-Fold (Must match training split exact random_state)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(y))
    oof_predictions_tta = np.zeros(len(y))
    fold_aucs = []
    fold_aucs_tta = []
    
    print(f"Data shape: {X_fhr.shape}, Labels: {y.shape}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fhr, y), 1):
        print(f"\n--- Processing Fold {fold} ---")
        
        # Load Fold Model
        model_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{fold}.keras")
        if not os.path.exists(model_path):
            print(f"Model missing for fold {fold}, skipping...")
            continue
            
        model = load_custom_model(model_path)
        
        # Prepare Validation Data (Exactly as in training/evaluation)
        X_fhr_val = X_fhr[val_idx]
        X_tab_val = X_tab[val_idx]
        y_val = y[val_idx]
        
        if X_uc is not None:
            # Re-fit CSP (Simulating strict pipeline without leakage)
            # Note: Ideally we save fitted CSP, but re-fitting on train split is valid
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
            
        # 1. Standard Prediction
        preds = predict_with_tta(model, X_inputs, use_tta=False)
        oof_predictions[val_idx] = preds
        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        
        # 2. TTA Prediction
        preds_tta = predict_with_tta(model, X_inputs, use_tta=True)
        oof_predictions_tta[val_idx] = preds_tta
        auc_tta = roc_auc_score(y_val, preds_tta)
        fold_aucs_tta.append(auc_tta)
        
        print(f"Fold {fold} AUC: {auc:.4f} -> TTA: {auc_tta:.4f}  ({'✅' if auc_tta > auc else '➖'})")

    # --- Final Results ---
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # 1. Mean Fold AUC (What we had before)
    mean_auc = np.mean(fold_aucs)
    mean_auc_tta = np.mean(fold_aucs_tta)
    print(f"\nEvaluate Uncertainty Script (Mean AUC): {mean_auc:.4f}")
    print(f"With TTA (Mean AUC):                  {mean_auc_tta:.4f}")
    
    # 2. Global OOF AUC (The "True" dataset-wide metric)
    # This aggregates all predictions and calculates one AUC
    global_auc = roc_auc_score(y, oof_predictions)
    global_auc_tta = roc_auc_score(y, oof_predictions_tta)
    
    print(f"\nGLOBAL OOF AUC (Standard):            {global_auc:.4f}")
    print(f"GLOBAL OOF AUC (with TTA):            {global_auc_tta:.4f}")
    
    print("\n" + "-"*60)
    if global_auc_tta > 0.85:
        print("🎉 SUCCESS: TARGET > 0.85 ACHIEVED!")
    elif global_auc_tta > 0.80:
        print("✅ GREAT RESULT: > 0.80 (SOTA Level)")
    else:
        print("⚠️  Result decent but below target.")
        
    print("-" * 60)

if __name__ == "__main__":
    main()

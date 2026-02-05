"""
Enhanced Training Pipeline for NeuroFetal AI
=============================================
Training script with:
- 3-input model support (FHR + Tabular + CSP)
- CSP fitting INSIDE cross-validation (no data leakage!)
- Focal Loss for class imbalance
- Logging and checkpointing

Critical Fix: CSP extractor is fitted on training data only within each fold.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
from datetime import datetime

# Local imports (from utils folder)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model import build_fusion_resnet, build_enhanced_fusion_resnet, build_attention_fusion_resnet
from focal_loss import get_focal_loss
from csp_features import MultimodalFeatureExtractor
from augmentation import TimeSeriesAugmentor, apply_label_smoothing


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
LOG_DIR = os.path.join(BASE_DIR, "Reports", "training_logs")

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 75  # Increased from 50 for better convergence
LEARNING_RATE = 0.001
N_FOLDS = 5

# Focal Loss parameters (for extreme class imbalance: ~7% positive in CTU-UHB)
USE_FOCAL_LOSS = True  # Re-enabled with lower gamma for stability
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 1.0  # Reduced from 2.0 to prevent extreme gradients early
FOCAL_LOSS_POS_WEIGHT = 5.0

# Model configuration
USE_ENHANCED_MODEL = True  # Use 3-input model with attention
USE_SE_BLOCKS = True
USE_ATTENTION = True
USE_CSP = None  # Auto-detect: True only if real UC data exists (set in main())
USE_CROSS_MODAL_ATTENTION = True  # NOVEL: Enable cross-modal attention fusion
MC_DROPOUT = False  # NOVEL: Enable MC Dropout for uncertainty (set True for inference)

# NOVEL: Data Augmentation Configuration
USE_AUGMENTATION = True  # Enable time-series augmentation
AUGMENT_EXPAND_FACTOR = 2  # 2x data expansion (original + augmented)
USE_LABEL_SMOOTHING = True  # Enable label smoothing regularization
LABEL_SMOOTHING = 0.1  # Smoothing factor (0.1 = soft labels [0.05, 0.95])


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data():
    """Load preprocessed data from disk."""
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    # Try to load UC signal for CSP features
    try:
        X_uc = np.load(os.path.join(PROCESSED_DATA_DIR, "X_uc.npy"))
    except FileNotFoundError:
        print("Warning: X_uc.npy not found. CSP features will use synthetic UC.")
        X_uc = None
    
    # Ensure FHR has channel dimension
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
        
    return X_fhr, X_tabular, y, X_uc


def extract_csp_features_for_fold(
    X_fhr_train, X_uc_train, y_train,
    X_fhr_val, X_uc_val,
    n_csp_components=4
):
    """
    Extract CSP features with proper train/val split.
    
    CRITICAL: CSP is fitted ONLY on training data to prevent data leakage.
    
    Args:
        X_fhr_train, X_uc_train, y_train: Training data
        X_fhr_val, X_uc_val: Validation data
        n_csp_components: Number of CSP components
        
    Returns:
        X_csp_train, X_csp_val: CSP features for train and val
    """
    # Create extractor
    extractor = MultimodalFeatureExtractor(n_csp_components=n_csp_components)
    
    # Split training data by class
    normal_mask = (y_train == 0)
    path_mask = (y_train == 1)
    
    # FHR needs to be 2D for CSP: (n_samples, signal_length)
    X_fhr_train_2d = X_fhr_train.squeeze()
    X_fhr_val_2d = X_fhr_val.squeeze()
    
    # Fit CSP on TRAINING DATA ONLY (prevents data leakage)
    extractor.fit(
        X_fhr_train_2d[normal_mask], X_uc_train[normal_mask],
        X_fhr_train_2d[path_mask], X_uc_train[path_mask]
    )
    
    # Transform both train and validation
    X_csp_train = extractor.extract_batch(X_fhr_train_2d, X_uc_train)
    X_csp_val = extractor.extract_batch(X_fhr_val_2d, X_uc_val)
    
    return X_csp_train, X_csp_val, extractor


def train_fold(
    fold_num,
    X_fhr_train, X_tab_train, X_csp_train, y_train,
    X_fhr_val, X_tab_val, X_csp_val, y_val,
    use_enhanced=True
):
    """
    Train a single fold with data augmentation.
    
    Args:
        fold_num: Fold number for logging
        X_*_train, X_*_val: Training and validation data
        y_train, y_val: Labels
        use_enhanced: Whether to use enhanced 3-input model
        
    Returns:
        history: Training history
        model: Trained model
        metrics: Validation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_num}")
    print(f"{'='*60}")
    
    # =========================================================================
    # NOVEL: Apply Data Augmentation
    # =========================================================================
    if USE_AUGMENTATION:
        print(f"Applying data augmentation (expand_factor={AUGMENT_EXPAND_FACTOR})...")
        augmentor = TimeSeriesAugmentor(
            p=0.5,
            time_warp_sigma=0.2,
            jitter_sigma=0.03,
            scale_sigma=0.1,
            mixup_alpha=0.2
        )
        
        # Augment FHR signal
        X_fhr_train, y_train = augmentor.augment_batch(
            X_fhr_train, y_train, 
            expand_factor=AUGMENT_EXPAND_FACTOR
        )
        
        # Replicate tabular and CSP features to match augmented FHR
        n_original = X_tab_train.shape[0]
        n_augmented = X_fhr_train.shape[0]
        if n_augmented > n_original:
            repeat_factor = n_augmented // n_original
            X_tab_train = np.tile(X_tab_train, (repeat_factor, 1))
            if X_csp_train is not None:
                X_csp_train = np.tile(X_csp_train, (repeat_factor, 1))
        
        print(f"  Augmented training samples: {n_original} → {n_augmented}")
    
    # Apply label smoothing
    if USE_LABEL_SMOOTHING:
        y_train = apply_label_smoothing(y_train, smoothing=LABEL_SMOOTHING)
        print(f"Applied label smoothing (factor={LABEL_SMOOTHING})")
    
    # =========================================================================
    # Build model
    # =========================================================================
    if use_enhanced and X_csp_train is not None:
        # Use NOVEL Cross-Modal Attention model for publication
        if USE_CROSS_MODAL_ATTENTION:
            model = build_attention_fusion_resnet(
                input_shape_fhr=(X_fhr_train.shape[1], X_fhr_train.shape[2]),
                input_shape_tabular=(X_tab_train.shape[1],),
                input_shape_csp=(X_csp_train.shape[1],),
                use_se_blocks=USE_SE_BLOCKS,
                use_temporal_attention=USE_ATTENTION,
                use_cross_modal_attention=True,
                mc_dropout=MC_DROPOUT
            )
            print("Using NOVEL AttentionFusionResNet with Cross-Modal Attention")
        else:
            model = build_enhanced_fusion_resnet(
                input_shape_fhr=(X_fhr_train.shape[1], X_fhr_train.shape[2]),
                input_shape_tabular=(X_tab_train.shape[1],),
                input_shape_csp=(X_csp_train.shape[1],),
                use_se_blocks=USE_SE_BLOCKS,
                use_attention=USE_ATTENTION
            )
        train_inputs = [X_fhr_train, X_tab_train, X_csp_train]
        val_inputs = [X_fhr_val, X_tab_val, X_csp_val]
    else:
        model = build_fusion_resnet(
            input_shape_ts=(X_fhr_train.shape[1], X_fhr_train.shape[2]),
            input_shape_tab=(X_tab_train.shape[1],)
        )
        train_inputs = [X_fhr_train, X_tab_train]
        val_inputs = [X_fhr_val, X_tab_val]
    
    # Metrics
    metrics = [
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.SensitivityAtSpecificity(0.85, name='sens_at_spec_85')
    ]
    
    # Optimizer
    if hasattr(tf.keras.optimizers, 'AdamW'):
        optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Loss function
    if USE_FOCAL_LOSS:
        loss_fn = get_focal_loss(
            alpha=FOCAL_LOSS_ALPHA,
            gamma=FOCAL_LOSS_GAMMA,
            use_weighted=True,
            pos_weight=FOCAL_LOSS_POS_WEIGHT
        )
        print(f"Using Focal Loss (α={FOCAL_LOSS_ALPHA}, γ={FOCAL_LOSS_GAMMA})")
    else:
        loss_fn = 'binary_crossentropy'
        print("Using Binary Cross-Entropy")
    
    # Compile
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    
    # Callbacks
    checkpoint_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{fold_num}.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_auc', verbose=1, 
                       save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_auc', patience=15, mode='max',  # Increased from 10
                     verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, 
                         min_lr=1e-6, verbose=1)
    ]
    
    # Train
    print(f"\nTraining on {len(y_train)} samples, validating on {len(y_val)} samples")
    print(f"Class balance - Train: {np.mean(y_train):.2%} positive, Val: {np.mean(y_val):.2%} positive")
    
    history = model.fit(
        train_inputs, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_inputs, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_metrics = model.evaluate(val_inputs, y_val, verbose=0)
    metric_names = ['loss'] + [m.name for m in metrics]
    results = dict(zip(metric_names, val_metrics))
    
    print(f"\nFold {fold_num} Results:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")
    
    return history, model, results


def main():
    """Main training function with 5-fold cross-validation."""
    ensure_dir(MODEL_DIR)
    ensure_dir(LOG_DIR)
    
    # Load Data
    print("="*60)
    print("NeuroFetal AI - Enhanced Training Pipeline")
    print("="*60)
    print("\nLoading data...")
    
    X_fhr, X_tabular, y, X_uc = load_data()
    
    # Check for NaN/Inf in data (critical for debugging)
    def check_data(arr, name):
        if np.isnan(arr).any():
            print(f"  WARNING: {name} contains NaN values!")
            arr = np.nan_to_num(arr, nan=0.0)
        if np.isinf(arr).any():
            print(f"  WARNING: {name} contains Inf values!")
            arr = np.nan_to_num(arr, posinf=1e6, neginf=-1e6)
        return arr
    
    X_fhr = check_data(X_fhr, "X_fhr")
    X_tabular = check_data(X_tabular, "X_tabular")
    y = check_data(y, "y")
    if X_uc is not None:
        X_uc = check_data(X_uc, "X_uc")
    
    print(f"Data loaded:")
    print(f"  FHR: {X_fhr.shape}")
    print(f"  Tabular: {X_tabular.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  UC: {'Available' if X_uc is not None else 'Not available (will use synthetic)'}")
    print(f"  Class balance: {np.mean(y):.2%} positive (compromised)")
    
    # Determine whether to use CSP features
    global USE_CSP
    if X_uc is None:
        print("\n⚠️  Real UC data not available - DISABLING CSP features.")
        print("   (Synthetic UC produces meaningless CSP; using 2-input model instead)")
        USE_CSP = False
        # Create dummy X_uc to avoid errors (won't be used)
        X_uc = np.zeros((len(X_fhr), X_fhr.shape[1]))
    else:
        print("\n✓ Real UC data available - CSP features ENABLED.")
        USE_CSP = True
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    all_results = []
    fold_num = 1
    
    for train_idx, val_idx in skf.split(X_fhr, y):
        # Split data
        X_fhr_train, X_fhr_val = X_fhr[train_idx], X_fhr[val_idx]
        X_tab_train, X_tab_val = X_tabular[train_idx], X_tabular[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_uc_train, X_uc_val = X_uc[train_idx], X_uc[val_idx]
        
        # Extract CSP features (fit on training only!) - ONLY if real UC is available
        if USE_ENHANCED_MODEL and USE_CSP:
            print(f"\nExtracting CSP features for Fold {fold_num}...")
            X_csp_train, X_csp_val, csp_extractor = extract_csp_features_for_fold(
                X_fhr_train, X_uc_train, y_train,
                X_fhr_val, X_uc_val
            )
            # Check CSP features for NaN/Inf
            X_csp_train = check_data(X_csp_train, "X_csp_train")
            X_csp_val = check_data(X_csp_val, "X_csp_val")
            print(f"  CSP features: train={X_csp_train.shape}, val={X_csp_val.shape}")
        else:
            if USE_ENHANCED_MODEL and not USE_CSP:
                print(f"\nFold {fold_num}: Skipping CSP (no real UC data)")
            X_csp_train, X_csp_val = None, None
        
        # Train fold
        history, model, fold_results = train_fold(
            fold_num,
            X_fhr_train, X_tab_train, X_csp_train, y_train,
            X_fhr_val, X_tab_val, X_csp_val, y_val,
            use_enhanced=USE_ENHANCED_MODEL
        )
        
        fold_results['fold'] = fold_num
        all_results.append(fold_results)
        
        fold_num += 1
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    aucs = [r['auc'] for r in all_results]
    print(f"\nAUC across folds: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Individual folds: {[f'{a:.4f}' for a in aucs]}")
    
    # Save results
    log_file = os.path.join(LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_file, 'w') as f:
        json.dump({
            'config': {
                'use_enhanced_model': USE_ENHANCED_MODEL,
                'use_se_blocks': USE_SE_BLOCKS,
                'use_attention': USE_ATTENTION,
                'use_focal_loss': USE_FOCAL_LOSS,
                'focal_loss_alpha': FOCAL_LOSS_ALPHA,
                'focal_loss_gamma': FOCAL_LOSS_GAMMA,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE
            },
            'results': all_results,
            'summary': {
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs))
            }
        }, f, indent=2)
    print(f"\nResults saved to: {log_file}")
    
    print("\n" + "="*60)
    print("✓ Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

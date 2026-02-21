"""
Diverse Ensemble Training for NeuroFetal AI (SOTA Phase 5)
==========================================================
Trains 3 diverse model architectures for stacking ensemble:
  Model A: AttentionFusionResNet (current deep model)
  Model B: 1D-InceptionNet (lighter, multi-scale patterns)
  Model C: XGBoost on extracted features (tabular+CSP)

The script generates out-of-fold (OOF) predictions for stacking.

Usage:
    python Code/scripts/train_diverse_ensemble.py
"""

import os
import sys
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
LOG_DIR = os.path.join(BASE_DIR, "Reports", "training_logs")

N_FOLDS = 5
RANDOM_STATE = 42
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "synthetic")
USE_TIMEGAN_AUG = True  # V4.0: inject TimeGAN synthetic data into training folds


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data():
    """Load preprocessed data."""
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))

    try:
        X_uc = np.load(os.path.join(PROCESSED_DATA_DIR, "X_uc.npy"))
    except FileNotFoundError:
        X_uc = None

    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)

    return X_fhr, X_tabular, y, X_uc


# ============================================================================
# Model B: 1D InceptionNet (Lighter, Multi-Scale)
# ============================================================================

def build_inception_model(input_shape_fhr, input_shape_tabular, input_shape_csp, dropout_rate=0.3):
    """
    1D InceptionNet: captures multi-scale temporal patterns with fewer parameters.
    
    Uses parallel convolutions with different kernel sizes (5, 15, 40) to capture
    short-term variability, accelerations/decelerations, and long-term trends
    simultaneously.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input

    # FHR Branch: Inception-style multi-scale
    input_fhr = Input(shape=input_shape_fhr, name='input_fhr')

    # Initial conv
    x = layers.Conv1D(32, 7, strides=2, padding='same')(input_fhr)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Inception Block 1
    def inception_block(x, f1, f2, f3, name_prefix=''):
        branch1 = layers.Conv1D(f1, 5, padding='same', name=f'{name_prefix}_conv5')(x)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Activation('relu')(branch1)

        branch2 = layers.Conv1D(f2, 15, padding='same', name=f'{name_prefix}_conv15')(x)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Activation('relu')(branch2)

        branch3 = layers.Conv1D(f3, 40, padding='same', name=f'{name_prefix}_conv40')(x)
        branch3 = layers.BatchNormalization()(branch3)
        branch3 = layers.Activation('relu')(branch3)

        return layers.Concatenate()([branch1, branch2, branch3])

    x = inception_block(x, 32, 32, 16, 'inc1')  # 80 channels
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = inception_block(x, 64, 64, 32, 'inc2')  # 160 channels
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = inception_block(x, 64, 64, 32, 'inc3')  # 160 channels
    fhr_out = layers.GlobalAveragePooling1D()(x)
    fhr_out = layers.Dense(96, activation='relu', name='fhr_proj')(fhr_out)

    # Tabular Branch
    input_tabular = Input(shape=input_shape_tabular, name='input_tabular')
    tab = layers.Dense(48, activation='relu')(input_tabular)
    tab = layers.Dropout(dropout_rate)(tab)
    tab = layers.Dense(96, activation='relu')(tab)

    # CSP Branch
    input_csp = Input(shape=input_shape_csp, name='input_csp')
    csp = layers.Dense(48, activation='relu')(input_csp)
    csp = layers.Dropout(dropout_rate)(csp)
    csp = layers.Dense(96, activation='relu')(csp)

    # Fusion: simple concatenation works well for diverse models
    fusion = layers.Concatenate()([fhr_out, tab, csp])
    x = layers.Dense(64, activation='relu')(fusion)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(
        inputs=[input_fhr, input_tabular, input_csp],
        outputs=output,
        name='InceptionFusionNet'
    )
    return model


# ============================================================================
# Model C: XGBoost/LightGBM on Extracted Features
# ============================================================================

def extract_cnn_features(X_fhr, model_path=None):
    """
    Extract hand-crafted features from FHR for XGBoost.
    Since we already have rich tabular + CSP features, we add simple
    statistical features from the FHR signal.
    """
    features = []
    for fhr in X_fhr:
        fhr_1d = fhr.flatten()
        valid = fhr_1d[fhr_1d > 0]

        feat = []
        # Basic stats
        feat.append(np.mean(valid) if len(valid) > 0 else 0)
        feat.append(np.std(valid) if len(valid) > 0 else 0)
        feat.append(np.median(valid) if len(valid) > 0 else 0)

        # Percentiles
        for p in [5, 25, 75, 95]:
            feat.append(np.percentile(valid, p) if len(valid) > 0 else 0)

        # Variability
        feat.append(np.mean(np.abs(np.diff(valid))) if len(valid) > 1 else 0)

        # Zero-crossing rate (proxy for oscillation)
        if len(valid) > 1:
            mean_centered = valid - np.mean(valid)
            zcr = np.sum(np.diff(np.sign(mean_centered)) != 0) / len(valid)
            feat.append(zcr)
        else:
            feat.append(0)

        # Peak/trough count
        from scipy.signal import find_peaks
        if len(valid) > 10:
            peaks, _ = find_peaks(valid, distance=30)
            troughs, _ = find_peaks(-valid, distance=30)
            feat.append(len(peaks))
            feat.append(len(troughs))
        else:
            feat.extend([0, 0])

        features.append(feat)

    return np.array(features, dtype=np.float32)


def train_xgboost_model(X_tab, X_csp, X_fhr_features, y_train,
                         X_tab_val, X_csp_val, X_fhr_features_val, y_val):
    """Train XGBoost on combined tabular + CSP + hand-crafted FHR features."""
    try:
        import xgboost as xgb
    except ImportError:
        try:
            import lightgbm as lgb
            print("  Using LightGBM (XGBoost not available)")
            return train_lightgbm_model(X_tab, X_csp, X_fhr_features, y_train,
                                       X_tab_val, X_csp_val, X_fhr_features_val, y_val)
        except ImportError:
            print("  WARNING: Neither XGBoost nor LightGBM available. Skipping Model C.")
            return None, None

    # Combine all features
    X_train_combined = np.hstack([X_tab, X_csp, X_fhr_features])
    X_val_combined = np.hstack([X_tab_val, X_csp_val, X_fhr_features_val])

    # Handle NaN
    X_train_combined = np.nan_to_num(X_train_combined, nan=0.0)
    X_val_combined = np.nan_to_num(X_val_combined, nan=0.0)

    # Class weight
    pos_ratio = np.mean(y_train)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
        use_label_encoder=False
    )

    model.fit(
        X_train_combined, y_train,
        eval_set=[(X_val_combined, y_val)],
        verbose=False
    )

    y_pred = model.predict_proba(X_val_combined)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"    XGBoost validation AUC: {auc:.4f}")

    return model, y_pred


def train_lightgbm_model(X_tab, X_csp, X_fhr_features, y_train,
                         X_tab_val, X_csp_val, X_fhr_features_val, y_val):
    """Fallback: LightGBM."""
    import lightgbm as lgb

    X_train_combined = np.hstack([X_tab, X_csp, X_fhr_features])
    X_val_combined = np.hstack([X_tab_val, X_csp_val, X_fhr_features_val])
    X_train_combined = np.nan_to_num(X_train_combined, nan=0.0)
    X_val_combined = np.nan_to_num(X_val_combined, nan=0.0)

    pos_ratio = np.mean(y_train)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1
    )

    model.fit(
        X_train_combined, y_train,
        eval_set=[(X_val_combined, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )

    y_pred = model.predict_proba(X_val_combined)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"    LightGBM validation AUC: {auc:.4f}")

    return model, y_pred


# ============================================================================
# Out-of-Fold Prediction Generator
# ============================================================================

def generate_oof_predictions(X_fhr, X_tabular, X_csp, y, n_folds=5):
    """
    Generate out-of-fold predictions from all 3 model types.
    
    Returns:
        oof_preds: (N, 3) array — OOF predictions from each model
        y_true: (N,) ground truth labels
    """
    import tensorflow as tf
    from model import build_attention_fusion_resnet, CrossModalAttention
    from attention_blocks import SEBlock, TemporalAttentionBlock
    from focal_loss import get_focal_loss
    from csp_features import MultimodalFeatureExtractor

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    n_samples = len(y)
    oof_model_a = np.zeros(n_samples)  # AttentionFusionResNet
    oof_model_b = np.zeros(n_samples)  # InceptionNet
    oof_model_c = np.zeros(n_samples)  # XGBoost

    fold_num = 1
    fold_aucs = {'model_a': [], 'model_b': [], 'model_c': []}

    for train_idx, val_idx in skf.split(X_fhr, y):
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{n_folds}")
        print(f"{'='*60}")

        X_fhr_train, X_fhr_val = X_fhr[train_idx], X_fhr[val_idx]
        X_tab_train, X_tab_val = X_tabular[train_idx], X_tabular[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CSP features — slice BEFORE augmentation so we can pad later
        if X_csp is not None:
            X_csp_train, X_csp_val = X_csp[train_idx], X_csp[val_idx]
        else:
            X_csp_train = np.zeros((len(train_idx), 19))
            X_csp_val = np.zeros((len(val_idx), 19))

        # V4.0: TimeGAN Augmentation (inject synthetic pathological traces)
        if USE_TIMEGAN_AUG:
            from train import apply_timegan_augmentation
            X_uc_for_aug = None
            # Try to load UC for augmentation shape matching
            uc_path = os.path.join(PROCESSED_DATA_DIR, "X_uc.npy")
            if os.path.exists(uc_path):
                X_uc_all = np.load(uc_path)
                if X_uc_all.ndim == 2:
                    X_uc_all = np.expand_dims(X_uc_all, axis=-1)
                X_uc_for_aug = X_uc_all[train_idx]

            n_before = len(y_train)
            print(f"  Applying TimeGAN augmentation...")
            print(f"    Before: {int(y_train.sum())} positives / {n_before} total")
            X_fhr_train, X_tab_train, X_uc_for_aug, y_train = apply_timegan_augmentation(
                X_fhr_train, X_tab_train, X_uc_for_aug, y_train, random_state=RANDOM_STATE
            )
            n_after = len(y_train)
            n_synthetic = n_after - n_before
            print(f"  TimeGAN Augmentation: {int(y_train[:n_before].sum())} → {int(y_train.sum())} positives / {n_after} total")
            print(f"  Injected {n_synthetic} synthetic traces (from {1410} available)")
            print(f"    After:  {int(y_train.sum())} positives / {n_after} total")

            # Pad CSP features to match augmented sample count
            # Resample CSP rows from existing pathological samples for synthetic traces
            if n_synthetic > 0:
                patho_mask = y_train[:n_before] == 1
                X_csp_patho = X_csp_train[patho_mask]
                rng = np.random.RandomState(RANDOM_STATE)
                resample_idx = rng.choice(len(X_csp_patho), size=n_synthetic, replace=True)
                X_csp_synthetic = X_csp_patho[resample_idx]
                X_csp_train = np.vstack([X_csp_train, X_csp_synthetic])
                print(f"  CSP features padded: {n_before} → {X_csp_train.shape[0]} rows")

        input_shapes = {
            'fhr': (X_fhr_train.shape[1], X_fhr_train.shape[2]),
            'tab': (X_tab_train.shape[1],),
            'csp': (X_csp_train.shape[1],)
        }

        # ------------------------------------------------------------------
        # Model A: AttentionFusionResNet (load from saved folds)
        # ------------------------------------------------------------------
        print("\n  [Model A] AttentionFusionResNet...")
        model_a_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{fold_num}.keras")
        if os.path.exists(model_a_path):
            custom_objects = {
                'SEBlock': SEBlock,
                'TemporalAttentionBlock': TemporalAttentionBlock,
                'CrossModalAttention': CrossModalAttention
            }
            try:
                model_a = tf.keras.models.load_model(
                    model_a_path, custom_objects=custom_objects, compile=False
                )
                preds_a = model_a.predict([X_fhr_val, X_tab_val, X_csp_val], verbose=0).flatten()
                auc_a = roc_auc_score(y_val, preds_a)
                fold_aucs['model_a'].append(auc_a)
                oof_model_a[val_idx] = preds_a
                print(f"    Model A fold {fold_num} AUC: {auc_a:.4f}")
            except Exception as e:
                print(f"    Model A failed for fold {fold_num}: {e}")
                oof_model_a[val_idx] = 0.5
        else:
            print(f"    Model A not found at {model_a_path}")
            oof_model_a[val_idx] = 0.5

        # ------------------------------------------------------------------
        # Model B: InceptionNet
        # ------------------------------------------------------------------
        print("\n  [Model B] InceptionNet...")
        try:
            model_b = build_inception_model(
                input_shapes['fhr'], input_shapes['tab'], input_shapes['csp']
            )

            loss_fn = get_focal_loss(alpha=0.65, gamma=2.0, use_weighted=True, pos_weight=5.0)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=5e-4)
            model_b.compile(optimizer=optimizer, loss=loss_fn,
                          metrics=[tf.keras.metrics.AUC(name='auc')])

            inception_path = os.path.join(MODEL_DIR, f"inception_model_fold_{fold_num}.keras")
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    inception_path, monitor='val_auc', save_best_only=True, mode='max'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc', patience=25, mode='max', restore_best_weights=True
                ),
            ]

            model_b.fit(
                [X_fhr_train, X_tab_train, X_csp_train], y_train,
                epochs=100, batch_size=32,
                validation_data=([X_fhr_val, X_tab_val, X_csp_val], y_val),
                callbacks=callbacks, verbose=0
            )

            preds_b = model_b.predict([X_fhr_val, X_tab_val, X_csp_val], verbose=0).flatten()
            auc_b = roc_auc_score(y_val, preds_b)
            fold_aucs['model_b'].append(auc_b)
            oof_model_b[val_idx] = preds_b
            print(f"    Model B fold {fold_num} AUC: {auc_b:.4f}")
        except Exception as e:
            print(f"    Model B training failed: {e}")
            oof_model_b[val_idx] = 0.5

        tf.keras.backend.clear_session()

        # ------------------------------------------------------------------
        # Model C: XGBoost
        # ------------------------------------------------------------------
        print("\n  [Model C] XGBoost/LightGBM...")
        X_fhr_feat_train = extract_cnn_features(X_fhr_train)
        X_fhr_feat_val = extract_cnn_features(X_fhr_val)

        model_c, preds_c = train_xgboost_model(
            X_tab_train, X_csp_train, X_fhr_feat_train, y_train,
            X_tab_val, X_csp_val, X_fhr_feat_val, y_val
        )

        if preds_c is not None:
            auc_c = roc_auc_score(y_val, preds_c)
            fold_aucs['model_c'].append(auc_c)
            oof_model_c[val_idx] = preds_c

            # Save XGBoost model
            xgb_path = os.path.join(MODEL_DIR, f"xgboost_model_fold_{fold_num}.pkl")
            with open(xgb_path, 'wb') as f:
                pickle.dump(model_c, f)
        else:
            oof_model_c[val_idx] = 0.5

        fold_num += 1

    # Stack OOF predictions
    oof_preds = np.column_stack([oof_model_a, oof_model_b, oof_model_c])

    # Save OOF predictions for stacking
    np.save(os.path.join(MODEL_DIR, "oof_predictions.npy"), oof_preds)
    np.save(os.path.join(MODEL_DIR, "oof_labels.npy"), y)

    print(f"\n{'='*60}")
    print("OOF AUC Summary")
    print(f"{'='*60}")
    for name, aucs in fold_aucs.items():
        if aucs:
            print(f"  {name}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return oof_preds, y, fold_aucs


# ============================================================================
# Stacking Meta-Learner
# ============================================================================

def train_stacking_meta_learner(oof_preds, y, weights=None):
    """
    Train a logistic regression meta-learner on OOF predictions.
    
    This learns optimal combination weights for the 3 model types,
    automatically finding which models are most trustworthy.
    
    Args:
        oof_preds: (N, 3) OOF predictions from 3 models
        y: (N,) ground truth
        weights: Optional manual weights for fallback
        
    Returns:
        meta_model: Trained logistic regression
        meta_auc: AUC of the stacked ensemble
    """
    # Remove samples where all predictions are 0.5 (model failed)
    valid_mask = ~np.all(oof_preds == 0.5, axis=1)
    oof_valid = oof_preds[valid_mask]
    y_valid = y[valid_mask]

    if len(y_valid) < 10:
        print("  Not enough valid OOF predictions for stacking. Using weighted average.")
        return None, 0.0

    # Logistic Regression meta-learner (Platt scaling built-in)
    meta_model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    )
    meta_model.fit(oof_valid, y_valid)

    meta_preds = meta_model.predict_proba(oof_valid)[:, 1]
    meta_auc = roc_auc_score(y_valid, meta_preds)

    print(f"\n  Stacking Meta-Learner AUC: {meta_auc:.4f}")
    print(f"  Meta-learner weights: {meta_model.coef_.flatten()}")
    print(f"  Meta-learner bias: {meta_model.intercept_}")

    # Compare with simple weighted average
    if weights is None:
        weights = [0.4, 0.3, 0.3]
    weighted_avg = np.average(oof_valid, axis=1, weights=weights)
    weighted_auc = roc_auc_score(y_valid, weighted_avg)
    print(f"  Weighted Average AUC (w={weights}): {weighted_auc:.4f}")

    # Save meta-learner
    meta_path = os.path.join(MODEL_DIR, "stacking_meta_learner.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_model, f)
    print(f"  Meta-learner saved to: {meta_path}")

    return meta_model, meta_auc


# ============================================================================
# Main
# ============================================================================

def main():
    ensure_dir(MODEL_DIR)
    ensure_dir(LOG_DIR)

    print("=" * 60)
    print("NeuroFetal AI — Diverse Ensemble Training (Phase 5)")
    print("=" * 60)

    # Load data
    X_fhr, X_tabular, y, X_uc = load_data()

    print(f"\nData: FHR={X_fhr.shape}, Tab={X_tabular.shape}, y={y.shape}")
    print(f"Class balance: {np.mean(y):.1%} positive")

    # For simplicity, load precomputed CSP or set to None
    csp_path = os.path.join(MODEL_DIR, "oof_csp_features.npy")
    X_csp = None
    if os.path.exists(csp_path):
        X_csp = np.load(csp_path)
        print(f"CSP features loaded: {X_csp.shape}")

    # Generate OOF predictions from all 3 model types
    oof_preds, y_true, fold_aucs = generate_oof_predictions(
        X_fhr, X_tabular, X_csp, y, n_folds=N_FOLDS
    )

    # Train stacking meta-learner
    meta_model, meta_auc = train_stacking_meta_learner(oof_preds, y_true)

    # Save results
    results = {
        'fold_aucs': {k: [float(v) for v in vals] for k, vals in fold_aucs.items()},
        'stacking_auc': float(meta_auc),
        'timestamp': datetime.now().isoformat()
    }

    log_path = os.path.join(LOG_DIR, f"ensemble_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {log_path}")
    print("\n✓ Diverse ensemble training complete!")


if __name__ == "__main__":
    main()

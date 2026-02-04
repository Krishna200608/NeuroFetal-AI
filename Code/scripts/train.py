import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_fusion_resnet
from focal_loss import get_focal_loss

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Focal Loss parameters (for extreme class imbalance: ~7% positive in CTU-UHB)
USE_FOCAL_LOSS = True
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_POS_WEIGHT = 5.0

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    # Ensure dimensions
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
        
    return X_fhr, X_tabular, y

def main():
    ensure_dir(MODEL_DIR)
    
    # Load Data
    print("Loading data...")
    X_fhr, X_tabular, y = load_data()
    print(f"Data loaded: FHR {X_fhr.shape}, Tabular {X_tabular.shape}, Labels {y.shape}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_var = 1
    
    for train_index, val_index in skf.split(X_fhr, y):
        print(f"\n--- Training Fold {fold_var} ---")
        
        X_fhr_train, X_fhr_val = X_fhr[train_index], X_fhr[val_index]
        X_tab_train, X_tab_val = X_tabular[train_index], X_tabular[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Build Model
        # Input shape from data
        input_shape_ts = (X_fhr.shape[1], X_fhr.shape[2])
        input_shape_tab = (X_tabular.shape[1],)
        
        model = build_fusion_resnet(input_shape_ts, input_shape_tab)
        
        # Metrics: AUC and Sensitivity at Specificity 0.85 (Recall @ 15% FPR)
        metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.SensitivityAtSpecificity(0.85, name='recall_at_15_fpr')
        ]
        
        if hasattr(tf.keras.optimizers, 'AdamW'):
            optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
        else:
            # Fallback for older TF
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        # Select loss function (Focal Loss for better imbalance handling)
        if USE_FOCAL_LOSS:
            loss_fn = get_focal_loss(
                alpha=FOCAL_LOSS_ALPHA,
                gamma=FOCAL_LOSS_GAMMA,
                use_weighted=True,
                pos_weight=FOCAL_LOSS_POS_WEIGHT
            )
            print(f"\nUsing Focal Loss (α={FOCAL_LOSS_ALPHA}, γ={FOCAL_LOSS_GAMMA}, pos_weight={FOCAL_LOSS_POS_WEIGHT})")
        else:
            loss_fn = 'binary_crossentropy'
            print("\nUsing Binary Cross-Entropy loss")
        
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=metrics)
        
        # Callbacks
        checkpoint_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold_var}.keras")
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_auc', patience=10, mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        # Note: Focal Loss is more robust to class imbalance than class_weight alone
        # But we still use class_weight as an additional regularization
        history = model.fit(
            [X_fhr_train, X_tab_train], y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([X_fhr_val, X_tab_val], y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1, 1: 3} if not USE_FOCAL_LOSS else None  # Reduce weight when using Focal Loss
        )
        
        # Optional: Evaluate best model
        print(f"Fold {fold_var} Training Complete.")
        
        fold_var += 1
        
    print("All folds completed.")

if __name__ == "__main__":
    main()

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_fusion_resnet

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

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
            optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4) # Standard for modern TF
        else:
            # Fallback for older TF (though Colab usually has latest)
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=metrics)
        
        # Callbacks
        checkpoint_path = os.path.join(MODEL_DIR, f"best_model_fold_{fold_var}.keras")
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_auc', patience=10, mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        history = model.fit(
            [X_fhr_train, X_tab_train], y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([X_fhr_val, X_tab_val], y_val),
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1, 1: 5} # Handle imbalance? User said Stratified CV, but usually we also weight classes or oversample.
                                      # "Stratified 5-Fold ... to handle class imbalance"
                                      # Usually just Stratified KFold is not enough for training, just for evaluation.
                                      # I'll add class weights just in case given the extreme imbalance (40 compromised cases?).
                                      # If 40 cases total, 5 folds = 8 cases per fold.
                                      # This is TINY.
        )
        
        # Optional: Evaluate best model
        print(f"Fold {fold_var} Training Complete.")
        
        fold_var += 1
        
    print("All folds completed.")

if __name__ == "__main__":
    main()

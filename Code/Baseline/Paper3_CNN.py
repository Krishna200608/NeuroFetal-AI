
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# Add scripts directory to path to import data_ingestion
# Assuming this script is in Code/Baseline/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
try:
    import data_ingestion
except ImportError:
    # Fallback if running from a different CWD
    sys.path.append(os.path.abspath(os.path.join("..", "scripts")))
    import data_ingestion

print(f"TensorFlow Version: {tf.__version__}")

# --- Load Data ---
# Logic: Load pre-processed .npy files from Datasets/processed
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Datasets", "processed")

print(f"Loading data from: {PROCESSED_DATA_DIR}")

try:
    X_signal = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    print("Loaded pre-processed data (X_fhr.npy).")
except FileNotFoundError:
    print("Error: Pre-processed .npy files not found.")
    print("Please run 'python Code/scripts/data_ingestion.py' first.")
    sys.exit(1)

print(f"X_signal shape: {X_signal.shape}")
print(f"y shape: {y.shape}")

# --- Define Model ---
def build_baseline_cnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Block 2
    x = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Block 3
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")

# --- Training Loop ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
accs = []

X = X_signal
# Ensure 3D shape (Samples, Time, Channels)
if X.ndim == 2:
    X = np.expand_dims(X, axis=-1)

print("\nStarting 5-Fold Cross-Validation (Paper 3 Baseline)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold+1}/5")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Class weights
    y_train = y_train.astype(int)
    neg, pos = np.bincount(y_train)
    class_weight = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}
    
    model = build_baseline_cnn(input_shape=(X.shape[1], X.shape[2]))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'accuracy'])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=5, mode='max', restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, # Reduced epochs for faster turnaround
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0 # Silent training
    )
    
    y_pred = model.predict(X_val, verbose=0)
    try:
        auc = roc_auc_score(y_val, y_pred)
    except ValueError:
        auc = 0.5 # Handle single class edge case
        
    acc = accuracy_score(y_val, (y_pred > 0.5).astype(int))
    
    aucs.append(auc)
    accs.append(acc)
    print(f"Fold {fold+1} Result -> AUC: {auc:.4f}, Acc: {acc:.4f}")
    
    # Save Model (Best Fold or Last)
    # Saving every fold might be excessive, let's save the best one based on AUC
    if len(aucs) == 1 or auc > max(aucs[:-1]):
        model_save_path = os.path.join(os.path.dirname(__file__), "Models", "baseline_paper3_best_cnn.keras")
        # Ensure dir exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"  Saved best model to {model_save_path}")

# --- Results ---
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
mean_acc = np.mean(accs)

print("\n=== Final Results (Paper 3) ===")
print(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
print(f"Mean Acc: {mean_acc:.4f}")

# Save to file
results_path = os.path.join(os.path.dirname(__file__), "baseline_paper3_results.txt")
with open(results_path, "w") as f:
    f.write(f"Ref: Paper 3 (Spilka 2016) - CNN Baseline\n")
    f.write(f"Mean AUC: {mean_auc:.4f}\n")
    f.write(f"Std Dev: {std_auc:.4f}\n")
    f.write(f"Mean Acc: {mean_acc:.4f}\n")

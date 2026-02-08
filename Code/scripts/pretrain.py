"""
SSL Pretraining Script
======================
Trains the Masked Autoencoder on the FHR dataset (unsupervised).
Saves the pretrained encoder weights for downstream fine-tuning.
"""

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import sys

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model import build_fhr_encoder
from ssl_models import MaskedAutoencoder

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
PRETRAIN_WEIGHTS_PATH = os.path.join(MODEL_DIR, "pretrained_fhr_encoder.weights.h5")

BATCH_SIZE = 64
EPOCHS = 50  # Pretraining usually requires many epochs, but dataset is small.
LEARNING_RATE = 1e-3

def load_data():
    """Load FHR data only (ignoring labels for SSL)."""
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
    
    # Check for NaN/Inf
    X_fhr = np.nan_to_num(X_fhr, nan=0.0)
    
    return X_fhr

def main():
    print("="*60)
    print("SSL Pretraining: Masked Autoencoder")
    print("="*60)
    
    # 1. Load Data
    print("Loading data...")
    X_fhr = load_data()
    print(f"Data shape: {X_fhr.shape}")
    
    # 2. Build Models
    print("Building MAE...")
    encoder = build_fhr_encoder(use_attention=True)
    mae = MaskedAutoencoder(
        encoder=encoder,
        input_shape=(1200, 1),
        masking_ratio=0.5,
        mask_block_size=20
    )
    
    mae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # 3. Train
    print(f"Starting training for {EPOCHS} epochs...")
    mae.fit(
        X_fhr, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )
    
    # 4. Save Encoder Weights
    print(f"Saving encoder weights to {PRETRAIN_WEIGHTS_PATH}...")
    # We save ONLY the encoder weights, not the full MAE
    mae.encoder.save_weights(PRETRAIN_WEIGHTS_PATH)
    print("✓ Pretraining complete!")

if __name__ == "__main__":
    main()

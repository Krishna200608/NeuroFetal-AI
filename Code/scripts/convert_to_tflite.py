"""
Convert Trained Keras Model to TFLite (Int8 Quantized)
======================================================
DeepFetal AI - Edge Optimization Module
---------------------------------------
Converts the best performing fold model to TFLite format for mobile deployment.
Includes:
1.  **Representative Dataset Generation**: On-the-fly CSP fitting and feature extraction for accurate calibration.
2.  **Full Integer Quantization (Int8)**: Optimization for NPU/DSP acceleration.
3.  **Float Fallback**: Keeps standard model for compatibility.

Input Model: 3-Branch Fusion ResNet (FHR, Tabular, Features)
Output: .tflite files (Standard & Int8)
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add utils to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
UTILS_DIR = os.path.join(CODE_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, CODE_DIR)

# Robust fallback imports
try:
    from model import CrossModalAttention
    from attention_blocks import SEBlock, TemporalAttentionBlock as TemporalAttention
    from csp_features import MultimodalFeatureExtractor
    from focal_loss import FocalLoss
except ImportError:
    print("⚠️  Warning: Direct imports failed. Check PYTHONPATH.")
    sys.exit(1)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code", "models", "tflite")
FOLD_TO_CONVERT = 1
CALIBRATION_SAMPLES = 300  # Number of samples for Int8 calibration

def load_calibration_data():
    """
    Loads and prepares data for TFLite calibration.
    Fits CSP on the fly to generate the 'input_csp' features.
    """
    print("Loading calibration data...")
    try:
        X_fhr = np.load(os.path.join(DATA_DIR, "X_fhr.npy"))
        X_tabular = np.load(os.path.join(DATA_DIR, "X_tabular.npy"))
        y = np.load(os.path.join(DATA_DIR, "y.npy"))
        try:
            X_uc = np.load(os.path.join(DATA_DIR, "X_uc.npy"))
        except:
            print("  ⚠️  X_uc.npy not found using random noise for calibration.")
            X_uc = np.random.normal(0, 1, X_fhr.shape)
            
        # Ensure dimensions
        if X_fhr.ndim == 2: X_fhr = np.expand_dims(X_fhr, -1)
        if X_uc.ndim == 2: X_uc = np.expand_dims(X_uc, -1)
        
        # Fit CSP Extractor for Feature Generation
        print("  Fitting CSP Feature Extractor for calibration...")
        extractor = MultimodalFeatureExtractor(n_csp_components=4)
        
        # Squeeze for extractor (expects N, T)
        X_fhr_sq = X_fhr.squeeze()
        X_uc_sq = X_uc.squeeze()
        
        # Split by class
        mask_norm = (y == 0)
        mask_path = (y == 1)
        
        extractor.fit(
            X_fhr_sq[mask_norm], X_uc_sq[mask_norm],
            X_fhr_sq[mask_path], X_uc_sq[mask_path]
        )
        
        # Extract features for all samples
        print("  Extracting multi-modal features...")
        X_csp = extractor.extract_batch(X_fhr_sq, X_uc_sq)
        
        print(f"  Data Loaded: FHR {X_fhr.shape}, Tab {X_tabular.shape}, Features {X_csp.shape}")
        return X_fhr, X_tabular, X_csp
        
    except Exception as e:
        print(f"❌ Error loading calibration data: {e}")
        return None, None, None

def convert_model():
    model_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{FOLD_TO_CONVERT}.keras")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    print(f"\nModel: {model_path}")
    
    # Custom Object Scope
    custom_objects = {
        'CrossModalAttention': CrossModalAttention,
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss()
    }

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✓ Keras Model Loaded")
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    # Prepare Calibration Data
    X_fhr, X_tab, X_csp = load_calibration_data()
    
    if X_fhr is None:
        print("❌ Cannot proceed with Int8 quantization without data.")
        return

    # Representative Dataset Generator
    def representative_data_gen():
        for i in range(min(CALIBRATION_SAMPLES, len(X_fhr))):
            # Yield list of inputs corresponding to model.inputs: [input_fhr, input_tabular, input_csp]
            # Ensure float32
            yield [
                X_fhr[i:i+1].astype(np.float32),
                X_tab[i:i+1].astype(np.float32),
                X_csp[i:i+1].astype(np.float32)
            ]

    # Initialize Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 1. Standard Conversion (Float32/Dynamic)
    print("\n[1/2] Converting Standard TFLite Model...")
    tflite_model = converter.convert()
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    std_path = os.path.join(OUTPUT_DIR, "neurofetal_model.tflite")
    with open(std_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved: {std_path} ({len(tflite_model)/1024:.1f} KB)")

    # 2. Int8 Quantization
    print("\n[2/2] Converting Int8 Quantized Model (Edge Optimized)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Ensure full integer quantization for TPU/NPU compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_quant_model = converter.convert()
        quant_path = os.path.join(OUTPUT_DIR, "neurofetal_model_quant_int8.tflite")
        with open(quant_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"  Saved: {quant_path} ({len(tflite_quant_model)/1024:.1f} KB)")
        print(f"  Compression: {len(tflite_model)/len(tflite_quant_model):.1f}x smaller")
    except Exception as e:
        print(f"⚠️ Int8 Conversion Failed: {e}")
        print("  (This often happens if model has ops not supported in Int8. Check logs.)")

if __name__ == "__main__":
    convert_model()

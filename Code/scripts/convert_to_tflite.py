"""
Convert Trained Keras Model to TFLite
=====================================
Converts the best performing fold model to TFLite format for mobile deployment.
Handles custom layers (CrossModalAttention) and architectural constraints.
"""

import os
import tensorflow as tf
import numpy as np
import sys

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from model import CrossModalAttention
except ImportError:
    print("Warning: Could not import CrossModalAttention from utils. Defining dummy for loading.")
    # Fallback if utils not found (e.g. slight path mismatch)
    class CrossModalAttention(tf.keras.layers.Layer):
        def __init__(self, embed_dim=256, num_heads=4, dropout=0.1, **kwargs):
            super().__init__(**kwargs)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Code", "models", "tflite")
FOLD_TO_CONVERT = 1  # Convert Fold 1 (usually representative)

def convert_model():
    model_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{FOLD_TO_CONVERT}.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run training first.")
        return

    print(f"Loading Keras model from: {model_path}")
    
    # Load model with custom objects
    # Note: custom objects are needed even if we just convert, as Keras needs to reconstruct the graph
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'CrossModalAttention': CrossModalAttention},
            compile=False # We don't need the optimizer/loss for inference conversion
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded successfully.")
    print("Converting to TFLite...")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 1. Standard Conversion
    tflite_model = converter.convert()
    
    # Ensure output dir exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, "neurofetal_model.tflite")
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"✓ Success! Standard TFLite model saved to: {output_path}")
    print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

    # 2. Optimized Conversion (Quantization) - Optional but recommended for mobile
    print("\nCreating Quantized version (Generic)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    quant_output_path = os.path.join(OUTPUT_DIR, "neurofetal_model_quant.tflite")
    with open(quant_output_path, "wb") as f:
        f.write(tflite_quant_model)
        
    print(f"✓ Success! Quantized TFLite model saved to: {quant_output_path}")
    print(f"  Size: {len(tflite_quant_model) / 1024:.2f} KB")

if __name__ == "__main__":
    convert_model()

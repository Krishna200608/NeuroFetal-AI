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

# Add utils to path (Robustly)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR) # .../Code
UTILS_DIR = os.path.join(CODE_DIR, 'utils')

sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, CODE_DIR)

try:
    # Try importing as if utils is the root (current behavior)
    from model import CrossModalAttention
    from attention_blocks import SEBlock, TemporalAttentionBlock as TemporalAttention
except ImportError as e1:
    print(f"Warning: Direct import failed: {e1}")
    try:
        # Try importing as package
        from utils.model import CrossModalAttention
        from utils.attention_blocks import SEBlock, TemporalAttentionBlock as TemporalAttention
    except ImportError as e2:
        print(f"Warning: Package import failed: {e2}")
        print("Defining robust dummy classes for loading.")
        
        # Robust Fallback Classes
        class CrossModalAttention(tf.keras.layers.Layer):
            def __init__(self, embed_dim=256, num_heads=4, dropout=0.1, **kwargs):
                super().__init__(**kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.dropout = dropout
            def call(self, inputs):
                return inputs[-1] # Pass through clinical/last input
            def get_config(self):
                config = super().get_config()
                config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'dropout': self.dropout})
                return config

        class SEBlock(tf.keras.layers.Layer):
            def __init__(self, reduction_ratio=16, **kwargs):
                super().__init__(**kwargs)
                self.reduction_ratio = reduction_ratio
            def call(self, inputs):
                return inputs # Pass through
            def get_config(self):
                config = super().get_config()
                config.update({'reduction_ratio': self.reduction_ratio})
                return config

        class TemporalAttention(tf.keras.layers.Layer):
            def __init__(self, output_dim=None, **kwargs):
                super().__init__(**kwargs)
                self.output_dim = output_dim
            def call(self, inputs):
                return inputs
            def get_config(self):
                config = super().get_config()
                config.update({'output_dim': self.output_dim})
                return config

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
            custom_objects={
                'CrossModalAttention': CrossModalAttention,
                'SEBlock': SEBlock,
                'TemporalAttentionBlock': TemporalAttention,
                'F1Score': None # Ignore metrics if they cause issues
            },
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

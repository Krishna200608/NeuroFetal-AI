import os
import tensorflow as tf
import streamlit as st
from utils.attention_blocks import SEBlock, TemporalAttentionBlock
from utils.model import CrossModalAttention
from utils.focal_loss import FocalLoss

@st.cache_resource
def load_model():
    """Load the enhanced 3-input model with fallback to legacy model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust path: utils/model_loader.py -> Code/models
    model_dir = os.path.join(current_dir, "..", "models")
    
    # Enhanced model paths (3-input: FHR + tabular_16 + CSP_19)
    enhanced_local = os.path.join(model_dir, "enhanced_model_fold_1.keras")
    enhanced_colab = '/content/drive/MyDrive/Research_Project/Code/models/enhanced_model_fold_1.keras'
    
    # Legacy model paths (2-input: FHR + tabular_3)
    legacy_local = os.path.join(model_dir, "best_model_fold_5.keras")
    legacy_colab = '/content/drive/MyDrive/Research_Project/Code/models/best_model_fold_5.keras'
    
    custom_objects = {
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'CrossModalAttention': CrossModalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.5, alpha=0.75)
    }
    
    model = None
    path_used = None
    is_enhanced = False
    
    # Try enhanced model first
    for path in [enhanced_local, enhanced_colab]:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                path_used = path
                is_enhanced = True
                break
            except Exception as e:
                print(f"Warning: Enhanced model load failed ({path}): {e}")
    
    # Fallback to legacy model
    if model is None:
        for path in [legacy_local, legacy_colab]:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                    path_used = path
                    is_enhanced = False
                    break
                except Exception as e:
                    print(f"Warning: Legacy model load failed ({path}): {e}")
    
    if model is None:
        st.error("Critical Error: No model found. Please verify model files exist in Code/models/")
    
    return model, path_used, is_enhanced

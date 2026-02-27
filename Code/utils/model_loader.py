import os
import tensorflow as tf
import streamlit as st
import pickle
import numpy as np
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


@st.cache_resource
def load_ensemble_models():
    """
    Load all 3 ensemble models for variance-based uncertainty estimation.
    
    Returns:
        models: dict with keys 'resnet', 'inception', 'xgboost' (values may be None)
        meta_learner: Calibrated stacking meta-learner (or None)
        n_loaded: number of models successfully loaded
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models")
    
    custom_objects = {
        'SEBlock': SEBlock,
        'TemporalAttentionBlock': TemporalAttentionBlock,
        'CrossModalAttention': CrossModalAttention,
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': FocalLoss(gamma=2.5, alpha=0.75)
    }
    
    models = {'resnet': None, 'inception': None, 'xgboost': None}
    
    # Model A: AttentionFusionResNet (fold 1)
    resnet_path = os.path.join(model_dir, "enhanced_model_fold_1.keras")
    if os.path.exists(resnet_path):
        try:
            models['resnet'] = tf.keras.models.load_model(
                resnet_path, custom_objects=custom_objects, compile=False
            )
            print("[OK] Loaded ResNet model")
        except Exception as e:
            print(f"[ERROR] ResNet load failed: {e}")
    
    # Model B: InceptionNet (fold 1)
    inception_path = os.path.join(model_dir, "inception_model_fold_1.keras")
    if os.path.exists(inception_path):
        try:
            models['inception'] = tf.keras.models.load_model(
                inception_path, custom_objects=custom_objects, compile=False
            )
            print("[OK] Loaded InceptionNet model")
        except Exception as e:
            print(f"[ERROR] InceptionNet load failed: {e}")
    
    # Model C: XGBoost (fold 1)
    xgb_path = os.path.join(model_dir, "xgboost_model_fold_1.pkl")
    if os.path.exists(xgb_path):
        try:
            with open(xgb_path, 'rb') as f:
                models['xgboost'] = pickle.load(f)
            print("[OK] Loaded XGBoost model")
        except Exception as e:
            print(f"[ERROR] XGBoost load failed: {e}")
    
    # Meta-learner (calibrated stacking)
    meta_learner = None
    meta_path = os.path.join(model_dir, "stacking_meta_learner.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                meta_learner = pickle.load(f)
            print("[OK] Loaded calibrated meta-learner")
        except Exception as e:
            print(f"[ERROR] Meta-learner load failed: {e}")
    
    n_loaded = sum(1 for v in models.values() if v is not None)
    print(f"Ensemble: {n_loaded}/3 models loaded")
    
    return models, meta_learner, n_loaded

"""
Enhanced Fusion ResNet Model for NeuroFetal AI
==============================================
3-input architecture with advanced attention mechanisms.

Architecture:
- Input 1: FHR signal (1200, 1) - 1D ResNet with SE Blocks + Attention
- Input 2: Tabular features (16,) - Dense network
- Input 3: CSP features (19,) - Dense network

Fusion Strategy (Hybrid):
- FHR × Tabular via Multiply (Paper 7 proven mechanism)
- Combined + CSP via Concatenate (new modality)

Key Enhancements:
- Squeeze-and-Excitation (SE) Blocks for channel attention
- Multi-Head Self-Attention for temporal patterns
- Multi-Scale feature extraction option
- Dropout and BatchNorm for regularization

Reference: 
- Paper 7: Fusion ResNet baseline
- Hu et al., 2018: SE Blocks
- Vaswani et al., 2017: Attention
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Import attention blocks
try:
    from attention_blocks import SEBlock, TemporalAttentionBlock, MultiScaleBlock
except ImportError:
    # Inline definition if import fails (for standalone testing)
    SEBlock = None
    TemporalAttentionBlock = None
    MultiScaleBlock = None


# ============================================================================
# Building Blocks
# ============================================================================

def se_block_inline(x, ratio=16):
    """Inline SE block if attention_blocks not available."""
    filters = x.shape[-1]
    reduced = max(1, filters // ratio)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(reduced, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    
    return layers.Multiply()([x, se])


def residual_block(x, filters, kernel_size=3, stride=1, use_se=True):
    """
    Residual block with optional SE attention.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Convolution kernel size
        stride: Stride for downsampling
        use_se: Whether to apply SE block
    """
    shortcut = x
    
    # First Conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # SE Block (channel attention)
    if use_se:
        if SEBlock is not None:
            x = SEBlock(reduction_ratio=16)(x)
        else:
            x = se_block_inline(x, ratio=16)
    
    # Adjust shortcut if shapes differ
    if x.shape[-1] != shortcut.shape[-1] or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def temporal_attention_inline(x, num_heads=4, key_dim=32):
    """Inline temporal attention if attention_blocks not available."""
    # Simple multi-head attention with residual
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=0.1
    )(x, x)
    attn = layers.Dropout(0.1)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


# ============================================================================
# Original Model (Backward Compatibility)
# ============================================================================

def build_fusion_resnet(input_shape_ts=(1200, 1), input_shape_tab=(3,)):
    """
    Original 2-input Fusion ResNet (Paper 7 baseline).
    Kept for backward compatibility and ablation studies.
    """
    # --- Branch 1: Time Series (ResNet 1D) ---
    input_ts = Input(shape=input_shape_ts, name='input_fhr')
    
    x1 = layers.Conv1D(64, 7, strides=2, padding='same')(input_ts)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.MaxPooling1D(3, strides=2, padding='same')(x1)
    
    # Residual Blocks
    x1 = residual_block(x1, 64, use_se=False)
    x1 = residual_block(x1, 128, stride=2, use_se=False)
    x1 = residual_block(x1, 128, use_se=False)
    
    x1 = layers.GlobalAveragePooling1D()(x1)  # (Batch, 128)
    
    # --- Branch 2: Tabular ---
    input_tab = Input(shape=input_shape_tab, name='input_tabular')
    
    x2 = layers.Dense(10, activation='relu')(input_tab)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(128, activation='relu')(x2)
    x2 = layers.Dropout(0.3)(x2)
    
    # --- Fusion (Multiply) ---
    fusion = layers.Multiply()([x1, x2])
    
    # --- Output ---
    output = layers.Dense(1, activation='sigmoid', name='output')(fusion)
    
    model = models.Model(inputs=[input_ts, input_tab], outputs=output)
    return model


# ============================================================================
# Enhanced 3-Input Model (New Architecture)
# ============================================================================

def build_enhanced_fusion_resnet(
    input_shape_fhr=(1200, 1),
    input_shape_tabular=(16,),
    input_shape_csp=(19,),
    use_se_blocks=True,
    use_attention=True,
    use_multi_scale=False,
    dropout_rate=0.3
):
    """
    Enhanced 3-input Fusion ResNet with attention mechanisms.
    
    Architecture Diagram:
    
    FHR (1200,1)          Tabular (16,)         CSP (19,)
         |                     |                    |
    Conv1D(64) + SE       Dense(32)            Dense(64)
         |                     |                    |
    ResBlock(64) + SE     Dense(128)           Dense(128)
         |                     |                    |
    ResBlock(128) + SE         |                    |
         |                     |                    |
    [MultiHeadAttn]            |                    |
         |                     |                    |
    GlobalAvgPool          (128-dim)            (128-dim)
         |                     |                    |
      (128-dim)                |                    |
         |___________×_________|                    |
                    |                               |
              (Multiply: 128-dim)                   |
                    |_____________concat____________|
                                |
                         (256-dim fusion)
                                |
                          Dense(64) + Dropout
                                |
                          Dense(1, sigmoid)
    
    Args:
        input_shape_fhr: Shape of FHR signal (time_steps, 1)
        input_shape_tabular: Shape of tabular features
        input_shape_csp: Shape of CSP features
        use_se_blocks: Enable SE attention in ResBlocks
        use_attention: Enable temporal self-attention
        use_multi_scale: Enable multi-scale feature extraction
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras Model with 3 inputs
    """
    
    # =========================================================================
    # Branch 1: FHR Signal (ResNet + SE + Attention)
    # =========================================================================
    input_fhr = Input(shape=input_shape_fhr, name='input_fhr')
    
    # Initial convolution
    x1 = layers.Conv1D(64, 7, strides=2, padding='same', name='fhr_conv1')(input_fhr)
    x1 = layers.BatchNormalization(name='fhr_bn1')(x1)
    x1 = layers.Activation('relu', name='fhr_relu1')(x1)
    x1 = layers.MaxPooling1D(3, strides=2, padding='same', name='fhr_pool1')(x1)
    
    # Multi-scale feature extraction (optional)
    if use_multi_scale and MultiScaleBlock is not None:
        x1 = MultiScaleBlock(filters=64)(x1)
    
    # Residual blocks with SE attention
    x1 = residual_block(x1, 64, use_se=use_se_blocks)
    x1 = residual_block(x1, 128, stride=2, use_se=use_se_blocks)
    x1 = residual_block(x1, 128, use_se=use_se_blocks)
    
    # Temporal self-attention (captures long-range dependencies)
    if use_attention:
        if TemporalAttentionBlock is not None:
            x1 = TemporalAttentionBlock(num_heads=4, key_dim=32)(x1)
        else:
            x1 = temporal_attention_inline(x1, num_heads=4, key_dim=32)
    
    # Global pooling
    fhr_features = layers.GlobalAveragePooling1D(name='fhr_gap')(x1)  # (batch, 128)
    
    # =========================================================================
    # Branch 2: Tabular Features (Dense Network)
    # =========================================================================
    input_tabular = Input(shape=input_shape_tabular, name='input_tabular')
    
    x2 = layers.Dense(32, activation='relu', name='tab_dense1')(input_tabular)
    x2 = layers.Dropout(dropout_rate, name='tab_drop1')(x2)
    x2 = layers.Dense(128, activation='relu', name='tab_dense2')(x2)
    x2 = layers.Dropout(dropout_rate, name='tab_drop2')(x2)
    tabular_features = x2  # (batch, 128)
    
    # =========================================================================
    # Branch 3: CSP Features (Dense Network)
    # =========================================================================
    input_csp = Input(shape=input_shape_csp, name='input_csp')
    
    x3 = layers.Dense(64, activation='relu', name='csp_dense1')(input_csp)
    x3 = layers.Dropout(dropout_rate, name='csp_drop1')(x3)
    x3 = layers.Dense(128, activation='relu', name='csp_dense2')(x3)
    x3 = layers.Dropout(dropout_rate, name='csp_drop2')(x3)
    csp_features = x3  # (batch, 128)
    
    # =========================================================================
    # Hybrid Fusion Strategy
    # =========================================================================
    # Step 1: Multiply FHR × Tabular (Paper 7 proven mechanism)
    # This acts as "attention gating" - tabular features modulate FHR features
    fhr_tab_fusion = layers.Multiply(name='multiply_fusion')([fhr_features, tabular_features])
    
    # Step 2: Concatenate with CSP (new modality)
    # CSP features add discriminative power from FHR-UC correlation patterns
    fusion = layers.Concatenate(name='concat_fusion')([fhr_tab_fusion, csp_features])  # (batch, 256)
    
    # =========================================================================
    # Classification Head
    # =========================================================================
    x = layers.Dense(64, activation='relu', name='head_dense1')(fusion)
    x = layers.Dropout(dropout_rate, name='head_drop1')(x)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # =========================================================================
    # Build Model
    # =========================================================================
    model = models.Model(
        inputs=[input_fhr, input_tabular, input_csp],
        outputs=output,
        name='EnhancedFusionResNet'
    )
    
    return model


# ============================================================================
# Model Factory (For Ablation Studies)
# ============================================================================

def build_model_for_ablation(config):
    """
    Factory function for creating model variants for ablation studies.
    
    Args:
        config: Dict with model configuration:
            - 'type': 'baseline' | 'enhanced'
            - 'use_se': bool
            - 'use_attention': bool
            - 'use_csp': bool
            - 'input_shapes': dict
    
    Returns:
        Configured Keras Model
    """
    if config.get('type') == 'baseline':
        return build_fusion_resnet(
            input_shape_ts=config.get('input_shape_fhr', (1200, 1)),
            input_shape_tab=config.get('input_shape_tabular', (3,))
        )
    else:
        return build_enhanced_fusion_resnet(
            input_shape_fhr=config.get('input_shape_fhr', (1200, 1)),
            input_shape_tabular=config.get('input_shape_tabular', (16,)),
            input_shape_csp=config.get('input_shape_csp', (19,)),
            use_se_blocks=config.get('use_se', True),
            use_attention=config.get('use_attention', True),
            use_multi_scale=config.get('use_multi_scale', False),
            dropout_rate=config.get('dropout_rate', 0.3)
        )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing NeuroFetal AI Model Architectures")
    print("="*60)
    
    # Test 1: Original Baseline Model
    print("\n1. Original Fusion ResNet (2-input baseline):")
    model_baseline = build_fusion_resnet()
    print(f"   Inputs: {[i.name for i in model_baseline.inputs]}")
    print(f"   Parameters: {model_baseline.count_params():,}")
    
    # Test 2: Enhanced 3-Input Model
    print("\n2. Enhanced Fusion ResNet (3-input with attention):")
    model_enhanced = build_enhanced_fusion_resnet(
        input_shape_fhr=(1200, 1),
        input_shape_tabular=(16,),
        input_shape_csp=(19,),
        use_se_blocks=True,
        use_attention=True
    )
    print(f"   Inputs: {[i.name for i in model_enhanced.inputs]}")
    print(f"   Parameters: {model_enhanced.count_params():,}")
    
    # Test 3: Model Summary
    print("\n3. Enhanced Model Architecture:")
    print("-"*60)
    model_enhanced.summary(line_length=100)
    
    # Test 4: Forward pass
    print("\n4. Forward Pass Test:")
    import numpy as np
    batch_size = 4
    
    x_fhr = np.random.randn(batch_size, 1200, 1).astype(np.float32)
    x_tabular = np.random.randn(batch_size, 16).astype(np.float32)
    x_csp = np.random.randn(batch_size, 19).astype(np.float32)
    
    predictions = model_enhanced.predict([x_fhr, x_tabular, x_csp], verbose=0)
    print(f"   Input shapes: FHR={x_fhr.shape}, Tab={x_tabular.shape}, CSP={x_csp.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions.flatten()}")
    
    print("\n" + "="*60)
    print("All model tests passed!")
    print("="*60)

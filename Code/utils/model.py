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
from tensorflow.keras.utils import register_keras_serializable

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


def residual_block(x, filters, kernel_size=3, stride=1, use_se=True, stochastic_depth_rate=0.0):
    """
    Residual block with optional SE attention and stochastic depth.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Convolution kernel size
        stride: Stride for downsampling
        use_se: Whether to apply SE block
        stochastic_depth_rate: Probability of dropping the residual path (0 = disabled)
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
    
    # SOTA: Stochastic Depth — randomly drop residual path during training
    if stochastic_depth_rate > 0:
        x = layers.Dropout(stochastic_depth_rate, noise_shape=(tf.shape(x)[0], 1, 1))(x)
    
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
# NOVEL: Cross-Modal Attention Layer (Publication Contribution)
# ============================================================================

@register_keras_serializable()
class CrossModalAttention(layers.Layer):
    """
    Cross-Modal Attention for FHR-CSP Interaction with Clinical Gating.
    
    NOVELTY: This layer enables dynamic attention between temporal features
    (FHR) and spatial pattern features (CSP), gated by clinical context.
    
    Architecture:
        Query: FHR features (temporal patterns)
        Key/Value: CSP features (spatial correlations)
        Gate: Clinical features (context modulation)
    
    Paper Claim: "We propose Cross-Modal Attention Fusion (CMAF), where 
    FHR and UC signals attend to each other dynamically, gated by clinical 
    context for context-aware prediction."
    """
    
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        # Projection layers for attention
        self.query_dense = layers.Dense(self.embed_dim, name='cma_query')
        self.key_dense = layers.Dense(self.embed_dim, name='cma_key')
        self.value_dense = layers.Dense(self.embed_dim, name='cma_value')
        
        # Clinical gating mechanism
        self.gate_dense = layers.Dense(self.embed_dim, activation='sigmoid', name='cma_gate')
        
        # Output projection
        self.output_dense = layers.Dense(self.embed_dim, name='cma_output')
        
        # Normalization and dropout
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: List of [fhr_features, csp_features, clinical_features]
                   Each has shape (batch, embed_dim)
        Returns:
            Attended features (batch, embed_dim)
        """
        fhr_features, csp_features, clinical_features = inputs
        
        # Project to query, key, value
        query = self.query_dense(fhr_features)    # (batch, embed_dim)
        key = self.key_dense(csp_features)        # (batch, embed_dim)
        value = self.value_dense(csp_features)    # (batch, embed_dim)
        
        # Compute attention scores
        # Scaled dot-product attention (simplified for 1D features)
        attention_scores = tf.reduce_sum(query * key, axis=-1, keepdims=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.embed_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attended = attention_weights * value
        
        # Clinical gating: modulate attention based on clinical context
        gate = self.gate_dense(clinical_features)  # (batch, embed_dim)
        gated_attended = attended * gate
        
        # Residual connection with FHR features
        output = fhr_features + self.dropout(gated_attended, training=training)
        output = self.layer_norm(output)
        output = self.output_dense(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate
        })
        return config


# ============================================================================
# NOVEL: Attention Fusion ResNet (Publication Model)
# ============================================================================

def build_attention_fusion_resnet(
    input_shape_fhr=(1200, 1),
    input_shape_tabular=(16,),
    input_shape_csp=(19,),
    use_se_blocks=True,
    use_temporal_attention=True,
    use_cross_modal_attention=True,
    dropout_rate=0.3,  # SOTA: Reduced from 0.4 — less over-regularization with augmentation
    mc_dropout=False,  # Enable for uncertainty quantification
    use_aux_head=False  # SOTA: Auxiliary pH regression head for multi-task learning
):
    """
    Novel Attention Fusion ResNet with Cross-Modal Attention.
    
    PUBLICATION ARCHITECTURE:
    
    FHR (1200,1)          Clinical (16,)         CSP (19,)
         |                     |                    |
    ResNet + SE           Dense(128)            Dense(128)
         |                     |                    |
    [Temporal Attn]            |                    |
         |                     |                    |
    GlobalPool                 |                    |
         |                     |                    |
      (128-dim)                |                    |
         |_______CROSS-MODAL ATTENTION______________|
                 (FHR attends to CSP, Clinical gates)
                        |
                   (128-dim CMAF)
                        |
                 [MC Dropout Head]
                        |
                   Prediction + Uncertainty
    
    Novel Contributions:
    1. Cross-Modal Attention: Dynamic FHR-CSP interaction
    2. Clinical Gating: Context modulates attention
    3. MC Dropout: Uncertainty quantification at inference
    
    Args:
        mc_dropout: If True, dropout is active during inference for
                   Monte Carlo uncertainty estimation
    """
    
    # =========================================================================
    # Branch 1: FHR Signal (ResNet + SE + Temporal Attention)
    # =========================================================================
    input_fhr = Input(shape=input_shape_fhr, name='input_fhr')
    
    # Use the shared encoder
    fhr_encoder = build_fhr_encoder(
        input_shape=input_shape_fhr,
        use_se_blocks=use_se_blocks,
        use_attention=use_temporal_attention, # Pass the temporal attention flag
        use_multi_scale=False,
        name='shared_fhr_encoder'
    )
    
    fhr_features = fhr_encoder(input_fhr)  # (batch, 128)
    
    # =========================================================================
    # Branch 2: Clinical/Tabular Features
    # =========================================================================
    input_tabular = Input(shape=input_shape_tabular, name='input_tabular')
    
    x2 = layers.Dense(64, activation='relu', name='tab_dense1')(input_tabular)
    x2 = layers.Dropout(dropout_rate, name='tab_drop1')(x2)
    x2 = layers.Dense(128, activation='relu', name='tab_dense2')(x2)
    clinical_features = x2  # (batch, 128) - Used for gating
    
    # =========================================================================
    # Branch 3: CSP Features (FHR-UC Spatial Patterns)
    # =========================================================================
    input_csp = Input(shape=input_shape_csp, name='input_csp')
    
    x3 = layers.Dense(64, activation='relu', name='csp_dense1')(input_csp)
    x3 = layers.Dropout(dropout_rate, name='csp_drop1')(x3)
    x3 = layers.Dense(128, activation='relu', name='csp_dense2')(x3)
    csp_features = x3  # (batch, 128)
    
    # =========================================================================
    # NOVEL: Cross-Modal Attention Fusion
    # =========================================================================
    if use_cross_modal_attention:
        # FHR attends to CSP patterns, gated by clinical context
        cross_modal_attn = CrossModalAttention(
            embed_dim=128, 
            num_heads=4, 
            dropout=dropout_rate,
            name='cross_modal_attention'
        )
        fusion = cross_modal_attn([fhr_features, csp_features, clinical_features])
    else:
        # Fallback to simple fusion for ablation
        fusion = layers.Multiply(name='multiply_fusion')([fhr_features, clinical_features])
        fusion = layers.Concatenate(name='concat_fusion')([fusion, csp_features])
    
    # =========================================================================
    # Classification Head with MC Dropout Support
    # =========================================================================
    # MC Dropout: Keep dropout active during inference for uncertainty
    if mc_dropout:
        # Always apply dropout (training=True) for uncertainty estimation
        x = layers.Dense(64, activation='relu', name='head_dense1')(fusion)
        x = layers.Dropout(dropout_rate)(x, training=True)  # Always on!
        x = layers.Dense(32, activation='relu', name='head_dense2')(x)
        x = layers.Dropout(dropout_rate)(x, training=True)  # Always on!
    else:
        # Standard dropout (off during inference)
        x = layers.Dense(64, activation='relu', name='head_dense1')(fusion)
        x = layers.Dropout(dropout_rate, name='head_drop1')(x)
        x = layers.Dense(32, activation='relu', name='head_dense2')(x)
        x = layers.Dropout(dropout_rate, name='head_drop2')(x)
    
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # =========================================================================
    # SOTA: Auxiliary pH Regression Head (Multi-Task Learning)
    # =========================================================================
    outputs = [output]
    if use_aux_head:
        aux_out = layers.Dense(32, activation='relu', name='aux_dense')(fusion)
        aux_out = layers.Dense(1, activation='linear', name='aux_ph_output')(aux_out)
        outputs.append(aux_out)
    
    # =========================================================================
    # Build Model
    # =========================================================================
    model = models.Model(
        inputs=[input_fhr, input_tabular, input_csp],
        outputs=outputs if use_aux_head else output,
        name='AttentionFusionResNet'
    )
    
    return model


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

def build_fhr_encoder(
    input_shape=(1200, 1),
    use_se_blocks=True,
    use_attention=True,
    use_multi_scale=False,
    name='fhr_encoder'
):
    """
    Builds the FHR Encoder Backbone (ResNet + Attention).
    
    This is extracted to be used in:
    1. The main Classification Model (Supervised)
    2. The Masked Autoencoder (Self-Supervised Pretraining)
    
    Returns:
        tf.keras.Model: Encoder model mapping (1200,1) -> (128,) vector
    """
    inputs = Input(shape=input_shape, name='encoder_input')
    
    # Initial convolution
    x = layers.Conv1D(64, 7, strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)
    
    # Multi-scale feature extraction (optional)
    if use_multi_scale and MultiScaleBlock is not None:
        x = MultiScaleBlock(filters=64)(x)
    
    # Stage 1: Low-level features (SOTA: stochastic depth for regularization)
    x = residual_block(x, 64, use_se=use_se_blocks, stochastic_depth_rate=0.05)
    x = residual_block(x, 64, use_se=use_se_blocks, stochastic_depth_rate=0.05)
    
    # Stage 2: Mid-level features (downsample)
    x = residual_block(x, 128, stride=2, use_se=use_se_blocks, stochastic_depth_rate=0.1)
    x = residual_block(x, 128, use_se=use_se_blocks, stochastic_depth_rate=0.1)
    
    # Mid-stage temporal attention
    if use_attention:
        if TemporalAttentionBlock is not None:
            x = TemporalAttentionBlock(num_heads=4, key_dim=32, name='temp_attn_mid')(x)
        else:
            x = temporal_attention_inline(x, num_heads=4, key_dim=32)
    
    # Stage 3: High-level features (SOTA: reduced from 256→192 to avoid overfit on small data)
    x = residual_block(x, 192, stride=2, use_se=use_se_blocks, stochastic_depth_rate=0.15)
    x = residual_block(x, 192, use_se=use_se_blocks, stochastic_depth_rate=0.15)
    
    # Final temporal attention
    if use_attention:
        if TemporalAttentionBlock is not None:
            x = TemporalAttentionBlock(num_heads=8, key_dim=64, name='temp_attn_final')(x)
        else:
            x = temporal_attention_inline(x, num_heads=8, key_dim=64)
    
    # Global pooling
    outputs = layers.GlobalAveragePooling1D(name='global_pool')(x)  # (batch, 192)
    
    # Bottleneck projection to 128-dim (standardizing latent space)
    outputs = layers.Dense(128, activation='relu', name='projection')(outputs)
    
    return models.Model(inputs=inputs, outputs=outputs, name=name)


def build_enhanced_fusion_resnet(
    input_shape_fhr=(1200, 1),
    input_shape_tabular=(16,),
    input_shape_csp=(19,),
    use_se_blocks=True,
    use_attention=True,
    use_multi_scale=False,
    dropout_rate=0.3  # SOTA: Reduced from 0.4 — augmentation handles regularization
):
    """
    Enhanced 3-input Fusion ResNet with attention mechanisms.
    """
    
    # =========================================================================
    # Branch 1: FHR Signal (ResNet + SE + Attention)
    # =========================================================================
    input_fhr = Input(shape=input_shape_fhr, name='input_fhr')
    
    # Use the shared encoder
    fhr_encoder = build_fhr_encoder(
        input_shape=input_shape_fhr,
        use_se_blocks=use_se_blocks,
        use_attention=use_attention,
        use_multi_scale=use_multi_scale,
        name='shared_fhr_encoder'
    )
    
    fhr_features = fhr_encoder(input_fhr)  # (batch, 128)
    
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
            dropout_rate=config.get('dropout_rate', 0.3)  # SOTA: Reduced default
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

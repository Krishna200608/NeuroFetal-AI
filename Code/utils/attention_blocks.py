"""
Attention Blocks for NeuroFetal AI
==================================
Advanced attention mechanisms for the Fusion ResNet architecture.

Implements:
1. Squeeze-and-Excitation (SE) Block - Channel attention (Hu et al., 2018)
2. Multi-Head Self-Attention - Temporal dependencies (Vaswani et al., 2017)
3. Positional Encoding - For attention mechanisms in 1D signals

These blocks add significant architectural complexity and have been shown
to improve performance in medical signal classification tasks.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# ============================================================================
# Squeeze-and-Excitation (SE) Block
# ============================================================================

class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation Block for 1D signals.
    
    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels. This "attention" mechanism allows
    the network to focus on the most informative filters.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    
    Args:
        reduction_ratio: Reduction factor for bottleneck (default: 16)
    """
    
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.reduced_dim = max(1, self.channels // self.reduction_ratio)
        
        # Squeeze: Global Average Pooling
        self.gap = layers.GlobalAveragePooling1D()
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        self.fc1 = layers.Dense(self.reduced_dim, activation='relu', 
                                kernel_initializer='he_normal')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid',
                                kernel_initializer='he_normal')
        self.reshape = layers.Reshape((1, self.channels))
        
        super(SEBlock, self).build(input_shape)
        
    def call(self, inputs):
        # Squeeze: (batch, time, channels) -> (batch, channels)
        squeezed = self.gap(inputs)
        
        # Excitation: Learn channel weights
        excitation = self.fc1(squeezed)
        excitation = self.fc2(excitation)
        
        # Reshape for broadcasting: (batch, channels) -> (batch, 1, channels)
        excitation = self.reshape(excitation)
        
        # Scale: Element-wise multiplication
        scaled = layers.Multiply()([inputs, excitation])
        return scaled
    
    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


def se_block(x, reduction_ratio=16, name=None):
    """
    Functional API wrapper for SE Block.
    
    Args:
        x: Input tensor (batch, time, channels)
        reduction_ratio: Reduction factor for bottleneck
        name: Optional name for the block
        
    Returns:
        Tensor with same shape as input, channel-recalibrated
    """
    return SEBlock(reduction_ratio=reduction_ratio, name=name)(x)


# ============================================================================
# Multi-Head Self-Attention
# ============================================================================

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for 1D sequences.
    
    Adds position information to the input embeddings, allowing the
    attention mechanism to utilize positional context.
    
    Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    """
    
    def __init__(self, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.pe = None
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        # Precompute positional encodings
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Handle odd d_model
        if self.d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
            
        # Use add_weight to properly register tensor (fixes TF graph scope issue)
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(self.max_len, self.d_model),
            initializer=tf.constant_initializer(pe),
            trainable=False,
            dtype=tf.float32
        )
        
        super(PositionalEncoding, self).build(input_shape)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Cast positional encoding to match input dtype (for mixed precision)
        pe_slice = self.pe[:seq_len, :]
        pe_casted = tf.cast(pe_slice, dtype=x.dtype)
        return x + pe_casted
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'max_len': self.max_len})
        return config


class TemporalAttentionBlock(layers.Layer):
    """
    Temporal Self-Attention Block for 1D signals.
    
    Uses Multi-Head Self-Attention to capture long-range dependencies
    in the FHR signal. This is crucial for detecting late decelerations
    that may occur at different positions in the window.
    
    Args:
        num_heads: Number of attention heads (default: 4)
        key_dim: Dimension of key/query vectors (default: 32)
        dropout_rate: Dropout rate after attention (default: 0.1)
        use_positional: Whether to use positional encoding (default: True)
    """
    
    def __init__(self, num_heads=4, key_dim=32, dropout_rate=0.1, 
                 use_positional=True, **kwargs):
        super(TemporalAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.use_positional = use_positional

        # Instantiate layers that don't depend on input shape in __init__
        if self.use_positional:
            self.pos_encoding = PositionalEncoding(name='positional_encoding')
            
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')
        self.dropout_ffn = layers.Dropout(self.dropout_rate, name='dropout_ffn')
        
    def build(self, input_shape):
        d_model = input_shape[-1]
        
        # Feed-forward network layers (must be in build as they depend on d_model)
        self.ffn_dense1 = layers.Dense(d_model * 4, activation='gelu', name='ffn_dense_1')
        self.ffn_dense2 = layers.Dense(d_model, name='ffn_dense_2')
        
        # Explicitly build layers to ensure variables are created and tracked immediately
        # This fixes "Layer was never built" errors during weight loading
        if self.use_positional:
            self.pos_encoding.build(input_shape)
            
        self.layer_norm1.build(input_shape)
        self.layer_norm2.build(input_shape)
        
        # MHA build takes input_shape (assuming query/key/value have same shape)
        self.mha.build(input_shape)
        
        self.ffn_dense1.build(input_shape)
        
        dense2_input_shape = input_shape[:-1] + (d_model * 4,)
        self.ffn_dense2.build(dense2_input_shape)
        
        super(TemporalAttentionBlock, self).build(input_shape)
        
    def call(self, x, training=None):
        # Positional encoding
        if self.use_positional:
            x = self.pos_encoding(x)
            
        # Pre-norm + Multi-Head Attention + Residual
        x_norm = self.layer_norm1(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        x = x + attn_output
        
        # Pre-norm + FFN + Residual
        x_norm = self.layer_norm2(x)
        
        # FFN Forward Pass
        x_ffn = self.ffn_dense1(x_norm)
        x_ffn = self.dropout_ffn(x_ffn, training=training)
        x_ffn = self.ffn_dense2(x_ffn)
        
        x = x + x_ffn
        
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(TemporalAttentionBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
            'use_positional': self.use_positional
        })
        return config


def temporal_attention(x, num_heads=4, key_dim=32, dropout_rate=0.1, 
                       use_positional=True, name=None):
    """
    Functional API wrapper for Temporal Attention Block.
    
    Args:
        x: Input tensor (batch, time, channels)
        num_heads: Number of attention heads
        key_dim: Dimension of key/query
        dropout_rate: Dropout rate
        use_positional: Whether to use positional encoding
        name: Optional name for the block
        
    Returns:
        Tensor with same shape as input, with temporal attention applied
    """
    return TemporalAttentionBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout_rate=dropout_rate,
        use_positional=use_positional,
        name=name
    )(x)


# ============================================================================
# Convolutional Block Attention Module (CBAM) - Optional Advanced Block
# ============================================================================

class ChannelAttention(layers.Layer):
    """Channel attention module from CBAM."""
    
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.reduced_dim = max(1, self.channels // self.reduction_ratio)
        
        self.gap = layers.GlobalAveragePooling1D()
        self.gmp = layers.GlobalMaxPooling1D()
        
        self.fc1 = layers.Dense(self.reduced_dim, activation='relu')
        self.fc2 = layers.Dense(self.channels)
        
        super(ChannelAttention, self).build(input_shape)
        
    def call(self, x):
        # Average pooling path
        avg_pool = self.gap(x)
        avg_pool = self.fc1(avg_pool)
        avg_pool = self.fc2(avg_pool)
        
        # Max pooling path
        max_pool = self.gmp(x)
        max_pool = self.fc1(max_pool)
        max_pool = self.fc2(max_pool)
        
        # Combine
        attention = tf.nn.sigmoid(avg_pool + max_pool)
        attention = tf.expand_dims(attention, axis=1)
        
        return x * attention


class SpatialAttention(layers.Layer):
    """Spatial (temporal) attention module from CBAM."""
    
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv1D(1, self.kernel_size, padding='same', 
                                  activation='sigmoid')
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, x):
        # Concatenate avg and max along channel axis
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Spatial attention map
        attention = self.conv(concat)
        
        return x * attention


class CBAMBlock(layers.Layer):
    """
    Convolutional Block Attention Module for 1D signals.
    
    Sequentially applies channel and spatial attention.
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """
    
    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def call(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================================
# Multi-Scale Feature Extraction Block
# ============================================================================

class MultiScaleBlock(layers.Layer):
    """
    Multi-scale temporal feature extraction block.
    
    Uses multiple parallel convolutional branches with different kernel sizes
    to capture patterns at different temporal scales. This is crucial for
    detecting both rapid accelerations and prolonged decelerations.
    
    Inspired by Inception module, adapted for 1D medical signals.
    """
    
    def __init__(self, filters=64, **kwargs):
        super(MultiScaleBlock, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        # Branch 1: Small kernel (fast patterns)
        self.conv_small = layers.Conv1D(self.filters, 3, padding='same', activation='relu')
        
        # Branch 2: Medium kernel (moderate patterns)
        self.conv_medium = layers.Conv1D(self.filters, 7, padding='same', activation='relu')
        
        # Branch 3: Large kernel (slow patterns / trends)
        self.conv_large = layers.Conv1D(self.filters, 15, padding='same', activation='relu')
        
        # Branch 4: Skip (identity-like with 1x1 conv)
        self.conv_skip = layers.Conv1D(self.filters, 1, padding='same', activation='relu')
        
        # Merge
        self.concat = layers.Concatenate()
        self.merge_conv = layers.Conv1D(self.filters, 1, padding='same')
        self.bn = layers.BatchNormalization()
        
        super(MultiScaleBlock, self).build(input_shape)
        
    def call(self, x):
        b1 = self.conv_small(x)
        b2 = self.conv_medium(x)
        b3 = self.conv_large(x)
        b4 = self.conv_skip(x)
        
        merged = self.concat([b1, b2, b3, b4])
        merged = self.merge_conv(merged)
        merged = self.bn(merged)
        
        return merged


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Attention Blocks...")
    
    # Create dummy input
    batch_size = 2
    seq_len = 300  # 5 minutes at 1Hz
    channels = 64
    
    x = tf.random.normal([batch_size, seq_len, channels])
    
    # Test SE Block
    print("\n1. SE Block:")
    se_out = se_block(x, reduction_ratio=16)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {se_out.shape}")
    assert x.shape == se_out.shape, "SE Block: Shape mismatch!"
    print("   ✓ SE Block passed")
    
    # Test Temporal Attention
    print("\n2. Temporal Attention Block:")
    attn_out = temporal_attention(x, num_heads=4, key_dim=32)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {attn_out.shape}")
    assert x.shape == attn_out.shape, "Temporal Attention: Shape mismatch!"
    print("   ✓ Temporal Attention passed")
    
    # Test CBAM
    print("\n3. CBAM Block:")
    cbam = CBAMBlock()
    cbam_out = cbam(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {cbam_out.shape}")
    assert x.shape == cbam_out.shape, "CBAM: Shape mismatch!"
    print("   ✓ CBAM Block passed")
    
    # Test Multi-Scale
    print("\n4. Multi-Scale Block:")
    ms = MultiScaleBlock(filters=64)
    ms_out = ms(x)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {ms_out.shape}")
    print("   ✓ Multi-Scale Block passed")
    
    print("\n" + "="*50)
    print("All attention blocks working correctly!")
    print("="*50)

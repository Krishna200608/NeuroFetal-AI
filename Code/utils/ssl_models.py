"""
Self-Supervised Learning Models for NeuroFetal AI
=================================================

Implements Masked Autoencoder (MAE) adaptation for 1D-CNN ResNet.

Strategy: Use a "Context Encoder" / Inpainting approach suitable for CNNs.
1. Mask random segments of the input FHR signal.
2. Encode the masked signal using the backbone (build_fhr_encoder).
3. Decode the latent vector back to the original signal.
4. Minimize reconstruction error (MSE).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

class MaskedAutoencoder(models.Model):
    """
    Masked Autoencoder for 1D Time-Series Pretraining.
    
    Args:
        encoder: Instantiated FHR Encoder model (outputs 128-dim vector)
        input_shape: Shape of input signal (default: (1200, 1))
        masking_ratio: Percentage of signal to mask (default: 0.5)
        mask_block_size: Size of contiguous masked blocks (default: 10)
    """
    
    def __init__(self, encoder, input_shape=(1200, 1), masking_ratio=0.5, mask_block_size=20, **kwargs):
        super(MaskedAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.input_shape_ts = input_shape
        self.masking_ratio = masking_ratio
        self.mask_block_size = mask_block_size
        
        # Build Decoder
        self.decoder = self._build_decoder()
        
    def _build_decoder(self):
        """Builds a lightweight decoder to reconstruct signal from latent space."""
        latent_dim = self.encoder.output.shape[-1] # Should be 128
        
        inp = Input(shape=(latent_dim,))
        
        # Project and Reshape to start upsampling
        # Target length 1200. Start small and upsample.
        # 1200 / 8 = 150.
        x = layers.Dense(150 * 64, activation='relu')(inp)
        x = layers.Reshape((150, 64))(x)
        x = layers.BatchNormalization()(x)
        
        # Upsample Block 1 (150 -> 300)
        x = layers.Conv1DTranspose(64, 7, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Upsample Block 2 (300 -> 600)
        x = layers.Conv1DTranspose(32, 7, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Upsample Block 3 (600 -> 1200)
        x = layers.Conv1DTranspose(16, 7, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Final refinement
        x = layers.Conv1D(1, 7, padding='same', activation='linear')(x)
        
        return models.Model(inp, x, name='fhr_decoder')

    def compile(self, optimizer, loss_fn=None):
        super(MaskedAutoencoder, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn or tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def _apply_mask(self, inputs):
        """
        Applies random masking to the input batch.
        Replaces masked regions with zeros (or a specific token value).
        
        Returns:
            masked_inputs, mask
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = self.input_shape_ts[0]
        
        # Generate random mask
        # We mask 'masking_ratio' of the signal in blocks
        # Simplified approach: Create random binary mask
        
        # Create random noise with lower resolution to simulate blocks
        num_blocks = seq_len // self.mask_block_size
        rand_noise = tf.random.uniform((batch_size, num_blocks, 1))
        
        # Create mask: 1 = keep, 0 = mask
        mask_small = tf.cast(rand_noise > self.masking_ratio, dtype=tf.float32)
        
        # Upsample mask to full resolution
        mask = tf.repeat(mask_small, repeats=self.mask_block_size, axis=1)
        
        # Handle residual length if any (shouldn't be for 1200/20)
        curr_len = tf.shape(mask)[1]
        pad_len = seq_len - curr_len
        
        mask = tf.cond(
            pad_len > 0,
            lambda: tf.pad(mask, [[0, 0], [0, pad_len], [0, 0]], constant_values=1.0),
            lambda: mask
        )
             
        # Apply mask
        masked_inputs = inputs * mask
        
        return masked_inputs, mask

    def train_step(self, data):
        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data
            
        # Add channel dim if missing
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, -1)

        with tf.GradientTape() as tape:
            # 1. Masking
            masked_inputs, mask = self._apply_mask(inputs)
            
            # 2. Encode
            latent = self.encoder(masked_inputs)
            
            # 3. Decode
            reconstruction = self.decoder(latent)
            
            # 4. Compute Loss
            # We compute loss on ALL pixels or just MASKED ones?
            # MAE usually computes loss only on masked patches for efficiency, 
            # but for CNN inpainting, global MSE is often used or MSE on masked regions.
            # Let's use MSE on masked regions specifically to force it to learn structure.
            
            diff = (inputs - reconstruction) ** 2
            
            # Loss on masked tokens: (1 - mask)
            # We want to minimize error where mask == 0
            mask_loss = tf.reduce_sum(diff * (1 - mask)) / (tf.reduce_sum(1 - mask) + 1e-6)
            
            loss = mask_loss

        # Gradients
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
    def call(self, inputs):
        # For inference/testing (no masking)
        latent = self.encoder(inputs)
        return self.decoder(latent)

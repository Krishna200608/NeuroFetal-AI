import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TimeGANModel:
    """
    TimeGAN (WGAN-GP) wrapper class for generating synthetic FHR and UC traces.
    Adapted for per-fold training to prevent data leakage.
    """
    def __init__(self, noise_dim=128, seq_len=1200, n_channels=2, batch_size=64, gp_weight=10.0):
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.gp_weight = gp_weight
        
        self.generator = self._build_generator()
        self.critic = self._build_critic()
        
        self.gen_optimizer = None
        self.critic_optimizer = None
        
        # We need these to denormalize the generated data
        self.data_min = None
        self.data_range = None
        
    def _build_generator(self):
        model = keras.Sequential(name="Generator")
        model.add(layers.Dense(75 * 256, input_dim=self.noise_dim))
        model.add(layers.Reshape((75, 256)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Block 1: 75 → 150
        model.add(layers.Conv1DTranspose(256, kernel_size=5, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Block 2: 150 → 300
        model.add(layers.Conv1DTranspose(128, kernel_size=5, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Block 3: 300 → 600
        model.add(layers.Conv1DTranspose(64, kernel_size=5, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Block 4: 600 → 1200
        model.add(layers.Conv1DTranspose(32, kernel_size=5, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Output: (1200, 2) — tanh for normalized signals
        model.add(layers.Conv1D(self.n_channels, kernel_size=7, padding='same', activation='tanh'))
        return model

    def _build_critic(self):
        model = keras.Sequential(name="Critic")

        # Block 1: 1200 → 600
        model.add(layers.Conv1D(32, kernel_size=5, strides=2, padding='same', input_shape=(self.seq_len, self.n_channels)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # Block 2: 600 → 300
        model.add(layers.Conv1D(64, kernel_size=5, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # Block 3: 300 → 150
        model.add(layers.Conv1D(128, kernel_size=5, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # Block 4: 150 → 75
        model.add(layers.Conv1D(256, kernel_size=5, strides=2, padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # Flatten → score
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    @tf.function
    def _gradient_penalty(self, real_samples, fake_samples):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = real_samples + alpha * (fake_samples - real_samples)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def _train_critic_step(self, real_batch):
        noise = tf.random.normal((tf.shape(real_batch)[0], self.noise_dim))
        with tf.GradientTape() as tape:
            fake_batch = self.generator(noise, training=True)
            real_score = self.critic(real_batch, training=True)
            fake_score = self.critic(fake_batch, training=True)

            w_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            gp = self._gradient_penalty(real_batch, fake_batch)
            critic_loss = w_loss + self.gp_weight * gp

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return critic_loss, w_loss

    @tf.function
    def _train_generator_step(self):
        noise = tf.random.normal((self.batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            fake_batch = self.generator(noise, training=True)
            fake_score = self.critic(fake_batch, training=True)
            gen_loss = -tf.reduce_mean(fake_score)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss

    def train(self, X_patho_stacked, epochs=1500, lr_g=1e-4, lr_d=1e-4, n_critic=5, verbose=1):
        """
        Trains the WGAN-GP on the provided stacked pathological samples.
        X_patho_stacked: (N, 1200, 2) array containing real pathological FHR and UC traces.
        """
        print(f"  [TimeGAN] Initializing training on {len(X_patho_stacked)} pathological samples...")
        
        # Normalization
        self.data_min = X_patho_stacked.min(axis=(0, 1), keepdims=True)
        data_max = X_patho_stacked.max(axis=(0, 1), keepdims=True)
        self.data_range = data_max - self.data_min
        self.data_range[self.data_range == 0] = 1.0
        
        X_normalized = 2.0 * (X_patho_stacked - self.data_min) / self.data_range - 1.0
        X_normalized = X_normalized.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices(X_normalized)
        dataset = dataset.shuffle(buffer_size=len(X_normalized)).batch(self.batch_size, drop_remainder=True)
        
        self.gen_optimizer = keras.optimizers.Adam(lr_g, beta_1=0.0, beta_2=0.9)
        self.critic_optimizer = keras.optimizers.Adam(lr_d, beta_1=0.0, beta_2=0.9)
        
        for epoch in range(1, epochs + 1):
            epoch_d_loss, epoch_g_loss, epoch_w_dist = [], [], []
            
            for step, real_batch in enumerate(dataset):
                d_loss, w_dist = self._train_critic_step(real_batch)
                epoch_d_loss.append(float(d_loss))
                epoch_w_dist.append(float(w_dist))
                
                if step % n_critic == 0:
                    g_loss = self._train_generator_step()
                    epoch_g_loss.append(float(g_loss))
                    
            if verbose and (epoch % 100 == 0 or epoch == 1):
                avg_d = np.mean(epoch_d_loss)
                avg_g = np.mean(epoch_g_loss) if epoch_g_loss else 0
                avg_w = np.mean(epoch_w_dist)
                print(f"  [TimeGAN] Epoch {epoch}/{epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | W Dist: {avg_w:.4f}")
                
        print(f"  [TimeGAN] Training complete for this fold.")

    def generate(self, n_samples):
        """
        Generates n_samples synthetic pathological traces.
        Returns: (n_samples, 1200, 2) un-normalized numpy array.
        """
        if self.data_min is None or self.data_range is None:
            raise ValueError("Model must be trained before generating data to ensure correct denormalization.")
            
        synthetic_batches = []
        gen_batch_size = 128
        
        for i in range(0, n_samples, gen_batch_size):
            batch_n = min(gen_batch_size, n_samples - i)
            noise = tf.random.normal((batch_n, self.noise_dim))
            synthetic_batch = self.generator(noise, training=False).numpy()
            synthetic_batches.append(synthetic_batch)
            
        X_synthetic_normalized = np.concatenate(synthetic_batches, axis=0)
        
        # Denormalize
        X_synthetic = (X_synthetic_normalized + 1.0) / 2.0 * self.data_range + self.data_min
        return X_synthetic

def apply_per_fold_timegan_augmentation(X_fhr_train, X_uc_train, X_tab_train, y_train, epochs=1500, batch_size=64):
    """
    1. Isolates pathological samples from the current fold's training data.
    2. Trains a new TimeGAN model on these samples only.
    3. Generates enough synthetic samples to balance the classes.
    4. Concatenates synthetic data with existing real data and shuffles.
    """
    patho_idx = np.where(y_train == 1)[0]
    X_fhr_patho = X_fhr_train[patho_idx]
    X_uc_patho = X_uc_train[patho_idx]
    
    # Expand dims if needed
    if X_fhr_patho.ndim == 2: X_fhr_patho = np.expand_dims(X_fhr_patho, -1)
    if X_uc_patho.ndim == 2: X_uc_patho = np.expand_dims(X_uc_patho, -1)
        
    X_patho_stacked = np.concatenate([X_fhr_patho, X_uc_patho], axis=-1)
    
    # Train TimeGAN
    gan = TimeGANModel(batch_size=batch_size)
    gan.train(X_patho_stacked, epochs=epochs, verbose=1)
    
    # Calculate how many to generate
    n_positive = len(patho_idx)
    n_negative = len(y_train) - n_positive
    n_needed = max(0, n_negative - n_positive)
    
    if n_needed == 0:
        return X_fhr_train, X_uc_train, X_tab_train, y_train
        
    print(f"  [TimeGAN] Generating {n_needed} synthetic samples to balance classes...")
    X_syn_stacked = gan.generate(n_needed)
    
    X_fhr_syn = X_syn_stacked[:, :, 0]
    X_uc_syn = X_syn_stacked[:, :, 1]
    
    if X_fhr_train.ndim == 2:
        X_fhr_syn = X_fhr_syn.reshape(n_needed, -1)
        X_uc_syn = X_uc_syn.reshape(n_needed, -1)
        
    # Generate tabular features by resampling from real pathological
    syn_tab_indices = np.random.choice(patho_idx, size=n_needed, replace=True)
    X_tab_syn = X_tab_train[syn_tab_indices]
    
    # Concat
    X_fhr_aug = np.concatenate([X_fhr_train, X_fhr_syn], axis=0)
    X_uc_aug = np.concatenate([X_uc_train, X_uc_syn], axis=0)
    X_tab_aug = np.concatenate([X_tab_train, X_tab_syn], axis=0)
    y_aug = np.concatenate([y_train, np.ones(n_needed)], axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(y_aug))
    return X_fhr_aug[perm], X_uc_aug[perm], X_tab_aug[perm], y_aug[perm]

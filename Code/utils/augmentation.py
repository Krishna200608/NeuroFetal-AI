"""
Time-Series Data Augmentation for CTG Signals
==============================================
Novel augmentation techniques for FHR/UC signal classification.

Implements:
1. Time Warping - Simulates varying recording speeds
2. Gaussian Jittering - Simulates sensor noise
3. Magnitude Scaling - Handles gain variations
4. Window Slicing - Creates temporal variations
5. Mixup - Regularization through interpolation

Reference:
- Um et al., 2017: Data augmentation for time series
- Zhang et al., 2018: Mixup regularization

Author: NeuroFetal AI Team
"""

import numpy as np
from scipy.interpolate import CubicSpline


class TimeSeriesAugmentor:
    """
    Augmentation pipeline for 1D physiological signals.
    
    Usage:
        augmentor = TimeSeriesAugmentor(p=0.5)
        X_aug, y_aug = augmentor.augment_batch(X, y)
    """
    
    def __init__(
        self,
        p=0.5,  # Probability of applying each augmentation
        time_warp_sigma=0.2,
        jitter_sigma=0.03,
        scale_sigma=0.1,
        mixup_alpha=0.2,
        use_time_warp=True,
        use_jitter=True,
        use_scaling=True,
        use_mixup=True,
        seed=None
    ):
        """
        Args:
            p: Probability of applying each augmentation
            time_warp_sigma: Controls severity of time warping
            jitter_sigma: Std dev of Gaussian noise (fraction of signal range)
            scale_sigma: Std dev of magnitude scaling factor
            mixup_alpha: Beta distribution parameter for mixup
            use_*: Enable/disable specific augmentations
            seed: Random seed for reproducibility
        """
        self.p = p
        self.time_warp_sigma = time_warp_sigma
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.mixup_alpha = mixup_alpha
        
        self.use_time_warp = use_time_warp
        self.use_jitter = use_jitter
        self.use_scaling = use_scaling
        self.use_mixup = use_mixup
        
        if seed is not None:
            np.random.seed(seed)
    
    def time_warp(self, x, sigma=None):
        """
        Time warping via smooth random displacement.
        
        Simulates variations in CTG recording speed without
        changing the overall signal duration.
        """
        sigma = sigma or self.time_warp_sigma
        orig_length = x.shape[0]
        
        # Generate random warping curve
        n_knots = 4
        knot_positions = np.linspace(0, orig_length - 1, n_knots)
        knot_values = knot_positions + np.random.normal(0, sigma * orig_length / n_knots, n_knots)
        knot_values = np.clip(knot_values, 0, orig_length - 1)
        
        # Create smooth warping function
        cs = CubicSpline(knot_positions, knot_values)
        warped_indices = cs(np.arange(orig_length))
        warped_indices = np.clip(warped_indices, 0, orig_length - 1).astype(int)
        
        # Apply warping
        if x.ndim == 1:
            return x[warped_indices]
        else:
            return x[warped_indices, :]
    
    def jitter(self, x, sigma=None):
        """
        Add Gaussian noise to simulate sensor noise.
        
        Noise is proportional to signal range to maintain
        physiological plausibility.
        """
        sigma = sigma or self.jitter_sigma
        signal_range = np.ptp(x) + 1e-8  # Prevent division by zero
        noise = np.random.normal(0, sigma * signal_range, x.shape)
        return x + noise
    
    def scaling(self, x, sigma=None):
        """
        Random magnitude scaling.
        
        Simulates CTG baseline variations and gain differences
        between recording devices.
        """
        sigma = sigma or self.scale_sigma
        scale_factor = np.random.normal(1.0, sigma)
        scale_factor = np.clip(scale_factor, 0.8, 1.2)  # Keep reasonable bounds
        return x * scale_factor
    
    def window_slice(self, x, reduce_ratio=0.9):
        """
        Random window slicing with resize.
        
        Extracts a random subset of the signal and resizes
        to original length, creating temporal variations.
        """
        orig_length = x.shape[0]
        target_length = int(orig_length * reduce_ratio)
        
        if target_length >= orig_length:
            return x
        
        start = np.random.randint(0, orig_length - target_length)
        sliced = x[start:start + target_length]
        
        # Resize back to original length
        indices = np.linspace(0, target_length - 1, orig_length).astype(int)
        if x.ndim == 1:
            return sliced[indices]
        else:
            return sliced[indices, :]
    
    def augment_single(self, x):
        """Apply random augmentations to a single sample."""
        x_aug = x.copy()
        
        if self.use_time_warp and np.random.random() < self.p:
            x_aug = self.time_warp(x_aug)
        
        if self.use_jitter and np.random.random() < self.p:
            x_aug = self.jitter(x_aug)
        
        if self.use_scaling and np.random.random() < self.p:
            x_aug = self.scaling(x_aug)
        
        return x_aug
    
    def mixup(self, X, y, alpha=None):
        """
        Mixup augmentation - interpolate between samples.
        
        Creates virtual training examples by linear interpolation,
        which acts as strong regularization.
        """
        alpha = alpha or self.mixup_alpha
        batch_size = X.shape[0]
        
        # Sample mixup ratio from Beta distribution
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)  # Ensure lam >= 0.5
        
        # Random shuffle for mixing partners
        indices = np.random.permutation(batch_size)
        
        # Mixup
        lam_x = lam.reshape(-1, 1, 1) if X.ndim == 3 else lam.reshape(-1, 1)
        X_mixed = lam_x * X + (1 - lam_x) * X[indices]
        
        lam_y = lam.reshape(-1, 1) if y.ndim == 2 else lam
        y_mixed = lam_y * y + (1 - lam_y) * y[indices]
        
        return X_mixed, y_mixed
    
    def augment_batch(self, X, y=None, expand_factor=2):
        """
        Augment a batch of samples.
        
        Args:
            X: Input samples (batch, time, channels) or (batch, time)
            y: Labels (optional)
            expand_factor: How many augmented copies per original
        
        Returns:
            X_aug: Augmented samples (includes originals)
            y_aug: Corresponding labels
        """
        batch_size = X.shape[0]
        
        # Start with original samples
        X_list = [X]
        y_list = [y] if y is not None else []
        
        # Generate augmented copies
        for _ in range(expand_factor - 1):
            X_aug = np.array([self.augment_single(x) for x in X])
            X_list.append(X_aug)
            if y is not None:
                y_list.append(y.copy())
        
        X_combined = np.concatenate(X_list, axis=0)
        
        if y is not None:
            y_combined = np.concatenate(y_list, axis=0)
            
            # Apply mixup to combined batch
            if self.use_mixup and np.random.random() < self.p:
                X_combined, y_combined = self.mixup(X_combined, y_combined)
            
            return X_combined, y_combined
        
        return X_combined


def apply_label_smoothing(y, smoothing=0.1):
    """
    Apply label smoothing for regularization.
    
    Converts hard labels [0, 1] to soft labels [smoothing, 1-smoothing].
    Helps prevent overconfidence and improves generalization.
    
    Args:
        y: Binary labels (0 or 1)
        smoothing: Smoothing factor (0 = no smoothing)
    
    Returns:
        Smoothed labels
    """
    y_smooth = y.copy().astype(np.float32)
    y_smooth = y_smooth * (1 - smoothing) + smoothing / 2
    return y_smooth


# Quick test
if __name__ == "__main__":
    # Test with dummy FHR signal
    X_dummy = np.random.randn(10, 1200, 1)  # 10 samples, 1200 timesteps, 1 channel
    y_dummy = np.random.randint(0, 2, 10).astype(np.float32)
    
    aug = TimeSeriesAugmentor(p=0.5)
    X_aug, y_aug = aug.augment_batch(X_dummy, y_dummy, expand_factor=2)
    
    print(f"Original: {X_dummy.shape}, {y_dummy.shape}")
    print(f"Augmented: {X_aug.shape}, {y_aug.shape}")
    
    # Test label smoothing
    y_smooth = apply_label_smoothing(y_dummy, smoothing=0.1)
    print(f"Original labels: {y_dummy[:5]}")
    print(f"Smoothed labels: {y_smooth[:5]}")

"""
Time-Series Data Augmentation for CTG Signals (v2 — SOTA)
=========================================================
Novel augmentation techniques for FHR/UC signal classification.

Implements:
1. Time Warping - Simulates varying recording speeds
2. Gaussian Jittering - Simulates sensor noise
3. Magnitude Scaling - Handles gain variations
4. Window Slicing - Creates temporal variations
5. Mixup - Regularization through interpolation
6. [NEW] SpecAugment - Time masking for signal robustness
7. [NEW] CutMix - Segment swapping between samples

v2 Changes (SOTA Strategy):
- Added SpecAugment (time masking)
- Added CutMix (segment swapping between signals)
- Both proven on small medical datasets to improve generalization by 0.02-0.05 AUC

Reference:
- Um et al., 2017: Data augmentation for time series
- Zhang et al., 2018: Mixup regularization
- Park et al., 2019: SpecAugment
- Yun et al., 2019: CutMix

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
        use_spec_augment=True,    # NEW: SpecAugment
        use_cutmix=True,          # NEW: CutMix
        spec_augment_max_mask=0.15,  # Max fraction of signal to mask
        spec_augment_n_masks=2,      # Number of masks to apply
        cutmix_alpha=1.0,            # Beta distribution param for CutMix
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
            spec_augment_max_mask: Max fraction of signal to zero-mask
            spec_augment_n_masks: Number of time masks per sample
            cutmix_alpha: Beta distribution parameter for CutMix
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
        self.use_spec_augment = use_spec_augment
        self.use_cutmix = use_cutmix
        
        self.spec_augment_max_mask = spec_augment_max_mask
        self.spec_augment_n_masks = spec_augment_n_masks
        self.cutmix_alpha = cutmix_alpha
        
        if seed is not None:
            np.random.seed(seed)
    
    def time_warp(self, x, sigma=None):
        """
        Time warping via smooth random displacement.
        Simulates variations in CTG recording speed.
        """
        sigma = sigma or self.time_warp_sigma
        orig_length = x.shape[0]
        
        n_knots = 4
        knot_positions = np.linspace(0, orig_length - 1, n_knots)
        knot_values = knot_positions + np.random.normal(0, sigma * orig_length / n_knots, n_knots)
        knot_values = np.clip(knot_values, 0, orig_length - 1)
        
        cs = CubicSpline(knot_positions, knot_values)
        warped_indices = cs(np.arange(orig_length))
        warped_indices = np.clip(warped_indices, 0, orig_length - 1).astype(int)
        
        if x.ndim == 1:
            return x[warped_indices]
        else:
            return x[warped_indices, :]
    
    def jitter(self, x, sigma=None):
        """Add Gaussian noise to simulate sensor noise."""
        sigma = sigma or self.jitter_sigma
        signal_range = np.ptp(x) + 1e-8
        noise = np.random.normal(0, sigma * signal_range, x.shape)
        return x + noise
    
    def scaling(self, x, sigma=None):
        """Random magnitude scaling."""
        sigma = sigma or self.scale_sigma
        scale_factor = np.random.normal(1.0, sigma)
        scale_factor = np.clip(scale_factor, 0.8, 1.2)
        return x * scale_factor
    
    def window_slice(self, x, reduce_ratio=0.9):
        """Random window slicing with resize."""
        orig_length = x.shape[0]
        target_length = int(orig_length * reduce_ratio)
        
        if target_length >= orig_length:
            return x
        
        start = np.random.randint(0, orig_length - target_length)
        sliced = x[start:start + target_length]
        
        indices = np.linspace(0, target_length - 1, orig_length).astype(int)
        if x.ndim == 1:
            return sliced[indices]
        else:
            return sliced[indices, :]
    
    def spec_augment(self, x):
        """
        SpecAugment for 1D signals: zero-mask random contiguous time segments.
        
        Forces the model to use all parts of the signal, preventing over-reliance
        on a few salient patterns. Proven effective on small medical datasets.
        
        Does NOT modify y — this is purely input-level masking.
        
        Args:
            x: Single sample (time,) or (time, channels)
            
        Returns:
            Augmented sample with masked segments
        """
        x_aug = x.copy()
        length = x_aug.shape[0]
        max_mask_len = int(length * self.spec_augment_max_mask)
        
        for _ in range(self.spec_augment_n_masks):
            mask_len = np.random.randint(1, max(2, max_mask_len))
            start = np.random.randint(0, length - mask_len)
            if x_aug.ndim == 1:
                x_aug[start:start + mask_len] = 0.0
            else:
                x_aug[start:start + mask_len, :] = 0.0
        
        return x_aug
    
    def cutmix_batch(self, X, y, alpha=None):
        """
        CutMix for time series: swap random segments between sample pairs.
        
        For each sample, randomly select a partner sample and swap a contiguous
        segment. Labels are mixed proportionally to the swapped length.
        
        This is much stronger than Mixup for time series because it preserves
        local signal structure while creating novel global combinations.
        
        Args:
            X: Batch (batch, time, channels) or (batch, time)
            y: Labels (batch,) or (batch, 1)
            alpha: Beta distribution parameter
            
        Returns:
            X_mixed, y_mixed
        """
        alpha = alpha or self.cutmix_alpha
        batch_size = X.shape[0]
        length = X.shape[1]
        
        # Sample mixing ratio
        lam = np.random.beta(alpha, alpha, batch_size)
        
        # Random partner indices
        indices = np.random.permutation(batch_size)
        
        X_mixed = X.copy()
        y_flat = y.flatten() if y.ndim > 1 else y.copy()
        y_mixed = y_flat.copy().astype(np.float32)
        
        for i in range(batch_size):
            # Segment length from lam
            cut_len = int(length * (1 - lam[i]))
            if cut_len < 1:
                continue
            start = np.random.randint(0, length - cut_len)
            
            # Swap segment
            if X.ndim == 3:
                X_mixed[i, start:start + cut_len, :] = X[indices[i], start:start + cut_len, :]
            else:
                X_mixed[i, start:start + cut_len] = X[indices[i], start:start + cut_len]
            
            # Mix labels proportionally
            actual_lam = 1 - cut_len / length
            y_mixed[i] = actual_lam * y_flat[i] + (1 - actual_lam) * y_flat[indices[i]]
        
        if y.ndim > 1:
            y_mixed = y_mixed.reshape(y.shape)
        
        return X_mixed, y_mixed
    
    def augment_single(self, x):
        """Apply random augmentations to a single sample."""
        x_aug = x.copy()
        
        if self.use_time_warp and np.random.random() < self.p:
            x_aug = self.time_warp(x_aug)
        
        if self.use_jitter and np.random.random() < self.p:
            x_aug = self.jitter(x_aug)
        
        if self.use_scaling and np.random.random() < self.p:
            x_aug = self.scaling(x_aug)
        
        # NEW: SpecAugment (applied per-sample)
        if self.use_spec_augment and np.random.random() < self.p:
            x_aug = self.spec_augment(x_aug)
        
        return x_aug
    
    def mixup(self, X, y, alpha=None):
        """
        Mixup augmentation - interpolate between samples.
        """
        alpha = alpha or self.mixup_alpha
        batch_size = X.shape[0]
        
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = np.maximum(lam, 1 - lam)
        
        indices = np.random.permutation(batch_size)
        
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
        
        # Generate augmented copies (per-sample augmentations)
        for _ in range(expand_factor - 1):
            X_aug = np.array([self.augment_single(x) for x in X])
            X_list.append(X_aug)
            if y is not None:
                y_list.append(y.copy())
        
        X_combined = np.concatenate(X_list, axis=0)
        
        if y is not None:
            y_combined = np.concatenate(y_list, axis=0)
            
            # Apply CutMix to combined batch (stronger than Mixup for time series)
            if self.use_cutmix and np.random.random() < self.p:
                X_combined, y_combined = self.cutmix_batch(X_combined, y_combined)
            # Fallback to Mixup if CutMix not enabled
            elif self.use_mixup and np.random.random() < self.p:
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
    
    aug = TimeSeriesAugmentor(
        p=0.5,
        use_spec_augment=True,
        use_cutmix=True
    )
    X_aug, y_aug = aug.augment_batch(X_dummy, y_dummy, expand_factor=4)
    
    print(f"Original: {X_dummy.shape}, {y_dummy.shape}")
    print(f"Augmented: {X_aug.shape}, {y_aug.shape}")
    
    # Test SpecAugment alone
    x_single = X_dummy[0].copy()
    x_masked = aug.spec_augment(x_single)
    masked_count = np.sum(x_masked == 0) - np.sum(x_single == 0)
    print(f"SpecAugment masked {masked_count} values")
    
    # Test CutMix
    X_cut, y_cut = aug.cutmix_batch(X_dummy, y_dummy)
    print(f"CutMix output: {X_cut.shape}, labels changed: {np.sum(y_cut != y_dummy)}")
    
    # Test label smoothing
    y_smooth = apply_label_smoothing(y_dummy, smoothing=0.1)
    print(f"Original labels: {y_dummy[:5]}")
    print(f"Smoothed labels: {y_smooth[:5]}")

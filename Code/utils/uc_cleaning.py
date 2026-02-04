"""
UC Artifact Cleaning Module
============================
Implements advanced uterine contraction (UC) signal cleaning following Paper 2 (Fridman et al., 2026).

Features:
- Sensor loss detection via rolling standard deviation
- Linear interpolation for gaps < 15 seconds
- Zero-padding for longer gaps
- Artifact removal (spikes, flatlines)

Reference: Paper 2 - "A Foundation Model Approach for Fetal Stress Prediction During Labor"
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

class UCCleaner:
    """
    Cleans Uterine Contraction signals to remove artifacts and sensor noise.
    
    Parameters:
    - fs: Sampling frequency (default: 4Hz)
    - gap_threshold_sec: Threshold for gap size (default: 15 seconds)
    - sensor_loss_threshold: Std dev threshold for sensor loss (default: 1e-5)
    - smoothing_window: Median filter window size (default: 5)
    """
    
    def __init__(self, fs=4, gap_threshold_sec=15, sensor_loss_threshold=1e-5, smoothing_window=5):
        self.fs = fs
        self.gap_threshold_sec = gap_threshold_sec
        self.gap_threshold_samples = int(gap_threshold_sec * fs)
        self.sensor_loss_threshold = sensor_loss_threshold
        self.smoothing_window = smoothing_window
    
    def detect_sensor_loss(self, signal, window_sec=5):
        """
        Detect periods of sensor loss using rolling standard deviation.
        Low std-dev indicates flatline (sensor not working).
        
        Args:
            signal: UC signal (1D array)
            window_sec: Window size for rolling std calculation
            
        Returns:
            is_sensor_loss: Boolean array indicating sensor loss periods
        """
        window_samples = int(window_sec * self.fs)
        if window_samples < 1:
            window_samples = 1
            
        is_loss = np.zeros(len(signal), dtype=bool)
        
        for i in range(len(signal) - window_samples):
            window = signal[i:i+window_samples]
            if np.std(window) < self.sensor_loss_threshold:
                is_loss[i:i+window_samples] = True
        
        return is_loss
    
    def remove_spikes(self, signal, threshold_percentile=99):
        """
        Remove outlier spikes using percentile-based thresholding.
        
        Args:
            signal: UC signal
            threshold_percentile: Percentile for outlier detection
            
        Returns:
            cleaned_signal: Signal with spikes removed (replaced by median)
        """
        percentile_val = np.percentile(signal, threshold_percentile)
        cleaned = signal.copy()
        spike_mask = cleaned > percentile_val
        
        if np.any(spike_mask):
            # Replace spikes with median of neighborhood
            for idx in np.where(spike_mask)[0]:
                start = max(0, idx - 2)
                end = min(len(signal), idx + 3)
                cleaned[idx] = np.median(signal[start:end])
        
        return cleaned
    
    def interpolate_gaps(self, signal):
        """
        Interpolate gaps smaller than threshold, zero-pad larger gaps.
        
        Args:
            signal: UC signal
            
        Returns:
            interpolated_signal: Signal with gaps handled
        """
        # Identify zero values (missing data)
        is_gap = (signal == 0)
        
        if not np.any(is_gap):
            return signal  # No gaps
        
        # Find continuous gap regions
        diff = np.diff(np.concatenate(([0], is_gap.astype(int), [0])))
        gap_starts = np.where(diff == 1)[0]
        gap_ends = np.where(diff == -1)[0]
        
        interpolated = signal.copy()
        
        for start, end in zip(gap_starts, gap_ends):
            gap_size = end - start
            
            if gap_size <= self.gap_threshold_samples:
                # Interpolate short gaps
                if start > 0 and end < len(signal):
                    left_val = signal[start - 1]
                    right_val = signal[end]
                    interpolated[start:end] = np.linspace(left_val, right_val, gap_size)
            # Else: keep as zero (long gap = real sensor loss)
        
        return interpolated
    
    def smooth(self, signal):
        """
        Apply median filtering for smoothing.
        
        Args:
            signal: UC signal
            
        Returns:
            smoothed_signal: Median-filtered signal
        """
        if self.smoothing_window < 2:
            return signal
        return medfilt(signal, kernel_size=self.smoothing_window if self.smoothing_window % 2 == 1 else self.smoothing_window + 1)
    
    def clean(self, signal):
        """
        Full cleaning pipeline: sensor loss → spikes → gaps → smooth.
        
        Args:
            signal: Raw UC signal (4Hz)
            
        Returns:
            cleaned_signal: Cleaned UC signal ready for feature extraction
        """
        # Step 1: Detect and zero-out sensor loss regions
        sensor_loss_mask = self.detect_sensor_loss(signal)
        signal_step1 = signal.copy()
        signal_step1[sensor_loss_mask] = 0
        
        # Step 2: Remove spikes
        signal_step2 = self.remove_spikes(signal_step1)
        
        # Step 3: Interpolate gaps
        signal_step3 = self.interpolate_gaps(signal_step2)
        
        # Step 4: Smooth
        signal_step4 = self.smooth(signal_step3)
        
        # Step 5: Normalize to [0, 1]
        _min, _max = np.min(signal_step4), np.max(signal_step4)
        if _max > _min:
            signal_step4 = (signal_step4 - _min) / (_max - _min)
        else:
            signal_step4 = np.zeros_like(signal_step4)
        
        return signal_step4
    
    def get_quality_score(self, signal):
        """
        Return quality score (0-1) indicating how much of signal is valid.
        High score = clean signal. Low score = lots of artifacts.
        
        Args:
            signal: Raw UC signal
            
        Returns:
            quality_score: Float [0, 1]
        """
        sensor_loss_mask = self.detect_sensor_loss(signal)
        zero_mask = (signal == 0)
        
        invalid_samples = np.sum(sensor_loss_mask) + np.sum(zero_mask)
        quality = 1.0 - (invalid_samples / len(signal))
        
        return max(0, min(1, quality))


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate synthetic noisy UC signal
    t = np.linspace(0, 60, 240)  # 60 seconds @ 4Hz
    signal = 50 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 5, len(t))
    signal = np.clip(signal, 0, 100)
    
    # Add sensor loss
    signal[50:60] = 0
    signal[150:170] = 0.001
    
    # Add spikes
    signal[100] = 150
    signal[120] = 160
    
    # Clean
    cleaner = UCCleaner()
    cleaned = cleaner.clean(signal)
    quality = cleaner.get_quality_score(signal)
    
    print(f"Signal Quality Score: {quality:.2%}")
    print(f"Cleaned signal shape: {cleaned.shape}")
    print(f"Cleaned signal range: [{cleaned.min():.3f}, {cleaned.max():.3f}]")

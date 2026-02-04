"""
CSP (Common Spatial Pattern) Feature Extraction
================================================
Extracts discriminative features from FHR-UC correlation using CSP.
Adapted from Paper 6 (Alqahtani et al., 2025) for multimodal signals.

Features extracted:
- CSP spatial filters (maximize variance between Normal vs Pathological)
- Log-variance features per filter
- Cross-correlation metrics between FHR and UC
- MAD (Median Absolute Deviation) - From Paper 4 (DeepCTG 1.0)
- Beta_0 (Baseline intercept) - From Paper 7 (Fusion ResNet)
- Labor Stage flag - From Paper 5
- Signal Quality Index (SQI)

Reference: Paper 6 - "Fetal Hypoxia Classification from Cardiotocography Signals 
Using Instantaneous Frequency and Common Spatial Pattern"

Enhancements:
- Paper 4: MAD and Beta_0 for baseline variability
- Paper 5: Labor stage flag for temporal context
- Paper 7: Signal quality for robustness
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import correlate
from scipy.stats import skew, kurtosis


class CSPFeatureExtractor:
    """
    Common Spatial Pattern (CSP) feature extractor for multimodal signals.
    
    Parameters:
    - n_components: Number of CSP filters (default: 4)
    - fs: Sampling frequency (default: 1Hz for preprocessed data)
    """
    
    def __init__(self, n_components=4, fs=1):
        self.n_components = n_components
        self.fs = fs
        self.csp_filters = None
        self.is_fitted = False
    
    def _compute_covariance(self, signal):
        """
        Compute covariance matrix for multichannel signal.
        
        Args:
            signal: Shape (n_samples, n_channels) [e.g., (1200, 2) for FHR + UC]
            
        Returns:
            cov: Covariance matrix (n_channels, n_channels)
        """
        signal_centered = signal - np.mean(signal, axis=0)
        cov = np.dot(signal_centered.T, signal_centered) / signal.shape[0]
        return cov
    
    def fit(self, X_normal, X_pathological):
        """
        Fit CSP filters using labeled training data.
        
        Args:
            X_normal: Array of shape (n_normal_samples, signal_length, 2) [Normal cases]
            X_pathological: Array of shape (n_pathological, signal_length, 2) [Pathological cases]
        """
        n_channels = X_normal.shape[2]
        
        # Compute average covariance for each class
        cov_normal = np.zeros((n_channels, n_channels))
        cov_pathological = np.zeros((n_channels, n_channels))
        
        for X in X_normal:
            cov_normal += self._compute_covariance(X)
        cov_normal /= len(X_normal)
        
        for X in X_pathological:
            cov_pathological += self._compute_covariance(X)
        cov_pathological /= len(X_pathological)
        
        # Solve generalized eigenvalue problem
        # Maximize: trace(w^T * Cov_pathological * w)
        # Subject to: w^T * Cov_normal * w = 1
        eigenvalues, eigenvectors = eigh(cov_pathological, cov_normal)
        
        # Select top and bottom components (most discriminative)
        idx = np.argsort(eigenvalues)
        idx_selected = np.concatenate([idx[:self.n_components//2], idx[-self.n_components//2:]])
        
        self.csp_filters = eigenvectors[:, idx_selected].T  # Shape: (n_components, n_channels)
        self.is_fitted = True
    
    def transform(self, X):
        """
        Extract CSP features from signal(s).
        
        Args:
            X: Signal array, shape (signal_length, 2) or (n_samples, signal_length, 2)
            
        Returns:
            features: CSP log-variance features, shape (n_components,) or (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("CSP must be fitted before transform. Call fit() first.")
        
        if X.ndim == 2:
            # Single sample: (signal_length, 2)
            X = np.expand_dims(X, axis=0)
            return self._transform_batch(X)[0]
        else:
            # Batch: (n_samples, signal_length, 2)
            return self._transform_batch(X)
    
    def _transform_batch(self, X):
        """
        Extract features from batch of samples.
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_components))
        
        for i in range(n_samples):
            signal = X[i]  # (signal_length, 2)
            # Apply CSP filters
            csp_signal = np.dot(self.csp_filters, signal.T)  # (n_components, signal_length)
            # Log-variance feature
            variances = np.var(csp_signal, axis=1)
            features[i] = np.log(variances + 1e-6)  # Add epsilon for numerical stability
        
        return features
    
    def fit_transform(self, X_normal, X_pathological):
        """
        Fit and transform in one call.
        """
        self.fit(X_normal, X_pathological)
        # Combine for transform
        X_all = np.concatenate([X_normal, X_pathological], axis=0)
        return self.transform(X_all)


# ============================================================================
# Enhanced Statistical Features (Paper 4, 5, 7)
# ============================================================================

def calculate_mad(signal):
    """
    Median Absolute Deviation - robust variability measure.
    From Paper 4 (DeepCTG 1.0) and clinical practice.
    
    MAD is more robust to outliers than standard deviation,
    making it suitable for FHR signals with artifacts.
    """
    median = np.median(signal)
    return np.median(np.abs(signal - median))


def calculate_beta0(signal):
    """
    Baseline intercept from linear regression.
    From Paper 7 (Fusion ResNet) - represents the signal baseline.
    
    Beta_0 captures the overall level of the signal,
    useful for detecting bradycardia or tachycardia trends.
    """
    x = np.arange(len(signal))
    try:
        coeffs = np.polyfit(x, signal, 1)
        return coeffs[1]  # Intercept
    except:
        return np.mean(signal)  # Fallback


def calculate_signal_quality(signal, min_valid=50, max_valid=200):
    """
    Signal Quality Index (SQI) - proportion of valid samples.
    
    Returns the fraction of samples within physiologically valid range.
    Low SQI indicates poor signal quality requiring UC cleaning.
    """
    valid_mask = (signal > min_valid) & (signal < max_valid)
    return np.mean(valid_mask)


def calculate_labor_stage_flag(window_start_sec, total_duration_sec):
    """
    Labor Stage proxy flag from Paper 5.
    
    Returns 1 if window is in the last 30 minutes before delivery
    (proxy for Stage 2 labor), otherwise 0.
    
    This feature is important because fetal compromise risk
    increases significantly during Stage 2 labor.
    """
    time_to_delivery = total_duration_sec - window_start_sec
    return 1.0 if time_to_delivery <= 30 * 60 else 0.0


class MultimodalFeatureExtractor:
    """
    Complete multimodal feature extraction combining:
    - CSP spatial filters
    - Statistical features (mean, std, min, max)
    - Advanced features (MAD, Beta_0, SQI)
    - Temporal context (labor stage)
    - Cross-correlation metrics
    
    Total features: 17 (vs original 13)
    """
    
    def __init__(self, n_csp_components=4, total_duration_sec=3600):
        self.csp = CSPFeatureExtractor(n_components=n_csp_components)
        self.n_csp_components = n_csp_components
        self.total_duration_sec = total_duration_sec  # Default: 60 min recording
        self.is_fitted = False
    
    @staticmethod
    def get_feature_names(n_csp=4):
        """
        Returns list of feature names for reference.
        Useful for SHAP explainability.
        """
        names = [
            # Basic FHR stats
            'fhr_mean', 'fhr_std', 'fhr_min', 'fhr_max',
            # Advanced FHR stats (Paper 4 & 7)
            'fhr_mad', 'fhr_beta0', 'fhr_skewness', 'fhr_kurtosis', 'fhr_sqi',
            # UC stats
            'uc_mean', 'uc_std', 'uc_count',
            # Cross-correlation
            'cross_corr_max', 'cross_corr_mean',
            # Temporal context (Paper 5)
            'labor_stage_flag',
        ]
        # CSP features
        names.extend([f'csp_{i}' for i in range(n_csp)])
        return names
    
    def extract_statistical_features(self, fhr, uc, window_start_sec=0):
        """
        Extract classical + advanced statistical features from FHR and UC.
        
        Args:
            fhr: FHR signal (1D array)
            uc: UC signal (1D array)
            window_start_sec: Start time of this window in seconds (for labor stage)
        
        Returns:
            features: Dict of extracted features
        """
        features = {}
        
        # ====== FHR Basic Features ======
        features['fhr_mean'] = np.mean(fhr)
        features['fhr_std'] = np.std(fhr)
        features['fhr_min'] = np.min(fhr)
        features['fhr_max'] = np.max(fhr)
        
        # ====== FHR Advanced Features (Paper 4 & 7) ======
        features['fhr_mad'] = calculate_mad(fhr)
        features['fhr_beta0'] = calculate_beta0(fhr)
        features['fhr_skewness'] = skew(fhr) if len(fhr) > 2 else 0.0
        features['fhr_kurtosis'] = kurtosis(fhr) if len(fhr) > 2 else 0.0
        features['fhr_sqi'] = calculate_signal_quality(fhr)
        
        # ====== UC Features ======
        features['uc_mean'] = np.mean(uc)
        features['uc_std'] = np.std(uc)
        features['uc_count'] = np.sum(uc > 0.5)  # Number of contractions
        
        # ====== Cross-correlation (FHR response to UC) ======
        cross_corr = correlate(fhr, uc, mode='same')
        features['cross_corr_max'] = np.max(np.abs(cross_corr))
        features['cross_corr_mean'] = np.mean(np.abs(cross_corr))
        
        # ====== Labor Stage Flag (Paper 5) ======
        features['labor_stage_flag'] = calculate_labor_stage_flag(
            window_start_sec, self.total_duration_sec
        )
        
        return features
    
    def fit(self, X_fhr_normal, X_uc_normal, X_fhr_path, X_uc_path):
        """
        Fit CSP on normal vs pathological samples.
        
        Args:
            X_fhr_normal: FHR signals for normal cases (n_normal, signal_len)
            X_uc_normal: UC signals for normal cases (n_normal, signal_len)
            X_fhr_path: FHR signals for pathological cases
            X_uc_path: UC signals for pathological cases
        """
        # Stack into (n_samples, signal_length, 2)
        X_normal = np.stack([X_fhr_normal, X_uc_normal], axis=2)
        X_path = np.stack([X_fhr_path, X_uc_path], axis=2)
        
        self.csp.fit(X_normal, X_path)
        self.is_fitted = True
    
    def extract(self, fhr, uc):
        """
        Extract all features from a single FHR-UC pair.
        
        Args:
            fhr: FHR signal (1D array)
            uc: UC signal (1D array)
            
        Returns:
            features_dict: Dictionary of extracted features
        """
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before extraction.")
        
        features = {}
        
        # Statistical features
        stat_feats = self.extract_statistical_features(fhr, uc)
        features.update(stat_feats)
        
        # CSP features
        multimodal_signal = np.stack([fhr, uc], axis=1)  # (signal_length, 2)
        csp_features = self.csp.transform(multimodal_signal)
        for i, csp_val in enumerate(csp_features):
            features[f'csp_{i}'] = csp_val
        
        return features
    
    def extract_batch(self, X_fhr, X_uc):
        """
        Extract features from multiple samples.
        
        Args:
            X_fhr: Shape (n_samples, signal_length)
            X_uc: Shape (n_samples, signal_length)
            
        Returns:
            features: Shape (n_samples, n_features)
        """
        n_samples = X_fhr.shape[0]
        
        # Statistical features
        stat_features_list = []
        for i in range(n_samples):
            stat_feats = self.extract_statistical_features(X_fhr[i], X_uc[i])
            stat_features_list.append(list(stat_feats.values()))
        
        stat_array = np.array(stat_features_list)  # (n_samples, n_stat_features)
        
        # CSP features
        multimodal_signals = np.stack([X_fhr, X_uc], axis=2)  # (n_samples, signal_len, 2)
        csp_array = self.csp.transform(multimodal_signals)  # (n_samples, n_csp)
        
        # Combine
        features = np.hstack([stat_array, csp_array])
        
        return features


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic normal vs pathological signals
    n_samples = 10
    signal_length = 100
    
    # Normal: FHR stable, UC regular
    X_fhr_normal = np.random.normal(130, 10, (n_samples, signal_length))
    X_uc_normal = np.sin(np.linspace(0, 4*np.pi, signal_length)) + np.random.normal(0, 0.1, signal_length)
    X_uc_normal = np.clip(X_uc_normal, 0, 1)
    
    # Pathological: FHR variable, UC irregular
    X_fhr_path = np.random.normal(120, 20, (n_samples, signal_length))
    X_uc_path = np.random.uniform(0, 1, (n_samples, signal_length))
    
    # Extract features
    extractor = MultimodalFeatureExtractor(n_csp_components=4)
    extractor.fit(X_fhr_normal, X_uc_normal, X_fhr_path, X_uc_path)
    
    features_normal = extractor.extract_batch(X_fhr_normal, X_uc_normal)
    features_path = extractor.extract_batch(X_fhr_path, X_uc_path)
    
    print(f"Normal features shape: {features_normal.shape}")
    print(f"Pathological features shape: {features_path.shape}")
    print(f"Feature names: FHR Mean, FHR Std, FHR Min, FHR Max, UC Mean, UC Std, UC Count, CrossCorr Max, CrossCorr Mean, CSP_0, CSP_1, CSP_2, CSP_3")

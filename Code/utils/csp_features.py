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
        Compute covariance matrix with Ledoit-Wolf shrinkage for numerical stability.
        
        This is a NOVEL enhancement using shrinkage estimator to prevent singular
        covariance matrices, especially important for highly correlated FHR-UC signals.
        
        Args:
            signal: Shape (n_samples, n_channels) [e.g., (1200, 2) for FHR + UC]
            
        Returns:
            cov: Regularized covariance matrix (n_channels, n_channels)
        """
        # Handle NaN/Inf in input signal
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
        
        signal_centered = signal - np.mean(signal, axis=0)
        n_samples, n_channels = signal.shape
        
        # Compute sample covariance
        sample_cov = np.dot(signal_centered.T, signal_centered) / max(n_samples - 1, 1)
        
        # Ledoit-Wolf shrinkage: cov = (1-shrinkage)*sample_cov + shrinkage*target
        # Target is a scaled identity matrix (spherical estimate)
        trace_cov = np.trace(sample_cov)
        target = (trace_cov / n_channels) * np.eye(n_channels) if trace_cov > 0 else np.eye(n_channels)
        
        # Adaptive shrinkage intensity based on condition number
        try:
            cond_num = np.linalg.cond(sample_cov)
            if cond_num > 1e6 or np.isnan(cond_num) or np.isinf(cond_num):
                shrinkage = 0.5  # High shrinkage for ill-conditioned matrices
            elif cond_num > 1e3:
                shrinkage = 0.2  # Moderate shrinkage
            else:
                shrinkage = 0.1  # Light shrinkage for well-conditioned matrices
        except:
            shrinkage = 0.5  # Default to high shrinkage on error
        
        # Apply shrinkage
        cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        # Additional regularization for guaranteed positive definiteness
        cov = cov + 1e-4 * np.eye(n_channels)
        
        # Final NaN check
        cov = np.nan_to_num(cov, nan=1e-4, posinf=1e4, neginf=-1e4)
        
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
        
        # Solve generalized eigenvalue problem with multi-layer fallback
        # Maximize: trace(w^T * Cov_pathological * w)
        # Subject to: w^T * Cov_normal * w = 1
        
        eigenvectors = None
        eigenvalues = None
        
        # Layer 1: Direct eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = eigh(cov_pathological, cov_normal)
            if np.isnan(eigenvectors).any() or np.isinf(eigenvectors).any():
                raise ValueError("NaN/Inf in eigenvectors")
        except Exception as e:
            pass  # Try Layer 2
        
        # Layer 2: Enhanced regularization and retry
        if eigenvectors is None:
            try:
                reg = 1e-2 * np.eye(n_channels)
                eigenvalues, eigenvectors = eigh(cov_pathological + reg, cov_normal + reg)
                if np.isnan(eigenvectors).any() or np.isinf(eigenvectors).any():
                    raise ValueError("NaN/Inf after regularization")
            except Exception as e:
                pass  # Try Layer 3
        
        # Layer 3: Standard eigendecomposition on difference matrix
        if eigenvectors is None:
            try:
                diff_cov = cov_pathological - cov_normal
                eigenvalues, eigenvectors = np.linalg.eigh(diff_cov)
                if np.isnan(eigenvectors).any() or np.isinf(eigenvectors).any():
                    raise ValueError("NaN/Inf in difference eigendecomp")
            except Exception as e:
                pass  # Use Layer 4 fallback
        
        # Layer 4: Ultimate fallback - orthogonal random projections
        if eigenvectors is None:
            print("CSP: Using fallback orthogonal projections (all decomposition methods failed)")
            eigenvectors = np.eye(n_channels)
            eigenvalues = np.ones(n_channels)
        
        # Select top and bottom components (most discriminative)
        idx = np.argsort(eigenvalues)
        n_select = max(self.n_components // 2, 1)
        idx_selected = np.concatenate([idx[:n_select], idx[-n_select:]])[:self.n_components]
        
        # Handle case where we don't have enough components
        while len(idx_selected) < self.n_components:
            idx_selected = np.concatenate([idx_selected, [idx_selected[-1]]])
        
        self.csp_filters = eigenvectors[:, idx_selected].T  # Shape: (n_components, n_channels)
        
        # Normalize filters to unit norm
        norms = np.linalg.norm(self.csp_filters, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        self.csp_filters = self.csp_filters / norms
        
        # Final NaN/Inf cleanup
        self.csp_filters = np.nan_to_num(self.csp_filters, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
        Extract features from batch of samples with robust NaN handling.
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_components))
        
        for i in range(n_samples):
            signal = X[i]  # (signal_length, 2)
            
            # Clean input signal
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply CSP filters
            csp_signal = np.dot(self.csp_filters, signal.T)  # (n_components, signal_length)
            
            # Robust variance calculation
            variances = np.var(csp_signal, axis=1)
            variances = np.clip(variances, 1e-10, 1e10)  # Bound variances
            
            # Log-variance feature with safe log
            log_vars = np.log(variances + 1e-8)
            
            # Aggressive NaN/Inf cleanup with bounded output
            log_vars = np.nan_to_num(log_vars, nan=0.0, posinf=5.0, neginf=-5.0)
            log_vars = np.clip(log_vars, -10.0, 10.0)  # Bound final output
            
            features[i] = log_vars
        
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
        
        # Fit MNE CSP
        # X shape: (n_epochs, n_channels, n_times) -> MNE expects features across time
        # MNE CSP expects: (n_epochs, n_channels, n_times)
        # We stacked as (n, len, 2) -> (n, 2, len)
        # UPDATE: The custom CSPFeatureExtractor expects (n_samples, time, channels)
        # So we do NOT transpose.
        
        X_train = np.concatenate([X_normal, X_path])
        y_train = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_path))])
        
        self.csp.fit(X_train, y_train)
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
        
        # NOVEL: Robust NaN imputation - replace NaN with column median
        # This prevents training instability from CSP numerical issues
        for col in range(features.shape[1]):
            col_data = features[:, col]
            nan_mask = np.isnan(col_data) | np.isinf(col_data)
            if nan_mask.any():
                # Use median of non-NaN values, or 0 if all NaN
                valid_data = col_data[~nan_mask]
                if len(valid_data) > 0:
                    fill_value = np.median(valid_data)
                else:
                    fill_value = 0.0
                features[nan_mask, col] = fill_value
        
        # Final clip to prevent extreme values
        features = np.clip(features, -100.0, 100.0)
        
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

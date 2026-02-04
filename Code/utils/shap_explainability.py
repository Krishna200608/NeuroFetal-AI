"""
SHAP Explainability for NeuroFetal AI
=====================================
SHAP-based feature importance analysis for the tabular branch.

Provides:
- Feature importance ranking
- SHAP summary plots
- Individual prediction explanations
- Clinical feature interpretation

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP Explainer for the Enhanced Fusion ResNet.
    
    Since SHAP works on tabular data, we create a wrapper that:
    1. Feeds fixed FHR through the FHR branch to get embeddings
    2. Then uses SHAP on the tabular + CSP features
    
    For full model interpretation, we can also use GradientExplainer
    on the entire model.
    """
    
    def __init__(self, model, feature_names=None, background_size=100):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained Keras model (3-input enhanced model)
            feature_names: List of feature names for tabular + CSP
            background_size: Number of background samples for SHAP
        """
        self.model = model
        self.feature_names = feature_names
        self.background_size = background_size
        self.explainer = None
        self.is_fitted = False
        
    def _create_tabular_wrapper(self, X_fhr_background):
        """
        Create a wrapper function that takes only tabular+CSP features.
        FHR is averaged over background samples.
        """
        # Average FHR embedding (fixed context)
        self.fhr_background = X_fhr_background
        
        def predict_tabular_only(X_combined):
            """
            Predict using fixed FHR background and variable tabular+CSP.
            X_combined: [tabular, csp] concatenated
            """
            n_samples = X_combined.shape[0]
            n_tabular = 16  # Adjust based on your actual tabular size
            
            # Split tabular and CSP
            X_tabular = X_combined[:, :n_tabular]
            X_csp = X_combined[:, n_tabular:]
            
            # Use mean FHR for all samples (or repeat)
            X_fhr = np.tile(self.fhr_background.mean(axis=0, keepdims=True), (n_samples, 1, 1))
            
            # Predict
            return self.model.predict([X_fhr, X_tabular, X_csp], verbose=0).flatten()
        
        return predict_tabular_only
    
    def fit(self, X_fhr, X_tabular, X_csp, sample_size=None):
        """
        Fit the SHAP explainer with background data.
        
        Args:
            X_fhr: Background FHR samples
            X_tabular: Background tabular samples
            X_csp: Background CSP samples
            sample_size: Number of samples to use (for efficiency)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
            
        # Sample if needed
        if sample_size and sample_size < len(X_fhr):
            idx = np.random.choice(len(X_fhr), sample_size, replace=False)
            X_fhr = X_fhr[idx]
            X_tabular = X_tabular[idx]
            X_csp = X_csp[idx]
        
        # Combine tabular + CSP for SHAP
        X_combined = np.hstack([X_tabular, X_csp])
        
        # Create wrapper function
        predict_fn = self._create_tabular_wrapper(X_fhr)
        
        # Create SHAP explainer
        # KernelExplainer is model-agnostic but slower
        # For speed, use a subset as background
        background_idx = np.random.choice(
            len(X_combined), 
            min(self.background_size, len(X_combined)), 
            replace=False
        )
        background = X_combined[background_idx]
        
        self.explainer = shap.KernelExplainer(predict_fn, background)
        self.is_fitted = True
        print(f"SHAP Explainer fitted with {len(background)} background samples")
        
    def explain(self, X_tabular, X_csp, n_samples=10):
        """
        Compute SHAP values for given samples.
        
        Args:
            X_tabular: Tabular features to explain
            X_csp: CSP features to explain
            n_samples: Number of samples to explain (for efficiency)
            
        Returns:
            shap_values: SHAP values array
        """
        if not self.is_fitted:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        X_combined = np.hstack([X_tabular[:n_samples], X_csp[:n_samples]])
        
        print(f"Computing SHAP values for {len(X_combined)} samples...")
        shap_values = self.explainer.shap_values(X_combined, nsamples=100)
        
        return shap_values
    
    def plot_summary(self, shap_values, X_combined, save_path=None):
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: Computed SHAP values
            X_combined: Feature matrix [tabular | csp]
            save_path: Optional path to save the plot
        """
        if not PLT_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
            
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_combined, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP summary plot saved to: {save_path}")
        else:
            plt.show()
            
    def plot_force(self, shap_values, X_combined, sample_idx=0, save_path=None):
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            shap_values: SHAP values
            X_combined: Feature matrix
            sample_idx: Index of sample to explain
            save_path: Optional path to save
        """
        if not PLT_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
            
        # Get expected value
        expected_value = self.explainer.expected_value
        
        # Force plot
        shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            X_combined[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP force plot saved to: {save_path}")
        else:
            plt.show()
            
    def get_feature_importance(self, shap_values):
        """
        Get feature importance ranking from SHAP values.
        
        Args:
            shap_values: Computed SHAP values
            
        Returns:
            importance_df: DataFrame with feature importance
        """
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        
        print("\n" + "="*50)
        print("Feature Importance (Mean |SHAP|)")
        print("="*50)
        for i in sorted_idx:
            print(f"{feature_names[i]:25s}: {importance[i]:.4f}")
        print("="*50)
        
        return {
            'feature_names': [feature_names[i] for i in sorted_idx],
            'importance': importance[sorted_idx]
        }


# ============================================================================
# Gradient-based Explainability (For FHR Signal)
# ============================================================================

class GradCAMExplainer:
    """
    Grad-CAM style explainability for the FHR signal branch.
    
    Highlights which parts of the FHR signal are most important
    for the model's prediction.
    """
    
    def __init__(self, model, layer_name='fhr_gap'):
        """
        Args:
            model: Trained Keras model
            layer_name: Name of the layer to compute gradients for
        """
        self.model = model
        self.layer_name = layer_name
        
    def compute_gradcam(self, X_fhr, X_tabular, X_csp):
        """
        Compute Grad-CAM heatmap for FHR signal.
        
        Returns:
            heatmap: Importance weights per time step
        """
        import tensorflow as tf
        
        # Get the target layer
        try:
            target_layer = self.model.get_layer(self.layer_name)
        except:
            print(f"Layer {self.layer_name} not found. Using last conv layer.")
            # Find last Conv1D layer
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    target_layer = layer
                    break
        
        # Create gradient model
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model([X_fhr, X_tabular, X_csp])
            loss = predictions[:, 0]  # For binary classification
            
        grads = tape.gradient(loss, conv_output)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Weight the output by gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SHAP Explainability Module Test")
    print("="*60)
    
    if not SHAP_AVAILABLE:
        print("\nSHAP not installed. Skipping tests.")
        print("Install with: pip install shap")
    else:
        print("\n✓ SHAP is available")
        
        # Test feature names from CSP extractor
        try:
            from csp_features import MultimodalFeatureExtractor
            feature_names = MultimodalFeatureExtractor.get_feature_names(n_csp=4)
            print(f"\nFeature names ({len(feature_names)} total):")
            for i, name in enumerate(feature_names):
                print(f"  {i+1:2d}. {name}")
        except ImportError:
            print("\nCould not import csp_features module")
            feature_names = None
            
    print("\n" + "="*60)
    print("Module loaded successfully!")
    print("="*60)

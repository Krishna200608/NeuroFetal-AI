#!/usr/bin/env python3
"""
MC Dropout Uncertainty Evaluation for NeuroFetal AI
====================================================
NOVEL contribution: Uncertainty quantification for clinical decision support.

This script evaluates prediction confidence using Monte Carlo Dropout,
providing calibrated uncertainty estimates for fetal distress predictions.

Key Features:
1. MC Dropout ensemble for predictive uncertainty
2. Expected Calibration Error (ECE) for calibration assessment
3. Uncertainty-aware decision thresholds
4. AUROC stratified by uncertainty (high vs low confidence)

Author: NeuroFetal AI Team
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from csp_features import MultimodalFeatureExtractor
# ============================================================================
# Configuration
# ============================================================================

N_MC_SAMPLES = 50  # Number of forward passes for MC Dropout
CONFIDENCE_THRESHOLD = 0.3  # Uncertainty threshold for clinical flagging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "Code", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "Reports", "uncertainty_analysis")


def enable_mc_dropout(model):
    """
    Enable dropout at inference time for MC Dropout.
    
    This is done by setting training=True for Dropout layers during prediction.
    """
    # Create a function that calls the model with training=True
    @tf.function
    def mc_predict(inputs):
        return model(inputs, training=True)
    return mc_predict


def mc_dropout_predict(model, X, n_samples=N_MC_SAMPLES):
    """
    Get predictions with uncertainty using MC Dropout.
    
    Performs multiple stochastic forward passes and returns:
    - Mean prediction (point estimate)
    - Predictive uncertainty (epistemic + aleatoric)
    
    Args:
        model: Trained Keras model with Dropout layers
        X: Input data (list of arrays for multi-input model)
        n_samples: Number of MC forward passes
    
    Returns:
        mean_pred: Mean prediction probability
        uncertainty: Standard deviation of predictions (uncertainty)
        all_preds: All individual predictions for analysis
    """
    predictions = []
    
    for i in range(n_samples):
        # Run with training=True to enable dropout
        pred = model.predict(X, verbose=0)
        predictions.append(pred.flatten())
    
    predictions = np.array(predictions)  # (n_samples, n_data_points)
    
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    return mean_pred, uncertainty, predictions


def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match observed frequencies.
    Lower ECE = better calibrated model.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
    
    Returns:
        ece: Expected Calibration Error
        bin_data: Dictionary with bin-wise calibration info
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = {'bins': [], 'accuracy': [], 'confidence': [], 'count': []}
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_pred[in_bin])
            ece += np.abs(bin_accuracy - bin_confidence) * (bin_count / len(y_true))
            
            bin_data['bins'].append((bin_lower, bin_upper))
            bin_data['accuracy'].append(bin_accuracy)
            bin_data['confidence'].append(bin_confidence)
            bin_data['count'].append(bin_count)
    
    return ece, bin_data


def stratify_by_uncertainty(y_true, y_pred, uncertainty, threshold=0.15):
    """
    Analyze performance stratified by prediction uncertainty.
    
    High-confidence predictions should have better AUC than low-confidence ones.
    
    Args:
        y_true: True labels
        y_pred: Mean predictions
        uncertainty: Prediction uncertainties
        threshold: Uncertainty threshold for stratification
    
    Returns:
        results: Dictionary with stratified metrics
    """
    low_uncertainty_mask = uncertainty < threshold
    high_uncertainty_mask = ~low_uncertainty_mask
    
    results = {
        'n_low_uncertainty': np.sum(low_uncertainty_mask),
        'n_high_uncertainty': np.sum(high_uncertainty_mask),
        'mean_uncertainty': np.mean(uncertainty),
        'threshold': threshold
    }
    
    # AUC for low uncertainty (high confidence) predictions
    if np.sum(low_uncertainty_mask) > 10 and len(np.unique(y_true[low_uncertainty_mask])) > 1:
        results['auc_low_uncertainty'] = roc_auc_score(
            y_true[low_uncertainty_mask], 
            y_pred[low_uncertainty_mask]
        )
    else:
        results['auc_low_uncertainty'] = None
    
    # AUC for high uncertainty (low confidence) predictions
    if np.sum(high_uncertainty_mask) > 10 and len(np.unique(y_true[high_uncertainty_mask])) > 1:
        results['auc_high_uncertainty'] = roc_auc_score(
            y_true[high_uncertainty_mask], 
            y_pred[high_uncertainty_mask]
        )
    else:
        results['auc_high_uncertainty'] = None
    
    return results


def plot_calibration_curve(y_true, y_pred, save_path=None):
    """Plot reliability diagram for calibration assessment."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Model calibration
    ax.plot(prob_pred, prob_true, 'o-', color='#E24A33', label='Model calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Calibration plot saved to: {save_path}")
    
    plt.close()


def plot_uncertainty_histogram(uncertainty, y_pred, y_true, save_path=None):
    """Plot uncertainty distribution with correct/incorrect stratification."""
    correct = (y_pred > 0.5).astype(int) == y_true
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(uncertainty[correct], bins=30, alpha=0.6, label='Correct predictions', color='#348ABD')
    ax.hist(uncertainty[~correct], bins=30, alpha=0.6, label='Incorrect predictions', color='#E24A33')
    
    ax.axvline(np.mean(uncertainty), color='black', linestyle='--', 
               label=f'Mean uncertainty: {np.mean(uncertainty):.3f}')
    
    ax.set_xlabel('Prediction Uncertainty (Std Dev)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect Predictions', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Uncertainty histogram saved to: {save_path}")
    
    plt.close()


def evaluate_uncertainty(model_path, X_val, y_val, output_dir=None):
    """
    Comprehensive uncertainty evaluation for a trained model.
    
    Args:
        model_path: Path to trained Keras model
        X_val: Validation inputs (list for multi-input model)
        y_val: Validation labels
        output_dir: Directory to save plots
    
    Returns:
        results: Dictionary with all uncertainty metrics
    """
    print("\n" + "="*60)
    print("MC Dropout Uncertainty Evaluation")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Get MC Dropout predictions
    print(f"\nPerforming {N_MC_SAMPLES} MC Dropout forward passes...")
    mean_pred, uncertainty, all_preds = mc_dropout_predict(model, X_val)
    
    # Basic stats
    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction: {np.mean(mean_pred):.4f}")
    print(f"  Mean uncertainty: {np.mean(uncertainty):.4f}")
    print(f"  Max uncertainty: {np.max(uncertainty):.4f}")
    
    # Calculate metrics
    auc = roc_auc_score(y_val, mean_pred)
    ece, bin_data = expected_calibration_error(y_val, mean_pred)
    stratified = stratify_by_uncertainty(y_val, mean_pred, uncertainty)
    
    print(f"\nPerformance Metrics:")
    print(f"  AUC (MC mean): {auc:.4f}")
    print(f"  Expected Calibration Error: {ece:.4f}")
    print(f"\nUncertainty Stratification (threshold={stratified['threshold']}):")
    print(f"  Low uncertainty samples: {stratified['n_low_uncertainty']}")
    print(f"  High uncertainty samples: {stratified['n_high_uncertainty']}")
    if stratified['auc_low_uncertainty']:
        print(f"  AUC (low uncertainty): {stratified['auc_low_uncertainty']:.4f}")
    if stratified['auc_high_uncertainty']:
        print(f"  AUC (high uncertainty): {stratified['auc_high_uncertainty']:.4f}")
    
    # Clinical interpretation
    high_risk_uncertain = np.sum((mean_pred > 0.5) & (uncertainty > 0.15))
    print(f"\n⚠️  Clinical Alert: {high_risk_uncertain} high-risk predictions with high uncertainty")
    print(f"   These cases should be flagged for additional clinical review.")
    
    # Generate plots if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plot_calibration_curve(
            y_val, mean_pred, 
            save_path=os.path.join(output_dir, 'calibration_curve.png')
        )
        plot_uncertainty_histogram(
            uncertainty, mean_pred, y_val,
            save_path=os.path.join(output_dir, 'uncertainty_histogram.png')
        )
    
    results = {
        'auc': auc,
        'ece': ece,
        'mean_uncertainty': np.mean(uncertainty),
        'stratified_metrics': stratified,
        'predictions': mean_pred,
        'uncertainties': uncertainty
    }
    
    return results




def main():
    """Main entry point for uncertainty evaluation."""
    
    # Load all data
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
    
    print("Loading data...")
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tab = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    # Load UC features for CSP
    uc_path = os.path.join(PROCESSED_DATA_DIR, "X_uc.npy")
    if os.path.exists(uc_path):
        print("Detailed: Loaded UC signals for CSP.")
        X_uc = np.load(uc_path)
    else:
        print("Warning: UC signals not found. Using 2-input mode (No CSP).")
        X_uc = None
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Replicate StratifiedKFold to get the same validation splits as training
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fhr, y), 1):
        print(f"\nProcessing Fold {fold}...")
        
        model_path = os.path.join(MODEL_DIR, f"enhanced_model_fold_{fold}.keras")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Skipping.")
            continue
            
        # Prepare data for this fold
        X_fhr_train = X_fhr[train_idx]
        X_fhr_val = X_fhr[val_idx]
        
        X_tab_val = X_tab[val_idx]
        y_val_fold = y[val_idx]
        y_train_fold = y[train_idx]
        
        if X_uc is not None:
             X_uc_train = X_uc[train_idx]
             X_uc_val = X_uc[val_idx]
             
             # Squeeze for CSP (N, L)
             X_fhr_train_2d = X_fhr_train.squeeze()
             X_uc_train_2d = X_uc_train.squeeze()
             X_fhr_val_2d = X_fhr_val.squeeze()
             X_uc_val_2d = X_uc_val.squeeze()
             
             # Create and Fit Multimodal Extractor
             # Default n_csp=4 + 15 stats = 19 features (Matches model input)
             extractor = MultimodalFeatureExtractor(n_csp_components=4)
             
             # Normal/Pathologic masks for CSP
             normal_mask = (y_train_fold == 0)
             path_mask = (y_train_fold == 1)
             
             extractor.fit(
                X_fhr_train_2d[normal_mask], X_uc_train_2d[normal_mask],
                X_fhr_train_2d[path_mask], X_uc_train_2d[path_mask]
             )
             
             # Transform Validation Data
             X_csp_val = extractor.extract_batch(X_fhr_val_2d, X_uc_val_2d)
             
             X_val_inputs = [X_fhr_val, X_tab_val, X_csp_val]
        else:
             X_val_inputs = [X_fhr_val, X_tab_val]
             
        # Evaluate
        fold_output_dir = os.path.join(RESULTS_DIR, f"fold_{fold}")
        results = evaluate_uncertainty(
            model_path, X_val_inputs, y_val_fold,
            output_dir=fold_output_dir
        )
        fold_metrics.append(results)
        print(f"Fold {fold} AUC: {results['auc']:.4f}")

    if fold_metrics:
        # Average metrics
        mean_auc = np.mean([r['auc'] for r in fold_metrics])
        mean_ece = np.mean([r['ece'] for r in fold_metrics])
        mean_unc = np.mean([r['mean_uncertainty'] for r in fold_metrics])
        
        print(f"\n{'='*60}")
        print(f"Overall Evaluation Results (Mean across {len(fold_metrics)} folds)")
        print(f"{'='*60}")
        print(f"Mean AUC: {mean_auc:.4f}")
        print(f"Mean ECE: {mean_ece:.4f}")
        print(f"Mean Uncertainty: {mean_unc:.4f}")
    
    print("\nUncertainty evaluation complete!")




if __name__ == "__main__":
    main()

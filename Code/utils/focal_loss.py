"""
Focal Loss Implementation
==========================
Implements Focal Loss for handling extreme class imbalance in fetal compromise detection.

Focal Loss formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
- α_t: Weighting factor for class imbalance
- γ: Focusing parameter (typically 2.0)
- (1 - p_t)^γ: Down-weights easy examples, focuses on hard negatives

Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
Adapted for medical imbalanced classification (~7% positive rate in CTU-UHB)
"""

import tensorflow as tf
from tensorflow.keras import losses


class FocalLoss(losses.Loss):
    """
    Focal Loss for binary classification with extreme class imbalance.
    
    Parameters:
    - alpha: Weight for positive class (default: 0.25, since ~7% positive in CTU-UHB)
    - gamma: Focusing parameter (default: 2.0)
    - reduction: Reduction method ('auto', 'sum', 'none')
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction="sum_over_batch_size", name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        """
        Compute focal loss with improved numerical stability.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities [0, 1]
            
        Returns:
            focal_loss: Computed loss
        """
        # Cast to float32 and flatten
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Use larger epsilon for stability (especially with mixed precision)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute p_t (probability of correct class)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Compute binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * bce
        
        return focal_loss


class WeightedFocalLoss(losses.Loss):
    """
    Focal Loss with class weighting for extreme imbalance.
    Combines benefits of Focal Loss + class weighting.
    
    Parameters:
    - alpha: Weight for positive class
    - gamma: Focusing parameter
    - pos_weight: Additional weight multiplier for positive class
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=5.0, reduction="sum_over_batch_size"):
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        """
        Compute weighted focal loss with improved numerical stability.
        """
        # Cast to float32 and flatten
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Use larger epsilon for stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute p_t (probability of correct class)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Weighted focal loss (alpha for balance, pos_weight for class imbalance)
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        class_weight = y_true * self.pos_weight + (1 - y_true) * 1.0
        
        focal_loss = alpha_weight * class_weight * focal_weight * bce
        
        return focal_loss


def get_focal_loss(alpha=0.25, gamma=2.0, use_weighted=False, pos_weight=5.0):
    """
    Factory function to create focal loss.
    
    Args:
        alpha: Alpha parameter
        gamma: Gamma parameter
        use_weighted: Whether to use weighted variant
        pos_weight: Positive class weight (only if use_weighted=True)
        
    Returns:
        loss: Focal loss instance
    """
    if use_weighted:
        return WeightedFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    else:
        return FocalLoss(alpha=alpha, gamma=gamma)


# Comparison functions
def binary_crossentropy_loss(y_true, y_pred):
    """Standard binary cross-entropy for baseline comparison."""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)


def weighted_bce_loss(y_true, y_pred, pos_weight=5.0):
    """
    Weighted BCE (traditional approach).
    Used as baseline comparison.
    """
    bce = binary_crossentropy_loss(y_true, y_pred)
    return y_true * pos_weight * bce + (1 - y_true) * bce


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Demonstration: Compare loss curves
    y_true_positive = np.ones(100)  # True positive class
    y_true_negative = np.zeros(100)  # True negative class
    
    # Range of predictions
    p_range = np.linspace(0.01, 0.99, 100)
    
    # Compute losses
    focal_losses_pos = []
    focal_losses_neg = []
    bce_losses_pos = []
    bce_losses_neg = []
    
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    for p in p_range:
        # Positive class (y_true=1)
        focal_pos = loss_fn(tf.constant([1.0]), tf.constant([p])).numpy()
        bce_pos = binary_crossentropy_loss(tf.constant([1.0]), tf.constant([p])).numpy()
        
        # Negative class (y_true=0)
        focal_neg = loss_fn(tf.constant([0.0]), tf.constant([p])).numpy()
        bce_neg = binary_crossentropy_loss(tf.constant([0.0]), tf.constant([p])).numpy()
        
        focal_losses_pos.append(focal_pos)
        focal_losses_neg.append(focal_neg)
        bce_losses_pos.append(bce_pos)
        bce_losses_neg.append(bce_neg)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(p_range, bce_losses_pos, label='BCE (y=1)', linestyle='--')
    plt.plot(p_range, focal_losses_pos, label='Focal (y=1)', linewidth=2)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Positive Class (y=1)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(p_range, bce_losses_neg, label='BCE (y=0)', linestyle='--')
    plt.plot(p_range, focal_losses_neg, label='Focal (y=0)', linewidth=2)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Negative Class (y=0)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    print("Focal Loss vs BCE comparison plot (visual demonstration only)")
    print("\nKey insight: Focal Loss (solid) down-weights easy negatives (left) more aggressively")
    print("than BCE (dashed), focusing training on hard examples.")

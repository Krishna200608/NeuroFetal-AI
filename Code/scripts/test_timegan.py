import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from timegan_utils import apply_per_fold_timegan_augmentation

def test_augmentation():
    print("Testing Per-Fold TimeGAN Augmentation...")
    
    # Create dummy data
    N = 100
    N_patho = 20
    
    # 80 normal, 20 patho -> needs 20 synthetic to balance (to reach 1:2 ratio? No, SMOTE was 1:2, wait. 
    # Ah, in apply_per_fold_timegan, n_negative - n_positive -> 80 - 20 = 60 synthetic.
    # So 20 patho + 60 synth = 80 positive, 80 negative. 1:1 balance.)
    
    X_fhr = np.random.randn(N, 1200, 1).astype(np.float32)
    X_uc = np.random.randn(N, 1200, 1).astype(np.float32)
    X_tab = np.random.randn(N, 18).astype(np.float32)
    
    y = np.zeros(N)
    y[:N_patho] = 1  # First 20 are pathological
    
    # Run augmentation
    print("Running augmentation (2 epochs)...")
    X_fhr_aug, X_uc_aug, X_tab_aug, y_aug = apply_per_fold_timegan_augmentation(
        X_fhr, X_uc, X_tab, y, epochs=2, batch_size=16
    )
    
    print(f"Original shape: FHR {X_fhr.shape}, UC {X_uc.shape}, Target {y.shape}")
    print(f"Augmented shape: FHR {X_fhr_aug.shape}, UC {X_uc_aug.shape}, Target {y_aug.shape}")
    print(f"Target distribution: {np.sum(y_aug)} pathological, {len(y_aug) - np.sum(y_aug)} normal")
    
    assert X_fhr_aug.shape[0] == len(y_aug)
    assert X_fhr_aug.shape[1:] == (1200, 1)
    print("Test passed successfully!")

if __name__ == "__main__":
    test_augmentation()

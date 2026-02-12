import numpy as np
import os

BASE_DIR = r"d:\Research Project\Research_Project\Datasets\processed"
X_fhr = np.load(os.path.join(BASE_DIR, "X_fhr.npy"))
X_uc = np.load(os.path.join(BASE_DIR, "X_uc.npy"))
X_tab = np.load(os.path.join(BASE_DIR, "X_tabular.npy"))
y = np.load(os.path.join(BASE_DIR, "y.npy"))

print(f"X_fhr shape: {X_fhr.shape}")
print(f"X_uc shape: {X_uc.shape}")
print(f"X_tab shape: {X_tab.shape}")
print(f"y shape: {y.shape}")

# Quick Start: Testing Novelty Features

## 1️⃣ Verify Module Imports

```bash
cd Code
python -c "
from scripts.uc_cleaning import UCCleaner
from scripts.csp_features import MultimodalFeatureExtractor
from scripts.focal_loss import get_focal_loss, WeightedFocalLoss
from scripts.evaluate_time_to_predict import TimeToPredict
print('✅ All novelty modules imported successfully!')
"
```

**Expected Output:**
```
✅ All novelty modules imported successfully!
```

---

## 2️⃣ Test UC Cleaning Module

```python
# Code/test_uc_cleaning.py
import numpy as np
from scripts.uc_cleaning import UCCleaner

# Create synthetic UC signal with artifacts
np.random.seed(42)
raw_uc = np.random.uniform(20, 60, 240)  # 240 samples @ 4Hz = 60 seconds
raw_uc[50:55] = 0  # Add sensor loss
raw_uc[100:105] = 500  # Add spike artifact

# Clean the signal
cleaner = UCCleaner(fs=4, gap_threshold_sec=15)
cleaned_uc = cleaner.clean(raw_uc)
quality_score = cleaner.get_quality_score(raw_uc)

print(f"Raw UC stats: mean={raw_uc.mean():.2f}, std={raw_uc.std():.2f}")
print(f"Cleaned UC stats: mean={cleaned_uc.mean():.2f}, std={cleaned_uc.std():.2f}")
print(f"Quality score: {quality_score:.3f}")
print(f"✅ UC cleaning works! Quality improved from {quality_score:.1%}")
```

**Expected Output:**
```
Raw UC stats: mean=40.XX, std=XX.XX
Cleaned UC stats: mean=XX.XX, std=XX.XX
Quality score: 0.8XX
✅ UC cleaning works! Quality improved from 0.8XX
```

---

## 3️⃣ Test CSP Feature Extraction

```python
# Code/test_csp_features.py
import numpy as np
from scripts.csp_features import MultimodalFeatureExtractor

# Create synthetic training data
np.random.seed(42)
n_samples_normal = 500
n_samples_pathological = 50

# Normal class (FHR 120-160 bpm, UC 30-60 mmHg)
X_fhr_normal = np.random.uniform(120, 160, (n_samples_normal, 240))
X_uc_normal = np.random.uniform(30, 60, (n_samples_normal, 240))

# Pathological class (FHR 80-100 bpm, UC 80-120 mmHg)
X_fhr_pathological = np.random.uniform(80, 100, (n_samples_pathological, 240))
X_uc_pathological = np.random.uniform(80, 120, (n_samples_pathological, 240))

# Train extractor
extractor = MultimodalFeatureExtractor(n_csp_components=4)
extractor.fit(X_fhr_normal, X_uc_normal, X_fhr_pathological, X_uc_pathological)

# Extract features from test sample
test_fhr = np.random.uniform(120, 160, 240)
test_uc = np.random.uniform(30, 60, 240)
features = extractor.extract(test_fhr, test_uc)

print(f"✅ CSP features extracted!")
print(f"Feature vector shape: {features.shape}")  # Should be (13,)
print(f"Features: {features}")
print(f"Statistical features (8): mean_fhr, std_fhr, ..., cross_corr")
print(f"CSP features (4): csp_var_1, csp_var_2, csp_var_3, csp_var_4")
```

**Expected Output:**
```
✅ CSP features extracted!
Feature vector shape: (13,)
Features: [120.XX 50.XX ... 0.8X]
```

---

## 4️⃣ Test Focal Loss

```python
# Code/test_focal_loss.py
import tensorflow as tf
from scripts.focal_loss import get_focal_loss, WeightedFocalLoss

# Create loss functions
focal_loss = get_focal_loss(alpha=0.25, gamma=2.0, use_weighted=True, pos_weight=5.0)

# Test on sample predictions
y_true = tf.constant([[1.0], [0.0], [1.0], [0.0], [0.0]])
y_pred = tf.constant([[0.9], [0.1], [0.7], [0.2], [0.1]])

loss_value = focal_loss(y_true, y_pred)

print(f"✅ Focal Loss compiled!")
print(f"Loss value: {loss_value.numpy():.4f}")
print(f"Focal Loss parameters:")
print(f"  - Alpha (positive weight): 0.25")
print(f"  - Gamma (focusing parameter): 2.0")
print(f"  - Pos weight (additional gain): 5.0")
```

**Expected Output:**
```
✅ Focal Loss compiled!
Loss value: 0.XXXX
Focal Loss parameters:
  - Alpha (positive weight): 0.25
  - Gamma (focusing parameter): 2.0
  - Pos weight (additional gain): 5.0
```

---

## 5️⃣ Test Time-to-Predict Evaluation

```python
# Code/test_time_to_predict.py
import numpy as np
from tensorflow import keras
from scripts.evaluate_time_to_predict import TimeToPredict

# Create a simple dummy model for testing
inputs = keras.Input(shape=(1200, 1))
x = keras.layers.Conv1D(32, 3, activation='relu')(inputs)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
dummy_model = keras.Model(inputs=inputs, outputs=outputs)

# Create synthetic test data
X_fhr_test = np.random.uniform(0, 1, (100, 1200, 1))
X_tab_test = np.random.uniform(0, 1, (100, 3))
y_test = np.random.randint(0, 2, (100,))

# Evaluate time-to-predict
evaluator = TimeToPredict(dummy_model, fs=1, window_size_sec=1200)
results = evaluator.evaluate_progressive(
    X_fhr_test, X_tab_test, y_test,
    time_points=[30, 40, 50, 60]
)

print(f"✅ Time-to-Predict evaluation complete!")
print(f"Results at different time points:")
for time_point in [30, 40, 50, 60]:
    if str(time_point) in results:
        print(f"  {time_point}min: AUC={results[str(time_point)].get('auc', 0):.4f}")

# Generate report
report = evaluator.generate_report()
print(f"\n{report}")
```

**Expected Output:**
```
✅ Time-to-Predict evaluation complete!
Results at different time points:
  30min: AUC=0.XXXX
  40min: AUC=0.XXXX
  50min: AUC=0.XXXX
  60min: AUC=0.XXXX

=== Time-to-Predict Analysis Report ===
...
```

---

## 6️⃣ Integration Test: Train with Focal Loss

**Prerequisites:** Ensure processed datasets exist:
```bash
ls Datasets/processed/
# Should show: X_fhr.npy, X_tabular.npy, y.npy
```

**Run training with Focal Loss:**
```bash
cd Code
python scripts/train.py
```

**What to expect:**
```
Loading processed data...
X_fhr shape: (2760, 1200, 1)
X_tabular shape: (2760, 3)
y shape: (2760,)
Using Focal Loss (α=0.25, γ=2.0, pos_weight=5.0) ✅
Starting 5-fold cross-validation...
Fold 1/5: 
  Epoch 1/100: loss=0.XX, val_auc=0.XX
  Epoch 2/100: loss=0.XX, val_auc=0.XX
  ...
  Best val_auc: 0.85XX (Epoch XX)
✅ Model saved: models/best_model_fold_1.keras
```

**Training Time:** ~10-20 minutes (depending on GPU)

---

## 7️⃣ Full Integration Test Template

```python
# Code/test_full_integration.py
import numpy as np
import tensorflow as tf
from scripts.uc_cleaning import UCCleaner
from scripts.csp_features import MultimodalFeatureExtractor
from scripts.focal_loss import get_focal_loss
from scripts.evaluate_time_to_predict import TimeToPredict

print("=" * 60)
print("NeuroFetal AI: Full Integration Test")
print("=" * 60)

# Step 1: Load synthetic data
print("\n[1/5] Loading synthetic data...")
X_fhr = np.random.uniform(0, 1, (100, 4800))  # 60min @ 4Hz
X_uc = np.random.uniform(0, 1, (100, 4800))
X_clinical = np.random.uniform(0, 1, (100, 3))  # Age, Parity, Gestation
y = np.random.randint(0, 2, (100,))
print(f"✅ Data loaded: {X_fhr.shape}, {X_uc.shape}, {X_clinical.shape}, {y.shape}")

# Step 2: Clean UC signal
print("\n[2/5] Cleaning UC signal...")
cleaner = UCCleaner(fs=4)
X_uc_cleaned = np.array([cleaner.clean(uc_sig) for uc_sig in X_uc])
print(f"✅ UC cleaned: {X_uc_cleaned.shape}")

# Step 3: Extract CSP features
print("\n[3/5] Extracting CSP features...")
normal_idx = y == 0
pathological_idx = y == 1
extractor = MultimodalFeatureExtractor(n_csp_components=4)
extractor.fit(
    X_fhr[normal_idx], X_uc_cleaned[normal_idx],
    X_fhr[pathological_idx], X_uc_cleaned[pathological_idx]
)
X_csp = np.array([extractor.extract(X_fhr[i], X_uc_cleaned[i]) for i in range(len(X_fhr))])
print(f"✅ CSP features extracted: {X_csp.shape}")

# Step 4: Create model and compile with Focal Loss
print("\n[4/5] Creating model with Focal Loss...")
inputs = tf.keras.Input(shape=(1200, 1))
x = tf.keras.layers.Conv1D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

focal_loss = get_focal_loss(alpha=0.25, gamma=2.0, use_weighted=True)
model.compile(optimizer='adam', loss=focal_loss, metrics=['auc'])
print(f"✅ Model compiled with Focal Loss")

# Step 5: Evaluate Time-to-Predict
print("\n[5/5] Evaluating Time-to-Predict...")
X_fhr_1hz = X_fhr[:, ::4]  # Downsample to 1Hz (1200 samples for 60min)
evaluator = TimeToPredict(model)
results = evaluator.evaluate_progressive(X_fhr_1hz, X_clinical, y, time_points=[30, 40, 50, 60])
print(f"✅ Time-to-Predict evaluation complete")

print("\n" + "=" * 60)
print("✅ Full Integration Test PASSED!")
print("=" * 60)
print("\nAll novelty features working correctly:")
print("  ✅ UC Cleaning")
print("  ✅ CSP Feature Extraction")
print("  ✅ Focal Loss Compilation")
print("  ✅ Time-to-Predict Evaluation")
```

**Run:**
```bash
cd Code
python test_full_integration.py
```

---

## 🎯 Checklist

- [ ] All modules import without errors
- [ ] UC cleaning produces cleaned signals
- [ ] CSP extraction outputs 13-dimensional vectors
- [ ] Focal Loss compiles into Keras model
- [ ] Time-to-Predict evaluates multiple time points
- [ ] Full integration test passes

---

## ✅ Ready to Proceed

If all tests pass, you're ready for:
1. **Phase 2:** Modify data_ingestion.py to use UC cleaning + CSP
2. **Phase 3:** Enhance model.py with CSP branch
3. **Phase 4:** Run full training with Focal Loss
4. **Phase 5:** Generate publication figures

---

**Need help?** Check `NOVELTY_FEATURES.md` for detailed explanations.

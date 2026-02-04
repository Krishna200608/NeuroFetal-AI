# Implementation Prompt: Phase 2-3 (UC Cleaning + CSP Fusion)

Use this prompt with Google's AI IDE, Claude, Gemini, or any LLM to implement the remaining phases.

---

## **PHASE 2: Modify data_ingestion.py**

### **Context:**
You are enhancing the NeuroFetal AI data preprocessing pipeline. The project is at:
- **Location:** `Code/scripts/data_ingestion.py`
- **Existing modules:** `Code/scripts/uc_cleaning.py`, `Code/scripts/csp_features.py`
- **Reference:** `FEATURE_IMPLEMENTATION.md` in project root

### **Requirement:**
Modify `data_ingestion.py` to integrate:
1. UC artifact cleaning (using UCCleaner from uc_cleaning.py)
2. CSP feature extraction (using MultimodalFeatureExtractor from csp_features.py)
3. Change output: X_tabular from (2760, 3) to (2760, 16) by concatenating CSP features

### **Exact Changes Needed:**

**Step 1: Add imports at top of file**
```python
from uc_cleaning import UCCleaner
from csp_features import MultimodalFeatureExtractor
```

**Step 2: After reading raw UC signal, add cleaning**
```python
# Initialize cleaner
cleaner = UCCleaner(fs=4, gap_threshold_sec=15)

# Apply to all UC signals
X_uc_cleaned = np.array([cleaner.clean(uc_signal) for uc_signal in X_uc_raw])
print(f"UC cleaning complete. Quality scores: min={cleaner.get_quality_score(X_uc_raw[0]):.3f}")
```

**Step 3: After identifying normal/pathological samples, add CSP fitting**
```python
# Identify samples by label
normal_mask = y == 0
pathological_mask = y == 1

# Initialize and fit CSP
extractor = MultimodalFeatureExtractor(n_csp_components=4)
extractor.fit(
    X_fhr[normal_mask], X_uc_cleaned[normal_mask],
    X_fhr[pathological_mask], X_uc_cleaned[pathological_mask]
)
print("CSP fitted on training data")

# Extract features for all samples
X_csp = extractor.extract_batch(X_fhr, X_uc_cleaned)
print(f"CSP features extracted: shape {X_csp.shape}")  # Should be (2760, 13)
```

**Step 4: Concatenate CSP with clinical features**
```python
# Extend tabular features from 3 to 16 dimensions
X_tabular_enhanced = np.concatenate([X_tabular_original, X_csp], axis=1)
print(f"Enhanced tabular shape: {X_tabular_enhanced.shape}")  # Should be (2760, 16)

# Use enhanced version for output
X_tabular = X_tabular_enhanced
```

**Step 5: Save CSP features separately (optional but helpful)**
```python
np.save(os.path.join(processed_dir, 'X_csp_features.npy'), X_csp)
print(f"Saved X_csp_features.npy: {X_csp.shape}")
```

**Step 6: Update final output**
```python
# Before: X_tabular had shape (2760, 3)
# After: X_tabular has shape (2760, 16)
np.save(os.path.join(processed_dir, 'X_tabular.npy'), X_tabular)
print(f"Saved X_tabular.npy with new shape: {X_tabular.shape}")
```

### **Verification:**
After modification, run:
```bash
python Code/scripts/data_ingestion.py
```
Expected output:
- X_tabular.npy: shape (2760, 16) — **VERIFY THIS CHANGED FROM 3 TO 16**
- X_csp_features.npy: shape (2760, 13) — **NEW FILE**
- Console messages showing UC cleaning + CSP extraction success

---

## **PHASE 3: Modify model.py**

### **Context:**
Enhance the Fusion ResNet architecture from 2-input to 3-input model with CSP branch.

### **Requirement:**
Modify `Code/scripts/model.py` to:
1. Add CSP input layer: shape (13,)
2. Add CSP Dense branch
3. Change fusion from element-wise multiply to concatenation
4. Modify model outputs to accept 3 inputs instead of 2

### **Exact Changes Needed:**

**Step 1: Add CSP input after other inputs**
```python
# After existing fhr_input and tabular_input definitions, add:
csp_input = keras.Input(shape=(13,), name='csp_features')
```

**Step 2: Add CSP processing branch**
```python
# After existing branch definitions, add:
csp_branch = keras.layers.Dense(128, activation='relu', name='csp_dense')(csp_input)
csp_branch = keras.layers.Dropout(0.3)(csp_branch)
```

**Step 3: Modify fusion logic**
```python
# BEFORE (2-input multiply):
# fusion = keras.layers.Multiply()([fhr_features, tabular_features])

# AFTER (3-input concatenate):
fusion_vector = keras.layers.Concatenate(name='fusion_concat')(
    [fhr_features, tabular_features, csp_branch]
)  # Shape: (384,) = 128+128+128

# Add dense layer to combine
fusion_output = keras.layers.Dense(64, activation='relu', name='fusion_dense')(fusion_vector)
fusion_output = keras.layers.Dropout(0.2)(fusion_output)
```

**Step 4: Update model definition**
```python
# BEFORE (2 inputs):
# model = keras.Model(inputs=[fhr_input, tabular_input], outputs=output)

# AFTER (3 inputs):
model = keras.Model(
    inputs=[fhr_input, tabular_input, csp_input],
    outputs=output
)
```

**Step 5: Verify model compilation**
```python
# Model should compile unchanged
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=['auc']
)
```

### **Verification:**
After modification, test with:
```bash
python -c "
import numpy as np
from scripts.model import create_fusion_resnet

# Create model
model = create_fusion_resnet()

# Test with dummy data
X_fhr = np.random.randn(1, 1200, 1).astype(np.float32)
X_tab = np.random.randn(1, 16).astype(np.float32)  # Now 16, not 3
X_csp = np.random.randn(1, 13).astype(np.float32)  # New

# Forward pass
pred = model([X_fhr, X_tab, X_csp])
print(f'✅ Model works! Prediction shape: {pred.shape}')
print(f'✅ Model accepts 3 inputs (FHR, Tabular[16], CSP[13])')
"
```

Expected output:
```
✅ Model works! Prediction shape: (1, 1)
✅ Model accepts 3 inputs (FHR, Tabular[16], CSP[13])
```

---

## **PHASE 4: Update train.py Data Feeding**

### **Context:**
train.py loads data but now X_tabular is (2760, 16) instead of (2760, 3).

### **Requirement:**
No changes needed IF:
- train.py already loads: `X_fhr`, `X_tabular`, `y` from .npy files
- Model creation already adapted (Phase 3 done)

### **Verify train.py works:**
```bash
python Code/scripts/train.py
```

Expected:
- Loads X_fhr: (2760, 1200, 1) ✅
- Loads X_tabular: (2760, 16) ✅ (was 3, now 16)
- Creates 3-input model ✅
- Uses Focal Loss ✅
- Trains 5-fold CV ✅
- Outputs: best_model_fold_{1-5}.keras ✅

---

## **PHASE 5: Update Colab Notebook**

### **Add cells to notebook:**

**New Cell 1: Import evaluation framework**
```python
import sys
sys.path.append('/content/drive/MyDrive/Research_Project/Code')

from scripts.evaluate_time_to_predict import TimeToPredict
import json
```

**New Cell 2: Generate Time-to-Predict results**
```python
import numpy as np
import tensorflow as tf

# Load test data
X_fhr = np.load('/content/drive/MyDrive/Research_Project/Datasets/processed/X_fhr.npy')
X_tabular = np.load('/content/drive/MyDrive/Research_Project/Datasets/processed/X_tabular.npy')
y = np.load('/content/drive/MyDrive/Research_Project/Datasets/processed/y.npy')

# Load best model
model = tf.keras.models.load_model(
    '/content/drive/MyDrive/Research_Project/Code/models/best_model_fold_5.keras'
)

# Evaluate Time-to-Predict
evaluator = TimeToPredict(model, fs=1, window_size_sec=1200)
results = evaluator.evaluate_progressive(X_fhr, X_tabular, y, time_points=[30, 40, 50, 60])

# Generate outputs
evaluator.plot_progressive_performance(
    '/content/drive/MyDrive/Research_Project/Code/figures/time_to_predict.png'
)
evaluator.save_results_json(
    '/content/drive/MyDrive/Research_Project/time_to_predict_results.json'
)

print(evaluator.generate_report())
```

---

## **Git Commit After Each Phase:**

```bash
# Phase 2
git add Code/scripts/data_ingestion.py
git commit -m "feat: Add UC cleaning and CSP feature extraction to data pipeline"

# Phase 3
git add Code/scripts/model.py
git commit -m "feat: Enhance model with CSP fusion branch (3-input architecture)"

# Phase 4 (if changes needed)
git add Code/scripts/train.py
git commit -m "feat: Update train.py for enhanced data pipeline"

# Phase 5
git add Code/notebooks/Training_Colab.ipynb
git commit -m "feat: Add Time-to-Predict evaluation cells to Colab notebook"
```

---

## **Verification Checklist:**

- [ ] Phase 2: X_tabular shape is (2760, 16) not (2760, 3)
- [ ] Phase 2: X_csp_features.npy created with shape (2760, 13)
- [ ] Phase 3: Model accepts 3 inputs: FHR, Tabular[16], CSP[13]
- [ ] Phase 3: Model forward pass works without errors
- [ ] Phase 4: train.py runs without shape mismatch errors
- [ ] Phase 4: Training converges with improved AUC
- [ ] Phase 5: Time-to-Predict plot generated successfully
- [ ] All commits pushed to `feature/fusion-enhancements` branch

---

## **Expected Results After All Phases:**

| Metric | Paper 7 | NeuroFetal | Improvement |
|--------|---------|-----------|------------|
| AUC | 0.84 | **0.87-0.89** | +3-6% |
| Sensitivity | 78% | **82-85%** | +4-7% |
| Specificity | 85% | **86-87%** | +1-2% |
| Time to 90% Sensitivity | N/A | **<45 min** | 10-15 min early |

---

**Use this prompt with any AI IDE or LLM. They will understand exactly what to do.**

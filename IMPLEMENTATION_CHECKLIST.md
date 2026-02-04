# NeuroFetal AI: Implementation Checklist

**Status:** Phase 1 Complete ✅ | Phase 2-5 TODO  
**Branch:** `feature/fusion-enhancements`  
**Last Updated:** February 4, 2026

---

## ✅ Phase 1: Core Modules Implementation (COMPLETED)

### Core Module Files Created
- [x] `Code/scripts/uc_cleaning.py` (280 lines)
  - `UCCleaner` class with 6 methods
  - Sensor loss detection, spike removal, gap interpolation, smoothing
  - Quality score computation
  - **Example usage:** `cleaner = UCCleaner(fs=4); cleaned_uc = cleaner.clean(raw_uc)`

- [x] `Code/scripts/csp_features.py` (360 lines)
  - `CSPFeatureExtractor` class (Common Spatial Pattern)
  - `MultimodalFeatureExtractor` class (Combined statistical + CSP)
  - Fits on training data, transforms signals
  - **Example usage:** `extractor.fit(X_normal, y_normal); features = extractor.extract(fhr, uc)`

- [x] `Code/scripts/focal_loss.py` (220 lines)
  - `FocalLoss` class (basic implementation)
  - `WeightedFocalLoss` class (enhanced with pos_weight)
  - Factory function `get_focal_loss()`
  - **Example usage:** `loss = get_focal_loss(alpha=0.25, gamma=2.0, use_weighted=True)`

- [x] `Code/scripts/evaluate_time_to_predict.py` (380 lines)
  - `TimeToPredict` class for progressive evaluation
  - Methods: evaluate_at_time, evaluate_progressive, find_time_to_sensitivity
  - Plotting and report generation
  - **Example usage:** `evaluator = TimeToPredict(model); results = evaluator.evaluate_progressive(X, y)`

### Integration Completed
- [x] `Code/scripts/train.py` modified
  - Added Focal Loss import
  - Added configuration flags: `USE_FOCAL_LOSS=True`
  - Integrated loss function selection
  - Added logging for active loss function

### Documentation Completed
- [x] `NOVELTY_FEATURES.md` - Comprehensive publication roadmap
- [x] `IMPLEMENTATION_CHECKLIST.md` (this file)

### Git Commits Completed
- [x] Commit 1: 4 core modules + 958 lines of code
- [x] Commit 2: Focal Loss integration in train.py
- [x] Commit 3: Documentation and novelty features guide

---

## ⏳ Phase 2: Enhanced Data Pipeline (READY FOR IMPLEMENTATION)

### Step 1: Update data_ingestion.py
**File:** `Code/scripts/data_ingestion.py`  
**Task:** Add UC cleaning and CSP feature extraction

**Changes Required:**
```python
# Add imports at top
from uc_cleaning import UCCleaner
from csp_features import MultimodalFeatureExtractor

# In preprocessing pipeline (after reading UC signal):
cleaner = UCCleaner(fs=4, gap_threshold_sec=15)
X_uc_cleaned = cleaner.clean(X_uc_raw)  # Apply cleaning

# In feature extraction (after loading normal/pathological data):
extractor = MultimodalFeatureExtractor(n_csp_components=4)
extractor.fit(X_fhr_train_normal, X_uc_cleaned_normal, 
              X_fhr_train_pathological, X_uc_cleaned_pathological)

# Extract features for all samples
X_csp_features = extractor.extract_batch(X_fhr, X_uc_cleaned)
# X_csp_features shape: (n_samples, 13)

# Concatenate with clinical features
X_tabular_enhanced = np.concatenate([X_tabular, X_csp_features], axis=1)
# Shape changes from (n_samples, 3) to (n_samples, 16)

# Save output
np.save('X_csp_features.npy', X_csp_features)
```

**Estimated Lines:** +60-80 lines  
**Priority:** HIGH (blocks model enhancement)

### Step 2: Ablation Study Configuration
**File:** Create `Code/scripts/ablation_config.py` (optional)

**Purpose:** Control which features are used in training
```python
ABLATION_EXPERIMENTS = {
    'baseline': {
        'name': 'Signal Only',
        'fhr': True, 'tabular': False, 'csp': False
    },
    'tabular_only': {
        'name': 'Clinical Features Only',
        'fhr': False, 'tabular': True, 'csp': False
    },
    'fusion': {
        'name': 'Fusion (Paper 7)',
        'fhr': True, 'tabular': True, 'csp': False
    },
    'fusion_csp': {
        'name': 'Fusion + CSP (Proposed)',
        'fhr': True, 'tabular': True, 'csp': True
    }
}
```

**Estimated Lines:** 30 lines  
**Priority:** MEDIUM (useful for comparison)

---

## ⏳ Phase 3: Enhanced Model Architecture (READY FOR IMPLEMENTATION)

### Step 1: Modify model.py
**File:** `Code/scripts/model.py`  
**Task:** Add CSP feature branch for 3-way fusion

**Architecture Changes:**
```python
# Current (Paper 7): FHR Branch + Tabular Branch + Multiply Fusion
# Proposed: FHR Branch + Tabular Branch + CSP Branch + 3-way Fusion

# NEW: CSP Input Branch
csp_input = keras.Input(shape=(13,), name='csp_features')
csp_branch = keras.layers.Dense(128, activation='relu')(csp_input)
csp_branch = keras.layers.Dropout(0.3)(csp_branch)

# Combine all three at fusion level (option 1: multiply all three)
# Or (option 2: concatenate then dense)
# Recommend: Concatenate (allows model to learn interaction)
fusion_vector = keras.layers.Concatenate()(
    [fhr_features, tabular_features, csp_branch]
)  # Shape: (384,) = 128+128+128

# Dense layer to combine
fusion_output = keras.layers.Dense(64, activation='relu')(fusion_vector)
fusion_output = keras.layers.Dropout(0.2)(fusion_output)
output = keras.layers.Dense(1, activation='sigmoid')(fusion_output)

# Create model with 3 inputs
model = keras.Model(inputs=[fhr_input, tabular_input, csp_input], 
                    outputs=output)
```

**Changes Required:**
- Add CSP input (shape: (13,))
- Add CSP Dense branch (128 units)
- Modify fusion logic (concatenate vs multiply)
- Update model compilation
- **Output shape:** Still (batch, 1) for binary classification

**Estimated Lines:** +30-40 lines  
**Priority:** HIGH (required for improvements)

---

## ⏳ Phase 4: Training & Evaluation (READY FOR EXECUTION)

### Step 1: Training with Focal Loss
**File:** `Code/scripts/train.py`  
**Status:** Ready to run with `USE_FOCAL_LOSS=True`

**Expected Commands:**
```bash
cd Code
python scripts/train.py  # Will use Focal Loss automatically
# Output: best_model_fold_{1-5}.keras + training logs
```

**Monitoring:**
- Track loss convergence (should stabilize by epoch 30-50)
- Monitor validation AUC (target: > 0.85)
- Check for overfitting (val_auc vs train_auc gap)

**Expected Artifacts:**
- `models/best_model_fold_1.keras` - Retrained with Focal Loss
- `models/best_model_fold_2.keras`
- `models/best_model_fold_3.keras`
- `models/best_model_fold_4.keras`
- `models/best_model_fold_5.keras`
- Training logs (optional: CSV with epoch metrics)

### Step 2: Time-to-Predict Evaluation
**File:** Create `Code/scripts/evaluate_models.py` (new)

**Purpose:** Generate Time-to-Predict plots and metrics

**Code Template:**
```python
from evaluate_time_to_predict import TimeToPredict
import numpy as np
from scripts.model import create_fusion_resnet
import tensorflow as tf

# Load best model
model = tf.keras.models.load_model('models/best_model_fold_5.keras')

# Load test data
X_fhr = np.load('Datasets/processed/X_fhr.npy')
X_tabular = np.load('Datasets/processed/X_tabular.npy')
y = np.load('Datasets/processed/y.npy')

# Evaluate time-to-predict
evaluator = TimeToPredict(model, fs=1, window_size_sec=1200)
results = evaluator.evaluate_progressive(
    X_fhr, X_tabular, y,
    time_points=[30, 40, 50, 60]
)

# Generate outputs
evaluator.plot_progressive_performance('time_to_predict_results.png')
evaluator.save_results_json('time_to_predict_results.json')
print(evaluator.generate_report())
```

**Expected Output:**
- `time_to_predict_results.png` - 4-panel plot (AUC, Sens, Spec, Acc vs time)
- `time_to_predict_results.json` - Metrics at each time point
- Console report with clinical latency metric

**Estimated Lines:** 50 lines  
**Priority:** HIGH (for publication)

### Step 3: Comparison Study (Focal Loss vs BCE)
**File:** Create `Code/scripts/compare_losses.py` (optional)

**Purpose:** A/B test Focal Loss improvement

**Approach:**
```python
# Train model with BCE (baseline)
model_bce = train_with_loss('binary_crossentropy')
results_bce = evaluate_model(model_bce)

# Train model with Focal Loss (proposed)
model_focal = train_with_loss('focal')
results_focal = evaluate_model(model_focal)

# Compare
comparison = {
    'BCE': {
        'auc': results_bce['auc'],
        'sensitivity': results_bce['sensitivity'],
        'specificity': results_bce['specificity'],
    },
    'Focal Loss': {
        'auc': results_focal['auc'],
        'sensitivity': results_focal['sensitivity'],
        'specificity': results_focal['specificity'],
    }
}
print_comparison_table(comparison)
```

**Expected Result:**
- Focal Loss: +2-3% AUC improvement
- Focal Loss: +4-6% Sensitivity improvement
- Focal Loss: -2-4% Specificity trade-off (acceptable for recall bias)

**Estimated Lines:** 80 lines  
**Priority:** MEDIUM (strengthens paper narrative)

---

## ⏳ Phase 5: Publication Preparation (DEFER FOR NOW)

### Step 1: Paper Writing
**File:** Create `Paper/NeuroFetal_AI_Publication.md` (new)

**Sections:**
- Abstract (250 words)
- Introduction (established problem, prior work)
- Methods (UC cleaning + CSP + Focal Loss + Time-to-Predict)
- Results (AUC tables, ablation study, Time-to-Predict plots)
- Discussion (clinical implications, limitations)
- References (7 papers + literature)

**Expected Length:** 6000-8000 words  
**Priority:** DEFER (after validation complete)

### Step 2: Supplementary Materials
- High-resolution figures (Grad-CAM on real cases)
- Hyperparameter sensitivity analysis
- Failure case analysis
- GitHub repository with reproducible code

---

## 🎯 Recommended Next Actions

### Immediate (Next 30 minutes):
1. **Review** `NOVELTY_FEATURES.md` for context
2. **Understand** the 4 core modules by reading docstrings
3. **Verify** all modules import correctly:
   ```bash
   cd Code
   python -c "from scripts.uc_cleaning import UCCleaner; from scripts.csp_features import MultimodalFeatureExtractor; from scripts.focal_loss import get_focal_loss; from scripts.evaluate_time_to_predict import TimeToPredict; print('✅ All modules import successfully')"
   ```

### Short-term (Next 2-4 hours):
1. **Implement** Phase 2: Modify `data_ingestion.py` with UC cleaning + CSP
2. **Implement** Phase 3: Enhance `model.py` with CSP branch
3. **Commit** changes to `feature/fusion-enhancements` branch

### Medium-term (Next 4-8 hours):
1. **Execute** Phase 4: Run training with Focal Loss
2. **Generate** Time-to-Predict evaluation plots
3. **Create** comparison study (Focal Loss vs BCE)

### Long-term (Next 1-2 days):
1. **Analyze** results and ablation study
2. **Validate** improvements match expectations (AUC > 0.85)
3. **Prepare** publication draft

---

## 📊 Success Metrics

### Minimum Targets (Conservative)
- [ ] AUC: > 0.85 (vs Paper 7's 0.84)
- [ ] Sensitivity: > 80% (vs Paper 7's 78%)
- [ ] Specificity: > 86% (vs Paper 7's 85%)

### Stretch Targets (Ambitious)
- [ ] AUC: > 0.87 (3-4% improvement)
- [ ] Sensitivity: > 82% (4-5% improvement)
- [ ] Specificity: > 87% (2% improvement)
- [ ] Time to 90% Sensitivity: < 45 minutes (early warning)

### Publication Targets
- [ ] 4+ novel contributions documented
- [ ] Ablation study showing each contribution's impact
- [ ] Clinical validation of Grad-CAM
- [ ] Open-source code release on GitHub

---

## 📝 Git Workflow

**Current Branch:** `feature/fusion-enhancements`  
**Commit Pattern:** `feat:`, `fix:`, `docs:`, `test:` prefixes

**For Phase 2:**
```bash
git commit -m "feat: Add UC cleaning and CSP feature extraction to data pipeline"
```

**For Phase 3:**
```bash
git commit -m "feat: Enhance model architecture with CSP fusion branch"
```

**For Phase 4:**
```bash
git commit -m "feat: Add Time-to-Predict evaluation and comparison studies"
```

**When ready to merge:**
```bash
git checkout main
git pull origin main
git checkout feature/fusion-enhancements
git merge main  # Resolve conflicts if any
git checkout main
git merge feature/fusion-enhancements
git push origin main
```

---

## ❓ Troubleshooting

### Issue: ModuleNotFoundError when importing new modules
**Solution:** Ensure you're in `Code` directory:
```bash
cd Code
python -c "from scripts.uc_cleaning import UCCleaner"
```

### Issue: Model input shape mismatch after CSP integration
**Solution:** Check X_csp_features shape is (n_samples, 13):
```python
print("X_csp_features shape:", X_csp_features.shape)  # Should be (2760, 13)
```

### Issue: Focal Loss not converging
**Solution:** Try reducing learning rate or adjusting γ (focusing parameter):
```python
FOCAL_LOSS_GAMMA = 1.5  # Instead of 2.0
```

---

## 🚀 Ready to Begin?

**All Phase 1 modules are ready to use. Phase 2-3 are straightforward modifications that you can implement with the provided code templates.**

**Next step: Implement Phase 2 (data pipeline enhancement) and Phase 3 (model architecture) to enable training with full novelty features.**

---

**Branch Status:** `feature/fusion-enhancements` (4 commits, ~1,500 lines added)  
**Ready for:** Phase 2 Implementation  
**Estimated Time to Completion:** 8-12 hours (all phases)

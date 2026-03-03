# NeuroFetal AI Project Context (V5.0 Calibrated Ensemble)
**Commit/Version:** `v5.0`
**Date:** March 2026

## 1. Project Overview
NeuroFetal AI is a State-of-the-Art (SOTA) Clinical Decision Support System designed to predict intrapartum fetal compromise. It fuses three data streams (Fetal Heart Rate [FHR], Uterine Contractions [UC], and Maternal Clinical Data) using a **Stacking Ensemble** of diverse models. V5.0 introduces **Platt Scaling Calibration** and **Information Theory Uncertainty (Entropy/Mutual Information)**, augmenting the previous V4.0 **TimeGAN** (WGAN-GP) synthetic generation.

## 2. Current Performance Status (V5.0)
Evaluated on the public **CTU-UHB dataset (552 records)** using Stratified 5-Fold Cross-Validation:
- **Stacking Ensemble Accuracy:** 96.34%
- **F1-Score:** 95.22%
- **AUC:** 0.8639
- **Brier Score:** 0.0460 (Highly calibrated forecast)
- **Expected Calibration Error (ECE):** 0.0543

## 3. Evaluation Against Baselines
We conducted a comprehensive literature review of 10+ papers and actively implemented/benchmarked several prior approaches:
- **1D-CNN (Spilka approach):** 0.564 AUC (Proves raw FHR alone fails without UC context).
- **Tabular Classical ML (Petrozziello approach - Random Forest):** 0.837 AUC (Fails to capture temporal trace dynamics).
- **Mendis et al. (Previous SOTA):** 0.84 AUC (Used private data, ignored the UC signal entirely).

## 4. Key Architectural Innovations
### A. Tri-Modal Data Fusion
The system extracts and fuses **35 total features**:
1. **FHR Signal (1200, 1)**: Resampled to 1Hz, windowed over 20-minutes.
2. **Tabular Context (16 features):** 3 demographic + 13 signal-derived statistical features (LTV, STV, etc.).
3. **Common Spatial Patterns (CSP, 19 features):** Extracts spatial variance filters exclusively from the FHR-UC interaction.

### B. Stacking Ensemble
Combines three distinct architectures via a **Logistic Regression Meta-Learner** (with Rank Averaging):
1. **AttentionFusionResNet (Deep Branch):** 1D ResNet + SE blocks + **Cross-Modal Attention Fusion (CMAF)** to logically gate signal vectors based on clinical tabular risk profiles.
2. **1D-InceptionNet:** Captures multi-scale temporal patterns (kernels 3, 5, 7).
3. **XGBoost:** Gradient-boosted trees performing on Tabular + CSP + FHR tabular features.

### C. TimeGAN Data Augmentation
- Mitigates the extreme 7.25% class imbalance.
- **WGAN-GP (1D Transposed Convolutions)** trained exclusively on pathological FHR+UC traces.
- Generates **1,410 physiologically realistic synthetic minority-class traces**, preserving vital temporal delays (e.g., late decelerations).

### D. Uncertainty & Calibration
- **Monte Carlo (MC) Dropout:** 20 forward passes with $p=0.3$ active at inference to calculate Epistemic Uncertainty.
- **Platt Scaling (`CalibratedClassifierCV`):** Shifts raw model logits into trustworthy clinical probabilities.
- Explicit "Ambiguous Zone: REQUIRES HUMAN REVIEW" triggering for high-variance sweeps.

### E. Edge Deployment & Explainability
- **TFLite Int8 Quantization:** Massive 27MB ensemble strictly compressed to a **1.9 MB** edge model, executing in <30ms on $60 commodity Android hardware without cellular dependence.
- **Grad-CAM (Explainable AI):** Generates physical color displacement heatmaps highlighting the exact window that triggered distress alerts within the Streamlit dashboard UI.

## 5. Repository Structure
- **`Code/scripts/`**: `train.py`, `train_diverse_ensemble.py`, `evaluate_ensemble.py`, `evaluate_uncertainty.py`, `convert_to_tflite.py`, `app.py` (Streamlit).
- **`Code/utils/`**: Custom layers (`attention_blocks.py`), Model definitions, `csp_features.py`, `focal_loss.py`.
- **`Code/models/`**: `.keras` folds, `.pkl` meta-learners, `tflite/` edge models.
- **`Datasets/`**: CTU-UHB raw `.dat`/`.hea`, processed `.npy`, and `synthetic/` TimeGAN traces.
- **`Reports/`**: Weekly reports, Mid-sem report (`Content/report_content.md`), presentation (`Content/PPT_content.md`), and final LaTeX `Paper/`.

## 6. End-Semester Roadmap
- Full Sub-System Integration of TimeGAN limits into Stratified 5-Fold grid sweeps.
- Finalize execution and compilation parameters for SOTA evaluations against baseline tests.
- Boot the finalized Streamlit clinical dashboard integrating the edge-executed `.tflite` bundles and robust MC Dropout confidence tracking.

# NeuroFetal AI Project Context (as of Release v3.0)
**Commit:** `v3.0`
**Date:** Feb 13, 2026

## 1. Project Overview
NeuroFetal AI is a clinical decision support system utilizing deep learning to predict fetal compromise during labor. It fuses three data streams (FHR, Contractions, Clinical Data) through a **Stacking Ensemble** of diverse models to achieve State-of-the-Art (SOTA) performance with clinical uncertainty quantification.

## 2. Current Performance Status
- **AUC Score:** 0.87 (Public Data Only — CTU-UHB, 552 records)
- **Baseline Comparison:** Exceeds Mendis et al. (0.84, which used 10k+ private samples). Previous project AUC was 0.74.
- **Ensemble Method:** Stacking Meta-Learner over 3 diverse base models (AttentionFusionResNet, 1D-InceptionNet, XGBoost) with Rank Averaging calibration.

## 3. Key Architecture Features
### A. Triple-Modal Fusion (Stacking Ensemble)
The system ensembles three architecturally diverse models:
1.  **AttentionFusionResNet:** FHR (1D ResNet + SE Attention) + 16 Tabular features + 19 CSP features.
2.  **1D-InceptionNet:** Multi-scale temporal convolutions on FHR + Tabular + CSP.
3.  **XGBoost:** Gradient-boosted trees on Tabular + CSP + FHR statistical features.
A **Logistic Regression Meta-Learner** (stacking) combines the base model OOF predictions.

### B. Feature Engineering
- **Tabular (16 features):** 3 demographic (age, parity, gestation) + 13 signal-derived (baseline FHR, STV, LTV, accelerations, decelerations, entropy, etc.).
- **CSP (19 features):** Common Spatial Patterns extracted from FHR + UC signals for spatial filtering.

### C. Uncertainty Quantification
- Implements **Monte Carlo Dropout** (20 forward passes).
- Provides a **Confidence Score** alongside prediction probability.
- High variance flags cases for human review.

### D. Edge Deployment
- **TFLite Model:** Quantized to **Int8** for mobile/embedded inference.
- **Offline Capability:** Runs on low-end Android devices without internet (<30ms inference).

## 4. Repository Structure
- **`Code/scripts/train.py`**: Main training pipeline with SMOTE, Focal Loss, and Stratified Group K-Fold.
- **`Code/scripts/train_diverse_ensemble.py`**: Diverse ensemble training (InceptionNet, XGBoost, Stacking).
- **`Code/notebooks/Training_Colab.ipynb`**: Primary experiment notebook.
- **`Code/models/`**: Contains `enhanced_model_fold_*.keras`, `inception_model_fold_*.keras`, `xgboost_model_fold_*.pkl`, `stacking_meta_learner.pkl`, and TFLite models.
- **`Datasets/`**: Contains processed `.npy` files (`X_fhr`, `X_tab`, `y`, `X_uc`, `X_csp`).
- **`Reports/final_report.md`**: Detailed analysis and results.

## 5. Next Steps & Future Roadmap
A rigorous, regressive plan has been established to evolve NeuroFetal AI from a prototype to a globally deployed Clinical Decision Support System. 

For the complete multi-year plan (Levels 1 through 5), see: **[`Future_Roadmap_NeuroFetal_AI.md`](Future_Roadmap_NeuroFetal_AI.md)**

Immediate technical priorities (Level 1) include:
1.  **External Validation:** Test zero-shot generalization on a different dataset to measure domain shift.
2.  **Advanced Generative Augmentation:** Develop a Time-Series GAN to replace SMOTE for class imbalance.
3.  **XAI Upgrade:** Implement SHAP or Integrated Gradients for better temporal feature attribution.
4.  **Streaming Architecture:** Begin transitioning from static window analysis to a continuous 1Hz streaming API.

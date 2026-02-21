# NeuroFetal AI Project Context (as of Release v4.0)
**Commit:** `v4.0`
**Date:** Feb 21, 2026

## 1. Project Overview
NeuroFetal AI is a clinical decision support system utilizing deep learning to predict fetal compromise during labor. It fuses three data streams (FHR, Contractions, Clinical Data) through a **Stacking Ensemble** of diverse models. V4.0 introduces **TimeGAN augmentation** — replacing SMOTE with a WGAN-GP time-series GAN for class imbalance.

## 2. Current Performance Status (V4.0)
- **Stacking Ensemble AUC:** 0.8639 (TimeGAN Augmentation, Public Data Only — CTU-UHB, 552 records)
- **Primary Model AUC:** 0.7910 ± 0.0322 (AttentionFusionResNet, 5-fold)
- **XGBoost AUC:** 0.8512 (strongest single model)
- **InceptionNet AUC:** 0.7886
- **Low-Uncertainty AUC:** 0.8471 (well-calibrated confidence)
- **V3.0 Baseline (SMOTE):** 0.87 AUC
- **Ensemble Method:** Stacking Meta-Learner over 3 diverse base models with Rank Averaging calibration.

## 3. V4.0 Changes (TimeGAN Augmentation)
### A. TimeGAN Architecture
- **Type:** WGAN-GP with 1D Transposed Convolutions
- **Training Data:** Pathological FHR+UC traces (~470 samples), stacked as `(N, 1200, 2)`
- **Training:** 500 epochs, gradient penalty λ=10
- **Output:** 1,410 synthetic traces (3x minority class) → `X_fhr_synthetic.npy`, `X_uc_synthetic.npy`
- **Notebook:** `Code/notebooks/TimeGAN_Colab.ipynb`

### B. Pipeline Integration
- `train.py` now accepts `--augmentation [timegan|smote|none]` flag (default: `timegan`)
- `train_diverse_ensemble.py` integrated with TimeGAN and CSP feature alignment fix
- `Training_Colab.ipynb` updated to V4.0 branding with `--augmentation timegan --epochs 150`

## 4. Key Architecture Features
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
- Low-uncertainty predictions: AUC 0.8471. High-uncertainty: AUC 0.6802.
- High variance flags cases for human review.

### D. Edge Deployment
- **TFLite Model:** Quantized to **Int8** for mobile/embedded inference.
- **Offline Capability:** Runs on low-end Android devices without internet (<30ms inference).

## 5. Repository Structure
- **`Code/scripts/train.py`**: Main training pipeline with TimeGAN/SMOTE augmentation, Focal Loss, and Stratified K-Fold.
- **`Code/scripts/train_diverse_ensemble.py`**: Diverse ensemble training (InceptionNet, XGBoost, Stacking).
- **`Code/notebooks/Training_Colab.ipynb`**: Primary experiment notebook (V4.0 TimeGAN).
- **`Code/notebooks/TimeGAN_Colab.ipynb`**: TimeGAN training notebook.
- **`Code/models/`**: Contains `enhanced_model_fold_*.keras`, `inception_model_fold_*.keras`, `xgboost_model_fold_*.pkl`, `stacking_meta_learner.pkl`, and TFLite models.
- **`Datasets/`**: Contains processed `.npy` files and `synthetic/` directory with TimeGAN-generated traces.
- **`Reports/final_report.md`**: Detailed analysis and results.

## 6. Next Steps & Future Roadmap
Immediate technical priorities include:
1.  **External Validation:** Test zero-shot generalization on a different dataset to measure domain shift.
2.  **Hyperparameter Tuning:** Optimize TimeGAN training (epochs, architecture, noise distribution) to improve primary model AUC.
3.  **XAI Upgrade:** Implement SHAP or Integrated Gradients for better temporal feature attribution.
4.  **Streaming Architecture:** Begin transitioning from static window analysis to a continuous 1Hz streaming API.

For the complete multi-year plan, see: **[`Future_Roadmap_NeuroFetal_AI.md`](Future_Roadmap_NeuroFetal_AI.md)**

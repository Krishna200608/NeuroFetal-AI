# NeuroFetal AI Project Context (as of Release v2.0)
**Commit:** `Phase_2_Complete`
**Date:** Feb 10, 2026

## 1. Project Overview
NeuroFetal AI is a clinical decision support system utilizing deep learning to predict fetal compromise during labor. It fuses three data streams (FHR, Contractions, Clinical Data) to achieve State-of-the-Art (SOTA) performance with clinical uncertainty quantification.

## 2. Current Performance Status
- **AUC Score:** 0.78 (Public Data)
- **Baseline Comparison:** Outperforms previous state (0.74) and is comparable to Mendis et al. (0.84 on larger private data).

## 3. Key Architecture Features
### A. Triple-Modal Fusion
The model (`AttentionFusionResNet`) integrates:
1.  **Fetal Heart Rate (FHR):** Processed via Common Spatial Patterns (CSP) and a 6-Block ResNet.
2.  **Uterine Contractions (UC):** Analyzed for stress response patterns.
3.  **Clinical Data:** Maternal age, parity, and gestation processed via a Dense network.

### B. Uncertainty Quantification
- Implements **Monte Carlo Dropout** (20 passes).
- Provides a **Confidence Score** alongside prediction probability.
- High variance flags cases for human review.

### C. Edge Deployment
- **TFLite Model:** Quantized to **2.6 MB** (Int8).
- **Offline Capability:** Runs on low-end Android devices without internet (<30ms inference).

## 4. Repository Structure
- **`Code/scripts/train.py`**: Main training pipeline with SMOTE, Focal Loss, and Stratified Group K-Fold.
- **`Code/notebooks/Training_Colab.ipynb`**: Primary experiment notebook.
- **`Datasets/`**: Contains processed `.npy` files (`X_fhr`, `X_tab`, `y`, `X_uc`).
- **`Reports/final_report.md`**: Detailed analysis and results.

## 5. Next Steps
- **Clinical Validation**: Run the quantized model on a larger retrospective dataset.
- **Hardware Benchmarking**: Test on Coral Edge TPU.
- **Publication**: Prepare manuscript for IEEE EMBC.


# Final Project Report: NeuroFetal AI

**Date:** March 13, 2026

**Status:** Completed & Validated (V6.0 Leak-Free Ensemble — Per-Fold TimeGAN)

**Final Metrics:** Ensemble AUC 0.8566, Base Model AUC 0.7640 ± 0.0619, Brier Score 0.046, ECE 0.0543 (Stacking Ensemble with Per-Fold TimeGAN Augmentation & Platt Scaling, Public Data Only, Strict No-Leakage)

---

## 1. Executive Summary
**NeuroFetal AI** has successfully evolved from a basic replication study into a **State-of-the-Art (SOTA)** Clinical Decision Support System. The project progressed through multiple phases: a Diverse Stacking Ensemble of three architecturally distinct models (AttentionFusionResNet, 1D-InceptionNet, XGBoost) that initially achieved AUC 0.87 using SMOTE, the **V4.0 release** replaced SMOTE with **TimeGAN-based data augmentation** — generating synthetic pathological FHR+UC traces using a WGAN-GP architecture.

Critically, the **V6.0 release** identified and corrected a **data leakage flaw** in the previous TimeGAN integration: the GAN had been trained globally on all pathological samples, allowing validation data to influence synthetic generation. The fix moves TimeGAN training **inside** each CV fold, training a fresh WGAN-GP exclusively on the current fold's training-set pathological samples (1500 epochs per fold). Combined with **Platt Scaling Calibration** and **Information Theory Uncertainty Metrics**, the final V6.0 ensemble achieves a defensible **AUC of 0.8566** and a base model **AUC of 0.7640 ± 0.0619** — both mathematically rigorous and suitable for peer-reviewed publication.

The system is not just a predictor but a **highly trustworthy clinical assistant**, directly presenting clinicians with reliable calibrated probabilities and explicitly flagging Epistemic Uncertainty. This is combined with **Grad-CAM Explainability** for transparent decision-making, and an enhanced **Edge Deployment** UI for low-resource clinical settings.

---

## 2. Methodology Evolution

### A. Phase 1-2: Single-Model Tri-Modal Fusion (AUC 0.74)
1.  **Input 1: Tabular Clinical Data**: 3 demographic features (Age, Parity, Gestation) via Dense Network.
2.  **Input 2: Time-Series (FHR + UC)**: 6-Block Residual Network with Bottleneck Layers and SE Attention.
3.  **Advanced Feature Extraction**: Common Spatial Patterns (CSP) from FHR/UC signals.
4.  **Cross-Modal Attention Fusion**: Clinical data queries signal embeddings.

### B. Phase 3-6: Diverse Stacking Ensemble (AUC 0.87)
Recognizing the ceiling of a single model, we introduced architectural diversity:
1.  **Model A — AttentionFusionResNet**: FHR signal encoder + 16 Tabular features + 19 CSP features via Cross-Modal Attention. (Enhanced from Phase 2 with 13 additional signal-derived tabular features.)
2.  **Model B — 1D-InceptionNet**: Multi-scale temporal convolutions (kernel sizes 3/5/7) + Tabular + CSP.
3.  **Model C — XGBoost**: Gradient-boosted trees on hand-crafted Tabular + CSP + FHR statistical features.
4.  **Stacking Meta-Learner**: A Logistic Regression model trained on out-of-fold (OOF) predictions from the 3 base models, with Rank Averaging calibration for robust probability estimates.

### C. Addressing Imbalance (The 7% Problem)
The dataset is heavily imbalanced (only 7.25% pathological cases). We addressed this with an evolving strategy:

**V3.0 (SMOTE):**
1.  **SMOTE**: Synthetic Minority Over-sampling applied to the fused feature space.
2.  **Focal Loss**: ($\gamma=2.5, \alpha=0.75$) forces the model to focus on "hard" examples.
3.  **Rank Averaging**: Normalizes prediction ranks across folds for robust global calibration.

**V6.0 (Per-Fold TimeGAN + Calibrated Ensemble — Current):**
1.  **Per-Fold TimeGAN Augmentation**: WGAN-GP time-series GAN trained **inside each CV fold** exclusively on that fold's training-set pathological traces (1500 epochs, batch size 64). Generates synthetic samples dynamically to balance each fold's class distribution. This eliminates the data leakage present in V4.0 where the GAN was trained globally.
2.  **Model Calibration**: Platt Scaling via `CalibratedClassifierCV` maps the continuous output score directly to reliable clinical probabilities, scoring highly on Brier (0.046) and ECE (0.054).
3.  **Uncertainty Quantification**: Distinguishes between Epistemic and Aleatoric uncertainty using Mutual Information and Predictive Entropy derived from MC Dropout passes.

### D. Feature Engineering (The 16+19 Upgrade)
- **Tabular Features (16)**: Expanded from 3 demographic-only features to 16 (3 demographic + 13 signal-derived: baseline FHR, STV, LTV, acceleration/deceleration counts, entropy, etc.).
- **CSP Features (19)**: Common Spatial Patterns extracted from multi-channel FHR+UC for spatial variance filtering — a novel application from BCI/EEG domain to fetal monitoring.

### E. Data Augmentation (The 5x Multiplier)
To overcome the small dataset size (552 raw recordings), we implemented a **Overlap-Windowing Strategy (20-min window, 10-min stride)**.
*   **Result**: This technique effectively multiplied our training data by ~5x (**552 Recordings → ~2,760 Training Samples**).

---

## 3. Quantitative Results

### Benchmarking against SOTA
| Model Approach | Metrics | Augmentation | Calibration | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Mendis et al.)** | 0.84 AUC (10k+ Private) | N/A | None | Previous SOTA |
| **Mendis Reproduced (Fair)** | **0.7983 AUC (same 552 data)** | None | None | Fair Baseline |
| Our Phase 1 (Basic Fusion) | 0.74 AUC | None | None | Surpassed |
| Our V3.0 (SMOTE Ensemble) | 0.87 AUC | SMOTE | None | Previous Best |
| Our V4.0 (TimeGAN Ensemble)| 0.8639 AUC | Global TimeGAN | None | **Had Data Leakage** |
| **Our V6.0 (Per-Fold TimeGAN)**| **Ensemble AUC 0.8566** | **Per-Fold TimeGAN** | **Platt Scaling (Brier: 0.046)** | **Current SOTA (Leak-Free)** |

### Robustness, Calibration & Uncertainty (V6.0)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|---|---|---|---|---|---|---|
| AttentionFusionResNet | 0.8089 | 0.7268 | 0.8256 | 0.6602 | 0.7983 | **0.7640 ± 0.0619** |
| InceptionNet | 0.7707 | 0.7707 | 0.8264 | 0.7872 | 0.8240 | **0.7958 ± 0.0263** |
| XGBoost | 0.8354 | 0.8701 | 0.8788 | 0.8388 | 0.8328 | **0.8512 ± 0.0210** |
| **Stacking Ensemble** | — | — | — | — | — | **0.8566** |

*   **Brier Score**: 0.0460 (highly reliable forecast)
*   **Expected Calibration Error (ECE)**: 0.0543
*   **Epistemic vs Aleatoric Uncertainty**: Mutual Information effectively identifies out-of-distribution or noisy signals, prompting clinical review automatically.

---

## 4. Key Novelties Delivered

1.  **Diverse Stacking Ensemble for CTG**: Combined deep learning (ResNet, InceptionNet) with gradient boosting (XGBoost) under a meta-learner — a first for public-dataset fetal monitoring.
2.  **Per-Fold TimeGAN for Fetal Monitoring (V6.0)**: Replaced global TimeGAN with a per-fold WGAN-GP that trains inside each cross-validation fold exclusively on that fold's pathological samples — ensuring mathematically strict no-leakage evaluation.
3.  **CSP for Fetal Monitoring**: One of the first applications of Common Spatial Patterns for single-channel FHR analysis, treating temporal variance as a spatial feature.
4.  **Fair Baseline Reproduction**: Reproduced the Mendis et al. architecture on identical data (AUC 0.7983), confirming their reported 0.84 relied on their private 10k-sample dataset.
5.  **Rank-Normalized Ensembling**: Proved that for medical datasets with varying fold calibrations, Rank Averaging is superior to probability averaging.
6.  **Uncertainty-Aware Dashboard**: The system says "Pathological (High Confidence)" or "Pathological (Low Confidence)" alongside Grad-CAM explanations — emulating a consultative second opinion.
7.  **Edge-Ready Deployment**: TFLite Int8 quantization enables real-time inference on a low-cost smartphone, ensuring accessibility in low-resource settings.

---

## 5. Phase 2: Uncertainty & Edge Optimization

### A. Uncertainty Quantification
1.  **Calibration Curves**: Model's predicted probabilities align with observed empirical accuracy.
2.  **Uncertainty Histograms**: Distribution of prediction confidence highlights "grey zone" cases for clinician review.

### B. TFLite Int8 Quantization
*   **Method**: Full Integer Quantization with representative dataset calibration.
*   **Result**:
    *   **Size**: Int8 quantized for edge deployment.
    *   **Accuracy**: Retained ~99% of original AUC.
    *   **Inference Speed**: <30ms on standard mobile CPU.

### C. Dashboard v4.0
*   **3-Input Model Support**: Loads `enhanced_model_fold_*.keras` with real-time feature extraction (16 tabular + 19 CSP).
*   **Integrated XAI**: Grad-CAM heatmaps with 3-input support for signal pattern explanation.
*   **Real-Time Reliability**: MC Dropout uncertainty metrics displayed alongside every prediction.
*   **Medical-Grade UI**: Dark mode with Material Design icons for low-light labor ward usability.

---

## 6. Conclusion

The **NeuroFetal AI** project has evolved to address both performance and scientific rigor. The V6.0 release identified and corrected a **data leakage flaw** in the TimeGAN augmentation pipeline — now training the WGAN-GP inside each cross-validation fold to ensure strict methodological correctness. The resulting leak-free **ensemble AUC of 0.8566** still significantly outperforms the reproduced Mendis baseline (0.7983 on identical data), confirming the genuine value of our Tri-Modal architecture, CSP features, and TimeGAN augmentation. Combined with Platt Scaling calibration (Brier: 0.046), reliable epistemic uncertainty bounds, and edge deployment (1.9 MB TFLite model), the system delivers a highly trustworthy, interpretable, and deployable clinical decision support system for intrapartum fetal monitoring.

**Final Verdict**: V6.0 objectives achieved — Per-fold TimeGAN integration, leak-free evaluation, fair baseline reproduction, and rigorous Uncertainty Quantification formally integrated.

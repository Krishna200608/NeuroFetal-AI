# Research Proposal: NeuroFetal AI: Intelligent Intrapartum Monitoring System

**Proposed Project:** NeuroFetal AI  
**Date:** 1 February 2026

---

## 1. Literature Review & Problem Identification

We have analyzed 7 state-of-the-art papers in the field of Fetal Heart Rate (FHR) analysis to determine the optimal architecture for our proposed system. Our analysis reveals two critical gaps in current methodologies:

1.  **The "Context Gap"**: Existing Deep Learning models (Table 1) rely solely on the FHR signal, ignoring maternal clinical context (Age, Parity, Gestation) that serves as a critical baseline for obstetricians.
2.  **The "Interpretability Gap"**: High-performing models (like Transformers) act as "black boxes," failing to provide the visual justification required for clinical trust.

Based on this evaluation, we propose a novel **Multi-Modal Fusion Architecture** that builds directly upon the concepts introduced in **Paper 7** (Spilka et al., Fusing Tabular Features...), but with significant enhancements to address its limitations.

---

## 2. Proposed Methodology

### A. Feature Selection Strategy
For the tabular component of our proposed fusion model, we intend to select only **Maternal Age**, **Parity**, and **Gestation**. This selection is justified by:

1.  **Clinical Relevance**: These three factors constitute the baseline "Risk Triad" that alters how a physician interprets a heart rate trace.
    *   *Example*: A predefined deceleration pattern may be tolerable in a term fetus but critical in a preterm one.
2.  **Unavailability of Other Data**: Other potential features (e.g., pH, Lactate) are retrospective and not available during the critical intrapartum monitoring window.
3.  **Prevention of Overfitting**: Given the dataset size (CTU-UHB, ~552 samples), introducing high-dimensional sparse data would likely cause model variance.

### B. Proposed Architecture (Fusion ResNet)
We propose to design a dual-branch neural network:
*   **Branch 1 (Signal)**: A **1D Residual Network (ResNet)** to capture temporal dependencies in the FHR signal. This improves upon the simpler C-Net used in Paper 7 by addressing the vanishing gradient problem in deeper layers.
*   **Branch 2 (Tabular)**: A **Dense Network (DenseNet)** to process the normalized clinical features.
*   **Fusion Layer**: These branches will limit concatenating feature vectors before the final classification head.

---

## 3. Comparative Gap Analysis

The table below summarizes our evaluation of current SOTA papers and highlights where our proposed **NeuroFetal-AI** system aims to contribute.

| Feature | **Paper 1: Rapid Detection** <br> (Input Length Invariant) | **Paper 2: Foundation Model** <br> (Stress Prediction) | **Paper 3: Cross-Database** <br> (Deep Learning Eval) | **Paper 4: DeepCTG 1.0** <br> (Interpretable Model) | **Paper 5: Both Stages** <br> (Classification) | **Paper 6: Spatial Pattern** <br> (Freq & CSP) | **Paper 7 (BASE): Fusion** <br> (Tabular + DL) | **PROPOSED SYSTEM** <br> (NeuroFetal-AI) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Focus** | Length Invariant CNN | Large-Scale Pretraining | Generalizability Test | Hypoxia Detection | Labor Stage 1 & 2 | Signal Frequency Analysis | **Signal + Clinical Fusion** | **Optimized Fusion + Edge XAI** |
| **Input Type** | Raw Signal Only | Raw Signal Only | Raw Signal Only | Raw Signal Only | Raw Signal Only | Frequency/Spatial Features | **Signal + Tabular** | **Signal + Tabular (Refined)** |
| **Preprocessing** | Length Normalization | Heavy Pre-training | Standard Integration | Linear Interpolation | Standard Filtering | CSP Extraction | Linear Interp + Imputation | **Proposed: Linear Interp + MinMax** |
| **Sampling Rate** | 4 Hz | 4 Hz | 4 Hz | 4 Hz | 4 Hz | N/A (Feature Based) | 4 Hz | **Target: 1 Hz (Edge Efficiency)** |
| **Augmentation** | Window Slicing | Masking (BERT-style) | None Reported | None Reported | SMOTE | Noise Injection | SMOTE (Tabular) | **Proposed: Cost-Sensitive Learning** |
| **Model Arch** | Custom 1D-CNN | Transformer (PatchTST) | Various CNNs | TCN (Temporal Conv) | 2D-CNN | SVM / CSP-Linear | **Fusion C-Net** | **Proposed: Fusion ResNet-1D** |
| **Loss Function** | Binary Cross Entropy | MSE -> BCE | Weighted BCE | Weighted BCE | Cross Entropy | Hinge Loss | Binary Cross Entropy | **Weighted Binary Cross Entropy** |
| **Explainability** | None | Attention Maps (Complex) | None | Filters (Limited) | None | Feature Weights | Feature Importance | **Proposed: Grad-CAM + SHAP** |
| **Deployment** | Research Only | Cloud Only (Heavy) | Research Only | Research Only | Research Only | Research Only | Research Code | **Goal: TFLite (Mobile Ready)** |
| **Context Aware** | No | No | No | No | No | No | **Yes** | **Yes (Enhanced Logic)** |

---

## 4. Expected Contributions

We aim for this project to demonstrate superiority in three key areas:

1.  **Architecture Upgrade**: By replacing the C-Net in Paper 7 with a **1D-ResNet**, we anticipate better feature extraction from noisy signal data.
2.  **Explainability Integration**: We plan to implement **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Unlike Paper 7, which effectively only explains the tabular importance, Grad-CAM will allow us to visualize which specific parts of the FHR signal (e.g., late decelerations) the model is focusing on.
3.  **Feasibility for Low-Resource Settings**: Unlike large Foundation models (Paper 2) or Image-based models (Paper 5) requiring GPUs, we aim to quantize our model for **Edge Deployment**, making it suitable for portable monitoring devices.

---

# Paper 7 Analysis: Fusing Tabular Features and Deep Learning (Fusion ResNet)

**Paper:** Fusing Tabular Features and Deep Learning for Fetal Heart Rate Analysis: A Clinically Interpretable Model for Fetal Compromise Detection (Mendis et al., 2023)
**Reviewer Role:** Senior Research Reviewer & ML Systems Architect
**Context:** Comparison vs. NeuroFetal AI (Existing Deployed System)

---

## 1. One-Paragraph Summary
This study presents **Fusion ResNet**, a novel multimodal architecture that fuses **raw FHR time series** (1D-ResNet branch) with **tabular clinical features** (Dense branch) to predict fetal compromise. It introduces a massive new private dataset (**MHW-pH**, 9,887 recordings) and validates on the public **CTU-UHB** dataset. The model achieves **SOTA performance (AUC 0.84)** on CTU-UHB, outperforming purely signal-based models (like their own work in Paper 3) and purely tabular models (SVM). Crucially, it integrates **SHAP** and **Grad-CAM** to provide clinical interpretability, showing that the model correctly attends to "late decelerations" and weighs clinical risks like "Nulliparity" and "Meconium" appropriately.

## 2. Key Contributions
*   **Multimodal Fusion Architecture:** Effectively combined a 1D-ResNet (for FHR signals) with a Dense Network (for Tabular Clinical Data) using late fusion.
*   **Huge Private Dataset:** Introduced MHW-pH (9,887 cases), one of the largest labeled CTG datasets in existence, used for robust pre-training.
*   **State-of-the-Art Results:** Achieved **AUC 0.84** on the public CTU-UHB dataset, beating the previous benchmark (MCNN, AUC 0.81).
*   **Explainability (XAI):** Applied **SHAP** (for tabular feature importance) and **Grad-CAM** (for signal localization) to validate that the model learns physiologically relevant patterns.

## 3. Dataset & Preprocessing
*   **Dataset 1 (Private):** MHW-pH (9,887 cases, 1.1% compromise).
*   **Dataset 2 (Public):** CTU-UHB (552 cases).
*   **Input:** Last 60 minutes of FHR signal (at **1Hz**).
*   **Tabular Features (12 extracted, top 5 used):**
    1.  **Parity** (Nulliparous vs Multiparous)
    2.  **Maternal Age**
    3.  **Gestation** (Weeks)
    4.  **Beta_0** (Baseline Intercept)
    5.  **MAD_dtrd** (Median Absolute Deviation of detrended FHR)
*   **Preprocessing:**
    *   **Imputation:** Mean imputation for tabular data.
    *   **Signal:** Artifact removal, interpolation (<15s gaps), downsampled to **1Hz**.
    *   **Normalization:** MinMax scaler for signals, Z-score for tabular.

## 4. Model Architecture (Fusion ResNet)
*   **Branch 1 (Signal):** 1D-ResNet (3 Residual Blocks) -> GAP -> Feature Vector (128d).
    *   *Note:* This is the same winning architecture from Paper 3.
*   **Branch 2 (Tabular):** 2 FC Layers (10 -> 128 neurons) -> ReLU/Sigmoid.
*   **Fusion:** **Element-wise Multiplication** was found to be the superior fusion operator (vs Addition or Concatenation).
*   **Head:** Final Sigmoid classification layer.

## 5. Results & Evaluation
*   **Internal (MHW-pH):** AUC 0.77 (Fusion) vs 0.73 (Signal Only). *Fusion adds +4% AUC.*
*   **External (CTU-UHB):** **AUC 0.84** (Fusion) vs 0.80 (Signal Only). *Fusion adds +4% AUC.*
*   **Benchmarking:**
    *   Beat MCNN (AUC 0.81).
    *   Beat Sparse SVM on tabular data (AUC 0.79).
    *   Beat CNN (Paper 1 & 2 architectures).

## 6. Strengths
*   **Proof of Concept for NeuroFetal:** This paper is the **scientific validation** of your exact architecture (ResNet + Clinical Data). It proves that adding clinical data *works* and improves performance by significantly handling cases where the signal is ambiguous.
*   **Fusion Strategy:** Systematic evaluation of fusion operators (Mul > Add > Concat) provides a concrete design guideline.
*   **Interpretability:** The SHAP plots (Fig 7) are brilliant. They show visually how "High Maternal Age" + "Low Variability" = "High Risk".

## 7. Weaknesses & Gaps
*   **UC Signal Ignored:** Like Paper 3, they explicitly discarded Uterine Comtractions due to poor quality. NeuroFetal could potentially innovate here if you have better quality control or cleaning (Paper 2).
*   **Hand-Crafted Feature Reliance:** The "Tabular" branch isn't just raw EHR data; it includes 2 mathematically derived signal features (beta_0 and MAD). This blurs the line between "Clinical Data" and "Signal Features".

## 8. Direct Comparison: Paper 7 vs NeuroFetal AI

| Feature | Paper 7 (Fusion ResNet) | NeuroFetal AI |
| :--- | :--- | :--- |
| **Architecture** | **1D-ResNet** + Tabular MLP | **1D-ResNet** + DenseNet |
| **Fusion Type** | **Multiplication** (Late) | Concatenation (Likely?) |
| **Input Signal** | FHR Only (1Hz) | FHR + UC (4Hz?) |
| **Tabular Input** | 5 Features (inc. derived) | Clinical Data (Raw?) |
| **Performance** | **AUC 0.84 (SOTA)** | To be determined |
| **Validation** | Cross-Database | Single Database |
| **XAI** | **SHAP + Grad-CAM** | Grad-CAM (Signal) |

## 9. Concrete Improvements for NeuroFetal
*   **Switch Fusion Operator:**
    *   *Insight:* They proved **Element-wise Multiplication** beats Concatenation for this specific task.
    *   *Action:* Change your fusion layer from `torch.cat([x, y])` to `x * y`.
*   **Add Derived Features:**
    *   *Insight:* Their "Tabular" branch wasn't just age/parity. It included **MAD** (Variability) and **Beta_0** (Baseline).
    *   *Action:* Calculate these 2 scalar values from your signal and append them to your clinical tabular input vector. It gives the tabular branch "signal awareness".
*   **Adopt SHAP:**
    *   *Insight:* Grad-CAM only explains the *signal*. SHAP explains the *clinical risk factors*.
    *   *Action:* Implement SHAP for your Tabular branch. It answers "Why did the model predict High Risk? -> Because Age > 35 and Parity = 0".

## 10. Proposed Ablation Experiments
*   **Exp:** `Fusion Operator: Mult vs Concat`.
    *   *Hypothesis:* Multiplication forces interaction between signal features and clinical context (e.g. "Low variability is only bad if Age > 40").
*   **Exp:** `Signal Only vs Fusion`.
    *   *Hypothesis:* Fusion should yield higher AUC (approx +0.04 gain predicted by Paper 7).

## 11. Defense Prep: 6 Likely Viva Questions
1.  **Q:** "Your project looks very similar to Mendis et al. (Paper 7). What is novel?"
    *   *A:* "While Mendis et al. established Fusion ResNet, NeuroFetal AI extends this by [1] Incorporating Uterine Contractions (ignored by them), [2] Deploying to Edge/RPi, and [3] Integrating a real-time alerting interface for clinicians."
2.  **Q:** "Why did you use Late Fusion?"
    *   *A:* "Paper 7 demonstrated that processing heterogeneous modalities (time-series vs static tabular) in separate streams allows each branch to learn optimal representations before fusing, preventing the dominant modality (signal) from washing out subtle clinical clues."
3.  **Q:** "How do you explain your model's decisions?"
    *   *A:* "We adopted the Dual-XAI framework validated in Paper 7: Grad-CAM for signal localization (showing decelerations) and SHAP for clinical feature importance (showing risk factors like Age)."
4.  **Q:** "Why element-wise multiplication?"
    *   *A:* "Multiplication acts as an attention mechanism. The clinical features effectively 'gate' or amplify different parts of the signal feature vector, allowing the model to be more sensitive to signals in high-risk patients."

## 12. Final Recommendation
**This is your Holy Grail.**
Paper 7 is the **strongest scientific validation** for your project structure.
**Cite it heavily.** Use it to justify: 1D-ResNet, 1Hz sampling, Late Fusion, and omitting UC (or treat UC as your "Novel Addition").
**Copy their Fusion:** Switch to element-wise multiplication if you haven't already.
**Adopt their XAI:** Adding SHAP for the tabular part will make your dashboard look incredible ("AI Risk Factors: Age, Nulliparity").

---

## TL;DR for Paper 7
*   **What they did:** Created **Fusion ResNet**: A state-of-the-art model fusing **1Hz FHR signals** (ResNet) with **Clinical Data** (Age, Parity, etc.).
*   **Key Win:** Achieved **AUC 0.84** (highest reliable score on CTU-UHB), proving that **Clinical Data + Signal > Signal Alone**.
*   **For NeuroFetal:** This is the blueprint. **Validate** your architecture against this. **Adopt** their fusion method (Multiplication). **Use (or exceed)** their XAI approach (SHAP + Grad-CAM). This paper justifies your entire project's existence.

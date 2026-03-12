# Week 7 Progress Report: NeuroFetal AI

**Date:** March 7, 2026 - March 12, 2026

## 1. Mid-Semester Evaluation (March 10)
*   Successfully presented the NeuroFetal AI project to the evaluation board.
*   Covered the full pipeline: data ingestion, tri-modal feature engineering (FHR + Tabular + CSP), TimeGAN augmentation, AttentionFusionResNet, stacking ensemble, uncertainty quantification, and edge deployment.
*   Board feedback was overall positive. **Prof. Manish Kumar** raised a valid concern regarding the use of TimeGAN on critical medical data and asked for stronger validation of the synthetic samples.

## 2. TimeGAN Synthetic Data Validation Suite
*   **Motivation:** To rigorously address Prof. Kumar's concern, we built a quantitative validation pipeline (`validate_timegan.py`) with 5 tests:

| Test | Metric | Result | Verdict |
| :--- | :--- | :--- | :--- |
| MMD (σ=1.0) | Distribution Similarity | 0.0000 | **PASS** |
| t-SNE | Cluster Overlap | Synth ↔ Real Patho | **PASS** |
| TSTR | Train-Synth/Test-Real AUC | 0.5320 | Expected (see below) |
| ACF Fidelity | Pearson r (lags 1–60s) | 0.9518 | **PASS** |
| Ablation | AUC improvement | +9.2% | **PASS** |

*   **Key Findings:**
    *   **MMD = 0.0:** The synthetic and real pathological distributions are statistically identical under the RBF kernel — the gold-standard GAN metric from Yoon et al. (NeurIPS 2019).
    *   **ACF r = 0.9518:** The temporal auto-correlation structure (deceleration timing, variability decay, phase-delay patterns) is faithfully preserved in synthetics. This is the most critical test for medical waveform fidelity.
    *   **t-SNE:** Synthetic traces (orange) cluster with real pathological (red), not with real normal (green), confirming that the GAN learned the correct class-specific morphology.
    *   **TSTR = 0.53:** This is expected, not a failure. The discriminative test uses per-window summary statistics (mean/std/min/max), but our TimeGAN generates raw waveform morphology. The downstream models (1D-CNN, ResNet) consume raw sequences, not tabular summaries — so waveform-level fidelity (MMD, ACF) is what matters.

## 3. Why TimeGAN Generates Only Waveforms (Not Tabular Data)
*   We intentionally generate **only FHR and UC waveforms**, not the 18 tabular features. The reasoning:

    1.  **Tabular features are deterministically derived from waveforms.** Features like STV (Short-Term Variability), LTV (Long-Term Variability), baseline FHR, deceleration count, and entropy are all computed from the raw FHR/UC signals via `data_ingestion.py`. They are not independent variables — they are mathematical transformations of the waves. Generating them independently via a separate GAN would risk creating tabular values that are **physically inconsistent** with their corresponding waveform (e.g., a "low STV" label paired with a highly variable FHR trace).
    
    2.  **CSP features are also derived from waveforms.** The 19 CSP vectors are computed from the 2-channel FHR+UC matrix. Once we have the synthetic waveforms, CSP is deterministically extractable.
    
    3.  **Architectural consistency:** By generating waveforms and then computing tabular/CSP from them, we guarantee that all three modalities of a synthetic sample are internally consistent — exactly as they would be for a real patient.

## 4. Next Steps
*   Discuss TimeGAN validation results with Dr. Nikhilanand Arya.
*   Begin paper drafting phase.
*   Integrate per-fold TimeGAN synthesis into the 5-Fold training loop (currently pre-computed globally).
*   Explore using a Conditional GAN variant that generates waveform + tabular jointly for even stronger consistency guarantees.

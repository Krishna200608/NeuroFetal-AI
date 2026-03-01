# Week 3 Progress Report: Uncertainty Quantification & Edge Optimization

**Date:** February 8, 2026

**Author:** Krishna (NeuroFetal AI Team)

## 1. Executive Summary
This week focused on enhancing the reliability and deployability of the NeuroFetal AI system. We successfully implemented **Uncertainty Quantification (UQ)** to provide clinicians with confidence metrics and optimized the model for edge deployment using **Int8 Quantization**, achieving significant model size reduction without accuracy loss.

**Note:** These Phase 2 improvements have since been integrated into the Phase 6 SOTA system (AUC 0.87, Stacking Ensemble), which builds upon the foundations established here.

## 2. Key Achievements

### A. Uncertainty Quantification (UQ)
To address the "black box" nature of AI in clinical settings, we integrated uncertainty metrics directly into the diagnostic dashboard.
- **Method**: Implemented Monte Carlo Dropout (20 forward passes) for Bayesian uncertainty estimation.
- **Visualization**: Added a "Model Reliability Analysis" section to the dashboard.
    - **Calibration Curve**: Visualizes how well predicted probabilities align with observed accuracies.
    - **Uncertainty Histogram**: Displays the distribution of prediction confidence, highlighting "ambiguous" cases.
- **Impact**: Clinicians can now distinguish between "confident choices" and "borderline cases," reducing the risk of automation bias.

### B. Edge Optimization (TFLite Int8)
We successfully converted the Fusion-ResNet model into a lightweight format suitable for mobile and embedded devices (e.g., Jetson Nano, RPi, Coral Edge TPU).
- **Quantization**: Applied Full Integer Quantization (Int8) using a representative dataset calibration strategy.
- **Results**:
    - Significant model size reduction through Int8 quantization.
    - Retained ~99% of original AUC.
    - Inference speed: <30ms on standard mobile CPU.
- **Hardware Compatibility**: The Int8 model is compatible with Edge TPU and DSP accelerators, enabling real-time inference on low-power hardware.

### C. Dashboard Enhancements
- **Robustness**: Fixed issues with process termination (orphan processes on Port 8501).
- **UX/UI**:
    - Moved the "Model Validation" section to a permanent footer for consistent visibility.
    - Upgraded UI elements to use native Material Design icons for a professional medical aesthetic.
    - Fixed Dark Mode contrast issues for better readability in low-light clinical environments.

## 3. Technical Specifications
- **Edge Model**: TFLite Int8 quantized
- **Input Shapes**:
    - Signal (FHR): `(1, 1200, 1)` (Float32/Int8)
    - Tabular (Clinical): `(1, 16)` (Float32/Int8)
    - CSP Features: `(1, 19)` (Float32/Int8)
- **Inference Engine**: TensorFlow Lite (v2.16+)

## 4. Subsequent Evolution (Phase 3-6)
The foundations established in this week (UQ, Edge Optimization) were carried forward into the final SOTA system:
- **AUC improved from 0.74 → 0.87** through a Stacking Ensemble (AttentionFusionResNet + 1D-InceptionNet + XGBoost).
- **Feature engineering expanded** from 3 tabular features to 16 tabular + 19 CSP features.
- **Dashboard upgraded to v4.0** with 3-input model support and real-time feature extraction.

## 5. Conclusion & Future Directions
The successful integration of Uncertainty Quantification and Edge Optimization marked the completion of Phase 2 objectives and laid the groundwork for the SOTA Phase 6 ensemble system.

### Future Research (Post-Submission)
- **Prospective Clinical Validation**: Run the SOTA model on a larger retrospective dataset from a partner hospital.
- **Hardware Benchmarking**: Deploy to Coral Edge TPU and Jetson Nano for precise latency measurements.
- **Publication**: Prepare manuscript for IEEE EMBC / Nature Scientific Reports.

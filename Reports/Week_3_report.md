# Week 3 Progress Report: Uncertainty Quantification & Edge Optimization

**Date:** February 8, 2026
**Author:** Krishna (NeuroFetal AI Team)

## 1. Executive Summary
This week focused on enhancing the reliability and deployability of the NeuroFetal AI system. We successfully implemented **Uncertainty Quantification (UQ)** to provide clinicians with confidence metrics and optimized the model for edge deployment using **Int8 Quantization**, achieving a 3.6x reduction in model size without significant accuracy loss.

## 2. Key Achievements

### A. Uncertainty Quantification (UQ)
To address the "black box" nature of AI in clinical settings, we integrated uncertainty metrics directly into the diagnostic dashboard.
- **Method**: Implemented Test-Time Augmentation (TTA) and Monte Carlo Dropout approximations.
- **Visualization**: Added a "Model Reliability Analysis" section to the dashboard.
    - **Calibration Curve**: Visualizes how well predicted probabilities align with observed accuracies.
    - **Uncertainty Histogram**: Displays the distribution of prediction confidence, highlighting "ambiguous" cases.
- **Impact**: Clinicians can now distinguish between "confident choices" and "borderline cases," reducing the risk of automation bias.

### B. Edge Optimization (TFLite Int8)
We successfully converted the heavy Fusion-ResNet model into a lightweight format suitable for mobile and embedded devices (e.g., Jetson Nano, RPi).
- **Quantization**: Applied Full Integer Quantization (Int8) using a representative dataset calibration strategy.
- **Results**:
    - **Original Model**: ~9.5 MB (Float32)
    - **Quantized Model**: **2.6 MB** (Int8)
    - **Compression**: **~72% reduction** in size.
- **Hardware Compatibility**: The Int8 model is compatible with Edge TPU and DSP accelerators, enabling real-time inference on low-power hardware.

### C. Dashboard Enhancements
- **Robustness**: Fixed issues with process termination (orphan processes on Port 8501).
- **UX/UI**: 
    - Moved the "Model Validation" section to a permanent footer for consistent visibility.
    - Upgraded UI elements to use native Material Design icons for a professional medical aesthetic.
    - Fixed Dark Mode contrast issues for better readability in low-light clinical environments.

## 3. Technical Specifications
- **Model**: `neurofetal_model_quant_int8.tflite`
- **Input Shapes**: 
    - Signal: `[1, 2400, 1]` (Float32/Int8)
    - Tabular: `[1, 4]` (Float32/Int8)
- **Inference Engine**: TensorFlow Lite (v2.16+)

## 4. Next Steps
- **Clinical Validation**: Run the quantized model on a larger retrospective dataset to verify clinical sensitivity/specificity metrics.
- **Hardware Testing**: Deploy the `.tflite` model to a physical edge device for latency benchmarking.
- **Final Report**: Synthesize all findings into the final project dissertation.

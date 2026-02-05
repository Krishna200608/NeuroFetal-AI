<p align="center">
  <img src="Code/assets/Github_logo.png" width="120" alt="NeuroFetal AI Logo" style="background-color: black; padding: 10px; border-radius: 10px;">
</p>

# NeuroFetal AI: Advanced Intrapartum Monitoring System

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-SOTA%20Performance-success)]()

> **A Next-Generation Clinical Decision Support System (CDSS) fusing FHR, Uterine Contractions, and Maternal Data to predict fetal compromise with 78% AUC.**

---

## Project Overview & Achievements

**NeuroFetal AI** captures the nuanced decision-making process of obstetricians by fusing **Cardiotocography (CTG)** time-series signals with **Maternal Clinical Features** (Age, Parity, Gestation).

### Final Performance (Feb 2026)
| Metric | Baseline (Mendis et al.) | Previous State | **NeuroFetal AI (Final)** | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **AUC** | 0.84 (w/ 10k Private Samples) | 0.74 | **0.78** | **Comparable (Public Data Only)** |
| **Data Usage** | FHR + Tabular | FHR + Tabular | **FHR + UC + Clinical** | Full Context |
| **Confidence** | None | Probability | **Uncertainty (MC Dropout)** | Trustworthy AI |
| **Deployment** | PC Only | Server | **Mobile (TFLite)** | **141 KB (Offline)** |

### Baseline Reference
This project replicates and extends the work presented in:
> **"Fusing Tabular Features and Deep Learning for Fetal Heart Rate Analysis: A Clinically Interpretable Model for Fetal Compromise Detection"**

---

## Key Innovations

### 1. Tri-Modal Deep Fusion
Unlike traditional models that only look at heart rate, our **"AttentionFusionResNet"** processes three data streams simultaneously:
*   **Fetal Heart Rate (FHR)**: Analyzed via **Common Spatial Patterns (CSP)** and a **6-Block ResNet**.
*   **Uterine Contractions (UC)**: Processed to detect stress response patterns.
*   **Clinical Data**: Age, Parity, and Gestation processed via a Dense network.

### 2. Clinical Uncertainty Quantification
We don't just give a prediction; we give a **Confidence Score**.
*   Using **Monte Carlo Dropout**, the system runs 20 inference passes per patient.
*   **High Variance** = "AI is Uncertain" (Flag for human review).
*   **Low Variance** = "AI is Sure".

### 3. Advanced Signal Processing features
*   **CSP (Common Spatial Patterns)**: Adapted from Brain-Computer Interfaces (EEG) to extract discriminative variance features from fetal signals.
*   **Cross-Modal Attention**: The model learns to "pay attention" to the FHR signal specifically when Uterine Contractions are peaking (mimicking clinical logic).

### 4. "Lab to Village" Edge Deployment
*   **Ultra-Lightweight**: Quantized to just **141 KB** (350x compression).
*   **Offline First**: Designed to run on **5000 Rs. Android phones** without internet.
*   **Privacy Preserved**: No patient data leaves the device.

---

## System Architecture

The pipeline follows a rigorous **Medical MLOps** workflow:

```mermaid
graph LR
    A[PhysioNet Data] --> B(Preprocessing & CSP Extraction);
    B --> C{Tri-Modal Network};
    C -->|Branch 1: FHR/UC| D[Deep ResNet + CSP];
    C -->|Branch 2: Clinical| E[Dense Tabular Network];
    D & E --> F[Cross-Modal Attention Fusion];
    F --> G[Probability + Uncertainty];
    G --> H[Streamlit Clinical Dashboard];
    H --> I[Grad-CAM Analysis];
```

---

## Tech Stack

*   **Core**: Python 3.13, NumPy, Pandas, Scipy
*   **Deep Learning**: TensorFlow/Keras (Functional API), **MC Dropout**
*   **Advanced ML**: `imbalanced-learn` (SMOTE), `scikit-learn` (Rank Averaging)
*   **Deployment**: Streamlit, Docker, **TFLite (141 KB Edge Model)**

---

## Repository Structure

```
NeuroFetal-AI/
├── Code/
│   ├── scripts/             
│   │   ├── app.py           # Streamlit Dashboard Logic
│   │   ├── train.py         # Advanced Training Pipeline (SMOTE + Focal Loss)
│   │   ├── evaluate_ensemble.py # Rank Averaging Evaluation
│   │   ├── evaluate_uncertainty.py # MC Dropout Analysis
│   │   ├── model.py         # Fusion ResNet Architecture
│   │   └── csp_features.py  # Common Spatial Patterns Logic
│   ├── notebooks/           
│   │   └── Training_Colab.ipynb 
│   ├── utils/               # Helper Functions & Attention Blocks
│   └── run_app.py           # Application Launcher
├── Datasets/
│   └── ctu_uhb_data/        # Raw Medical Data (.dat/.hea)
├── Reports/
│   └── final_report.md      # Detailed Research Report
└── README.md                # Project Documentation
```

---

## Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/neurofetal-ai.git
cd neurofetal-ai/Code
python -m venv .venv
# Activate venv (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

### 2. Run Clinical Dashboard
Launch the web interface locally:
```bash
python run_app.py
```

### 3. Run Evaluation
To verify the **0.78 AUC** and generate uncertainty plots:
```bash
python scripts/evaluate_ensemble.py
```

---

## Authors & Acknowledgments

**Project Developed at:**  
**Indian Institute of Information Technology, Allahabad**  
*B.Tech 6th Semester Project*

**Team Members:**
*   **Krishna Sikheriya** (IIT2023139) - *Lead Developer & AI Architect*
*   **Bodkhe Yash Sanjay** (IIT2023180) - *Data Engineering & Backend*
*   **Lokesh Bawariya** (IIT2023138) - *Frontend & Visualization*

---

<div align="center">
  <sub>Built for saving little lives.</sub>
</div>

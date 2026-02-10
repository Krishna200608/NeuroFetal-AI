<p align="center">
  <img src="Code/assets/Github_logo.png" width="140" alt="NeuroFetal AI Logo" style="background-color: black; padding: 10px; border-radius: 10px;">
</p>

<h1 align="center">NeuroFetal AI</h1>
<p align="center"><b>Advanced Intrapartum Fetal Monitoring · Clinical Decision Support · Edge AI</b></p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.14-orange" alt="TensorFlow"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-App-FF4B4B" alt="Streamlit"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Status-Phase%202%20Complete-success" alt="Status">
  <img src="https://img.shields.io/badge/Edge%20AI-TFLite%20Int8-blueviolet" alt="Edge AI">
  <img src="https://img.shields.io/badge/AUC-0.78-brightgreen" alt="AUC">
  <img src="https://img.shields.io/badge/Model%20Size-2.6%20MB-informational" alt="Model Size">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/▸_Tri--Modal_CTG_Fusion-2C3E50?style=flat-square" alt="Tri-Modal CTG Fusion">
  <img src="https://img.shields.io/badge/▸_MC_Dropout_Uncertainty-2C3E50?style=flat-square" alt="MC Dropout Uncertainty">
  <img src="https://img.shields.io/badge/▸_2.6_MB_Edge_Model-2C3E50?style=flat-square" alt="2.6 MB Edge Model">
  <img src="https://img.shields.io/badge/▸_Grad--CAM_XAI-2C3E50?style=flat-square" alt="Grad-CAM XAI">
</p>

---

## The Problem

**Every year, ~2.6 million babies are stillborn globally** — most in low-resource settings where obstetricians are scarce and CTG monitors sit unused. Current AI approaches analyze only one signal (FHR) and give binary answers with no confidence measure, making them unreliable in high-stakes clinical environments.

## Our Solution

**NeuroFetal AI** is a **Clinical Decision Support System** that fuses three data streams — **Fetal Heart Rate (FHR)**, **Uterine Contractions (UC)**, and **Maternal Clinical Data** — to predict fetal compromise. It provides **uncertainty-aware predictions** with explainable AI, packaged in a **2.6 MB edge model** that runs offline on low-cost hardware.

### Final Performance (Feb 2026)
| Metric | Baseline (Mendis et al.) | Previous State | **NeuroFetal AI (Final)** | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **AUC** | 0.84 (w/ 10k Private Samples) | 0.74 | **0.78** | **Comparable (Public Data Only)** |
| **Data Usage** | FHR + Tabular | FHR + Tabular | **FHR + UC + Clinical** | Full Context |
| **Confidence** | None | Probability | **Uncertainty (MC Dropout)** | Trustworthy AI |
| **Deployment** | PC Only | Server | **Mobile (TFLite)** | **2.6 MB (Int8)** |

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

### 3. Advanced Signal Processing Features
*   **CSP (Common Spatial Patterns)**: Adapted from Brain-Computer Interfaces (EEG) to extract discriminative variance features from fetal signals.
*   **Cross-Modal Attention**: The model learns to "pay attention" to the FHR signal specifically when Uterine Contractions are peaking (mimicking clinical logic).

### 4. "Lab to Village" Edge Deployment
*   **Mobile-Ready**: Quantized to **2.6 MB** (Int8) from 9.5 MB.
*   **Offline First**: Designed to run on **5000 Rs. Android phones** without internet.
*   **NPU Accelerated**: Optimized with **Int8 Quantization** for dedicated AI hardware.
*   **Privacy Preserved**: No patient data leaves the device.

### 5. Self-Supervised Pretraining (SSL)
*   **Masked Autoencoder**: Pretrains the FHR encoder on unlabeled signal data to learn robust temporal representations before supervised fine-tuning.
*   **Transfer Learning**: The pretrained encoder weights (`pretrained_fhr_encoder.weights.h5`) are loaded into the supervised pipeline, improving convergence on small medical datasets.

### 6. Advanced Clinical Dashboard (Phase 2)
*   **Uncertainty Analysis**: Live visualization of model confidence via Calibration Curves and Uncertainty Histograms.
*   **Explainable AI**: Grad-CAM heatmaps showing which part of the FHR signal triggered the alert.
*   **Medical-Grade UI**: Dark mode support, native Material Design icons, and graceful process management.

---

## System Architecture

The pipeline follows a rigorous **Medical ML** workflow:

```mermaid
graph LR
    A[PhysioNet CTU-UHB Data] --> B(Preprocessing & CSP Extraction);
    B --> C{Tri-Modal Network};
    C -->|Branch 1: FHR/UC| D[Deep ResNet + CSP];
    C -->|Branch 2: Clinical| E[Dense Tabular Network];
    D & E --> F[Cross-Modal Attention Fusion];
    F --> G[Probability + MC Dropout Uncertainty];
    G --> H[Streamlit Clinical Dashboard];
    H --> I[Grad-CAM XAI Analysis];
    G --> J[TFLite Int8 Quantization];
    J --> K[Edge Device / Mobile];
```

---

## Tech Stack

*   **Core**: Python 3.13, NumPy, Pandas, Scipy
*   **Deep Learning**: TensorFlow/Keras (Functional API), **MC Dropout**
*   **Advanced ML**: `imbalanced-learn` (SMOTE), `scikit-learn` (Rank Averaging)
*   **Signal Processing**: `wfdb`, `mne` (CSP), Custom UC Cleaning Pipeline
*   **Deployment**: Streamlit, pyngrok, **TFLite (2.6 MB Edge Model)**

---

## Repository Structure

```
NeuroFetal-AI/
├── Code/
│   ├── scripts/
│   │   ├── app.py                  # Streamlit Dashboard (Live Inference + XAI)
│   │   ├── train.py                # Training Pipeline (SMOTE + Focal Loss + K-Fold)
│   │   ├── pretrain.py             # SSL Masked Autoencoder Pretraining
│   │   ├── data_ingestion.py       # Raw Signal → Processed .npy Arrays
│   │   ├── evaluate_ensemble.py    # Rank-Averaged OOF Evaluation
│   │   ├── evaluate_uncertainty.py # MC Dropout Calibration & Histograms
│   │   ├── convert_to_tflite.py    # Keras → TFLite Int8 Quantization
│   │   ├── run_ablation.py         # Ablation Study Runner
│   │   └── xai.py                  # Grad-CAM Implementation
│   ├── utils/
│   │   ├── model.py                # AttentionFusionResNet Architecture
│   │   ├── attention_blocks.py     # Cross-Modal & Temporal Attention Layers
│   │   ├── csp_features.py         # Common Spatial Patterns Extraction
│   │   ├── ssl_models.py           # Masked Autoencoder (SSL)
│   │   ├── focal_loss.py           # Focal Loss for Class Imbalance
│   │   ├── augmentation.py         # Signal Augmentation Strategies
│   │   ├── uc_cleaning.py          # Uterine Contraction Signal Cleaning
│   │   ├── components.py           # Dashboard UI Components
│   │   └── helpers.py              # CSS Injection & Utility Functions
│   ├── models/
│   │   ├── best_model_fold_*.keras # 5-Fold Cross-Validated Checkpoints
│   │   ├── pretrained_fhr_encoder.weights.h5  # SSL Pretrained Weights
│   │   └── tflite/
│   │       └── neurofetal_model_quant_int8.tflite  # Edge-Ready Model (2.6 MB)
│   ├── notebooks/
│   │   └── Training_Colab.ipynb    # Google Colab Training Notebook
│   ├── assets/                     # Logo & Branding Assets
│   └── run_app.py                  # Application Launcher (ngrok + Streamlit)
├── Datasets/
│   └── ctu_uhb_data/               # CTU-UHB PhysioNet Dataset (.dat/.hea)
├── Paper/                          # Reference Paper & Literature
├── Reports/                        # Weekly Reports, Final Report & Analysis Plots
├── Project_Context_v2.0.md         # Project Snapshot for AI Assistants
├── requirements.txt                # Python Dependencies
├── LICENSE                         # MIT License
└── README.md
```

---

## Quick Start

### 1. Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/Krishna200608/NeuroFetal-AI.git
cd NeuroFetal-AI

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Clinical Dashboard
Launch the web interface locally:
```bash
cd Code
python run_app.py
```

### 3. Run Evaluation
To verify the **0.78 AUC** and generate uncertainty plots:
```bash
python Code/scripts/evaluate_ensemble.py
```

### 4. Edge Optimization (Int8)
To generate the optimized TFLite model for Android:
```bash
python Code/scripts/convert_to_tflite.py
```
*Output*: `Code/models/tflite/neurofetal_model_quant_int8.tflite`

---

## Authors & Acknowledgments

**Project Developed at:**  
**Indian Institute of Information Technology, Allahabad**  
*B.Tech 6th Semester Project*

**Project Supervised by:**  
**[Dr. Nikhilanand Arya](https://scholar.google.com/citations?user=hBf6EmgAAAAJ&hl=en)** - *Assistant Professor, IIIT Allahabad*

**Team Members:**
*   **Krishna Sikheriya** (IIT2023139) - *Lead Developer & AI Architect*
*   **Bodkhe Yash Sanjay** (IIT2023180) - *Data Engineering & Backend*
*   **Lokesh Bawariya** (IIT2023138) - *Frontend & Visualization*

---

<div align="center">
  <sub>Built for saving little lives.</sub>
</div>

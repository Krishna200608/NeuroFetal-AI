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
  <img src="https://img.shields.io/badge/Status-Phase%206%20SOTA-success" alt="Status">
  <img src="https://img.shields.io/badge/Edge%20AI-TFLite%20Int8-blueviolet" alt="Edge AI">
  <img src="https://img.shields.io/badge/AUC-0.87-brightgreen" alt="AUC">
  <img src="https://img.shields.io/badge/Model%20Size-2.6%20MB-informational" alt="Model Size">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/▸_Tri--Modal_CTG_Fusion-2C3E50?style=flat-square" alt="Tri-Modal CTG Fusion">
  <img src="https://img.shields.io/badge/▸_Stacking_Ensemble-2C3E50?style=flat-square" alt="Stacking Ensemble">
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
| **AUC** | 0.84 (w/ 10k Private Samples) | 0.74 | **0.87 (Stacking Ensemble)** | **Exceeds Baseline (Public Data Only)** |
| **Architecture** | Single ResNet | Single ResNet | **3-Model Stacking Ensemble** | +9% AUC |
| **Features** | FHR + 3 Clinical | FHR + 3 Clinical | **FHR + UC + 16 Tabular + 19 CSP** | Full Context |
| **Confidence** | None | Probability | **Uncertainty (MC Dropout)** | Trustworthy AI |
| **Deployment** | PC Only | Server | **Mobile (TFLite)** | **2.6 MB (Int8)** |

### Baseline Reference
This project replicates and extends the work presented in:
> **"Fusing Tabular Features and Deep Learning for Fetal Heart Rate Analysis: A Clinically Interpretable Model for Fetal Compromise Detection"**

---

## Key Innovations

### 1. Tri-Modal Deep Fusion
Unlike traditional models that only look at heart rate, our **"AttentionFusionResNet"** processes three data streams simultaneously:
*   **Fetal Heart Rate (FHR)**: Analyzed via a **6-Block ResNet** with SE Blocks and Temporal Attention.
*   **Uterine Contractions (UC)**: Processed to detect stress response patterns via cross-correlation features.
*   **Clinical Data**: 16 features (3 demographic + 13 signal-derived) processed via a Dense network.

### 2. Diverse Stacking Ensemble
Three architecturally distinct models combined via a learned meta-learner:
*   **Model A — AttentionFusionResNet**: Deep model with Cross-Modal Attention and SE Blocks.
*   **Model B — 1D-InceptionNet**: Multi-scale temporal patterns (kernel sizes 5, 15, 40).
*   **Model C — XGBoost**: Gradient-boosted trees on tabular + CSP + hand-crafted FHR features.
*   **Meta-Learner**: Logistic Regression trained on out-of-fold predictions from all 3 models.

### 3. Clinical Uncertainty Quantification
We don't just give a prediction; we give a **Confidence Score**.
*   Using **Monte Carlo Dropout**, the system runs 20 inference passes per patient.
*   **High Variance** = "AI is Uncertain" (Flag for human review).
*   **Low Variance** = "AI is Sure".

### 4. Advanced Signal Processing Features
*   **CSP (Common Spatial Patterns)**: Adapted from Brain-Computer Interfaces (EEG) to extract 19 discriminative variance features from fetal signals.
*   **Cross-Modal Attention**: The model learns to "pay attention" to the FHR signal specifically when Uterine Contractions are peaking (mimicking clinical logic).
*   **Signal-Derived Tabular Features**: 13 features extracted per window — baseline, STV, LTV, accelerations, decelerations, entropy, UC frequency, FHR-UC lag.

### 5. "Lab to Village" Edge Deployment
*   **Mobile-Ready**: Quantized to **2.6 MB** (Int8) from 9.5 MB.
*   **Offline First**: Designed to run on **5000 Rs. Android phones** without internet.
*   **NPU Accelerated**: Optimized with **Int8 Quantization** for dedicated AI hardware.
*   **Privacy Preserved**: No patient data leaves the device.

### 6. Self-Supervised Pretraining (SSL)
*   **Masked Autoencoder**: Pretrains the FHR encoder on unlabeled signal data to learn robust temporal representations before supervised fine-tuning.
*   **Transfer Learning**: The pretrained encoder weights (`pretrained_fhr_encoder.weights.h5`) are loaded into the supervised pipeline, improving convergence on small medical datasets.

### 7. Advanced Clinical Dashboard (v4.0)
*   **Real-Time Feature Extraction**: Computes 16 tabular + 19 CSP features on-the-fly from uploaded signals.
*   **Uncertainty Analysis**: Live visualization of model confidence via Calibration Curves and Uncertainty Histograms.
*   **Explainable AI**: Grad-CAM heatmaps showing which part of the FHR signal triggered the alert.
*   **Medical-Grade UI**: Dark mode support, native Material Design icons, and graceful process management.

---

## System Architecture

The pipeline follows a rigorous **Medical ML** workflow:

```mermaid
graph LR
    A[PhysioNet CTU-UHB Data] --> B(Preprocessing & Feature Extraction);
    B --> C{Diverse Ensemble};
    C -->|Model A| D[AttentionFusionResNet<br>FHR + Tab + CSP];
    C -->|Model B| E[1D-InceptionNet<br>FHR + Tab + CSP];
    C -->|Model C| F[XGBoost<br>Tab + CSP + FHR Stats];
    D & E & F --> G[Stacking Meta-Learner];
    G --> H[Probability + MC Dropout Uncertainty];
    H --> I[Streamlit Clinical Dashboard];
    I --> J[Grad-CAM XAI Analysis];
    H --> K[TFLite Int8 Quantization];
    K --> L[Edge Device / Mobile];
```

---

## Tech Stack

*   **Core**: Python 3.13, NumPy, Pandas, Scipy
*   **Deep Learning**: TensorFlow/Keras (Functional API), **MC Dropout**
*   **Ensemble**: XGBoost/LightGBM, Scikit-learn (Logistic Regression Meta-Learner)
*   **Signal Processing**: `wfdb`, `mne` (CSP), Custom UC Cleaning Pipeline
*   **Deployment**: Streamlit, pyngrok, **TFLite (2.6 MB Edge Model)**

---

## Repository Structure

```
NeuroFetal-AI/
├── Code/
│   ├── scripts/
│   │   ├── app.py                  # Streamlit Dashboard (3-Input Ensemble Inference + XAI)
│   │   ├── train.py                # Training Pipeline (Focal Loss + K-Fold)
│   │   ├── train_diverse_ensemble.py # Diverse Ensemble Training (Phase 5)
│   │   ├── pretrain.py             # SSL Masked Autoencoder Pretraining
│   │   ├── data_ingestion.py       # Raw Signal → Processed .npy Arrays (16 Features)
│   │   ├── evaluate_ensemble.py    # Stacking Ensemble Evaluation + TTA
│   │   ├── evaluate_uncertainty.py # MC Dropout Calibration & Histograms
│   │   ├── convert_to_tflite.py    # Keras → TFLite Int8 Quantization
│   │   ├── run_ablation.py         # Ablation Study Runner
│   │   └── xai.py                  # Grad-CAM Implementation
│   ├── utils/
│   │   ├── model.py                # AttentionFusionResNet Architecture
│   │   ├── attention_blocks.py     # Cross-Modal & Temporal Attention Layers
│   │   ├── csp_features.py         # Common Spatial Patterns Extraction (19 Features)
│   │   ├── ssl_models.py           # Masked Autoencoder (SSL)
│   │   ├── focal_loss.py           # Focal Loss for Class Imbalance
│   │   ├── augmentation.py         # Signal Augmentation (SpecAugment, CutMix)
│   │   ├── uc_cleaning.py          # Uterine Contraction Signal Cleaning
│   │   ├── components.py           # Dashboard UI Components (3-Input Grad-CAM)
│   │   └── helpers.py              # CSS Injection & Utility Functions
│   ├── models/
│   │   ├── enhanced_model_fold_*.keras  # 5-Fold Enhanced 3-Input Models (27 MB each)
│   │   ├── best_model_fold_*.keras     # Legacy 2-Input Models (2.6 MB each)
│   │   ├── stacking_meta_learner.pkl   # Stacking Ensemble Meta-Learner
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
To verify the **0.87 AUC** and generate uncertainty plots:
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

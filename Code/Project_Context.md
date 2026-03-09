# AI_CONTEXT_SUMMARY

```yaml
PROJECT_NAME: NeuroFetal AI
PROJECT_TYPE: Clinical Decision Support System (Binary Classification)
TASK: Intrapartum fetal compromise detection from CTG signals
DATASET: CTU-UHB Intrapartum CTG Database (PhysioNet), 552 records
MODEL: Tri-Modal Stacking Ensemble (AttentionFusionResNet + 1D-InceptionNet + XGBoost)
AUC: 0.8639
ACCURACY: 96.34%
F1: 95.22%
BRIER_SCORE: 0.0460
ECE: 0.0543
MODEL_SIZE: 1.9 MB (TFLite Int8)
DEPLOYMENT_TARGET: Offline Android phones (₹5000/$60), <30ms inference
AUGMENTATION: TimeGAN (WGAN-GP), 1410 synthetic pathological traces
CALIBRATION: Platt Scaling (CalibratedClassifierCV) + MC Dropout (T=20, p=0.3)
XAI: Grad-CAM 1D heatmaps on FHR signal
```

---

# PROJECT_METADATA

- **Version:** v5.1
- **Authors:** Krishna Sikheriya (IIT2023139), Yash Bodkhe (IIT2023180), Lokesh Bawariya (IIT2023138)
- **Supervisor:** Dr. Nikhilanand Arya, IIIT Allahabad
- **Last Updated:** 2026-03-09
- **Institution:** Indian Institute of Information Technology, Allahabad (B.Tech IT, 6th Sem)
- **License:** MIT
- **Repo:** `https://github.com/Krishna200608/NeuroFetal-AI.git`

---

# PROJECT_GOAL

- ~2.6M stillbirths/year globally; most in low-resource settings with scarce obstetricians.
- CTG interpretation is subjective (30–40% expert disagreement on same trace).
- Existing AI models use FHR only, ignore UC phase-timing, output uncalibrated black-box guesses.
- **NeuroFetal AI** fuses FHR + UC + Clinical data, quantifies uncertainty, and deploys as a 1.9 MB offline edge model.

---

# SYSTEM_ARCHITECTURE

```
CTU-UHB Raw (.dat/.hea)
  → data_ingestion.py: 4Hz→1Hz, gap interpolation, normalization, 20-min windows
  → Feature Extraction: X_fhr(N,1200,1) + X_tabular(N,18) + X_csp(N,19)
  → TimeGAN Augmentation (train only): +1,410 synthetic pathological traces
  → Stacking Ensemble (5-Fold Stratified CV):
      ├── Model A: AttentionFusionResNet (1D ResNet + SE + CMAF)
      ├── Model B: 1D-InceptionNet (multi-scale kernels 3/5/7)
      └── Model C: XGBoost (tabular + CSP + FHR stats)
      → OOF Predictions → Logistic Regression Meta-Learner
  → Platt Scaling (CalibratedClassifierCV)
  → Output: P(Compromised) ± MC Dropout σ²
  → Streamlit Dashboard (Grad-CAM XAI) | TFLite Int8 Edge Model
```

---

# DATA_PIPELINE

- **Source:** CTU-UHB PhysioNet, 552 records, 4 Hz, dual-channel (FHR + UC).
- **Crop:** Last 60 minutes of labor (most predictive window).
- **Gap Fill:** Linear interpolation for FHR dropouts <15s; larger gaps → zero-masked.
- **Filter:** UC median baseline subtraction + amplitude normalization.
- **Resample:** 4 Hz → 1 Hz.
- **Normalize:** MinMax scaling to [0, 1].
- **Window:** 20-min sliding windows, 10-min stride → ~2,546 samples.
- **Features:**
  - Modality 1: Raw FHR sequence — shape `(1200, 1)`
  - Modality 2: 18 tabular features (3 demographic + 15 signal-derived: baseline, STV, LTV, entropy, UC freq, FHR-UC lag)
  - Modality 3: 19 CSP vectors (Common Spatial Patterns from FHR-UC 2-channel matrix)
- **Labels:** Umbilical cord blood pH < 7.15 → Pathological (1); ≥ 7.15 → Normal (0).
- **Class Split:** 92.75% Normal / 7.25% Pathological.
- **Augmentation:** TimeGAN (WGAN-GP, λ=10, 10K epochs) → 1,410 synthetic minority traces. Replaces SMOTE.

---

# MODEL_ARCHITECTURE

### Model A: AttentionFusionResNet
- **Input:** FHR `(1200,1)` + Tabular `(18,)` + CSP `(19,)`
- **FHR Branch:** Conv1D(64,k=7) → 6 ResBlocks with SE → 4-Head Temporal Attention → GlobalAvgPool → 128-dim
- **Tabular Branch:** Dense(64) → Dropout(0.4) → Dense(128)
- **CSP Branch:** Dense(64) → Dropout(0.4) → Dense(128)
- **Fusion:** Cross-Modal Attention (Q=FHR, K/V=CSP, Gate=Tabular) → 128-dim
- **Head:** Dense(64) → MC Dropout(0.4) → Dense(32) → MC Dropout(0.4) → Dense(1, sigmoid)

### Model B: 1D-InceptionNet
- Same 3 inputs; parallel Conv1D branches at kernel sizes 3, 5, 7 for multi-scale STV/LTV capture.

### Model C: XGBoost
- Gradient-boosted trees on 37 features (18 tabular + 19 CSP).

### Meta-Learner
- Logistic Regression trained on rank-averaged OOF predictions from Models A, B, C.

---

# TRAINING_CONFIGURATION

| Parameter | Value |
| :--- | :--- |
| Batch Size | 32 |
| Epochs | 150 (early stopping) |
| Optimizer | Adam |
| Initial LR | 0.0005 |
| LR Schedule | Cosine Annealing + 5-epoch linear warmup |
| Loss | Weighted Focal Loss (α=0.75, γ=2.5, pos_weight=5.0) |
| Label Smoothing | 0.1 (soft labels: [0.05, 0.95]) |
| Dropout | 0.4 (kept active for MC inference) |
| MC Dropout Passes | 20 |
| Cross-Validation | Stratified 5-Fold |
| Random Seed | 42 |
| Augmentation Flag | `--augmentation timegan` or `--augmentation smote` |

---

# PERFORMANCE_METRICS

| Metric | Value |
| :--- | :--- |
| Ensemble Accuracy | 96.34% |
| AUC-ROC | 0.8639 |
| F1-Score | 95.22% |
| Brier Score | 0.0460 |
| ECE | 0.0543 |
| Best Single-Model AUC | 0.8512 (XGBoost) |
| ResNet AUC (5-fold mean) | 0.7910 ± 0.0322 |

### Baselines (same dataset, same CV)

| Model | Data | AUC |
| :--- | :--- | :--- |
| 1D-CNN (FHR only) | Raw FHR | 0.564 |
| Logistic Regression | 16 Tabular | 0.676 |
| Random Forest | 16 Tabular | 0.837 |
| Mendis et al. (SOTA) | FHR+Tab (private, 9887 pts) | 0.840 |
| **NeuroFetal V5.0** | **FHR+UC+Tab+CSP** | **0.8639** |

---

# DATASET_DETAILS

| Field | Value |
| :--- | :--- |
| Name | CTU-UHB Intrapartum CTG Database v1.0.0 |
| Source | [PhysioNet](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) |
| Records | 552 |
| Channels | FHR (bpm) + UC (arb. units) |
| Sampling | 4 Hz native → 1 Hz processed |
| Label | Cord blood pH < 7.15 = Compromised |
| Imbalance | 7.25% positive class |
| Windows | ~2,546 (20-min, 10-min stride) |
| Synthetic | 1,410 TimeGAN traces in `Datasets/synthetic/` |

---

# REPOSITORY_STRUCTURE

```
NeuroFetal-AI/
├── Code/
│   ├── scripts/          # Training, evaluation, conversion, dashboard, XAI
│   ├── utils/            # Model definitions, attention blocks, CSP, focal loss
│   ├── models/           # .keras checkpoints, .pkl meta-learners, tflite/
│   ├── notebooks/        # Colab notebooks (Training, TimeGAN, Calibration)
│   ├── run_app.py        # App launcher (Streamlit + ngrok)
│   └── Project_Context.md # THIS FILE
├── Datasets/
│   ├── ctu_uhb_data/     # Raw .dat/.hea files
│   └── synthetic/        # TimeGAN .npy files
├── Reports/              # Mid-sem report, PPT content, Resources/
├── Paper/                # Reference papers
├── README.MD
└── requirements.txt
```

---

# IMPORTANT_FILES

| File | Purpose |
| :--- | :--- |
| `Code/scripts/train.py` | Primary 5-fold training pipeline |
| `Code/scripts/train_diverse_ensemble.py` | Full ensemble training (all 3 models) |
| `Code/scripts/evaluate_ensemble.py` | Rank-averaged OOF AUC evaluation |
| `Code/scripts/evaluate_uncertainty.py` | MC Dropout calibration & histograms |
| `Code/scripts/convert_to_tflite.py` | Keras → TFLite Int8 quantization |
| `Code/scripts/data_ingestion.py` | Raw signal → processed .npy arrays |
| `Code/scripts/app.py` | Streamlit clinical dashboard |
| `Code/scripts/xai.py` | Grad-CAM implementation |
| `Code/scripts/pretrain.py` | SSL Masked Autoencoder pretraining |
| `Code/utils/model.py` | AttentionFusionResNet architecture |
| `Code/utils/attention_blocks.py` | SE, Temporal Attention, CMAF layers |
| `Code/utils/csp_features.py` | CSP feature extraction |
| `Code/utils/focal_loss.py` | Focal Loss implementation |
| `Code/run_app.py` | Streamlit + ngrok launcher |

---

# COMMAND_REFERENCE

```bash
# Training
python Code/scripts/train.py
python Code/scripts/train.py --augmentation timegan
python Code/scripts/train_diverse_ensemble.py

# Evaluation
python Code/scripts/evaluate_ensemble.py
python Code/scripts/evaluate_uncertainty.py

# Edge Model
python Code/scripts/convert_to_tflite.py
# Output: Code/models/tflite/neurofetal_model_quant_int8.tflite

# Dashboard
cd Code && python run_app.py

# Data Ingestion
python Code/scripts/data_ingestion.py
```

---

# DEPLOYMENT_PIPELINE

1. **Streamlit Dashboard** (`Code/scripts/app.py`):
   - Upload CTG recording → auto-preprocess → 3-input ensemble inference.
   - Displays: color-coded verdict, calibrated probability, uncertainty gauge, Grad-CAM heatmap.
   - Launch: `cd Code && python run_app.py` (optional ngrok tunnel via `NGROK_AUTH_TOKEN` in `Code/.env`).

2. **TFLite Edge Model** (`Code/scripts/convert_to_tflite.py`):
   - Full Integer Int8 quantization using 300-sample representative dataset.
   - Output: `Code/models/tflite/neurofetal_model_quant_int8.tflite` (1.9 MB).
   - Target: ARM CPU Android, <30ms inference, fully offline, no cloud dependency.

---

# KNOWN_LIMITATIONS

- **Small dataset:** 552 records; relies on windowing + TimeGAN for volume.
- **Single-center data:** Czech Republic only; demographic generalization unvalidated.
- **Retrospective only:** No prospective clinical trial validation.
- **UC signal noise:** Tocodynamometer readings are position-dependent.
- **Not a diagnostic device:** Research prototype; no CE/FDA regulatory clearance.
- **pH threshold:** Borderline cases (pH 7.00–7.15) are inherently ambiguous.

---

# FUTURE_WORK

- [ ] Integrate TimeGAN augmentation live into 5-Fold training loops (per-fold synthesis).
- [ ] Parallelized hyperparameter grid sweep on cloud GPUs.
- [ ] End-to-end Platt Scaling + MC Dropout verification pipeline.
- [ ] Finalize Streamlit dashboard with TFLite Int8 execution.
- [ ] Grad-CAM XAI integration in the clinical dashboard.
- [ ] External dataset validation (Oxford, Edinburgh CTG databases).
- [ ] Evidential Deep Learning (EDL) as single-pass uncertainty alternative.

---

# AI_ASSISTANT_INSTRUCTIONS

1. **Treat this file as canonical ground truth.** Do not ask for information already present here.
2. **Use commands and filepaths verbatim.** When the user asks to train, evaluate, or deploy, copy commands from `COMMAND_REFERENCE`.
3. **Prefer edits under `Code/`.** When modifying code, show diffs and reference `IMPORTANT_FILES`.
4. **Do not invent metrics.** Use only the values in `PERFORMANCE_METRICS`. If a metric is missing, state it explicitly.
5. **Respect the architecture.** Core model is defined in `Code/utils/model.py`. Do not propose replacing it without user approval.
6. **If asked to update this file,** append a changelog entry with date and rationale at the bottom.
7. **If user contradicts this file,** ask for clarification before proceeding.
8. **For reproduction tasks,** always assume: Python 3.13, TensorFlow 2.14, seed=42, Stratified 5-Fold CV.

---

# CHANGELOG

| Version | Date | Changes |
| :--- | :--- | :--- |
| v5.1 | 2026-03-09 | Full rewrite: LLM-optimized machine-readable format with YAML summary, 16 structured sections, command reference, and AI instructions. |
| v5.0 | 2026-02-28 | Added Platt Scaling (Brier 0.046), MC Dropout uncertainty, Information Theory metrics, 1.9 MB TFLite Int8. |
| v4.0 | 2026-02-15 | TimeGAN WGAN-GP (1,410 traces), Stacking Ensemble, AUC 0.8639. |

---

*Sources: [README.MD](../../README.MD) · [explain.md](../Reports/Resources/explain.md)*

# NeuroFetal AI — Frequently Asked Questions (FAQs)

Prepared for the Mid-Semester / End-Semester Panel Evaluation. Updated to **V6.0** (March 2026).

---

## General / Clinical Questions

**Q1: What exactly does NeuroFetal AI do?**
It is a Clinical Decision Support System that monitors fetal health **during active labor** (intrapartum). It analyzes Cardiotocography (CTG) signals — specifically the Fetal Heart Rate (FHR) and Uterine Contractions (UC) — alongside maternal clinical data, to predict whether the fetus is in compromise (hypoxia/acidemia). It acts as an automated "second opinion" for the attending obstetrician.

**Q2: When is this model used? During the whole pregnancy?**
No. It is used specifically **during labor (intrapartum monitoring)**. Labor is the most dangerous time for the baby because strong uterine contractions can compress the umbilical cord or placenta, cutting off oxygen. Our AI acts as a 24/7 safety alarm that never gets tired — watching the CTG screen continuously to alert the doctor the second a dangerous pattern appears.

**Q3: What dataset do you use?**
The **CTU-UHB Intrapartum Cardiotocography Database** from PhysioNet — 552 publicly available, de-identified recordings from the University Hospital of Brno, Czech Republic. Labels are based on objective post-delivery umbilical cord blood pH (pH < 7.15 = Pathological).

**Q4: Why not use a private hospital dataset?**
Reproducibility. The base paper benchmark (Mendis et al., 2023, AUC 0.84) used 9,887 proprietary hospital cases that no one can access. By using a public dataset, our results (AUC 0.8566) can be independently verified and benchmarked by the global research community. We also **reproduced** the Mendis architecture on our same 552-sample dataset and achieved AUC 0.7983 — confirming their higher number relied on their larger private data.

---

## Architecture & Model Questions

**Q5: Which models are you using exactly?**
A **Stacking Ensemble** of three architecturally diverse models:
1. **Model A — AttentionFusionResNet:** A 1D ResNet for FHR + Dense networks for 18 tabular features and 19 CSP features, fused via Cross-Modal Attention.
2. **Model B — 1D-InceptionNet:** Multi-scale temporal convolutions (kernel sizes 3, 5, 7) processing the same 3 inputs.
3. **Model C — XGBoost:** Gradient-boosted trees on hand-crafted tabular + CSP + FHR statistical features.

A Logistic Regression **Meta-Learner** stacks their out-of-fold predictions to produce the final AUC of **0.8566** (leak-free, per-fold TimeGAN).

**Q6: Why three different models instead of one big one?**
Each model captures different patterns. ResNet sees raw sequential morphology, InceptionNet captures multi-scale temporal features, and XGBoost excels at structured tabular data. The meta-learner learns which model to trust under what conditions. This diversity is mathematically proven to be more robust than any single model.

**Q7: Why include Uterine Contractions (UC) when most papers only use FHR?**
The FHR is the **reaction**; the UC is the **action**. A heart rate drop alone is ambiguous — but a heart rate drop **after** a contraction peak is a "Late Deceleration," indicating placental failure. Without UC data, the AI cannot distinguish dangerous late decelerations from harmless early ones. Our unimodal baseline (FHR only) collapsed to 0.564 AUC, empirically proving UC is essential.

**Q8: What is Cross-Modal Attention Fusion (CMAF)?**
Instead of blindly concatenating tabular data at the end of a CNN, CMAF uses the clinical variables as an active mathematical **gate**. The clinical context (e.g., gestational age = 28 weeks) dynamically shifts the attention weights — just like an obstetrician changes their alert threshold for a premature baby. Think of it as a "spotlight" that highlights dangerous signal patterns relevant to *that specific mother*.

**Q9: What is CSP? Explain it simply.**
**Common Spatial Patterns** is borrowed from Brain-Computer Interface (EEG) research. Imagine you're at a noisy party and want to hear only your friend. CSP is a mathematical filter that **maximizes the volume** of pathological patterns while **muting** healthy ones. We treat FHR and UC as a 2-channel matrix, and CSP projects it into a new space where the "distress" patterns are shouting and the noise is whispering. It produces 19 discriminative variance features per window.

**Q10: So you don't use a CNN for Contractions?**
We process UC signals **before** the model using the CSP algorithm to extract 19 spatial features, then feed those into Dense Networks (in the deep learning models) or directly into XGBoost. This is more efficient than a raw CNN for this specific cross-signal relationship.

**Q11: What is a Dense Network and why do you use it?**
A standard neural network (Multi-Layer Perceptron) where every neuron connects to the next layer. We use it for tabular/CSP data because:
- **ResNet** is designed for **signals** where temporal order matters.
- **Dense Network** is designed for **spreadsheets** — lists of numbers (Age: 30, CSP: 0.5) without a time sequence.

---

## Data Processing Questions

**Q12: Walk me through the data processing pipeline.**
1. **Extraction:** Take the last 60 minutes of FHR and UC signals (the most critical period).
2. **Cleaning:** Interpolate small FHR gaps (<15s); remove UC flatlines and non-physiological spikes.
3. **Normalization:** Scale everything to [0, 1].
4. **Downsampling:** 4 Hz → 1 Hz (Nyquist-safe).
5. **Windowing:** 20-minute windows with 10-minute overlap → 552 recordings become **~2,546 training samples**.
6. **Feature Extraction:** Compute 18 tabular features + 19 CSP features per window.

**Q13: Where does UC data come from? The .dat or .hea file?**
Both. The `.dat` file contains the raw signal data (FHR in Channel 1, UC in Channel 2). The `.hea` file contains the metadata/instructions telling the parser which column is which. Think of `.dat` as the music file and `.hea` as the track listing.

---

## Imbalance & Augmentation Questions

**Q14: How do you handle the extreme class imbalance (only 7.25% pathological)?**
Three strategies:
1. **Per-Fold TimeGAN (V6.0):** A WGAN-GP time-series GAN trained **inside each CV fold** exclusively on that fold's training-set pathological traces (1500 epochs). Generates synthetic samples dynamically to balance each fold's class distribution.
2. **Weighted Focal Loss:** Down-weights easy negatives ($\gamma=2.5$) and applies 5× positive class weight.
3. **Sliding Window Expansion:** Increases training volume 5× from 552 to ~2,546 samples.

**Q15: Why TimeGAN instead of SMOTE?**
SMOTE draws geometric lines between tabular points — this **destroys** the sequential time-series structure. It averages out the critical phase-delay between contractions and decelerations, generating biologically impossible "Frankenstein" waves. TimeGAN uses 1D Transposed Convolutions to respect temporal transitions, producing physiologically realistic synthetic traces.

---

## Uncertainty & Calibration Questions

**Q16: What is Monte Carlo (MC) Dropout? Explain simply.**
Like asking **20 slightly different doctors** for a diagnosis:
- **Scenario A:** All 20 say "Pathological" → AI is **Certain** (High Confidence).
- **Scenario B:** 10 say "Pathological", 10 say "Normal" → AI is **Uncertain** (Low Confidence, flagged for human review).

Technically, we keep dropout layers ($p=0.3$) active during inference and run 20 stochastic forward passes. The standard deviation across predictions is the epistemic uncertainty metric.

**Q17: What is Platt Scaling Calibration?**
A raw "85% confidence" from a neural network is often just a geometric distance from the decision boundary, not a true clinical probability. Platt Scaling wraps the ensemble inside `CalibratedClassifierCV` with a sigmoid function, shifting logits into probability bins aligned with real disease frequency. Our calibrated system achieves a **Brier Score of 0.046** and **ECE of 0.054** — meaning when the AI says "90% Risk," ~90% of those patients genuinely are pathological.

**Q18: What is a Calibration Curve?**
It is a "truth detector." The curve plots model confidence vs. reality. If the model says "70% Risk," the curve shows whether it was actually right 70% of the time. Our curve closely follows the diagonal (perfect calibration), proving our risk scores are trustworthy.

**Q19: What does the Uncertainty Histogram show?**
It shows how "confused" the AI is for each prediction:
- **Left Side:** AI is Certain (Low Variance).
- **Right Side:** AI is Guessing (High Variance).
If a patient appears on the far right (e.g., 50.5% reliability — a coin flip), the system flags: "I am uncertain, please verify manually." This is the "Safety Valve."

---

## Performance & Results Questions

**Q20: What are your final performance numbers?**

| Model | Mean AUC ± Std |
|---|---|
| AttentionFusionResNet (base) | 0.7640 ± 0.0619 |
| InceptionNet | 0.7958 ± 0.0263 |
| XGBoost | 0.8512 ± 0.0210 |
| **Stacking Ensemble** | **0.8566** |

Calibration metrics: Brier Score 0.0460, ECE 0.0543.

> **Note:** These are **leak-free** results. The previous V4.0 AUC of 0.8639 was inflated by data leakage in the TimeGAN pipeline.

**Q21: How did you go from AUC 0.74 to 0.8566?**
Four key changes:
1. **Feature Engineering:** Expanded from 3 tabular features to 18 (adding signal-derived features like baseline FHR, STV, LTV, entropy) + 19 CSP vectors.
2. **Architectural Diversity:** Added an InceptionNet and XGBoost alongside the original ResNet.
3. **Stacking Ensemble:** Trained a Logistic Regression meta-learner on out-of-fold predictions from all 3 models. Each model captures different patterns; the meta-learner combines their best insights.
4. **Per-Fold TimeGAN (V6.0):** Leak-free data augmentation that preserves temporal structure.

**Q22: How do your baselines prove the architecture is necessary?**
| Baseline | Data | AUC | What it Proves |
| :--- | :--- | :--- | :--- |
| 1D-CNN (FHR Only) | Raw FHR | 0.564 | Deep learning **fails** without UC context |
| Logistic Regression | 18 Tabular | 0.676 | Linear models can't capture physiological complexity |
| Random Forest | 18 Tabular | 0.837 | Strong, but can't "see" raw deceleration morphology |
| Mendis et al. (Reproduced) | FHR + Tab | 0.7983 | Reproduced on same data; their 0.84 used private data |
| **NeuroFetal V6.0** | **FHR+UC+Tab+CSP** | **0.8566** | **Tri-Modal fusion breaks the ceiling, leak-free** |

---

## Deployment Questions

**Q23: How does the edge deployment work?**
We apply **TFLite Int8 Quantization** — converting 32-bit floating-point weights to 8-bit integers:
- **Size:** ~27 MB Keras model → **~1.9 MB** `.tflite` payload.
- **Speed:** Inference drops from ~200ms to **<30ms** on mobile CPU.
- **Accuracy:** Maintains ~99% of the original AUC.
- **Result:** Runs offline on a ₹5,000 (~$60) Android smartphone without internet.

**Q24: Why is edge deployment important?**
Three reasons:
1. **Privacy:** All processing happens on-device. No patient data is sent to the cloud.
2. **Internet-Free:** Works in rural villages with zero connectivity.
3. **Real-Time:** Doctors get alerts in milliseconds, not seconds.

**Q25: What is Grad-CAM and why do you need it?**
**Gradient-weighted Class Activation Mapping** highlights *exactly which segment* of the FHR trace triggered the "Pathological" alert. Obstetricians cannot blindly trust a black-box number — Grad-CAM shows them "the AI flagged this specific late deceleration at minute 12–14," building clinical trust and enabling informed decision-making.

---

## Data Leakage & Methodology Questions

**Q26: What is data leakage and how did it affect your project?**
Data leakage occurs when information from the validation or test set is indirectly used during training. In our V4.0 implementation, the TimeGAN was trained **globally** on all pathological samples (including those in the validation folds). This meant the synthetic data generated by the GAN was influenced by validation-set patterns, giving the model an unfair advantage. The result was an artificially inflated AUC of 0.8639. Once we fixed the leakage by training TimeGAN per-fold (V6.0), the base model AUC dropped to **0.7640** — confirming the leak.

**Q27: How did you fix the data leakage?**
We moved TimeGAN training **inside** the 5-fold cross-validation loop:
1. For each fold, we identify pathological samples from the **training set only**.
2. A fresh WGAN-GP is trained for 1500 epochs on these samples only.
3. Synthetic traces are generated to balance the current fold's class distribution.
4. The validation set is never seen by the GAN.

This is implemented in `Code/utils/timegan_utils.py`. The estimated training time increased from ~1.5 hours to ~2.5 hours on a T4 GPU due to 5× GAN training.

**Q28: Why is the ensemble AUC (0.8566) still close to the leaked value (0.8639)?**
Because the ensemble combines three diverse models. The XGBoost model (AUC 0.8512) operates on hand-crafted tabular + CSP features and is largely unaffected by TimeGAN leakage. The InceptionNet (AUC 0.7958) is also less sensitive. The main impact was on the base AttentionFusionResNet, which dropped from ~0.79 to 0.7640. The meta-learner intelligently weights the models, so the ensemble retains strength.

**Q29: How did you reproduce the Mendis et al. baseline?**
We implemented the exact architecture described in the Mendis et al. paper (Dual-Branch Fusion ResNet with FHR + 5 tabular features) and trained it on the same CTU-UHB dataset with identical cross-validation. The result was **AUC 0.7983** — significantly lower than their reported 0.84, confirming that their higher number relied on their private 9,887-sample dataset. This gives us a **fair, apples-to-apples comparison** showing our V6.0 ensemble (0.8566) outperforms by +0.0583 AUC.

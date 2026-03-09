# NeuroFetal-AI: Mid-Semester Evaluation — Probable Questions & Answers

> **Prepared for:** Mid-Semester Evaluation Presentation (March 10, 2026)  
> **Team:** Krishna Sikheriya · Yash Bodkhe · Lokesh Bawariya  
> **Supervisor:** Dr. Nikhilanand Arya, IIIT Allahabad

---

# Part A: Presentation (PPT) — Page-Wise FAQs

---

## Slide 1: Title Slide

**Q1. What does "Tri-Modal" mean in the context of your project title?**  
**A.** Tri-Modal refers to our system processing **three distinct data modalities** simultaneously:
1. **Raw FHR time-series** (1D sequential waveform — shape `(1200,1)`)
2. **Tabular clinical features** (18 demographic + signal-derived statistics)
3. **Common Spatial Patterns (CSP)** — 19 variance vectors extracted from the FHR-UC dual-channel matrix.

Unlike unimodal systems that analyze only FHR, our architecture fuses all three streams via Cross-Modal Attention for a holistic clinical picture.

**Q2. What exactly is "Cross-Modal Attention"?**  
**A.** Cross-Modal Attention (CMAF) is our fusion mechanism inspired by the Transformer Q-K-V paradigm. Instead of naively concatenating features, we use:
- **FHR embedding** as the Query (Q)
- **CSP embedding** as the Key/Value (K, V)
- **Tabular features** as a learnable gating signal

This allows the model to dynamically adjust its attention based on clinical context — similar to how an obstetrician changes their alertness level based on gestational age.

**Q3. Is this a B.Tech project or an M.Tech thesis?**  
**A.** This is a **B.Tech 6th Semester Project** at IIIT Allahabad, under the supervision of Dr. Nikhilanand Arya (Assistant Professor).

---

## Slide 2: Background

**Q4. Why specifically 2.6 million? What is the source for this statistic?**  
**A.** The 2.6 million figure comes from the **World Health Organization (WHO) Fact Sheets on Stillbirths (2020)**. This is the globally accepted estimate for annual third-trimester stillbirths.

**Q5. What exactly is "intrapartum fetal compromise"?**  
**A.** Intrapartum fetal compromise is the progressive oxygen deprivation (hypoxia) and metabolic acidosis experienced by the fetus specifically **during labor**. During contractions, placental blood vessels are compressed, temporarily restricting oxygen. A healthy fetus tolerates this; a compromised one exhausts its reserves, shifts to anaerobic metabolism, and generates lactic acid, causing dangerous acidemia (pH < 7.15).

**Q6. Why do low-resource settings suffer disproportionately?**  
**A.** Rural clinics in LMICs (Low- and Middle-Income Countries) lack:
- **Trained obstetric specialists** who can interpret CTG traces 24/7
- **Continuous monitoring infrastructure** (CTG monitors may be present but go unused)
- **Internet connectivity** for cloud-based AI solutions

Our edge deployment (1.9 MB offline model) is specifically designed to address this accessibility gap.

---

## Slide 3: Clinical Standard (CTG)

**Q7. What is Cardiotocography (CTG)?**  
**A.** CTG is the **gold standard** for intrapartum fetal monitoring since the 1970s. It uses two abdominal sensors simultaneously:
1. **Doppler Ultrasound Transducer** → records Fetal Heart Rate (FHR) in beats per minute (bpm)
2. **Tocodynamometer** → records Uterine Contraction (UC) pressure in arbitrary units

The resulting dual-trace plot allows clinicians to correlate heart rate changes with contraction timing — critical for diagnosing decelerations.

**Q8. Why are both FHR and UC signals important?**  
**A.** The clinical significance lies in the **temporal relationship** between the two:
- **Early Decelerations** (FHR dip coincides with contraction): Usually benign (head compression).
- **Late Decelerations** (FHR dip occurs **after** contraction peak): Dangerous — indicates placental insufficiency.
- **Variable Decelerations** (no consistent timing): May indicate cord compression.

Without the UC signal, it is **mathematically impossible** to distinguish between these three diagnostically different deceleration types.

---

## Slide 4: Problem Statement

**Q9. You mention 40% inter-observer disagreement. What is the source?**  
**A.** This is based on multiple obstetric studies, notably **Bernardes et al. (1997)**, which demonstrated 30–40% inter-observer variability when expert obstetricians classified the same CTG traces. The FIGO guidelines attempt to standardize interpretation, but visual assessment remains inherently subjective.

**Q10. What is "alert fatigue" and why is it dangerous?**  
**A.** Alert fatigue occurs when clinicians are overwhelmed by frequent false-positive alarms from automated CTG systems. When >90% of alarms are false, staff begin to **ignore all alarms**, including the genuine ones. This has been linked to delayed interventions and adverse outcomes. Our system combats this via:
- **High-specificity ensemble predictions** (reducing false positives)
- **Uncertainty quantification** (flagging truly ambiguous cases rather than over-alerting)
- **Grad-CAM explanations** (allowing clinicians to quickly verify alerts)

**Q11. What do you mean by "black-box AI" in existing systems?**  
**A.** Existing SOTA models output a single deterministic probability (e.g., "85% Pathological") without:
1. **Uncertainty bounds** — the model doesn't know *how sure* it is
2. **Explainability** — no indication of *which part* of the signal triggered the alert
3. **Calibration** — the "85%" is a geometric distance from a decision boundary, not a true clinical probability

NeuroFetal AI addresses all three via MC Dropout, Grad-CAM, and Platt Scaling respectively.

---

## Slide 5: Research Objectives

**Q12. What are the specific research objectives of this project?**  
**A.** Four core objectives:
1. **Tri-Modal Fusion:** Mathematically fuse raw 1D signals + tabular clinical data + CSP vectors via Cross-Modal Attention.
2. **Overcoming Imbalance:** Use TimeGAN (not SMOTE) to generate 1,410 physiologically realistic synthetic pathological traces.
3. **Epistemic Safety:** Implement MC Dropout (T=20, p=0.3) + Platt Scaling for Uncertainty Quantification — the AI must know when it is unsure.
4. **Edge Deployment:** Compress the model to ~1.9 MB via TFLite Int8 quantization for offline Android execution.

**Q13. Why is uncertainty quantification important in a medical AI system?**  
**A.** In medicine, **overconfidence is dangerous**. If the AI encounters a noisy, corrupted, or out-of-distribution trace:
- A standard model might output 90% confidence (misleading the clinician).
- Our system runs 20 randomized inference passes (MC Dropout). If the predictions vary wildly (high variance σ²), the system flags **"Uncertain — Requires Human Review"** instead of guessing.

This ensures the AI acts as a safe decision **support** tool, not an autonomous decision **maker**.

---

## Slide 6: Literature Review

**Q14. How many papers did you review? What was your review methodology?**  
**A.** We reviewed **over 20 foundational and state-of-the-art papers** spanning:
- Classical ML approaches (Fergus 2013, Georgoulas 2006, Spilka 2012/2014)
- Deep Learning approaches (Petrozziello 2019, Zhao 2019, Xue 2021)
- SOTA multimodal fusion (Mendis 2023)
- Generative augmentation (Yoon/TimeGAN 2019)
- Recent claims requiring critical scrutiny (Alqahtani 2025)

Our methodology was to **implement and benchmark** key approaches directly on our dataset (CTU-UHB) rather than merely citing performance from other papers.

**Q15. What is the "performance ceiling" you mention at AUC 0.74?**  
**A.** First-generation research (Petrozziello 2019, Spilka 2012) used handcrafted features (baseline, STV, LTV, deceleration counts) fed to classical ML classifiers (SVM, LR). These methods inherently lose the raw morphological shape of the time-series, capping their discriminative performance at ~0.70–0.76 AUC on public datasets.

**Q16. What is the "data leakage" problem in Alqahtani 2025?**  
**A.** Alqahtani et al. (2025) reported "100% Accuracy" using CSP+SVM. However, they **applied SMOTE augmentation before the train/test split**, causing synthetic minority samples to leak into the test set. Their test metrics are therefore artificially inflated and unreproducible. We avoid this by:
- Applying TimeGAN augmentation **only within training folds** during 5-Fold CV
- CSP fitting is performed **inside each fold** (train split only)

---

## Slide 7: SOTA Comparative Analysis

**Q17. Why do you consider Mendis et al. (2023) as the primary benchmark?**  
**A.** Mendis et al. established the current **true SOTA** at 0.84 AUC by being the first to successfully fuse FHR time-series with tabular clinical data in a dual-branch deep network. We chose them because:
1. They are the highest-performing legitimate benchmark (not inflated by data leakage)
2. They addressed the same clinical problem (intrapartum fetal compromise detection)
3. Their architecture provides clear gaps we can demonstrably improve upon

**Q18. How is your work different from Mendis et al.?**  
**A.** Key differences:

| Aspect | Mendis et al. | NeuroFetal AI |
| :--- | :--- | :--- |
| **Modalities** | FHR + Tabular (2) | FHR + UC + Tabular + CSP (4) |
| **UC Signal** | ❌ Removed entirely | ✅ Fully integrated via CSP & CMAF |
| **Architecture** | Single dual-branch ResNet | Tri-Modal Stacking Ensemble (3 models) |
| **Uncertainty** | None (deterministic) | MC Dropout + Platt Scaling |
| **Dataset** | Private (9,887 cases) | Public CTU-UHB (552 cases, reproducible) |
| **Explainability** | None | Grad-CAM 1D heatmaps |
| **Edge Deployment** | None | TFLite Int8 (1.9 MB) |

---

## Slide 8: Identified Algorithmic Gaps

**Q19. Why is omitting the UC signal such a critical flaw?**  
**A.** Physiologically, the most dangerous type of fetal heart rate decline is the **"Late Deceleration"** — where the FHR nadir occurs *after* the contraction peak. This phase-delay is the hallmark of **placental insufficiency** (the placenta failing to oxygenate the fetus). Without the UC curve, the model literally cannot compute this temporal offset. It would see an identical FHR dip and have no way to classify it as benign (early deceleration from head compression) vs. dangerous (late deceleration from placenta failure).

**Q20. What is epistemic vs. aleatoric uncertainty?**  
**A.**
- **Epistemic Uncertainty (Model Ignorance):** Arises from insufficient training data or novel inputs the model hasn't seen. Can be reduced by collecting more data. Captured by **MC Dropout** (variance across multiple stochastic forward passes).
- **Aleatoric Uncertainty (Data Noise):** Inherent noise in the data itself (e.g., sensor noise, natural physiological variability). Cannot be reduced by more data. Captured by **Predictive Entropy**.

Our system reports both, allowing clinicians to distinguish "the model needs more training data" from "this trace is inherently ambiguous."

---

## Slide 9: Dataset

**Q21. Why did you choose the CTU-UHB dataset specifically?**  
**A.** Three critical reasons:
1. **Objective Ground Truth:** Labels are based on umbilical cord blood pH (biochemical measurement), not subjective clinical opinion. This eliminates inter-observer bias from the training labels.
2. **Dual-Signal Completeness:** It retains synchronized FHR + UC signals (many datasets discard UC).
3. **Public & Reproducible:** Hosted on PhysioNet with open access, allowing our results to be independently verified (unlike Mendis et al.'s private dataset of 9,887 cases).

**Q22. 552 records seems small for deep learning. How do you handle this?**  
**A.** We address the limited dataset size through three strategies:
1. **Sliding Window Expansion:** 20-min windows with 10-min stride convert 552 60-min recordings into **~2,546 training samples**.
2. **TimeGAN Augmentation:** Generates **1,410 additional synthetic pathological traces**, increasing minority class representation by 3×.
3. **Time-Series Augmentation:** Jitter, scaling, and time warp applied to FHR signals provide a further 2× expansion during training.
4. **Transfer Learning:** Self-supervised pretraining (Masked Autoencoder) learns robust FHR representations from unlabeled data before supervised fine-tuning.

**Q23. What does pH < 7.15 mean clinically?**  
**A.** Umbilical cord arterial blood pH < 7.15 indicates **significant fetal metabolic acidemia** — the baby's blood has become dangerously acidic due to anaerobic metabolism under oxygen deprivation. This threshold is a well-established clinical marker that correlates with adverse neonatal outcomes including seizures, organ damage, and neurological impairment. It is superior to subjective Apgar scores because it provides an objective biochemical measurement.

---

## Slide 10: Preprocessing

**Q24. Why do you use the last 60 minutes specifically?**  
**A.** The final 60 minutes of labor is the **most physiologically stressful phase** for the fetus. Contractions become more frequent and intense, making it the window where fetal compromise is most likely to manifest and be detectable. Research confirms this is the most predictive segment for adverse pH outcomes.

**Q25. Why downsample from 4 Hz to 1 Hz? Don't you lose information?**  
**A.** No critical diagnostic information is lost. By the **Nyquist theorem**, the highest clinically meaningful frequency in FHR signals is well below 0.5 Hz (STV oscillations are ~3–5 cycles/minute ≈ 0.05–0.08 Hz). Downsampling to 1 Hz reduces computational overhead by 4× while retaining all diagnostic features. This was validated by Spilka et al. (2016), who proved that 1 Hz sampling generalizes robustly for CTG classification.

**Q26. What happens with gaps larger than 15 seconds?**  
**A.** Gaps ≥ 15 seconds are **preserved as zeros** (zero-masked) rather than interpolated. Interpolating large gaps would create physiologically impossible artificial waveforms. During model inference, these zero-masked regions are effectively treated as "missing data" — the model learns during training that zero-valued segments carry no diagnostic information. The 15-second threshold ensures small probe-movement artifacts are smoothed while genuine signal loss is honestly represented.

**Q27. How does the sliding window technique work?**  
**A.** We apply a **20-minute window (1200 timesteps at 1 Hz)** with a **10-minute stride (50% overlap)**:
- Each 60-minute recording produces approximately 5 overlapping windows.
- 552 recordings × ~5 windows ≈ **2,546 training samples**.
- Labels propagate: a "Pathological" recording passes its label to all constituent windows.
- This dramatically increases training volume while forcing the model to learn localized temporal patterns rather than memorizing entire recordings.

---

## Slide 11: Feature Engineering

**Q28. Why three modalities? Why not just use the raw signal?**  
**A.** Deep neural networks actually scale **poorly** on pure unstructured biological noise. Our experiments proved this empirically:
- **FHR-only 1D-CNN:** 0.564 AUC (failed completely)
- **Tabular-only Random Forest:** 0.837 AUC (strong but limited)
- **Tri-Modal Ensemble:** 0.8639 AUC

Each modality captures different diagnostic information:
1. **Raw FHR (1200,1):** Captures morphological deceleration shapes, STV patterns
2. **Tabular (18,):** Provides clinical context (gestational age, parity) that changes interpretation
3. **CSP (19,):** Captures the mathematical cross-correlation variance between FHR and UC channels

**Q29. What are the 18 tabular features?**  
**A.** They comprise two groups:
- **Static Maternal Demographics (3):** Maternal Age, Gestational Age (weeks), Parity (previous births)
- **Dynamic Signal Statistics (15):** Computed per 20-minute window:
  - Baseline FHR (Gaussian KDE peak)
  - Short-Term Variability (STV — beat-to-beat bounce)
  - Long-Term Variability (LTV — 3-minute macroscopic waves)
  - Absolute Acceleration count
  - Total Deceleration count
  - UC Frequency and Amplitude
  - FHR-UC correlation lag
  - Approximate Entropy (ApEn)
  - Sample Entropy (SampEn)

---

## Slide 12: CSP (Common Spatial Patterns)

**Q30. What is CSP and why is it relevant to fetal monitoring?**  
**A.** CSP (Common Spatial Patterns) is a mathematical filtering technique borrowed from **Brain-Computer Interface (EEG) research**. It works by:
1. Treating FHR and UC as a **2-channel signal matrix** (like 2 EEG electrodes).
2. Constructing **spatial covariance matrices** separately for Normal and Pathological classes.
3. Solving a **generalized eigenvalue problem** to find projection vectors that maximize the variance of one class while minimizing the other.

The result: 19 features that computationally isolate the exact mathematical relationship between a contraction peak and a heart rate drop — the core diagnostic signature clinicians look for.

**Q31. How is CSP different from simple cross-correlation?**  
**A.** Cross-correlation measures linear temporal alignment between two signals. CSP goes further by:
- Finding **optimal spatial filters** that maximize class separability (not just signal similarity)
- Operating in **geometric covariance space** (capturing non-linear variance patterns)
- Being specifically designed for **binary classification** of multi-channel signals

CSP was chosen because it has been rigorously validated in EEG-based BCI systems for distinguishing between two mental states — analogous to our Normal vs. Pathological classification.

---

## Slide 13: Augmentation Challenges (SMOTE)

**Q32. Why did SMOTE fail for your problem?**  
**A.** SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples by drawing linear interpolations between nearest-neighbor points in **tabular feature space**. For time-series data, this:
1. **Destroys temporal structure:** Interpolating between two pathological waveforms creates "Frankenstein" traces with physiologically impossible transitions.
2. **Eliminates phase-delay:** The critical time-lag between a UC peak and an FHR trough gets averaged out.
3. **Confuses deep learning models:** The generated waves don't follow autoregressive temporal rules, introducing contradictory training signals.

We empirically verified this: SMOTE initially yielded 0.87 AUC (V3.0), but the model was learning artifacts — leading us to develop TimeGAN (V4.0) which achieved 0.8639 AUC with genuine temporal fidelity.

**Q33. If SMOTE gave higher AUC (0.87), why switch to TimeGAN (0.8639)?**  
**A.** The 0.87 AUC with SMOTE was **artificially inflated**. SMOTE creates feature-space interpolations that leak implicit class boundary information. When we validated rigorously with rank averaging and stacking ensembles in V4.0/V5.0, the honest AUC stabilized at 0.8639 with TimeGAN — which represents genuinely learned temporal patterns, not augmentation artifacts.

---

## Slide 14: Generative Methodology (TimeGAN)

**Q34. How does TimeGAN differ from a standard GAN?**  
**A.** Standard GANs (e.g., DCGAN) generate data in a single pass without temporal constraints. TimeGAN adds:
1. **Autoregressive mechanism:** Forces the generator to learn **step-by-step temporal transitions** (each timestep is conditioned on previous timesteps via GRU cells).
2. **Embedding + Recovery networks:** Learns a lower-dimensional representation of the temporal dynamics before generating in that latent space.
3. **Supervised loss component:** Explicitly penalizes the generator if it fails to predict the next timestep correctly.

This ensures synthesized FHR traces maintain realistic **sequential physiological dynamics** — deceleration curves develop gradually, not instantaneously.

**Q35. Why WGAN-GP specifically? Why not vanilla GAN?**  
**A.** Vanilla GANs suffer from two critical issues:
1. **Mode Collapse:** The generator memorizes a single "safe" output and repeats it (we'd get 1,410 identical traces).
2. **Training Instability:** Gradient explosions/vanishing during long (10,000 epoch) training cycles.

WGAN-GP (Wasserstein GAN with Gradient Penalty, λ=10) solves both by:
- Using the **Wasserstein distance** instead of Jensen-Shannon divergence (smoother loss landscape)
- Enforcing a **Gradient Penalty** instead of weight clipping (stable, well-behaved gradients)

**Q36. How do you verify that TimeGAN traces are physiologically realistic?**  
**A.** We validate through:
1. **Visual comparison:** Overlaying real vs. synthetic traces — synthetic traces show realistic deceleration shapes phase-locked to contraction peaks.
2. **Statistical similarity:** Comparing distribution statistics (mean, variance, autocorrelation) between real and synthetic pathological traces.
3. **Training diagnostics:** Monitoring discriminator loss convergence — the discriminator should struggle to distinguish real from fake (indicating high-quality synthesis).

---

## Slide 15: Core Architecture (AttentionFusionResNet)

**Q37. Why use a 1D-ResNet instead of an LSTM/RNN for time-series?**  
**A.** Three reasons:
1. **Parallelization:** ResNets process the entire sequence in parallel via convolutions, while LSTMs process sequentially (much slower).
2. **Long-range dependencies:** RNNs suffer from vanishing gradients over 1200 timesteps. ResNets use skip connections to maintain gradient flow.
3. **Proven effectiveness:** Spilka et al. (2016) demonstrated that 1D-ResNets at 1Hz sampling are highly robust for CTG classification, and our architecture builds directly on this validated approach.

We added **Squeeze-and-Excitation (SE) blocks** for channel recalibration and **Multi-Head Temporal Attention** for capturing global patterns, making it more powerful than a vanilla ResNet.

**Q38. What are Squeeze-and-Excitation (SE) Blocks?**  
**A.** SE blocks are channel attention mechanisms (Hu et al., CVPR 2018) that:
1. **Squeeze:** Global average pool across the temporal dimension → per-channel summary.
2. **Excite:** Two FC layers learn to recalibrate channel importance → sigmoid activation.
3. **Scale:** Multiply each channel by its learned importance weight.

This allows the ResNet to automatically learn which convolutional filters are most important for detection — effectively performing feature selection within the network.

**Q39. How many parameters does the model have?**  
**A.** Each full AttentionFusionResNet (with all three input branches and head) is approximately **27 MB** in float32 format. After Int8 quantization via TFLite, it compresses to approximately **1.9 MB**.

---

## Slide 16: Cross-Modal Attention (CMAF)

**Q40. How is CMAF different from simple concatenation?**  
**A.** In standard multimodal networks, different data streams are simply **concatenated** into a long vector before the final classifier. This treats all features equally regardless of clinical context.

CMAF performs **dynamic, context-aware fusion:**
- The FHR embedding serves as a *Query* — "what am I looking for?"
- The CSP embedding provides *Key/Value* — "where is the FHR-UC correlation evidence?"
- The Tabular embedding acts as a *Gate* — "given this patient's clinical profile, how should I weigh the evidence?"

**Example:** For a 28-week premature fetus (from tabular data), CMAF shifts attention weights to be hypersensitive to minor heart rate drops that would be normal at 40 weeks.

**Q41. How does the gating mechanism work mathematically?**  
**A.** The tabular features ($v_{tab}$) are passed through a sigmoid-activated dense layer producing a gate vector $g \in [0,1]^{128}$. This gate is element-wise multiplied with the attention output:
$$\text{output} = g \odot \text{Attention}(Q=v_{FHR}, K=v_{CSP}, V=v_{CSP})$$
When $g_i \approx 0$, the $i$-th attention dimension is suppressed. When $g_i \approx 1$, it is fully passed through. The gate values are learned end-to-end from the clinical context.

---

## Slide 17: Ensemble Strategy

**Q42. Why use three different models instead of a single deep model?**  
**A.** No single algorithmic architecture reliably generalizes across chaotic clinical noise. Each model captures different aspects:

| Model | Strength | Weakness |
| :--- | :--- | :--- |
| **AttentionFusionResNet** | Deep sequential morphology extraction | Can overfit on small tabular sets |
| **1D-InceptionNet** | Multi-scale STV/LTV simultaneous analysis | Less effective on long-range patterns |
| **XGBoost** | Extremely robust on structured tabular+CSP data | Cannot process raw sequential signals |

By stacking them, the meta-learner learns which model to trust for different types of traces.

**Q43. What is "stacking" and how does the meta-learner work?**  
**A.** Stacking is an ensemble technique where:
1. During 5-Fold CV, each model produces **Out-of-Fold (OOF) predictions** — predictions on data the model never saw during training.
2. These OOF predictions from all 3 models are collected into a matrix of shape `(N, 3)`.
3. A **Logistic Regression Meta-Learner** is trained on this matrix to learn the optimal weighted combination of the three models' opinions.

This is superior to simple majority voting or averaging because the meta-learner adaptively learns model-specific trustworthiness.

**Q44. What is "rank averaging" and why do you use it?**  
**A.** Raw probability outputs from different folds/models have different calibration scales (fold 1 might output probabilities in [0.2, 0.8] while fold 2 outputs [0.1, 0.9]). **Rank averaging** normalizes predictions by:
1. Ranking all predictions within each fold from 0 to 1.
2. Averaging the ranks (not the raw probabilities) across folds.

This eliminates inter-fold calibration bias and produces more stable, reliable ensemble predictions.

---

## Slide 18: Uncertainty Quantification (MC Dropout)

**Q45. How does MC Dropout work in practice?**  
**A.** During standard training, Dropout (p=0.4) is used to prevent overfitting. In MC Dropout, we **keep dropout active during inference**:
1. For each patient's 20-minute window, the model runs **T=20 forward passes**.
2. Each pass randomly drops different neurons, producing slightly different predictions.
3. The **mean** of 20 predictions = final probability estimate.
4. The **standard deviation (σ)** across 20 predictions = epistemic uncertainty.
5. If σ > 0.05 → flag as "UNCERTAIN: REQUIRES HUMAN REVIEW."

This is mathematically equivalent to **approximate Bayesian inference** (Gal & Ghahramani, 2016).

**Q46. Why 20 passes? Is that computationally expensive?**  
**A.** 20 passes is the empirically validated sweet spot:
- Fewer passes (e.g., 5) produce unstable uncertainty estimates.
- More passes (e.g., 100) add computational cost with diminishing returns.
- With 20 passes on the Keras model, inference takes ~1 second total per sample — acceptable for our clinical use case (decision is made over minutes, not milliseconds).

**Q47. What happens when the model is "uncertain"?**  
**A.** When σ² > 0.05:
- The dashboard displays an **amber "CONFIDENCE LOW: REQUIRES HUMAN REVIEW"** banner.
- The numerical probability is still shown but accompanied by the uncertainty range (e.g., "75% ± 12%").
- The system does **not** override the clinician — it explicitly defers to human judgment.
- This is critical for patient safety: the AI admits its limitations rather than guessing.

---

## Slide 19: Platt Scaling Calibration

**Q48. What is the difference between model confidence and calibrated probability?**  
**A.** A raw neural network output of "85%" is merely a geometric distance from a learned decision boundary — it does **not** mean 85 out of 100 similar patients are truly pathological. Platt Scaling applies a sigmoid mapping to these raw logits:
$$P_{\text{calibrated}} = \frac{1}{1 + e^{-(Aw + B)}}$$
where $A$ and $B$ are learned parameters that align model outputs with true class frequencies. After calibration, "85%" genuinely means ~85% of patients with that score are truly pathological.

**Q49. What is the Brier Score and ECE?**  
**A.**
- **Brier Score (0.0460):** Mean squared error between predicted probabilities and actual binary outcomes. Lower is better (0 = perfect, 1 = worst). Our 0.046 indicates near-ideal probability estimates.
- **Expected Calibration Error / ECE (0.0543):** Measures the average gap between predicted probabilities and observed frequencies across probability bins. Our 0.0543 indicates <6% average calibration error — meaning our probability estimates are highly trustworthy.

**Q50. How is Platt Scaling implemented?**  
**A.** We wrap the entire Stacking Ensemble inside scikit-learn's `CalibratedClassifierCV` with:
- **Method:** `sigmoid` (Platt Scaling)
- **Cross-Validation:** 5-Fold (ensuring the calibration mapping doesn't overfit)
- This is applied as the **final post-processing layer** after the meta-learner, ensuring all ensemble outputs are properly calibrated.

---

## Slide 20: Baseline Validation

**Q51. Why are baselines important? Can't you just show your final model's results?**  
**A.** Baselines are scientifically essential because they:
1. **Justify architectural complexity:** We prove that simpler approaches fail, validating the need for Tri-Modal fusion.
2. **Establish a fair benchmark:** All baselines use the **exact same dataset, splits, and CV strategy**, ensuring an apples-to-apples comparison.
3. **Frame the contribution:** Our improvement from 0.564 (FHR-only) to 0.8639 (Tri-Modal) demonstrates the precise value each modality adds.

**Q52. Why does the 1D-CNN (FHR Only) perform so poorly at 0.564 AUC?**  
**A.** 0.564 AUC is barely better than random (0.50). The 1D-CNN fails because:
- Without UC signals, it cannot detect **late decelerations** (the most dangerous pattern).
- Many FHR deceleration shapes look identical without contraction timing context.
- The model essentially learns to detect only extreme FHR anomalies (flat line, severe bradycardia), missing subtle pathological patterns.

This definitively proves FHR-only models are **clinically insufficient** and justifies our Tri-Modal approach.

---

## Slide 21: Deployment Optimization (TFLite)

**Q53. What is TFLite Int8 quantization?**  
**A.** TFLite Int8 quantization converts model weights from **32-bit floating point (FP32)** to **8-bit integers (Int8)**:
- Each weight goes from 4 bytes → 1 byte (4× compression).
- A **representative dataset** (300 calibration samples) determines the optimal int8 scaling factors.
- The I/O pipeline uses Int8 throughout for maximum performance on ARM CPUs and NPU/DSP accelerators.

**Q54. How much does quantization affect accuracy?**  
**A.** Int8 quantization retains **~99% of original float32 AUC**. The slight accuracy drop is clinically negligible because:
- We use a carefully selected representative dataset (300 training samples) to minimize quantization error.
- The ensemble's robustness (3 diverse models) absorbs minor individual model perturbations.

**Q55. What are the exact deployment specifications?**  
**A.**

| Specification | Value |
| :--- | :--- |
| **Model Size** | ~1.9 MB (down from 27 MB per fold) |
| **Target Hardware** | ARM CPU Android phones (₹5,000 / ~$60) |
| **Inference Time** | < 30 ms per prediction |
| **Internet Required** | No (fully offline) |
| **Privacy** | No patient data ever leaves the device |

---

## Slides 22–24: Key Novelties, Tech Stack & Conclusion

**Q56. What are the three key novelties of NeuroFetal AI?**  
**A.**
1. **Tri-Modal Deep Fusion:** First system to fuse FHR + UC + Tabular + CSP via Cross-Modal Attention (CMAF), breaking past the unimodal performance ceiling.
2. **TimeGAN Synthesis:** Replaced SMOTE with a temporal GAN that preserves phase-delay dynamics, generating 1,410 physiologically realistic pathological traces.
3. **Epistemic Safety + Edge AI:** Combined MC Dropout uncertainty quantification + Platt Scaling calibration within a 1.9 MB deployable edge payload — trustworthy AI that runs offline.

**Q57. What is completed at mid-semester vs. what remains?**  
**A.**
- ✅ **Completed:** Full data pipeline, Tri-Modal feature extraction (CSP/Tabular), TimeGAN synthesis (1,410 traces), core AttentionFusionResNet + CMAF coded, baseline validation.
- 🚀 **Remaining (End-Semester):** Final 5-Fold evaluation loops with TimeGAN, Brier-Score calibration verification, Grad-CAM XAI integration in Streamlit dashboard, live trace simulation demo.

**Q58. What technology stack are you using?**  
**A.**
- **Deep Learning Core:** Python 3.13, TensorFlow 2.14, Keras (Functional API)
- **Ensemble/Statistical:** Scikit-Learn 1.8.0, XGBoost 3.2.0
- **Signal Processing:** wfdb (PhysioNet parser), SciPy, NumPy, mne (CSP)
- **Deployment:** Streamlit (≥1.35.0), TensorFlow Lite, Pyngrok (ngrok tunneling)

---

# Part B: Report — Chapter-Wise FAQs

---

## Chapter 1: Introduction & Clinical Motivation

**Q59. What is the difference between intrapartum and antepartum monitoring?**  
**A.**
- **Antepartum:** Monitoring before the onset of labor (weeks before delivery). Uses non-stress tests (NST), biophysical profiles.
- **Intrapartum:** Monitoring **during active labor and delivery**. This is our focus because contractions create acute mechanical stress that can unmask previously hidden fetal compromise.

Our system specifically targets the intrapartum window because this is when fetal hypoxia most rapidly progresses to irreversible damage.

**Q60. What is the difference between aerobic and anaerobic metabolism in the fetus?**  
**A.**
- **Aerobic metabolism (normal):** The fetus uses oxygen from placental blood to produce ATP (energy) efficiently. Produces CO₂ and water as byproducts.
- **Anaerobic metabolism (compromised):** When oxygen is depleted, the fetus switches to glucose-only metabolism. This produces **lactic acid**, causing blood pH to drop (acidemia). It is also far less efficient (2 ATP vs. 36 ATP per glucose molecule), so the fetus exhausts energy reserves rapidly.

The pH < 7.15 threshold directly measures the severity of this anaerobic acid accumulation.

**Q61. What are the FIGO guidelines for CTG interpretation?**  
**A.** FIGO (International Federation of Gynecology and Obstetrics) provides standardized rubrics classifying CTG traces into:
1. **Normal:** Baseline 110–160 bpm, moderate variability, accelerations present, no significant decelerations → Continue monitoring.
2. **Suspicious:** Lacking at least one normal feature → Closer observation, possible additional testing.
3. **Pathological:** Severely abnormal features → Urgent intervention required.

Despite these guidelines, inter-observer agreement remains only 60–70%, which is why automated systems like ours are needed.

**Q62. What is "Defender's Bias" in obstetrics?**  
**A.** Defender's Bias is the clinical tendency toward **false-positive over-diagnosis** of fetal distress. Because the consequences of missing a genuine case are catastrophic (stillbirth), clinicians err heavily on the side of caution. This has directly contributed to a global surge in **unnecessary emergency cesarean sections**, which carry surgical risks for the mother (hemorrhage, infection, future pregnancy complications).

---

## Chapter 2: Problem Statement & Objectives

**Q63. Can you state your project's problem definition concisely?**  
**A.** *"How can a multi-modal deep learning system integrate volatile FHR time-series, synchronized UC signals, and static maternal clinical features to accurately detect intrapartum fetal compromise, given severe class imbalance (7.25% pathological), while providing uncertainty estimates, model explainability, and an offline edge-deployable footprint?"*

**Q64. How does Objective 5 (SOTA Validation) ensure scientific rigor?**  
**A.** Objective 5 mandates:
1. **Public dataset only** (CTU-UHB, 552 records) — anyone can reproduce and verify.
2. **Stratified 5-Fold CV** — no cherry-picked train/test splits.
3. **Fixed seed (42)** — deterministic reproducibility.
4. **OOF predictions** — every sample is predicted by a model that never saw it.
5. **Explicit baselines** — we prove improvement over defined reference points, not just report numbers in isolation.

---

## Chapter 3: Literature Survey & Gap Analysis

**Q65. What are the two "generational shifts" in automated CTG analysis?**  
**A.**
1. **First Generation (Classical ML, ~2006–2015):** Extracted morphological FIGO features (baseline, STV, LTV, deceleration counts) → fed to SVM, RF, LR. Discarded raw signal shape. **Ceiling: ~0.70–0.76 AUC.**
2. **Second Generation (Deep Sequence Models, ~2016–present):** Directly processed raw 1D FHR via CNNs, LSTMs. Captured signal morphology but ignored clinical context. **Ceiling: ~0.80 AUC.** Mendis (2023) pushed to 0.84 by adding tabular features.

NeuroFetal AI represents a **Third Generation** — Tri-Modal fusion with CSP, uncertainty quantification, and edge deployment.

**Q66. How did you validate the gaps you identified in the literature?**  
**A.** We didn't just cite weaknesses — we **actively implemented and benchmarked** the competing approaches:
- Built a 1D-CNN on FHR-only → 0.564 AUC (proving UC omission is catastrophic)
- Built LR on tabular → 0.676 AUC (proving linear models are insufficient)
- Built RF on tabular → 0.837 AUC (proving classical ML has a ceiling)

This empirical validation on the same dataset (CTU-UHB, same splits, same CV) provides irrefutable evidence that Tri-Modal fusion is necessary.

---

## Chapter 4: Dataset Description

**Q67. Why is pH < 7.15 used instead of pH < 7.05 or pH < 7.20?**  
**A.** The pH < 7.15 threshold represents **clinically significant fetal acidemia** — the point where neonatal morbidity risk substantially increases. It is the most widely used research threshold in the CTG classification literature. Note:
- pH < 7.20: Some studies use this more conservative threshold but it captures too many borderline cases.
- pH < 7.05: Used in some of our documentation (explain.md) — represents **severe** acidemia. Different documents may reference slightly different thresholds depending on the severity level being discussed.
- pH < 7.00: Represents critical/near-fatal acidemia.

**Q68. What is a tocodynamometer and how accurate is it?**  
**A.** A tocodynamometer is a pressure sensor placed on the maternal abdomen (uterine fundus) that measures the **relative intensity** of uterine contractions. Important limitations:
- It measures relative pressure, not absolute intrauterine pressure (unlike an intrauterine pressure catheter/IUPC).
- Readings are **position-dependent** — if the sensor shifts, amplitude changes.
- It reliably captures **contraction timing and frequency** (which is what our model primarily uses) but amplitude measurements can be noisy.
- Our preprocessing applies median baseline subtraction and amplitude normalization to mitigate these artifacts.

**Q69. How do you handle missing FHR data during probe disconnection?**  
**A.** Two-tiered approach:
1. **Gaps < 15 seconds:** Linearly interpolated (the physiological reality is that FHR doesn't change dramatically in 15 seconds, so interpolation is safe).
2. **Gaps ≥ 15 seconds:** Preserved as zero values. These represent genuine signal loss that we don't want to fabricate. The model learns during training that zero-valued segments carry no diagnostic information and should be ignored.

---

## Chapter 5: Advanced Feature Engineering

**Q70. What is Short-Term Variability (STV) and why is it important?**  
**A.** STV measures the **beat-to-beat fluctuations** in FHR (millisecond-scale differences between consecutive heartbeats). It reflects the integrity of the fetal autonomic nervous system:
- **Normal STV:** Indicates healthy neurological function and adequate oxygenation.
- **Reduced STV:** One of the earliest warning signs of fetal hypoxia — the autonomic nervous system is being suppressed.
- Clinically, reduced STV is a stronger predictor of acidemia than deceleration patterns alone.

**Q71. What is Long-Term Variability (LTV)?**  
**A.** LTV measures the **macroscopic oscillations** in FHR baseline over 3–5 minute periods (cycling between 110–160 bpm). It reflects the balance between the sympathetic and parasympathetic nervous systems:
- **Normal LTV:** 5–25 bpm oscillations → healthy neurological cycling.
- **Reduced LTV (< 5 bpm) or Sinusoidal pattern:** May indicate severe fetal anemia or terminal hypoxia.

**Q72. What is Approximate Entropy (ApEn) and why is it used?**  
**A.** ApEn is a **non-linear complexity measure** that quantifies the regularity/predictability of a time-series:
- **High ApEn:** Signal is complex and unpredictable → usually indicates a healthy, responsive fetal cardiovascular system.
- **Low ApEn:** Signal is regular and predictable → may indicate a failing autonomic system unable to dynamically respond to stress.

We include both ApEn and Sample Entropy (SampEn) as features because they capture information about fetal well-being that linear statistics (mean, variance) cannot.

---

## Chapter 6: Addressing Imbalance (TimeGAN)

**Q73. What is "majority-class collapse" and how does it manifest?**  
**A.** When 92.75% of training samples are "Normal," a neural network can achieve **92.75% accuracy** by simply predicting "Normal" for every single patient. This trivial strategy:
- Has 100% sensitivity for Normal class and 0% sensitivity for Pathological.
- Is clinically **useless** — it never detects the cases we actually care about.
- Is exactly what happens with standard cross-entropy loss on imbalanced datasets.

We combat this via: TimeGAN augmentation (3× minority class), Focal Loss (α=0.75, γ=2.5, pos_weight=5.0), and label smoothing (0.1).

**Q74. How does Focal Loss address class imbalance?**  
**A.** Focal Loss (Lin et al., 2017) modifies standard cross-entropy by adding a **modulating factor**:
$$FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$
- When the model is **confident and correct** (easy examples): $(1-p_t)^\gamma$ approaches 0, reducing the loss contribution → model stops wasting gradient on already-learned cases.
- When the model is **wrong** (hard examples, typically minority class): Loss remains large → gradients focus on improving these cases.
- **α = 0.75:** Increases loss weight for the minority (pathological) class.
- **γ = 2.5:** Aggressively down-weights easy negatives.
- **pos_weight = 5.0:** Additional 5× multiplier for positive class samples.

**Q75. What is Mode Collapse in GANs and how does WGAN-GP prevent it?**  
**A.** Mode Collapse occurs when the GAN generator "collapses" to producing only one or a few outputs that fool the discriminator, ignoring the diversity of the training data. For us, this would mean generating 1,410 nearly identical synthetic traces.

WGAN-GP prevents this by:
1. **Wasserstein distance:** Provides meaningful gradients even when distributions don't overlap (unlike JS divergence which saturates).
2. **Gradient Penalty (λ=10):** Enforces a 1-Lipschitz constraint on the discriminator, ensuring smooth, stable gradients throughout training.
3. This allows stable training over 10,000 epochs without collapse.

---

## Chapter 7: Proposed Architecture

**Q76. What is a Residual Block and why does it prevent vanishing gradients?**  
**A.** A Residual Block adds a **skip connection** (identity mapping) that bypasses one or more layers:
$$\text{output} = F(x) + x$$
Instead of learning the full mapping $H(x) = F(x)$, the network only needs to learn the **residual** $F(x) = H(x) - x$. This:
1. Provides a direct gradient pathway during backpropagation (gradients can flow through the skip connection unchanged).
2. Makes it easy to learn identity mappings (just set F(x) = 0) — deeper layers cannot hurt performance.
3. Enables training of very deep networks (our 6-block architecture spans 1200 timesteps).

**Q77. How does the Multi-Head Temporal Attention work?**  
**A.** After the 6 ResBlocks compress the 1200-timestep FHR signal into a feature map, **4-Head Temporal Attention** (Vaswani et al., 2017) is applied:
1. **Q, K, V** are all derived from the same feature map (self-attention).
2. **4 parallel attention heads** learn different temporal relationships simultaneously.
3. Each head computes: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
4. Outputs are concatenated and linearly projected → 128-dimensional representation.

This captures **global long-range dependencies** — e.g., recognizing that a deceleration at minute 3 and another at minute 17 constitute a recurring pathological pattern.

**Q78. How does the 1D-InceptionNet differ from the ResNet?**  
**A.** While ResNet uses fixed-size convolutional kernels, InceptionNet applies **three parallel kernel sizes simultaneously** (3, 5, 7):
- **Kernel 3:** Captures rapid, microsecond-scale STV beat-to-beat variations.
- **Kernel 5:** Captures medium-scale deceleration morphology.
- **Kernel 7:** Captures slow, macroscopic LTV baseline oscillations.

The outputs of all three scales are concatenated, giving the model a multi-resolution view of the FHR signal. This architectural diversity is precisely why it complements the ResNet in our ensemble.

---

## Chapter 8: Uncertainty & Calibration (V5.0)

**Q79. How is MC Dropout mathematically equivalent to Bayesian inference?**  
**A.** Gal & Ghahramani (2016) proved that training a neural network with Dropout and performing T stochastic forward passes at inference is equivalent to **Variational Inference with Bernoulli approximate distributions** over the network weights:
- Each forward pass with different dropped neurons effectively samples from an approximate posterior distribution $q(W)$.
- The mean prediction $\bar{y} = \frac{1}{T}\sum_{t=1}^{T}f^{W_t}(x)$ approximates the predictive mean.
- The variance $\sigma^2 = \frac{1}{T}\sum_{t=1}^{T}(f^{W_t}(x) - \bar{y})^2$ approximates the epistemic uncertainty.

This gives us Bayesian uncertainty quantification without the computational cost of true Bayesian neural networks.

**Q80. What are the Info-Theory metrics (Predictive Entropy, Mutual Information)?**  
**A.**
- **Predictive Entropy:** $H[\bar{y}] = -\bar{y}\log\bar{y} - (1-\bar{y})\log(1-\bar{y})$  
  Captures **total uncertainty** (epistemic + aleatoric). High entropy = high overall uncertainty.
- **Mutual Information:** $MI = H[\bar{y}] - \frac{1}{T}\sum_t H[y_t]$  
  Captures **epistemic uncertainty only** (model ignorance). High MI = model is unsure because of its own limitations (not data noise).

By reporting both, clinicians can distinguish between "this trace is genuinely ambiguous" (high aleatoric) vs. "the model hasn't seen enough similar cases" (high epistemic).

---

## Chapter 9: Baseline Evaluation & Metrics

**Q81. Why report AUC-ROC instead of Accuracy?**  
**A.** With 92.75% class imbalance, a model that predicts "Normal" for every patient achieves 92.75% accuracy — while being clinically useless. **AUC-ROC** is threshold-independent and measures the model's ability to **rank** pathological cases higher than normal cases across all possible thresholds. An AUC of 0.5 = random, 1.0 = perfect discrimination.

**Q82. What does the jump from 0.564 (1D-CNN) to 0.8639 (Ensemble) tell us?**  
**A.** The improvement quantifies the exact contribution of each architectural component:
- **0.564 → 0.837 (+0.273 AUC):** Adding tabular features via Random Forest adds massive clinical context.
- **0.837 → 0.8639 (+0.027 AUC):** Adding deep sequential processing (ResNet) + CSP + CMAF fusion breaks past the tabular ceiling.
- **0.840 → 0.8639 (+0.024 AUC):** Our system surpasses Mendis et al.'s SOTA (which used 9,887 private samples) using only 552 public samples — proving our architecture is more efficient.

---

## Chapter 10: Edge Deployment & XAI Plans

**Q83. What is Grad-CAM and how does it work for 1D signals?**  
**A.** Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., ICCV 2017) generates visual explanations by:
1. Computing the gradient of the predicted class score with respect to the **last convolutional layer** activations.
2. Global-average-pooling these gradients to obtain per-channel importance weights.
3. Computing a weighted combination of the activation maps → 1D heatmap over the 1200-sample input.

For 1D FHR signals, this produces a heatmap overlay showing **exactly which temporal segment** (e.g., minutes 12–14 where a late deceleration occurs) most influenced the "Pathological" prediction. Clinicians can visually verify: "Yes, that's exactly the late deceleration I was also concerned about."

**Q84. Why is offline deployment critical for this project's impact?**  
**A.** The highest burden of intrapartum fetal mortality is in rural LMICs where:
- There is **no internet connectivity** (eliminates cloud-based AI)
- There are **no GPU servers** (eliminates heavy Keras models)
- Only **cheap Android phones** (₹5,000/$60) may be available
- There are **no trained obstetricians** to interpret CTG traces

Our 1.9 MB TFLite model runs entirely offline, requires no cloud connection, executes in <30 ms on commodity ARM CPUs, and preserves patient privacy (no data ever leaves the device). This is what "Lab to Village" means — democratizing access to expert-level fetal monitoring.

---

## Chapter 11: Current Status & End-Semester Roadmap

**Q85. What exactly has been completed at mid-semester?**  
**A.** All foundational engineering and architectural work:
1. ✅ **Data Pipeline:** 552 CTU-UHB recordings → 2,546 windowed training matrices
2. ✅ **Feature Engineering:** 18 tabular features + 19 CSP vectors fully extracted
3. ✅ **TimeGAN:** WGAN-GP trained (10K epochs), 1,410 synthetic traces generated
4. ✅ **Architecture:** AttentionFusionResNet, CMAF, 1D-InceptionNet, XGBoost — all coded and locally verified
5. ✅ **Baselines:** 3 empirical baselines established (1D-CNN: 0.564, LR: 0.676, RF: 0.837)
6. ✅ **Streamlit Dashboard:** Working clinical UI with prediction + uncertainty visualization

**Q86. What is planned for the end-semester evaluation?**  
**A.** Phase 2 focuses on integration, validation, and deployment:
1. 🚀 **Full Training Loop:** Bind TimeGAN outputs live into 5-Fold CV (per-fold synthesis preventing data leakage)
2. 🚀 **Hyperparameter Optimization:** Cloud GPU sweep for final metrics
3. 🚀 **Platt Scaling + MC Dropout Verification:** Complete calibration pipeline with Brier Score and ECE validation
4. 🚀 **Grad-CAM Integration:** XAI heatmaps natively in the Streamlit dashboard
5. 🚀 **Live Demo:** Real trace simulation mimicking a genuine labor ward environment

---

# Part C: General / Cross-Cutting FAQs

---

**Q87. What is the clinical significance of your 0.8639 AUC?**  
**A.** An AUC of 0.8639 means that if we randomly select one pathological fetus and one normal fetus, the model correctly ranks the pathological case as higher risk **86.39% of the time**. In clinical context:
- This surpasses the SOTA (Mendis: 0.84 AUC) using only 552 public samples (vs. their 9,887 private samples).
- Combined with our 96.34% overall accuracy and 95.22% F1-score, the system provides highly reliable assistance.
- The Brier Score of 0.046 confirms the predicted probabilities accurately reflect real-world risk.

**Q88. Is this a diagnostic tool or a decision support system?**  
**A.** Strictly a **Clinical Decision Support System (CDSS)** — not a diagnostic device. Key distinctions:
- **No autonomous decisions:** All predictions are presented as advisory information.
- **Always defers to clinicians:** Especially when uncertainty is high.
- **No regulatory clearance:** Not CE/FDA/CDSCO approved. Research prototype only.
- **Cannot replace obstetricians:** Designed to augment, not replace, clinical judgment.

**Q89. What are the known limitations of your system?**  
**A.**
1. **Small dataset:** 552 records (single hospital, Czech Republic) — demographic generalization unvalidated.
2. **No prospective validation:** All evaluation is retrospective. Real-world performance may differ.
3. **UC signal quality dependency:** Tocodynamometer readings are position-dependent and noisy.
4. **pH threshold ambiguity:** Borderline pH cases (7.00–7.15) are inherently diagnostically ambiguous.
5. **Not regulatory-cleared:** Would require prospective clinical trials and ISO/IEC certification for real clinical use.

**Q90. How does your project compare to existing commercial CTG analysis systems?**  
**A.** Commercial systems (e.g., K2 Guardian, GE Maternal-Infant Care) typically:
- Use rule-based FIGO feature extraction (not deep learning)
- Analyze FHR only (no UC signal fusion)
- Output categorical labels (Normal/Suspicious/Pathological) without probability calibration
- Do not provide uncertainty quantification
- Require expensive proprietary hardware

NeuroFetal AI advances beyond these by offering tri-modal deep fusion, calibrated probabilities, uncertainty awareness, explainability via Grad-CAM, and runs on ₹5,000 phones offline.

**Q91. What is the role of each team member?**  
**A.**
- **Krishna Sikheriya (IIT2023139):** Lead Developer & AI Architect — model architecture design, training pipeline, ensemble integration.
- **Bodkhe Yash Sanjay (IIT2023180):** Data Engineering & Backend — data ingestion, feature extraction, TimeGAN implementation.
- **Lokesh Bawariya (IIT2023138):** Frontend & Visualization — Streamlit dashboard, Grad-CAM visualization, deployment pipeline.

**Q92. What would you do differently if you had more time or resources?**  
**A.**
1. **Multi-center validation:** Test on Oxford/Edinburgh CTG databases to assess demographic generalization.
2. **Prospective clinical pilot:** Deploy in a controlled hospital setting with concurrent expert evaluation.
3. **Real-time streaming:** Process live CTG feeds (currently limited to 20-min fixed windows).
4. **Evidential Deep Learning:** Explore single-pass uncertainty (Dirichlet prior) as a faster alternative to MC Dropout.
5. **Federated Learning:** Train across hospitals without sharing patient data.

**Q93. Why not use a Transformer / Foundation Model directly?**  
**A.** We considered this but concluded:
1. **Data volume:** 2,546 samples is insufficient for training a Transformer from scratch (they need orders of magnitude more data).
2. **Inductive bias:** 1D-ResNets with convolutions have strong inductive bias for local pattern detection (decelerations), which is appropriate for physiological signals.
3. **We do use attention:** Our Multi-Head Temporal Attention and CMAF layers incorporate the Transformer's core mechanism (Q-K-V attention) within a CNN backbone, getting the best of both worlds.
4. **Computational cost:** Full Transformers would be too heavy for our 1.9 MB edge deployment target.

**Q94. How do you ensure there is no data leakage in your pipeline?**  
**A.** Multiple safeguards:
1. **Stratified 5-Fold CV:** Each fold is split at the **patient level** (not window level) — all windows from one patient are in the same fold.
2. **TimeGAN augmentation:** Applied **only to training folds**, never to validation.
3. **CSP fitting:** CSP spatial filters are computed **inside each fold** using only training data.
4. **Feature extraction:** Tabular features are extracted independently per window before splitting.
5. **OOF predictions:** Every sample is predicted by a model that never saw it during training.

**Q95. What is the reproducibility guarantee of your results?**  
**A.** Our results are fully reproducible:
- **Fixed random seed:** 42 across all stochastic operations (Python, NumPy, TensorFlow, scikit-learn)
- **Public dataset:** CTU-UHB is freely available on PhysioNet
- **Open-source code:** All scripts, model definitions, and training configurations are in our GitHub repository
- **Stratified 5-Fold CV:** Deterministic splits with the same seed produce identical folds
- **Documented hyperparameters:** Every training parameter is recorded in `Project_Context.md`

---

> **All the best for the mid-semester evaluation! 🚀**

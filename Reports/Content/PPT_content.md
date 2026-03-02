# NeuroFetal AI: Mid-Semester Presentation (PPT Content)
The following outline provides comprehensive text and speaker notes for a 20+ slide pitch deck detailing the project.

---
### **Slide 1: Title Slide**
**Title:** NeuroFetal AI: A Tri-Modal Architecture for CTG Classification
**Subtitle:** Mid-Semester Evaluation
**Team:** Krishna Sikheriya, Bodkhe Yash Sanjay, Lokesh Bawariya
**Supervisor:** Dr. Nikhilanand Arya (Research Supervisor)
**Institution:** Indian Institute of Information Technology, Allahabad

---
### **Slide 2: Introduction**
**Body:**
- Intrapartum fetal monitoring relies globally on Cardiotocography (CTG).
- Two Simultaneous Signals recorded:
  - **FHR**: Fetal Heart Rate via Doppler ultrasound.
  - **UC**: Uterine Contractions via a pressure tocodynamometer.
- ~2.6 million instances of stillbirths occur globally every year, with Fetal Compromise (hypoxia/acidosis) being a primary trigger.

---
### **Slide 3: Problem Statement**
**Body:**
- Extracting meaning from CTG paper strips is visually demanding and highly flawed.
- **Inter-Observer Disagreement**: Different obstetricians looking at the exact same trace disagree 30-40% of the time.
- **Alert Fatigue**: Conventional automated threshold systems generate intense false-positive alarms.
- **Real-World Impact**: This subjectivity drives up unnecessary surgical interventions or tragically delays critical action in low-resource settings.

---
### **Slide 4: Literature Review (10+ Papers Analyzed)**
**Body:**
- **Scope of Review**: We comprehensively analyzed over 10 recent research papers spanning classical ML and modern Deep Learning applied to CTG analysis.
- **Classical ML Approaches (e.g., Spilka et al., 2014)**: Early work focused heavily on manually extracting morphological features. (Classifiers like SVMs and Random Forests peaked at ~0.76 AUC).
- **Deep Learning Era (e.g., Petrozziello et al., 2019)**: 1D Convolutional Neural Networks and LSTMs bypassed manual feature extraction, processing the FHR wave directly (achieving ~0.80 AUC).
- **Recent SOTA (Mendis et al., 2023)**: Pioneered multimodal analysis, fusing a 1D-ResNet for FHR and a Dense Network for Maternal Tabular data to achieve a 0.84 AUC.

---
### **Slide 5: Identified Gaps**
**Body:**
- **UC Signal Omission**: Mendis et al. completely discarded the uterine contraction channel, ignoring the vital FHR–contraction temporal delay necessary for spotting late decelerations.
- **No Uncertainty Quantification**: Prior models provided deterministic point predictions, which is medically dangerous when an AI encounters a trace it doesn't recognize.
- **Dependence on Private Data**: State-of-the-Art models (0.84 AUC) were validated on massive, closed corporate datasets (9,887 cases), making them non-reproducible and potentially biased.
- **Hardware Limitations**: Deep learning models historically require high-end GPUs, making them useless in rural, low-resource labor wards.

---
### **Slide 6: Our Solution – NeuroFetal AI**
**Body:**
- **Tri-Modal Deep Fusion**: Processes FHR sequences, UC contraction rhythms, and static Maternal details actively.
- **Uncertainty Aware Engineering**: Designing systems to calculate concrete Clinical Confidence metrics. 
- **TimeGAN Implementation**: Synthesizing authentic pathological traces to fight heavy dataset skew.
- **Edge Deployable Architecure**: Blueprinting for offline processing via an Int8 Quantized TFLite payload.

---
### **Slide 7: CTU-UHB Benchmark Dataset**
**Body:**
- Public PhysioNet Database (Czech Republic).
- **552 Strict Inclusion Patients**: Singleton, >36 weeks gestation.
- **Ultimate Ground Truth Truth**: Arterial Cord Blood pH values (< 7.05 represents True Compromise).
- **The Imbalance Hurdle**: Only ~7.25% of the data reflects true pathological/hypoxic scenarios. 

---
### **Slide 8: Robust Preprocessing Pipeline**
**Body:**
- **Filtering**: Median baseline subtraction on heavily unstable UC signals.
- **Interpolation**: Dropout windows under 15 seconds linearly bridged.
- **Temporal Alignment**: Resampling down to 1 Hz. 
- **Window Strategy**: Sliding 20-minute windows operating with a 50% 10-minute stride (Expanded 552 recordings to 2,760 independent segments).

---
### **Slide 9: Advanced Feature Engineering (35 Total)**
**Body:**
- **Demographic (3)**: Age, Parity, Gestation limit.
- **Heart/UC Metrics (13)**: Baseline values, Short Term Variability, Deceleration counts, approximate entropy.
- **Common Spatial Patterns (CSP - 19)**: BCI technique adopted to analyze the variance filter interactions exclusively between FHR and UC curves. 

---
### **Slide 10: Addressing Data Starvation: SMOTE vs TimeGAN**
**Body:**
- **Phase 3 (Legacy)**: Attempted SMOTE. Flat tabular balancing destroys all active temporal relationships (late decelerations).
- **Phase 4 (Current)**: Swapped to a **Time-Series Generative Adversarial Network**.
- Utilized a WGAN-GP 1D Convolution framework to maintain physiologically feasible waveform delay dynamics.
- Resulted in 1,410 synthetic additions.

---
### **Slide 11: Base Model A: AttentionFusionResNet**
**Body:**
- The foundational anchor of the project.
- **Structure**: 6 parallel cascading residual blocks injected with Squeeze-and-Excitation channel balancers.
- Capable of modeling long term wave mechanics across 1,200 positional units. 
- Collapsed via Global Average Pooling into a tight unified signature vector.

---
### **Slide 12: Groundbreaking: Cross-Modal Attention Fusion (CMAF)**
**Body:**
- Instead of basic concatenation, we force the network to apply clinical logic.
- Calculates scaled dot-product Attention using FHR as queries and CSP metrics as Keys/Values.
- **The Gating Interface**: The 16 Tabular features output a deterministic active sigmoid map, telling the network which spatial patterns to ignore or penalize strictly based on the mother’s personal physiological age or parity.

---
### **Slide 13: Stacking Ensemble Strategy**
**Body:**
- A single architecture rarely generalizes optimally across extreme medical noise.
- We run 3 parallel pipelines:
  - Model A: AttentionFusionResNet (Deep)
  - Model B: 1D-InceptionNet (Multi-scale kernel sweeps)
  - Model C: XGBoost (Gradient tracking on Tabular + CSP sets)
- Fed into an inclusive Meta-Learner via Rank-Averaged Out-of-Fold merging.

---
### **Slide 14: Overcoming Metric Illusion: Calibration (V5.0)**
**Body:**
- Standard Neural Networks are frequently highly inaccurately confident.
- Applied **Platt Scaling** logic post-ensemble processing.
- Ensured continuous output scores directly mirror verified physiological probability thresholds.
- Result: A Brier Score sitting remarkably low at 0.0460.

---
### **Slide 15: Deep Epistemic Uncertainty via MC Dropout**
**Body:**
- When a doctor isn’t sure, they consult another doctor. Our model mimics this.
- Target network pathways are randomly blacked out inside live assessment ($p=0.3$).
- The system executes 20 repetitive isolated inference walks.
- High operational variance ($\sigma^2$) directly triggers an "Ambiguous Zone" alarm.

---
### **Slide 16: Future Proofing: 'Lab to Village' Edge Processing**
**Body:**
- Cloud reliance breaks down in third-world rural hospital complexes. 
- **Our Goal**: Apply active **TensorFlow Lite Full Integer Quantization** prior to deployment.
- This will compress massive parallel deep learning architectures straight into an Int8 execution bundle.
- **Why?** To ensure the entire AI suite performs efficiently on low-cost $60 commodity android CPUs without *any* cellular connection.

---
### **Slide 17: Proposed Interface: Clinical Dashboard (Streamlit)**
**Body:**
- A fully reactive Dark-Mode specific UI tailored directly to strict labor ward lighting standards.
- Designed so clinicians can drag-and-drop raw `.dat` or CSV signal inputs natively.
- Features are calculated on the loop exactly parallel with the AI inference layer. 
- Will integrate active diagnostic "progress tracking" wheels corresponding to the calibrated safety confidence.

---
### **Slide 18: Future Work: Explainable AI mapping (Grad-CAM)**
**Body:**
- Doctors do not trust black boxes. In Phase 2, we implement localized Explainable AI.
- Generating physical color displacement heatmaps linked straight down to the trailing network convolution layers. 
- This will specifically permit obstetricians utilizing our Dashboard to visualize the exact 60-second window (the precise heart-rate dip or anomaly) that activated the machine learning's physiological distress alerts.

---
### **Slide 19: End-Sem Evaluation Plan (Phase 2)**
**Body:**
- The final metric evaluation will be executed using rigorous **Stratified 5-Fold Cross Validation** (preventing data leakage).
- **Objective Benchmark**: The public CTU-UHB dataset limits. 
- **The Baseline to Beat**: Mendis et al. (0.84 AUC).
- We will generate and chart Accuracy, AUC, and F1-Scores for the base models vs the meta-learning stacked ensemble. 

---
### **Slide 20: Mid-Semester Current Status**
**Body:**
- **Completed Steps**: Data Ingestion Pipeline mapped out, Physiological baseline and CSP processing completed, 1D ResNet architecture coded, TimeGAN augmentation WGAN-GP training successfully tested. 
- **Current Bottlenecks**: Merging massive multi-channel tensor spaces accurately through the CMAF attention loops over lengthy computational timeframes.
- **Status Summary**: On track. Core architecture design is solidified; shifting into intensive training and compilation phase.

---
### **Slide 21: Conclusion & Project Trajectory**
**Body:**
- Visual CTG evaluation represents a dangerous and highly subjective diagnostic flaw.
- NeuroFetal AI presents an architecturally complete blueprint designed specifically to bridge this gap intelligently.
- By moving to Phase II (Execution, Testing, Calibration, & Edge App Delivery), we aim to deliver an uncertainty-aware, hardware-optimized safeguard setting a new benchmark for fetal monitoring accessibility.

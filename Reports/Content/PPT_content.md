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
### **Slide 2: Clinical Motivation – The Global Tragedy**
**Body:**
- ~2.6 million instances of stillbirths occur globally every single year.
- Fetal Compromise (hypoxia/acidosis) is a primary trigger.
- The burden disproportionately hits low-resource geographic setups.
- **The Need**: Early intervention gives doctors the time necessary to perform successful emergency instrumental deliveries.

---
### **Slide 3: Cardiotocography (CTG) – The Clinical Standard**
**Body:**
- CTG is the frontline method for fetal monitoring during labor.
- Two Simultaneous Signals recorded:
  - **FHR**: Fetal Heart Rate via Doppler ultrasound.
  - **UC**: Uterine Contractions via a pressure tocodynamometer.
- **Goal**: Doctors analyze complex timing between the contraction peak and the heartrate nadir.

---
### **Slide 4: The Flaws in Subjective Analysis**
**Body:**
- Extracting meaning from CTG paper strips is visually demanding and highly flawed.
- **Inter-Observer Disagreement**: Different obstetricians looking at the exact same trace disagree 30-40% of the time.
- **Alert Fatigue**: Conventional automated threshold systems generate intense false-positive alarms.
- **Result**: Excessive unnecessary C-section surgeries, or tragically missed diagnoses.

---
### **Slide 5: Limitations of Current AI (Gap Analysis)**
**Body:**
- **Unimodal Limitation**: Prior neural networks (ResNets, 1D-CNNs) ignore the UC string entirely, looking only at heart rates.
- **Private Dependencies**: Heavy baseline papers (Mendis et al., 0.84 AUC) rely on 9,800+ closed corporate traces.
- **Black-Box AI**: Lack of uncertainty metrics makes current predictions medically unsafe to trust implicitly.

---
### **Slide 6: Our Solution – NeuroFetal AI**
**Body:**
- **Tri-Modal Deep Fusion**: Processes FHR sequences, UC contraction rhythms, and static Maternal details actively.
- **Uncertainty Aware**: Calculates concrete Clinical Confidence. 
- **TimeGAN Implementation**: Synthesizes authentic pathological traces.
- **Edge Deployable**: Offline processing via an Int8 Quantized TFLite payload running at 1.9MB.

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
### **Slide 16: 'Lab to Village' Edge Processing**
**Body:**
- Cloud reliance breaks down in third-world rural hospital complexes. 
- Ran active **TensorFlow Lite Full Integer Quantization**.
- Compressed massive 27MB parallel architectures straight into an Int8 1.9MB execution bundle.
- Validated to perform flawlessly in <30ms on 5,000 RS commodity android CPUs without any cellular connection.

---
### **Slide 17: Clinical Dashboard Implementation (Streamlit)**
**Body:**
- Fully reactive Dark-Mode specific UI tailored to strict labor ward lighting standards.
- Clinicians drop raw `.dat` or CSV inputs natively.
- Features are calculated on the loop exactly parallel with TFLite interpretation thresholds. 
- Integrated diagnostic "progress tracking" wheels corresponding to calibrated safety confidence.

---
### **Slide 18: Explainable AI mapping (Grad-CAM)**
**Body:**
- Unpacking the black box.
- Generating physical color displacement heatmaps directly linked down to the absolute trailing network convolution layers. 
- Specifically allows obstetricians to visualize the literal 60-second window gap that activated the machine learning's physiological distress alerts.

---
### **Slide 19: Final Validated CTU-UHB Outcomes**
**Body:**
- Verified utilizing strict Stratified 5-Fold evaluation (Seed: 42).
- **Public Baseline (Mendis)**: 0.84 (Relying strictly on 9,800 private inputs).
- **NeuroFetal AI V4.0 (TimeGAN)**: 0.8639 AUC.
- **NeuroFetal AI V5.0 (Calibrated TimeGAN Ensemble)**: **96.34% Accuracy / 95.22% F1 / 0.046 Brier Score**.
- Fully surmounts state-of-the-art standards actively utilizing accessible public frameworks.

---
### **Slide 20: Development Timeline & Project Status**
**Body:**
- **Completed Steps**: Data Pipeline, ResNet implementation, CSP integration, TimeGAN, Stacking Meta-Ensemble, TFLite Optimization. 
- **Current Live Status**: Dashboard V4 completed; testing live uncertainty rendering bounds.
- **Pending (End-Sem Goals)**: Formal Android apk wrapper buildout; potentially linking active bluetooth sensor streams for hardware integration validation. 

---
### **Slide 21: Conclusion**
**Body:**
- NeuroFetal AI successfully closes the dangerous discrepancy gap inherent in visual CTG evaluation.
- Delivers an uncertainty-aware, hardware-optimized safeguard against subjective misdiagnosis. 
- Sets a brand new public evaluation standard aiming to universally reduce intrapartum stillbirth ratios.

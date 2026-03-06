# NeuroFetal-AI: Mid-Semester Evaluation Presentation (PPT Content)
The following outline provides comprehensive text and speaker notes for a **20-slide academic pitch deck**, structured for a formal panel evaluation (15+ mins). It is balanced for a 3-member team presentation.

**Critical Note:** This presentation is restricted to showcasing the *methodology, data engineering, architecture, and baseline validation*. The final V5.0 ensemble empirical results (0.864 AUC) are deliberately omitted, as this is a Mid-Semester review.

---

## Part 1: Clinical Problem & Biological Foundation (Speaker 1: Krishna)

### **Slide 1: Title Slide**
*(Visual: Very clean white/slate design. "NeuroFetal-AI" prominent at the center. Clean fonts like Inter or Roboto.)*
* **Title:** NeuroFetal-AI: A Tri-Modal Architecture with Cross-Modal Attention for CTG Classification
* **Subtitle:** Mid-Semester Project Evaluation
* **Team:** Krishna Sikheriya | Yash Sanjay Bodkhe | Lokesh Bawariya
* **Institution:** Indian Institute of Information Technology, Allahabad

**🗣️ Speaker 1 (Krishna) Notes:**  
"Good morning, esteemed panel members. Today, our team is proud to present NeuroFetal-AI, an ongoing clinical decision support system designed to automate the detection of fetal compromise during labor using advanced multi-modal deep learning."

---

### **Slide 2: Background**
*(Visual: Left: Minimalist bullet points. Right: An image or map summarizing the global burden.)*
* **Heading:** Background
* **The Scale:** ~2.6 million babies are stillborn globally every year.
* **The Cause:** A significant driver is undetected intrapartum fetal compromise—progressive hypoxia and metabolic acidosis under the mechanical stress of labor.
* **The Vulnerability:** The overwhelming burden falls heavily on under-resourced rural clinics lacking continuous access to trained obstetric specialists.

**🗣️ Speaker 1 Notes:**  
"To understand our motivation, we must look at the clinical reality. Every year, 2.6 million stillbirths occur globally, largely driven by undetected progressive fetal hypoxia—where a baby runs out of oxygen during the mechanical stress of labor. Our system is designed specifically to act as an automated safeguard against this."

---

### **Slide 3: Clinical Standard**
*(Visual: A clean graphic showing the two sensors on a maternal abdomen leading to a dual-trace monitor.)*
* **Heading:** Clinical Standard
* **The Technology:** Cardiotocography (CTG) has been the global standard since the 1970s.
* **Dual Simultaneous Streams:**
    * **FHR (Fetal Heart Rate):** Measured via Doppler Ultrasound (capturing autonomic nervous system stress).
    * **UC (Uterine Contractions):** Measured via a Tocodynamometer (capturing the physical compressive force applied to the fetus).

**🗣️ Speaker 1 Notes:**  
"The current standard for monitoring this is Cardiotocography (CTG). It utilizes dual abdominal sensors to simultaneously plot the baby's heart rate against the mother's uterine contractions. Understanding the interaction between these two curves is the foundation of obstetric diagnostics."

---

### **Slide 4: Problem Statement**
*(Visual: A complex, noisy CTG paper trace highlighting how difficult it is to read.)*
* **Heading:** Problem Statement
* **The Limitation:** Extracting diagnostic meaning from analog paper traces is highly subjective and visually demanding.
* **Expert Disagreement:** Studies show that expert clinicians analyzing the *exact same trace* disagree 30% to 40% of the time.
* **The "Defender's Bias":** The fear of a fatal miss leads to extreme alert-fatigue, driving massive worldwide spikes in unnecessary emergency Cesarean sections.

**🗣️ Speaker 1 Notes:**  
"However, CTG analysis is deeply flawed. Extracting meaning by eye is subjective. The literature proves that experts disagree up to 40% of the time when reading the exact same strip. This lack of automated, objective consensus directly leads to unnecessary C-sections or tragic delays in true emergency intervention."

---

### **Slide 5: Research Objectives**
*(Visual: 4 minimalist chevron arrows pointing forward.)*
* **Heading:** Research Objectives
* **1. Tri-Modal Engineering:** To mathematically fuse raw 1D signals, clinical tabular data, and Common Spatial Patterns (CSP).
* **2. Overcoming Imbalance:** To utilize Generative AI (GANs) to synthesize strictly time-delayed minority class pathological traces.
* **3. Epistemic Safety:** To enforce concrete uncertainty-quantification bounds preventing dangerous "black-box" clinical guesses.
* **4. Edge Democratization:** To compress heavy cloud models into offline, Android-deployable payloads for rural access.

**🗣️ Speaker 1 Notes:**  
"Therefore, our project mapped out four strict computational objectives: Engineering a tri-modal fusion network, overcoming massive clinical data starvation via Generative AI, enforcing mathematical uncertainty bounds so the AI knows when it is confused, and ultimately compressing this system into an offline edge payload."

---

### **Slide 6: Literature Review (Historical Progression)**
*(Visual: A clean horizontal timeline graphic showing the evolution from visual to deep learning.)*
* **Heading:** Literature Review: Evolution of CTG Analysis
* **1970s - 2000s (Visual Heuristics):** Reliance entirely on human interpretation of paper traces (FIGO guidelines). Characterized by high subjective error rates.
* **2010s (Classical Machine Learning):** Introduction of Support Vector Machines (SVMs) and Random Forests extracting basic time-domain features. Plagued by poor generalization.
* **2020+ (Deep Sequential Learning):** The shift to 1D-CNNs and LSTMs. Systems began learning directly from raw waveforms but struggled heavily with the 7% class imbalance.

**🗣️ Speaker 1 Notes:**  
"Before designing our architecture, we analyzed the historical progression of CTG analysis. For decades, the field relied solely on flawed visual heuristics. In the 2010s, classical machine learning was introduced, but it failed to generalize. Recently, the focus shifted to deep sequential learning, which improved raw wave extraction but hit a hard ceiling due to extreme data starvation."

---

### **Slide 7: State-of-the-Art (SOTA) Comparative Analysis**
*(Visual: A clean, 3-column comparative table highlighting Mendis et al. against older methods.)*
* **Heading:** Analyzing the Modern Benchmark

| Research Phase | Core Architecture | Key Limitation |
| :--- | :--- | :--- |
| **Traditional CNNs (2019)** | Purely 1D FHR Sequences | Ignored maternal clinical history entirely. |
| **Early Multimodal (2021)** | Simple FHR + Static Variables | Failed to capture cross-modal temporal dependencies. |
| **Mendis et al. (2023)** | CNN + Tabular Stacking (0.84 AUC) | Deliberately removed the Uterine Contraction channel. |

**🗣️ Speaker 1 Notes:**  
"This brings us to the modern State-of-the-Art comparative analysis. The most impressive contemporary baseline was established by Mendis et al. in 2023. They achieved a strong 0.84 AUC by fusing Fetal Heart Rates with Tabular data. However, as shown in the table, our team identified a massive, fundamental flaw in their algorithm: the deliberate deletion of the contraction channel."

---

### **Slide 8: Identified Algorithmic Gaps**
*(Visual: Two large "warning" style boxes or highlighted text blocks breaking down the two specific gaps.)*
* **Heading:** Identified Algorithmic Gaps
* **Gap 1: The Omitted Contraction (Physiological Failure):**
    * Detecting a fatal "Late Deceleration" dynamically requires measuring the *time-delay phase shift* between the Heart Rate trough and the Uterine Contraction peak. Deleting the UC curve makes this mathematically impossible.
* **Gap 2: Point Predictions (Clinical Safety Failure):**
    * Current SOTA models output solely deterministic, "confident" probabilities even on heavily uncalibrated, out-of-distribution noise, offering zero Epistemic Safety to the physician.

**🗣️ Speaker 1 Notes:**  
"We recognized two foundational gaps in the current SOTA. First, omitting the contraction channel is physiologically dangerous; you cannot identify a late deceleration without knowing when the contraction occurred. Second, these current models suffer from overconfidence. They provide single-point deterministic predictions without giving the clinician any measure of mathematical uncertainty. I will now hand over to Yash to explain how we engineer the data to solve this."

---

## Part 2: Data Engineering & Generative Synthesis (Speaker 2: Yash)

### **Slide 9: Dataset**
*(Visual: Split slide displaying CTU-UHB stats and the extreme imbalance ratio pie chart.)*
* **Heading:** Dataset
* **The Source:** PhysioNet open-access clinical database (Czech Republic) - 552 Patients.
* **Objective Ground Truth:** Unlike subjective medical labels, our target is the post-delivery umbilical cord arterial Blood pH ($\text{pH} < 7.15$ = True Pathology).
* **The Imbalance Hurdle:** 
    * Normal Instances: ~92.75%
    * True Pathological Instances: ~7.25%

**🗣️ Speaker 2 (Yash) Notes:**  
"Thank you. Our foundation is the open-access CTU-UHB dataset. We selected this heavily because its labels are not based on human opinion, but absolute biochemical reality: cord blood pH values beneath 7.15. Our primary engineering bottleneck, however, is extreme real-world class imbalance, with only 7.2% of the dataset representing true pathology."

---

### **Slide 10: Preprocessing**
*(Visual: Insert the `preprocessing_flowchart.png` plot here, showing the 4-step horizontal pipeline.)*
* **Heading:** Preprocessing
* **Artifact Interpolation:** Missing probe data under 15 seconds is linearly bridged; larger gaps are strictly zero-masked.
* **Temporal Alignment & Filtering:** Median baseline subtractions on noisy UC signals and downsampled to 1 Hz.
* **Sliding Window Strategy:** Using a 20-minute window with a 10-minute stride—expanding 552 patient recordings into exactly 2,546 localized training matrices.

**🗣️ Speaker 2 Notes:**  
"Raw intrapartum signals are incredibly noisy. Our preprocessing interpolates small sensor dropouts, normalizes amplitudes, and downsamples the data to 1 Hertz. Crucially, we apply a sliding window technique, multiplying our 552 patient traces into over 2,500 independent 20-minute sequence matrices for neural network ingestion."

---

### **Slide 11: Feature Engineering**
*(Visual: 3 distinct data streams feeding into a central "NeuroFetal" hub.)*
* **Heading:** Feature Engineering
* Deep Neural Networks scale poorly on pure unstructured biological noise. We transform the data into three separate mathematical modalities:
* **Modality 1 (Temporal):** The localized 1D raw sequences of the FHR and UC.
* **Modality 2 (Tabular):** 18 discrete static metrics (e.g., Maternal Parity, Gestation) alongside computed dynamic signal statistics (LTV, STV, Entropy).
* **Modality 3 (CSP):** 19 Common Spatial Patterns matrices extracting non-linear physiological variance.

**🗣️ Speaker 2 Notes:**  
"Standard neural nets fail on pure noise. To fix this, we engineered a tri-modal matrix. First, the 1D sequential wave. Second, 18 distinct tabular variables, including the mother’s age and physiological entropy. Third, 19 Common Spatial Pattern vectors."

---

### **Slide 12: Specific Algorithm (CSP)**
*(Visual: A conceptual diagram showing two waves combining into a geometric plane representation.)*
* **Heading:** Specific Algorithm (CSP)
* **The Clinical Requirement:** Finding the exact intersection between a heart rate drop and a uterine squeeze.
* **The CSP Approach:** Borrowed directly from Brain-Computer Interface research.
* **The Implementation:** Projects the 2-Channel (FHR+UC) matrix into a new geometric plane that computationally maximizes the variance disparity strictly between healthy and hypoxic fetuses.

**🗣️ Speaker 2 Notes:**  
"I want to highlight CSP specifically. Borrowed from Brain-Computer Interface technology, we treat the FHR and Contraction signals as a dual-channel matrix. CSP projects this data onto a new plane, mathematically forcing our models to recognize the exact cross-correlated variance between a physiological contraction and a delayed fetal heart rate collapse."

---

### **Slide 13: Augmentation Challenges**
*(Visual: Conceptual graph showing SMOTE interpolating an impossible "middle" wave between two different decelerations. *Note: Can use the script generated plot*.)*
* **Heading:** Augmentation Challenges
* **Phase 3 (Legacy):** We initially implemented SMOTE (Synthetic Minority Oversampling).
* **The Flaw:** SMOTE draws static geometric lines in tabular space to balance classes.
* **Biological Destruction:** This completely obliterates the sequential time-series structure, generating "Frankenstein" waves that destroy the physiological phase-delay required for diagnosis.

**🗣️ Speaker 2 Notes:**  
"To solve the 7% class imbalance, we initially attempted SMOTE. The problem is that interpolating tabular points destroys the sequential structure of physiological waves. SMOTE basically averages out the critical time-delay between contractions and decelerations, generating waves that are biologically impossible and confusing the deep learning models."

---

### **Slide 14: Generative Methodology**
*(Visual: Clear TimeGAN architecture blocks: Generator, Discriminator with GRU layers.)*
* **Heading:** Generative Methodology
* **The Architecture:** Swapped SMOTE for a 1D Convolutional WGAN-GP.
* **Autoregressive Constraints:** Forces the generator to learn and respect step-by-step temporal transitions over massive 10,000-epoch cycles.
* **The Output:** Successfully synthesized 1,410 physiologically authentic pathological traces.

**🗣️ Speaker 2 Notes:**  
"We completely scrapped SMOTE and architected a Time-Series GAN. Using a Wasserstein-GAN with a Gradient Penalty, we actively forced a generator to recreate pathological traces using recurrent layers. This successfully synthesized 1,410 new pathological cases, perfectly maintaining strict physiological phase dynamics. I’ll hand it to Lokesh to discuss putting this all together architecturally."

---

## Part 3: Architecture, Baselines, & Edge (Speaker 3: Lokesh)

### **Slide 15: Core Architecture**
*(Visual: Clean block diagram of the ResNet branch showing sequence processing.)*
* **Heading:** Core Architecture
* To process the massive 1200-timestep time series (1D sequences).
* **Residual Mechanics:** 6 parallel cascading residual blocks preventing vanishing gradients.
* **Squeeze-and-Excitation (SE):** Channel balancing layers inside the skip connections.
* **Temporal Tracking:** Capable of modeling long-term physiological deceleration loops spanning 20 full minutes.

**🗣️ Speaker 3 (Lokesh) Notes:**  
"Thank you Yash. The core of our architecture is a custom 1-Dimensional Residual Network we named AttentionFusionResNet. Using 6 cascading blocks and Squeeze-and-Excitation routing, this branch is explicitly designed to look solely at the raw shapes of the fetal heart rate curves, tracking deep deceleration patterns without human bias."

---

### **Slide 16: Cross-Modal Attention**
*(Visual: A glowing diagram showing 'Tabular Metadata' acting as a gate/key over a neural layer.)*
* **Heading:** Cross-Modal Attention
* **The Problem:** Standard multi-modal networks just blindly "concatenate" tabular data at the end of a CNN.
* **Our Innovation (CMAF):** The Tabular metadata acts as an active mathematical *Gate* before sequence processing.
* **The Logic:** Tabular context directly shifts Attention layers. Example: If gestation = 28-weeks (extremely premature), CMAF instantly modifies neural weights to hypersensitize the model to minor heart rate drops.

**🗣️ Speaker 3 Notes:**  
"The most critical architectural innovation is our Cross-Modal Attention, or CMAF. Standard networks just stick tabular data at the end. We use the clinical variables to actively structure the attention layers. Just like an obstetrician changes their alert threshold if a mother is severely premature, our network shifts its weights on the fly based on the tabular clinical context."

---

### **Slide 17: Ensemble Strategy**
*(Visual: Insert the `architecture_flowchart.png` plot here, clearly showing the three models dropping into the Meta-Learner.)*
* **Heading:** Ensemble Strategy
* A single algorithmic architecture rarely generalizes across chaotic clinical noise.
* **Model A:** AttentionFusionResNet (Deep Sequential Extraction)
* **Model B:** 1D-InceptionNet (Multi-scale Kernel sweeps analyzing STV and LTV simultaneously)
* **Model C:** XGBoost (Classical gradient tracking strictly for Tabular + CSP sets)
* **The Meta-Learner:** A logistic regressor utilizing Out-Of-Fold Stacked probabilities.

**🗣️ Speaker 3 Notes:**  
"We combine three diverse models into a unified Stacking Ensemble. We run the ResNet, a multi-scale Inception net, and an XGBoost tree in parallel. A Meta-Learner then takes the independent probabilities of all three models and mathematically constructs the ultimate, most reliable inference output."

---

### **Slide 18: Uncertainty Quantification**
*(Visual: A scatter plot mock or gauge showing a 'Doubt/Confidence' metric.)*
* **Heading:** Uncertainty Quantification
* Providing an 85% "Confident" prediction on unrecognizable noise is clinically dangerous.
* **Our Execution:** Monte Carlo (MC) Dropout layers are intentionally left *active* during live inference.
* **The Loop:** The system executes 20 completely randomized inference walks per trace.
* **The Trigger:** High mathematical variance ($\sigma^2$) forces an "Ambiguous Zone: Requires Human Review" system override.

**🗣️ Speaker 3 Notes:**  
"Because giving confident predictions on extremely noisy traces is medically dangerous, we implemented Monte Carlo Dropout. When the model looks at a trace, it runs 20 randomized inference passes. If the answers are wildly different from each other, the variance spikes, and the AI natively tells the clinician: 'Uncertain: I require a human review'. "

---

### **Slide 19: Baseline Validation**
*(Visual: A clean table showing the three evaluated comparative baseline metrics.)*
* **Heading:** Baseline Validation
* To prove the necessity of the Tri-Modal structure, we actively implemented unimodal and classical architectures directly against the exact dataset:

| Architecture Approach | Target Data Modality | Validation AUC |
| :--- | :--- | :--- |
| **Unimodal Deep Learning** | Raw 1D-FHR Only | 0.564 |
| **Classical Linear Logic** | 16 Tabular Variants | 0.676 |
| **Classical Decisional Tree** | 16 Tabular Variants | 0.837 |

**🗣️ Speaker 3 Notes:**  
"We have not blindly assumed our Tri-Modal concept is better; we proved it requires fusion. We tested a Deep CNN exclusively on the Fetal Heart Rate, and it yielded a 0.56 AUC—it failed completely without Uterine Contractions. Classical trees hit an absolute ceiling at 0.83. This confirms definitively that breaking past the 0.84 SOTA barrier requires multi-modal ensembles."

---

### **Slide 20: Deployment Optimization**
*(Visual: A flowchart of a Keras file transitioning via Int8 Quantization to a TFLite mobile phone icon.)*
* **Heading:** Deployment Optimization
* **The Bottleneck:** Massive Keras TensorFlow networks require GPUs; rural wards lack internet and servers.
* **The Solution (TFLite):** Compressing float32 weights strictly down to 8-bit integers using Representative Dataset bounds.
* **The Result:** We successfully compress the backend architecture into an incredibly lightweight **deployable edge payload** bound for commodity Android integration.

**🗣️ Speaker 3 Notes:**  
"Finally, impact requires accessibility. We are applying TensorFlow Lite Int8 Quantization to our heavy backend. We crush massive floating-point tensors into 8-bit integers, shrinking the entire decision support system down to a highly compressed offline file, ensuring it runs rapidly on cheap android processors without cellular connections."

---

### **Slide 21: Technology Stack**
*(Visual: High quality logos arranged in their respective operational stacks.)*
* **Heading:** Technology Stack
* **Deep Learning Core:** Python 3.13 | TensorFlow 2.14 | Keras (Functional API)
* **Statistical / Ensembling:** Scikit-Learn (1.8.0) | XGBoost (3.2.0)
* **Physiological Signal Processing:** `wfdb` (Waveform DB PhysioNet) | SciPy | NumPy
* **Deployment & Inference Layer:** Streamlit (>=1.35.0) | TFLite | Pyngrok

**🗣️ Speaker 3 Notes:**  
"Our entire framework is strictly open-sourced and optimized. The heavy lifting is handled natively in Python 3.13 with TensorFlow 2.14, utilizing XGBoost for ensemble boosting, customized `wfdb` parsing for physiological signal handling, and Streamlit serving as our local clinical dashboard interface for ultimate end-device deployment."

---

### **Slide 22: Conclusion & Roadmap**
*(Visual: Checkmarks vs rocket icons outlining a brief roadmap list.)*
* **Heading:** Conclusion & Roadmap
* ✅ **Completed Engineering:** Full dataset ingestion, Tri-Modal feature extraction (CSP/Tabular), and TimeGAN Generative synthesis stability.
* ✅ **Completed Architecture:** Core AttentionFusionResNet and CMAF pipelines coded, Baseline limits established.
* 🚀 **End-Semester Roadmap:**
    * Execute the ultimate 5-Fold Evaluation Loops utilizing the final Meta-Learner stacking framework.
    * Output rigorous Brier-Score calibration models for true disease probability mapping.
    * Implement **Grad-CAM (Explainable AI)** heatmaps in the local Streamlit dashboard to visually highlight *why* the AI alerted the attending obstetrician. 

**🗣️ Speaker 3 Notes:**  
"To summarize our mid-semester status: The hardest data engineering, generative TimeGAN augmentation, and core architectural coding are complete. As we enter the final Phase 2, we will finalize our 5-Fold evaluation metrics, calibrate our risk probabilities, and deploy Explainable AI heatmaps in our dashboard—allowing the clinical physician to directly view the visual logic behind the AI's physiological warnings. Thank you for your time, the theoretical and architectural floor is now open for questions."

---

## Appendices

### **Slide 23: References (1/3) - Obstetrical & Dataset Foundations**
* **Heading:** References
* Petrozziello, A., et al. (2019). "Deep learning for continuous fetal heart rate monitoring in labor." *IEEE CBMS.* [DOI: 10.1109/CBMS.2019.00115](https://ieeexplore.ieee.org/document/8787383)
* Goldberger, A., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*. [DOI: 10.1161/01.cir.101.23.e215](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
* Ayres-de-Campos, D., et al. (2015). "FIGO consensus guidelines on intrapartum fetal monitoring: Cardiotocography." *Int J Gynaecol Obstet.* [DOI: 10.1016/j.ijgo.2015.06.020](https://pubmed.ncbi.nlm.nih.gov/26433401/)

### **Slide 24: References (2/3) - Generative Augmentation**
* **Heading:** References
* Yoon, J., Jarrett, D., & van der Schaar, M. (2019). "Time-series Generative Adversarial Networks." *NeurIPS.* [Link: arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
* Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein Generative Adversarial Networks." *ICML.* [Link: arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
* Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR.* [DOI: 10.1613/jair.953](https://arxiv.org/abs/1106.1813)

### **Slide 25: References (3/3) - Architectures & Optimization**
* **Heading:** References
* He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR.* [Link: arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
* Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS.* [Link: arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* Szegedy, C., et al. (2015). "Going Deeper with Convolutions." *CVPR.* [Link: arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
* Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR.* [Link: arXiv:1712.05877](https://arxiv.org/abs/1712.05877)

---

### **Slide 26: Thank You / Q&A**
*(Visual: Clean, minimal closing slide with the project abstract logo or title centered.)*
* **Heading:** Thank You
* **Sub-Heading:** "From Concept to Clinical Code: Engineering NeuroFetal-AI"
* **Presenters:** [Speaker 1 Name], Yashvardhan Singh, Lokesh
* **Prompt:** We welcome your questions regarding our Generative Architecture, the Tri-Modal Fusion Pipeline, or our Phase 2 Clinical Deployment strategy.

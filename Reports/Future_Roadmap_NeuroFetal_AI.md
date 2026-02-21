# NeuroFetal AI: Academic Research Roadmap

**Objective:** To outline a realistic, high-impact, 1.5-month research roadmap evolving NeuroFetal AI from its current SOTA baseline (v3.0) into a novel, publishable academic project for an undergraduate (6th Semester B.Tech) curriculum.

**Constraints & Context:**
- **Hardware:** Laptops without discrete GPUs; reliance on free-tier Google Colab (T4 GPUs).
- **Timeline:** Mid-semester presentation in ~15 days; End-semester presentation in ~1.5 months.
- **Goal:** Emphasize algorithmic novelty and research contributions for an academic paper, eschewing prospective clinical/hospital validation.

---

## 🎯 The Ultimate Endpoint: Research Paper Publication (Month 2)
**The Vision:** A compelling manuscript demonstrating a novel method for identifying fetal compromise in highly imbalanced CTG datasets using Tri-Modal Fusion, Advanced Augmentation, and Explainable AI. Target venues: IEEE EMBC, IEEE JBHI, or Nature Scientific Reports.

To achieve this, we must first pass the end-semester evaluation...

## 🏁 Milestone 2: End-Semester Presentation (1.5 Months)
The project represents a complete, finished, and highly novel AI system that clearly distinguishes itself from existing literature (like Spilka 2016 and Petrozziello 2018).

**Required Technical Deliverables (Working Backwards):**
1. **Evidential Deep Learning (EDL) Integration:**
    *   *The Novelty:* Replace MC Dropout (computationally expensive multiple forward passes) with EDL. EDL requires only a single forward pass, where the network outputs a distribution (Dirichlet parameters) to quantify uncertainty directly.
    *   *The Impact:* Allows the 1.9MB edge model to run even faster while still calculating a strict "Confidence Score." This is a highly publishable architectural tweak.
2. **Advanced XAI (Explainable AI) Benchmarking:**
    *   *The Novelty:* Grad-CAM is common. Move to **SHAP** (SHapley Additive exPlanations) or **Integrated Gradients** for the 1D CNN FHR branch.
    *   *The Impact:* Provide concrete feature-level attribution graphs (e.g., "The model predicted *Compromised* because STV was < 3.0 AND LTV variance was high"). These plots are essential for a strong research paper.
3. **Manuscript Draft "Results" Section:** Complete the tables comparing Baseline vs. V3.0 Ensemble vs. V4.0 (Enhanced Novelty) models.

To get the results needed for the final presentation, the mid-semester evaluation requires a breakthrough...

## 🚀 Milestone 1: Mid-Semester Presentation (15 Days)
Show that the team has moved beyond mere model-building and has tackled the hardest problem in the CTU-UHB dataset: the 7% minority class imbalance. SMOTE is a linear data science trick; we need deep learning novelty.

**Required Technical Deliverables (Working Backwards for the next two weeks):**
1. **TimeGAN for Synthetic CTG Generation:**
    *   *The Problem:* We only have ~40 pathological recordings (7.25% of 552). Algorithms struggle to learn the pattern of compromise.
    *   *The Novelty (Mid-Sem Focus):* Instead of standard tabular SMOTE, train a **Time-Series Generative Adversarial Network (TimeGAN)** purely on the pathological FHR/UC signals to generate realistic, synthetic fetal distress traces.
    *   *Feasibility:* Training a 1D TimeGAN on just 40 traces using free Colab is computationally feasible and highly impressive for an academic presentation.
2. **Train "Ensemble V4.0" with Synthetic Data:**
    *   Inject the TimeGAN-generated pathological traces into the training pipeline.
    *   *Goal:* Prove that using Deep Generative Augmentation improves the AUC beyond the current 0.87 benchmark. Even a 0.02 lift is a significant research finding.
3. **Mid-Semester Presentation Deck:**
    *   Slide 1: Problem & Existing SOTA Benchmark (AUC 0.84-0.87).
    *   Slide 2: The 7% Imbalance Bottleneck.
    *   Slide 3: Our Novel Solution: TimeGAN for Synthetic Distress Generation.
    *   Slide 4: Early results showing morphological similarity between real and synthetic traces.

---

## 🛠️ Compute Strategy (Colab Free-Tier Management)
Given the constraints, efficient use of compute is critical. We must avoid out-of-memory (OOM) errors and timeout disconnects.

1. **Pre-compute Everything:** The `data_ingestion.py` pipeline runs locally to produce the `.npy` files. Only upload the processed `X_fhr.npy`, `X_tabular.npy`, etc., to Google Drive to mount in Colab.
2. **Modular Colab Workflows:**
    *   *Notebook 1:* Train TimeGAN (save synthetic `.npy` generator locally).
    *   *Notebook 2:* Train the 1D-InceptionNet and ResNet separately (save weights `.keras`).
    *   *Notebook 3:* Train the XGBoost and Meta-Learner quickly on CPU using pre-extracted OOF (Out-of-Fold) probabilities.
3. **Avoid Colab Pro Requirements:** TimeGAN on 1D sequences of length 1200 will fit within the 15GB VRAM limit of a free T4 GPU if the batch size is kept small (e.g., 16 or 32).

---

## 📋 Summary of the "Next Right Steps" (Next 15 Days)
1. **Research TimeGAN Implementations:** Find an open-source TensorFlow implementation of TimeGAN or a suitable 1D VAE (Variational Autoencoder) for time-series generation.
2. **Filter Pathological Data:** Write a script to isolate the 40 existing pathological FHR/UC signal windows to feed to the GAN discriminator.
3. **Begin GAN Training:** Run iterations on Google Colab to generate synthetic traces and visually inspect them against real traces using Matplotlib.
4. **Draft the Mid-Semester Presentation.**

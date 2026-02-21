# Claude V4.0 Implementation Plan: NeuroFetal AI TimeGAN Synthesis

**Context for Claude:**
You are acting as an expert Machine Learning Researcher and Engineer. Your task is to implement the next major breakthrough for our 6th-semester B.Tech research project, "NeuroFetal AI".

Currently, our system achieved a State-of-the-Art (SOTA) AUC of 0.87 on the CTU-UHB dataset using a Tri-Modal Stacking Ensemble (FHR + UC + Clinical tabular data). However, we are bounded by a severe data imbalance: only 7.25% of our 552 recordings are "Pathological" (fetal compromise). We previously used tabular SMOTE, but this is a linear interpolation trick that destroys temporal dynamics.

**Your Objective: Milestone 1 (Mid-Semester Priority)**
Implement a **Time-Series Generative Adversarial Network (TimeGAN)** (or a highly robust 1D Convolutional GAN/Variational Autoencoder) strictly for the Pathological (minority) FHR and UC traces. 

By generating synthetic, physiologically realistic fetal distress traces, we will dynamically augment our training set to artificially balance the classes, with the goal of pushing our AUC > 0.90.

---

### Phase 1: Environment & Constraints
You will write code to be executed in our new Colab Notebook: `Code/notebooks/TimeGAN_Colab.ipynb`.
*   **Hardware:** Free-Tier Google Colab (Tesla T4 GPU, 15GB VRAM). 
*   **Data Shapes:** 
    *   FHR Input: `(N, 1200, 1)` -> 20 minutes at 1Hz.
    *   UC Input: `(N, 1200, 1)` -> 20 minutes at 1Hz.
    *   *Note:* The number of actual pathological windows `N` is small (~800 segmented windows from ~40 patients).
*   **Library:** You may use `ydata-synthetic` (which has a TimeGAN implementation), or build a custom `tf.keras` 1D GAN/VAE if that gives better control over the 1200-timestep length without OOM errors.

### Phase 2: Technical Requirements for Claude to Implement

**1. Data Preparation (Isolating the Minority Class):**
Write the data loading logic. We only want to feed the Pathological data (`y == 1`) into the GAN. Both FHR and UC should be synthesized simultaneously to preserve their physiological cross-correlation (e.g., late decelerations occurring after contractions).
*   *Implementation thought:* Stack FHR and UC into a `(N, 1200, 2)` shape array to feed into the generator.

**2. Generator Architecture:**
*   Design a Generator capable of outputting `(1200, 2)`.
*   Given the length (1200), standard RNNs might suffer from vanishing gradients. An approach using **1D Transposed Convolutions** (DCGAN style) or **Temporal Convolutional Networks (TCNs)** is highly recommended. Alternatively, a Transformer-based generator.

**3. Discriminator Architecture:**
*   Design a Discriminator that takes `(1200, 2)` and outputs a real/fake probability.
*   Use 1D Convolutions with LeakyReLU and Dropout.

**4. Training Loop & Validation:**
*   Write a robust custom training loop `train_step()` using `tf.GradientTape()`.
*   Include visualization functions: Every 50 epochs, plot a real Pathological trace (FHR + UC) next to a Generated trace using Matplotlib, so we can visually track morphological convergence.
*   Save the generator weights (`generator_v4.keras`) upon completion.
*   Generate 3x the size of the original minority class and save to `X_fhr_synthetic.npy` and `X_uc_synthetic.npy`.

**5. Integration with Existing Pipeline (Future Phase):**
*   Provide a stub/script on how to concatenate these synthetic `.npy` files with our original training data during the K-Fold cross-validation loop in `train_diverse_ensemble.py`.

### Instructions for Claude Output:
Please begin by outlining your proposed GAN architecture (Generator and Discriminator layers) designed specifically for 1D sequences of length 1200. Then, provide the complete, copy-pasteable Python code for the Colab notebook cells to execute this training pipeline. Keep Colab memory limitations in mind (use small batch sizes like 16 or 32).

# Week 6 Progress Report: NeuroFetal AI

**Date:** February 21, 2026 - February 27, 2026

## 1. Streamlit Cloud Deployment Fixes
*   **Identified Model Loading Bug:** We discovered a divergence between the local `localhost:8501` Streamlit app and the Cloud-deployed version. `localhost` loaded an "Ensemble (3 models)" while the Cloud instance showed "Ensemble (2 models)".
*   **Root Cause:** The `xgboost` model was trained and saved as a Pickled `.pkl` file with an `XGBClassifier` wrapper. However, on the cloud deployment environment (due to mismatched `scikit-learn` or `xgboost` versions), the Streamlit engine struggled to load it properly alongside the Meta-Learner (which had a `CalibratedClassifierCV` wrapper). This caused silent fallback to a 2-model ensemble.
*   **Fix Implemented:** We completely reviewed and restructured `model_loader.py` to add robust error-handling when loading `.pkl` models. We also pinned exactly working framework versions in `requirements.txt` to synchronize the cloud environment with our local environments.

## 2. Phase 8: Rigorous Model Calibration (V5.0)
*   **Why It Mattered:** Clinical models require more than accuracy—they need reliable probability outputs. A 90% confidence score must mean the model is right 9 out of 10 times.
*   **Action:** Retrained and evaluated the ensemble meta-learner using **Platt Scaling** via `CalibratedClassifierCV` in our Colab evaluation pipeline (`Calibration_Retrain_Colab.ipynb`). 
*   **Results Achieved:** 
    *   **Brier Score:** Improved to a stunning **0.0460** (closer to 0 is better).
    *   **Expected Calibration Error (ECE):** Reduced to **0.0543**.
    *   **Accuracy:** Soared to **96.34%**, with an F1 Score of **95.22%**.

## 3. Advanced Information Theory Uncertainty
*   We upgraded our Monte Carlo Dropout uncertainty tracking.
*   Instead of just showing standard deviation, the app now explicitly calculates:
    1.  **Predictive Entropy:** Captures the total uncertainty of the prediction.
    2.  **Mutual Information:** Isolates the *Epistemic Uncertainty* (model's "ignorance" about out-of-distribution data).
*   The Streamlit dashboard now features gauge charts to visually articulate Epistemic vs Aleatoric uncertainty to clinicians in real time.

## 4. Documentation & Version Bump
*   Since the integration of Model Calibration and Advanced Uncertainty represents a fundamental upgrade to the reliability and UI payload of the system, we bumped the project version to **V5.0 (Calibrated Ensemble)**.
*   Updated `README.MD`, `Our_project.md`, `explain.md`, and `final_report.md` to reflect the completed V5.0 metrics, Brier scores, and accuracy milestones.

**Objective Complete:** The NeuroFetal AI pipeline is now fully operational, calibrated, state-of-the-art, and deployed to the cloud with robust clinical XAI features.

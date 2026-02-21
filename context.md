# NeuroFetal AI — Continuation Context (V4.0 TimeGAN Phase)

## Project Overview
- **Repo:** `Krishna200608/NeuroFetal-AI` — 6th-semester B.Tech research project
- **Branch:** `feat/v4.0-timegan` (current working branch, `main` has V3.0 baseline)
- **Goal:** Fetal distress detection from CTG (Cardiotocography) signals using deep learning

## What's Been Completed

### V3.0 Baseline (on `main`)
- **Data Pipeline:** `Code/scripts/data_ingestion.py` — extracts FHR/UC from CTU-UHB PhysioNet DB, outputs `(2546, 1200)` arrays to `Datasets/processed/*.npy`
- **Architecture:** AttentionFusionResNet with 3 inputs (FHR signal + Tabular features + CSP features)
- **Training:** `Code/scripts/train.py` — 5-fold CV with SMOTE, Focal Loss, Cosine Annealing, SSL pretraining
- **Ensemble:** `Code/scripts/train_diverse_ensemble.py` — Stacking (AttentionFusionResNet + InceptionNet + XGBoost)
- **Baseline AUC:** 0.87

### V4.0 TimeGAN (on `feat/v4.0-timegan`) — DONE
- **Notebook:** `Code/notebooks/TimeGAN_Colab.ipynb` — Ran successfully on Colab T4 GPU
- **Architecture:** 1D Convolutional WGAN-GP (Generator: noise→Conv1DTranspose×4→(1200,2), Critic: Conv1D×4→linear)
- **Training:** 500 epochs, batch 16, 5:1 critic ratio, GP λ=10
- **Output:** 1,410 synthetic pathological traces saved to `Datasets/synthetic/X_fhr_synthetic.npy` and `X_uc_synthetic.npy`
- **Generator weights:** `Code/models/generator_v4.keras` (12 MB)
- **Bug fixed:** `data_ingestion.py` saves 2D `(N,1200)` arrays — added `expand_dims` guard before GAN stacking

### V4.0 Integration (on `feat/v4.0-timegan`) — DONE, PUSHED
Modified the training scripts to replace SMOTE with TimeGAN augmentation:

#### `Code/scripts/train.py` changes:
1. Added `AUGMENTATION = "timegan"` global and `SYNTHETIC_DATA_DIR` config
2. Added `--augmentation` CLI flag with choices `["smote", "timegan", "none"]` (default: `"timegan"`)
3. Added `apply_timegan_augmentation()` function (~80 lines) that:
   - Loads pre-generated synthetic data from `Datasets/synthetic/`
   - Matches array dimensionality (2D vs 3D) automatically
   - Injects synthetic traces to reach 1:2 minority ratio (same as SMOTE target)
   - Resamples tabular features from existing pathological samples
   - Falls back to SMOTE if synthetic files not found
4. Updated K-fold loop (line ~592) to branch: `smote` / `timegan` / `none`

#### `Code/scripts/train_diverse_ensemble.py` changes:
1. Added `USE_TIMEGAN_AUG = True` flag and `SYNTHETIC_DATA_DIR`
2. Added TimeGAN augmentation block inside `generate_oof_predictions()` fold loop (imports `apply_timegan_augmentation` from `train.py`)

### V4.0 Colab Notebook (on `feat/v4.0-timegan`) — DONE, NEEDS PUSH
Updated `Code/notebooks/Training_Colab.ipynb` for V4.0:
- Header updated to V4.0 TimeGAN branding
- Clone cell now checks out `feat/v4.0-timegan` branch
- Training cell uses `--augmentation timegan --epochs 150`
- Added optional SMOTE comparison cell
- All stale V3.0 outputs cleared for clean re-run

## What's Next (Priority Order)

### 1. Push V4.0 Notebook Update & Run on Colab
```bash
# Push the updated notebook
git add -A && git commit -m "docs: update Training_Colab.ipynb for V4.0 TimeGAN augmentation" && git push origin feat/v4.0-timegan
```

On Colab, open `Code/notebooks/Training_Colab.ipynb` and run all cells sequentially:
```bash
# Single-model training (train.py with TimeGAN as default)
!python Code/scripts/train.py --augmentation timegan --epochs 150

# Or fall back to SMOTE for comparison
!python Code/scripts/train.py --augmentation smote --epochs 150

# Diverse ensemble (automatically uses TimeGAN)
!python Code/scripts/train_diverse_ensemble.py
```
**Goal:** Compare AUC vs 0.87 SMOTE baseline

### 2. End-Semester Novelty Features (Future)
- Evidential Deep Learning (EDL) for uncertainty quantification
- SHAP feature attribution for interpretability

## Key File Locations
| File | Purpose |
|------|---------|
| `Code/scripts/data_ingestion.py` | Raw CTU-UHB → processed `.npy` |
| `Code/scripts/train.py` | Main 5-fold training with SMOTE/TimeGAN |
| `Code/scripts/train_diverse_ensemble.py` | 3-model stacking ensemble |
| `Code/notebooks/TimeGAN_Colab.ipynb` | WGAN-GP training (completed with outputs) |
| `Code/notebooks/Training_Colab.ipynb` | Main training notebook for Colab (V4.0) |
| `Code/utils/model.py` | AttentionFusionResNet architecture |
| `Code/utils/focal_loss.py` | Focal Loss implementation |
| `Code/utils/csp_features.py` | CSP feature extraction |
| `Datasets/processed/*.npy` | X_fhr, X_uc, X_tabular, y |
| `Datasets/synthetic/*.npy` | TimeGAN-generated X_fhr_synthetic, X_uc_synthetic |
| `Code/models/generator_v4.keras` | Trained WGAN-GP generator |

## Important Technical Notes
- **Array shapes:** `data_ingestion.py` saves FHR/UC as 2D `(N, 1200)`. `train.py:load_data()` expands to 3D `(N, 1200, 1)`. The TimeGAN augmentation function handles both.
- **Class balance:** 470/2546 = 18.5% pathological. TimeGAN targets 1:2 ratio same as SMOTE.
- **Colab auth:** Uses `google.colab.userdata.get('GITHUB_TOKEN')` from Secrets.

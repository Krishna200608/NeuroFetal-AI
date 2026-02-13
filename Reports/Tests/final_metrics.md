# Final Test Metrics (Full Dataset)

**Date:** 2026-02-13 12:39

**Model:** *Fold 1 — SOTA Enhanced Model (AttentionFusionResNet + Stacking Ensemble)*

**Total Samples:** **1513 (20-min windows)**

---

## Overall Performance

| Metric       | Score      |
| ------------ | ---------- |
| **Accuracy** | **74.49%** |
| **AUC-ROC**  | **0.8439** |

---

## Classification Report

```text
              precision    recall  f1-score   support

      Normal       0.94      0.73      0.82      1235
 Compromised       0.40      0.79      0.53       278

    accuracy                           0.74      1513
   macro avg       0.67      0.76      0.68      1513
weighted avg       0.84      0.74      0.77      1513

```

---

## Confusion Matrix

### Matrix Form

```text
          Predicted
          0     1
Actual 0  906   329
       1  57   221
```

### Tabular Form

| Actual \ Predicted  | 0    | 1  |
| ------------------- | ---- | -- |
| **0 (Normal)**      | 906 | 329 |
| **1 (Compromised)** | 57 | 221 |

---

## Input Details

| Feature Type      | Shape           |
| ----------------- | --------------- |
| **FHR Input**     | (1513, 1200, 1) |
| **Tabular Input** | (1513, 18) |
| **CSP Features**  | (1513, 19) |

---

*End of report*

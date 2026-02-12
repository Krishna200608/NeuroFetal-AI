# Final Test Metrics (Full Dataset)

**Date:** 2026-02-12 19:29

**Model:** *Fold 1 — AttentionFusionResNet*

**Total Samples:** **1656 (20-min windows)**

---

## Overall Performance

| Metric       | Score      |
| ------------ | ---------- |
| **Accuracy** | **90.28%** |
| **AUC-ROC**  | **0.7386** |

---

## Classification Report

```text
              precision    recall  f1-score   support

      Normal       0.94      0.95      0.95      1536
 Compromised       0.30      0.25      0.27       120

    accuracy                           0.90      1656
   macro avg       0.62      0.60      0.61      1656
weighted avg       0.90      0.90      0.90      1656

```

---

## Confusion Matrix

### Matrix Form

```text
          Predicted
          0     1
Actual 0  1465   71
       1  90   30
```

### Tabular Form

| Actual \ Predicted  | 0    | 1  |
| ------------------- | ---- | -- |
| **0 (Normal)**      | 1465 | 71 |
| **1 (Compromised)** | 90 | 30 |

---

## Input Details

| Feature Type      | Shape           |
| ----------------- | --------------- |
| **FHR Input**     | (1656, 1200, 1) |
| **Tabular Input** | (1656, 3) |
| **CSP Features**  | N/A |

---

*End of report*

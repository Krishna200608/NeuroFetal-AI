
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Add scripts directory to path (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
try:
    import data_ingestion
except ImportError:
    pass

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Datasets", "processed")

print(f"Loading tabular data from: {PROCESSED_DATA_DIR}")

try:
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    print("Loaded tabular features.")
except FileNotFoundError:
    print("X_tabular.npy not found.")
    sys.exit(1)

print(f"X_tabular shape: {X_tabular.shape}")
print(f"y shape: {y.shape}")

# --- Training Loop ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

print("\nStarting Benchmark (Paper 4)...")

results = {name: {'auc': [], 'acc': []} for name in models}
X = np.nan_to_num(X_tabular, nan=0.0)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    for name, model in models.items():
        clf = model
        # RF doesn't strictly need scaling but consistent usage is fine
        if name == "Random Forest":
            clf.fit(X_train, y_train) # Fit on unscaled for RF usually better/same
            input_val = X_val
        else:
            clf.fit(X_train_scaled, y_train)
            input_val = X_val_scaled
            
        if hasattr(clf, "predict_proba"):
            y_pred_prob = clf.predict_proba(input_val)[:, 1]
        else:
            y_pred_prob = clf.predict(input_val)
            
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_val, y_pred_prob)
        acc = accuracy_score(y_val, y_pred_class)
        
        results[name]['auc'].append(auc)
        results[name]['acc'].append(acc)

        # Save Model (Last Fold)
        import joblib
        model_dir = os.path.join(os.path.dirname(__file__), "Models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"baseline_paper4_{name.replace(' ', '_')}.pkl")
        joblib.dump(clf, model_path)


print("\n=== Final Results (Paper 4) ===")
with open(os.path.join(os.path.dirname(__file__), "baseline_paper4_results.txt"), "w") as f:
    f.write(f"Ref: Paper 4 (Petrozziello 2018) - Classical ML Baseline\n")
    for name in models:
        mean_auc = np.mean(results[name]['auc'])
        std_auc = np.std(results[name]['auc'])
        mean_acc = np.mean(results[name]['acc'])
        print(f"{name}: AUC {mean_auc:.4f} +/- {std_auc:.4f}, Acc {mean_acc:.4f}")
        f.write(f"{name}: AUC {mean_auc:.4f} +/- {std_auc:.4f}, Acc {mean_acc:.4f}\n")

print("Results saved.")

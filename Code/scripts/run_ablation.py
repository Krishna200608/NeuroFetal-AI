"""
Ablation Study Runner for NeuroFetal AI
========================================
Run this script in Colab after pulling the latest changes.

Usage:
1. Pull latest code: !git pull origin main
2. Run data ingestion (if not done): !python Code/scripts/data_ingestion.py
3. Run this script: !python Code/scripts/run_ablation.py

The script will:
1. Run all 5 ablation configurations
2. Generate comparison table
3. Save results to JSON
"""

import os
import sys
import numpy as np

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'Code', 'utils'))
sys.path.insert(0, os.path.join(BASE_DIR, 'Code', 'scripts'))

from ablation_study import AblationStudy

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*60)
    print("NeuroFetal AI - Ablation Study")
    print("="*60)
    
    # Load data
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
    
    print("\n1. Loading data...")
    X_fhr = np.load(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"))
    X_tabular = np.load(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    
    # Try to load UC for CSP
    try:
        X_uc = np.load(os.path.join(PROCESSED_DATA_DIR, "X_uc.npy"))
        print(f"   UC data loaded: {X_uc.shape}")
    except FileNotFoundError:
        X_uc = None
        print("   UC data not found - CSP experiments will be skipped")
    
    # Ensure FHR has channel dimension
    if X_fhr.ndim == 2:
        X_fhr = np.expand_dims(X_fhr, axis=-1)
    
    print(f"   FHR: {X_fhr.shape}")
    print(f"   Tabular: {X_tabular.shape}")
    print(f"   Labels: {y.shape}, {np.mean(y):.1%} positive")
    
    # Initialize ablation study
    print("\n2. Initializing ablation study...")
    study = AblationStudy()
    
    # Run all experiments
    print("\n3. Running experiments...")
    print("   This will take a while - grab a coffee ☕")
    
    # For quick testing, just run a subset
    # Uncomment the line below to run all experiments
    results = study.run_all_experiments(X_fhr, X_tabular, y, X_csp=None)  # CSP disabled for now
    
    # Print summary
    print("\n4. Results Summary")
    study.print_summary()
    
    # Save results
    print("\n5. Saving results...")
    study.save_results('ablation_results.json')
    
    # Generate LaTeX table
    print("\n6. LaTeX Table for Paper:")
    study.generate_latex_table()
    
    print("\n" + "="*60)
    print("Ablation study complete!")
    print("="*60)


if __name__ == "__main__":
    main()

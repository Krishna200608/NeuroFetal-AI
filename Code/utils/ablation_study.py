"""
Ablation Study Framework for NeuroFetal AI
==========================================
Systematic ablation experiments to isolate the contribution of each enhancement.

Experiments:
1. Baseline (Paper 7): FHR + Tabular, Multiply fusion
2. + SE Blocks: Add channel attention
3. + Temporal Attention: Add self-attention
4. + CSP Features: Add multimodal features
5. + All Enhancements: Full model

Each experiment is run with 5-fold CV and results are logged.
This is required for publication to demonstrate the value of each contribution.
"""

import os
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    use_se_blocks: bool = True
    use_attention: bool = True
    use_csp: bool = True
    use_focal_loss: bool = True
    input_shape_fhr: tuple = (1200, 1)
    input_shape_tabular: tuple = (16,)
    input_shape_csp: tuple = (19,)
    n_folds: int = 5
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    config_name: str
    fold: int
    auc: float
    sensitivity: float
    specificity: float
    precision: float
    f1_score: float
    training_time_sec: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AblationStudy:
    """
    Framework for running systematic ablation experiments.
    """
    
    # Predefined ablation configurations
    CONFIGS = {
        'baseline': AblationConfig(
            name='baseline',
            description='Paper 7 Baseline: FHR + Tabular, Multiply fusion, no attention',
            use_se_blocks=False,
            use_attention=False,
            use_csp=False,
            use_focal_loss=False,
            input_shape_tabular=(3,)  # Original 3 tabular features
        ),
        'baseline_focal': AblationConfig(
            name='baseline_focal',
            description='Baseline + Focal Loss for class imbalance',
            use_se_blocks=False,
            use_attention=False,
            use_csp=False,
            use_focal_loss=True,
            input_shape_tabular=(3,)
        ),
        'se_only': AblationConfig(
            name='se_only',
            description='Baseline + SE Blocks (channel attention)',
            use_se_blocks=True,
            use_attention=False,
            use_csp=False,
            use_focal_loss=True,
            input_shape_tabular=(3,)
        ),
        'attention_only': AblationConfig(
            name='attention_only',
            description='Baseline + Temporal Self-Attention',
            use_se_blocks=False,
            use_attention=True,
            use_csp=False,
            use_focal_loss=True,
            input_shape_tabular=(3,)
        ),
        'se_attention': AblationConfig(
            name='se_attention',
            description='SE Blocks + Temporal Attention',
            use_se_blocks=True,
            use_attention=True,
            use_csp=False,
            use_focal_loss=True,
            input_shape_tabular=(3,)
        ),
        'csp_only': AblationConfig(
            name='csp_only',
            description='Baseline + CSP Features (no attention)',
            use_se_blocks=False,
            use_attention=False,
            use_csp=True,
            use_focal_loss=True
        ),
        'full_model': AblationConfig(
            name='full_model',
            description='Full Enhanced Model: SE + Attention + CSP + Focal Loss',
            use_se_blocks=True,
            use_attention=True,
            use_csp=True,
            use_focal_loss=True
        ),
    }
    
    def __init__(self, output_dir: str = None):
        """
        Initialize ablation study.
        
        Args:
            output_dir: Directory to save results
        """
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, "Reports", "ablation_study")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results: List[AblationResult] = []
        self.current_config: Optional[AblationConfig] = None
        
    def build_model_for_config(self, config: AblationConfig):
        """
        Build model based on ablation configuration.
        
        Args:
            config: AblationConfig specifying which components to include
            
        Returns:
            Keras Model
        """
        try:
            from model import build_fusion_resnet, build_enhanced_fusion_resnet
        except ImportError:
            raise ImportError("Cannot import model module")
        
        if not config.use_csp:
            # Use original 2-input model
            return build_fusion_resnet(
                input_shape_ts=config.input_shape_fhr,
                input_shape_tab=config.input_shape_tabular
            )
        else:
            # Use enhanced 3-input model
            return build_enhanced_fusion_resnet(
                input_shape_fhr=config.input_shape_fhr,
                input_shape_tabular=config.input_shape_tabular,
                input_shape_csp=config.input_shape_csp,
                use_se_blocks=config.use_se_blocks,
                use_attention=config.use_attention
            )
    
    def run_single_experiment(
        self, 
        config: AblationConfig,
        X_fhr: np.ndarray,
        X_tabular: np.ndarray,
        y: np.ndarray,
        X_csp: np.ndarray = None
    ) -> List[AblationResult]:
        """
        Run a single ablation experiment with cross-validation.
        
        Args:
            config: Experiment configuration
            X_fhr: FHR signal data
            X_tabular: Tabular feature data
            y: Labels
            X_csp: CSP features (optional, used if config.use_csp=True)
            
        Returns:
            List of AblationResult for each fold
        """
        import tensorflow as tf
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        import time
        
        print(f"\n{'='*60}")
        print(f"Running Ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        self.current_config = config
        fold_results = []
        
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_fhr, y), 1):
            print(f"\n--- Fold {fold}/{config.n_folds} ---")
            
            # Split data
            X_fhr_train, X_fhr_val = X_fhr[train_idx], X_fhr[val_idx]
            X_tab_train, X_tab_val = X_tabular[train_idx], X_tabular[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if config.use_csp and X_csp is not None:
                X_csp_train, X_csp_val = X_csp[train_idx], X_csp[val_idx]
            
            # Build model
            model = self.build_model_for_config(config)
            
            # Compile
            if config.use_focal_loss:
                try:
                    from focal_loss import get_focal_loss
                    loss_fn = get_focal_loss(alpha=0.25, gamma=2.0)
                except ImportError:
                    loss_fn = 'binary_crossentropy'
            else:
                loss_fn = 'binary_crossentropy'
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss=loss_fn,
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            
            # Prepare inputs
            if config.use_csp:
                train_inputs = [X_fhr_train, X_tab_train, X_csp_train]
                val_inputs = [X_fhr_val, X_tab_val, X_csp_val]
            else:
                train_inputs = [X_fhr_train, X_tab_train]
                val_inputs = [X_fhr_val, X_tab_val]
            
            # Train
            start_time = time.time()
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc', patience=10, mode='max', restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6
                )
            ]
            
            history = model.fit(
                train_inputs, y_train,
                validation_data=(val_inputs, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred_proba = model.predict(val_inputs, verbose=0).flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Metrics
            auc = roc_auc_score(y_val, y_pred_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            
            # Sensitivity = Recall (True Positive Rate)
            sensitivity = recall
            
            # Specificity (True Negative Rate)
            tn = np.sum((y_val == 0) & (y_pred == 0))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Store result
            result = AblationResult(
                config_name=config.name,
                fold=fold,
                auc=auc,
                sensitivity=sensitivity,
                specificity=specificity,
                precision=precision,
                f1_score=f1,
                training_time_sec=training_time
            )
            fold_results.append(result)
            self.results.append(result)
            
            print(f"   AUC: {auc:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")
            
            # Clear session for next fold
            tf.keras.backend.clear_session()
            
        return fold_results
    
    def run_all_experiments(self, X_fhr, X_tabular, y, X_csp=None):
        """
        Run all predefined ablation experiments.
        
        Args:
            X_fhr, X_tabular, y, X_csp: Input data
            
        Returns:
            Dict of results per configuration
        """
        all_results = {}
        
        for config_name, config in self.CONFIGS.items():
            fold_results = self.run_single_experiment(
                config, X_fhr, X_tabular, y, X_csp
            )
            all_results[config_name] = fold_results
            
        # Save results
        self.save_results()
        self.print_summary()
        
        return all_results
    
    def save_results(self, filename: str = 'ablation_results.json'):
        """Save all results to JSON file."""
        results_dict = [asdict(r) for r in self.results]
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        print(f"\nResults saved to: {output_path}")
        
    def print_summary(self):
        """Print summary table of all experiments."""
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        
        # Group by config
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in self.results:
            grouped[r.config_name].append(r)
            
        print(f"\n{'Config':<20} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'Time(s)':>10}")
        print("-"*80)
        
        for config_name, results in grouped.items():
            aucs = [r.auc for r in results]
            sens = [r.sensitivity for r in results]
            specs = [r.specificity for r in results]
            f1s = [r.f1_score for r in results]
            times = [r.training_time_sec for r in results]
            
            print(f"{config_name:<20} "
                  f"{np.mean(aucs):.4f}±{np.std(aucs):.2f} "
                  f"{np.mean(sens):.4f} "
                  f"{np.mean(specs):.4f} "
                  f"{np.mean(f1s):.4f} "
                  f"{np.mean(times):>8.1f}")
                  
        print("="*80)
        
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in self.results:
            grouped[r.config_name].append(r)
            
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & AUC & Sensitivity & Specificity & F1 \\",
            r"\midrule"
        ]
        
        for config_name, results in grouped.items():
            aucs = [r.auc for r in results]
            sens = [r.sensitivity for r in results]
            specs = [r.specificity for r in results]
            f1s = [r.f1_score for r in results]
            
            lines.append(
                f"{config_name} & "
                f"{np.mean(aucs):.3f}$\\pm${np.std(aucs):.2f} & "
                f"{np.mean(sens):.3f} & "
                f"{np.mean(specs):.3f} & "
                f"{np.mean(f1s):.3f} \\\\"
            )
            
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        latex = "\n".join(lines)
        
        # Save to file
        output_path = os.path.join(self.output_dir, 'ablation_table.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")
        
        return latex


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Ablation Study Framework Test")
    print("="*60)
    
    # List available configurations
    print("\nAvailable Ablation Configurations:")
    for name, config in AblationStudy.CONFIGS.items():
        print(f"  - {name}: {config.description}")
        
    # Test framework initialization
    study = AblationStudy()
    print(f"\nOutput directory: {study.output_dir}")
    
    print("\n" + "="*60)
    print("Framework loaded successfully!")
    print("To run ablations, use: study.run_all_experiments(X_fhr, X_tab, y, X_csp)")
    print("="*60)

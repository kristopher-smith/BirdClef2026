"""
Automatic experiment tracking utilities for BirdClef2026.

Provides decorators and callbacks for automatic metric and artifact logging.
Works with or without MLflow (graceful fallback).
"""

import json
import os
import sys
from pathlib import Path
from functools import wraps
from typing import Optional

import numpy as np

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MetricsLogger:
    """
    Unified metrics logger that works with or without MLflow.
    
    When MLflow is available: logs to MLflow + console
    When MLflow unavailable: logs to console + JSON files
    """
    
    def __init__(self, 
                 experiment_name: str = "birdclef2026",
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 run_dir: str = "runs"):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.run_dir = Path(run_dir)
        self.mlflow_available = MLFLOW_AVAILABLE
        self._run = None
        self._run_id = None
        self._epoch_metrics = []
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def start_run(self):
        """Start an MLflow run if available, otherwise prepare fallback."""
        if self.mlflow_available:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            self._run_id = self._run.info.run_id
            print(f"[MLflow] Started run: {self._run_id}")
        else:
            print(f"[Tracking] MLflow not available - using fallback logging")
            
    def log_params(self, params: dict):
        """Log hyperparameters."""
        if self.mlflow_available and self._run:
            mlflow.log_params(params)
            
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to MLflow (if available) + console + JSON."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
        print(f"  Metrics: {metrics_str}")
        
        self._save_metrics_json(metrics, step)
        
        if self.mlflow_available and self._run:
            mlflow.log_metrics(metrics, step=step)
            
        if step is not None:
            self._epoch_metrics.append({**metrics, 'step': step})
            
    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None):
        """Log an artifact file."""
        artifact_name = artifact_name or Path(local_path).name
        
        if self.mlflow_available and self._run:
            mlflow.log_artifact(local_path)
            
    def log_text(self, text: str, artifact_name: str):
        """Log text content as artifact."""
        path = self.run_dir / artifact_name
        path.write_text(text)
        
        if self.mlflow_available and self._run:
            mlflow.log_artifact(str(path))
            
    def log_confusion_matrix(self, cm: np.ndarray, labels: list[str], step: Optional[int] = None):
        """Log confusion matrix as artifact (only for present classes)."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        present_classes = [i for i in range(len(labels)) 
                          if cm[i].sum() > 0 or cm[:, i].sum() > 0]
        
        if len(present_classes) == 0:
            return
            
        cm_filtered = cm[np.ix_(present_classes, present_classes)]
        labels_filtered = [labels[i] for i in present_classes]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_filtered, annot=False, fmt='d', 
                   xticklabels=labels_filtered, yticklabels=labels_filtered, ax=ax,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Present Classes Only)')
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        
        cm_path = self.run_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        
        self.log_artifact(str(cm_path))
        
        cm_data = {
            'matrix': cm_filtered.tolist(),
            'labels': labels_filtered
        }
        cm_json_path = self.run_dir / "confusion_matrix.json"
        cm_json_path.write_text(json.dumps(cm_data, indent=2))
        self.log_artifact(str(cm_json_path))
        
    def log_spectrogram(self, spec: np.ndarray, artifact_name: str = "sample_spectrogram.png"):
        """Log a spectrogram image."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel bins')
        plt.colorbar(im, ax=ax, label='Magnitude')
        
        path = self.run_dir / artifact_name
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        
        self.log_artifact(str(path))
        
    def log_training_curves(self):
        """Log training curves plot from accumulated metrics."""
        import matplotlib.pyplot as plt
        
        if not self._epoch_metrics:
            return
            
        epochs = [m['step'] for m in self._epoch_metrics if 'val_loss' in m]
        if not epochs:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        val_metrics = [m for m in self._epoch_metrics if 'val_loss' in m]
        train_metrics = [m for m in self._epoch_metrics if 'train_loss' in m or 'loss' in m]
        
        if train_metrics:
            loss_vals = [m.get('loss', m.get('train_loss', 0)) for m in train_metrics]
            axes[0, 0].plot(range(1, len(loss_vals) + 1), loss_vals, label='Train')
        if val_metrics:
            val_loss_vals = [m['val_loss'] for m in val_metrics]
            axes[0, 0].plot(epochs, val_loss_vals, label='Val')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
        
        if val_metrics:
            map_vals = [m.get('map_at_10', 0) for m in val_metrics]
            f1_vals = [m.get('f1_at_10', 0) for m in val_metrics]
            axes[0, 1].plot(epochs, map_vals, label='mAP@10')
            axes[0, 1].plot(epochs, f1_vals, label='F1@10')
            axes[0, 1].set_title('Metrics')
            axes[0, 1].legend()
        
        if train_metrics:
            lr_vals = [m.get('lr', 0) for m in train_metrics]
            if any(lr_vals):
                axes[1, 0].plot(range(1, len(lr_vals) + 1), lr_vals)
                axes[1, 0].set_title('Learning Rate')
        
        plt.tight_layout()
        
        path = self.run_dir / "training_curves.png"
        plt.savefig(path)
        plt.close()
        
        self.log_artifact(str(path))
        
    def log_model_checkpoint(self, checkpoint_path: Path, metrics: Optional[dict] = None):
        """Log a model checkpoint as artifact."""
        import shutil
        
        if not checkpoint_path.exists():
            print(f"[Tracking] Warning: checkpoint {checkpoint_path} not found")
            return
            
        dest_path = self.run_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, dest_path)
        self.log_artifact(str(dest_path))
        
        if metrics:
            metrics_path = self.run_dir / f"{checkpoint_path.stem}_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2))
            self.log_artifact(str(metrics_path))
            
    def log_submission(self, submission_path: Path):
        """Log a submission CSV as artifact."""
        if not submission_path.exists():
            print(f"[Tracking] Warning: submission {submission_path} not found")
            return
        self.log_artifact(str(submission_path))
        
    def log_predictions(self, predictions: np.ndarray, labels: list[str], artifact_name: str = "predictions.csv"):
        """Log predictions as CSV."""
        import pandas as pd
        
        df = pd.DataFrame(predictions, columns=labels)
        path = self.run_dir / artifact_name
        df.to_csv(path, index=False)
        self.log_artifact(str(path))
        
    def log_class_distribution(self, labels_df, label_cols, artifact_name: str = "class_distribution.png"):
        """Log class distribution bar chart."""
        import matplotlib.pyplot as plt
        
        class_counts = labels_df[label_cols].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        class_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        plt.tight_layout()
        
        path = self.run_dir / artifact_name
        plt.savefig(path)
        plt.close()
        
        self.log_artifact(str(path))
        
    def end_run(self):
        """End the run."""
        self.log_training_curves()
        
        if self.mlflow_available and self._run:
            mlflow.end_run()
            print(f"[MLflow] Ended run: {self._run_id}")
            
    def _save_metrics_json(self, metrics: dict, step: Optional[int]):
        """Save metrics to JSON file (fallback)."""
        step_str = f"epoch_{step}" if step is not None else "final"
        path = self.run_dir / f"metrics_{step_str}.json"
        path.write_text(json.dumps(metrics, indent=2))


def mlflow_track(metric_names: list[str], 
                 prefix: str = ""):
    """
    Decorator to automatically track metrics from train/eval functions.
    
    Args:
        metric_names: List of metric names to extract from return dict
        prefix: Prefix for metrics (e.g., 'train_' or 'val_')
        
    Usage:
        @mlflow_track(['loss', 'accuracy', 'map_at_10'], prefix='train_')
        def train_one_epoch(model, dataloader, ...):
            # ... training code ...
            return {'loss': epoch_loss, 'accuracy': epoch_acc, 'map_at_10': map_score}
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                prefixed_metrics = {}
                for name in metric_names:
                    key = f"{prefix}{name}" if prefix else name
                    if name in result:
                        prefixed_metrics[key] = result[name]
                
                step = kwargs.get('epoch', kwargs.get('step', None))
                
                if hasattr(wrapper, '_logger') and wrapper._logger:
                    wrapper._logger.log_metrics(prefixed_metrics, step=step)
                    
            return result
            
        wrapper._logger = None
        return wrapper
    return decorator


def attach_logger(func, logger: MetricsLogger):
    """Attach a logger instance to a decorated function."""
    func._logger = logger
    return func

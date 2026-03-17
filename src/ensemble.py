"""Model Ensemble for BirdClef 2026."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json

from model import BirdClefModel, get_device

try:
    from model_perch import create_embedding_model, BirdClefPERCHModel
    PERCH_AVAILABLE = True
except ImportError:
    PERCH_AVAILABLE = False


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with averaging."""

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        aggregation: str = "average",
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of PyTorch models
            weights: Weights for each model (default: equal weights)
            aggregation: 'average' or 'max' for combining predictions
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.aggregation = aggregation
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all models and aggregate."""
        all_logits = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                logits = model(x)
                all_logits.append(logits * weight)
        
        if self.aggregation == "average":
            return torch.stack(all_logits).sum(dim=0)
        elif self.aggregation == "max":
            return torch.stack(all_logits).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class EnsemblePredictor:
    """Predictor that handles ensemble of models."""

    def __init__(
        self,
        model_paths: List[str],
        num_classes: int = 234,
        device: str = "cpu",
        weights: Optional[List[float]] = None,
        aggregation: str = "average",
        model_types: Optional[List[str]] = None,
        embedding_models: Optional[List[str]] = None,
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of paths to model checkpoints
            num_classes: Number of output classes
            device: Device to run on
            weights: Weights for each model
            aggregation: 'average' or 'max'
            model_types: List of backbone types (efficientnet_b0, efficientnet_b2, etc.)
            embedding_models: List of embedding model types (yamnet, perch, simple)
        """
        self.model_paths = model_paths
        self.num_classes = num_classes
        self.device = device
        self.weights = weights
        self.aggregation = aggregation
        self.model_types = model_types or ["efficientnet_b0"] * len(model_paths)
        self.embedding_models = embedding_models or [None] * len(model_paths)
        
        self.models = self._load_models()
        
    def _load_model(
        self,
        path: str,
        backbone: str,
        embedding_model: Optional[str],
    ) -> nn.Module:
        """Load a single model from checkpoint."""
        
        if embedding_model and embedding_model in ["yamnet", "simple", "perch"]:
            if not PERCH_AVAILABLE:
                print(f"Warning: {embedding_model} not available, using efficientnet_b0")
                backbone = "efficientnet_b0"
            else:
                model = create_embedding_model(
                    model_type=embedding_model,
                    num_classes=self.num_classes,
                    pretrained=False,
                )
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.to(self.device)
                model.eval()
                return model
        
        model = BirdClefModel(
            num_classes=self.num_classes,
            backbone=backbone,
            pretrained=False,
        )
        
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        
        return model

    def _load_models(self) -> List[nn.Module]:
        """Load all models from checkpoints."""
        models = []
        
        for path, backbone, emb_model in zip(
            self.model_paths, self.model_types, self.embedding_models
        ):
            print(f"Loading model: {path} ({backbone})")
            model = self._load_model(path, backbone, emb_model)
            models.append(model)
        
        return models

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with ensemble.
        
        Args:
            x: Input tensor (batch, 1, freq, time)
        
        Returns:
            Averaged probabilities
        """
        all_probs = []
        
        x = x.to(self.device)
        
        for model, weight in zip(self.models, self.weights or [1.0] * len(self.models)):
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits) * weight
                all_probs.append(probs)
        
        if self.aggregation == "average":
            return torch.stack(all_probs).sum(dim=0)
        elif self.aggregation == "max":
            return torch.stack(all_probs).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def predict_batch(self, dataloader) -> np.ndarray:
        """Predict on a dataloader."""
        all_probs = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            probs = self.predict(inputs)
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)


def create_ensemble_from_dir(
    checkpoint_dir: str,
    num_classes: int = 234,
    device: str = "cpu",
    pattern: str = "*.pt",
    weights: Optional[List[float]] = None,
    aggregation: str = "average",
) -> EnsemblePredictor:
    """
    Create ensemble from all models in a directory.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        num_classes: Number of classes
        device: Device to run on
        pattern: Glob pattern for model files
        weights: Model weights
        aggregation: 'average' or 'max'
    
    Returns:
        EnsemblePredictor instance
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_paths = sorted(checkpoint_dir.glob(pattern))
    
    if not model_paths:
        raise ValueError(f"No models found in {checkpoint_dir} matching {pattern}")
    
    print(f"Found {len(model_paths)} models in {checkpoint_dir}")
    
    return EnsemblePredictor(
        model_paths=[str(p) for p in model_paths],
        num_classes=num_classes,
        device=device,
        weights=weights,
        aggregation=aggregation,
    )


def create_ensemble_from_config(
    config_path: str,
    num_classes: int = 234,
    device: str = "cpu",
) -> EnsemblePredictor:
    """
    Create ensemble from JSON config.
    
    Config format:
    ```json
    {
        "models": [
            {"path": "models/best_b0.pt", "weight": 0.3, "backbone": "efficientnet_b0"},
            {"path": "models/best_b2.pt", "weight": 0.3, "backbone": "efficientnet_b2"},
            {"path": "models/best_short.pt", "weight": 0.2, "backbone": "efficientnet_b0", "embedding_model": null},
            {"path": "models/perch.pt", "weight": 0.2, "embedding_model": "perch"}
        ],
        "aggregation": "average"
    }
    ```
    
    Args:
        config_path: Path to JSON config
        num_classes: Number of classes
        device: Device to run on
    
    Returns:
        EnsemblePredictor instance
    """
    with open(config_path) as f:
        config = json.load(f)
    
    model_paths = []
    weights = []
    model_types = []
    embedding_models = []
    
    for model_config in config.get("models", []):
        model_paths.append(model_config["path"])
        weights.append(model_config.get("weight", 1.0))
        model_types.append(model_config.get("backbone", "efficientnet_b0"))
        embedding_models.append(model_config.get("embedding_model", None))
    
    return EnsemblePredictor(
        model_paths=model_paths,
        num_classes=num_classes,
        device=device,
        weights=weights,
        aggregation=config.get("aggregation", "average"),
        model_types=model_types,
        embedding_models=embedding_models,
    )


def create_perch_ensemble(
    model_paths: List[str],
    num_classes: int = 234,
    device: str = "cpu",
    weights: Optional[List[float]] = None,
    aggregation: str = "average",
) -> EnsemblePredictor:
    """
    Create an ensemble of PERCH models.
    
    Args:
        model_paths: List of paths to PERCH model checkpoints
        num_classes: Number of classes
        device: Device to run on
        weights: Weights for each model
        aggregation: 'average' or 'max'
    
    Returns:
        EnsemblePredictor with PERCH models
    """
    if not PERCH_AVAILABLE:
        raise RuntimeError("PERCH not available. Install audioclass[perch,tensorflow]")
    
    embedding_models = ["perch"] * len(model_paths)
    model_types = ["efficientnet_b0"] * len(model_paths)
    
    return EnsemblePredictor(
        model_paths=model_paths,
        num_classes=num_classes,
        device=device,
        weights=weights,
        aggregation=aggregation,
        model_types=model_types,
        embedding_models=embedding_models,
    )


class MixedEnsemblePredictor:
    """Ensemble that handles mixed input types (spectrogram + waveform).
    
    This predictor automatically routes inputs to appropriate models:
    - Spectrogram models: EfficientNet, etc.
    - Waveform models: PERCH, YAMNet
    """
    
    def __init__(
        self,
        spectrogram_models: List[nn.Module],
        waveform_models: List[nn.Module],
        weights: Optional[List[float]] = None,
        aggregation: str = "average",
        device: str = "cpu",
    ):
        """
        Initialize mixed ensemble predictor.
        
        Args:
            spectrogram_models: Models that expect spectrograms (batch, 1, freq, time)
            waveform_models: Models that expect waveforms (batch, 160000)
            weights: Weights for each model
            aggregation: 'average' or 'max'
            device: Device to run on
        """
        self.spectrogram_models = nn.ModuleList(spectrogram_models)
        self.waveform_models = nn.ModuleList(waveform_models)
        self.device = device
        
        total_models = len(spectrogram_models) + len(waveform_models)
        self.weights = weights if weights else [1.0 / total_models] * total_models
        self.aggregation = aggregation
        
        if len(self.weights) != total_models:
            raise ValueError("Number of weights must match total number of models")
    
    def predict(
        self,
        spectrograms: Optional[torch.Tensor] = None,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict with mixed ensemble.
        
        Args:
            spectrograms: Input spectrograms (batch, 1, freq, time)
            waveforms: Input waveforms (batch, 160000)
        
        Returns:
            Averaged probabilities
        """
        all_probs = []
        
        for i, model in enumerate(self.spectrogram_models):
            if spectrograms is None:
                continue
            weight = self.weights[i]
            with torch.no_grad():
                spectrograms = spectrograms.to(self.device)
                logits = model(spectrograms)
                probs = torch.sigmoid(logits) * weight
                all_probs.append(probs)
        
        for i, model in enumerate(self.waveform_models):
            if waveforms is None:
                continue
            weight = self.weights[len(self.spectrogram_models) + i]
            with torch.no_grad():
                waveforms = waveforms.to(self.device)
                logits = model(waveforms)
                probs = torch.sigmoid(logits) * weight
                all_probs.append(probs)
        
        if not all_probs:
            raise ValueError("No inputs provided")
        
        if self.aggregation == "average":
            return torch.stack(all_probs).sum(dim=0)
        elif self.aggregation == "max":
            return torch.stack(all_probs).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

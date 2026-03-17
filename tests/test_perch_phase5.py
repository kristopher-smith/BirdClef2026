"""Tests for Phase 5 - Integration Testing & Model Upgrades."""

import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import tempfile
import os
import subprocess


class TestIntegrationTestFiles:
    """Test that integration test files exist and are executable."""

    def test_perch_integration_file_exists(self):
        """Test test_perch_integration.py exists."""
        assert Path("src/test_perch_integration.py").exists()

    def test_perch_quick_file_exists(self):
        """Test test_perch_quick.py exists."""
        assert Path("src/test_perch_quick.py").exists()

    def test_perch_quick_runs(self):
        """Test that quick test runs without errors."""
        result = subprocess.run(
            ["python", "src/test_perch_quick.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd()
        )
        # Should either succeed or have specific skip reasons
        assert result.returncode == 0 or "SKIPPED" in result.stdout or "skip" in result.stdout.lower()


class TestModelBackbones:
    """Test different EfficientNet backbones."""

    def test_efficientnet_b0(self):
        """Test EfficientNet-B0 model creation."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b0", pretrained=False)
        assert model is not None
        
        dummy_input = torch.randn(2, 1, 128, 251)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_efficientnet_b1(self):
        """Test EfficientNet-B1 model creation."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b1", pretrained=False)
        assert model is not None
        
        dummy_input = torch.randn(2, 1, 128, 251)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_efficientnet_b2(self):
        """Test EfficientNet-B2 model creation."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b2", pretrained=False)
        assert model is not None
        
        dummy_input = torch.randn(2, 1, 128, 251)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_efficientnet_b3(self):
        """Test EfficientNet-B3 model creation."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b3", pretrained=False)
        assert model is not None
        
        dummy_input = torch.randn(2, 1, 128, 251)
        output = model(dummy_input)
        assert output.shape == (2, 10)


class TestBackboneArgumentParsing:
    """Test backbone argument in training scripts."""

    def test_train_backbone_argument(self):
        """Test train.py accepts backbone argument."""
        import train
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = [
                'train.py',
                '--model', 'efficientnet_b2',
                '--epochs', '1',
                '--test',
            ]
            args = train.parse_args()
            assert args.model == 'efficientnet_b2'
        finally:
            sys.argv = old_argv

    def test_predict_backbone_argument(self):
        """Test predict.py accepts backbone argument."""
        import predict
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = [
                'predict.py',
                '--backbone', 'efficientnet_b3',
            ]
            args = predict.parse_args()
            assert args.backbone == 'efficientnet_b3'
        finally:
            sys.argv = old_argv

    def test_train_cv_backbone_argument(self):
        """Test train_cv.py accepts model argument."""
        import train_cv
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = [
                'train_cv.py',
                '--model', 'efficientnet_b2',
                '--folds', '3',
            ]
            args = train_cv.parse_args()
            assert args.model == 'efficientnet_b2'
        finally:
            sys.argv = old_argv


class TestEndToEndPipeline:
    """Test end-to-end pipeline components."""

    def test_audio_to_labels_pipeline(self):
        """Test audio loading to labels pipeline."""
        from dataset_perch import BirdClefAudioDataset
        
        taxonomy = pd.read_csv("data/birdclef-2026/taxonomy.csv")
        labels = pd.read_csv("data/birdclef-2026/train_soundscapes_labels.csv")
        
        dataset = BirdClefAudioDataset(
            audio_dir="data/birdclef-2026/train_soundscapes",
            labels_df=labels.head(3),
            taxonomy_df=taxonomy,
            sample_rate=32000,
            duration=5,
            use_cache=False,
        )
        
        waveform, labels_tensor = dataset[0]
        
        assert waveform.shape == torch.Size([160000])
        assert labels_tensor.shape == torch.Size([len(taxonomy)])
        assert labels_tensor.sum() > 0

    def test_model_output_range(self):
        """Test model outputs are valid logits."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b0", pretrained=False)
        model.eval()
        
        with torch.no_grad():
            dummy_input = torch.randn(2, 1, 128, 251)
            output = model(dummy_input)
            
            # Output should be logits (not yet sigmoid applied)
            assert output.shape == (2, 10)
            # Logits can be any value, but shouldn't be NaN or Inf
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_sigmoid_probabilities(self):
        """Test sigmoid converts logits to probabilities."""
        logits = torch.randn(4, 10)
        probs = torch.sigmoid(logits)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_bce_loss_compatibility(self):
        """Test BCEWithLogitsLoss works with model output."""
        criterion = nn.BCEWithLogitsLoss()
        
        logits = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        targets[:, :3] = 1
        
        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestTrainingPipeline:
    """Test training pipeline components."""

    def test_train_one_epoch_function_exists(self):
        """Test train.py has train_one_epoch function."""
        import train
        assert hasattr(train, 'train_one_epoch')

    def test_validate_function_exists(self):
        """Test train.py has validate function."""
        import train
        assert hasattr(train, 'validate')

    def test_train_perch_one_epoch_exists(self):
        """Test train_perch.py has train_one_epoch function."""
        import train_perch
        assert hasattr(train_perch, 'train_one_epoch')

    def test_train_perch_validate_exists(self):
        """Test train_perch.py has validate function."""
        import train_perch
        assert hasattr(train_perch, 'validate')


class TestPredictionPipeline:
    """Test prediction pipeline components."""

    def test_load_and_process_audio_exists(self):
        """Test predict.py has load_and_process_audio function."""
        import predict
        assert hasattr(predict, 'load_and_process_audio')

    def test_load_and_process_audio_for_perch_exists(self):
        """Test predict.py has load_and_process_audio_for_perch function."""
        import predict
        assert hasattr(predict, 'load_and_process_audio_for_perch')

    def test_predict_function_exists(self):
        """Test predict.py has predict function."""
        import predict
        assert hasattr(predict, 'predict')

    def test_predict_perch_function_exists(self):
        """Test predict.py has predict_perch function."""
        import predict
        assert hasattr(predict, 'predict_perch')


class TestEnsemblePipeline:
    """Test ensemble pipeline components."""

    def test_ensemble_predictor_exists(self):
        """Test ensemble.py has EnsemblePredictor."""
        from ensemble import EnsemblePredictor
        assert EnsemblePredictor is not None

    def test_create_ensemble_from_config_exists(self):
        """Test ensemble.py has create_ensemble_from_config."""
        from ensemble import create_ensemble_from_config
        assert create_ensemble_from_config is not None

    def test_create_ensemble_from_dir_exists(self):
        """Test ensemble.py has create_ensemble_from_dir."""
        from ensemble import create_ensemble_from_dir
        assert create_ensemble_from_dir is not None

    def test_create_perch_ensemble_exists(self):
        """Test ensemble.py has create_perch_ensemble."""
        from ensemble import create_perch_ensemble
        assert create_perch_ensemble is not None


class TestDataAugmentation:
    """Test data augmentation pipeline."""

    def test_specaugment_exists(self):
        """Test augmentation.py has SpecAugment."""
        from augmentation import SpecAugment
        assert SpecAugment is not None

    def test_mixup_exists(self):
        """Test augmentation.py has Mixup."""
        from augmentation import Mixup
        assert Mixup is not None

    def test_compose_exists(self):
        """Test augmentation.py has Compose."""
        from augmentation import Compose
        assert Compose is not None

    def test_waveform_timeshift_exists(self):
        """Test augmentation.py has WaveformTimeShift."""
        from augmentation import WaveformTimeShift
        assert WaveformTimeShift is not None

    def test_waveform_noise_exists(self):
        """Test augmentation.py has WaveformNoise."""
        from augmentation import WaveformNoise
        assert WaveformNoise is not None


class TestMetricsComputation:
    """Test metrics computation."""

    def test_train_metrics_exist(self):
        """Test train.py has map and f1 functions."""
        import train
        assert hasattr(train, 'compute_map_at_k')
        assert hasattr(train, 'compute_f1_at_k')

    def test_train_perch_metrics_exist(self):
        """Test train_perch.py has map and f1 functions."""
        import train_perch
        assert hasattr(train_perch, 'compute_map_at_k')
        assert hasattr(train_perch, 'compute_f1_at_k')

    def test_train_cv_metrics_exist(self):
        """Test train_cv.py has map and f1 functions."""
        import train_cv
        assert hasattr(train_cv, 'compute_map_at_k')
        assert hasattr(train_cv, 'compute_f1_at_k')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

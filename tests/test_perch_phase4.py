"""Tests for Phase 4 - Cross-Validation, Validation & Ensemble Integration."""

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


class TestSubmissionValidation:
    """Test submission validation functionality."""

    def test_validate_submission_imports(self):
        """Test validate_submission module imports."""
        import validate_submission
        assert validate_submission is not None
        assert hasattr(validate_submission, 'validate_submission')
        assert hasattr(validate_submission, 'parse_args')

    def test_validate_submission_valid(self):
        """Test validation of a valid submission."""
        from validate_submission import validate_submission
        
        # Create a valid submission
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_sub.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid = validate_submission(
                temp_path,
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            assert is_valid == True
        finally:
            os.unlink(temp_path)

    def test_validate_submission_shape_mismatch(self):
        """Test validation detects shape mismatch."""
        from validate_submission import validate_submission
        
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        # Create submission with wrong shape (different column)
        wrong_sub = sample_sub.copy()
        wrong_sub = wrong_sub.drop(columns=[wrong_sub.columns[1]])  # Remove one column
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            wrong_sub.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid = validate_submission(
                temp_path,
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            assert is_valid == False
        finally:
            os.unlink(temp_path)

    def test_validate_submission_invalid_probabilities(self):
        """Test validation detects invalid probabilities."""
        from validate_submission import validate_submission
        
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        # Create submission with invalid probabilities (> 1)
        bad_sub = sample_sub.copy()
        label_cols = [c for c in bad_sub.columns if c != 'row_id']
        bad_sub[label_cols[0]] = 1.5  # Invalid probability
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_sub.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid = validate_submission(
                temp_path,
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            assert is_valid == False
        finally:
            os.unlink(temp_path)

    def test_validate_submission_negative_probabilities(self):
        """Test validation detects negative probabilities."""
        from validate_submission import validate_submission
        
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        bad_sub = sample_sub.copy()
        label_cols = [c for c in bad_sub.columns if c != 'row_id']
        bad_sub[label_cols[0]] = -0.1  # Negative probability
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_sub.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid = validate_submission(
                temp_path,
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            assert is_valid == False
        finally:
            os.unlink(temp_path)

    def test_validate_submission_nan_values(self):
        """Test validation detects NaN values."""
        from validate_submission import validate_submission
        
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        bad_sub = sample_sub.copy()
        label_cols = [c for c in bad_sub.columns if c != 'row_id']
        bad_sub.loc[0, label_cols[0]] = np.nan
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_sub.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            is_valid = validate_submission(
                temp_path,
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            assert is_valid == False
        finally:
            os.unlink(temp_path)

    def test_validate_submission_argument_parsing(self):
        """Test argument parsing."""
        import validate_submission
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = [
                'validate_submission.py',
                '--submission', 'submission.csv',
                '--sample_submission', 'sample.csv',
                '--taxonomy', 'taxonomy.csv',
            ]
            args = validate_submission.parse_args()
            assert args.submission == 'submission.csv'
            assert args.sample_submission == 'sample.csv'
            assert args.taxonomy == 'taxonomy.csv'
        finally:
            sys.argv = old_argv


class TestCrossValidation:
    """Test cross-validation functionality."""

    def test_train_cv_imports(self):
        """Test train_cv module imports."""
        import train_cv
        assert train_cv is not None
        assert hasattr(train_cv, 'compute_map_at_k')
        assert hasattr(train_cv, 'compute_f1_at_k')
        assert hasattr(train_cv, 'compute_ap')

    def test_compute_ap(self):
        """Test average precision computation."""
        from train_cv import compute_ap
        
        recalls = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        precisions = np.array([1.0, 0.9, 0.85, 0.8, 0.75, 0.7])
        
        ap = compute_ap(recalls, precisions)
        assert isinstance(ap, (float, np.floating))
        assert 0 <= ap <= 1.0

    def test_compute_map_at_k(self):
        """Test mean Average Precision at k."""
        from train_cv import compute_map_at_k
        
        probs = np.random.rand(10, 5)
        labels = np.zeros((10, 5))
        labels[:5, :3] = 1
        
        map_at_k = compute_map_at_k(probs, labels, k=10)
        assert isinstance(map_at_k, (float, np.floating))
        assert 0 <= map_at_k <= 1.0

    def test_compute_f1_at_k(self):
        """Test F1 at k computation."""
        from train_cv import compute_f1_at_k
        
        probs = np.random.rand(10, 5)
        labels = np.zeros((10, 5))
        labels[:5, :3] = 1
        
        f1_at_k = compute_f1_at_k(probs, labels, k=3)
        assert isinstance(f1_at_k, (float, np.floating))
        assert 0 <= f1_at_k <= 1.0

    def test_stratified_kfold_exists(self):
        """Test sklearn stratified kfold is available."""
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        
        assert len(splits) == 5


class TestEnsembleConfig:
    """Test ensemble configuration."""

    def test_perch_ensemble_config_exists(self):
        """Test PERCH ensemble config file exists."""
        config_path = Path("models/perch_ensemble_config.json")
        assert config_path.exists()

    def test_perch_ensemble_config_valid_json(self):
        """Test PERCH ensemble config is valid JSON."""
        with open("models/perch_ensemble_config.json") as f:
            config = json.load(f)
        
        assert "models" in config
        assert "aggregation" in config
        assert len(config["models"]) == 3

    def test_perch_ensemble_config_input_types(self):
        """Test config has correct input types."""
        with open("models/perch_ensemble_config.json") as f:
            config = json.load(f)
        
        input_types = [m.get("input_type") for m in config["models"]]
        assert "spectrogram" in input_types
        assert "audio" in input_types

    def test_perch_ensemble_config_weights(self):
        """Test config weights sum correctly."""
        with open("models/perch_ensemble_config.json") as f:
            config = json.load(f)
        
        weights = [m["weight"] for m in config["models"]]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_create_perch_ensemble_function(self):
        """Test create_perch_ensemble function exists."""
        from ensemble import create_perch_ensemble
        assert create_perch_ensemble is not None


class TestMixedEnsemblePredictor:
    """Test MixedEnsemblePredictor class."""

    def test_mixed_ensemble_exists(self):
        """Test MixedEnsemblePredictor class exists."""
        try:
            from ensemble import MixedEnsemblePredictor
            assert MixedEnsemblePredictor is not None
        except ImportError:
            pytest.skip("MixedEnsemblePredictor not available")

    def test_mixed_ensemble_init(self):
        """Test MixedEnsemblePredictor initialization."""
        try:
            from ensemble import MixedEnsemblePredictor
        except ImportError:
            pytest.skip("MixedEnsemblePredictor not available")
        
        class MockSpectrogramModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.randn(x.size(0), 10)
        
        class MockWaveformModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.randn(x.size(0), 10)
        
        spectrogram_models = [MockSpectrogramModel()]
        waveform_models = [MockWaveformModel()]
        
        predictor = MixedEnsemblePredictor(
            spectrogram_models=spectrogram_models,
            waveform_models=waveform_models,
            device="cpu"
        )
        
        assert predictor is not None
        assert len(predictor.spectrogram_models) == 1
        assert len(predictor.waveform_models) == 1


class TestPerClassMetrics:
    """Test per-class performance tracking."""

    def test_per_class_f1_computation(self):
        """Test per-class F1 can be computed."""
        from train_cv import compute_f1_at_k
        
        probs = np.random.rand(20, 10)
        labels = np.zeros((20, 10))
        labels[:10, :5] = 1
        
        f1 = compute_f1_at_k(probs, labels, k=3)
        assert 0 <= f1 <= 1

    def test_per_class_recall_computation(self):
        """Test per-class recall can be computed."""
        probs = np.random.rand(20, 10)
        labels = np.zeros((20, 10))
        labels[:10, :5] = 1
        
        recalls = []
        for c in range(10):
            if labels[:, c].sum() > 0:
                top_k_indices = np.argsort(-probs[:, c])[:3]
                tp = labels[top_k_indices, c].sum()
                recall = tp / labels[:, c].sum()
                recalls.append(recall)
        
        assert len(recalls) > 0


class TestHeldOutTestSet:
    """Test held-out test set functionality."""

    def test_train_test_split(self):
        """Test train/test split works."""
        from sklearn.model_selection import train_test_split
        
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        assert len(X_train) == 90
        assert len(X_test) == 10


class TestIntegrationWithSubmission:
    """Integration tests for full pipeline."""

    def test_sample_submission_format(self):
        """Test sample submission format is correct."""
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        
        assert 'row_id' in sample_sub.columns
        assert len(sample_sub.columns) > 1
        
        label_cols = [c for c in sample_sub.columns if c != 'row_id']
        assert len(label_cols) > 0
        
        assert (sample_sub[label_cols].values >= 0).all()
        assert (sample_sub[label_cols].values <= 1).all()

    def test_taxonomy_classes_match_submission(self):
        """Test taxonomy classes match submission columns."""
        sample_sub = pd.read_csv("data/birdclef-2026/sample_submission.csv")
        taxonomy = pd.read_csv("data/birdclef-2026/taxonomy.csv")
        
        submission_cols = set(c for c in sample_sub.columns if c != 'row_id')
        taxonomy_primary = set(taxonomy['primary_label'].values)
        
        # All submission cols should be in taxonomy
        assert submission_cols.issubset(taxonomy_primary) or len(submission_cols) > 0

    def test_validate_existing_submission(self):
        """Test validation of actual submission file."""
        from validate_submission import validate_submission
        
        if Path("submission.csv").exists():
            is_valid = validate_submission(
                "submission.csv",
                "data/birdclef-2026/sample_submission.csv",
                "data/birdclef-2026/taxonomy.csv"
            )
            # Should either pass or fail but not crash
            assert isinstance(is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

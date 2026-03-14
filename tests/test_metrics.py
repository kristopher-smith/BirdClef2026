"""Tests for metrics calculations."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from train import compute_ap, compute_map_at_k, compute_f1_at_k


class TestAveragePrecision:
    """Tests for compute_ap function."""

    def test_perfect_recall_precision(self):
        """Test AP with perfect recall."""
        recalls = np.array([0.0, 0.5, 1.0])
        precisions = np.array([1.0, 0.8, 0.5])
        ap = compute_ap(recalls, precisions)
        assert 0.0 <= ap <= 1.0

    def test_zero_recall(self):
        """Test AP with zero recall."""
        recalls = np.array([0.0, 0.0, 0.0])
        precisions = np.array([0.0, 0.0, 0.0])
        ap = compute_ap(recalls, precisions)
        assert ap == 0.0


class TestMAP:
    """Tests for compute_map_at_k function."""

    def test_map_perfect_predictions(self):
        """Test mAP with perfect predictions."""
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        labels = np.array([
            [1, 0],
            [1, 0],
        ])
        map_score = compute_map_at_k(probs, labels, k=2)
        assert 0.0 <= map_score <= 1.0

    def test_map_no_positives(self):
        """Test mAP with no positive labels."""
        probs = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        labels = np.array([
            [0, 0],
            [0, 0],
        ])
        map_score = compute_map_at_k(probs, labels, k=2)
        assert map_score == 0.0

    def test_map_different_k(self):
        """Test mAP with different k values."""
        probs = np.array([
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ])
        labels = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        
        map_1 = compute_map_at_k(probs, labels, k=1)
        map_2 = compute_map_at_k(probs, labels, k=2)
        map_3 = compute_map_at_k(probs, labels, k=3)
        
        assert 0.0 <= map_1 <= 1.0
        assert 0.0 <= map_2 <= 1.0
        assert 0.0 <= map_3 <= 1.0


class TestF1Score:
    """Tests for compute_f1_at_k function."""

    def test_f1_perfect(self):
        """Test F1 with perfect predictions."""
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
        ])
        labels = np.array([
            [1, 0],
            [1, 0],
        ])
        f1 = compute_f1_at_k(probs, labels, k=1)
        assert f1 == 1.0

    def test_f1_no_predictions(self):
        """Test F1 with no predictions above threshold."""
        probs = np.array([
            [0.1, 0.1],
            [0.1, 0.1],
        ])
        labels = np.array([
            [1, 1],
            [1, 1],
        ])
        f1 = compute_f1_at_k(probs, labels, k=1)
        assert 0.0 <= f1 <= 1.0

    def test_f1_different_k(self):
        """Test F1 with different k values."""
        probs = np.array([
            [0.9, 0.5, 0.1],
            [0.8, 0.3, 0.1],
        ])
        labels = np.array([
            [1, 0, 0],
            [1, 1, 0],
        ])
        
        f1_1 = compute_f1_at_k(probs, labels, k=1)
        f1_2 = compute_f1_at_k(probs, labels, k=2)
        
        assert 0.0 <= f1_1 <= 1.0
        assert 0.0 <= f1_2 <= 1.0


class TestMetricsEdgeCases:
    """Edge case tests for metrics."""

    def test_single_sample(self):
        """Test with single sample."""
        probs = np.array([[0.9, 0.1]])
        labels = np.array([[1, 0]])
        map_score = compute_map_at_k(probs, labels, k=2)
        f1_score = compute_f1_at_k(probs, labels, k=1)
        assert 0.0 <= map_score <= 1.0
        assert 0.0 <= f1_score <= 1.0

    def test_large_k(self):
        """Test with k larger than number of classes."""
        probs = np.array([
            [0.9, 0.1, 0.0, 0.0],
        ])
        labels = np.array([
            [1, 0, 0, 0],
        ])
        map_score = compute_map_at_k(probs, labels, k=10)
        assert 0.0 <= map_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

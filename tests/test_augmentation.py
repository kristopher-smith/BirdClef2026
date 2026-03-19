"""Tests for augmentation.py"""

import sys
sys.path.insert(0, 'src')

import torch
import pytest
from augmentation import (
    SpecAugment,
    TimeShift,
    TimeStretch,
    Mixup,
    Compose,
)


class TestSpecAugment:
    """Tests for SpecAugment."""

    def test_output_shape(self):
        """Test that output shape is preserved."""
        aug = SpecAugment(freq_mask_param=10, time_mask_param=20)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_values_clipped(self):
        """Test that masked values are zero."""
        aug = SpecAugment(freq_mask_param=30, time_mask_param=50, num_freq_masks=2, num_time_masks=2)
        x = torch.ones(1, 128, 313)
        out = aug(x)
        assert (out <= 1.0).all()

    def test_deterministic(self):
        """Test that augmentation is stochastic."""
        aug = SpecAugment(freq_mask_param=10, time_mask_param=10)
        x = torch.ones(1, 128, 313)
        out1 = aug(x.clone())
        out2 = aug(x.clone())
        assert not torch.equal(out1, out2)


class TestTimeShift:
    """Tests for TimeShift."""

    def test_output_shape(self):
        """Test that output shape is preserved."""
        aug = TimeShift(max_shift=30)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_no_shift(self):
        """Test with zero shift (or very small range)."""
        aug = TimeShift(max_shift=1)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_preserves_values(self):
        """Test that values are just shifted, not modified."""
        aug = TimeShift(max_shift=10)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert torch.allclose(out.sum(), x.sum(), rtol=1e-5)


class TestTimeStretch:
    """Tests for TimeStretch."""

    def test_output_shape(self):
        """Test that output shape is preserved."""
        aug = TimeStretch(min_rate=0.9, max_rate=1.1)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_no_stretch(self):
        """Test with rate close to 1."""
        aug = TimeStretch(min_rate=0.99, max_rate=1.01)
        x = torch.randn(1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape


class TestMixup:
    """Tests for Mixup."""

    def test_output_shape(self):
        """Test that output shape is preserved."""
        mixup = Mixup(alpha=0.4)
        x = torch.randn(4, 1, 128, 313)
        y = torch.randn(4, 234)
        x_mixed, y_mixed = mixup(x, y)
        assert x_mixed.shape == x.shape
        assert y_mixed.shape == y.shape

    def test_labels_in_range(self):
        """Test that mixed labels are in valid range."""
        mixup = Mixup(alpha=0.4)
        x = torch.randn(4, 1, 128, 313)
        y = torch.eye(234)[:4]
        x_mixed, y_mixed = mixup(x, y)
        assert (y_mixed >= 0).all()
        assert (y_mixed <= 1).all()

    def test_no_mixup(self):
        """Test with alpha=0 (no mixing)."""
        mixup = Mixup(alpha=0.0)
        x = torch.randn(4, 1, 128, 313)
        y = torch.randn(4, 234)
        x_mixed, y_mixed = mixup(x, y)
        assert torch.equal(x, x_mixed)
        assert torch.equal(y, y_mixed)

    def test_single_sample(self):
        """Test with single sample (no mixing possible)."""
        mixup = Mixup(alpha=0.4)
        x = torch.randn(1, 1, 128, 313)
        y = torch.randn(1, 234)
        x_mixed, y_mixed = mixup(x, y)
        assert x_mixed.shape == x.shape
        assert y_mixed.shape == y.shape


class TestCompose:
    """Tests for Compose."""

    def test_compose_order(self):
        """Test that composes are applied in order."""
        transforms = [
            TimeShift(max_shift=1),
            TimeShift(max_shift=1),
        ]
        compose = Compose(transforms)
        x = torch.randn(1, 128, 313)
        out = compose(x)
        assert out.shape == x.shape

    def test_empty_compose(self):
        """Test with empty compose."""
        compose = Compose([])
        x = torch.randn(1, 128, 313)
        out = compose(x)
        assert torch.equal(out, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for model.py"""

import sys
sys.path.insert(0, 'src')

import torch
import pytest
from model import BirdClefModel, BirdClefModelWithPool, SimpleCNN, get_device


class TestBirdClefModel:
    """Tests for BirdClefModel."""

    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"])
    def test_model_forward_shape(self, backbone):
        """Test that model outputs correct shape."""
        model = BirdClefModel(backbone=backbone, pretrained=False, num_classes=234)
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 234)

    def test_model_single_input(self):
        """Test model with single input."""
        model = BirdClefModel(backbone='efficientnet_b0', pretrained=False, num_classes=100)
        x = torch.randn(1, 1, 128, 313)
        out = model(x)
        assert out.shape == (1, 100)

    def test_model_three_channel_input(self):
        """Test model with 3-channel input (grayscale repeated)."""
        model = BirdClefModel(backbone='efficientnet_b0', pretrained=False, num_classes=50)
        x = torch.randn(2, 3, 128, 313)
        out = model(x)
        assert out.shape == (2, 50)

    def test_model_backbone_equivalence(self):
        """Test that different backbones produce same output shape."""
        x = torch.randn(1, 1, 128, 313)
        outputs = {}
        for backbone in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            model = BirdClefModel(backbone=backbone, pretrained=False, num_classes=234)
            with torch.no_grad():
                outputs[backbone] = model(x).shape
        assert all(v == (1, 234) for v in outputs.values())


class TestBirdClefModelWithPool:
    """Tests for BirdClefModelWithPool."""

    @pytest.mark.parametrize("pool_type", ["avg", "max"])
    def test_pool_types(self, pool_type):
        """Test different pooling types."""
        model = BirdClefModelWithPool(
            backbone='efficientnet_b0',
            pretrained=False,
            num_classes=234,
            pool_type=pool_type
        )
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 234)


class TestSimpleCNN:
    """Tests for SimpleCNN."""

    def test_simple_cnn_output_shape(self):
        """Test SimpleCNN output shape."""
        model = SimpleCNN(num_classes=234, input_channels=1)
        x = torch.randn(2, 1, 128, 313)
        out = model(x)
        assert out.shape == (2, 234)

    def test_simple_cnn_different_classes(self):
        """Test with different number of classes."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 10)


class TestGetDevice:
    """Tests for get_device."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a valid device."""
        device = get_device()
        assert device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

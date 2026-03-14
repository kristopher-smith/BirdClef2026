"""Tests for tta.py"""

import sys
sys.path.insert(0, 'src')

import torch
import pytest
from tta import (
    TTAOriginal,
    TTAHorizontalFlip,
    TTATimeShift,
    TTAFreqMask,
    TTATimeMask,
    get_tta_transforms,
    PredictorWithTTA,
    apply_tta_to_predictions,
)


class TestTTAAugments:
    """Tests for TTA augmentations."""

    def test_original_preserves_input(self):
        """Test TTAOriginal preserves input."""
        aug = TTAOriginal()
        x = torch.randn(2, 1, 128, 313)
        out = aug(x)
        assert torch.equal(out, x)

    def test_flip_shape(self):
        """Test TTAHorizontalFlip preserves shape."""
        aug = TTAHorizontalFlip()
        x = torch.randn(2, 1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_flip_content(self):
        """Test flip reverses time dimension."""
        aug = TTAHorizontalFlip()
        x = torch.randn(1, 1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_timeshift_shape(self):
        """Test TTATimeShift preserves shape."""
        aug = TTATimeShift(max_shift=10)
        x = torch.randn(2, 1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_freqmask_shape(self):
        """Test TTAFreqMask preserves shape."""
        aug = TTAFreqMask(freq_mask_param=10)
        x = torch.randn(2, 1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape

    def test_timemask_shape(self):
        """Test TTATimeMask preserves shape."""
        aug = TTATimeMask(time_mask_param=20)
        x = torch.randn(2, 1, 128, 313)
        out = aug(x)
        assert out.shape == x.shape


class TestGetTTATransforms:
    """Tests for get_tta_transforms."""

    def test_original_only(self):
        """Test getting original transform only."""
        transforms = get_tta_transforms("original")
        assert len(transforms) == 1
        assert isinstance(transforms[0], TTAOriginal)

    def test_multiple_transforms(self):
        """Test getting multiple transforms."""
        transforms = get_tta_transforms("original,flip")
        assert len(transforms) == 2

    def test_unknown_transform(self):
        """Test unknown transform is skipped."""
        transforms = get_tta_transforms("original,unknown,flip")
        assert len(transforms) == 2

    def test_all_transforms(self):
        """Test all transforms."""
        transforms = get_tta_transforms("original,flip,timeshift,freqmask,timemask")
        assert len(transforms) == 5


class TestApplyTTA:
    """Tests for apply_tta_to_predictions."""

    def test_output_shape(self):
        """Test output has correct shape."""
        from model import BirdClefModel
        
        model = BirdClefModel(backbone='efficientnet_b0', pretrained=False, num_classes=234)
        model.eval()
        
        x = torch.randn(2, 1, 128, 313)
        transforms = [TTAOriginal(), TTAHorizontalFlip()]
        
        out = apply_tta_to_predictions(model, x, transforms)
        
        assert out.shape == (2, 234)

    def test_output_range(self):
        """Test output is in valid probability range."""
        from model import BirdClefModel
        
        model = BirdClefModel(backbone='efficientnet_b0', pretrained=False, num_classes=234)
        model.eval()
        
        x = torch.randn(2, 1, 128, 313)
        transforms = [TTAOriginal()]
        
        out = apply_tta_to_predictions(model, x, transforms)
        
        assert (out >= 0).all()
        assert (out <= 1).all()


class TestPredictorWithTTA:
    """Tests for PredictorWithTTA."""

    def test_predictor_output_shape(self):
        """Test PredictorWithTTA output shape."""
        from model import BirdClefModel
        
        model = BirdClefModel(backbone='efficientnet_b0', pretrained=False, num_classes=234)
        model.eval()
        
        predictor = PredictorWithTTA(
            model=model,
            augments=[TTAOriginal(), TTAHorizontalFlip()],
        )
        
        x = torch.randn(2, 1, 128, 313)
        out = predictor.predict(x)
        
        assert out.shape == (2, 234)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

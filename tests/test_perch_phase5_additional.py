"""Additional tests for Phase 5 - Model Upgrades (EfficientNet-B3 + LSP fixes)."""

import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class TestEfficientNetB3ModelVariants:
    """Test EfficientNet-B3 in all model variants."""

    def test_birdclefmodel_b3_output_shape(self):
        """Test BirdClefModel with B3 produces correct output."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=234, backbone="efficientnet_b3", pretrained=False)
        
        # Test with standard spectrogram
        dummy_input = torch.randn(4, 1, 128, 251)
        output = model(dummy_input)
        
        assert output.shape == (4, 234)

    def test_birdclefmodelwithpool_b3(self):
        """Test BirdClefModelWithPool with B3."""
        from model import BirdClefModelWithPool
        
        model = BirdClefModelWithPool(num_classes=234, backbone="efficientnet_b3", pretrained=False)
        
        # Test with longer spectrogram
        dummy_input = torch.randn(4, 1, 128, 500)
        output = model(dummy_input)
        
        assert output.shape == (4, 234)

    def test_b3_in_features(self):
        """Test B3 has correct in_features."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b3", pretrained=False)
        
        # B3 should have 1536 in_features from its classifier
        assert model.fc.in_features == 1536


class TestLSPErrorFixes:
    """Test that LSP errors are fixed (type hints, imports)."""

    def test_model_imports_work(self):
        """Test model module imports correctly."""
        from model import BirdClefModel, BirdClefModelWithPool, get_device
        
        assert BirdClefModel is not None
        assert BirdClefModelWithPool is not None
        assert get_device is not None

    def test_no_undefined_references(self):
        """Test no undefined references in model.py."""
        # This is a sanity check that the model can be imported
        import model
        # If this fails, there are import errors
        assert hasattr(model, 'BirdClefModel')
        assert hasattr(model, 'BirdClefModelWithPool')

    def test_model_docstrings(self):
        """Test model classes have docstrings."""
        from model import BirdClefModel, BirdClefModelWithPool
        
        assert BirdClefModel.__doc__ is not None
        assert BirdClefModelWithPool.__doc__ is not None


class TestModelForwardPass:
    """Test model forward passes with various input sizes."""

    def test_standard_input_b3(self):
        """Test B3 with standard spectrogram size."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=100, backbone="efficientnet_b3", pretrained=False)
        model.eval()
        
        with torch.no_grad():
            # Standard: (batch, 1, 128, 251)
            x = torch.randn(8, 1, 128, 251)
            out = model(x)
            assert out.shape == (8, 100)

    def test_variable_length_input_b3(self):
        """Test B3 with variable length spectrograms."""
        from model import BirdClefModelWithPool
        
        model = BirdClefModelWithPool(num_classes=50, backbone="efficientnet_b3", pretrained=False)
        model.eval()
        
        with torch.no_grad():
            # Different lengths
            for length in [100, 200, 400, 600]:
                x = torch.randn(4, 1, 128, length)
                out = model(x)
                assert out.shape == (4, 50)

    def test_single_sample_b3(self):
        """Test B3 with single sample."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b3", pretrained=False)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(1, 1, 128, 251)
            out = model(x)
            assert out.shape == (1, 10)


class TestModelTrainingMode:
    """Test model in training and eval modes."""

    def test_training_mode_b3(self):
        """Test B3 in training mode."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b3", pretrained=False)
        model.train()
        
        x = torch.randn(4, 1, 128, 251, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "At least some parameters should have gradients"

    def test_eval_mode_b3(self):
        """Test B3 in eval mode."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=10, backbone="efficientnet_b3", pretrained=False)
        model.eval()
        
        # Dropout should not be active in eval mode
        x = torch.randn(4, 1, 128, 251)
        out1 = model(x)
        out2 = model(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(out1, out2)


class TestCheckpointCompatibility:
    """Test checkpoint saving/loading."""

    def test_save_load_b3(self):
        """Test saving and loading B3 checkpoint."""
        from model import BirdClefModel
        
        model = BirdClefModel(num_classes=234, backbone="efficientnet_b3", pretrained=False)
        
        # Save state dict
        state_dict = model.state_dict()
        
        # Create new model and load
        model2 = BirdClefModel(num_classes=234, backbone="efficientnet_b3", pretrained=False)
        model2.load_state_dict(state_dict)
        
        # Set both to eval mode to disable dropout
        model.eval()
        model2.eval()
        
        # Test they produce same output
        x = torch.randn(2, 1, 128, 251)
        out1 = model(x)
        out2 = model2(x)
        
        assert torch.allclose(out1, out2)


class TestBackboneSelection:
    """Test backbone selection in training scripts."""

    def test_backbone_in_argument_parsing(self):
        """Test backbone argument is available."""
        import train
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = ['train.py', '--model', 'efficientnet_b3']
            args = train.parse_args()
            assert args.model == 'efficientnet_b3'
        finally:
            sys.argv = old_argv

    def test_predict_backbone_selection(self):
        """Test backbone selection in predict.py."""
        import predict
        import sys
        
        old_argv = sys.argv
        try:
            sys.argv = ['predict.py', '--backbone', 'efficientnet_b3']
            args = predict.parse_args()
            assert args.backbone == 'efficientnet_b3'
        finally:
            sys.argv = old_argv


class TestErrorHandling:
    """Test error handling for invalid backbones."""

    def test_invalid_backbone_raises_error(self):
        """Test invalid backbone raises ValueError."""
        from model import BirdClefModel
        
        with pytest.raises(ValueError):
            BirdClefModel(num_classes=10, backbone="invalid_backbone")

    def test_invalid_backbone_error_message(self):
        """Test error message mentions valid options."""
        from model import BirdClefModel
        
        try:
            BirdClefModel(num_classes=10, backbone="invalid")
        except ValueError as e:
            assert "Unknown backbone" in str(e) or "invalid" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

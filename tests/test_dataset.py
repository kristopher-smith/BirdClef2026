"""Tests for dataset.py"""

import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDatasetStructure:
    """Tests for dataset module structure."""

    def test_dataset_imports(self):
        """Test that dataset module can be imported."""
        from dataset import BirdClefDataset, BirdClefShortClipDataset
        assert BirdClefDataset is not None
        assert BirdClefShortClipDataset is not None

    def test_dataset_class_exists(self):
        """Test that dataset classes exist."""
        from dataset import BirdClefDataset, BirdClefShortClipDataset, BirdClefTestDataset
        assert hasattr(BirdClefDataset, '__init__')
        assert hasattr(BirdClefShortClipDataset, '__init__')
        assert hasattr(BirdClefTestDataset, '__init__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

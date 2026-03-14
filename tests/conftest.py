"""Pytest configuration and fixtures."""

import sys
sys.path.insert(0, 'src')

import pytest
import torch


@pytest.fixture
def sample_spectrogram():
    """Sample spectrogram for testing."""
    return torch.randn(1, 1, 128, 313)


@pytest.fixture
def sample_batch():
    """Sample batch of spectrograms."""
    return torch.randn(4, 1, 128, 313)


@pytest.fixture
def sample_labels():
    """Sample multi-label targets."""
    return torch.randint(0, 2, (4, 234)).float()


@pytest.fixture
def num_classes():
    """Number of classes for testing."""
    return 234


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')

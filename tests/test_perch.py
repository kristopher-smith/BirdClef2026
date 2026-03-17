"""Tests for PERCH Integration - Raw audio waveform support."""

import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import torch


class TestAudioUtilities:
    """Test audio utility functions."""

    def test_load_audio_for_perch_basic(self):
        """Test basic audio loading for PERCH."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform = load_audio_for_perch(audio_path, sr=32000, duration=5.0, offset=0.0)
        
        assert waveform is not None
        assert waveform.dtype == np.float32
        assert len(waveform) == 160000  # 5 sec × 32kHz
        assert waveform.max() <= 1.0
        assert waveform.min() >= -1.0

    def test_load_audio_for_perch_with_offset(self):
        """Test audio loading with offset."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform1 = load_audio_for_perch(audio_path, sr=32000, duration=5.0, offset=0.0)
        waveform2 = load_audio_for_perch(audio_path, sr=32000, duration=5.0, offset=5.0)
        
        assert not np.array_equal(waveform1, waveform2), "Different offsets should produce different waveforms"

    def test_load_audio_segments(self):
        """Test loading multiple segments from audio file."""
        from audio import load_audio_segments
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        segments = load_audio_segments(audio_path, segment_duration=5.0, sr=32000)
        
        assert len(segments) > 0
        for waveform, row_id in segments:
            assert waveform.dtype == np.float32
            assert len(waveform) == 160000
            assert isinstance(row_id, str)

    def test_audio_normalization(self):
        """Test that audio is properly normalized to [-1, 1]."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform = load_audio_for_perch(audio_path, sr=32000, duration=5.0, normalize=True)
        
        max_val = np.abs(waveform).max()
        assert max_val <= 1.0 + 1e-6, "Waveform should be normalized to [-1, 1]"
        if max_val > 0:
            assert np.isclose(max_val, 1.0, atol=1e-6), "Max should be 1.0 after normalization"


class TestDatasetClasses:
    """Test dataset classes for PERCH."""

    @pytest.fixture
    def taxonomy_df(self):
        """Load taxonomy dataframe."""
        taxonomy_path = "data/birdclef-2026/taxonomy.csv"
        return pd.read_csv(taxonomy_path)

    @pytest.fixture
    def labels_df(self):
        """Load soundscape labels dataframe."""
        labels_path = "data/birdclef-2026/train_soundscapes_labels.csv"
        return pd.read_csv(labels_path)

    def test_bird_clef_audio_dataset_init(self, taxonomy_df, labels_df):
        """Test BirdClefAudioDataset initialization."""
        from dataset_perch import BirdClefAudioDataset
        
        dataset = BirdClefAudioDataset(
            audio_dir="data/birdclef-2026/train_soundscapes",
            labels_df=labels_df.head(10),
            taxonomy_df=taxonomy_df,
            sample_rate=32000,
            duration=5,
            use_cache=False,
        )
        
        assert dataset.sample_rate == 32000
        assert dataset.duration == 5
        assert dataset.expected_samples == 160000

    def test_bird_clef_audio_dataset_getitem(self, taxonomy_df, labels_df):
        """Test BirdClefAudioDataset __getitem__."""
        from dataset_perch import BirdClefAudioDataset
        
        dataset = BirdClefAudioDataset(
            audio_dir="data/birdclef-2026/train_soundscapes",
            labels_df=labels_df.head(5),
            taxonomy_df=taxonomy_df,
            sample_rate=32000,
            duration=5,
            use_cache=False,
        )
        
        waveform, labels = dataset[0]
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == (160000,)  # 5 sec × 32kHz
        assert waveform.dtype == torch.float32
        assert isinstance(labels, torch.Tensor)
        assert labels.dtype == torch.float32
        assert labels.sum() > 0, "Should have at least one positive label"

    def test_bird_clef_audio_clip_dataset_init(self, taxonomy_df):
        """Test BirdClefAudioClipDataset initialization."""
        from dataset_perch import BirdClefAudioClipDataset
        
        dataset = BirdClefAudioClipDataset(
            csv_path="data/birdclef-2026/train.csv",
            audio_dir="data/birdclef-2026/train_audio",
            taxonomy_df=taxonomy_df,
            sample_rate=32000,
            duration=5,
            use_cache=False,
        )
        
        assert dataset.sample_rate == 32000
        assert dataset.duration == 5
        assert dataset.expected_samples == 160000

    def test_bird_clef_test_audio_dataset_init(self):
        """Test BirdClefTestAudioDataset initialization."""
        from dataset_perch import BirdClefTestAudioDataset
        
        dataset = BirdClefTestAudioDataset(
            audio_dir="data/birdclef-2026/test_soundscapes",
            sample_rate=32000,
            duration=5,
        )
        
        assert dataset.sample_rate == 32000
        assert dataset.duration == 5
        assert dataset.expected_samples == 160000

    def test_bird_clef_test_audio_dataset_segments(self):
        """Test BirdClefTestAudioDataset segment generation."""
        from dataset_perch import BirdClefTestAudioDataset
        
        dataset = BirdClefTestAudioDataset(
            audio_dir="data/birdclef-2026/train_soundscapes",
            sample_rate=32000,
            duration=5,
        )
        
        if len(dataset.audio_files) == 0:
            pytest.skip("No audio files available")
        
        audio_path = dataset.audio_files[0]
        segments = dataset.get_segments(audio_path)
        
        assert len(segments) > 0
        for offset, row_id in segments:
            assert isinstance(offset, float)
            assert isinstance(row_id, str)


class TestCacheDirectories:
    """Test cache directory functionality."""

    def test_cache_directory_creation(self, tmp_path):
        """Test that cache directories are created properly."""
        from dataset_perch import BirdClefAudioDataset
        import pandas as pd
        
        taxonomy_df = pd.read_csv("data/birdclef-2026/taxonomy.csv")
        labels_df = pd.read_csv("data/birdclef-2026/train_soundscapes_labels.csv")
        
        cache_dir = tmp_path / "cache_audio"
        
        dataset = BirdClefAudioDataset(
            audio_dir="data/birdclef-2026/train_soundscapes",
            labels_df=labels_df.head(5),
            taxonomy_df=taxonomy_df,
            sample_rate=32000,
            duration=5,
            use_cache=True,
            cache_dir=str(cache_dir),
        )
        
        assert cache_dir.exists(), "Cache directory should be created"
        assert cache_dir.is_dir(), "Cache directory should be a directory"


class TestAudioFormatRequirements:
    """Test that audio format meets PERCH requirements."""

    def test_sample_rate_32khz(self):
        """Test that loaded audio has correct sample rate."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform = load_audio_for_perch(audio_path, sr=32000, duration=5.0)
        
        assert len(waveform) == 160000, "5 seconds at 32kHz should be 160000 samples"

    def test_duration_5_seconds(self):
        """Test that audio duration is exactly 5 seconds."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform = load_audio_for_perch(audio_path, sr=32000, duration=5.0)
        
        expected_samples = 5 * 32000  # 5 sec × 32kHz
        assert len(waveform) == expected_samples, f"Expected {expected_samples} samples, got {len(waveform)}"

    def test_float32_format(self):
        """Test that audio is returned as float32."""
        from audio import load_audio_for_perch
        
        audio_path = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
        
        if not Path(audio_path).exists():
            pytest.skip("Audio file not found")
        
        waveform = load_audio_for_perch(audio_path, sr=32000, duration=5.0)
        
        assert waveform.dtype == np.float32, f"Expected float32, got {waveform.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

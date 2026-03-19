"""Dataset classes for PERCH - raw audio waveform support."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path


class BirdClefAudioDataset(Dataset):
    """Dataset for BirdClef returning raw audio waveforms (for PERCH)."""

    def __init__(
        self,
        audio_dir: str,
        labels_df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        sample_rate: int = 32000,
        duration: int = 5,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "data/cache_audio",
        normalize: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.labels_df = labels_df.copy()
        self.sample_rate = sample_rate
        self.duration = duration
        self.expected_samples = sample_rate * duration
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.normalize = normalize

        self.label_cols = [c for c in taxonomy_df['primary_label'].values]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_cols)}

        self._prepare_labels()

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_labels(self):
        """Convert multi-label strings to one-hot encoding."""
        label_cols = self.label_cols

        label_data = {}
        for label in label_cols:
            label_data[label] = self.labels_df['primary_label'].apply(
                lambda x: 1 if label in str(x).split(';') else 0
            )

        label_df = pd.DataFrame(label_data)
        self.labels_df = pd.concat([self.labels_df, label_df], axis=1)

    def _load_audio(self, filename: str, start: str) -> np.ndarray:
        """Load a 5-second segment from audio file as waveform."""
        audio_path = self.audio_dir / filename

        if not audio_path.exists():
            return np.zeros(self.expected_samples, dtype=np.float32)

        try:
            start_sec = self._parse_time(start)
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=start_sec,
                duration=self.duration,
                mono=True
            )

            if len(y) < self.expected_samples:
                y = np.pad(y, (0, self.expected_samples - len(y)))

            return y.astype(np.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.expected_samples, dtype=np.float32)

    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '00:00:05' to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def _normalize_waveform(self, y: np.ndarray) -> np.ndarray:
        """Normalize waveform to [-1, 1] range."""
        if self.normalize:
            max_val = np.abs(y).max()
            if max_val > 0:
                y = y / max_val
        return y

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get waveform and labels for a given index."""
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        start = row['start']

        cache_key = f"{filename}_{start.replace(':', '_')}"
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if self.use_cache and cache_path.exists():
            waveform = np.load(cache_path)
        else:
            waveform = self._load_audio(filename, start)
            waveform = self._normalize_waveform(waveform)

            if self.use_cache:
                np.save(cache_path, waveform)

        waveform = torch.from_numpy(waveform)

        label_values = [row[label] for label in self.label_cols]
        labels = torch.tensor(label_values, dtype=torch.float32)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, labels

    def get_raw_audio_path(self, idx: int) -> Path:
        """Get the audio file path for a given index."""
        row = self.labels_df.iloc[idx]
        return self.audio_dir / row['filename']


class BirdClefAudioClipDataset(Dataset):
    """Dataset for short audio clips (from train.csv) returning raw waveforms."""

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        taxonomy_df: pd.DataFrame,
        sample_rate: int = 32000,
        duration: int = 5,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "data/cache_audio_clips",
        normalize: bool = True,
    ):
        self.csv_path = csv_path
        self.audio_dir = Path(audio_dir)
        self.taxonomy_df = taxonomy_df
        self.sample_rate = sample_rate
        self.duration = duration
        self.expected_samples = sample_rate * duration
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.normalize = normalize

        self.label_cols = [c for c in taxonomy_df['primary_label'].values]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_cols)}

        self._load_csv()

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_csv(self):
        """Load and prepare the training CSV."""
        df = pd.read_csv(self.csv_path)
        df = df[df['primary_label'].isin(self.label_cols)]
        self.df = df.reset_index(drop=True)
        self.labels_df = self.df  # Alias for compatibility

    def _get_audio_path(self, primary_label: str, filename: str) -> Path:
        """Find the audio file path given primary_label and filename."""
        audio_path = self.audio_dir / primary_label / filename
        if audio_path.exists():
            return audio_path

        for ext in ['.ogg', '.wav', '.m4a', '.mp3']:
            alt_path = self.audio_dir / primary_label / (
                filename.replace('.ogg', ext).replace('.wav', ext)
                .replace('.m4a', ext).replace('.mp3', ext)
            )
            if alt_path.exists():
                return alt_path

        return audio_path

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file as waveform."""
        if not audio_path.exists():
            return np.zeros(self.expected_samples, dtype=np.float32)

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            if len(y) < self.expected_samples:
                y = np.pad(y, (0, self.expected_samples - len(y)))
            elif len(y) > self.expected_samples:
                y = y[:self.expected_samples]

            return y.astype(np.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.expected_samples, dtype=np.float32)

    def _normalize_waveform(self, y: np.ndarray) -> np.ndarray:
        """Normalize waveform to [-1, 1] range."""
        if self.normalize:
            max_val = np.abs(y).max()
            if max_val > 0:
                y = y / max_val
        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get waveform and label for a given index."""
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = row['filename']

        audio_path = self._get_audio_path(primary_label, filename)

        safe_filename = filename.replace('/', '_').replace('.', '_')
        cache_key = f"{primary_label}_{safe_filename}"
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if self.use_cache and cache_path.exists():
            waveform = np.load(cache_path)
        else:
            waveform = self._load_audio(audio_path)
            waveform = self._normalize_waveform(waveform)

            if self.use_cache:
                np.save(cache_path, waveform)

        waveform = torch.from_numpy(waveform)

        label_idx = self.label_to_idx.get(primary_label, -1)
        labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        if label_idx >= 0:
            labels[label_idx] = 1.0

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, labels


class BirdClefTestAudioDataset(Dataset):
    """Dataset for test soundscape prediction with raw audio."""

    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 32000,
        duration: int = 5,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.expected_samples = sample_rate * duration

        self.audio_files = list(Path(audio_dir).glob("*.ogg"))
        if not self.audio_files:
            self.audio_files = list(Path(audio_dir).glob("*.mp3"))

    def _load_audio(self, audio_path: Path, offset: float) -> np.ndarray:
        """Load a segment from audio file."""
        try:
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=self.duration,
                mono=True
            )

            if len(y) < self.expected_samples:
                y = np.pad(y, (0, self.expected_samples - len(y)))

            return y.astype(np.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.expected_samples, dtype=np.float32)

    def _normalize_waveform(self, y: np.ndarray) -> np.ndarray:
        """Normalize waveform to [-1, 1] range."""
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        return y

    def get_segments(self, audio_path: Path) -> list[tuple[float, str]]:
        """Get all 5-second segments for an audio file."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            segments = []

            for start in range(0, int(duration), 5):
                row_id = f"{audio_path.stem}_{start:05d}"
                segments.append((float(start), row_id))

            return segments
        except:
            return []

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        """Get all segments from an audio file."""
        audio_path = self.audio_files[idx]
        segments = self.get_segments(audio_path)

        waveforms = []
        row_ids = []

        for offset, row_id in segments:
            waveform = self._load_audio(audio_path, offset)
            waveform = self._normalize_waveform(waveform)
            waveforms.append(waveform)
            row_ids.append(row_id)

        return waveforms, row_ids, audio_path.stem


def load_audio_segment(
    audio_path: str | Path,
    offset: float = 0.0,
    duration: float = 5.0,
    sample_rate: int = 32000,
    normalize: bool = True,
) -> np.ndarray:
    """Load a single audio segment as waveform.
    
    Args:
        audio_path: Path to audio file
        offset: Start offset in seconds
        duration: Duration in seconds
        sample_rate: Target sample rate
        normalize: Whether to normalize to [-1, 1]
    
    Returns:
        Waveform as numpy array
    """
    audio_path = Path(audio_path)
    expected_samples = int(sample_rate * duration)
    
    try:
        y, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            offset=offset,
            duration=duration,
            mono=True
        )
        
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)))
        
        y = y.astype(np.float32)
        
        if normalize:
            max_val = np.abs(y).max()
            if max_val > 0:
                y = y / max_val
                
        return y
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(expected_samples, dtype=np.float32)

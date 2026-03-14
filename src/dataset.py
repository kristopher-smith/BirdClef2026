"""Dataset classes for BirdClef 2026."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path


class BirdClefDataset(Dataset):
    """Dataset for BirdClef soundscape training data."""

    def __init__(
        self,
        audio_dir: str,
        labels_df: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        sample_rate: int = 32000,
        duration: int = 5,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "data/cache",
    ):
        self.audio_dir = Path(audio_dir)
        self.labels_df = labels_df
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

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
        """Load a 5-second segment from audio file."""
        audio_path = self.audio_dir / filename
        
        if not audio_path.exists():
            return np.zeros((self.sample_rate * self.duration,))

        try:
            start_sec = self._parse_time(start)
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=start_sec,
                duration=self.duration,
                mono=True
            )
            
            if len(y) < self.sample_rate * self.duration:
                y = np.pad(y, (0, self.sample_rate * self.duration - len(y)))
            
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros((self.sample_rate * self.duration,))

    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '00:00:05' to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def _compute_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        return mel_db.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get spectrogram and labels for a given index."""
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        start = row['start']

        cache_path = self.cache_dir / f"{filename}_{start.replace(':', '_')}.npy"
        
        if self.use_cache and cache_path.exists():
            spectrogram = np.load(cache_path)
        else:
            y = self._load_audio(filename, start)
            spectrogram = self._compute_spectrogram(y)
            
            if self.use_cache:
                np.save(cache_path, spectrogram)

        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
        
        label_values = [row[label] for label in self.label_cols]
        labels = torch.tensor(label_values, dtype=torch.float32)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, labels


class BirdClefTestDataset(Dataset):
    """Dataset for test soundscape prediction."""

    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 32000,
        duration: int = 5,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.audio_files = list(Path(audio_dir).glob("*.ogg"))

    def _load_audio(self, audio_path: Path, offset: float) -> np.ndarray:
        """Load a 5-second segment from audio file."""
        try:
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=self.duration,
                mono=True
            )
            
            if len(y) < self.sample_rate * self.duration:
                y = np.pad(y, (0, self.sample_rate * self.duration - len(y)))
            
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros((self.sample_rate * self.duration,))

    def _compute_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        return mel_db.astype(np.float32)

    def __len__(self) -> int:
        return len(self.audio_files)

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


class BirdClefShortClipDataset(Dataset):
    """Dataset for short audio clips from train.csv (single-label)."""

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        taxonomy_df: pd.DataFrame,
        sample_rate: int = 32000,
        duration: int = 5,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "data/cache_short",
    ):
        self.csv_path = csv_path
        self.audio_dir = Path(audio_dir)
        self.taxonomy_df = taxonomy_df
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

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

    def _get_audio_path(self, primary_label: str, filename: str) -> Path:
        """Find the audio file path given primary_label and filename."""
        audio_path = self.audio_dir / primary_label / filename
        if audio_path.exists():
            return audio_path
        
        for ext in ['.ogg', '.wav', '.m4a', '.mp3']:
            alt_path = self.audio_dir / primary_label / (filename.replace('.ogg', ext).replace('.wav', ext).replace('.m4a', ext).replace('.mp3', ext))
            if alt_path.exists():
                return alt_path
        
        return audio_path

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file with robust format handling."""
        if not audio_path.exists():
            return np.zeros((self.sample_rate * self.duration,))

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            if len(y) < self.sample_rate * self.duration:
                y = np.pad(y, (0, self.sample_rate * self.duration - len(y)))
            elif len(y) > self.sample_rate * self.duration:
                y = y[:self.sample_rate * self.duration]
            
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros((self.sample_rate * self.duration,))

    def _compute_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        return mel_db.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get spectrogram and label for a given index."""
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = row['filename']

        audio_path = self._get_audio_path(primary_label, filename)
        
        safe_filename = filename.replace('/', '_').replace('.', '_')
        cache_key = f"{primary_label}_{safe_filename}"
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        if self.use_cache and cache_path.exists():
            spectrogram = np.load(cache_path)
        else:
            y = self._load_audio(audio_path)
            spectrogram = self._compute_spectrogram(y)
            
            if self.use_cache:
                np.save(cache_path, spectrogram)

        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
        
        label_idx = self.label_to_idx.get(primary_label, -1)
        labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        if label_idx >= 0:
            labels[label_idx] = 1.0

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, labels

"""Audio processing utilities for BirdClef 2026."""

import numpy as np
import librosa


def load_audio(path: str, sr: int = 32000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 32000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute mel spectrogram from audio waveform."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_mfcc(
    y: np.ndarray,
    sr: int = 32000,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Compute MFCC features from audio waveform."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

"""Audio processing utilities for BirdClef 2026."""

import numpy as np
import librosa
from pathlib import Path


def load_audio(path: str, sr: int = 32000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def load_audio_for_perch(
    path: str | Path,
    sr: int = 32000,
    duration: float = 5.0,
    offset: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Load audio file optimized for PERCH input.
    
    PERCH expects:
    - 32kHz sample rate
    - Mono channel
    - Normalized to [-1, 1]
    - Fixed length (160000 samples for 5 seconds at 32kHz)
    
    Args:
        path: Path to audio file
        sr: Sample rate (default 32000 for PERCH)
        duration: Duration in seconds
        offset: Start offset in seconds
        normalize: Normalize to [-1, 1]
    
    Returns:
        Waveform as float32 numpy array
    """
    expected_samples = int(sr * duration)
    
    y, _ = librosa.load(
        path,
        sr=sr,
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


def load_audio_segments(
    path: str | Path,
    segment_duration: float = 5.0,
    sr: int = 32000,
    normalize: bool = True,
) -> list[tuple[np.ndarray, str]]:
    """Load all consecutive segments from an audio file.
    
    Args:
        path: Path to audio file
        segment_duration: Duration of each segment in seconds
        sr: Sample rate
        normalize: Normalize to [-1, 1]
    
    Returns:
        List of (waveform, row_id) tuples
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    
    if normalize:
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
    
    total_duration = len(y) / sr
    segment_samples = int(sr * segment_duration)
    
    segments = []
    for start in range(0, int(total_duration), int(segment_duration)):
        start_sample = int(start * sr)
        end_sample = int((start + segment_duration) * sr)
        
        y_seg = y[start_sample:end_sample]
        
        if len(y_seg) < segment_samples:
            y_seg = np.pad(y_seg, (0, segment_samples - len(y_seg)))
        
        row_id = f"{Path(path).stem}_{start:05d}"
        segments.append((y_seg.astype(np.float32), row_id))
    
    return segments


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

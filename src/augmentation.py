"""Data augmentation for spectrograms."""

import torch
import torch.nn as nn
import numpy as np


class SpecAugment:
    """SpecAugment: A Simple Data Augmentation Method for ASR."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.
        
        Args:
            x: Spectrogram tensor of shape (1, freq, time)
        """
        _, n_freq, n_time = x.shape

        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_freq - f))
            x[:, f0:f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            x[:, :, t0:t0 + t] = 0

        return x


class TimeShift:
    """Random time shift augmentation."""

    def __init__(self, max_shift: int = 50):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time shift to spectrogram."""
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        if shift > 0:
            return torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)
        else:
            return torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)


class TimeStretch:
    """Random time stretch (simplified version)."""

    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time stretch to spectrogram."""
        rate = np.random.uniform(self.min_rate, self.max_rate)
        if abs(rate - 1.0) < 0.05:
            return x

        _, n_freq, n_time = x.shape
        new_n_time = int(n_time * rate)
        
        x_stretched = torch.nn.functional.interpolate(
            x, size=(n_freq, new_n_time), mode='bilinear', align_corners=False
        )

        if new_n_time > n_time:
            start = (new_n_time - n_time) // 2
            return x_stretched[:, :, start:start + n_time]
        else:
            pad = n_time - new_n_time
            start = pad // 2
            return torch.nn.functional.pad(x_stretched, (start, pad - start))


class Mixup:
    """Mixup augmentation for multi-label classification."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup to batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

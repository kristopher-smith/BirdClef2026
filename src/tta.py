"""Test-Time Augmentation (TTA) for BirdClef 2026."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Optional


class TTAAugment:
    """Base class for TTA augmentations."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TTAOriginal(TTAAugment):
    """Original image (no augmentation)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TTAHorizontalFlip(TTAAugment):
    """Horizontal flip of spectrogram."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[-1])


class TTATimeShift(TTAAugment):
    """Time shift augmentation."""

    def __init__(self, max_shift: int = 10):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        if x.dim() == 4:
            if shift > 0:
                return torch.cat([x[:, :, :, shift:], x[:, :, :, :shift]], dim=3)
            else:
                return torch.cat([x[:, :, :, shift:], x[:, :, :, :shift]], dim=3)
        else:
            if shift > 0:
                return torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)
            else:
                return torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)


class TTAFreqMask(TTAAugment):
    """Frequency masking augmentation."""

    def __init__(self, freq_mask_param: int = 10, num_masks: int = 1):
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            _, _, n_freq, n_time = x.shape
        else:
            _, n_freq, n_time = x.shape
        
        for _ in range(self.num_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_freq - f))
            x = x.clone()
            if x.dim() == 4:
                x[:, :, f0:f0 + f, :] = 0
            else:
                x[:, f0:f0 + f, :] = 0
        
        return x


class TTATimeMask(TTAAugment):
    """Time masking augmentation."""

    def __init__(self, time_mask_param: int = 20, num_masks: int = 1):
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            _, _, n_freq, n_time = x.shape
        else:
            _, n_freq, n_time = x.shape
        
        for _ in range(self.num_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_time - t))
            x = x.clone()
            if x.dim() == 4:
                x[:, :, :, t0:t0 + t] = 0
            else:
                x[:, :, t0:t0 + t] = 0
        
        return x


class TTACompose:
    """Compose multiple TTA augmentations."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [t(x) for t in self.transforms]


def get_tta_transforms(augments: str = "original,flip") -> List[TTAAugment]:
    """
    Get TTA transforms based on string specification.
    
    Args:
        augments: Comma-separated list of augmentations:
            - original: No augmentation
            - flip: Horizontal flip
            - timeshift: Random time shift
            - freqmask: Frequency masking
            - timemask: Time masking
    
    Returns:
        List of TTA augmentation callables
    """
    augments = augments.lower().split(",")
    transforms = []
    
    for aug in augments:
        aug = aug.strip()
        if aug == "original":
            transforms.append(TTAOriginal())
        elif aug == "flip":
            transforms.append(TTAHorizontalFlip())
        elif aug == "timeshift":
            transforms.append(TTATimeShift(max_shift=10))
        elif aug == "freqmask":
            transforms.append(TTAFreqMask(freq_mask_param=10))
        elif aug == "timemask":
            transforms.append(TTATimeMask(time_mask_param=20))
        else:
            print(f"Warning: Unknown TTA augmentation '{aug}', skipping")
    
    return transforms


class PredictorWithTTA:
    """Wrapper for model with Test-Time Augmentation."""

    def __init__(
        self,
        model: nn.Module,
        augments: Optional[List[TTAAugment]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.augments = augments if augments else [TTAOriginal()]
        self.device = device
        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA.
        
        Args:
            x: Input tensor of shape (batch, 1, freq, time)
        
        Returns:
            Averaged predictions over all augmentations
        """
        all_predictions = []
        
        with torch.no_grad():
            for aug in self.augments:
                augmented = aug(x)
                augmented = augmented.to(self.device)
                outputs = self.model(augmented)
                probs = torch.sigmoid(outputs)
                all_predictions.append(probs)
        
        avg_predictions = torch.stack(all_predictions).mean(dim=0)
        return avg_predictions

    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_probs: bool = True,
    ) -> np.ndarray:
        """
        Predict on a dataloader with TTA.
        
        Args:
            dataloader: DataLoader with spectrograms
            return_probs: If True, return probabilities; else return logits
        
        Returns:
            Numpy array of predictions
        """
        all_probs = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            probs = self.predict(inputs)
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)


def apply_tta_to_predictions(
    model: nn.Module,
    inputs: torch.Tensor,
    augments: List[TTAAugment],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Apply TTA and average predictions.
    
    Args:
        model: PyTorch model
        inputs: Input tensor (batch, 1, freq, time)
        augments: List of augmentation callables
        device: Device to run on
    
    Returns:
        Averaged predictions
    """
    model.eval()
    all_preds = []
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        for aug in augments:
            augmented = aug(inputs)
            outputs = model(augmented)
            all_preds.append(torch.sigmoid(outputs))
    
    avg_preds = torch.stack(all_preds).mean(dim=0)
    return avg_preds

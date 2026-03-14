"""Model definitions for BirdClef 2026."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class BirdClefModel(nn.Module):
    """EfficientNet-based model for bird species classification."""

    def __init__(
        self,
        num_classes: int = 234,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BirdClefModelWithPool(nn.Module):
    """EfficientNet with additional pooling for longer spectrograms."""

    def __init__(
        self,
        num_classes: int = 234,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        pool_type: str = "avg",
    ):
        super().__init__()
        
        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.pool_type = pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_model(num_classes: int = 234, **kwargs) -> BirdClefModel:
    """Create a BirdClef model with default settings."""
    return BirdClefModel(num_classes=num_classes, **kwargs)


class ConvBlock(nn.Module):
    """Basic convolutional block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """Simple CNN baseline for quick testing."""

    def __init__(self, num_classes: int = 234, input_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

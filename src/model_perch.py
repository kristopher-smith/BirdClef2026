"""Model definitions using PERCH/YAMNet embeddings for BirdClef 2026."""

import torch
import torch.nn as nn
import numpy as np

PERCH_AVAILABLE = False
YAMNET_AVAILABLE = False

try:
    import tensorflow as tf
    import tf_keras as keras
    YAMNET_AVAILABLE = True
except ImportError:
    pass

try:
    import audioclass
    PERCH_AVAILABLE = True
except ImportError:
    pass


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class YAMNetEmbedding(nn.Module):
    """YAMNet audio embedding model."""

    def __init__(self, num_embeddings: int = 1024):
        super().__init__()
        if not YAMNET_AVAILABLE:
            raise ImportError("TensorFlow and tf-keras required for YAMNet. Install with: pip install tensorflow tf-keras")
        
        import tensorflow_hub as hub
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        self.num_embeddings = num_embeddings
        self._embedding_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract YAMNet embeddings from spectrograms.
        
        Note: YAMNet expects raw audio (waveform), not spectrograms.
        This is a placeholder that returns zeros - for proper YAMNet embeddings,
        raw audio needs to be passed to the model.
        
        Args:
            x: Spectrogram tensor of shape (batch, 1, freq, time)
        
        Returns:
            Embeddings of shape (batch, num_embeddings)
        """
        batch_size = x.size(0)
        emb = torch.zeros(batch_size, self._embedding_dim, device=x.device)
        return emb


class BirdClefYAMNetModel(nn.Module):
    """YAMNet embeddings + classification head for BirdClef."""

    def __init__(
        self,
        num_classes: int = 234,
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: int = 1024,
    ):
        super().__init__()
        
        if not YAMNET_AVAILABLE:
            raise ImportError("TensorFlow and tf-keras required for YAMNet")
        
        self.embedding = YAMNetEmbedding(num_embeddings=embedding_dim)
        actual_embedding_dim = self.embedding._embedding_dim  # 1024 for YAMNet
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(actual_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.classifier(embeddings)


class PERCHEmbedding(nn.Module):
    """PERCH audio embedding model using audioclass."""

    def __init__(self, embedding_dim: int = 1280):
        super().__init__()
        if not PERCH_AVAILABLE:
            raise ImportError("audioclass required for PERCH. Install with: pip install audioclass[perch,tensorflow]")
        
        try:
            from audioclass.models.perch import Perch
            self.perch = Perch.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PERCH model: {e}. Make sure you have internet access and Kaggle credentials if needed.")
        
        self.embedding_dim = embedding_dim
        self._embedding_dim = 1280  # PERCH embeddings are 1280-dim (output_1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract PERCH embeddings from raw audio waveforms.
        
        PERCH expects:
        - Raw audio waveform (not spectrogram)
        - 32kHz sample rate
        - 160000 samples (5 seconds)
        - float32, normalized to [-1, 1]
        
        Args:
            x: Waveform tensor of shape (batch, 160000)
        
        Returns:
            Embeddings of shape (batch, 1280)
        """
        batch_size = x.size(0)
        embeddings = []
        
        for i in range(batch_size):
            audio_np = x[i].cpu().numpy()
            
            result = self.perch.process_array(audio_np)
            emb = result.features
            
            embeddings.append(emb)
        
        return torch.tensor(np.array(embeddings), dtype=torch.float32, device=x.device)


class BirdClefPERCHModel(nn.Module):
    """PERCH embeddings + classification head for BirdClef."""

    def __init__(
        self,
        num_classes: int = 234,
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: int = 1280,
    ):
        super().__init__()
        
        if not PERCH_AVAILABLE:
            raise ImportError("audioclass required for PERCH")
        
        self.embedding = PERCHEmbedding(embedding_dim=embedding_dim)
        actual_embedding_dim = self.embedding._embedding_dim  # 1280 for PERCH
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(actual_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.classifier(embeddings)


class SimpleEmbeddingModel(nn.Module):
    """Simple CNN-based embedding model as fallback."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x


class BirdClefSimpleEmbeddingModel(nn.Module):
    """Simple CNN embeddings + classification head."""

    def __init__(
        self,
        num_classes: int = 234,
        dropout: float = 0.3,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        self.embedding = SimpleEmbeddingModel(embedding_dim=embedding_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.classifier(embeddings)


def create_embedding_model(
    model_type: str = "yamnet",
    num_classes: int = 234,
    pretrained: bool = True,
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """
    Create an embedding-based model.
    
    Args:
        model_type: Type of embedding model ('yamnet', 'perch', 'simple')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained embeddings
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_type == "yamnet":
        return BirdClefYAMNetModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif model_type == "perch":
        return BirdClefPERCHModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif model_type == "simple":
        return BirdClefSimpleEmbeddingModel(
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'yamnet', 'perch', or 'simple'")

# BirdClef 2026 - Training & Inference Pipeline

Bird species identification from audio recordings in the Pantanal, South America.

## Overview

| Component | Details |
|-----------|---------|
| **Model** | EfficientNet-B0 (pretrained on ImageNet) |
| **Training Data** | 1,478 soundscape segments + 35,549 short clips |
| **Classes** | 234 bird species |
| **Primary Metric** | mAP@10 |

## Quick Start

```bash
# Train basic model
python src/train.py --epochs 20 --batch_size 16 --use_augment

# Make predictions
python src/predict.py --model models/best_model.pt --output submission.csv
```

---

## Training Pipeline

### Training Scripts

| Script | Purpose |
|--------|---------|
| `src/train.py` | Train on soundscape data |
| `src/train_short.py` | Pre-train on short clips (35k samples) |
| `src/train_cv.py` | Cross-validation with held-out test |
| `src/train_perch.py` | Train with PERCH embeddings (raw audio) |

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 1e-4 |
| `--num_workers` | DataLoader workers | 4 |
| `--model` | Backbone model | `efficientnet_b0` |
| `--embedding_model` | Alternative embedding model | None |
| `--dropout` | Dropout rate | 0.3 |
| `--cache_dir` | Spectrogram cache directory | `data/cache` |
| `--use_cache` | Use cached spectrograms | True |
| `--test` | Quick test run (1 epoch) | False |

#### Augmentation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_augment` | Enable SpecAugment + TimeShift | False |
| `--mixup_alpha` | Mixup alpha (0=disabled) | 0.0 |
| `--label_smoothing` | Label smoothing factor | 0.0 |

#### Regularization & Training Enhancements

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_class_weights` | Use inverse frequency weights | False |
| `--warmup_epochs` | Learning rate warmup epochs | 0 |
| `--early_stopping_patience` | Early stopping patience (0=disabled) | 0 |

#### Pretrained & Transfer Learning

| Argument | Description | Default |
|----------|-------------|---------|
| `--pretrained` | Path to pretrained checkpoint | None |
| `--upload_to_kaggle` | Upload model to Kaggle after training | False |

#### MLflow Tracking

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow_experiment` | MLflow experiment name | `birdclef2026` |
| `--mlflow_run_name` | MLflow run name | None |
| `--mlflow_tracking_uri` | MLflow tracking URI | None |

### Example Training Commands

```bash
# Basic training with augmentations
python src/train.py --epochs 20 --batch_size 16 --use_augment

# Full augmentation + regularization
python src/train.py --epochs 20 --batch_size 16 --use_augment \
    --mixup_alpha 0.4 --label_smoothing 0.1 --use_class_weights

# Pre-train on short clips
python src/train_short.py --epochs 15 --batch_size 32 --use_augment \
    --mixup_alpha 0.4 --label_smoothing 0.1 --use_class_weights

# Fine-tune pretrained model on soundscapes
python src/train.py --epochs 5 --pretrained models/best_short_clip_model.pt \
    --use_augment

# Train with EfficientNet-B2, warmup, and early stopping
python src/train.py --model efficientnet_b2 --epochs 30 --batch_size 12 \
    --use_augment --warmup_epochs 3 --early_stopping_patience 5

# Cross-validation with held-out test set
python src/train_cv.py --folds 5 --held_out_ratio 0.1 --epochs 15 --use_augment
```

---

## Inference Pipeline

### Prediction Script

```bash
python src/validate_submission.py --submission submission.csv \
    --sample_submission data/birdclef-2026/sample_submission.csv \
    --taxonomy data/birdclef-2026/taxonomy.csv
```

---

## Changelog

### Phase 1: Audio Dataset (Complete)
- Added `src/dataset_perch.py` with raw waveform dataset classes
- Added `load_audio_for_perch()` and `load_audio_segments()` to `src/audio.py`
- Added PERCH integration section to README

### Prediction Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model checkpoint path | `models/best_model.pt` |
| `--output` | Output submission file | `submission.csv` |
| `--batch_size` | Batch size for inference | 16 |
| `--sample_rate` | Audio sample rate | 32000 |
| `--n_mels` | Number of mel bins | 128 |
| `--n_fft` | FFT window size | 2048 |
| `--hop_length` | Hop length | 512 |

### Test-Time Augmentation (TTA)

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_tta` | Enable TTA | False |
| `--tta_augments` | Comma-separated augments | `original,flip` |

**Available TTA Augments:**
- `original` - No augmentation
- `flip` - Horizontal flip
- `timeshift` - Random time shift (±10 frames)
- `freqmask` - Frequency masking
- `timemask` - Time masking

```bash
# TTA with flip only (recommended for speed/accuracy)
python src/predict.py --model models/best_model.pt --output submission.csv \
    --use_tta

# Multiple TTA augments
python src/predict.py --model models/best_model.pt --output submission.csv \
    --use_tta --tta_augments "original,flip,timeshift"

# All TTA augments (slowest but most accurate)
python src/predict.py --model models/best_model.pt --output submission.csv \
    --use_tta --tta_augments "original,flip,timeshift,freqmask,timemask"
```

### Model Ensemble

| Argument | Description | Default |
|----------|-------------|---------|
| `--ensemble` | Enable ensemble | False |
| `--ensemble_config` | Path to ensemble JSON config | None |
| `--ensemble_dir` | Directory containing models | `models` |
| `--ensemble_pattern` | Glob pattern for models | `*.pt` |
| `--ensemble_weights` | Comma-separated weights | None |
| `--ensemble_aggregation` | Aggregation method (`average`/`max`) | `average` |

**Ensemble Config Format:**
```json
{
    "models": [
        {"path": "models/b0.pt", "weight": 0.3, "backbone": "efficientnet_b0"},
        {"path": "models/b2.pt", "weight": 0.3, "backbone": "efficientnet_b2"},
        {"path": "models/short.pt", "weight": 0.2, "backbone": "efficientnet_b0"},
        {"path": "models/perch.pt", "weight": 0.2, "embedding_model": "perch"}
    ],
    "aggregation": "average"
}
```

```bash
# Use ensemble from config
python src/predict.py --ensemble --ensemble_config models/ensemble_config.json \
    --output submission.csv

# Auto-discover models in directory
python src/predict.py --ensemble --ensemble_dir models --ensemble_pattern "*.pt" \
    --output submission.csv
```

---

## Available Models

### EfficientNet Backbones

| Model | Description |
|-------|-------------|
| `efficientnet_b0` | Lightweight, fast training (default) |
| `efficientnet_b1` | Slightly larger |
| `efficientnet_b2` | Better accuracy |
| `efficientnet_b3` | Best accuracy, slower |

### Embedding Models

| Model | Description | Dependencies |
|-------|-------------|--------------|
| `yamnet` | YAMNet embeddings (1024-dim from AudioSet) | tensorflow, tf-keras, tensorflow-hub |
| `perch` | PERCH embeddings (recommended) | audioclass[perch,tensorflow] |
| `simple` | Lightweight CNN fallback | None |

```bash
# Train with simple embeddings
python src/train.py --embedding_model simple --epochs 20 --use_augment

# Train with PERCH embeddings
python src/train.py --embedding_model perch --epochs 20 --use_augment

# Train with YAMNet embeddings
python src/train.py --embedding_model yamnet --epochs 20 --use_augment
```

---

## Tracking System

The project uses **MLflow** for experiment tracking with a graceful fallback to local JSON logging.

### Features

- **Metrics Logging**: mAP@10, F1@10, loss, per-class metrics
- **Artifact Logging**: Models, spectrograms, confusion matrices, training curves
- **Parameter Tracking**: All hyperparameters logged automatically
- **Fallback Mode**: Works without MLflow (saves to local JSON)

### Usage

```python
from tracking import MetricsLogger

# Create logger
logger = MetricsLogger(
    experiment_name="birdclef2026",
    run_name="experiment_001",
    tracking_uri="http://localhost:5000"  # Optional MLflow server
)

# Start run
logger.start_run()

# Log parameters
logger.log_params({
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "model": "efficientnet_b0"
})

# Log metrics
logger.log_metrics({"loss": 0.5, "map_at_10": 0.7}, step=1)

# Log artifacts
logger.log_artifact("models/best_model.pt")
logger.log_confusion_matrix(cm, labels)
logger.log_training_curves()

# End run
logger.end_run()
```

### Decorator Usage

```python
from tracking import mlflow_track, attach_logger

@mlflow_track(['loss', 'map_at_10'], prefix='train_')
def train_one_epoch(model, dataloader, epoch):
    # ... training code ...
    return {'loss': epoch_loss, 'map_at_10': map_score}

# Attach logger
attach_logger(train_one_epoch, logger)
```

### Metrics Saved

| Metric | Description |
|--------|-------------|
| `mAP@10` | Primary competition metric |
| `F1@10` | Secondary metric |
| `Per-class recall` | Identify weak species |
| `BCE Loss` | Training stability |
| `Element-wise accuracy` | Quick sanity check |

---

## Data Summary

| Dataset | Samples | Species | Notes |
|---------|---------|---------|-------|
| `train.csv` (short clips) | 35,549 | 206 | Single species per file |
| `train_soundscapes` | 1,478 | 75 | Multi-label segments |
| Taxonomy | 234 | - | All classes in competition |

### Recommended Training Strategy

1. **Pre-train**: Short clips (35k samples) for 10-15 epochs
2. **Fine-tune**: Soundscapes (1.5k samples) for 5 epochs

This leverages more data for representation learning while adapting to multi-label format.

---

## PERCH Integration

### Audio Format Requirements

PERCH requires raw audio waveforms instead of spectrograms:

| Parameter | Value |
|-----------|-------|
| Sample Rate | 32 kHz |
| Duration | 5 seconds |
| Samples | 160,000 (5 sec × 32kHz) |
| Format | float32, normalized to [-1, 1] |

### Dataset Classes

| Class | Purpose | Input |
|-------|---------|-------|
| `BirdClefAudioDataset` | Training on soundscapes | Raw waveform |
| `BirdClefAudioClipDataset` | Training on short clips | Raw waveform |
| `BirdClefTestAudioDataset` | Test inference | Raw waveform |

### Audio Utilities

```python
from audio import load_audio_for_perch, load_audio_segments

# Load single 5-second segment
waveform = load_audio_for_perch("audio.ogg", sr=32000, duration=5.0, offset=10.0)

# Load all 5-second segments from file
segments = load_audio_segments("audio.ogg", segment_duration=5.0, sr=32000)
# Returns: [(waveform, row_id), ...]
```

### Cache Directories

| Input Type | Cache Directory |
|------------|----------------|
| Spectrograms | `data/cache/` |
| Raw audio (PERCH) | `data/cache_audio/` |

---

## Validation

```bash
# Validate submission format
python src/validate_submission.py --submission submission.csv \
    --sample_submission data/birdclef-2026/sample_submission.csv \
    --taxonomy data/birdclef-2026/taxonomy.csv
```

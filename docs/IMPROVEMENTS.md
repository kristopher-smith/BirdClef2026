# BirdClef 2026 Improvement Plan

## Current Baseline

| Component | Current State |
|-----------|---------------|
| **Model** | EfficientNet-B0 (pretrained on ImageNet) |
| **Training Data** | 1,478 soundscape segments (~1.5k samples) |
| **Validation** | Element-wise accuracy only |
| **Augmentations** | Defined in `augmentation.py` but NOT used |
| **Loss** | BCEWithLogitsLoss |
| **Epochs** | 10 |

---

## Data Summary

| Dataset | Samples | Species | Notes |
|---------|---------|---------|-------|
| `train.csv` (short clips) | 35,549 | 206 | Single species per file |
| `train_soundscapes` | 1,478 | 75 | Multi-label segments |
| Taxonomy | 234 | - | All classes in competition |

**Key insight**: Short clips provide ~30x more training data but only 206 species vs 234 in taxonomy.

---

## Implementation Plan

### Phase 1: Augmentations + Metrics (Week 1)

#### 1.1 Enable Data Augmentations
**Files**: `src/train.py`

**Changes**:
- Import augmentations from `src/augmentation.py`
- Apply SpecAugment (freq/time masking) during training
- Add Mixup for multi-label classification
- Apply TimeShift augmentation

**Code structure**:
```python
# In train.py, add to training loop
transform = Compose([
    SpecAugment(freq_mask_param=15, time_mask_param=35),
    TimeShift(max_shift=30),
])
```

#### 1.2 Add Proper Validation Metrics
**Files**: `src/train.py`

**Changes**:
- Add mean Average Precision (mAP@10) calculation
- Add per-species F1-score tracking
- Log metrics per epoch

**Priority metrics**:
- `mAP@10`: Primary competition metric
- `F1@10`: Secondary metric
- Per-class recall: Identify weak species

---

### Phase 2: Data Expansion (Week 2)

#### 2.1 Create Short Audio Dataset
**Files**: Create `src/dataset_short.py` or extend `src/dataset.py`

**Changes**:
- Create `BirdClefShortClipDataset` class
- Handle 35,549 short clips from `train.csv`
- Use filename format: `{primary_label}/iNat{id}.ogg`
- Map to files in `train_audio/{primary_label}/`
- Single-label classification

**Key considerations**:
- Handle multiple audio formats (.ogg, .wav, .m4a)
- Use full audio or random 5-second slice
- Build caching mechanism

#### 2.2 Mixed Training Strategy
**Files**: Create `src/train_pretrained.py`

**Approach**:
1. **Pre-train**: Short clips (35k samples) for 10-15 epochs
2. **Fine-tune**: Soundscapes (1.5k samples) for 5 epochs

This leverages more data for representation learning while adapting to multi-label format.

---

### Phase 3: Model Upgrades (Week 3)

#### 3.1 Upgrade Backbone
**Files**: `src/model.py`

**Changes**:
- Add EfficientNet-B2/B3 support
- Update model selection with `--model` flag in train.py

#### 3.2 Training Enhancements
**Files**: `src/train.py`

**Changes**:
- Add label smoothing (0.1) for multi-label
- Implement learning rate warmup (3-5 epochs)
- Add early stopping (patience=5)
- Train for 20-30 epochs with better scheduling

#### 3.3 Class Imbalance Handling
**Changes**:
- Apply class weights inversely proportional to frequency
- Track per-class F1 to identify weak classes

---

### Phase 4: Robust Testing (Week 4)

#### 4.1 Cross-Validation
- Implement 5-fold stratified cross-validation
- Use k-fold for reliable performance estimates

#### 4.2 Held-Out Test Set
- Reserve 10% of data as unseen test set
- Evaluate mAP, F1, precision, recall on held-out set

#### 4.3 Per-Species Analysis
- Track per-class performance
- Identify weak species and investigate

#### 4.4 Submission Verification
- Validate submission format matches `sample_submission.csv`
- Check probability ranges [0, 1]
- Verify all species present in predictions

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Audio format variations (.ogg, .wav, .m4a) | Robust loading with error handling |
| Class overlap (206 vs 234 species) | Pre-train + fine-tune workflow |
| Memory constraints | Efficient caching of spectrograms |
| Overfitting on small soundscape data | Augmentation + early stopping |

---

## Augmentation Priority

| Augmentation | Priority | Reason |
|-------------|----------|--------|
| SpecAugment | High | Proven for audio |
| Mixup | High | Multi-label regularization |
| TimeShift | Medium | Adds robustness |
| TimeStretch | Low | May distort bird calls |

---

## Metrics to Track

| Metric | Purpose |
|--------|---------|
| `mAP@10` | Primary competition metric |
| `F1@10` | Secondary metric |
| Per-class recall | Identify weak species |
| BCE Loss | Training stability |
| Element-wise accuracy | Quick sanity check |

---

## Command Reference

### Phase 1: Train Soundscapes with Augmentations
```bash
# Basic training with augmentations
python src/train.py --epochs 20 --batch_size 16 --use_augment

# Full augmentation + regularization
python src/train.py --epochs 20 --batch_size 16 --use_augment --mixup_alpha 0.4 --label_smoothing 0.1 --use_class_weights
```

### Phase 2: Pre-train on Short Clips (35k samples)
```bash
# Pre-train on short clips with full augmentation
python src/train_short.py --epochs 15 --batch_size 32 --use_augment --mixup_alpha 0.4 --label_smoothing 0.1 --use_class_weights
```

### Phase 2: Fine-tune on Soundscapes
```bash
# Fine-tune pretrained model on soundscapes
python src/train.py --epochs 5 --pretrained models/best_short_clip_model.pt --use_augment
```

### Phase 3: Larger Backbone + Training Enhancements
```bash
# Train with EfficientNet-B2, warmup, and early stopping
python src/train.py --model efficientnet_b2 --epochs 30 --batch_size 12 --use_augment --warmup_epochs 3 --early_stopping_patience 5
```

### Available Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size | 16/32 |
| `--lr` | Learning rate | 1e-4 |
| `--model` | Backbone (efficientnet_b0/b1/b2/b3) | efficientnet_b0 |
| `--dropout` | Dropout rate | 0.3 |
| `--use_augment` | Enable SpecAugment + TimeShift | False |
| `--mixup_alpha` | Mixup alpha (0=disabled) | 0.0 |
| `--label_smoothing` | Label smoothing factor | 0.0 |
| `--use_class_weights` | Use inverse frequency weights | False |
| `--pretrained` | Path to pretrained checkpoint | None |
| `--warmup_epochs` | Learning rate warmup epochs | 0 |
| `--early_stopping_patience` | Early stopping patience (0=disabled) | 0 |
| `--cache_dir` | Spectrogram cache directory | data/cache |
| `--num_workers` | DataLoader workers | 4 |

### Phase 4: Cross-Validation & Validation
```bash
# 5-fold cross-validation with held-out test set
python src/train_cv.py --folds 5 --held_out_ratio 0.1 --epochs 15 --use_augment

# Validate submission file
python src/validate_submission.py --submission submission.csv
```

### CV Script Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--folds` | Number of CV folds | 5 |
| `--held_out_ratio` | Held-out test set ratio | 0.1 |
| `--epochs` | Epochs per fold | 10 |
| `--use_augment` | Enable augmentations | False |

### Validate Submission Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--submission` | Path to submission CSV | (required) |
| `--sample_submission` | Path to sample submission | data/birdclef-2026/sample_submission.csv |
| `--taxonomy` | Path to taxonomy CSV | data/birdclef-2026/taxonomy.csv |

---

## Implementation Order

1. `src/train.py`: ✅ Add augmentation support + mAP metrics
2. `src/dataset.py`: ✅ Add ShortClipDataset class  
3. `src/train_short.py`: ✅ Support training on short clips
4. `src/train.py`: ✅ Add label smoothing, class weights, pretrained loading
5. `src/model.py`: ✅ Support EfficientNet-B2 (Phase 3)
6. `src/train.py`: ✅ Add warmup + early stopping (Phase 3)
7. `src/train_cv.py`: ✅ 5-fold cross-validation + held-out test (Phase 4)
8. `src/validate_submission.py`: ✅ Submission validation (Phase 4)
9. `src/model.py`: ✅ Add EfficientNet-B3 + fix LSP errors (Phase 5)
10. `src/model_perch.py`: ✅ Add PERCH/YAMNet embeddings (Phase 6)
11. `src/train.py`: ✅ Add --embedding_model support (Phase 6)
12. `src/tta.py`: ✅ Test-Time Augmentation (Phase 7)
13. `src/predict.py`: ✅ Add TTA support (Phase 7)
14. `src/ensemble.py`: ✅ Model Ensemble (Phase 8)
15. `src/predict.py`: ✅ Add ensemble support (Phase 8)

---

## Phase 6: PERCH/YAMNet Embeddings

### Overview
Add support for pretrained audio embedding models (YAMNet, PERCH) alongside EfficientNet.

### Changes

#### 6.1 Create PERCH/YAMNet Model
**Files**: `src/model_perch.py` (new)

- `YAMNetEmbedding`: YAMNet audio embedding extraction
- `BirdClefYAMNetModel`: YAMNet embeddings + classification head
- `PERCHEmbedding`: PERCH audio embedding (via audioclass)
- `BirdClefPERCHModel`: PERCH embeddings + classification head
- `BirdClefSimpleEmbeddingModel`: Simple CNN fallback (no TensorFlow needed)
- `create_embedding_model()`: Factory function

#### 6.2 Update Training Scripts
**Files**: `src/train.py`

- Add `--embedding_model` argument (yamnet, perch, simple)
- Falls back to efficientnet_b0 if dependencies not available

### Requirements
```bash
# For PERCH (recommended):
pip install audioclass[perch,tensorflow]

# For YAMNet only:
pip install tensorflow tf-keras tensorflow-hub
```

### Usage
```bash
# Train with simple CNN embeddings (no extra deps)
python src/train.py --embedding_model simple --epochs 20 --use_augment

# Train with PERCH embeddings (recommended)
python src/train.py --embedding_model perch --epochs 20 --use_augment

# Train with YAMNet embeddings
python src/train.py --embedding_model yamnet --epochs 20 --use_augment

# Original EfficientNet training still works
python src/train.py --model efficientnet_b2 --epochs 20
```

### Testing
```bash
# Test simple embedding model
python -c "from src.model_perch import BirdClefSimpleEmbeddingModel; import torch; m = BirdClefSimpleEmbeddingModel(); print(m(torch.randn(2,1,128,313)).shape)"

# Test YAMNet (requires TensorFlow)
python -c "from src.model_perch import BirdClefYAMNetModel; import torch; m = BirdClefYAMNetModel(); print(m(torch.randn(2,1,128,313)).shape)"
```

### Notes
- YAMNet provides 1024-dim embeddings from pretrained AudioSet
- Simple embedding model is a lightweight CNN fallback
- YAMNet is slower (requires TensorFlow) but more accurate
- Both models handle multi-label classification

---

## Phase 7: Test-Time Augmentation (TTA)

### Overview
Add Test-Time Augmentation to improve prediction quality by averaging predictions over multiple augmented versions of each input.

### Changes

#### 7.1 Create TTA Module
**Files**: `src/tta.py` (new)

- `TTAOriginal`: No augmentation
- `TTAHorizontalFlip`: Flip spectrogram horizontally
- `TTATimeShift`: Random time shift
- `TTAFreqMask`: Frequency masking
- `TTATimeMask`: Time masking
- `TTACompose`: Compose multiple augmentations
- `PredictorWithTTA`: Wrapper class for TTA inference
- `get_tta_transforms()`: Factory function
- `apply_tta_to_predictions()`: Apply TTA to predictions

#### 7.2 Update Prediction Script
**Files**: `src/predict.py`

- Add `--use_tta` flag
- Add `--tta_augments` argument

### Usage
```bash
# Predict without TTA (baseline)
python src/predict.py --model models/best_model.pt --output submission.csv

# Predict with TTA (original + flip)
python src/predict.py --model models/best_model.pt --output submission.csv --use_tta

# Predict with TTA (multiple augments)
python src/predict.py --model models/best_model.pt --output submission.csv --use_tta --tta_augments "original,flip,timeshift"

# Predict with TTA (all augments)
python src/predict.py --model models/best_model.pt --output submission.csv --use_tta --tta_augments "original,flip,timeshift,freqmask,timemask"
```

### Available TTA Augments
| Augment | Description |
|---------|-------------|
| `original` | No augmentation |
| `flip` | Horizontal flip |
| `timeshift` | Random time shift (±10 frames) |
| `freqmask` | Frequency masking |
| `timemask` | Time masking |

### Testing
```python
# Test TTA
python -c "
import torch
from src.tta import get_tta_transforms, apply_tta_to_predictions
from src.model import BirdClefModel

transforms = get_tta_transforms('original,flip')
model = BirdClefModel(backbone='efficientnet_b0', pretrained=False)
model.eval()

x = torch.randn(2, 1, 128, 313)
probs = apply_tta_to_predictions(model, x, transforms)
print(f'Output shape: {probs.shape}')
"

# Compare inference time
# Without TTA: ~1x (baseline)
# With TTA (2 augments): ~2x
# With TTA (5 augments): ~5x
```

### Notes
- TTA typically improves mAP by 1-3%
- More augments = better accuracy but slower inference
- Recommended: `original,flip` for best speed/accuracy tradeoff

---

## Phase 8: Model Ensemble

### Overview
Combine multiple models (EfficientNet-B0, B2, short-clip pretrained, PERCH, etc.) to improve predictions through ensemble averaging.

### Changes

#### 8.1 Create Ensemble Module
**Files**: `src/ensemble.py` (new)

- `EnsembleModel`: Core ensemble class with weighted averaging
- `EnsemblePredictor`: Predictor wrapper for ensemble inference
- `create_ensemble_from_dir()`: Auto-discover models in directory
- `create_ensemble_from_config()`: Load ensemble from JSON config

#### 8.2 Update Prediction Script
**Files**: `src/predict.py`

- Add `--ensemble` flag
- Add `--ensemble_config` for JSON config
- Add `--ensemble_dir` for model directory
- Add `--ensemble_weights` for custom weights
- Add `--ensemble_aggregation` (average/max)

#### 8.3 Create Sample Config
**Files**: `models/ensemble_config.json` (new)

### Usage

**Option 1: Use JSON config**
```bash
python src/predict.py --ensemble --ensemble_config models/ensemble_config.json --output submission.csv
```

**Option 2: Auto-discover models in directory**
```bash
python src/predict.py --ensemble --ensemble_dir models --ensemble_pattern "*.pt" --output submission.csv
```

**Option 3: Single model (baseline)**
```bash
python src/predict.py --model models/best_model.pt --output submission.csv
```

### Ensemble Config Format
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

### Testing
```bash
# Test ensemble loading
python -c "
from src.ensemble import create_ensemble_from_config
ensemble = create_ensemble_from_config('models/ensemble_config.json', num_classes=234)
print(f'Loaded {len(ensemble.models)} models')
"
```

### Expected Improvements
- 2-5% mAP improvement over best single model
- Combining diverse models (different backbones, pretraining) gives best results
- Equal weights work well; custom weights can help if one model is stronger

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

### Phase 3: Larger Backbone
```bash
# Train with EfficientNet-B2
python src/train.py --model efficientnet_b2 --epochs 30 --batch_size 12 --use_augment
```

### Available Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch_size` | Batch size | 16/32 |
| `--lr` | Learning rate | 1e-4 |
| `--model` | Backbone (efficientnet_b0/b1/b2) | efficientnet_b0 |
| `--dropout` | Dropout rate | 0.3 |
| `--use_augment` | Enable SpecAugment + TimeShift | False |
| `--mixup_alpha` | Mixup alpha (0=disabled) | 0.0 |
| `--label_smoothing` | Label smoothing factor | 0.0 |
| `--use_class_weights` | Use inverse frequency weights | False |
| `--pretrained` | Path to pretrained checkpoint | None |
| `--cache_dir` | Spectrogram cache directory | data/cache |
| `--num_workers` | DataLoader workers | 4 |

---

## Implementation Order

1. `src/train.py`: ✅ Add augmentation support + mAP metrics
2. `src/dataset.py`: ✅ Add ShortClipDataset class  
3. `src/train_short.py`: ✅ Support training on short clips
4. `src/train.py`: ✅ Add label smoothing, class weights, pretrained loading
5. `src/model.py`: ✅ Support EfficientNet-B2 (Phase 3)
6. `tests/`: Add evaluation tests (mAP, F1, submission validation)

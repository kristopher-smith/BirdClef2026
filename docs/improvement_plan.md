# BirdClef 2026 Model Improvement Plan

## Project Context
- **Environment**: Apple Silicon (MPS acceleration)
- **Training Data**: 1,478 samples, 234 classes
- **Current Baseline**: EfficientNet-B0, BCEWithLogitsLoss, no augmentation
- **Target**: 0.50+ mAP
- **Training**: 20-30 epochs per phase
- **Deadline**: June 3, 2026

---

## Phase 1: Foundation & Baseline Establishment
**Goal**: Establish baseline metrics with augmentation enabled

### Tasks

| # | Task | Expected Impact |
|---|------|----------------|
| 1.1 | Add mAP@3 metric to training loop | Required for evaluation |
| 1.2 | Stratified validation split (70/15/15) | Better validation |
| 1.3 | Enable SpecAugment | +2-4% mAP |
| 1.4 | Enable Mixup during training | +2-5% mAP |
| 1.5 | Add label smoothing / class weights | Better calibration |

### Implementation Details

#### 1.1 - Metrics (`src/metrics.py`)
Create new file with mAP calculation:
- `map_at_k(preds, targets, k=3)`: Mean Average Precision at 3
- `compute_metrics(preds, targets)`: Returns mAP, F1, AUC

#### 1.2 - Stratified Split
Update `train.py`:
```python
# Use multi-label stratification
from sklearn.model_selection import MultilabelStratifiedKFold

# Or simple approach: stratify by primary label count bins
train_test_split(..., stratify=pd.cut(label_sums, bins=5))
```

#### 1.3 - SpecAugment
Add to `BirdClefDataset` transform:
```python
transform=Compose([
    SpecAugment(freq_mask_param=15, time_mask_param=35),
])
```

#### 1.4 - Mixup
Add to `train_one_epoch()`:
```python
if random.random() > 0.5:
    inputs, labels = mixup(inputs, labels)
```

### Testing Protocol
- **Split**: 70% train, 15% val, 15% test (stratified)
- **Seeds**: Run with seeds 42, 123, 456
- **Duration**: ~30 min per seed on MPS
- **Success Criteria**: mAP@3 > 0.25

---

## Phase 2: Model Architecture
**Goal**: Improve model capacity

### Tasks

| # | Task | Expected Impact |
|---|------|----------------|
| 2.1 | Upgrade to EfficientNet-B2 | +3-5% mAP |
| 2.2 | Class-weighted loss (pos_weight) | +2-3% mAP |
| 2.3 | Adjust learning rate (5e-5 for B2) | Faster convergence |
| 2.4 | Increase dropout (0.4-0.5) | Prevent overfitting |

### Implementation Details

#### 2.2 - Class-Weighted Loss
```python
# Compute pos_weight from training data
label_counts = labels_df[label_cols].sum()
pos_weight = (len(labels_df) - label_counts) / (label_counts + 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight.values))
```

#### 2.3 - Learning Rate Schedule
```python
# Warmup + Cosine for B2
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-5,
    epochs=epochs, steps_per_epoch=len(train_loader)
)
```

### Testing Protocol
- **Compare**: B0+aug vs B2+aug (same split, same seeds)
- **Duration**: ~45 min per seed
- **Success Criteria**: mAP@3 > 0.35

---

## Phase 3: Advanced Techniques
**Goal**: Push towards 0.50+ mAP

### Tasks

| # | Task | Expected Impact |
|---|------|----------------|
| 3.1 | Test-Time Augmentation (TTA) | +2-4% mAP |
| 3.2 | Longer audio context (10s) | +3-5% mAP |
| 3.3 | Multi-scale input (128+256 n_mels) | +2-3% mAP |
| 3.4 | Ensemble B0 + B2 | +3-5% mAP |

### Implementation Details

#### 3.1 - TTA
Add to `predict.py`:
```python
# Original
preds = model(x)

# TTA with horizontal flip
preds_flip = model(torch.flip(x, dims=[3]))
final_preds = (preds + preds_flip) / 2
```

#### 3.2 - 10-second Audio
Modify dataset config:
```python
BirdClefDataset(..., duration=10, hop_length=1024)
```
Note: Will require recomputing spectrogram cache (~30 min)

#### 3.4 - Ensemble
```python
# Load both models
preds_b0 = model_b0(x)
preds_b2 = model_b2(x)
final = 0.4 * preds_b0 + 0.6 * preds_b2  # Weight by validation mAP
```

### Testing Protocol
- **Compare**: Each technique A/B tested individually
- **Duration**: Variable
- **Success Criteria**: Final mAP@3 > 0.50

---

## Experiment Tracking Structure

```
experiments/
├── phase1/
│   ├── phase1_seed42/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   └── best_model.pt
│   ├── phase1_seed123/
│   └── phase1_seed456/
├── phase2/
│   ├── phase2_b2_seed42/
│   └── ...
└── phase3/
    ├── phase3_tta/
    ├── phase3_10s/
    └── phase3_ensemble/
```

### config.yaml Template
```yaml
model:
  backbone: efficientnet_b2
  dropout: 0.4
  pretrained: true

training:
  epochs: 25
  batch_size: 16
  lr: 5.0e-5
  weight_decay: 0.01
  label_smoothing: 0.1

augmentation:
  spec_augment: true
  mixup: true
  mixup_alpha: 0.4

data:
  duration: 5
  n_mels: 128
  sample_rate: 32000

results:
  val_map: 0.38
  test_map: 0.37
  seed: 42
```

---

## Quick Reference Commands

```bash
# Phase 1 - Quick test
python src/train.py --test --epochs 1

# Phase 1 - Full run
python src/train.py --epochs 25 --batch_size 16 --lr 1e-4

# Phase 2 - EfficientNet-B2
python src/train.py --model efficientnet_b2 --epochs 25 --lr 5e-5 --dropout 0.4

# Generate predictions
python src/predict.py --checkpoint models/best_model.pt --output submission.csv
```

---

## Progress Checklist

- [ ] Phase 1.1: Add mAP@3 metric
- [ ] Phase 1.2: Stratified validation split
- [ ] Phase 1.3: Enable SpecAugment
- [ ] Phase 1.4: Enable Mixup
- [ ] Phase 1.5: Add class weights
- [ ] Phase 2.1: Upgrade to EfficientNet-B2
- [ ] Phase 2.2: Class-weighted loss
- [ ] Phase 2.3: Adjust learning rate
- [ ] Phase 2.4: Increase dropout
- [ ] Phase 3.1: Test-Time Augmentation
- [ ] Phase 3.2: Longer audio (10s)
- [ ] Phase 3.3: Multi-scale input
- [ ] Phase 3.4: Ensemble models

---

## Notes

- Apple Silicon uses MPS backend: `torch.device("mps")`
- Cache directory: `data/cache/`
- Model checkpoints: `models/`
- All augmentations defined in `src/augmentation.py` but not yet used in training

"""Training script for BirdClef 2026."""
## To run:  python src/train.py --num_workers 4

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import BirdClefDataset
from model import BirdClefModel, get_device
from augmentation import SpecAugment, TimeShift, Mixup, Compose


def parse_args():
    parser = argparse.ArgumentParser(description="Train BirdClef 2026 model")
    parser.add_argument("--data_dir", type=str, default="data/birdclef-2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", help="Quick test run with 1 epoch")
    parser.add_argument("--use_augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha (0 to disable)")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    return parser.parse_args()


def get_model(backbone: str, num_classes: int, dropout: float):
    """Create model based on backbone name."""
    return BirdClefModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=True,
        dropout=dropout,
    )


def compute_ap(recalls, precisions):
    """Compute average precision given recall and precision curves."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_map_at_k(probs, labels, k=10):
    """Compute mean Average Precision at k."""
    n_samples, n_classes = labels.shape
    aps = []
    
    for c in range(n_classes):
        if labels[:, c].sum() == 0:
            continue
        
        sorted_indices = np.argsort(-probs[:, c])
        top_k = sorted_indices[:k]
        
        tp = labels[top_k, c].sum()
        fp = k - tp
        
        if tp + fp == 0:
            continue
        
        precisions = labels[top_k, c].cumsum() / (np.arange(k) + 1)
        recalls = labels[top_k, c].cumsum() / labels[:, c].sum()
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def compute_f1_at_k(probs, labels, k=10):
    """Compute F1 score at k (top-k predictions per sample)."""
    n_samples, n_classes = labels.shape
    f1_scores = []
    
    for i in range(n_samples):
        top_k_indices = np.argsort(-probs[i])[:k]
        preds = np.zeros(n_classes)
        preds[top_k_indices] = 1
        
        tp = (preds * labels[i]).sum()
        fp = ((preds == 1) & (labels[i] == 0)).sum()
        fn = ((preds == 0) & (labels[i] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def get_augmentation_transform():
    """Create augmentation pipeline."""
    return Compose([
        SpecAugment(freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2),
        TimeShift(max_shift=30),
    ])


def compute_class_weights(labels_df, label_cols, device):
    """Compute inverse frequency class weights for imbalanced data."""
    class_counts = labels_df[label_cols].sum().values
    class_counts = np.maximum(class_counts, 1)
    weights = len(labels_df) / (len(label_cols) * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, augment_transform=None, mixup_alpha=0.0, label_smoothing=0.0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    mixup = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if augment_transform is not None:
            for i in range(inputs.size(0)):
                inputs[i] = augment_transform(inputs[i])

        if mixup is not None and inputs.size(0) > 1:
            inputs, labels = mixup(inputs, labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        if label_smoothing > 0:
            labels_smooth = labels * (1 - label_smoothing) + label_smoothing / labels.size(1)
            loss = criterion(outputs, labels_smooth)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_corrects += (preds == labels).sum().item()
        running_total += labels.numel()

        pbar.set_postfix({
            'loss': loss.item(),
            'acc': running_corrects / running_total
        })

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / running_total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model with mAP and F1 metrics."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            running_corrects += (preds == labels).sum().item()
            running_total += labels.numel()

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / running_total

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    map_at_10 = compute_map_at_k(all_probs, all_labels, k=10)
    f1_at_10 = compute_f1_at_k(all_probs, all_labels, k=10)

    return epoch_loss, epoch_acc, map_at_10, f1_at_10


def main():
    args = parse_args()

    print(f"Training with:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Epochs: {args.epochs if not args.test else 1}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model: {args.model}")
    print(f"  Augmentations: {args.use_augment}")
    print(f"  Mixup alpha: {args.mixup_alpha}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Class weights: {args.use_class_weights}")

    device = get_device()
    print(f"  Device: {device}")

    data_dir = Path(args.data_dir)
    train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    train_audio = data_dir / "train_soundscapes"

    print(f"\nTraining samples: {len(train_labels)}")
    print(f"Number of classes: {len(taxonomy)}")

    dataset = BirdClefDataset(
        audio_dir=str(train_audio),
        labels_df=train_labels,
        taxonomy_df=taxonomy,
        sample_rate=32000,
        duration=5,
        n_mels=128,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
    )

    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = get_model(args.model, len(taxonomy), args.dropout)
    model = model.to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(dataset.labels_df, dataset.label_cols, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        print(f"  Using class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs if not args.test else 1
    )

    augment_transform = get_augmentation_transform() if args.use_augment else None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_map = 0.0

    epochs = args.epochs if not args.test else 1

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*50)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            augment_transform=augment_transform,
            mixup_alpha=args.mixup_alpha,
            label_smoothing=args.label_smoothing
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, map_at_10, f1_at_10 = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val mAP@10: {map_at_10:.4f}, Val F1@10: {f1_at_10:.4f}")

        scheduler.step()

        if map_at_10 > best_map:
            best_map = map_at_10
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'map_at_10': map_at_10,
                'f1_at_10': f1_at_10,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mAP@10: {best_map:.4f}")


if __name__ == "__main__":
    main()

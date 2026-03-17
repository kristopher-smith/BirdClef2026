"""Training script for BirdClef 2026 with PERCH embeddings."""
## To run:
## python src/train_perch.py --num_workers 4 --use_augment --mixup_alpha 0.4 --label_smoothing 0.1 --use_class_weights --epochs 30 --batch_size 4 --use_augment --warmup_epochs 3 --early_stopping_patience 5
## NOTE: PERCH is slow - batch_size is small by default (4) due to embedding extraction time

import argparse
import os
import sys
import time
import json
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset_perch import BirdClefAudioDataset, BirdClefAudioClipDataset
from model_perch import BirdClefPERCHModel, create_embedding_model, PERCH_AVAILABLE
from augmentation import SpecAugment, TimeShift, Mixup, Compose, WaveformTimeShift, WaveformNoise
from tracking import MetricsLogger, mlflow_track, attach_logger
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="Train BirdClef 2026 with PERCH embeddings")
    parser.add_argument("--data_dir", type=str, default="data/birdclef-2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4, help="Small batch size due to PERCH slow processing")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2, help="Fewer workers to avoid memory issues")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default="data/cache_audio")
    parser.add_argument("--checkpoint_dir", type=str, default="models/perch")
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", help="Quick test run with 1 epoch")
    parser.add_argument("--use_augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha (0 to disable)")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint for fine-tuning")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Early stopping patience (0 to disable)")
    parser.add_argument("--upload_to_kaggle", action="store_true", help="Upload model to Kaggle after training")
    parser.add_argument("--kaggle_dataset_slug", type=str, default="birdclef2026-perch", help="Kaggle dataset slug")
    parser.add_argument("--mlflow_experiment", type=str, default="birdclef2026_perch", help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--use_short_clips", action="store_true", help="Train on short clips instead of soundscapes")
    parser.add_argument("--embedding_dim", type=int, default=1280, help="PERCH embedding dimension")
    return parser.parse_args()


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
    """Create augmentation pipeline for audio waveforms."""
    return Compose([
        WaveformTimeShift(max_shift=10000),  # ~0.3 seconds at 32kHz
        WaveformNoise(noise_level=0.005),
    ])


def compute_class_weights(labels_df, label_cols, device):
    """Compute inverse frequency class weights for imbalanced data."""
    class_counts = labels_df[label_cols].sum().values
    class_counts = np.maximum(class_counts, 1)
    weights = len(labels_df) / (len(label_cols) * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    return weights


@mlflow_track(['loss', 'acc'], prefix='train_')
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

    return {'loss': epoch_loss, 'acc': epoch_acc}


@mlflow_track(['loss', 'acc', 'map_at_10', 'f1_at_10'], prefix='val_')
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

    return {'loss': epoch_loss, 'acc': epoch_acc, 'map_at_10': map_at_10, 'f1_at_10': f1_at_10, 'all_probs': all_probs, 'all_labels': all_labels}


def main():
    args = parse_args()

    if not PERCH_AVAILABLE:
        print("ERROR: PERCH (audioclass) is not available.")
        print("Install with: pip install audioclass[perch,tensorflow]")
        sys.exit(1)

    logger = MetricsLogger(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        tracking_uri=args.mlflow_tracking_uri
    )
    logger.start_run()

    logger.log_params({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'embedding_dim': args.embedding_dim,
        'dropout': args.dropout,
        'use_augment': args.use_augment,
        'mixup_alpha': args.mixup_alpha,
        'label_smoothing': args.label_smoothing,
        'use_class_weights': args.use_class_weights,
        'warmup_epochs': args.warmup_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'use_short_clips': args.use_short_clips,
    })

    attach_logger(train_one_epoch, logger)
    attach_logger(validate, logger)

    print(f"Training with PERCH embeddings:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Epochs: {args.epochs if not args.test else 1}")
    print(f"  Batch size: {args.batch_size} (small due to PERCH processing time)")
    print(f"  Learning rate: {args.lr}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Augmentations: {args.use_augment}")
    print(f"  Data source: {'short clips' if args.use_short_clips else 'soundscapes'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    data_dir = Path(args.data_dir)
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")

    if args.use_short_clips:
        train_csv = data_dir / "train.csv"
        train_audio = data_dir / "train_audio"
        dataset = BirdClefAudioClipDataset(
            csv_path=str(train_csv),
            audio_dir=str(train_audio),
            taxonomy_df=taxonomy,
            sample_rate=32000,
            duration=5,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
        )
    else:
        train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
        train_audio = data_dir / "train_soundscapes"
        dataset = BirdClefAudioDataset(
            audio_dir=str(train_audio),
            labels_df=train_labels,
            taxonomy_df=taxonomy,
            sample_rate=32000,
            duration=5,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
        )

    print(f"Training samples: {len(dataset)}")
    print(f"Number of classes: {len(taxonomy)}")

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

    model = BirdClefPERCHModel(
        num_classes=len(taxonomy),
        pretrained=True,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
    )
    model = model.to(device)

    if args.use_class_weights:
        class_weights = compute_class_weights(dataset.labels_df, dataset.label_cols, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        print(f"  Using class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    warmup_epochs = args.warmup_epochs if not args.test else 0
    train_epochs = args.epochs if not args.test else 1
    
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_epochs
        )

    augment_transform = get_augmentation_transform() if args.use_augment else None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_map = 0.0
    checkpoint_path = checkpoint_dir / "best_model.pt"
    epochs = train_epochs

    early_stopping_counter = 0
    early_stop = False

    print("\nWARNING: PERCH embedding extraction is slow.")
    print("Consider using fewer epochs or pre-computing embeddings for production.")

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*50)

        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            augment_transform=augment_transform,
            mixup_alpha=args.mixup_alpha,
            label_smoothing=args.label_smoothing
        )
        train_loss = train_result['loss']
        train_acc = train_result['acc']
        current_lr = optimizer.param_groups[0]['lr']
        train_result['lr'] = current_lr
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_result = validate(model, val_loader, criterion, device)
        val_loss = val_result['loss']
        val_acc = val_result['acc']
        map_at_10 = val_result['map_at_10']
        f1_at_10 = val_result['f1_at_10']
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val mAP@10: {map_at_10:.4f}, Val F1@10: {f1_at_10:.4f}")

        scheduler.step()

        if map_at_10 > best_map:
            best_map = map_at_10
            best_val_loss = val_loss
            early_stopping_counter = 0
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
            
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'map_at_10': map_at_10,
                'f1_at_10': f1_at_10,
            }
            logger.log_model_checkpoint(checkpoint_path, best_metrics)
        else:
            if args.early_stopping_patience > 0:
                early_stopping_counter += 1
                print(f"No improvement. Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    early_stop = True

        if early_stop:
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mAP@10: {best_map:.4f}")

    logger.end_run()


if __name__ == "__main__":
    main()

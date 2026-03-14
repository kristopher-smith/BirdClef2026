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
    return parser.parse_args()


def get_model(backbone: str, num_classes: int, dropout: float):
    """Create model based on backbone name."""
    return BirdClefModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=True,
        dropout=dropout,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
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
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += (preds == labels).sum().item()
            running_total += labels.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / running_total

    return epoch_loss, epoch_acc


def main():
    args = parse_args()

    print(f"Training with:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Epochs: {args.epochs if not args.test else 1}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model: {args.model}")

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs if not args.test else 1
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    epochs = args.epochs if not args.test else 1

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*50)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

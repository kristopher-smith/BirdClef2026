"""Cross-validation training script for BirdClef 2026."""
## To run: python src/train_cv.py --folds 5

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from dataset import BirdClefDataset
from model import BirdClefModel, get_device
from augmentation import SpecAugment, TimeShift, Mixup, Compose


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validation training for BirdClef 2026")
    parser.add_argument("--data_dir", type=str, default="data/birdclef-2026")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cache_dir", type=str, default="data/cache_cv")
    parser.add_argument("--checkpoint_dir", type=str, default="models/cv")
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", help="Quick test run with 1 epoch")
    parser.add_argument("--use_augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha (0 to disable)")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--held_out_ratio", type=float, default=0.1, help="Held-out test set ratio")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Early stopping patience")
    return parser.parse_args()


def get_model(backbone: str, num_classes: int, dropout: float):
    return BirdClefModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=True,
        dropout=dropout,
    )


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_map_at_k(probs, labels, k=10):
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


def compute_per_class_metrics(probs, labels, label_cols, k=10):
    """Compute per-class precision, recall, and F1."""
    n_classes = len(label_cols)
    metrics = {col: {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0} for col in label_cols}
    
    for c, col in enumerate(label_cols):
        support = labels[:, c].sum()
        if support == 0:
            continue
        
        top_k_indices = np.argsort(-probs[:, c])[:k]
        tp = labels[top_k_indices, c].sum()
        fp = k - tp
        fn = support - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / support if support > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[col] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(support)}
    
    return metrics


def compute_macro_metrics(probs, labels, k=10):
    """Compute macro-averaged precision, recall."""
    n_samples, n_classes = labels.shape
    
    precisions = []
    recalls = []
    
    for c in range(n_classes):
        if labels[:, c].sum() == 0:
            continue
        
        top_k_indices = np.argsort(-probs[:, c])[:k]
        tp = labels[top_k_indices, c].sum()
        fp = k - tp
        fn = labels[:, c].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / labels[:, c].sum() if labels[:, c].sum() > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return {
        'macro_precision': np.mean(precisions) if precisions else 0,
        'macro_recall': np.mean(recalls) if recalls else 0,
    }


def get_augmentation_transform():
    return Compose([
        SpecAugment(freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2),
        TimeShift(max_shift=30),
    ])


def compute_class_weights(labels_df, label_cols, device):
    class_counts = labels_df[label_cols].sum().values
    class_counts = np.maximum(class_counts, 1)
    weights = len(labels_df) / (len(label_cols) * class_counts)
    return torch.FloatTensor(weights).to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device, augment_transform=None, mixup_alpha=0.0, label_smoothing=0.0):
    model.train()
    running_loss = 0.0
    
    mixup = Mixup(alpha=mixup_alpha) if mixup_alpha > 0 else None
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
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
    
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device, label_cols):
    model.eval()
    running_loss = 0.0
    
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
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    map_at_10 = compute_map_at_k(all_probs, all_labels, k=10)
    f1_at_10 = compute_f1_at_k(all_probs, all_labels, k=10)
    per_class = compute_per_class_metrics(all_probs, all_labels, label_cols, k=10)
    macro = compute_macro_metrics(all_probs, all_labels, k=10)
    
    return {
        'loss': running_loss / len(dataloader.dataset),
        'map_at_10': map_at_10,
        'f1_at_10': f1_at_10,
        'per_class': per_class,
        'macro_precision': macro['macro_precision'],
        'macro_recall': macro['macro_recall'],
        'all_probs': all_probs,
        'all_labels': all_labels,
    }


def main():
    args = parse_args()
    
    print(f"Cross-validation training:")
    print(f"  Folds: {args.folds}")
    print(f"  Held-out ratio: {args.held_out_ratio}")
    print(f"  Epochs: {args.epochs if not args.test else 1}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model: {args.model}")
    print(f"  Augmentations: {args.use_augment}")
    
    device = get_device()
    print(f"  Device: {device}")
    
    data_dir = Path(args.data_dir)
    train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    train_audio = data_dir / "train_soundscapes"
    
    label_cols = [c for c in taxonomy['primary_label'].values]
    print(f"\nTotal samples: {len(train_labels)}")
    print(f"Number of classes: {len(label_cols)}")
    
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
    labels_array = train_labels[label_cols].values
    
    stratify_labels = labels_array.argmax(axis=1)
    
    held_out_size = int(len(indices) * args.held_out_ratio)
    train_val_indices, held_out_indices = train_test_split(
        indices, test_size=held_out_size, random_state=42, stratify=stratify_labels
    )
    
    held_out_dataset = Subset(dataset, held_out_indices)
    held_out_loader = DataLoader(
        held_out_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f"Training/validation samples: {len(train_val_indices)}")
    print(f"Held-out test samples: {len(held_out_indices)}")
    
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_held_out_probs = None
    all_held_out_labels = None
    
    augment_transform = get_augmentation_transform() if args.use_augment else None
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_indices, np.array(stratify_labels)[train_val_indices])):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.folds}")
        print('='*60)
        
        train_dataset = Subset(dataset, [train_val_indices[i] for i in train_idx])
        val_dataset = Subset(dataset, [train_val_indices[i] for i in val_idx])
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )
        
        model = get_model(args.model, len(taxonomy), args.dropout)
        model = model.to(device)
        
        if args.use_class_weights:
            train_labels_fold = train_labels.iloc[[train_val_indices[i] for i in train_idx]]
            class_weights = compute_class_weights(train_labels_fold, label_cols, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        train_epochs = args.epochs if not args.test else 1
        
        if args.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_epochs - args.warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
        
        best_map = 0.0
        best_model_state = None
        early_stopping_counter = 0
        
        for epoch in range(1, train_epochs + 1):
            print(f"\nEpoch {epoch}/{train_epochs}")
            
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                augment_transform=augment_transform,
                mixup_alpha=args.mixup_alpha,
                label_smoothing=args.label_smoothing
            )
            
            val_results = validate(model, val_loader, criterion, device, label_cols)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}, mAP@10: {val_results['map_at_10']:.4f}, F1@10: {val_results['f1_at_10']:.4f}")
            
            scheduler.step()
            
            if val_results['map_at_10'] > best_map:
                best_map = val_results['map_at_10']
                best_model_state = model.state_dict().copy()
                early_stopping_counter = 0
            else:
                if args.early_stopping_patience > 0:
                    early_stopping_counter += 1
                    if early_stopping_counter >= args.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        model.load_state_dict(best_model_state)
        
        fold_checkpoint = checkpoint_dir / f"fold_{fold + 1}_model.pt"
        torch.save({'model_state_dict': model.state_dict()}, fold_checkpoint)
        
        val_results = validate(model, val_loader, criterion, device, label_cols)
        
        held_out_results = validate(model, held_out_loader, criterion, device, label_cols)
        
        if all_held_out_probs is None:
            all_held_out_probs = held_out_results['all_probs']
            all_held_out_labels = held_out_results['all_labels']
        else:
            all_held_out_probs = np.concatenate([all_held_out_probs, held_out_results['all_probs']])
            all_held_out_labels = np.concatenate([all_held_out_labels, held_out_results['all_labels']])
        
        fold_results.append({
            'fold': fold + 1,
            'val_map': val_results['map_at_10'],
            'val_f1': val_results['f1_at_10'],
            'val_loss': val_results['loss'],
            'held_out_map': held_out_results['map_at_10'],
            'held_out_f1': held_out_results['f1_at_10'],
            'held_out_macro_precision': held_out_results['macro_precision'],
            'held_out_macro_recall': held_out_results['macro_recall'],
        })
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Val mAP@10: {val_results['map_at_10']:.4f}, F1@10: {val_results['f1_at_10']:.4f}")
        print(f"  Held-out mAP@10: {held_out_results['map_at_10']:.4f}, F1@10: {held_out_results['f1_at_10']:.4f}")
    
    avg_results = {
        'val_map': np.mean([r['val_map'] for r in fold_results]),
        'val_f1': np.mean([r['val_f1'] for r in fold_results]),
        'held_out_map': np.mean([r['held_out_map'] for r in fold_results]),
        'held_out_f1': np.mean([r['held_out_f1'] for r in fold_results]),
        'held_out_macro_precision': np.mean([r['held_out_macro_precision'] for r in fold_results]),
        'held_out_macro_recall': np.mean([r['held_out_macro_recall'] for r in fold_results]),
    }
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print('='*60)
    print(f"Average Val mAP@10: {avg_results['val_map']:.4f} ± {np.std([r['val_map'] for r in fold_results]):.4f}")
    print(f"Average Val F1@10: {avg_results['val_f1']:.4f} ± {np.std([r['val_f1'] for r in fold_results]):.4f}")
    print(f"Average Held-out mAP@10: {avg_results['held_out_map']:.4f}")
    print(f"Average Held-out F1@10: {avg_results['held_out_f1']:.4f}")
    print(f"Average Held-out Macro Precision: {avg_results['held_out_macro_precision']:.4f}")
    print(f"Average Held-out Macro Recall: {avg_results['held_out_macro_recall']:.4f}")
    
    results = {
        'fold_results': fold_results,
        'average_results': avg_results,
    }
    
    results_path = checkpoint_dir / "cv_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    if all_held_out_probs is not None:
        per_class_metrics = compute_per_class_metrics(all_held_out_probs, all_held_out_labels, label_cols, k=10)
        
        sorted_by_f1 = sorted(per_class_metrics.items(), key=lambda x: x[1]['f1'])
        
        print(f"\n{'='*60}")
        print("Per-Species Analysis (Bottom 10 by F1)")
        print('='*60)
        for species, metrics in sorted_by_f1[:10]:
            print(f"  {species}: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, Support={metrics['support']}")
        
        per_class_path = checkpoint_dir / "per_class_metrics.json"
        with open(per_class_path, 'w') as f:
            json.dump(per_class_metrics, f, indent=2)
        print(f"\nPer-class metrics saved to {per_class_path}")


if __name__ == "__main__":
    main()

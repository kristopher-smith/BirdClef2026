"""Prediction script for BirdClef 2026."""

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
from tqdm import tqdm

from model import BirdClefModel, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with BirdClef 2026 model")
    parser.add_argument("--data_dir", type=str, default="data/birdclef-2026")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    return parser.parse_args()


def compute_spectrogram(y, sample_rate, n_mels, n_fft, hop_length):
    """Compute mel spectrogram from audio."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def load_and_process_audio(audio_path, sample_rate, n_mels, n_fft, hop_length, duration=5):
    """Load audio and create spectrograms for each 5-second segment."""
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return [], []

    total_samples = len(y)
    segment_samples = sample_rate * duration
    
    spectrograms = []
    row_ids = []
    
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment = y[start:end]
        
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        spec = compute_spectrogram(segment, sample_rate, n_mels, n_fft, hop_length)
        spectrograms.append(spec)
        
        filename = audio_path.stem
        start_sec = start // sample_rate
        row_id = f"{filename}_{start_sec:05d}"
        row_ids.append(row_id)
    
    return spectrograms, row_ids


def predict(model, dataloader, device):
    """Run inference on a batch of spectrograms."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def main():
    args = parse_args()
    
    print(f"Predicting with:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    
    device = get_device()
    print(f"  Device: {device}")
    
    data_dir = Path(args.data_dir)
    
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    
    label_cols = [col for col in sample_sub.columns if col != 'row_id']
    print(f"Number of classes: {len(label_cols)}")
    
    model = BirdClefModel(num_classes=len(label_cols), pretrained=False)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    test_dir = data_dir / "test_soundscapes"
    
    if not test_dir.exists() or len(list(test_dir.glob("*.ogg"))) == 0:
        print(f"\nNo test audio found in {test_dir}")
        print("The test data is hidden on Kaggle. This script will work when test data is available.")
        print("\nCreating sample submission with uniform probabilities...")
        
        row_ids = sample_sub['row_id'].tolist()
        n_rows = len(row_ids)
        
        uniform_prob = 1.0 / len(label_cols)
        probs_df = pd.DataFrame(
            uniform_prob * np.ones((n_rows, len(label_cols))),
            columns=label_cols
        )
        submission = pd.concat([pd.DataFrame({'row_id': row_ids}), probs_df], axis=1)
        
    else:
        print(f"Found test files in {test_dir}")
        
        all_predictions = []
        all_row_ids = []
        
        test_files = list(test_dir.glob("*.ogg"))
        print(f"Processing {len(test_files)} test files...")
        
        for audio_path in tqdm(test_files, desc="Processing audio"):
            spectrograms, row_ids = load_and_process_audio(
                audio_path,
                args.sample_rate,
                args.n_mels,
                args.n_fft,
                args.hop_length
            )
            
            if not spectrograms:
                continue
            
            specs_tensor = torch.tensor(np.array(spectrograms)).unsqueeze(1)
            
            batch_size = args.batch_size
            predictions = []
            
            for i in range(0, len(specs_tensor), batch_size):
                batch = specs_tensor[i:i+batch_size].to(device)
                with torch.no_grad():
                    outputs = model(batch)
                    probs = torch.sigmoid(outputs)
                predictions.append(probs.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            
            all_predictions.append(predictions)
            all_row_ids.extend(row_ids)
        
        predictions_array = np.concatenate(all_predictions, axis=0)
        
        submission = pd.DataFrame({'row_id': all_row_ids})
        
        for idx, col in enumerate(label_cols):
            if idx < predictions_array.shape[1]:
                submission[col] = predictions_array[:, idx]
            else:
                submission[col] = 1.0 / len(label_cols)
        
        for col in label_cols:
            if col not in submission.columns:
                submission[col] = 1.0 / len(label_cols)
        
        existing_cols = [c for c in submission.columns if c != 'row_id']
        submission = submission[['row_id'] + existing_cols]
    
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()

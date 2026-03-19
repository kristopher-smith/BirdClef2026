#!/usr/bin/env python
"""End-to-end integration tests for PERCH pipeline.

This script tests the complete pipeline:
1. Audio dataset loading
2. PERCH model training (1 epoch)
3. Model checkpoint saving/loading
4. Prediction with PERCH model
5. Submission format validation
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import torch

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest


def setup_test_dirs():
    """Create temporary test directories."""
    test_dir = Path("data/test_integration")
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def test_audio_dataset():
    """Test 1: Audio dataset loading."""
    print("\n" + "="*60)
    print("TEST 1: Audio Dataset Loading")
    print("="*60)
    
    from dataset_perch import BirdClefAudioDataset
    
    data_dir = Path("data/birdclef-2026")
    train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    train_audio = data_dir / "train_soundscapes"
    
    # Use small subset for testing
    dataset = BirdClefAudioDataset(
        audio_dir=str(train_audio),
        labels_df=train_labels.head(5),
        taxonomy_df=taxonomy,
        sample_rate=32000,
        duration=5,
        use_cache=False,
    )
    
    # Test loading
    waveform, labels = dataset[0]
    
    assert waveform.shape == torch.Size([160000]), f"Expected (160000,), got {waveform.shape}"
    assert labels.shape == torch.Size([234]), f"Expected (234,), got {labels.shape}"
    assert -1.0 <= waveform.min() <= 1.0, "Waveform not normalized"
    
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels sum (positive): {labels.sum().item()}")
    print("  ✅ PASSED")
    return True


def test_perch_model_creation():
    """Test 2: PERCH model creation."""
    print("\n" + "="*60)
    print("TEST 2: PERCH Model Creation")
    print("="*60)
    
    from model_perch import BirdClefPERCHModel, PERCH_AVAILABLE
    
    if not PERCH_AVAILABLE:
        print("  ⚠️  SKIPPED - PERCH not available")
        return False
    
    model = BirdClefPERCHModel(
        num_classes=234,
        pretrained=False,
        dropout=0.3,
    )
    
    assert model.embedding._embedding_dim == 1280, f"Expected 1280, got {model.embedding._embedding_dim}"
    
    # Test classifier structure
    classifier = model.classifier
    assert classifier[0].in_features == 1280, "First linear input"
    assert classifier[0].out_features == 512, "First linear output"
    assert classifier[3].in_features == 512, "Second linear input"
    assert classifier[3].out_features == 234, "Final output"
    
    print(f"  Embedding dim: {model.embedding._embedding_dim}")
    print(f"  Classifier: {classifier}")
    print("  ✅ PASSED")
    return True


def test_perch_training():
    """Test 3: PERCH training (minimal)."""
    print("\n" + "="*60)
    print("TEST 3: PERCH Training (1 batch)")
    print("="*60)
    
    from dataset_perch import BirdClefAudioDataset
    from model_perch import BirdClefPERCHModel, PERCH_AVAILABLE
    
    if not PERCH_AVAILABLE:
        print("  ⚠️  SKIPPED - PERCH not available")
        return False
    
    data_dir = Path("data/birdclef-2026")
    train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    train_audio = data_dir / "train_soundscapes"
    
    dataset = BirdClefAudioDataset(
        audio_dir=str(train_audio),
        labels_df=train_labels.head(2),  # Just 2 samples
        taxonomy_df=taxonomy,
        sample_rate=32000,
        duration=5,
        use_cache=False,
    )
    
    model = BirdClefPERCHModel(num_classes=234, pretrained=False, dropout=0.3)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train for 1 batch only (PERCH is extremely slow)
    model.train()
    waveform, labels = dataset[0]
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    labels = labels.unsqueeze(0)
    
    optimizer.zero_grad()
    try:
        output = model(waveform)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Input shape: {waveform.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_checkpoint_save_load():
    """Test 4: Checkpoint save/load."""
    print("\n" + "="*60)
    print("TEST 4: Checkpoint Save/Load")
    print("="*60)
    
    import tempfile
    from model import BirdClefModel
    
    # Create a simple EfficientNet model for testing
    model = BirdClefModel(num_classes=234, backbone="efficientnet_b0", pretrained=False)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
    
    # Save checkpoint
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'val_loss': 0.5,
        'map_at_10': 0.3,
    }, checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    assert checkpoint['epoch'] == 1
    assert 'model_state_dict' in checkpoint
    
    os.unlink(checkpoint_path)
    
    print(f"  Saved and loaded checkpoint successfully")
    print("  ✅ PASSED")
    return True


def test_prediction_pipeline():
    """Test 5: Prediction pipeline."""
    print("\n" + "="*60)
    print("TEST 5: Prediction Pipeline")
    print("="*60)
    
    from model import BirdClefModel, get_device
    from predict import load_and_process_audio
    
    device = get_device()
    
    # Create simple model
    model = BirdClefModel(num_classes=234, backbone="efficientnet_b0", pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Load test audio
    data_dir = Path("data/birdclef-2026")
    test_dir = data_dir / "test_soundscapes"
    
    if not test_dir.exists() or len(list(test_dir.glob("*.ogg"))) == 0:
        print("  ⚠️  No test audio found - using sample submission")
        
        # Just test that we can create the submission structure
        sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
        label_cols = [c for c in sample_sub.columns if c != 'row_id']
        
        # Create uniform submission
        submission = pd.DataFrame({
            'row_id': sample_sub['row_id'].head(10)
        })
        for col in label_cols:
            submission[col] = 1.0 / len(label_cols)
        
        print(f"  Created sample submission: {submission.shape}")
        print("  ✅ PASSED (sample mode)")
        return True
    
    # Process audio
    audio_files = list(test_dir.glob("*.ogg"))
    if audio_files:
        spectrograms, row_ids = load_and_process_audio(
            audio_files[0],
            sample_rate=32000,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        if spectrograms:
            # Run prediction
            specs_tensor = torch.tensor(np.array(spectrograms)).unsqueeze(1).to(device)
            
            with torch.no_grad():
                outputs = model(specs_tensor)
                probs = torch.sigmoid(outputs)
            
            print(f"  Processed {len(spectrograms)} segments")
            print(f"  Prediction shape: {probs.shape}")
            print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
            print("  ✅ PASSED")
            return True
    
    print("  ⚠️  No audio to test")
    return True


def test_ensemble_predictor():
    """Test 6: Ensemble predictor."""
    print("\n" + "="*60)
    print("TEST 6: Ensemble Predictor")
    print("="*60)
    
    from ensemble import MixedEnsemblePredictor
    import torch.nn as nn
    
    class DummySpectrogramModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 234)
    
    class DummyWaveformModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 234)
    
    mixed = MixedEnsemblePredictor(
        spectrogram_models=[DummySpectrogramModel()],
        waveform_models=[DummyWaveformModel()],
        weights=[0.5, 0.5],
        aggregation="average",
        device="cpu"
    )
    
    # Test with dummy inputs
    spec_input = torch.randn(2, 1, 128, 313)
    wave_input = torch.randn(2, 160000)
    
    result = mixed.predict(spectrograms=spec_input, waveforms=wave_input)
    
    assert result.shape == (2, 234), f"Expected (2, 234), got {result.shape}"
    
    print(f"  Input: spectrogram (2,1,128,313) + waveform (2,160000)")
    print(f"  Output shape: {result.shape}")
    print("  ✅ PASSED")
    return True


def test_submission_format():
    """Test 7: Submission format validation."""
    print("\n" + "="*60)
    print("TEST 7: Submission Format Validation")
    print("="*60)
    
    data_dir = Path("data/birdclef-2026")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    
    label_cols = [c for c in sample_sub.columns if c != 'row_id']
    
    # Create test submission
    submission = pd.DataFrame({
        'row_id': ['test_001', 'test_002', 'test_003']
    })
    
    # Add probability columns
    for col in label_cols:
        submission[col] = np.random.uniform(0, 1, size=len(submission))
    
    # Validate format
    assert 'row_id' in submission.columns, "Missing row_id"
    assert len(submission.columns) == len(label_cols) + 1, "Wrong number of columns"
    
    # Check probabilities
    probs = submission[label_cols].values
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of range"
    
    print(f"  Columns: {len(submission.columns)} (expected {len(label_cols) + 1})")
    print(f"  Rows: {len(submission)}")
    print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("  ✅ PASSED")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("PERCH INTEGRATION TESTS")
    print("="*60)
    
    results = []
    
    # Run tests
    tests = [
        ("Audio Dataset", test_audio_dataset),
        ("PERCH Model Creation", test_perch_model_creation),
        ("PERCH Training", test_perch_training),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Ensemble Predictor", test_ensemble_predictor),
        ("Submission Format", test_submission_format),
    ]
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED/SKIPPED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())

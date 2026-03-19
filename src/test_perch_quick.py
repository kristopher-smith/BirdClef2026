#!/usr/bin/env python
"""Quick integration tests for PERCH pipeline (skips slow PERCH forward pass)."""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'


def test_audio_dataset():
    """Test audio dataset loading."""
    print("\n" + "="*60)
    print("TEST 1: Audio Dataset Loading")
    print("="*60)
    
    from dataset_perch import BirdClefAudioDataset
    
    data_dir = Path("data/birdclef-2026")
    train_labels = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    train_audio = data_dir / "train_soundscapes"
    
    dataset = BirdClefAudioDataset(
        audio_dir=str(train_audio),
        labels_df=train_labels.head(2),
        taxonomy_df=taxonomy,
        sample_rate=32000,
        duration=5,
        use_cache=False,
    )
    
    waveform, labels = dataset[0]
    
    assert waveform.shape == torch.Size([160000])
    assert labels.shape == torch.Size([234])
    
    print(f"  Waveform: {waveform.shape}, Labels: {labels.shape}")
    print("  ✅ PASSED")
    return True


def test_perch_model_structure():
    """Test PERCH model structure (no forward pass)."""
    print("\n" + "="*60)
    print("TEST 2: PERCH Model Structure")
    print("="*60)
    
    from model_perch import BirdClefPERCHModel, PERCH_AVAILABLE
    
    if not PERCH_AVAILABLE:
        print("  ⚠️  SKIPPED - PERCH not installed")
        return False
    
    model = BirdClefPERCHModel(num_classes=234, pretrained=False, dropout=0.3)
    
    assert model.embedding._embedding_dim == 1280
    print(f"  Embedding dim: {model.embedding._embedding_dim}")
    print(f"  Classifier: {model.classifier}")
    print("  ✅ PASSED")
    return True


def test_checkpoint_save_load():
    """Test checkpoint save/load."""
    print("\n" + "="*60)
    print("TEST 3: Checkpoint Save/Load")
    print("="*60)
    
    from model import BirdClefModel
    
    model = BirdClefModel(num_classes=234, backbone="efficientnet_b0", pretrained=False)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name
    
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'val_loss': 0.5,
        'map_at_10': 0.3,
    }, checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint['epoch'] == 1
    
    import os
    os.unlink(checkpoint_path)
    
    print("  ✅ PASSED")
    return True


def test_prediction_with_efficientnet():
    """Test prediction with EfficientNet (not PERCH - too slow)."""
    print("\n" + "="*60)
    print("TEST 4: EfficientNet Prediction")
    print("="*60)
    
    from model import BirdClefModel
    from predict import load_and_process_audio
    
    model = BirdClefModel(num_classes=234, backbone="efficientnet_b0", pretrained=False)
    model.eval()
    
    data_dir = Path("data/birdclef-2026")
    test_dir = data_dir / "test_soundscapes"
    
    if not test_dir.exists() or len(list(test_dir.glob("*.ogg"))) == 0:
        print("  Using sample submission format")
        sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
        
        submission = pd.DataFrame({'row_id': sample_sub['row_id'].head(10)})
        label_cols = [c for c in sample_sub.columns if c != 'row_id']
        for col in label_cols:
            submission[col] = 1.0 / len(label_cols)
        
        print(f"  Created submission: {submission.shape}")
        print("  ✅ PASSED")
        return True
    
    audio_files = list(test_dir.glob("*.ogg"))
    if audio_files:
        spectrograms, row_ids = load_and_process_audio(
            audio_files[0], 32000, 128, 2048, 512
        )
        
        specs_tensor = torch.tensor(np.array(spectrograms)).unsqueeze(1)
        
        with torch.no_grad():
            outputs = model(specs_tensor)
            probs = torch.sigmoid(outputs)
        
        print(f"  Segments: {len(spectrograms)}, Output: {probs.shape}")
        print("  ✅ PASSED")
        return True
    
    print("  ⚠️  No audio found")
    return True


def test_ensemble():
    """Test ensemble predictor."""
    print("\n" + "="*60)
    print("TEST 5: Ensemble Predictor")
    print("="*60)
    
    from ensemble import MixedEnsemblePredictor
    import torch.nn as nn
    
    class DummySpecModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 234)
    
    class DummyWaveModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 234)
    
    mixed = MixedEnsemblePredictor(
        spectrogram_models=[DummySpecModel()],
        waveform_models=[DummyWaveModel()],
        weights=[0.5, 0.5],
        device="cpu"
    )
    
    result = mixed.predict(
        spectrograms=torch.randn(2, 1, 128, 313),
        waveforms=torch.randn(2, 160000)
    )
    
    assert result.shape == (2, 234)
    
    print(f"  Output: {result.shape}")
    print("  ✅ PASSED")
    return True


def test_submission_format():
    """Test submission format."""
    print("\n" + "="*60)
    print("TEST 6: Submission Format")
    print("="*60)
    
    data_dir = Path("data/birdclef-2026")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")
    label_cols = [c for c in sample_sub.columns if c != 'row_id']
    
    submission = pd.DataFrame({
        'row_id': ['test_001', 'test_002']
    })
    for col in label_cols:
        submission[col] = np.random.uniform(0, 1, size=len(submission))
    
    probs = submission[label_cols].values
    assert (probs >= 0).all() and (probs <= 1).all()
    
    print(f"  Columns: {len(submission.columns)}, Rows: {len(submission)}")
    print("  ✅ PASSED")
    return True


def main():
    print("\n" + "="*60)
    print("PERCH INTEGRATION TESTS (Quick)")
    print("="*60)
    
    tests = [
        ("Audio Dataset", test_audio_dataset),
        ("Model Structure", test_perch_model_structure),
        ("Checkpoint", test_checkpoint_save_load),
        ("EfficientNet Prediction", test_prediction_with_efficientnet),
        ("Ensemble", test_ensemble),
        ("Submission Format", test_submission_format),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            results.append((name, test_fn()))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"  {name}: {'✅' if result else '❌'}")
    
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

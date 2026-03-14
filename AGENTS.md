# BirdClef 2026 Project

## Competition
- **Task**: Acoustic Species Identification in the Pantanal, South America
- **Goal**: Identify bird species from audio recordings
- **Deadline**: June 3, 2026

## Project Structure
```
├── data/           # Raw and processed data
├── src/            # Source code
├── notebooks/      # EDA and experimentation
├── models/         # Saved model checkpoints
└── requirements.txt
```

## Key Approaches
- Audio → Spectrogram conversion (mel spectrograms)
- Pre-trained models: Perch, EfficientNet, YAMNet
- Transfer learning for bird classification

## Commands
- Install dependencies: `pip install -r requirements.txt`
- Run training: `python src/train.py`
- Make predictions: `python src/predict.py`

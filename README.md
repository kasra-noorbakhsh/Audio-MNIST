# Spoken Digit Recognition with Audio MNIST

## Overview

This project implements a speaker-independent spoken digit recognition system using the [Audio MNIST dataset](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist) from Kaggle. The dataset contains ~30,000 WAV audio files of spoken digits (0-9) from 60 speakers. We use PyTorch to build a CNN model that processes log-Mel spectrograms, achieving high accuracy (~96-98% on test set).

Key focuses:
- **Reproducibility**: Seeded operations, cached features/metadata, version logging.
- **Efficiency**: Feature caching, GPU support, early stopping.
- **Performance**: Audio preprocessing (resampling, trimming, normalization), CNN with BatchNorm/Dropout.
- **Export**: Trained model (.pth) and local inference script for real-time predictions on host machine.

## Dataset

- **Source**: Kaggle Audio MNIST (~30k mono WAV files at 48kHz, ~1s each).
- **Structure**: Files organized by speaker (e.g., `data/01/0_01_0.wav` = digit 0, speaker 01, iteration 0).
- **Preprocessing**: Resampled to 16kHz, trimmed silence, normalized, padded/truncated to 1s, converted to log-Mel spectrograms (128x32).

## Features and Improvements

- **Preprocessing Pipeline**: Librosa for loading/resampling/trimming/normalization and Mel spectrogram extraction.
- **Model**: Lightweight CNN (2 conv layers, ~4.2M params) with Xavier init, BatchNorm, Dropout (0.3), dynamic FC layers.
- **Training**: AdamW optimizer, LR scheduler, early stopping, gradient clipping. Speaker-independent splits (70/15/15%).
- **Evaluation**: Test accuracy, confusion matrix, classification report (saved as PNG/TXT).
- **Reproducibility**: Random seeds (42), cached .npy features/.pt tensors/.csv metadata, library version logging.
- **Local Use**: Standalone inference script for predicting digits from any WAV file.

## Requirements

- Python 3.8+ (tested on 3.12)
- Libraries (from `requirements.txt` generated on Kaggle):
  - torch
  - torchvision
  - torchaudio
  - librosa
  - soundfile
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - tqdm
  - kaggle (for local dataset download)

Install via: `pip install -r requirements.txt`

## Setup

### On Kaggle
1. Create a new notebook.
2. Add the Audio MNIST dataset via "Add Input".
3. Copy-paste the refined code blocks sequentially into cells.
4. Run all: Processes data, trains model, evaluates, saves artifacts (e.g., `best_model.pth`, plots, logs).
5. Download outputs: Right-click files in /kaggle/working (e.g., .pth, .png, requirements.txt).

### Locally
1. Download dataset: Install Kaggle CLI (`pip install kaggle`), set API token (~/.kaggle/kaggle.json), run `kaggle datasets download -d sripaadsrinivasan/audio-mnist --unzip -p data/`.
2. Adjust paths: Set `DATA_DIR = Path("data/data")` (nested due to unzip).
3. Install requirements.
4. Run the script/notebook: Features will cache in `./cache`, model trains/saves to `./best_model.pth`.

## Usage

### Training
- Run the full script/notebook.
- Monitors val acc, saves best model.
- Outputs: Training log (`training_log.csv`), metrics plot (`training_metrics.png`), CM (`confusion_matrix.png`), report (`classification_report.txt`).

### Inference (Local)
Use `local_inference.py` (provided in conversation):
- Update `MODEL_PATH` to your .pth file.
- Hardcode mean/std from training output (e.g., from console: "Normalization: mean=..., std=...").
- Run: `python local_inference.py path/to/your_digit.wav`
- Expects ~1s mono WAV (spoken digit 0-9); outputs predicted digit.

Example:
```
Predicted Digit: 7
```

## Project Structure
- **Code Blocks**: Modular (imports, preprocessing, model, training, eval).
- **Cache Dir**: Features (.npy), metadata/splits (.csv), tensors (.pt).
- **Working Dir**: Model (.pth), plots (.png), logs (.csv/.txt).
- **Inference Script**: Self-contained with model class and preprocess functions.

## Results
- **Expected Performance**: ~96-98% test accuracy (easy dataset; CNN on Mel specs).
- **Improvements Potential**: Add data augmentation (noise/shift) for 99%+, or use ResNet/Transformers.
- **Runtime**: ~10-20min on Kaggle GPU for full training.

## Credits
- Dataset: Sripaad Srinivasan on Kaggle.
- Libraries: PyTorch, Librosa, etc.
- Built with iterative refinements for reproducibility and local deployment.

For issues, check logs or re-run with debug prints. Contributions welcome!

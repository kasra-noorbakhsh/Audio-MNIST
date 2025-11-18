import torch
import numpy as np
import librosa
import argparse

# Params (match project)
TARGET_SR = 16000
TARGET_LEN = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
NUM_CLASSES = 10
MODEL_PATH = "best_model.pth"

# Preprocess functions (copy from project)
def load_and_preprocess(path, target_sr=TARGET_SR, target_len=TARGET_LEN):
    try:
        y, sr = librosa.load(path, sr=None, mono=True, dtype='float32')
        y, _ = librosa.effects.trim(y, top_db=30)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        if np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y, norm=np.inf)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        elif len(y) > target_len:
            y = y[:target_len]
        return y, target_sr
    except Exception as e:
        raise ValueError(f"Error processing {path}: {e}")

def to_logmel(y, sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=sr/2)
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return logmel

# Model (copy class from project)
class CNN_Small(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_shape=(1, N_MELS, (TARGET_LEN // HOP_LENGTH) + 1)):
        super(CNN_Small, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3)
        )
        with torch.no_grad():
            dummy = torch.zeros((1,) + input_shape)  # (1, channels, height, width)
            flattened = self.features(dummy).view(1, -1).size(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(flattened, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Inference
def predict_digit(audio_path, model_path=MODEL_PATH, device='cpu'):
    model = CNN_Small()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    y, sr = load_and_preprocess(audio_path)
    logmel = to_logmel(y, sr)
    
    # Normalize (use train mean/std; hardcode or load if available)
    mean, std = -60.8624, 19.5190  # update from your run
    logmel = (logmel - mean) / (std + 1e-6)
    
    input_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,height,width)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoken Digit Recognition")
    parser.add_argument("audio_path", type=str, help="Path to WAV file")
    args = parser.parse_args()
    digit = predict_digit(args.audio_path)
    print(f"Predicted Digit: {digit}")

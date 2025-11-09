import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Xavier initializer (from notebook)
# -------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------------
# CNN_Small (exact same as notebook)
# -------------------------
class CNN_Small(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Small, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = None  # lazy init
        self.fc2 = None
        self.num_classes = num_classes

        # Apply deterministic initialization to conv layers immediately
        self.apply(init_weights)

    def _initialize_fc(self, x):
        """Initialize fully connected layers dynamically based on input shape."""
        n_features = x.view(x.size(0), -1).shape[1]
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.to(x.device)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        # Lazy initialization
        if self.fc1 is None:
            self._initialize_fc(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------
# Load model
# -------------------------
def load_trained_model(checkpoint_path="cnn_digit_recognition.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CNN_Small(num_classes=10)

    # Load only matching keys (skip fc1/fc2, which are lazily initialized)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    model.to(device)
    return model, device

# -------------------------
# Audio preprocessing (same as Kaggle)
# -------------------------
def preprocess_audio(file_path, sr=8000, n_mels=128, n_fft=512, hop_length=160):
    y, orig_sr = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    tensor = torch.tensor(logmel).unsqueeze(0).unsqueeze(0)  # shape (1, 1, n_mels, time)
    return tensor

# -------------------------
# Prediction
# -------------------------
def predict_digit(file_path, model_path="cnn_digit_recognition.pth"):
    model, device = load_trained_model(model_path)
    x = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        preds = model(x)
        pred_digit = torch.argmax(preds, dim=1).item()
    print(f"Predicted digit: {pred_digit}")

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    audio_path = "data/43/8_43_8.wav"
    predict_digit(audio_path)

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
MODEL_PATH = "best_snn_model.pth"  # Update to your SNN model path
TIME_STEPS = 20  # SNN-specific
DECAY_MULTIPLIER = 0.9
THRESHOLD = 1.0
PENALTY_THRESHOLD = 1.5

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

def to_logmel(y, sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT, positive_shift=True):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=sr/2)
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if positive_shift:
        logmel = (logmel + 80) / 80  # Shift to [0,1] for SNN
    return logmel

# Model (copy SNN class from project)
class InputToSpiking(torch.nn.Module):
    """Adds temporal noise (flickering) to input for rate coding."""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        random_activation = torch.rand_like(x).to(self.device)
        return random_activation * x  # Flicker for spiking

class SpikingConvLayer(torch.nn.Module):
    """Spiking conv layer with LIF dynamics."""
    def __init__(self, device, in_channels, out_channels, kernel_size=3, padding=1,
                 decay_multiplier=DECAY_MULTIPLIER, threshold=THRESHOLD, penalty_threshold=PENALTY_THRESHOLD):
        super().__init__()
        self.device = device
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.decay_multiplier = decay_multiplier
        self.threshold = threshold
        self.penalty_threshold = penalty_threshold
        self.reset_state()

    def reset_state(self):
        self.prev_inner = None

    def forward(self, x):
        batch_size, _, h, w = x.shape
        if self.prev_inner is None:
            self.prev_inner = torch.zeros(batch_size, self.conv.out_channels, h, w).to(self.device)

        input_excitation = self.bn(self.conv(x))

        inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier

        outer_excitation = torch.nn.functional.relu(inner_excitation - self.threshold)

        do_penalize_gate = (outer_excitation > 0).float()
        inner_excitation = inner_excitation - do_penalize_gate * (
            self.penalty_threshold / self.threshold * inner_excitation)

        delayed_return_state = self.prev_inner
        self.prev_inner = inner_excitation
        return delayed_return_state

class OutputPooling(torch.nn.Module):
    """Pools outputs over time (sum for rate)."""
    def __init__(self, average_output=False):
        super().__init__()
        self.reducer = lambda x, dim: x.sum(dim=dim) if not average_output else x.mean(dim=dim)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        return self.reducer(x, 0)

class SNN(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, time_steps=TIME_STEPS, device='cpu'):
        super().__init__()
        self.time_steps = time_steps
        self.device = device
        self.input_to_spiking = InputToSpiking(device)
        self.layer1 = SpikingConvLayer(device, 1, 32)  # Channels=1; change to 3 if using deltas
        self.pool1 = torch.nn.MaxPool2d(2)
        self.layer2 = SpikingConvLayer(device, 32, 64)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.dropout = torch.nn.Dropout(0.3)
        
        # Hardcoded flattened (fixed shape after pools: 64 ch, H=128/4=32, W=32/4=8)
        flattened = 64 * 32 * 8  # 16384

        self.fc1 = torch.nn.Linear(flattened, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.output_pooling = OutputPooling(average_output=False)
        
        self.to(self.device)  # Move to device

    def reset_state(self):
        self.layer1.reset_state()
        self.layer2.reset_state()

    def forward(self, x):
        self.reset_state()
        batch_size = x.size(0)
        outs = []

        for t in range(self.time_steps):
            xi = self.input_to_spiking(x)
            layer1_state = self.layer1(xi)
            cur = self.pool1(layer1_state)
            layer2_state = self.layer2(cur)
            cur = self.pool2(layer2_state)
            cur = self.dropout(cur)
            cur = cur.view(batch_size, -1)
            cur = torch.nn.functional.relu(self.fc1(cur))
            cur = self.fc2(cur)
            outs.append(cur)

        out = self.output_pooling(outs)
        return out  # For CrossEntropyLoss

# Inference
def predict_digit(audio_path, model_path=MODEL_PATH, device='cpu'):
    model = SNN(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    y, sr = load_and_preprocess(audio_path)
    logmel = to_logmel(y, sr)
    
    # Normalize (use train mean/std; hardcode or load if available)
    mean, std = 0.2392, 0.2440  # Update from your run
    logmel = (logmel - mean) / (std + 1e-6)
    
    input_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,height,width)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoken Digit Recognition (SNN)")
    parser.add_argument("audio_path", type=str, help="Path to WAV file")
    args = parser.parse_args()
    digit = predict_digit(args.audio_path)
    print(f"Predicted Digit: {digit}")

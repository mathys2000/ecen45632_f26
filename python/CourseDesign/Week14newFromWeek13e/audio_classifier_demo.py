"""
Simple PyTorch demo: classify 3 kinds of audio signals with a small neural network.

Classes (timbre / waveform shape):
    0 = sine wave      (pure tone, one frequency)
    1 = square wave   (odd harmonics)
    2 = sawtooth wave  (all harmonics)

Pipeline:  raw audio -> FFT magnitude spectrum (features) -> MLP -> 3 class scores

Compares a network with ONE hidden layer vs. TWO hidden layers.
Requires: torch, numpy, (optional) matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import square, sawtooth  # waveform generators

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------
# 1. Generate synthetic audio
# ----------------------------------------------------------------------
SR = 8000          # sample rate (Hz)
DURATION = 0.25    # seconds per clip
N_SAMPLES = int(SR * DURATION)
N_BINS = 64        # number of spectrum features fed to the network
CLASS_NAMES = ["sine", "square", "sawtooth"]


def make_clip(label):
    """One random clip of the given class: random pitch, phase, and noise."""
    f0 = np.random.uniform(100, 400)           # fundamental frequency (Hz)
    phase = np.random.uniform(0, 2 * np.pi)
    t = np.arange(N_SAMPLES) / SR
    arg = 2 * np.pi * f0 * t + phase
    if label == 0:
        x = np.sin(arg)
    elif label == 1:
        x = square(arg)
    else:
        x = sawtooth(arg)
    return x + 0.2 * np.random.randn(N_SAMPLES)  # add background noise


def to_features(x):
    """Waveform -> log-magnitude spectrum (0..2 kHz), reduced to N_BINS values."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spectrum = spectrum[freqs <= 2000]
    # average neighbouring bins down to N_BINS features
    chunks = np.array_split(spectrum, N_BINS)
    feats = np.array([c.mean() for c in chunks])
    feats = np.log1p(feats)
    return (feats - feats.mean()) / (feats.std() + 1e-8)  # normalise


def make_dataset(n_per_class):
    X, y = [], []
    for label in range(3):
        for _ in range(n_per_class):
            X.append(to_features(make_clip(label)))
            y.append(label)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


X_train, y_train = make_dataset(300)
X_test, y_test = make_dataset(100)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ----------------------------------------------------------------------
# 2. Define the networks
# ----------------------------------------------------------------------
# One hidden layer:  64 -> 16 -> 3
one_hidden = nn.Sequential(
    nn.Linear(N_BINS, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

# Two hidden layers: 64 -> 32 -> 16 -> 3
two_hidden = nn.Sequential(
    nn.Linear(N_BINS, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)


# ----------------------------------------------------------------------
# 3. Train + evaluate
# ----------------------------------------------------------------------
def train(model, epochs=60, lr=0.01):
    loss_fn = nn.CrossEntropyLoss()  # includes softmax
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)  # forward pass
        loss.backward()                           # backpropagation
        opt.step()                                # update weights
        history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}  loss = {loss.item():.4f}")
    return history


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


results = {}
for name, model in [("1 hidden layer", one_hidden), ("2 hidden layers", two_hidden)]:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} ({n_params} parameters) ===")
    results[name] = train(model)
    print(f"  train accuracy: {accuracy(model, X_train, y_train):.1%}")
    print(f"  test accuracy:  {accuracy(model, X_test, y_test):.1%}")

# ----------------------------------------------------------------------
# 4. Try it on brand-new clips
# ----------------------------------------------------------------------
print("\n=== Predictions on new clips (2-hidden-layer model) ===")
two_hidden.eval()
for true_label in range(3):
    feats = torch.tensor(to_features(make_clip(true_label)), dtype=torch.float32)
    probs = torch.softmax(two_hidden(feats), dim=0)
    guess = probs.argmax().item()
    print(f"  true: {CLASS_NAMES[true_label]:9s} -> predicted: {CLASS_NAMES[guess]:9s} "
          f"(confidence {probs[guess]:.0%})")

# ----------------------------------------------------------------------
# 5. Optional: plots for the classroom
# ----------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(3):
        x = make_clip(i)
        axes[0, i].plot(x[:200])
        axes[0, i].set_title(f"{CLASS_NAMES[i]} waveform")
        axes[1, i].plot(to_features(x))
        axes[1, i].set_title(f"{CLASS_NAMES[i]} spectrum features")
    plt.tight_layout()
    plt.show()

    plt.figure()
    for name, hist in results.items():
        plt.plot(hist, label=name)
    plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend()
    plt.title("Training loss: 1 vs 2 hidden layers")
    plt.show()
except ImportError:
    print("\n(matplotlib not installed - skipping plots)")

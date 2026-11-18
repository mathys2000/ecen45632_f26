"""Lecture 28: MLPs for Signal Classification
DSP features -> PyTorch tensors -> MLP -> cross-entropy -> Adam -> evaluation.
Uses the Week 12 synthetic kick/snare/cymbal feature dataset.
"""
from pathlib import Path
import copy
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = Path('/mnt/data/week12_dsp_ml/synthetic_audio_features.csv')
OUT_DIR = Path('/mnt/data/Week14newFromWeek13e_lecture26/assets')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Load engineered DSP features
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_names = [c for c in df.columns if c != 'Class']
X = df[feature_names].to_numpy(dtype=np.float32)
y_text = df['Class'].to_numpy()

encoder = LabelEncoder()
y = encoder.fit_transform(y_text).astype(np.int64)
class_names = encoder.classes_.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

# Scale using TRAINING data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64,
    shuffle=True,
)

# -----------------------------------------------------------------------------
# 2) Define a small MLP
# -----------------------------------------------------------------------------
class SignalMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),   # raw logits; no softmax here
        )

    def forward(self, x):
        return self.net(x)

model = SignalMLP(len(feature_names), len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------------------------------------------------------
# 3) Training loop: forward -> loss -> zero_grad -> backward -> step
# -----------------------------------------------------------------------------
history = {'loss': [], 'train_acc': [], 'test_acc': []}
best_state = None
best_test_acc = -1.0

epochs = 120
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    n_seen = 0

    for xb, yb in train_loader:
        # 1. Forward pass
        logits = model(xb)
        # 2. Compute loss
        loss = criterion(logits, yb)
        # 3. Clear old gradients
        optimizer.zero_grad()
        # 4. Backpropagate
        loss.backward()
        # 5. Update parameters
        optimizer.step()

        running_loss += loss.item() * xb.shape[0]
        n_seen += xb.shape[0]

    epoch_loss = running_loss / n_seen

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).argmax(dim=1)
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(dim=1)
        train_acc = (train_pred == y_train_t).float().mean().item()
        test_acc = (test_pred == y_test_t).float().mean().item()

    history['loss'].append(epoch_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.softmax(test_logits, dim=1)
    test_pred = test_logits.argmax(dim=1).cpu().numpy()

acc = accuracy_score(y_test, test_pred)
cm = confusion_matrix(y_test, test_pred)

print('Features:', feature_names)
print('Classes:', class_names)
print('Train shape:', X_train_t.shape, 'Test shape:', X_test_t.shape)
print(model)
print(f'Best held-out test accuracy: {acc:.3f}')
print('Confusion matrix:\n', cm)
print(classification_report(y_test, test_pred, target_names=class_names, digits=3))

# Example logits / probabilities for one test clip
sample_logits = test_logits[0]
sample_probs = test_probs[0]
print('Example true class:', class_names[y_test[0]])
print('Example logits:', sample_logits.numpy())
print('Example probabilities:', sample_probs.numpy())
print('Predicted class:', class_names[int(sample_probs.argmax())])

# -----------------------------------------------------------------------------
# 4) Figures for slides / notebook
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(history['loss'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Training cross-entropy loss')
ax.set_title('MLP training loss')
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / 'training_loss.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(history['train_acc'], label='Train')
ax.plot(history['test_acc'], label='Held-out test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.0, 1.05)
ax.set_title('Classification accuracy during training')
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / 'training_accuracy.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 4.7))
im = ax.imshow(cm)
ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha='right')
ax.set_yticks(range(len(class_names)), class_names)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title('Held-out confusion matrix')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT_DIR / 'confusion_matrix.png', dpi=180)
plt.close(fig)

# -----------------------------------------------------------------------------
# 5) A tiny gradient demonstration for one sample
# -----------------------------------------------------------------------------
small = nn.Linear(len(feature_names), len(class_names))
small_loss = nn.CrossEntropyLoss()
xb = X_train_t[:1]
yb = y_train_t[:1]
logits = small(xb)
loss = small_loss(logits, yb)
small.zero_grad()
loss.backward()
print('One-sample linear-classifier loss:', float(loss))
print('Weight-gradient shape:', tuple(small.weight.grad.shape))
print('Bias-gradient shape:', tuple(small.bias.grad.shape))

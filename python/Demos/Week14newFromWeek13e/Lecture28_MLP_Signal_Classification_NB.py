# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-py
# ---

# %% [markdown]
# # Lecture 28 — MLPs for Signal Classification
#
# This notebook follows the lecture flow: engineered DSP features → MLP → cross-entropy → Adam → five-step training loop.

# %% [markdown]
# ## 1. Imports, reproducibility, and DSP-feature dataset
#
# The dataset contains spectral centroid, zero-crossing rate, spectral roll-off, RMS energy, spectral flux, and three MFCCs for three synthetic signal classes.

# %%
from pathlib import Path
import copy, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

SEED=7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATA_PATH=Path('/mnt/data/week12_dsp_ml/synthetic_audio_features.csv')

# %%
df = pd.read_csv(DATA_PATH)
feature_names = [c for c in df.columns if c != 'Class']
X = df[feature_names].to_numpy(dtype=np.float32)
y_text = df['Class'].to_numpy()
encoder = LabelEncoder()
y = encoder.fit_transform(y_text).astype(np.int64)
class_names = encoder.classes_.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

print(feature_names)
print(class_names)
print(X_train_t.shape, X_test_t.shape)


# %% [markdown]
# ## 2. Define the MLP
#
# For one example, each affine layer computes $\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}$, followed by ReLU in the hidden layers. The final layer returns raw logits.

# %%
class SignalMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )
    def forward(self, x):
        return self.net(x)

model = SignalMLP(len(feature_names), len(class_names))
print(model)
print('Total trainable parameters:', sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 3. Logits, softmax, and cross-entropy
#
# `CrossEntropyLoss` expects raw logits. For a single correct class $y$, the loss is $L=-\log p_y$, where $p_y$ comes from the softmax of the logits.

# %%
criterion = nn.CrossEntropyLoss()
xb = X_train_t[:4]
yb = y_train_t[:4]
logits = model(xb)
probs = torch.softmax(logits, dim=1)
loss = criterion(logits, yb)
print('logits shape:', logits.shape)
print('probability row sums:', probs.sum(dim=1))
print('loss:', loss.item())

# %% [markdown]
# ## 4. Adam and the five-step training loop
#
# Each minibatch follows: forward → loss → clear gradients → backward → parameter update.

# %%
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = {'loss': [], 'train_acc': [], 'test_acc': []}
best_state = None
best_test_acc = -1

for epoch in range(120):
    model.train()
    running_loss = 0.0
    n_seen = 0
    for xb, yb in train_loader:
        logits = model(xb)                 # 1. forward
        loss = criterion(logits, yb)       # 2. loss
        optimizer.zero_grad()              # 3. clear old gradients
        loss.backward()                    # 4. backpropagate
        optimizer.step()                   # 5. update parameters
        running_loss += loss.item()*xb.shape[0]
        n_seen += xb.shape[0]

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).argmax(dim=1)
        test_pred = model(X_test_t).argmax(dim=1)
        train_acc = (train_pred == y_train_t).float().mean().item()
        test_acc = (test_pred == y_test_t).float().mean().item()
    history['loss'].append(running_loss/n_seen)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
print('best test accuracy:', best_test_acc)

# %% [markdown]
# ## 5. Training curves

# %%
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(history['loss'])
ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss')
ax.grid(True, alpha=.25)
plt.show()

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(history['train_acc'], label='Train')
ax.plot(history['test_acc'], label='Held-out test')
ax.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy during training', ylim=(0,1.05))
ax.legend(); ax.grid(True, alpha=.25)
plt.show()

# %% [markdown]
# ## 6. Held-out evaluation and confusion matrix

# %%
model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.softmax(test_logits, dim=1)
    test_pred_t = test_logits.argmax(dim=1)

test_pred = test_pred_t.numpy()
print('accuracy:', accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_test, test_pred)
print(cm)

fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm)
ax.set_xticks(range(len(class_names)), class_names)
ax.set_yticks(range(len(class_names)), class_names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.show()

# %% [markdown]
# ## 7. Inspect one prediction
#
# Logits are unconstrained real numbers. Softmax converts them to positive class probabilities that sum to one.

# %%
i=0
print('true:', class_names[y_test[i]])
print('logits:', test_logits[i])
print('probabilities:', test_probs[i])
print('predicted:', class_names[int(test_probs[i].argmax())])

# %% [markdown]
# ## 8. Gradient shape check
#
# For a linear classifier with $K=3$ classes and $D=8$ features, the weight-gradient shape is $(3,8)$ because there is one gradient for every learnable weight.

# %%
small = nn.Linear(len(feature_names), len(class_names))
loss = nn.CrossEntropyLoss()(small(X_train_t[:1]), y_train_t[:1])
small.zero_grad(); loss.backward()
print('weight grad:', small.weight.grad.shape)
print('bias grad:', small.bias.grad.shape)

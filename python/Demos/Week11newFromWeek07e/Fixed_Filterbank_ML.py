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
# # Fixed_Filterbank_ML
#
# Fixed filter-bank features + Scikit-Learn

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import torch


# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)


# %%
def _make_filterbank(fs: float, taps: int = 63) -> list[np.ndarray]:
    nyq = fs / 2
    return [
        ss.firwin(taps, 700 / nyq),
        ss.firwin(taps, [800 / nyq, 1800 / nyq], pass_zero=False),
        ss.firwin(taps, [1900 / nyq, 3200 / nyq], pass_zero=False),
        ss.firwin(taps, 3300 / nyq, pass_zero=False),
    ]



# %%
def _subband_features(x: np.ndarray, bank: list[np.ndarray]) -> np.ndarray:
    feats = []
    for h in bank:
        y = ss.lfilter(h, [1.0], x)
        feats.append(np.log(np.mean(y**2) + 1e-9))
    return np.asarray(feats)



# %%
# Parameters
rng = np.random.default_rng(42)
fs = 8000.0
N = 256
t = np.arange(N) / fs   # time axis
bank = _make_filterbank(fs)
centers = [400, 1300, 2700]


# %%
# Generate dataset and labels
C, ix = 1, 0
sigix = np.zeros((3, N))
X, y = [], []
for cls, f0 in enumerate(centers):
    for i in range(180):
        #t = np.arange(N) / fs
        #f = f0 + rng.normal(scale=70)
        f = f0 + rng.normal(scale=300)
        phase = rng.uniform(0, 2*np.pi)
        sig = np.sin(2*np.pi*f*t + phase)
        if cls==C and i==ix:
            sigix[0] = sig
            fix = f
            phaseix = phase
        sig += 0.4 * np.sin(2*np.pi*(f*1.18)*t + rng.uniform(0, 2*np.pi))
        if cls==C and i==ix:
            sigix[1] = sig
        sig += rng.normal(scale=0.7, size=N)
        if cls==C and i==ix:
            sigix[2] = sig
        X.append(_subband_features(sig, bank))
        y.append(cls)
        if cls==C and i==ix:
            featsix = _subband_features(sig, bank)
            clsix = cls


# %%
print(f'features: {featsix}, class: {clsix}')
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(t, sigix[0], label='1st Sinusoid')
axs[0].set_title(f'Signal Sample, Class={C}, ix={ix}, f={fix:.1f} Hz, $\\theta$={180*phaseix/np.pi:.1f} deg')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(t, sigix[1], label='2nd Sinusoid')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(t, sigix[2], label='Noise Added')
axs[2].set_xlabel('time [sec]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# Logistic regression using Scikit-Learn
X = np.asarray(X); y = np.asarray(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("Fixed filter-bank feature accuracy =", accuracy_score(yte, pred))
print("confusion matrix:\n", confusion_matrix(yte, pred))
print("Only 4 fixed subband-energy features feed the classifier.")


# %%
# Plot Signal energy matrix
fig, ax = plt.subplots(figsize=(8.8, 4.6))
fig.canvas.toolbar_position = 'top'
means = np.vstack([X[y == c].mean(axis=0) for c in range(3)])
im = ax.imshow(means, aspect='auto')
ax.set_xticks(range(4), ['low', 'low-mid', 'high-mid', 'high'])
ax.set_yticks(range(3), ['class 0', 'class 1', 'class 2'])
ax.set_title('Mean log-energy features from a fixed four-band filter bank')
fig.colorbar(im, ax=ax, label='mean log energy')
plt.tight_layout()
plt.show()


# %%
"""Tiny PyTorch classifier on fixed filter-bank features."""
rng = np.random.default_rng(9)
# Synthetic already-extracted 4-D subband features: one dominant band per class.
X = []; y = []
for c in range(3):      # 3 classes
    for _ in range(120):     # 120 observations per class
        v = rng.normal(scale=0.6, size=4)    # 4 random features
        v[c] += 3.0                          # bump up feature of class c
        #v[c] += 1.0                          # bump up feature of class c
        X.append(v); y.append(c)      # feature vectors and labels


# %%
# Tiny PyTorch multiclass logistic regression model
Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)
model = torch.nn.Linear(4, 3)
opt = torch.optim.Adam(model.parameters(), lr=0.04)
for _ in range(120):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(Xt), yt)
    loss.backward(); opt.step()
acc = (model(Xt).argmax(1) == yt).float().mean().item()
print(f"Tiny 4→3 PyTorch classifier training accuracy: {acc:.3f}")
print("DSP front end is fixed; only the small classifier is learned.")


# %%

# %%

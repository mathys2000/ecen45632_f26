"""Week 7 Polyphase Structures — DSP + ML demonstrations.

Requires: numpy, scipy, matplotlib, scikit-learn, torch.
All demonstrations use fixed random seeds for repeatability.
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def polyphase_components(h: np.ndarray, M: int) -> list[np.ndarray]:
    """Return e_k[n] = h[nM+k], k=0,...,M-1."""
    h = np.asarray(h, dtype=float)
    if M < 1:
        raise ValueError("M must be a positive integer")
    return [h[k::M].copy() for k in range(M)]


def reconstruct_from_polyphase(phases: list[np.ndarray], M: int) -> np.ndarray:
    """Interleave polyphase components to reconstruct h[n]."""
    N = max(k + M * (len(e) - 1) for k, e in enumerate(phases) if len(e)) + 1
    h = np.zeros(N)
    for k, e in enumerate(phases):
        h[k:k + M * len(e):M] = e
    return h


def demo_polyphase_decomposition() -> None:
    h = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    M = 3
    phases = polyphase_components(h, M)
    h_rec = reconstruct_from_polyphase(phases, M)
    print("h =", h)
    for k, e in enumerate(phases):
        print(f"e_{k}[n] =", e)
    print("reconstructed =", h_rec)
    print("exact reconstruction:", np.allclose(h, h_rec))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    markers = ['o', 's', '^']
    for k, e in enumerate(phases):
        idx = k + M * np.arange(len(e))
        ax.stem(idx, e, linefmt='-', markerfmt=markers[k % len(markers)], basefmt=' ', label=f'phase {k}')
    ax.set_title('One FIR impulse response split into M = 3 polyphase subsequences')
    ax.set_xlabel('original tap index n')
    ax.set_ylabel('h[n]')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def demo_efficient_decimation() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=80)
    h = signal.firwin(15, 0.28)
    M = 4

    # Naive conceptual implementation: compute every FIR output, then discard 3/4.
    v = signal.convolve(x, h, mode='full')
    y_naive = v[::M]

    # SciPy upfirdn uses the multirate/polyphase idea internally.
    y_poly = signal.upfirdn(h, x, up=1, down=M)

    L = min(len(y_naive), len(y_poly))
    print("max |naive - polyphase| =", np.max(np.abs(y_naive[:L] - y_poly[:L])))
    N = len(h)
    print(f"Approximate multiplication-rate ratio naive/polyphase ≈ M = {M}")
    print(f"Naive: about {N} multiplies/input sample; efficient: about {N/M:.2f} multiplies/input sample on average")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    n = np.arange(L)
    ax.plot(n, y_naive[:L], 'o-', label='filter then downsample')
    ax.plot(n, y_poly[:L], 'x--', label='polyphase / upfirdn')
    ax.set_title('Efficient decimation gives the same retained output samples')
    ax.set_xlabel('output index m')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def demo_first_noble_identity() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=24)
    g = np.array([0.2, 0.6, 0.2])
    M = 3

    # H(z^M): insert M-1 zeros between taps of g.
    h_expanded = np.zeros((len(g) - 1) * M + 1)
    h_expanded[::M] = g

    left = signal.convolve(x, h_expanded, mode='full')[::M]
    right = signal.convolve(x[::M], g, mode='full')
    L = min(len(left), len(right))
    print("First Noble Identity numerical check")
    print("max error =", np.max(np.abs(left[:L] - right[:L])))
    print("Important: an arbitrary H(z) cannot be commuted through ↓M; the identity uses H(z^M).")


def demo_grouped_conv1d() -> None:
    import torch

    torch.manual_seed(3)
    layer = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=2, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        # Outputs 0,1 see only input channels 0,1; outputs 2,3 see only 2,3.
        layer.weight[:2] = 0.5
        layer.weight[2:] = -0.25

    x = torch.zeros(1, 4, 16)
    x[:, 0, 7] = 1.0
    y = layer(x)
    affected = y.abs().amax(dim=-1).squeeze(0).detach().numpy()
    print("Peak response in each output channel after perturbing input channel 0:")
    print(affected)
    print("Only outputs in the same group respond.")
    print("Analogy to polyphase: partition work into independent branches; difference: grouped conv partitions CHANNELS, polyphase partitions SAMPLE PHASES.")


def demo_second_noble_identity() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=12)
    g = np.array([0.25, 0.5, 0.25])
    L = 4

    up = np.zeros((len(x) - 1) * L + 1)
    up[::L] = x
    h_expanded = np.zeros((len(g) - 1) * L + 1)
    h_expanded[::L] = g
    left = signal.convolve(up, h_expanded, mode='full')

    low = signal.convolve(x, g, mode='full')
    right = np.zeros((len(low) - 1) * L + 1)
    right[::L] = low

    print("Second Noble Identity numerical check")
    print("max error =", np.max(np.abs(left - right)))
    print("Important: the filter that commutes with ↑L has the special H(z^L) form.")


def demo_efficient_interpolation() -> None:
    rng = np.random.default_rng(22)
    x = rng.normal(size=30)
    L = 3
    h = L * signal.firwin(17, 1 / L * 0.88)

    # Naive zero insertion + full-rate filtering.
    xu = np.zeros((len(x) - 1) * L + 1)
    xu[::L] = x
    y_naive = signal.convolve(xu, h, mode='full')

    # Efficient polyphase interpolation.
    y_poly = signal.upfirdn(h, x, up=L, down=1)
    Lc = min(len(y_naive), len(y_poly))
    print("max |naive - polyphase| =", np.max(np.abs(y_naive[:Lc] - y_poly[:Lc])))
    print(f"Ideal arithmetic saving is approximately a factor of L = {L} relative to multiplying by inserted zeros.")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(y_naive[:45], 'o-', label='zero insertion + FIR')
    ax.plot(y_poly[:45], 'x--', label='polyphase / upfirdn')
    ax.set_title('Efficient interpolation avoids multiplying by inserted zeros')
    ax.set_xlabel('high-rate output index')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def haar_analysis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Critically sampled two-channel Haar analysis for an even-length vector."""
    x = np.asarray(x, dtype=float)
    if len(x) % 2:
        x = np.pad(x, (0, 1))
    a = (x[0::2] + x[1::2]) / math.sqrt(2)
    d = (x[0::2] - x[1::2]) / math.sqrt(2)
    return a, d


def haar_synthesis(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    x = np.empty(2 * len(a))
    x[0::2] = (a + d) / math.sqrt(2)
    x[1::2] = (a - d) / math.sqrt(2)
    return x


def demo_haar_filter_bank() -> None:
    n = np.arange(64)
    x = np.sin(0.12 * np.pi * n) + 0.35 * np.cos(0.78 * np.pi * n)
    a, d = haar_analysis(x)
    xr = haar_synthesis(a, d)
    print("Haar perfect-reconstruction max error =", np.max(np.abs(x - xr)))
    print("input energy =", np.sum(x**2), "subband energy =", np.sum(a**2) + np.sum(d**2))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(n, x, label='input x[n]')
    ax.plot(n, xr, '--', label='reconstructed')
    ax.set_title('Two-channel Haar analysis/synthesis: exact reconstruction')
    ax.set_xlabel('n')
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _make_filterbank(fs: float, taps: int = 63) -> list[np.ndarray]:
    nyq = fs / 2
    return [
        signal.firwin(taps, 700 / nyq),
        signal.firwin(taps, [800 / nyq, 1800 / nyq], pass_zero=False),
        signal.firwin(taps, [1900 / nyq, 3200 / nyq], pass_zero=False),
        signal.firwin(taps, 3300 / nyq, pass_zero=False),
    ]


def _subband_features(x: np.ndarray, bank: list[np.ndarray]) -> np.ndarray:
    feats = []
    for h in bank:
        y = signal.lfilter(h, [1.0], x)
        feats.append(np.log(np.mean(y**2) + 1e-9))
    return np.asarray(feats)


def demo_fixed_filterbank_ml() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, confusion_matrix

    rng = np.random.default_rng(42)
    fs = 8000.0
    N = 256
    bank = _make_filterbank(fs)
    X, y = [], []
    centers = [400, 1300, 2700]
    for cls, f0 in enumerate(centers):
        for _ in range(180):
            t = np.arange(N) / fs
            f = f0 + rng.normal(scale=70)
            phase = rng.uniform(0, 2*np.pi)
            sig = np.sin(2*np.pi*f*t + phase)
            sig += 0.4 * np.sin(2*np.pi*(f*1.18)*t + rng.uniform(0, 2*np.pi))
            sig += rng.normal(scale=0.7, size=N)
            X.append(_subband_features(sig, bank))
            y.append(cls)
    X = np.asarray(X); y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Fixed filter-bank feature accuracy =", accuracy_score(yte, pred))
    print("confusion matrix:\n", confusion_matrix(yte, pred))
    print("Only 4 fixed subband-energy features feed the classifier.")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    means = np.vstack([X[y == c].mean(axis=0) for c in range(3)])
    im = ax.imshow(means, aspect='auto')
    ax.set_xticks(range(4), ['low', 'low-mid', 'high-mid', 'high'])
    ax.set_yticks(range(3), ['class 0', 'class 1', 'class 2'])
    ax.set_title('Mean log-energy features from a fixed four-band filter bank')
    fig.colorbar(im, ax=ax, label='mean log energy')
    plt.tight_layout()
    plt.show()


def demo_pytorch_lightweight_classifier() -> None:
    """Tiny PyTorch classifier on fixed filter-bank features."""
    import torch
    rng = np.random.default_rng(9)
    # Synthetic already-extracted 4-D subband features: one dominant band per class.
    X = []; y = []
    for c in range(3):
        for _ in range(120):
            v = rng.normal(scale=0.6, size=4)
            v[c] += 3.0
            X.append(v); y.append(c)
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.04)
    for _ in range(120):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), y)
        loss.backward(); opt.step()
    acc = (model(X).argmax(1) == y).float().mean().item()
    print(f"Tiny 4→3 PyTorch classifier training accuracy: {acc:.3f}")
    print("DSP front end is fixed; only the small classifier is learned.")


def main() -> None:
    demo_polyphase_decomposition()
    demo_efficient_decimation()
    demo_first_noble_identity()
    demo_grouped_conv1d()
    demo_second_noble_identity()
    demo_efficient_interpolation()
    demo_haar_filter_bank()
    demo_fixed_filterbank_ml()
    demo_pytorch_lightweight_classifier()


if __name__ == '__main__':
    main()

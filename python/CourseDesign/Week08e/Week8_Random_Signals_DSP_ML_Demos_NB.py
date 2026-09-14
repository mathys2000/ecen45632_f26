# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""Week 8: Random Signals — DSP + ML demos.

Designed to accompany:
  Lecture 15: Characterizing Random Signals
  Lecture 16: LTI Systems with Random Inputs

Dependencies: numpy, scipy, matplotlib, scikit-learn, torch
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F

RNG = np.random.default_rng(8)


def biased_autocorrelation(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Biased autocorrelation estimate normalized so r_hat[0] = 1."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full", method="fft")
    lags = signal.correlation_lags(len(x), len(x), mode="full")
    keep = (lags >= -max_lag) & (lags <= max_lag)
    r = corr[keep] / len(x)
    r = r / r[lags[keep] == 0][0]
    return lags[keep], r


def demo_ensemble_realizations() -> None:
    """A stochastic process is an ensemble; a recording is one realization."""
    n = np.arange(180)
    plt.figure(figsize=(9, 5))
    for i in range(6):
        phase = RNG.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * 0.05 * n + phase) + 0.35 * RNG.normal(size=n.size)
        plt.plot(n, x + 2.7 * i, label=f"realization {i+1}")
    plt.xlabel("sample n")
    plt.ylabel("amplitude (offset for display)")
    plt.title("Six realizations of the same stochastic process")
    plt.grid(alpha=0.2)
    plt.show()


def demo_stationarity_window_statistics() -> None:
    """Compare local means/variances for stationary and nonstationary data."""
    N = 1200
    n = np.arange(N)
    x_wss = RNG.normal(0, 1, N)
    x_nonstationary = 0.002 * n + (0.4 + 0.0015 * n) * RNG.normal(size=N)

    def block_stats(x: np.ndarray, block: int = 100):
        blocks = x[: len(x) // block * block].reshape(-1, block)
        return blocks.mean(axis=1), blocks.var(axis=1)

    for name, x in [("approximately stationary", x_wss), ("nonstationary", x_nonstationary)]:
        means, variances = block_stats(x)
        print(f"\n{name}")
        print("block means    =", np.round(means, 3))
        print("block variances=", np.round(variances, 3))

    plt.figure(figsize=(9, 4))
    plt.plot(n, x_wss, label="approximately stationary")
    plt.plot(n, x_nonstationary, label="changing mean / variance", alpha=0.8)
    plt.xlabel("sample n")
    plt.ylabel("amplitude")
    plt.title("Stationary-looking and nonstationary realizations")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def demo_autocorrelation_memory() -> None:
    """White noise has little lag memory; filtered noise has substantial lag memory."""
    N = 40000
    w = RNG.normal(size=N)
    ar1 = signal.lfilter([1.0], [1.0, -0.9], w)
    lags, rw = biased_autocorrelation(w, 30)
    _, ra = biased_autocorrelation(ar1, 30)

    plt.figure(figsize=(9, 4.5))
    plt.stem(lags, rw, basefmt=" ", label="white noise")
    plt.plot(lags, ra, "o-", markersize=3, label="AR(1) filtered noise")
    plt.xlabel("lag m")
    plt.ylabel("normalized autocorrelation")
    plt.title("Autocorrelation reveals statistical memory")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def demo_autocorrelation_matrix_psd() -> None:
    """Construct a Toeplitz correlation matrix and verify positive semidefiniteness."""
    rho = 0.82
    K = 10
    R = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (K, K))
    eigvals = np.linalg.eigvalsh(R)
    print("minimum eigenvalue =", eigvals.min())
    print("all eigenvalues >= 0 (within roundoff)?", np.all(eigvals >= -1e-12))

    a = RNG.normal(size=K)
    print("example quadratic form a^T R a =", float(a @ R @ a))

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(R, origin="lower", aspect="equal")
    plt.colorbar(label="correlation")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title("Toeplitz autocorrelation matrix")
    plt.show()


def demo_template_matching() -> None:
    """Use cross-correlation to estimate an unknown template location."""
    N = 300
    t = np.arange(41)
    template = np.sin(np.pi * t / 40) ** 2 * np.sin(2 * np.pi * 0.1 * t)
    true_start = 167
    x = 0.35 * RNG.normal(size=N)
    x[true_start : true_start + len(template)] += 2.0 * template

    score = signal.correlate(x, template, mode="valid")
    estimated_start = int(np.argmax(score))
    print("true start     =", true_start)
    print("estimated start=", estimated_start)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    axes[0].plot(x)
    axes[0].axvline(true_start, linestyle="--")
    axes[0].set_title("Noisy observation")
    axes[0].grid(alpha=0.2)
    axes[1].plot(score)
    axes[1].axvline(estimated_start, linestyle="--")
    axes[1].set_title("Cross-correlation template score")
    axes[1].set_xlabel("candidate start index")
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def demo_sklearn_correlation_features() -> None:
    """Correlation-based alignment/feature extraction before a Scikit-Learn classifier.

    Class 0 contains one short template; class 1 contains another. Each event appears
    at a random time. We turn a variable-time waveform into fixed correlation features.
    """
    n_samples = 900
    length = 160
    t = np.arange(31)
    template0 = np.sin(np.pi * t / 30) ** 2 * np.sin(2 * np.pi * 0.08 * t)
    template1 = np.sin(np.pi * t / 30) ** 2 * np.sin(2 * np.pi * 0.16 * t)

    X_features = []
    y_labels = []
    for _ in range(n_samples):
        label = int(RNG.integers(0, 2))
        template = template0 if label == 0 else template1
        x = 0.55 * RNG.normal(size=length)
        start = int(RNG.integers(15, length - len(template) - 15))
        x[start : start + len(template)] += RNG.uniform(1.2, 2.1) * template

        c0 = signal.correlate(x, template0, mode="valid")
        c1 = signal.correlate(x, template1, mode="valid")
        # Fixed-size features: strongest match to each template and corresponding lag.
        X_features.append([c0.max(), np.argmax(c0), c1.max(), np.argmax(c1)])
        y_labels.append(label)

    X_features = np.asarray(X_features)
    y_labels = np.asarray(y_labels)
    Xtr, Xte, ytr, yte = train_test_split(
        X_features, y_labels, test_size=0.3, random_state=8, stratify=y_labels
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("correlation-feature classification accuracy =", accuracy_score(yte, pred))


def demo_lti_mean_and_variance() -> None:
    """Verify mean/DC-gain and white-noise output-variance relationships by simulation."""
    N = 300000
    mu_x = 3.0
    sigma2_x = 4.0
    x = mu_x + np.sqrt(sigma2_x) * RNG.normal(size=N)

    filters = {
        "moving average [1/2, 1/2]": np.array([0.5, 0.5]),
        "difference [1, -1]": np.array([1.0, -1.0]),
    }
    for name, h in filters.items():
        y = signal.lfilter(h, [1.0], x)[100:]
        predicted_mean = mu_x * h.sum()
        predicted_variance = sigma2_x * np.sum(h**2)  # valid here because noise part is white
        print(f"\n{name}")
        print("predicted mean / measured mean =", predicted_mean, np.mean(y))
        print("predicted var  / measured var  =", predicted_variance, np.var(y))


def demo_correlation_convention() -> None:
    """Numerically verify the input-output cross-correlation formula for our convention.

    Convention: r_xy[m] = E{x[n] y[n+m]} for real signals.
    With y = h*x, this gives r_xy = r_xx * h (not h reversed).
    The reversed-h expression belongs to the alternate lag convention.
    """
    N = 100000
    x = signal.lfilter([1.0], [1.0, -0.7], RNG.normal(size=N))
    h = np.array([1.0, 0.4, -0.2])
    y = signal.lfilter(h, [1.0], x)
    x = x[200:]
    y = y[200:]

    maxlag = 15
    # Direct estimate r_xy[m] = mean x[n] y[n+m]
    rxy = []
    for m in range(-maxlag, maxlag + 1):
        if m >= 0:
            rxy.append(np.mean(x[: len(x)-m] * y[m:]))
        else:
            rxy.append(np.mean(x[-m:] * y[: len(y)+m]))
    rxy = np.asarray(rxy)

    # Direct r_xx on a wider lag set, then convolve with h.
    wide = maxlag + len(h) + 3
    rxx_lags = np.arange(-wide, wide + 1)
    rxx = []
    for m in rxx_lags:
        if m >= 0:
            rxx.append(np.mean(x[: len(x)-m] * x[m:]))
        else:
            rxx.append(np.mean(x[-m:] * x[: len(x)+m]))
    rxx = np.asarray(rxx)
    predicted_full = np.convolve(rxx, h, mode="full")
    predicted_lags = np.arange(rxx_lags[0], rxx_lags[-1] + len(h))
    target_lags = np.arange(-maxlag, maxlag + 1)
    predicted = np.array([predicted_full[predicted_lags == m][0] for m in target_lags])

    print("max |direct r_xy - r_xx*h| =", np.max(np.abs(rxy - predicted)))
    plt.figure(figsize=(9, 4))
    plt.plot(target_lags, rxy, "o", label="direct estimate")
    plt.plot(target_lags, predicted, "-", label="r_xx * h prediction")
    plt.xlabel("lag m")
    plt.ylabel("cross-correlation")
    plt.title("Cross-correlation convention check")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def demo_white_noise_coloring() -> None:
    """Filter white noise and compare its autocorrelation and PSD before/after."""
    N = 50000
    x = RNG.normal(size=N)
    h = np.ones(9) / 9
    y = signal.lfilter(h, [1.0], x)

    lags, rx = biased_autocorrelation(x, 30)
    _, ry = biased_autocorrelation(y[100:], 30)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].stem(lags, rx, basefmt=" ", label="white input")
    axes[0].plot(lags, ry, "o-", markersize=3, label="filtered output")
    axes[0].set_title("Filtering introduces lag correlation")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    f, Pxx = signal.welch(x, nperseg=1024)
    _, Pyy = signal.welch(y, nperseg=1024)
    axes[1].plot(f, 10 * np.log10(Pxx + 1e-12), label="white input")
    axes[1].plot(f, 10 * np.log10(Pyy + 1e-12), label="colored output")
    axes[1].set_xlabel("normalized cycles/sample")
    axes[1].set_ylabel("PSD (dB)")
    axes[1].set_title("The filter shapes the noise spectrum")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def add_noise_at_snr_torch(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise at a target RMS SNR."""
    signal_rms = torch.sqrt(torch.mean(x**2) + 1e-12)
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = torch.randn_like(x)
    noise = noise / (torch.sqrt(torch.mean(noise**2)) + 1e-12)
    return x + noise_rms * noise


def colored_noise_torch(length: int, taps: torch.Tensor) -> torch.Tensor:
    """Generate FIR-colored noise and normalize it to unit RMS."""
    w = torch.randn(1, 1, length + taps.numel() - 1)
    h = taps.flip(0).view(1, 1, -1)  # conv1d performs correlation; flip for convolution
    y = F.conv1d(w, h).view(-1)[:length]
    return y / (torch.sqrt(torch.mean(y**2)) + 1e-12)


def demo_pytorch_noise_augmentation() -> None:
    """Compare white and colored noise augmentation with the same RMS."""
    torch.manual_seed(8)
    fs = 1000.0
    n = torch.arange(600)
    clean = torch.sin(2 * torch.pi * 45 * n / fs) + 0.35 * torch.sin(2 * torch.pi * 110 * n / fs)

    white_aug = add_noise_at_snr_torch(clean, snr_db=12.0)
    taps = torch.tensor([0.08, 0.18, 0.28, 0.18, 0.08])
    colored = colored_noise_torch(clean.numel(), taps)
    signal_rms = torch.sqrt(torch.mean(clean**2))
    noise_rms = signal_rms / (10.0 ** (12.0 / 20.0))
    colored_aug = clean + noise_rms * colored

    plt.figure(figsize=(9, 4.5))
    k = 240
    plt.plot(n[:k] / fs, clean[:k].numpy(), label="clean")
    plt.plot(n[:k] / fs, white_aug[:k].numpy(), label="white augmentation", alpha=0.8)
    plt.plot(n[:k] / fs, colored_aug[:k].numpy(), label="colored augmentation", alpha=0.8)
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.title("PyTorch augmentation: same approximate SNR, different noise statistics")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


if __name__ == "__main__":
    print("Week 8 random-signal demos loaded.")
    print("Run individual demo_* functions interactively, or execute the Jupyter notebook.")

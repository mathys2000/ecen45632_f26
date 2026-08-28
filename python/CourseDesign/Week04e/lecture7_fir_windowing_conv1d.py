"""Lecture 7 demonstrations: FIR design, windowing, and PyTorch Conv1d.

Designed for a senior/first-year graduate DSP + ML course.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)


def ideal_lowpass(length: int, wc: float, window: str | None = None) -> np.ndarray:
    """Return a length-N linear-phase lowpass FIR by sampling the ideal sinc.

    Parameters
    ----------
    length:
        Number of taps. Odd length makes the symmetry center an integer.
    wc:
        Digital cutoff in rad/sample, 0 < wc < pi.
    window:
        Optional SciPy window name: boxcar, bartlett, hann, hamming, blackman.
    """
    if length % 2 == 0:
        raise ValueError("Use an odd length for this centered demonstration.")
    if not 0 < wc < np.pi:
        raise ValueError("wc must lie between 0 and pi.")

    m = (length - 1) / 2
    n = np.arange(length)
    r = n - m
    hd = np.empty(length, dtype=float)
    nonzero = r != 0
    hd[nonzero] = np.sin(wc * r[nonzero]) / (np.pi * r[nonzero])
    hd[~nonzero] = wc / np.pi

    if window is not None:
        w = signal.get_window(window, length, fftbins=False)
        hd = hd * w
    return hd


def plot_ideal_and_truncated() -> None:
    wc = 0.4 * np.pi
    n_inf = np.arange(-40, 41)
    h_inf = np.empty_like(n_inf, dtype=float)
    h_inf[n_inf == 0] = wc / np.pi
    nz = n_inf != 0
    h_inf[nz] = np.sin(wc * n_inf[nz]) / (np.pi * n_inf[nz])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    markerline, stemlines, baseline = ax.stem(n_inf, h_inf, basefmt=" ")
    ax.set_title("Ideal lowpass impulse response: infinite, two-sided sinc")
    ax.set_xlabel("n")
    ax.set_ylabel("$h_d[n]$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fir_ideal_sinc.png", dpi=180)
    plt.close(fig)


def plot_windows() -> None:
    N = 51
    windows = {
        "Rectangular": signal.windows.boxcar(N),
        "Bartlett": signal.windows.bartlett(N),
        "Hann": signal.windows.hann(N, sym=True),
        "Hamming": signal.windows.hamming(N, sym=True),
        "Blackman": signal.windows.blackman(N, sym=True),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for name, w in windows.items():
        ax.plot(w, label=name, linewidth=1.8)
    ax.set_title("Common windows in the time domain")
    ax.set_xlabel("Tap index")
    ax.set_ylabel("$w[n]$")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fir_windows_time.png", dpi=180)
    plt.close(fig)


def plot_windowed_responses() -> None:
    wc = 0.4 * np.pi
    N = 51
    names = ["boxcar", "bartlett", "hann", "hamming", "blackman"]
    labels = ["Rectangular", "Bartlett", "Hann", "Hamming", "Blackman"]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for name, label in zip(names, labels):
        h = ideal_lowpass(N, wc, name)
        w, H = signal.freqz(h, worN=4096)
        Hdb = 20 * np.log10(np.maximum(np.abs(H), 1e-8))
        ax.plot(w / np.pi, Hdb, label=label, linewidth=1.6)
    ax.axvline(wc / np.pi, linestyle="--", linewidth=1.2)
    ax.set_ylim(-110, 5)
    ax.set_xlim(0, 1)
    ax.set_title("Window choice trades transition width against stopband ripple")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fir_window_responses.png", dpi=180)
    plt.close(fig)


def plot_gibbs_zoom() -> None:
    wc = 0.4 * np.pi
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for N in [21, 51, 101]:
        h = ideal_lowpass(N, wc, "boxcar")
        w, H = signal.freqz(h, worN=8192)
        ax.plot(w / np.pi, np.abs(H), label=f"N={N}")
    ax.axvline(wc / np.pi, linestyle="--", linewidth=1.2)
    ax.set_xlim(0.25, 0.55)
    ax.set_ylim(-0.12, 1.15)
    ax.set_title("Rectangular truncation: the ripple narrows, but the overshoot persists")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel(r"$|H(e^{j\omega})|$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fir_gibbs_zoom.png", dpi=180)
    plt.close(fig)


def plot_phase_group_delay() -> None:
    N = 41
    h = ideal_lowpass(N, 0.35 * np.pi, "hamming")
    w, H = signal.freqz(h, worN=2048)
    phase = np.unwrap(np.angle(H))
    gd_w, gd = signal.group_delay((h, 1), w=1024)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(w / np.pi, phase, linewidth=1.8)
    ax.set_title("Symmetric FIR: approximately linear phase in the passband")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fir_linear_phase.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(gd_w / np.pi, gd, linewidth=1.8)
    ax.axhline((N - 1) / 2, linestyle="--", linewidth=1.2,
               label=f"$(N-1)/2={(N-1)/2:g}$ samples")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, N)
    ax.set_title("Linear phase implies nearly constant group delay")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel("Group delay (samples)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fir_group_delay.png", dpi=180)
    plt.close(fig)


def torch_fixed_fir_demo() -> tuple[float, float]:
    rng = np.random.default_rng(7)
    fs = 8000
    t = np.arange(0, 0.04, 1 / fs)
    x = (np.sin(2 * np.pi * 450 * t)
         + 0.45 * np.sin(2 * np.pi * 2100 * t)
         + 0.05 * rng.standard_normal(t.size)).astype(np.float32)

    h = ideal_lowpass(31, 0.30 * np.pi, "hamming").astype(np.float32)
    y_np = np.convolve(x, h, mode="full")[:len(x)]

    xt = torch.from_numpy(x).view(1, 1, -1)
    ht = torch.from_numpy(h)
    kernel = torch.flip(ht, dims=[0]).view(1, 1, -1)
    xpad = F.pad(xt, (len(h) - 1, 0))
    y_t = F.conv1d(xpad, kernel).view(-1).numpy()

    max_error = float(np.max(np.abs(y_np - y_t)))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    nshow = 150
    ax.plot(x[:nshow], label="input", linewidth=1.1)
    ax.plot(y_np[:nshow], label="NumPy DSP convolution", linewidth=1.8)
    ax.plot(y_t[:nshow], "--", label="PyTorch Conv1d result", linewidth=1.4)
    ax.set_title("A fixed FIR can be implemented exactly with PyTorch Conv1d")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fir_torch_match.png", dpi=180)
    plt.close(fig)

    # Demonstrate fixed versus learnable weights.
    weights = torch.nn.Parameter(kernel.clone())
    grad_flag = float(weights.requires_grad)
    return max_error, grad_flag


def main() -> None:
    plot_ideal_and_truncated()
    plot_windows()
    plot_windowed_responses()
    plot_gibbs_zoom()
    plot_phase_group_delay()
    max_error, grad_flag = torch_fixed_fir_demo()
    print(f"PyTorch/NumPy max convolution mismatch: {max_error:.3e}")
    print(f"Example learnable kernel requires_grad: {bool(grad_flag)}")


if __name__ == "__main__":
    main()

"""Week 2 DSP + ML teaching demos: sampling, aliasing, reconstruction, ZOH, quantization.

Designed for a combined senior / first-year graduate ECE course using
Oppenheim & Schafer, Discrete-Time Signal Processing.

Run as a script to generate figures and WAV files in ./generated_week2.
The companion notebook presents the same ideas interactively.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

try:
    import torch
except ImportError:  # keep most demos usable without torch
    torch = None


def normalized_sinc(x: np.ndarray) -> np.ndarray:
    """sinc(x) = sin(pi x)/(pi x), with sinc(0)=1."""
    return np.sinc(x)


def sample_sinusoid(f0_hz: float = 900.0, fs_hz: float = 4000.0, duration_s: float = 0.006):
    """Dense 'continuous-time' grid plus ideal sample values."""
    t = np.linspace(0.0, duration_s, 3000, endpoint=False)
    x_c = np.cos(2 * np.pi * f0_hz * t)
    n = np.arange(int(np.floor(duration_s * fs_hz)))
    t_n = n / fs_hz
    x_n = np.cos(2 * np.pi * f0_hz * t_n)
    return t, x_c, n, t_n, x_n


def alias_to_baseband(f_hz: float, fs_hz: float) -> float:
    """Fold a real sinusoid frequency into [0, fs/2]."""
    f_mod = np.mod(f_hz, fs_hz)
    return float(min(f_mod, fs_hz - f_mod))


def sampled_spectrum_copies(freq: np.ndarray, baseband_center_hz: float, bandwidth_hz: float,
                            fs_hz: float, kmax: int = 2) -> np.ndarray:
    """Simple triangular CT spectrum and its sampled replicas (teaching visualization)."""
    def tri(z):
        return np.maximum(1.0 - np.abs(z - baseband_center_hz) / bandwidth_hz, 0.0) + \
               np.maximum(1.0 - np.abs(z + baseband_center_hz) / bandwidth_hz, 0.0)
    out = np.zeros_like(freq, dtype=float)
    for k in range(-kmax, kmax + 1):
        out += tri(freq - k * fs_hz)
    return out


def synth_voice_like(fs_hz: int = 48000, duration_s: float = 2.5) -> np.ndarray:
    """Deterministic voice-like waveform with harmonics + an HF component for alias demos."""
    t = np.arange(int(fs_hz * duration_s)) / fs_hz
    # Slowly varying F0 around 180 Hz.
    f0 = 180.0 + 15.0 * np.sin(2 * np.pi * 2.2 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs_hz
    x = np.zeros_like(t)
    # Harmonic stack with a smooth spectral tilt.
    for k in range(1, 35):
        x += (1.0 / k**1.15) * np.sin(k * phase + 0.15 * k)
    # Formant-like shaping via resonant bandpass filters.
    sos1 = signal.iirpeak(700 / (fs_hz / 2), Q=5)
    sos2 = signal.iirpeak(1500 / (fs_hz / 2), Q=7)
    sos3 = signal.iirpeak(2600 / (fs_hz / 2), Q=9)
    shaped = 0.8 * signal.lfilter(*sos1, x) + 0.55 * signal.lfilter(*sos2, x) + 0.35 * signal.lfilter(*sos3, x)
    # Add a controlled high-frequency component that will alias after 4x decimation.
    hf = 0.16 * np.sin(2 * np.pi * 10000.0 * t)
    # Syllabic amplitude envelope.
    env = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.8 * t))**1.8
    y = env * shaped + hf
    y /= np.max(np.abs(y)) + 1e-12
    return 0.9 * y


def decimate_naive(x: np.ndarray, factor: int) -> np.ndarray:
    return x[::factor]


def decimate_antialias(x: np.ndarray, factor: int) -> np.ndarray:
    # Polyphase FIR performs anti-aliasing and decimation.
    return signal.resample_poly(x, up=1, down=factor)


def sinc_reconstruct(xn: np.ndarray, fs_hz: float, t_eval: np.ndarray, n0: int = 0) -> np.ndarray:
    """Finite sinc interpolation from a finite sequence of samples."""
    n = np.arange(n0, n0 + len(xn))
    tau = t_eval[:, None] * fs_hz - n[None, :]
    return normalized_sinc(tau) @ xn


def zoh_waveform(xn: np.ndarray, fs_hz: float, oversample: int = 30):
    """Simple zero-order hold visualization."""
    y = np.repeat(xn, oversample)
    t = np.arange(len(y)) / (fs_hz * oversample)
    return t, y


def uniform_quantize(x: np.ndarray, bits: int, full_scale: float = 1.0):
    """Mid-tread uniform quantizer over approximately [-full_scale, full_scale]."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    levels = 2**bits
    delta = 2 * full_scale / levels
    x_clip = np.clip(x, -full_scale, full_scale - delta)
    q = delta * np.round(x_clip / delta)
    q = np.clip(q, -full_scale, full_scale - delta)
    return q, delta


def measured_sqnr_db(x: np.ndarray, xq: np.ndarray) -> float:
    noise = xq - x
    return 10 * np.log10(np.mean(x**2) / np.mean(noise**2))


def theoretical_sqnr_db(bits: int) -> float:
    return 6.02 * bits + 1.76


def stft_power_db(x: np.ndarray, fs_hz: float, nperseg: int = 1024):
    f, t, Z = signal.stft(x, fs=fs_hz, nperseg=nperseg, noverlap=3*nperseg//4,
                          window="hann", boundary=None)
    p = 20 * np.log10(np.maximum(np.abs(Z), 1e-8))
    return f, t, p


def torch_stft_feature(x: np.ndarray, n_fft: int = 1024, hop: int = 256):
    """Magnitude STFT using PyTorch, demonstrating float conversion for ML."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    xt = torch.as_tensor(x, dtype=torch.float32)
    window = torch.hann_window(n_fft)
    X = torch.stft(xt, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return X.abs()


def int16_to_float32(x_int16: np.ndarray) -> np.ndarray:
    """Typical audio-to-ML conversion: int16 PCM to roughly [-1, 1)."""
    if x_int16.dtype != np.int16:
        raise TypeError("expected int16 PCM")
    return x_int16.astype(np.float32) / 32768.0


def write_wav(path: Path, fs_hz: int, x: np.ndarray):
    x = np.asarray(x)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 1.0:
        x = x / peak
    pcm = np.int16(np.clip(x, -1, 1 - 1/32768) * 32767)
    wavfile.write(path, fs_hz, pcm)


def make_demo_outputs(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fs = 48000
    x = synth_voice_like(fs)
    factor = 4
    fs_ds = fs // factor
    x_naive = decimate_naive(x, factor)
    x_aa = decimate_antialias(x, factor)
    xq8, _ = uniform_quantize(x, 8)
    xq4, _ = uniform_quantize(x, 4)

    write_wav(outdir / "01_clean_48k.wav", fs, x)
    write_wav(outdir / "02_decimated_naive_12k_alias.wav", fs_ds, x_naive)
    write_wav(outdir / "03_decimated_antialias_12k.wav", fs_ds, x_aa)
    write_wav(outdir / "04_quantized_8bit.wav", fs, xq8)
    write_wav(outdir / "05_quantized_4bit.wav", fs, xq4)

    print(f"Wrote demo WAV files to {outdir}")
    print(f"10 kHz at fs={fs_ds} Hz aliases to {alias_to_baseband(10000, fs_ds):.0f} Hz")
    for b in [4, 8, 12, 16]:
        t = np.arange(fs) / fs
        sine = 0.999 * np.sin(2*np.pi*997*t)
        q, _ = uniform_quantize(sine, b)
        print(f"B={b:2d}: theory={theoretical_sqnr_db(b):6.2f} dB, measured={measured_sqnr_db(sine,q):6.2f} dB")


if __name__ == "__main__":
    make_demo_outputs(Path(__file__).with_name("generated_week2"))

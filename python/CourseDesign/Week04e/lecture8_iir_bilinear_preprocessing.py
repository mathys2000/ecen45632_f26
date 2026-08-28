"""Lecture 8 demonstrations: analog prototypes, bilinear transform, and IIR preprocessing."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)


def plot_analog_prototypes() -> None:
    N = 5
    wc = 1.0
    designs = {
        "Butterworth": signal.butter(N, wc, analog=True, output="ba"),
        "Chebyshev I (1 dB ripple)": signal.cheby1(N, 1, wc, analog=True, output="ba"),
        "Chebyshev II (40 dB stop)": signal.cheby2(N, 40, 1.6 * wc, analog=True, output="ba"),
        "Elliptic (1/40 dB)": signal.ellip(N, 1, 40, wc, analog=True, output="ba"),
    }
    Om = np.logspace(-1.2, 1.1, 1500)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for name, (b, a) in designs.items():
        _, H = signal.freqs(b, a, worN=Om)
        ax.semilogx(Om, 20 * np.log10(np.maximum(np.abs(H), 1e-8)), label=name)
    ax.axvline(wc, linestyle="--", linewidth=1.1)
    ax.set_ylim(-90, 5)
    ax.set_title("Classical analog lowpass prototypes: same goal, different tradeoffs")
    ax.set_xlabel(r"Analog frequency $\Omega$ (rad/s, normalized)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "iir_analog_prototypes.png", dpi=180)
    plt.close(fig)


def plot_bilinear_mapping() -> None:
    T = 0.2
    # Representative stable analog poles.
    s = np.array([-0.5 + 2j, -1 + 1j, -2 + 0j, -3 - 2j, -0.3 - 3j])
    z = (1 + s * T / 2) / (1 - s * T / 2)

    theta = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.3, label="unit circle")
    ax.scatter(z.real, z.imag, s=55, label="mapped stable poles")
    for sk, zk in zip(s, z):
        ax.annotate(f"{sk.real:.1f}{sk.imag:+.1f}j", (zk.real, zk.imag),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Bilinear transform maps the analog LHP inside $|z|<1$")
    ax.set_xlabel("Re{$z$}")
    ax.set_ylabel("Im{$z$}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "iir_bilinear_pole_map.png", dpi=180)
    plt.close(fig)


def plot_frequency_warping() -> None:
    omega = np.linspace(0, 0.985 * np.pi, 1000)
    T = 1.0
    Omega = (2 / T) * np.tan(omega / 2)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(omega / np.pi, Omega, linewidth=2)
    ax.plot(omega / np.pi, omega / T, "--", linewidth=1.2,
            label="small-frequency linear approximation")
    ax.set_ylim(0, 20)
    ax.set_xlim(0, 1)
    ax.set_title("Bilinear transform frequency warping")
    ax.set_xlabel(r"Digital frequency $\omega/\pi$")
    ax.set_ylabel(r"Analog frequency $\Omega$ for $T=1$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "iir_frequency_warping.png", dpi=180)
    plt.close(fig)


def prewarp_example() -> tuple[float, float]:
    T = 1.0
    omega_c = 0.4 * np.pi
    Omega_c = (2 / T) * np.tan(omega_c / 2)

    # Analog Butterworth prototype at the prewarped cutoff, then bilinear transform.
    b_a, a_a = signal.butter(4, Omega_c, analog=True, output="ba")
    b_z, a_z = signal.bilinear(b_a, a_a, fs=1 / T)
    w, H = signal.freqz(b_z, a_z, worN=4096)

    # Find approximate -3 dB location.
    db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    idx = np.argmin(np.abs(db + 3.0103))
    omega_meas = w[idx]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(w / np.pi, db, linewidth=1.8)
    ax.axvline(omega_c / np.pi, linestyle="--", linewidth=1.2,
               label="desired digital cutoff")
    ax.axhline(-3.0103, linestyle=":", linewidth=1.1)
    ax.set_ylim(-80, 3)
    ax.set_xlim(0, 1)
    ax.set_title("Prewarping places the Butterworth cutoff at the desired digital frequency")
    ax.set_xlabel(r"Normalized digital frequency $\omega/\pi$")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "iir_prewarp_design.png", dpi=180)
    plt.close(fig)
    return Omega_c, omega_meas


def plot_iir_phase() -> None:
    b, a = signal.butter(5, 0.25)
    w, H = signal.freqz(b, a, worN=2048)
    phase = np.unwrap(np.angle(H))
    gd_w, gd = signal.group_delay((b, a), w=1024)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(w / np.pi, phase, linewidth=1.8)
    ax.set_title("A typical IIR lowpass has nonlinear phase")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "iir_nonlinear_phase.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(gd_w / np.pi, gd, linewidth=1.8)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, np.nanpercentile(gd[np.isfinite(gd)], 95) * 1.2)
    ax.set_title("Nonlinear phase appears as frequency-dependent group delay")
    ax.set_xlabel(r"Normalized frequency $\omega/\pi$")
    ax.set_ylabel("Group delay (samples)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "iir_group_delay.png", dpi=180)
    plt.close(fig)


def plot_filtfilt_demo() -> None:
    rng = np.random.default_rng(11)
    fs = 1000.0
    t = np.arange(0, 1.0, 1 / fs)
    # A transient plus useful 12-Hz oscillation and high-frequency interference.
    clean = np.exp(-((t - 0.42) / 0.018) ** 2) + 0.35 * np.sin(2 * np.pi * 12 * t)
    noisy = clean + 0.25 * np.sin(2 * np.pi * 150 * t) + 0.07 * rng.standard_normal(t.size)

    sos = signal.butter(4, 40, btype="low", fs=fs, output="sos")
    causal = signal.sosfilt(sos, noisy)
    zero_phase = signal.sosfiltfilt(sos, noisy)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    mask = (t > 0.30) & (t < 0.56)
    ax.plot(t[mask], noisy[mask], label="noisy input", linewidth=0.9, alpha=0.7)
    ax.plot(t[mask], clean[mask], label="reference", linewidth=1.3)
    ax.plot(t[mask], causal[mask], label="causal IIR", linewidth=1.6)
    ax.plot(t[mask], zero_phase[mask], label="forward-backward (zero phase)", linewidth=1.8)
    ax.set_title("Offline forward-backward filtering removes phase delay but is noncausal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "iir_filtfilt_transient.png", dpi=180)
    plt.close(fig)

    w, H = signal.sosfreqz(sos, worN=2048, fs=fs)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    one_pass_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))
    two_pass_db = 20 * np.log10(np.maximum(np.abs(H) ** 2, 1e-10))
    ax.plot(w, one_pass_db, label="one pass: $|H|$", linewidth=1.8)
    ax.plot(w, two_pass_db, label="forward-backward: $|H|^2$", linewidth=1.8)
    ax.set_xlim(0, 180)
    ax.set_ylim(-100, 3)
    ax.set_title("Forward-backward filtering squares the magnitude response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "iir_filtfilt_magnitude.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_analog_prototypes()
    plot_bilinear_mapping()
    plot_frequency_warping()
    Omega_c, omega_meas = prewarp_example()
    plot_iir_phase()
    plot_filtfilt_demo()
    print(f"Prewarped analog cutoff for omega_c=0.4*pi, T=1: {Omega_c:.6f} rad/s")
    print(f"Measured digital -3 dB cutoff: {omega_meas/np.pi:.4f} * pi rad/sample")


if __name__ == "__main__":
    main()

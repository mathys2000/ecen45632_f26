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

# %% [markdown]
# # Lecture 9 — Time/Frequency Responses and Group Delay
#
# Numerical demonstrations for the lecture.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

OUT = Path('/mnt/data/week5_lectures9_10/assets')
OUT.mkdir(exist_ok=True)


def savefig(name):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    return path

# Example system: H(z) = (1 - 0.8 z^-1)/(1 - 0.5 z^-1)
b = np.array([1.0, -0.8])
a = np.array([1.0, -0.5])
w, H = signal.freqz(b, a, worN=2048)
mag = np.abs(H)
phase = np.unwrap(np.angle(H))
w_gd, gd = signal.group_delay((b, a), w=2048)

# Magnitude/phase/group delay
plt.figure(figsize=(7.2, 4.4))
plt.plot(w/np.pi, 20*np.log10(np.maximum(mag, 1e-12)))
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Magnitude (dB)')
plt.title('Example Pole–Zero System: Magnitude Response')
plt.grid(True, alpha=0.3)
savefig('L9_mag.png')

plt.figure(figsize=(7.2, 4.4))
plt.plot(w/np.pi, phase)
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Unwrapped phase (rad)')
plt.title('Example Pole–Zero System: Phase Response')
plt.grid(True, alpha=0.3)
savefig('L9_phase.png')

plt.figure(figsize=(7.2, 4.4))
plt.plot(w_gd/np.pi, gd)
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Group delay (samples)')
plt.title('Frequency-Dependent Group Delay')
plt.grid(True, alpha=0.3)
savefig('L9_group_delay.png')

# Pole-zero plot and geometric vectors to one point on unit circle
z = np.roots(b)
p = np.roots(a)
omega0 = 0.55*np.pi
pt = np.exp(1j*omega0)
t = np.linspace(0, 2*np.pi, 500)
plt.figure(figsize=(5.5, 5.5))
plt.plot(np.cos(t), np.sin(t), '--', linewidth=1)
plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
plt.scatter(z.real, z.imag, marker='o', s=90, facecolors='none', linewidths=2, label='zero')
plt.scatter(p.real, p.imag, marker='x', s=90, linewidths=2, label='pole')
plt.scatter([pt.real], [pt.imag], s=50, label=r'$e^{j\omega_0}$')
for zz in z:
    plt.plot([zz.real, pt.real], [zz.imag, pt.imag])
for pp in p:
    plt.plot([pp.real, pt.real], [pp.imag, pt.imag])
plt.xlim(-1.35, 1.35); plt.ylim(-1.35, 1.35)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Real'); plt.ylabel('Imaginary')
plt.title('Geometric Evaluation on the Unit Circle')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.25)
savefig('L9_pz_geometry.png')

# All-pass system: H_ap(z) = (z^-1 - a)/(1 - a z^-1)
ap = 0.7
b_ap = np.array([-ap, 1.0])
a_ap = np.array([1.0, -ap])
w_ap, H_ap = signal.freqz(b_ap, a_ap, worN=2048)
_, gd_ap = signal.group_delay((b_ap, a_ap), w=2048)
plt.figure(figsize=(7.2, 4.4))
plt.plot(w_ap/np.pi, np.abs(H_ap))
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Magnitude')
plt.ylim(0.96, 1.04)
plt.title('First-Order All-Pass: Unit Magnitude')
plt.grid(True, alpha=0.3)
savefig('L9_allpass_mag.png')

plt.figure(figsize=(7.2, 4.4))
plt.plot(w_ap/np.pi, gd_ap)
plt.xlabel(r'Normalized frequency $\omega/\pi$')
plt.ylabel('Group delay (samples)')
plt.title('All-Pass Changes Phase / Group Delay')
plt.grid(True, alpha=0.3)
savefig('L9_allpass_gd.png')

# Waveform dispersion: pulse through causal Butterworth vs zero-phase
fs = 1000.0
tm = np.arange(0, 1.0, 1/fs)
x = np.exp(-0.5*((tm-0.35)/0.008)**2) - 0.55*np.exp(-0.5*((tm-0.37)/0.012)**2)
sos = signal.butter(5, 90, btype='low', fs=fs, output='sos')
y_causal = signal.sosfilt(sos, x)
y_zero = signal.sosfiltfilt(sos, x)
plt.figure(figsize=(7.4, 4.4))
plt.plot(tm*1000, x, label='original')
plt.plot(tm*1000, y_causal, label='causal IIR')
plt.plot(tm*1000, y_zero, label='zero-phase offline')
plt.xlim(300, 450)
plt.xlabel('Time (ms)'); plt.ylabel('Amplitude')
plt.title('Phase Distortion Changes Transient Timing and Shape')
plt.legend(); plt.grid(True, alpha=0.3)
savefig('L9_transient_dispersion.png')

print('Lecture 9 assets generated in', OUT)


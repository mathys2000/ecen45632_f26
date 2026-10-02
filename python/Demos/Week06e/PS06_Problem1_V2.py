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
# # PS06_Problem1_V2
#
# Conv1d and the sliding dot product, Version 2

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile


# %%
# Read .wav file
def wav16read(f_name):
    fs, xn16 = wavfile.read(f_name)        # read sampling rate Fs and data xn from .wav file
    return fs, xn16.astype(np.float32)/2**15        # convert to 32-bit floating point in range -1.0 to +1.0


# %%
# Write .wav file
def wav16write(f_name, fs, yn):
    yn16 = np.array(2**15*yn, np.int16)
    wavfile.write(f_name, fs, yn16)


# %%
# Load chirp signal
fs, xn = wav16read('audio/chirp_001.wav')

# %%
# FIR Lowpass Filter
Ntaps = 51
M = Ntaps - 1   # filter order
wc = np.pi/3    # cutoff frequency in radians
hlp = ss.firwin(numtaps=Ntaps, cutoff=wc/np.pi, window='hamming')  # impulse response

# %%
# Plot time and frequency response of lowpass filter
nn = np.arange(Ntaps)
ww, Hejw = ss.freqz(hlp, 1, worN=2000)
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].stem(nn, hlp, label='Time Domain')
axs[0].set_title(f'FIR LPF $h[n]\\Leftrightarrow H(e^{{j\\omega}})$, $\\omega_c/\\pi$={wc/np.pi:0.2f}, $N$={Ntaps}')
axs[0].set_ylabel('$h[n]$')
axs[0].set_xlabel('n')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww/np.pi, np.abs(Hejw), label='Frequency Domain')
axs[1].axvline(wc/np.pi, linestyle='--', color='red', label=f'$\\omega_c/\\pi$={wc/np.pi:0.2f}')
axs[1].set_ylabel('$|H(e^{{j\\omega}})|$')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww/np.pi, np.unwrap(np.angle(Hejw)), label='Frequency Domain')
axs[2].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[2].set_ylabel('$\\angle H(e^{{j\\omega}})$ [rad]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# DSP convolution to low-pass filter the chirp signal
yDSP = np.convolve(xn, hlp, mode='full')
wav16write('audio/chirp_001_yDSP.wav', fs, yDSP)
print(yDSP.size)

# %%
# Importing PyTorch modules
import torch
import torch.nn.functional as F

# %%
# Flipped filter coefficients w_torch and padded xn signal
# w[k] = hlp[M-k]:
w_torch = torch.tensor(np.array(hlp, np.float32)).view(1, 1, -1).flip(dims=[2])
# x_padded[n] = x[n-M], M is filter order of hlp[n] lowpass filter
xn_padded = F.pad(torch.tensor(xn).view(1, 1, -1), pad=(M, M))

# %%
# Execute (full) convolution in PyTorch
y_torch = F.conv1d(xn_padded, w_torch).squeeze()
wav16write('audio/chirp_001_ytorch.wav', fs, y_torch)
print(y_torch.shape)

# %%
# Compare DSP convolution and conv1d result
print(f'max error: {np.max(np.abs(yDSP - y_torch.tolist()))}')

# %%
# Original signal
ipd.Audio(xn, rate=fs)

# %%
# DSP filtered signal
ipd.Audio(yDSP, rate=fs)

# %%
# conv1d filtered signal
ipd.Audio(y_torch, rate=fs)

# %%
y_torch.shape

# %%

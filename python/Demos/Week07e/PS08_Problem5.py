# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ecen45632 [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-ecen45632
# ---

# %% [markdown]
# # PS08 Problem5
#
# Problem 5: Zero Padding and Frequency Plot Detail

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
fs = 2048
f0 = 403.3
Lx = 128     # number of samples
N = [128, 512, 4096]

# %%
# Waveform
nn = np.arange(128)    # sample axis
xn = np.exp(2j*np.pi*f0*nn/fs)
X128k = np.fft.fft(xn, n=N[0])
ff128 = np.fft.fftfreq(N[0], 1/fs)
X512k = np.fft.fft(xn, n=N[1])
ff512 = np.fft.fftfreq(N[1], 1/fs)
X4096k = np.fft.fft(xn, n=N[2])
ff4096 = np.fft.fftfreq(N[2], 1/fs)
            

# %%
# Plot magnitude of frequency response
fig, axs = plt.subplots(1, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs.plot(ff4096, np.abs(X4096k)/128, '.-', label=f'N={N[2]}')
axs.plot(ff512, np.abs(X512k)/128, '.-', label=f'N={N[1]}')
axs.plot(ff128, np.abs(X128k)/128, '.-', label=f'N={N[0]}')
axs.grid(alpha=0.5)
axs.set_xlim([350, 450])
axs.legend()
plt.show()

# %%
print(np.mod(f0, 16))
print(f0/0.5)

# %%
25*16

# %%
x8 = [1,2,3,4,0,0,0,0]
X8 = np.fft.fft(x8)
Y8 = X8*np.exp(-2j*np.pi*6*np.arange(8)/8)
print(Y8)

# %%
y8 = np.fft.ifft(Y8)
print(y8.real)

# %%

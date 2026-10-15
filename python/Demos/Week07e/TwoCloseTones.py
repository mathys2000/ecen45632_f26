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
# # Two Close Tones
#
# Two close tones, observation duration and intermittent tones

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
fs = 2048    # sampling rate, samples/sec
L1, L2 = 128, 1024    # signal length in samples
f0, f1 = 400, 406     # frequencies in Hz
N1, N2 = 128, 4096    # FFT blocklengths

# %%
# Waveforms
tt1 = (np.arange(L1)-L1/2)/fs    # time axis 1, seconds
tt2 = (np.arange(L2)-L2/2)/fs    # time axis 2, seconds
x1 = np.exp(2j*np.pi*f0*tt1) + np.exp(2j*np.pi*f1*tt1)  # signal 1, length N1
x2 = np.exp(2j*np.pi*f0*tt2) + np.exp(2j*np.pi*f1*tt2)  # signal 1, length N2


# %%
# Waveforms for testing
x1a = np.exp(2j*np.pi*f0*tt1)
x1b = np.exp(2j*np.pi*f1*tt1)


# %%
fig, axs = plt.subplots(2, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
axs[0].plot(tt1, x1.real, '-', label=f'L1={L1}')
axs[0].set_title(f'Sum of two Sinusoids, f0={f0} Hz, f1={f1} Hz, fs={fs} Hz')
axs[0].grid(alpha=0.5)
axs[0].set_ylabel('x1(t)')
axs[0].set_xlim([-0.3, 0.3])
axs[0].legend()
axs[1].plot(tt1, x1.real, '-')
axs[1].plot(tt2, x2.real, '-', label=f'L2={L2}')
axs[1].grid(alpha=0.5)
axs[1].set_ylabel('x2(t)')
axs[1].set_xlabel('t [sec]')
axs[1].set_xlim([-0.3, 0.3])
axs[1].legend()
plt.show()

# %%
# DFT/FFTs
X1_N1 = np.fft.fft(x1, n=N1)/L1
ff1 = np.fft.fftfreq(N1, 1/fs)
X1_N2 = np.fft.fft(x1, n=N2)/L1
ff2 = np.fft.fftfreq(N2, 1/fs)
X2_N2 = np.fft.fft(x2, n=N2)/L2


# %%
# Plot magnitude of frequency response
fig, axs = plt.subplots(1, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
axs.plot(ff1, np.abs(X1_N1), '.-', label=f'L1={L1}, N1={N1}')
axs.plot(ff2, np.abs(X1_N2), '.-', label=f'L1={L1}, N2={N2}')
axs.plot(ff2, np.abs(X2_N2), '.-', label=f'L2={L2}, N2={N2}')
axs.set_title(f'DFT of Sum of two Sinusoids, f0={f0} Hz, f1={f1} Hz, fs={fs} Hz')
axs.grid(alpha=0.5)
axs.set_ylabel('|X(f)|')
axs.set_xlabel('f [Hz]')
axs.set_xlim([350, 450])
axs.legend()
plt.show()

# %%

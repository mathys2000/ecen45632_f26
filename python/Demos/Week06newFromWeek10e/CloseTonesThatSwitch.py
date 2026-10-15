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
# # Close Tones That Switch
#
# A signal contains 500 Hz for one half and 550 Hz for the other half of 100 ms

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
fs = 4000
tlen = 0.1
f0, f1 = 500, 550


# %%
# Waveforms
tt = np.arange(np.round(tlen*fs))/fs
x1t = np.cos(2*np.pi*f0*tt)
ix = np.where(tt>=tlen/2)[0]
x1t[ix] = np.cos(2*np.pi*f1*tt[ix])
x2t = np.cos(2*np.pi*f1*tt)
x2t[ix] = np.cos(2*np.pi*f0*tt[ix])
Lx = x1t.size                 

# %%
# DFT/FFTs
X1 = np.fft.fft(x1t)/Lx
X2 = np.fft.fft(x2t)/Lx
ff = np.fft.fftfreq(Lx, 1/fs)


# %%
# Plot magnitude of frequency response
fig, axs = plt.subplots(2, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
axs[0].plot(ff, np.abs(X1), '-', label=f'X1(f)')
axs[0].set_title(f'Close Tones that Switch, f0={f0} Hz, f1={f1} Hz, fs={fs} Hz')
axs[0].grid(alpha=0.5)
axs[0].set_ylabel('|X1(f)|')
#axs[0].set_xlim([400, 600])
axs[0].legend()
axs[1].plot(ff, np.abs(X2), '-', label=f'X2(f)')
axs[1].grid(alpha=0.5)
axs[1].set_ylabel('|X2(f)|')
axs[1].set_xlabel('f [Hz]')
axs[1].legend()
plt.show()

# %%
# Spectrogram
Ba, Bb = 10, 200
ff2a, tt2a, Za = ss.stft(x1t, fs=fs, window='hann', nperseg=Ba, noverlap=Ba/2, nfft=512, boundary=None)
ff2b, tt2b, Zb = ss.stft(x1t, fs=fs, window='hann', nperseg=Bb, noverlap=Bb/2, nfft=512, boundary=None)
#ff2a, tt2a, Za = ss.stft(x1t, fs=fs, window='boxcar', nperseg=Ba, noverlap=Ba/2, nfft=512, boundary=None)
#ff2b, tt2b, Zb = ss.stft(x1t, fs=fs, window='boxcar', nperseg=Bb, noverlap=Bb/2, nfft=512, boundary=None)
fig, axs = plt.subplots(2, 1, figsize=(8,5))
fig.canvas.toolbar_position = 'top'
im0 = axs[0].pcolormesh(tt2a, ff2a, 20*np.log10(np.abs(Za)+1e-6), shading='auto', label=f'B={Ba}')
axs[0].set_title(f'STFT / spectrogram view, B={Ba}')
axs[0].set_ylim(0,1800)
axs[0].set_ylabel('Hz')
cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label('dB')
im1 = axs[1].pcolormesh(tt2b, ff2b, 20*np.log10(np.abs(Zb)+1e-6), shading='auto', label=f'B={Bb}')
axs[1].set_title(f'STFT / spectrogram view, B={Bb}')
axs[1].set_ylim(0,1800)
axs[1].set_ylabel('Hz')
axs[1].set_xlabel('time (s)')
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label('dB')
plt.tight_layout()
plt.show()



# %%

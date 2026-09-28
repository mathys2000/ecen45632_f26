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
# # Chirp 001
#
# Generate linear chirp signal

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile


# %%
# Parameters
Fs = 8000       # sampling rate
tlen = 2        # duration in sec
f0, f1 = 100, 3900    # start and end frequency
tt = np.arange(np.round(tlen*Fs))/Fs    # time axis

# %%
# Generate waveform
fit = np.zeros(tt.size)    # instantaneous frequency
ix = np.where(tt<tlen/2)[0]
fit[ix] = f0 + 2*(f1-f0)/tlen*tt[ix]
ix = np.where(tt>=tlen/2)[0]
fit[ix] = f0 + 2*(f1-f0) - 2*(f1-f0)/tlen*tt[ix]
psi = 2*np.pi*np.cumsum(fit)/Fs     # phase of chirp signal
xt = np.cos(psi)      # chirp signal

# %%
fig, axs = plt.subplots(2, 1, figsize=(7,6))
axs[0].plot(tt, fit)
axs[0].grid()
axs[1].plot(tt, xt)
axs[1].grid()
plt.show()

# %%
# Play signal
ipd.Audio(xt, rate=Fs)

# %%
# Write 16-bit .wav file
xt16 = np.array(2**15*xt, np.int16)
wavfile.write('audio/chirp_001.wav', Fs, xt16)


# %%

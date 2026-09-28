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
# # PS06_Problem1
#
# Conv1d and the sliding dot product

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile

# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
Nx = 51    # filter/window length
M2 = (Nx-1)/2     # offset to make hlp causal
nn = np.arange(Nx)     # index axis
wc = np.pi/3     # cutoff frequency
Nw = 5000   # number of frequency points to evaluate

# %%
# Ideal LPF, truncated
hlp = wc/np.pi*np.sinc(wc/np.pi*(nn-M2))

# %%
# Window function
W = ss.get_window('hamm', Nx, fftbins=False)
hlpW = hlp*W   # windowed filter coefficients
ww, HejwW = ss.freqz(hlpW, 1, Nw)


# %%
# Read audio signal
Fs, xt16 = wavfile.read('audio/chirp_001.wav')
xt = xt16.astype(np.float32)/2**15
tt = np.arange(xt.size)/Fs    # time axis

# %%
# Filter audio signal
#yt = ss.lfilter(hlpW, 1, xt)
yt = np.convolve(xt, hlpW, mode='full')

# %%
# Original signal
print(xt.size)
ipd.Audio(xt, rate=Fs)

# %%
# Filtered signal
print(yt.size)
ipd.Audio(yt, rate=Fs)

# %%
# Write filtered 16-bit .wav file
yt16 = np.array(2**15*yt, np.int16)
wavfile.write('audio/chirp_001_filtered.wav', Fs, yt16)


# %%
# PyTorch conv1d implementation
import torch
import torch.nn.functional as F

# %%
# Flipped filter coefficients and padded signal
hW = torch.tensor(np.array(hlpW, np.float32))
hlpW_tensor = hW.view(1, 1, -1)
hlpW_flipped = torch.flip(hlpW_tensor, dims=[-1])
x = torch.tensor(xt)
x_tensor = x.view(1, 1, -1)
x_padded = F.pad(x_tensor, pad=(Nx-1, Nx-1, 0, 0), mode='constant', value=0)

# %%
print(hlpW_flipped.dtype)
print(x_padded.dtype)

# %%
# conv1d computation (correlation)
y_out = F.conv1d(x_padded, hlpW_flipped)
y_conv1d = y_out.squeeze()

# %%
# conv1d filtered signal
print(y_conv1d.numpy().size)
ipd.Audio(y_conv1d.numpy(), rate=Fs)

# %%
# Compare DSP convolution and conv1d result
print(f'max error: {np.max(np.abs(yt - y_conv1d.numpy()))}')

# %%

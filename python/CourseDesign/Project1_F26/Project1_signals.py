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
# # Project1_signals
#
# Generate Project 1 waveforms using fixtures() in common.py

# %%
import numpy as np
from pathlib import Path
from common import fixtures, read_wav
import IPython.display as ipd


# %%
FS = 16000    # default sampling rate

# %%
a = fixtures(Path("signals"))


# %%
print(a['melody'].shape)
print(a['jammer'].shape)
print(a['mixture'].shape)
print(a['probe'].shape)

# %%
ipd.Audio(a['melody'], rate=FS)

# %%
ipd.Audio(a['jammer'], rate=FS)

# %%
ipd.Audio(a['mixture'], rate=FS)

# %%
ipd.Audio(a['probe'], rate=FS)

# %%
x1 = read_wav('signals/leakage.wav')
x2 = read_wav('signals/qpsk_iq.wav')

# %%
ipd.Audio('signals/leakage.wav', rate=FS)

# %%
ipd.Audio('signals/qpsk_iq.wav', rate=FS)

# %%
with np.load('signals/truth.npz') as data:
    # 1. See what keys (array names) are inside the file
    print("Available arrays:", data.files)
    my_data_dict = dict(data)
 

# %%
print(my_data_dict['bits'].shape)
print(my_data_dict['bits'])

# %%

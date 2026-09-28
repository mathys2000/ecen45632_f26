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
# # WavFile_ReadWrite
#
# Jupyter notebook for reading and writing 16-bit .wav files

# %%
import numpy as np
import IPython.display as ipd
from scipy.io import wavfile

# %%
# Read audio signal
Fs, xn16 = wavfile.read('chirp_001.wav')   # read sampling rate Fs and data xn from .wav file
xn = xn16.astype(np.float32)/2**15         # convert to 32-bit floating point in range -1.0 to +1.0

# %%
# Play xt signal
ipd.Audio(xn, rate=Fs)

# %%
# Modify xn signal
yn = 0.5*xn    # change amplitude

# %%
# Write modified yn signal to 16-bit .wav file
yn16 = np.array(2**15*yn, np.int16)
wavfile.write('chirp_001_modified.wav', Fs, yn16)

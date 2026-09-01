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
# # PS 2, Problem 3
#
# Use scipy.signal.freqz to check your reasoning

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# %%
# Compute frequency response of H(ejw)
b = [1, -1]     # numerator of H(ejw)
a = [1, -0.8]   # denominator of H(ejw)
[w, Hejw] = ss.freqz(b, a)    # frequency response

# %%
fig = plt.figure(figsize=(9, 3))
plt.plot(w/np.pi, np.abs(Hejw))
plt.xlabel(r'$\omega/\pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.title(r'$H(e^{j\omega})$ magnitude from 0 to $\pi$')
plt.grid(True, alpha=0.5)
plt.show()


# %%

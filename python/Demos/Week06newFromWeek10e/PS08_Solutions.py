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
# # PS08 Solutions
#
# Problem Set 8 (DFT/FFTs) Solutions

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# x, h sequences
x = np.array([1, 2, 3, 4])
h = np.array([1, -1, 1])

# %%
# N1 = 4 FFT convolution
N1 = 4
y4 = np.fft.ifft(np.fft.fft(x, n=N1)*np.fft.fft(h, n=N1))
print(y4.real)

# %%
# N2 = 5 FFT convolution
N2 = 5
y5 = np.fft.ifft(np.fft.fft(x, n=N2)*np.fft.fft(h, n=N2))
print(y5.real)

# %%
# N3 = 6 FFT convolution, minimum N that results in full linear convolution
N3 = 6
y6 = np.fft.ifft(np.fft.fft(x, n=N3)*np.fft.fft(h, n=N3))
print(y6.real)

# %%
# N4 = 8 FFT convolution, full length linear convolution using power of 2 FFT
N4 = 8
y8 = np.fft.ifft(np.fft.fft(x, n=N4)*np.fft.fft(h, n=N4))
y8z = np.zeros(y8.size)
ix = np.where(np.abs(y8.real)>1e-14)[0]
y8z[ix] = y8[ix].real
print(y8z)

# %%
# ----------- optimized code -----------------------------------------------------

# %%
# N1 = 4 FFT convolution
N1 = 4
y4 = np.fft.irfft(np.fft.rfft(x, n=N1) * np.fft.rfft(h, n=N1), n=N1)
print(y4)

# %%
# N2 = 5 FFT convolution
N2 = 5
y5 = np.fft.irfft(np.fft.rfft(x, n=N2) * np.fft.rfft(h, n=N2), n=N2)
print(y5)

# %%
# N3 = 6 FFT convolution, minimum N that results in full linear convolution
N3 = 6
y6 = np.fft.irfft(np.fft.rfft(x, n=N3) * np.fft.rfft(h, n=N3), n=N3)
print(y6)

# %%
# N4 = 8 FFT convolution, full length linear convolution using power of 2 FFT
N4 = 8
y8 = np.fft.irfft(np.fft.rfft(x, n=N4) * np.fft.rfft(h, n=N4), n=N4)
y8[np.abs(y8) <= 1e-14] = 0.0   # zero out round-off noise
print(y8)

# %%

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
import scipy.signal as ss
import matplotlib.pyplot as plt

# %%
# ---------------- Problem 3 ----------------------------------------------------

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
# ------------- Problem 4 --------------------------------------------------

# %%
def mag_dB(X, llim, norm=True):
    """
    Computes normalized magnitude of X in dB with lower limit llim dB
    Returns X_dB
    """
    mag = np.abs(X)
    if norm:
        mag = np.maximum(mag / mag.max(), 10 ** (llim / 20))
    else:
        mag = np.maximum(mag, 10 ** (llim / 20))
    X_dB = 20 * np.log10(mag)
    return X_dB



# %%
# Parameters
fs = 1024
f1, f2 = 128, 136
Lx = 64
d_floor = -60   # display floor in dB


# %%
# Generate sequences in time and frequency
nn = np.arange(Lx)
x1n = np.exp(2j*np.pi*f1*nn/fs)   # x[n] with first frequency
x2n = np.exp(2j*np.pi*f2*nn/fs)   # x[n] with second frequency
ww, X1ejw = ss.freqz(x1n, 1)    # DTFT of x1n
ww, X2ejw = ss.freqz(x2n, 1)    # DTFT of x2n
X1ejw_dB = mag_dB(X1ejw, d_floor)
X2ejw_dB = mag_dB(X2ejw, d_floor)
X1k = np.fft.fft(x1n)      # DFT of x1n
X2k = np.fft.fft(x2n)      # DFT of x2n
X1k_dB = mag_dB(X1k, d_floor)
X2k_dB = mag_dB(X2k, d_floor)
ff = fs*np.fft.fftfreq(Lx)      # DFT frequencies in Hz 
ix_pos = np.where(ff>=0)[0]

# %%
# Plot DTFT and DFT
plt.figure(3, figsize=(9, 4))
plt.plot(ww*fs/(2*np.pi), X1ejw_dB, '--', linewidth=0.7, label=f'f={f1} Hz, DTFT')
plt.plot(ww*fs/(2*np.pi), X2ejw_dB, '-.', linewidth=0.7, label=f'f={f2} Hz, DTFT')
plt.plot(ff[ix_pos], X1k_dB[ix_pos], 'x', label=f'f={f1} Hz, DFT')
plt.plot(ff[ix_pos], X2k_dB[ix_pos]-4, 'o', label=f'f={f2} Hz, DFT')
plt.title(f'DTFT and DFT Magnitudes, f1={f1} Hz, f2={f2} Hz, Lx={Lx}, fs={fs} samples/sec')
plt.ylabel('dB normalized')
plt.xlabel('Frequency [Hz]')
plt.xlim([0, 250])
plt.grid()
plt.legend()
plt.show()

# %%
# Hann window function
win = ss.get_window('hann', Lx, fftbins=True)   # periodic (vs symmetric) window for DFT/FFT
x2n_win = x2n*win    # windowed x2n sequence
ww, X2ejw_win = ss.freqz(x2n_win, 1)    # DTFT of x2n_win
X2ejw_win_dB = mag_dB(X2ejw_win, d_floor)


# %%
# Plot DTFT of rectangular and Hann windowed x2n (f2=136 Hz)
plt.figure(7, figsize=(9, 4))
plt.plot(ww*fs/(2*np.pi), X2ejw_dB, label=f'f={f1} Hz, Rectangular')
plt.plot(ww*fs/(2*np.pi), X2ejw_win_dB, label=f'f={f1} Hz, Periodic Hann')
plt.title(f'Main Lobe Width and Leakage for Different Windows, f2={f2} Hz, Lx={Lx}')
plt.grid()

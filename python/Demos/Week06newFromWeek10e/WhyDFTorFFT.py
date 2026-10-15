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
# # Why DFT or FFT
#
# DTFT and DFT/FFT of some audio signals

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
# Read .wav file
def wav16read(f_name):
    fs, xn16 = wavfile.read(f_name)        # read sampling rate Fs and data xn from .wav file
    return fs, xn16.astype(np.float32)/2**15        # convert to 32-bit floating point in range -1.0 to +1.0


# %%
# Write .wav file
def wav16write(f_name, fs, yn):
    yn16 = np.array(2**15*yn, np.int16)
    wavfile.write(f_name, fs, yn16)


# %%
# Signal x1[n]
Nx1 = 41    # filter/window length
wc1 = np.pi/3     # cutoff frequency
x1n = ss.firwin(Nx1, wc1/np.pi, window='hamming')
nn1 = np.arange(x1n.size)     # index axis
Nw1 = 5000   # number of frequency points to evaluate
ww1, X1ejw = ss.freqz(x1n, 1, Nw1)   # compute DTFT using scipy.signal.freqz


# %%
# Plot signal 1
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].stem(nn1, x1n, label='Time Domain')
axs[0].set_title(f'Signal $x_1[n]\\Leftrightarrow X_1(e^{{j\\omega}})$, $\\omega_c/\\pi$={wc1/np.pi:0.2f}')
axs[0].set_ylabel('$x[n]$')
axs[0].set_xlabel('n')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww1/np.pi, np.abs(X1ejw), label='Frequency Domain')
axs[1].axvline(wc1/np.pi, linestyle='--', color='red', label=f'$\\omega_c/\\pi$={wc1/np.pi:0.2f}')
axs[1].set_ylabel('$|X(e^{{j\\omega}})|$')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww1/np.pi, np.unwrap(np.angle(X1ejw)), label='Frequency Domain')
axs[2].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[2].set_ylabel('$\\angle X(e^{{j\\omega}})$ [rad]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# Signal x2[n]
Nx2 = 1000
nn2 = np.arange(1*1000)
x2n = np.exp(-0.5*((nn2/Nx2-0.35)/0.008)**2) - 0.55*np.exp(-0.5*((nn2/Nx2-0.37)/0.012)**2)
#x2n = x2n + np.exp(-0.5*((nn2/Nx2-0.7)/0.008)**2) - 0.55*np.exp(-0.5*((nn2/Nx2-0.72)/0.012)**2)
#x2n = np.exp(-0.5*((nn2/Nx2-0.35)/0.002)**2) - 0.55*np.exp(-0.5*((nn2/Nx2-0.355)/0.003)**2)
Nw2 = 5000   # number of frequency points to evaluate
ww2, X2ejw = ss.freqz(x2n, 1, Nw2)   # compute DTFT using scipy.signal.freqz


# %%
# Plot signal 2
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(nn2, x2n, label='Time Domain')
axs[0].set_title(f'Signal $x_2[n]\\Leftrightarrow X_2(e^{{j\\omega}})$')
axs[0].set_ylabel('$x[n]$')
axs[0].set_xlabel('n')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww2/np.pi, np.abs(X2ejw), label='Frequency Domain')
axs[1].set_ylabel('$|X(e^{{j\\omega}})|$')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww2/np.pi, np.unwrap(np.angle(X2ejw)), label='Frequency Domain')
axs[2].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[2].set_ylabel('$\\angle X(e^{{j\\omega}})$ [rad]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# Signal x3[n]
fs3, x3n = wav16read('audio/chirp_001.wav')
nn3 = np.arange(x3n.size)     # index axis
Nw3 = 5000   # number of frequency points to evaluate
ww3, X3ejw = ss.freqz(x3n, 1, Nw3)   # compute DTFT using scipy.signal.freqz


# %%
# Play x3n signal
ipd.Audio(x3n, rate=fs3)

# %%
# Plot signal 3
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(nn3, x3n, label='Time Domain')
axs[0].set_title(f'Signal $x_3[n]\\Leftrightarrow X_3(e^{{j\\omega}})$, fs={fs3} Hz')
axs[0].set_ylabel('$x[n]$')
axs[0].set_xlabel('n')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww3/np.pi, np.abs(X3ejw), label='Frequency Domain')
axs[1].set_ylabel('$|X(e^{{j\\omega}})|$')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww3/np.pi, np.unwrap(np.angle(X3ejw)), label='Frequency Domain')
axs[2].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[2].set_ylabel('$\\angle X(e^{{j\\omega}})$ [rad]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# Signal x4[n]
fs4, x4n = wav16read('audio/Zest_8000_mono.wav')
nn4 = np.arange(x4n.size)     # index axis
Nw4 = 5000   # number of frequency points to evaluate
ww4, X4ejw = ss.freqz(x4n, 1, Nw4)   # compute DTFT using scipy.signal.freqz


# %%
# Play x4n signal
ipd.Audio(x4n, rate=fs4)

# %%
# Plot signal 4
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(nn4, x4n, label='Time Domain')
axs[0].set_title(f'Signal $x_4[n]\\Leftrightarrow X_4(e^{{j\\omega}})$, fs={fs4} Hz')
axs[0].set_ylabel('$x[n]$')
axs[0].set_xlabel('n')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww4/np.pi, np.abs(X4ejw), label='Frequency Domain')
axs[1].set_ylabel('$|X(e^{{j\\omega}})|$')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww4/np.pi, np.unwrap(np.angle(X4ejw)), label='Frequency Domain')
axs[2].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[2].set_ylabel('$\\angle X(e^{{j\\omega}})$ [rad]')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.tight_layout()
plt.show()

# %%
# Spectrogram
f4, tt4, Z4 = ss.stft(x4n, fs=fs4, window='hann', nperseg=256, noverlap=192, nfft=512, boundary=None)
fig = plt.figure(figsize=(9,3))
fig.canvas.toolbar_position = 'top'
plt.pcolormesh(tt4, f4, 20*np.log10(np.abs(Z4)+1e-6), shading='auto')
plt.ylim(0,4000)
plt.xlabel('time (s)')
plt.ylabel('Hz')
plt.title('STFT / spectrogram view')
plt.colorbar(label='dB');
plt.tight_layout()
plt.show()


# %%
# Play x4n signal
ipd.Audio(x4n, rate=fs4)

# %%
# Chirp signal x5t
fs5 = 4000     # sampling rate
tlen5 = 1.2    # duration in seconds
#tlen5 = 12    # duration in seconds
tt5 = np.arange(np.round(tlen5*fs5))/fs5   # time axis
x5t = ss.chirp(tt5, f0=150, f1=1500, t1=tt5[-1])     # linear chirp signal
x5t = x5t + 0.1*np.random.default_rng(3).normal(size=len(tt5))  # add some Gaussian noise
ff5 = np.fft.rfftfreq(len(x5t), 1/fs5);   # frequency axis
PSD5 = np.abs(np.fft.rfft(x5t*np.hanning(len(x5t))))**2     # power spectral density
#PSD5 = np.abs(np.fft.rfft(x5t*np.ones(len(x5t))))**2


# %%
print(ff5[0], ff5[-1])
print(ff5[:20])

# %%
# Play x4n signal
ipd.Audio(x5t, rate=fs5)

# %%
fig = plt.figure(figsize=(8,3))
fig.canvas.toolbar_position = 'top'
plt.semilogy(ff5, PSD5+1e-10)
#plt.plot(ff5, PSD5+1e-10)
plt.grid(alpha=0.5)
plt.xlim(0,1800)
plt.title('Global spectrum')
plt.xlabel('Hz')
plt.tight_layout()
plt.show()


# %%
# Spectrogram
f5, tt5, Z5 = ss.stft(x5t, fs=fs5, window='hann', nperseg=256, noverlap=192, nfft=512, boundary=None)
fig = plt.figure(figsize=(8,3))
fig.canvas.toolbar_position = 'top'
plt.pcolormesh(tt5, f5, 20*np.log10(np.abs(Z5)+1e-6), shading='auto')
plt.ylim(0,1800)
plt.xlabel('time (s)')
plt.ylabel('Hz')
plt.title('STFT / spectrogram view')
plt.colorbar(label='dB');
plt.tight_layout()
plt.show()


# %%

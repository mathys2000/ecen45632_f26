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
# # Overlap-Save
#
# Overlap-Save (OLS) method for block processing

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile
rng = np.random.default_rng()

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
# Circular convolution using DFT/FFT

def circular_convolution_fft(signal, kernel):
    # Both signals must be of equal length for standard circular convolution
    # Take FFT of both, multiply them, and take the inverse FFT
    fft_signal = np.fft.fft(signal)
    fft_kernel = np.fft.fft(kernel)
    
    # Return only the real part (imaginary parts should be negligible noise)
    return np.real(np.fft.ifft(fft_signal * fft_kernel))

# Example usage:
#x = np.array([1, 2, 3, 4])
#h = np.array([1, -1, 1, -1])

#result = circular_convolution_fft(x, h)
#print("FFT Method Result:", np.round(result, 2))
# Output: [0. 0. 0. 0.]



# %%
# Generate test signal
fs = 8000
tlen = 0.5
tt = np.arange(np.round(tlen*fs))/fs   # time axis
f_sig = 500
st = np.cos(2*np.pi*f_sig*tt - np.pi/2)   # sinusoidal signal
std_dev = 0.2
nt = rng.normal(loc=0, scale=std_dev, size=st.size)   # Gaussian noise
snt = st + nt    # signal plus noise

# %%
t0, t1 = 0, 0.1
fig, axs = plt.subplots(3, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(tt, st, label='Signal')
axs[0].set_title(f'Test Signal with Noise, std_dev={std_dev}, fs={fs} Hz')
axs[0].set_ylabel('$x(t)$')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[0].set_xlim([t0, t1])
axs[1].plot(tt, nt, label='Noise')
axs[1].set_ylabel('$n(t)$')
axs[1].grid(alpha=0.5)
#axs[1].set_xlabel('t')
axs[1].legend()
axs[1].set_xlim([t0, t1])
axs[2].plot(tt, snt, label='Signal + Noise')
axs[2].set_ylabel('$s(t) + n(t)$')
axs[2].grid(alpha=0.5)
axs[2].set_xlabel('$t$ [sec]')
axs[2].legend()
axs[2].set_xlim([t0, t1])
plt.show();

# %%
# Define LPF
Lh = 21    # filter length
M = Lh - 1   # filter order
wc = 1.5*np.pi*2*f_sig/fs    # cutoff frequency in radians
hlp = ss.firwin(numtaps=Lh, cutoff=wc/np.pi, window='hamming')  # impulse response
ww, Hlpejw = ss.freqz(hlp, 1, 1000)   # frequency response of filter

# %%
fig, axs = plt.subplots(1, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
axs.axvline(f_sig, linestyle='--', color='magenta')
axs.plot(fs/2*ww/np.pi, np.abs(Hlpejw))
axs.set_title(f'LPF at $f_c$={wc/np.pi*fs/2} Hz, $L_h$={Lh}')
axs.grid(alpha=0.5)
axs.set_ylabel('$|H(e^{{j\\omega}})|$')
axs.set_xlabel('$f=\\frac{{\\omega}}{{2\\pi}}f_s$ [Hz]')
plt.tight_layout()
plt.show()

# %%
# Overlap and save method parameters
Nfft = int(2**np.round(np.log2(4*Lh)))   # nearest power of 2
B = Nfft-Lh+1
print(Nfft, B, Lh)

# %%
# Overlap and save implementation
NB = int(np.ceil(snt.size/B))    # number of blocks
Lpad = B*NB - snt.size
xn = np.hstack((np.zeros(Lh-1), snt, np.zeros(Lpad)))  # input sequence
hlp_padded = np.hstack((hlp, np.zeros(Nfft-hlp.size)))   # hlp zero-padded
Y = np.zeros((NB, B))   # output array
x_ptr = 0   # pointer into xn array
for i in range(NB):
    y_aux = circular_convolution_fft(xn[i*B:(i+1)*B+Lh-1], hlp_padded)
    Y[i] = y_aux[Lh-1:]
yn_OLS = np.reshape(Y, (-1, 1))

# %%
# np.convolve implementation
yn_conv = np.convolve(snt, hlp)
print(yn_OLS.size)
print(yn_conv.size)

# %%
t0, t1 = 0, 0.05
fig, axs = plt.subplots(2, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
axs[0].plot(tt, snt[:tt.size], label='Signal + Noise')
axs[0].set_title(f'Low-Pass Filtered Test Signal, $f_c$={wc/np.pi*fs/2} Hz, N={Nfft}, B={B}, $L_h$={Lh}')
axs[0].grid(alpha=0.5)
axs[0].set_ylabel('$s(t)+n(t)$')
axs[0].set_xlim([t0, t1])
axs[0].legend()
axs[1].plot(tt, yn_conv[:tt.size], label='np.convolve')
axs[1].plot(tt, yn_OLS[:tt.size], linestyle='--', label='OLS')
axs[1].grid(alpha=0.5)
axs[1].set_ylabel('$y(t)$')
axs[1].set_xlabel('$t$ [sec]')
axs[1].set_xlim([t0, t1])
axs[1].legend()
plt.show()

# %%
np.shape(yn_OLS)

# %%

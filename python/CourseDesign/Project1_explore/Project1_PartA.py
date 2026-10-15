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
# # Project1_PartA
#
# Time Frequency Analysis and Windows

# %% [markdown]
# ## A1 Window duration and uncertainty
#
# Implement analysis. On `tf_probe.wav` compare Hann windows of B = 256, 1024 and 4096 samples, each with 75 percent overlap and FFT length equal to B. Plot three spectrograms with identical frequency limits and dB ranges. Report frame duration, bin spacing and hop time. Inspect the two close tones away from transients and quantify the observed transient width using an explicitly stated threshold or energy criterion.
#
# $$T_w=B/f_s, \qquad \Delta f_{bin}=f_s/N, \qquad T_H=H/f_s$$
#
# Explain why long windows help distinguish close steady tones but smear short events and broaden the apparent sweep of a chirp within each frame. FFT bin spacing is not the true resolving power. At B=1024, repeat with N=4096 while keeping L and H fixed. Explain what zero padding changes and what it cannot change. Use the time frequency uncertainty idea qualitatively; a continuous time Gaussian bound should not be applied blindly to finite periodic DFT samples.

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
Fs = 16000    # default sampling rate


# %%
# Read .wav file
def wav16read(f_name):
    fs, xn16 = wavfile.read(f_name)        # read sampling rate Fs and data xn from .wav file
    return fs, xn16.astype(np.float32)/2**15        # convert to 32-bit floating point in range -1.0 to +1.0
    


# %%
def _layout(Lx, B, hop):
    """
    Lx: length of x (total data length)
    B: frame (block) length
    hop: hop size (B/4 for 75% overlap)
    Front padding: pad_f = B - hop zeros in front so early samples get full overlap
    Back padding: pad_b = total-Lx-pad_f
    Enough padding at the end so that every retained sample is covered by the same number of frames
    """
    pad_f = B - hop    # front padding
    M = int(np.ceil((Lx + pad_f - hop)/hop) + 1)    # number of frames (blocks)
    total = (M - 1)*hop + B     # total length with padding (pad_f and pad_b)
    return pad_f, total-Lx-pad_f, M


# %%
def analysis(x, B=1024, hop=None, N=None, window="hann", fs=Fs):
    """
    B: frame (data block) length
    hop: sample advance between frames
    N: FFT length (default is B)
    fs: sampling rate
    Use "boxcar" for rectangular window
    Returns (X, meta)
    X has shape (frames, N//2 + 1) for real x
    X has shape (frames, N) for complex x
    """
    x = np.asarray(x)    # convert x to an array if necessary
    cplx = np.iscomplexobj(x)   # check if x is complex-valued
    hop = B//4 if hop is None else int(hop)     # hop length
    N = B if N is None else int(N)     # FFT blocksize N
    if N < B:
        raise ValueError("N must be >= B")
    if not 1 <= hop <= B:
        raise ValueError("hop must be in range [1, B]")
    w = ss.get_window(window, B, fftbins=True)   # periodic (vs symmetric) window
    pad_f, pad_b, M = _layout(x.size, B, hop)    # padding for x, M is number of frames
    xp = np.concatenate([np.zeros(pad_f, x.dtype), x, np.zeros(pad_b, x.dtype)])
    idx = hop*np.arange(M)[:, None] + np.arange(B)[None, :]  # 2D array of indexes into xp
    frames = xp[idx] * w    # windowed frames (M x B array)
    if cplx:
        X = np.fft.fft(frames, n=N, axis=1)
    else:
        X = np.fft.rfft(frames, n=N, axis=1)
    meta = dict(B=B, hop=hop, N=N, window=window, w=w, Lx=x.size, pad_front=pad_f,
                pad_back=pad_b, M=M, complex=cplx, fs=fs)
    return X, meta
    


# %%
def frame_times(meta):
    """
    Frame center times in seconds, with the leading padding offset removed.
    """
    return (meta["hop"]*np.arange(meta["M"]) + meta["B"]/2 - meta["pad_front"])/meta["fs"]
    


# %%
def frame_freqs(meta):
    """
    FFT bin frequencies in Hz: One-sided for real x, two-sided for complex x.
    """
    T = 1/meta["fs"]    # sampling time interval
    if meta["complex"]:
        ff = np.fft.fftfreq(meta["N"], T)
    else:
        ff = np.fft.rfftfreq(meta["N"], T)
    return ff
    


# %%
# Import and play x1 signal
f_name1 = "signals/tf_probe.wav"
fs1, x1 = wav16read(f_name1)
print(fs1)
ipd.Audio(x1, rate=fs1)

# %%
#B, N = 256, 256
B, N = 1024, 1024
X1, meta1 = analysis(x1, B=B, hop=B/4, N=N, window="hann", fs=fs1)
X1mag = np.abs(X1)
X1mag = X1mag/X1mag.max()   # normalized
min1dB = -80
X1dB = np.maximum(20*np.log10(X1mag+1e-12), min1dB)
f1 = frame_freqs(meta1)
t1 = frame_times(meta1)

# %%
fig, axs = plt.subplots(1, 1, figsize=(7, 4))
fig.canvas.toolbar_position = 'top'
#im = show(axs, X1, meta1, "Relative dB")
axs.pcolormesh(t1, f1, X1dB.T)

# %%

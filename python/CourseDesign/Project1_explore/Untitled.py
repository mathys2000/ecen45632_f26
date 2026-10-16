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
# # Project1_PartA_V2
#
# Time Frequency Analysis and Windows, verification code from ClaudeAI

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
def synthesis(X, meta, tol=1e-8):
    """
    Weighted overlap-add with squared-window normalization.
    Returns (y, denominator over retained samples).
    Raises NOLAError if NOLA (non-zero overlap add) fails.
    """
    B, hop, N, w = meta["B"], meta["hop"], meta["N"], meta["w"]    # meta data
    M = X.shape[0]    # number of frames
    if meta["complex"]:
        v = np.fft.ifft(X, n=N, axis=1)[:,:B]    # recovered frames (complex-valued)
    else:
        v = np.fft.irfft(X, n=N, axis=1)[:,:B]   # recovered frames (real-valued)
    total = (M - 1)*hop + B    # total length of output sequence (with pad_f, pad_b padding)
    idx = hop*np.arange(M)[:, None] + np.arange(B)[None, :]  # 2D array of indexes from v array
    num = np.zeros(total, v.dtype)   # prepare numerator
    den = np.zeros(total)   # prepare denominator
    np.add.at(num, idx, v*w)     # add v*w as many times as an index appears in idx (accounting for all overlaps)
    np.add.at(den, idx, np.broadcast_to(w**2, idx.shape))   # applies w**2 elements to each frame
    sl = slice(meta["pad_front"], meta["pad_front"] + meta["Lx"])  # slicing for num, den
    d = den[sl]
    if d.min() <= tol*w.max()**2:
        raise ValueError(f"NOLA violated: min sum w^2 = {d.min():.3g} (window '{meta['window']}'")
    return num[sl] / d, d    # return y and denominator d



# %%
def ola_filter(x, h, B=512):
    """
    Exact linear convolution by FFT overlap-add.
    Output length len(x)+len(h)-1.
    """
    x = np.asarray(x)   # data sequence
    h = np.asarray(h)   # filter impulse response
    Lx, Lh = x.size, h.size
    Nfft = 1 << int(np.ceil(np.log2(B + Lh - 1)))  # Nfft is power of 2 greater or equal to B+Lh-1
    cplx = np.iscomplexobj(x) or np.iscomplexobj(h)
    if cplx:
        fft, ifft = (np.fft.fft, np.fft.ifft)    # shortcuts for np.fft.fft and np.fft.ifft
    else:
        fft, ifft = (np.fft.rfft, np.fft.irfft)   # shortcuts for np.fft.rfft and np.fft.irfft
    Hf = fft(h, Nfft)
    if cplx:
        y = np.zeros(Lx+Lh-1, np.complex128)    # prepare output array
    else:
        y = np.zeros(Lx+Lh-1, np.float64)    # prepare output array
    for s in range(0, Lx, B):
        blk = x[s:s+B]         # last block may be partial
        if cplx:
            yb = ifft(fft(blk, Nfft) * Hf)
        else:
            yb = ifft(fft(blk, Nfft) * Hf, Nfft)    # full block convolution length
        n = len(blk) + Lh -1    # length of block convolution
        y[s:s+n] += yb[:n]    # overlap add step
    return y   # return linear convolution of x and h
    


# %%
# From ClaudeAI
import json, time
from pathlib import Path
from common import FS, fixtures, read_wav, write_wav, relerr, snr, receive


# %%
# From ClaudeAI
SEED = 2735
rng = np.random.default_rng(SEED)
out = Path("results"); out.mkdir(exist_ok=True)
a = fixtures(Path("signals"))
R = {}

def spec_db(X, floor=-80):
    m = np.abs(X); m = m / m.max()
    return np.maximum(20 * np.log10(m + 1e-12), floor)

def show(ax, X, meta, title, fmax=4000):
    ##f = dsp.frame_freqs(meta); t = dsp.frame_times(meta)
    f = frame_freqs(meta); t = frame_times(meta)
    im = ax.pcolormesh(t, f, spec_db(X).T, vmin=-80, vmax=0, shading="auto")
    ax.set_ylim(0, fmax); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title}\nfs={meta['fs']} Hz, {meta['window']}, L={meta['B']}, H={meta['hop']}, N={meta['N']}", fontsize=8)
    return im



# %%
# ---------------------------------------------------------------- A1
# From ClaudeAI
p = a["probe"]; A1 = {}
fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, (L, N) in zip(axs, [(256, 256), (1024, 1024), (4096, 4096), (1024, 4096)]):
    ## X, m = dsp.analysis(p, L=L, hop=L // 4, N=N)
    X, m = analysis(p, B=L, hop=L // 4, N=N)
    im = show(ax, X, m, "Relative dB")
    ## t = dsp.frame_times(m); f = dsp.frame_freqs(m)
    t = frame_times(m); f = frame_freqs(m)
    # transient (5 ms Hann pulse) is lowpass: use the 50-250 Hz band, away from tones/chirp, t in 0.7-1.1 s
    win_t = (t > 0.7) & (t < 1.1)
    lo = np.sum(np.abs(X[:, (f > 50) & (f < 250)]) ** 2, axis=1) * win_t
    above = t[lo > 0.1 * lo.max()]                              # -10 dB frame-energy threshold
    mid = (t > 1.5) & (t < 3.0)
    avg = np.abs(X[mid]).mean(axis=0)
    g = lambda fq: np.interp(fq, f, avg)
    i1, i2 = np.argmin(abs(f - 1000)), np.argmin(abs(f - 1080))     # nearest bins to each tone
    peaks = min(avg[i1], avg[i2]); valley = avg[i1:i2 + 1].min()
    A1[f"L{L}_N{N}"] = dict(Tw_ms=1e3 * L / FS, bin_Hz=FS / N, Th_ms=1e3 * L / 4 / FS,
        transient_width_ms_at_minus10dB=1e3 * (above.max() - above.min()) if len(above) else 0,
        tone_valley_dB_below_peaks=float(20 * np.log10(peaks / valley)),
        chirp_spread_Hz_per_frame=float(750 * L / FS))
fig.colorbar(im, ax=axs, label="dB rel. max"); fig.savefig(out / "A1_spectrograms.png", dpi=110); plt.close(fig)
R["A1"] = A1


# %%
print(R)

# %%
print(a["probe"].shape)
print(a["probe"])

# %%
ipd.Audio(p, rate=Fs)

# %%
# ---------------------------------------------------------------- A2
# From ClaudeAI
_, x = read_wav("signals/leakage.wav"); L, N = 1024, 65536
fN = np.fft.rfftfreq(N, 1 / FS); A2 = {}
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
for name in ["rect", "hann", "hamming", "blackman"]:
    ## w = dsp.make_window(name, L)
    w = ss.get_window(name, L, fftbins=True)   # periodic (vs symmetric) window
    S = np.abs(np.fft.rfft(x * w, N)) / w.sum() * 2          # coherent-gain corrected amplitude
    axs[0].plot(fN, 20 * np.log10(S + 1e-12), label=name)
    W = np.abs(np.fft.rfft(w, N)); W /= W[0]
    k = 0
    while W[k + 1] < W[k]: k += 1                            # first null = first local minimum
    side = 20 * np.log10(W[k:].max())
    axs[1].plot(fN, 20 * np.log10(W + 1e-12), label=name)
    # leakage of the strong tone alone at 1140 Hz vs. the weak tone's true level
    strong = .65 * np.cos(2 * np.pi * 1003 * np.arange(L) / FS)
    Ss = np.abs(np.fft.rfft(strong * w, N)) / w.sum() * 2
    i = np.argmin(abs(fN - 1140))
    A2[name] = dict(mainlobe_null_to_null_Hz=2 * fN[k], highest_sidelobe_dB=float(side),
        ENBW_Hz=FS * np.sum(w ** 2) / w.sum() ** 2, ENBW_bins=L * np.sum(w ** 2) / w.sum() ** 2,
        strong_tone_leakage_at_1140_dB=float(20 * np.log10(Ss[i] + 1e-12)),
        weak_tone_true_dB=float(20 * np.log10(.015)),
        weak_tone_measured_dB=float(20 * np.log10(S[np.argmax(S * ((fN > 1120) & (fN < 1160)))])))
axs[0].set(xlim=(900, 1300), ylim=(-120, 0), xlabel="Frequency (Hz)", ylabel="Amplitude (dB re 1)",
           title="Two-tone, L=1024, N=65536, fs=16 kHz"); axs[0].legend()
axs[1].set(xlim=(0, 200), ylim=(-120, 0), xlabel="Frequency (Hz)", ylabel="dB re DC", title="Window transforms alone")
fig.savefig(out / "A2_leakage.png", dpi=110); plt.close(fig); R["A2"] = A2


# %%
print(R)

# %%
print(x.shape)

# %%
ipd.Audio(x, rate=Fs)

# %%
# ---------------------------------------------------------------- B1
# From ClaudeAI
B1 = []
for win, ovs in [("hann", [0, .25, .5, .75, .875]), ("rect", [0])]:
    for ov in ovs:
        hop = int(round(1024 * (1 - ov))); row = dict(window=win, overlap=ov, hop=hop)
        try:
            ## t0 = time.perf_counter(); X, m = dsp.analysis(p, 1024, hop, window=win)
            ## y, d = dsp.synthesis(X, m); row.update(runtime_ms=1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter(); X, m = analysis(p, 1024, hop, window=win)
            y, d = synthesis(X, m); row.update(runtime_ms=1e3 * (time.perf_counter() - t0))
            row.update(frames=X.shape[0], relerr=relerr(p, y), den_min=float(d.min()), den_max=float(d.max()))
        ## except dsp.NOLAError as e:
            ## row.update(status="NOLA failure", message=str(e))
        except:
            row.update(status="NOLA failure", message="NOLA failure")
        B1.append(row)
write_wav(out / "_tmp.wav", p); _, back = read_wav(out / "_tmp.wav"); (out / "_tmp.wav").unlink()
R["B1"] = dict(table=B1, wav_roundtrip_relerr=relerr(p, back))


# %%
print(R)

# %%
print(back.shape)

# %%
ipd.Audio(back, rate=Fs)

# %%
# ---------------------------------------------------------------- B2
# From ClaudeAI
## h = signal.firwin(129, 1200, fs=FS, window="hamming"); ref = np.convolve(a["mixture"], h); B2 = []
h = ss.firwin(129, 1200, fs=Fs, window="hamming"); ref = np.convolve(a["mixture"], h); B2 = []
for B in [128, 512, 2048]:
    ## y = dsp.ola_filter(a["mixture"], h, B)
    y = ola_filter(a["mixture"], h, B)
    B2.append(dict(B=B, nfft=1 << int(np.ceil(np.log2(B + 128))), blocks=int(np.ceil(len(a["mixture"]) / B)),
                   relerr=relerr(ref, y), len_ok=len(y) == len(ref)))
# aliasing demo: FFT shorter than B+M-1 circularly wraps the tail
B, nf = 512, 512; blk = a["mixture"][:B]
alias = np.fft.irfft(np.fft.rfft(blk, nf) * np.fft.rfft(h, nf), nf)
B2.append(dict(demo="nfft=B (too short)", relerr_vs_linear_first512=relerr(np.convolve(blk, h)[:nf], alias)))
R["B2"] = B2


# %%
print(R)

# %%
print(a["mixture"].shape)
print(y.shape)
print(alias.shape)

# %%
ipd.Audio(a["mixture"], rate=Fs)

# %%
ipd.Audio(y, rate=Fs)

# %%
ipd.Audio(alias, rate=Fs)

# %%

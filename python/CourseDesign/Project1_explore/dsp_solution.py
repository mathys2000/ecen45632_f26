"""Instructor reference algorithms. FFT convention: local index, no hidden phase shift."""
import numpy as np
from scipy import signal, ndimage
from common import FS

def analysis(x, L=1024, hop=256, window='hann', nfft=None):
    nfft = L if nfft is None else nfft
    if not (0 < hop <= L <= nfft):
        raise ValueError('Require 0 < hop <= L <= nfft')
    w = signal.get_window(window, L, fftbins=True)
    # Padding before sample zero covers window endpoints; crop precisely at synthesis.
    left = L
    xp = np.pad(x, (left, L))
    count = 1 + int(np.ceil((len(xp)-L)/hop))
    needed = (count-1)*hop + L
    xp = np.pad(xp, (0, needed-len(xp)))
    frames = np.array([xp[m*hop:m*hop+L]*w for m in range(count)])
    X = np.fft.fft(frames, n=nfft, axis=1)
    meta = dict(w=w, L=L, hop=hop, nfft=nfft, left=left, length=len(x),
                real_input=not np.iscomplexobj(x))
    return X, meta

def synthesis(X, meta):
    w, L, hop = meta['w'], meta['L'], meta['hop']
    frames = np.fft.ifft(X, axis=1)[:, :L]
    N = (len(frames)-1)*hop + L
    y = np.zeros(N, complex); d = np.zeros(N)
    for m, frame in enumerate(frames):
        idx = slice(m*hop, m*hop+L)
        y[idx] += frame*w
        d[idx] += w*w
    crop = slice(meta['left'], meta['left']+meta['length'])
    if np.any(d[crop] < 1e-12):
        raise ValueError('NOLA failure in retained samples')
    y = y[crop]/d[crop]
    if meta['real_input']:
        if np.max(np.abs(y.imag)) > 1e-9:
            raise ValueError('Mask destroyed Hermitian symmetry')
        y = y.real
    return y, d[crop]

def ola_filter(x, h, B=512, nfft=None):
    """Exact full linear convolution using nonoverlapping input blocks."""
    M = len(h)
    if nfft is None:
        nfft = 1 << (B+M-2).bit_length()
    if nfft < B+M-1:
        raise ValueError('FFT is too short for linear convolution')
    H = np.fft.fft(h, nfft)
    y = np.zeros(len(x)+M-1, complex)
    for start in range(0, len(x), B):
        block = x[start:start+B]
        z = np.fft.ifft(np.fft.fft(block, nfft)*H)[:len(block)+M-1]
        y[start:start+len(z)] += z
    return y if np.iscomplexobj(x) or np.iscomplexobj(h) else y.real

def remove_jammer(x, L=1024, hop=256, width_hz=60, floor=.03,
                  search_band=(1600, 3400)):
    """No clean reference or known trajectory used. Dominant ridge within prior band."""
    X, meta = analysis(x, L, hop)
    f = np.fft.fftfreq(meta['nfft'], 1/FS)
    bins = np.flatnonzero((f>=search_band[0]) & (f<=search_band[1]))
    ridge = f[bins[np.argmax(np.abs(X[:, bins]), axis=1)]]
    # Padded empty frames may choose arbitrary bins; they do not affect retained data.
    ridge = ndimage.median_filter(ridge, size=5, mode='nearest')
    distance = np.abs(np.abs(f)[None, :]-ridge[:, None])
    G = 1-(1-floor)*np.exp(-.5*(distance/width_hz)**2)
    Y = G*X  # Real nonnegative even mask keeps each surviving coefficient's phase.
    y, _ = synthesis(Y, meta)
    return y, dict(X=X, Y=Y, G=G, meta=meta, ridge=ridge, f=f)

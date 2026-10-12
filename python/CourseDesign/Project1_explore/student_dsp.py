"""Fill the four required algorithms. common.py supplies fixtures and the receiver."""
import numpy as np
from scipy import signal
from common import FS

def analysis(x, L=1024, hop=256, window='hann', nfft=None):
    # TODO: periodic window, padded frames, local-index two-sided FFT, metadata.
    raise NotImplementedError

def synthesis(X, meta):
    # TODO: IFFT, synthesis window, overlap sum, samplewise denominator, crop.
    # Reject zero denominator and preserve complex I/Q.
    raise NotImplementedError

def ola_filter(x, h, B=512, nfft=None):
    # TODO: FFT >= B+len(h)-1, full convolution including final short block and tail.
    raise NotImplementedError

def remove_jammer(x, L=1024, hop=256, width_hz=60, floor=.03,
                  search_band=(1600, 3400)):
    # TODO: estimate ridge from mixture only; smooth it; apply a tapered symmetric
    # real mask to COMPLEX coefficients; synthesize; return diagnostic arrays.
    raise NotImplementedError

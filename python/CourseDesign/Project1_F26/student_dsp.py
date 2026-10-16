"""Fill the three required algorithms. common.py supplies fixtures and the receiver."""
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


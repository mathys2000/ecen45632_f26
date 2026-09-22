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
# # Waveform Dispersion
#
# Pulse through IIR filter and filtfilt response

# %%
import numpy as np
import scipy.signal as ss
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt

# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
fs = 1000
tlen = 1      # time length in sec

# %%
# Generate waveform
tt = np.arange(np.round(tlen*fs))/fs    # time axis
#xt = np.exp(-0.5*((tt-0.35)/0.008)**2) - 0.55*np.exp(-0.5*((tt-0.37)/0.012)**2)
#xt = np.exp(-0.5*((tt-0.35)/0.004)**2) - 0.55*np.exp(-0.5*((tt-0.36)/0.006)**2)
#xt = np.exp(-0.5*((tt-0.35)/0.002)**2) - 0.55*np.exp(-0.5*((tt-0.355)/0.003)**2)
xt = np.exp(-0.5*((tt-0.3472)/0.002)**2) - 0.55*np.exp(-0.5*((tt-0.3522)/0.003)**2)
Xf = np.fft.fft(xt)/fs     # approx to FT X(f)
Nfft = Xf.size
ff = fs*np.arange(Nfft)/Nfft


# %%
# Plot test waveform
fig, axs = plt.subplots(2, 1, figsize=(7, 5))
fig.canvas.toolbar_position = 'top'
axs[0].plot(tt*1e3, xt, label='Time Domain')
axs[0].set_xlim([300, 500])
axs[0].set_title(f'Test Waveform, fs={fs} Hz')
axs[0].set_ylabel('$x(t)$')
axs[0].set_xlabel('Time [ms]')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ff[:int(Nfft/2+1)], np.abs(Xf[:int(Nfft/2+1)]), label='Frequency Domain')
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('$X(f)$')
axs[1].grid(alpha=0.5)
axs[1].legend()
plt.tight_layout()
plt.show()

# %%
# Generate IIR filters
Nfilt = 7
wc, rp, rs = 80, 1, 40
ww = np.linspace(0, np.pi, 1000)
dw = (ww[-1]-ww[0])/(ww.size-1)
sos_butt = ss.butter(Nfilt, wc, btype='low', fs=fs, output='sos')
yt_butt_causal = ss.sosfilt(sos_butt, xt)
yt_butt_zero = ss.sosfiltfilt(sos_butt, xt)
Hejw_butt = ss.freqz_sos(sos_butt, ww)[1]
sos_ellip = ss.ellip(Nfilt, rp, rs, wc, btype='low', fs=fs, output='sos')
yt_ellip_causal = ss.sosfilt(sos_ellip, xt)
yt_ellip_zero = ss.sosfiltfilt(sos_ellip, xt)
Hejw_ellip = ss.freqz_sos(sos_ellip, ww)[1]

# %%
# Plot frequency response
fig, axs = plt.subplots(3, 1, figsize=(7,6))
fig.canvas.toolbar_position = 'top'
axs[0].plot(fs/2*ww/np.pi, np.abs(Hejw_butt), label='Butterworth')
axs[0].plot(fs/2*ww/np.pi, np.abs(Hejw_ellip), label='Elliptic')
axs[0].set_title(f'IIR Filters, wc={wc} Hz, rp={rp}, rs={rs}, N={Nfilt}, fs={fs}')
axs[0].set_ylabel('$|H|$')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(fs/2*ww[:-2]/np.pi, np.unwrap(np.angle(Hejw_butt[:-2])), label='Butterworth')
axs[1].plot(fs/2*ww[:-2]/np.pi, np.unwrap(np.angle(Hejw_ellip[:-2])), label='Elliptic')
axs[1].set_ylabel('$\\angle H$ [rad]')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(fs/2*ww[1:-2]/np.pi, -np.diff(np.unwrap(np.angle(Hejw_butt)))[:-2]/dw, label='Butterworth')
axs[2].plot(fs/2*ww[1:-2]/np.pi, -np.diff(np.unwrap(np.angle(Hejw_ellip)))[:-2]/dw, label='Elliptic')
axs[2].set_ylabel('$\\tau_g$')
axs[2].set_xlabel('Frequency [Hz]')
axs[2].set_ylim([-5, 100])
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.show()

# %%
# Plot time domain results
fig, axs = plt.subplots(3, 1, figsize=(7, 6))
fig.canvas.toolbar_position = 'top'
axs[0].plot(tt*1e3, xt, label='Original')
axs[0].set_title(f'Filtered Testsignals, fs={fs} Hz')
axs[0].set_ylabel('$x(t)$')
axs[0].set_xlim([300, 500])
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(tt*1e3, yt_butt_causal, label='Butterworth, causal')
axs[1].plot(tt*1e3, yt_ellip_causal, label='Elliptic, causal')
axs[1].set_ylabel('$y(t)$')
axs[1].set_xlim([300, 500])
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(tt*1e3, yt_butt_zero, label='Butterworth, filtfilt')
axs[2].plot(tt*1e3, yt_ellip_zero, label='Elliptic, filtfilt')
axs[2].set_ylabel('$y(t)$, offline')
axs[2].set_xlabel('Time [ms]')
axs[2].set_xlim([300, 500])
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.show()

# %%

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
# # IIR Prototype LPFs
#
# Frequency (magnitude and phase) response of analog IIR prototype lowpassm filters

# %%
import numpy as np
import scipy.signal as ss
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt

# %%
# %matplotlib widget
fsz = (9,5)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
filts = ['butt', 'cheby1', 'cheby2', 'ellip']
N = 5           # filter order
Wc = 1          # cutoff frequency
#Wc = 1.453085     # prewarp for wc=0.4*pi at T=1
rp = 1          # passband ripple in dB
rs = 40         # stopband ripple in dB
WW = np.logspace(-1, 1, 1000)    # frequency in rad/s

# %%
# Filter designs
b_butt, a_butt = ss.butter(N, Wc, analog=True, output='ba')
b_cheby1, a_cheby1 = ss.cheby1(N, rp, Wc, analog=True, output='ba')
b_cheby2, a_cheby2 = ss.cheby2(N, rs, 1.6*Wc, analog=True, output='ba')
b_ellip, a_ellip = ss.ellip(N, rp, rs, Wc, analog=True, output='ba')

# %%
# Frequency response
HjW_butt = ss.freqs(b_butt, a_butt, WW)[1]
HjW_cheby1 = ss.freqs(b_cheby1, a_cheby1, WW)[1]
HjW_cheby2 = ss.freqs(b_cheby2, a_cheby2, WW)[1]
HjW_ellip = ss.freqs(b_ellip, a_ellip, WW)[1]


# %%
# Poles and zeros
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
z_butt = npp.polyroots(b_butt[::-1])    # zeros
p_butt = npp.polyroots(a_butt[::-1])    # poles
z_cheby1 = npp.polyroots(b_cheby1[::-1])    # zeros
p_cheby1 = npp.polyroots(a_cheby1[::-1])    # poles
z_cheby2 = npp.polyroots(b_cheby2[::-1])    # zeros
p_cheby2 = npp.polyroots(a_cheby2[::-1])    # poles
z_ellip = npp.polyroots(b_ellip[::-1])    # zeros
p_ellip = npp.polyroots(a_ellip[::-1])    # poles


# %%
fig = plt.figure(figsize=fsz, layout='constrained')
fig.canvas.toolbar_position = 'top'
axs = fig.add_gridspec(2, 2)
ax00 = fig.add_subplot(axs[0,0])
if 'butt' in filts:
    ax00.semilogx(WW, 20*np.log10(np.abs(HjW_butt)), color='blue', label='Butterworth')
if 'cheby1' in filts:
    ax00.semilogx(WW, 20*np.log10(np.abs(HjW_cheby1)), color='orange', label='Chebyshev I')
if 'cheby2' in filts:
    ax00.semilogx(WW, 20*np.log10(np.abs(HjW_cheby2)), color='green', label='Chebyshev II')
if 'ellip' in filts:
    ax00.semilogx(WW, 20*np.log10(np.abs(HjW_ellip)), color='red', label='Elliptic')
ax00.grid(alpha=0.5)
ax00.legend()
ax00.set_title(f'Classical Analog Lowpass Prototypes, N={N}')
ax00.set_ylabel(f'$|H(j\\Omega)|$')
ax00.set_ylim([-90, 10])
ax10 = fig.add_subplot(axs[1,0])
if 'butt' in filts:
    ax10.semilogx(WW, 180/np.pi*np.unwrap(np.angle(HjW_butt)), color='blue', label='Butterworth')
if 'cheby1' in filts:
    ax10.semilogx(WW, 180/np.pi*np.unwrap(np.angle(HjW_cheby1)), color='orange', label='Chebyshev I')
if 'cheby2' in filts:
    ax10.semilogx(WW, 180/np.pi*np.unwrap(np.angle(HjW_cheby2)), color='green', label='Chebyshev II')
if 'ellip' in filts:
    ax10.semilogx(WW, 180/np.pi*np.unwrap(np.angle(HjW_ellip)), color='red', label='Elliptic')
ax10.grid(alpha=0.5)
ax10.legend()
ax10.set_ylabel(f'$\\angle H(j\\Omega)$ [deg, unwrapped]')
ax10.set_xlabel(f'Analog frequency $\\Omega$ [rad/sec], $\\Omega_c$={Wc:1.3f} [rad/sec]')
axx1 = fig.add_subplot(axs[:,1])
axx1.plot(Wc*cir.real, Wc*cir.imag, '--k', linewidth=0.5)
axx1.axvline(0, color='black', linewidth=0.5)
if 'butt' in filts:
    axx1.plot(z_butt.real, z_butt.imag, 'o', mec='blue')
    axx1.plot(p_butt.real, p_butt.imag, 'x', mec='blue', label='Butterworth')
if 'cheby1' in filts:
    axx1.plot(z_cheby1.real, z_cheby1.imag, 'o', mec='orange')
    axx1.plot(p_cheby1.real, p_cheby1.imag, 'x', mec='orange', label='Chebyshev I')
if 'cheby2' in filts:
    axx1.plot(z_cheby2.real, z_cheby2.imag, 'o', mec='green')
    axx1.plot(p_cheby2.real, p_cheby2.imag, 'x', mec='green', label='Chebyshev II')
if 'ellip' in filts:
    axx1.plot(z_ellip.real, z_ellip.imag, 'o', mec='red')
    axx1.plot(p_ellip.real, p_ellip.imag, 'x', mec='red', label='Elliptic')
axx1.set_title(f'Analog Design Pole-Zero Plot, $\\Omega_c$={Wc:1.3f}')
axx1.set_ylim([-2, 2])
axx1.grid(alpha=0.5)
axx1.set_aspect('equal')
axx1.legend()
plt.show()


# %%
# DT filter parameters
T = 1
#T = 0.1
wc = 2*np.atan(Wc*T/2)     # frequency warping
ww = np.logspace(np.log10(np.pi*1e-2), np.log10(np.pi), 1000)
ww_lin = np.linspace(0, np.pi, 1000)
dw = (ww_lin[-1]-ww_lin[0])/(np.size(ww_lin)-1)

# %%
# Use bilinear transform to compute DT filter coefficients
b_butt_DT, a_butt_DT = ss.bilinear(b_butt, a_butt, fs=1/T)
b_cheby1_DT, a_cheby1_DT = ss.bilinear(b_cheby1, a_cheby1, fs=1/T)
b_cheby2_DT, a_cheby2_DT = ss.bilinear(b_cheby2, a_cheby2, fs=1/T)
b_ellip_DT, a_ellip_DT = ss.bilinear(b_ellip, a_ellip, fs=1/T)

# %%
# DT frequency response
Hejw_butt = ss.freqz(b_butt_DT, a_butt_DT, ww)[1]
Hejw_cheby1 = ss.freqz(b_cheby1_DT, a_cheby1_DT, ww)[1]
Hejw_cheby2 = ss.freqz(b_cheby2_DT, a_cheby2_DT, ww)[1]
Hejw_ellip = ss.freqz(b_ellip_DT, a_ellip_DT, ww)[1]


# %%
# DT poles and zeros
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
z_butt_DT = npp.polyroots(b_butt_DT[::-1])    # zeros
p_butt_DT = npp.polyroots(a_butt_DT[::-1])    # poles
z_cheby1_DT = npp.polyroots(b_cheby1_DT[::-1])    # zeros
p_cheby1_DT = npp.polyroots(a_cheby1_DT[::-1])    # poles
z_cheby2_DT = npp.polyroots(b_cheby2_DT[::-1])    # zeros
p_cheby2_DT = npp.polyroots(a_cheby2_DT[::-1])    # poles
z_ellip_DT = npp.polyroots(b_ellip_DT[::-1])    # zeros
p_ellip_DT = npp.polyroots(a_ellip_DT[::-1])    # poles


# %%
fig = plt.figure(figsize=fsz, layout='constrained')
fig.canvas.toolbar_position = 'top'
axs = fig.add_gridspec(2, 2)
ax00 = fig.add_subplot(axs[0,0])
if 'butt' in filts:
    ax00.semilogx(ww/np.pi, 20*np.log10(np.abs(Hejw_butt)), color='blue', label='Butterworth')
if 'cheby1' in filts:
    ax00.semilogx(ww/np.pi, 20*np.log10(np.abs(Hejw_cheby1)), color='orange', label='Chebyshev I')
if 'cheby2' in filts:
    ax00.semilogx(ww/np.pi, 20*np.log10(np.abs(Hejw_cheby2)), color='green', label='Chebyshev II')
if 'ellip' in filts:
    ax00.semilogx(ww/np.pi, 20*np.log10(np.abs(Hejw_ellip)), color='red', label='Elliptic')
ax00.grid(alpha=0.5)
ax00.legend()
ax00.set_title(f'DT IIR LPFs from Classical Analog Prototypes, N={N}, T={T}')
ax00.set_ylabel(f'$|H(e^{{j\\omega)}}|$')
ax00.set_ylim([-90, 10])
ax10 = fig.add_subplot(axs[1,0])
if 'butt' in filts:
    ax10.semilogx(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_butt)), color='blue', label='Butterworth')
if 'cheby1' in filts:
    ax10.semilogx(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_cheby1)), color='orange', label='Chebyshev I')
if 'cheby2' in filts:
    ax10.semilogx(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_cheby2)), color='green', label='Chebyshev II')
if 'ellip' in filts:
    ax10.semilogx(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_ellip)), color='red', label='Elliptic')
ax10.grid(alpha=0.5)
ax10.legend()
ax10.set_ylabel(f'$\\angle H(e^{{j\\omega}})$ [deg, unwrapped]')
ax10.set_xlabel(f'Normalized frequency $\\omega/\\pi$ [rad/sample], $\\omega_c/\\pi$={wc/np.pi:1.3f}')
axx1 = fig.add_subplot(axs[:,1])
axx1.plot(cir.real, cir.imag, '--k', linewidth=0.5)
if 'butt' in filts:
    axx1.plot(z_butt_DT.real, z_butt_DT.imag, 'o', mec='blue')
    axx1.plot(p_butt_DT.real, p_butt_DT.imag, 'x', mec='blue', label='Butterworth')
if 'cheby1' in filts:
    axx1.plot(z_cheby1_DT.real, z_cheby1_DT.imag, 'o', mec='orange')
    axx1.plot(p_cheby1_DT.real, p_cheby1_DT.imag, 'x', mec='orange', label='Chebyshev I')
if 'cheby2' in filts:
    axx1.plot(z_cheby2_DT.real, z_cheby2_DT.imag, 'o', mec='green')
    axx1.plot(p_cheby2_DT.real, p_cheby2_DT.imag, 'x', mec='green', label='Chebyshev II')
if 'ellip' in filts:
    axx1.plot(z_ellip_DT.real, z_ellip_DT.imag, 'o', mec='red')
    axx1.plot(p_ellip_DT.real, p_ellip_DT.imag, 'x', mec='red', label='Elliptic')
axx1.set_title(f'DT LPF Pole-Zero Plot, $\\omega_c/\\pi$={wc/np.pi:1.3f}')
#axx1.set_ylim([-2, 2])
axx1.grid(alpha=0.5)
axx1.set_aspect('equal')
axx1.legend()
plt.show()


# %%
# DT frequency response, linear omega
Hejw_butt_lin = ss.freqz(b_butt_DT, a_butt_DT, ww_lin)[1]
Hejw_cheby1_lin = ss.freqz(b_cheby1_DT, a_cheby1_DT, ww_lin)[1]
Hejw_cheby2_lin = ss.freqz(b_cheby2_DT, a_cheby2_DT, ww_lin)[1]
Hejw_ellip_lin = ss.freqz(b_ellip_DT, a_ellip_DT, ww_lin)[1]


# %%
# Group delay
taug_butt_DT = -np.diff(np.unwrap(np.angle(Hejw_butt_lin)))/dw
taug_cheby1_DT = -np.diff(np.unwrap(np.angle(Hejw_cheby1_lin)))/dw
taug_cheby2_DT = -np.diff(np.unwrap(np.angle(Hejw_cheby2_lin)))/dw
taug_ellip_DT = -np.diff(np.unwrap(np.angle(Hejw_ellip_lin)))/dw


# %%
# Plot group delays
fig, axs = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
if 'butt' in filts:
    axs.plot(ww_lin[1:-2]/np.pi, taug_butt_DT[:-2], color='blue', label='Butterworth')
if 'cheby1' in filts:
    axs.plot(ww_lin[1:-2]/np.pi, taug_cheby1_DT[:-2], color='orange', label='Chebyshev I')
if 'cheby2' in filts:
    axs.plot(ww_lin[1:-2]/np.pi, taug_cheby2_DT[:-2], color='green', label='Chebyshev II')
if 'ellip' in filts:
    axs.plot(ww_lin[1:-2]/np.pi, taug_ellip_DT[:-2], color='red', label='Elliptic')
axs.set_title(f'Group Delay of DT LPFs from Analog Prototypes, N={N}, T={T}')
axs.set_ylabel(f'Group delay (samples)')
axs.set_xlabel(f'Normalized frequency $\\omega/\\pi$ [rad/sample], $\\omega_c/\\pi$={wc/np.pi:1.3f}')
axs.grid(alpha=0.5)
axs.set_ylim([0, 30/T])
axs.legend()
plt.show()

# %%

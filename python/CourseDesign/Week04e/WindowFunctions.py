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
# # Window Functions
#
# Time and frequency domain plots of window functions for filter design

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
Nx = 51    # filter/window length
M2 = (Nx-1)/2     # offset to make hlp causal
nn = np.arange(Nx)     # index axis
wc = 0.4*np.pi     # cutoff frequency
Nw = 5000   # number of frequency points to evaluate

# %%
# Ideal LPF, truncated
hlp = wc/np.pi*np.sinc(wc/np.pi*(nn-M2))

# %%
# Window functions
kbeta = 3.86
Wrect = ss.get_window('rect', Nx, fftbins=False)
Wbart = ss.get_window('bart', Nx, fftbins=False)
Whann = ss.get_window('hann', Nx, fftbins=False)
Whamm = ss.get_window('hamm', Nx, fftbins=False)
Wblack = ss.get_window('black', Nx, fftbins=False)
Wkais = ss.get_window(('kaiser',kbeta), Nx, fftbins=False)
ww, Hejw_rect = ss.freqz(hlp*Wrect, 1, Nw)
ww, Hejw_bart = ss.freqz(hlp*Wbart, 1, Nw)
ww, Hejw_hann = ss.freqz(hlp*Whann, 1, Nw)
ww, Hejw_hamm = ss.freqz(hlp*Whamm, 1, Nw)
ww, Hejw_black = ss.freqz(hlp*Wblack, 1, Nw)
ww, Hejw_kais = ss.freqz(hlp*Wkais, 1, Nw)


# %%
# Time domain plot
fig, axs = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
axs.plot(nn, Wrect, label='Rectangular')
axs.plot(nn, Wbart, label='Bartlett')
axs.plot(nn, Whann, label='Hann')
axs.plot(nn, Whamm, label='Hamming')
axs.plot(nn, Wblack, label='Blackman')
axs.plot(nn, Wkais, linewidth=2.0, label=f'Kaiser $\\beta$={kbeta}')
axs.set_title(f'Common Windows in the Time Domain')
axs.set_ylabel(f'$w[n]$')
axs.set_xlabel(f'Tap index $n$')
axs.grid(alpha=0.5)
axs.legend()
plt.show()

# %%
# Frequency domain plot of LPF
fig, axs = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_rect)), label='Rectangular')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_bart)), label='Bartlett')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_hann)), label='Hann')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_hamm)), label='Hamming')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_black)), label='Blackman')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Hejw_kais)), linewidth=2.0, label=f'Kaiser $\\beta$={kbeta}')
axs.set_title(f'Choice of Window Tradeoff: Transition Width vs. Stopband Ripple')
axs.set_ylabel(f'$|H(e^{{j\\omega}})|$')
axs.set_xlabel(f'Normalized frequency $\\omega/\\pi$, $\\omega_c/\\pi$={wc/np.pi}')
axs.set_ylim([-100, 10])
axs.grid(alpha=0.5)
axs.legend()
plt.show()

# %%
# Window functions in frequency domain
ww, Wejw_rect = ss.freqz(Wrect, 1, Nw)
ww, Wejw_bart = ss.freqz(Wbart, 1, Nw)
ww, Wejw_hann = ss.freqz(Whann, 1, Nw)
ww, Wejw_hamm = ss.freqz(Whamm, 1, Nw)
ww, Wejw_black = ss.freqz(Wblack, 1, Nw)
ww, Wejw_kais = ss.freqz(Wkais, 1, Nw)


# %%
# Frequency domain plot of Window
fig, axs = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_rect/np.max(np.abs(Wejw_rect)))), label='Rectangular')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_bart/np.max(np.abs(Wejw_bart)))), label='Bartlett')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_hann/np.max(np.abs(Wejw_hann)))), label='Hann')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_hamm/np.max(np.abs(Wejw_hamm)))), label='Hamming')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_black/np.max(np.abs(Wejw_black)))), label='Blackman')
axs.plot(ww/np.pi, 20*np.log10(np.abs(Wejw_kais/np.max(np.abs(Wejw_kais)))), linewidth=2.0, label=f'Kaiser $\\beta$={kbeta}')
axs.set_title(f'Normalized DTFTs of Common Window Functions')
axs.set_ylabel(f'$|W(e^{{j\\omega}})|$')
axs.set_xlabel(f'Normalized frequency $\\omega/\\pi$')
axs.set_ylim([-100, 10])
axs.set_xlim([0, 0.6])
axs.grid(alpha=0.5)
axs.legend(ncol=3)
plt.show()

# %%

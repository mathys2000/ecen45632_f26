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
# # FIR Type 1to4
#
# FIR filters of Type I, II, III, and IV

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
Nx1 = 11    # Type I number of taps
Nx2 = 10    # Type II number of taps
M2x1 = (Nx1-1)/2     # offset to make h[n] causal
M2x2 = (Nx2-1)/2     # offset top make h[n] causal
nn1 = np.arange(Nx1)     # index axis
nn2 = np.arange(Nx2)
wc = 0.4*np.pi     # cutoff frequency
Nw = 5000   # number of frequency points to evaluate

# %%
# Ideal LPF, truncated
hlp1 = wc/np.pi*np.sinc(wc/np.pi*(nn1-M2x1))
hlp2 = wc/np.pi*np.sinc(wc/np.pi*(nn2-M2x2))
ww, Hejw1 = ss.freqz(hlp1, 1, Nw)
ww, Hejw2 = ss.freqz(hlp2, 1, Nw)
dw = (ww[-1]-ww[0])/(ww.size-1)

# %%
# Time and frequency domain plots
fig, axs = plt.subplots(4, 2, figsize=(9,7))
fig.canvas.toolbar_position = 'top'
axs[0,0].stem(nn1, hlp1, label=f'N={Nx1}')
axs[0,0].set_title(f'FIR Type I LPF, $\\omega_c/\\pi$={wc/np.pi}')
axs[0,0].set_ylabel(f'$h[n]$')
axs[0,0].set_xlabel(f'$n$')
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc=1)
axs[0,1].stem(nn2, hlp2, label=f'N={Nx2}')
axs[0,1].set_title(f'FIR Type II LPF, $\\omega_c/\\pi$={wc/np.pi}')
axs[0,1].set_ylabel(f'$h[n]$')
axs[0,1].set_xlabel(f'$n$')
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc=1)
axs[1,0].plot(ww/np.pi, np.abs(Hejw1))
axs[1,0].set_ylabel('$|H(e^{{j\\omega}}|$')
axs[1,0].grid(alpha=0.5)
axs[2,0].plot(ww/np.pi, np.unwrap(np.angle(Hejw1)))
axs[2,0].set_ylabel('$\\angle H(e^{{j\\omega}})$')
axs[2,0].grid(alpha=0.5)
axs[3,0].plot(ww[1:]/np.pi, -np.diff(np.unwrap(np.angle(Hejw1)))/dw)
axs[3,0].set_ylabel('$\\tau_g(\\omega)$')
axs[3,0].set_xlabel('$\\omega/\\pi$')
axs[3,0].set_ylim([-1, Nx1])
axs[3,0].grid(alpha=0.5)
axs[1,1].plot(ww/np.pi, np.abs(Hejw2))
axs[1,1].set_ylabel('$|H(e^{{j\\omega}}|$')
axs[1,1].grid(alpha=0.5)
axs[2,1].plot(ww/np.pi, np.unwrap(np.angle(Hejw2)))
axs[2,1].set_ylabel('$\\angle H(e^{{j\\omega}})$')
axs[2,1].grid(alpha=0.5)
axs[3,1].plot(ww[1:]/np.pi, -np.diff(np.unwrap(np.angle(Hejw2)))/dw)
axs[3,1].set_ylabel('$\\tau_g(\\omega)$')
axs[3,1].set_xlabel('$\\omega/\\pi$')
axs[3,1].set_ylim([-1, Nx2])
axs[3,1].grid(alpha=0.5)
plt.show()

# %%
# FIR type III and IV filters
ix = np.where(nn1-M2x1 != 0)[0]
hhilb3 = np.zeros(Nx1)
hhilb3[ix] = (1-np.cos(np.pi*(nn1[ix]-M2x1)))/(np.pi*(nn1[ix]-M2x1))
hhilb4 = (1-np.cos(np.pi*(nn2-M2x2)))/(np.pi*(nn2-M2x2))
ww, Hejw3 = ss.freqz(hhilb3, 1, Nw)
ww, Hejw4 = ss.freqz(hhilb4, 1, Nw)
dw = (ww[-1]-ww[0])/(ww.size-1)

# %%
fig, axs = plt.subplots(4, 2, figsize=(9,7))
fig.canvas.toolbar_position = 'top'
axs[0,0].stem(nn1, hhilb3, label=f'N={Nx1}')
axs[0,0].grid(alpha=0.5)
axs[0,0].set_title(f'FIR Type III Hilbert Filter')
axs[0,0].set_ylabel(f'$h[n]$')
axs[0,0].set_xlabel(f'$n$')
axs[0,0].grid(alpha=0.5)
axs[0,0].legend(loc=1)
axs[0,1].stem(nn2, hhilb4, label=f'N={Nx2}')
axs[0,1].set_title(f'FIR Type IV Hilbert Filter')
axs[0,1].set_ylabel(f'$h[n]$')
axs[0,1].set_xlabel(f'$n$')
axs[0,1].grid(alpha=0.5)
axs[0,1].legend(loc=1)
axs[1,0].plot(ww/np.pi, np.abs(Hejw3))
axs[1,0].set_ylabel('$|H(e^{{j\\omega}}|$')
axs[1,0].grid(alpha=0.5)
axs[2,0].plot(ww/np.pi, np.unwrap(np.angle(Hejw3)))
axs[2,0].set_ylabel('$\\angle H(e^{{j\\omega}})$')
axs[2,0].grid(alpha=0.5)
axs[3,0].plot(ww[1:]/np.pi, -np.diff(np.unwrap(np.angle(Hejw3)))/dw)
axs[3,0].set_ylabel('$\\tau_g(\\omega)$')
axs[3,0].set_xlabel('$\\omega/\\pi$')
axs[3,0].set_ylim([-1, Nx1])
axs[3,0].grid(alpha=0.5)
axs[1,1].plot(ww/np.pi, np.abs(Hejw4))
axs[1,1].set_ylabel('$|H(e^{{j\\omega}}|$')
axs[1,1].grid(alpha=0.5)
axs[2,1].plot(ww/np.pi, np.unwrap(np.angle(Hejw4)))
axs[2,1].set_ylabel('$\\angle H(e^{{j\\omega}})$')
axs[2,1].grid(alpha=0.5)
axs[3,1].plot(ww[1:]/np.pi, -np.diff(np.unwrap(np.angle(Hejw4)))/dw)
axs[3,1].set_ylabel('$\\tau_g(\\omega)$')
axs[3,1].set_xlabel('$\\omega/\\pi$')
axs[3,1].set_ylim([-1, Nx2])
axs[3,1].grid(alpha=0.5)
plt.show()

# %%

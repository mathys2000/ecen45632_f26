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
# # FIR LPF Demo
#
# Demonstration of Gibbs' phenomenon for truncated ideal LPF

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
Nx1 = 51    # filter/window length
Nx2 = 511
M21 = (Nx1-1)/2     # offset to make hlp causal
M22 = (Nx2-1)/2
nn1 = np.arange(Nx1)     # index axis
nn2 = np.arange(Nx2)
wc = 0.4*np.pi     # cutoff frequency
Nw = 5000   # number of frequency points to evaluate

# %%
# Ideal LPF, truncated
hlp1 = wc/np.pi*np.sinc(wc/np.pi*(nn1-M21))
hlp2 = wc/np.pi*np.sinc(wc/np.pi*(nn2-M22))
ww, Hejw1 = ss.freqz(hlp1, 1, Nw)
ww, Hejw2 = ss.freqz(hlp2, 1, Nw)

# %%
# Time domain plot
fig, axs = plt.subplots(2, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
axs[0].stem(nn1-M21, hlp1, label=f'N={Nx1}')
axs[0].set_title(f'Truncated Impulse Response of Ideal LPF, $\\omega_c/\\pi$={wc/np.pi}')
axs[0].set_ylabel(f'$h[n]$')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(nn2-M22, hlp2)
axs[1].plot(nn2-M22, hlp2, label=f'N={Nx2}')
axs[1].set_ylabel(f'$h[n]$')
axs[1].set_xlabel(f'$n$')
axs[1].grid(alpha=0.5)
axs[1].legend()
plt.show()

# %%
# Frequency domain plot of LPF
fig, axs = plt.subplots(1, 1, figsize=fsz)
fig.canvas.toolbar_position = 'top'
axs.plot(ww/np.pi, np.abs(Hejw1), label=f'N={Nx1}')
axs.plot(ww/np.pi, np.abs(Hejw2), label=f'N={Nx2}')
axs.set_title(f'Truncation of $h[n]$: Ripple narrows as N increases, Overshoot persists')
axs.set_ylabel(f'$|H(e^{{j\\omega}})|$')
axs.set_xlabel(f'Normalized frequency $\\omega/\\pi$, $\\omega_c/\\pi$={wc/np.pi}')
axs.set_xlim([0.25, 0.55])
axs.grid(alpha=0.5)
axs.legend()
plt.show()

# %%

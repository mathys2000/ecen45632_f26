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
# # PS06_Problem 7
#
# Adapted from Problem 5.10 in Oppenheim/Schafer Book

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Parameters
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
zeros = 0.6*np.array([np.exp(1j*135/180*np.pi), np.exp(-1j*135/180*np.pi)])
poles = 0.75*np.array([np.exp(1j*np.pi), np.exp(1j*np.pi/3), np.exp(-1j*np.pi/3)])

# %%
# Pole-zero plot
fig, axs = plt.subplots(1, 1, figsize=(4,4))
#fig.canvas.toolbar_position = 'top'
axs.axhline(0, linewidth=0.7, color='black')
axs.axvline(0, linewidth=0.7, color='black')
axs.plot(cir.real, cir.imag, linestyle='--', linewidth=0.7, color='black')
#axs.plot(zeros.real, zeros.imag, marker='o', color='black')
axs.plot(zeros.real, zeros.imag, 'ok', label='zero')
#axs.plot(zeros.real, zeros.imag, marker='o', color='black')
axs.plot(poles.real, poles.imag, 'xk', label='pole')
axs.set_title(f'Pole-Zero Plot, z-Plane')
axs.set_ylabel('Im(z)')
axs.set_xlabel('Re(z)')
#axs.grid(alpha=0.5)
#axs.legend(loc=3)
axs.legend(loc=2)
axs.set_aspect('equal')
plt.show()

# %%

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
# # PS05_Problem4
#
# Prewarping design exercise

# %%
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

# %%
# Parameters
wp, ws = 0.3*np.pi, 0.6*np.pi
Ap, As = 1, 30
T = 1

# %%
# Prewarp and order estimate
Wp, Ws = 2/T*np.tan(wp/2), 2/T*np.tan(ws/2)
N = np.ceil(np.log10((10**(As/10)-1)/(10**(Ap/10)-1))/(2*np.log10(Ws/Wp)))

# %%
# Analog filter with prewarp
b_CT, a_CT = ss.butter(N, 1.15*Wp, analog=True, output='ba')

# %%
# Digital filter
ww = np.linspace(0, np.pi-1e-2, 1000)
b_DT, a_DT = ss.bilinear(b_CT, a_CT, fs=1/T)   # bilinear transform
Hejw = ss.freqz(b_DT, a_DT, ww)[1]   # DTFT

# %%
fig, axs = plt.subplots(2, 1, figsize=(7,6))
axs[0].plot(ww/np.pi, 20*np.log10(np.abs(Hejw)), label='Magnitude, enlarged')
axs[0].axhline(-Ap, color='k', linestyle='--', linewidth=0.7)
axs[0].axvline(wp/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[0].axvline(ws/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[0].set_title(f'Butterworth LPF, prewarped, $N$={N}, $\\omega_p/\\pi$={wp/np.pi}, $\\omega_s/\\pi$={ws/np.pi}, T={T}')
axs[0].set_ylabel('$|H(e^{{j\\omega}})|$')
axs[0].grid(alpha=0.5)
axs[0].set_ylim([-3, 1])
axs[0].legend()
axs[1].plot(ww/np.pi, 20*np.log10(np.abs(Hejw)), label='Magnitude')
axs[1].axhline(-Ap, color='k', linestyle='--', linewidth=0.7)
axs[1].axhline(-As, color='k', linestyle='--', linewidth=0.7)
axs[1].axvline(wp/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[1].axvline(ws/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[1].set_ylabel('$|H(e^{{j\\omega}})|$')
axs[1].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[1].grid(alpha=0.5)
axs[1].set_ylim([-80, 10])
axs[1].legend(loc=1)
plt.show()

# %%
fig, axs = plt.subplots(2, 1)
axs[0].plot(ww/np.pi, np.unwrap(np.angle(Hejw)), label='Phase [rad]')
axs[0].set_title(f'Butterworth LPF, prewarped, $N$={N}, $\\omega_p/\\pi$={wp/np.pi}, $\\omega_s/\\pi$={ws/np.pi}, T={T}')
axs[0].set_ylabel('$\\angle H(e^{{j\\omega}})$ [rad] (unwrapped)')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww/np.pi, ss.group_delay((b_DT,a_DT), ww)[1], label='Group Delay [samples]')
axs[1].set_ylabel('$\\tau_g(\\omega)$ [samples]')
axs[1].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[1].grid(alpha=0.5)
axs[1].set_ylim([0, 6])
axs[1].legend()
plt.show()

# %%
# Analog filter, no prewarp (np)
N_np = np.ceil(np.log10((10**(As/10)-1)/(10**(Ap/10)-1))/(2*np.log10(ws/wp)))
b_CT_np, a_CT_np = ss.butter(N_np, 1.15*wp, analog=True, output='ba')

# %%
# Digital filter, no prewarp
b_DT_np, a_DT_np = ss.bilinear(b_CT_np, a_CT_np, fs=1/T)   # bilinear transform
Hejw_np = ss.freqz(b_DT_np, a_DT_np, ww)[1]   # DTFT

# %%
fig, axs = plt.subplots(2, 1, figsize=(7,6))
axs[0].plot(ww/np.pi, 20*np.log10(np.abs(Hejw)), label='Magnitude, prewarped')
axs[0].plot(ww/np.pi, 20*np.log10(np.abs(Hejw_np)), label='Magnitude, no prewarp')
axs[0].axhline(-Ap, color='k', linestyle='--', linewidth=0.7)
axs[0].axvline(wp/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[0].axvline(0.955*wp/np.pi, color='r', linestyle='--', linewidth=0.7)
axs[0].axvline(ws/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[0].set_title(f'Butterworth LPF, $N$={N_np}, $\\omega_p/\\pi$={wp/np.pi}, $\\omega_s/\\pi$={ws/np.pi}, T={T}')
axs[0].set_ylabel('$|H(e^{{j\\omega}})|$')
axs[0].grid(alpha=0.5)
axs[0].set_ylim([-3, 1])
axs[0].legend()
axs[1].plot(ww/np.pi, 20*np.log10(np.abs(Hejw)), label='Magnitude, prewarped')
axs[1].plot(ww/np.pi, 20*np.log10(np.abs(Hejw_np)), label='Magnitude, no prewarp')
axs[1].axhline(-Ap, color='k', linestyle='--', linewidth=0.7)
axs[1].axhline(-As, color='k', linestyle='--', linewidth=0.7)
axs[1].axvline(wp/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[1].axvline(ws/np.pi, color='k', linestyle='--', linewidth=0.7)
axs[1].axvline(0.81*ws/np.pi, color='r', linestyle='--', linewidth=0.7)
axs[1].axvline(0.915*ws/np.pi, color='b', linestyle='--', linewidth=0.7)
axs[1].set_ylabel('$|H(e^{{j\\omega}})|$')
axs[1].set_xlabel('Normalized frequency $\\omega/\\pi$')
axs[1].grid(alpha=0.5)
axs[1].set_ylim([-80, 10])
axs[1].legend(loc=1)
plt.show()

# %%
2*np.tan(0.3*np.pi)

# %%
print(Wp, Ws)

# %%
(np.log10(999/(10**0.1-1)))/(2*np.log10(Ws/Wp))

# %%
np.log10((10**(As/10)-1)/(10**(Ap/10)-1))/(2*np.log10(ws/wp))

# %%

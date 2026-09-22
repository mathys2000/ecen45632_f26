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
# # Pole-Zero Geometry
#
# Simple one pole, one zero HPF

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
z1, p1 = 0.8, 0.5
#z1, p1 = 0.8, 0.95
b1 = [1, -z1]
a1 = [1, -p1]
b2 = [-z1, 1]
a2 = [1, -p1]
ww = np.linspace(0, np.pi, 1000)

# %%
# DT filter
Hejw1 = ss.freqz(b1, a1, ww)[1]   # DTFT
Hejw2 = ss.freqz(b2, a2, ww)[1]   # DTFT

# %%
# Plots
fig, axs = plt.subplots(3, 1, figsize=(8,6))
fig.canvas.toolbar_position = 'top'
axs[0].plot(ww/np.pi, 20*np.log10(np.abs(Hejw1)), label='Magnitude [dB]')
#axs[0].plot(ww/np.pi, 20*np.log10(np.abs(Hejw2)), linestyle='--', label='Magnitude [dB]')
axs[0].set_title(f'Example Pole-Zero System, Zero: $z$={z1}, Pole: $z$={p1}')
axs[0].set_ylabel('$|H(e^{{j\\omega}})|$ [dB]')
axs[0].grid(alpha=0.5)
axs[0].legend()
axs[1].plot(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw1)), label='Phase [deg]')
#axs[1].plot(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw2)), label='Phase [deg]')
axs[1].set_ylabel('$\\angle H(e^{{j\\omega}})$ [deg]')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww/np.pi, ss.group_delay((b1,a1), ww)[1], label='Group Delay [samples]')
#axs[2].plot(ww/np.pi, ss.group_delay((b2,a2), ww)[1], label='Group Delay [samples]')
axs[2].set_ylabel('$\\tau_g(\\omega)$ [samples]')
axs[2].set_xlabel('Normalized Frequency $\\omega/\\pi$')
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.show()

# %%
# Geometry
#w0_deg = 100               # w0 frequency in degrees
#w0_deg = 0
#w0_deg = 5
#w0_deg = 22               # w0 frequency in degrees
#w0_deg = 50
w0_deg = 120
#w0_deg = 180
w0 = w0_deg*np.pi/180         # w0 normalized test frequency in radians
ixw0 = np.argmin(np.abs(ww-w0))
ejw0 = np.exp(1j*w0)    
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
zeros1 = npp.polyroots(b1[::-1])    # zeros
poles1 = npp.polyroots(a1[::-1])    # poles
v_num1 = np.array([zeros1[0], ejw0])     # numerator vector
v_den1 = np.array([poles1[0], ejw0])     # denominator vector
v_num1_abs, v_num1_arg_deg = np.abs(np.diff(v_num1))[0], 180/np.pi*np.angle(np.diff(v_num1))[0]
v_den1_abs, v_den1_arg_deg = np.abs(np.diff(v_den1))[0], 180/np.pi*np.angle(np.diff(v_den1))[0]
print(f'num: {v_num1_abs:.3f}, {v_num1_arg_deg:.2f} [deg]')
print(f'den: {v_den1_abs:.3f}, {v_den1_arg_deg:0.2f} [deg]')
zeros2 = npp.polyroots(b2[::-1])    # zeros
poles2 = npp.polyroots(a2[::-1])    # poles


# %%
# Pole-zero plot geometry
fig, axs = plt.subplots(1, 1, figsize=(6,5))
fig.canvas.toolbar_position = 'top'
axs.axhline(0, linewidth=0.5, color='black')
axs.axvline(0, linewidth=0.5, color='black')
axs.plot(cir.real, cir.imag, linestyle='--', linewidth=0.7, color='black')
axs.plot(ejw0.real, ejw0.imag, marker='o', color='magenta', label='$e^{{j\\omega_0}}$')
axs.plot(zeros1.real, zeros1.imag, marker='o', color='blue', label='zero')
axs.plot(poles1.real, poles1.imag, marker='x', color='red', label='pole')
axs.plot(v_num1.real, v_num1.imag, color='blue')
axs.plot(v_den1.real, v_den1.imag, color='red')
#axs.set_title(f'Geometric Evaluation on Unit Circle, $\\omega_0$={w0_deg} [deg]')
axs.set_title(f'Geometric Evaluation on Unit Circle')
axs.set_ylabel('Imaginary')
axs.set_xlabel('Real')
axs.grid(alpha=0.5)
axs.legend(loc=3)
axs.set_aspect('equal')
plt.show()

# %%
# Pole-zero plot geometry
fig, axs = plt.subplots(1, 1, figsize=(6,5))
fig.canvas.toolbar_position = 'top'
axs.axhline(0, linewidth=0.5, color='black')
axs.axvline(0, linewidth=0.5, color='black')
axs.plot(cir.real, cir.imag, linestyle='--', linewidth=0.7, color='black')
axs.plot(zeros1.real, zeros1.imag, marker='o', color='blue')
axs.plot(poles1.real, poles1.imag, marker='x', color='red')
axs.plot(ejw0.real, ejw0.imag, marker='o', color='magenta', label='$e^{{j\\omega_0}}$')
axs.plot(v_num1.real, v_num1.imag, color='blue', label=f'{v_num1_abs:.3f} $\\angle${v_num1_arg_deg:.2f}$^o$')
axs.plot(v_den1.real, v_den1.imag, color='red', label=f'{v_den1_abs:.3f} $\\angle${v_den1_arg_deg:.2f}$^o$')
axs.set_title(f'Geometric Evaluation on Unit Circle, $\\omega_0$={w0_deg}$^o$')
axs.set_ylabel('Imaginary')
axs.set_xlabel('Real')
axs.grid(alpha=0.5)
axs.legend(loc=3)
axs.set_aspect('equal')
plt.show()

# %%
fig = plt.figure(figsize=(8,4), layout='constrained')
fig.canvas.toolbar_position = 'top'
axs = fig.add_gridspec(2, 2)
ax0x = fig.add_subplot(axs[:,0])
ax0x.axhline(0, linewidth=0.5, color='black')
ax0x.axvline(0, linewidth=0.5, color='black')
ax0x.plot(cir.real, cir.imag, linestyle='--', linewidth=0.7, color='black')
ax0x.plot(zeros1.real, zeros1.imag, marker='o', color='blue')
ax0x.plot(poles1.real, poles1.imag, marker='x', color='red')
ax0x.plot(ejw0.real, ejw0.imag, marker='o', color='magenta', label='$e^{{j\\omega_0}}$')
ax0x.plot(v_num1.real, v_num1.imag, color='blue', label=f'{v_num1_abs:.3f} $\\angle${v_num1_arg_deg:.2f}$^o$')
ax0x.plot(v_den1.real, v_den1.imag, color='red', label=f'{v_den1_abs:.3f} $\\angle${v_den1_arg_deg:.2f}$^o$')
ax0x.set_title(f'Geometric Evaluation on Unit Circle, $\\omega_0$={w0_deg}$^o$')
ax0x.set_ylabel('Imaginary')
ax0x.set_xlabel('Real')
ax0x.grid(alpha=0.5)
ax0x.legend(loc=3)
ax0x.set_aspect('equal')
ax01 = fig.add_subplot(axs[0,1])
ax01.axvline(w0/np.pi, linestyle='--', color='magenta')
ax01.plot(ww/np.pi, 20*np.log10(np.abs(Hejw1)), label=f'$|H(e^{{j\\omega_0}})|$={np.abs(Hejw1[ixw0]):.3f}')
ax01.set_title(f'$H(z)$, Zero: $z$={z1}, Pole: $z$={p1}')
ax01.set_ylabel('$|H(e^{{j\\omega}})|$ [dB]')
ax01.grid(alpha=0.8)
ax01.legend()
ax11 = fig.add_subplot(axs[1,1])
ax11.axvline(w0/np.pi, linestyle='--', color='magenta')
ax11.plot(ww/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw1)), label=f'$\\angle H(e^{{j\\omega_0}})$={180/np.pi*np.angle(Hejw1[ixw0]):.2f}')
ax11.set_ylabel('$\\angle H(e^{{j\\omega}})$ [deg]')
ax11.set_xlabel('Normalized Frequency $\\omega/\\pi$')
ax11.grid(alpha=0.8)
ax11.legend()
plt.show()

# %%

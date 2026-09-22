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
# # First Order Allpass
#
# First order allpass filter

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
ap = 0.7
#ap = 0.9
b_ap = np.array([-ap, 1.0])
a_ap = np.array([1.0, -ap])

# %%
# Frequency response
ww = np.linspace(0, np.pi, 10000)
_, Hejw_ap = ss.freqz(b_ap, a_ap, ww)
_, gd_ap = ss.group_delay((b_ap, a_ap), ww)


# %%
# Plot frequency response
fig, axs = plt.subplots(3, 1, figsize=(7,6))
fig.canvas.toolbar_position = 'top'
axs[0].plot(ww/np.pi, np.abs(Hejw_ap), label='Magnitude')
axs[0].set_title(f'First Order All-Pass, a={ap}')
axs[0].set_ylabel('$|H_{{ap}}(e^{{j\\omega}})|$')
axs[0].grid(alpha=0.5)
axs[0].set_ylim([0, 1.1])
axs[0].legend(loc=4)
axs[1].plot(ww[:-2]/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_ap[:-2])), label='Phase [deg]')
axs[1].set_ylabel('$\\angle H_{{ap}}(e^{{j\\omega}})$ [deg]')
axs[1].grid(alpha=0.5)
axs[1].legend()
axs[2].plot(ww[:-2]/np.pi, gd_ap[:-2], label='Group Delay [samples]')
axs[2].set_ylabel('$\\tau_g(\\omega)$')
axs[2].set_xlabel('Normalized Frequency $\\omega/\\pi$')
#axs[2].set_ylim([-5, 100])
axs[2].grid(alpha=0.5)
axs[2].legend()
plt.show()

# %%
# Geometry
#w0_deg = 100               # w0 frequency in degrees
#w0_deg = 0
#w0_deg = 5
#w0_deg = 22               # w0 frequency in degrees
w0_deg = 50
#w0_deg = 120
#w0_deg = 180
w0 = w0_deg*np.pi/180         # w0 normalized test frequency in radians
ixw0 = np.argmin(np.abs(ww-w0))
ejw0 = np.exp(1j*w0)    
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
zeros1 = npp.polyroots(b_ap[::-1])    # zeros
poles1 = npp.polyroots(a_ap[::-1])    # poles
v_num1 = np.array([ejw0, zeros1[0]])     # numerator vector
v_den1 = np.array([poles1[0], ejw0])     # denominator vector
v_num1_abs, v_num1_arg_deg = np.abs(np.diff(v_num1))[0], 180/np.pi*np.angle(np.diff(v_num1))[0]
v_den1_abs, v_den1_arg_deg = np.abs(np.diff(v_den1))[0], 180/np.pi*np.angle(np.diff(v_den1))[0]
print(f'num: {ap*v_num1_abs:.3f}, {v_num1_arg_deg:.2f} [deg]')
print(f'den: {v_den1_abs:.3f}, {v_den1_arg_deg:0.2f} [deg]')


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
ax0x.plot(v_num1.real, v_num1.imag, color='blue', label=f'{ap*v_num1_abs:.3f} $\\angle${v_num1_arg_deg:.2f}$^o$')
ax0x.plot(v_den1.real, v_den1.imag, color='red', label=f'{v_den1_abs:.3f} $\\angle${v_den1_arg_deg:.2f}$^o$')
ax0x.set_title(f'Geometric Evaluation on Unit Circle, $\\omega_0$={w0_deg}$^o$')
ax0x.set_ylabel('Imaginary')
ax0x.set_xlabel('Real')
ax0x.grid(alpha=0.5)
ax0x.legend(loc=3)
ax0x.set_aspect('equal')
ax01 = fig.add_subplot(axs[0,1])
ax01.axvline(w0/np.pi, linestyle='--', color='magenta')
ax01.plot(ww/np.pi, np.abs(Hejw_ap), label=f'$|H_{{ap}}(e^{{j\\omega_0}})|$={np.abs(Hejw_ap[ixw0]):.3f}')
ax01.set_title(f'First Order All-Pass, a={ap}')
ax01.set_ylabel('$|H_{{ap}}(e^{{j\\omega}})|$')
ax01.set_ylim([0, 1.1])
ax01.grid(alpha=0.8)
ax01.legend()
ax11 = fig.add_subplot(axs[1,1])
ax11.axvline(w0/np.pi, linestyle='--', color='magenta')
ax11.plot(ww[:-2]/np.pi, 180/np.pi*np.unwrap(np.angle(Hejw_ap[:-2])),
          label=f'$\\angle H_{{ap}}(e^{{j\\omega_0}})$={180/np.pi*np.angle(Hejw_ap[ixw0]):.2f}')
ax11.set_ylabel('$\\angle H_{{ap}}(e^{{j\\omega}})$ [deg]')
ax11.set_xlabel('Normalized Frequency $\\omega/\\pi$')
ax11.grid(alpha=0.8)
ax11.legend()
plt.show()

# %%
180/np.pi*ww[ixw0-1]

# %%

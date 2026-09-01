# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lecture 8 — IIR Design, Bilinear Transformation, and ML Preprocessing

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# %% [markdown]
# ## 1. Analog Butterworth prototype
#
# $$|H_c(j\Omega)|^2=\frac{1}{1+(\Omega/\Omega_c)^{2N}}.$$

# %%
N=4; Oc=1.0
Om=np.logspace(-1,1,1000)
mag=1/np.sqrt(1+(Om/Oc)**(2*N))
plt.figure(figsize=(9,4)); plt.semilogx(Om,20*np.log10(mag)); plt.axvline(Oc,ls='--'); plt.grid(alpha=.25,which='both'); plt.ylabel('Magnitude (dB)'); plt.xlabel(r'$\Omega$'); plt.show()

# %% [markdown]
# ## 2. Bilinear transform and frequency warping
#
# $$s=\frac{2}{T}\frac{1-z^{-1}}{1+z^{-1}},$$
#
# which maps the analog imaginary axis to the digital unit circle. Substituting $z=e^{j\omega}$ gives
#
# $$\Omega=\frac{2}{T}\tan\left(\frac{\omega}{2}\right).$$

# %%
omega=np.linspace(0,.98*np.pi,1000); T=1.0
Omega=2/T*np.tan(omega/2)
plt.figure(figsize=(9,4)); plt.plot(omega/np.pi,Omega); plt.ylim(0,20); plt.grid(alpha=.25); plt.xlabel(r'$\omega/\pi$'); plt.ylabel(r'$\Omega$'); plt.show()

# %% [markdown]
# ## 3. Prewarped Butterworth example
#
# Desired digital cutoff: $\omega_c=0.4\pi$. We first compute
#
# $$\Omega_c=2\tan(0.2\pi).$$

# %%
wc=.4*np.pi; Oc=2*np.tan(wc/2)
print('prewarped cutoff =',Oc)
ba,aa=signal.butter(4,Oc,analog=True)
bz,az=signal.bilinear(ba,aa,fs=1.0)
w,H=signal.freqz(bz,az,worN=4096)
plt.figure(figsize=(9,4)); plt.plot(w/np.pi,20*np.log10(np.maximum(abs(H),1e-10))); plt.axvline(.4,ls='--'); plt.axhline(-3.0103,ls=':'); plt.ylim(-80,3); plt.grid(alpha=.25); plt.xlabel(r'$\omega/\pi$'); plt.ylabel('Magnitude (dB)'); plt.show()

# %% [markdown]
# ## 4. Causal IIR versus forward-backward filtering
#
# For offline preprocessing, forward-backward filtering cancels phase but produces an effective magnitude approximately $|H(e^{j\omega})|^2$ and cannot be used as a causal streaming operation.

# %%
rng=np.random.default_rng(3); fs=1000.; t=np.arange(0,1,1/fs)
clean=np.exp(-((t-.42)/.018)**2)+.35*np.sin(2*np.pi*12*t)
noisy=clean+.25*np.sin(2*np.pi*150*t)+.05*rng.standard_normal(t.size)
sos=signal.butter(4,40,btype='low',fs=fs,output='sos')
causal=signal.sosfilt(sos,noisy); zero=signal.sosfiltfilt(sos,noisy)
mask=(t>.32)&(t<.53)
plt.figure(figsize=(9,4)); plt.plot(t[mask],clean[mask],label='reference'); plt.plot(t[mask],causal[mask],label='causal IIR'); plt.plot(t[mask],zero[mask],label='forward-backward'); plt.grid(alpha=.25); plt.legend(); plt.show()

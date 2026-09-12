# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ecen45632 [conda env:ecen45632]
#     language: python
#     name: conda-env-ecen45632-ecen45632
# ---

# %% [markdown]
# # Quad Filter Demo
#
# Visualize and listen to simple 2'nd order filters

# %%
import numpy as np
import scipy.signal as ss
import numpy.polynomial.polynomial as npp
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy.io import wavfile


# %%
# Read audio signal
#Fs, audio16 = wavfile.read('audio/Zest_8000_mono.wav')
Fs, audio16 = wavfile.read('audio/Zest_8000_mono_800tone.wav')
audio = audio16.astype(np.float32)/2**15
tt = np.arange(audio.size)/Fs    # time axis

# %%
# Parameters
num_type = 'ccz'       # numerator type: 'rez' for real zeros, 'ccz' for complex-conjugate zeros
# 'rez' parameters
r1z, r2z = 0, -0.3      # numerator r1, r2
#r1z, r2z = -1, 1      # numerator r1, r2
#r1z, r2z = -1, -1      # numerator r1, r2
# 'ccz' parameters
#rhoz, thz = 1, 0     # numerator rho, theta (deg)
rhoz, thz = 1, 36     # numerator rho, theta (deg)
#rhoz, thz = 1, 180     # numerator rho, theta (deg)

den_type = 'ccp'       # Denominator type: 'rep' for real poles, 'ccp' for complex-conjugate poles
# 'rep' parameters
r1p, r2p = 0.4, 0.8    # denominator r1, r2
# 'ccp' parameters
rhop, thp = 0.98, 36     # denominator rho, theta (deg)

# %%
# Generate filter
if num_type=='ccz':
    bz = [1, -2*rhoz*np.cos(np.pi/180*thz), rhoz**2]     # numerator
else:
    bz = [1, -(r1z+r2z), r1z*r2z]

if den_type=='ccp':    
    ap = [1, -2*rhop*np.cos(np.pi/180*thp), rhop**2]     # denominator
else:
    ap = [1, -(r1p+r2p), r1p*r2p]

# Frequency response
ww, Hejw = ss.freqz(bz, ap, worN=500)

# Poles and zeros
cir = np.exp(2j*np.pi*np.arange(360)/360.0)  # unit circle
z_roots = npp.polyroots(bz[::-1])    # numerator roots (zeros)
p_roots = npp.polyroots(ap[::-1])    # denominator roots (poles)

# %%
#fig, axs = plt.subplots(2, 2, figsize=(7, 4), layout='constrained')
fig = plt.figure(figsize=(7, 4), layout='constrained')
axs = fig.add_gridspec(2, 2)
ax00 = fig.add_subplot(axs[0,0])
ax00.plot(ww, np.abs(Hejw), '-b', label='Magnitude')
ax00.grid(alpha=0.5)
ax00.legend()
ax10 = fig.add_subplot(axs[1,0])
ax10.plot(ww, 180/np.pi*np.angle(Hejw), '-r', label='Phase [deg]')
ax10.grid(alpha=0.5)
ax10.legend()
ax10.set_xlabel('$\\omega$')
axx1 = fig.add_subplot(axs[:,1])
axx1.plot(cir.real, cir.imag, '--k', linewidth=0.5)
axx1.plot(z_roots.real, z_roots.imag, 'ob')
axx1.plot(p_roots.real, p_roots.imag, 'xr')
axx1.set_title('Pole-Zero Plot')
axx1.grid(alpha=0.5)
axx1.set_aspect('equal')
plt.show()


# %%
# Filter audio signal
audio_filt = ss.lfilter(bz, ap, audio)
#audio_filt = 0.25*np.cos(2*np.pi*800*tt) + audio  # add 800 hz tone 

# %%
# Original signal
ipd.Audio(audio, rate=Fs)

# %%
# Filtered and normalized signal
audio_filt_norm = 0.8*audio_filt/np.max(abs(audio_filt))   # normalized, scaled voice signal
ipd.Audio(audio_filt_norm, rate=Fs)

# %%
# Write 16-bit .wav file
audio16_filt = np.array(2**15*audio_filt_norm, np.int16)
wavfile.write('audio/audio_filt.wav', Fs, audio16_filt)
#wavfile.write('audio/Zest_8000_mono_800tone.wav', Fs, audio16_filt)


# %%

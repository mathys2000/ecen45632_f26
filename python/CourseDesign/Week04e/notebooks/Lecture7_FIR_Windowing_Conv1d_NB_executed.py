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
# # Lecture 7 — FIR Filter Design, Windowing, and 1D Convolutions
#
# This notebook connects classical window-method FIR design with fixed-weight PyTorch `Conv1d` operations.

# %%
from pathlib import Path
import sys
BASE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(BASE))
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn.functional as F
from lecture7_fir_windowing_conv1d import ideal_lowpass

# %% [markdown]
# ## 1. Ideal lowpass impulse response
#
# For cutoff $\omega_c$,
#
# $$h_d[n]=\begin{cases}\dfrac{\sin(\omega_c n)}{\pi n},&n\neq0,\\\dfrac{\omega_c}{\pi},&n=0.\end{cases}$$

# %%
wc = 0.4*np.pi
n = np.arange(-30,31)
h = np.empty_like(n,dtype=float)
h[n==0] = wc/np.pi
h[n!=0] = np.sin(wc*n[n!=0])/(np.pi*n[n!=0])
plt.figure(figsize=(9,4))
plt.stem(n,h,basefmt=" ")
plt.xlabel("n"); plt.ylabel("h_d[n]"); plt.grid(alpha=.25); plt.show()

# %% [markdown]
# ## 2. Window method
#
# A practical FIR is obtained from
#
# $$h_w[n]=h_d[n]w[n].$$
#
# The window controls the transition-width / sidelobe tradeoff.

# %%
N=51
for name in ["boxcar","bartlett","hann","hamming","blackman"]:
    h=ideal_lowpass(N,wc,name)
    w,H=signal.freqz(h,worN=4096)
    plt.plot(w/np.pi,20*np.log10(np.maximum(abs(H),1e-8)),label=name)
plt.ylim(-100,5); plt.xlim(0,1); plt.grid(alpha=.25); plt.legend();
plt.xlabel(r"$\omega/\pi$"); plt.ylabel("Magnitude (dB)"); plt.show()

# %% [markdown]
# ## 3. Fixed FIR as PyTorch `Conv1d`
#
# Textbook convolution is
#
# $$y[n]=\sum_{k=0}^{M}h[k]x[n-k].$$
#
# PyTorch `conv1d` uses cross-correlation ordering, so we flip the FIR coefficients and left-pad the signal to reproduce causal DSP convolution.

# %%
fs=8000
t=np.arange(0,.03,1/fs)
x=(np.sin(2*np.pi*400*t)+.4*np.sin(2*np.pi*2000*t)).astype(np.float32)
h=ideal_lowpass(31,.3*np.pi,"hamming").astype(np.float32)
y_np=np.convolve(x,h,mode="full")[:len(x)]
xt=torch.from_numpy(x).view(1,1,-1)
ht=torch.from_numpy(h)
kernel=torch.flip(ht,[0]).view(1,1,-1)
y_t=F.conv1d(F.pad(xt,(len(h)-1,0)),kernel).view(-1).numpy()
print("max difference =",np.max(np.abs(y_np-y_t)))
plt.figure(figsize=(9,4)); plt.plot(y_np[:150],label="NumPy"); plt.plot(y_t[:150],'--',label="PyTorch"); plt.legend(); plt.grid(alpha=.25); plt.show()

# %% [markdown]
# ## 4. Fixed versus learnable coefficients
#
# Traditional DSP supplies $h[k]$ from a design procedure. A neural convolution layer treats the kernel coefficients as parameters and changes them using gradients.

# %%
conv=torch.nn.Conv1d(1,1,kernel_size=len(h),bias=False)
with torch.no_grad():
    conv.weight[:] = kernel
print("learnable by default:",conv.weight.requires_grad)
conv.weight.requires_grad_(False)
print("frozen fixed DSP front-end:",conv.weight.requires_grad)

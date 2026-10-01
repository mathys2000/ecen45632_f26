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
# # Week 6 — DFT, Fast Convolution, STFT, and Spectrograms
#
# These demos support Lectures 11 and 12.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. DFT samples the finite-record DTFT

# %%
N=32
n=np.arange(N)
x=np.cos(0.31*np.pi*n)+0.5*np.cos(0.62*np.pi*n)
w=np.linspace(0,2*np.pi,3000,endpoint=False)
DT=np.array([np.sum(x*np.exp(-1j*ww*n)) for ww in w])
X=np.fft.fft(x)
plt.figure(figsize=(8,3)); plt.plot(w/np.pi,np.abs(DT),label='DTFT'); plt.scatter((2*np.pi*np.arange(N)/N)/np.pi,np.abs(X),s=18,label='DFT'); plt.xlabel('ω/π'); plt.ylabel('magnitude'); plt.legend(); plt.show()

# %% [markdown]
# ## 2. Linear versus circular convolution

# %%
x=np.array([1.,2.,-1.,1.]); h=np.array([1.,1.,0.5])
y_lin=np.convolve(x,h)
N=4
y_circ=np.fft.ifft(np.fft.fft(x,N)*np.fft.fft(h,N)).real
Nok=len(x)+len(h)-1
y_fft=np.fft.ifft(np.fft.fft(x,Nok)*np.fft.fft(h,Nok)).real
print('linear:',y_lin)
print('N=4 circular:',y_circ)
print('properly padded FFT:',y_fft)


# %% [markdown]
# ## 3. Overlap-add and overlap-save implementations

# %%

def fft_linear_convolution(x, h, nfft=None):
    x=np.asarray(x); h=np.asarray(h)
    need=len(x)+len(h)-1
    if nfft is None:
        nfft=1 << int(np.ceil(np.log2(need)))
    if nfft < need:
        raise ValueError(f"nfft must be at least {need}")
    y=np.fft.irfft(np.fft.rfft(x,nfft)*np.fft.rfft(h,nfft),nfft)
    return y[:need]


def overlap_add(x,h,block_size=256,nfft=None):
    x=np.asarray(x,float); h=np.asarray(h,float)
    if nfft is None:
        need=block_size+len(h)-1
        nfft=1 << int(np.ceil(np.log2(need)))
    if nfft < block_size+len(h)-1:
        raise ValueError('nfft too small for overlap-add')
    H=np.fft.rfft(np.pad(h,(0,nfft-len(h))))
    y=np.zeros(len(x)+len(h)-1)
    for start in range(0,len(x),block_size):
        xb=x[start:start+block_size]
        X=np.fft.rfft(np.pad(xb,(0,nfft-len(xb))))
        yb=np.fft.irfft(X*H,nfft)
        end=min(start+nfft,len(y))
        y[start:end]+=yb[:end-start]
    return y


def overlap_save(x,h,nfft=512):
    x=np.asarray(x,float); h=np.asarray(h,float); M=len(h)
    if nfft < M:
        raise ValueError('nfft must be >= filter length')
    L=nfft-M+1
    H=np.fft.rfft(np.pad(h,(0,nfft-M)))
    padded=np.concatenate([np.zeros(M-1),x,np.zeros(M-1)])
    out=[]
    for start in range(0,len(padded)-M+1,L):
        block=padded[start:start+nfft]
        if len(block)<nfft: block=np.pad(block,(0,nfft-len(block)))
        z=np.fft.irfft(np.fft.rfft(block)*H,nfft)
        out.append(z[M-1:])
    y=np.concatenate(out)
    return y[:len(x)+M-1]




# %%
rng=np.random.default_rng(2)
x=rng.normal(size=3000); h=signal.firwin(129,0.18)
direct=np.convolve(x,h)
one=fft_linear_convolution(x,h); ola=overlap_add(x,h,384); ols=overlap_save(x,h,512)
print('one-shot FFT error:',np.max(np.abs(one-direct)))
print('OLA error:',np.max(np.abs(ola-direct)))
print('OLS error:',np.max(np.abs(ols-direct)))

# %% [markdown]
# ## 4. Non-stationary signal: global spectrum versus STFT

# %%
fs=4000
t=np.arange(0,1.2,1/fs)
x=signal.chirp(t,f0=150,f1=1500,t1=1.2)+0.1*np.random.default_rng(3).normal(size=len(t))
F=np.fft.rfftfreq(len(x),1/fs); P=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
plt.figure(figsize=(8,3)); plt.semilogy(F,P+1e-10); plt.xlim(0,1800); plt.title('Global spectrum'); plt.xlabel('Hz'); plt.show()
f,tt,Z=signal.stft(x,fs=fs,window='hann',nperseg=256,noverlap=192,nfft=512,boundary=None)
plt.figure(figsize=(8,3)); plt.pcolormesh(tt,f,20*np.log10(np.abs(Z)+1e-6),shading='auto'); plt.ylim(0,1800); plt.xlabel('time (s)'); plt.ylabel('Hz'); plt.title('STFT / spectrogram view'); plt.colorbar(label='dB'); plt.show()

# %% [markdown]
# ## 5. Window length: time–frequency trade-off

# %%
fig,axs=plt.subplots(1,3,figsize=(12,3),sharey=True)
for ax,L in zip(axs,[64,256,768]):
    f0,t0,Z0=signal.stft(x,fs=fs,window='hann',nperseg=L,noverlap=int(.75*L),nfft=max(1024,L),boundary=None)
    ax.pcolormesh(t0,f0,20*np.log10(np.abs(Z0)+1e-6),shading='auto'); ax.set_ylim(0,1800); ax.set_title(f'window={L}'); ax.set_xlabel('s')
axs[0].set_ylabel('Hz'); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. PyTorch spectrogram tensor and tiny CNN

# %%
try:
    import torch
    from torch import nn
    xt=torch.tensor(x,dtype=torch.float32).unsqueeze(0)
    win=torch.hann_window(256)
    Zt=torch.stft(xt,n_fft=512,hop_length=64,win_length=256,window=win,return_complex=True)
    S=torch.log1p(torch.abs(Zt)**2).unsqueeze(1)
    net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(16,3))
    logits=net(S)
    print('spectrogram tensor:',S.shape)
    print('CNN logits:',logits.shape)
except Exception as e:
    print('PyTorch demo skipped:',e)

# %% [markdown]
# ## 7. Suggested experiments
#
# 1. Reduce the OLA FFT length below the required value and explain the error.
# 2. Compare 10-ms and 100-ms STFT windows on a short transient.
# 3. Change hop length while holding the window fixed; distinguish denser frame sampling from improved fundamental resolution.
# 4. Try log-power versus linear-power spectrogram inputs and compare their dynamic ranges.

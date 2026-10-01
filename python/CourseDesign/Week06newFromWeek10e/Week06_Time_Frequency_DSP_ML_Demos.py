"""Week 6: DFT, fast convolution, STFT, spectrograms, and PyTorch bridge."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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


def demo_fast_convolution():
    rng=np.random.default_rng(2)
    x=rng.normal(size=3000)
    h=signal.firwin(129,0.18)
    direct=np.convolve(x,h)
    one=fft_linear_convolution(x,h)
    ola=overlap_add(x,h,block_size=384)
    ols=overlap_save(x,h,nfft=512)
    print('max errors vs direct:',np.max(abs(one-direct)),np.max(abs(ola-direct)),np.max(abs(ols-direct)))


def demo_stft():
    fs=4000
    t=np.arange(0,1.2,1/fs)
    x=signal.chirp(t,150,t[-1],1500)+0.1*np.random.default_rng(3).normal(size=len(t))
    f,tt,Z=signal.stft(x,fs=fs,window='hann',nperseg=256,noverlap=192,nfft=512,boundary=None)
    plt.figure(); plt.pcolormesh(tt,f,20*np.log10(abs(Z)+1e-6),shading='auto'); plt.ylim(0,1800)
    plt.xlabel('time (s)'); plt.ylabel('Hz'); plt.title('Chirp spectrogram'); plt.colorbar(label='dB'); plt.show()
    return x,fs


def torch_spectrogram_and_cnn(x,fs):
    try:
        import torch
        from torch import nn
    except Exception as e:
        print('PyTorch unavailable:',e); return
    xt=torch.tensor(x,dtype=torch.float32).unsqueeze(0)
    win=torch.hann_window(256)
    Z=torch.stft(xt,n_fft=512,hop_length=64,win_length=256,window=win,return_complex=True)
    S=torch.log1p(torch.abs(Z)**2).unsqueeze(1)
    net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(16,3))
    y=net(S)
    print('spectrogram tensor:',tuple(S.shape),'logits:',tuple(y.shape))


if __name__=='__main__':
    demo_fast_convolution()
    x,fs=demo_stft()
    torch_spectrogram_and_cnn(x,fs)

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
# # Week 3 — z-Transform, ROC, Transfer Functions, and Pole–Zero Geometry
#
# These examples support Lectures 5–6. Predict each result from the math **before** running the code.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import Audio, display


# %% [markdown]
# ## Lecture 5.1 — Why the z-transform can converge when the DTFT does not
# For $x[n]=2^n u[n]$, the z-transform series is $\sum_{n=0}^\infty (2/z)^n$. Convergence requires $|z|>2$.

# %%
def partial_sum(a=2.0, r=1.0, omega=0.0, N=30):
    n=np.arange(N); z=r*np.exp(1j*omega); return np.cumsum((a**n)*z**(-n))

plt.figure(figsize=(8,4))
for r in [1.0,2.1,2.5,3.0]:
    s=partial_sum(r=r)
    plt.plot(np.arange(1,len(s)+1),np.abs(s),label=f'r={r}')
plt.yscale('log');plt.grid(True,alpha=.25);plt.xlabel('N');plt.ylabel('|partial sum|');plt.legend();plt.show()

# %% [markdown]
# **Good interpretation:** the DTFT evaluates at $r=1$, outside the ROC here. Moving radially outward adds enough $r^{-n}$ damping to make the geometric series converge.

# %% [markdown]
# ## Lecture 5.2 — Same rational expression, different ROC
# Both $a^n u[n]$ and $-a^n u[-n-1]$ have algebraic expression $1/(1-az^{-1})$. The ROC tells us which power-series expansion—and therefore which time support—is intended.

# %%
a=.7
n=np.arange(-12,16)
x_right=(a**n)*(n>=0)
x_left=-(a**n)*(n<=-1)
fig,axs=plt.subplots(1,2,figsize=(10,3))
axs[0].stem(n,x_right);axs[0].set_title('right-sided: ROC |z|>.7')
axs[1].stem(n,x_left);axs[1].set_title('left-sided: ROC |z|<.7')
for ax in axs: ax.grid(True,alpha=.2);ax.set_xlabel('n')
plt.show()

# %% [markdown]
# ## Lecture 5.3 — Partial fractions with `scipy.signal.residuez`

# %%
b=[1.0]
a=np.convolve([1,-.5],[1,-.8])
r,p,k=signal.residuez(b,a)
print('a =',a)
print('residues =',r)
print('poles =',p)
print('direct =',k)

# %% [markdown]
# For ROC $|z|>0.8$, both terms are right-sided. For $0.5<|z|<0.8$, the 0.5-pole term is right-sided and the 0.8-pole term is left-sided. `residuez` gives the algebra; **you supply the ROC interpretation**.

# %% [markdown]
# ## Lecture 6.1 — Difference equation to poles and zeros
# Example: $y[n]-1.2y[n-1]+0.32y[n-2]=x[n]+0.3x[n-1]$.

# %%
b=np.array([1.,.3]);a=np.array([1.,-1.2,.32])
print('zeros:',np.roots(b))
print('poles:',np.roots(a))


# %%
def pzplot(b,a,title='Pole-zero plot'):
    z=np.roots(b) if len(b)>1 else np.array([]);p=np.roots(a) if len(a)>1 else np.array([])
    fig,ax=plt.subplots(figsize=(5,5));ax.add_patch(plt.Circle((0,0),1,fill=False,ls='--'));
    if len(z): ax.scatter(z.real,z.imag,s=90,facecolors='none',edgecolors='C0',linewidths=2,label='zeros')
    if len(p): ax.scatter(p.real,p.imag,s=90,marker='x',linewidths=2,label='poles')
    ax.axhline(0,lw=.7);ax.axvline(0,lw=.7);ax.set_aspect('equal');ax.set_xlim(-1.25,1.25);ax.set_ylim(-1.25,1.25);ax.grid(True,alpha=.2);ax.set_title(title);ax.legend();plt.show()

pzplot(b,a)


# %% [markdown]
# ## Lecture 6.2 — Geometric evaluation: compare distances with `freqz`

# %%
def geometric_magnitude(b,a,omega):
    z=np.roots(b) if len(b)>1 else np.array([]);p=np.roots(a) if len(a)>1 else np.array([]);q=np.exp(1j*omega)
    num=abs(b[0])*np.prod(np.abs(q-z)) if len(z) else abs(b[0])
    den=abs(a[0])*np.prod(np.abs(q-p)) if len(p) else abs(a[0])
    return num/den

b=np.array([1.,1.]);a=np.array([1.,-.75])
for omega in [0,.25*np.pi,.5*np.pi,np.pi]:
    _,H=signal.freqz(b,a,worN=[omega])
    print(f'omega/pi={omega/np.pi:.2f}: geometry={geometric_magnitude(b,a,omega):.6f}, freqz={abs(H[0]):.6f}')

# %% [markdown]
# ## Lecture 6.3 — Pole radius sandbox
# The pole angle sets the resonance center. Moving the pole closer to the unit circle sharpens the response and lengthens the decay.

# %%
theta=.35*np.pi
plt.figure(figsize=(8,4))
for rr in [.6,.85,.95,.985]:
    poles=rr*np.exp(1j*np.array([theta,-theta]));a=np.poly(poles).real;b=[1.]
    w,H=signal.freqz(b,a,worN=4096);H=H/np.max(np.abs(H))
    plt.plot(w/np.pi,20*np.log10(np.maximum(np.abs(H),1e-5)),label=f'r={rr}')
plt.ylim(-60,2);plt.grid(True,alpha=.25);plt.legend();plt.xlabel('omega/pi');plt.ylabel('normalized dB');plt.show()


# %% [markdown]
# ## Lecture 6.4 — 60 Hz notch filter

# %%
def notch_coefficients(f0=60.,fs=8000.,r=.985):
    w0=2*np.pi*f0/fs; z=np.exp(1j*np.array([w0,-w0])); p=r*z
    b=np.poly(z).real;a=np.poly(p).real;b*=np.sum(a)/np.sum(b);return b,a

fs=8000.;b,a=notch_coefficients(fs=fs)
f,H=signal.freqz(b,a,worN=32768,fs=fs)
plt.figure(figsize=(8,4));plt.plot(f,20*np.log10(np.maximum(np.abs(H),1e-8)));plt.xlim(0,300);plt.ylim(-90,3);plt.axvline(60,ls='--');plt.grid(True,alpha=.25);plt.show()
pzplot(b,a,'60 Hz notch')

# %% [markdown]
# ## Lecture 6.5 — Listen to a synthetic hum-contaminated signal

# %%
duration=3;t=np.arange(int(fs*duration))/fs
voice=(.35*np.sin(2*np.pi*220*t)+.18*np.sin(2*np.pi*440*t)+.10*np.sin(2*np.pi*660*t))*(.75+.25*np.sin(2*np.pi*2.3*t))
hum=.28*np.sin(2*np.pi*60*t)+.08*np.sin(2*np.pi*120*t)
x=voice+hum;y=signal.lfilter(b,a,x)
print('Before:');display(Audio(x,rate=int(fs)))
print('After 60 Hz notch:');display(Audio(y,rate=int(fs)))

# %% [markdown]
# ## ML bridge
# A pole/zero front end changes the spectral structure presented to a model. A notch may remove a known nuisance; a resonator may emphasize a band; an unstable filter can destroy all downstream features. Treat front-end DSP as part of the ML system, not just preprocessing.

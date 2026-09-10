# Week 6: Multi-rate DSP and ML Demos

# %% [markdown]
# # Week 6: Multi-rate DSP and ML Demos
# 
# Companion notebook for:
# 
# - **Lecture 11:** Downsampling, Decimation, and CNN Strides
# - **Lecture 12:** Upsampling, Interpolation, Rational Rate Changes, and Transposed Convolutions
# 
# The notebook uses NumPy/SciPy for classical DSP and PyTorch for the ML connections. Each section is designed to be runnable live in class.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn as nn

np.set_printoptions(precision=3, suppress=True)
torch.manual_seed(7)

print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)

# %% [markdown]
# ## 1. Downsampling in the time domain
# 
# For downsampling by an integer factor $M$,
# 
# $$x_d[n] = x[nM].$$
# 
# Only every $M$th input sample is retained.

# %%
M = 3
n = np.arange(30)
x = np.sin(0.18*np.pi*n) + 0.35*np.sin(0.62*np.pi*n)
xd = x[::M]

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.stem(n, x, basefmt=" ", label="x[n]")
ax.scatter(n[::M], xd, s=70, facecolors="none", edgecolors="tab:red", linewidths=2,
           label="samples retained")
ax.set_title(f"Downsampling by M = {M}")
ax.set_xlabel("n")
ax.set_ylabel("amplitude")
ax.grid(alpha=0.25)
ax.legend()
plt.show()

print("First 8 output samples:", xd[:8])

# %% [markdown]
# ## 2. A tone that aliases after downsampling
# 
# A digital frequency $\omega_0$ becomes $M\omega_0$ modulo $2\pi$ after downsampling by $M$. This makes aliasing easy to demonstrate with a single sinusoid.

# %%
M = 2
N = 64
n = np.arange(N)
omega0 = 0.72*np.pi
x = np.cos(omega0*n)
xd = x[::M]

# The new normalized frequency is M*omega0, wrapped to [-pi, pi).
omega_alias = ((M*omega0 + np.pi) % (2*np.pi)) - np.pi
print(f"Input frequency = {omega0/np.pi:.2f}π rad/sample")
print(f"Aliased output frequency = {abs(omega_alias)/np.pi:.2f}π rad/output-sample")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.stem(np.arange(len(xd)), xd, basefmt=" ")
ax.set_title("Downsampled sinusoid")
ax.set_xlabel("output sample index")
ax.set_ylabel("amplitude")
ax.grid(alpha=0.25)
plt.show()

# %% [markdown]
# ## 3. Decimation: lowpass before throwing samples away
# 
# For decimation by $M$, the anti-aliasing filter should suppress input content above approximately $\pi/M$. In a real FIR design, leave a transition band rather than attempting an ideal brick-wall cutoff.

# %%
fs = 4000
M = 2
t = np.arange(0, 0.08, 1/fs)

# 400 Hz is safe after decimation to 2 kHz; 1400 Hz is not.
x = np.sin(2*np.pi*400*t) + 0.55*np.sin(2*np.pi*1400*t)

raw = x[::M]

# FIR cutoff is normalized to the input Nyquist frequency for scipy.signal.firwin.
h = signal.firwin(63, cutoff=0.46)
xf = signal.lfilter(h, [1], x)
dec = xf[::M]

f_raw, P_raw = signal.periodogram(raw, fs=fs/M)
f_dec, P_dec = signal.periodogram(dec, fs=fs/M)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.semilogy(f_raw, P_raw + 1e-12, label="raw ↓2")
ax.semilogy(f_dec, P_dec + 1e-12, label="LPF then ↓2")
ax.set_xlim(0, fs/(2*M))
ax.set_title("Anti-alias filtering before decimation")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD")
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 4. `scipy.signal.resample_poly`: the practical rational-rate tool
# 
# `resample_poly(x, up=L, down=M)` implements the conceptual chain
# 
# $$\uparrow L \rightarrow H(z) \rightarrow \downarrow M$$
# 
# using an efficient polyphase structure.

# %%
fs_in = 48_000
fs_out = 44_100
L, M = 147, 160

t = np.arange(0, 0.03, 1/fs_in)
x = np.sin(2*np.pi*1000*t) + 0.25*np.sin(2*np.pi*7000*t)
y = signal.resample_poly(x, L, M)

print("Input samples:", len(x))
print("Output samples:", len(y))
print("Expected ratio:", L/M)
print("Measured ratio:", len(y)/len(x))

f0, P0 = signal.periodogram(x, fs=fs_in)
f1, P1 = signal.periodogram(y, fs=fs_out)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.semilogy(f0, P0 + 1e-12, label="48 kHz input")
ax.semilogy(f1, P1 + 1e-12, label="44.1 kHz output")
ax.set_xlim(0, 15_000)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("48 kHz → 44.1 kHz using L/M = 147/160")
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 5. Strided `Conv1d` is convolution followed by subsampling
# 
# Ignoring padding conventions, a stride-$M$ convolution computes only every $M$th convolution output. That is exactly a filtering-plus-downsampling structure.

# %%
# Make a deterministic kernel so the comparison is transparent.
x_np = np.array([0., 1., 2., 1., 0., -1., -2., -1., 0., 1., 0.], dtype=np.float32)
h_np = np.array([0.25, 0.5, 0.25], dtype=np.float32)
stride = 2

# NumPy: correlation convention to match PyTorch Conv1d.
full_valid = np.correlate(x_np, h_np, mode="valid")
y_numpy = full_valid[::stride]

conv = nn.Conv1d(1, 1, kernel_size=3, stride=stride, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor(h_np).view(1, 1, -1)

x_t = torch.tensor(x_np).view(1, 1, -1)
y_torch = conv(x_t).detach().numpy().squeeze()

print("NumPy filter then ↓2:", y_numpy)
print("PyTorch Conv1d stride=2:", y_torch)
print("Maximum difference:", np.max(np.abs(y_numpy - y_torch)))
assert np.allclose(y_numpy, y_torch, atol=1e-6)

# %% [markdown]
# ## 6. Shift sensitivity caused by subsampling phase
# 
# A one-sample shift can move signal energy onto a different sampling phase before $\downarrow 2$. The example below compares raw subsampling with blur-then-subsample.

# %%
def shift_right(x, amount=1):
    y = np.zeros_like(x)
    y[amount:] = x[:-amount]
    return y

N = 80
n = np.arange(N)
# Include a strong near-Nyquist component to make phase sensitivity visible.
x = np.sin(0.18*np.pi*n) + 0.65*np.sin(0.82*np.pi*n)
xs = shift_right(x, 1)

raw0, raw1 = x[::2], xs[::2]
blur = np.array([1, 2, 1], dtype=float) / 4
xb0 = np.convolve(x, blur, mode="same")[::2]
xb1 = np.convolve(xs, blur, mode="same")[::2]

raw_change = np.linalg.norm(raw0 - raw1) / np.linalg.norm(raw0)
blur_change = np.linalg.norm(xb0 - xb1) / np.linalg.norm(xb0)
print(f"Relative output change, raw ↓2:  {raw_change:.3f}")
print(f"Relative output change, blur+↓2: {blur_change:.3f}")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(raw0[:24], marker='o', label='raw ↓2')
ax.plot(raw1[:24], marker='o', label='raw ↓2 after 1-sample shift')
ax.set_title('Small input shift can produce a large phase-dependent change')
ax.set_xlabel('output index')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 7. Upsampling by zero insertion
# 
# For upsampling by $L$, insert $L-1$ zeros between input samples.

# %%
L = 3
x = np.array([1.0, 0.5, -0.5, -1.0, 0.25])
xi = np.zeros(len(x)*L)
xi[::L] = x

print("x  =", x)
print("xi =", xi)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.stem(np.arange(len(xi)), xi, basefmt=" ")
ax.set_title(f"Zero insertion for L = {L}")
ax.set_xlabel("high-rate sample index")
ax.grid(alpha=0.25)
plt.show()

# %% [markdown]
# ## 8. Interpolation filtering and the gain of $L$
# 
# The interpolation filter removes images and should have DC gain $L$. A constant input is a good sanity test.

# %%
L = 4
x = np.ones(30)
xi = np.zeros(len(x)*L)
xi[::L] = x

h_unit = signal.firwin(65, cutoff=1/L)
h_scaled = L*h_unit

y_unit = signal.lfilter(h_unit, [1], xi)
y_scaled = signal.lfilter(h_scaled, [1], xi)

# Ignore the transient when comparing means.
sl = slice(70, 100)
print("Mean with unit-gain LPF:  ", np.mean(y_unit[sl]))
print("Mean with gain-L LPF:     ", np.mean(y_scaled[sl]))

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(y_unit, label='unit DC gain')
ax.plot(y_scaled, label=f'DC gain L={L}')
ax.axhline(1.0, linestyle='--', linewidth=1, label='desired constant level')
ax.set_ylim(-0.1, 1.25)
ax.set_title('Why the interpolation filter needs gain L')
ax.set_xlabel('high-rate sample index')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# %% [markdown]
# ## 9. `ConvTranspose1d` as zero insertion + learned filtering
# 
# For a simple no-padding case, transposed convolution can be reproduced by inserting zeros at the stride and convolving with the kernel. Indexing details change with padding/output padding, but the DSP picture remains useful.

# %%
stride = 2
kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
x = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)

ct = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=stride, bias=False)
with torch.no_grad():
    ct.weight[:] = kernel.view(1, 1, -1)

y_ct = ct(x.view(1,1,-1)).detach().numpy().squeeze()

# Explicit zero insertion. For ConvTranspose1d's cross-correlation convention,
# convolving with the kernel reproduces this simple case.
up = np.zeros((len(x)-1)*stride + 1, dtype=np.float32)
up[::stride] = x.numpy()
y_explicit = np.convolve(up, kernel.numpy(), mode='full')

print('ConvTranspose1d:', y_ct)
print('zero insert + conv:', y_explicit)
print('max difference:', np.max(np.abs(y_ct-y_explicit)))
assert np.allclose(y_ct, y_explicit, atol=1e-6)

# %% [markdown]
# ## 10. Periodic overlap variation in transposed convolution
# 
# One mechanism behind checkerboard-like artifacts is **uneven overlap**: some output positions receive more kernel contributions than others.

# %%
stride = 2
kernel_len = 5
x = np.ones(8)
up = np.zeros((len(x)-1)*stride + 1)
up[::stride] = x
overlap = np.convolve(up, np.ones(kernel_len), mode='full')

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.stem(np.arange(len(overlap)), overlap, basefmt=" ")
ax.set_title(f"Overlap counts: stride={stride}, kernel length={kernel_len}")
ax.set_xlabel("output index")
ax.set_ylabel("number of overlapping contributions")
ax.grid(alpha=0.25)
plt.show()

print("Interior overlap values:", overlap[4:-4])

# %% [markdown]
# ## 11. Suggested in-class extensions
# 
# 1. Change the high-frequency component in Section 6 from $0.82\pi$ to $0.3\pi$. Does blur-before-stride still help as much?
# 2. Change the decimation factor in Section 3 from $M=2$ to $M=4$ and redesign the FIR cutoff.
# 3. In Section 8, remove the factor $L$ from the interpolation filter and explain the amplitude error using a constant input.
# 4. In Section 10, compare kernel lengths 4, 5, and 6 for stride 2. Which choices create the most even overlap?
# 5. Replace the fixed blur in Section 6 with a learned Conv1d kernel and inspect its frequency response after training on a simple classification task.

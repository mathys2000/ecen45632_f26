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
# # Week 14 — Lecture 27: Tensors and Automatic Differentiation
#
# **DSP → PyTorch bridge for senior / first-year graduate ECE**
#
# Learning goals:
#
# 1. Map familiar DSP objects into tensor shapes.
# 2. Use matrix multiplication, broadcasting, reshaping, `unsqueeze`, and `permute` safely.
# 3. Explain the dynamic computational graph built during a forward pass.
# 4. Use the chain rule and `torch.autograd` to compute gradients such as
#
# $$
# \nabla_{\boldsymbol{\theta}}L
# =
# \begin{bmatrix}
# \partial L/\partial \theta_1 & \cdots & \partial L/\partial \theta_P
# \end{bmatrix}^{T}.
# $$
#
# 5. Connect automatic differentiation to a familiar DSP object: a learnable FIR filter.

# %%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)
print(torch.__version__)

# %% [markdown]
# ## 1. From a discrete-time sequence to a tensor
#
# In DSP we write a finite observation of a signal as
#
# $$
# \mathbf{x} = [x[0],x[1],\ldots,x[L-1]]^T.
# $$
#
# In PyTorch the same data can be stored as a rank-1 tensor with shape `(L,)`. A tensor also carries a **dtype**, a **device**, and potentially gradient-tracking information.

# %%
fs = 8000
n = torch.arange(2000)
t = n/fs
x = 0.7*torch.sin(2*torch.pi*440*t) + 0.25*torch.sin(2*torch.pi*880*t)
print("shape =", x.shape)
print("dtype =", x.dtype)
print("device =", x.device)
print("requires_grad =", x.requires_grad)

plt.figure(figsize=(8,2.5))
plt.plot(t[:500]*1000, x[:500])
plt.xlabel("time (ms)")
plt.ylabel("amplitude")
plt.title("1-D waveform tensor")
plt.tight_layout()

# %% [markdown]
# ### A spectrogram is naturally a rank-2 tensor
#
# An STFT maps one waveform into a complex matrix
#
# $$
# X[k,m]
# =
# \sum_n x[n]w[n-mR]e^{-j2\pi kn/N_{\mathrm{FFT}}}.
# $$
#
# A common tensor shape is `(frequency_bin, frame)`.

# %%
window = torch.hann_window(256)
X = torch.stft(x, n_fft=256, hop_length=64, win_length=256, window=window, return_complex=True)
P = X.abs().square()
print("STFT shape:", X.shape, "dtype:", X.dtype)

plt.figure(figsize=(7.5,3))
plt.imshow(10*torch.log10(P+1e-10), origin="lower", aspect="auto")
plt.xlabel("frame")
plt.ylabel("frequency bin")
plt.title("Power spectrogram tensor")
plt.colorbar(label="dB")
plt.tight_layout()

# %% [markdown]
# ## 2. NumPy array versus PyTorch tensor
#
# Both support vectorized numerical operations. PyTorch additionally integrates:
#
# - device placement such as CPU or accelerator,
# - automatic differentiation,
# - neural-network layers and optimizers.
#
# On CPU, `torch.from_numpy(a)` can share memory with the NumPy array.

# %%
a_np = np.arange(5, dtype=np.float32)
a_t = torch.from_numpy(a_np)
print("before:", a_np, a_t)
a_np[0] = 99
print("after modifying NumPy:", a_np, a_t)

# %% [markdown]
# ## 3. Batches, channels, and length
#
# PyTorch's common 1-D convolution convention is
#
# $$
# (N,C,L),
# $$
#
# where
#
# - $N$ = batch size,
# - $C$ = number of channels,
# - $L$ = samples per item.
#
# For 32 stereo one-second clips at $16\,\mathrm{kHz}$:
#
# $$
# \text{shape}=(32,2,16000).
# $$

# %%
N, C, L = 32, 2, 16000
audio = torch.randn(N, C, L)
print(audio.shape)

# A mono batch stored as (N,L) can be converted to (N,1,L).
mono = torch.randn(8, 4000)
mono_ncl = mono.unsqueeze(1)
print(mono.shape, "->", mono_ncl.shape)

# %% [markdown]
# ## 4. Broadcasting
#
# Suppose `audio` has shape `(N,C,L)` and we want one gain per channel. Reshape the gain vector to
#
# $$
# (1,C,1),
# $$
#
# so PyTorch can broadcast it across batch and time.

# %%
small = torch.randn(4, 2, 16)
gain = torch.tensor([0.5, 1.25]).view(1,2,1)
y = small * gain
print("audio:", small.shape)
print("gain: ", gain.shape)
print("result:", y.shape)

# %% [markdown]
# ## 5. Matrix multiplication as a DSP transform
#
# Many DSP feature transforms are linear mappings. If
#
# $$
# \mathbf{P}\in\mathbb{R}^{F\times T}
# $$
#
# is a power spectrogram and
#
# $$
# \mathbf{W}\in\mathbb{R}^{M\times F}
# $$
#
# is a filterbank matrix, then
#
# $$
# \mathbf{E}=\mathbf{W}\mathbf{P}
# $$
#
# has shape
#
# $$
# M\times T.
# $$

# %%
F_bins, frames, bands = 129, 20, 40
P = torch.rand(F_bins, frames)
W = torch.rand(bands, F_bins)
E = W @ P
print(W.shape, "@", P.shape, "=", E.shape)

# %% [markdown]
# ## 6. Chain rule before autograd
#
# Let
#
# $$
# u=3\theta+1,
# $$
#
# and
#
# $$
# L=u^2.
# $$
#
# The chain rule gives
#
# $$
# \frac{dL}{d\theta}
# =
# \frac{dL}{du}\frac{du}{d\theta}
# =
# (2u)(3)
# =
# 6(3\theta+1).
# $$
#
# At $\theta=2$,
#
# $$
# \frac{dL}{d\theta}=42.
# $$

# %%
theta = torch.tensor(2.0, requires_grad=True)
u = 3*theta + 1
L = u**2
L.backward()
print("L =", L.item())
print("dL/dtheta =", theta.grad.item())

# %% [markdown]
# ## 7. What PyTorch records
#
# During the forward pass, PyTorch records differentiable operations connecting tensors that require gradients. Conceptually,
#
# $$
# \theta\longrightarrow u=3\theta+1\longrightarrow L=u^2.
# $$
#
# Calling `L.backward()` traverses this directed acyclic graph in reverse and repeatedly applies local derivatives. This is **reverse-mode automatic differentiation**, which is especially efficient when one scalar loss depends on many parameters.

# %% [markdown]
# ## 8. A vector-valued example
#
# Let
#
# $$
# \hat y_i=wx_i+b,
# $$
#
# with mean-squared error
#
# $$
# L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2.
# $$
#
# Autograd computes both
#
# $$
# \frac{\partial L}{\partial w}
# \quad\text{and}\quad
# \frac{\partial L}{\partial b}
# $$
#
# in one backward pass.

# %%
xv = torch.linspace(-2,2,41)
yv = 2.5*xv - 0.4
w = torch.tensor(0.2, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y_hat = w*xv + b
loss = torch.mean((y_hat-yv)**2)
loss.backward()
print(f"loss={loss.item():.4f}")
print(f"dL/dw={w.grad.item():.4f}, dL/db={b.grad.item():.4f}")

# %% [markdown]
# ### Gradients accumulate
#
# PyTorch **adds** new gradients into the `.grad` field. Therefore a normal training loop contains
#
# ```python
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# ```
#
# before the next iteration.

# %%
theta = torch.tensor(1.0, requires_grad=True)
(theta**2).backward()
print(theta.grad)
(theta**2).backward()
print(theta.grad, "<- accumulated")
theta.grad.zero_()
print(theta.grad, "<- cleared")


# %% [markdown]
# ## 9. DSP-native example: learn an FIR filter
#
# For a causal FIR system,
#
# $$
# y[n]=\sum_{k=0}^{M-1}h[k]x[n-k].
# $$
#
# Treat the tap vector
#
# $$
# \mathbf{h}=[h[0],\ldots,h[M-1]]^T
# $$
#
# as the learnable parameter vector and minimize
#
# $$
# L(\mathbf{h})
# =
# \frac{1}{N}\sum_n(\hat y[n]-y_{\text{target}}[n])^2.
# $$
#
# For this simple model, one can derive
#
# $$
# \frac{\partial L}{\partial h[k]}
# =
# \frac{2}{N}\sum_n e[n]x[n-k],
# $$
#
# where
#
# $$
# e[n]=\hat y[n]-y_{\text{target}}[n].
# $$
#
# Autograd arrives at the same gradient automatically.

# %%
def fir_apply(x, h):
    M = h.numel()
    x_pad = F.pad(x.view(1,1,-1), (M-1,0))
    # conv1d performs cross-correlation; flip h to match DSP convolution.
    return F.conv1d(x_pad, h.flip(0).view(1,1,-1)).view(-1)

torch.manual_seed(7)
x = torch.randn(512)
h_true = torch.tensor([0.15, 0.50, 0.25, -0.10])
with torch.no_grad():
    y_target = fir_apply(x, h_true)

h = torch.nn.Parameter(torch.zeros_like(h_true))
optimizer = torch.optim.SGD([h], lr=0.08)
history = []
for step in range(80):
    optimizer.zero_grad()
    y_hat = fir_apply(x, h)
    loss = torch.mean((y_hat-y_target)**2)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

print("true   :", h_true)
print("learned:", h.detach())
print("final loss:", history[-1])
plt.semilogy(history)
plt.xlabel("step")
plt.ylabel("MSE")
plt.title("Autograd learns FIR coefficients")
plt.tight_layout()

# %% [markdown]
# ## 10. `no_grad()` and `detach()`
#
# Use `torch.no_grad()` when computations should not build a gradient graph, for example evaluation or target generation.
#
# Use `.detach()` when you need a tensor sharing the same numerical values but disconnected from the existing graph.
#
# Do **not** use either one inside a computation when you still need gradients to flow through that operation.

# %%
z = torch.tensor(2.0, requires_grad=True)
y = 5*z
print(y.requires_grad)
print(y.detach().requires_grad)
with torch.no_grad():
    q = 5*z
print(q.requires_grad)

# %% [markdown]
# ## 11. Take-away
#
# The core transition is
#
# $$
# \boxed{
# \text{DSP arrays}
# \rightarrow
# \text{tensors}
# \rightarrow
# \text{computational graph}
# \rightarrow
# \nabla_{\theta}L
# }
# $$
#
# A PyTorch tensor is not inherently a neural-network object. It can represent the same signals, spectra, filterbanks, and linear transforms used throughout DSP. Gradient tracking adds the ability to **learn parameters of those operations from a loss function**.

# %%

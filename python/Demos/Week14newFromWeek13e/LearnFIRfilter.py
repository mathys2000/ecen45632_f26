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
# # Learn FIR Filter
#
# Learn FIR filter coefficients from response of filter to random input

# %% [markdown]
# ## DSP-native example: learn an FIR filter
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
#

# %%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)
print(torch.__version__)


# %%
# Convolution h[n]*x[n]
def fir_apply(x, h):
    M = h.numel()    # filter order
    x_pad = F.pad(x.view(1,1,-1), (M-1,0))
    # conv1d performs cross-correlation; flip h to match DSP convolution.
    return F.conv1d(x_pad, h.flip(0).view(1,1,-1)).view(-1)



# %%
# Setup of parameters
torch.manual_seed(7)     # fix seed for repeatability
x = torch.randn(512)     # initialize x[n] (length 512) with random data 
h_true = torch.tensor([0.15, 0.50, 0.25, -0.10])
with torch.no_grad():    # compute target response
    y_target = fir_apply(x, h_true)
print(f'x: {x.shape}')
print(f'h_true: {h_true.shape}')

# %%
# Initialize Stochastic Gradient Descent (SGD) algorithm
h = torch.nn.Parameter(torch.zeros_like(h_true))  # initialize h[n] with all zeros
#h = torch.nn.Parameter(torch.zeros(3))
optimizer = torch.optim.SGD([h], lr=0.08)
history = []     # keep track of loss history
print(f'h: {h}')

# %%
# Run SGD algorithm to learn h[n] coefficients
for step in range(80):
    optimizer.zero_grad()
    y_hat = fir_apply(x, h)
    loss = torch.mean((y_hat-y_target)**2)
    loss.backward()
    optimizer.step()
    history.append(loss.item())


# %%
# Report and plot results
print("true   :", h_true)
print("learned:", h.detach())
print("final loss:", history[-1])
plt.semilogy(history)
plt.grid(alpha=0.5)
plt.xlabel("step")
plt.ylabel("MSE")
plt.title("Autograd learns FIR coefficients")
plt.tight_layout()


# %%

# %%
# Convolution h[n]*x[n]
def fir_apply(x, h):
    M = h.numel()    # filter order
    x_pad = F.pad(x.view(1,1,-1), (M-1,0))
    # conv1d performs cross-correlation
    # flip h to match DSP convolution.
    return F.conv1d(x_pad, h.flip(0).view(1,1,-1)).view(-1)


# %%
# Setup of parameters
torch.manual_seed(7)     # fix seed for repeatability
x = torch.randn(512)     # initialize x[n] with random data 
h_true = torch.tensor([0.15, 0.50, 0.25, -0.10])
with torch.no_grad():    # compute target response
    y_target = fir_apply(x, h_true)
print(f'x: {x.shape}')
print(f'h_true: {h_true.shape}')

# %%
# Initialize Stochastic Gradient Descent (SGD) algorithm
# initialize h[n] with all zeros
h = torch.nn.Parameter(torch.zeros_like(h_true))  
optimizer = torch.optim.SGD([h], lr=0.08)
history = []     # keep track of loss history
print(f'h: {h}')

# %%
# Run SGD algorithm to learn h[n] coefficients
for step in range(80):
    optimizer.zero_grad()
    y_hat = fir_apply(x, h)
    loss = torch.mean((y_hat-y_target)**2)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

# %%
# Report and plot results
print("true   :", h_true)
print("learned:", h.detach())
print("final loss:", history[-1])
plt.figure(figsize=(6, 3))
plt.semilogy(history)
plt.grid(alpha=0.5)
plt.xlabel("step")
plt.ylabel("MSE")
plt.title("Autograd learns FIR coefficients")
plt.tight_layout()

# %%

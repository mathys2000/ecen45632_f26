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
# # PyTorch Strides
#
# Upsampling and downsampling in CNNs using PyTorch

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# %%
# ----- Downsampling -----------------------------------------------------------------

# %%
# Use deterministic kernel to make comparisons transparent.
x_np = np.array([0., 1., 2., 1., 0., -1., -2., -1., 0.], dtype=np.float32)
#h_np = 1/4*np.array([1, 2, 1], dtype=np.float32)
h_np = np.array([1, 0, 0], dtype=np.float32)
#h_np = np.array([0, 1, 0], dtype=np.float32)
stride = 2


# %%
# NumPy: correlation convention to match PyTorch Conv1d.
full_valid = np.correlate(x_np, h_np, mode="valid")
y_numpy = full_valid[::stride]


# %%
# Weight matrix multiply
m = int(1 + np.ceil((x_np.size - h_np.size) / stride))  # number of rows
W = np.zeros((m, (m-1)*stride + h_np.size))
for i in range(m):
    W[i, i*stride:i*stride + h_np.size] = h_np
W = W[:,:x_np.size]
print(f'W =\n{W}')
y_W = W @ x_np


# %%
# PyTorch Conv1d
conv = nn.Conv1d(1, 1, kernel_size=h_np.size, stride=stride, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor(h_np).view(1, 1, -1)

x_t = torch.tensor(x_np).view(1, 1, -1)
y_torch = conv(x_t).detach().numpy().squeeze()


# %%
# Compare results
print(f'Numpy:    y = {y_numpy}')
print(f'W matrix: y = {y_W}')
print(f'PyTorch:  y = {y_torch}')

# %%
print(f'x = {x_np}')

# %%
# ----- Upsampling -----------------------------------------------------------------

# %%
# Define inputs for upsampling from downsampling outputs
x_up_W = y_W
x_up_t = y_torch

# %%
# Transpose of weight matrix multiply
print(f'W.T =\n{W.T}')
y_up_W = W.T @ x_up_W


# %%
conv_transpose = nn.ConvTranspose1d(1, 1, kernel_size=h_np.size, stride=stride, bias=False)
with torch.no_grad():
    conv_transpose.weight[:] = torch.tensor(h_np).view(1, 1, -1)

y_up_t = torch.tensor(x_up_t).view(1, 1, -1)
y_up_torch = conv_transpose(y_up_t).detach().numpy().squeeze()


# %%
# Compare results
print(f'y_up_W:     y={y_up_W}')
print(f'y_up_torch: y={y_up_torch}')

# %%
print(f'x = {x_up_W}')

# %%

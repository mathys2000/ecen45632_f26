# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:dlpytorch]
#     language: python
#     name: conda-env-dlpytorch-py
# ---

# %% [markdown]
# # PS1_Problem 3
#
# Verify convolution

# %%
import numpy as np

# %%
# Convolution using np.convolve
x = [1,-1,2]; h = [2,1]
y = np.convolve(x, h, mode='full')
print(f'x*h = {y}')

# %%
import torch
import torch.nn.functional as F

# %%
# DSP convolution using torch conv1d
x2 = torch.tensor([[[0,1,-1,2,0]]])   # note the padding with zeros
w2 = torch.tensor([[[1,2]]])          # this is h[n] flipped
y2 = F.conv1d(x2, w2)
print(f'torch: x*w = {y2}')

# %%

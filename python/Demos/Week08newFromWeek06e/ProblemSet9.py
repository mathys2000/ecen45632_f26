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
# # Problem Set 9
#
# Python code for Problem Set 9

# %%
import numpy as np
import torch
import torch.nn as nn


# %%
x = np.array([1,2,0,-1,3,1,0], np.float32)
h = 1/4*np.array([1,2,1], np.float32)
stride = 1

# %%
conv = nn.Conv1d(1, 1, kernel_size=h.size, stride=stride, bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor(h).view(1, 1, -1)
x_t = torch.tensor(x).view(1, 1, -1)
y_t = conv(x_t).detach().numpy().squeeze()
print(f'stride={stride}, y[n]={y_t}')

# %%

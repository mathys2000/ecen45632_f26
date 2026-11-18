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
# # Tensors 001
#
# Defining and viewing tensors in PyTorch

# %%
import numpy as np
import torch

# %%
# From Python lists, 1-dimensional
data01 = np.arange(8)
t_data01 = torch.tensor(data01)
print(f'data = {t_data01}')
print(f'shape = {t_data01.shape}')
print(f'dtype = {t_data01.dtype}')
print(f'device = {t_data01.device}')
print(f'requires_grad = {t_data01.requires_grad}')


# %%
# From Python lists, 2-dimensional
data02 = [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]
t_data02 = torch.tensor(data02)
print(f'data = {t_data02}')
print(f'shape = {t_data02.shape}')
print(f'dtype = {t_data02.dtype}')
print(f'device = {t_data02.device}')
print(f'requires_grad = {t_data02.requires_grad}')


# %%
t_data03 = torch.zeros(3,7)
t_data03[1].copy_(torch.from_numpy(np.arange(7)))
#t_data03[:,1].copy_(torch.from_numpy(np.arange(3)))
print(t_data03)
print(t_data03.shape)
t_data04 = t_data03.unsqueeze(1)
print(t_data04)
print(t_data04.shape)
print(t_data04[0][0][0])

# %%
#copy_(torch.from_numpy(numpy_arr[1]))

# %%
theta = torch.tensor([2.0, -3.1], requires_grad=True)
#theta = torch.tensor([-3.1], requires_grad=True)
u = 3*theta + 1
print(u**2)
print((u**2).mean())
loss = (u**2).mean()
print(loss)
loss.backward()
print(theta.grad)


# %%
t_data05 = torch.tensor([np.arange(5),np.arange(5,10),np.arange(10,15)])

# %%
print(t_data05.shape)
print(t_data05)

# %%
t_data06 = t_data05.unsqueeze(1)
print(t_data06.shape)
print(t_data06)

# %%
chan_gain = torch.tensor([0.5, 1.25])
print(chan_gain.shape)
print(chan_gain)

# %%
C = 2
chan_gain2 = chan_gain.view(1, C, 1)
print(chan_gain2.shape)
print(chan_gain2)

# %%
torch.linspace(-2, 2, 5)

# %%

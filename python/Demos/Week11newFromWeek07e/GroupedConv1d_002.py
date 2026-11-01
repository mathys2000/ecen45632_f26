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
# # Grouped Conv1d 002
#
# Grouped and ungrouped convolution in pytorch, small example

# %%
import numpy as np
import torch

# %%
layer1 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=0, groups=1, bias=False)
with torch.no_grad():
    # Outputs 0,1,2,3 see all input channels 0,1,2,3
    layer1.weight[0] = 0.5
    #layer1.weight[0] = torch.from_numpy(np.array([0.5, 0.4, 0.3]))
    layer1.weight[1] = 0.25
    layer1.weight[2] = -0.25
    layer1.weight[3] = -0.5
    print(layer1.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 1, 3] = 1.0
#x[:, 1, 3] = 2.0
print(x)
y1 = layer1(x)
y1_np = y1.detach().numpy()
print(y1_np)

# %%
layer2 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=0, groups=2, bias=False)
with torch.no_grad():
    # Outputs 0,1 see only input channels 0,1; outputs 2,3 see only 2,3.
    layer2.weight[0] = 0.5
    layer2.weight[1] = 0.25
    layer2.weight[2] = -0.25
    layer2.weight[3] = -0.5
    print(layer2.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 0, 3] = 1.0
#x[:, 1, 3] = 2.0
print(x)
y2 = layer2(x)
y2_np = y2.detach().numpy()
print(y2_np)

# %%
layer4 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=0, groups=4, bias=False)
with torch.no_grad():
    # Output 0 sees only input channel 0; output 1 sees only 1; output 2 sees only 2; output 3 sees only 3.
    layer4.weight[0] = 0.5
    layer4.weight[1] = 0.25
    layer4.weight[2] = -0.25
    layer4.weight[3] = -0.5
    print(layer4.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 1, 3] = 1.0
#x[:, 1, 3] = 2.0
print(x)
y4 = layer4(x)
y4_np = y4.detach().numpy()
print(y4_np)

# %%

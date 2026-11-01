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
# # Grouped Conv1d 001
#
# Grouped and ungrouped convolution in pytorch

# %%
import numpy as np
import torch

# %%
#torch.manual_seed(3)
layer1 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=1, bias=False)
with torch.no_grad():
    #layer1.weight.zero_()
    #print(layer1.weight)
    # Outputs 0,1 see only input channels 0,1; outputs 2,3 see only 2,3.
    layer1.weight[0] = 0.5
    layer1.weight[1] = 0.25
    layer1.weight[2] = -0.25
    layer1.weight[3] = -0.5
    print(layer1.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 0, 3] = 1.0
#x[:, 1, 3] = 2.0
print(x)
y1 = layer1(x)
y1_np = y1.detach().numpy()
affected1 = y1.abs().amax(dim=-1).squeeze(0).detach().numpy()
print(y1_np)

# %%
#torch.manual_seed(3)
layer2 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=2, bias=False)
with torch.no_grad():
    #layer2.weight.zero_()
    #print(layer2.weight)
    # Outputs 0,1 see only input channels 0,1; outputs 2,3 see only 2,3.
    layer2.weight[0] = 0.5
    layer2.weight[1] = 0.25
    layer2.weight[2] = -0.25
    layer2.weight[3] = -0.5
    print(layer2.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 0, 3] = 1.0
x[:, 2, 3] = 2.0
print(x)
y2 = layer2(x)
y2_np = y2.detach().numpy()
affected2 = y2.abs().amax(dim=-1).squeeze(0).detach().numpy()
print(y2_np)

# %%
#torch.manual_seed(3)
layer4 = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)
with torch.no_grad():
    #layer4.weight.zero_()
    #print(layer4.weight)
    # Outputs 0,1 see only input channels 0,1; outputs 2,3 see only 2,3.
    layer4.weight[0] = 0.5
    layer4.weight[1] = 0.25
    layer4.weight[2] = -0.25
    layer4.weight[3] = -0.5
    print(layer4.weight)


# %%
x = torch.zeros(1, 4, 8)
x[:, 0, 3] = 1.0
x[:, 2, 3] = 2.0
print(x)
y4 = layer4(x)
y4_np = y4.detach().numpy()
affected4 = y4.abs().amax(dim=-1).squeeze(0).detach().numpy()
print(y4_np)

# %%

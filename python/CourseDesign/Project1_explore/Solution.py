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
# # Worked DSP project solution
# The implementation is in `dsp_solution.py`. This notebook runs all experiments and displays measured evidence. Parameters and theoretical interpretation are in the accompanying solution document. The saved outputs below are from a completed run.

# %%
from run_solution import main
metrics = main()


# %% [markdown]
# ## Time frequency tradeoff

# %%
print(metrics['time_frequency'])


# %%
from IPython.display import display, Image
display(Image(filename="results/uncertainty.png"))


# %% [markdown]
# ## Zero padding

# %%
print(metrics['time_frequency'])


# %%
from IPython.display import display, Image
display(Image(filename="results/zero_padding.png"))


# %% [markdown]
# ## Windows and leakage

# %%
print(metrics['windows'])


# %%
from IPython.display import display, Image
display(Image(filename="results/windows.png"))


# %% [markdown]
# ## Overlap and exact FFT convolution

# %%
print(metrics['overlap'])


# %% [markdown]
# ## Moving jammer

# %%
print(metrics['jammer_sweep'])


# %%
from IPython.display import display, Image
display(Image(filename="results/jammer.png"))


# %% [markdown]
# ## Phase coherent QPSK

# %%
print(metrics['communications'])


# %%
from IPython.display import display, Image
display(Image(filename="results/constellations.png"))


# %% [markdown]
# ## Acceptance checks
# `main()` checks identity reconstruction, expected Hann NOLA failure, real and complex full FFT convolution, and zero BER after coherent QPSK reassembly. EVM includes finite RRC pulse truncation. WAV round-trip error is separate from float64 identity error. Runtime values vary across machines. See the solution document for the derivations and grading notes.

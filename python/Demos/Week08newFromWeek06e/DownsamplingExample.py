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
# # Downsampling Example
#
# Sum of two sinusoids, downsample by M=2 or M=3

# %%
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd


# %%
# Parameters
Fs1 = 6000
M = 3
tlen = 1
wa, wb = 0.18*np.pi, 0.46*np.pi

# %%
# Generate waveform
nn = np.arange(np.round(tlen*Fs1))
xn = np.sin(wa*nn) + 0.35*np.sin(wb*nn)
xnd = xn[::M]


# %%
# Waveform for display
N = 40
n = np.arange(N)
x = np.sin(wa*n) + 0.35*np.sin(wb*n)
xd = x[::M]

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.stem(n, x, basefmt=" ", label="x[n]")
ax.scatter(n[::M], xd, s=70, facecolors="none", edgecolors="tab:red", linewidths=2,
           label="samples retained")
ax.set_title(f"Downsampling by M={M}, fa={wa/(2*np.pi)*Fs1} Hz, fb={wb/(2*np.pi)*Fs1} Hz, Fs2={Fs1//M} [samples/sec]")
ax.set_xlabel("n")
ax.set_ylabel("amplitude")
ax.grid(alpha=0.25)
ax.legend()
plt.show()

#print("First 8 output samples:", xd[:8])

# %%
# Original signal
print(xn.size)
ipd.Audio(xn, rate=Fs1)


# %%
# Downsampled signal
print(xnd.size)
ipd.Audio(M*xnd, rate=Fs1/M)


# %%

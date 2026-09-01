# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dlpytorch [conda env:dlpytorch]
#     language: python
#     name: conda-env-dlpytorch-dlpytorch
# ---

# %% [markdown]
# # Week 2 — Sampling, Aliasing, Reconstruction, and Quantization
#
# **DSP + ML course | senior / first-year graduate ECE**
#
# This notebook supports the two 75-minute Week 2 lectures and follows the notation used in Oppenheim & Schafer.
#
# Learning goals:
# 1. Connect a continuous-time acoustic/electrical signal $x_c(t)$ to samples $x[n]=x_c(nT)$.
# 2. Visualize spectral replication and aliasing.
# 3. Compare naive decimation with anti-aliased decimation.
# 4. Reconstruct samples with sinc interpolation and model a zero-order hold (ZOH).
# 5. Quantize audio, verify the $6.02B+1.76$ dB SQNR rule, and connect PCM integers to PyTorch `float32` tensors.
#
# The notebook synthesizes a deterministic voice-like signal so it runs without an external audio file. Replace it later with a classroom vocal recording if desired.

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from IPython.display import Audio, display
import torch

# Import the companion helper module when this notebook is next to it.
HERE = Path.cwd()
if not (HERE / "Week2_DSP_ML_Code.py").exists():
    # Jupyter may run with a different working directory.
    candidate = Path("/mnt/data/week2_dsp_ml")
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))

from Week2_DSP_ML_Code import (
    sample_sinusoid, alias_to_baseband, synth_voice_like,
    decimate_naive, decimate_antialias, sinc_reconstruct, zoh_waveform,
    uniform_quantize, measured_sqnr_db, theoretical_sqnr_db,
    stft_power_db, torch_stft_feature, int16_to_float32, write_wav
)

plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True
print("NumPy", np.__version__, "| PyTorch", torch.__version__)

# %% [markdown]
# ## 1. The C/D interface: $x[n]=x_c(nT)$
#
# A sample is **not** a little rectangle and it is not an approximation to the analog waveform. Mathematically, it is the exact value of the continuous-time signal at one instant. The approximation enters when we try to infer the values *between* samples without enough bandwidth information.

# %%
f0, fs = 900.0, 4000.0

t, xc, n, tn, xn = sample_sinusoid(f0, fs, duration_s=0.006)

plt.plot(t * 1000, xc, label=r"$x_c(t)$")
plt.stem(tn * 1000, xn, basefmt=" ", label=r"$x[n]$")
plt.xlabel("time (ms)"); plt.ylabel("amplitude")
plt.title(f"{f0:.0f} Hz sinusoid sampled at {fs:.0f} Hz")
plt.legend(); plt.show()

omega0 = 2*np.pi*f0/fs
print(f"Digital frequency: omega0 = {omega0:.3f} rad/sample = {omega0/np.pi:.3f} pi rad/sample")

# %%
f00, fs = 900.0, 4000.0

t, xc, n, tn, xn = sample_sinusoid(f00, fs, duration_s=0.004)

plt.plot(t * 1000, xc, label=r"$x_c(t)$")
plt.stem(tn * 1000, xn, basefmt=" ", label=r"$x[n]$")

f01, fs = 3100.0, 4000.0
t, xc, n, tn, xn = sample_sinusoid(f01, fs, duration_s=0.004)

plt.plot(t * 1000, xc, label=r"$x_c(t)$")
plt.stem(tn * 1000, xn, basefmt=" ", label=r"$x[n]$")

plt.xlabel("time (ms)"); plt.ylabel("amplitude")
plt.title(f"{f00:.0f} Hz, {f01:.0f} Hz sinusoids sampled at {fs:.0f} Hz")
plt.legend(); plt.show()

omega00 = 2*np.pi*f00/fs
print(f"Digital frequency: omega00 = {omega00:.3f} rad/sample = {omega00/np.pi:.3f} pi rad/sample")
omega01 = 2*np.pi*f01/fs
print(f"Digital frequency: omega01 = {omega01:.3f} rad/sample = {omega01/np.pi:.3f} pi rad/sample")

# %% [markdown]
# ## 2. Aliasing is non-uniqueness
#
# For a sampled sinusoid,
# $$e^{j(\omega+2\pi k)n}=e^{j\omega n}.$$
# So continuous-time frequencies separated by integer multiples of $f_s$ can become the **same discrete-time sequence**. For real cosines there is also the mirror symmetry $\cos(\omega n)=\cos(-\omega n)$.
#
# This is why aliasing is permanent information loss: after sampling, there may be no evidence of which analog frequency created the samples.

# %%
fs = 12_000
for f in [2_000, 10_000, 14_000, 22_000]:
    print(f"{f/1000:5.1f} kHz -> alias {alias_to_baseband(f, fs)/1000:4.1f} kHz at fs={fs/1000:.0f} kHz")

n = np.arange(30)
x2 = np.cos(2*np.pi*2000*n/fs)
x10 = np.cos(2*np.pi*10000*n/fs)
print("max |x_2k[n] - x_10k[n]| =", np.max(np.abs(x2-x10)))

# %% [markdown]
# ## 3. Decimation: wrong way and right way
#
# Suppose we reduce 48 kHz audio to 12 kHz by keeping every fourth sample.
#
# - New Nyquist frequency: 6 kHz.
# - Any energy above 6 kHz must be removed **before** downsampling.
# - A 10 kHz component aliases to 2 kHz if we skip the anti-aliasing filter.
#
# The example below deliberately includes a 10 kHz component so the alias is obvious.

# %%
fs_hi = 48_000
M = 4
fs_lo = fs_hi // M
x = synth_voice_like(fs_hi, duration_s=2.5)
x_naive = decimate_naive(x, M)
x_aa = decimate_antialias(x, M)

print("Original fs:", fs_hi, "Hz | Downsampled fs:", fs_lo, "Hz")
print("10 kHz aliases to", alias_to_baseband(10_000, fs_lo), "Hz")

print("Clean synthetic voice-like signal:")
display(Audio(x, rate=fs_hi))
print("Naive decimation (aliasing):")
display(Audio(x_naive, rate=fs_lo))
print("Anti-aliased decimation:")
display(Audio(x_aa, rate=fs_lo))


# %%
def show_spec(sig, fs, title, nperseg):
    f, tt, p = stft_power_db(sig, fs, nperseg=nperseg)
    plt.figure(figsize=(9,4))
    plt.pcolormesh(tt, f/1000, p, shading="auto", cmap="magma", vmin=-80, vmax=-15)
    plt.ylim(0, min(12, fs/2000))
    plt.xlabel("time (s)"); plt.ylabel("frequency (kHz)")
    plt.title(title); plt.colorbar(label="magnitude (dB)"); plt.show()

show_spec(x, fs_hi, "Original at 48 kHz", 1024)
show_spec(x_naive, fs_lo, "Naive 4x decimation: false low-frequency structure", 512)
show_spec(x_aa, fs_lo, "Polyphase anti-alias filtering + decimation", 512)

# %% [markdown]
# ### ML connection
# A spectrogram is often treated as an image by a CNN. But an alias is not random measurement noise; it is **structured energy in the wrong frequency bin**. A model can therefore learn a stable but physically false feature. Changing the sampling hardware or sample rate later can cause a large distribution shift.

# %% [markdown]
# ## 4. Ideal D/C reconstruction: sinc interpolation
#
# For a bandlimited signal sampled above the Nyquist rate,
# $$x_r(t)=\sum_{n=-\infty}^{\infty}x[n]\,\mathrm{sinc}\!\left(\frac{t-nT}{T}\right),$$
# where $\mathrm{sinc}(u)=\sin(\pi u)/(\pi u)$.
#
# Every shifted sinc is 1 at its own sample instant and 0 at every other integer sample instant. The infinite sum therefore passes through all sample values while enforcing bandlimitation.

# %%
fs = 1000.0
n = np.arange(6)
xn = np.array([0.0, 0.8, -0.3, 1.0, 0.2, -0.6])
t_eval = np.linspace(-0.001, 0.006, 2500)
xr = sinc_reconstruct(xn, fs, t_eval)

plt.plot(t_eval*1000, xr, label="sinc interpolation")
plt.stem(n/fs*1000, xn, basefmt=" ", label="samples")
plt.xlabel("time (ms)"); plt.ylabel("amplitude")
plt.title("Finite illustration of ideal sinc interpolation")
plt.legend(); plt.show()

# Check interpolation at the sample instants.
x_check = sinc_reconstruct(xn, fs, n/fs)
print("maximum sample-point interpolation error =", np.max(np.abs(x_check-xn)))

# %% [markdown]
# ## 5. Practical D/A: zero-order hold
#
# A physical DAC commonly holds each sample value for one sample period. If the hold pulse is
# $$h_{\mathrm{ZOH}}(t)=u(t)-u(t-T),$$
# then
# $$H_{\mathrm{ZOH}}(j\Omega)=T e^{-j\Omega T/2}\frac{\sin(\Omega T/2)}{\Omega T/2}.$$
#
# That sinc envelope causes high-frequency **droop**. At the digital Nyquist frequency, the magnitude is $2/\pi$, about $-3.92$ dB.

# %%
xn = np.array([0.0, .9, .3, -.6, -.2, .7, .1])
tz, yz = zoh_waveform(xn, fs_hz=1000, oversample=80)
plt.step(tz*1000, yz, where="post", label="ZOH")
plt.stem(np.arange(len(xn)), xn, basefmt=" ", label="samples")
plt.xlabel("time (ms)"); plt.ylabel("amplitude"); plt.title("Zero-order hold")
plt.legend(); plt.show()

omega = np.linspace(0, np.pi, 1000)
mag = np.ones_like(omega)
mag[1:] = np.abs(np.sin(omega[1:]/2)/(omega[1:]/2))
plt.plot(omega/np.pi, 20*np.log10(mag))
plt.xlabel(r"$\omega/\pi$"); plt.ylabel("magnitude (dB)")
plt.title("ZOH sinc droop")
plt.show()
print("ZOH droop at Nyquist =", 20*np.log10(2/np.pi), "dB")

# %% [markdown]
# ## 6. Quantization: time is discrete *and* amplitude is discrete
#
# For a uniform $B$-bit quantizer spanning approximately $[-A,A]$,
# $$\Delta=\frac{2A}{2^B}.$$
# Under the familiar high-resolution noise model,
# $$e[n]\sim \mathcal U(-\Delta/2,\Delta/2),\qquad \sigma_e^2=\frac{\Delta^2}{12}.$$
# For a full-scale sine wave this gives
# $$\mathrm{SQNR}\approx 6.02B+1.76\ \mathrm{dB}.$$
#
# **Important caveat:** the additive white-noise model is an approximation. It can fail for very low-level or highly periodic signals, and it does not cover overload clipping.

# %%
fs = 48_000
t = np.arange(fs) / fs
sine = 0.999*np.sin(2*np.pi*997*t)

for B in [4, 8, 12, 16]:
    q, delta = uniform_quantize(sine, B)
    print(f"B={B:2d}  Delta={delta:.7f}  theory={theoretical_sqnr_db(B):6.2f} dB  measured={measured_sqnr_db(sine,q):6.2f} dB")

bits = np.arange(2,17)
theory = 6.02*bits + 1.76
meas = []
for B in bits:
    q,_ = uniform_quantize(sine, int(B))
    meas.append(measured_sqnr_db(sine,q))
plt.plot(bits, theory, label="6.02B + 1.76 dB")
plt.plot(bits, meas, "o", label="measured")
plt.xlabel("bits"); plt.ylabel("SQNR (dB)"); plt.title("About 6 dB per additional bit")
plt.legend(); plt.show()

# %%
xq8, _ = uniform_quantize(x, 8)
xq4, _ = uniform_quantize(x, 4)
print("8-bit quantized:"); display(Audio(xq8, rate=fs_hi))
print("4-bit quantized:"); display(Audio(xq4, rate=fs_hi))

show_spec(x, fs_hi, "Clean waveform", 1024)
show_spec(xq4, fs_hi, "4-bit quantization raises a broadband error floor", 1024)

# %% [markdown]
# ## 7. PCM integers → PyTorch float tensors
#
# A common audio file stores 16-bit PCM integers. Neural-network pipelines normally convert them to floating point and scale them near $[-1,1)$. **Casting to float does not restore precision that was lost during the ADC quantization step.** It only changes the numerical representation used for subsequent computation.
#
# The rough “6 dB per bit” rule gives about 96 dB across 16 bits; the full-scale sine SQNR formula gives about 98.1 dB.

# %%
pcm = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
xf = int16_to_float32(pcm)
xt = torch.from_numpy(xf)
print("PCM:", pcm)
print("float32:", xf)
print("PyTorch dtype:", xt.dtype)

# A spectrogram feature tensor has no idea whether aliasing happened earlier.
mag = torch_stft_feature(x_naive, n_fft=512, hop=128)
print("STFT magnitude shape (frequency bins x frames):", tuple(mag.shape))
print("The tensor faithfully represents the *aliased* waveform it was given.")

# %% [markdown]
# ## 8. Export files for the Lecture 4 live demo
#
# The following cell writes five WAV files. Play them in sequence during class:
# 1. clean 48 kHz signal,
# 2. naive 12 kHz decimation,
# 3. anti-aliased 12 kHz decimation,
# 4. 8-bit quantization,
# 5. 4-bit quantization.

# %%
demo_dir = HERE / "audio"
demo_dir.mkdir(exist_ok=True)
write_wav(demo_dir / "01_clean_48k.wav", fs_hi, x)
write_wav(demo_dir / "02_decimated_naive_12k_alias.wav", fs_lo, x_naive)
write_wav(demo_dir / "03_decimated_antialias_12k.wav", fs_lo, x_aa)
write_wav(demo_dir / "04_quantized_8bit.wav", fs_hi, xq8)
write_wav(demo_dir / "05_quantized_4bit.wav", fs_hi, xq4)
print("Wrote:")
for p in sorted(demo_dir.glob("*.wav")):
    print(" ", p.name)

# %% [markdown]
# ## Suggested in-class prompts
#
# - “Why can no digital algorithm undo aliasing after the samples are taken?”
# - “Which operation must happen before decimation, and why does its order matter?”
# - “Why is an ideal sinc interpolator mathematically clean but physically inconvenient?”
# - “What assumption is hidden in the phrase *quantization noise*?”
# - “Why does converting `int16` audio to `float32` not give us 32-bit audio precision?”
#
# ### Graduate extension
# Derive the ZOH frequency response from $h_{\mathrm{ZOH}}(t)=u(t)-u(t-T)$ and design a digital pre-emphasis response that approximately compensates the ZOH droop over $0\le\omega\le0.8\pi$.

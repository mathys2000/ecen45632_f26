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
# # Make Audio Dataset 001
#
# Generate dataset (low tone, high tone, chirp) for simple MFCC mean/std audio classifier

# %%
import numpy as np
from pathlib import Path
from scipy.io import wavfile


# %%
# Parameters
SR = 16000
DURATION = 1
SEED = 42
CLASSES = ('low_tone', 'high_tone', 'chirp')
OUT = Path('mfcc_project_output')


# %%
# Synthesize sample of class 'label'
def synthesize(label, rng):
    """
    Generate one independent, slightly varied, 1-second example of type 'label'
    """
    t = np.arange(int(SR*DURATION))/SR    # time axis
    amplitude = rng.uniform(0.3, 0.9)
    phase = rng.uniform(0, 2*np.pi)
    if label == 'low_tone':
        f = rng.uniform(180, 320)
        x = np.cos(2*np.pi*f*t + phase)
    elif label == 'high_tone':
        f = rng.uniform(1200, 1700)
        x = np.cos(2*np.pi*f*t + phase)
    elif label == 'chirp':
        f0 = rng.uniform(250, 450)
        f1 = rng.uniform(2200, 2800)
        k = (f1-f0)/DURATION
        x = np.cos(2*np.pi*(f0*t + 0.5*k*t**2) + phase)
    else:
        raise ValueError(label)
    # Add independently sampled sensor/background noise.
    x = amplitude*x + rng.uniform(0.015, 0.07)*rng.normal(size=len(t))
    x = np.clip(x, -1, 1)     # clip amplitude to -1...+1 range
    return x.astype(np.float32)
    


# %%
# Generate dataset with 'examples_per_class' samples
def make_dataset(base_dir, examples_per_class=40):
    rng = np.random.default_rng(seed=SEED)
    rows = []
    for label in CLASSES:
        folder = base_dir/label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(examples_per_class):
            x = synthesize(label, rng)
            path = folder/f'{label}_{i:03d}.wav'
            wavfile.write(path, SR, np.round(x*(2**15-1)).astype(np.int16))
            rows.append((path, label))
    return rows
    


# %%
# Main script to generate dataset
OUT.mkdir(parents=True, exist_ok=True)
rows = make_dataset(OUT/'wav')


# %%
print(np.shape(rows))

# %%
print(type(rows))
print(rows[0])

# %%

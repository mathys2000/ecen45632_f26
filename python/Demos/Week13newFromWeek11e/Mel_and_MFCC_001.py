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
# # Mel and MFCC 001
#
# Mel spectrograms and Mel-frequency cepstral coefficients (MFCCs)
#
# Using Librosa and building from scratch

# %%
import argparse
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd


# %%
# %matplotlib widget
fsz = (7,4)
fsz2 = (fsz[0], 1.5*fsz[1]/2)

# %%
# Parameters
Fs = 16000    # target sampling rate
#n_fft = 512
n_fft = 1024
n_mels = 128
n_mfcc = 20
hop_length = 128
win_length = n_fft
window = 'hann'
f_min = 0
f_max = Fs/2

# %%
# Load waveform and standardize sample rate
#f_name, disp_name = 'Zest_8000_mono.wav', 'Speech Signal'
f_name, disp_name = 'Zest_44100_stereo.wav', 'Speech Signal' 
#f_name, disp_name = 'UrbanSoundSamples/100648-1-0-0_fold10_class1.wav', 'car horn'
#f_name, disp_name = 'UrbanSoundSamples/100795-3-0-0_fold10_class3.wav', 'dog bark'
#f_name, disp_name = 'UrbanSoundSamples/101382-2-0-10_fold10_class2.wav', 'children playing'

x, sr = librosa.load(f_name, sr=Fs, mono=True)

print(f'{len(x)/Fs} sec')

# %%
ipd.Audio(x, rate=Fs)


# %%
# Compute STFT
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                 window=window, center=False)

# Power spectrogram
P = np.abs(X) ** 2


# %%
print(np.shape(X))
print(np.min(np.abs(X)))
print(np.max(np.abs(X)))


# %%
# Plot linear and log spectrograms
(m, n) = np.shape(X)
tt = (hop_length*np.arange(n) + win_length//2)/Fs
ff = Fs/n_fft*np.arange(n_fft//2+1)
fig, axs = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
im0 = axs[0].pcolormesh(tt, ff, np.abs(X), vmin=0, vmax=2, cmap='magma', shading='auto')
axs[0].set_title(f'Spectrogram (Linear Amplitude), Nfft={n_fft}, hop={hop_length}, Fs={Fs} Hz')
axs[0].set_ylabel('Frequency [Hz]')
fig.colorbar(im0, ax=axs[0], label='Linear')   # add a color bar
im1 = axs[1].pcolormesh(tt, ff, 10*np.log10(np.maximum(P, 1e-6)), cmap='magma', shading='auto')
axs[1].set_title('Spectrogram in dB')
axs[1].set_ylabel('Frequnecy [Hz]')
axs[1].set_xlabel('Time [sec]')
fig.colorbar(im1, ax=axs[1], label='Decibels (dB)')

plt.show()


# %%
# ------------ Mel Spectrogram from scratch ----------------------------------------------------

# %%
# Hz to/from Mel conversion
def hz_to_mel(hz):
    """
    Convert frequency in Hz to Mel scale HTK-style
    m = 2595*log10(1 + f/700)
    """
    return 2595*np.log10(1 + hz/700.0)

def mel_to_hz(mel):
    """
    Convert HTK-style Mel scale to frequency in Hz
    f = 700*(10**(m/2595) - 1)
    """
    return 700*(10**(mel/2595) - 1)



# %%
# Mel Filterbank
def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    Build a triangular Mel-filterbank matrix.
    """
    if fmax is None:
        fmax = sr/2.0    # Nyquist frequency

    # 1. Map min/max Hz frequencies to Mel scale
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # 2. Generate linearly spaced points in the Mel scale (plus 2 for the edges)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    #print(mel_points)

    # 3. Convert Mel points back to Hz
    hz_points = mel_to_hz(mel_points)
    #print(hz_points)

    # 4. Map Hz points to FFT bin numbers
    bins = np.floor((n_fft + 1)*hz_points/sr).astype(int)
    #print(bins)

    # 5. Construct the triangular filter banks
    num_fft_bins = int(n_fft//2 + 1)
    fbank = np.zeros((n_mels, num_fft_bins))

    for m in range(1, n_mels + 1):
        f_m_minus = bins[m-1]     # left edge
        f_m = bins[m]             # Center peak
        f_m_plus = bins[m+1]      # Right edge

        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - bins[m-1])/(bins[m] - bins[m-1])
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bins[m+1] - k)/(bins[m+1] - bins[m])

    # 6. Normalize area under filter
        hz_left = hz_points[m-1]
        hz_right = hz_points[m+1]
        
        # Avoid division by zero just in case frequencies overlap perfectly
        if hz_right > hz_left:
            fbank[m-1] *= 2.0/(hz_right - hz_left)

    return fbank, bins
    


# %%
Fs2 = Fs
n_fft2 = 2*n_fft
n_mels2 = 16
mel_fbank2, bins2 = create_mel_filterbank(Fs2, n_fft2, n_mels2)


# %%
ff_fbank2 = Fs2/n_fft2 * np.arange(n_fft2//2 + 1)
fig, axs = plt.subplots(1, 1, figsize=(9, 4))
fig.canvas.toolbar_position = 'top'
for i in range(np.shape(mel_fbank2)[0]):
    axs.plot(ff_fbank2, mel_fbank2[i])
axs.set_title(f'Triangular Mel Filterbank, n_mels={n_mels2}, Fs={Fs2} Hz')
axs.set_ylabel('Weight')
axs.set_xlabel('Frequency [Hz]')
axs.grid(alpha = 0.5)
plt.show()

# %%
# Generate Mel filterbank
mel_fbank, bins = create_mel_filterbank(Fs, n_fft, n_mels, f_min, f_max)
#mel_fbank, bins = create_mel_filterbank(Fs, n_fft, 32)

# Dot product in frequency domain maps linear frequency bins to Mel bins
mel_spectrogram = np.dot(mel_fbank, P[:n_fft//2+1,:])

# Convert to Log scale (dB) to match human perception
log_mel_spectrogram = 10*np.log10(mel_spectrogram + 1e-12)

# %%
# Plot linear and log Mel spectrograms
(m, n) = np.shape(X)
tt = (hop_length*np.arange(n) + win_length//2)/Fs
ff_mel = hz_to_mel(Fs/n_fft*bins[1:-1])
fig, axs = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
im0 = axs[0].pcolormesh(tt, ff_mel, np.abs(mel_spectrogram), vmin=0, vmax=0.1, cmap='magma', shading='auto')
axs[0].set_title(f'Mel Spectrogram (Linear Amplitude), Nfft={n_fft}, hop={hop_length}, Fs={Fs} Hz')
axs[0].set_ylabel('Frequency [Mel]')
fig.colorbar(im0, ax=axs[0], label='Linear')   # add a color bar
im1 = axs[1].pcolormesh(tt, ff_mel, np.maximum(log_mel_spectrogram, -60), cmap='magma', shading='auto')
#im1 = axs[1].pcolormesh(tt, bins[1:-1], np.maximum(log_mel_spectrogram, -80), cmap='magma', shading='auto')
#im1 = axs[1].imshow(log_mel_spectrogram, aspect='auto', origin='lower', cmap='magma')
axs[1].set_title('Mel Spectrogram in dB')
axs[1].set_ylabel('Frequency [Mel]')
axs[1].set_xlabel('Time [sec]')
fig.colorbar(im1, ax=axs[1], label='Decibels (dB)')

plt.show()

# %%
# Plot Log spectrogram and log Mel spectrogram for comparison
(m, n) = np.shape(X)
tt = (hop_length*np.arange(n) + win_length//2)/Fs
ff_mel = hz_to_mel(Fs/n_fft*bins[1:-1])
fig, axs = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
im0 = axs[0].pcolormesh(tt, ff, 10*np.log10(np.maximum(P, 1e-6)), cmap='magma', shading='auto')
axs[0].set_title('Spectrogram in dB')
axs[0].set_ylabel('Frequnecy [Hz]')
axs[0].set_xlabel('Time [sec]')
fig.colorbar(im0, ax=axs[0], label='Decibels (dB)')
im1 = axs[1].pcolormesh(tt, ff_mel, np.maximum(log_mel_spectrogram, -60), cmap='magma', shading='auto')
#axs[1].yaxis.set_major_formatter(lambda val, pos: f'{mel_to_hz(val):.0f}')
axs[1].set_title('Mel Spectrogram in dB')
axs[1].set_ylabel('Frequency [Mel]')
axs[1].set_xlabel('Time [sec]')
fig.colorbar(im1, ax=axs[1], label='Decibels (dB)')

plt.show()


# %%
# ------------- Generate MFCCs ------------------------------------------------------------------------

# %%
# Type II Discrete Cosine Transform
def dct_type2_orthogonal(matrix, n_mfcc=20):
    """
    Computes the Type-II DCT along the rows of a matrix with orthogonal normalization.
    
    matrix: Log-mel spectrogram matrix of shape (n_mels, num_frames)
    n_mfcc: Number of cepstral coefficients to retain (usually 13 to 20)
    """
    n_mels, num_frames = matrix.shape
    mfcc = np.zeros((n_mfcc, num_frames))
    
    # Precompute indices for vectorization over frames
    m_indices = np.arange(n_mels)
    
    for c in range(n_mfcc):
        # Scale factor for orthogonal normalization
        if c == 0:
            ortho_scale = np.sqrt(1 / n_mels)
        else:
            ortho_scale = np.sqrt(2 / n_mels)
            
        # Compute the DCT equation for coefficient 'c' across all time frames
        # Cosine term: cos( pi * c * (2m + 1) / (2 * n_mels) )
        cos_term = np.cos(np.pi * c * (2 * m_indices + 1) / (2 * n_mels))
        
        # Multiply along the Mel-axis and scale
        mfcc[c, :] = ortho_scale * np.dot(cos_term, matrix)
        
    return mfcc



# %%
# Compute MFCCs from log_mel_spectrogram
mfccs = dct_type2_orthogonal(log_mel_spectrogram, n_mfcc)
mfccs_avg = mfccs.mean(axis=1)    # MFCCs mean
mfccs_std = mfccs.std(axis=1)     # MFCCs standard deviation


# %%
# Plot Waveform, Log Mel Spectrogram, MFCC coefficients
fig, axs = plt.subplots(3, 1, figsize=(9, 6), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
axs[0].plot(np.arange(len(x))/Fs, x, linewidth=0.6)
axs[0].set_title(f'Original Audio Waveform, "{disp_name}"')
axs[0].grid(alpha = 0.5)
axs[0].set_ylabel('Amplitude')
im1 = axs[1].pcolormesh(tt, ff_mel, np.maximum(log_mel_spectrogram, -60), cmap='magma', shading='auto')
#axs[1].yaxis.set_major_formatter(lambda val, pos: f'{mel_to_hz(val):.0f}')
axs[1].set_title(f'Mel Spectrogram in dB, n_mels={n_mels}, n_fft={n_fft}, Fs={Fs} Hz')
axs[1].set_ylabel('Frequency [Mel]')
fig.colorbar(im1, ax=axs[1], label='Decibels (dB)')
im2 = axs[2].pcolormesh(tt, np.arange(n_mfcc), mfccs, vmin=-100, vmax=100, cmap='viridis', shading='auto')
#im2 = axs[2].imshow(mfccs, aspect='auto', origin='lower', cmap='magma')
axs[2].set_title(f'MFCC Coefficients, n_mfcc={n_mfcc}')
axs[2].set_ylabel('MFCC Index')
axs[2].set_xlabel('Time [sec]')
fig.colorbar(im2, ax=axs[2], label='Linear')
plt.show()

# %%
ipd.Audio(x, rate=Fs)

# %%
print(f'mfccs_avg:\n{mfccs_avg}')
print(f'mfccs_std:\n{mfccs_std}')

# %%
# Plot mfccs_avg and mfccs_std
fig, axs = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
markerline, stemlines, baseline = axs[0].stem(np.log10(np.abs(mfccs_avg)), label='log10(mfccs_avg)')
plt.setp(stemlines, linewidth=2.0)
markerline, stemlines, baseline = axs[0].stem(np.sign(mfccs_avg), '--r', markerfmt='+', label='sign(mfccs_avg)')
plt.setp(stemlines, linewidth=0.4)
plt.setp(markerline, markersize=10)
axs[0].grid(alpha=0.5)
axs[0].set_title(f'Mean and Std Dev of MFCCs, n_mfcc={n_mfcc}, "{disp_name}"')
axs[0].legend()
axs[1].stem(mfccs_std, label='mfccs_std')
axs[1].grid(alpha=0.5)
axs[1].set_xlabel('MFCC Index')
axs[1].legend()
plt.show()

# %%
# ---------- Mel Spectrogram and MFCCs Using Librosa --------------------------------------------------

# %%
# Mel-frequency filter bank
mel_power = librosa.feature.melspectrogram(S=P, sr=Fs, n_fft=n_fft, hop_length=hop_length,
                                           n_mels=n_mels, fmin=f_min, fmax=f_max, htk=False)

# Convert Mel power to decibels
mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)


# %%
# Mel-Frequency Cepstral Coefficients
# DCT-II of the log-Mel representation
mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc, dct_type=2, norm="ortho")
mfcc_avg = mfcc.mean(axis=1)    # MFCC mean
mfcc_std = mfcc.std(axis=1)     # MFCC standard deviation


# %%
# Check matrix shapes
print(f"STFT shape: {X.shape}")
print(f"Mel spectrogram shape: {mel_db.shape}")
print(f"MFCC shape: {mfcc.shape}")


# %%
# Plot Waveform, Librosa Log Mel Spectrogram and MFCC coefficients
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'

# Waveform
axes[0].plot(np.arange(len(x))/Fs, x, linewidth=0.6)
axes[0].set_title(f'Librosa: Original Audio Waveform, "{disp_name}"')
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

# Mel spectrogram
img1 = librosa.display.specshow(mel_db, sr=Fs, hop_length=hop_length, x_axis="time", y_axis="mel",
                                ax=axes[1], cmap="magma")
axes[1].set_title("Librosa: Log-Mel Power Spectrogram")
fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

# MFCCs
img2 = librosa.display.specshow(mfcc, sr=Fs, hop_length=hop_length, x_axis="time",
                                ax=axes[2], cmap="coolwarm")

axes[2].set_title("Librosa: MFCC Coefficients")
axes[2].set_ylabel("MFCC Index")
fig.colorbar(img2, ax=axes[2])

plt.show()

# %%
# Plot Librosa mfcc_avg and mfcc_std
fig, axs = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)
fig.canvas.toolbar_position = 'top'
markerline, stemlines, baseline = axs[0].stem(np.log10(np.abs(mfcc_avg)), label='log10(mfcc_avg)')
plt.setp(stemlines, linewidth=2.0)
markerline, stemlines, baseline = axs[0].stem(np.sign(mfcc_avg), '--r', markerfmt='+', label='sign(mfcc_avg)')
plt.setp(stemlines, linewidth=0.4)
plt.setp(markerline, markersize=10)
axs[0].grid(alpha=0.5)
axs[0].set_title(f'Librosa: Mean and Std Dev of MFCCs, n_mfcc={n_mfcc}, "{disp_name}"')
axs[0].legend()
axs[1].stem(mfccs_std, label='mfcc_std')
axs[1].grid(alpha=0.5)
axs[1].set_xlabel('MFCC Index')
axs[1].legend()
plt.show()

# %%
print(f'mfcc_avg:\n{mfcc_avg}')
print(f'mfcc_std:\n{mfcc_std}')

# %%

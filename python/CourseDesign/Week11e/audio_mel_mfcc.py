
"""
audio_mel_mfcc.py

Read a WAV file, calculate a Mel spectrogram and MFCCs,
plot the results, and optionally save features.

Usage:
    python audio_mel_mfcc.py recording.wav

    python audio_mel_mfcc.py recording.wav --save-dir results
"""

import argparse
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def extract_features(
    filename,
    target_sr=16000,
    n_fft=512,
    hop_length=160,
    n_mels=64,
    n_mfcc=13,
):
    # -------------------------------------------------
    # 1. Load waveform and standardize sample rate
    # -------------------------------------------------
    x, sr = librosa.load(
        filename,
        sr=target_sr,
        mono=True
    )

    if len(x) < n_fft:
        raise ValueError(
            f"Audio needs at least {n_fft} samples."
        )

    duration = len(x) / sr

    print(f"File: {filename}")
    print(f"Sample rate: {sr} Hz")
    print(f"Samples: {len(x)}")
    print(f"Duration: {duration:.3f} seconds")

    # -------------------------------------------------
    # 2. Short-Time Fourier Transform (STFT)
    # -------------------------------------------------
    X = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False
    )

    # Power spectrogram
    P = np.abs(X) ** 2

    # -------------------------------------------------
    # 3. Mel-frequency filter bank
    # -------------------------------------------------
    mel_power = librosa.feature.melspectrogram(
        S=P,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr / 2
    )

    # Convert Mel power to decibels
    mel_db = librosa.power_to_db(
        mel_power,
        ref=np.max,
        top_db=80.0
    )

    # -------------------------------------------------
    # 4. Mel-Frequency Cepstral Coefficients
    # -------------------------------------------------
    # DCT-II of the log-Mel representation
    mfcc = librosa.feature.mfcc(
        S=mel_db,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho"
    )

    print(f"STFT shape: {X.shape}")
    print(f"Mel spectrogram shape: {mel_db.shape}")
    print(f"MFCC shape: {mfcc.shape}")

    return x, sr, mel_db, mfcc


def plot_features(x, sr, mel_db, mfcc, hop_length):
    # -------------------------------------------------
    # 5. Plot waveform, Mel spectrogram, and MFCCs
    # -------------------------------------------------
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 10),
        constrained_layout=True
    )

    # Waveform
    t = np.arange(len(x)) / sr

    axes[0].plot(t, x, linewidth=0.6)
    axes[0].set_title("Original Audio Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Mel spectrogram
    img1 = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma"
    )

    axes[1].set_title("Log-Mel Power Spectrogram")
    fig.colorbar(
        img1, ax=axes[1], format="%+2.0f dB"
    )

    # MFCCs
    img2 = librosa.display.specshow(
        mfcc,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        ax=axes[2],
        cmap="coolwarm"
    )

    axes[2].set_title("MFCC Coefficients")
    axes[2].set_ylabel("MFCC Index")
    fig.colorbar(img2, ax=axes[2])

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="WAV to Mel spectrogram and MFCCs"
    )

    parser.add_argument(
        "wav_file",
        help="Path to input WAV file"
    )

    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory for output files"
    )

    args = parser.parse_args()

    x, sr, mel_db, mfcc = extract_features(
        args.wav_file
    )

    fig = plot_features(
        x, sr, mel_db, mfcc, hop_length=160
    )

    if args.save_dir:
        output_dir = Path(args.save_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_dir / "mel_spectrogram.npy",
            mel_db
        )

        np.save(
            output_dir / "mfcc.npy",
            mfcc
        )

        fig.savefig(
            output_dir / "audio_features.png",
            dpi=200
        )

        print(f"Saved results to {output_dir}")

    plt.show()


if __name__ == "__main__":
    main()
